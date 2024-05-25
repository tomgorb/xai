import os
import re
import sys
import time
import datetime
import argparse
import logging
import random
import yaml

from google.cloud import bigquery, storage
from google.oauth2 import service_account
from googleapiclient import discovery
from google_pandas_load import Loader

# Import params from python file
import purchase_probability.parameters as parameters

from purchase_probability.utils import export_bq_to_gs, create_dataset_if_not_exists

logger = logging.getLogger(__name__)


def is_success(ml_engine_service, project_id, job_id):
    # Doc: https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#State
    wait = 60 # seconds
    timeout_preparing = datetime.timedelta(seconds=900)
    timeout_running = datetime.timedelta(hours=6)
    api_call_time = datetime.datetime.now()
    api_job_name = "projects/{project_id}/jobs/{job_name}".format(project_id=project_id, job_name=job_id)
    job_description = ml_engine_service.projects().jobs().get(name=api_job_name).execute()
    while not job_description["state"] in ["SUCCEEDED", "FAILED", "CANCELLED"]:
        # check here the PREPARING and RUNNING state to detect the abnormalities of ML Engine service
        if job_description["state"] == "PREPARING":
            delta = datetime.datetime.now() - api_call_time
            if delta > timeout_preparing:
                logger.error("[ML] PREPARING stage timeout after %ss --> CANCEL job '%s'" %(delta.seconds, job_id))
                ml_engine_service.projects().jobs().cancel(name=api_job_name, body={}).execute()
                raise Exception
        if job_description["state"] == "RUNNING":
            delta = datetime.datetime.now() - api_call_time
            if delta > timeout_running + timeout_preparing:
                logger.error("[ML] RUNNING stage timeout after %ss --> CANCEL job '%s'" %(delta.seconds, job_id))
                ml_engine_service.projects().jobs().cancel(name=api_job_name, body={}).execute()
                raise Exception

        logger.info("[ML] NEXT UPDATE for job '%s' IN %ss (%ss ELAPSED IN %s STAGE)" %(job_id,
                                                                                       wait,
                                                                                       delta.seconds,
                                                                                       job_description["state"]))
        job_description = ml_engine_service.projects().jobs().get(name=api_job_name).execute()
        time.sleep(wait)
    logger.info("Job '%s' done" % job_id)
    # Check the job state
    if job_description["state"] == "SUCCEEDED":
        logger.info("Job '%s' succeeded!" % job_id)
        return True
    else:
        logger.error(job_description["errorMessage"])
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trigger a Purchase Probability action.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action",       help="action switch : train, evaluate or predict")
    parser.add_argument("--account_id", dest="account_id", required=True,  help="Account ID to work on.")
    parser.add_argument("--ds_nodash",  dest="ds_nodash", required=True,  help="Airflow execution date (20190101).")
    parser.add_argument("--env",        dest="env", required=False, default="local", help="Environment : local or cloud.")
    parser.add_argument("--mode",       dest="mode", required=False, default="prod", help="Mode: test or prod.")
    parser.add_argument("--conf",       dest="conf", required=False, default="/etc/conf/purchase-probability.yaml", help="Absolute or relative path to configuration file.")

    args = parser.parse_args()

    ENV = args.env

    root_logger = logging.getLogger()
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient').setLevel(logging.WARNING)
    logging.getLogger('google_auth_httplib2').setLevel(logging.WARNING)

    if ENV == 'local':
        root_logger.setLevel(logging.DEBUG)

    account_id = args.account_id
    execution_date = datetime.datetime.strptime(args.ds_nodash, "%Y%m%d")

    logger.info("[RUNNING] purchase-probability for account %s (job: %s)" % (account_id, args.action))
    if args.action not in ['check_model', 'train', 'evaluate', 'predict']:
        logger.error("action %s does not exist!" % args.action)
        logger.info("[CRASHED] purchase-probability for account %s (job: %s)" % (account_id, args.action))
        sys.exit(1)

    with open(args.conf, 'r') as f:
        config = yaml.safe_load(f)

    working_directory = 'purchase-probability/train'.format(account_id=account_id)

    path_data = '/files'
    path_model = '/model'

    project_id = config['google_cloud']['project_prefix'] + account_id
    if config['google_cloud']['credentials_json_file'] != "":
        credentials = service_account.Credentials.from_service_account_file(
                        config['google_cloud']['credentials_json_file'])
        gs_client = storage.Client(project=project_id, credentials=credentials)
        bq_client_account = bigquery.Client(project=project_id, credentials=credentials)
    else:
        credentials = None
        gs_client = storage.Client(project=project_id)
        bq_client_account = bigquery.Client(project=project_id)

    dataset_ref = bq_client_account.dataset(parameters.dataset_name)
    bucket = gs_client.bucket(config['google_gcs']['bucket_prefix']+account_id)

    if args.action == 'check_model':
        p = re.compile("([0-9]{8})")
        blobs = bucket.list_blobs(prefix=working_directory+path_model+'/model_last_execution_date_')
        models = list({p.search(blob.name).group(1) for blob in blobs})
        logger.info(models)
        if len(models) == 0:
            logger.info("no rnn found --> train")
            time.sleep(5)
            sys.exit(5)
        else:
            latest = datetime.datetime.strptime(max(models), "%Y%m%d")
            logger.info(latest.date())
            if (execution_date-latest).days <= random.randint(5,7):
                logger.info("rnn found")
                time.sleep(3)
                sys.exit(3)
            else:
                logger.info("rnn re-train")
                time.sleep(5)
                sys.exit(5)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir_path,"queries.yaml")) as query_file:
        query_dict = yaml.safe_load(query_file)

    # Query to retrieve all data dates.
    QUERY = ('SELECT DISTINCT FORMAT_DATE("%Y%m%d", DATE(event_timestamp)) as table_date '
             'FROM `{}.data.events` where event_env = "product_page" '.format(project_id))

    query_job = bq_client_account.query(QUERY)  # API request
    dates_iterator = query_job.result()  # Waits for query to finish
    available_dates = sorted([x[0] for x in dates_iterator])
    min_date = datetime.datetime.strptime(min(available_dates), '%Y%m%d')
    max_date = datetime.datetime.strptime(max(available_dates), '%Y%m%d')
    logger.info("Min and Max of available dates in our BQ datasets: {}, {}.".format(min_date.date(), max_date.date()))

    if execution_date > max_date+datetime.timedelta(days=1):
        logger.warning("The execution date is too recent compared to max date available in the dataset: {} > {}".format(
            execution_date.date(), max_date.date()))
        sys.exit(0)

    else:
        if execution_date == max_date+datetime.timedelta(days=1):
            execution_date = max_date

    # Test in the training history if there is missing data
    days_before_execution_date = [(execution_date - datetime.timedelta(days=x)).date().strftime("%Y%m%d") for x in range(parameters.training_history)]
    days_missing = len(days_before_execution_date) - len([x for x in days_before_execution_date if x in available_dates])
    if days_missing > 0:
        logger.warning("{} daily table(s) are missing in BQ "
                       "from the previous {} days before the execution date ({}).".format(days_missing, parameters.training_history, execution_date.date()))
        if days_missing > int(0.1*parameters.training_history):
            logger.warning("[ABORTED] too many days missing (should be at most {}).".format(int(0.1*parameters.training_history)))
            sys.exit(0)

    # Now it's execution date minus training_history plus forward_prediction (to have some history)
    first_training_date = execution_date - \
                          datetime.timedelta(days=parameters.training_history) + \
                          datetime.timedelta(days=parameters.forward_prediction)
    first_training_date = max(min_date+datetime.timedelta(days=parameters.forward_prediction),first_training_date)
    logger.info('First date of training considered: {}'.format(first_training_date.date()))

    if len(list(bucket.list_blobs(prefix=working_directory + path_data + '/Sequences_'))) == 0:
        logger.info("""First run of the marvelous preprocessing before LSTM network!!!
            --> Preprocess & Train the network with all available data (DO NOT PANIC, BE PATIENT)""")

    nb_training_days = max(0, (execution_date - datetime.timedelta(days=parameters.forward_prediction-1) - first_training_date).days)
    logger.info('{} training days'.format(nb_training_days))

    #####
    if nb_training_days <= 0:
        logger.warning('Not enough data: we need to be at least {} days after the first training date, '
                       'so {} days after {}. Exiting'.format(parameters.forward_prediction, parameters.forward_prediction, first_training_date.date()))
        logger.info("[ABORTED] purchase-probability for account %s (job: %s)" % (account_id, args.action))
        sys.exit(0)
    #####

    if args.action == 'train':
        list_data_date = [(first_training_date + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(nb_training_days)]
    elif args.action == 'predict':
        list_data_date = [execution_date.strftime("%Y%m%d")]
    elif args.action == 'evaluate':
        list_data_date = [(execution_date - datetime.timedelta(days=parameters.forward_prediction)).strftime("%Y%m%d")]

    if args.mode == 'test':
        list_data_date = [list_data_date[-1]]

    # Check if we already did preprocessing for dates in list_data_dates
    # Preprocessed files have a different name when we do it for predictions (hence the big 'if')
    date_to_preprocess = []
    for data_date in list_data_date:
        if ((args.action == 'predict') & \
            (len(list(bucket.list_blobs(prefix=working_directory + path_data + '/Sequences_for_prediction_' + data_date))) != 0)) | \
           ((args.action != 'predict') & \
            (len(list(bucket.list_blobs(prefix=working_directory + path_data + '/Sequences_' + data_date))) != 0)):
            continue
        else:
            date_to_preprocess.append(data_date)
    logger.info('Dates which will be preprocessed: {}'.format(date_to_preprocess))

    if date_to_preprocess:

        create_dataset_if_not_exists(bq_client=bq_client_account, dataset_name=parameters.dataset_name)

        # Check if we have csv for dates in list_data_dates
        date_to_export = []
        for data_date in date_to_preprocess:
            if list(bucket.list_blobs(prefix=working_directory + path_data + '/df_' + data_date)):
                continue
            else:
                date_to_export.append(data_date)

        logger.info('Dates which will be exported from bq: {}'.format(date_to_export))
        logger.info('Export function from bq to gs...')

        export_bq_to_gs(bq_client=bq_client_account,
                        gs_client=gs_client,
                        project_id=project_id,
                        dataset_name=dataset_ref.dataset_id,
                        bucket=bucket.name,
                        query_dict=query_dict,
                        list_data_date=date_to_export,
                        directory=working_directory + path_data,
                        action=args.action,
                        min_date=min_date)

    logger.info('Checking if packages needed by ML engine are available')
    packages = {p: 'gs://{bucket}/purchase-probability/packages'
                   '/{package}'.format(bucket=bucket.name,
                                       account_id=account_id,
                                       package=p
                                       ) for p in os.listdir(os.path.join(dir_path, 'packages'))}

    logger.debug("package URIs: %s " % list(packages.values()))
    for package_name, uri in packages.items():
        package_uri = uri.strip().split("gs://{bucket}/".format(bucket=bucket.name))[1]
        blob = bucket.blob(package_uri)
        if not blob.exists():
            logger.warning("blob %s does not exist on Google Storage, uploading..." % blob)
            blob.upload_from_filename(os.path.join(dir_path, 'packages', package_name))
            logger.info("blob %s available on Google Storage" % blob)
        else:
            logger.info("blob %s does exist on Google Storage, re-uploading..." % blob)
            blob.delete()
            blob.upload_from_filename(os.path.join(dir_path, 'packages', package_name))
            logger.info("blob %s available on Google Storage" % blob)

    logger.info('Preprocessing function...')
    if date_to_preprocess:
        if args.mode == 'prod' and ENV=='cloud':
            ml_engine_service = discovery.build('ml', 'v1', credentials=credentials, cache_discovery=False)

            job_parent = "projects/{project}".format(project=project_id)
            job_id_list = []
            job_counter = 0
            for data_date in date_to_preprocess:
                logger.info(data_date)
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                job_id = "purchase_probability_preprocess_{}_{}".format(account_id, now_str)
                job_id_list.append(job_id)

                job_body = {'trainingInput':
                            {'pythonVersion': parameters.ml_pythonVersion,
                             'runtimeVersion': parameters.ml_runtimeVersion,
                             'scaleTier': parameters.ml_preprocess['ml_scaleTier_train'],
                             'region': parameters.ml_region,
                             'pythonModule': 'purchase_probability.preprocessing',
                             'args': ["--data_date", data_date,
                                      "--bucket_name", bucket.name,
                                      "--directory", working_directory + path_data,
                                      "--action", args.action],
                             'packageUris': list(packages.values()),
                             'masterType': parameters.ml_preprocess['ml_masterType']
                             },
                            'jobId': job_id}

                logging.info("job_body: %s" % job_body)
                logging.info("job_parent: %s" % job_parent)
                logging.info("creating a job ml: %s" % job_id)
                job_ml = ml_engine_service.projects().jobs().create(parent=job_parent, body=job_body).execute()
                time.sleep(3)
                job_counter += 1
                if job_counter % parameters.max_nb_parallel_jobs == 0 or job_counter == len(date_to_preprocess):
                    try:
                        succeeded_jobs = all([is_success(ml_engine_service, project_id,
                                                         job_id) for job_id in job_id_list])
                        if succeeded_jobs:
                            logger.info('All previous jobs done, running some more')
                        else:
                            logger.error('At least one job failed :(')
                            sys.exit(1)
                    except Exception as e:
                        logger.error(e)
                        sys.exit(1)
                    job_id_list = []

        elif args.mode == 'test' or ENV =='local':
            from purchase_probability.preprocessing import preprocessing_rnn
            preprocessing_rnn(data_date=list_data_date[0],
                              bucket=bucket.name,
                              directory=working_directory + path_data,
                              action=args.action)
    else:
        logging.info('No date to preprocess, moving on to training or prediction')

    if args.action == 'train':

        # Ml-jobs have to be finished in order to process to the next step
        logger.info('Training function...')

        if ENV == 'local':
            from purchase_probability.train_eval_predict import train_rnn
            train_rnn(bucket_name=bucket.name,
                      execution_date=execution_date,
                      first_training_date=first_training_date,
                      directory=working_directory,
                      path_data=path_data,
                      path_model=path_model,
                      max_step=500)
        elif ENV == 'cloud':
            # Size of machine we will do RNN on. Can be S, M or L
            # Fix to 'M' for now.
            # TODO We should use total size of tfr to choose between S, M or L
            mlmachine_size = 'L'

            logger.info('Start training on the cloud')
            ml_engine_service = discovery.build('ml', 'v1', credentials=credentials, cache_discovery=False)

            job_parent = "projects/{project}".format(project=project_id)

            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            job_id = "purchase_probability_train_{}_{}".format(account_id, now_str)
            job_body = {'trainingInput':
                        {'pythonVersion': parameters.ml_pythonVersion,
                         'runtimeVersion': parameters.ml_runtimeVersion,
                         'scaleTier': parameters.typology_machine[mlmachine_size]['ml_scaleTier_train'],
                         'region': parameters.ml_region,
                         'pythonModule': 'purchase_probability.train_eval_predict',
                         'args': ["--bucket_name", bucket.name,
                                  "--dataset_name", dataset_ref.dataset_id,
                                  "--execution_date", execution_date.strftime("%Y%m%d"),
                                  "--first_training_date", first_training_date.strftime("%Y%m%d"),
                                  "--action", args.action,
                                  "--working_directory", working_directory,
                                  "--path_data", path_data,
                                  "--path_model", path_model],
                         'packageUris': list(packages.values()),
                         'masterType': parameters.typology_machine[mlmachine_size]['ml_masterType'],
                         'workerType': parameters.typology_machine[mlmachine_size]['ml_workerType'],
                         'workerCount': parameters.typology_machine[mlmachine_size]['ml_workerCount'],
                         'parameterServerCount': parameters.typology_machine[mlmachine_size
                                                                             ]['ml_parameterServerCount'],
                         'parameterServerType': parameters.typology_machine[mlmachine_size
                                                                            ]['ml_parameterServerType']
                         },
                        'jobId': job_id}

            logging.info("job_body: %s" % job_body)
            logging.info("job_parent: %s" % job_parent)
            logging.info("creating a job ml: %s" % job_id)
            job_ml = ml_engine_service.projects().jobs().create(parent=job_parent, body=job_body).execute()
            time.sleep(5)

            try:
                succeeded_job = is_success(ml_engine_service, project_id, job_id)
                if succeeded_job:
                    logger.info('Training job done')
                else:
                    logger.error('Training job failed')
                    sys.exit(1)
            except Exception as e:
                logger.error(e)
                sys.exit(1)

        logger.info("[FINISHED] purchase-probability for account %s (job: %s)" % (account_id, args.action))

    elif args.action == 'predict':
        # Do predictions
        pred_file = ['gs://' + bucket.name + '/' + working_directory + path_data +
                     '/Sequences_for_prediction_' + execution_date.strftime("%Y%m%d") + '_*.tfr'
                     ]

        if ENV == 'local':
            from purchase_probability.train_eval_predict import predict_rnn
            predictions = predict_rnn(bucket_name=bucket.name,
                                      pred_files=pred_file,
                                      directory=working_directory,
                                      path_data=path_data,
                                      path_model=path_model,
                                      execution_date=execution_date)

        elif ENV == 'cloud':
            ml_engine_service = discovery.build('ml', 'v1', credentials=credentials, cache_discovery=False)

            job_parent = "projects/{project}".format(project=project_id)

            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            job_id = "purchase_probability_predict_{}_{}".format(account_id, now_str)

            job_body = {'trainingInput':
                        {'pythonVersion': parameters.ml_pythonVersion,
                         'runtimeVersion': parameters.ml_runtimeVersion,
                         'scaleTier': parameters.ml_predict['ml_scaleTier_train'],
                         'region': parameters.ml_region,
                         'pythonModule': 'purchase_probability.train_eval_predict',
                         'args': ["--bucket_name", bucket.name,
                                  "--dataset_name", dataset_ref.dataset_id,
                                  "--train_or_pred_files", pred_file,
                                  "--action", args.action,
                                  "--execution_date", execution_date.strftime("%Y%m%d"),
                                  "--working_directory", working_directory,
                                  "--path_data", path_data,
                                  "--path_model", path_model],
                         'packageUris': list(packages.values()),
                         'masterType': parameters.ml_predict['ml_masterType']
                         },
                        'jobId': job_id}

            logging.info("job_body: %s" % job_body)
            logging.info("job_parent: %s" % job_parent)
            logging.info("creating a job ml: %s" % job_id)
            job_ml = ml_engine_service.projects().jobs().create(parent=job_parent, body=job_body).execute()
            time.sleep(5)

            try:
                succeeded_job = is_success(ml_engine_service, project_id, job_id)
                if succeeded_job:
                    logger.info('Predicting job done')
                else:
                    logger.error('Predicting job failed')
                    sys.exit(1)
            except Exception as e:
                logger.error(e)
                sys.exit(1)

        dataset_ref_account = bq_client_account.dataset(parameters.dataset_name)

        gpl = Loader(bq_client=bq_client_account,
                     dataset_ref=dataset_ref_account,
                     bucket=bucket,
                     gs_dir_path_in_bucket=working_directory + path_data,
                     logger=logger)

        gpl.load(source='gs',
                 destination='bq',
                 data_name='predictions_' + execution_date.strftime("%Y%m%d"),
                 bq_schema=[bigquery.SchemaField('id', 'STRING'),
                            bigquery.SchemaField('purchase_proba', 'FLOAT')],
                 write_disposition='WRITE_TRUNCATE')
        logger.info(pred_file[0])

    elif args.action == 'evaluate':

        if ENV == 'local':
            logger.info('Test function...')
            from purchase_probability.train_eval_predict import test_rnn
            predictions, labels = test_rnn(bucket_name=bucket.name,
                                           directory=working_directory,
                                           path_data=path_data,
                                           path_model=path_model,
                                           execution_date=execution_date)

            logger.info('Test visualization of output')
            from purchase_probability.train_eval_predict import visualize_rnn
            visualize_rnn(predictions=predictions,
                          labels=labels,
                          execution_date=execution_date,
                          bucket_name=bucket.name,
                          directory=working_directory,
                          path_model=path_model)
        elif ENV == 'cloud':
            ml_engine_service = discovery.build('ml', 'v1', credentials=credentials, cache_discovery=False)

            job_parent = "projects/{project}".format(project=project_id)

            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            job_id = "purchase_probability_evaluate_{}_{}".format(account_id, now_str)

            job_body = {'trainingInput':
                            {'pythonVersion': parameters.ml_pythonVersion,
                             'runtimeVersion': parameters.ml_runtimeVersion,
                             'scaleTier': parameters.ml_predict['ml_scaleTier_train'],
                             'region': parameters.ml_region,
                             'pythonModule': 'purchase_probability.train_eval_predict',
                             'args': ["--bucket_name", bucket.name,
                                      "--dataset_name", dataset_ref.dataset_id,
                                      "--action", args.action,
                                      "--execution_date", execution_date.strftime("%Y%m%d"),
                                      "--working_directory", working_directory,
                                      "--path_data", path_data,
                                      "--path_model", path_model],
                             'packageUris': list(packages.values()),
                             'masterType': parameters.ml_predict['ml_masterType']
                             },
                        'jobId': job_id}

            logging.info("job_body: %s" % job_body)
            logging.info("job_parent: %s" % job_parent)
            logging.info("creating a job ml: %s" % job_id)
            job_ml = ml_engine_service.projects().jobs().create(parent=job_parent, body=job_body).execute()
            time.sleep(5)

            try:
                succeeded_job = is_success(ml_engine_service, project_id, job_id)
                if succeeded_job:
                    logger.info('Evaluating job done')
                else:
                    logger.error('Evaluating job failed')
                    sys.exit(1)
            except Exception as e:
                logger.error(e)
                sys.exit(1)

    else:
        pass

    # Delete preprocessed files older than execution_date - 13 months:
    if args.action != 'evaluate':
        suffix = '/Sequences_' if args.action == 'train' else '/Sequences_for_prediction_'
        bucket_path = working_directory + path_data + suffix

        preprocessed_files = list(bucket.list_blobs(prefix=bucket_path))

        files_to_delete = [x.name for x in preprocessed_files
                           if x.name[len(bucket_path):len(bucket_path)+8
                                     ] < (execution_date - datetime.timedelta(days=395)).date().strftime("%Y%m%d")]
        for file in files_to_delete:
            bucket.delete_blobs(blobs=list(bucket.list_blobs(prefix=file)))

    logger.info("[FINISHED] purchase-probability for account %s (job: %s)" % (account_id, args.action))
