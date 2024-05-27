import logging, os, sys, time, datetime, argparse
import yaml
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from googleapiclient import discovery
from explainer.model import explainer_fn
from explainer.tests.test_preprocessing import test_preprocessing
import explainer.parameters as param

logger = logging.getLogger(__name__)


def is_success(ml_engine_service, project_id, job_id):
    # Doc: https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#State
    wait = 60  # seconds
    timeout_preparing = datetime.timedelta(seconds=900)
    timeout_running = datetime.timedelta(hours=24)
    api_call_time = datetime.datetime.now()
    api_job_name = "projects/{project_id}/jobs/{job_name}".format(project_id=project_id, job_name=job_id)
    job_description = ml_engine_service.projects().jobs().get(name=api_job_name).execute()
    while not job_description["state"] in ["SUCCEEDED", "FAILED", "CANCELLED"]:
        # check here the PREPARING and RUNNING state to detect the abnormalities of ML Engine service
        if job_description["state"] == "PREPARING":
            delta = datetime.datetime.now() - api_call_time
            if delta > timeout_preparing:
                logger.error("[ML] PREPARING stage timeout after %ss --> CANCEL job '%s'" % (delta.seconds, job_id))
                ml_engine_service.projects().jobs().cancel(name=api_job_name, body={}).execute()
                raise Exception
        if job_description["state"] == "RUNNING":
            delta = datetime.datetime.now() - api_call_time
            if delta > timeout_running + timeout_preparing:
                logger.error("[ML] RUNNING stage timeout after %ss --> CANCEL job '%s'" %(delta.seconds, job_id))
                ml_engine_service.projects().jobs().cancel(name=api_job_name, body={}).execute()
                raise Exception

        logger.info("[ML] NEXT UPDATE for job '%s' IN %ss (%ss ELAPSED IN %s STAGE)" % (job_id,
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

    parser = argparse.ArgumentParser(description="Running LIME explainer on instance.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--account_id", dest="account", help="account id to work on.")
    parser.add_argument("--date", dest="date", help="data date to work on.")
    parser.add_argument("--conf", dest="config", default="/etc/conf/explainer.yaml",
                        help="absolute or relative path of configuration file")
    parser.add_argument("--env", dest="env", help="environment : local or cloud", default="cloud")
    parser.add_argument("--usage", dest="usage", help="explanation mode : single or extensive", default="single")

    if len(sys.argv) < 4:
        parser.print_help()
        sys.exit(1)
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

    account_id = args.account

    logger.info("[RUNNING] LIME explainer for account %s" % (account_id))

    # Directory where are saved all the results of the explainer
    # working_directory = 'purchase-probability/train'

    # path_data = '/files'
    # path_model = '/model'

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    project_global = config['google_cloud']['project_id']
    project_account = config['google_cloud']['project_prefix'] + account_id
    if config['google_cloud']['credentials_json_file'] != "":
        credentials = service_account.Credentials.from_service_account_file(
            config['google_cloud']['credentials_json_file'])
        gs_client = storage.Client(project=project_global, credentials=credentials)
        bq_client = bigquery.Client(project=project_global, credentials=credentials)
        bq_client_account = bigquery.Client(project=project_account, credentials=credentials)
    else:
        credentials = None
        gs_client = storage.Client(project=project_global)
        bq_client = bigquery.Client(project=project_global)
        bq_client_account = bigquery.Client(project=project_account)

    dataset_ref = bq_client.dataset(config['google_bq']['dataset_prefix'] + account_id)
    bucket = gs_client.bucket(config['google_gcs']['bucket_prefix'] + account_id)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(dir_path)

    if ENV == 'local':

        if args.usage == 'test':

            test_preprocessing(project_id=project_global,
                               dataset_id=dataset_ref.dataset_id,
                               bucket_name=bucket.name,
                               directory=param.gpp['directory'],
                               path_data=param.gpp['files_path'],
                               )

        else:

            explainer = explainer_fn(project_id=project_global,
                                     dataset_id=dataset_ref.dataset_id,
                                     bucket_name=bucket.name,
                                     directory=param.gpp['directory'],
                                     # path_model=param.gpp['model_path'],
                                     path_data=param.gpp['files_path'],
                                     date=args.date,
                                     usage=args.usage)
    else:
        logger.info('Checking if packages needed by ML engine are available')

        packages = {p: 'gs://{bucket}/purchase-probability/packages/{package}'.format(bucket=bucket.name,
                                                                                    account_id=account_id,
                                                                                    package=p
                                                                                    )
                    for p in os.listdir(os.path.join(parent_path, 'packages'))}

        logger.debug("package URIs: %s " % list(packages.values()))

        for package_name, uri in packages.items():

            package_uri = uri.strip().split("gs://{bucket}/".format(bucket=bucket.name))[1]

            blob = bucket.blob(package_uri)

            if not blob.exists():

                logger.warning("blob %s does not exist on Google Storage, uploading..." % blob)

                blob.upload_from_filename(os.path.join(parent_path, 'packages', package_name))

                logger.info("blob %s available on Google Storage" % blob)

            else:

                logger.info("blob %s does exist on Google Storage, re-uploading..." % blob)

                blob.delete()

                blob.upload_from_filename(os.path.join(parent_path, 'packages', package_name))

                logger.info("blob %s available on Google Storage" % blob)

        # Size of machine we will do RNN on. Can be S, M or L
        # Fix to 'S' for now.
        mlmachine_size = 'S'

        logger.info('Start training on the cloud')

        ml_engine_service = discovery.build('ml', 'v1', credentials=credentials)

        job_parent = "projects/{project}".format(project=project_account)

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        job_id = "job_{}_{}".format(account_id, now_str)

        job_body = {'trainingInput':
                    {'pythonVersion': param.ml_pythonVersion,
                     'runtimeVersion': param.ml_runtimeVersion,
                     'scaleTier': param.typology_machine[mlmachine_size]['ml_scaleTier'],
                     'region': param.ml_region,
                     'pythonModule': 'Explainer.model',
                     'args': ["--global_proj_id", project_global,
                              "--bucket_name", bucket.name,
                              "--dataset_name", dataset_ref.dataset_id,
                              "--account_id", account_id,
                              "--data_date", args.date,
                              "--min_date", param.min_date,
                              "--working_directory", param.gpp['directory'],
                              "--path_data", param.gpp['files_path'],
                              "--usage", args.usage],
                     'packageUris': list(packages.values()),
                     'masterType': param.typology_machine[mlmachine_size]['ml_masterType']
                     # 'workerType': param.typology_machine[mlmachine_size]['ml_workerType'],
                     # 'workerCount': param.typology_machine[mlmachine_size]['ml_workerCount'],
                     # 'parameterServerCount': param.typology_machine[mlmachine_size]['ml_parameterServerCount'],
                     # 'parameterServerType': param.typology_machine[mlmachine_size]['ml_parameterServerType']
                     },
                    'jobId': job_id}

        logging.info("job_body: %s" % job_body)

        logging.info("job_parent: %s" % job_parent)

        logging.info("creating a job ml: %s" % job_id)

        job_ml = ml_engine_service.projects().jobs().create(parent=job_parent, body=job_body).execute()

        time.sleep(5)

        try:

            succeeded_job = is_success(ml_engine_service, project_account, job_id)

            if succeeded_job:

                logger.info('Training job done')

            else:

                logger.error('Training job failed')

                sys.exit(1)

        except Exception as e:

            logger.error(e)

            sys.exit(1)

        logger.info("[FINISHED] LIME explainer for account %s" % (account_id))

        time.sleep(5)
