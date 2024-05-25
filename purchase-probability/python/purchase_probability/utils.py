    # Package to build the model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Package to build the app
import logging
import datetime
import tempfile
import sys
import json
import pickle
import os.path
from google.cloud import bigquery, storage, exceptions
from google_pandas_load import Loader, LoadConfig

# Import params from python file
import purchase_probability.parameters as parameters

# Instantiate logger
logger = logging.getLogger(__name__)

def create_dataset_if_not_exists(bq_client, dataset_name):
    dataset_ref = bq_client.dataset(dataset_name)
    try:
        bq_client.get_dataset(dataset_ref)
        logger.info("%s already exists" % dataset_ref)
        return True
    except exceptions.NotFound:
        logger.info("%s does not exist" % dataset_ref)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = 'EU'
        bq_client.create_dataset(dataset)
        logger.info("Dataset {} created in BQ".format(dataset_ref))
        return True
    except Exception as e:
        logger.error("Error when creating dataset {}: {}".format(dataset_ref, e))
        sys.exit(1)

# Function to retrieve distinct values of dummy features
def get_dummy_feature(bq_client,
                      gs_client,
                      dummy,
                      project_id,
                      bucket,
                      path):

    dataset_ref = bigquery.dataset.DatasetReference(project=project_id,
                                                    dataset_id=parameters.dataset_name)
    bucket = storage.bucket.Bucket(client=gs_client, name=bucket)

    gpl = Loader(bq_client=bq_client,
                 dataset_ref=dataset_ref,
                 bucket=bucket,
                 local_dir_path=path,
                 logger=logger)
    try:
        res_query = gpl.load(source='query',
                             destination='dataframe',
                             query='SELECT DISTINCT ' + dummy + " FROM `" + project_id + ".data.events`",
                             delete_in_bq=True
                            )
    except Exception:
        res_query = gpl.load(source='query',
                             destination='dataframe',
                             query='SELECT DISTINCT ' + dummy + " FROM `" + project_id + ".data.visits`",
                             delete_in_bq=True
                             )

    # Remove white space in the beginning and in the end, and lower case every name
    res_query[dummy] = res_query[dummy].str.strip().str.lower()
    # Remove 'unknown' from distinct values
    res_query = res_query[(res_query[dummy] != 'unknown') &
                          (res_query[dummy] != '?')]
    # Unify mobile and smartphone
    res_query[dummy].replace(['mobile'], ['mobile phone'], inplace=True)
    res_query[dummy].replace(['computer'], ['desktop'], inplace=True)
    res_query[dummy].replace(['game_console'], ['other'], inplace=True)
    res_query[dummy].replace(['dmr'], ['other'], inplace=True)
    res_query[dummy].replace(['wearable'], ['mobile device'], inplace=True)
    # keep only distinct values
    res_query.drop_duplicates(inplace=True)
    return res_query[dummy]


def preprocessing_params(bq_client,
                         gs_client,
                         feat_to_embed,
                         dict_params):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Import
        dataset_ref = bigquery.dataset.DatasetReference(project=dict_params['project_id'],
                                                        dataset_id=parameters.dataset_name)
        bucket = storage.bucket.Bucket(client=gs_client, name=dict_params['bucket_name'])

        gpl = Loader(bq_client=bq_client,
                     dataset_ref=dataset_ref,
                     bucket=bucket,
                     local_dir_path=tmpdir,
                     logger=logger)

        # Sizes of vocabulary come from this query
        categorical_values = {}
        voc_sizes = {}
        logger.info('Requesting bigquery to retrieve values of categorical features to embed...')

        for cat in feat_to_embed:
            query_cat = dict_params['query_dict']['categories'].format(category=cat, **dict_params)
            res_query = gpl.load(source='query',
                                 destination='dataframe',
                                 query=query_cat,
                                 delete_in_bq=True).append({cat: 0}, ignore_index=True)
            categorical_values.update({cat: res_query
                                       })
            voc_sizes.update({cat: len(res_query)
                              })
        logger.info('Size of vocabularies: {}'.format(voc_sizes))

        # Values for each dummy features in a dict
        dummy_cat = {parameters.features_to_dummify[0]: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                                         'Friday', 'Saturday'],
                     parameters.features_to_dummify[1]: list(range(1, 13)),
                     parameters.features_to_custom_dummify[0]: list(
                         get_dummy_feature(bq_client,
                                           gs_client,
                                           parameters.features_to_custom_dummify[0],
                                           project_id=dict_params['project_id'],
                                           bucket=dict_params['bucket_name'],
                                           path=tmpdir).values),
                     parameters.features_to_custom_dummify[1]: list(
                         get_dummy_feature(bq_client,
                                           gs_client,
                                           parameters.features_to_custom_dummify[1],
                                           project_id=dict_params['project_id'],
                                           bucket=dict_params['bucket_name'],
                                           path=tmpdir).values),
                     }

    return categorical_values, dummy_cat, voc_sizes

def export_bq_to_gs(bq_client,
                    gs_client,
                    project_id,
                    dataset_name,
                    bucket,
                    query_dict,
                    list_data_date,
                    directory,
                    action,
                    min_date):
    """ Export data from Bigquery to Google Storage """

    # Create params dictionary
    params = {
        'selected_columns': ', '.join(parameters.selected_columns),
        'features_event': ', '.join(parameters.features_event),
        'features_visit': ', '.join(parameters.features_visit),
        'features_product': ', '.join(parameters.features_product),
        'project_id': project_id,
        'dataset_name': dataset_name,
        'bucket_name': bucket,
        'query_dict': query_dict}

    # Add known = 1 to the query if action is 'train', nothing if action is 'predict' (we take everyone for prediction)
    if (action == 'train') | (action == 'evaluate'):
        params['known_filter'] = 'AND known'
    elif action == 'predict':
        params['known_filter'] = ''

    # Instantiate bq_client and gs_client
    dataset_ref = bigquery.dataset.DatasetReference(project=project_id,
                                                    dataset_id=parameters.dataset_name)
    bucket = storage.bucket.Bucket(client=gs_client, name=bucket)

    with tempfile.TemporaryDirectory() as tmpdir:
        gpl = Loader(bq_client=bq_client,
                     dataset_ref=dataset_ref,
                     bucket=bucket,
                     local_dir_path=tmpdir,
                     gs_dir_path_in_bucket=directory,
                     logger=logger)

        # Check if we already created a param file in gs (from previous preprocessing)
        # If not, query bigquery
        if list(bucket.list_blobs(prefix=directory + '/preprocessing_params.pkl')):
            logger.info('Preprocessing params already in gs...')
        else:
            logger.info('Preprocessing params not in gs, querying bigquery...')
            categorical_values, dict_dummy_cat, voc_sizes = preprocessing_params(bq_client, gs_client,
                                                                                 feat_to_embed=parameters.features_to_embed,
                                                                                 dict_params=params)

            with open(os.path.join(tmpdir, 'preprocessing_params.pkl'), 'wb') as f:
                pickle.dump((categorical_values, dict_dummy_cat, voc_sizes), f, 4)
            gpl.load(source='local',
                     destination='gs',
                     data_name='preprocessing_params.pkl')
            logger.info('Preprocessing params created in gs (file name: preprocessing_params.pkl)')

        # Create rnn_params.json file if not already created
        if list(bucket.list_blobs(prefix=directory + '/rnn_params.json')):
            logger.info('Rnn params already in gs, continue')
        else:
            logger.info('Rnn params not in gs')
            # Download the params
            gpl.load(source='gs',
                     destination='local',
                     data_name='preprocessing_params.pkl',
                     delete_in_gs=False)
            with open(os.path.join(tmpdir, 'preprocessing_params.pkl'), 'rb') as f:
                categorical_values, dict_dummy_cat, voc_sizes = pickle.load(f)

            # Create dict with params, which will be loaded when training the rnn
            rnn_params = {'N_INPUT': len(parameters.features_numerical) +
                                     len(parameters.features_engineered) +
                                     len(dict_dummy_cat[parameters.features_to_dummify[0]]) +
                                     len(dict_dummy_cat[parameters.features_to_dummify[1]]) +
                                     parameters.n_values_custom_dummies[parameters.features_to_custom_dummify[0]] +
                                     parameters.n_values_custom_dummies[parameters.features_to_custom_dummify[1]]
                          }
            rnn_params.update(voc_sizes)

            rnn_params['min_date'] = min_date

            with open(os.path.join(tmpdir, 'rnn_params.json'), 'w') as fp:
                json.dump(rnn_params, fp)
            gpl.load(source='local',
                     destination='gs',
                     data_name='rnn_params.json')
            logger.info('Rnn params created in gs (file name: rnn_params.json)')

        job_configs = []
        for date in list_data_date:
            # Transform data_date to real datetime format
            date_datetime = datetime.datetime.strptime(date, '%Y%m%d')
            params['data_date'] = date

            # Create date for beginning of history and end of data (including data for creating labels)
            # Add test: check diff between oldest data and begin_hist?
            # History length can be shorter than 90 days expected
            params['date_begin_hist'] = (date_datetime - datetime.timedelta(days=parameters.length_history)
                                         ).strftime('%Y%m%d')
            params['date_end_hist'] = (date_datetime + datetime.timedelta(days=parameters.forward_prediction)
                                       ).strftime('%Y%m%d')

            logger.info('Begin date of history is {}'.format(params['date_begin_hist']))
            logger.info('End date (including {} days ahead to calculate labels) is {}'.format(parameters.forward_prediction, params['date_end_hist']))

            # Query in queries.yaml.
            query_features = query_dict['retrieve_features'].format(**params)

            job_configs.append(LoadConfig(source='query',
                                          destination='gs',
                                          query=query_features,
                                          data_name='df_' + date
                                          )
                               )
        logger.info('Query jobs created, submit list of job to bigquery:')
        gpl.mload(configs=job_configs)
        logger.info('Files created in Google storage')
