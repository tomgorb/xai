# Package to build the model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Package to build the app
import logging
import argparse
import datetime
import tempfile
import sys
import json
import pickle
import os.path

from google.cloud import storage
from google_pandas_load import Loader

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

# Import params from python file
import purchase_probability.parameters as parameters

# Instantiate logger
logger = logging.getLogger(__name__)

def calculate_labels(df, cutoff_date):
    # Create a dataframe with one label per id in ids
    train_labels = pd.DataFrame({'id': df['id'].unique(), 'labels': [0] * len(df['id'].unique())})

    # Calculate the labels: select events between cutoff_date + 1 and cutoff_date  + parameters.forward_prediction, for each id in ids
    # select unique id in df where:
    #   _ event dates are between cutoff_date + 1 and cutoff_date + parameters.forward_prediction,
    #   _ id in ids,
    #   _ at least one event is an order.
    id_with_orders = df[(df['event_timestamp'] > cutoff_date) &
                        (df['event_timestamp'] <= cutoff_date + datetime.timedelta(days=parameters.forward_prediction)) &
                        (df['event_env'] == 'purchase')
                        ]['id'].unique()

    # Calculate labels: put 1 for ids in id_with_orders.
    # id in ids and not in id_with_orders have already the label 0
    train_labels.loc[train_labels.id.isin(id_with_orders), ['labels']] = 1
    # Sort value by id to insure compatibility with the feature dataframe
    train_labels.sort_values(by=['id'], inplace=True)
    return train_labels


def custom_embed_categorical_features(df, dict_categories, bucket, path, directory):
    # dict_categories should be a dictionary with keys equals to categorical features and values the distinct values
    # of this category

    gpl = Loader(bucket=bucket,
                 local_dir_path=path,
                 gs_dir_path_in_bucket=directory,
                 separator='\t')

    logger.info('Categories to embed: {}'.format(dict_categories.keys()))

    # Instantiate a dataframe
    df_cat_enc = pd.DataFrame()

    for feat in dict_categories.keys():
        # Check if we already saved a mapping between categories and integers
        if list(bucket.list_blobs(prefix=directory + '/metadata_' + feat + '.csv')):

            logger.info('Mapping for categorical feature {} is in gs, downloading ...'.format(feat))
            # Download the mapping
            metadata = gpl.load(source='gs',
                                destination='dataframe',
                                data_name='metadata_' + feat + '.csv',
                                delete_in_gs=False)
        else:
            logger.info('No mapping for categorical feature {} in gs'.format(feat))
            # Create a dictionary with integer 0 for blank string
            metadata = pd.DataFrame(list({'0': 0}.items()),
                                    columns=['Categorical_values', 'Labels'])

        # Compare categories with saved ones (or the one with just a value for blank string)
        cat_diff = list(set(df[feat].unique()) - set(metadata['Categorical_values']))
        cat_diff = [x for x in cat_diff if x != '' and x != '?']
        # If new categories, append to old_categories and save this to gs
        if len(cat_diff) > 0:
            df_diff = pd.DataFrame([[x, y]
                                    for x, y in zip(cat_diff,
                                                    list(range(max(max(metadata['Labels'] + 1), 0),
                                                               max(max(metadata['Labels'] + 1), 0) + len(cat_diff))))
                                    ], columns=['Categorical_values', 'Labels']
                                   )
            metadata = metadata.append(df_diff, ignore_index=True)

            metadata.to_csv(path + '/metadata_' + feat + '.csv', sep='\t', encoding='utf-8', index=False)

            gpl.load(source='local',
                     destination='gs',
                     data_name=os.path.basename(path + '/metadata_' + feat + '.csv'))
            logger.info('New values added to feature {}.'.format(feat))

        # Apply the mapping to feat, put type to 'category' to ensure order is preserved
        df_cat_enc[feat] = df[feat].astype(pd.api.types.CategoricalDtype(categories=metadata['Categorical_values'])).cat.codes

    return df_cat_enc


def calculate_inactivity(df, date):
    # Find timestamp of the most recent event for every id in ids
    fixed_features_train = df[df['event_date'] <= date
                              ].groupby(['id'], sort=False)['event_timestamp'].max().reset_index()

    fixed_features_train.loc[:, 'inactivity_time'] = (pd.to_datetime(date) + datetime.timedelta(days=1)
                                                      - fixed_features_train['event_timestamp'
                                                                             ]
                                                      ).dt.total_seconds() / (3600 * 24)

    return fixed_features_train


def make_sequence_example(id, num_inputs, cat_inputs, fixed_inputs, label):
    # inputs (num and cat features): A list of input vectors, each input vector is a list of float32
    # labels, inputs_fixed: A list of int64
    ids = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf-8')]))]
    input_num_features = [tf.train.Feature(float_list=tf.train.FloatList(value=input_)) for input_ in num_inputs]
    input_cat_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=input_)) for input_ in cat_inputs]
    label_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))]
    fixed_features = [tf.train.Feature(float_list=tf.train.FloatList(value=[fixed_inputs]))]

    feature_list = {
        'inputs_num': tf.train.FeatureList(feature=input_num_features),
        'inputs_cat': tf.train.FeatureList(feature=input_cat_features),
        'labels': tf.train.FeatureList(feature=label_features),
        'inputs_fixed': tf.train.FeatureList(feature=fixed_features),
        'ids': tf.train.FeatureList(feature=ids)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)


def write_instances_to_tfr(output_file, ids, num_feats, cat_feats, fixed_feats, labels):
    options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    writer = tf.io.TFRecordWriter(output_file, options=options)

    for id, seq_num, seq_cat, fixed_feat, label in zip(ids,
                                                              num_feats,
                                                              cat_feats,
                                                              fixed_feats,
                                                              labels):
        ex = make_sequence_example(id, seq_num, seq_cat, fixed_feat, label)
        writer.write(ex.SerializeToString())
    writer.close()


def group_features_by_id(dataset, cutoff_date, col_numerical, col_embed, ids):
    # Only keep up to the last 200 events by id, before the cutoff date

    logger.info("Group by ID : numerical and embedded columns...")

    temp_list = dataset[(dataset['event_date'] <= cutoff_date) &
                        (dataset['id'].isin(ids))
                        ].groupby('id').tail(200)[['id'] + col_numerical].values

    idx = np.unique(temp_list[:, 0], return_index=True)

    temp_list_num = np.split(temp_list[:, 1:], idx[1][1:])

    temp_list = dataset[(dataset['event_date'] <= cutoff_date) &
                        (dataset['id'].isin(ids))
                        ].groupby('id').tail(200)[['id'] + col_embed].values

    temp_list_embed = np.split(temp_list[:, 1:], idx[1][1:])

    # Data frame: features grouped by id, and one column for the sequence length
    logger.info("DataFrame creation...")
    feat_by_id = pd.DataFrame({'id': idx[0],
                                      'features_num': temp_list_num,
                                      'features_cat': temp_list_embed,
                                      'Seq_length': [len(temp_list[i]) for i in range(len(idx[0]))]})
    del temp_list, temp_list_num, temp_list_embed
    # Sort value by id to insure compatibility with the labels
    logger.info("DataFrame creation OK")

    logger.info("Sorting DataFrame by ID...")
    feat_by_id.sort_values(by=['id'], inplace=True)
    logger.info("Sorting DataFrame by ID OK")

    return feat_by_id


# Function to make timestamps format uniform in the dataframe
def event_timestamp_ms(x):
    if len(x) == 23:
        return datetime.datetime.strptime(x.replace(' UTC', '.000 UTC'), '%Y-%m-%d %H:%M:%S.%f UTC')
    else:
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f UTC')


##########################
# Preprocessing function
##########################

def preprocessing_rnn(data_date,
                      bucket,
                      directory,
                      action
                      ):

    logger.info('Starting preprocessing for rnn on date {}'.format(data_date))

    with tempfile.TemporaryDirectory() as tmpdir:
        gs_client = storage.Client()
        bucket = storage.bucket.Bucket(client=gs_client, name=bucket)

        gpl = Loader(bucket=bucket,
                     local_dir_path=tmpdir,
                     gs_dir_path_in_bucket=directory)

        # If we already did this step, just load the pickle file, otherwise calculate the preprocessing params
        # Once it is calculated, preprocessing_params are not updated
        if list(bucket.list_blobs(prefix=directory + '/preprocessing_params.pkl')):
            logger.info('Preprocessing params in gs, loading them...')
            # Download the params
            gpl.load(source='gs',
                     destination='local',
                     data_name='preprocessing_params.pkl',
                     delete_in_gs=False)
            with open(os.path.join(tmpdir, 'preprocessing_params.pkl'), 'rb') as f:
                categorical_values, dict_dummy_cat, voc_sizes = pickle.load(f)
        else:
            logger.error('Preprocessing params not in gs, you should do the export from BigQuery first.')
            sys.exit(1)

        # Transform data_date to real datetime format
        date_datetime = datetime.datetime.strptime(data_date, '%Y%m%d')

        # Load data from gs, who was previously export from bigquery
        logger.info('Dataframe not present: querying gs...')

        # Retrieve data from gs for the date data_date
        df = gpl.load(source='gs',
                      destination='dataframe',
                      data_name='df_' + data_date,
                      delete_in_gs=True  # put to true if we want to delete input files from gs
                      )

        if len(df) == 0:
            logger.warning('Input dataset is empty.')
        # Reset index: when we download the csv from bigquery, several dataframes are concatenated together,
        # so the index can be redundant.
        logger.info('Drop useless col')
        df.reset_index(drop=True, inplace=True)
        if ['Unnamed: 0'] in np.array(df.columns):
            df.drop('Unnamed: 0', axis=1, inplace=True)

        logger.info('event_timestamp normalization')
        # Change timestamp to have the same format everywhere
        df.loc[:, 'event_timestamp'] = pd.to_datetime(df.event_timestamp.str[:19],
                                                      format='%Y-%m-%d %H:%M:%S')

        logger.info('Sort dataframe')
        # Sort the dataframe by id and event_timestamp, in order to calculate the delta in time between events
        # (by id)
        df.sort_values(['id', 'event_timestamp'], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info('Calculate delta_time')
        # Calculate the time difference between events
        series_delta_time = df['event_timestamp'].diff()

        logger.info('Mask data')
        # Create a mask for the first event of every id
        mask = df.id != df.id.shift(1)

        # Use the mask to put the delta time to 0 (first event should have 0 time diff)
        series_delta_time[mask] = datetime.timedelta(0)

        logger.info('Concat delta_time')
        # Put the delta time inside a column in the dataframe
        df = pd.concat((series_delta_time.to_frame(name='delta_time'), df), axis=1)

        logger.info('Convert time delta to seconds')
        # Convert time delta to seconds
        df.loc[:, parameters.features_engineered[0]] = df.delta_time.dt.total_seconds()

        logger.info('Create event_date variable')
        # Create a event_date column without the hours, minutes, seconds etc
        df.loc[:, 'event_date'] = [x.date() for x in df.event_timestamp]
        # df.loc[:, 'event_date'] = pd.to_datetime(df.event_date)

        logger.info('Fill na')
        # Fill nan values
        df[parameters.features_id_dates + parameters.features_numerical] = df[
            parameters.features_id_dates + parameters.features_numerical].fillna(0)
        df[parameters.features_to_dummify + parameters.features_to_custom_dummify] = df[
            parameters.features_to_dummify + parameters.features_to_custom_dummify].fillna('')
        df[parameters.features_to_embed] = df[parameters.features_to_embed].fillna('0')

        # Embeddings
        # Vectorized categorical variables
        logger.info('Embed categorical features...')

        df_categorical_encoded = custom_embed_categorical_features(df,
                                                                   categorical_values,
                                                                   bucket=bucket,
                                                                   path=tmpdir,
                                                                   directory=directory
                                                                   )

        for col in df_categorical_encoded.columns:
            df_categorical_encoded[col].replace([-1], [0], inplace=True)

        # Drop useless categorical features
        df.drop(parameters.features_to_embed, inplace=True, axis=1)

        # Add vectorized categorical variables
        df = pd.concat([df, df_categorical_encoded], axis=1)

        logger.info('Number of columns in df: ' + str(len(df.columns)))

        logger.info('Calculate labels...')
        labels = calculate_labels(df, date_datetime)
        logger.info('Calculate inactivity time...')
        inactivity = calculate_inactivity(df, date_datetime.date())

        percentage_win = 100 * sum(labels['labels']) / len(labels['labels'])

        logger.info("Percentage of purchase: {0}".format(percentage_win))

        logger.info('Mean of inactivity: {} days'.format(inactivity['inactivity_time'].mean()))

        # Names of numerical columns + id column & timestamp column
        col_num_and_embed = parameters.features_numerical + parameters.features_engineered + \
            parameters.features_to_embed

        # Get dummy variables for some categorical variables
        dict_dummy = {}

        # For each category to dummify, change the type of that category column to category,
        # with the list of category contained in dict_dummy_cat
        for col in parameters.features_to_dummify:
            categories = dict_dummy_cat[col]
            df[col] = df[col].astype(pd.api.types.CategoricalDtype(categories=categories))
            dummy = pd.get_dummies(df[col], prefix=col)
            dict_dummy.update({col: dummy})

        # For each category to custom dummify
        for col in parameters.features_to_custom_dummify:
            # Check if we already saved a mapping between categories and integers
            if list(bucket.list_blobs(
                    prefix=directory + '/mapping_dummy_feature_'+col+'.json')):
                logger.info('Mapping for dummy feature {} is in gs, downloading ...'.format(col))
                # Download the mapping
                gpl.load(source='gs',
                         destination='local',
                         data_name='mapping_dummy_feature_'+col+'.json',
                         delete_in_gs=False)
                old_categories = json.load(open(os.path.join(tmpdir, "mapping_dummy_feature_"+col+'.json')))
            else:
                logger.info('No mapping for dummy feature {} in gs'.format(col))
                # Create a dictionary with integer 0 for blank string
                old_categories = {'': 0}

            # Retrieve categories for 'col' feature
            categories = dict_dummy_cat[col]

            # Compare categories with saved ones (or the one with just a value for blank string)
            cat_diff = list(set(categories) - set(old_categories))

            # If new categories, append to old_categories and save this to gs
            if len(cat_diff) > 0:
                for new_cat in cat_diff:
                    old_categories[new_cat] = len(old_categories.keys())
                with open(os.path.join(tmpdir, 'mapping_dummy_feature_'+col+'.json'), 'w') as fp:
                    json.dump(old_categories, fp)
                gpl.load(source='local',
                         destination='gs',
                         data_name='mapping_dummy_feature_'+col+'.json')

            # Remove white space in the beginning and in the end, and lower case every name
            df[col] = df[col].str.strip().str.lower()
            # Remove 'unknown' and consider it as null
            df[col].replace(['unknown'], [''], inplace=True)
            # Remove '?' and consider it as null
            df[col].replace(['?'], [''], inplace=True)
            # Unify mobile and smartphone
            df[col].replace(['mobile'], ['mobile phone'], inplace=True)
            # Unify computer and desktop
            df[col].replace(['computer'], ['desktop'], inplace=True)
            # Change wearable to mobile device
            df[col].replace(['wearable'], ['mobile device'], inplace=True)
            # ...
            df[col].replace(['game_console'], ['other'], inplace=True)
            df[col].replace(['dmr'], ['other'], inplace=True)

            # Create list with integers corresponding to categorical value in df[col]
            df_numerized = [[old_categories[x]] for x in df[col]]

            # Create a OneHotEncoder with a fixed number of values
            # Old version
            #  encoder = OneHotEncoder(n_values=parameters.n_values_custom_dummies[col],
            #                        handle_unknown='error')
            # New working version
            encoder = OneHotEncoder(categories=[range(parameters.n_values_custom_dummies[col])],
                                    handle_unknown='error')

            col_names = categories + ['Na_' + str(x) for x in range(
                parameters.n_values_custom_dummies[col] - len(categories))]
            col_names = [col + '_' + x for x in col_names]
            # Create a dataframe with dummy values
            # old version
            #  dummy = pd.DataFrame.from_items(zip(col_names,
            #                                     encoder.fit_transform(df_numerized).transpose().toarray()))
            # New working version
            dummy = pd.DataFrame.from_dict(dict(zip(col_names,
                                                    encoder.fit_transform(df_numerized).transpose().toarray())))

            logger.info('Number of rows in dummy dataframe for features {} '
                        '(should be the same as number of rows in df): {}'.format(col, len(dummy)))
            dict_dummy.update({col: dummy})

        # Concat dummy variables with numerical variables
        # Be sure we concatenate each features in dict_dummy
        df = pd.concat([df['event_date'],
                        df[parameters.features_id_dates],
                        dict_dummy[parameters.features_to_dummify[0]],
                        dict_dummy[parameters.features_to_dummify[1]],
                        dict_dummy[parameters.features_to_custom_dummify[0]],
                        dict_dummy[parameters.features_to_custom_dummify[1]],
                        df[col_num_and_embed]
                        ], axis=1, copy=False)

        logger.info('Number of columns in df: {}'.format(len(df.columns)))

        # Column names we want to keep: only numerical ones without dummy ones and embedded ones: useful later
        col_except_id_dates = [i for i in df.columns if i not in parameters.features_id_dates + ['event_date']]
        col_num_wo_embed = [i for i in col_except_id_dates if i not in parameters.features_to_embed]

        logger.info('Number of inputs for each timestamp: {} (including {} embedded categories)'.format(
                        len(col_except_id_dates), len(parameters.features_to_embed)))

        logger.info('Group features by ids from date {}'.format(data_date))

        # Write all examples into a TFRecords file
        if action == 'predict':
            output_file = os.path.join(tmpdir, 'Sequences_for_prediction_' + data_date)
        else:
            output_file = os.path.join(tmpdir, 'Sequences_' + data_date)

        # Hack to retrieve the same ids in the same order.
        # Joining dataframes would be too costly in terms of memory
        labels.sort_values(by='id', inplace=True)

        inactivity.sort_values(by='id', inplace=True)

        unique_mi = df.sort_values(by='id')['id'].unique()
        counter = 0
        batch_size = 300000

        while batch_size * counter < len(unique_mi):
            id_subset = unique_mi[batch_size * counter:batch_size * (counter + 1)]
            feat_by_ids_train = group_features_by_id(df,
                                                                   date_datetime.date(),
                                                                   col_num_wo_embed,
                                                                   parameters.features_to_embed,
                                                                   id_subset)

            options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
            writer = tf.io.TFRecordWriter(output_file + '_' + str(counter) + '.tfr',
                                                 options=options)
            for id, sequence_num, sequence_cat, fixed_feature, \
                sequence_label in zip(feat_by_ids_train['id'],
                                      feat_by_ids_train['features_num'],
                                      feat_by_ids_train['features_cat'],
                                      inactivity['inactivity_time'][inactivity['id'].isin(id_subset)],
                                      labels['labels'][labels['id'].isin(id_subset)]
                                      ):
                ex = make_sequence_example(id, sequence_num, sequence_cat, fixed_feature, sequence_label)
                writer.write(ex.SerializeToString())
            writer.close()

            gpl.load(source='local',
                     destination='gs',
                     data_name=os.path.basename(output_file + '_' + str(counter) + '.tfr'))

            counter += 1

        logger.info('Day ' + data_date + ' done, successfully written in gs:' + directory + '/'
                    + os.path.basename(output_file) + '!')

        logger.info('Deleting ' + tmpdir + ' directory...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_date', help='data_date')
    parser.add_argument('--bucket_name', help='bucket_name')
    parser.add_argument("--directory", help="directory where we do stuff")
    parser.add_argument("--action", help="Action type")

    args = parser.parse_args()

    preprocessing_rnn(data_date=args.data_date,
                      bucket=args.bucket_name,
                      directory=args.directory,
                      action=args.action)
