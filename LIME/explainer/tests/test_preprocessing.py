import logging
import argparse
from purchase_probability.train_eval_predict import create_input_fn
from purchase_probability import parameters
from explainer.model import file_loader
from explainer.preprocessing import mapping, reshaping
from explainer.utilities import gplLoader
from explainer import parameters as param
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import json
import os

logger = logging.getLogger(__name__)


def test_preprocessing(project_id, dataset_id, bucket_name, directory, path_data):

    # Setting the google client
    gpl = gplLoader(project_id, dataset_id, bucket_name, directory, path_data)

    # Loading the rnn_params file (necessary for using the functions from ml_buy_probability)
    gpl.load(source='gs',
             destination='local',
             data_name='rnn_params.json',
             delete_in_gs=False)

    rnn_params_json = json.load(open(os.path.join('/tmp/', 'rnn_params.json')))

    parameters.model_params['N_INPUT'] = rnn_params_json['N_INPUT']

    train_file = ['/tmp/purchase-probability_train_files_Sequences_20200303_0.tfr']

    # Creating the input function for the train set

    train_input_fn = create_input_fn(path=train_file)

    train_iterator = train_input_fn(shuffle_data=True)

    logger.info('Starting loading train files')

    # The training files are chosen among those having a sequence length equal to the desired one

    train_feat, _ = file_loader(train_iterator)

    logger.info('Ended loading train files')

    length_max = max([max(train_feat[i]['Lengths_features']) for i in range(len(train_feat))])

    mapping_index = {'num_to_num':
                         [(param.lstm_features.index(i), e)
                          for e, i in enumerate(param.mapping_features['num_to_num'])],
                     'num_to_cat_encoded':
                         [(param.encoded_num2cat_idx[i], e)
                          for e, i in enumerate(param.mapping_features['num_to_cat_encoded'])],
                     'num_to_cat':
                         [(param.lstm_features.index(i), param.new_feature_names['Categorical_features'].index(i))
                          for i in param.mapping_features['num_to_cat']]
                     }

    sample_data = train_feat[0]

    pad_data = {'Numerical_features': np.pad(sample_data['Numerical_features'],
                                             ((0, 0),
                                              (0, length_max - sample_data['Numerical_features'].shape[1]),
                                              (0, 0)),
                                             constant_values=-1.
                                             ),
                'Categorical_features': np.pad(sample_data['Categorical_features'],
                                               ((0, 0),
                                                (0, length_max - sample_data['Numerical_features'].shape[1]),
                                                (0, 0)),
                                               constant_values=-1
                                               ),
                'Lengths_features': sample_data['Lengths_features'],
                'Fixed_features': sample_data['Fixed_features'],
                'Ids': sample_data['Ids']}

    logger.info('TESTING MAPPING FUNCTION')

    mapped_data = mapping(param.lstm_features, pad_data)

    logger.info('Checking proper shape')

    logger.info('Shape Numerical features: ' + str(mapped_data['Numerical_features'].shape))

    logger.info('Expected number of Numerical features: ' + str(len(param.new_feature_names['Numerical_features'])))

    logger.info('Shape Categorical features: ' + str(mapped_data['Categorical_features'].shape))

    logger.info('Expected number of Categorical features: ' + str(len(param.new_feature_names['Categorical_features'])))

    logger.info('Checking consistency of values')

    for e, i in enumerate(mapping_index['num_to_num']):

        logger.info('Number of different values from data to mapped data for '
                    + str(param.mapping_features['num_to_num'][e]) + ': ' +
                    str(np.sum(mapped_data['Numerical_features'][:, :, i[1]] !=
                               pad_data['Numerical_features'][:, :, i[0]])))

    for e, i in enumerate(mapping_index['num_to_cat']):

        logger.info('Number of different values from data to mapped data for '
                    + str(param.mapping_features['num_to_cat'][e]) + ': ' +
                    str(np.sum(mapped_data['Categorical_features'][:, :, i[1]] !=
                               pad_data['Numerical_features'][:, :, i[0]])))

    for e, i in enumerate(mapping_index['num_to_cat_encoded']):

        enc = OneHotEncoder(handle_unknown='ignore')

        enc.fit(np.arange(i[0][1] - i[0][0]).reshape(-1, 1))

        nb_in = 0

        for j in range(mapped_data['Categorical_features'].shape[0]):

            nb_in += np.sum(pad_data['Numerical_features'][j,
                           :pad_data['Lengths_features'][j], i[0][0]:i[0][1]] != enc.transform(
                mapped_data['Categorical_features'][j, :pad_data['Lengths_features'][j], i[1]].reshape(-1, 1)))

        logger.info('Number of different values from data to mapped data for '
                    + str(param.mapping_features['num_to_cat_encoded'][e]) + ': ' +
                    str(nb_in))

    logger.info('TESTING RESHAPING FUNCTION')

    reshaped_data = reshaping(mapped_data)

    logger.info('Checking proper shape')

    row, cols = reshaped_data.shape

    logger.info('Features in data: ' + str(cols))

    logger.info('Expected features: ' + str(length_max * (len(param.new_feature_names['Numerical_features']) +
                                                          len(param.new_feature_names['Categorical_features'])) + 2))

    logger.info('Checking consistency of values')

    for i in param.new_feature_names['Numerical_features']:

        nb = 0

        for j in range(length_max):

            nb += np.sum(~np.isclose(pad_data['Numerical_features'][:, j, param.lstm_features.index(i)],
                                     reshaped_data[:, param.new_feature_names['Numerical_features'].index(i)
                                                      + j * len(param.new_feature_names['Numerical_features'])]))

        logger.info('Number of different values for feature ' + str(i) + ': ' + str(nb))

    for i in param.mapping_features['num_to_cat']:

        nb = 0

        for j in range(length_max):

            nb += np.sum(~np.isclose(pad_data['Numerical_features'][:, j, param.lstm_features.index(i)],
                                     reshaped_data[:, param.new_feature_names['Categorical_features'].index(i) +
                                     len(param.new_feature_names['Numerical_features']) * length_max
                                                      + j * len(param.new_feature_names['Categorical_features'])]))

        logger.info('Number of different values for feature ' + str(i) + ': ' + str(nb))

    for i in param.mapping_features['num_to_cat_encoded']:

        nb = 0

        enc = OneHotEncoder(handle_unknown='ignore')

        enc.fit(np.arange(param.encoded_num2cat_idx[i][1] - param.encoded_num2cat_idx[i][0]).reshape(-1, 1))

        for k in range(pad_data['Lengths_features'].shape[0]):

            for j in range(pad_data['Lengths_features'][k]):

                transformed_data = enc.transform(reshaped_data[k, len(param.new_feature_names['Numerical_features'])
                                                              * length_max +
                                                              param.new_feature_names['Categorical_features'].index(i)
                                                              + j * len(param.new_feature_names['Categorical_features'])
                                             ].reshape(-1, 1)).toarray()

            nb += np.sum(~np.isclose(pad_data['Numerical_features'][k, j,
                                     param.encoded_num2cat_idx[i][0]:param.encoded_num2cat_idx[i][1]],
                                     transformed_data))

        logger.info('Number of different values for feature ' + str(i) + ': ' + str(nb))

    logger.info('FINISHED TEST PREPROCESSING')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_id', help='project_id')
    parser.add_argument('--dataset_id', help='dataset_id')
    parser.add_argument('--bucket_name', help='bucket_name')
    parser.add_argument("--working_directory", help="Directory where we do stuff")
    parser.add_argument("--path_data", help="Path for files in gs bucket")

    args = parser.parse_args()

    test_preprocessing(project_id=args.project_id,
                       dataset_id=args.dataset_id,
                       bucket_name=args.bucket_name,
                       directory=args.working_directory,
                       path_data=args.path_data)
