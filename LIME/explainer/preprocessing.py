from explainer import parameters as param
import numpy as np
import logging

logger = logging.getLogger(__name__)


def mapping(original_num_feat, data):
    """
    :param original_num_feat: list. A list with the name of each feature used in the LSTM model
    :param data: numpy array. The data one needs to preprocess
    :return:
    """

    # Map the original data to a form more suitable for LIME

    mapping_index = {'num_to_num':
                         [(original_num_feat.index(i), e) for e, i in enumerate(param.mapping_features['num_to_num'])],
                     'num_to_cat_encoded':
                         [(param.encoded_num2cat_idx[i], e)
                          for e, i in enumerate(param.mapping_features['num_to_cat_encoded'])],
                     'num_to_cat':
                         [(original_num_feat.index(i), param.new_feature_names['Categorical_features'].index(i))
                          for i in param.mapping_features['num_to_cat']]
                     }

    # Defining the size of the new numerical and categorical inputs
    new_n_inputs = len(param.new_feature_names['Numerical_features'])

    new_c_inputs = len(param.new_feature_names['Categorical_features'])

    n_samples, n_sequences, n_inputs = data['Numerical_features'].shape

    # Defining a new dictionary with the modified features
    shaped_train_data = {'Numerical_features': np.full((n_samples, n_sequences, new_n_inputs), -1, dtype='float32'),
                         'Categorical_features': np.zeros((n_samples, n_sequences, new_c_inputs)),
                         'Fixed_features': data['Fixed_features'],
                         'Lengths_features': data['Lengths_features']}

    # Filling the new numerical features and defining the map between new and old indices
    for (i, j) in mapping_index['num_to_num']:

        shaped_train_data['Numerical_features'][:, :, j] = data['Numerical_features'][:, :, i]

    for (i, j) in mapping_index['num_to_cat_encoded']:

        row, col, val = np.where(data['Numerical_features'][:, :, i[0]:i[1]].astype('int') == 1)

        # Shifting the value of week days, month, event_env and device by 1 so that 0 is the padding value

        shaped_train_data['Categorical_features'][row, col, j] = val + 1

    for (i, j) in mapping_index['num_to_cat']:

        if i == param.lstm_features.index(param.mapping_features['qty']):

            for k in range(data['Lengths_features'].shape[0]):

                neg_idx = np.nonzero(data['Numerical_features'][k, :data['Lengths_features'][k], i] < 0)

                shaped_train_data['Categorical_features'][k, :, j] = data['Numerical_features'][k, :, i] + 1

                shaped_train_data['Categorical_features'][k, neg_idx, j] = data['Numerical_features'][k, neg_idx, i] - 2

        else:

            # Shifting the value of product_qty, visit_nb_pages, visit_unique_pages, product_rating,
            # product_number_of_ratings by 1 so that 0 is the padding value

            shaped_train_data['Categorical_features'][:, :, j] = data['Numerical_features'][:, :, i] + 1

    shaped_train_data['Categorical_features'][:, :, -3:] = data['Categorical_features'][:, :, 0:3]

    return shaped_train_data


def reshaping(data):
    """
    :param data: numpy array. The data one needs to preprocess
    :return:
    """

    for i in ['Numerical_features', 'Categorical_features']:

        n_samples, n_sequences, n_inputs = data[i].shape

        data[i] = data[i].reshape(n_samples, int(n_sequences * n_inputs))

    lime_data = np.concatenate((data['Numerical_features'],
                                data['Categorical_features'],
                                data['Fixed_features']), axis=1)

    return lime_data


def preprocessing(data, length_max):
    """
    :param data: numpy array. The data to preprocess
    :param training_sess: bool. True if the data are for training the explainer, False if are the instances to explain
    :return:
    """

    # Create the new data as expected by LIME. Some of the numerical features are also transformed to categorical
    # features (for the LIME interpreter, not for the network)

    pad_data = {'Numerical_features': np.pad(data['Numerical_features'],
                                             ((0, 0),
                                              (0, length_max - data['Numerical_features'].shape[1]),
                                              (0, 0)),
                                             constant_values=-1.
                                             ),
                'Categorical_features': np.pad(data['Categorical_features'],
                                               ((0, 0),
                                                (0, length_max - data['Numerical_features'].shape[1]),
                                                (0, 0)),
                                               constant_values=0
                                               ),
                'Lengths_features': data['Lengths_features'],
                'Fixed_features': data['Fixed_features'],
                'Ids': data['Ids']}

    shaped_train_data = mapping(param.lstm_features, pad_data)

    # Reshaping the input of the LSTM from 3d to 2d

    lime_data = reshaping(shaped_train_data)

    return lime_data
