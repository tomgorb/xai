import tensorflow as tf
import numpy as np
from purchase_probability.train_eval_predict import model_fn
from purchase_probability.train_eval_predict import bucketing
from purchase_probability.train_eval_predict import padd_and_batch
from purchase_probability import parameters
from sklearn.preprocessing import OneHotEncoder
import copy
import logging
import shap

logger = logging.getLogger(__name__)


class ShapPredictor:

    def __init__(self, cloud_parameters, feature_names, lstm_features, mapping_features, encoded_features, length_max):

        self.cloud_parameters = cloud_parameters
        self.feature_names = feature_names
        self.lstm_features = lstm_features
        self.mapping_features = mapping_features
        self.encoded_features = encoded_features

        self.length_max = length_max

        self.internal_memory = []

        self.shap_explainers = []

        self.shap_values = []

    def update_memory(self, data):

        self.internal_memory.append(data)

    def clean_memory(self):

        self.internal_memory = []

    def prediction_reshaping(self, data):
        """
        :param data: numpy array. The perturbed data
        :return:
        """

        # Defining the number of numerical and categorical features (new version), the number of sequence and the
        # number of samples

        original_inputs = {'Numerical_features': len(self.feature_names['Numerical_features']),
                           'Categorical_features': len(self.feature_names['Categorical_features']),
                           'lstm_inputs': parameters.model_params['N_INPUT']}

        mapping_index = {'num_to_num':
                             [(self.lstm_features.index(i), e) for e, i in
                              enumerate(self.mapping_features['num_to_num'])],
                         'num_to_cat_encoded': [(self.encoded_features[i], e)
                                                for e, i in enumerate(self.mapping_features[
                                                                          'num_to_cat_encoded'])],
                         'num_to_cat': [(self.lstm_features.index(i), self.feature_names[
                             'Categorical_features'].index(i))
                                        for i in self.mapping_features['num_to_cat']]
                         }

        n_samples = data.shape[0]

        data = data.reshape(n_samples, 1, -1)

        num_feat = data[:, :, 0:original_inputs['Numerical_features']]

        cat_feat = data[:, :, original_inputs['Numerical_features']:original_inputs['Numerical_features'] +
                                                                    original_inputs['Categorical_features']]

        fix_feat = data[:, 0, -1]

        # Shifting back the categorical features when needed

        shift_idx = [self.feature_names['Categorical_features'].index(i)
                     for i in self.feature_names['Categorical_features'] if i not in self.mapping_features['cat_to_cat']]

        prod_qty_idx = self.feature_names['Categorical_features'].index(self.mapping_features['qty'])

        cat_feat[:, :, shift_idx] -= 1

        cat_feat[:, :, prod_qty_idx][np.nonzero(cat_feat[:, :, prod_qty_idx] < 0)] += 2

        # Defining the array for the numerical data suitable for the LSTM network

        numerical_data = np.zeros((n_samples, 1, original_inputs['lstm_inputs']))

        enc = OneHotEncoder(handle_unknown='ignore')

        # Converting the categorical features encoded to numerical features by a one hot encoder

        for i in mapping_index['num_to_cat_encoded']:
            enc.fit(np.arange(i[0][1] - i[0][0]).reshape(-1, 1))

            temp = enc.transform(cat_feat[:, :, i[1]].reshape(-1, 1)).toarray()

            numerical_data[:, :, i[0][0]:i[0][1]] = temp.reshape(n_samples, 1, i[0][1] - i[0][0])

            # Fixing the padding value

            numerical_data[np.nonzero(np.sum(numerical_data[:, :, i[0][0]:i[0][1]], axis=-1) == 0.)] = -1.

        # Taking the categorical features for the LSTM

        categorical_data = cat_feat[:, :, -3:]

        # Taking the numerical features from the LIME shape to the LSTM shape

        for i in mapping_index['num_to_num']:
            numerical_data[:, :, i[0]] = num_feat[:, :, i[1]]

        # Taking the categorical features from LIME to the numerical features for the LSTM

        for i in mapping_index['num_to_cat']:
            numerical_data[:, :, i[0]] = cat_feat[:, :, i[1]]

        # Defining the dictionary for the input of the LSTM model

        sample_data = {'Ids': np.transpose([np.arange(n_samples).astype('str')]),
                       'Numerical_features': numerical_data.astype('float32'),
                       'Categorical_features': categorical_data.astype('int64'),
                       'Fixed_features': np.abs(fix_feat, dtype='float32').reshape(-1, 1),
                       'Lengths_features': np.ones((n_samples, 1), dtype='int32'),
                       'Labels': (np.repeat(np.array([[1., 0.]]), n_samples, axis=0)).astype('float32')
                       }

        return sample_data

    def predict_fn(self, sample_data):
        """
        :param sample_data: numpy array. The perturbed data from the instance to explain
        :return:
        """

        data = copy.deepcopy(sample_data)

        # data = custom_utils.prediction_reshaping(data)

        data = self.prediction_reshaping(data)

        extended_data = data.copy()

        if self.internal_memory:

            copy_memory = copy.deepcopy(self.internal_memory)

            # reshaped_memory = [custom_utils.prediction_reshaping(copy_memory[i]) for i in range(len(copy_memory))]

            reshaped_memory = [self.prediction_reshaping(copy_memory[i]) for i in
                               range(len(copy_memory))]

            for c in ['Numerical_features', 'Categorical_features']:
                extended_data[c] = np.concatenate([extended_data[c]] + [np.broadcast_to(reshaped_memory[i][c],
                                                                                        extended_data[c].shape)
                                                                        for i in range(len(self.internal_memory))],
                                                  axis=1)

            extended_data['Lengths_features'] = np.full((extended_data['Numerical_features'].shape[0], 1),
                                                        fill_value=extended_data['Numerical_features'].shape[1],
                                                        dtype='int32')

        tf.reset_default_graph()

        logger.info('Constructing the dataset')

        with tf.name_scope('Dataset_constructor'):

            def gen_pred():
                for i in range(extended_data['Ids'].shape[0]):
                    yield extended_data['Ids'][i], \
                          extended_data['Numerical_features'][i, :extended_data['Lengths_features'][i, 0], :], \
                          extended_data['Categorical_features'][i, :extended_data['Lengths_features'][i, 0], :], \
                          extended_data['Labels'][i], \
                          extended_data['Lengths_features'][i, 0], \
                          extended_data['Fixed_features'][i]

            dataset = tf.data.Dataset.from_generator(gen_pred,
                                                     output_types=(tf.string, tf.float32, tf.int64, tf.float32,
                                                                   tf.int32, tf.float32),
                                                     output_shapes=((None,),
                                                                    (None, parameters.model_params['N_INPUT']),
                                                                    (None,
                                                                     len(parameters.model_params[
                                                                             'vocabulary_sizes'].keys())
                                                                     ),
                                                                    (None,),
                                                                    (),
                                                                    (None,)))

            window_size = 10 * parameters.hp_params['batch_size']
            dataset = dataset.apply(tf.contrib.data.group_by_window(
                key_func=lambda mi, x_num, x_cat, y, z, fixed: bucketing(z, parameters.bucket_boundaries),
                reduce_func=lambda _, x: padd_and_batch(x, parameters.hp_params['batch_size']),
                window_size=window_size))

            # Prefetch dataset (working with tf.__version__ >= 1.4)
            dataset = dataset.prefetch(parameters.buffer_size)

            iterator = dataset.make_one_shot_iterator()

            # Get next elements from the iterator for train and validation
            next_mi, next_num_example, next_cat_example, next_label, next_length, next_fixed, = iterator.get_next()

            features_dict = {'Numerical_features': next_num_example,
                             'Categorical_features': next_cat_example,
                             'Lengths_features': next_length,
                             'Fixed_features': next_fixed,
                             'Ids': next_mi
                             }

        parameters_predictor = parameters.hp_params

        parameters_predictor.update({'bucket_name': self.cloud_parameters['bucket_name'],
                                     'directory': self.cloud_parameters['directory'] +
                                                  self.cloud_parameters['model_path']})

        # parameters_predictor.update({'bucket_name': param.gpp['bucket_name'],
        #                              'directory': param.gpp['directory'] + param.gpp['model_path']})

        # Calling the model function
        predictions_est = model_fn(features_dict, next_label, mode=tf.estimator.ModeKeys.PREDICT,
                                   params=parameters_predictor)

        # Manually load the latest checkpoint
        saver = tf.train.Saver()

        with tf.Session() as sess:

            logger.info('Loading checkpoint')

            # ckp_path = 'gs://' + param.gpp['bucket_name'] + '/' + param.gpp['directory'] + param.gpp['model_path']

            ckp_path = 'gs://' + self.cloud_parameters['bucket_name'] + '/' + self.cloud_parameters['directory'] \
                       + self.cloud_parameters['model_path']

            logger.info(ckp_path)

            ckpt = tf.train.get_checkpoint_state('gs://' + self.cloud_parameters['bucket_name'] + '/' +
                                                 self.cloud_parameters['directory']
                                                 + self.cloud_parameters['model_path'])

            saver.restore(sess, ckpt.model_checkpoint_path)

            predictions = predictions_est.predictions

            # Loop through the batches and store predictions and labels
            prediction_values = []

            label_values = []

            # Added ids list to be filled later
            logger.info('Constructing output')

            ids = []

            compl_prediction_values = []

            while True:

                try:

                    preds, lbls = sess.run([predictions, next_label])

                    # Filling ids
                    ids = np.append(ids, [x for x in preds['Ids']])

                    prediction_values = np.append(prediction_values, [x[1] for x in preds['Predictions']])

                    compl_prediction_values = np.append(compl_prediction_values, [x[0] for x in preds['Predictions']])

                    label_values = np.append(label_values, [x[1] for x in lbls])

                except tf.errors.OutOfRangeError:

                    break

            logger.info('Explanation concluded')

        return np.transpose(np.array([compl_prediction_values, prediction_values]))

    def explainer_initializer(self, data):

        for i in range(self.length_max):

            if i > 0:

                self.update_memory(data[:, i - 1, :])

            self.shap_explainers.append(shap.KernelExplainer(self.predict_fn,
                                                             data[:, i, :], link="logit"))

        print(len(self.shap_explainers))

    def explain(self, instance, instance_length):

        self.clean_memory()

        for i in range(instance_length):

            print(i)

            if i > 0:

                self.update_memory(instance[:, i - 1, :])

            self.shap_values.append(self.shap_explainers[i].shap_values(instance[:, i, :])[0])
