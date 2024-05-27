import lime.lime_tabular
import logging
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances
import copy
import warnings

logger = logging.getLogger(__name__)


class LimeCustomExplainer(lime.lime_tabular.LimeTabularExplainer):
    def __init__(self,
                 training_data,
                 mapping_index=None,
                 mapping_features=None,
                 length_max=None,
                 product_subset=None,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 resampling_features=None,
                 positive_features=None,
                 original_inputs=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 random_state=None,
                 training_data_stats=None,
                 sample_around_instance=False,
                 discretize_continuous=False):

        super(LimeCustomExplainer, self).__init__(
            training_data,
            mode=mode,
            training_labels=training_labels,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_names=categorical_names,
            kernel_width=kernel_width,
            kernel=kernel,
            verbose=verbose,
            class_names=class_names,
            feature_selection=feature_selection,
            discretize_continuous=discretize_continuous,
            discretizer='quartile',
            random_state=random_state,
            training_data_stats=training_data_stats,
            sample_around_instance=sample_around_instance)

        self.mapping_index = mapping_index
        self.mapping_features = mapping_features

        self.length_max = length_max

        self.product_lvl_subset = product_subset

        self.resampling_features = resampling_features

        self.positive_features = positive_features

        self.original_inputs = original_inputs

        if self.training_data_stats:

            self.scaler.mean_ = training_data_stats['means']

            self.scaler.scale_ = training_data_stats['stds']

            self.scaler.mean_[self.categorical_features] = 0

            self.scaler.scale_[self.categorical_features] = 1

    def resampling(self, data):
        """
        :param data: numpy array. The perturbed data
        :return:
        """

        idx = self.resampling_features['prod_lvl_idx']

        idx_pag = self.resampling_features['nb_pag_idx']

        for i in range(1, data.shape[0]):

            for j in idx:

                if data[i, j] not in self.product_lvl_subset.keys():

                    data[i, j] = np.random.choice(self.feature_values[j],
                                                  p=self.feature_frequencies[j])

                if data[i, j + 1] not in self.product_lvl_subset[data[i, j]].keys():

                    subset = [k for k in self.feature_values[j + 1]
                              if k in self.product_lvl_subset[int(data[i, j])].keys()]

                    proba = [self.feature_frequencies[j + 1][self.feature_values[j + 1].index(k)] for k in subset]

                    data[i, j + 1] = np.random.choice(subset,
                                                      p=[k/sum(proba) for k in proba])

                if data[i, j + 2] not in self.product_lvl_subset[int(data[i, j])][int(data[i, j + 1])]:

                    subset = [k for k in self.feature_values[j + 2]
                              if k in self.product_lvl_subset[int(data[i, j])][int(data[i, j + 1])]]

                    if len(subset) == 0:

                        data[i, j + 2] = np.random.choice(self.product_lvl_subset[int(data[i, j])][int(data[i, j + 1])])

                    else:

                        proba = [self.feature_frequencies[j + 2][self.feature_values[j + 2].index(k)] for k in subset]

                        data[i, j + 2] = np.random.choice(subset,
                                                          p=[k/sum(proba) for k in proba])

            # Resampling the variable visit_u_pages in accordance to visit_n_pages

            for j in idx_pag:

                if data[i, j] == 1.:

                    data[i, j + 1] = 1

                elif data[i, j + 1] > data[i, j] or data[i, j + 1] == 1.:

                    data[i, j + 1] = np.random.choice(np.arange(2, int(data[i, j]) + 1))

        return data

    def reshaping(self, data, instance_length):
        """
        :param data: numpy array. The perturbed data
        :return:
        """

        # Defining the number of numerical and categorical features (new version), the number of sequence and the
        # number of samples

        n_nin = self.original_inputs['Numerical_features']

        n_cat = self.original_inputs['Categorical_features']

        n_steps = int((data.shape[1] - 1) / (n_nin + n_cat))

        n_samples = data.shape[0]

        logger.info('Reshaping the instance')

        # Splitting array for numerical, categorical and length features

        split_lst = np.split(data, [n_steps * n_nin, n_steps * (n_nin + n_cat), n_steps * (n_nin + n_cat) + 1],
                             axis=1)

        num_feat = split_lst[0].reshape(n_samples, n_steps, n_nin)

        cat_feat = split_lst[1]

        fix_feat = split_lst[2]

        # Defining the array for the numerical data suitable for the LSTM network

        numerical_data = np.zeros((n_samples, n_steps, self.original_inputs['lstm_inputs']))

        enc = OneHotEncoder(handle_unknown='ignore')

        # Converting the categorical features encoded to numerical features by a one hot encoder

        for i in self.mapping_index['num_to_cat_encoded']:

            idx = [i[1] + j * n_cat for j in range(n_steps)]

            enc.fit(np.arange(i[0][1] - i[0][0]).reshape(-1, 1))

            temp = np.concatenate([enc.transform(cat_feat[:, j].reshape(-1, 1)).toarray() for j in idx], axis=1)

            numerical_data[:, :, i[0][0]:i[0][1]] = temp.reshape(n_samples, n_steps, i[0][1] - i[0][0])

            # Fixing the padding value

            numerical_data[np.nonzero(np.sum(numerical_data[:, :, i[0][0]:i[0][1]], axis=-1) == 0.)] = -1.

        # Reshaping the comb_feat to the 3d dimensions for the LSTM (needs to feed the actual dimensions as parameters)

        cat_feat = cat_feat.reshape(n_samples, n_steps, n_cat)

        # Taking the absolute value of the features that needs to be positive

        num_feat[:, :, [0, 1, 4]] = np.abs(num_feat[:, :, [0, 1, 4]])

        # Taking the categorical features for the LSTM

        categorical_data = cat_feat[:, :, -3:]

        # Taking the numerical features from the LIME shape to the LSTM shape

        for i in self.mapping_index['num_to_num']:

            numerical_data[:, :, i[0]] = num_feat[:, :, i[1]]

        # Taking the categorical features from LIME to the numerical features for the LSTM

        for i in self.mapping_index['num_to_cat']:

            numerical_data[:, :, i[0]] = cat_feat[:, :, i[1]]

        # Defining the dictionary for the input of the LSTM model

        sample_data = {'Ids': np.transpose([np.arange(n_samples).astype('str')]),
                       'Numerical_features': numerical_data.astype('float32'),
                       'Categorical_features': categorical_data.astype('int64'),
                       'Fixed_features': np.abs(fix_feat, dtype='float32'),
                       'Lengths_features': np.tile(np.array(instance_length).reshape(1, -1), (n_samples, 1)),
                       'Labels': (np.repeat(np.array([[1., 0.]]), n_samples, axis=0)).astype('float32')
                       }

        return sample_data

    def __custom_data_inverse(self,
                              data_row,
                              num_samples,
                              mask):

        num_cols = data_row.shape[0]

        instance_sample = data_row

        scale = self.scaler.scale_

        mean = self.scaler.mean_

        data = self.random_state.normal(0, 1, num_samples * num_cols
                                        ).reshape(num_samples, num_cols)

        data = np.array(data)

        if self.sample_around_instance:

            data = data * scale + instance_sample

        else:

            data = data * scale + mean

        # Fixing the features that should be positive by applying the abs

        data[:, self.positive_features] = np.abs(data[:, self.positive_features])

        data[:, mask] = -1.

        data[0] = data_row.copy()

        inverse = data.copy()

        for column in self.categorical_features:

            values = self.feature_values[column]

            freqs = self.feature_frequencies[column]

            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)

            inverse[:, column] = inverse_column if not mask[column] else np.zeros((num_samples,))

        # Applying resampling in data_inverse

        inverse = self.resampling(inverse)

        for column in self.categorical_features:

            inverse_column = inverse[:, column]

            binary_column = (inverse_column == data_row[column]).astype(int)

            binary_column[0] = 1

            inverse_column[0] = data[0, column]

            data[:, column] = binary_column if not mask[column] else 1

            inverse[:, column] = inverse_column if not mask[column] else 0

        inverse[0] = data_row
        # Shifting back the categorical features when needed
        shift_idx = [i for i in self.categorical_features if i not in
                     [e for e, j in enumerate(self.feature_names) if j in self.mapping_features['cat_to_cat']]]

        prod_qty_idx = [i for i in self.categorical_features if i in
                        [e for e, j in enumerate(self.feature_names) if j==self.mapping_features['qty']]]

        inverse[:, shift_idx] -= 1

        inverse[:, prod_qty_idx][np.nonzero(inverse[:, prod_qty_idx] < 0)] += 2

        return data, inverse

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         instance_length=None):
        """
        :param data_row: numpy array. The entry to explain
        :param predict_fn: function. The function returning the probabilities for each class (for classifiers)
        :param labels: tuple. Labels to be explained.
        :param top_labels: int. If not None, ignore labels and produce explanations for the top_labels with highest
        prediction probabilities.
        :param num_features: int. Maximum number of features present in explanation
        :param num_samples: int. Size of the neighborhood to learn the linear model
        :param distance_metric: sklearn.distance_metric. The distance metric to use for weights.
        :param model_regressor: sklearn regressor. Defaults to Ridge regression in LimeBase.
        :return:
        """
        mask = data_row == -1.

        categorical_mask = data_row[self.categorical_features] == 0

        mask[self.categorical_features] = categorical_mask

        data, inverse = self.__custom_data_inverse(data_row, num_samples, mask)

        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        # Fixing the scaled_data
        for i in range(mask.shape[0]):

            if i not in self.categorical_features and mask[i]:

                scaled_data[:, i] = -1.

        logger.info('Calling predict_fn')

        inverse = self.reshaping(inverse, instance_length)

        yss = predict_fn(inverse)

        distances = pairwise_distances(scaled_data, scaled_data[0].reshape(1, -1),
                                       metric=distance_metric).ravel()

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":

            if len(yss.shape) == 1:

                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")

            elif len(yss.shape) == 2:

                if self.class_names is None:

                    self.class_names = [str(x) for x in range(yss[0].shape[0])]

                else:

                    self.class_names = list(self.class_names)

                if not np.allclose(yss.sum(axis=1), 1.0):

                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)

            else:

                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:

            try:

                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1

            except AssertionError:

                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]

            min_y = min(yss)

            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)

        if feature_names is None:

            feature_names = [str(x) for x in range(data_row.shape[0])]

        values = self.convert_and_round(data_row)

        feature_indexes = None

        for i in self.categorical_features:

            if self.discretizer is not None and i in self.discretizer.lambdas:

                continue

            name = int(data_row[i])

            if i in self.categorical_names:

                name = self.categorical_names[i][name]

            feature_names[i] = '%s=%s' % (feature_names[i], name)

            values[i] = 'True'

        categorical_features = self.categorical_features

        discretized_feature_names = None

        if self.discretizer is not None:

            categorical_features = range(data.shape[1])

            discretized_instance = self.discretizer.discretize(data_row)

            discretized_feature_names = copy.deepcopy(feature_names)

            for f in self.discretizer.names:

                discretized_feature_names[f] = self.discretizer.names[f][int(discretized_instance[f])]

        domain_mapper = lime.lime_tabular.TableDomainMapper(feature_names,
                                                            values,
                                                            scaled_data[0],
                                                            categorical_features=categorical_features,
                                                            discretized_feature_names=discretized_feature_names,
                                                            feature_indexes=feature_indexes)

        ret_exp = lime.explanation.Explanation(domain_mapper,
                                               mode=self.mode,
                                               class_names=self.class_names)

        ret_exp.scaled_data = scaled_data

        if self.mode == "classification":

            ret_exp.predict_proba = yss[0]

            if top_labels:

                labels = np.argsort(yss[0])[-top_labels:]

                ret_exp.top_labels = list(labels)

                ret_exp.top_labels.reverse()

        else:

            ret_exp.predicted_value = predicted_value

            ret_exp.min_value = min_y

            ret_exp.max_value = max_y

            labels = [0]

        for label in labels:

            (ret_exp.intercept[label], ret_exp.local_exp[label], ret_exp.score, ret_exp.local_pred) =\
                self.base.explain_instance_with_data(scaled_data, yss, distances, label, num_features,
                                                     model_regressor=model_regressor,
                                                     feature_selection=self.feature_selection)

        if self.mode == "regression":

            ret_exp.intercept[1] = ret_exp.intercept[0]

            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]

            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp
