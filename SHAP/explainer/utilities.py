from google.cloud import bigquery, storage
from google_pandas_load import Loader
import tensorflow as tf
import numpy as np
import numpy.ma as ma
import explainer.parameters as param
from explainer.preprocessing import preprocessing
import logging
import dill
import pandas as pd
import collections

logger = logging.getLogger(__name__)


def gplLoader(project_id, dataset, bucket_name, directory, path):

    bq_client = bigquery.Client(project=project_id, credentials=None)

    dataset_ref = bigquery.dataset.DatasetReference(project=project_id, dataset_id=dataset)

    gs_client = storage.Client(project=project_id, credentials=None)

    bucket = storage.bucket.Bucket(client=gs_client, name=bucket_name)

    gpl = Loader(bq_client=bq_client,
                 dataset_ref=dataset_ref,
                 bucket=bucket,
                 gs_dir_path_in_bucket=directory + path,
                 local_dir_path='/tmp')

    return gpl


def file_loader(iterator):
    """
    :param iterator: tf.one_shot_iterator. An iterator for the different features
    :param n_seq: int. The sequence length for which we want to build an explainer
    :return:
    """
    # The function loads a dictionary defined in the tf iterator into a list

    with tf.Session() as sess:

        while True:

            try:

                feat, lbl = sess.run(iterator)

                yield feat, lbl

            except tf.errors.OutOfRangeError:

                break


def train_stats_generator(train_iterator, length_max):

    iter_shap_data = map(lambda x: preprocessing(x[0], length_max), file_loader(train_iterator))

    categorical_index = [i for i in range(len(param.new_feature_names['Numerical_features']),
                                          len(param.new_feature_names['Numerical_features']) +
                                          len(param.new_feature_names['Categorical_features']))]

    logger.info('Defining the training set statistics')

    online_scaler = OnlineStats_3d(categorical_index, [-1., 0.0])

    for i in iter_shap_data:

        training_data_stats, feature_counter = online_scaler.fit(i)

    training_data_stats['stds'] = np.sqrt(training_data_stats['stds'])

    for step in range(length_max):

        for feature in categorical_index:

            training_data_stats['feature_values'][step][feature],\
            training_data_stats['feature_frequencies'][step][feature] = map(list,
                                                                            zip(*(sorted(feature_counter[step][
                                                                                             feature].items()))))

    return training_data_stats


def train_data_generator(training_data_stats, samples):

    logger.info('Loading product_lvl subset from storage')

    gpl = gplLoader(param.gpp['project_id'], param.gpp['dataset_id'], param.gpp['bucket_name'],
                    param.gpp['directory'], param.gpp['output_path'])

    file_subset = 'product_lvl_subset'

    gpl.load(source='gs', destination='local', data_name=file_subset, delete_in_gs=False)

    with open('/tmp/' + file_subset, 'rb') as f:

        product_lvl_subset = dill.load(f)

    # Generating a training dataset for SHAP based on the characteristics of the full training data

    logger.info('Generating a dummy training set for the explainer')

    shap_training = np.random.normal(loc=training_data_stats['means'].reshape(1,
                                                                              training_data_stats['means'].shape[0],
                                                                              training_data_stats['means'].shape[1]),
                                     scale=training_data_stats['stds'].reshape(1,
                                                                               training_data_stats['stds'].shape[0],
                                                                               training_data_stats['stds'].shape[1]),
                                     size=(samples,
                                           training_data_stats['means'].shape[0],
                                           training_data_stats['means'].shape[1]))

    # Fixing the features that should be positive by applying the abs

    positive_idx = [0, 1, 4]

    shap_training[:, :, positive_idx] = np.abs(shap_training[:, :, positive_idx])

    for i in training_data_stats['feature_values'].keys():

        for j in training_data_stats['feature_values'][i].keys():

            shap_training[:, i, j] = np.random.choice(training_data_stats['feature_values'][i][j],
                                                      size=samples,
                                                      p=training_data_stats['feature_frequencies'][i][j] / np.sum(
                                                          training_data_stats['feature_frequencies'][i][j]))

    # Resampling product_lvl and nb_unique_pages so that it is consistent

    resampling_idx = [14, 10]

    shap_training = resampling(shap_training, resampling_idx,
                               training_data_stats['feature_values'],
                               training_data_stats['feature_frequencies'],
                               product_lvl_subset)

    return shap_training


def feature_names_creation(length):

    lime_feat_names = {i: [] for i in ['Numerical_features', 'Categorical_features']}

    for i in ['Numerical_features', 'Categorical_features']:
        lime_feat_names[i] = [y + '_t' + str(x) for x in range(length)
                              for y in param.new_feature_names[i]]

    # lime_feat_names['Lengths_features'] = ['sequence_length']

    lime_feat_names['Fixed_features'] = ['inactivity_time']

    return lime_feat_names


def categorical_names_creation(lime_feat_names, length):

    gpl = gplLoader(param.gpp['project_id'], param.gpp['dataset_id'], param.gpp['bucket_name'],
                    param.gpp['directory'], param.gpp['files_path'])

    for i in param.mapping_features['cat_to_cat']:

        file_name = 'metadata_' + i + '.csv'

        gpl.load(source='gs', destination='local', data_name=file_name, delete_in_gs=False)

        df_temp = pd.read_csv('/tmp/' + file_name, sep='\t', engine='python')

        param.label_category[i] = list(df_temp.Categorical_values.values)

    categorical_names = {len(param.new_feature_names['Numerical_features']) * length +
                         lime_feat_names['Categorical_features'].index(i): param.label_category[j]
                         for j in param.mapping_features['num_to_cat_encoded'] +
                         param.mapping_features['cat_to_cat']
                         for i in lime_feat_names['Categorical_features'] if j in i}

    return categorical_names


def product_lvl_subset_creation(data, product_lvl_subset):

    lvl_1 = np.unique(data['Categorical_features'][:, :, -3])

    for i in lvl_1:

        if i not in product_lvl_subset.keys():

            product_lvl_subset[i] = {}

    for i in product_lvl_subset.keys():

        idx_lvl_1 = np.nonzero(data['Categorical_features'][:, :, -3] == i)

        for k in np.unique(data['Categorical_features'][idx_lvl_1][:, -2]):

            if k not in product_lvl_subset[i].keys():

                product_lvl_subset[i][k] = []

            idx_lvl_2 = np.nonzero(data['Categorical_features'][idx_lvl_1][:, -2] == k)

            product_lvl_subset[i][k].extend([j for j in np.unique(
                data['Categorical_features'][idx_lvl_1][idx_lvl_2][:, -1])
                                             if j not in product_lvl_subset[i][k]])

    return product_lvl_subset


class OnlineStats_3d:

    def __init__(self,
                 cat_idx,
                 padding_values):

        self.cat_idx = cat_idx

        self.padding_values = padding_values

        self.count = np.zeros((1, 3))

    def fit(self, data):

        if len(data.shape) == 1:

            data.reshape(1, -1)

        # n_feat = data.shape[1]

        _, n_dims, n_feat = data.shape

        if not hasattr(self, 'total_count'):

            self.total_count = np.zeros((n_dims, n_feat))

            self.output_dict = {'means': np.zeros((n_dims, n_feat)),
                                'mins': np.full((n_dims, n_feat), np.inf),
                                'maxs': np.full((n_dims, n_feat), -np.inf),
                                'stds': np.zeros((n_dims, n_feat)),
                                'feature_values': {i: {j: [] for j in self.cat_idx} for i in range(n_dims)},
                                'feature_frequencies': {i: {j: [] for j in self.cat_idx} for i in range(n_dims)}
                                }

            self.feature_counter = {i: {j: collections.Counter() for j in self.cat_idx} for i in range(n_dims)}

        mask = data == self.padding_values[0]

        masked_batch = ma.array(data, mask=mask)

        current_mean = np.mean(masked_batch, axis=0)

        current_var = np.var(masked_batch, axis=0)

        sample_count = np.sum(~mask, axis=0)

        num_mean = self.total_count * self.output_dict['means'] + sample_count * current_mean.filled(fill_value=0.)

        weight_var = self.total_count * self.output_dict['stds'] + sample_count * current_var.filled(fill_value=0.)

        weight_mean = self.total_count * sample_count * (self.output_dict['means'] -
                                                         current_mean.filled(fill_value=0.)) ** 2

        self.total_count += sample_count

        np.divide(num_mean, self.total_count, out=self.output_dict['means'], where=self.total_count != 0)

        np.divide(weight_var * self.total_count + weight_mean, self.total_count ** 2,
                  out=self.output_dict['stds'], where=self.total_count != 0)

        self.output_dict['mins'] = np.fmin(self.output_dict['mins'], np.min(data, axis=0))

        self.output_dict['maxs'] = np.fmax(self.output_dict['maxs'], np.max(data, axis=0))

        for step in range(n_dims):

            for feature in self.cat_idx:

                column = data[:, step, feature]

                self.feature_counter[step][feature].update(column)

                del self.feature_counter[step][feature][self.padding_values[1]]

        return self.output_dict, self.feature_counter


def resampling(data, resampling_features, feature_values, feature_frequencies, product_lvl_subset):
    """
    :param data: numpy array. The perturbed data
    :return:
    """

    idx = resampling_features[0]

    idx_pag = resampling_features[1]

    for i in range(data.shape[0]):

        for j in range(data.shape[1]):

            if data[i, j, idx] not in product_lvl_subset.keys():

                data[i, j, idx] = np.random.choice(feature_values[j][idx],
                                                   p=feature_frequencies[j][idx])

            if data[i, j, idx + 1] not in product_lvl_subset[data[i, j, idx]].keys():

                subset = [k for k in feature_values[j][idx + 1]
                          if k in product_lvl_subset[int(data[i, j, idx])].keys()]

                proba = [feature_frequencies[j][idx + 1][feature_values[j][idx + 1].index(k)] for k in subset]

                data[i, j, idx + 1] = np.random.choice(subset,
                                                       p=[k/sum(proba) for k in proba])

            if data[i, j, idx + 2] not in product_lvl_subset[int(data[i, j, idx])][int(data[i, j, idx + 1])]:

                subset = [k for k in feature_values[j][idx + 2]
                          if k in product_lvl_subset[int(data[i, j, idx])][int(data[i, j, idx + 1])]]

                if len(subset) == 0:

                    data[i, j, idx + 2] = np.random.choice(product_lvl_subset[int(data[i, j,
                                                                                       idx])][int(data[i, j, idx + 1])])

                else:

                    proba = [feature_frequencies[j][idx + 2][feature_values[j][idx + 2].index(k)] for k in subset]

                    data[i, j, idx + 2] = np.random.choice(subset,
                                                           p=[k/sum(proba) for k in proba])

            # Resampling the variable visit_u_pages in accordance to visit_n_pages

            if data[i, j, idx_pag] == 1.:

                data[i, j, idx_pag + 1] = 1

            elif data[i, j, idx_pag + 1] > data[i, j, idx_pag] or data[i, j, idx_pag + 1] == 1.:

                data[i, j, idx_pag + 1] = np.random.choice(np.arange(2, int(data[i, j, idx_pag]) + 1))

    return data
