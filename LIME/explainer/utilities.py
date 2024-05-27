from google.cloud import bigquery, storage
from google_pandas_load import Loader
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import explainer.parameters as param
import logging
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


def custom_plot(explainer):
    exp = explainer.as_list(label=1)
    fig = plt.figure(figsize=(25., 25.))
    vals = [x[1] for x in exp]
    names = [x[0] for x in exp]
    vals.reverse()
    names.reverse()
    colors = ['green' if x > 0 else 'red' for x in vals]
    pos = np.arange(len(exp)) + .5
    plt.barh(pos, vals, align='center', color=colors)
    plt.yticks(pos, names, rotation=45, fontsize=12)
    title = 'Local explanation for class %s' % explainer.class_names[1]
    plt.title(title, fontsize=25)
    return fig


def plotting(explanation, n_seq=None):

    def splitting(explainer, lbl=1):
        n_nin = len(param.new_feature_names['Numerical_features'])
        n_cat = len(param.new_feature_names['Categorical_features'])

        logger.info('Reshaping the instance')
        # Splitting array for numerical, categorical and length features
        split_map = np.split(sorted(explainer.as_map()[lbl]), [n_seq*n_nin, n_seq*(n_nin + n_cat)], axis=0)

        # Summing up along the sequences

        total_n_feat = np.sum(split_map[0][:, 1].reshape(n_seq, n_nin), axis=0)
        total_c_feat = np.sum(split_map[1][:, 1].reshape(n_seq, n_cat), axis=0)
        total_weight = np.concatenate([total_n_feat, total_c_feat, split_map[2][:, 1]])

        return total_weight

    def window_plot(ax, vals, names, lbl, class_names, n_f, inverted_y, sorted_weights=None):

        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(vals[0:n_f])) + .5
        ax.barh(pos, vals, align='center', color=colors)
        rot = 45
        if inverted_y:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            rot = -rot
        ax.set_yticks(pos)
        ax.set_yticklabels(names, rotation=rot, fontsize=10)
        title = 'Explanation for the class {}'.format(class_names[lbl])
        ax.set_title(title, fontsize=12)

    # Box with informations on the LIME explainer results and on the instance explained

    def summary_box(ax, explainer):
        ax.tick_params(axis='both', left=False, top=False, right=False,
                       bottom=False, labelleft=False, labeltop=False,
                       labelright=False, labelbottom=False)
        text = 'LIME stats\n\nDay: {}\n\nLoc_pred: {:.2};  Bias: {:.2};  Real_pred: {:.2}\n\n' \
               'R2: {:.2}\n\nN_samples: {}\n\nN_seq: {}\n\nModel: linear;  Regressor: default; Kernel: default;  ' \
               'Width: default'.format(param.date, explainer.local_pred[0], explainer.intercept[1],
                                       explainer.predict_proba[1], explainer.score, param.n_samples, n_seq)
        ax.text(0.02, 0.20, text)

    # Standard plot with the results of the LIME explainer

    def full_plot(ax, explainer, lbl=1, n_f=10, inverted_y=False):
        exp = explainer.as_list(label=lbl)
        vals = [x[1] for x in exp[0:n_f]]
        names = [x[0] for x in exp[0:n_f]]
        window_plot(ax, vals, names, lbl, explainer.class_names, n_f, inverted_y)

    # Standard plot with weights summed up along the sequences

    def feat_plot(ax, explainer, lbl=1, n_f=10, inverted_y=False):

        total_weight = splitting(explainer, lbl=1)

        # Sorting the new weights

        sorted_weights = sorted([(i, np.abs(k)) for i, k in enumerate(total_weight)],
                               key=lambda weight: weight[1], reverse=True)

        # Creating the new features names

        feat_names = param.new_feature_names['Numerical_features'] + param.new_feature_names['Categorical_features'] +\
                     ['inactivity', 'sequence']
        vals = [total_weight[x[0]] for x in sorted_weights[0:n_f]]
        names = [feat_names[x[0]] for x in sorted_weights[0:n_f]]
        window_plot(ax, vals, names, lbl, explainer.class_names, n_f, inverted_y, sorted_weights)

    def feat_agg_plot(ax, explainer, lbl=1, n_f=10, inverted_y=False):

        total_weight = splitting(explainer, lbl=1)
        new_weight = []
        new_weight.append(sum(total_weight[0:7]))
        new_weight.append(sum(total_weight[7:19]))
        new_weight.append(sum(total_weight[19:29]))
        new_weight.append(sum(total_weight[29:35]))
        new_weight.extend(total_weight[35:43])
        new_weight.append(sum(total_weight[43:45]))
        new_weight.extend(total_weight[45:])
        feat_names_t = ['week_days', 'months', 'event_env', 'devices'] + param.num_feat_rem[0:8] + ['geo_loc'] +\
                       [param.num_feat_rem[-1]]
        feat_names = feat_names_t + param.cat_feat_names + ['inactivity', 'sequence']
        sorted_weights = sorted([(i, np.abs(k)) for i, k in enumerate(new_weight)],
                                key=lambda weight: weight[1], reverse=True)
        vals = [new_weight[x[0]] for x in sorted_weights[0:n_f]]
        names = [feat_names[x[0]] for x in sorted_weights[0:n_f]]
        window_plot(ax, vals, names, lbl, explainer.class_names, n_f, inverted_y, sorted_weights)

    fig = plt.figure(figsize=(18, 9))

    # ax1 = plt.subplot2grid((2, 7), (0, 0), colspan=3)
    # ax2 = plt.subplot2grid((2, 7), (0, 3), colspan=3)
    # ax3 = plt.subplot2grid((2, 7), (1, 0), colspan=3)
    # ax4 = plt.subplot2grid((2, 7), (1, 3), colspan=3)

    ax1 = plt.subplot2grid((2, 7), (0, 0), colspan=3, rowspan=2)
    ax2 = plt.subplot2grid((2, 7), (0, 3), colspan=3)

    full_plot(ax1, explanation, lbl=1, n_f=10)
    # feat_plot(ax2, explanation, lbl=1)
    # feat_agg_plot(ax3, explanation, lbl=1)
    summary_box(ax2, explanation)

    # fig.suptitle(
    #     'LIME explainer for Master_id {} on day {}'.format(pred_element['Ids'][explained_instance][0].decode(),
    #                                                        date))

    plt.tight_layout()

    return fig


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


class OnlineStats:

    def __init__(self,
                 cat_idx,
                 padding_values):

        self.cat_idx = cat_idx

        self.padding_values = padding_values

        self.count = np.zeros((1, 3))

    def fit(self, data):

        if len(data.shape) == 1:

            data.reshape(1, -1)

        n_feat = data.shape[1]

        if not hasattr(self, 'total_count'):

            self.total_count = np.zeros(n_feat)

            self.output_dict = {'means': np.zeros(n_feat),
                                'mins': np.full(n_feat, np.inf),
                                'maxs': np.full(n_feat, -np.inf),
                                'stds': np.zeros(n_feat),
                                'feature_values': {i: [] for i in self.cat_idx},
                                'feature_frequencies': {i: [] for i in self.cat_idx}
                                }

            self.feature_counter = {i: collections.Counter() for i in self.cat_idx}

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

        for feature in self.cat_idx:

            column = data[:, feature]

            self.feature_counter[feature].update(column)

            del self.feature_counter[feature][self.padding_values[1]]

        return self.output_dict, self.feature_counter
