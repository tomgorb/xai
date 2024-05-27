import argparse, datetime, logging, sys
import tensorflow as tf
import numpy as np
import numpy.ma as ma
from explainer import parameters as param
from explainer.utilities import gplLoader, plotting
import explainer.utilities as custom_utils
from explainer.preprocessing import preprocessing
from explainer.custom_explainer import LimeCustomExplainer
import dill
import sklearn
import json
import os
from purchase_probability.train_eval_predict import create_input_fn, model_fn, bucketing, padd_and_batch
from purchase_probability import parameters


logger = logging.getLogger(__name__)

# Defining a custom LimeExplainer inheriting the class TabularExplainer. The main differences are the overriding of the
# explain instance function, and the output of the predict_fn. Also we define a new distance function.


def explain(explainer, instance, predict_fn, **kwargs):
    """
    :param explainer: CustomExplainer. A customized, trained, lime tabular explainer
    :param instance: numpy array. The sample for which one wants an explanation of the prediction
    :param predict_fn: function. The function calling the tf-model for prediction
    :param kwargs:
    :return:
    """
    explainer.random_state = sklearn.utils.check_random_state(param.rnd_seed)
    return explainer.explain_instance(instance, predict_fn, **kwargs)


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


def predict_fn(sample_data):
    """
    :param sample_data: numpy array. The perturbed data from the instance to explain
    :return:
    """

    # Saving the perturbed data for later analysis
    if param.dump_perturbed_data:

        logger.info('Saving the perturbed data')

        data_file = 'perturbed_data_kw_' + str(param.kernel_width)

        with open('/tmp/' + data_file, 'wb') as f:

            dill.dump(sample_data, f)

        gpl = gplLoader(param.gpp['project_id'], param.gpp['dataset_id'], param.gpp['bucket_name'],
                        param.gpp['directory'], param.gpp['output_path'])

        gpl.load(source='local', destination='gs', data_name=data_file)

    tf.reset_default_graph()

    logger.info('Constructing the dataset')

    with tf.name_scope('Dataset_constructor'):

        def gen_pred():

            for i in range(sample_data['Ids'].shape[0]):

                yield sample_data['Ids'][i],\
                      sample_data['Numerical_features'][i, :sample_data['Lengths_features'][i][0], :],\
                      sample_data['Categorical_features'][i, :sample_data['Lengths_features'][i][0], :], \
                      sample_data['Labels'][i], \
                      sample_data['Lengths_features'][i][0],\
                      sample_data['Fixed_features'][i]

        dataset = tf.data.Dataset.from_generator(gen_pred, output_types=(tf.string, tf.float32, tf.int64, tf.float32,
                                                                         tf.int32, tf.float32),
                                                 output_shapes=((None,),
                                                                (None, parameters.model_params['N_INPUT']),
                                                                (None,
                                                                 len(parameters.model_params['vocabulary_sizes'].keys())
                                                                 ),
                                                                (None, ),
                                                                (),
                                                                (None,)))

        window_size = 10 * parameters.hp_params['batch_size']
        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func=lambda id, x_num, x_cat, y, z, fixed: bucketing(z, parameters.bucket_boundaries),
            reduce_func=lambda _, x: padd_and_batch(x, parameters.hp_params['batch_size']),
            window_size=window_size))

        # Prefetch dataset (working with tf.__version__ >= 1.4)
        dataset = dataset.prefetch(parameters.buffer_size)

        iterator = dataset.make_one_shot_iterator()

        # Get next elements from the iterator for train and validation
        next_id, next_num_example, next_cat_example, next_label, next_length, next_fixed, = iterator.get_next()

        features_dict = {'Numerical_features': next_num_example,
                         'Categorical_features': next_cat_example,
                         'Lengths_features': next_length,
                         'Fixed_features': next_fixed,
                         'Ids': next_id
                         }

    parameters_predictor = parameters.hp_params

    parameters_predictor.update({'bucket_name': param.gpp['bucket_name'],
                                 'directory': param.gpp['directory'] + param.gpp['model_path']})

    # Calling the model function
    predictions_est = model_fn(features_dict, next_label, mode=tf.estimator.ModeKeys.PREDICT,
                               params=parameters_predictor)

    # Manually load the latest checkpoint
    saver = tf.train.Saver()

    with tf.Session() as sess:

        logger.info('Loading checkpoint')

        ckp_path = 'gs://' + param.gpp['bucket_name'] + '/' + param.gpp['directory'] + param.gpp['model_path']

        logger.info(ckp_path)

        ckpt = tf.train.get_checkpoint_state('gs://' + param.gpp['bucket_name'] + '/' + param.gpp['directory']
                                             + param.gpp['model_path'])

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


def explainer_fn(project_id, dataset_id, directory, path_data, bucket_name, date, usage):
    """
    :param project_id: str. The name of the project id
    :param dataset_id:str. The dataset to work with
    :param directory: str. The main folder in gs of the project
    :param path_data: str. The folder in gs with the training data
    :param bucket_name: str. The name of the bucket where to store data
    :param date: str. Date for which we want the explanation
    :param usage: str. Type of explanation to perform
    :return:
    """
    # The training files are necessary in LIME to evaluate mean values and variances in order then to scale the variable
    # we want to explain

    np.random.seed(param.rnd_seed)

    # Setting the google client
    gpl = gplLoader(project_id, dataset_id, bucket_name, directory, path_data)

    # Loading the rnn_params file (necessary for using the functions from ml_buy_probability)
    gpl.load(source='gs',
             destination='local',
             data_name='rnn_params.json',
             delete_in_gs=False)

    rnn_params_json = json.load(open(os.path.join('/tmp/', 'rnn_params.json')))

    parameters.model_params['N_INPUT'] = rnn_params_json['N_INPUT']

    logger.info('Checking existence of explainer in bucket')

    gpl = gplLoader(project_id, dataset_id, bucket_name, directory, param.gpp['output_path'])

    logger.info('gs://' + bucket_name + '/' + directory + param.gpp['output_path'])

    file_explainer = 'custom_explainer_kw_{}_V3'.format(param.kernel_width)

    if gpl.exist_in_gs(file_explainer):

        logger.info('Loading explainer from gs')

        gpl.load(source='gs', destination='local', data_name=file_explainer, delete_in_gs=False)

        with open('/tmp/'+file_explainer, 'rb') as f:

            explainer = dill.load(f)

    else:

        # Defining the first training day
        first_training_date = datetime.datetime.strptime(date, '%Y%m%d') - \
                              datetime.timedelta(days=parameters.training_history) + \
                              datetime.timedelta(days=parameters.forward_prediction)

        first_training_date = max(datetime.datetime.strptime(param.min_date, '%Y%m%d') +
                                  datetime.timedelta(days=parameters.forward_prediction),
                                  first_training_date)

        logger.info('First date of training considered: {}'.format(first_training_date.date()))

        # Creating a list with the date composing the training set
        nb_training_days = datetime.datetime.strptime(date, '%Y%m%d') - \
                           (datetime.datetime.strptime(param.min_date, '%Y%m%d') +
                            datetime.timedelta(days=parameters.forward_prediction - 1))

        logger.info('Number of training days: ' + str(nb_training_days))

        training_days = [(first_training_date + datetime.timedelta(days=x)).date().strftime("%Y%m%d")
                         for x in range(nb_training_days.days)]

        # Creating a list with the location of the training files
        train_files = ['gs://' + bucket_name + '/' + directory + path_data + '/Sequences_' + x + '_*.tfr'
                       for x in training_days[: int(len(training_days) * 0.8)]]

        logger.info('Number of training files: ' + str(len(train_files)))

        logger.info('Creating explainer')

        # Creating the input function for the train set

        train_input_fn = create_input_fn(path=train_files)

        train_iterator = train_input_fn(shuffle_data=True)

        logger.info('Checking existence of product_lvl_subset')

        gpl = gplLoader(param.gpp['project_id'], param.gpp['dataset_id'], param.gpp['bucket_name'],
                        param.gpp['directory'], param.gpp['output_path'])

        file_subset = 'product_lvl_subset'

        if not gpl.exist_in_gs(data_name=file_subset):

            logger.info('Creating product_lvl subset')

            product_lvl_subset = {}

            for (i, j) in file_loader(train_iterator):

                product_lvl_subset = custom_utils.product_lvl_subset_creation(i, product_lvl_subset)

            logger.info('Pushing product_lvl subset on storage')

            with open('/tmp/' + file_subset, 'wb') as f:

                dill.dump(product_lvl_subset, f)

            gpl.load(source='local', destination='gs', data_name=file_subset, delete_in_gs=False)

        else:

            logger.info('Loading product_lvl subset from storage')

            gpl.load(source='gs', destination='local', data_name=file_subset, delete_in_gs=False)

            with open('/tmp/' + file_subset, 'rb') as f:

                product_lvl_subset = dill.load(f)

        length_max = 0

        for (i, j) in file_loader(train_iterator):

            length_max = max(length_max, max(i['Lengths_features']))

        iter_lime_data = map(lambda x: preprocessing(x[0], length_max), file_loader(train_iterator))

        categorical_index = [i for i in range(len(param.new_feature_names['Numerical_features']) * length_max,
                                              (len(param.new_feature_names['Numerical_features']) +
                                               len(param.new_feature_names['Categorical_features'])) * length_max)]

        online_scaler = custom_utils.OnlineStats(categorical_index, [-1., 0.0])

        for i in iter_lime_data:

            training_data_stats, feature_counter = online_scaler.fit(i)

        training_data_stats['stds'] = np.sqrt(training_data_stats['stds'])

        for feature in categorical_index:

            training_data_stats['feature_values'][feature],\
                training_data_stats['feature_frequencies'][feature] = map(list,
                                                                          zip(*(sorted(feature_counter[
                                                                                           feature].items()))))

        # Defining the LIME explainer

        logger.info('Defining the explainer')

        logger.info('Creating feature names')

        lime_feat_names = custom_utils.feature_names_creation(length_max)

        logger.info('Creating categorical names')

        categorical_names = custom_utils.categorical_names_creation(lime_feat_names, length_max)

        lime_feature_names = np.concatenate([lime_feat_names[i] for i in ['Numerical_features',
                                                                          'Categorical_features',
                                                                          'Fixed_features']])

        mapping_index = {'num_to_num':
                             [(param.lstm_features.index(i), e) for e, i in
                              enumerate(param.mapping_features['num_to_num'])],
                         'num_to_cat_encoded':
                             [(param.encoded_num2cat_idx[i], e)
                              for e, i in enumerate(param.mapping_features['num_to_cat_encoded'])],
                         'num_to_cat':
                             [(param.lstm_features.index(i), param.new_feature_names['Categorical_features'].index(i))
                              for i in param.mapping_features['num_to_cat']]
                         }

        custom_kw = param.kernel_width * np.sqrt(lime_feature_names.shape[0])

        resampling_idx = {'prod_lvl_idx': [categorical_index[i]
                                           for i in range(param.new_feature_names[
                                                              'Categorical_features'].index('product_department'),
                                                          len(categorical_index),
                                                          len(param.new_feature_names['Categorical_features']))],
                          'nb_pag_idx': [categorical_index[i] for i in range(param.new_feature_names[
                                                                                 'Categorical_features'].index(
                              'visit_nb_pages'),
                              len(categorical_index),
                              len(param.new_feature_names['Categorical_features']))]}

        positive_idx = []

        for i in ['product_price', 'visit_duration', 'Delta_time']:

            positive_idx += [j for j in range(param.new_feature_names['Numerical_features'].index(i),
                                              len(param.new_feature_names['Numerical_features'] * length_max),
                                              len(param.new_feature_names['Numerical_features']))]

        n_inputs = {'Numerical_features': len(param.new_feature_names['Numerical_features']),
                    'Categorical_features': len(param.new_feature_names['Categorical_features']),
                    'lstm_inputs': parameters.model_params['N_INPUT']}

        logger.info('Creating the explainer')

        explainer = LimeCustomExplainer(np.zeros((1, lime_feature_names.shape[0])),
                                        mapping_index=mapping_index,
                                        mapping_features=param.mapping_features,
                                        length_max=length_max,
                                        product_subset=product_lvl_subset,
                                        feature_names=lime_feature_names,
                                        categorical_names=categorical_names,
                                        class_names=['No_purchase', 'Purchase'],
                                        categorical_features=categorical_index,
                                        resampling_features=resampling_idx,
                                        positive_features=positive_idx,
                                        original_inputs=n_inputs,
                                        kernel_width=custom_kw,
                                        verbose=False,
                                        training_data_stats=training_data_stats,
                                        discretize_continuous=False,
                                        sample_around_instance=False)

        logger.info('Explainer created')

        logger.info('Pushing the explainer to storage')

        gpl = gplLoader(project_id, dataset_id, bucket_name, directory, param.gpp['output_path'])

        with open('/tmp/' + file_explainer, 'wb') as f:

            dill.dump(explainer, f)

        gpl.load(source='local', destination='gs', data_name=file_explainer, delete_in_gs=False)

    if usage == 'single':

        logger.info('Defining the prediction file')

        # Defining the prediction dates
        prediction_file = ['gs://' + bucket_name + '/' + directory + path_data + '/Sequences_' + date + '_*.tfr']

        # Creating the input function for the test set
        pred_input_fn = create_input_fn(path=prediction_file)

        pred_iterator = pred_input_fn(shuffle_data=False)

        logger.info('Loading prediction file')

        pred_feat_lst, pred_lbl_lst = next(file_loader(pred_iterator))

        logger.info('Preprocessing prediction file')

        iter_pred_data = map(lambda x: preprocessing(x, explainer.length_max), [pred_feat_lst])

        logger.info('Sampling an id')

        pred_final_array = next(iter_pred_data)

        # Defining the instance interested to explain
        instance = pred_final_array[10]

        # Defining the maximum complexity for LIME, as the maximum number of features
        max_complexity = pred_final_array.shape[1]

        logger.info('Explaining the instance')

        final_exp = explain(explainer, instance, predict_fn, num_features=max_complexity, labels=(1,),
                            num_samples=param.n_samples, instance_length=pred_feat_lst['Lengths_features'][10])

        logger.info('Plotting the explanation')

        fig_explain = plotting(final_exp)

        figname = 'LIME_OldDist.png'

        fig_explain.savefig('/tmp/'+figname)

        gpl.load(source='local', destination='gs', data_name=figname, delete_in_gs=False)

        logger.info('Pushing the explanation to storage')

        explanation_name = 'explanation_kw_' + str(param.kernel_width)

        with open('/tmp/' + explanation_name, 'wb') as f:

            dill.dump(final_exp, f)

        gpl.load(source='local', destination='gs', data_name=explanation_name, delete_in_gs=False)

        logger.info('Ended the explanation')

    elif usage == 'parametric':

        logger.info('Performing parametric analysis on a single instance')

        logger.info('Defining the prediction file')

        # Defining the prediction dates
        prediction_file = ['gs://' + bucket_name + '/' + directory + path_data + '/Sequences_' + date + '_*.tfr']

        # Creating the input function for the test set
        pred_input_fn = create_input_fn(path=prediction_file)

        pred_iterator = pred_input_fn(shuffle_data=False)

        logger.info('Loading prediction file')

        pred_feat_lst, pred_lbl_lst = next(file_loader(pred_iterator))

        logger.info('Preprocessing prediction file')

        iter_pred_data = map(lambda x: preprocessing(x, explainer.length_max), [pred_feat_lst])

        logger.info('Sampling a id')

        pred_final_array = next(iter_pred_data)

        # Defining the instance interested to explain
        instance = pred_final_array[10]

        logger.info('Defining the parameters for the analysis')

        kernels = [0.25, 0.5, 0.75]

        complexities = [300, 400, 500, 600, 900]

        samples = [5000, 10000, 15000]

        logger.info('Loading the explainers')

        explainers = []

        for i in kernels:

            file_explainer = 'custom_explainer_kw_{}_V3'.format(i)

            gpl.load(source='gs', destination='local', data_name=file_explainer, delete_in_gs=False)

            with open('/tmp/' + file_explainer, 'rb') as f:

                explainer = dill.load(f)

            explainers.append(explainer)

        logger.info('Performing parameters analysis with kernel widths = {},'
                    'complexities = {}, number of samples = {}'.format(' '.join(map(str, kernels)),
                                                                       ' '.join(map(str, complexities)),
                                                                       ' '.join(map(str, samples))))

        for i in range(len(kernels)):

            for j in complexities:

                for k in samples:

                    logger.info('Performing for kernel_width = {}, '
                                'complexity = {} and number of samples = {}'.format(kernels[i], j, k))

                    explanation = explain(explainers[i], instance, predict_fn, num_features=j, labels=(1,),
                                          num_samples=k,
                                          instance_length=pred_feat_lst['Lengths_features'][10])

                    logger.info('Pushing the explanation to storage')

                    explanation_name = 'explanation_kw_{}_c_{}_ns_{}'.format(kernels[i], j, k)

                    with open('/tmp/' + explanation_name, 'wb') as f:

                        dill.dump(explanation, f)

                    gpl.load(source='local', destination='gs', data_name=explanation_name, delete_in_gs=False)

                    logger.info('Ended the explanation')

    else:

        logger.error('Usage has to be either single or dev')

        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_date', help='data_date')
    parser.add_argument('--global_proj_id', help='global_project_id')
    parser.add_argument('--account_proj_id', help='account_project_id')
    parser.add_argument('--bucket_name', help='bucket_name')
    parser.add_argument('--dataset_name', help='dataset_name')
    parser.add_argument("--account_id", dest="account", help="account id to work on.")
    parser.add_argument("--min_date", dest="min_date", help="min date to work on")
    parser.add_argument("--working_directory", help="Directory where we do stuff")
    parser.add_argument("--path_data", help="Path for files in gs bucket")
    # parser.add_argument("--path_model", help="Path for model in gs bucket")
    parser.add_argument("--usage", help="explanation mode : single or extensive")

    args = parser.parse_args()

    global_project_id = args.global_proj_id
    account_project_id = args.account_proj_id
    dataset_name = args.dataset_name
    account_id = args.account

    explainer_fn(project_id=global_project_id,
                 dataset_id=dataset_name,
                 bucket_name=args.bucket_name,
                 directory=args.working_directory,
                 path_data=args.path_data,
                 date=args.data_date,
                 usage=args.usage)
