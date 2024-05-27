import argparse, datetime, logging, sys
import numpy as np
from explainer import parameters as param
import explainer.utilities as custom_utils
from explainer.preprocessing import preprocessing
import dill
import json
import os
from purchase_probability.train_eval_predict import create_input_fn
from purchase_probability import parameters
from explainer.custom_explainer import ShapPredictor


logger = logging.getLogger(__name__)


def explainer_fn(project_id, dataset_id, directory, path_data, bucket_name, date, mode, usage):
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
    # The training files are necessary in SHAP to evaluate mean values and variances in order then to scale the variable
    # we want to explain

    np.random.seed(param.rnd_seed)

    logger.info('Downloading the model parameters')

    # Setting the google client
    gpl = custom_utils.gplLoader(project_id, dataset_id, bucket_name, directory, path_data)

    # Loading the rnn_params file (necessary for using the functions from ml_buy_probability)
    gpl.load(source='gs',
             destination='local',
             data_name='rnn_params.json',
             delete_in_gs=False)

    rnn_params_json = json.load(open(os.path.join('/tmp/', 'rnn_params.json')))

    parameters.model_params['N_INPUT'] = rnn_params_json['N_INPUT']

    length_max = parameters.bucket_boundaries[-1]

    if mode == 'train':

        logger.info('Defining the training days for the explainer')

        # Defining the first training day
        first_training_date = datetime.datetime.strptime(date, '%Y%m%d') -\
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
                       for x in training_days[: int(len(training_days) * 0.01)]]

        # train_files = ['gs://' + bucket_name + '/' + directory + path_data + '/Sequences_' + x + '_*.tfr'
        #                for x in training_days[: int(len(training_days) * 0.8)]]

        logger.info('Number of training files: ' + str(len(train_files)))

        logger.info('Creating explainer')

        # Creating the input function for the train set

        train_input_fn = create_input_fn(path=train_files)

        train_iterator = train_input_fn(shuffle_data=True)

        # Loading the explainer from gs if exists

        explainer_name = 'shap_explainers_' + str(param.n_samples) + '_local_training'

        # Defining the SHAP explainer

        logger.info('Defining the explainer')

        # Generating fake training data for the baseline for SHAP

        # Checking for training stats in gs

        training_stats_name = 'training_stats_local_training'

        # gpl = custom_utils.gplLoader(project_id, dataset_id, bucket_name, directory,
        #                              param.gpp['output_path'] + '/results_shap')
        gpl = custom_utils.gplLoader(project_id, dataset_id, bucket_name, directory,
                                     param.gpp['output_path'])

        training_data_stats = custom_utils.train_stats_generator(train_iterator, length_max)

        logger.info('Pushing the training stats in gs')

        with open('/tmp/' + training_stats_name, 'wb') as f:

            dill.dump(training_data_stats, f)

        gpl.load(source='local', destination='gs', data_name=training_stats_name)

        shap_training = custom_utils.train_data_generator(training_data_stats, param.n_samples)

        # Defining the explainer

        shap_training = shap_training[:, :, :]

        predictor = ShapPredictor(cloud_parameters=param.gpp,
                                  feature_names=param.new_feature_names,
                                  lstm_features=param.lstm_features,
                                  mapping_features=param.mapping_features,
                                  encoded_features=param.encoded_num2cat_idx,
                                  length_max=length_max)

        predictor.explainer_initializer(shap_training)

        logger.info('Pickling the explainers')

        with open('/tmp/' + explainer_name, 'wb') as f:

            dill.detect.trace(True)

            dill.dump(predictor, f)

            # dill.dump(shap_explainers, f)

        logger.info('Pushing the explainers to storage')

        gpl = custom_utils.gplLoader(project_id, dataset_id, bucket_name, directory,
                                     param.gpp['output_path'])

        # gpl = custom_utils.gplLoader(project_id, dataset_id, bucket_name, directory,
        #                              param.gpp['output_path'] + '/results_shap')

        gpl.load(source='local', destination='gs', data_name=explainer_name)

        logger.info('SHAP Explainer trained')

        sys.exit(0)

    if mode == 'explain':

        # explainer_name = 'shap_explainers_' + str(param.n_samples)

        explainer_name = 'shap_explainers_' + str(param.n_samples) + '_local_training'

        # gpl = custom_utils.gplLoader(project_id, dataset_id, bucket_name, directory,
        #                              param.gpp['output_path'] + '/results_shap')

        gpl = custom_utils.gplLoader(project_id, dataset_id, bucket_name, directory,
                                     param.gpp['output_path'])

        if gpl.exist_in_gs(explainer_name):

            logger.info('Loading explainer')

            gpl.load(source='gs', destination='local', data_name=explainer_name, delete_in_gs=False)

            with open('/tmp/' + explainer_name, 'rb') as f:

                shap_explainers = dill.load(f)

        else:

            logger.error('No model in storage. Run train')

            sys.exit(1)

        if usage == 'single':

            logger.info('Defining the prediction file')

            # Defining the prediction dates
            prediction_file = ['gs://' + bucket_name + '/' + directory + path_data + '/Sequences_' + date + '_*.tfr']

            # Creating the input function for the test set
            pred_input_fn = create_input_fn(path=prediction_file)

            pred_iterator = pred_input_fn(shuffle_data=False)

            logger.info('Loading prediction file')

            pred_feat_lst, pred_lbl_lst = next(custom_utils.file_loader(pred_iterator))

            instance_length = pred_feat_lst['Lengths_features'][param.instance_to_explain]

            iter_pred_data = map(lambda x: preprocessing(x, length_max), [pred_feat_lst])

            logger.info('Preprocessing prediction file')

            logger.info('Sampling an id')

            pred_final_array = next(iter_pred_data)

            # Defining the instance interested to explain

            instance = pred_final_array[param.instance_to_explain].reshape(1, length_max, -1)

            logger.info('Explaining the instance')

            shap_values = shap_explainers.explain(instance, instance_length)

            logger.info('Ended the explanation')

        else:

            logger.error('Usage has to be either single or dev')

            sys.exit(1)

    else:

        logger.error('Mode has to be either train or explain')

        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_date', help='data_date')
    parser.add_argument('--global_proj_id', help='global_project_id')
    parser.add_argument('--bucket_name', help='bucket_name')
    parser.add_argument('--dataset_name', help='dataset_name')
    parser.add_argument("--working_directory", help="Directory where we do stuff")
    parser.add_argument("--path_data", help="Path for files in gs bucket")
    parser.add_argument("--mode", help="running mode : train or explain")
    parser.add_argument("--usage", help="explanation mode : single or extensive")

    args = parser.parse_args()

    global_project_id = args.global_proj_id
    dataset_name = args.dataset_name

    explainer_fn(project_id=global_project_id,
                 dataset_id=dataset_name,
                 bucket_name=args.bucket_name,
                 directory=args.working_directory,
                 path_data=args.path_data,
                 date=args.data_date,
                 mode=args.mode,
                 usage=args.usage)
