# Package to build the model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Package to build the app

import logging
import os
import re
import json
import tempfile
import argparse
import datetime
from math import floor
import pandas as pd
import numpy as np

from google.cloud import storage
from google_pandas_load import Loader

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins.pr_curve import summary
from tensorflow.python.tools import inspect_checkpoint

from sklearn.metrics import roc_curve, precision_recall_curve, auc, brier_score_loss

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

import purchase_probability.parameters as parameters

# Instantiate logger
logger = logging.getLogger(__name__)

#####################
# Utility functions
#####################


def interval_number(x, n):
    res = floor(x * n)
    if res == n:
        return n - 1
    else:
        return res


def partition_proba_dico(y_test, y_proba, n_bins):
    dict_proba = {}
    for i in range(n_bins):
        dict_proba[i] = {'count': 0, 'proba_true': 0, 'proba_pred': 0, 'std_err': 0}
        y_proba_i = [x for x in y_proba if interval_number(x, n_bins) == i]
        dict_proba[i]['std_err'] = np.std(y_proba_i)

    for outcome, proba in zip(y_test, y_proba):
        i = interval_number(proba, n_bins)
        dict_proba[i]['count'] += 1
        dict_proba[i]['proba_true'] += outcome
        dict_proba[i]['proba_pred'] += proba

    for i in range(n_bins):
        counter = dict_proba[i]['count']
        if counter > 0:
            dict_proba[i]['proba_true'] = round(dict_proba[i]['proba_true'] / float(counter), 2)
            dict_proba[i]['proba_pred'] = round(dict_proba[i]['proba_pred'] / float(counter), 2)
        else:
            dict_proba[i]['proba_true'] = '/'
            dict_proba[i]['proba_pred'] = '/'
    return dict_proba


def partition_proba_curve(y_test, y_proba, n_bins):
    dict_proba = partition_proba_dico(y_test, y_proba, n_bins)
    counter = []
    proba_true = []
    proba_pred = []
    std = []
    for i in range(n_bins):
        if dict_proba[i]['count'] > 10:
            counter.append(dict_proba[i]['count'])
            proba_true.append(dict_proba[i]['proba_true'])
            proba_pred.append(dict_proba[i]['proba_pred'])
            std.append(dict_proba[i]['std_err'])
    return [counter, proba_true, proba_pred, std]


def partition_proba_table(y_test, y_proba, n_bins):
    dict_proba = partition_proba_dico(y_test, y_proba, n_bins)
    counter = []
    proba_true = []
    proba_pred = []
    for i in range(n_bins):
        counter.append(dict_proba[i]['count'])
        proba_true.append(dict_proba[i]['proba_true'])
        proba_pred.append(dict_proba[i]['proba_pred'])
    return [counter, proba_true, proba_pred]


def plot_calibration_curve(name, fig_index, labels, logits, n_bins, n_std=1):
    plt.figure(fig_index, figsize=(10, 7))
    ax = plt.subplot2grid((1, 1), (0, 0))

    ax.set_title('Reliability curve')

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    clf_score = brier_score_loss(labels, logits, pos_label=1)

    _, proba_true_curve, proba_pred_curve, std = partition_proba_curve(labels, logits, n_bins)

    ax.plot(proba_pred_curve, proba_true_curve, "s-", label="%s (%1.3f)" % (name, clf_score))

    ax.plot([x+n_std*y for x, y in zip(proba_pred_curve, std)], proba_true_curve, "+-", label="Value + std")
    ax.plot([x-n_std*y for x, y in zip(proba_pred_curve, std)], proba_true_curve, "+-", label="Value - std")

    ax.legend(loc="upper left")

    ax.set_ylabel("Fraction of positives")
    ax.set_ylim([-0.05, 1.05])

    ax.set_xlabel("Predicted probability")
    ax.xaxis.set_label_coords(0.5, 0.04)
    ax.set_xticklabels(labels=["", 0.2, 0.4, 0.6, 0.8, ""], position=(0, 0.04))
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_xaxis().tick_bottom()

    count_table, proba_true_table, proba_pred_table = partition_proba_table(labels, logits, n_bins)

    table = ax.table(cellText=[proba_pred_table, proba_true_table, count_table],
                     rowLabels=['Predicted probability', 'Fraction of positives', 'Count'],
                     loc='bottom',
                     bbox=[0, -0.19, 1, 0.18])
    table.set_fontsize(17)
    plt.tight_layout()


def display_results(labels, preds):
    nb_ones = labels.sum()
    nb_zeros = labels.shape[0] - nb_ones

    # ROC CURVE
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label='ROC curve for LSTM (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate/proportion de zeros mal classifies')
    plt.ylabel('True Positive Rate/Recall')
    plt.title(
        'Receiver operating characteristic for LSTM - count 0 : {0}     count 1 : {1}'.format(nb_zeros, nb_ones))
    plt.legend(loc="lower right")
    plt.savefig('Curve_ROC_auc.png')

    # PRECISION / RECALL CURVE
    precision, recall, _ = precision_recall_curve(labels, preds)
    plt.figure(figsize=(10, 10))
    plt.plot(precision, recall, label='Precision/Recall curve for LSTM')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Precision')
    plt.ylabel('True Positive Rate/Recall')
    plt.title('precision/recall curve for LSTM')
    plt.legend(loc="lower right")
    plt.savefig('Curve_precision_recall.png')


########################
# Processing functions
########################


def input_parser(serialized_example):
    # Define how to parse an example
    # "inputs_num" and "inputs_cat" will be a sequence of list of length "n_input",
    # i.e. number of columns per time step
    # 'labels" will be a sequence of integer (0 or 1 depending on purchase or not purchase)
    sequence_features = {"ids": tf.FixedLenSequenceFeature([], dtype=tf.string),
                         "inputs_num": tf.FixedLenSequenceFeature([parameters.model_params['N_INPUT']],
                                                                  dtype=tf.float32),
                         "inputs_cat": tf.FixedLenSequenceFeature(
                             [len(parameters.model_params['vocabulary_sizes'].keys())], dtype=tf.int64),
                         "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                         "inputs_fixed": tf.FixedLenSequenceFeature([], dtype=tf.float32)
                         }

    # Parse the example 'serialized_example' with the schema 'sequence_features' previously defined
    _, sequence = tf.parse_single_sequence_example(serialized=serialized_example,
                                                   sequence_features=sequence_features)

    actual_length = tf.shape(sequence["inputs_num"])[0]

    seq_labels = tf.reshape(tf.one_hot(sequence['labels'], depth=parameters.NUM_CLASSES, axis=-1),
                            [parameters.NUM_CLASSES],
                            name='One_hot_encoding')

    return sequence['ids'], sequence['inputs_num'], sequence['inputs_cat'], seq_labels, actual_length,\
        sequence['inputs_fixed']


def bucketing(sequence_length, buckets):
    # Given a sequence_length returns a bucket id
    # Clip the buckets at sequence length and return the first argmax, the bucket id
    t = tf.clip_by_value(buckets, 0, sequence_length)
    return tf.argmax(t)


def padd_and_batch(grouped_dataset, batchsize):
    # Elements of grouped_dataset are padded up to padded_shapes, and batch to batchsize (max)
    return grouped_dataset.padded_batch(batchsize,
                                        padded_shapes=([None],
                                                       [None, parameters.model_params['N_INPUT']],
                                                       [None, len(parameters.model_params['vocabulary_sizes'].keys())],
                                                       [None],
                                                       [],
                                                       [None]),
                                        padding_values=('-1', -1., tf.cast(0, dtype=tf.int64), -1., -1, -1.))


def create_input_fn(path):
    # This function needs input_parser, bucketing, and padd_and_batch functions
    # TO DO: Optimization of cycle_length, block_length, num_parallel_calls ?
    def input_fn(shuffle_data=False, buckets=parameters.bucket_boundaries, buffer_size=parameters.buffer_size,
                 batch_size=parameters.BATCH_SIZE, epochs=parameters.NUM_EPOCHS):
        with tf.name_scope('Dataset_reader'):
            files = tf.data.Dataset.list_files(path)

            # Create Tensorflow Dataset objects
            # Use interleave() and prefetch() to read many files concurrently.
            dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x,
                                                                         compression_type='GZIP'
                                                                         ).prefetch(batch_size),
                                       cycle_length=8,
                                       block_length=500)
            # Todo try if it increases the speed
            # dataset = files.apply(tf.contrib.data.parallel_interleave(
            #                        tf.data.TFRecordDataset, cycle_length=8))

            # Map the tfrecord example to tensor using the input_parser function
            dataset = dataset.map(input_parser, num_parallel_calls=8)

            # Number of elements per bucket.
            window_size = 10 * batch_size

            # Group the dataset according to a bucket key (see bucketing).
            # Every element in the dataset is attributed a key (i.e. a bucket id)
            # The elements are then bucketed according to these keys. A group of
            # `window_size` having the same keys are given to the reduce_func.
            dataset = dataset.apply(tf.contrib.data.group_by_window(
                                        key_func=lambda mi, x_num, x_cat, y, z, fixed: bucketing(z, buckets),
                                        reduce_func=lambda _, x: padd_and_batch(x, batch_size),
                                        window_size=window_size))

            if shuffle_data:
                # Shuffle the dataset
                dataset = dataset.shuffle(buffer_size=buffer_size)

            # Repeat the input NUM_EPOCHS times. useful with the iterator chosen
            dataset = dataset.repeat(epochs)

            # Prefetch dataset (working with tf.__version__ >= 1.4)
            dataset = dataset.prefetch(buffer_size)

            # Create iterator. Can be modified to match use case
            iterator = dataset.make_one_shot_iterator()

            # Get next elements from the iterator for train and validation
            next_mi, next_num_example, next_cat_example, next_label, next_length, next_fixed = iterator.get_next()

            features_dict = {'Numerical_features': next_num_example,
                             'Categorical_features': next_cat_example,
                             'Lengths_features': next_length,
                             'Fixed_features': next_fixed,
                             'ids': next_mi}
        return features_dict, next_label

    return input_fn


# Construct a custom model function for tf.Estimator
def model_fn(features, labels, mode, params):
    # Logic to do the following:
    # 1. Configure the model via TensorFlow operations
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object

    # Work around to match our custom configuration
    num_example = features['Numerical_features']
    cat_example = features['Categorical_features']
    length = features['Lengths_features']
    fixed_features = features['Fixed_features']

    with tf.variable_scope('lstm_numerical_feat'):
        # Basic LSTM Cell for our neural network
        cell = tf.contrib.rnn.LSTMCell(params['num_hidden'])

        # Define the neural network. Val is the output value, state is the state value
        # (irrelevant, each time we look at a new sequence)
        val, state = tf.nn.dynamic_rnn(cell, num_example, dtype=tf.float32, sequence_length=length)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'val' has already the right dimension: [batch_size, n_timestep, n_hidden]

        # Hack to build the indexing and retrieve the right output.
        batch_sizes = tf.shape(val)[0]

        # Start indices for each sample
        index = tf.range(0, batch_sizes) * tf.shape(val)[1] + (length - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(val, [-1, params['num_hidden']]), index)

    with tf.variable_scope('lstm_embed_product_department'):
        # Create embeddings for product_department. Start with a random matrix
        embedding_matrix_1 = tf.Variable(
            tf.random_uniform([parameters.model_params['vocabulary_sizes']['product_department'],
                               params['embedding_size_1']],
                              -1.0, 1.0),
            name="Embeddings_product_department")
        embed_1 = tf.nn.embedding_lookup(embedding_matrix_1, cat_example[:, :, 0], name='Embedding_lookup_1')

        cell_embed_1 = tf.contrib.rnn.LSTMCell(params['embedding_size_1'])
        # state_embed_1 = cell_embed_1.zero_state(tf.shape(cat_example)[0], tf.float32)

        # LSTM for embedding part
        val_embed_1, state_embed_1 = tf.nn.dynamic_rnn(cell_embed_1, embed_1, dtype=tf.float32,
                                                       sequence_length=length)

        batch_sizes_embed_1 = tf.shape(val_embed_1)[0]

        # Start indices for each sample
        index_embed_1 = tf.range(0, batch_sizes_embed_1) * tf.shape(val_embed_1)[1] + (length - 1)
        # Indexing
        outputs_embed_1 = tf.gather(tf.reshape(val_embed_1, [-1, params['embedding_size_1']]), index_embed_1)

    with tf.variable_scope('lstm_embed_product_category'):
        # Create embeddings for product_category. Start with a random matrix
        embedding_matrix_2 = tf.Variable(
            tf.random_uniform([parameters.model_params['vocabulary_sizes']['product_category'],
                               params['embedding_size_2']],
                              -1.0, 1.0),
            name="Embeddings_product_category")
        embed_2 = tf.nn.embedding_lookup(embedding_matrix_2, cat_example[:, :, 1], name='Embedding_lookup_2')

        cell_embed_2 = tf.contrib.rnn.LSTMCell(params['embedding_size_2'])

        # LSTM for embedding part
        val_embed_2, state_embed_2 = tf.nn.dynamic_rnn(cell_embed_2, embed_2, dtype=tf.float32,
                                                       sequence_length=length)

        batch_sizes_embed_2 = tf.shape(val_embed_2)[0]

        # Start indices for each sample
        index_embed_2 = tf.range(0, batch_sizes_embed_2) * tf.shape(val_embed_2)[1] + (length - 1)
        # Indexing
        outputs_embed_2 = tf.gather(tf.reshape(val_embed_2, [-1, params['embedding_size_2']]), index_embed_2)

    with tf.variable_scope('lstm_embed_product_sub_category'):
        # Create embeddings for product_sub_category. Start with a random matrix
        embedding_matrix_3 = tf.Variable(
            tf.random_uniform([parameters.model_params['vocabulary_sizes']['product_sub_category'],
                               params['embedding_size_3']],
                              -1.0, 1.0),
            name="Embeddings_product_sub_category")
        embed_3 = tf.nn.embedding_lookup(embedding_matrix_3, cat_example[:, :, 2], name='Embedding_lookup_3')

        cell_embed_3 = tf.contrib.rnn.LSTMCell(params['embedding_size_3'])

        # LSTM for embedding part
        val_embed_3, state_embed_3 = tf.nn.dynamic_rnn(cell_embed_3, embed_3, dtype=tf.float32,
                                                       sequence_length=length)

        batch_sizes_embed_3 = tf.shape(val_embed_3)[0]

        # Start indices for each sample
        index_embed_3 = tf.range(0, batch_sizes_embed_3) * tf.shape(val_embed_3)[1] + (length - 1)
        # Indexing
        outputs_embed_3 = tf.gather(tf.reshape(val_embed_3, [-1, params['embedding_size_3']]), index_embed_3)

    with tf.name_scope('Concatenate'):
        # concat outputs of LSTM
        outputs = tf.concat((outputs, outputs_embed_1, outputs_embed_2, outputs_embed_3), axis=1)

    # Add non-sequential layer: fixed features (like time before last events, weekday ...)
    concat_feat = tf.concat((outputs, tf.reshape(fixed_features, [-1, 1])), axis=1)

    with tf.name_scope('Dense_layer'):
        merged = tf.layers.dense(concat_feat, params['num_hidden'] + params['embedding_size_1']
                                                                   + params['embedding_size_2']
                                                                   + params['embedding_size_3']
                                 )
    # Make prediction:
    with tf.variable_scope("prediction"):
        # Define weight and bias
        weight = tf.Variable(tf.random.truncated_normal([params['num_hidden'] + params['embedding_size_1']
                                                                              + params['embedding_size_2']
                                                                              + params['embedding_size_3'],
                                                                              parameters.model_params['num_classes']]),
                             name='Weight')

        bias = tf.Variable(tf.constant(0.1, shape=[2]), name='Bias')
        prediction = tf.add(tf.matmul(merged, weight), bias)  # Add Relu ? Dropout ?
        prediction_softmaxed = tf.nn.softmax(prediction, name='Prediction_probas')

    # Activation - softmax
    with tf.variable_scope('softmax-cross_entropy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=prediction))

    # Set up learning rate with exp decay
    learn_rate = tf.compat.v1.train.exponential_decay(params['learning_rate'], tf.compat.v1.train.get_or_create_global_step(),
                                                      10000, 0.96, staircase=True)

    loss = tf.compat.v1.train.AdamOptimizer(learning_rate=learn_rate).minimize(
        cost, global_step=tf.compat.v1.train.get_global_step())

    # Calculation of error on test data
    with tf.variable_scope('Error'):
        mistakes = tf.not_equal(tf.argmax(labels, 1), tf.argmax(prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    accuracy, accuracy_op = tf.compat.v1.metrics.accuracy(labels=tf.argmax(labels, 1),
                                                          predictions=tf.argmax(prediction, 1))

    # Calculate AUC metric
    auc_live, auc_update_op = tf.compat.v1.metrics.auc(predictions=tf.nn.softmax(prediction),
                                                       labels=labels,
                                                       name='AUC')

    summary.op(name='Precision_recall',
               labels=tf.cast(labels, tf.bool),
               predictions=tf.nn.softmax(prediction),
               num_thresholds=20)

    eval_metric_ops = {'Accuracy': (accuracy, accuracy_op),
                       'Area_under_curve': (auc_live, auc_update_op)}

    # create a summary for our cost and accuracy
    tf.compat.v1.summary.scalar("cost", cost)
    tf.compat.v1.summary.scalar("error", error)
    tf.compat.v1.summary.scalar("accuracy", accuracy_op)

    # merge all summaries into a single "operation" which we can execute in a session
    writer = tf.compat.v1.summary.FileWriter('gs://' + params['bucket_name'] + "/" + params['directory'],
                                             graph=tf.compat.v1.get_default_graph())

    summary_hook = tf.estimator.SummarySaverHook(save_steps=100,
                                                 summary_writer=writer,
                                                 summary_op=tf.compat.v1.summary.merge_all())

    # Config projector for embeddings
    projector_config = projector.ProjectorConfig()
    # You can add multiple embeddings. Here we add only one.
    added_embedding = projector_config.embeddings.add()
    added_embedding.tensor_name = "lstm_embed_product_department/Embeddings_product_department:0"  # Name of the embedding matrix
    # Link this tensor to its metadata file.
    added_embedding.metadata_path = 'metadata_product_department.csv'

    # Write embeddings to summary_writer
    projector.visualize_embeddings(writer, projector_config)

    export_outputs = {'Predictions': tf.estimator.export.PredictOutput({"Predictions": prediction_softmaxed})}

    if mode != tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions={"Predictions": prediction_softmaxed,
                                                       "ids": features['ids']},
                                          loss=cost,
                                          train_op=loss,
                                          eval_metric_ops=eval_metric_ops,
                                          training_hooks=[summary_hook])
    else:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions={"Predictions": prediction_softmaxed,
                                                       "ids": features['ids']})


##########################
# Serving input function
##########################

def tfr_serving_input_fn():
    """ Build the serving inputs."""

    feature_placeholders = {
        "Numerical_features": tf.placeholder(tf.float32, [None, None, parameters.model_params['N_INPUT']]),
        "Categorical_features": tf.placeholder(tf.int32,
                                               [None, None, len(parameters.model_params['vocabulary_sizes'].keys())]),
        "Fixed_features": tf.placeholder(tf.float32, [None]),
        "Lengths_features": tf.placeholder(tf.int32, [None]),
        "labels": tf.placeholder(tf.float32, [None]),
        "Predictions": tf.placeholder(tf.float32, [None])
                                }

    feature_spec = {"Numerical_features": tf.FixedLenSequenceFeature([parameters.model_params['N_INPUT']],
                                                                     dtype=tf.float32,
                                                                     allow_missing=True),
                    "Categorical_features": tf.FixedLenSequenceFeature(
                        [len(parameters.model_params['vocabulary_sizes'].keys())],
                        dtype=tf.int64,
                        allow_missing=True),
                    # "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                    "Fixed_features": tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
                    "Lengths_features": tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)
                    }

    features = {key: tensor for key, tensor in feature_placeholders.items()}

    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[parameters.BATCH_SIZE],
                                           name='input_tensor')
    receiver_tensors = {'predictor_inputs': serialized_tf_example}
    # features = tf.parse_example(serialized_tf_example, feature_spec)

    # features["Lengths_features"] = tf.squeeze(features["Lengths_features"], axis=[1])
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


#####################
# Training function
#####################

def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters set.
    """
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if tf_config and 'task' in tf_config and 'type' in tf_config['task'] and 'index' in tf_config['task']:
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=['/job:ps',
                                                  '/job:worker/task:%d' % tf_config['task']['index']
                                                  ])
    return None


def train_rnn(bucket_name, execution_date, first_training_date, directory, path_data, path_model, max_step):

    # Enable logging
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.reset_default_graph()

    # Load params from preprocessing and add it to model_params
    with tempfile.TemporaryDirectory() as tmpdir:
        #  project_id is the "global" one, same for dataset_name
        gs_client = storage.Client()
        bucket = storage.bucket.Bucket(client=gs_client, name=bucket_name)

        gpl = Loader(bucket=bucket,
                     local_dir_path=tmpdir,
                     gs_dir_path_in_bucket=directory + path_data,
                     logger=logger
                     )

        gpl.load(source='gs',
                 destination='local',
                 data_name='rnn_params.json',
                 delete_in_gs=False)

        rnn_params_json = json.load(open(os.path.join(tmpdir, 'rnn_params.json')))

    parameters.model_params['N_INPUT'] = rnn_params_json['N_INPUT']

    nb_training_days = max(0, (execution_date - datetime.timedelta(days=parameters.forward_prediction-1) - first_training_date).days)
    training_days = [(first_training_date + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(nb_training_days)]

    train_files = ['gs://' + bucket.name + '/' + directory + path_data + '/Sequences_' + x + '_*.tfr'
                   for x in training_days[: int(len(training_days) * 0.8)]]  # TODO: 0.8 AS A PARAMETER?
    eval_files = ['gs://' + bucket.name + '/' + directory + path_data + '/Sequences_' + x + '_*.tfr'
                  for x in training_days[int(len(training_days) * 0.8):]]

    tf.logging.info('Creating train input function...')
    # Create train input function and evaluation input function
    train_input_fn = create_input_fn(path=train_files)
    eval_input_fn = create_input_fn(path=eval_files)

    # Retrieve last checkpoint and find number of past iterations
    latest_ckpt = tf.train.latest_checkpoint('gs://' + bucket.name + '/' + directory + path_model + '/')

    if latest_ckpt is not None:
        pattern_match = re.search(r'[0-9]{1,}$', latest_ckpt)
        past_nb_iter = int(pattern_match.group(0))
    else:
        past_nb_iter = 0

    logger.info(past_nb_iter)

    """ If we want to do early stopping
    # Early stopping hook
    early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(nn_lstm,
                                                                   metric_name='loss',
                                                                   max_steps_without_decrease=1000,
                                                                   min_steps=10000)
    """

    # Define training and eval specs
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(shuffle_data=True),
                                        max_steps=past_nb_iter + max_step
                                        )  # hooks=[early_stopping])

    # For eval specs, define also a servinginputreceiver,which will be passed to an exporter
    exporter = tf.estimator.FinalExporter('features',
                                          tf.estimator.export.build_parsing_serving_input_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(epochs=1),
                                      steps=100,
                                      # exporters=[exporter],
                                      name='purchase_preds_eval')

    tf.logging.info('Instantiate Estimator...')
    # Instantiate an Estimator
    config_estimator = tf.estimator.RunConfig(log_step_count_steps=1000,
                                              model_dir='gs://' + bucket_name + "/" + directory + path_model,
                                              session_config=_get_session_config_from_env_var())

    params_estimator = parameters.hp_params

    params_estimator.update({'bucket_name': bucket_name,
                             'directory': directory + path_model})

    nn_lstm = tf.estimator.Estimator(model_fn=model_fn,
                                     params=params_estimator,
                                     model_dir='gs://' + bucket_name + "/" + directory + path_model,
                                     config=config_estimator)

    tf.logging.info('Start training RNN model ...')
    # Train and evaluate the Estimator
    profiler_hook = tf.train.ProfilerHook(save_steps=50,
                                          output_dir=directory + path_model,
                                          show_dataflow=False,
                                          show_memory=True)  # working but not really useful
    # nn_lstm.train(input_fn=lambda: train_input_fn(shuffle_data=True),
    #              steps=max_step,
    #              hooks=[profiler_hook])
    tf.estimator.train_and_evaluate(nn_lstm, train_spec, eval_spec)
    tf.logging.info('Training done !')

    with tempfile.TemporaryDirectory() as tmpdir:
        file_date_execution = "model_last_execution_date_{0}".format(execution_date.strftime("%Y%m%d"))
        with open(file_date_execution):
            pass

        gpl = Loader(bucket=bucket,
                     local_dir_path=tmpdir,
                     gs_dir_path_in_bucket=directory + path_model,
                     logger=logger
                     )

        gpl.load(source='local',
                 destination='gs',
                 data_name=file_date_execution
                 )

    feature_spec = {"Numerical_features": tf.FixedLenSequenceFeature([parameters.model_params['N_INPUT']],
                                                                     dtype=tf.float32,
                                                                     allow_missing=True),
                    "Categorical_features": tf.FixedLenSequenceFeature(
                        [len(parameters.model_params['vocabulary_sizes'].keys())],
                        dtype=tf.int64,
                        allow_missing=True),
                    # "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                    "Fixed_features": tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
                    "Lengths_features": tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)
                    }

    feature_placeholders = {
        "Numerical_features": tf.placeholder(tf.float32, [None, None, parameters.model_params['N_INPUT']]),
        "Categorical_features": tf.placeholder(tf.int32,
                                               [None, None, len(parameters.model_params['vocabulary_sizes'].keys())]),
        "Fixed_features": tf.placeholder(tf.float32, [None]),
        "Lengths_features": tf.placeholder(tf.int32, [None]),
        "labels": tf.placeholder(tf.float32, [None]),
        "Predictions": tf.placeholder(tf.float32, [None])
                                }
    # nn_lstm.export_savedmodel('gs://' + bucket_name + '/' + directory,
    #                           serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(
    #                           features=feature_placeholders)
    #                           )


#########################
# Optimization function
#########################

def hyperparameter_optim(train_files, eval_files):
    count = 0

    space = [
        Integer(64, 512, name='batch_size'),
        Real(1e-7, 1e-6, name='learning_rate'),
        Integer(10, 150, name='embedding_size_1'),
        Integer(10, 150, name='embedding_size_2'),
        Integer(10, 150, name='embedding_size_3'),
        Integer(50, 150, name='num_hidden')
    ]

    config_estimator = tf.estimator.RunConfig(log_step_count_steps=1000)

    @use_named_args(space)
    def score(**params):
        global count
        params['directory'] = '/test_skopt_' + str(count)

        train_input_fn = create_input_fn(path=train_files)
        eval_input_fn = create_input_fn(path=eval_files)

        nn_lstm = tf.estimator.Estimator(model_fn=model_fn,
                                         params=params,
                                         model_dir=params['directory'],  # check directory
                                         config=config_estimator)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(shuffle_data=True),
                                            max_steps=20000)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(epochs=1))

        try:
            tf.estimator.train_and_evaluate(nn_lstm, train_spec, eval_spec)
            eval_metrics = nn_lstm.evaluate(lambda: eval_input_fn(epochs=1))

            count += 1

            return eval_metrics['loss']
        except (tf.errors.ResourceExhaustedError, tf.train.NanLossDuringTrainingError):
            return 1e9

    gp_minimize(score, space)


def inspect_checkpoint_file(file_name):
    # Print all tensors in checkpoint file
    inspect_checkpoint.print_tensors_in_checkpoint_file(file_name=file_name,
                                                        tensor_name='',
                                                        all_tensors=False,
                                                        all_tensor_names=False)


#################
# Test function
#################

def test_rnn(bucket_name, directory, path_data, path_model, execution_date, batch_size=20000):
    tf.reset_default_graph()

    # Load params from preprocessing and add it to model_params
    with tempfile.TemporaryDirectory() as tmpdir:
        gs_client = storage.Client()
        bucket = storage.bucket.Bucket(client=gs_client, name=bucket_name)

        gpl = Loader(bucket=bucket,
                     local_dir_path=tmpdir,
                     gs_dir_path_in_bucket=directory + path_data,
                     logger=logger
                     )

        gpl.load(source='gs',
                 destination='local',
                 data_name='rnn_params.json',
                 delete_in_gs=False)

        rnn_params_json = json.load(open(os.path.join(tmpdir, 'rnn_params.json')))
    parameters.model_params['N_INPUT'] = rnn_params_json['N_INPUT']

    parameters.hp_params.update({'bucket_name': bucket_name,
                                 'directory': directory + path_model + '/tests'})

    test_prediction_dates = [(execution_date - datetime.timedelta(days=x + parameters.forward_prediction)).strftime("%Y%m%d") for x in range(parameters.forward_prediction)]

    test_files = ['gs://' + bucket.name + '/' + directory + path_data + '/Sequences_' + x + '_*.tfr'
                  for x in test_prediction_dates]

    # Create test function
    test_input_fn = create_input_fn(path=test_files)

    # Rebuild the input pipeline
    features, labels = test_input_fn(batch_size=batch_size, epochs=1)

    # Rebuild the model
    predictions_est = model_fn(features,
                               labels,
                               mode=tf.estimator.ModeKeys.EVAL,
                               params=parameters.hp_params)

    # Manually load the latest checkpoint
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('gs://' + bucket_name + '/' + directory + path_model)
        logger.info(ckpt)
        saver.restore(sess, ckpt.model_checkpoint_path)

        predictions = predictions_est.predictions

        # Loop through the batches and store predictions and labels
        prediction_values = []
        label_values = []
        while True:
            try:
                preds, lbls = sess.run([predictions, labels])
                prediction_values = np.append(prediction_values, [x[1] for x in preds['Predictions']])
                label_values = np.append(label_values, [x[1] for x in lbls])

                # confusion = tf.confusion_matrix(labels=tf.argmax(lbls, 1),
                #                                 predictions=tf.argmax(preds['Predictions'], 1), num_classes=2)

            except tf.errors.OutOfRangeError:
                break
    return prediction_values, label_values


def visualize_rnn(labels, predictions, execution_date, bucket_name, directory, path_model):

    with tempfile.TemporaryDirectory() as tmpdir:

        # Plot calibration curve
        plot_calibration_curve('model_hesitation', 1, labels, predictions, 10)
        plt.savefig(tmpdir + '/' + execution_date.strftime("%Y%m%d") + '_Curve_calibration.png',
                    bbox_inches='tight')

        # Plot distribution of probabilities
        hist_data = pd.DataFrame(predictions)
        fig, ax = plt.subplots()
        hist_data.hist(bins=15, bottom=0.1, ax=ax)
        ax.set_yscale('log')
        plt.savefig(tmpdir + '/' + execution_date.strftime("%Y%m%d") + '_Curve_Probability_distribution.png')

        # Display ROC_curve, Precision-recall curve and confusion matrix
        display_results(labels, predictions)

        itemindex = np.where(labels == 1)

        label_one = pd.DataFrame(predictions[itemindex])
        _, ax = plt.subplots()
        label_one.hist(bins=15, bottom=0.1, ax=ax)
        ax.set_yscale('log')
        plt.savefig(tmpdir + '/' + execution_date.strftime("%Y%m%d") + '_Curve_Probability_distribution_label1.png')

        itemindex = np.where(labels == 0)

        label_zero = pd.DataFrame(predictions[itemindex])
        _, ax = plt.subplots()
        label_zero.hist(bins=15, bottom=0.1, ax=ax)
        ax.set_yscale('log')
        plt.savefig(tmpdir + '/' + execution_date.strftime("%Y%m%d") + '_Curve_Probability_distribution_label0.png')

        gs_client = storage.Client()
        bucket = storage.bucket.Bucket(client=gs_client, name=bucket_name)

        gpl = Loader(bucket=bucket,
                     local_dir_path=tmpdir,
                     gs_dir_path_in_bucket=directory + path_model + '/tests/figures',
                     logger=logger
                     )

        gpl.load(source='local',
                 destination='gs',
                 data_name=execution_date.strftime("%Y%m%d"),
                 delete_in_local=True)

        logger.info('Curve files created in gs.')


def test_and_vis(bucket_name, directory, path_data, path_model, execution_date, batch_size=20000):
    predictions, labels = test_rnn(bucket_name=bucket_name,
                                   directory=directory,
                                   execution_date=execution_date,
                                   path_data=path_data,
                                   path_model=path_model,
                                   batch_size=batch_size)

    visualize_rnn(predictions=predictions,
                  labels=labels,
                  execution_date=execution_date,
                  bucket_name=bucket_name,
                  directory=directory,
                  path_model=path_model)

#################
# Predict function
#################


def predict_rnn(bucket_name, pred_files, directory, path_data, path_model, execution_date):
    # Load params from preprocessing and add it to model_params
    with tempfile.TemporaryDirectory() as tmpdir:
        gs_client = storage.Client()
        bucket = storage.bucket.Bucket(client=gs_client, name=bucket_name)

        gpl = Loader(bucket=bucket,
                     local_dir_path=tmpdir,
                     gs_dir_path_in_bucket=directory + path_data,
                     logger=logger
                     )

        gpl.load(source='gs',
                 destination='local',
                 data_name='rnn_params.json',
                 delete_in_gs=False)

        rnn_params_json = json.load(open(os.path.join(tmpdir, 'rnn_params.json')))
    parameters.model_params['N_INPUT'] = rnn_params_json['N_INPUT']

    # Create prediction input function
    pred_input_fn = create_input_fn(path=pred_files)

    params_estimator = parameters.hp_params

    params_estimator.update({'bucket_name': bucket_name,
                             'directory': directory + path_model})

    # Rebuild the input pipeline
    b_size = 10000
    features, labels = pred_input_fn(batch_size=b_size, epochs=1)

    # Rebuild the model
    predictions_est = model_fn(features,
                               labels,
                               mode=tf.estimator.ModeKeys.PREDICT,
                               params=params_estimator)

    # Manually load the latest checkpoint
    saver = tf.train.Saver()
    logger.info("Starting predictions by batch of {}.".format(b_size))
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('gs://' + bucket_name + '/' + directory + path_model)
        logger.info(ckpt)
        saver.restore(sess, ckpt.model_checkpoint_path)

        predictions = predictions_est.predictions

        # Loop through the batches and store predictions and labels
        prediction_values = []
        id_values = []
        while True:
            try:
                preds_mi = sess.run([predictions])
                prediction_values = np.append(prediction_values, [x[1] for x in preds_mi[0]['Predictions']])
                id_values = np.append(id_values, preds_mi[0]['ids'])

            except tf.errors.OutOfRangeError as e:  # ResourceExhaustedError:
                logger.warning(e)
                break

    result_predict = pd.DataFrame({'id': [x.decode('utf-8') for x in id_values],
                                   'purchase_probas': prediction_values})

    logger.info("Loading predictions into BigQuery")
    with tempfile.TemporaryDirectory() as tmpdir:
        gs_client = storage.Client()
        bucket = storage.bucket.Bucket(client=gs_client, name=bucket_name)

        gpl = Loader(bucket=bucket,
                     local_dir_path=tmpdir,
                     gs_dir_path_in_bucket=directory + path_data,
                     logger=logger
                     )

        gpl.load(source='dataframe',
                 destination='gs',
                 dataframe=result_predict,
                 data_name='predictions_' + execution_date.strftime("%Y%m%d"))

    return result_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execution_date', help='execution_date')
    parser.add_argument('--bucket_name', help='bucket_name')
    parser.add_argument('--dataset_name', help='dataset_name')
    parser.add_argument("--first_training_date", dest="first_training_date", help="first training date")
    parser.add_argument("--train_or_pred_files", help="train files or prediction files for the model.")
    parser.add_argument("--action", help="train, evaluate or predict.")
    parser.add_argument("--working_directory", help="Directory where we do stuff")
    parser.add_argument("--path_data", help="Path for files in gs bucket")
    parser.add_argument("--path_model", help="Path for model in gs bucket")

    args = parser.parse_args()

    execution_date = datetime.datetime.strptime(args.execution_date, '%Y%m%d')
    bucket_name = args.bucket_name
    dataset_name = args.dataset_name
    files = args.train_or_pred_files
    action = args.action
    first_training_date = datetime.datetime.strptime(args.first_training_date, '%Y%m%d') if args.first_training_date else None
    path_data = args.path_data
    path_model = args.path_model
    working_directory = args.working_directory

    if action == 'train':
        train_rnn(bucket_name=bucket_name,
                  execution_date=execution_date,
                  first_training_date=first_training_date,
                  directory=working_directory,
                  path_data=path_data,
                  path_model=path_model,
                  max_step=parameters.max_step_training)

    elif action == 'predict':
        predict_rnn(bucket_name=bucket_name,
                    pred_files=files,
                    directory=working_directory,
                    path_data=path_data,
                    path_model=path_model,
                    execution_date=execution_date)

    elif action == 'evaluate':
        test_and_vis(bucket_name=bucket_name,
                     directory=working_directory,
                     path_data=path_data,
                     path_model=path_model,
                     execution_date=execution_date,
                     batch_size=20000)

    else:
        logger.info("Action '%s' unrecognized --> BYE BYE!" % action)
