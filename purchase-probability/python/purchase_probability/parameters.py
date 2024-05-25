# Fonctionnel configuration

# Cloud ML parameters
ml_pythonVersion = '3.7'
ml_runtimeVersion = '1.15'
ml_region = 'europe-west1'
# Useful for preprocessing and predict, not for training. ml_scalerTier_train is used for training
# ml_scaleTier = 'BASIC'
ml_preprocess = {'ml_scaleTier_train': 'CUSTOM',
                 'ml_masterType': 'large_model'}

ml_predict = {'ml_scaleTier_train': 'CUSTOM',
              'ml_masterType': 'large_model'}

typology_machine = {'S': {'ml_scaleTier_train': 'BASIC',
                          'ml_masterType': 'standard',
                          'ml_workerType': 'large_model',
                          'ml_workerCount': 1,
                          'ml_parameterServerType': 'standard',
                          'ml_parameterServerCount': 0},
                    'M': {'ml_scaleTier_train': 'CUSTOM',
                          'ml_workerCount': 3,
                          'ml_workerType': 'standard_gpu',
                          'ml_masterType': 'complex_model_m_gpu',
                          'ml_parameterServerCount': 2,
                          'ml_parameterServerType': 'standard_gpu'},
                    'L': {'ml_scaleTier_train': 'CUSTOM',
                          'ml_workerCount': 4,
                          'ml_workerType': 'complex_model_m_gpu',
                          'ml_masterType': 'complex_model_m_gpu',
                          'ml_parameterServerCount': 4,
                          'ml_parameterServerType': 'complex_model_m_gpu'}
                   }

# Google parameter
dataset_name = 'purchase_probability'

# Maximum number of parallel jobs we can launch
max_nb_parallel_jobs = 20

# Path directory in gs:
gs_dir_path = 'train_files'

# Lists of features

# Names of id column & timestamp columns
features_id_dates = ['id', 'event_timestamp']

features_num_event = ['product_qty', 'product_price']
features_to_custom_dummify_event = ['event_env']
features_event = features_num_event + features_to_custom_dummify_event + features_id_dates + \
                 ['FORMAT_DATE("%A",DATE(event_timestamp)) as week_day',
                  'FORMAT_DATE("%m",DATE(event_timestamp)) as month']

features_num_visit = ['duration', 'nb_pages', 'unique_pages', 'latitude', 'longitude']
features_to_custom_dummify_visit = ['device']
features_visit = features_num_visit + features_to_custom_dummify_visit

features_num_product = ['product_rating', 'product_nb_of_ratings']
features_to_embed_product = ['product_department', 'product_category', 'product_sub_category']
features_product = features_num_product + features_to_embed_product

# Features to dummify
# If values are added, we should concatenate them in the preprocessing (after dict_dummy creation)
features_to_dummify = ['week_day', 'month']

# Features to hash (we don't dummify it because the number of distinct values can change)
# If values are added, we should concatenate them in the preprocessing (after dict_dummy creation)
features_to_custom_dummify = features_to_custom_dummify_event + features_to_custom_dummify_visit
# We fix max number of event_env to 10 max number of device to 10
n_values_custom_dummies = dict(zip(features_to_custom_dummify, [10, 10]))


# Define features to embed
features_to_embed = features_to_embed_product

# Names of numerical columns
features_numerical = features_num_event + features_num_visit + features_num_product


# Features we create during the preprocessing
features_engineered = ['delta_time']

# Features to select from bigquery
# All features except engineered features during preprocessing,
# and to_dummify features (we create these values manually)
# We are not taking into account ['product_id' 'product_name', 'store_type', 'store_id'] for now
selected_columns = features_id_dates + features_numerical + features_to_custom_dummify + \
                   features_to_embed + ['FORMAT_DATE("%A",DATE(event_timestamp)) as week_day',
                                        'FORMAT_DATE("%m",DATE(event_timestamp)) as month']

selected_columns = ', '.join(selected_columns)


# Define how many days of history we consider (for a specific id)
length_history = 91

# Define how many days forward we want to predict
forward_prediction = 28

training_history = 210 # in days

##################
# RNN Parameters
##################

# Max step training
max_step_training = 200000

# Input Parameters

# Number of distinct outputs
NUM_CLASSES = 2

# Size of vocabulary for embedded features
VOCABULARY_SIZE = {'product_department': 100,
                   'product_category': 1000,
                   'product_sub_category': 10000}

# Define the bucket boundaries
bucket_boundaries = [0, 10, 25, 45, 75, 120, 200]

# Buffer size (how many example we read from files before launching the RNN pipeline)
buffer_size = 1200


# Hyperparameters
# Number of input sequences feed into the RNN
BATCH_SIZE = 161
# Number of epochs
NUM_EPOCHS = 1
# LSTM parameters
NUM_HIDDEN = 124
# Embedding parameters
EMBEDDING_SIZE = [10, 30, 100]


# Set model params: dict of params
model_params = {'num_classes': NUM_CLASSES,
                'vocabulary_sizes': VOCABULARY_SIZE
                }

hp_params = {'batch_size': BATCH_SIZE,
             'learning_rate': 1e-6,
             'embedding_size_1': EMBEDDING_SIZE[0],
             'embedding_size_2': EMBEDDING_SIZE[1],
             'embedding_size_3': EMBEDDING_SIZE[2],
             'num_epoch': NUM_EPOCHS,
             'num_hidden': NUM_HIDDEN
             }
