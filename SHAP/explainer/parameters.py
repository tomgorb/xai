rnd_seed = 16

dump_perturbed_data = False

min_date = '20200303'

# Parameters of google platform

gpp = {'project_id': '',
       'dataset_id': '',
       'bucket_name': '',
       'directory': 'purchase-probability/train',
       'files_path': '/files',
       'model_path': '/model',
       'output_path': '/explainer'
       }

# Cloud ML parameters
ml_pythonVersion = '3.7'
ml_runtimeVersion = '1.15'
ml_region = 'europe-west1'

# Useful for preprocessing and predict, not for training. ml_scalerTier_train is used for training
# ml_scaleTier = 'BASIC'

typology_machine = {'S': {'ml_scaleTier': 'CUSTOM',
                          'ml_masterType': 'n1-standard-8',
                          'ml_workerType': 'large_model',
                          'ml_workerCount': 0,
                          'ml_parameterServerType': 'standard',
                          'ml_parameterServerCount': 0},
                    'M': {'ml_scaleTier': 'CUSTOM',
                          'ml_workerCount': 0,
                          'ml_workerType': 'n1-highmem-8',
                          'ml_masterType': 'n1-highmem-8',
                          'ml_parameterServerCount': 0,
                          'ml_parameterServerType': 'n1-highmem-8'}
                    }

# PARAMETERS FOR THE EXPLAINER

lstm_features = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'January', 'February',
                 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
                 'None_event', 'product_page', 'purchase', 'add_to_cart', 'event_Na_0', 'event_Na_1', 'event_Na_2',
                 'event_Na_3', 'event_Na_4', 'event_Na_5', 'None_device', 'tv_device', 'ebook_reader', 'tablet',
                 'other', 'mobile_device', 'desktop', 'mobile_phone', 'device_Na_0', 'device_Na_1', 'product_qty',
                 'product_price', 'visit_duration', 'visit_nb_pages', 'visit_unique_pages', 'Latitude',
                 'Longitude', 'product_rating', 'product_number_of_ratings', 'Delta_time']

# Number of samples

n_samples = 100

# m_id to explain

instance_to_explain = 10

# PARAMETERS FOR PREPROCESSING

new_feature_names = {'Numerical_features':
                         ['product_price', 'visit_duration', 'Latitude', 'Longitude', 'Delta_time'],
                     'Categorical_features':
                         ['week_day', 'month', 'event_env', 'device', 'product_qty', 'visit_nb_pages',
                          'visit_unique_pages', 'product_rating', 'product_number_of_ratings', 'product_department',
                          'product_category', 'product_sub_category'],
                     'Fixed_features':
                         ['inactivity']
                     }

mapping_features = {'num_to_num': ['product_price', 'visit_duration', 'Latitude', 'Longitude', 'Delta_time'],
                    'num_to_cat_encoded': ['week_day', 'month', 'event_env', 'device'],
                    'num_to_cat': ['product_qty', 'visit_nb_pages', 'visit_unique_pages', 'product_rating',
                                   'product_number_of_ratings'],
                    'cat_to_cat': ['product_department', 'product_category', 'product_sub_category'],
                    'qty': 'product_qty'}

encoded_num2cat_idx = {'week_day': [0, 7],
                       'month': [7, 19],
                       'event_env': [19, 29],
                       'device': [29, 39]}

label_category = {'week_day': ['PAD', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
                  'month': ['PAD', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                            'October', 'November', 'December'],
                  'event_env': ['PAD', 'None_event', 'product_page_view', 'purchase', 'add_to_cart', 'event_Na_0',
                                'event_Na_1', 'event_Na_2', 'event_Na_3', 'event_Na_4', 'event_Na_5'],
                  'device': ['PAD', 'None_device', 'tv_device', 'ebook_reader', 'tablet', 'other', 'mobile_device', 'desktop',
                             'mobile_phone', 'device_Na_0', 'device_Na_1']}
