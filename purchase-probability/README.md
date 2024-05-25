## Legacy Python and TensorFlow versions on Cloud AI Platform

**A Runtime Version will be supported for one year from the date that it is released for AI Platform Training and Prediction. You can find the release date in the Runtime Version documentation. The support policy will be enforced starting January 13, 2020.
**


## Recurrent Neural Network to predict purchase probabilities

#### Features

Preprocess data, train an RNN LSTM model, and predict purchase
probabilities 7 days ahead.

One preprocessed file per day, training every 7 days.

#### Prerequisite:
* At least 14 days of past history are needed,
* Sequences with 90 days of past history maximum,
* Preprocessing can be done for every day before the current date,
but labels are correct only for days before current date minus 7 days.
So training sequences are only the ones before current date minus 7 days (others are for prediction).


#### Quick start

* **Actions**: train, predict, or evaluate.

* **Train step**: should be the first one. Command line:

```python main.py train --account_id='account_number' --ds_nodash='date' ```

* **Predict step**: if a model has already been trained:

``` python main.py predict --account_id='account_number' --ds_nodash='date' ```

* **Test step (only internally)**: useful to see the precision of the model.

``` python main.py evaluate --account_id='account_number' --ds_nodash='date' ```

#### Params

* **date**: date of most recent available data (often date day - 1)

* **env**: 'local' (default) or 'cloud'.

    * If 'local': train or predict in a local env (even the preprocessing part)
    * If 'cloud': train or predict in the cloud with ml jobs (even the preprocessing part)

* **mode**: 'prod'(default) or 'test'.
    * if 'test': we test our workflow on one date and we do preprocessing in local
    * if 'prod': we execute our workflow on every available dates and preprocess is executed in ml jobs

* **mlmachine_size**: specify the typology of machine we will use on ml-engine. 3 possibilities:S, M or L.

* **conf**: see conf/purchase-probability.yaml


#### General workflow:

* Preliminary:
    * Retrieve the oldest date in the dataset;
    * Start to preprocess data only if we have 14 days of history;
    * Calcul first date with at least 14 days of history (dataset oldest date + 14 days): first_training_date
    * List of dates, 2 possibilities:
        * 'train': we work on days between first_training_date and date -7 (we need 7 days to be able to calculate labels). If 0 days: ABORT (not enough data)
        * 'predict': prediction on date (if model exists)
    * Export:
        * BQ → GS. Data export of dates we can work on (if we did not do it already);
        * Package → GS: push package purchase-probability on GS if it is not there yet.
   * For each date (see list of dates), preprocess data
* 'Train': if last training was done 7 days ago, do the training, otherwise pass.
* 'Predict': predict on date  

#### How to restart

If something is wrong (bad data, poor model, etc) and we want to start from scratch, we have to delete all data and model files in the corresponding Google Cloud Storage bucket.

#### Issues remaining

* Hyperparameters optimization
* Optimization of parameters to improve speed (prefetch, parallel calls, etc) ? Located in input_fn.
