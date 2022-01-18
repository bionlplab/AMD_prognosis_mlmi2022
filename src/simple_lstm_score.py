from numpy.lib.npyio import save
import tensorflow as tf
import numpy as np
import time
import os
import random
from utils.utils import *

class SimpleLSTM(tf.keras.Model):
    def __init__(self, config):
        super(SimpleLSTM, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        self.masking_layer = tf.keras.layers.Masking(mask_value=-1.0, name="masking_layer")
        self.lstm = tf.keras.layers.LSTM(units=config["lstm_unit"], return_sequences=True, name="lstm")
        self.concatenation_layer = tf.keras.layers.Concatenate(axis=-1, name="concatenation")
        self.prediction_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.relu, kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))

    def call(self, x, d):
        """
        --x: batch * max_length * feature_size
        --d: batch * feature
        """

        x = self.masking_layer(x)
        x = self.lstm(x)

        return self.prediction_layer(self.concatenation_layer([x, d]))

def compute_loss(model, x, d, y):

    y_hat = model(x, d)
    y_hat = tf.reshape(y_hat, shape=(-1))
    tf_mask = tf.reshape((y != -1.), shape=(-1))
    y = tf.reshape(y, shape=(-1))

    y_hat = y_hat[tf_mask]
    y = y[tf_mask]

    return tf.sqrt(tf.reduce_mean(tf.math.pow((y - y_hat), 2)))

def calculate_rmse(model, test_x, test_d, test_y, config, tracking=False):
    batch_size = config["batch_size"]
    num_batch = int(np.ceil(float(len(test_x)) / float(batch_size)))
    tracking_prediction = []
    tracking_true = []
    rmse = []
    
    for t in range(num_batch):
        x = test_x[t * batch_size:(t+1) * batch_size]
        d = test_d[t * batch_size:(t+1) * batch_size]
        y = test_y[t * batch_size:(t+1) * batch_size]

        x, d, y = pad_data(x, d, y, config)
        y_hat = model(x, d)
        y_hat = tf.reshape(y_hat, shape=(-1))
        tf_mask = tf.reshape((y != -1.), shape=(-1))
        y = tf.reshape(y, shape=(-1))

        y_hat = y_hat[tf_mask]
        y = y[tf_mask]
        rmse_batch = tf.sqrt(tf.reduce_mean(tf.math.pow((y - y_hat), 2)))
        rmse.append(rmse_batch.numpy())
        tracking_prediction.append(y_hat)
        tracking_true.append(y)

    if tracking:
        return np.mean(rmse), tracking_true, tracking_prediction
    else:
        return np.mean(rmse)

def calculate_mad(model, test_x, test_d, test_y, config, tracking=False):
    """
    calculate mean absolute difference
    """
    batch_size = config["batch_size"]
    num_batch = int(np.ceil(float(len(test_x)) / float(batch_size)))
    tracking_prediction = []
    tracking_true = []
    mad = []
    
    for t in range(num_batch):
        x = test_x[t * batch_size:(t+1) * batch_size]
        d = test_d[t * batch_size:(t+1) * batch_size]
        y = test_y[t * batch_size:(t+1) * batch_size]

        x, d, y = pad_data(x, d, y, config)
        y_hat = model(x, d)
        y_hat = tf.reshape(y_hat, shape=(-1))
        tf_mask = tf.reshape((y != -1.), shape=(-1))
        y = tf.reshape(y, shape=(-1))

        y_hat = y_hat[tf_mask]
        y = y[tf_mask]
        mad_batch = tf.reduce_mean(tf.math.abs((y - y_hat)))
        mad.append(mad_batch.numpy())
        tracking_prediction.append(y_hat)
        tracking_true.append(y)

    if tracking:
        return np.mean(mad), tracking_true, tracking_prediction
    else:
        return np.mean(mad)

def train_rnn_kfold(output_path, data_path, max_epoch, batch_size, lstm_unit, cfp_feature_size, clinical_feature_size,
                    l2_reg=0.001, learning_rate=0.001, k=5):

    config = locals().copy()

    print("load data...")
    amd_eye_data_dict = load_data(data_path)

    print("split the dataset into k-fold...")
    clinical_feature_chunks, cfp_feature_chunks, severe_score_chunks = data_to_chunk(amd_eye_data_dict, k)
    
    kfold_training_loss = []
    kfold_validation_mad = []
    kfold_test_mad = []
    kfold_time_elapsed = []
    kfold_test_prediction = []
    kfold_test_true = []
    
    for i in range(k):
        print("build and initialize model for {k}-th fold...".format(k=i+1))
        simple_lstm = SimpleLSTM(config)
    
        train_x, valid_x, test_x = flatten_list(cfp_feature_chunks[i:i+3]), cfp_feature_chunks[i+4], cfp_feature_chunks[i+5]
        train_d, valid_d, test_d = flatten_list(clinical_feature_chunks[i:i+3]), clinical_feature_chunks[i+4], clinical_feature_chunks[i+5]
        train_y, valid_y, test_y = flatten_list(severe_score_chunks[i:i+3]), severe_score_chunks[i+4], severe_score_chunks[i+5]
        
        training_loss = []
        validation_mad = []
        time_elapsed = []
        best_mad = np.inf
        best_epoch = 0
        best_model = None
        num_batch = int(np.ceil(float(len(train_x)) / float(batch_size)))
        
        print("start training...")
        for epoch in range(max_epoch):
            start_time = time.time()
            loss_record = []
            progbar = tf.keras.utils.Progbar(num_batch)
        
            for t in random.sample(range(num_batch), num_batch):
                x = train_x[t * batch_size:(t+1) * batch_size]
                d = train_d[t * batch_size:(t+1) * batch_size]
                y = train_y[t * batch_size:(t+1) * batch_size]
                x, d, y = pad_data(x, d, y, config)
                
                with tf.GradientTape() as tape:
                    batch_loss = compute_loss(simple_lstm, x, d, y)
                gradients = tape.gradient(batch_loss, simple_lstm.trainable_variables)
                simple_lstm.optimizer.apply_gradients(zip(gradients, simple_lstm.trainable_variables))
                
                loss_record.append(batch_loss.numpy())
                progbar.add(1)
                
            end_time = time.time()
            time_elapsed.append(end_time - start_time)
            training_loss.append(np.mean(loss_record))
            print('epoch:{e}, training loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
            
            current_mad = calculate_mad(simple_lstm, valid_x, valid_d, valid_y, config)
            print('epoch:{e}, validation MAD:{l:.6f}'.format(e=epoch+1, l=current_mad))
            validation_mad.append(current_mad)
            
            if current_mad < best_mad: 
                best_mad = current_mad
                best_epoch = epoch+1
                best_model = simple_lstm.get_weights()

        kfold_training_loss.append(training_loss)
        kfold_validation_mad.append(np.mean(validation_mad))
        kfold_time_elapsed.append(np.mean(time_elapsed))

        print('Best model of {k}-th fold: at epoch {e}, best model MAD:{l:.6f}'.format(k=i+1, e=best_epoch, l=best_mad))
        print("calculate MAD using the best model on the test set")
        simple_lstm.set_weights(best_model)
        test_mad, test_true, test_prediction = calculate_mad(simple_lstm, test_x, test_d, test_y, config, tracking=True)
        print("MAD of {k}-th fold: {auc:.6f}".format(k=i+1, auc=test_mad))
        kfold_test_mad.append(test_mad)
        kfold_test_prediction.append(test_prediction)
        kfold_test_true.append(test_true)

    print("saving results...")
    save_data(os.path.join(output_path, "config.pkl"), config)
    save_data(os.path.join(output_path, "training_loss.pkl"), kfold_training_loss)
    save_data(os.path.join(output_path, "validation_mad.pkl"), kfold_validation_mad)
    save_data(os.path.join(output_path, "test_mad.pkl"), kfold_test_mad)
    save_data(os.path.join(output_path, "time_elapsed.pkl"), kfold_time_elapsed)
    save_data(os.path.join(output_path, "test_prediction.pkl"), kfold_test_prediction)
    save_data(os.path.join(output_path, "test_true.pkl"), kfold_test_true)