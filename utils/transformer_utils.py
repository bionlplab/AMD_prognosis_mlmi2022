import tensorflow as tf
import numpy as np
import random
import pickle

def load_data(data_path):
    data = pickle.load(open(data_path, 'rb'))

    return data

def save_data(output_path, mydata):
    with open(output_path, 'wb') as f:
        
        pickle.dump(mydata, f)

def shuffle_data(mydata):
    mydata = np.array(mydata)
    idx = np.arange(len(mydata))
    random.shuffle(idx)
    
    return mydata[idx]

def prepare_batch(eye_list_batch, label_list_batch, feature_dict, config):

    max_len = 0
    
    # find max length of the batch
    for eye_list in eye_list_batch:

        if len(eye_list) > max_len:
            max_len = len(eye_list)

    batch_x_tensor = np.zeros(shape=(len(eye_list_batch), max_len, config["feature_dim"]))
    batch_y_tensor = np.zeros(shape=(len(label_list_batch), max_len))
    mask = np.zeros(shape=(len(label_list_batch), max_len))
    
    for b, eye_list in enumerate(eye_list_batch):
        
        for i, eye in enumerate(eye_list):
            batch_x_tensor[b, i, :] = feature_dict[eye]

    for b, label_list in enumerate(label_list_batch):

        batch_y_tensor[b, :len(label_list)] = label_list
        mask[b, len(label_list):] = 1. # masked value=1.0

    mask = tf.reshape(mask, shape=(mask.shape[0], 1, 1, mask.shape[1]))

    return tf.cast(batch_x_tensor, dtype=tf.float32), tf.cast(batch_y_tensor, dtype=tf.float32), tf.cast(mask, dtype=tf.float32)

def flatten_with_mask(y_batch, y_hat, mask):
    """
    y_batch: (batch_size, max_seq_len)
    y_hat: (batch_size, max_seq_len, 1)
    mask: (batch_size, 1, 1, max_seq_len)
    """
    mask = tf.reshape(mask, shape=-1)
    boolean_mask = tf.where(mask == 0, True, False)
    y_batch = tf.reshape(y_batch, shape=-1)
    y_hat = tf.reshape(y_hat, shape=-1)

    y_batch_flatten = tf.boolean_mask(y_batch, boolean_mask)
    y_hat_flatten = tf.boolean_mask(y_hat, boolean_mask)

    return y_batch_flatten, y_hat_flatten

def calculate_auc(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))
    AUC.reset_states()

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        label_list_batch = label_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch, mask = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)

        y_hat = model(x_batch, training=True, mask=mask) # (batch_size, max_seq_len, 1)
        y_batch_flatten, y_hat_flatten = flatten_with_mask(y_batch, y_hat, mask)
        AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

    return AUC.result().numpy()

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, class_weight):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.class_weight = class_weight
        self.eps = 10e-07 
    
    def call(self, y_true, y_pred):
        """
        y_true and y_pred must be flattened with mask
        """
        # Clipping y_pred for numerical stability 
        y_pred = tf.clip_by_value(y_pred, self.eps, 1-self.eps)

        loss_for_true = tf.negative(tf.multiply(y_true, tf.math.log(y_pred)) * self.class_weight[1])
        loss_for_false = tf.negative(tf.multiply(1. - y_true, tf.math.log(1.0 - y_pred)) * self.class_weight[0])
        weighted_loss_sum = tf.add(loss_for_true, loss_for_false)
        
        return tf.reduce_mean(weighted_loss_sum, axis=0)