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

def shuffle_data_pair(mydata1, mydata2):
    assert len(mydata1) == len(mydata2), "the length of each data in the data pair must be the same"
    mydata1_array = np.array(mydata1)
    mydata2_array = np.array(mydata2)
    idx = np.arange(len(mydata1_array))
    random.shuffle(idx)
    
    return list(mydata1_array[idx]), list(mydata2_array[idx])

def build_stratified_batch(eye_list, label_list, batch_size):
    
    late_amd_eye_list = []
    non_late_amd_eye_list = []
    late_amd_label_list = []
    non_late_amd_label_list = []
    
    for eyes, labels in zip(eye_list, label_list):
        
        if labels[-1] == 1:
            late_amd_eye_list.append(eyes)
            late_amd_label_list.append(labels)
        else:
            non_late_amd_eye_list.append(eyes)
            non_late_amd_label_list.append(labels)
            
    late_amd_eye_list, late_amd_label_list = shuffle_data_pair(late_amd_eye_list, late_amd_label_list)
    non_late_amd_eye_list, non_late_amd_label_list = shuffle_data_pair(non_late_amd_eye_list, non_late_amd_label_list)
    
    non_late_amd_batch = int(np.ceil(len(non_late_amd_eye_list)/(batch_size-1)))
    
    total_batch = np.max([len(late_amd_eye_list), non_late_amd_batch])
    
    if total_batch > len(late_amd_eye_list):
        addition_length = total_batch - len(late_amd_eye_list)
        late_amd_eye_list.extend(late_amd_eye_list[:addition_length])
        late_amd_label_list.extend(late_amd_label_list[:addition_length])
        
    elif total_batch > int(np.ceil(len(non_late_amd_eye_list)/(batch_size-1))):
        addition_length = int((total_batch - np.ceil(len(non_late_amd_eye_list)/(batch_size-1))) * 31)
        non_late_amd_eye_list.extend(non_late_amd_eye_list[:addition_length])
        non_late_amd_label_list.extend(non_late_amd_label_list[:addition_length])
    
    stratified_eye_list = []
    stratified_label_list = []
    
    for i in range(total_batch):
        non_late_amd_eyes = non_late_amd_eye_list[(batch_size-1)*i:(batch_size-1)*(i+1)]
        non_late_amd_labels = non_late_amd_label_list[(batch_size-1)*i:(batch_size-1)*(i+1)]
        late_amd_eyes = late_amd_eye_list[i:(i+1)]
        late_amd_labels = late_amd_label_list[i:(i+1)]
        
        stratified_eye_list.extend(non_late_amd_eyes)
        stratified_label_list.extend(non_late_amd_labels)
        stratified_eye_list.extend(late_amd_eyes)
        stratified_label_list.extend(late_amd_labels)
    
    return stratified_eye_list, stratified_label_list

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

def prepare_batch_segment(eye_list_batch, label_list_batch, feature_dict, config):

    max_len = 0
    
    # find max length of the batch
    for eye_list in eye_list_batch:

        if len(eye_list) > max_len:
            max_len = len(eye_list)

    batch_x_tensor = np.zeros(shape=(len(eye_list_batch), max_len, config["feature_dim"]))
    batch_y_tensor = np.zeros(shape=(len(label_list_batch), max_len))
    mask = np.zeros(shape=(len(label_list_batch), max_len))
    batch_segment_tensor = np.zeros(shape=(len(eye_list_batch), max_len))
    
    for b, eye_list in enumerate(eye_list_batch):
        
        for i, eye in enumerate(eye_list):
            batch_x_tensor[b, i, :] = feature_dict[eye]

    for b, label_list in enumerate(label_list_batch):

        batch_y_tensor[b, :len(label_list)] = label_list
        mask[b, len(label_list):] = 1. # masked value=1.0

    if config["use_segment_embedding"] == "bisegment":
        for i in range(max_len):
            if i % 2 != 0:
                batch_segment_tensor[:,i] = 1.
    elif config["use_segment_embedding"] == "separate":
        for i in range(max_len):
            batch_segment_tensor[:,i] = i

    mask = tf.reshape(mask, shape=(mask.shape[0], 1, 1, mask.shape[1]))

    return tf.cast(batch_x_tensor, dtype=tf.float32), tf.cast(batch_y_tensor, dtype=tf.float32), tf.cast(mask, dtype=tf.float32), tf.cast(batch_segment_tensor, dtype=tf.float32)

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

def flatten_with_last(y_batch, y_hat, mask):
    """
    y_batch: (batch_size, max_seq_len)
    y_hat: (batch_size, max_seq_len, 1)
    mask: (batch_size, 1, 1, max_seq_len)
    """
    mask = tf.reshape(mask, shape=(mask.shape[0], mask.shape[-1]))
    max_len = mask.shape[1]
    last_ind = max_len - (tf.reduce_sum(mask, axis=1) + 1.)
    last_ind = tf.cast(tf.reshape(last_ind, shape=(last_ind.shape[0], 1)), tf.int32)

    y_hat = tf.reshape(y_hat, shape=(y_hat.shape[0], y_hat.shape[1]))
    y_batch_flatten = tf.reshape(tf.gather(y_batch, last_ind, axis=1, batch_dims=1), shape=-1)
    y_hat_flatten = tf.reshape(tf.gather(y_hat, last_ind, axis=1, batch_dims=1), shape=-1)

    return y_batch_flatten, y_hat_flatten

def flatten_with_token(y_batch, y_hat, mask):
    """
    y_batch: (batch_size, max_seq_len)
    y_hat: (batch_size, max_seq_len, 1)
    mask: (batch_size, 1, 1, max_seq_len)
    """
    mask = tf.reshape(mask, shape=(mask.shape[0], mask.shape[-1]))
    max_len = mask.shape[1]
    last_ind = max_len - (tf.reduce_sum(mask, axis=1) + 1.)
    last_ind = tf.cast(tf.reshape(last_ind, shape=(last_ind.shape[0], 1)), tf.int32)

    y_hat = tf.reshape(y_hat, shape=(y_hat.shape[0], y_hat.shape[1]))
    y_batch_flatten = tf.reshape(tf.gather(y_batch, last_ind, axis=1, batch_dims=1), shape=-1) # the last label after removing masked label
    y_hat_flatten = tf.reshape(y_hat[:,0], shape=-1) # the first output is for pred token

    return y_batch_flatten, y_hat_flatten

def calculate_auc(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        label_list_batch = label_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch, mask = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)

        y_hat = model(x_batch, training=False, mask=mask) # (batch_size, max_seq_len, 1)
        y_batch_flatten, y_hat_flatten = flatten_with_last(y_batch, y_hat, mask)
        AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

    return AUC.result().numpy()

def calculate_auc_token(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        label_list_batch = label_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch, mask = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)

        y_hat = model(x_batch, training=False, mask=mask) # (batch_size, max_seq_len, 1)
        y_batch_flatten, y_hat_flatten = flatten_with_token(y_batch, y_hat, mask)
        AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

    return AUC.result().numpy()

def calculate_auc_segment(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        label_list_batch = label_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch, mask, segment_batch = prepare_batch_segment(eye_list_batch, label_list_batch, feature_dict, config)
        y_hat = model(x_batch, segment_batch, training=False, mask=mask) # (batch_size, max_seq_len, 1)
        y_batch_flatten, y_hat_flatten = flatten_with_last(y_batch, y_hat, mask)
        AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

    return AUC.result().numpy()

def calculate_auc_perlength(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    result_dict = dict()
    batch_size = config["batch_size"]

    for length, length_dict in dataset_dict.items():

        this_length_eye_list = length_dict["eye_list"]
        this_length_label_list = length_dict["label_list"]
        num_batch = int(np.ceil(float(len(this_length_eye_list)) / float(batch_size)))
        prediction_count = 0

        AUC.reset_states()

        for i in range(num_batch):
            eye_list_batch = this_length_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = this_length_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch, mask = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)
            y_hat = model(x_batch, training=False, mask=mask) # (batch_size, length, 1)
            y_batch_flatten, y_hat_flatten = flatten_with_last(y_batch, y_hat, mask)
            prediction_count += y_hat_flatten.shape[0]
            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

        result_dict[length] = {"AUC" : AUC.result().numpy(), "prediction_count" : prediction_count}

    return result_dict

def calculate_auc_perlength_token(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    result_dict = dict()
    batch_size = config["batch_size"]

    for length, length_dict in dataset_dict.items():

        this_length_eye_list = length_dict["eye_list"]
        this_length_label_list = length_dict["label_list"]
        num_batch = int(np.ceil(float(len(this_length_eye_list)) / float(batch_size)))
        prediction_count = 0

        AUC.reset_states()

        for i in range(num_batch):
            eye_list_batch = this_length_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = this_length_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch, mask = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)
            y_hat = model(x_batch, training=False, mask=mask) # (batch_size, length, 1)
            y_batch_flatten, y_hat_flatten = flatten_with_token(y_batch, y_hat, mask)
            prediction_count += y_hat_flatten.shape[0]
            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

        result_dict[length] = {"AUC" : AUC.result().numpy(), "prediction_count" : prediction_count}

    return result_dict

def calculate_auc_perlength_segment(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    result_dict = dict()
    batch_size = config["batch_size"]

    for length, length_dict in dataset_dict.items():

        this_length_eye_list = length_dict["eye_list"]
        this_length_label_list = length_dict["label_list"]
        num_batch = int(np.ceil(float(len(this_length_eye_list)) / float(batch_size)))
        prediction_count = 0

        AUC.reset_states()

        for i in range(num_batch):
            eye_list_batch = this_length_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = this_length_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch, mask, segment_batch = prepare_batch_segment(eye_list_batch, label_list_batch, feature_dict, config)
            y_hat = model(x_batch, segment_batch, training=False, mask=mask) # (batch_size, length, 1)
            y_batch_flatten, y_hat_flatten = flatten_with_last(y_batch, y_hat, mask)
            prediction_count += y_hat_flatten.shape[0]
            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

        result_dict[length] = {"AUC" : AUC.result().numpy(), "prediction_count" : prediction_count}

    return result_dict

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, class_weight):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.class_weight = class_weight
        self.eps = 10e-07 
    
    def call(self, y_true, y_pred):
        """
        y_true and y_pred must be flattened
        """
        # Clipping y_pred for numerical stability 
        y_pred = tf.clip_by_value(y_pred, self.eps, 1-self.eps)

        loss_for_true = tf.negative(tf.multiply(y_true, tf.math.log(y_pred)) * self.class_weight[1])
        loss_for_false = tf.negative(tf.multiply(1. - y_true, tf.math.log(1.0 - y_pred)) * self.class_weight[0])
        weighted_loss_sum = tf.add(loss_for_true, loss_for_false)
        
        return tf.reduce_mean(weighted_loss_sum, axis=0)