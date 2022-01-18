import numpy as np
from numpy.lib.function_base import unwrap
import tensorflow as tf
import imgaug.augmenters as iaa
import cv2
import random
import pickle
from src.resnet_src import *

def shuffle_data(mydata):
    mydata = np.array(mydata)
    idx = np.arange(len(mydata))
    random.shuffle(idx)
    
    return mydata[idx]

def load_data(data_path):
    data = pickle.load(open(data_path, 'rb'))

    return data

def save_data(output_path, mydata):
    with open(output_path, 'wb') as f:
        
        pickle.dump(mydata, f)

def parse_image(filename, resizing, scale, augmenter=None):
    
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image)
    image = image.numpy()

    if augmenter != None:
        image = augmenter(image=image)

    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)
    image = tf.image.resize(image, (resizing, resizing)) # 256 * 256
    image = tf.image.central_crop(image, scale)
    image = tf.cast(image, tf.uint8)
    
    return image

def set_augmenter():
    
    augmenter = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips with 0.5 probability 
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, 
        rotate=(-5, 5), shear=(-2, 2)
        )])

    return augmenter

def prepare_batch_resnet(eye_batch, dataset_dict, config, augment, one_hot=False):

    img_size = int(config["resizing"]*config["scale"])
    batch_x_tensor = np.zeros(shape=(len(eye_batch), img_size, img_size, 3))

    if augment:
        augmeter = set_augmenter()
    else:
        augmeter = None

    if one_hot:
        batch_y_tensor = np.zeros(shape=(len(eye_batch), config["num_class"]))
    else:
        batch_y_tensor = np.zeros(shape=(len(eye_batch), 1))

    for idx, eye in enumerate(eye_batch):
        eye_filename = config["data_root_path"] + eye.split(" ")[0] + "/" + eye
        eye_img = parse_image(eye_filename, resizing=config["resizing"], scale=config["scale"], augmenter=augmeter)
        batch_x_tensor[idx, :, :, :] = eye_img
        if one_hot:
            batch_y_tensor[idx, :] = tf.keras.utils.to_categorical(dataset_dict[eye] - 1, num_classes=config["num_class"])
        else:
            batch_y_tensor[idx, :] = dataset_dict[eye]
    
    batch_x_tensor =  tf.math.divide(batch_x_tensor, 255.)
    
    return batch_x_tensor, batch_y_tensor

def prepare_batch_resnetlstm_module(eye_list_batch, label_list_batch, config, augment):
    
    max_len = 0
    img_size = int(config["resizing"]*config["scale"])
    
    # find the max length of the batch
    for eye_list in eye_list_batch:
        if len(eye_list) > max_len:
            max_len = len(eye_list)
            
    batch_x_tensor = np.zeros(shape=(len(eye_list_batch), max_len, img_size, img_size, 3))
    batch_y_tensor = np.zeros(shape=(len(eye_list_batch), max_len))
    
    if augment:
        augmeter = set_augmenter()
    else:
        augmeter = None

    for list_idx, eye_list in enumerate(eye_list_batch):
        
        for eye_idx, eye in enumerate(eye_list):
            eye_filename = config["data_root_path"] + eye.split(" ")[0] + "/" + eye
            eye_img = parse_image(eye_filename, resizing=config["resizing"], scale=config["scale"], augmenter=augmeter)
            batch_x_tensor[list_idx, eye_idx, :, :] = eye_img

    for idx, label_list in enumerate(label_list_batch):
        batch_y_tensor[idx, :len(label_list)] = np.array(label_list)

    batch_x_tensor =  tf.math.divide(batch_x_tensor, 255.)

    return batch_x_tensor, batch_y_tensor

def prepare_batch_lstm_module(eye_list_batch, feature_dict, config):

    max_len = 0
    
    # find max length of the batch
    for eye_list in eye_list_batch:
        if len(eye_list) > max_len:
            max_len = len(eye_list)

    batch_x_tensor = np.zeros(shape=(len(eye_list_batch), max_len, config["feature_dim"]))
    
    for b, eye_list in enumerate(eye_list_batch):
        
        for i, eye in enumerate(eye_list):
            batch_x_tensor[b, i, :] = feature_dict[eye]

    return batch_x_tensor

def prepare_batch_lstm_sequential_module(eye_list_batch, label_list_batch, feature_dict, config):

    max_len = 0
    
    # find max length of the batch
    for eye_list in eye_list_batch:

        if len(eye_list) > max_len:
            max_len = len(eye_list)

    batch_x_tensor = np.zeros(shape=(len(eye_list_batch), max_len, config["feature_dim"]))
    batch_y_tensor = np.zeros(shape=(len(label_list_batch), max_len))
    
    for b, eye_list in enumerate(eye_list_batch):
        
        for i, eye in enumerate(eye_list):
            batch_x_tensor[b, i, :] = feature_dict[eye]

    for b, label_list in enumerate(label_list_batch):

        batch_y_tensor[b, :len(label_list)] = label_list

    return batch_x_tensor, batch_y_tensor

def apply_mask(result_tensor):
    """
    result tensor must be masked
    """
    masking =  result_tensor._keras_mask
    masking_index = np.sum(masking, axis=-1) - 1
    masking_onehot = tf.one_hot(masking_index, depth=masking.shape[1])
    result_tensor = tf.reshape(result_tensor, shape=(result_tensor.shape[0], result_tensor.shape[1]))
    masked_result_tensor = tf.multiply(result_tensor, masking_onehot)
    masked_result_flatten = tf.reduce_sum(masked_result_tensor, axis=-1)
    
    return masked_result_flatten

def apply_sequential_mask(y_batch, y_hat):
    """
    y_hat must be masked
    remove masked elements and flatten tensors
    """
    mask =  y_hat._keras_mask
    y_hat = tf.reshape(y_hat, shape=(y_batch.shape[0], y_batch.shape[1]))
    y_hat_flatten = tf.boolean_mask(y_hat, mask)
    y_batch_flatten = tf.boolean_mask(y_batch, mask)

    return y_batch_flatten, y_hat_flatten

def apply_sequential_last_mask(y_batch, y_hat):
    """
    y_hat must be masked
    remove the elements except the very last element in each sequence
    """
    masking =  y_hat._keras_mask
    masking_index = np.sum(masking, axis=-1) - 1
    masking_onehot = tf.one_hot(masking_index, depth=masking.shape[1])
    y_hat = tf.reshape(y_hat, shape=(y_batch.shape[0], y_batch.shape[1]))
    masked_y_hat = tf.multiply(y_hat, masking_onehot)
    masked_y_batch = tf.multiply(y_batch, masking_onehot)
    y_hat_flatten = tf.reduce_sum(masked_y_hat, axis=-1)
    y_batch_flatten = tf.reduce_sum(masked_y_batch, axis=-1)
    
    return y_batch_flatten, y_hat_flatten

def calculate_auc(model, dataset_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))
    AUC.reset_states()

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        y_batch = np.array(label_list[i*batch_size:(i+1)*batch_size])
        x_batch, this_batch_size = prepare_batch(eye_list_batch, config, augment=False)
        y_hat = model(x_batch, this_batch_size, training=False)
        y_hat = apply_mask(y_hat)
        AUC.update_state(y_true=y_batch, y_pred=y_hat)

    return AUC.result().numpy()

def calculate_auc_lstm_module(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))
    AUC.reset_states()

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        y_batch = np.array(label_list[i*batch_size:(i+1)*batch_size])
        x_batch = prepare_batch_lstm_module(eye_list_batch, feature_dict, config)

        y_hat = model(x_batch, training=False)
        y_hat = apply_mask(y_hat)
        AUC.update_state(y_true=y_batch, y_pred=y_hat)

    return AUC.result().numpy()

def calculate_auc_lstm_sequential_module(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))
    AUC.reset_states()

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        label_list_batch = label_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch = prepare_batch_lstm_sequential_module(eye_list_batch, label_list_batch, feature_dict, config)

        y_hat = model(x_batch, training=False)
        y_batch_flatten, y_hat_flatten = apply_sequential_mask(y_batch, y_hat)
        AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

    return AUC.result().numpy()

def calculate_auc_resnetlstm_module(model, dataset_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))
    AUC.reset_states()

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        label_list_batch = label_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch = prepare_batch_resnetlstm_module(eye_list_batch, label_list_batch, config, augment=False)

        y_hat = model(x_batch, training=False)
        y_batch_flatten, y_hat_flatten = apply_sequential_mask(y_batch, y_hat)
        AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

    return AUC.result().numpy()

def calculate_auc_lstm_attention_module(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))
    AUC.reset_states()

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        y_batch = np.array(label_list[i*batch_size:(i+1)*batch_size])
        x_batch = prepare_batch_lstm_module(eye_list_batch, feature_dict, config)

        y_hat = model(x_batch, training=False)
        AUC.update_state(y_true=y_batch, y_pred=y_hat)

    return AUC.result().numpy()

def calculate_metrics_peryear(model, dataset_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    result_dict = dict()
    batch_size = config["batch_size"]

    for year, year_dict in dataset_dict.items():

        AUC.reset_states()
        this_year_eye_list = year_dict["eye_list"]
        this_year_label_list = year_dict["label_list"]
        num_batch = int(np.ceil(float(len(this_year_eye_list)) / float(batch_size)))

        if len(this_year_eye_list) > 0:
            
            LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            NON_LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            
            for i in range(num_batch):
                eye_list_batch = this_year_eye_list[i*batch_size:(i+1)*batch_size]
                y_batch = np.array(this_year_label_list[i*batch_size:(i+1)*batch_size])
                x_batch, this_batch_size = prepare_batch(eye_list_batch, config, augment=False)
                y_hat = model(x_batch, this_batch_size, training=True)
                y_hat = apply_mask(y_hat)
                AUC.update_state(y_true=y_batch, y_pred=y_hat)
                LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=y_batch)
                NON_LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=1.-y_batch)

            result_dict[year] = {"AUC" : AUC.result().numpy(), 
                                "late_amd_score_mean" : LATE_AMD_SCORE_MEAN.result().numpy(),
                                "non_late_amd_score_mean" : NON_LATE_AMD_SCORE_MEAN.result().numpy()}
        else:
            continue

    return result_dict

def calculate_metrics_peryear_lstm_module(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    result_dict = dict()
    batch_size = config["batch_size"]

    for year, year_dict in dataset_dict.items():

        AUC.reset_states()
        this_year_eye_list = year_dict["eye_list"]
        this_year_label_list = year_dict["label_list"]
        num_batch = int(np.ceil(float(len(this_year_eye_list)) / float(batch_size)))

        if len(this_year_eye_list) > 0:
            
            LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            NON_LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            
            for i in range(num_batch):
                eye_list_batch = this_year_eye_list[i*batch_size:(i+1)*batch_size]
                y_batch = np.array(this_year_label_list[i*batch_size:(i+1)*batch_size])
                x_batch = prepare_batch_lstm_module(eye_list_batch, feature_dict, config)
                y_hat = model(x_batch, training=False)
                y_hat = apply_mask(y_hat)
                AUC.update_state(y_true=y_batch, y_pred=y_hat)
                LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=y_batch)
                NON_LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=1.-y_batch)

            result_dict[year] = {"AUC" : AUC.result().numpy(), 
                                "late_amd_score_mean" : LATE_AMD_SCORE_MEAN.result().numpy(),
                                "non_late_amd_score_mean" : NON_LATE_AMD_SCORE_MEAN.result().numpy()}
        else:
            continue

    return result_dict

def calculate_metrics_peryear_lstm_attention_module(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    result_dict = dict()
    batch_size = config["batch_size"]

    for year, year_dict in dataset_dict.items():

        AUC.reset_states()
        this_year_eye_list = year_dict["eye_list"]
        this_year_label_list = year_dict["label_list"]
        num_batch = int(np.ceil(float(len(this_year_eye_list)) / float(batch_size)))

        if len(this_year_eye_list) > 0:
            
            LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            NON_LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            
            for i in range(num_batch):
                eye_list_batch = this_year_eye_list[i*batch_size:(i+1)*batch_size]
                y_batch = np.array(this_year_label_list[i*batch_size:(i+1)*batch_size])
                x_batch = prepare_batch_lstm_module(eye_list_batch, feature_dict, config)
                y_hat = model(x_batch, training=False)
                AUC.update_state(y_true=y_batch, y_pred=y_hat)
                LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=y_batch)
                NON_LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=1.-y_batch)

            result_dict[year] = {"AUC" : AUC.result().numpy(), 
                                "late_amd_score_mean" : LATE_AMD_SCORE_MEAN.result().numpy(),
                                "non_late_amd_score_mean" : NON_LATE_AMD_SCORE_MEAN.result().numpy()}
        else:
            continue

    return result_dict

def calculate_metrics_perlength(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    result_dict = dict()
    batch_size = config["batch_size"]

    for length, length_dict in dataset_dict.items():

        this_length_eye_list = length_dict["eye_list"]
        this_length_label_list = length_dict["label_list"]
        num_batch = int(np.ceil(float(len(this_length_eye_list)) / float(batch_size)))

        AUC.reset_states()
        LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
        NON_LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            
        for i in range(num_batch):
            eye_list_batch = this_length_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = this_length_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch = prepare_batch_lstm_sequential_module(eye_list_batch, label_list_batch, feature_dict, config)
            y_hat = model(x_batch, training=False)
            y_batch_flatten, y_hat_flatten = apply_sequential_last_mask(y_batch, y_hat)
            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

            LATE_AMD_SCORE_MEAN.update_state(y_hat_flatten, sample_weight=y_batch_flatten)
            NON_LATE_AMD_SCORE_MEAN.update_state(y_hat_flatten, sample_weight=1.-y_batch_flatten)

            result_dict[length] = {"AUC" : AUC.result().numpy(), 
                                "late_amd_score_mean" : LATE_AMD_SCORE_MEAN.result().numpy(),
                                "non_late_amd_score_mean" : NON_LATE_AMD_SCORE_MEAN.result().numpy()}

    return result_dict

def calculate_metrics_perlength_resnetlstm(model, dataset_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    result_dict = dict()
    batch_size = config["batch_size"]

    for length, length_dict in dataset_dict.items():

        this_length_eye_list = length_dict["eye_list"]
        this_length_label_list = length_dict["label_list"]
        num_batch = int(np.ceil(float(len(this_length_eye_list)) / float(batch_size)))

        AUC.reset_states()
        LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
        NON_LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            
        for i in range(num_batch):
            eye_list_batch = this_length_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = this_length_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch = prepare_batch_resnetlstm_module(eye_list_batch, label_list_batch, config, augment=False)
            y_hat = model(x_batch, training=False)
            y_batch_flatten, y_hat_flatten = apply_sequential_last_mask(y_batch, y_hat)
            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

            LATE_AMD_SCORE_MEAN.update_state(y_hat_flatten, sample_weight=y_batch_flatten)
            NON_LATE_AMD_SCORE_MEAN.update_state(y_hat_flatten, sample_weight=1.-y_batch_flatten)

            result_dict[length] = {"AUC" : AUC.result().numpy(), 
                                "late_amd_score_mean" : LATE_AMD_SCORE_MEAN.result().numpy(),
                                "non_late_amd_score_mean" : NON_LATE_AMD_SCORE_MEAN.result().numpy()}

    return result_dict

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, class_weight):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.class_weight = np.reshape(class_weight, newshape=(2,1))
        self.eps = 10e-07 
    
    def call(self, y_true, y_pred):  
        # Clipping y_pred for numerical stability 
        y_pred = tf.clip_by_value(y_pred, self.eps, 1-self.eps)

        loss_for_true = tf.reshape(tf.multiply(y_true, tf.math.log(y_pred)), shape=(1,-1))
        loss_for_false = tf.reshape(tf.multiply(1. - y_true, tf.math.log(1.0 - y_pred)), shape=(1,-1))
        loss_tensor = tf.concat([loss_for_true, loss_for_false], axis=-1)
        weighted_loss_tensor = tf.negative(tf.multiply(self.class_weight, loss_tensor))
        
        return tf.reduce_mean(tf.reduce_sum(weighted_loss_tensor, axis=0))

def load_pretrained_resent(config):

    pretrained_config = config["pretrained_config"]

    # load data
    data_dict = load_data(pretrained_config["data_dict_path"])
    pretrained_weights = np.load(config["pretrained_weights_path"], allow_pickle=True)
    validation_set_dict = data_dict["validation_set"]

    #"build and initialize ResNet

    if config["use_pretrain"] == "random":
        print("use pretrained weights based on random initialization...")
        resnet = ResNet(pretrained_config)
    if config["use_pretrain"] == "imagenet":
        print("use pretrained weights based on imagenet...")
        resnet = ResNetPretrained(pretrained_config)

    validation_set = list(validation_set_dict.keys())
    batch_size = config["batch_size"]
    eye_batch = validation_set[:batch_size]

    # initialize the model's weight
    x, y = prepare_batch_resnet(eye_batch, validation_set_dict, pretrained_config, augment=False, one_hot=False)
    _ = resnet(x, training=False, extraction=False)
    
    # load the best model's weight
    resnet.set_weights(pretrained_weights)

    return resnet

class WeightedBinaryCrossEntropySequential(tf.keras.losses.Loss):
    def __init__(self, class_weight):
        super(WeightedBinaryCrossEntropySequential, self).__init__()
        self.class_weight = class_weight
        self.eps = 10e-07 
    
    def call(self, y_true, y_pred):
        """
        y_true: (batch_size, max_len)
        """

        mask = y_pred._keras_mask
        onehot_mask = tf.where(mask == True, 1., 0.)

        y_pred = tf.clip_by_value(y_pred, self.eps, 1-self.eps) # clipping y_pred for numerical stability 
        y_pred = tf.reshape(y_pred, shape=(y_true.shape[0], y_true.shape[1])) # make the shape of y_true and y_pred the same

        loss_for_true = self.class_weight[1] * tf.multiply(y_true, tf.math.log(y_pred))
        loss_for_false = self.class_weight[0] * tf.multiply(1. - y_true, tf.math.log(1.0 - y_pred))

        loss_tensor = tf.negative(tf.add(loss_for_true, loss_for_false))
        loss_tensor = tf.multiply(loss_tensor, onehot_mask)
        avg_loss_tensor = tf.divide(tf.reduce_sum(loss_tensor, axis=-1), tf.reduce_sum(onehot_mask, axis=-1))

        return tf.reduce_mean(avg_loss_tensor)