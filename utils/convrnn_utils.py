import numpy as np
from numpy.lib.function_base import unwrap
import tensorflow as tf
import imgaug.augmenters as iaa
import cv2
import random
import pickle
from src.resnet_src import *

def count_by_unfold(list_batch):
    
    count = 0
    for elem in list_batch:
        count += len(elem)
    
    return count

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

def prepare_batch_lstm(eye_list_batch, label_list_batch, feature_dict, config):

    max_len = 0
    
    # find the max length of the batch
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

def flatten_with_mask(y_batch, y_hat):
    """
    y_hat must be masked
    remove all the masked elements
    y_batch: (batch_size, max_seq_len)
    y_hat: (batch_size, max_seq_len, 1)
    """

    mask =  y_hat._keras_mask
    y_hat = tf.reshape(y_hat, shape=(y_batch.shape[0], y_batch.shape[1]))
    y_hat_flatten = tf.boolean_mask(y_hat, mask)
    y_batch_flatten = tf.boolean_mask(y_batch, mask)

    return tf.cast(y_batch_flatten, dtype=tf.float32), tf.cast(y_hat_flatten, dtype=tf.float32)

def flatten_with_last(y_batch, y_hat):
    """
    y_hat must be masked
    remove the elements except the very last element in each sequence
    y_batch: (batch_size, max_seq_len)
    y_hat: (batch_size, max_seq_len, 1)
    """
    mask = y_hat._keras_mask.numpy()
    mask_ind = np.sum(mask, axis=-1) - 1
    mask_onehot = tf.one_hot(mask_ind, depth=mask.shape[1])

    y_hat = tf.reshape(y_hat, shape=(y_batch.shape[0], y_batch.shape[1]))
    masked_y_hat = tf.multiply(y_hat, mask_onehot)
    masked_y_batch = tf.multiply(y_batch, mask_onehot)
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

def calculate_auc_lstm(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))
    AUC.reset_states()

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        label_list_batch = label_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch = prepare_batch_lstm(eye_list_batch, label_list_batch, feature_dict, config)

        y_hat = model(x_batch, training=False)
        y_batch_flatten, y_hat_flatten = flatten_with_last(y_batch, y_hat)
        AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

    return AUC.result().numpy()

def calculate_auc_lstm_auxlabel(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))
    AUC.reset_states()

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        label_list_batch = label_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch = prepare_batch_lstm(eye_list_batch, label_list_batch, feature_dict, config)

        y_hat = model(x_batch, training=False)
        y_batch_flatten, y_hat_flatten = flatten_with_mask(y_batch, y_hat)
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

def calculate_auc_lstm_perlength(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    sequential_last_prediction_num = 0
    sequential_last_point_num = 0
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
            x_batch, y_batch = prepare_batch_lstm(eye_list_batch, label_list_batch, feature_dict, config)
            sequential_last_point_num += y_batch.shape[0]
            y_hat = model(x_batch, training=False)
            y_batch_flatten, y_hat_flatten = flatten_with_last(y_batch, y_hat)
            prediction_count += y_hat_flatten.shape[0]
            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

            result_dict[length] = {"AUC" : AUC.result().numpy(), "prediction_count" : prediction_count}

    return result_dict

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