import numpy as np
from numpy.lib.function_base import unwrap
import tensorflow as tf
import imgaug.augmenters as iaa
import cv2
import random
import pickle

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

def prepare_batch(eye_batch, dataset_dict, config, augmenter=None, one_hot=False):

    img_size = int(config["resizing"]*config["scale"])
    batch_x_tensor = np.zeros(shape=(len(eye_batch), img_size, img_size, 3))

    if one_hot:
        batch_y_tensor = np.zeros(shape=(len(eye_batch), config["num_class"]))
    else:
        batch_y_tensor = np.zeros(shape=(len(eye_batch), 1))

    for idx, eye in enumerate(eye_batch):
        eye_filename = config["data_root_path"] + eye.split(" ")[0] + "/" + eye
        eye_img = parse_image(eye_filename, resizing=config["resizing"], scale=config["scale"], augmenter=augmenter)
        batch_x_tensor[idx, :, :, :] = eye_img
        if one_hot:
            batch_y_tensor[idx, :] = tf.keras.utils.to_categorical(dataset_dict[eye] - 1, num_classes=config["num_class"])
        else:
            batch_y_tensor[idx, :] = dataset_dict[eye]
    
    batch_x_tensor =  tf.math.divide(batch_x_tensor, 255.)
    
    return tf.cast(batch_x_tensor, dtype=tf.float32), tf.cast(batch_y_tensor, dtype=tf.float32)

def prepare_batch_to_extract(eye_batch, config):

    img_size = int(config["resizing"]*config["scale"])
    batch_x_tensor = np.zeros(shape=(len(eye_batch), img_size, img_size, 3))

    for idx, eye in enumerate(eye_batch):
        eye_filename = config["data_root_path"] + eye.split(" ")[0] + "/" + eye
        eye_img = parse_image(eye_filename, resizing=config["resizing"], scale=config["scale"], augmenter=None)
        batch_x_tensor[idx, :, :, :] = eye_img
    
    batch_x_tensor =  tf.math.divide(batch_x_tensor, 255.)

    return batch_x_tensor

def update_feature_dict(extracted_feature_dict, eye_batch, extracted_features):
    """
    extracted_features: batch * feature_dim
    """
    update_dict = dict()

    for idx, eye in enumerate(eye_batch):
        this_feature = extracted_features[idx, :]
        update_dict[eye] = this_feature.numpy()

    extracted_feature_dict.update(update_dict)

    return extracted_feature_dict

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

class WeightedCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, class_weight):
        super(WeightedCategoricalCrossEntropy, self).__init__()
        self.class_weight = class_weight
        self.eps = 10e-07 
    
    def call(self, y_true, y_pred):
        
        # Clipping y_pred for numerical stability 
        y_pred = tf.clip_by_value(y_pred, self.eps, 1-self.eps)
        
        y_true = tf.multiply(self.class_weight, y_true)
        y_true = tf.cast(y_true, dtype=tf.float32)
        weighted_loss_tensor = tf.negative(tf.multiply(y_true, tf.math.log(y_pred)))

        return tf.reduce_mean(tf.reduce_sum(weighted_loss_tensor, axis=-1))

def calculate_metrics_binary(model, dataset_dict, pixel_data, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    WBCE = WeightedBinaryCrossEntropy(config["class_weight"])
    # WKL = tfa.losses.WeightedKappaLoss(num_classes=config["num_class"], weightage="quadratic")
    batch_size = config["batch_size"]
    num_batch = int(np.ceil(float(len(dataset_dict)) / float(batch_size)))
    data_set = list(dataset_dict.keys())
    AUC.reset_states()
    loss = []

    for i in range(num_batch):
        eye_batch = data_set[i*batch_size:(i+1)*batch_size]
        x, y = prepare_batch(eye_batch, dataset_dict, pixel_data, config, normalize=False, augment=False, one_hot=False)
        y_hat = model(x, training=False)
        AUC.update_state(y_true=y, y_pred=y_hat)
        loss.append(WBCE(y_true=y, y_pred=y_hat))

    return np.mean(loss), AUC.result().numpy()

def calculate_metrics_score(model, dataset_dict, pixel_data, config):

    ACC = tf.keras.metrics.TopKCategoricalAccuracy(k=config["k"])
    WCCE = WeightedCategoricalCrossEntropy(config["class_weight"])
    # WKL = tfa.losses.WeightedKappaLoss(num_classes=config["num_class"], weightage="quadratic")
    batch_size = config["batch_size"]
    num_batch = int(np.ceil(float(len(dataset_dict)) / float(batch_size)))
    data_set = list(dataset_dict.keys())
    ACC.reset_states()
    loss = []

    for i in range(num_batch):
        eye_batch = data_set[i*batch_size:(i+1)*batch_size]
        x, y = prepare_batch(eye_batch, dataset_dict, pixel_data, config, normalize=False, augment=False, one_hot=True)
        y_hat = model(x, training=False)
        ACC.update_state(y_true=y, y_pred=y_hat)
        loss.append(WCCE(y_true=y, y_pred=y_hat))

    return np.mean(loss), ACC.result().numpy()

def calculate_accuracy(model, dataset_dict, config):

    ACC = tf.keras.metrics.TopKCategoricalAccuracy(k=config["k"])
    batch_size = config["batch_size"]
    num_batch = int(np.ceil(float(len(dataset_dict)) / float(batch_size)))
    data_set = list(dataset_dict.keys())
    ACC.reset_states()

    for i in range(num_batch):
        eye_batch = data_set[i*batch_size:(i+1)*batch_size]
        x, y = prepare_batch(eye_batch, dataset_dict, config, augment=False, one_hot=True)
        y_hat = model(x, training=False)
        ACC.update_state(y_true=y, y_pred=y_hat)

    return ACC.result().numpy()

def calculate_auc(model, dataset_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    num_batch = int(np.ceil(float(len(dataset_dict)) / float(batch_size)))
    data_set = list(dataset_dict.keys())
    AUC.reset_states()

    for i in range(num_batch):
        eye_batch = data_set[i*batch_size:(i+1)*batch_size]
        x, y = prepare_batch(eye_batch, dataset_dict, config, augmenter=None, one_hot=False)
        y_hat = model(x, training=False)
        AUC.update_state(y_true=y, y_pred=y_hat)

    return AUC.result().numpy()

def calculate_metrics_pertime(model, dataset_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    TP = tf.keras.metrics.TruePositives()
    TN = tf.keras.metrics.TrueNegatives()
    FP = tf.keras.metrics.FalsePositives()
    FN = tf.keras.metrics.FalseNegatives()
    result_dict = dict()
    batch_size = config["batch_size"]

    for year, year_dict in dataset_dict.items():

        AUC.reset_states()
        TP.reset_states()
        TN.reset_states()
        FP.reset_states()
        FN.reset_states()
        num_batch = int(np.ceil(float(len(year_dict)) / float(batch_size)))
        data_set = list(year_dict.keys())

        if len(data_set) > 0:
            
            LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            NON_LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            
            for i in range(num_batch):
                eye_batch = data_set[i*batch_size:(i+1)*batch_size]
                x, y = prepare_batch(eye_batch, year_dict, config, augment=False, one_hot=False)
                y_hat = model(x, training=False)
                AUC.update_state(y_true=y, y_pred=y_hat)
                TP.update_state(y_true=y, y_pred=y_hat)
                TN.update_state(y_true=y, y_pred=y_hat)
                FP.update_state(y_true=y, y_pred=y_hat)
                FN.update_state(y_true=y, y_pred=y_hat)
                LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=y)
                NON_LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=1.-y)

            result_dict[year] = {"AUC" : AUC.result().numpy(), 
                                "TP" : TP.result().numpy(),
                                "TN" : TN.result().numpy(),
                                "FP" : FP.result().numpy(),
                                "FN" : FN.result().numpy(),
                                "late_amd_score_mean" : LATE_AMD_SCORE_MEAN.result().numpy(),
                                "non_late_amd_score_mean" : NON_LATE_AMD_SCORE_MEAN.result().numpy()}
        else:
            continue

    return result_dict

def calculate_metrics_perlength(model, test_set_dict, per_length_test_set_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    result_dict = dict()
    batch_size = config["batch_size"]

    for length, length_dict in per_length_test_set_dict.items():

        num_batch = int(np.ceil(float(len(length_dict)) / float(batch_size)))
        data_list = list(length_dict.keys())

        AUC.reset_states()
        prediction_count = 0
        LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
        NON_LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
            
        for i in range(num_batch):
            eye_batch = data_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch = prepare_batch(eye_batch, length_dict, config, augment=False, one_hot=False)
            prediction_count += x_batch.shape[0]
            y_hat = model(x_batch, training=False)
            AUC.update_state(y_true=y_batch, y_pred=y_hat)
            LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=y_batch)
            NON_LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=1.-y_batch)

        result_dict[length] = {"AUC" : AUC.result().numpy(),
        "late_amd_score_mean" : LATE_AMD_SCORE_MEAN.result().numpy(),
        "non_late_amd_score_mean" : NON_LATE_AMD_SCORE_MEAN.result().numpy(),
        "prediction_count" : prediction_count}

    num_batch = int(np.ceil(float(len(test_set_dict)) / float(batch_size)))
    data_list = list(test_set_dict.keys())

    AUC.reset_states()
    LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
    NON_LATE_AMD_SCORE_MEAN = tf.keras.metrics.Mean()
    prediction_count = 0

    for i in range(num_batch):
        eye_batch = data_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch = prepare_batch(eye_batch, test_set_dict, config, augment=False, one_hot=False)
        prediction_count += x_batch.shape[0]
        y_hat = model(x_batch, training=False)
        AUC.update_state(y_true=y_batch, y_pred=y_hat)
        LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=y_batch)
        NON_LATE_AMD_SCORE_MEAN.update_state(y_hat, sample_weight=1.-y_batch)

    result_dict["overall"] = {"AUC" : AUC.result().numpy(), 
                            "late_amd_score_mean" : LATE_AMD_SCORE_MEAN.result().numpy(),
                            "non_late_amd_score_mean" : NON_LATE_AMD_SCORE_MEAN.result().numpy(),
                            "prediction_count" : prediction_count}
    
    return result_dict
