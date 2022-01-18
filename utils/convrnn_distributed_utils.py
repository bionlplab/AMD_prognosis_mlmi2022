import tensorflow as tf
import numpy as np
import os
import pickle

def load_data(data_path):
    data = pickle.load(open(data_path, 'rb'))

    return data

def save_data(output_path, mydata):
    with open(output_path, 'wb') as f:
        
        pickle.dump(mydata, f)

def remove_hidden_file(listdir):

    newlistdir = []
    
    for dir in listdir:

        if dir.startswith('.'):
            continue
        else:
            newlistdir.append(dir)

    return newlistdir

def load_dataset_split(dataset_split_path, element_spec):
    dataset_split_list = os.listdir(dataset_split_path)
    dataset_split_list = remove_hidden_file(dataset_split_list)
    concatenated_dataset = None

    for idx, dataset_split in enumerate(dataset_split_list):

        ds = tf.data.experimental.load(os.path.join(dataset_split_path, dataset_split), element_spec=element_spec)

        if idx == 0:
            concatenated_dataset = ds
        else:
            concatenated_dataset = concatenated_dataset.concatenate(ds)

    return concatenated_dataset

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
    y_batch = tf.cast(y_batch, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    masking_index = np.sum(masking, axis=-1) - 1
    masking_onehot = tf.one_hot(masking_index, depth=masking.shape[1])
    y_hat = tf.reshape(y_hat, shape=(y_batch.shape[0], y_batch.shape[1]))
    masked_y_hat = tf.multiply(y_hat, masking_onehot)
    masked_y_batch = tf.multiply(y_batch, masking_onehot)
    y_hat_flatten = tf.reduce_sum(masked_y_hat, axis=-1)
    y_batch_flatten = tf.reduce_sum(masked_y_batch, axis=-1)
    
    return y_batch_flatten, y_hat_flatten

def train_step(model, optimizer, loss_object, auc_object, inputs, global_batch_size):
    x_batch, y_batch = inputs
    x_batch = x_batch / 255.

    with tf.GradientTape() as tape:
        y_hat = model(x_batch, training=True)
        loss = compute_loss(loss_object, y_batch, y_hat, global_batch_size)

    gradients = tape.gradient(loss, model.trainable_variables)
    y_batch_flatten, y_hat_flatten = apply_sequential_mask(y_batch, y_hat)
    auc_object.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss 

def evaluation_step(model, auc_object, inputs):
    x_batch, y_batch = inputs
    x_batch = x_batch / 255.

    y_hat = model(x_batch, training=False)
    y_batch_flatten, y_hat_flatten = apply_sequential_mask(y_batch, y_hat)
    auc_object.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

def compute_loss(loss_object, y_batch, y_hat, global_batch_size):
    
    per_replica_loss = loss_object(y_true=y_batch, y_pred=y_hat)
    return tf.nn.compute_average_loss(per_replica_loss, global_batch_size=global_batch_size)

@tf.function
def distributed_train_step(strategy, model, optimizer, loss_object, auc_object, global_input, global_batch_size):
    per_replica_losses = strategy.run(train_step, args=(model, optimizer, loss_object, auc_object, global_input, global_batch_size,))
    
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def distributed_evaluation_step(strategy, model, auc_object, global_input):
  return strategy.run(evaluation_step, args=(model, auc_object, global_input,))

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, class_weight, **kwargs):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.class_weight = tf.cast(np.reshape(class_weight, newshape=(2,1)), tf.float32)
        self.eps = 10e-07 
        self.reduction = tf.keras.losses.Reduction.NONE
    
    def call(self, y_true, y_pred):  
        # Clipping y_pred for numerical stability 
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.reshape(y_pred, shape=(y_true.shape[0], y_true.shape[1])) # make the shape of y_true and y_pred the same
        y_pred = tf.clip_by_value(y_pred, self.eps, 1-self.eps)

        loss_for_true = tf.negative(tf.reshape(tf.multiply(y_true, tf.math.log(y_pred)), shape=(1,-1)) * self.class_weight[1])
        loss_for_false = tf.negative(tf.reshape(tf.multiply(1. - y_true, tf.math.log(1.0 - y_pred)), shape=(1,-1)) * self.class_weight[0])
        weighted_loss = tf.add(loss_for_true, loss_for_false)
        
        return weighted_loss