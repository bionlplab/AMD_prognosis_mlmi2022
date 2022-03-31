import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
from src.resnet_src import *
from utils.convrnn_utils import *

class ResNetLSTM(tf.keras.Model):
    def __init__(self, config):
        super(ResNetLSTM, self).__init__()

        if config["use_pretrain"] == None:
            print("initialize with random weights...")
            config["prediction"] = "None"
            self.resnet = ResNet(config)
        else:
            self.resnet = load_pretrained_resent(config)
            self.resnet.fc.trainable = False # FC layer in the resnet will not be trained

        self.lstm = LSTM(config)
        self.batch_size = config["batch_size"]
        self.units = config["units"]
        self.masking_layer = tf.keras.layers.Masking(mask_value=0.0)
        self.fc = tf.keras.layers.Dense(units=config["units"], activation="relu")
        self.prediction = tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))

    def call(self, inputs, training=None):

        """
        inputs: (batch_size, max_len, width, height, channel) does not compatible
        inputs: (batch_size * max_len, width, height, channel) converted tensor and sliced later
        """
        x = tf.reshape(inputs, shape=(inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4])) 
        x = self.resnet(x, training=training, extraction=True) # (batch_size * max_len, 2048)
        x = self.fc(x) # (batch_size * max_len, units)
        x = tf.reshape(x, shape=(inputs.shape[0], inputs.shape[1], -1)) # (batch_size, max_len, units)
        x = self.masking_layer(x) # (batch_size, max_len, units) masked
        x = self.lstm(x) # (batch_size, max_len, units) masked
        
        return self.prediction(x) # (batch_size, max_len, 1) masked

class LSTMModule(tf.keras.Model):
    """
    LSTM on top of the features extracted from images
    """
    def __init__(self, config):
        super(LSTMModule, self).__init__()
        self.lstm = LSTM(config)
        self.masking_layer = tf.keras.layers.Masking(mask_value=0.0)
        self.fc = tf.keras.layers.Dense(units=config["units"], activation="relu")
        self.prediction = tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))

    def call(self, inputs, training=None):
        
        x = self.masking_layer(inputs)
        x = self.fc(x) # (batch_size, max_len, units)
        x = self.lstm(x) # (batch_size, max_len, units)

        return self.prediction(x) # (batch_size, max_len, 1)

class LSTMAttentionModule(tf.keras.Model):
    def __init__(self, config):
        super(LSTMAttentionModule, self).__init__()
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.lstm = LSTM(config)
        self.attention_weight = tf.Variable(self.initializer(shape=(config["units"],1)))
        self.attention_softmax = tf.keras.layers.Softmax(axis=1)
        self.masking_layer = tf.keras.layers.Masking(mask_value=0.0)
        self.fc = tf.keras.layers.Dense(units=config["units"], activation="relu")
        self.prediction = tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))

    def call(self, inputs, training=None):
        
        x = self.fc(inputs) # batch_size * max_len * units
        x = self.masking_layer(x)
        x = self.lstm(x) # batch_size * max_len * units
        x_attention = tf.matmul(x, self.attention_weight)
        x_attention = tf.reshape(x_attention, shape=(x.shape[0], x.shape[1]))
        attention_mask = tf.not_equal(x_attention, 0.)
        a_attention = self.attention_softmax(x_attention, mask=attention_mask)
        a_attention = tf.reshape(a_attention, shape=(a_attention.shape[0], a_attention.shape[1], 1))
        a_attention = tf.tile(a_attention, (1, 1, x.shape[2]))

        x = tf.multiply(x, a_attention)
        x = tf.reduce_sum(x, axis=1) # batch_size * units

        return self.prediction(x) # batch_size * 1

class LSTM(tf.keras.layers.Layer):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.lstm = tf.keras.layers.LSTM(units=config["units"], return_sequences=True)
        
    def call(self, inputs):
        
        return self.lstm(inputs)

