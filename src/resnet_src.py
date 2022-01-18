import tensorflow as tf
import numpy as np
import os

class ResNet(tf.keras.Model):
    def __init__(self, config):
        super(ResNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="SAME")
        self.bn = tf.keras.layers.BatchNormalization()
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.stacked_layer1 = stack_bottleneck_blocks(filters=64, blocks=config["architecture"][0], stride1=1)
        self.stacked_layer2 = stack_bottleneck_blocks(filters=128, blocks=config["architecture"][1], stride1=2) # downsampling at the first block
        self.stacked_layer3 = stack_bottleneck_blocks(filters=256, blocks=config["architecture"][2], stride1=2) # downsampling at the first block
        self.stacked_layer4 = stack_bottleneck_blocks(filters=512, blocks=config["architecture"][3], stride1=2) # downsampling at the first block

        if config["prediction"] == "binary":
            self.fc = tf.keras.layers.Dense(units=config["num_class"], activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))
        elif config["prediction"] == "score":
            self.fc = tf.keras.layers.Dense(units=config["num_class"], activation="softmax", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))
        elif config["prediction"] == "None":
            self.fc = None
        else:
            raise ValueError("prediction type does not exist")

    def call(self, x_input, training=None, extraction=None):

        x = self.conv(x_input)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.maxpool(x)
        x = self.stacked_layer1(x, training=training)
        x = self.stacked_layer2(x, training=training)
        x = self.stacked_layer3(x, training=training)
        x = self.stacked_layer4(x, training=training)
        x = self.avgpool(x)

        if extraction:
            return x
        else:
            return self.fc(x)

class BottleneckBlockConvShortcut(tf.keras.layers.Layer):
    """ResNet Bottlenect Block that has convolutional shortcut conection"""
    def __init__(self, filters, kernel_size=3, stride=1):
        super(BottleneckBlockConvShortcut, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='SAME')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=1, strides=1)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv_shortcut = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=1, strides=stride)
        self.bn_shortcut = tf.keras.layers.BatchNormalization()

    def call(self, x_input, training=None, **kwargs):

        x_shortcut = self.conv_shortcut(x_input)
        x_shortcut = self.bn_shortcut(x_shortcut, training=training)

        x = self.conv1(x_input)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        return tf.nn.relu(tf.math.add(x, x_shortcut))

class BottleneckBlockIdentityShortcut(tf.keras.layers.Layer):
    """ResNet Bottlenect Block that has identity shortcut conection"""
    def __init__(self, filters, kernel_size=3, stride=1):
        super(BottleneckBlockIdentityShortcut, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='SAME')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=1, strides=1)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, x_input, training=None, **kwargs):

        x = self.conv1(x_input)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        return tf.nn.relu(tf.math.add(x, x_input))

def stack_bottleneck_blocks(filters, blocks, stride1=2):
    """A set of stacked residual blocks.
    Args:
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    Returns:
    layer of stacked bottleneck blocks
    """
    stacked_blocks = tf.keras.Sequential()
    stacked_blocks.add(BottleneckBlockConvShortcut(filters, stride=stride1))

    for _ in range(1, blocks):
      stacked_blocks.add(BottleneckBlockIdentityShortcut(filters, stride=1))
    
    return stacked_blocks
    
class ResNetPretrained(tf.keras.Model):
    def __init__(self, config):
        super(ResNetPretrained, self).__init__()
        if config["architecture"] == [3, 4, 23, 3]:
            self.resnet_pretrained = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
        elif config["architecture"] == [3, 4, 6, 3]:
            self.resnet_pretrained = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
        
        if config["prediction"] == "binary":
            self.fc = tf.keras.layers.Dense(units=config["num_class"], activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))
        elif config["prediction"] == "score":
            self.fc = tf.keras.layers.Dense(units=config["num_class"], activation="softmax", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))
        else:
            raise ValueError("prediction type does not exist")
        
    def call(self, x_input, training=None, extraction=None):

        x = self.resnet_pretrained(x_input, training=training)

        if extraction:
            return x
        else:
            return self.fc(x)

