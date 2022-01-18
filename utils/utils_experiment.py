import tensorflow as tf
import numpy as np

def load_cifar10() :

    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_data, test_data = train_data / 255.0, test_data / 255.0

    return train_data, train_labels, test_data, test_labels

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def calculate_accuracy(model, x, y, config):

    ACC = tf.keras.metrics.SparseCategoricalAccuracy()
    ACC.reset_states()
    batch_size = config["batch_size"]
    num_batch = int(np.ceil(x.shape[0] / batch_size))

    for i in range(num_batch):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]
        y_hat = model(x_batch, training=False)
        ACC.update_state(y_true=y_batch, y_pred=y_hat)

    return ACC.result().numpy()