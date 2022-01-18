import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
from src.resnet_src import *
from utils.resnet_utils import *

def train_resnet_binary_detection(output_path, data_root_path, data_dict_path, architecture, epoch, batch_size, 
                            learning_rate, l2_reg, num_class, resizing, scale, class_weight, prediction, augment, lr_scheduling,
                            use_pretrain=None):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]

    print("build and initialize ResNet...")
    if use_pretrain == None:
        print("initialize with random weights...")
        resnet = ResNet(config)
    if use_pretrain == "imagenet":
        print("use ResNet pretrained with imagenet...")
        resnet = ResNetPretrained(config)

    if lr_scheduling:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                                        decay_steps=3000, decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:    
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    WBCE = WeightedBinaryCrossEntropy(class_weight)
    AUC = tf.keras.metrics.AUC(num_thresholds=200)

    training_loss = []
    training_auc = []
    validation_auc = []
    best_auc = 0.

    for e in range(epoch):

        AUC.reset_states()
        training_set = list(training_set_dict.keys())
        training_set = shuffle_data(training_set)
        num_batch = int(np.ceil(len(training_set) / batch_size))
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_batch = training_set[i*batch_size:(i+1)*batch_size]
            x, y = prepare_batch(eye_batch, training_set_dict, config, augment=augment, one_hot=False)
            
            with tf.GradientTape() as tape:
                y_hat = resnet(x, training=True)
                batch_loss = WBCE(y_true=y, y_pred=y_hat)

            AUC.update_state(y_true=y, y_pred=y_hat)
            gradients = tape.gradient(batch_loss, resnet.trainable_variables)
            optimizer.apply_gradients(zip(gradients, resnet.trainable_variables))
            loss_record.append(batch_loss.numpy())
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        training_loss.append(np.mean(loss_record))
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(loss_record)))

        current_validation_auc = calculate_auc(resnet, validation_set_dict, config)
        print('epoch:{e}, validation acc:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = resnet.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    resnet.set_weights(best_model)
    test_auc = calculate_auc(resnet, test_set_dict, config)
    print("test AUC: {auc:.6f}".format(auc=test_auc))

    print("saving results...")
    save_data(os.path.join(output_path, "resnet_config.pkl"), config)
    save_data(os.path.join(output_path, "resnet_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "resnet_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "resnet_validation_auc.pkl"), validation_auc)
    model_filepath = os.path.join(output_path, "resnet_best_model.npy")
    np.save(model_filepath, best_model)

def train_resnet_score_detection(output_path, data_root_path, data_mapping_dict_path, architecture, epoch, batch_size, 
                            learning_rate, l2_reg, num_class, k, resizing, scale, class_weight, prediction, augment,
                            lr_scheduling):

    config = locals().copy()

    print("load data...")
    data_mapping_dict = load_data(data_mapping_dict_path)
    training_set_dict = data_mapping_dict["train_set"]
    validation_set_dict = data_mapping_dict["validation_set"]
    test_set_dict = data_mapping_dict["test_set"]

    print("build and initialize ResNet")
    resnet = ResNet(config)
    if lr_scheduling:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                                        decay_steps=3000, decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:    
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    WCCE = WeightedCategoricalCrossEntropy(class_weight)
    ACC = tf.keras.metrics.TopKCategoricalAccuracy(k=config["k"])

    training_loss = []
    training_acc = []
    validation_acc = []
    best_acc = 0.

    for e in range(epoch):

        ACC.reset_states()
        train_set = list(training_set_dict.keys())
        train_set = shuffle_data(train_set)
        num_batch = int(np.ceil(len(train_set) / batch_size))
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_batch = train_set[i*batch_size:(i+1)*batch_size]
            x, y = prepare_batch(eye_batch, training_set_dict, config, augment=augment, one_hot=True)
            
            with tf.GradientTape() as tape:
                y_hat = resnet(x, training=True)
                batch_loss = WCCE(y_true=y, y_pred=y_hat)
            
            ACC.update_state(y_true=y, y_pred=y_hat)
            gradients = tape.gradient(batch_loss, resnet.trainable_variables)
            optimizer.apply_gradients(zip(gradients, resnet.trainable_variables))
            loss_record.append(batch_loss.numpy())
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        training_loss.append(np.mean(loss_record))
        training_acc.append(ACC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(loss_record)))

        current_validation_acc = calculate_accuracy(resnet, validation_set_dict, config)
        print('epoch:{e}, validation acc:{l:.6f}'.format(e=e+1, l=current_validation_acc))
        validation_acc.append(current_validation_acc)

        if current_validation_acc > best_acc:
            best_acc = current_validation_acc
            best_epoch = e+1
            best_model = resnet.get_weights()

    print('Best model: at epoch {e}, accuracy:{l:.6f}'.format(e=best_epoch, l=best_acc))
    print("calculate accuracy using the best model on the test set")
    resnet.set_weights(best_model)
    test_acc = calculate_accuracy(resnet, test_set_dict, config)
    print("test Accuracy: {auc:.6f}".format(auc=test_acc))

    print("saving results...")
    save_data(os.path.join(output_path, "resnet_config.pkl"), config)
    save_data(os.path.join(output_path, "resnet_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "resnet_training_acc.pkl"), training_acc)
    save_data(os.path.join(output_path, "resnet_validation_acc.pkl"), validation_acc)

def train_resnet_binary_prediction(output_path, data_root_path, data_dict_path, architecture, epoch, batch_size, 
                            learning_rate, l2_reg, num_class, resizing, scale, class_weight, prediction, augment, lr_scheduling,
                            use_pretrain=None):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]

    print("build and initialize ResNet...")
    if use_pretrain == None:
        print("initialize with random weights...")
        resnet = ResNet(config)
    if use_pretrain == "imagenet":
        print("use ResNet pretrained with imagenet...")
        resnet = ResNetPretrained(config)

    if lr_scheduling:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                                        decay_steps=2000, decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:    
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    WBCE = WeightedBinaryCrossEntropy(class_weight)
    AUC = tf.keras.metrics.AUC(num_thresholds=200)

    training_loss = []
    training_auc = []
    validation_auc = []
    best_auc = 0.

    for e in range(epoch):

        AUC.reset_states()
        training_set = list(training_set_dict.keys())
        training_set = shuffle_data(training_set)
        num_batch = int(np.ceil(len(training_set) / batch_size))
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_batch = training_set[i*batch_size:(i+1)*batch_size]
            x, y = prepare_batch(eye_batch, training_set_dict, config, augment=augment, one_hot=False)
            
            with tf.GradientTape() as tape:
                y_hat = resnet(x, training=True)
                batch_loss = WBCE(y_true=y, y_pred=y_hat)

            AUC.update_state(y_true=y, y_pred=y_hat)
            gradients = tape.gradient(batch_loss, resnet.trainable_variables)
            optimizer.apply_gradients(zip(gradients, resnet.trainable_variables))
            loss_record.append(batch_loss.numpy())
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        training_loss.append(np.mean(loss_record))
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(loss_record)))

        current_validation_auc = calculate_auc(resnet, validation_set_dict, config)
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = resnet.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    resnet.set_weights(best_model)
    result_dict = calculate_metrics_pertime(resnet, test_set_dict, config)

    print("saving results...")
    save_data(os.path.join(output_path, "resnet_config.pkl"), config)
    save_data(os.path.join(output_path, "resnet_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "resnet_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "resnet_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "resnet_result_dict.pkl"), result_dict)
    model_filepath = os.path.join(output_path, "resnet_best_model.npy")
    np.save(model_filepath, best_model)

def extract_resnet_feature(output_path, model_path, entire_data_list_path, config_path):
    """
    extract features from images using trained ResNet
    """

    print("load data...")
    config = load_data(config_path)
    entire_data_list = load_data(entire_data_list_path)
    data_dict = load_data(config["data_dict_path"])
    best_model = np.load(model_path, allow_pickle=True)
    validation_set_dict = data_dict["validation_set"]

    print("build and initialize ResNet...")
    if config["use_pretrain"] == None:
        print("initialize with random weights...")
        resnet = ResNet(config)
    if config["use_pretrain"] == "imagenet":
        print("use ResNet pretrained with imagenet...")
        resnet = ResNetPretrained(config)

    validation_set = list(validation_set_dict.keys())
    batch_size = config["batch_size"]
    eye_batch = validation_set[:batch_size]
    # initialize the model's weight
    x, y = prepare_batch(eye_batch, validation_set_dict, config, augment=False, one_hot=False)
    y_hat = resnet(x, training=False, extraction=False)

    print("load weights on ResNet...")
    # load the best model's weight
    resnet.set_weights(best_model)

    print("extract features using trained ResNet...")
    extracted_feature_dict = dict()
    num_batch = int(np.ceil(len(entire_data_list) / batch_size))
    progbar = tf.keras.utils.Progbar(num_batch)

    for i in range(num_batch):
        eye_batch = entire_data_list[i*batch_size:(i+1)*batch_size]
        x = prepare_batch_to_extract(eye_batch, config)
        extracted_features = resnet(x, training=False, extraction=True)
        extracted_feature_dict = update_feature_dict(extracted_feature_dict, eye_batch, extracted_features)
        progbar.add(1)

    print("saving extracted feature dict")
    save_data(os.path.join(output_path, "extracted_feature_dict.pkl"), extracted_feature_dict)

def evaluate_resnet_peryear(model_path, data_dict_path, config_path, use_pretrain):

    print("load data...")
    data_dict = load_data(data_dict_path)
    config = load_data(config_path)
    best_model = np.load(model_path, allow_pickle=True)
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]

    print("build and initialize ResNet...")
    resnet = ResNet(config)
    if use_pretrain == None:
        print("initialize with random weights...")
        resnet = ResNet(config)
    if use_pretrain == "imagenet":
        print("use ResNet pretrained with imagenet...")
        resnet = ResNetPretrained(config)

    validation_set = list(validation_set_dict.keys())
    batch_size = config["batch_size"]
    eye_batch = validation_set[:batch_size]

    # initialize the model's weight
    x, y = prepare_batch(eye_batch, validation_set_dict, config, augment=False, one_hot=False)
    y_hat = resnet(x, training=False, extraction=False)
    
    # load the best model's weight
    resnet.set_weights(best_model)

    print("calculate AUC using the best model on the test set")
    result_dict = calculate_metrics_pertime(resnet, test_set_dict, config)

    return result_dict

def evaluate_resnet_perlength(model_path, data_dict_path, config_path, use_pretrain):

    print("load data...")
    data_dict = load_data(data_dict_path)
    config = load_data(config_path)
    best_model = np.load(model_path, allow_pickle=True)
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]
    per_length_test_set_dict = data_dict["per_length_test_set"]

    print("build and initialize ResNet...")
    resnet = ResNet(config)
    if use_pretrain == None:
        print("initialize with random weights...")
        resnet = ResNet(config)
    if use_pretrain == "imagenet":
        print("use ResNet pretrained with imagenet...")
        resnet = ResNetPretrained(config)

    validation_set = list(validation_set_dict.keys())
    batch_size = config["batch_size"]
    eye_batch = validation_set[:batch_size]

    # initialize the model's weight
    x_batch, y_batch = prepare_batch(eye_batch, validation_set_dict, config, augment=False, one_hot=False)
    y_hat = resnet(x_batch, training=False, extraction=False)

    # load the best model's weight
    resnet.set_weights(best_model)

    print("calculate AUC using the best model on the test set")
    result_dict = calculate_metrics_perlength(resnet, test_set_dict, per_length_test_set_dict, config)

    return result_dict