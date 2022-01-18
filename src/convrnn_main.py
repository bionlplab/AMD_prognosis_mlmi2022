from pickle import load
import tensorflow as tf
import numpy as np
import os
from src.convrnn_src import *
from utils.convrnn_utils import *

def train_lstm_module_timedelta(output_path, data_dict_path, feature_dict_path, epoch, batch_size, units, feature_dim,
                            learning_rate, l2_reg, class_weight, lr_scheduling):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    feature_dict = load_data(feature_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]

    print("build and initialize models...")
    lstm_module = LSTMModule(config)

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
        train_eye_list = training_set_dict["eye_list"]
        train_label_list = training_set_dict["label_list"]

        # shuffle data
        train_eye_list = shuffle_data(train_eye_list)
        train_label_list = shuffle_data(train_label_list)

        num_batch = int(np.ceil(len(train_eye_list) / batch_size))
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
            y_batch = np.array(train_label_list[i*batch_size:(i+1)*batch_size])
            x_batch = prepare_batch_lstm_module(eye_list_batch, feature_dict, config)
            
            with tf.GradientTape() as tape:
                y_hat = lstm_module(x_batch, training=True)
                y_hat = apply_mask(y_hat)
                batch_loss = WBCE(y_true=y_batch, y_pred=y_hat)

            AUC.update_state(y_true=y_batch, y_pred=y_hat)
            gradients = tape.gradient(batch_loss, lstm_module.trainable_variables)
            optimizer.apply_gradients(zip(gradients, lstm_module.trainable_variables))
            loss_record.append(batch_loss.numpy())
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        training_loss.append(np.mean(loss_record))
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(loss_record)))

        current_validation_auc = calculate_auc_lstm_module(lstm_module, validation_set_dict, feature_dict, config)
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = lstm_module.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    lstm_module.set_weights(best_model)
    result_dict = calculate_metrics_peryear_lstm_module(lstm_module, test_set_dict, feature_dict, config)

    print("saving results...")
    save_data(os.path.join(output_path, "convrnn_config.pkl"), config)
    save_data(os.path.join(output_path, "convrnn_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "convrnn_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "convrnn_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "convrnn_result_dict.pkl"), result_dict)

def train_lstm_sequential_module(output_path, data_dict_path, feature_dict_path, epoch, batch_size, units, feature_dim,
                            learning_rate, l2_reg, class_weight, lr_scheduling):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    feature_dict = load_data(feature_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]
    per_length_test_set_dict = data_dict["per_length_test_set"]

    print("build and initialize models...")
    lstm_module = LSTMModule(config)

    if lr_scheduling:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                                        decay_steps=300, decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:    
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    WBCES = WeightedBinaryCrossEntropySequential(class_weight)
    AUC = tf.keras.metrics.AUC(num_thresholds=200)

    training_loss = []
    training_auc = []
    validation_auc = []
    best_auc = 0.

    for e in range(epoch):

        AUC.reset_states()
        train_eye_list = training_set_dict["eye_list"]
        train_label_list = training_set_dict["label_list"]

        # need shuffle
        num_batch = int(np.ceil(len(train_eye_list) / batch_size))
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = train_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch = prepare_batch_lstm_sequential_module(eye_list_batch, label_list_batch, feature_dict, config)
            
            with tf.GradientTape() as tape:
                y_hat = lstm_module(x_batch, training=True) # batch_size * max_len * 1
                batch_loss = WBCES(y_true=y_batch, y_pred=y_hat)

            y_batch_flatten, y_hat_flatten = apply_sequential_mask(y_batch, y_hat)
            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)
            gradients = tape.gradient(batch_loss, lstm_module.trainable_variables)
            optimizer.apply_gradients(zip(gradients, lstm_module.trainable_variables))
            loss_record.append(batch_loss.numpy())
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        training_loss.append(np.mean(loss_record))
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(loss_record)))

        current_validation_auc = calculate_auc_lstm_sequential_module(lstm_module, validation_set_dict, feature_dict, config)
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = lstm_module.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    lstm_module.set_weights(best_model)
    test_auc = calculate_auc_lstm_sequential_module(lstm_module, test_set_dict, feature_dict, config)
    print('test auc:{l:.6f}'.format(e=best_epoch, l=test_auc))
    per_length_result_dict = calculate_metrics_perlength(lstm_module, per_length_test_set_dict, feature_dict, config)
    per_length_result_dict["average"] = test_auc

    print("saving results...")
    save_data(os.path.join(output_path, "convrnn_config.pkl"), config)
    save_data(os.path.join(output_path, "convrnn_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "convrnn_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "convrnn_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "convrnn_per_length_result_dict.pkl"), per_length_result_dict)

def train_lstm_attention_module_timedelta(output_path, data_dict_path, feature_dict_path, epoch, batch_size, units, feature_dim,
                            learning_rate, l2_reg, class_weight, lr_scheduling):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    feature_dict = load_data(feature_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]

    print("build and initialize models...")
    lstm_module = LSTMAttentionModule(config)

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
        train_eye_list = training_set_dict["eye_list"]
        train_label_list = training_set_dict["label_list"]

        # need shuffle
        num_batch = int(np.ceil(len(train_eye_list) / batch_size))
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
            y_batch = np.array(train_label_list[i*batch_size:(i+1)*batch_size])
            x_batch = prepare_batch_lstm_module(eye_list_batch, feature_dict, config)
            
            with tf.GradientTape() as tape:
                y_hat = lstm_module(x_batch, training=True)
                batch_loss = WBCE(y_true=y_batch, y_pred=y_hat)

            AUC.update_state(y_true=y_batch, y_pred=y_hat)
            gradients = tape.gradient(batch_loss, lstm_module.trainable_variables)
            optimizer.apply_gradients(zip(gradients, lstm_module.trainable_variables))
            loss_record.append(batch_loss.numpy())
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        training_loss.append(np.mean(loss_record))
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(loss_record)))

        current_validation_auc = calculate_auc_lstm_attention_module(lstm_module, validation_set_dict, feature_dict, config)
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = lstm_module.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    lstm_module.set_weights(best_model)
    result_dict = calculate_metrics_peryear_lstm_attention_module(lstm_module, test_set_dict, feature_dict, config)

    print("saving results...")
    save_data(os.path.join(output_path, "convrnn_config.pkl"), config)
    save_data(os.path.join(output_path, "convrnn_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "convrnn_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "convrnn_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "convrnn_result_dict.pkl"), result_dict)

def train_resnetlstm_module(output_path, data_root_path, data_dict_path, architecture, epoch, batch_size, units,
                            learning_rate, l2_reg, resizing, scale, class_weight, augment, lr_scheduling, use_pretrain, 
                            pretrained_weights_path, pretrained_config_path, testing):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]
    per_length_test_set_dict = data_dict["per_length_test_set"]

    if use_pretrain != None:
        pretrained_config = load_data(pretrained_config_path)
        config["pretrained_config"] = pretrained_config

    if testing:
        training_set_dict = data_dict["validation_set"]

    print("build and initialize models...")
    resnetlstm = ResNetLSTM(config)

    if lr_scheduling:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                                        decay_steps=2000, decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:    
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    WBCES = WeightedBinaryCrossEntropySequential(class_weight)
    AUC = tf.keras.metrics.AUC(num_thresholds=200)

    training_loss = []
    training_auc = []
    validation_auc = []
    best_auc = 0.

    for e in range(epoch):

        AUC.reset_states()
        train_eye_list = training_set_dict["eye_list"]
        train_label_list = training_set_dict["label_list"]

        # need shuffle
        num_batch = int(np.ceil(len(train_eye_list) / batch_size))
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = train_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch = prepare_batch_resnetlstm_module(eye_list_batch, label_list_batch, config, augment=augment)
            
            with tf.GradientTape() as tape:
                y_hat = resnetlstm(x_batch, training=True)
                batch_loss = WBCES(y_true=y_batch, y_pred=y_hat)

            y_batch_flatten, y_hat_flatten = apply_sequential_mask(y_batch, y_hat)
            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)
            gradients = tape.gradient(batch_loss, resnetlstm.trainable_variables)
            optimizer.apply_gradients(zip(gradients, resnetlstm.trainable_variables))
            loss_record.append(batch_loss.numpy())
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        training_loss.append(np.mean(loss_record))
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(loss_record)))

        current_validation_auc = calculate_auc_resnetlstm_module(resnetlstm, validation_set_dict, config)
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = resnetlstm.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    resnetlstm.set_weights(best_model)
    test_auc = calculate_auc_resnetlstm_module(resnetlstm, test_set_dict, config)
    print('test auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    per_length_result_dict = calculate_metrics_perlength_resnetlstm(resnetlstm, per_length_test_set_dict, config)
    per_length_result_dict["average"] = test_auc

    print("saving results...")
    save_data(os.path.join(output_path, "convrnn_config.pkl"), config)
    save_data(os.path.join(output_path, "convrnn_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "convrnn_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "convrnn_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "convrnn_result_dict.pkl"), per_length_result_dict)
    model_filepath = os.path.join(output_path, "convrnn_best_model.npy")
    np.save(model_filepath, best_model)