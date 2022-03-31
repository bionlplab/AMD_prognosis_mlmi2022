import tensorflow as tf
import os
from utils.transformer_utils import *
from src.transformer_src import *

def train_transformer(output_path, data_dict_path, feature_dict_path, epoch, batch_size, num_layers, model_dim, num_heads, feature_dim,
                            intermediate_dim, maximum_seq_length, dropout_rate, learning_rate, l2_reg, class_weight, lr_scheduling, patience,
                            specify_fold, testing):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    if specify_fold != None:
        data_dict = data_dict[specify_fold]
    feature_dict = load_data(feature_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]
    per_length_test_set_dict = data_dict["per_length_test_set"]

    if testing:
        training_set_dict = data_dict["validation_set"]

    print("build and initialize models...")
    tfencoder = TransformerEncoder(config)

    if lr_scheduling != None:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
            decay_steps=lr_scheduling[0], decay_rate=lr_scheduling[1])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:    
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    WBCE = WeightedBinaryCrossEntropy(class_weight)
    AUC = tf.keras.metrics.AUC(num_thresholds=200)

    training_loss = []
    training_auc = []
    validation_auc = []
    best_auc = 0.
    early_stopping_count = 0

    for e in range(epoch):

        AUC.reset_states()
        train_eye_list = training_set_dict["eye_list"]
        train_label_list = training_set_dict["label_list"]
        train_eye_list, train_label_list = build_stratified_batch(train_eye_list, train_label_list, batch_size)

        num_batch = int(np.ceil(len(train_eye_list) / batch_size))
        batch_loss_sum = 0
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = train_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch, mask = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)
            # mask: (batch_size, 1, 1, max_seq_len)
            
            with tf.GradientTape() as tape:
                y_hat = tfencoder(x_batch, training=True, mask=mask) # (batch_size, max_seq_len, 1)
                y_batch_flatten, y_hat_flatten = flatten_with_last(y_batch, y_hat, mask)
                batch_loss = WBCE(y_true=y_batch_flatten, y_pred=y_hat_flatten)

            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)
            gradients = tape.gradient(batch_loss, tfencoder.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tfencoder.trainable_variables))
            batch_loss_sum += batch_loss.numpy()
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        avg_epoch_loss = batch_loss_sum / num_batch
        training_loss.append(avg_epoch_loss)
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(avg_epoch_loss)))

        current_validation_auc = calculate_auc(tfencoder, validation_set_dict, feature_dict, config)
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = tfencoder.get_weights()

        if current_validation_auc < validation_auc[-1]:
            early_stopping_count += 1
        else:
            early_stopping_count = 0

        if early_stopping_count >= patience:
            print("training finished due to early stopping...")
            break

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    tfencoder.set_weights(best_model)
    test_auc = calculate_auc(tfencoder, test_set_dict, feature_dict, config)
    print('test auc:{l:.6f}'.format(e=best_epoch, l=test_auc))
    result_dict = calculate_auc_perlength(tfencoder, per_length_test_set_dict, feature_dict, config)
    result_dict["entire_AUC"] = test_auc

    print("saving results...")
    save_data(os.path.join(output_path, "transformer_config.pkl"), config)
    save_data(os.path.join(output_path, "transformer_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "transformer_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "transformer_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "transformer_result_dict.pkl"), result_dict)

def train_transformer_use_token(output_path, data_dict_path, feature_dict_path, epoch, batch_size, num_layers, model_dim, num_heads, feature_dim,
                            intermediate_dim, maximum_seq_length, dropout_rate, learning_rate, l2_reg, class_weight, lr_scheduling, 
                            use_sep_token, use_pred_token, testing):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    feature_dict = load_data(feature_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]
    per_length_test_set_dict = data_dict["per_length_test_set"]
    token_num = config["use_pred_token"] + config["use_sep_token"]

    if testing:
        training_set_dict = data_dict["validation_set"]

    print("build and initialize models...")
    tfencoder = TransformerEncoderToken(config)

    if lr_scheduling != None:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
            decay_steps=lr_scheduling[0], decay_rate=lr_scheduling[1])
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

        num_batch = int(np.ceil(len(train_eye_list) / batch_size))
        batch_loss_sum = 0
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = train_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch, mask = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)
            # mask: (batch_size, 1, 1, max_seq_len)
            
            with tf.GradientTape() as tape:
                y_hat = tfencoder(x_batch, training=True, mask=mask) # (batch_size, max_seq_len, 1)
                y_batch_flatten, y_hat_flatten = flatten_with_token(y_batch, y_hat, mask)
                batch_loss = WBCE(y_true=y_batch_flatten, y_pred=y_hat_flatten)

            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)
            gradients = tape.gradient(batch_loss, tfencoder.trainable_variables)
            if token_num == 1:
                gradients[-1] = tf.convert_to_tensor(gradients[-1])
            elif token_num == 2:
                gradients[-1] = tf.convert_to_tensor(gradients[-1])
                gradients[-2] = tf.convert_to_tensor(gradients[-2])
            optimizer.apply_gradients(zip(gradients, tfencoder.trainable_variables))
            batch_loss_sum += batch_loss.numpy()
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        avg_epoch_loss = batch_loss_sum / num_batch
        training_loss.append(avg_epoch_loss)
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(avg_epoch_loss)))

        current_validation_auc = calculate_auc_token(tfencoder, validation_set_dict, feature_dict, config)
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = tfencoder.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    tfencoder.set_weights(best_model)
    test_auc = calculate_auc_token(tfencoder, test_set_dict, feature_dict, config)
    print('test auc:{l:.6f}'.format(e=best_epoch, l=test_auc))
    result_dict = calculate_auc_perlength_token(tfencoder, per_length_test_set_dict, feature_dict, config)
    result_dict["entire_AUC"] = test_auc

    print("saving results...")
    save_data(os.path.join(output_path, "transformer_config.pkl"), config)
    save_data(os.path.join(output_path, "transformer_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "transformer_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "transformer_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "transformer_result_dict.pkl"), result_dict)

def train_transformer_use_segment_embedding(output_path, data_dict_path, feature_dict_path, epoch, batch_size, num_layers, model_dim, num_heads, feature_dim,
                            intermediate_dim, maximum_seq_length, dropout_rate, learning_rate, l2_reg, class_weight, lr_scheduling, 
                            use_segment_embedding, testing):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    feature_dict = load_data(feature_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]
    per_length_test_set_dict = data_dict["per_length_test_set"]

    if testing:
        training_set_dict = data_dict["validation_set"]

    print("build and initialize models...")
    tfencoder = TransformerEncoderSegment(config)

    if lr_scheduling != None:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
            decay_steps=lr_scheduling[0], decay_rate=lr_scheduling[1])
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

        num_batch = int(np.ceil(len(train_eye_list) / batch_size))
        batch_loss_sum = 0
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = train_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch, mask, segment_batch = prepare_batch_segment(eye_list_batch, label_list_batch, feature_dict, config)
            # mask: (batch_size, 1, 1, max_seq_len)
            
            with tf.GradientTape() as tape:
                y_hat = tfencoder(x_batch, segment_batch, training=True, mask=mask) # (batch_size, max_seq_len, 1)
                y_batch_flatten, y_hat_flatten = flatten_with_last(y_batch, y_hat, mask)
                batch_loss = WBCE(y_true=y_batch_flatten, y_pred=y_hat_flatten)

            AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)
            gradients = tape.gradient(batch_loss, tfencoder.trainable_variables)
            gradients[-1] = tf.convert_to_tensor(gradients[-1]) # collect sparse gradients for embedding layer
            optimizer.apply_gradients(zip(gradients, tfencoder.trainable_variables))
            batch_loss_sum += batch_loss.numpy()
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        avg_epoch_loss = batch_loss_sum / num_batch
        training_loss.append(avg_epoch_loss)
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(avg_epoch_loss)))

        current_validation_auc = calculate_auc_segment(tfencoder, validation_set_dict, feature_dict, config)
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = tfencoder.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    tfencoder.set_weights(best_model)
    test_auc = calculate_auc_segment(tfencoder, test_set_dict, feature_dict, config)
    print('test auc:{l:.6f}'.format(e=best_epoch, l=test_auc))
    result_dict = calculate_auc_perlength_segment(tfencoder, per_length_test_set_dict, feature_dict, config)
    result_dict["entire_AUC"] = test_auc

    print("saving results...")
    save_data(os.path.join(output_path, "transformer_config.pkl"), config)
    save_data(os.path.join(output_path, "transformer_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "transformer_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "transformer_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "transformer_result_dict.pkl"), result_dict)

def bootstrap_train_transformer(output_path, data_dict_path, feature_dict_path, repetition, epoch, batch_size, num_layers, 
                                model_dim, num_heads, feature_dim, intermediate_dim, maximum_seq_length, dropout_rate, 
                                learning_rate, l2_reg, class_weight, lr_scheduling, seed_list, testing):

    if seed_list != None:
        assert len(seed_list) == repetition, "the length of seed list must be the same with repetition"

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    feature_dict = load_data(feature_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]
    per_length_test_set_dict = data_dict["per_length_test_set"]

    if testing:
        training_set_dict = data_dict["validation_set"]

    entire_repetition_result_dict = dict()

    for r in range(repetition):

        print("{}-th repetition...".format(r+1))
        if seed_list != None:
            print("using bootstrap the train set...")
            current_seed = seed_list[r]
            np.random.seed(current_seed)
            bootstrap_ind = np.random.randint(len(training_set_dict["eye_list"]), size=len(training_set_dict["eye_list"]))
            train_eye_list = list(np.array(training_set_dict["eye_list"])[bootstrap_ind])
            train_label_list = list(np.array(training_set_dict["label_list"])[bootstrap_ind])
            train_eye_list, train_label_list = build_stratified_batch(train_eye_list, train_label_list, batch_size)
            
        else:
            print("using the full train set...")
            train_eye_list = training_set_dict["eye_list"]
            train_label_list = training_set_dict["label_list"]
            train_eye_list, train_label_list = build_stratified_batch(train_eye_list, train_label_list, batch_size)

        print("build and initialize models...")
        tfencoder = TransformerEncoder(config)

        if lr_scheduling != None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                                        decay_steps=lr_scheduling[0], decay_rate=lr_scheduling[1])
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
            num_batch = int(np.ceil(len(train_eye_list) / batch_size))
            batch_loss_sum = 0
            progbar = tf.keras.utils.Progbar(num_batch)

            for i in range(num_batch):
                eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
                label_list_batch = train_label_list[i*batch_size:(i+1)*batch_size]
                x_batch, y_batch, mask = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)
                # mask: (batch_size, 1, 1, max_seq_len)
            
                with tf.GradientTape() as tape:
                    y_hat = tfencoder(x_batch, training=True, mask=mask) # (batch_size, max_seq_len, 1)
                    y_batch_flatten, y_hat_flatten = flatten_with_last(y_batch, y_hat, mask)
                    batch_loss = WBCE(y_true=y_batch_flatten, y_pred=y_hat_flatten)

                AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)
                gradients = tape.gradient(batch_loss, tfencoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, tfencoder.trainable_variables))
                batch_loss_sum += batch_loss.numpy()
                print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
                progbar.add(1)

            # per epoch record
            avg_epoch_loss = batch_loss_sum / num_batch
            training_loss.append(avg_epoch_loss)
            training_auc.append(AUC.result().numpy())
            print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(avg_epoch_loss)))

            current_validation_auc = calculate_auc(tfencoder, validation_set_dict, feature_dict, config)
            print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
            validation_auc.append(current_validation_auc)

            if current_validation_auc > best_auc:
                best_auc = current_validation_auc
                best_epoch = e+1
                best_model = tfencoder.get_weights()

        print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
        print("calculate AUC using the best model on the test set")
        tfencoder.set_weights(best_model)
        test_auc = calculate_auc(tfencoder, test_set_dict, feature_dict, config)
        print('test auc:{l:.6f}'.format(e=best_epoch, l=test_auc))
        result_dict = calculate_auc_perlength(tfencoder, per_length_test_set_dict, feature_dict, config)
        result_dict["entire_AUC"] = test_auc

        # saving result of the current repetition
        entire_repetition_result_dict[r] = {"training_loss" : training_loss,
                                            "training_AUC" : training_auc,
                                            "validation_AUC" : validation_auc,
                                            "result" : result_dict}

    print("saving results...")
    save_data(os.path.join(output_path, "transformer_repetiton{}_config.pkl".format(repetition)), config)
    save_data(os.path.join(output_path, "transformer_repetiton{}_result_dict.pkl".format(repetition)), entire_repetition_result_dict)