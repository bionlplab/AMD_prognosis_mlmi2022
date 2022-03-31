import numpy as np
import tensorflow as tf
import random
import pickle
import os

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

def prepare_batch(eye_list_batch, label_list_batch, feature_dict, config):

    batch_x_tensor = np.zeros(shape=(len(eye_list_batch), config["feature_dim"]))
    batch_y_tensor = np.zeros(shape=(len(label_list_batch), 1))
    
    for b, eye_list in enumerate(eye_list_batch):

        batch_x_tensor[b, :] = feature_dict[eye_list[-1]] # use only the last image

    for b, label_list in enumerate(label_list_batch):

        batch_y_tensor[b, :] = label_list[-1]

    return tf.cast(batch_x_tensor, dtype=tf.float32), tf.cast(batch_y_tensor, dtype=tf.float32)

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

def calculate_auc(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    batch_size = config["batch_size"]
    eye_list = dataset_dict["eye_list"]
    label_list = dataset_dict["label_list"]
    num_batch = int(np.ceil(float(len(eye_list)) / float(batch_size)))
    AUC.reset_states()

    for i in range(num_batch):

        eye_list_batch = eye_list[i*batch_size:(i+1)*batch_size]
        label_list_batch = label_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)

        y_hat = model(x_batch)
        y_batch = tf.reshape(y_batch, shape=-1)
        y_hat = tf.reshape(y_hat, shape=-1)
        AUC.update_state(y_true=y_batch, y_pred=y_hat)

    return AUC.result().numpy()

def calculate_auc_perlength(model, dataset_dict, feature_dict, config):

    AUC = tf.keras.metrics.AUC(num_thresholds=200)
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
            x_batch, y_batch = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)
            y_hat = model(x_batch)
            y_batch = tf.reshape(y_batch, shape=-1)
            y_hat = tf.reshape(y_hat, shape=-1)
            prediction_count += y_hat.shape[0]
            AUC.update_state(y_true=y_batch, y_pred=y_hat)

        result_dict[length] = {"AUC" : AUC.result().numpy(), "prediction_count" : prediction_count}

    return result_dict

def bootstrap_resnet_feature_evaluation(output_path, data_dict_path, feature_dict_path, repetition, epoch, batch_size,
                            units, feature_dim, learning_rate, l2_reg, class_weight, lr_scheduling, seed_list):

    if seed_list != None:
        assert len(seed_list) == repetition, "the length of seed list must be the same with repetition"

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    feature_dict = load_data(feature_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]
    # per length test is not used

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
        fcprediction = FCPrediction(config)

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
            loss_record = []
            progbar = tf.keras.utils.Progbar(num_batch)

            for i in range(num_batch):
                eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
                label_list_batch = train_label_list[i*batch_size:(i+1)*batch_size]
                x_batch, y_batch = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)
            
                with tf.GradientTape() as tape:
                    y_hat = fcprediction(x_batch) # (batch_size, 1)
                    y_batch = tf.reshape(y_batch, shape=-1)
                    y_hat = tf.reshape(y_hat, shape=-1)
                    batch_loss = WBCE(y_true=y_batch, y_pred=y_hat)

                AUC.update_state(y_true=y_batch, y_pred=y_hat)
                gradients = tape.gradient(batch_loss, fcprediction.trainable_variables)
                optimizer.apply_gradients(zip(gradients, fcprediction.trainable_variables))
                loss_record.append(batch_loss.numpy())
                print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
                progbar.add(1)

            # per epoch record
            training_loss.append(np.mean(loss_record))
            training_auc.append(AUC.result().numpy())
            print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(loss_record)))

            current_validation_auc = calculate_auc(fcprediction, validation_set_dict, feature_dict, config)
            print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
            validation_auc.append(current_validation_auc)

            if current_validation_auc > best_auc:
                best_auc = current_validation_auc
                best_epoch = e+1
                best_model = fcprediction.get_weights()

        print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
        print("calculate AUC using the best model on the test set")
        fcprediction.set_weights(best_model)
        test_auc = calculate_auc(fcprediction, test_set_dict, feature_dict, config)
        print('test auc:{l:.6f}'.format(e=best_epoch, l=test_auc))

        # saving result of the current repetition
        entire_repetition_result_dict[r] = {"training_loss" : training_loss,
                                            "training_AUC" : training_auc,
                                            "validation_AUC" : validation_auc,
                                            "entire_AUC" : test_auc}

    print("saving results...")
    save_data(os.path.join(output_path, "resnet_feature_evaluation_repetiton{}_config.pkl".format(repetition)), config)
    save_data(os.path.join(output_path, "resnet_feature_evaluation_repetiton{}_result_dict.pkl".format(repetition)), entire_repetition_result_dict)

def resnet_feature_evaluation(output_path, data_dict_path, feature_dict_path, epoch, batch_size,
                            units, feature_dim, learning_rate, l2_reg, class_weight, lr_scheduling, specify_fold):

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

    print("build and initialize models...")
    fcprediction = FCPrediction(config)

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
        train_eye_list, train_label_list = build_stratified_batch(train_eye_list, train_label_list, batch_size)
        num_batch = int(np.ceil(len(train_eye_list) / batch_size))
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batch)

        for i in range(num_batch):
            eye_list_batch = train_eye_list[i*batch_size:(i+1)*batch_size]
            label_list_batch = train_label_list[i*batch_size:(i+1)*batch_size]
            x_batch, y_batch = prepare_batch(eye_list_batch, label_list_batch, feature_dict, config)
            
            with tf.GradientTape() as tape:
                y_hat = fcprediction(x_batch) # (batch_size, 1)
                y_batch = tf.reshape(y_batch, shape=-1)
                y_hat = tf.reshape(y_hat, shape=-1)
                batch_loss = WBCE(y_true=y_batch, y_pred=y_hat)

            AUC.update_state(y_true=y_batch, y_pred=y_hat)
            gradients = tape.gradient(batch_loss, fcprediction.trainable_variables)
            optimizer.apply_gradients(zip(gradients, fcprediction.trainable_variables))
            loss_record.append(batch_loss.numpy())
            print("batch loss: {l:.6f}".format(l=batch_loss.numpy()))
            progbar.add(1)

        # per epoch record
        training_loss.append(np.mean(loss_record))
        training_auc.append(AUC.result().numpy())
        print('epoch:{e}, training loss:{l:.6f}'.format(e=e+1, l=np.mean(loss_record)))

        current_validation_auc = calculate_auc(fcprediction, validation_set_dict, feature_dict, config)
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = fcprediction.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    print("calculate AUC using the best model on the test set")
    fcprediction.set_weights(best_model)
    test_auc = calculate_auc(fcprediction, test_set_dict, feature_dict, config)
    print('test auc:{l:.6f}'.format(e=best_epoch, l=test_auc))
    result_dict = calculate_auc_perlength(fcprediction, per_length_test_set_dict, feature_dict, config)
    result_dict["entire_AUC"] = test_auc

    # saving result of the current repetition
    print("saving results...")
    save_data(os.path.join(output_path, "resnet_feature_evaluation_config.pkl"), config)
    save_data(os.path.join(output_path, "resnet_feature_evaluation_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "resnet_feature_evaluation_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "resnet_feature_evaluation_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "resnet_feature_evaluation_result_dict.pkl"), result_dict)

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

class FCPrediction(tf.keras.Model):
    def __init__(self, config):
        super(FCPrediction, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=config["units"], activation="relu")
        self.fc2 = tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))
        
    def call(self, x_input):
        x = self.fc1(x_input)
        return self.fc2(x)