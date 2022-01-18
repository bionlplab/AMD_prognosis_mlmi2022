import tensorflow as tf
import os
from utils.transformer_utils import *
from src.transformer_src import *

def train_transformer(output_path, data_dict_path, feature_dict_path, epoch, batch_size, num_layers, model_dim, num_heads, feature_dim,
                            intermediate_dim, maximum_seq_length, dropout_rate, learning_rate, l2_reg, class_weight, lr_scheduling, testing):

    config = locals().copy()

    print("load data...")
    data_dict = load_data(data_dict_path)
    feature_dict = load_data(feature_dict_path)
    training_set_dict = data_dict["train_set"]
    validation_set_dict = data_dict["validation_set"]
    test_set_dict = data_dict["test_set"]

    if testing:
        training_set_dict = data_dict["validation_set"]

    print("build and initialize models...")
    tfencoder = TransformerEncoder(config)

    if lr_scheduling:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                                        decay_steps=600, decay_rate=0.9)
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
                y_batch_flatten, y_hat_flatten = flatten_with_mask(y_batch, y_hat, mask)
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
    result_dict = dict()
    result_dict["average"] = test_auc

    print("saving results...")
    save_data(os.path.join(output_path, "transformer_config.pkl"), config)
    save_data(os.path.join(output_path, "transformer_training_loss.pkl"), training_loss)
    save_data(os.path.join(output_path, "transformer_training_auc.pkl"), training_auc)
    save_data(os.path.join(output_path, "transformer_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "transformer_result_dict.pkl"), result_dict)