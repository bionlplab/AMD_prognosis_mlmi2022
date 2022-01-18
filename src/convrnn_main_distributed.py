
import tensorflow as tf
import os
from src.convrnn_src import *
from utils.convrnn_distributed_utils import *

def distributed_train_resnetlstm_module(output_path, data_path, architecture, epoch, per_replica_batch_size, units,
                            learning_rate, designate_gpu, l2_reg, class_weight, lr_scheduling, use_pretrain, 
                            pretrained_weights_path, pretrained_config_path, testing):

    config = locals().copy()
    config["batch_size"] = per_replica_batch_size
    
    os.environ["CUDA_VISIBLE_DEVICES"] = designate_gpu
    gpu_list = designate_gpu.split(",")
    global_batch_size = per_replica_batch_size * len(gpu_list)

    print("set mirrored strategy using {} GPUs...".format(len(gpu_list)))
    strategy = tf.distribute.MirroredStrategy()
    
    print("load data...")
    if testing:
        train_dataset_path = os.path.join(data_path, "validation_dataset_tf")
        train_dataset_elem_spec_path = os.path.join(data_path, "validation_dataset_tf_element_spec.pkl")
        train_dataset_elem_spec = load_data(train_dataset_elem_spec_path)
        train_dataset = tf.data.experimental.load(train_dataset_path, element_spec=train_dataset_elem_spec)
        train_dataset = train_dataset.shuffle(len(train_dataset)).batch(global_batch_size, drop_remainder=True)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    else:
        train_dataset_path = os.path.join(data_path, "train_dataset_tf_split")
        train_dataset_elem_spec_path = os.path.join(data_path, "train_dataset_tf_element_spec.pkl")
        train_dataset_elem_spec = load_data(train_dataset_elem_spec_path)
        train_dataset = load_dataset_split(train_dataset_path, train_dataset_elem_spec)
        train_dataset = train_dataset.shuffle(len(train_dataset)).batch(global_batch_size, drop_remainder=True)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    train_dataset_length = len(train_dataset)
    del(train_dataset) # for memory efficiency

    validation_dataset_path = os.path.join(data_path, "validation_dataset_tf")
    validation_dataset_elem_spec_path = os.path.join(data_path, "validation_dataset_tf_element_spec.pkl")
    validation_dataset_elem_spec = load_data(validation_dataset_elem_spec_path)
    validation_dataset = tf.data.experimental.load(validation_dataset_path, element_spec=validation_dataset_elem_spec)
    validation_dataset = validation_dataset.batch(per_replica_batch_size)

    if use_pretrain != None:
        pretrained_config = load_data(pretrained_config_path)
        config["pretrained_config"] = pretrained_config
    
    print("build and initialize models...")

    validation_AUC = tf.keras.metrics.AUC(num_thresholds=200)
    test_AUC = tf.keras.metrics.AUC(num_thresholds=200)
    per_length_test_AUC = tf.keras.metrics.AUC(num_thresholds=200)

    with strategy.scope():
        resnetlstm = ResNetLSTM(config)
        WBCE = WeightedBinaryCrossEntropy(class_weight)
        train_AUC = tf.keras.metrics.AUC(num_thresholds=200)
    
        if lr_scheduling:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                                         decay_steps=300, decay_rate=0.9)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:    
            optimizer = tf.keras.optimizers.Adam(learning_rate)
            
    train_loss = []
    validation_auc = []
    best_auc = 0.
            
    for e in range(epoch):
        
        train_AUC.reset_states()
        validation_AUC.reset_states()
        train_loss_sum = 0.
        total_train_num_batches = train_dataset_length
        count_num_batches = 0
        
        train_progbar = tf.keras.utils.Progbar(total_train_num_batches)
        
        for train_dist_dataset_batch in train_dist_dataset:
            try:
                train_loss_sum += distributed_train_step(strategy, resnetlstm, optimizer, WBCE, train_AUC, train_dist_dataset_batch, global_batch_size)
            except:
                return train_dist_dataset_batch
            count_num_batches += 1
            train_progbar.add(1)
        
        epoch_loss = train_loss_sum / count_num_batches
        train_loss.append(epoch_loss)
        print('epoch:{e}, train loss:{l:.6f}'.format(e=e+1, l=epoch_loss))
        
        for validation_dataset_batch in validation_dataset:
            
            x_batch, y_batch = validation_dataset_batch
            x_batch = x_batch / 255.
            y_hat = resnetlstm(x_batch, training=False)
            y_batch_flatten, y_hat_flatten = apply_sequential_mask(y_batch, y_hat)
            validation_AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

        current_validation_auc = validation_AUC.result().numpy()
        print('epoch:{e}, validation AUC:{l:.6f}'.format(e=e+1, l=current_validation_auc))
        validation_auc.append(current_validation_auc)

        if current_validation_auc > best_auc:
            best_auc = current_validation_auc
            best_epoch = e+1
            best_model = resnetlstm.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))

    print("calculate AUC using the best model on the test set")
    resnetlstm.set_weights(best_model)

    print("load data...")
    test_dataset_path = os.path.join(data_path, "test_dataset_tf")
    test_dataset_elem_spec_path = os.path.join(data_path, "test_dataset_tf_element_spec.pkl")
    test_dataset_elem_spec = load_data(test_dataset_elem_spec_path)
    test_dataset = tf.data.experimental.load(test_dataset_path, element_spec=test_dataset_elem_spec)
    test_dataset = test_dataset.batch(per_replica_batch_size)

    for test_datset_batch in test_dataset:
        
        x_batch, y_batch = test_datset_batch
        x_batch = x_batch / 255.
        y_hat = resnetlstm(x_batch, training=False)
        y_batch_flatten, y_hat_flatten = apply_sequential_mask(y_batch, y_hat)
        test_AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)

    test_auc = test_AUC.result().numpy()
    print('test auc:{auc:.6f}'.format(auc=test_auc))
    per_length_result_dict = dict()
    per_length_result_dict["entire_AUC"] = test_auc

    print("calculate AUC using the best model on the per length test set")

    unique_length = list(range(1, 12))

    for length in unique_length:

        this_length_test_dataset_path = os.path.join(data_path, "per_length_test_dataset_tf/", "length_{}_test_dataset_tf".format(length))
        this_length_test_dataset_elem_spec_path = os.path.join(data_path, "per_length_test_dataset_tf/", "length_{}_test_dataset_element_spec.pkl".format(length))
        this_length_test_dataset_elem_spec = load_data(this_length_test_dataset_elem_spec_path)
        this_length_test_dataset = tf.data.experimental.load(this_length_test_dataset_path, element_spec=this_length_test_dataset_elem_spec)
        this_length_test_dataset = this_length_test_dataset.batch(per_replica_batch_size)

        per_length_test_AUC.reset_states()

        for this_length_test_dataset_batch in this_length_test_dataset:
            x_batch, y_batch = this_length_test_dataset_batch
            x_batch = x_batch / 255.
            y_hat = resnetlstm(x_batch, training=False)
            y_batch_flatten, y_hat_flatten = apply_sequential_last_mask(y_batch, y_hat)
            per_length_test_AUC.update_state(y_true=y_batch_flatten, y_pred=y_hat_flatten)
            
        this_length_test_auc = per_length_test_AUC.result().numpy()
        per_length_result_dict[length] = this_length_test_auc
        
    print("saving results...")
    save_data(os.path.join(output_path, "convrnn_config.pkl"), config)
    save_data(os.path.join(output_path, "convrnn_training_loss.pkl"), train_loss)
    save_data(os.path.join(output_path, "convrnn_validation_auc.pkl"), validation_auc)
    save_data(os.path.join(output_path, "convrnn_result_dict.pkl"), per_length_result_dict)

    