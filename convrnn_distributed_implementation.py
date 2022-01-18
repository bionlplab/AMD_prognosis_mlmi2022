#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.convrnn_distributed_main_ex import *


# In[2]:


distributed_train_resnetlstm_module("/home/jl5307/current_research/AMD_prediction/results/convlstm/resnetlstm_module_u256_b16_e20_sequential_timedelta5_dist/",
                                   "/home/jl5307/current_research/AMD_prediction/img_data/numpy_data/longitudinal_sequential_prediction_timedelta5/dataset_tf/",
                                   [3, 4, 23, 3], 20, 4, 256, 0.0005, "0,1,2,3", 0.0001, [0.03, 0.97], lr_scheduling=True, use_pretrain="imagenet",
                                   pretrained_weights_path="/home/jl5307/current_research/AMD_prediction/results/resnet101_binary/b16_lr0005_e20_detection_imagenet/resnet_best_model.npy",
                                   pretrained_config_path="/home/jl5307/current_research/AMD_prediction/results/resnet101_binary/b16_lr0005_e20_detection_imagenet/resnet_config.pkl",
                                   testing=False)


# In[ ]:




