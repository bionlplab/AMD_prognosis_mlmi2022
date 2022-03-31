#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.convrnn_main import *


# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[3]:


train_lstm("/home/jl5307/current_research/AMD_prediction/results/convlstm/5_fold/td5/fold0/", 
            "/home/jl5307/current_research/AMD_prediction/img_data/5_fold/five_fold_unrolled_longitudinal_prediction_td5_min5_data_dict.pkl",
            "/home/jl5307/current_research/AMD_prediction/results/resnet101_amd_detection/fold0/extracted_feature_dict.pkl",
            30, 32, 256, 2048, 0.0002, 0.0001, [0.03, 0.97], lr_scheduling=[2000, 0.9], specify_fold=0, testing=False)


# In[ ]:


train_lstm("/home/jl5307/current_research/AMD_prediction/results/convlstm/5_fold/td5/fold1/", 
            "/home/jl5307/current_research/AMD_prediction/img_data/5_fold/five_fold_unrolled_longitudinal_prediction_td5_min5_data_dict.pkl",
            "/home/jl5307/current_research/AMD_prediction/results/resnet101_amd_detection/fold1/extracted_feature_dict.pkl",
            30, 32, 256, 2048, 0.0002, 0.0001, [0.03, 0.97], lr_scheduling=[2000, 0.9], specify_fold=1, testing=False)


# In[ ]:


train_lstm("/home/jl5307/current_research/AMD_prediction/results/convlstm/5_fold/td5/fold2/", 
            "/home/jl5307/current_research/AMD_prediction/img_data/5_fold/five_fold_unrolled_longitudinal_prediction_td5_min5_data_dict.pkl",
            "/home/jl5307/current_research/AMD_prediction/results/resnet101_amd_detection/fold2/extracted_feature_dict.pkl",
            30, 32, 256, 2048, 0.0002, 0.0001, [0.03, 0.97], lr_scheduling=[2000, 0.9], specify_fold=2, testing=False)


# In[ ]:


train_lstm("/home/jl5307/current_research/AMD_prediction/results/convlstm/5_fold/td5/fold3/", 
            "/home/jl5307/current_research/AMD_prediction/img_data/5_fold/five_fold_unrolled_longitudinal_prediction_td5_min5_data_dict.pkl",
            "/home/jl5307/current_research/AMD_prediction/results/resnet101_amd_detection/fold3/extracted_feature_dict.pkl",
            30, 32, 256, 2048, 0.0002, 0.0001, [0.03, 0.97], lr_scheduling=[2000, 0.9], specify_fold=3, testing=False)


# In[ ]:


train_lstm("/home/jl5307/current_research/AMD_prediction/results/convlstm/5_fold/td5/fold4/", 
            "/home/jl5307/current_research/AMD_prediction/img_data/5_fold/five_fold_unrolled_longitudinal_prediction_td5_min5_data_dict.pkl",
            "/home/jl5307/current_research/AMD_prediction/results/resnet101_amd_detection/fold4/extracted_feature_dict.pkl",
            30, 32, 256, 2048, 0.0002, 0.0001, [0.03, 0.97], lr_scheduling=[2000, 0.9], specify_fold=4, testing=False)

