import numpy as np
import random
import pickle

def pad_data(cfp_features, clinical_features, labels, config):
    n_patients = len(cfp_features)
    lengths = np.array([len(features) for features in cfp_features])
    max_len = np.max(lengths)
    cfp_feature_size = config["cfp_feature_size"]
    clinical_feature_size = config["clinical_feature_size"]

    x = np.full((n_patients, max_len, cfp_feature_size), -1.).astype(np.float32)
    d = np.zeros((n_patients, clinical_feature_size)).astype(np.float32)
    y = np.full((n_patients, max_len), -1.).astype(np.float32)
    
    for idx, features in enumerate(cfp_features):
        for i, feature in enumerate(features):
            x[idx, i, :] = feature
        
    for idx, clinical_feature in enumerate(clinical_features):
        d[idx, :] = clinical_feature
    
    d = np.tile(d, [1,max_len])
    d = np.reshape(d, newshape=(n_patients, max_len, clinical_feature_size))

    for idx, label in enumerate(labels):
        y[idx, :(len(label)-1)] = label[1:]
        
    return x, d, y

def shuffle_data(mydata):
    mydata = np.array(mydata)
    idx = np.arange(len(mydata))
    random.shuffle(idx)
    
    return mydata[idx]

def data_to_chunk(data_dict, k):

    # parse data dictionary to list
    data_list = []

    for _, value in data_dict.items():
        data_list.append(value)

    data_list = shuffle_data(data_list)

    chunk_size = int(np.ceil(len(data_dict) / k))
    clinical_feature_chunks = []
    cfp_feature_chunks = []
    severe_score_chunks = []

    for i in range(k):
        clinical_feature_chunk = []
        cfp_feature_chunk = []
        severe_score_chunk = []
        this_data_chunk = data_list[i*chunk_size:(i+1)*chunk_size]

        for item in this_data_chunk:
            clinical_feature_chunk.append(item["re"]["clinical_feature"])
            clinical_feature_chunk.append(item["le"]["clinical_feature"])
            cfp_feature_chunk.append(item["re"]["feature"])
            cfp_feature_chunk.append(item["le"]["feature"])
            severe_score_chunk.append(item["re"]["severe_score"])
            severe_score_chunk.append(item["le"]["severe_score"])

        clinical_feature_chunks.append(clinical_feature_chunk)
        cfp_feature_chunks.append(cfp_feature_chunk)
        severe_score_chunks.append(severe_score_chunk)

    return clinical_feature_chunks * 2, cfp_feature_chunks * 2, severe_score_chunks * 2

def flatten_list(mylist):
    
    newlist = []
    for item in mylist:
        newlist.extend(item)
        
    return newlist
    
def load_data(data_path):
    data = pickle.load(open(data_path, 'rb'))

    return data

def save_data(output_path, mydata):
    with open(output_path, 'wb') as f:
        
        pickle.dump(mydata, f)