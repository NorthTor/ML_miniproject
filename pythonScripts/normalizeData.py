
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import math
import random, os
import shutil 



def import_data_mfcc(data_path):

    print('Importing Json data... please wait')
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    Y = np.array(data["labels"])
    Z = np.array(data["mapping"])

    return X, Y, Z


def normalize_data(data_matrix_3D, labels, mapping, new_json_path):

    data = {
        "mapping": mapping.tolist(),
        "mfcc": [],
        "labels": labels.tolist()
    }

    print('type datamatrix:', type(data_matrix_3D), np.shape(data_matrix_3D))
    # subtract the mean and divide by std deviation
    # input is a 3D vector
    nbr_mfcc = data_matrix_3D[2]

    nbr_datapoints = np.shape(data_matrix_3D)[0] * np.shape(data_matrix_3D)[1] * np.shape(data_matrix_3D)[2]
    # mean and std deviation
    mean = np.mean(data_matrix_3D)
    std = np.std(data_matrix_3D)
    # normalize all datapoint in 3D matrix
    data_normalized = (data_matrix_3D - mean) / std

    # put normalized data back into json format 
    for i in range(np.shape(data_normalized)[0]):
        print(i + 1, 'of', np.shape(data_normalized)[0], 'audio samples normalized')
        mfcc_array = data_normalized[i,:,:] # 20x1 vector  or nx1 where n is the number of MFFCs used
        #print(np.shape(mfcc_array))
        data["mfcc"].append(mfcc_array.tolist())

    print('Saving Json data... please wait')
    # save to JSON
    with open(new_json_path, 'w') as fp:
        json.dump(data, fp, indent=4)


#--- End functions, start main -----------------------------------------------------------------------------

DATA_PATH = "../Features/mfcc_500.json"
NEW_JSON_PATH = "../Features/normalized_mfcc_500.json"

# import data
data, labels_data, mapping_data = import_data_mfcc(DATA_PATH)
print("type", type(data), np.shape(data))
# do normalization on complete dataset
normalize_data(data, labels_data, mapping_data, NEW_JSON_PATH)





