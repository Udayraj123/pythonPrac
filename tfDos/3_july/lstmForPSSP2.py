import tensorflow as tf
import numpy as np
import pandas as pd

data_train = np.load('./data/cullpdb+profile_6133_filtered.npy')
data_test = np.load('./data/cb513+profile_split1.npy')

print("\nData laoaded. ")
print("Train data shape : ", data_train.shape)
print("Test data shape : ", data_test.shape)

print("\nReshaping data. ")
data_train = np.reshape(data_train, [-1, 700, 57])
data_test = np.reshape(data_test, [-1, 700, 57])
print("Train data shape : ", data_train.shape)
print("Test data shape : ", data_test.shape)

train_data_input = []

class SingleLayerBRNNforPSSP():


