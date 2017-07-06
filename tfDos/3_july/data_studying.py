import numpy as np
import pandas as pd
import sklearn.metrics as skm

print("Loading the data : ")
# print(data.info())
train_data = np.load('./data/cullpdb+profile_6133_filtered.npy')
test_data = np.load('./data/cb513+profile_split1.npy')
# train1_data = train_data[:3,:57*4]
# test1_data = test_data[:3,:57*4]
# train2_data = np.copy(train1_data)
# test2_data = np.copy(test1_data)

# print(train1_data)
# print(test1_data)

# for i in range(3):
# 	for j in range(57*4):
# 		print("Train data ", i, j, j // 57, j%57, " : ", train1_data[i, j])

# for i in range(3):
# 	for j in range(57*4):
# 		print("Test  data ", i, j, j // 57, j%57, " : ", test1_data[i, j])

# train1_data = np.reshape(train1_data, [-1, 4, 57])
# test1_data = np.reshape(test1_data, [-1, 4, 57])

# for i in range(train1_data.shape[0]):
# 	for j in range(4):
# 		for k in range(57):
# 			print("Train data ", i, j*57+k, j, k, train2_data[i, j*57+k], train1_data[i, j, k])

# for i in range(test1_data.shape[0]):
# 	for j in range(4):
# 		for k in range(57):
# 			print("Test data ", i, j*57+k, j, k, test2_data[i, j*57+k], test1_data[i, j, k])

# # train_data_otput = train1_data[:, :, np.r_[22:30, 31:32]]
# train_data_otput = train1_data[:, :, np.r_[22:30]]
# test_data_otput = test1_data[:, :, 22:30]
# print(train_data_otput.shape)
# print(test_data_otput.shape)

# train_data_otput = np.reshape(train_data_otput, [-1, 8])
# test_data_otput = np.reshape(test_data_otput, [-1, 8])

# print(pd.DataFrame(train_data_otput).mean(axis = 1))
# print(pd.DataFrame(test_data_otput).mean(axis = 1))

print("Printing data shape : ")
print(train_data.shape)
print(test_data.shape)
print("Printing data shape : ")
print(train_data.shape)
print(test_data.shape)
train_data = np.reshape(train_data, [-1, 700, 57])
test_data = np.reshape(test_data, [-1, 700, 57])
print("Printing data shape : ")
print(train_data.shape)
print(test_data.shape)

train_data_input = train_data[:, :, np.r_[0:21, 36:57]]
train_data_otput = train_data[:, :, 22:30]
test_data_input = test_data[:, :, np.r_[0:21, 36:57]]
test_data_otput = test_data[:, :, 22:30]

print(train_data_input.shape)
print(train_data_otput.shape)
print(test_data_input.shape)
print(test_data_otput.shape)

train_data_otput = np.reshape(train_data_otput, [-1, 8])
test_data_otput = np.reshape(test_data_otput, [-1, 8])

# temp = pd.DataFrame(test_data_otput).mean(axis = 1)
# for i in range(1000):
# 	print(i, temp[i])

print(pd.DataFrame(test_data_otput).mean(axis = 1))
print(pd.DataFrame(train_data_otput).mean(axis = 1))
# print(pd.DataFrame(test_data_otput).count(axis = 1))
# print(pd.DataFrame(train_data_otput).count(axis = 1))


