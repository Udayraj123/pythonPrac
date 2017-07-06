# Trying to reproduce results from
# paper :    arXiv : 1412.7828v2 [q-bio.QM] 4 Jan 2015
# Experiment name : Protein secondary structure prediction using
# LSTM networks.

# Model :
#  -- Standard stacked bidirectional LSTM with 3 layers.
#  -- (300 or 500) LSTM units in each layer
#  -- There is a FFN between h_rec and h with a skip connection. h_rec = ffn(h) + h
#  -- FFN is a two layer ReLU network with 300 or 500 units,
#  -- Introduce a FFN to combine output from forward and backward RNN
#  -- Has a ReLU with 200 or 400 hidden units.
#  -- The concatenation is regularized with 50% dropout.

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pandas as pd

print(tf.__version__)

"""
cb513+profile_split1.npy.gz
cullpdb+profile_6133.npy.gz
cullpdb+profile_6133_filtered.npy.gz
"""

data = np.load('./data/cullpdb+profile_6133.npy.gz')
print(data.shape)
data = np.reshape(data, [6133, 700, 57])
print(data.shape)

# print(data.info())
cv_data = data[5600:5877, :]
train_data = np.load('./data/cullpdb+profile_6133_filtered.npy')
test_data = np.load('./data/cb513+profile_split1.npy')
train_data = np.reshape(train_data, [-1, 700, 57])
test_data = np.reshape(test_data, [-1, 700, 57])
print(train_data.shape)
print(test_data.shape)
"""
Source : arXiv:1403.1347v1  [q-bio.QM]  6 Mar 2014
:Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction
The resulting training data including both feature and la-
bels has 57 channels (22 for PSSM, 22 for sequence, 2 for
terminals,  8  for  secondary  structure  labels,  2  for  solvent
accessibility  labels),  and  the  overall  channel  size  is  700.
"""

"""
Source : http://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt
It is currently in numpy format as a (N protein x k features) matrix. You can reshape it to (N protein x 700 amino acids x 57 features) first.
The 57 features are:
"[0,22): amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F',
'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq'"
"[22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H',
'S', 'T','NoSeq'"
"[31,33): N- and C- terminals;"
"[33,35): relative and absolute solvent accessibility, used only for training.
(absolute accessibility is thresholded at 15; relative accessibility is normalized by the largest accessibility
value in a protein and thresholded at 0.15; original solvent accessibility is computed by DSSP)"
"[35,57): sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and
it is different from the order for amino acid residues"
The last feature of both amino acid residues and secondary structure labels just mark end of the protein sequence.
"[22,31) and [33,35) are hidden during testing."
"The dataset division for the first ""cullpdb+profile_6133.npy.gz"" dataset is"
"[0,5600) training"
"[5605,5877) test "
"[5877,6133) validation"
"""
# Split the train data
train_data_residues = train_data[:, :,  0:21]
train_data_secstruc = train_data[:, :, 22:30]
train_data_nctermin = train_data[:, :, 31:33]
train_data_rlabsolv = train_data[:, :, 33:35]
train_data_sequepro = train_data[:, :, 35:57]

# Checking shapes
print("Train data residues shape : ", train_data_residues.shape)
print("Train data secondary structue : ",train_data_secstruc.shape)
print("Train data n and c terminals : ", train_data_nctermin.shape)
print("Train data relative and absolute solvability : ", train_data_rlabsolv.shape)
print("Train data sequence profile : ", train_data_sequepro.shape)

train_data_input = train_data[:, :, np.r_[0:21, 36:57]]
train_data_otput = train_data[:, :, 23:31]
test_data_input = test_data[:, :, np.r_[0:21, 36:57]]
test_data_otput = test_data[:, :, 23:31]
# Checking shapes
# print("Train data input  shape : ", train_data_input.shape)
# print("Train data output shape : ", train_data_otput.shape)

learning_rate = 0.00000001
n_epochs = 1000
num_classes = 8
hidden_units = 300

class BrnnForPssp():

    def __init__(self, learning_rate, num_classes, hidden_units):

        # Initialize data and variables
        self.weights = tf.Variable(tf.random_uniform([hidden_units*2, num_classes], minval=-0.5, maxval=0.5))
        self.biases  = tf.Variable(tf.zeros([num_classes]))
        self.x = tf.placeholder("float", [None, 700, 42])
        self.y = tf.placeholder("float", [None, 700, 8])

        # Do the prediction

        # Remember to change activation to ReLU
        self.fw_rnn_cell1 = rnn.LSTMCell(hidden_units, forget_bias=1.0)
        self.fw_rnn_cell2 = rnn.LSTMCell(hidden_units, forget_bias=1.0)
        self.fw_rnn_cell3 = rnn.LSTMCell(hidden_units, forget_bias=1.0)
        self.bw_rnn_cell1 = rnn.LSTMCell(hidden_units, forget_bias=1.0)
        self.bw_rnn_cell2 = rnn.LSTMCell(hidden_units, forget_bias=1.0)
        self.bw_rnn_cell3 = rnn.LSTMCell(hidden_units, forget_bias=1.0)
        # self.fw_rnn_cells = [self.fw_rnn_cell1, self.fw_rnn_cell2, self.fw_rnn_cell3]
        # self.bw_rnn_cells = [self.bw_rnn_cell1, self.bw_rnn_cell2, self.bw_rnn_cell3]
        self.fw_rnn_cells = [self.fw_rnn_cell1]
        self.bw_rnn_cells = [self.bw_rnn_cell1]
        self.outputs, self.states_fw, self.states_bw = rnn.stack_bidirectional_dynamic_rnn(
                                                            self.fw_rnn_cells,
                                                            self.bw_rnn_cells,
                                                            self.x,
                                                            dtype=tf.float32)
        # self.output.shape is (?, 700, 600)
        self.outputs_reshaped = tf.reshape(self.outputs, [-1, 2*hidden_units])
        self.y_reshaped = tf.reshape(self.y, [-1, num_classes])
        # check importantFunctions.py : line-40 to see how it works
        # reference link  is :
        # https://stackoverflow.com/questions/38051143/no-broadcasting-for-tf-matmul-in-tensorflow
        self.y_predicted = (tf.matmul(self.outputs_reshaped, self.weights) + self.biases)

        # Define the loss function
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.y_reshaped)

        # Define the trainer and optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.trainer = self.optimizer.minimize(self.loss)

        # creating session and initializing variables
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        # get accuracy
        self.get_equal = tf.equal(tf.argmax(self.y_reshaped, 1), tf.argmax(self.y_predicted, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.get_equal, tf.float32))

    def predict(self, x, y):
        result = self.sess.run(self.y_predicted, feed_dict={self.x: x, self.y: y})
        return result

    def optimize(self, x, y):
        print("Accuracy and loss on next-to-be-trained batch : ", self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y}),
        	                                             self.sess.run(self.loss, feed_dict={self.x: x, self.y: y}))
        result = self.sess.run(self.trainer, feed_dict={self.x: x, self.y: y})
        print("Accuracy and loss obtained after training     : ", self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y}),
        	                                             self.sess.run(self.loss, feed_dict={self.x: x, self.y: y}))

    def cross_validate(self, x, y):
        result = self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y})
        return result

    def build_graph(self, x, y):
        writer = tf.summary.FileWriter('./graphs/lstmForPSSP',self.sess.graph)



model = BrnnForPssp(learning_rate=learning_rate, num_classes=8, hidden_units=hidden_units)
print("Successfully created the model")

for i in range(n_epochs):
    if i % 10 == 0:
        x = test_data_input
        y = test_data_otput
        print(i, "td - ", model.cross_validate(x=x, y=y))
    j = i%56 + 1
    x = train_data_input[j*100:j*100+100, :]
    y = train_data_otput[j*100:j*100+100, :]
    print("Iteration number & batch number: ", i, j)
    model.optimize(x=x, y=y)