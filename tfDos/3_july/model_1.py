import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from random import shuffle
import numpy as np
from sklearn.metrics import classification_report as c_metric

def get_data_train():
  file_path = './data/batch_wise_data.pkl'
  file_ip = open(file_path, 'rb')
  data_train = pickle.load(file_ip)
  file_ip.close()
  print("Data has been loaded. ")
  return data_train

class BrnnForPsspModelOne:
  def __init__(self,
  	num_classes = 8,
  	hidden_units = 100):
    self.input_x = tf.placeholder(tf.float64, [ 5, 800, 100])
    self.input_y = tf.placeholder(tf.int64, [ 5, 800])
    self.input_msks = tf.placeholder(tf.float64, [ 5, 800])
    self.input_seq_len = tf.placeholder(tf.int64, [ 5])
    self.input_y_o = tf.one_hot(indices = self.input_y,
      depth = num_classes,
      on_value = 1.0,
      off_value = 0.0,
      axis = -1)

    self.hidden_units = tf.constant(hidden_units, dtype = tf.float64)
    # define weights and biases here (4 weights + 4 biases)
    self.weight_f_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_f_p = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_p = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.biases_f_c = tf.Variable(0.0000001 * tf.random_uniform(shape=[num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_c = tf.Variable(0.0000001 * tf.random_uniform(shape=[num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_f_p = tf.Variable(0.0000001 * tf.random_uniform(shape=[num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_p = tf.Variable(0.0000001 * tf.random_uniform(shape=[num_classes], dtype=tf.float64), dtype=tf.float64)
		
    self.rnn_cell_f = rnn.GRUCell(num_units = hidden_units, 
 		activation = tf.tanh)
    self.rnn_cell_b = rnn.GRUCell(num_units = hidden_units, 
 		activation = tf.tanh)
    self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(
 		cell_fw = self.rnn_cell_f,
 		cell_bw = self.rnn_cell_b,
 		inputs = self.input_x,
 		sequence_length = self.input_seq_len,
 		dtype = tf.float64,
 		swap_memory = False)
    self.outputs_f = self.outputs[0]
    self.outputs_b = self.outputs[1]
    self.outputs_f_p_l = []
    self.outputs_b_p_l = []
    for i in range(700):
      # 50 dummies + seq + 50 dummies
      # For forward maxpooling, index i will have maxpool from i-50:i 
      # Loss due to dummies will get maske completely 
      self.outputs_f_p_l.append(tf.reduce_max(self.outputs_f[: , i:i+50, :],
        axis = 1))
      self.outputs_b_p_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+101, :],
      	axis = 1))
    self.outputs_f_p = tf.stack(self.outputs_f_p_l, axis = 1)
    self.outputs_b_p = tf.stack(self.outputs_b_p_l, axis = 1)
    self.outputs_f_c = tf.slice(self.outputs_f, [0, 50, 0], [ 5, 700, 100])
    self.outputs_b_c = tf.slice(self.outputs_b, [0, 50, 0], [ 5, 700, 100])

    self.outputs_f_c_r = tf.reshape(self.outputs_f_c, [-1, 100])
    self.outputs_b_c_r = tf.reshape(self.outputs_b_c, [-1, 100])
    self.outputs_f_p_r = tf.reshape(self.outputs_f_p, [-1, 100])
    self.outputs_b_p_r = tf.reshape(self.outputs_b_p, [-1, 100])

    self.y_predicted = ( tf.matmul(self.outputs_f_c_r, self.weight_f_c) + self.biases_f_c
                       + tf.matmul(self.outputs_b_c_r, self.weight_b_c) + self.biases_b_c
                       + tf.matmul(self.outputs_f_p_r, self.weight_f_p) + self.biases_f_p
                       + tf.matmul(self.outputs_b_p_r, self.weight_b_p) + self.biases_b_p )
    # [ 5*700, 8] <- self.y_predicted 
    self.input_y_o_s = tf.slice(self.input_y_o, [0, 50, 0], [ 5, 700, 8])
    self.input_msks_s = tf.slice(self.input_msks, [0, 50], [ 5, 700])
    # [ 5, 700, 8] <- self.input_y_o_s
    self.input_y_o_r = tf.reshape(self.input_y_o_s, [-1, 8])
    self.input_msks_r = tf.reshape(self.input_msks_s, [-1, 1])
    # [ 5*700, 8] <- self.input_y_o_r
    self.loss_unmasked = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.input_y_o_r), [3500, 1])
    #  dim: The class dimension. Defaulted to -1 
    #  which is the last dimension.
    self.loss_masked = tf.multiply(self.loss_unmasked, self.input_msks_r)
    self.no_of_entries_unmasked = tf.reduce_sum(self.input_msks_r)
    self.loss_reduced = ( tf.reduce_sum(self.loss_masked) / self.no_of_entries_unmasked )
	
    self.get_equal_unmasked = tf.reshape(tf.equal(tf.argmax(self.input_y_o_r, 1), tf.argmax(self.y_predicted, 1)), [3500, 1])
    self.get_equal = tf.multiply(tf.cast(self.get_equal_unmasked, tf.float64), self.input_msks_r)
    self.accuracy = ( tf.reduce_sum(tf.cast(self.get_equal, tf.float64)) / self.no_of_entries_unmasked)

    # define optimizer and trainer
    self.optimizer_1 = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    self.trainer_1 = self.optimizer_1.minimize(self.loss_reduced)

    self.optimizer_2 = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    self.trainer_2 = self.optimizer_2.minimize(self.loss_reduced)

    self.optimizer_3 = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
    self.trainer_3 = self.optimizer_3.minimize(self.loss_reduced)

    self.optimizer_mini = tf.train.AdamOptimizer(learning_rate = 0.000001)
    self.trainer_mini = self.optimizer_mini.minimize(self.loss_reduced)

    self.sess = tf.Session()
    self.init = tf.global_variables_initializer()
    self.sess.run(self.init)

  def optimize_mini(self, x, y, seq_len, msks):
    result, loss, accuracy, no_of_entries_unmasked = self.sess.run([self.trainer_mini,
		self.loss_reduced,
		self.accuracy,
		self.no_of_entries_unmasked],
		feed_dict={self.input_x:x, 
		self.input_y:y,
		self.input_seq_len:seq_len,
		self.input_msks:msks})
    return loss, accuracy, no_of_entries_unmasked

  def get_loss_and_predictions(self, x, y, seq_len, msks):
    loss_unmasked, loss_masked, loss_reduced, input_msks_r, y_predicted, input_y_o_r = self.sess.run([
    	self.loss_unmasked,
    	self.loss_masked,
    	self.loss_reduced,
    	self.input_msks_r,
    	self.y_predicted,
    	self.input_y_o_r],
    	feed_dict = {self.input_x:x, 
		self.input_y:y,
		self.input_seq_len:seq_len,
		self.input_msks:msks})
    return loss_unmasked, loss_masked, loss_reduced, input_msks_r, y_predicted, input_y_o_r 

  def get_shapes(self):
  	print("(self.loss_unmasked.shape)", self.loss_unmasked.shape)
  	print("(self.loss_masked.shape)", self.loss_masked.shape)
  	print("(self.loss_reduced.shape)", self.loss_reduced.shape)
  	print("(self.y_predicted.shape)", self.y_predicted.shape)
  	print("(self.input_y_o_r.shape)", self.input_y_o_r.shape)
  	# print(y.y_predicted.shape)
  	print("(self.input_msks_r.shape)", self.input_msks_r.shape)
  	print("(self.get_equal_unmasked.shape)", self.get_equal_unmasked.shape)
  	print("(self.get_equal.shape)", self.get_equal.shape)

if __name__=="__main__":
  data_train = get_data_train()
  # for batch_no in range(43):
  print("Creating model...")
  model = BrnnForPsspModelOne()
  print("Model creation finished. ")
  model.get_shapes()
  n_epochs = 1000
  for epoch in range(n_epochs):
    for batch_no in range(2):
      print("Epoch number and batch_no: ", epoch, batch_no)
      data = data_train[batch_no]
      x_inp = data[0]
      y_inp = data[1]
      m_inp = data[2]
      l_inp = data[3]
      x_inp = x_inp[:5]
      y_inp = y_inp[:5]
      m_inp = m_inp[:5]
      l_inp = l_inp[:5]
      loss_unmasked, loss_masked, loss_reduced, input_msks_r, y_predicted, input_y_o_r = model.get_loss_and_predictions(x_inp, y_inp, l_inp, m_inp)
      print("Loss before optimizing : ", loss_reduced)
      loss, accuracy, no_of_entries_unmasked = model.optimize_mini(x_inp, y_inp, l_inp, m_inp)
      no_of_entries_unmasked_inp = 0
      for i in range(5):
      	for j in range(len(m_inp[i])):
      	  no_of_entries_unmasked_inp += m_inp[i][j]
      # print(dtype(loss_unmasked), dtype(loss_masked), dtype(loss_reduced), dtype(input_msks_r))
      ans = True
      
      # debug portion
      # for i in range(3500):
      # 	print(loss_unmasked[i], loss_masked[i], input_msks_r[i], m_inp[i // 700][i % 700 + 50])
      # 	ans = ans and (input_msks_r[i] == m_inp[i // 700][i % 700 + 50])
      # 	ans = ans and (np.argmax(input_y_o_r[i], 0) == y_inp[i // 700][i % 700 + 50] or y_inp[i // 700][i % 700 + 50] == -1)
      # 	print(y_predicted[i])
      # 	print(input_y_o_r[i], y_inp[i // 700][i % 700 + 50])
      # 	if(ans == False):
      # 		debug = input()
      # 	if(i % 700 == 699):
      # 		debug = input()
      
      print("Loss, accuracy and verification results : ", loss, accuracy, ans)
      print("(no_of_entries_unmasked, no_of_entries_unmasked_inp)", no_of_entries_unmasked, no_of_entries_unmasked_inp)

"""
Epoch number and batch_no:  0 0
Loss, accuracy :  910.271072368 275.0
Epoch number and batch_no:  0 1
Loss, accuracy :  1281.51474569 291.0
Epoch number and batch_no:  1 0
Loss, accuracy :  890.325852686 279.0
Epoch number and batch_no:  1 1
Loss, accuracy :  1255.00815712 303.0
Epoch number and batch_no:  2 0
Loss, accuracy :  879.144031338 291.0
Epoch number and batch_no:  2 1
Loss, accuracy :  1239.58471894 314.0
Epoch number and batch_no:  3 0
Loss, accuracy :  874.015249401 278.0
Epoch number and batch_no:  3 1
Loss, accuracy :  1231.64421255 318.0
Epoch number and batch_no:  4 0
Loss, accuracy :  872.887020292 278.0
Epoch number and batch_no:  4 1
Loss, accuracy :  1228.63596089 326.0
Epoch number and batch_no:  5 0
Loss, accuracy :  874.31197826 283.0
Epoch number and batch_no:  5 1
Loss, accuracy :  1228.8423795 324.0
Epoch number and batch_no:  6 0
Loss, accuracy :  877.411856914 284.0
Epoch number and batch_no:  6 1
Loss, accuracy :  1231.18483386 323.0
Epoch number and batch_no:  7 0
Loss, accuracy :  881.312186267 283.0
Epoch number and batch_no:  7 1
Loss, accuracy :  1234.6607407 318.0
Epoch number and batch_no:  8 0
Loss, accuracy :  885.617966309 283.0
Epoch number and batch_no:  8 1
Loss, accuracy :  1238.65908374 318.0
Epoch number and batch_no:  9 0
Loss, accuracy :  889.887106462 280.0
Epoch number and batch_no:  9 1
Loss, accuracy :  1242.66251344 315.0
"""














