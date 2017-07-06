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
  	hidden_units = 100,
  	batch_size = 5):
    
    self.input_x = tf.placeholder(tf.float64, [ batch_size, 800, 100])
    self.input_y = tf.placeholder(tf.int64, [ batch_size, 800])
    self.input_msks = tf.placeholder(tf.float64, [ batch_size, 800])
    self.input_seq_len = tf.placeholder(tf.int64, [ batch_size])
    self.input_y_o = tf.one_hot(indices = self.input_y,
      depth = num_classes,
      on_value = 1.0,
      off_value = 0.0,
      axis = -1)

    self.hidden_units = tf.constant(hidden_units, dtype = tf.float64)
    # define weights and biases here (4 weights + 4 biases)
    self.weight_f_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_f_p_50 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_p_50 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_f_p_20 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_p_20 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_f_p_10 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_p_10 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_f_p_30 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.weight_b_p_30 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float64), dtype=tf.float64) 
    self.biases_f_c = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_c = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_f_p_50 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_p_50 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_f_p_20 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_p_20 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_f_p_10 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_p_10 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_f_p_30 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases_b_p_30 = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    self.biases = tf.Variable(tf.zeros([num_classes], dtype=tf.float64), dtype=tf.float64)
    
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
    self.outputs_f_p_50_l = []
    self.outputs_b_p_50_l = []
    self.outputs_f_p_20_l = []
    self.outputs_b_p_20_l = []
    # self.outputs_f_p_10_l = []
    # self.outputs_b_p_10_l = []
    # self.outputs_f_p_30_l = []
    # self.outputs_b_p_30_l = []
    for i in range(700):
      # 50 dummies + seq + 50 dummies
      # For forward maxpooling, index i will have maxpool from i-50:i 
      # Loss due to dummies will get maske completely 
      self.outputs_f_p_50_l.append(tf.reduce_max(self.outputs_f[: , i:i+50, :],
        axis = 1))
      self.outputs_b_p_50_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+101, :],
      	axis = 1))
      self.outputs_f_p_20_l.append(tf.reduce_max(self.outputs_f[: , i+30:i+50, :],
        axis = 1))
      self.outputs_b_p_20_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+71, :],
      	axis = 1))
      # self.outputs_f_p_10_l.append(tf.reduce_max(self.outputs_b[: , i+40:i+50, :],
      #   axis = 1))
      # self.outputs_b_p_10_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+61, :],
      #   axis = 1))
      # self.outputs_f_p_30_l.append(tf.reduce_max(self.outputs_b[: , i+20:i+50, :],
      #   axis = 1))
      # self.outputs_b_p_30_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+81, :],
      #   axis = 1))
    self.outputs_f_p_50 = tf.stack(self.outputs_f_p_50_l, axis = 1)
    self.outputs_b_p_50 = tf.stack(self.outputs_b_p_50_l, axis = 1)
    self.outputs_f_p_20 = tf.stack(self.outputs_f_p_20_l, axis = 1)
    self.outputs_b_p_20 = tf.stack(self.outputs_b_p_20_l, axis = 1)
    # self.outputs_f_p_10 = tf.stack(self.outputs_f_p_10_l, axis = 1)
    # self.outputs_b_p_10 = tf.stack(self.outputs_b_p_10_l, axis = 1)
    # self.outputs_f_p_30 = tf.stack(self.outputs_f_p_30_l, axis = 1)
    # self.outputs_b_p_30 = tf.stack(self.outputs_b_p_30_l, axis = 1)
    self.outputs_f_c = tf.slice(self.outputs_f, [0, 50, 0], [ batch_size, 700, 100])
    self.outputs_b_c = tf.slice(self.outputs_b, [0, 50, 0], [ batch_size, 700, 100])

    self.outputs_f_c_r = tf.reshape(self.outputs_f_c, [-1, 100])
    self.outputs_b_c_r = tf.reshape(self.outputs_b_c, [-1, 100])
    self.outputs_f_p_50_r = tf.reshape(self.outputs_f_p_50, [-1, 100])
    self.outputs_b_p_50_r = tf.reshape(self.outputs_b_p_50, [-1, 100])
    self.outputs_f_p_20_r = tf.reshape(self.outputs_f_p_20, [-1, 100])
    self.outputs_b_p_20_r = tf.reshape(self.outputs_b_p_20, [-1, 100])
    # self.outputs_f_p_30_r = tf.reshape(self.outputs_f_p_30, [-1, 100])
    # self.outputs_b_p_30_r = tf.reshape(self.outputs_b_p_30, [-1, 100])
    # self.outputs_f_p_10_r = tf.reshape(self.outputs_f_p_10, [-1, 100])
    # self.outputs_b_p_10_r = tf.reshape(self.outputs_b_p_10, [-1, 100])
    
    self.y_predicted = ( tf.matmul(self.outputs_f_c_r, self.weight_f_c)
                       + tf.matmul(self.outputs_b_c_r, self.weight_b_c)
                       + tf.matmul(self.outputs_f_p_50_r, self.weight_f_p_50)
                       + tf.matmul(self.outputs_b_p_50_r, self.weight_b_p_50) 
                       + tf.matmul(self.outputs_f_p_20_r, self.weight_f_p_20)
                       + tf.matmul(self.outputs_b_p_20_r, self.weight_b_p_20)
                       # + tf.matmul(self.outputs_f_p_10_r, self.weight_b_p_20)
                       # + tf.matmul(self.outputs_b_p_10_r, self.weight_b_p_20)
                       # + tf.matmul(self.outputs_f_p_30_r, self.weight_b_p_20)
                       # + tf.matmul(self.outputs_b_p_30_r, self.weight_b_p_20)
                       + self.biases)

    # self.y_predicted = ( tf.matmul(self.outputs_f_c_r, self.weight_f_c) + self.biases_f_c
    #                    + tf.matmul(self.outputs_b_c_r, self.weight_b_c) + self.biases_b_c
    #                    + tf.matmul(self.outputs_f_p_50_r, self.weight_f_p_50) + self.biases_f_p_50
    #                    + tf.matmul(self.outputs_b_p_50_r, self.weight_b_p_50) + self.biases_b_p_50 
    #                    + tf.matmul(self.outputs_f_p_20_r, self.weight_f_p_20) + self.biases_f_p_20
    #                    + tf.matmul(self.outputs_b_p_20_r, self.weight_b_p_20) + self.biases_b_p_20)
                       # + tf.matmul(self.outputs_f_p_30_r, self.weight_f_p_30) + self.biases_f_p_30
                       # + tf.matmul(self.outputs_b_p_30_r, self.weight_b_p_30) + self.biases_b_p_30
                       # + tf.matmul(self.outputs_f_p_10_r, self.weight_f_p_10) + self.biases_f_p_10
                       # + tf.matmul(self.outputs_b_p_10_r, self.weight_b_p_10) + self.biases_b_p_10)
    # [ batch_size*700, 8] <- self.y_predicted 
    self.input_y_o_s = tf.slice(self.input_y_o, [0, 50, 0], [ batch_size, 700, 8])
    self.input_msks_s = tf.slice(self.input_msks, [0, 50], [ batch_size, 700])
    # [ batch_size, 700, 8] <- self.input_y_o_s
    self.input_y_o_r = tf.reshape(self.input_y_o_s, [-1, 8])
    self.input_msks_r = tf.reshape(self.input_msks_s, [-1, 1])
    # [ batch_size*700, 8] <- self.input_y_o_r
    self.loss_unmasked = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.input_y_o_r), [batch_size*700, 1])
    #  dim: The class dimension. Defaulted to -1 
    #  which is the last dimension.
    self.loss_masked = tf.multiply(self.loss_unmasked, self.input_msks_r)
    self.no_of_entries_unmasked = tf.reduce_sum(self.input_msks_r)
    self.loss_reduced = ( tf.reduce_sum(self.loss_masked) / self.no_of_entries_unmasked )
	
    self.get_equal_unmasked = tf.reshape(tf.equal(tf.argmax(self.input_y_o_r, 1), tf.argmax(self.y_predicted, 1)), [batch_size*700, 1])
    self.get_equal = tf.multiply(tf.cast(self.get_equal_unmasked, tf.float64), self.input_msks_r)
    self.accuracy = ( tf.reduce_sum(tf.cast(self.get_equal, tf.float64)) / self.no_of_entries_unmasked)

    # define optimizer and trainer
    self.optimizer_1 = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    self.trainer_1 = self.optimizer_1.minimize(self.loss_reduced)

    self.optimizer_2 = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    self.trainer_2 = self.optimizer_2.minimize(self.loss_reduced)

    self.optimizer_3 = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
    self.trainer_3 = self.optimizer_3.minimize(self.loss_reduced)

    self.optimizer_mini = tf.train.AdamOptimizer(learning_rate = 1e-2)
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

  def print_biases(self, x, y, seq_len, msks):
    f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20, biases = self.sess.run([self.biases_f_c,
      self.biases_b_c,
      self.biases_f_p_50,
      self.biases_b_p_50,
      self.biases_f_p_20,
      self.biases_b_p_20,
      self.biases],
      feed_dict = {self.input_x:x, 
        self.input_y:y,
        self.input_seq_len:seq_len,
        self.input_msks:msks})
    # print("self.biases_f_c : ", f_c)
    # print("self.biases_b_c : ", b_c)
    # print("self.biases_f_p_50 : ", f_p_50)
    # print("self.biases_b_p_50 : ", b_p_50)
    # print("self.biases_f_p_20 : ", f_p_20)
    # print("self.biases_b_p_50 : ", b_p_20)
    print("self.biases : ", biases)

  def print_weights(self, x, y, seq_len, msks):
    f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20 = self.sess.run([self.weight_f_c,
      self.weight_b_c,
      self.weight_f_p_50,
      self.weight_b_p_50,
      self.weight_f_p_20,
      self.weight_b_p_20],
      feed_dict = {self.input_x:x, 
        self.input_y:y,
        self.input_seq_len:seq_len,
        self.input_msks:msks})
    print("self.weights_f_c : ", f_c)
    print("self.weights_b_c : ", b_c)
    print("self.weights_f_p_50 : ", f_p_50)
    print("self.weights_b_p_50 : ", b_p_50)
    print("self.weights_f_p_20 : ", f_p_20)
    print("self.weights_b_p_50 : ", b_p_20)

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
  
  def get_rnn_outputs(self, x, y, seq_len, msks):
    f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20 = self.sess.run([self.outputs_f_c_r,
      self.outputs_b_c_r,
      self.outputs_f_p_50_r,
      self.outputs_b_p_50_r,
      self.outputs_f_p_20_r,
      self.outputs_b_p_20_r],
      feed_dict = {self.input_x:x, 
        self.input_y:y,
        self.input_seq_len:seq_len,
        self.input_msks:msks})
    return f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20

def verify_accuracy(y_inp, y_pre, msk, epoch):
  total = 0
  correct = 0
  count_5 = 0
  count_5_inp = 0
  for i in range(len(y_pre)):
    if(i%700 == 699 and epoch > 25):
      print("\n\n")
    if(msk[i // 700] [i % 700 + 50] == 1):
      if(np.argmax(y_pre[i], 0) == 5):
        count_5 += 1
      if(y_inp[i // 700][i % 700 + 50] == 5):
        count_5_inp += 1
      total += 1
      if(epoch >= 25):
        print(i, np.argmax(y_pre[i], 0), y_inp[i // 700][i % 700 + 50])
      if(np.argmax(y_pre[i], 0) == y_inp[i // 700][i % 700 + 50]):
        correct += 1
  if(epoch > 25):
    debug = input()
  print("No of 5 s predicted, input", count_5, count_5/total, count_5_inp, count_5_inp/total)
  return correct/total

def get_c1_score(y_inp, y_pre, msk):
  y_predicted = []
  y_actual = []
  for i in range(len(y_pre)):
    if(msk[i // 700] [i % 700 + 50] == 1):
      y_predicted.append(np.argmax(y_pre[i], 0))
      y_actual.append(y_inp[i // 700][i % 700 + 50])
  print("F1 score results : \n", c_metric(y_actual, y_predicted))
  print("Predicted : \n", c_metric(y_predicted, y_predicted))
  

if __name__=="__main__":
  data_train = get_data_train()
  # for batch_no in range(43):
  print("Creating model...")
  model = BrnnForPsspModelOne()
  print("Model creation finished. ")
  model.get_shapes()
  n_epochs = 200
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
      # model.print_weights(x_inp, y_inp, l_inp, m_inp)
      # f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20 = model.get_rnn_outputs(x_inp, y_inp, l_inp, m_inp)
      # print("f_c : ", f_c)
      # print("b_c : ", b_c)
      # print("f_p_50 : ", f_p_50)
      # print("b_p_50 : ", b_p_50)
      # print("f_p_20 : ", f_p_20)
      # print("b_p_20 : ", b_p_20)

      loss_unmasked, loss_masked, loss_reduced, input_msks_r, y_predicted, input_y_o_r = model.get_loss_and_predictions(x_inp, y_inp, l_inp, m_inp)
      print("Loss before optimizing : ", loss_reduced)
      loss, accuracy, no_of_entries_unmasked = model.optimize_mini(x_inp, y_inp, l_inp, m_inp)
      # no_of_entries_unmasked_inp = 0
      # for i in range(5):
      # 	for j in range(len(m_inp[i])):
      # 	  no_of_entries_unmasked_inp += m_inp[i][j]
      # # print(dtype(loss_unmasked), dtype(loss_masked), dtype(loss_reduced), dtype(input_msks_r))
      ans = True
      # debugging snippet
      # for i in range(3500):
      #   print(loss_unmasked[i], loss_masked[i], input_msks_r[i], m_inp[i // 700][i % 700 + 50])
      #   ans = ans and (input_msks_r[i] == m_inp[i // 700][i % 700 + 50])
      #   ans = ans and (np.argmax(input_y_o_r[i], 0) == y_inp[i // 700][i % 700 + 50] or y_inp[i // 700][i % 700 + 50] == -1)
      #   print(y_predicted[i])
      #   print(input_y_o_r[i], y_inp[i // 700][i % 700 + 50])
      #   if(ans == False):
      #     debug = input()
      #   if(i % 700 == 699):
      #     debug = input()
      print("Loss, accuracy and verification results : ", loss, accuracy, ans)
      # print("no_of_entries_unmasked, no_of_entries_unmasked_inp", no_of_entries_unmasked, no_of_entries_unmasked_inp)
      # print("Verifying accuracy : ", verify_accuracy(y_inp, y_predicted, m_inp, epoch))
      get_c1_score(y_inp, y_predicted, m_inp)
      model.print_biases(x_inp, y_inp, l_inp, m_inp)
      # model.print_weights(x_inp, y_inp, l_inp, m_inp)

"""
2 batches with no of exmples 5 :
Epoch number and batch_no:  66 1
Loss before optimizing :  1.5238893197
Loss, accuracy and verification results :  1.5238893197 0.41252552757 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.64      0.13      0.22       305
          1       0.00      0.00      0.00        11
          2       0.42      0.39      0.40       289
          3       0.12      0.04      0.06        82
          5       0.40      0.93      0.56       475
          6       0.43      0.07      0.11       138
          7       0.00      0.00      0.00       169

avg / total       0.39      0.41      0.32      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        64
          2       1.00      1.00      1.00       266
          3       1.00      1.00      1.00        25
          5       1.00      1.00      1.00      1093
          6       1.00      1.00      1.00        21

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06100156 -0.09390066  0.02648934 -0.00381814 -0.13048878  0.05078882
 -0.03468548 -0.01644616]
Epoch number and batch_no:  67 0
Loss before optimizing :  1.32520188844
Loss, accuracy and verification results :  1.32520188844 0.537848605578 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.37      0.45      0.40       152
          1       0.00      0.00      0.00         4
          2       0.46      0.72      0.56       224
          3       0.67      0.26      0.37        70
          5       0.74      0.75      0.74       381
          6       0.13      0.09      0.10        82
          7       0.00      0.00      0.00        91

avg / total       0.50      0.54      0.50      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       189
          2       1.00      1.00      1.00       351
          3       1.00      1.00      1.00        27
          5       1.00      1.00      1.00       385
          6       1.00      1.00      1.00        52

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0613054  -0.09345114  0.02611104 -0.00314484 -0.13118774  0.05157372
 -0.03686981 -0.01679556]
Epoch number and batch_no:  67 1
Loss before optimizing :  1.46561302396
Loss, accuracy and verification results :  1.46561302396 0.44656228727 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.36      0.46      0.40       305
          1       0.00      0.00      0.00        11
          2       0.46      0.45      0.45       289
          3       0.25      0.02      0.04        82
          5       0.49      0.81      0.61       475
          6       0.00      0.00      0.00       138
          7       0.00      0.00      0.00       169

avg / total       0.34      0.45      0.37      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       396
          2       1.00      1.00      1.00       281
          3       1.00      1.00      1.00         8
          5       1.00      1.00      1.00       784

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0610774  -0.09282762  0.02608333 -0.00432961 -0.13182252  0.0522959
 -0.03887069 -0.01597929]
Epoch number and batch_no:  68 0
Loss before optimizing :  1.25539191959
Loss, accuracy and verification results :  1.25539191959 0.541832669323 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.41      0.31      0.35       152
          1       0.00      0.00      0.00         4
          2       0.56      0.57      0.56       224
          3       0.91      0.14      0.25        70
          5       0.55      0.94      0.70       381
          6       0.00      0.00      0.00        82
          7       0.00      0.00      0.00        91

avg / total       0.46      0.54      0.46      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       116
          2       1.00      1.00      1.00       227
          3       1.00      1.00      1.00        11
          5       1.00      1.00      1.00       650

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0603657  -0.0925301   0.02678621 -0.00475892 -0.13239892  0.05230062
 -0.03957987 -0.01537287]
Epoch number and batch_no:  68 1
Loss before optimizing :  1.50046203359
Loss, accuracy and verification results :  1.50046203359 0.421375085092 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.15      0.24       305
          1       0.00      0.00      0.00        11
          2       0.43      0.47      0.45       289
          3       1.00      0.01      0.02        82
          5       0.41      0.91      0.56       475
          6       0.00      0.00      0.00       138
          7       0.18      0.01      0.02       169

avg / total       0.42      0.42      0.33      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        80
          2       1.00      1.00      1.00       314
          3       1.00      1.00      1.00         1
          5       1.00      1.00      1.00      1063
          7       1.00      1.00      1.00        11

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06116985 -0.09231099  0.02740662 -0.00502483 -0.13292306  0.05102536
 -0.03844859 -0.01494597]
Epoch number and batch_no:  69 0
Loss before optimizing :  1.24377708764
Loss, accuracy and verification results :  1.24377708764 0.549800796813 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.48      0.30      0.37       152
          1       0.00      0.00      0.00         4
          2       0.43      0.85      0.57       224
          3       0.92      0.16      0.27        70
          5       0.70      0.79      0.74       381
          6       0.00      0.00      0.00        82
          7       0.20      0.04      0.07        91

avg / total       0.52      0.55      0.49      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        93
          2       1.00      1.00      1.00       447
          3       1.00      1.00      1.00        12
          5       1.00      1.00      1.00       432
          7       1.00      1.00      1.00        20

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06155766 -0.0925571   0.02699329 -0.00437529 -0.13339915  0.0509524
 -0.03672454 -0.01610752]
Epoch number and batch_no:  69 1
Loss before optimizing :  1.44705140975
Loss, accuracy and verification results :  1.44705140975 0.450646698434 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.37      0.41      0.39       305
          1       0.00      0.00      0.00        11
          2       0.47      0.40      0.43       289
          3       0.20      0.01      0.02        82
          5       0.48      0.87      0.62       475
          6       0.31      0.03      0.05       138
          7       0.33      0.01      0.01       169

avg / total       0.40      0.45      0.37      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       341
          2       1.00      1.00      1.00       246
          3       1.00      1.00      1.00         5
          5       1.00      1.00      1.00       861
          6       1.00      1.00      1.00        13
          7       1.00      1.00      1.00         3

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06186391 -0.09265426  0.02720587 -0.00491704 -0.13383144  0.0507567
 -0.03580634 -0.01678115]
Epoch number and batch_no:  70 0
Loss before optimizing :  1.20589666708
Loss, accuracy and verification results :  1.20589666708 0.567729083665 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.39      0.45      0.42       152
          1       0.00      0.00      0.00         4
          2       0.59      0.59      0.59       224
          3       0.87      0.19      0.31        70
          5       0.62      0.91      0.74       381
          6       0.27      0.11      0.16        82
          7       0.00      0.00      0.00        91

avg / total       0.51      0.57      0.51      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       174
          2       1.00      1.00      1.00       223
          3       1.00      1.00      1.00        15
          5       1.00      1.00      1.00       559
          6       1.00      1.00      1.00        33

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0611839  -0.09279936  0.02830575 -0.00499718 -0.13422356  0.05062864
 -0.03645549 -0.01688759]
Epoch number and batch_no:  70 1
Loss before optimizing :  1.42881541604
Loss, accuracy and verification results :  1.42881541604 0.467665078285 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.43      0.30      0.35       305
          1       0.00      0.00      0.00        11
          2       0.44      0.62      0.51       289
          3       1.00      0.01      0.02        82
          5       0.50      0.85      0.63       475
          6       0.25      0.07      0.10       138
          7       0.50      0.01      0.01       169

avg / total       0.47      0.47      0.39      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       216
          2       1.00      1.00      1.00       410
          3       1.00      1.00      1.00         1
          5       1.00      1.00      1.00       804
          6       1.00      1.00      1.00        36
          7       1.00      1.00      1.00         2

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06126599 -0.09272974  0.02899731 -0.00578099 -0.13457956  0.05052441
 -0.03815241 -0.0160821 ]
Epoch number and batch_no:  71 0
Loss before optimizing :  1.1788702178
Loss, accuracy and verification results :  1.1788702178 0.562749003984 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.57      0.23      0.33       152
          1       0.00      0.00      0.00         4
          2       0.44      0.88      0.59       224
          3       0.82      0.20      0.32        70
          5       0.66      0.84      0.74       381
          6       0.00      0.00      0.00        82
          7       0.00      0.00      0.00        91

avg / total       0.49      0.56      0.48      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        61
          2       1.00      1.00      1.00       444
          3       1.00      1.00      1.00        17
          5       1.00      1.00      1.00       482

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06155883 -0.09275578  0.02847188 -0.0055716  -0.1349025   0.05091682
 -0.03932638 -0.01544379]
Epoch number and batch_no:  71 1
Loss before optimizing :  1.40828694852
Loss, accuracy and verification results :  1.40828694852 0.461538461538 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.45      0.24      0.31       305
          1       0.00      0.00      0.00        11
          2       0.47      0.54      0.50       289
          3       1.00      0.01      0.02        82
          5       0.47      0.94      0.62       475
          6       0.00      0.00      0.00       138
          7       0.17      0.02      0.03       169

avg / total       0.41      0.46      0.37      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       166
          2       1.00      1.00      1.00       328
          3       1.00      1.00      1.00         1
          5       1.00      1.00      1.00       956
          7       1.00      1.00      1.00        18

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06269843 -0.09258163  0.02796876 -0.00579044 -0.13519571  0.05061918
 -0.03932881 -0.01507772]
Epoch number and batch_no:  72 0
Loss before optimizing :  1.14895460869
Loss, accuracy and verification results :  1.14895460869 0.580677290837 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.40      0.41      0.40       152
          1       0.00      0.00      0.00         4
          2       0.57      0.67      0.61       224
          3       0.80      0.17      0.28        70
          5       0.64      0.93      0.76       381
          6       0.00      0.00      0.00        82
          7       0.27      0.03      0.06        91

avg / total       0.51      0.58      0.51      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       156
          2       1.00      1.00      1.00       264
          3       1.00      1.00      1.00        15
          5       1.00      1.00      1.00       558
          7       1.00      1.00      1.00        11

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06286157 -0.09248298  0.02808311 -0.00542574 -0.13546153  0.05031781
 -0.03859469 -0.01553236]
Epoch number and batch_no:  72 1
Loss before optimizing :  1.40024224685
Loss, accuracy and verification results :  1.40024224685 0.475833900613 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.33      0.54      0.41       305
          1       0.00      0.00      0.00        11
          2       0.49      0.59      0.53       289
          3       0.50      0.04      0.07        82
          5       0.60      0.75      0.67       475
          6       0.80      0.03      0.06       138
          7       0.06      0.01      0.01       169

avg / total       0.47      0.48      0.42      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       497
          2       1.00      1.00      1.00       352
          3       1.00      1.00      1.00         6
          5       1.00      1.00      1.00       593
          6       1.00      1.00      1.00         5
          7       1.00      1.00      1.00        16

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06241449 -0.09222534  0.028359   -0.00650474 -0.13570272  0.05067717
 -0.03814532 -0.01627008]
Epoch number and batch_no:  73 0
Loss before optimizing :  1.12775396432
Loss, accuracy and verification results :  1.12775396432 0.589641434263 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.48      0.34      0.40       152
          1       0.00      0.00      0.00         4
          2       0.58      0.69      0.63       224
          3       0.80      0.23      0.36        70
          5       0.61      0.97      0.75       381
          6       0.20      0.02      0.04        82
          7       0.00      0.00      0.00        91

avg / total       0.51      0.59      0.51      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       108
          2       1.00      1.00      1.00       265
          3       1.00      1.00      1.00        20
          5       1.00      1.00      1.00       601
          6       1.00      1.00      1.00        10

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06185882 -0.09207689  0.02887329 -0.00645374 -0.13592157  0.05061433
 -0.03795658 -0.01650667]
Epoch number and batch_no:  73 1
Loss before optimizing :  1.40179257494
Loss, accuracy and verification results :  1.40179257494 0.462219196732 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.49      0.16      0.25       305
          1       0.00      0.00      0.00        11
          2       0.46      0.58      0.51       289
          3       0.40      0.05      0.09        82
          5       0.47      0.93      0.62       475
          6       0.32      0.12      0.17       138
          7       0.00      0.00      0.00       169

avg / total       0.40      0.46      0.37      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       102
          2       1.00      1.00      1.00       364
          3       1.00      1.00      1.00        10
          5       1.00      1.00      1.00       942
          6       1.00      1.00      1.00        50
          7       1.00      1.00      1.00         1

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06271493 -0.09183156  0.02919555 -0.0063575  -0.13612061  0.04985748
 -0.03856501 -0.01586372]
Epoch number and batch_no:  74 0
Loss before optimizing :  1.17211624054
Loss, accuracy and verification results :  1.17211624054 0.580677290837 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.49      0.41      0.45       152
          1       0.00      0.00      0.00         4
          2       0.45      0.91      0.60       224
          3       0.68      0.24      0.36        70
          5       0.76      0.77      0.77       381
          6       0.60      0.07      0.13        82
          7       0.00      0.00      0.00        91

avg / total       0.56      0.58      0.53      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       128
          2       1.00      1.00      1.00       453
          3       1.00      1.00      1.00        25
          5       1.00      1.00      1.00       385
          6       1.00      1.00      1.00        10
          7       1.00      1.00      1.00         3

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06297829 -0.09185986  0.02831566 -0.0058845  -0.13630109  0.05058591
 -0.03984441 -0.01553781]
Epoch number and batch_no:  74 1
Loss before optimizing :  1.49356742965
Loss, accuracy and verification results :  1.49356742965 0.413886997958 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.39      0.25      0.30       305
          1       0.00      0.00      0.00        11
          2       0.73      0.18      0.28       289
          3       0.13      0.06      0.08        82
          5       0.41      0.99      0.58       475
          6       0.60      0.02      0.04       138
          7       0.27      0.02      0.03       169

avg / total       0.45      0.41      0.32      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       191
          2       1.00      1.00      1.00        70
          3       1.00      1.00      1.00        38
          5       1.00      1.00      1.00      1154
          6       1.00      1.00      1.00         5
          7       1.00      1.00      1.00        11

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06384784 -0.09183276  0.02926949 -0.00603597 -0.13646499  0.04930613
 -0.0400761  -0.0148336 ]
Epoch number and batch_no:  75 0
Loss before optimizing :  1.18276037123
Loss, accuracy and verification results :  1.18276037123 0.581673306773 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.35      0.68      0.46       152
          1       0.00      0.00      0.00         4
          2       0.57      0.76      0.65       224
          3       0.77      0.24      0.37        70
          5       0.78      0.75      0.77       381
          6       0.00      0.00      0.00        82
          7       0.38      0.07      0.11        91

avg / total       0.56      0.58      0.54      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       298
          2       1.00      1.00      1.00       300
          3       1.00      1.00      1.00        22
          5       1.00      1.00      1.00       368
          7       1.00      1.00      1.00        16

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06240584 -0.09202406  0.03043092 -0.00572791 -0.13661353  0.04947662
 -0.04027704 -0.01546404]
Epoch number and batch_no:  75 1
Loss before optimizing :  1.4367397699
Loss, accuracy and verification results :  1.4367397699 0.434309053778 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.35      0.32      0.34       305
          1       0.00      0.00      0.00        11
          2       0.34      0.81      0.48       289
          3       0.45      0.06      0.11        82
          5       0.64      0.62      0.63       475
          6       0.00      0.00      0.00       138
          7       0.16      0.03      0.05       169

avg / total       0.39      0.43      0.38      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       284
          2       1.00      1.00      1.00       683
          3       1.00      1.00      1.00        11
          5       1.00      1.00      1.00       459
          6       1.00      1.00      1.00         1
          7       1.00      1.00      1.00        31

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06145677 -0.0921975   0.03024411 -0.00619442 -0.13674838  0.05067003
 -0.04050487 -0.01653196]
Epoch number and batch_no:  76 0
Loss before optimizing :  1.17004125148
Loss, accuracy and verification results :  1.17004125148 0.560756972112 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.18      0.28       152
          1       0.00      0.00      0.00         4
          2       0.49      0.76      0.60       224
          3       1.00      0.07      0.13        70
          5       0.60      0.94      0.73       381
          6       1.00      0.01      0.02        82
          7       0.25      0.01      0.02        91

avg / total       0.60      0.56      0.47      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        48
          2       1.00      1.00      1.00       347
          3       1.00      1.00      1.00         5
          5       1.00      1.00      1.00       599
          6       1.00      1.00      1.00         1
          7       1.00      1.00      1.00         4

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06142131 -0.09254323  0.02998817 -0.00522162 -0.13687077  0.05088794
 -0.04035101 -0.01712679]
Epoch number and batch_no:  76 1
Loss before optimizing :  1.46546212705
Loss, accuracy and verification results :  1.46546212705 0.42614023145 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.56      0.14      0.23       305
          1       0.00      0.00      0.00        11
          2       0.45      0.44      0.45       289
          3       0.38      0.04      0.07        82
          5       0.41      0.94      0.57       475
          6       0.57      0.03      0.06       138
          7       0.18      0.01      0.02       169

avg / total       0.43      0.43      0.33      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        77
          2       1.00      1.00      1.00       282
          3       1.00      1.00      1.00         8
          5       1.00      1.00      1.00      1084
          6       1.00      1.00      1.00         7
          7       1.00      1.00      1.00        11

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06321442 -0.09262039  0.03012939 -0.00471065 -0.13698203  0.04944417
 -0.0399747  -0.0165515 ]
Epoch number and batch_no:  77 0
Loss before optimizing :  1.20983549683
Loss, accuracy and verification results :  1.20983549683 0.551792828685 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.37      0.47      0.41       152
          1       0.00      0.00      0.00         4
          2       0.45      0.83      0.58       224
          3       0.81      0.24      0.37        70
          5       0.76      0.73      0.74       381
          6       1.00      0.01      0.02        82
          7       0.00      0.00      0.00        91

avg / total       0.58      0.55      0.50      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       196
          2       1.00      1.00      1.00       416
          3       1.00      1.00      1.00        21
          5       1.00      1.00      1.00       366
          6       1.00      1.00      1.00         1
          7       1.00      1.00      1.00         4

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06379348 -0.09273017  0.02936455 -0.00424975 -0.13708297  0.04953484
 -0.04016117 -0.01608869]
Epoch number and batch_no:  77 1
Loss before optimizing :  1.38141261429
Loss, accuracy and verification results :  1.38141261429 0.481960517359 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.34      0.58      0.43       305
          1       0.00      0.00      0.00        11
          2       0.55      0.52      0.53       289
          3       0.29      0.06      0.10        82
          5       0.59      0.77      0.67       475
          6       0.62      0.04      0.07       138
          7       0.19      0.03      0.05       169

avg / total       0.47      0.48      0.43      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       520
          2       1.00      1.00      1.00       270
          3       1.00      1.00      1.00        17
          5       1.00      1.00      1.00       628
          6       1.00      1.00      1.00         8
          7       1.00      1.00      1.00        26

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06349706 -0.09255676  0.02923354 -0.0052967  -0.13717474  0.04999964
 -0.04074517 -0.01557495]
Epoch number and batch_no:  78 0
Loss before optimizing :  1.13889248088
Loss, accuracy and verification results :  1.13889248088 0.589641434263 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.41      0.51      0.45       152
          1       0.00      0.00      0.00         4
          2       0.68      0.55      0.61       224
          3       0.80      0.23      0.36        70
          5       0.63      0.96      0.76       381
          6       0.20      0.01      0.02        82
          7       0.32      0.09      0.14        91

avg / total       0.55      0.59      0.53      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       188
          2       1.00      1.00      1.00       180
          3       1.00      1.00      1.00        20
          5       1.00      1.00      1.00       586
          6       1.00      1.00      1.00         5
          7       1.00      1.00      1.00        25

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06249442 -0.09257591  0.0304434  -0.00568366 -0.13725814  0.04993128
 -0.04113447 -0.01579663]
Epoch number and batch_no:  78 1
Loss before optimizing :  1.38216523109
Loss, accuracy and verification results :  1.38216523109 0.486725663717 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.47      0.30      0.36       305
          1       1.00      0.09      0.17        11
          2       0.48      0.58      0.53       289
          3       0.35      0.07      0.12        82
          5       0.52      0.90      0.66       475
          6       0.38      0.02      0.04       138
          7       0.24      0.11      0.15       169

avg / total       0.45      0.49      0.42      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       190
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       350
          3       1.00      1.00      1.00        17
          5       1.00      1.00      1.00       825
          6       1.00      1.00      1.00         8
          7       1.00      1.00      1.00        78

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06256535 -0.09288517  0.03156833 -0.00617198 -0.13733432  0.04943936
 -0.04106146 -0.01657365]
Epoch number and batch_no:  79 0
Loss before optimizing :  1.12247908712
Loss, accuracy and verification results :  1.12247908712 0.591633466135 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.61      0.30      0.40       152
          1       0.50      0.25      0.33         4
          2       0.44      0.91      0.60       224
          3       0.71      0.29      0.41        70
          5       0.74      0.85      0.79       381
          6       0.00      0.00      0.00        82
          7       0.40      0.02      0.04        91

avg / total       0.56      0.59      0.53      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        76
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       457
          3       1.00      1.00      1.00        28
          5       1.00      1.00      1.00       435
          6       1.00      1.00      1.00         1
          7       1.00      1.00      1.00         5

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06272379 -0.09367732  0.03119467 -0.00552625 -0.13740373  0.04959628
 -0.04051735 -0.01702879]
Epoch number and batch_no:  79 1
Loss before optimizing :  1.34484970517
Loss, accuracy and verification results :  1.34484970517 0.492171545269 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.42      0.32      0.37       305
          1       0.00      0.00      0.00        11
          2       0.49      0.66      0.56       289
          3       0.27      0.07      0.12        82
          5       0.53      0.88      0.66       475
          6       0.26      0.05      0.08       138
          7       0.20      0.01      0.01       169

avg / total       0.42      0.49      0.42      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       236
          2       1.00      1.00      1.00       390
          3       1.00      1.00      1.00        22
          5       1.00      1.00      1.00       789
          6       1.00      1.00      1.00        27
          7       1.00      1.00      1.00         5

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0634677  -0.09426461  0.03042389 -0.00520628 -0.13746712  0.0495113
 -0.04030679 -0.0163635 ]
Epoch number and batch_no:  80 0
Loss before optimizing :  1.12852387499
Loss, accuracy and verification results :  1.12852387499 0.585657370518 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.40      0.47      0.43       152
          1       0.00      0.00      0.00         4
          2       0.67      0.55      0.60       224
          3       0.63      0.27      0.38        70
          5       0.62      0.97      0.76       381
          6       0.33      0.07      0.12        82
          7       0.00      0.00      0.00        91

avg / total       0.52      0.59      0.52      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       182
          2       1.00      1.00      1.00       183
          3       1.00      1.00      1.00        30
          5       1.00      1.00      1.00       591
          6       1.00      1.00      1.00        18

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06339043 -0.09473155  0.03098353 -0.00468012 -0.13752483  0.04886624
 -0.04054115 -0.01538709]
Epoch number and batch_no:  80 1
Loss before optimizing :  1.33292933521
Loss, accuracy and verification results :  1.33292933521 0.510551395507 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.40      0.48      0.43       305
          1       0.00      0.00      0.00        11
          2       0.48      0.68      0.57       289
          3       0.39      0.11      0.17        82
          5       0.64      0.80      0.71       475
          6       0.23      0.07      0.10       138
          7       0.29      0.07      0.11       169

avg / total       0.46      0.51      0.46      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       364
          2       1.00      1.00      1.00       407
          3       1.00      1.00      1.00        23
          5       1.00      1.00      1.00       594
          6       1.00      1.00      1.00        40
          7       1.00      1.00      1.00        41

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06334038 -0.09475651  0.03117633 -0.00525422 -0.13757741  0.04903796
 -0.04155212 -0.01496018]
Epoch number and batch_no:  81 0
Loss before optimizing :  1.09991005533
Loss, accuracy and verification results :  1.09991005533 0.594621513944 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.46      0.43      0.44       152
          1       0.00      0.00      0.00         4
          2       0.51      0.80      0.62       224
          3       0.68      0.24      0.36        70
          5       0.72      0.86      0.78       381
          6       0.75      0.04      0.07        82
          7       0.28      0.08      0.12        91

avg / total       0.59      0.59      0.54      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       142
          2       1.00      1.00      1.00       353
          3       1.00      1.00      1.00        25
          5       1.00      1.00      1.00       455
          6       1.00      1.00      1.00         4
          7       1.00      1.00      1.00        25

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06284068 -0.09472921  0.03092996 -0.00541131 -0.13762524  0.04996223
 -0.04216208 -0.01578739]
Epoch number and batch_no:  81 1
Loss before optimizing :  1.34729424022
Loss, accuracy and verification results :  1.34729424022 0.460857726344 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.50      0.21      0.30       305
          1       0.00      0.00      0.00        11
          2       0.59      0.47      0.52       289
          3       0.55      0.07      0.13        82
          5       0.44      0.97      0.60       475
          6       0.27      0.02      0.04       138
          7       0.23      0.04      0.07       169

avg / total       0.44      0.46      0.38      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       127
          2       1.00      1.00      1.00       231
          3       1.00      1.00      1.00        11
          5       1.00      1.00      1.00      1059
          6       1.00      1.00      1.00        11
          7       1.00      1.00      1.00        30

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06377006 -0.09433098  0.03127593 -0.00547802 -0.13766895  0.04926727
 -0.04171614 -0.01650648]
Epoch number and batch_no:  82 0
Loss before optimizing :  1.05654585419
Loss, accuracy and verification results :  1.05654585419 0.616533864542 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.48      0.40      0.44       152
          1       0.00      0.00      0.00         4
          2       0.54      0.80      0.64       224
          3       0.77      0.33      0.46        70
          5       0.70      0.91      0.79       381
          6       0.60      0.07      0.13        82
          7       0.38      0.03      0.06        91

avg / total       0.59      0.62      0.56      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       128
          2       1.00      1.00      1.00       333
          3       1.00      1.00      1.00        30
          5       1.00      1.00      1.00       495
          6       1.00      1.00      1.00        10
          7       1.00      1.00      1.00         8

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06404347 -0.09395805  0.03170523 -0.00581875 -0.13770878  0.04886274
 -0.04134222 -0.01696294]
Epoch number and batch_no:  82 1
Loss before optimizing :  1.33427326731
Loss, accuracy and verification results :  1.33427326731 0.499659632403 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.38      0.55      0.45       305
          1       0.00      0.00      0.00        11
          2       0.44      0.72      0.55       289
          3       0.44      0.15      0.22        82
          5       0.70      0.70      0.70       475
          6       0.29      0.09      0.14       138
          7       0.40      0.01      0.02       169

avg / total       0.49      0.50      0.45      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       443
          2       1.00      1.00      1.00       473
          3       1.00      1.00      1.00        27
          5       1.00      1.00      1.00       476
          6       1.00      1.00      1.00        45
          7       1.00      1.00      1.00         5

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06382941 -0.09338802  0.03152695 -0.0074013  -0.13774517  0.04943029
 -0.04206774 -0.01653026]
Epoch number and batch_no:  83 0
Loss before optimizing :  1.0322760729
Loss, accuracy and verification results :  1.0322760729 0.627490039841 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.52      0.40      0.45       152
          1       0.00      0.00      0.00         4
          2       0.60      0.75      0.67       224
          3       0.86      0.26      0.40        70
          5       0.66      0.98      0.79       381
          6       0.36      0.06      0.10        82
          7       0.56      0.05      0.10        91

avg / total       0.60      0.63      0.56      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       117
          2       1.00      1.00      1.00       282
          3       1.00      1.00      1.00        21
          5       1.00      1.00      1.00       561
          6       1.00      1.00      1.00        14
          7       1.00      1.00      1.00         9

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06330239 -0.09298774  0.03175418 -0.0076091  -0.13777838  0.04952734
 -0.04284012 -0.01575902]
Epoch number and batch_no:  83 1
Loss before optimizing :  1.32374966329
Loss, accuracy and verification results :  1.32374966329 0.494213750851 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.47      0.28      0.35       305
          1       0.25      0.09      0.13        11
          2       0.53      0.52      0.53       289
          3       0.75      0.04      0.07        82
          5       0.50      0.97      0.66       475
          6       0.50      0.04      0.08       138
          7       0.33      0.11      0.17       169

avg / total       0.49      0.49      0.42      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       180
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       286
          3       1.00      1.00      1.00         4
          5       1.00      1.00      1.00       926
          6       1.00      1.00      1.00        12
          7       1.00      1.00      1.00        57

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06388298 -0.09266132  0.03244186 -0.00690932 -0.13780882  0.04835597
 -0.04282783 -0.01518918]
Epoch number and batch_no:  84 0
Loss before optimizing :  1.05590535387
Loss, accuracy and verification results :  1.05590535387 0.626494023904 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.50      0.43      0.46       152
          1       0.50      0.25      0.33         4
          2       0.53      0.88      0.66       224
          3       0.75      0.34      0.47        70
          5       0.76      0.86      0.81       381
          6       0.67      0.02      0.05        82
          7       0.35      0.12      0.18        91

avg / total       0.62      0.63      0.58      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       129
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       377
          3       1.00      1.00      1.00        32
          5       1.00      1.00      1.00       430
          6       1.00      1.00      1.00         3
          7       1.00      1.00      1.00        31

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06409842 -0.09277753  0.03208541 -0.00522894 -0.13783664  0.04814218
 -0.04219198 -0.01565027]
Epoch number and batch_no:  84 1
Loss before optimizing :  1.29775575307
Loss, accuracy and verification results :  1.29775575307 0.506466984343 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.37      0.46      0.41       305
          1       0.50      0.09      0.15        11
          2       0.46      0.72      0.57       289
          3       0.43      0.26      0.32        82
          5       0.66      0.76      0.71       475
          6       0.39      0.05      0.09       138
          7       0.22      0.03      0.05       169

avg / total       0.47      0.51      0.46      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       385
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       450
          3       1.00      1.00      1.00        49
          5       1.00      1.00      1.00       542
          6       1.00      1.00      1.00        18
          7       1.00      1.00      1.00        23

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06403927 -0.09282288  0.03115726 -0.0053269  -0.1378621   0.04881272
 -0.04173569 -0.01586479]
Epoch number and batch_no:  85 0
Loss before optimizing :  1.02262005085
Loss, accuracy and verification results :  1.02262005085 0.609561752988 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.43      0.46      0.45       152
          1       0.00      0.00      0.00         4
          2       0.67      0.60      0.64       224
          3       0.72      0.33      0.45        70
          5       0.65      0.98      0.78       381
          6       0.31      0.10      0.15        82
          7       0.33      0.03      0.06        91

avg / total       0.57      0.61      0.55      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       161
          2       1.00      1.00      1.00       201
          3       1.00      1.00      1.00        32
          5       1.00      1.00      1.00       575
          6       1.00      1.00      1.00        26
          7       1.00      1.00      1.00         9

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06361281 -0.09289379  0.03139121 -0.00574563 -0.13788531  0.04891104
 -0.0417074  -0.01570858]
Epoch number and batch_no:  85 1
Loss before optimizing :  1.28245106651
Loss, accuracy and verification results :  1.28245106651 0.503744043567 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.43      0.36      0.39       305
          1       0.00      0.00      0.00        11
          2       0.58      0.43      0.50       289
          3       0.59      0.16      0.25        82
          5       0.52      0.97      0.68       475
          6       0.43      0.14      0.22       138
          7       0.28      0.08      0.12       169

avg / total       0.48      0.50      0.45      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       258
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       212
          3       1.00      1.00      1.00        22
          5       1.00      1.00      1.00       883
          6       1.00      1.00      1.00        47
          7       1.00      1.00      1.00        46

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06381075 -0.09281635  0.03262702 -0.00632662 -0.13790654  0.04806418
 -0.04229014 -0.01533409]
Epoch number and batch_no:  86 0
Loss before optimizing :  1.03359693327
Loss, accuracy and verification results :  1.03359693327 0.626494023904 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.41      0.47       152
          1       0.00      0.00      0.00         4
          2       0.51      0.88      0.64       224
          3       0.78      0.30      0.43        70
          5       0.77      0.88      0.82       381
          6       1.00      0.01      0.02        82
          7       0.33      0.14      0.20        91

avg / total       0.65      0.63      0.58      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       113
          2       1.00      1.00      1.00       390
          3       1.00      1.00      1.00        27
          5       1.00      1.00      1.00       434
          6       1.00      1.00      1.00         1
          7       1.00      1.00      1.00        39

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06380864 -0.09301984  0.03256897 -0.00553626 -0.13792588  0.04805687
 -0.04225244 -0.01562611]
Epoch number and batch_no:  86 1
Loss before optimizing :  1.2597459123
Loss, accuracy and verification results :  1.2597459123 0.53982300885 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.43      0.51      0.47       305
          1       0.00      0.00      0.00        11
          2       0.46      0.75      0.57       289
          3       0.65      0.18      0.29        82
          5       0.69      0.80      0.74       475
          6       0.47      0.12      0.19       138
          7       0.35      0.05      0.08       169

avg / total       0.52      0.54      0.49      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       365
          2       1.00      1.00      1.00       470
          3       1.00      1.00      1.00        23
          5       1.00      1.00      1.00       554
          6       1.00      1.00      1.00        34
          7       1.00      1.00      1.00        23

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06380627 -0.09300782  0.03164484 -0.00506291 -0.13794362  0.04875306
 -0.04238468 -0.0158679 ]
Epoch number and batch_no:  87 0
Loss before optimizing :  0.996479103325
Loss, accuracy and verification results :  0.996479103325 0.633466135458 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.42      0.46       152
          1       0.00      0.00      0.00         4
          2       0.67      0.69      0.68       224
          3       0.66      0.53      0.59        70
          5       0.65      0.97      0.78       381
          6       0.29      0.06      0.10        82
          7       0.57      0.04      0.08        91

avg / total       0.60      0.63      0.58      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       126
          2       1.00      1.00      1.00       230
          3       1.00      1.00      1.00        56
          5       1.00      1.00      1.00       568
          6       1.00      1.00      1.00        17
          7       1.00      1.00      1.00         7

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06389339 -0.09293366  0.03172577 -0.0057272  -0.13795981  0.04863221
 -0.0422886  -0.01549848]
Epoch number and batch_no:  87 1
Loss before optimizing :  1.25814656186
Loss, accuracy and verification results :  1.25814656186 0.507147719537 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.42      0.36      0.39       305
          1       0.00      0.00      0.00        11
          2       0.71      0.43      0.53       289
          3       0.39      0.22      0.28        82
          5       0.51      0.97      0.67       475
          6       0.43      0.17      0.25       138
          7       0.29      0.06      0.10       169

avg / total       0.49      0.51      0.45      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       263
          2       1.00      1.00      1.00       173
          3       1.00      1.00      1.00        46
          5       1.00      1.00      1.00       897
          6       1.00      1.00      1.00        56
          7       1.00      1.00      1.00        34

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06452291 -0.09246053  0.03295515 -0.00700143 -0.13797467  0.04737829
 -0.04257997 -0.01452634]
Epoch number and batch_no:  88 0
Loss before optimizing :  1.04085215406
Loss, accuracy and verification results :  1.04085215406 0.627490039841 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.46      0.56      0.51       152
          1       1.00      0.25      0.40         4
          2       0.50      0.86      0.64       224
          3       0.79      0.31      0.45        70
          5       0.85      0.78      0.81       381
          6       0.67      0.20      0.30        82
          7       0.47      0.18      0.26        91

avg / total       0.66      0.63      0.61      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       183
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       383
          3       1.00      1.00      1.00        28
          5       1.00      1.00      1.00       351
          6       1.00      1.00      1.00        24
          7       1.00      1.00      1.00        34

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06399337 -0.0921074   0.03342805 -0.00718002 -0.13798825  0.04756989
 -0.04323843 -0.01477304]
Epoch number and batch_no:  88 1
Loss before optimizing :  1.24388158368
Loss, accuracy and verification results :  1.24388158368 0.541865214432 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.44      0.44      0.44       305
          1       0.00      0.00      0.00        11
          2       0.45      0.84      0.59       289
          3       0.62      0.12      0.20        82
          5       0.74      0.78      0.76       475
          6       0.39      0.08      0.13       138
          7       0.34      0.16      0.22       169

avg / total       0.53      0.54      0.50      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       300
          2       1.00      1.00      1.00       541
          3       1.00      1.00      1.00        16
          5       1.00      1.00      1.00       504
          6       1.00      1.00      1.00        28
          7       1.00      1.00      1.00        80

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06366228 -0.09162573  0.03279572 -0.00671531 -0.13800077  0.0484358
 -0.04369815 -0.01558807]
Epoch number and batch_no:  89 0
Loss before optimizing :  1.02123586348
Loss, accuracy and verification results :  1.02123586348 0.622509960159 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.53      0.34      0.41       152
          1       0.00      0.00      0.00         4
          2       0.69      0.67      0.68       224
          3       0.62      0.40      0.49        70
          5       0.63      0.99      0.77       381
          6       0.36      0.05      0.09        82
          7       0.39      0.13      0.20        91

avg / total       0.58      0.62      0.57      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        96
          2       1.00      1.00      1.00       219
          3       1.00      1.00      1.00        45
          5       1.00      1.00      1.00       602
          6       1.00      1.00      1.00        11
          7       1.00      1.00      1.00        31

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06360824 -0.09157077  0.03322295 -0.00614956 -0.13801228  0.04814663
 -0.0430602  -0.01655088]
Epoch number and batch_no:  89 1
Loss before optimizing :  1.21753735447
Loss, accuracy and verification results :  1.21753735447 0.550714771954 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.44      0.36      0.40       305
          1       0.00      0.00      0.00        11
          2       0.60      0.64      0.62       289
          3       0.40      0.33      0.36        82
          5       0.58      0.97      0.73       475
          6       0.50      0.12      0.20       138
          7       0.42      0.03      0.06       169

avg / total       0.52      0.55      0.49      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       251
          2       1.00      1.00      1.00       311
          3       1.00      1.00      1.00        68
          5       1.00      1.00      1.00       793
          6       1.00      1.00      1.00        34
          7       1.00      1.00      1.00        12

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06426239 -0.09141008  0.03378158 -0.007025   -0.13802289  0.04718204
 -0.04233591 -0.01602241]
Epoch number and batch_no:  90 0
Loss before optimizing :  1.01464837872
Loss, accuracy and verification results :  1.01464837872 0.631474103586 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.47      0.49       152
          1       0.33      0.25      0.29         4
          2       0.51      0.83      0.63       224
          3       0.66      0.39      0.49        70
          5       0.80      0.84      0.82       381
          6       0.48      0.28      0.35        82
          7       0.43      0.07      0.11        91

avg / total       0.62      0.63      0.60      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       140
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       360
          3       1.00      1.00      1.00        41
          5       1.00      1.00      1.00       398
          6       1.00      1.00      1.00        48
          7       1.00      1.00      1.00        14

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06414578 -0.09156296  0.03377637 -0.00738729 -0.13803262  0.04713941
 -0.04252764 -0.01518794]
Epoch number and batch_no:  90 1
Loss before optimizing :  1.25426783429
Loss, accuracy and verification results :  1.25426783429 0.528931245745 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.45      0.43      0.44       305
          1       0.00      0.00      0.00        11
          2       0.48      0.74      0.58       289
          3       0.86      0.07      0.13        82
          5       0.65      0.80      0.72       475
          6       0.41      0.19      0.26       138
          7       0.24      0.11      0.15       169

avg / total       0.51      0.53      0.49      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       286
          2       1.00      1.00      1.00       450
          3       1.00      1.00      1.00         7
          5       1.00      1.00      1.00       586
          6       1.00      1.00      1.00        64
          7       1.00      1.00      1.00        76

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06434912 -0.09159701  0.03303646 -0.00663682 -0.13804163  0.04754513
 -0.04338773 -0.01477631]
Epoch number and batch_no:  91 0
Loss before optimizing :  0.944233093629
Loss, accuracy and verification results :  0.944233093629 0.671314741036 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.47      0.52       152
          1       0.33      0.50      0.40         4
          2       0.60      0.78      0.68       224
          3       0.75      0.47      0.58        70
          5       0.76      0.96      0.85       381
          6       0.55      0.15      0.23        82
          7       0.39      0.18      0.24        91

avg / total       0.65      0.67      0.64      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       122
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       291
          3       1.00      1.00      1.00        44
          5       1.00      1.00      1.00       478
          6       1.00      1.00      1.00        22
          7       1.00      1.00      1.00        41

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06450887 -0.09203753  0.03238207 -0.00531439 -0.13804995  0.04791728
 -0.04348787 -0.01529622]
Epoch number and batch_no:  91 1
Loss before optimizing :  1.25238943253
Loss, accuracy and verification results :  1.25238943253 0.522123893805 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.43      0.33      0.38       305
          1       0.00      0.00      0.00        11
          2       0.68      0.51      0.58       289
          3       0.42      0.27      0.33        82
          5       0.52      0.99      0.68       475
          6       0.57      0.09      0.16       138
          7       0.35      0.09      0.14       169

avg / total       0.51      0.52      0.46      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       235
          2       1.00      1.00      1.00       216
          3       1.00      1.00      1.00        52
          5       1.00      1.00      1.00       900
          6       1.00      1.00      1.00        23
          7       1.00      1.00      1.00        43

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06557057 -0.09222897  0.03268735 -0.00449308 -0.13805777  0.0467663
 -0.04266968 -0.01529539]
Epoch number and batch_no:  92 0
Loss before optimizing :  0.953022846999
Loss, accuracy and verification results :  0.953022846999 0.666334661355 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.42      0.64      0.51       152
          1       0.00      0.00      0.00         4
          2       0.68      0.71      0.70       224
          3       0.69      0.57      0.62        70
          5       0.80      0.92      0.86       381
          6       0.49      0.21      0.29        82
          7       0.40      0.04      0.08        91

avg / total       0.65      0.67      0.63      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       231
          2       1.00      1.00      1.00       233
          3       1.00      1.00      1.00        58
          5       1.00      1.00      1.00       437
          6       1.00      1.00      1.00        35
          7       1.00      1.00      1.00        10

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06522657 -0.09238506  0.03335862 -0.00472124 -0.138065    0.04633945
 -0.04253146 -0.01480959]
Epoch number and batch_no:  92 1
Loss before optimizing :  1.19133858007
Loss, accuracy and verification results :  1.19133858007 0.565690946222 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.42      0.57      0.49       305
          1       0.00      0.00      0.00        11
          2       0.53      0.79      0.64       289
          3       0.53      0.24      0.33        82
          5       0.75      0.79      0.77       475
          6       0.34      0.20      0.25       138
          7       0.55      0.04      0.07       169

avg / total       0.56      0.57      0.52      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       413
          2       1.00      1.00      1.00       431
          3       1.00      1.00      1.00        38
          5       1.00      1.00      1.00       497
          6       1.00      1.00      1.00        79
          7       1.00      1.00      1.00        11

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06445286 -0.09218484  0.03321079 -0.00519322 -0.13807176  0.04677868
 -0.04339147 -0.01352987]
Epoch number and batch_no:  93 0
Loss before optimizing :  0.909711352501
Loss, accuracy and verification results :  0.909711352501 0.676294820717 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.44      0.50       152
          1       0.50      0.50      0.50         4
          2       0.65      0.78      0.71       224
          3       0.76      0.54      0.63        70
          5       0.76      0.95      0.84       381
          6       0.61      0.17      0.27        82
          7       0.33      0.24      0.28        91

avg / total       0.66      0.68      0.65      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       114
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       271
          3       1.00      1.00      1.00        50
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00        23
          7       1.00      1.00      1.00        67

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06398527 -0.09222293  0.03327992 -0.00516623 -0.13807805  0.04719798
 -0.04359093 -0.01407077]
Epoch number and batch_no:  93 1
Loss before optimizing :  1.17743729511
Loss, accuracy and verification results :  1.17743729511 0.541184479238 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.53      0.22      0.31       305
          1       0.25      0.09      0.13        11
          2       0.54      0.65      0.59       289
          3       0.64      0.26      0.37        82
          5       0.58      0.97      0.73       475
          6       0.54      0.11      0.18       138
          7       0.31      0.27      0.29       169

avg / total       0.53      0.54      0.49      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       124
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       346
          3       1.00      1.00      1.00        33
          5       1.00      1.00      1.00       791
          6       1.00      1.00      1.00        28
          7       1.00      1.00      1.00       143

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06523566 -0.09227253  0.03350174 -0.00503789 -0.13808403  0.04661098
 -0.04275144 -0.01567656]
Epoch number and batch_no:  94 0
Loss before optimizing :  0.909014860761
Loss, accuracy and verification results :  0.909014860761 0.666334661355 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.45      0.48       152
          1       0.29      0.50      0.36         4
          2       0.62      0.81      0.70       224
          3       0.72      0.63      0.67        70
          5       0.78      0.93      0.85       381
          6       0.34      0.15      0.21        82
          7       0.40      0.07      0.11        91

avg / total       0.62      0.67      0.63      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       134
          1       1.00      1.00      1.00         7
          2       1.00      1.00      1.00       294
          3       1.00      1.00      1.00        61
          5       1.00      1.00      1.00       458
          6       1.00      1.00      1.00        35
          7       1.00      1.00      1.00        15

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06592741 -0.09274301  0.03374945 -0.00535491 -0.13808958  0.04622518
 -0.04213996 -0.01636506]
Epoch number and batch_no:  94 1
Loss before optimizing :  1.14776678886
Loss, accuracy and verification results :  1.14776678886 0.586113002042 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.41      0.67      0.50       305
          1       0.00      0.00      0.00        11
          2       0.56      0.74      0.64       289
          3       0.77      0.24      0.37        82
          5       0.81      0.82      0.81       475
          6       0.43      0.23      0.30       138
          7       0.50      0.02      0.03       169

avg / total       0.60      0.59      0.55      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       503
          2       1.00      1.00      1.00       380
          3       1.00      1.00      1.00        26
          5       1.00      1.00      1.00       479
          6       1.00      1.00      1.00        75
          7       1.00      1.00      1.00         6

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06557663 -0.09297937  0.03365499 -0.00604642 -0.13809481  0.04652115
 -0.04290667 -0.01525493]
Epoch number and batch_no:  95 0
Loss before optimizing :  0.867771950549
Loss, accuracy and verification results :  0.867771950549 0.668326693227 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.48      0.49       152
          1       1.00      0.25      0.40         4
          2       0.60      0.82      0.69       224
          3       0.76      0.46      0.57        70
          5       0.77      0.95      0.85       381
          6       0.60      0.15      0.24        82
          7       0.36      0.09      0.14        91

avg / total       0.64      0.67      0.63      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       143
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       305
          3       1.00      1.00      1.00        42
          5       1.00      1.00      1.00       471
          6       1.00      1.00      1.00        20
          7       1.00      1.00      1.00        22

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06494462 -0.0933695   0.03327689 -0.00602604 -0.13809972  0.04669534
 -0.04300372 -0.01361868]
Epoch number and batch_no:  95 1
Loss before optimizing :  1.14621131059
Loss, accuracy and verification results :  1.14621131059 0.560925799864 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.31      0.41       305
          1       0.00      0.00      0.00        11
          2       0.61      0.65      0.63       289
          3       0.58      0.35      0.44        82
          5       0.61      0.94      0.74       475
          6       0.52      0.12      0.19       138
          7       0.26      0.30      0.28       169

avg / total       0.55      0.56      0.52      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       162
          2       1.00      1.00      1.00       305
          3       1.00      1.00      1.00        50
          5       1.00      1.00      1.00       729
          6       1.00      1.00      1.00        31
          7       1.00      1.00      1.00       192

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06577549 -0.09346883  0.03311053 -0.00621336 -0.13810461  0.04624478
 -0.04217545 -0.01380756]
Epoch number and batch_no:  96 0
Loss before optimizing :  0.87510047685
Loss, accuracy and verification results :  0.87510047685 0.68625498008 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.50      0.52       152
          1       0.00      0.00      0.00         4
          2       0.69      0.78      0.73       224
          3       0.78      0.61      0.69        70
          5       0.76      0.95      0.85       381
          6       0.47      0.18      0.26        82
          7       0.37      0.21      0.27        91

avg / total       0.65      0.69      0.66      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       141
          2       1.00      1.00      1.00       253
          3       1.00      1.00      1.00        55
          5       1.00      1.00      1.00       472
          6       1.00      1.00      1.00        32
          7       1.00      1.00      1.00        51

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06631142 -0.09351254  0.03327293 -0.00625641 -0.13810927  0.04587338
 -0.04119064 -0.0147611 ]
Epoch number and batch_no:  96 1
Loss before optimizing :  1.10095589077
Loss, accuracy and verification results :  1.10095589077 0.592920353982 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.44      0.57      0.50       305
          1       0.00      0.00      0.00        11
          2       0.62      0.68      0.65       289
          3       0.60      0.22      0.32        82
          5       0.78      0.87      0.82       475
          6       0.33      0.39      0.36       138
          7       0.46      0.09      0.16       169

avg / total       0.58      0.59      0.57      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       395
          2       1.00      1.00      1.00       314
          3       1.00      1.00      1.00        30
          5       1.00      1.00      1.00       531
          6       1.00      1.00      1.00       164
          7       1.00      1.00      1.00        35

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06636046 -0.09317237  0.03358892 -0.00627545 -0.13811371  0.04601641
 -0.04243856 -0.0150009 ]
Epoch number and batch_no:  97 0
Loss before optimizing :  0.86845576323
Loss, accuracy and verification results :  0.86845576323 0.691235059761 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.49      0.55      0.52       152
          1       0.00      0.00      0.00         4
          2       0.64      0.87      0.74       224
          3       0.71      0.59      0.64        70
          5       0.83      0.92      0.87       381
          6       0.54      0.24      0.34        82
          7       0.38      0.07      0.11        91

avg / total       0.66      0.69      0.66      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       169
          2       1.00      1.00      1.00       301
          3       1.00      1.00      1.00        58
          5       1.00      1.00      1.00       423
          6       1.00      1.00      1.00        37
          7       1.00      1.00      1.00        16

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06562591 -0.09286655  0.03367461 -0.00620866 -0.13811789  0.04644609
 -0.04336309 -0.01462562]
Epoch number and batch_no:  97 1
Loss before optimizing :  1.13483054894
Loss, accuracy and verification results :  1.13483054894 0.567733151804 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.47      0.34      0.39       305
          1       0.00      0.00      0.00        11
          2       0.57      0.72      0.64       289
          3       0.76      0.16      0.26        82
          5       0.61      0.98      0.75       475
          6       0.51      0.15      0.23       138
          7       0.38      0.14      0.20       169

avg / total       0.54      0.57      0.51      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       217
          2       1.00      1.00      1.00       364
          3       1.00      1.00      1.00        17
          5       1.00      1.00      1.00       770
          6       1.00      1.00      1.00        41
          7       1.00      1.00      1.00        60

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06606899 -0.09240803  0.03359985 -0.00532832 -0.13812189  0.0456944
 -0.04316576 -0.01372695]
Epoch number and batch_no:  98 0
Loss before optimizing :  0.821766969956
Loss, accuracy and verification results :  0.821766969956 0.709163346614 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.64      0.45      0.53       152
          1       0.00      0.00      0.00         4
          2       0.64      0.87      0.74       224
          3       0.71      0.64      0.68        70
          5       0.81      0.96      0.88       381
          6       0.54      0.23      0.32        82
          7       0.47      0.24      0.32        91

avg / total       0.68      0.71      0.68      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       106
          2       1.00      1.00      1.00       303
          3       1.00      1.00      1.00        63
          5       1.00      1.00      1.00       450
          6       1.00      1.00      1.00        35
          7       1.00      1.00      1.00        47

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06668352 -0.09220011  0.03330985 -0.00457491 -0.13812565  0.04505549
 -0.0424845  -0.01325443]
Epoch number and batch_no:  98 1
Loss before optimizing :  1.08332247922
Loss, accuracy and verification results :  1.08332247922 0.607896528251 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.47      0.52      0.50       305
          1       0.00      0.00      0.00        11
          2       0.58      0.77      0.66       289
          3       0.65      0.37      0.47        82
          5       0.81      0.84      0.83       475
          6       0.36      0.36      0.36       138
          7       0.44      0.18      0.26       169

avg / total       0.59      0.61      0.59      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       335
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       384
          3       1.00      1.00      1.00        46
          5       1.00      1.00      1.00       497
          6       1.00      1.00      1.00       135
          7       1.00      1.00      1.00        71

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06703703 -0.0919578   0.03284704 -0.0040613  -0.13812919  0.04516347
 -0.04307631 -0.01301056]
Epoch number and batch_no:  99 0
Loss before optimizing :  0.810859290476
Loss, accuracy and verification results :  0.810859290476 0.719123505976 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.47      0.61      0.53       152
          1       1.00      0.25      0.40         4
          2       0.81      0.72      0.76       224
          3       0.71      0.70      0.71        70
          5       0.82      0.96      0.89       381
          6       0.61      0.33      0.43        82
          7       0.50      0.29      0.36        91

avg / total       0.71      0.72      0.71      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       194
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       199
          3       1.00      1.00      1.00        69
          5       1.00      1.00      1.00       445
          6       1.00      1.00      1.00        44
          7       1.00      1.00      1.00        52

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06652243 -0.09185933  0.03337781 -0.00444719 -0.13813248  0.04528286
 -0.04355984 -0.01308031]
Epoch number and batch_no:  99 1
Loss before optimizing :  1.05533794484
Loss, accuracy and verification results :  1.05533794484 0.607215793057 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.50      0.50      0.50       305
          1       0.33      0.09      0.14        11
          2       0.64      0.71      0.68       289
          3       0.61      0.44      0.51        82
          5       0.67      0.95      0.78       475
          6       0.45      0.15      0.23       138
          7       0.40      0.15      0.22       169

avg / total       0.57      0.61      0.57      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       306
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       320
          3       1.00      1.00      1.00        59
          5       1.00      1.00      1.00       669
          6       1.00      1.00      1.00        47
          7       1.00      1.00      1.00        65

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06658664 -0.09181241  0.03387734 -0.00506365 -0.13813557  0.04485836
 -0.04333674 -0.01281631]
Epoch number and batch_no:  100 0
Loss before optimizing :  0.809521537781
Loss, accuracy and verification results :  0.809521537781 0.692231075697 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.43      0.49       152
          1       0.50      0.25      0.33         4
          2       0.59      0.90      0.71       224
          3       0.76      0.63      0.69        70
          5       0.85      0.93      0.89       381
          6       0.44      0.18      0.26        82
          7       0.36      0.15      0.22        91

avg / total       0.67      0.69      0.66      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       111
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       341
          3       1.00      1.00      1.00        58
          5       1.00      1.00      1.00       419
          6       1.00      1.00      1.00        34
          7       1.00      1.00      1.00        39

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06672143 -0.0921938   0.03342853 -0.00513642 -0.13813841  0.04481649
 -0.0425298  -0.01242674]
Epoch number and batch_no:  100 1
Loss before optimizing :  1.020000394
Loss, accuracy and verification results :  1.020000394 0.619469026549 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.52      0.50      0.51       305
          1       0.50      0.09      0.15        11
          2       0.65      0.72      0.68       289
          3       0.49      0.48      0.48        82
          5       0.78      0.89      0.83       475
          6       0.41      0.35      0.38       138
          7       0.33      0.24      0.27       169

avg / total       0.60      0.62      0.60      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       288
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       320
          3       1.00      1.00      1.00        79
          5       1.00      1.00      1.00       540
          6       1.00      1.00      1.00       117
          7       1.00      1.00      1.00       123

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06725544 -0.09252734  0.03327726 -0.00589375 -0.13814106  0.04496565
 -0.0427327  -0.01266723]
Epoch number and batch_no:  101 0
Loss before optimizing :  0.790313846415
Loss, accuracy and verification results :  0.790313846415 0.708167330677 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.47      0.66      0.55       152
          1       1.00      0.25      0.40         4
          2       0.78      0.68      0.73       224
          3       0.75      0.71      0.73        70
          5       0.82      0.97      0.89       381
          6       0.53      0.29      0.38        82
          7       0.44      0.18      0.25        91

avg / total       0.70      0.71      0.69      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       213
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       194
          3       1.00      1.00      1.00        67
          5       1.00      1.00      1.00       448
          6       1.00      1.00      1.00        45
          7       1.00      1.00      1.00        36

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06672724 -0.09277782  0.03411365 -0.00696433 -0.1381435   0.04487994
 -0.04294455 -0.01244561]
Epoch number and batch_no:  101 1
Loss before optimizing :  1.01462414432
Loss, accuracy and verification results :  1.01462414432 0.620830496937 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.55      0.53       305
          1       0.00      0.00      0.00        11
          2       0.56      0.82      0.67       289
          3       0.67      0.29      0.41        82
          5       0.77      0.93      0.84       475
          6       0.46      0.16      0.24       138
          7       0.34      0.13      0.19       169

avg / total       0.58      0.62      0.58      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       329
          2       1.00      1.00      1.00       418
          3       1.00      1.00      1.00        36
          5       1.00      1.00      1.00       574
          6       1.00      1.00      1.00        48
          7       1.00      1.00      1.00        64

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06644947 -0.09266575  0.03422525 -0.00720422 -0.13814584  0.04464909
 -0.04264657 -0.011657  ]
Epoch number and batch_no:  102 0
Loss before optimizing :  0.768478944699
Loss, accuracy and verification results :  0.768478944699 0.72609561753 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.71      0.45      0.55       152
          1       0.50      0.50      0.50         4
          2       0.66      0.85      0.74       224
          3       0.88      0.61      0.72        70
          5       0.85      0.94      0.89       381
          6       0.49      0.44      0.46        82
          7       0.43      0.34      0.38        91

avg / total       0.72      0.73      0.71      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        96
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       290
          3       1.00      1.00      1.00        49
          5       1.00      1.00      1.00       420
          6       1.00      1.00      1.00        73
          7       1.00      1.00      1.00        72

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06658771 -0.09267219  0.03408389 -0.00678848 -0.13814804  0.04465897
 -0.04275162 -0.01176193]
Epoch number and batch_no:  102 1
Loss before optimizing :  0.981935274615
Loss, accuracy and verification results :  0.981935274615 0.645336963921 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.46      0.52       305
          1       0.00      0.00      0.00        11
          2       0.64      0.83      0.72       289
          3       0.73      0.29      0.42        82
          5       0.77      0.94      0.85       475
          6       0.44      0.33      0.38       138
          7       0.38      0.30      0.34       169

avg / total       0.62      0.65      0.62      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       237
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       374
          3       1.00      1.00      1.00        33
          5       1.00      1.00      1.00       585
          6       1.00      1.00      1.00       104
          7       1.00      1.00      1.00       135

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06762867 -0.09248771  0.03374375 -0.00588671 -0.13815013  0.04456435
 -0.04322575 -0.01266477]
Epoch number and batch_no:  103 0
Loss before optimizing :  0.759689952717
Loss, accuracy and verification results :  0.759689952717 0.712151394422 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.59      0.55       152
          1       0.50      0.25      0.33         4
          2       0.74      0.78      0.76       224
          3       0.77      0.79      0.78        70
          5       0.80      0.97      0.88       381
          6       0.52      0.28      0.37        82
          7       0.23      0.03      0.06        91

avg / total       0.66      0.71      0.68      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       177
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       238
          3       1.00      1.00      1.00        71
          5       1.00      1.00      1.00       459
          6       1.00      1.00      1.00        44
          7       1.00      1.00      1.00        13

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06775158 -0.0923135   0.03378616 -0.00558508 -0.13815206  0.04419933
 -0.04310567 -0.01218348]
Epoch number and batch_no:  103 1
Loss before optimizing :  0.946443544177
Loss, accuracy and verification results :  0.946443544177 0.663036078965 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.48      0.68      0.56       305
          1       0.00      0.00      0.00        11
          2       0.70      0.78      0.74       289
          3       0.67      0.51      0.58        82
          5       0.79      0.94      0.86       475
          6       0.64      0.25      0.36       138
          7       0.60      0.11      0.18       169

avg / total       0.66      0.66      0.63      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       437
          2       1.00      1.00      1.00       321
          3       1.00      1.00      1.00        63
          5       1.00      1.00      1.00       565
          6       1.00      1.00      1.00        53
          7       1.00      1.00      1.00        30

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06688298 -0.09199454  0.03399745 -0.00543025 -0.13815386  0.04380135
 -0.04262723 -0.01058724]
Epoch number and batch_no:  104 0
Loss before optimizing :  0.717354821684
Loss, accuracy and verification results :  0.717354821684 0.760956175299 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.67      0.47      0.55       152
          1       0.33      0.25      0.29         4
          2       0.77      0.85      0.81       224
          3       0.76      0.84      0.80        70
          5       0.89      0.93      0.91       381
          6       0.54      0.52      0.53        82
          7       0.49      0.46      0.47        91

avg / total       0.75      0.76      0.75      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       108
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       247
          3       1.00      1.00      1.00        78
          5       1.00      1.00      1.00       402
          6       1.00      1.00      1.00        80
          7       1.00      1.00      1.00        86

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06643108 -0.09204787  0.03450559 -0.00617607 -0.13815553  0.04387345
 -0.04264885 -0.01069438]
Epoch number and batch_no:  104 1
Loss before optimizing :  0.937915390751
Loss, accuracy and verification results :  0.937915390751 0.645336963921 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.66      0.38      0.48       305
          1       0.00      0.00      0.00        11
          2       0.62      0.86      0.72       289
          3       0.68      0.41      0.52        82
          5       0.76      0.94      0.84       475
          6       0.46      0.38      0.42       138
          7       0.36      0.31      0.33       169

avg / total       0.63      0.65      0.62      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       174
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       404
          3       1.00      1.00      1.00        50
          5       1.00      1.00      1.00       585
          6       1.00      1.00      1.00       112
          7       1.00      1.00      1.00       143

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0673536  -0.09205802  0.03440966 -0.00636971 -0.13815713  0.04381748
 -0.04296517 -0.01145705]
Epoch number and batch_no:  105 0
Loss before optimizing :  0.713374693962
Loss, accuracy and verification results :  0.713374693962 0.733067729084 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.57      0.58      0.57       152
          1       0.00      0.00      0.00         4
          2       0.68      0.88      0.77       224
          3       0.82      0.66      0.73        70
          5       0.85      0.96      0.90       381
          6       0.60      0.29      0.39        82
          7       0.50      0.16      0.25        91

avg / total       0.71      0.73      0.70      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       155
          2       1.00      1.00      1.00       291
          3       1.00      1.00      1.00        56
          5       1.00      1.00      1.00       432
          6       1.00      1.00      1.00        40
          7       1.00      1.00      1.00        30

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06759501 -0.09217776  0.03396423 -0.00568909 -0.13815862  0.043663
 -0.04240668 -0.01126219]
Epoch number and batch_no:  105 1
Loss before optimizing :  0.91126823204
Loss, accuracy and verification results :  0.91126823204 0.662355343771 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.50      0.60      0.55       305
          1       0.00      0.00      0.00        11
          2       0.71      0.73      0.72       289
          3       0.50      0.77      0.61        82
          5       0.82      0.94      0.88       475
          6       0.52      0.32      0.39       138
          7       0.47      0.13      0.20       169

avg / total       0.64      0.66      0.63      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       367
          2       1.00      1.00      1.00       298
          3       1.00      1.00      1.00       125
          5       1.00      1.00      1.00       547
          6       1.00      1.00      1.00        85
          7       1.00      1.00      1.00        47

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0675073  -0.09199999  0.03401262 -0.00763829 -0.13816005  0.04359617
 -0.04184036 -0.01035678]
Epoch number and batch_no:  106 0
Loss before optimizing :  0.691431594489
Loss, accuracy and verification results :  0.691431594489 0.746015936255 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.51      0.55       152
          1       1.00      0.25      0.40         4
          2       0.79      0.82      0.80       224
          3       0.87      0.69      0.77        70
          5       0.86      0.96      0.91       381
          6       0.47      0.57      0.52        82
          7       0.46      0.29      0.35        91

avg / total       0.74      0.75      0.74      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       134
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       231
          3       1.00      1.00      1.00        55
          5       1.00      1.00      1.00       427
          6       1.00      1.00      1.00       100
          7       1.00      1.00      1.00        56

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06730914 -0.09202252  0.03463574 -0.0084021  -0.1381614   0.04351334
 -0.04251529 -0.01006186]
Epoch number and batch_no:  106 1
Loss before optimizing :  0.891108362106
Loss, accuracy and verification results :  0.891108362106 0.669162695711 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.56      0.52      0.54       305
          1       0.00      0.00      0.00        11
          2       0.64      0.88      0.74       289
          3       0.83      0.30      0.45        82
          5       0.83      0.93      0.88       475
          6       0.53      0.40      0.46       138
          7       0.40      0.29      0.34       169

avg / total       0.65      0.67      0.65      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       284
          2       1.00      1.00      1.00       396
          3       1.00      1.00      1.00        30
          5       1.00      1.00      1.00       535
          6       1.00      1.00      1.00       103
          7       1.00      1.00      1.00       121

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06742178 -0.09177899  0.03464704 -0.00776812 -0.1381627   0.04357214
 -0.04339664 -0.01015594]
Epoch number and batch_no:  107 0
Loss before optimizing :  0.67264403479
Loss, accuracy and verification results :  0.67264403479 0.75796812749 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.68      0.52      0.59       152
          1       0.33      0.25      0.29         4
          2       0.68      0.94      0.79       224
          3       0.74      0.77      0.76        70
          5       0.87      0.97      0.92       381
          6       0.58      0.23      0.33        82
          7       0.58      0.31      0.40        91

avg / total       0.74      0.76      0.73      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       116
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       309
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       422
          6       1.00      1.00      1.00        33
          7       1.00      1.00      1.00        48

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06773139 -0.09164163  0.03411743 -0.00750625 -0.13816392  0.0435498
 -0.04309941 -0.0099013 ]
Epoch number and batch_no:  107 1
Loss before optimizing :  0.894259135138
Loss, accuracy and verification results :  0.894259135138 0.674608577263 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.63      0.56       305
          1       0.50      0.09      0.15        11
          2       0.73      0.74      0.74       289
          3       0.65      0.60      0.62        82
          5       0.80      0.96      0.87       475
          6       0.55      0.32      0.40       138
          7       0.46      0.22      0.30       169

avg / total       0.66      0.67      0.65      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       373
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       292
          3       1.00      1.00      1.00        75
          5       1.00      1.00      1.00       565
          6       1.00      1.00      1.00        80
          7       1.00      1.00      1.00        82

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06786574 -0.0915755   0.03398172 -0.00792974 -0.13816508  0.0433599
 -0.04264262 -0.00940021]
Epoch number and batch_no:  108 0
Loss before optimizing :  0.669317897463
Loss, accuracy and verification results :  0.669317897463 0.780876494024 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.67      0.54      0.60       152
          1       0.50      0.25      0.33         4
          2       0.77      0.89      0.83       224
          3       0.83      0.79      0.81        70
          5       0.91      0.94      0.93       381
          6       0.55      0.61      0.58        82
          7       0.55      0.41      0.47        91

avg / total       0.77      0.78      0.77      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       123
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       260
          3       1.00      1.00      1.00        66
          5       1.00      1.00      1.00       395
          6       1.00      1.00      1.00        91
          7       1.00      1.00      1.00        67

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06788341 -0.09177682  0.03404859 -0.00816896 -0.13816616  0.04350671
 -0.04308186 -0.00940596]
Epoch number and batch_no:  108 1
Loss before optimizing :  0.837483535597
Loss, accuracy and verification results :  0.837483535597 0.68413886998 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.58      0.57       305
          1       0.00      0.00      0.00        11
          2       0.74      0.78      0.76       289
          3       0.67      0.55      0.60        82
          5       0.81      0.95      0.87       475
          6       0.52      0.41      0.46       138
          7       0.45      0.28      0.35       169

avg / total       0.66      0.68      0.67      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       325
          2       1.00      1.00      1.00       305
          3       1.00      1.00      1.00        67
          5       1.00      1.00      1.00       556
          6       1.00      1.00      1.00       109
          7       1.00      1.00      1.00       107

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06803403 -0.09155927  0.03412765 -0.00809252 -0.13816716  0.04361624
 -0.04384473 -0.00961464]
Epoch number and batch_no:  109 0
Loss before optimizing :  0.647286656519
Loss, accuracy and verification results :  0.647286656519 0.749003984064 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.53      0.56       152
          1       0.50      0.25      0.33         4
          2       0.76      0.86      0.81       224
          3       0.77      0.81      0.79        70
          5       0.84      0.97      0.90       381
          6       0.57      0.29      0.39        82
          7       0.50      0.31      0.38        91

avg / total       0.72      0.75      0.73      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       136
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       252
          3       1.00      1.00      1.00        74
          5       1.00      1.00      1.00       442
          6       1.00      1.00      1.00        42
          7       1.00      1.00      1.00        56

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06816485 -0.09144069  0.03428396 -0.00786179 -0.13816809  0.04335699
 -0.04385155 -0.00956888]
Epoch number and batch_no:  109 1
Loss before optimizing :  0.826713806482
Loss, accuracy and verification results :  0.826713806482 0.679373723622 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.59      0.56       305
          1       0.00      0.00      0.00        11
          2       0.70      0.85      0.77       289
          3       0.61      0.66      0.63        82
          5       0.79      0.95      0.86       475
          6       0.59      0.28      0.38       138
          7       0.49      0.20      0.28       169

avg / total       0.65      0.68      0.65      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       329
          2       1.00      1.00      1.00       349
          3       1.00      1.00      1.00        89
          5       1.00      1.00      1.00       570
          6       1.00      1.00      1.00        64
          7       1.00      1.00      1.00        68

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06843461 -0.09088401  0.03423868 -0.00789931 -0.13816895  0.04277497
 -0.0432797  -0.00882266]
Epoch number and batch_no:  110 0
Loss before optimizing :  0.629119418416
Loss, accuracy and verification results :  0.629119418416 0.783864541833 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.59      0.61      0.60       152
          1       0.50      0.25      0.33         4
          2       0.80      0.88      0.84       224
          3       0.89      0.84      0.87        70
          5       0.95      0.92      0.94       381
          6       0.61      0.51      0.56        82
          7       0.46      0.47      0.47        91

avg / total       0.79      0.78      0.78      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       158
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       248
          3       1.00      1.00      1.00        66
          5       1.00      1.00      1.00       368
          6       1.00      1.00      1.00        69
          7       1.00      1.00      1.00        93

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06823527 -0.09058865  0.03426382 -0.00795947 -0.13816974  0.0429336
 -0.04317244 -0.0092093 ]
Epoch number and batch_no:  110 1
Loss before optimizing :  0.826272678445
Loss, accuracy and verification results :  0.826272678445 0.671885636487 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.58      0.56       305
          1       0.43      0.27      0.33        11
          2       0.71      0.81      0.75       289
          3       0.70      0.55      0.62        82
          5       0.84      0.90      0.87       475
          6       0.49      0.42      0.45       138
          7       0.38      0.25      0.30       169

avg / total       0.65      0.67      0.66      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       325
          1       1.00      1.00      1.00         7
          2       1.00      1.00      1.00       331
          3       1.00      1.00      1.00        64
          5       1.00      1.00      1.00       510
          6       1.00      1.00      1.00       118
          7       1.00      1.00      1.00       114

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06803487 -0.09060266  0.03417118 -0.00757954 -0.13817048  0.04339173
 -0.04391306 -0.00964503]
Epoch number and batch_no:  111 0
Loss before optimizing :  0.624155473625
Loss, accuracy and verification results :  0.624155473625 0.78187250996 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.70      0.56      0.62       152
          1       0.60      0.75      0.67         4
          2       0.84      0.83      0.83       224
          3       0.69      0.97      0.81        70
          5       0.85      0.99      0.91       381
          6       0.58      0.39      0.47        82
          7       0.59      0.36      0.45        91

avg / total       0.77      0.78      0.76      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       122
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       222
          3       1.00      1.00      1.00        98
          5       1.00      1.00      1.00       446
          6       1.00      1.00      1.00        55
          7       1.00      1.00      1.00        56

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06805962 -0.09113171  0.03464493 -0.00860993 -0.13817116  0.04340936
 -0.04404608 -0.00982731]
Epoch number and batch_no:  111 1
Loss before optimizing :  0.844991065334
Loss, accuracy and verification results :  0.844991065334 0.692307692308 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.64      0.52      0.58       305
          1       0.25      0.09      0.13        11
          2       0.71      0.85      0.77       289
          3       0.57      0.78      0.66        82
          5       0.78      0.97      0.87       475
          6       0.65      0.20      0.31       138
          7       0.47      0.34      0.39       169

avg / total       0.67      0.69      0.66      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       250
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       345
          3       1.00      1.00      1.00       112
          5       1.00      1.00      1.00       593
          6       1.00      1.00      1.00        43
          7       1.00      1.00      1.00       122

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06880454 -0.09179282  0.03520894 -0.01085909 -0.13817178  0.04292936
 -0.04311099 -0.01002626]
Epoch number and batch_no:  112 0
Loss before optimizing :  0.699795188207
Loss, accuracy and verification results :  0.699795188207 0.738047808765 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.57      0.63      0.60       152
          1       0.50      0.25      0.33         4
          2       0.69      0.94      0.80       224
          3       0.93      0.39      0.55        70
          5       0.91      0.92      0.92       381
          6       0.52      0.41      0.46        82
          7       0.44      0.23      0.30        91

avg / total       0.73      0.74      0.72      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       169
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       305
          3       1.00      1.00      1.00        29
          5       1.00      1.00      1.00       386
          6       1.00      1.00      1.00        65
          7       1.00      1.00      1.00        48

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06853619 -0.09260218  0.03481871 -0.01005614 -0.13817236  0.04297695
 -0.042489   -0.00962549]
Epoch number and batch_no:  112 1
Loss before optimizing :  0.936167312095
Loss, accuracy and verification results :  0.936167312095 0.626276378489 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.48      0.59      0.53       305
          1       0.00      0.00      0.00        11
          2       0.70      0.76      0.73       289
          3       0.74      0.34      0.47        82
          5       0.83      0.85      0.84       475
          6       0.35      0.47      0.40       138
          7       0.35      0.15      0.21       169

avg / total       0.62      0.63      0.61      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       376
          2       1.00      1.00      1.00       311
          3       1.00      1.00      1.00        38
          5       1.00      1.00      1.00       485
          6       1.00      1.00      1.00       188
          7       1.00      1.00      1.00        71

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06765989 -0.09268524  0.03441584 -0.00808956 -0.13817291  0.04358872
 -0.04433764 -0.00867284]
Epoch number and batch_no:  113 0
Loss before optimizing :  0.69713963056
Loss, accuracy and verification results :  0.69713963056 0.743027888446 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.69      0.47      0.56       152
          1       1.00      0.25      0.40         4
          2       0.81      0.83      0.82       224
          3       0.63      0.73      0.68        70
          5       0.80      0.98      0.88       381
          6       0.68      0.26      0.37        82
          7       0.45      0.48      0.47        91

avg / total       0.74      0.74      0.72      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       103
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       227
          3       1.00      1.00      1.00        81
          5       1.00      1.00      1.00       464
          6       1.00      1.00      1.00        31
          7       1.00      1.00      1.00        97

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06744749 -0.09255262  0.03484941 -0.00696858 -0.13817341  0.04344632
 -0.04498115 -0.0090377 ]
Epoch number and batch_no:  113 1
Loss before optimizing :  0.946105038885
Loss, accuracy and verification results :  0.946105038885 0.636487406399 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.62      0.35      0.44       305
          1       0.00      0.00      0.00        11
          2       0.65      0.78      0.71       289
          3       0.45      0.78      0.57        82
          5       0.71      0.97      0.82       475
          6       0.57      0.12      0.19       138
          7       0.47      0.37      0.41       169

avg / total       0.62      0.64      0.59      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       172
          2       1.00      1.00      1.00       344
          3       1.00      1.00      1.00       143
          5       1.00      1.00      1.00       650
          6       1.00      1.00      1.00        28
          7       1.00      1.00      1.00       132

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06912463 -0.09192893  0.03499727 -0.00846464 -0.13817389  0.04244742
 -0.04379275 -0.00947611]
Epoch number and batch_no:  114 0
Loss before optimizing :  0.820026507778
Loss, accuracy and verification results :  0.820026507778 0.675298804781 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.42      0.55      0.48       152
          1       0.67      0.50      0.57         4
          2       0.61      0.89      0.72       224
          3       0.79      0.59      0.67        70
          5       0.91      0.83      0.87       381
          6       0.56      0.22      0.32        82
          7       0.43      0.16      0.24        91

avg / total       0.68      0.68      0.66      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       201
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       330
          3       1.00      1.00      1.00        52
          5       1.00      1.00      1.00       351
          6       1.00      1.00      1.00        32
          7       1.00      1.00      1.00        35

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06933513 -0.09166769  0.03400255 -0.00882828 -0.13817433  0.04248583
 -0.04212767 -0.00912448]
Epoch number and batch_no:  114 1
Loss before optimizing :  0.962807505472
Loss, accuracy and verification results :  0.962807505472 0.642614023145 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.55      0.55       305
          1       0.00      0.00      0.00        11
          2       0.66      0.76      0.71       289
          3       0.93      0.32      0.47        82
          5       0.82      0.87      0.85       475
          6       0.38      0.51      0.43       138
          7       0.42      0.28      0.34       169

avg / total       0.64      0.64      0.63      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       310
          2       1.00      1.00      1.00       333
          3       1.00      1.00      1.00        28
          5       1.00      1.00      1.00       500
          6       1.00      1.00      1.00       184
          7       1.00      1.00      1.00       114

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06956285 -0.091396    0.03315194 -0.00792084 -0.13817476  0.04302997
 -0.04322282 -0.0089523 ]
Epoch number and batch_no:  115 0
Loss before optimizing :  0.751251321711
Loss, accuracy and verification results :  0.751251321711 0.707171314741 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.48      0.57      0.52       152
          1       0.67      0.50      0.57         4
          2       0.82      0.63      0.71       224
          3       0.68      0.93      0.79        70
          5       0.82      0.95      0.88       381
          6       0.55      0.26      0.35        82
          7       0.44      0.34      0.39        91

avg / total       0.70      0.71      0.69      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       182
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       172
          3       1.00      1.00      1.00        95
          5       1.00      1.00      1.00       444
          6       1.00      1.00      1.00        38
          7       1.00      1.00      1.00        70

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0691745  -0.0913513   0.03376734 -0.00857454 -0.13817519  0.04315072
 -0.04349602 -0.0093787 ]
Epoch number and batch_no:  115 1
Loss before optimizing :  1.02288651987
Loss, accuracy and verification results :  1.02288651987 0.609257998639 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.49      0.52      0.50       305
          1       0.12      0.09      0.11        11
          2       0.64      0.70      0.67       289
          3       0.62      0.45      0.52        82
          5       0.70      0.94      0.80       475
          6       0.74      0.14      0.24       138
          7       0.33      0.17      0.22       169

avg / total       0.59      0.61      0.57      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       329
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       317
          3       1.00      1.00      1.00        60
          5       1.00      1.00      1.00       639
          6       1.00      1.00      1.00        27
          7       1.00      1.00      1.00        89

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0687995  -0.09139067  0.03437942 -0.0090137  -0.13817563  0.04258379
 -0.04203706 -0.009451  ]
Epoch number and batch_no:  116 0
Loss before optimizing :  0.735168454778
Loss, accuracy and verification results :  0.735168454778 0.723107569721 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.62      0.40      0.49       152
          1       0.50      0.25      0.33         4
          2       0.62      0.90      0.73       224
          3       0.75      0.70      0.73        70
          5       0.90      0.92      0.91       381
          6       0.54      0.45      0.49        82
          7       0.47      0.27      0.35        91

avg / total       0.71      0.72      0.71      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        98
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       327
          3       1.00      1.00      1.00        65
          5       1.00      1.00      1.00       390
          6       1.00      1.00      1.00        69
          7       1.00      1.00      1.00        53

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06883954 -0.09180425  0.0339707  -0.00913368 -0.13817606  0.04248296
 -0.04089929 -0.00912971]
Epoch number and batch_no:  116 1
Loss before optimizing :  0.9942187644
Loss, accuracy and verification results :  0.9942187644 0.611300204221 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.56      0.38      0.45       305
          1       0.17      0.09      0.12        11
          2       0.65      0.69      0.67       289
          3       0.56      0.67      0.61        82
          5       0.83      0.86      0.84       475
          6       0.32      0.51      0.39       138
          7       0.36      0.28      0.32       169

avg / total       0.61      0.61      0.61      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       207
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       306
          3       1.00      1.00      1.00        99
          5       1.00      1.00      1.00       495
          6       1.00      1.00      1.00       224
          7       1.00      1.00      1.00       132

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06988951 -0.09224327  0.03392312 -0.01095311 -0.13817646  0.04286314
 -0.04261024 -0.0089318 ]
Epoch number and batch_no:  117 0
Loss before optimizing :  0.743394952571
Loss, accuracy and verification results :  0.743394952571 0.715139442231 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.47      0.69      0.56       152
          1       0.00      0.00      0.00         4
          2       0.80      0.77      0.78       224
          3       0.87      0.47      0.61        70
          5       0.87      0.92      0.89       381
          6       0.56      0.34      0.42        82
          7       0.40      0.32      0.36        91

avg / total       0.72      0.72      0.71      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       225
          2       1.00      1.00      1.00       217
          3       1.00      1.00      1.00        38
          5       1.00      1.00      1.00       402
          6       1.00      1.00      1.00        50
          7       1.00      1.00      1.00        72

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06931986 -0.09267302  0.0345244  -0.01103122 -0.13817684  0.04329567
 -0.0440601  -0.00903452]
Epoch number and batch_no:  117 1
Loss before optimizing :  0.984018313554
Loss, accuracy and verification results :  0.984018313554 0.628318584071 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.50      0.58      0.53       305
          1       0.00      0.00      0.00        11
          2       0.63      0.76      0.69       289
          3       0.69      0.41      0.52        82
          5       0.71      0.95      0.81       475
          6       0.78      0.05      0.10       138
          7       0.49      0.21      0.29       169

avg / total       0.62      0.63      0.58      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       355
          2       1.00      1.00      1.00       348
          3       1.00      1.00      1.00        49
          5       1.00      1.00      1.00       636
          6       1.00      1.00      1.00         9
          7       1.00      1.00      1.00        72

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06838565 -0.09254489  0.03473101 -0.01008481 -0.13817719  0.04280809
 -0.04297416 -0.0081456 ]
Epoch number and batch_no:  118 0
Loss before optimizing :  0.7847950325
Loss, accuracy and verification results :  0.7847950325 0.715139442231 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.66      0.43      0.52       152
          1       1.00      0.25      0.40         4
          2       0.67      0.90      0.77       224
          3       0.64      0.87      0.73        70
          5       0.85      0.85      0.85       381
          6       0.59      0.35      0.44        82
          7       0.49      0.40      0.44        91

avg / total       0.71      0.72      0.70      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       100
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       303
          3       1.00      1.00      1.00        96
          5       1.00      1.00      1.00       382
          6       1.00      1.00      1.00        49
          7       1.00      1.00      1.00        73

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06791767 -0.09229078  0.03426585 -0.01014383 -0.13817753  0.04282999
 -0.04147743 -0.00788909]
Epoch number and batch_no:  118 1
Loss before optimizing :  1.05611857235
Loss, accuracy and verification results :  1.05611857235 0.586793737236 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.66      0.27      0.38       305
          1       0.00      0.00      0.00        11
          2       0.64      0.64      0.64       289
          3       0.39      0.76      0.52        82
          5       0.79      0.84      0.81       475
          6       0.34      0.49      0.40       138
          7       0.35      0.40      0.37       169

avg / total       0.61      0.59      0.58      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       122
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       290
          3       1.00      1.00      1.00       157
          5       1.00      1.00      1.00       508
          6       1.00      1.00      1.00       195
          7       1.00      1.00      1.00       196

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06946356 -0.09173834  0.03419952 -0.01359866 -0.13817787  0.04325275
 -0.04217867 -0.00881876]
Epoch number and batch_no:  119 0
Loss before optimizing :  0.922462117102
Loss, accuracy and verification results :  0.922462117102 0.645418326693 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.50      0.57      0.54       152
          1       1.00      0.25      0.40         4
          2       0.83      0.49      0.62       224
          3       0.87      0.39      0.53        70
          5       0.71      0.96      0.82       381
          6       0.38      0.48      0.42        82
          7       0.38      0.20      0.26        91

avg / total       0.66      0.65      0.63      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       173
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       133
          3       1.00      1.00      1.00        31
          5       1.00      1.00      1.00       516
          6       1.00      1.00      1.00       102
          7       1.00      1.00      1.00        48

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07028276 -0.09121802  0.03595589 -0.01459962 -0.13817823  0.04241812
 -0.04386246 -0.0094521 ]
Epoch number and batch_no:  119 1
Loss before optimizing :  1.05488352783
Loss, accuracy and verification results :  1.05488352783 0.596324029952 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.46      0.67      0.54       305
          1       0.00      0.00      0.00        11
          2       0.50      0.92      0.64       289
          3       0.89      0.10      0.18        82
          5       0.85      0.80      0.83       475
          6       0.61      0.10      0.17       138
          7       0.42      0.03      0.06       169

avg / total       0.62      0.60      0.54      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       445
          2       1.00      1.00      1.00       533
          3       1.00      1.00      1.00         9
          5       1.00      1.00      1.00       447
          6       1.00      1.00      1.00        23
          7       1.00      1.00      1.00        12

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06958077 -0.09049405  0.03514508 -0.01303335 -0.13817859  0.04235481
 -0.04388082 -0.00756294]
Epoch number and batch_no:  120 0
Loss before optimizing :  0.748314385192
Loss, accuracy and verification results :  0.748314385192 0.732071713147 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.52      0.53       152
          1       1.00      0.50      0.67         4
          2       0.78      0.90      0.83       224
          3       0.66      0.80      0.72        70
          5       0.87      0.89      0.88       381
          6       0.52      0.29      0.38        82
          7       0.45      0.38      0.42        91

avg / total       0.72      0.73      0.72      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       147
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       258
          3       1.00      1.00      1.00        85
          5       1.00      1.00      1.00       389
          6       1.00      1.00      1.00        46
          7       1.00      1.00      1.00        77

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06857251 -0.09006515  0.03430769 -0.0118933  -0.13817897  0.04270257
 -0.0433771  -0.00646051]
Epoch number and batch_no:  120 1
Loss before optimizing :  1.08727506125
Loss, accuracy and verification results :  1.08727506125 0.57726344452 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.60      0.31      0.41       305
          1       0.40      0.18      0.25        11
          2       0.75      0.56      0.64       289
          3       0.46      0.60      0.52        82
          5       0.69      0.89      0.78       475
          6       0.36      0.26      0.30       138
          7       0.31      0.50      0.38       169

avg / total       0.59      0.58      0.56      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       156
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       216
          3       1.00      1.00      1.00       106
          5       1.00      1.00      1.00       614
          6       1.00      1.00      1.00       101
          7       1.00      1.00      1.00       271

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0691817  -0.09017764  0.03477072 -0.01215595 -0.13817936  0.0425361
 -0.04314511 -0.00803337]
Epoch number and batch_no:  121 0
Loss before optimizing :  0.763893014437
Loss, accuracy and verification results :  0.763893014437 0.727091633466 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.50      0.53       152
          1       0.50      0.25      0.33         4
          2       0.77      0.85      0.81       224
          3       0.66      0.77      0.71        70
          5       0.86      0.91      0.89       381
          6       0.48      0.46      0.47        82
          7       0.43      0.26      0.33        91

avg / total       0.71      0.73      0.72      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       137
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       248
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       400
          6       1.00      1.00      1.00        79
          7       1.00      1.00      1.00        56

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06946052 -0.0907743   0.03548857 -0.01299007 -0.13817975  0.04286222
 -0.04414345 -0.00933513]
Epoch number and batch_no:  121 1
Loss before optimizing :  1.06127793265
Loss, accuracy and verification results :  1.06127793265 0.607896528251 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.46      0.57      0.51       305
          1       0.00      0.00      0.00        11
          2       0.53      0.88      0.66       289
          3       0.60      0.29      0.39        82
          5       0.79      0.84      0.82       475
          6       0.58      0.20      0.30       138
          7       0.75      0.07      0.13       169

avg / total       0.63      0.61      0.56      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       376
          2       1.00      1.00      1.00       482
          3       1.00      1.00      1.00        40
          5       1.00      1.00      1.00       507
          6       1.00      1.00      1.00        48
          7       1.00      1.00      1.00        16

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06913038 -0.09107562  0.03436398 -0.01259613 -0.13818013  0.04349207
 -0.04447354 -0.00799491]
Epoch number and batch_no:  122 0
Loss before optimizing :  0.873108513274
Loss, accuracy and verification results :  0.873108513274 0.676294820717 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.49      0.51      0.50       152
          1       0.00      0.00      0.00         4
          2       0.73      0.75      0.74       224
          3       0.79      0.37      0.50        70
          5       0.72      0.96      0.82       381
          6       0.74      0.17      0.28        82
          7       0.52      0.31      0.39        91

avg / total       0.67      0.68      0.65      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       159
          2       1.00      1.00      1.00       231
          3       1.00      1.00      1.00        33
          5       1.00      1.00      1.00       508
          6       1.00      1.00      1.00        19
          7       1.00      1.00      1.00        54

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06873424 -0.09133639  0.0338719  -0.01043771 -0.13818053  0.04296801
 -0.04333763 -0.00716741]
Epoch number and batch_no:  122 1
Loss before optimizing :  1.04324601073
Loss, accuracy and verification results :  1.04324601073 0.586793737236 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.47      0.47      0.47       305
          1       0.00      0.00      0.00        11
          2       0.73      0.52      0.61       289
          3       0.44      0.56      0.49        82
          5       0.71      0.89      0.79       475
          6       0.45      0.25      0.32       138
          7       0.35      0.38      0.37       169

avg / total       0.58      0.59      0.57      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       300
          2       1.00      1.00      1.00       208
          3       1.00      1.00      1.00       105
          5       1.00      1.00      1.00       599
          6       1.00      1.00      1.00        76
          7       1.00      1.00      1.00       181

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.068714   -0.09144485  0.0346522  -0.01008357 -0.13818094  0.04210785
 -0.04212632 -0.00768587]
Epoch number and batch_no:  123 0
Loss before optimizing :  0.896917363084
Loss, accuracy and verification results :  0.896917363084 0.688247011952 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.41      0.48       152
          1       0.50      0.50      0.50         4
          2       0.74      0.83      0.78       224
          3       0.55      0.79      0.65        70
          5       0.88      0.80      0.83       381
          6       0.42      0.61      0.50        82
          7       0.42      0.35      0.38        91

avg / total       0.70      0.69      0.69      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       108
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       252
          3       1.00      1.00      1.00       100
          5       1.00      1.00      1.00       345
          6       1.00      1.00      1.00       119
          7       1.00      1.00      1.00        76

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06845085 -0.09187899  0.03551689 -0.01185627 -0.13818135  0.04278949
 -0.04334563 -0.00863177]
Epoch number and batch_no:  123 1
Loss before optimizing :  0.95834929606
Loss, accuracy and verification results :  0.95834929606 0.631041524847 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.49      0.52       305
          1       0.00      0.00      0.00        11
          2       0.57      0.86      0.68       289
          3       0.70      0.39      0.50        82
          5       0.78      0.90      0.84       475
          6       0.38      0.25      0.30       138
          7       0.48      0.21      0.29       169

avg / total       0.61      0.63      0.60      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       275
          2       1.00      1.00      1.00       437
          3       1.00      1.00      1.00        46
          5       1.00      1.00      1.00       549
          6       1.00      1.00      1.00        89
          7       1.00      1.00      1.00        73

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06862294 -0.0919932   0.03539072 -0.0123692  -0.13818174  0.04340138
 -0.04475865 -0.00868883]
Epoch number and batch_no:  124 0
Loss before optimizing :  0.826621270105
Loss, accuracy and verification results :  0.826621270105 0.687250996016 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.63      0.45      0.52       152
          1       0.40      0.50      0.44         4
          2       0.69      0.88      0.78       224
          3       0.93      0.20      0.33        70
          5       0.72      0.99      0.83       381
          6       0.61      0.23      0.34        82
          7       0.36      0.13      0.19        91

avg / total       0.67      0.69      0.64      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       108
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       284
          3       1.00      1.00      1.00        15
          5       1.00      1.00      1.00       528
          6       1.00      1.00      1.00        31
          7       1.00      1.00      1.00        33

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06939517 -0.09235421  0.03479479 -0.00994024 -0.13818212  0.0427284
 -0.04485452 -0.00813883]
Epoch number and batch_no:  124 1
Loss before optimizing :  0.935626071488
Loss, accuracy and verification results :  0.935626071488 0.622872702519 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.45      0.62      0.52       305
          1       0.00      0.00      0.00        11
          2       0.70      0.66      0.68       289
          3       0.53      0.54      0.53        82
          5       0.81      0.90      0.85       475
          6       0.56      0.13      0.21       138
          7       0.34      0.27      0.30       169

avg / total       0.61      0.62      0.60      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       423
          2       1.00      1.00      1.00       273
          3       1.00      1.00      1.00        83
          5       1.00      1.00      1.00       527
          6       1.00      1.00      1.00        32
          7       1.00      1.00      1.00       131

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06916877 -0.0922666   0.03455468 -0.0081096  -0.13818248  0.04226287
 -0.04364386 -0.00835939]
Epoch number and batch_no:  125 0
Loss before optimizing :  0.751167285849
Loss, accuracy and verification results :  0.751167285849 0.735059760956 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.60      0.54      0.57       152
          1       0.50      0.50      0.50         4
          2       0.85      0.81      0.83       224
          3       0.56      0.86      0.67        70
          5       0.91      0.88      0.90       381
          6       0.44      0.46      0.45        82
          7       0.45      0.41      0.43        91

avg / total       0.74      0.74      0.74      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       136
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       215
          3       1.00      1.00      1.00       108
          5       1.00      1.00      1.00       372
          6       1.00      1.00      1.00        86
          7       1.00      1.00      1.00        83

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06872214 -0.09239494  0.03469537 -0.00879822 -0.13818284  0.04279383
 -0.04321668 -0.00939793]
Epoch number and batch_no:  125 1
Loss before optimizing :  0.978799092525
Loss, accuracy and verification results :  0.978799092525 0.624234172907 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.44      0.48       305
          1       1.00      0.09      0.17        11
          2       0.74      0.65      0.69       289
          3       0.46      0.66      0.54        82
          5       0.69      0.96      0.81       475
          6       0.39      0.47      0.43       138
          7       0.67      0.11      0.18       169

avg / total       0.63      0.62      0.59      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       245
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       253
          3       1.00      1.00      1.00       117
          5       1.00      1.00      1.00       659
          6       1.00      1.00      1.00       167
          7       1.00      1.00      1.00        27

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06912825 -0.09213207  0.03570183 -0.01078733 -0.13818319  0.04233417
 -0.04459498 -0.00823911]
Epoch number and batch_no:  126 0
Loss before optimizing :  0.737154022937
Loss, accuracy and verification results :  0.737154022937 0.727091633466 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.57      0.46      0.51       152
          1       0.60      0.75      0.67         4
          2       0.65      0.93      0.77       224
          3       0.83      0.57      0.68        70
          5       0.85      0.95      0.90       381
          6       0.55      0.29      0.38        82
          7       0.59      0.24      0.34        91

avg / total       0.71      0.73      0.70      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       122
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       319
          3       1.00      1.00      1.00        48
          5       1.00      1.00      1.00       429
          6       1.00      1.00      1.00        44
          7       1.00      1.00      1.00        37

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06933621 -0.09235644  0.03584995 -0.01127744 -0.13818353  0.04201548
 -0.04557199 -0.00657289]
Epoch number and batch_no:  126 1
Loss before optimizing :  0.949499983025
Loss, accuracy and verification results :  0.949499983025 0.643975493533 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.52      0.50      0.51       305
          1       0.00      0.00      0.00        11
          2       0.64      0.84      0.72       289
          3       0.78      0.35      0.49        82
          5       0.85      0.88      0.86       475
          6       0.63      0.09      0.15       138
          7       0.38      0.56      0.45       169

avg / total       0.65      0.64      0.62      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       294
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       381
          3       1.00      1.00      1.00        37
          5       1.00      1.00      1.00       489
          6       1.00      1.00      1.00        19
          7       1.00      1.00      1.00       248

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06943169 -0.09237212  0.03549336 -0.01054715 -0.13818386  0.04219978
 -0.04477229 -0.00777873]
Epoch number and batch_no:  127 0
Loss before optimizing :  0.707889473927
Loss, accuracy and verification results :  0.707889473927 0.723107569721 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.52      0.50      0.51       152
          1       0.40      0.50      0.44         4
          2       0.76      0.87      0.81       224
          3       0.69      0.67      0.68        70
          5       0.81      0.98      0.88       381
          6       0.53      0.23      0.32        82
          7       0.47      0.16      0.24        91

avg / total       0.69      0.72      0.69      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       145
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       257
          3       1.00      1.00      1.00        68
          5       1.00      1.00      1.00       461
          6       1.00      1.00      1.00        36
          7       1.00      1.00      1.00        32

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06928104 -0.09283864  0.03521956 -0.00962309 -0.13818421  0.04182943
 -0.043216   -0.00785358]
Epoch number and batch_no:  127 1
Loss before optimizing :  0.885422683853
Loss, accuracy and verification results :  0.885422683853 0.663036078965 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.60      0.50      0.54       305
          1       0.00      0.00      0.00        11
          2       0.69      0.81      0.74       289
          3       0.62      0.73      0.67        82
          5       0.80      0.91      0.85       475
          6       0.37      0.46      0.41       138
          7       0.49      0.20      0.28       169

avg / total       0.64      0.66      0.64      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       255
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       339
          3       1.00      1.00      1.00        96
          5       1.00      1.00      1.00       539
          6       1.00      1.00      1.00       169
          7       1.00      1.00      1.00        68

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.06965499 -0.09291782  0.03498815 -0.00981553 -0.13818456  0.04159055
 -0.04339224 -0.00692224]
Epoch number and batch_no:  128 0
Loss before optimizing :  0.63955115094
Loss, accuracy and verification results :  0.63955115094 0.774900398406 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.64      0.57      0.61       152
          1       1.00      0.25      0.40         4
          2       0.78      0.87      0.82       224
          3       0.76      0.81      0.79        70
          5       0.89      0.94      0.92       381
          6       0.54      0.60      0.57        82
          7       0.62      0.33      0.43        91

avg / total       0.77      0.77      0.76      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       135
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       249
          3       1.00      1.00      1.00        75
          5       1.00      1.00      1.00       405
          6       1.00      1.00      1.00        91
          7       1.00      1.00      1.00        48

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.06994417 -0.09298773  0.03488999 -0.01021446 -0.1381849   0.04162457
 -0.04439268 -0.00599602]
Epoch number and batch_no:  128 1
Loss before optimizing :  0.823926495094
Loss, accuracy and verification results :  0.823926495094 0.680054458816 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.57      0.50      0.53       305
          1       0.00      0.00      0.00        11
          2       0.71      0.78      0.74       289
          3       0.71      0.82      0.76        82
          5       0.83      0.95      0.88       475
          6       0.49      0.16      0.24       138
          7       0.43      0.51      0.47       169

avg / total       0.66      0.68      0.66      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       266
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       317
          3       1.00      1.00      1.00        95
          5       1.00      1.00      1.00       542
          6       1.00      1.00      1.00        45
          7       1.00      1.00      1.00       200

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07050914 -0.09302584  0.03508951 -0.01118595 -0.13818523  0.04176403
 -0.04449095 -0.00694078]
Epoch number and batch_no:  129 0
Loss before optimizing :  0.613211069402
Loss, accuracy and verification results :  0.613211069402 0.763944223108 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.66      0.60       152
          1       0.67      0.50      0.57         4
          2       0.76      0.88      0.81       224
          3       0.77      0.84      0.80        70
          5       0.87      0.98      0.92       381
          6       0.68      0.16      0.26        82
          7       0.66      0.23      0.34        91

avg / total       0.75      0.76      0.73      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       183
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       260
          3       1.00      1.00      1.00        77
          5       1.00      1.00      1.00       430
          6       1.00      1.00      1.00        19
          7       1.00      1.00      1.00        32

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0703535  -0.09318699  0.03506423 -0.0122414  -0.13818555  0.04180714
 -0.04346298 -0.00695252]
Epoch number and batch_no:  129 1
Loss before optimizing :  0.845905147917
Loss, accuracy and verification results :  0.845905147917 0.672566371681 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.54      0.60      0.57       305
          1       0.17      0.09      0.12        11
          2       0.65      0.85      0.74       289
          3       0.75      0.57      0.65        82
          5       0.79      0.94      0.86       475
          6       0.53      0.24      0.33       138
          7       0.56      0.18      0.28       169

avg / total       0.65      0.67      0.64      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       342
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       376
          3       1.00      1.00      1.00        63
          5       1.00      1.00      1.00       565
          6       1.00      1.00      1.00        62
          7       1.00      1.00      1.00        55

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07020135 -0.09310107  0.0344522  -0.01286191 -0.13818586  0.0416231
 -0.04217191 -0.00569993]
Epoch number and batch_no:  130 0
Loss before optimizing :  0.615936077414
Loss, accuracy and verification results :  0.615936077414 0.772908366534 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.73      0.39      0.51       152
          1       0.00      0.00      0.00         4
          2       0.81      0.85      0.83       224
          3       0.94      0.69      0.79        70
          5       0.86      0.97      0.92       381
          6       0.48      0.74      0.59        82
          7       0.56      0.49      0.53        91

avg / total       0.78      0.77      0.76      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        82
          2       1.00      1.00      1.00       236
          3       1.00      1.00      1.00        51
          5       1.00      1.00      1.00       429
          6       1.00      1.00      1.00       126
          7       1.00      1.00      1.00        80

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07064289 -0.0929036   0.03419581 -0.01233121 -0.13818616  0.04139671
 -0.04270748 -0.00512651]
Epoch number and batch_no:  130 1
Loss before optimizing :  0.773392641211
Loss, accuracy and verification results :  0.773392641211 0.706603131382 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.65      0.50      0.57       305
          1       0.40      0.18      0.25        11
          2       0.75      0.82      0.78       289
          3       0.90      0.52      0.66        82
          5       0.81      0.97      0.89       475
          6       0.51      0.49      0.50       138
          7       0.45      0.43      0.44       169

avg / total       0.70      0.71      0.69      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       235
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       317
          3       1.00      1.00      1.00        48
          5       1.00      1.00      1.00       567
          6       1.00      1.00      1.00       133
          7       1.00      1.00      1.00       164

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07186876 -0.09255805  0.03425677 -0.01081178 -0.13818645  0.04093049
 -0.04404709 -0.00576784]
Epoch number and batch_no:  131 0
Loss before optimizing :  0.599107532211
Loss, accuracy and verification results :  0.599107532211 0.782868525896 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.76      0.61       152
          1       1.00      0.25      0.40         4
          2       0.86      0.87      0.86       224
          3       0.68      1.00      0.81        70
          5       0.96      0.93      0.94       381
          6       0.73      0.29      0.42        82
          7       0.60      0.32      0.42        91

avg / total       0.80      0.78      0.77      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       226
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       226
          3       1.00      1.00      1.00       103
          5       1.00      1.00      1.00       367
          6       1.00      1.00      1.00        33
          7       1.00      1.00      1.00        48

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07153145 -0.09227897  0.0345934  -0.01117953 -0.13818673  0.04108994
 -0.04429399 -0.00600193]
Epoch number and batch_no:  131 1
Loss before optimizing :  0.742032709639
Loss, accuracy and verification results :  0.742032709639 0.71477195371 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.53      0.71      0.61       305
          1       0.43      0.27      0.33        11
          2       0.76      0.88      0.81       289
          3       0.60      0.82      0.69        82
          5       0.87      0.95      0.91       475
          6       0.71      0.18      0.29       138
          7       0.60      0.20      0.30       169

avg / total       0.71      0.71      0.68      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       406
          1       1.00      1.00      1.00         7
          2       1.00      1.00      1.00       335
          3       1.00      1.00      1.00       111
          5       1.00      1.00      1.00       518
          6       1.00      1.00      1.00        35
          7       1.00      1.00      1.00        57

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07041037 -0.09225977  0.03479242 -0.01309735 -0.13818701  0.04140834
 -0.04327856 -0.00517098]
Epoch number and batch_no:  132 0
Loss before optimizing :  0.547828534998
Loss, accuracy and verification results :  0.547828534998 0.803784860558 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.82      0.43      0.57       152
          1       0.67      1.00      0.80         4
          2       0.81      0.94      0.87       224
          3       0.84      0.84      0.84        70
          5       0.88      0.98      0.93       381
          6       0.59      0.59      0.59        82
          7       0.56      0.49      0.53        91

avg / total       0.80      0.80      0.79      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        80
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       261
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       426
          6       1.00      1.00      1.00        81
          7       1.00      1.00      1.00        80

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07038777 -0.09280873  0.03479931 -0.0142541  -0.13818729  0.04143315
 -0.04278446 -0.00468689]
Epoch number and batch_no:  132 1
Loss before optimizing :  0.750561224917
Loss, accuracy and verification results :  0.750561224917 0.709326072158 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.71      0.39      0.51       305
          1       0.50      0.36      0.42        11
          2       0.79      0.88      0.83       289
          3       0.91      0.38      0.53        82
          5       0.81      0.98      0.89       475
          6       0.45      0.61      0.52       138
          7       0.47      0.49      0.48       169

avg / total       0.72      0.71      0.69      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       170
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       322
          3       1.00      1.00      1.00        34
          5       1.00      1.00      1.00       573
          6       1.00      1.00      1.00       186
          7       1.00      1.00      1.00       176

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07199472 -0.09343863  0.03502302 -0.01360865 -0.13818756  0.04092275
 -0.04402464 -0.0053256 ]
Epoch number and batch_no:  133 0
Loss before optimizing :  0.517285759063
Loss, accuracy and verification results :  0.517285759063 0.827689243028 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.61      0.74      0.67       152
          1       0.67      0.50      0.57         4
          2       0.85      0.95      0.90       224
          3       0.90      0.80      0.85        70
          5       0.94      0.97      0.96       381
          6       0.71      0.48      0.57        82
          7       0.67      0.43      0.52        91

avg / total       0.83      0.83      0.82      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       186
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       248
          3       1.00      1.00      1.00        62
          5       1.00      1.00      1.00       392
          6       1.00      1.00      1.00        55
          7       1.00      1.00      1.00        58

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07238303 -0.0939428   0.03520068 -0.01199993 -0.13818782  0.04068686
 -0.0448941  -0.00578863]
Epoch number and batch_no:  133 1
Loss before optimizing :  0.704961608143
Loss, accuracy and verification results :  0.704961608143 0.73179033356 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.51      0.81      0.62       305
          1       0.00      0.00      0.00        11
          2       0.77      0.90      0.83       289
          3       0.71      0.85      0.77        82
          5       0.95      0.92      0.93       475
          6       0.85      0.17      0.28       138
          7       0.66      0.24      0.35       169

avg / total       0.76      0.73      0.70      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       484
          2       1.00      1.00      1.00       337
          3       1.00      1.00      1.00        99
          5       1.00      1.00      1.00       461
          6       1.00      1.00      1.00        27
          7       1.00      1.00      1.00        61

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07086349 -0.09384049  0.03520563 -0.01134909 -0.13818807  0.04097672
 -0.04419881 -0.00534382]
Epoch number and batch_no:  134 0
Loss before optimizing :  0.50617432614
Loss, accuracy and verification results :  0.50617432614 0.810756972112 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.77      0.43      0.55       152
          1       0.75      0.75      0.75         4
          2       0.80      0.96      0.87       224
          3       0.69      0.97      0.81        70
          5       0.89      0.98      0.94       381
          6       0.73      0.52      0.61        82
          7       0.66      0.51      0.57        91

avg / total       0.80      0.81      0.79      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        84
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       268
          3       1.00      1.00      1.00        98
          5       1.00      1.00      1.00       421
          6       1.00      1.00      1.00        59
          7       1.00      1.00      1.00        70

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07034192 -0.09370108  0.03493416 -0.01191019 -0.13818831  0.04104643
 -0.04317295 -0.00476498]
Epoch number and batch_no:  134 1
Loss before optimizing :  0.706617086578
Loss, accuracy and verification results :  0.706617086578 0.72157930565 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.74      0.30      0.42       305
          1       0.50      0.09      0.15        11
          2       0.80      0.89      0.84       289
          3       0.77      0.84      0.80        82
          5       0.81      0.99      0.89       475
          6       0.48      0.61      0.54       138
          7       0.50      0.54      0.52       169

avg / total       0.72      0.72      0.70      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       121
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       321
          3       1.00      1.00      1.00        90
          5       1.00      1.00      1.00       576
          6       1.00      1.00      1.00       176
          7       1.00      1.00      1.00       183

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07208274 -0.0933115   0.03482603 -0.01288383 -0.13818854  0.04050539
 -0.04346909 -0.004986  ]
Epoch number and batch_no:  135 0
Loss before optimizing :  0.482149206838
Loss, accuracy and verification results :  0.482149206838 0.826693227092 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.63      0.64      0.64       152
          1       0.60      0.75      0.67         4
          2       0.89      0.93      0.91       224
          3       0.95      0.81      0.88        70
          5       0.96      0.96      0.96       381
          6       0.57      0.67      0.61        82
          7       0.63      0.46      0.53        91

avg / total       0.83      0.83      0.82      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       156
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       235
          3       1.00      1.00      1.00        60
          5       1.00      1.00      1.00       384
          6       1.00      1.00      1.00        97
          7       1.00      1.00      1.00        67

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07304683 -0.09307274  0.03489773 -0.01285769 -0.13818876  0.0402959
 -0.04472479 -0.0049898 ]
Epoch number and batch_no:  135 1
Loss before optimizing :  0.714093543565
Loss, accuracy and verification results :  0.714093543565 0.718856364874 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.48      0.83      0.61       305
          1       1.00      0.09      0.17        11
          2       0.82      0.80      0.81       289
          3       0.85      0.61      0.71        82
          5       0.96      0.87      0.91       475
          6       0.70      0.33      0.45       138
          7       0.62      0.36      0.46       169

avg / total       0.76      0.72      0.72      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       532
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       284
          3       1.00      1.00      1.00        59
          5       1.00      1.00      1.00       431
          6       1.00      1.00      1.00        64
          7       1.00      1.00      1.00        98

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07140665 -0.09264804  0.03513924 -0.01193387 -0.13818897  0.04095798
 -0.04539463 -0.0048261 ]
Epoch number and batch_no:  136 0
Loss before optimizing :  0.464919158686
Loss, accuracy and verification results :  0.464919158686 0.830677290837 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.76      0.56      0.64       152
          1       0.50      0.75      0.60         4
          2       0.82      0.97      0.89       224
          3       0.85      0.91      0.88        70
          5       0.89      0.99      0.94       381
          6       0.75      0.50      0.60        82
          7       0.70      0.51      0.59        91

avg / total       0.82      0.83      0.82      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       112
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       265
          3       1.00      1.00      1.00        75
          5       1.00      1.00      1.00       425
          6       1.00      1.00      1.00        55
          7       1.00      1.00      1.00        66

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07053387 -0.09284233  0.03504942 -0.01119605 -0.13818917  0.04123369
 -0.0452435  -0.00458669]
Epoch number and batch_no:  136 1
Loss before optimizing :  0.723340957199
Loss, accuracy and verification results :  0.723340957199 0.710006807352 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.71      0.32      0.44       305
          1       0.42      0.45      0.43        11
          2       0.77      0.87      0.81       289
          3       0.63      0.95      0.76        82
          5       0.78      0.98      0.87       475
          6       0.63      0.42      0.50       138
          7       0.49      0.51      0.50       169

avg / total       0.70      0.71      0.68      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       139
          1       1.00      1.00      1.00        12
          2       1.00      1.00      1.00       327
          3       1.00      1.00      1.00       123
          5       1.00      1.00      1.00       599
          6       1.00      1.00      1.00        92
          7       1.00      1.00      1.00       177

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07200563 -0.09352934  0.03488703 -0.01200239 -0.13818937  0.0406178
 -0.04436978 -0.00498933]
Epoch number and batch_no:  137 0
Loss before optimizing :  0.432019190724
Loss, accuracy and verification results :  0.432019190724 0.851593625498 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.70      0.61      0.65       152
          1       0.75      0.75      0.75         4
          2       0.89      0.94      0.91       224
          3       0.83      0.99      0.90        70
          5       0.95      0.99      0.97       381
          6       0.69      0.66      0.68        82
          7       0.66      0.54      0.59        91

avg / total       0.84      0.85      0.85      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       131
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       236
          3       1.00      1.00      1.00        83
          5       1.00      1.00      1.00       398
          6       1.00      1.00      1.00        78
          7       1.00      1.00      1.00        74

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07342723 -0.09428711  0.03490035 -0.013207   -0.13818957  0.04006155
 -0.04396669 -0.00507679]
Epoch number and batch_no:  137 1
Loss before optimizing :  0.630794744129
Loss, accuracy and verification results :  0.630794744129 0.762423417291 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.55      0.82      0.66       305
          1       1.00      0.09      0.17        11
          2       0.85      0.87      0.86       289
          3       0.90      0.74      0.81        82
          5       0.95      0.91      0.93       475
          6       0.62      0.47      0.54       138
          7       0.69      0.35      0.46       169

avg / total       0.78      0.76      0.76      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       460
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       295
          3       1.00      1.00      1.00        68
          5       1.00      1.00      1.00       455
          6       1.00      1.00      1.00       104
          7       1.00      1.00      1.00        86

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0731005  -0.09444053  0.03507503 -0.01373734 -0.13818976  0.04012629
 -0.04417989 -0.0044359 ]
Epoch number and batch_no:  138 0
Loss before optimizing :  0.460695556556
Loss, accuracy and verification results :  0.460695556556 0.840637450199 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.61      0.70      0.65       152
          1       1.00      0.25      0.40         4
          2       0.91      0.93      0.92       224
          3       0.92      0.87      0.90        70
          5       0.97      0.94      0.95       381
          6       0.68      0.61      0.65        82
          7       0.64      0.64      0.64        91

avg / total       0.85      0.84      0.84      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       176
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       229
          3       1.00      1.00      1.00        66
          5       1.00      1.00      1.00       369
          6       1.00      1.00      1.00        73
          7       1.00      1.00      1.00        90

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07197229 -0.09440303  0.03537291 -0.01357634 -0.13818994  0.04063283
 -0.04480317 -0.00413033]
Epoch number and batch_no:  138 1
Loss before optimizing :  0.585268174062
Loss, accuracy and verification results :  0.585268174062 0.789652825051 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.75      0.55      0.63       305
          1       1.00      0.09      0.17        11
          2       0.76      0.96      0.85       289
          3       0.89      0.94      0.91        82
          5       0.92      0.97      0.94       475
          6       0.67      0.52      0.59       138
          7       0.58      0.63      0.60       169

avg / total       0.79      0.79      0.78      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       224
          1       1.00      1.00      1.00         1
          2       1.00      1.00      1.00       365
          3       1.00      1.00      1.00        87
          5       1.00      1.00      1.00       501
          6       1.00      1.00      1.00       107
          7       1.00      1.00      1.00       184

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07210385 -0.09393926  0.03514026 -0.01332876 -0.13819012  0.04103588
 -0.04533033 -0.00483352]
Epoch number and batch_no:  139 0
Loss before optimizing :  0.438730975172
Loss, accuracy and verification results :  0.438730975172 0.841633466135 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.78      0.55      0.64       152
          1       1.00      0.50      0.67         4
          2       0.86      0.97      0.91       224
          3       0.85      0.94      0.89        70
          5       0.90      0.99      0.94       381
          6       0.71      0.59      0.64        82
          7       0.66      0.57      0.61        91

avg / total       0.83      0.84      0.83      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       106
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       253
          3       1.00      1.00      1.00        78
          5       1.00      1.00      1.00       418
          6       1.00      1.00      1.00        68
          7       1.00      1.00      1.00        79

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0728797  -0.09365662  0.03470502 -0.01323621 -0.13819029  0.04104738
 -0.04547807 -0.00527623]
Epoch number and batch_no:  139 1
Loss before optimizing :  0.583258924506
Loss, accuracy and verification results :  0.583258924506 0.778080326753 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.64      0.69      0.66       305
          1       0.50      0.36      0.42        11
          2       0.92      0.86      0.89       289
          3       0.74      0.96      0.84        82
          5       0.84      0.99      0.91       475
          6       0.67      0.45      0.54       138
          7       0.67      0.41      0.51       169

avg / total       0.77      0.78      0.77      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       328
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       271
          3       1.00      1.00      1.00       107
          5       1.00      1.00      1.00       557
          6       1.00      1.00      1.00        93
          7       1.00      1.00      1.00       105

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07350872 -0.09345532  0.03486325 -0.01365475 -0.13819046  0.04043537
 -0.04519962 -0.00496264]
Epoch number and batch_no:  140 0
Loss before optimizing :  0.423072067133
Loss, accuracy and verification results :  0.423072067133 0.854581673307 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.63      0.79      0.70       152
          1       0.75      0.75      0.75         4
          2       0.92      0.91      0.92       224
          3       0.87      0.99      0.93        70
          5       0.95      0.98      0.96       381
          6       0.78      0.55      0.64        82
          7       0.76      0.49      0.60        91

avg / total       0.86      0.85      0.85      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       191
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       221
          3       1.00      1.00      1.00        79
          5       1.00      1.00      1.00       392
          6       1.00      1.00      1.00        58
          7       1.00      1.00      1.00        59

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07308558 -0.09341371  0.03531486 -0.01406212 -0.13819061  0.03999269
 -0.04461137 -0.00423215]
Epoch number and batch_no:  140 1
Loss before optimizing :  0.563861529798
Loss, accuracy and verification results :  0.563861529798 0.801225323349 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.67      0.67      0.67       305
          1       0.62      0.45      0.53        11
          2       0.84      0.92      0.88       289
          3       0.94      0.88      0.91        82
          5       0.94      0.96      0.95       475
          6       0.62      0.55      0.58       138
          7       0.63      0.59      0.61       169

avg / total       0.80      0.80      0.80      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       305
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       316
          3       1.00      1.00      1.00        77
          5       1.00      1.00      1.00       485
          6       1.00      1.00      1.00       122
          7       1.00      1.00      1.00       156

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0728279  -0.09351854  0.03558917 -0.01378769 -0.13819076  0.03978001
 -0.04439952 -0.00413804]
Epoch number and batch_no:  141 0
Loss before optimizing :  0.416274965548
Loss, accuracy and verification results :  0.416274965548 0.860557768924 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.82      0.55      0.66       152
          1       0.60      0.75      0.67         4
          2       0.83      0.98      0.90       224
          3       0.88      0.96      0.92        70
          5       0.96      0.97      0.96       381
          6       0.73      0.71      0.72        82
          7       0.70      0.70      0.70        91

avg / total       0.86      0.86      0.85      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       101
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       266
          3       1.00      1.00      1.00        76
          5       1.00      1.00      1.00       386
          6       1.00      1.00      1.00        79
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07311257 -0.09365065  0.03529829 -0.01348038 -0.13819092  0.03994922
 -0.04450343 -0.00456318]
Epoch number and batch_no:  141 1
Loss before optimizing :  0.526823644478
Loss, accuracy and verification results :  0.526823644478 0.808032675289 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.71      0.63      0.67       305
          1       0.60      0.27      0.37        11
          2       0.83      0.94      0.88       289
          3       0.90      0.96      0.93        82
          5       0.93      0.98      0.96       475
          6       0.62      0.62      0.62       138
          7       0.64      0.54      0.59       169

avg / total       0.80      0.81      0.80      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       272
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       328
          3       1.00      1.00      1.00        88
          5       1.00      1.00      1.00       498
          6       1.00      1.00      1.00       136
          7       1.00      1.00      1.00       142

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07382223 -0.09379437  0.03482607 -0.01309106 -0.13819106  0.04013162
 -0.04528933 -0.00474875]
Epoch number and batch_no:  142 0
Loss before optimizing :  0.392433713025
Loss, accuracy and verification results :  0.392433713025 0.867529880478 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.66      0.74      0.70       152
          1       1.00      1.00      1.00         4
          2       0.95      0.91      0.93       224
          3       0.87      0.99      0.93        70
          5       0.94      0.99      0.96       381
          6       0.83      0.54      0.65        82
          7       0.75      0.65      0.69        91

avg / total       0.87      0.87      0.86      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       170
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       214
          3       1.00      1.00      1.00        79
          5       1.00      1.00      1.00       405
          6       1.00      1.00      1.00        53
          7       1.00      1.00      1.00        79

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07386644 -0.09395081  0.03490832 -0.01315013 -0.1381912   0.04018235
 -0.04565006 -0.00469314]
Epoch number and batch_no:  142 1
Loss before optimizing :  0.523876885976
Loss, accuracy and verification results :  0.523876885976 0.806671204901 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.65      0.73      0.69       305
          1       0.50      0.27      0.35        11
          2       0.90      0.89      0.90       289
          3       0.82      0.96      0.89        82
          5       0.92      0.98      0.95       475
          6       0.77      0.46      0.57       138
          7       0.62      0.54      0.58       169

avg / total       0.80      0.81      0.80      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       343
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       285
          3       1.00      1.00      1.00        96
          5       1.00      1.00      1.00       508
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00       149

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07347784 -0.09402436  0.03532431 -0.01340722 -0.13819133  0.04009481
 -0.04547507 -0.00462586]
Epoch number and batch_no:  143 0
Loss before optimizing :  0.364182534725
Loss, accuracy and verification results :  0.364182534725 0.877490039841 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.75      0.70      0.72       152
          1       1.00      0.75      0.86         4
          2       0.91      0.97      0.94       224
          3       0.92      1.00      0.96        70
          5       0.94      0.98      0.96       381
          6       0.81      0.59      0.68        82
          7       0.70      0.69      0.70        91

avg / total       0.87      0.88      0.87      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       141
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       239
          3       1.00      1.00      1.00        76
          5       1.00      1.00      1.00       396
          6       1.00      1.00      1.00        59
          7       1.00      1.00      1.00        90

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07323275 -0.09412477  0.03561372 -0.01369513 -0.13819146  0.03996196
 -0.04509408 -0.00457511]
Epoch number and batch_no:  143 1
Loss before optimizing :  0.514662971665
Loss, accuracy and verification results :  0.514662971665 0.805990469707 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.72      0.65      0.69       305
          1       0.75      0.27      0.40        11
          2       0.82      0.96      0.88       289
          3       0.85      0.87      0.86        82
          5       0.93      0.98      0.95       475
          6       0.63      0.52      0.57       138
          7       0.63      0.58      0.60       169

avg / total       0.80      0.81      0.80      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       275
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       338
          3       1.00      1.00      1.00        84
          5       1.00      1.00      1.00       497
          6       1.00      1.00      1.00       115
          7       1.00      1.00      1.00       156

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07358276 -0.09405684  0.03551226 -0.01389926 -0.13819159  0.03984764
 -0.04496032 -0.00463413]
Epoch number and batch_no:  144 0
Loss before optimizing :  0.402395985267
Loss, accuracy and verification results :  0.402395985267 0.864541832669 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.76      0.64      0.70       152
          1       0.50      0.50      0.50         4
          2       0.88      0.97      0.92       224
          3       0.89      0.90      0.89        70
          5       0.94      0.98      0.96       381
          6       0.72      0.71      0.72        82
          7       0.72      0.62      0.66        91

avg / total       0.86      0.86      0.86      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       129
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       246
          3       1.00      1.00      1.00        71
          5       1.00      1.00      1.00       396
          6       1.00      1.00      1.00        80
          7       1.00      1.00      1.00        78

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07404756 -0.09400559  0.03541896 -0.01381061 -0.13819171  0.03986351
 -0.04554474 -0.00477771]
Epoch number and batch_no:  144 1
Loss before optimizing :  0.510574145126
Loss, accuracy and verification results :  0.510574145126 0.817562968005 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.66      0.78      0.71       305
          1       1.00      0.27      0.43        11
          2       0.89      0.94      0.91       289
          3       0.88      0.94      0.91        82
          5       0.94      0.97      0.95       475
          6       0.76      0.43      0.55       138
          7       0.66      0.55      0.60       169

avg / total       0.82      0.82      0.81      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       362
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       307
          3       1.00      1.00      1.00        88
          5       1.00      1.00      1.00       490
          6       1.00      1.00      1.00        78
          7       1.00      1.00      1.00       141

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07378616 -0.09385411  0.03541488 -0.01357975 -0.13819182  0.03990206
 -0.04554422 -0.0046542 ]
Epoch number and batch_no:  145 0
Loss before optimizing :  0.347416897656
Loss, accuracy and verification results :  0.347416897656 0.895418326693 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.76      0.77      0.77       152
          1       0.67      1.00      0.80         4
          2       0.94      0.96      0.95       224
          3       0.95      0.99      0.97        70
          5       0.95      0.99      0.97       381
          6       0.83      0.65      0.73        82
          7       0.75      0.67      0.71        91

avg / total       0.89      0.90      0.89      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       153
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       229
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       398
          6       1.00      1.00      1.00        64
          7       1.00      1.00      1.00        81

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0733539  -0.09387076  0.03555378 -0.01329788 -0.13819194  0.03984629
 -0.04541876 -0.00437618]
Epoch number and batch_no:  145 1
Loss before optimizing :  0.489826775508
Loss, accuracy and verification results :  0.489826775508 0.818924438393 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.73      0.64      0.68       305
          1       0.80      0.36      0.50        11
          2       0.86      0.96      0.91       289
          3       0.84      0.93      0.88        82
          5       0.94      0.96      0.95       475
          6       0.66      0.53      0.59       138
          7       0.64      0.72      0.68       169

avg / total       0.81      0.82      0.81      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       265
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       322
          3       1.00      1.00      1.00        91
          5       1.00      1.00      1.00       486
          6       1.00      1.00      1.00       111
          7       1.00      1.00      1.00       189

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07355299 -0.09384478  0.03543463 -0.0130825  -0.13819205  0.03984133
 -0.04534522 -0.00468417]
Epoch number and batch_no:  146 0
Loss before optimizing :  0.352877258152
Loss, accuracy and verification results :  0.352877258152 0.886454183267 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.81      0.68      0.74       152
          1       1.00      1.00      1.00         4
          2       0.95      0.96      0.96       224
          3       0.93      0.99      0.96        70
          5       0.95      0.99      0.97       381
          6       0.72      0.72      0.72        82
          7       0.67      0.68      0.68        91

avg / total       0.88      0.89      0.88      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       129
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       228
          3       1.00      1.00      1.00        74
          5       1.00      1.00      1.00       395
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        92

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07389909 -0.09391093  0.03538874 -0.01286109 -0.13819217  0.03974967
 -0.04569299 -0.00470842]
Epoch number and batch_no:  146 1
Loss before optimizing :  0.470831717963
Loss, accuracy and verification results :  0.470831717963 0.833219877468 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.70      0.74      0.72       305
          1       0.67      0.18      0.29        11
          2       0.89      0.95      0.92       289
          3       0.83      0.98      0.90        82
          5       0.93      0.99      0.96       475
          6       0.75      0.51      0.61       138
          7       0.73      0.60      0.66       169

avg / total       0.83      0.83      0.82      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       321
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       309
          3       1.00      1.00      1.00        96
          5       1.00      1.00      1.00       507
          6       1.00      1.00      1.00        95
          7       1.00      1.00      1.00       138

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07421989 -0.09369221  0.03527445 -0.01309158 -0.13819228  0.03954348
 -0.04577302 -0.00425207]
Epoch number and batch_no:  147 0
Loss before optimizing :  0.329135695194
Loss, accuracy and verification results :  0.329135695194 0.905378486056 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.74      0.80      0.77       152
          1       1.00      0.75      0.86         4
          2       0.96      0.96      0.96       224
          3       0.91      0.99      0.95        70
          5       0.98      1.00      0.99       381
          6       0.82      0.65      0.72        82
          7       0.78      0.73      0.75        91

avg / total       0.90      0.91      0.90      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       164
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       225
          3       1.00      1.00      1.00        76
          5       1.00      1.00      1.00       386
          6       1.00      1.00      1.00        65
          7       1.00      1.00      1.00        85

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07397468 -0.09357078  0.0354305  -0.01330948 -0.13819239  0.03939851
 -0.0456532  -0.00375391]
Epoch number and batch_no:  147 1
Loss before optimizing :  0.444858059012
Loss, accuracy and verification results :  0.444858059012 0.842750170184 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.72      0.73      0.73       305
          1       1.00      0.27      0.43        11
          2       0.91      0.94      0.93       289
          3       0.88      1.00      0.94        82
          5       0.96      0.98      0.97       475
          6       0.72      0.51      0.60       138
          7       0.67      0.73      0.70       169

avg / total       0.84      0.84      0.84      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       307
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       298
          3       1.00      1.00      1.00        93
          5       1.00      1.00      1.00       486
          6       1.00      1.00      1.00        97
          7       1.00      1.00      1.00       185

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07377144 -0.09325288  0.03568587 -0.01370948 -0.1381925   0.03939193
 -0.04543086 -0.00396781]
Epoch number and batch_no:  148 0
Loss before optimizing :  0.310504218741
Loss, accuracy and verification results :  0.310504218741 0.907370517928 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.82      0.70      0.76       152
          1       0.67      1.00      0.80         4
          2       0.93      1.00      0.97       224
          3       0.95      1.00      0.97        70
          5       0.98      0.99      0.99       381
          6       0.77      0.68      0.72        82
          7       0.75      0.80      0.78        91

avg / total       0.90      0.91      0.90      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       131
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       240
          3       1.00      1.00      1.00        74
          5       1.00      1.00      1.00       383
          6       1.00      1.00      1.00        73
          7       1.00      1.00      1.00        97

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07373821 -0.09317213  0.03580825 -0.01388329 -0.1381926   0.03949165
 -0.04553673 -0.00425265]
Epoch number and batch_no:  148 1
Loss before optimizing :  0.433702468784
Loss, accuracy and verification results :  0.433702468784 0.849557522124 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.75      0.71      0.73       305
          1       0.67      0.55      0.60        11
          2       0.86      0.96      0.91       289
          3       0.91      0.96      0.93        82
          5       0.96      0.99      0.98       475
          6       0.66      0.62      0.64       138
          7       0.79      0.66      0.72       169

avg / total       0.84      0.85      0.85      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       292
          1       1.00      1.00      1.00         9
          2       1.00      1.00      1.00       322
          3       1.00      1.00      1.00        87
          5       1.00      1.00      1.00       488
          6       1.00      1.00      1.00       130
          7       1.00      1.00      1.00       141

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07411063 -0.09323972  0.03568596 -0.01391193 -0.13819269  0.03959004
 -0.04613944 -0.00427144]
Epoch number and batch_no:  149 0
Loss before optimizing :  0.296700708514
Loss, accuracy and verification results :  0.296700708514 0.910358565737 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.79      0.79      0.79       152
          1       1.00      1.00      1.00         4
          2       0.93      0.99      0.96       224
          3       0.95      1.00      0.97        70
          5       0.97      1.00      0.98       381
          6       0.89      0.67      0.76        82
          7       0.77      0.69      0.73        91

avg / total       0.91      0.91      0.91      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       151
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       238
          3       1.00      1.00      1.00        74
          5       1.00      1.00      1.00       393
          6       1.00      1.00      1.00        62
          7       1.00      1.00      1.00        82

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07426266 -0.09340893  0.03550319 -0.01390276 -0.13819279  0.03964674
 -0.04639821 -0.00400006]
Epoch number and batch_no:  149 1
Loss before optimizing :  0.421504366
Loss, accuracy and verification results :  0.421504366 0.859768550034 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.72      0.78      0.75       305
          1       1.00      0.27      0.43        11
          2       0.95      0.94      0.95       289
          3       0.91      0.96      0.93        82
          5       0.96      0.99      0.97       475
          6       0.79      0.51      0.62       138
          7       0.71      0.77      0.74       169

avg / total       0.86      0.86      0.86      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       330
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       284
          3       1.00      1.00      1.00        87
          5       1.00      1.00      1.00       492
          6       1.00      1.00      1.00        89
          7       1.00      1.00      1.00       184

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07415752 -0.09344286  0.03572207 -0.01400206 -0.13819288  0.03965103
 -0.04618352 -0.00442376]
Epoch number and batch_no:  150 0
Loss before optimizing :  0.283008751656
Loss, accuracy and verification results :  0.283008751656 0.918326693227 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.85      0.76      0.80       152
          1       1.00      0.75      0.86         4
          2       0.94      1.00      0.97       224
          3       0.95      1.00      0.97        70
          5       0.97      1.00      0.98       381
          6       0.81      0.73      0.77        82
          7       0.80      0.77      0.78        91

avg / total       0.92      0.92      0.92      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       135
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       236
          3       1.00      1.00      1.00        74
          5       1.00      1.00      1.00       394
          6       1.00      1.00      1.00        74
          7       1.00      1.00      1.00        88

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07422198 -0.09345052  0.03587043 -0.01412063 -0.13819296  0.03956636
 -0.04618288 -0.00451769]
Epoch number and batch_no:  150 1
Loss before optimizing :  0.399101061618
Loss, accuracy and verification results :  0.399101061618 0.855003403676 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.75      0.73      0.74       305
          1       0.83      0.45      0.59        11
          2       0.90      0.97      0.93       289
          3       0.94      0.98      0.96        82
          5       0.95      0.99      0.97       475
          6       0.68      0.62      0.65       138
          7       0.76      0.65      0.70       169

avg / total       0.85      0.86      0.85      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       298
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       312
          3       1.00      1.00      1.00        85
          5       1.00      1.00      1.00       497
          6       1.00      1.00      1.00       127
          7       1.00      1.00      1.00       144

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07454827 -0.0933627   0.03590891 -0.01429166 -0.13819304  0.03940554
 -0.04660502 -0.00418658]
Epoch number and batch_no:  151 0
Loss before optimizing :  0.277529738215
Loss, accuracy and verification results :  0.277529738215 0.925298804781 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.83      0.76      0.79       152
          1       1.00      1.00      1.00         4
          2       0.94      1.00      0.97       224
          3       0.96      1.00      0.98        70
          5       0.99      0.99      0.99       381
          6       0.83      0.73      0.78        82
          7       0.79      0.84      0.81        91

avg / total       0.92      0.93      0.92      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       140
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       238
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        72
          7       1.00      1.00      1.00        96

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07478668 -0.09334801  0.03574452 -0.01409729 -0.13819312  0.0394063
 -0.04703386 -0.00396996]
Epoch number and batch_no:  151 1
Loss before optimizing :  0.389983575625
Loss, accuracy and verification results :  0.389983575625 0.859768550034 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.71      0.79      0.75       305
          1       0.62      0.45      0.53        11
          2       0.93      0.96      0.95       289
          3       0.90      0.99      0.94        82
          5       0.96      0.99      0.98       475
          6       0.85      0.43      0.57       138
          7       0.74      0.75      0.74       169

avg / total       0.86      0.86      0.85      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       340
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       297
          3       1.00      1.00      1.00        90
          5       1.00      1.00      1.00       491
          6       1.00      1.00      1.00        71
          7       1.00      1.00      1.00       172

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07478771 -0.09341732  0.0356204  -0.01407734 -0.13819319  0.0393875
 -0.04656813 -0.00410445]
Epoch number and batch_no:  152 0
Loss before optimizing :  0.265897100145
Loss, accuracy and verification results :  0.265897100145 0.92828685259 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.83      0.82      0.83       152
          1       1.00      1.00      1.00         4
          2       0.99      0.98      0.98       224
          3       0.90      1.00      0.95        70
          5       0.98      0.99      0.99       381
          6       0.86      0.70      0.77        82
          7       0.81      0.85      0.83        91

avg / total       0.93      0.93      0.93      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       151
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       223
          3       1.00      1.00      1.00        78
          5       1.00      1.00      1.00       387
          6       1.00      1.00      1.00        66
          7       1.00      1.00      1.00        95

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07466866 -0.09361459  0.03568113 -0.01437543 -0.13819326  0.0393971
 -0.04601001 -0.00432491]
Epoch number and batch_no:  152 1
Loss before optimizing :  0.369873917852
Loss, accuracy and verification results :  0.369873917852 0.878829135466 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.80      0.77      0.78       305
          1       0.71      0.45      0.56        11
          2       0.94      0.98      0.96       289
          3       0.92      0.99      0.95        82
          5       0.97      1.00      0.98       475
          6       0.70      0.69      0.69       138
          7       0.78      0.70      0.74       169

avg / total       0.87      0.88      0.88      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       294
          1       1.00      1.00      1.00         7
          2       1.00      1.00      1.00       301
          3       1.00      1.00      1.00        88
          5       1.00      1.00      1.00       491
          6       1.00      1.00      1.00       136
          7       1.00      1.00      1.00       152

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07492899 -0.09371942  0.0357251  -0.01479228 -0.13819333  0.03933216
 -0.04606453 -0.00428493]
Epoch number and batch_no:  153 0
Loss before optimizing :  0.255816702457
Loss, accuracy and verification results :  0.255816702457 0.926294820717 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.84      0.76      0.80       152
          1       1.00      1.00      1.00         4
          2       0.96      1.00      0.98       224
          3       0.96      0.99      0.97        70
          5       0.98      1.00      0.99       381
          6       0.80      0.80      0.80        82
          7       0.82      0.79      0.80        91

avg / total       0.92      0.93      0.92      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       138
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       232
          3       1.00      1.00      1.00        72
          5       1.00      1.00      1.00       387
          6       1.00      1.00      1.00        83
          7       1.00      1.00      1.00        88

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07523437 -0.09381923  0.03579334 -0.01499847 -0.1381934   0.03927146
 -0.04656496 -0.00409954]
Epoch number and batch_no:  153 1
Loss before optimizing :  0.358944677705
Loss, accuracy and verification results :  0.358944677705 0.887678692988 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.76      0.83      0.79       305
          1       1.00      0.18      0.31        11
          2       0.94      0.99      0.96       289
          3       0.98      0.98      0.98        82
          5       0.98      0.99      0.99       475
          6       0.83      0.57      0.67       138
          7       0.77      0.79      0.78       169

avg / total       0.89      0.89      0.88      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       334
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       303
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       481
          6       1.00      1.00      1.00        94
          7       1.00      1.00      1.00       173

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07511367 -0.09356807  0.03580593 -0.01493476 -0.13819347  0.03932671
 -0.04654249 -0.00423184]
Epoch number and batch_no:  154 0
Loss before optimizing :  0.24615864586
Loss, accuracy and verification results :  0.24615864586 0.934262948207 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.83      0.86      0.84       152
          1       1.00      1.00      1.00         4
          2       0.98      1.00      0.99       224
          3       0.96      1.00      0.98        70
          5       0.98      0.99      0.99       381
          6       0.90      0.70      0.79        82
          7       0.80      0.82      0.81        91

avg / total       0.93      0.93      0.93      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       156
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       227
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       387
          6       1.00      1.00      1.00        63
          7       1.00      1.00      1.00        94

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0748673  -0.09342334  0.03587979 -0.01488362 -0.13819353  0.03933413
 -0.04637189 -0.00424476]
Epoch number and batch_no:  154 1
Loss before optimizing :  0.345038033049
Loss, accuracy and verification results :  0.345038033049 0.891082368958 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.82      0.77      0.80       305
          1       1.00      0.55      0.71        11
          2       0.93      0.98      0.95       289
          3       0.95      1.00      0.98        82
          5       0.98      0.99      0.98       475
          6       0.75      0.69      0.72       138
          7       0.77      0.80      0.79       169

avg / total       0.89      0.89      0.89      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       287
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       305
          3       1.00      1.00      1.00        86
          5       1.00      1.00      1.00       483
          6       1.00      1.00      1.00       126
          7       1.00      1.00      1.00       176

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07503901 -0.09323454  0.03584121 -0.01484256 -0.13819359  0.03934486
 -0.04653728 -0.00441584]
Epoch number and batch_no:  155 0
Loss before optimizing :  0.239034337719
Loss, accuracy and verification results :  0.239034337719 0.927290836653 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.87      0.76      0.81       152
          1       1.00      1.00      1.00         4
          2       0.95      0.99      0.97       224
          3       0.97      1.00      0.99        70
          5       0.98      1.00      0.99       381
          6       0.82      0.79      0.81        82
          7       0.80      0.81      0.80        91

avg / total       0.93      0.93      0.93      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       134
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       234
          3       1.00      1.00      1.00        72
          5       1.00      1.00      1.00       388
          6       1.00      1.00      1.00        79
          7       1.00      1.00      1.00        93

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07540844 -0.09322687  0.03582432 -0.01493873 -0.13819365  0.03929315
 -0.04696622 -0.00437092]
Epoch number and batch_no:  155 1
Loss before optimizing :  0.335377775435
Loss, accuracy and verification results :  0.335377775435 0.901974132063 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.78      0.86      0.82       305
          1       0.89      0.73      0.80        11
          2       0.95      1.00      0.97       289
          3       0.95      1.00      0.98        82
          5       0.98      0.99      0.98       475
          6       0.83      0.60      0.70       138
          7       0.85      0.78      0.81       169

avg / total       0.90      0.90      0.90      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       334
          1       1.00      1.00      1.00         9
          2       1.00      1.00      1.00       303
          3       1.00      1.00      1.00        86
          5       1.00      1.00      1.00       483
          6       1.00      1.00      1.00       100
          7       1.00      1.00      1.00       154

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07547407 -0.09339459  0.0357909  -0.01499921 -0.1381937   0.03924231
 -0.04701211 -0.00412157]
Epoch number and batch_no:  156 0
Loss before optimizing :  0.260537567135
Loss, accuracy and verification results :  0.260537567135 0.921314741036 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.80      0.86      0.83       152
          1       0.80      1.00      0.89         4
          2       0.97      0.98      0.98       224
          3       0.96      1.00      0.98        70
          5       0.98      0.98      0.98       381
          6       0.92      0.67      0.77        82
          7       0.74      0.80      0.77        91

avg / total       0.92      0.92      0.92      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       163
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       226
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       379
          6       1.00      1.00      1.00        60
          7       1.00      1.00      1.00        98

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07522224 -0.09375894  0.03588367 -0.0150288  -0.13819375  0.03927949
 -0.04666412 -0.00419357]
Epoch number and batch_no:  156 1
Loss before optimizing :  0.354519685841
Loss, accuracy and verification results :  0.354519685841 0.880871341048 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.80      0.80      0.80       305
          1       1.00      0.27      0.43        11
          2       0.92      0.97      0.94       289
          3       0.96      0.95      0.96        82
          5       0.97      0.99      0.98       475
          6       0.77      0.62      0.68       138
          7       0.75      0.80      0.77       169

avg / total       0.88      0.88      0.88      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       305
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       304
          3       1.00      1.00      1.00        81
          5       1.00      1.00      1.00       483
          6       1.00      1.00      1.00       111
          7       1.00      1.00      1.00       182

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07517341 -0.09382399  0.03603736 -0.0148751  -0.1381938   0.03931096
 -0.04655911 -0.00471593]
Epoch number and batch_no:  157 0
Loss before optimizing :  0.238939000967
Loss, accuracy and verification results :  0.238939000967 0.935258964143 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.90      0.77      0.83       152
          1       1.00      1.00      1.00         4
          2       0.95      1.00      0.97       224
          3       0.97      0.99      0.98        70
          5       0.97      1.00      0.99       381
          6       0.82      0.82      0.82        82
          7       0.87      0.86      0.86        91

avg / total       0.93      0.94      0.93      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       130
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       237
          3       1.00      1.00      1.00        71
          5       1.00      1.00      1.00       390
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        90

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0754018  -0.09393232  0.03603002 -0.01457277 -0.13819385  0.03929842
 -0.04688383 -0.00491313]
Epoch number and batch_no:  157 1
Loss before optimizing :  0.38321606744
Loss, accuracy and verification results :  0.38321606744 0.86249149081 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.72      0.82      0.77       305
          1       0.80      0.36      0.50        11
          2       0.91      0.99      0.95       289
          3       0.92      0.98      0.95        82
          5       0.97      0.96      0.97       475
          6       0.74      0.56      0.64       138
          7       0.81      0.66      0.73       169

avg / total       0.86      0.86      0.86      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       348
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       315
          3       1.00      1.00      1.00        87
          5       1.00      1.00      1.00       472
          6       1.00      1.00      1.00       104
          7       1.00      1.00      1.00       138

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07510336 -0.09373388  0.03575075 -0.01425091 -0.1381939   0.03944171
 -0.04709999 -0.00431261]
Epoch number and batch_no:  158 0
Loss before optimizing :  0.237844947564
Loss, accuracy and verification results :  0.237844947564 0.933266932271 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.90      0.78      0.83       152
          1       1.00      0.75      0.86         4
          2       0.97      0.97      0.97       224
          3       0.95      1.00      0.97        70
          5       0.98      1.00      0.99       381
          6       0.88      0.77      0.82        82
          7       0.75      0.93      0.83        91

avg / total       0.94      0.93      0.93      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       131
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        74
          5       1.00      1.00      1.00       387
          6       1.00      1.00      1.00        72
          7       1.00      1.00      1.00       113

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07511489 -0.09352869  0.03573163 -0.01409942 -0.13819394  0.03947622
 -0.04723849 -0.00443195]
Epoch number and batch_no:  158 1
Loss before optimizing :  0.357492435389
Loss, accuracy and verification results :  0.357492435389 0.880871341048 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.80      0.79      0.79       305
          1       1.00      0.45      0.62        11
          2       0.95      0.98      0.96       289
          3       0.89      0.99      0.94        82
          5       0.94      0.98      0.96       475
          6       0.80      0.62      0.70       138
          7       0.77      0.78      0.78       169

avg / total       0.88      0.88      0.88      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       299
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       300
          3       1.00      1.00      1.00        91
          5       1.00      1.00      1.00       497
          6       1.00      1.00      1.00       106
          7       1.00      1.00      1.00       171

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07543957 -0.09319304  0.03573961 -0.01427266 -0.13819399  0.03930802
 -0.04701598 -0.00472851]
Epoch number and batch_no:  159 0
Loss before optimizing :  0.2213236257
Loss, accuracy and verification results :  0.2213236257 0.938247011952 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.88      0.84      0.86       152
          1       1.00      1.00      1.00         4
          2       0.97      0.99      0.98       224
          3       0.97      1.00      0.99        70
          5       0.98      0.99      0.99       381
          6       0.89      0.76      0.82        82
          7       0.80      0.86      0.83        91

avg / total       0.94      0.94      0.94      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       146
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       229
          3       1.00      1.00      1.00        72
          5       1.00      1.00      1.00       386
          6       1.00      1.00      1.00        70
          7       1.00      1.00      1.00        97

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07567296 -0.09306325  0.03581595 -0.01454791 -0.13819403  0.03917151
 -0.04683375 -0.00492199]
Epoch number and batch_no:  159 1
Loss before optimizing :  0.331435653689
Loss, accuracy and verification results :  0.331435653689 0.890401633764 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.77      0.85      0.81       305
          1       0.89      0.73      0.80        11
          2       0.96      0.97      0.96       289
          3       0.91      0.99      0.95        82
          5       0.98      0.98      0.98       475
          6       0.78      0.67      0.72       138
          7       0.82      0.74      0.78       169

avg / total       0.89      0.89      0.89      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       335
          1       1.00      1.00      1.00         9
          2       1.00      1.00      1.00       291
          3       1.00      1.00      1.00        89
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       118
          7       1.00      1.00      1.00       152

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07561834 -0.0930903   0.03585927 -0.01493756 -0.13819407  0.03919169
 -0.04686465 -0.0046589 ]
Epoch number and batch_no:  160 0
Loss before optimizing :  0.218281127763
Loss, accuracy and verification results :  0.218281127763 0.940239043825 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.87      0.88      0.87       152
          1       1.00      1.00      1.00         4
          2       0.97      0.99      0.98       224
          3       0.96      1.00      0.98        70
          5       0.99      0.99      0.99       381
          6       0.93      0.78      0.85        82
          7       0.79      0.81      0.80        91

avg / total       0.94      0.94      0.94      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       153
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       228
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       383
          6       1.00      1.00      1.00        69
          7       1.00      1.00      1.00        94

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07546321 -0.09328241  0.03588111 -0.01512734 -0.13819411  0.03928841
 -0.04698683 -0.00439573]
Epoch number and batch_no:  160 1
Loss before optimizing :  0.322619668742
Loss, accuracy and verification results :  0.322619668742 0.893805309735 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.83      0.79      0.81       305
          1       0.70      0.64      0.67        11
          2       0.97      0.97      0.97       289
          3       0.94      0.96      0.95        82
          5       0.97      0.99      0.98       475
          6       0.77      0.64      0.70       138
          7       0.76      0.86      0.81       169

avg / total       0.89      0.89      0.89      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       293
          1       1.00      1.00      1.00        10
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        84
          5       1.00      1.00      1.00       488
          6       1.00      1.00      1.00       114
          7       1.00      1.00      1.00       191

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07565171 -0.09351425  0.03600634 -0.01519978 -0.13819414  0.03930389
 -0.04704649 -0.00483277]
Epoch number and batch_no:  161 0
Loss before optimizing :  0.204165226138
Loss, accuracy and verification results :  0.204165226138 0.94422310757 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.84      0.89      0.87       152
          1       1.00      0.75      0.86         4
          2       0.99      1.00      0.99       224
          3       0.96      1.00      0.98        70
          5       0.98      1.00      0.99       381
          6       0.91      0.78      0.84        82
          7       0.86      0.80      0.83        91

avg / total       0.94      0.94      0.94      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       160
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       226
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       387
          6       1.00      1.00      1.00        70
          7       1.00      1.00      1.00        85

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0756777  -0.09368138  0.03613846 -0.01524741 -0.13819418  0.03928916
 -0.04700791 -0.00505982]
Epoch number and batch_no:  161 1
Loss before optimizing :  0.310436085595
Loss, accuracy and verification results :  0.310436085595 0.898570456093 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.80      0.85      0.82       305
          1       0.83      0.45      0.59        11
          2       0.96      0.99      0.98       289
          3       0.97      0.95      0.96        82
          5       0.97      1.00      0.98       475
          6       0.79      0.63      0.70       138
          7       0.82      0.79      0.80       169

avg / total       0.90      0.90      0.90      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       323
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       297
          3       1.00      1.00      1.00        80
          5       1.00      1.00      1.00       490
          6       1.00      1.00      1.00       110
          7       1.00      1.00      1.00       163

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07562596 -0.09377954  0.03621993 -0.01507687 -0.13819421  0.03923564
 -0.04706564 -0.00502434]
Epoch number and batch_no:  162 0
Loss before optimizing :  0.195223869253
Loss, accuracy and verification results :  0.195223869253 0.955179282869 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.89      0.86      0.88       152
          1       1.00      0.75      0.86         4
          2       0.97      1.00      0.99       224
          3       0.99      1.00      0.99        70
          5       0.99      1.00      1.00       381
          6       0.87      0.83      0.85        82
          7       0.89      0.90      0.90        91

avg / total       0.95      0.96      0.95      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       147
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       230
          3       1.00      1.00      1.00        71
          5       1.00      1.00      1.00       383
          6       1.00      1.00      1.00        78
          7       1.00      1.00      1.00        92

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07558965 -0.09377747  0.03618143 -0.01496085 -0.13819425  0.0392199
 -0.04717509 -0.00481399]
Epoch number and batch_no:  162 1
Loss before optimizing :  0.310026085781
Loss, accuracy and verification results :  0.310026085781 0.895847515317 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.81      0.82      0.81       305
          1       1.00      0.55      0.71        11
          2       0.94      0.98      0.96       289
          3       0.93      0.98      0.95        82
          5       0.98      0.98      0.98       475
          6       0.78      0.62      0.69       138
          7       0.80      0.86      0.83       169

avg / total       0.89      0.90      0.89      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       309
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       301
          3       1.00      1.00      1.00        86
          5       1.00      1.00      1.00       477
          6       1.00      1.00      1.00       109
          7       1.00      1.00      1.00       181

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07559357 -0.09355547  0.03617929 -0.01498981 -0.13819428  0.0392589
 -0.04709026 -0.00506392]
Epoch number and batch_no:  163 0
Loss before optimizing :  0.198570652451
Loss, accuracy and verification results :  0.198570652451 0.953187250996 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.93      0.85      0.89       152
          1       1.00      1.00      1.00         4
          2       0.97      0.99      0.98       224
          3       0.96      1.00      0.98        70
          5       0.99      1.00      0.99       381
          6       0.92      0.82      0.86        82
          7       0.84      0.93      0.89        91

avg / total       0.95      0.95      0.95      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       139
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       228
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       386
          6       1.00      1.00      1.00        73
          7       1.00      1.00      1.00       101

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07578426 -0.09337128  0.0361669  -0.01515669 -0.13819431  0.03924286
 -0.04703859 -0.00530802]
Epoch number and batch_no:  163 1
Loss before optimizing :  0.301362799776
Loss, accuracy and verification results :  0.301362799776 0.898570456093 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.78      0.86      0.82       305
          1       1.00      0.73      0.84        11
          2       0.98      0.98      0.98       289
          3       0.92      0.98      0.95        82
          5       0.97      0.99      0.98       475
          6       0.79      0.64      0.71       138
          7       0.85      0.77      0.81       169

avg / total       0.90      0.90      0.90      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       338
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       288
          3       1.00      1.00      1.00        87
          5       1.00      1.00      1.00       482
          6       1.00      1.00      1.00       113
          7       1.00      1.00      1.00       153

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07575888 -0.09342738  0.03622863 -0.01545158 -0.13819434  0.03922975
 -0.04712241 -0.00504534]
Epoch number and batch_no:  164 0
Loss before optimizing :  0.201545796392
^[[ALoss, accuracy and verification results :  0.201545796392 0.951195219124 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.89      0.87      0.88       152
          1       1.00      1.00      1.00         4
          2       0.99      0.99      0.99       224
          3       0.95      1.00      0.97        70
          5       0.98      1.00      0.99       381
          6       0.92      0.80      0.86        82
          7       0.86      0.89      0.88        91

avg / total       0.95      0.95      0.95      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       148
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       225
          3       1.00      1.00      1.00        74
          5       1.00      1.00      1.00       387
          6       1.00      1.00      1.00        72
          7       1.00      1.00      1.00        94

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0757003  -0.09353981  0.0363819  -0.01568248 -0.13819437  0.03911198
 -0.0471182  -0.00473543]
Epoch number and batch_no:  164 1
Loss before optimizing :  0.295003545473
Loss, accuracy and verification results :  0.295003545473 0.905377808033 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.82      0.81      0.82       305
          1       0.80      0.73      0.76        11
          2       0.95      0.99      0.97       289
          3       0.97      0.94      0.96        82
          5       0.97      0.99      0.98       475
          6       0.85      0.72      0.78       138
          7       0.80      0.85      0.83       169

avg / total       0.90      0.91      0.90      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       302
          1       1.00      1.00      1.00        10
          2       1.00      1.00      1.00       300
          3       1.00      1.00      1.00        79
          5       1.00      1.00      1.00       483
          6       1.00      1.00      1.00       116
          7       1.00      1.00      1.00       179

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07597452 -0.09384897  0.03646593 -0.01549984 -0.1381944   0.03893506
 -0.04717309 -0.00484694]
Epoch number and batch_no:  165 0
Loss before optimizing :  0.194984882684
Loss, accuracy and verification results :  0.194984882684 0.96015936255 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.90      0.91      0.91       152
          1       1.00      1.00      1.00         4
          2       0.99      1.00      0.99       224
          3       0.99      0.96      0.97        70
          5       0.99      0.99      0.99       381
          6       0.92      0.83      0.87        82
          7       0.87      0.95      0.91        91

avg / total       0.96      0.96      0.96      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       154
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       226
          3       1.00      1.00      1.00        68
          5       1.00      1.00      1.00       379
          6       1.00      1.00      1.00        74
          7       1.00      1.00      1.00        99

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07596442 -0.09424644  0.03657031 -0.01484683 -0.13819443  0.03890209
 -0.04731481 -0.00515056]
Epoch number and batch_no:  165 1
Loss before optimizing :  0.288524858714
Loss, accuracy and verification results :  0.288524858714 0.897889720899 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.81      0.84      0.83       305
          1       1.00      0.45      0.62        11
          2       0.94      0.99      0.96       289
          3       0.94      0.98      0.96        82
          5       0.96      0.99      0.98       475
          6       0.78      0.67      0.72       138
          7       0.85      0.76      0.80       169

avg / total       0.90      0.90      0.90      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       316
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       303
          3       1.00      1.00      1.00        85
          5       1.00      1.00      1.00       488
          6       1.00      1.00      1.00       120
          7       1.00      1.00      1.00       152

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07595468 -0.09429926  0.03650203 -0.01436861 -0.13819446  0.03882138
 -0.04741487 -0.00498919]
Epoch number and batch_no:  166 0
Loss before optimizing :  0.188123855506
Loss, accuracy and verification results :  0.188123855506 0.949203187251 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.90      0.86      0.88       152
          1       1.00      0.50      0.67         4
          2       0.99      0.99      0.99       224
          3       0.96      1.00      0.98        70
          5       0.99      0.99      0.99       381
          6       0.94      0.80      0.87        82
          7       0.80      0.93      0.86        91

avg / total       0.95      0.95      0.95      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       145
          1       1.00      1.00      1.00         2
          2       1.00      1.00      1.00       225
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       383
          6       1.00      1.00      1.00        70
          7       1.00      1.00      1.00       106

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07598164 -0.09409081  0.03654099 -0.01428167 -0.13819448  0.03873879
 -0.0472943  -0.0051334 ]
Epoch number and batch_no:  166 1
Loss before optimizing :  0.27204338593
Loss, accuracy and verification results :  0.27204338593 0.921715452689 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.85      0.85      0.85       305
          1       1.00      0.36      0.53        11
          2       0.98      0.99      0.98       289
          3       0.88      1.00      0.94        82
          5       0.99      0.99      0.99       475
          6       0.83      0.72      0.77       138
          7       0.85      0.92      0.88       169

avg / total       0.92      0.92      0.92      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       304
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       290
          3       1.00      1.00      1.00        93
          5       1.00      1.00      1.00       476
          6       1.00      1.00      1.00       119
          7       1.00      1.00      1.00       183

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0760698  -0.09364927  0.03666586 -0.01490752 -0.13819451  0.0387459
 -0.04713428 -0.00542126]
Epoch number and batch_no:  167 0
Loss before optimizing :  0.170862120625
Loss, accuracy and verification results :  0.170862120625 0.966135458167 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.90      0.94      0.92       152
          1       1.00      1.00      1.00         4
          2       0.99      1.00      0.99       224
          3       0.97      1.00      0.99        70
          5       1.00      1.00      1.00       381
          6       0.92      0.85      0.89        82
          7       0.93      0.87      0.90        91

avg / total       0.97      0.97      0.97      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       159
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       227
          3       1.00      1.00      1.00        72
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        76
          7       1.00      1.00      1.00        85

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07606023 -0.09338539  0.03666197 -0.01549375 -0.13819454  0.03880079
 -0.04712144 -0.00523362]
Epoch number and batch_no:  167 1
Loss before optimizing :  0.262194741503
Loss, accuracy and verification results :  0.262194741503 0.929203539823 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.86      0.87      0.87       305
          1       0.67      0.91      0.77        11
          2       0.98      0.99      0.98       289
          3       0.99      0.98      0.98        82
          5       0.98      0.99      0.99       475
          6       0.86      0.77      0.81       138
          7       0.88      0.85      0.87       169

avg / total       0.93      0.93      0.93      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       310
          1       1.00      1.00      1.00        15
          2       1.00      1.00      1.00       294
          3       1.00      1.00      1.00        81
          5       1.00      1.00      1.00       483
          6       1.00      1.00      1.00       123
          7       1.00      1.00      1.00       163

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07603347 -0.09372967  0.03659783 -0.01564568 -0.13819456  0.03885863
 -0.0473895  -0.00473016]
Epoch number and batch_no:  168 0
Loss before optimizing :  0.161847227308
Loss, accuracy and verification results :  0.161847227308 0.964143426295 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.93      0.89      0.91       152
          1       0.80      1.00      0.89         4
          2       1.00      0.99      0.99       224
          3       0.97      1.00      0.99        70
          5       0.99      1.00      1.00       381
          6       0.92      0.85      0.89        82
          7       0.87      0.95      0.91        91

avg / total       0.96      0.96      0.96      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       147
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       223
          3       1.00      1.00      1.00        72
          5       1.00      1.00      1.00       382
          6       1.00      1.00      1.00        76
          7       1.00      1.00      1.00        99

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07611789 -0.09414636  0.03658433 -0.01562039 -0.13819459  0.03890542
 -0.0475633  -0.00468415]
Epoch number and batch_no:  168 1
Loss before optimizing :  0.262473426745
Loss, accuracy and verification results :  0.262473426745 0.920353982301 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.82      0.90      0.86       305
          1       1.00      0.55      0.71        11
          2       0.99      0.96      0.97       289
          3       0.99      0.98      0.98        82
          5       0.99      0.99      0.99       475
          6       0.93      0.65      0.77       138
          7       0.79      0.89      0.84       169

avg / total       0.93      0.92      0.92      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       336
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       280
          3       1.00      1.00      1.00        81
          5       1.00      1.00      1.00       479
          6       1.00      1.00      1.00        97
          7       1.00      1.00      1.00       190

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07601706 -0.09420459  0.03678745 -0.01558861 -0.13819461  0.03891734
 -0.04732432 -0.0052102 ]
Epoch number and batch_no:  169 0
Loss before optimizing :  0.153342332882
Loss, accuracy and verification results :  0.153342332882 0.96812749004 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.92      0.93      0.92       152
          1       1.00      1.00      1.00         4
          2       0.99      1.00      1.00       224
          3       0.99      1.00      0.99        70
          5       0.99      1.00      0.99       381
          6       0.90      0.90      0.90        82
          7       0.96      0.86      0.91        91

avg / total       0.97      0.97      0.97      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       154
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       226
          3       1.00      1.00      1.00        71
          5       1.00      1.00      1.00       386
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        81

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07607144 -0.0942195   0.03684912 -0.01550001 -0.13819464  0.03882201
 -0.04744842 -0.00509883]
Epoch number and batch_no:  169 1
Loss before optimizing :  0.258649889344
Loss, accuracy and verification results :  0.258649889344 0.918311776719 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.85      0.87      0.86       305
          1       0.86      0.55      0.67        11
          2       0.96      1.00      0.98       289
          3       0.97      0.93      0.95        82
          5       0.98      0.99      0.98       475
          6       0.81      0.79      0.80       138
          7       0.88      0.79      0.83       169

avg / total       0.92      0.92      0.92      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       314
          1       1.00      1.00      1.00         7
          2       1.00      1.00      1.00       301
          3       1.00      1.00      1.00        78
          5       1.00      1.00      1.00       482
          6       1.00      1.00      1.00       135
          7       1.00      1.00      1.00       152

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07609909 -0.09408459  0.03675816 -0.01532526 -0.13819466  0.038689
 -0.04800525 -0.0042063 ]
Epoch number and batch_no:  170 0
Loss before optimizing :  0.148316006442
Loss, accuracy and verification results :  0.148316006442 0.969123505976 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.92      0.93      0.93       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      0.99      1.00       381
          6       0.98      0.79      0.88        82
          7       0.85      0.99      0.91        91

avg / total       0.97      0.97      0.97      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       155
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       379
          6       1.00      1.00      1.00        66
          7       1.00      1.00      1.00       106

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07606769 -0.09390225  0.03667124 -0.01510534 -0.13819468  0.03866979
 -0.04800906 -0.00413236]
Epoch number and batch_no:  170 1
Loss before optimizing :  0.263556051201
Loss, accuracy and verification results :  0.263556051201 0.918311776719 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.86      0.85      0.85       305
          1       0.80      0.73      0.76        11
          2       0.98      0.98      0.98       289
          3       0.95      0.95      0.95        82
          5       0.99      0.99      0.99       475
          6       0.91      0.66      0.76       138
          7       0.77      0.96      0.86       169

avg / total       0.92      0.92      0.92      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       302
          1       1.00      1.00      1.00        10
          2       1.00      1.00      1.00       288
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       100
          7       1.00      1.00      1.00       212

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0759958  -0.09372861  0.03674534 -0.01491428 -0.13819471  0.03877821
 -0.04712324 -0.00552766]
Epoch number and batch_no:  171 0
Loss before optimizing :  0.144000443571
Loss, accuracy and verification results :  0.144000443571 0.981075697211 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.98      0.93      0.96       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       0.95      1.00      0.97        70
          5       1.00      1.00      1.00       381
          6       0.90      0.95      0.92        82
          7       0.99      0.96      0.97        91

avg / total       0.98      0.98      0.98      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       145
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        74
          5       1.00      1.00      1.00       382
          6       1.00      1.00      1.00        87
          7       1.00      1.00      1.00        88

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0761322  -0.09367707  0.03678318 -0.01488122 -0.13819473  0.03878852
 -0.04686872 -0.00615243]
Epoch number and batch_no:  171 1
Loss before optimizing :  0.275640371728
Loss, accuracy and verification results :  0.275640371728 0.900612661675 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.85      0.85      0.85       305
          1       0.69      0.82      0.75        11
          2       0.93      0.98      0.96       289
          3       0.94      0.99      0.96        82
          5       0.95      1.00      0.97       475
          6       0.73      0.84      0.78       138
          7       0.97      0.60      0.74       169

avg / total       0.91      0.90      0.90      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       302
          1       1.00      1.00      1.00        13
          2       1.00      1.00      1.00       304
          3       1.00      1.00      1.00        86
          5       1.00      1.00      1.00       500
          6       1.00      1.00      1.00       159
          7       1.00      1.00      1.00       105

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07644006 -0.09399052  0.03665042 -0.01497997 -0.13819475  0.03854271
 -0.04774136 -0.00468555]
Epoch number and batch_no:  172 0
Loss before optimizing :  0.138873331724
Loss, accuracy and verification results :  0.138873331724 0.977091633466 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.93      0.98      0.95       152
          1       1.00      1.00      1.00         4
          2       1.00      0.99      0.99       224
          3       0.97      1.00      0.99        70
          5       1.00      1.00      1.00       381
          6       0.96      0.82      0.88        82
          7       0.94      0.98      0.96        91

avg / total       0.98      0.98      0.98      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       161
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       221
          3       1.00      1.00      1.00        72
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        70
          7       1.00      1.00      1.00        95

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07640034 -0.09423033  0.03672759 -0.01515452 -0.13819477  0.03838273
 -0.04813118 -0.00375247]
Epoch number and batch_no:  172 1
Loss before optimizing :  0.269625563823
Loss, accuracy and verification results :  0.269625563823 0.905377808033 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.84      0.84      0.84       305
          1       0.75      0.27      0.40        11
          2       0.99      0.95      0.97       289
          3       0.93      0.99      0.96        82
          5       0.99      0.98      0.99       475
          6       0.93      0.64      0.76       138
          7       0.70      0.97      0.81       169

avg / total       0.92      0.91      0.90      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       303
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       278
          3       1.00      1.00      1.00        87
          5       1.00      1.00      1.00       467
          6       1.00      1.00      1.00        95
          7       1.00      1.00      1.00       235

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07643341 -0.09406919  0.03707009 -0.01545192 -0.13819479  0.03850339
 -0.04755482 -0.00524038]
Epoch number and batch_no:  173 0
Loss before optimizing :  0.130782663228
^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OSLoss, accuracy and verification results :  0.130782663228 0.985059760956 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.95      0.97      0.96       152
          1       0.80      1.00      0.89         4
          2       0.99      1.00      0.99       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.99      0.89      0.94        82
          7       0.97      0.99      0.98        91

avg / total       0.99      0.99      0.98      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       154
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       227
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        74
          7       1.00      1.00      1.00        93

avg / total       1.00      1.00      1.00      1004

^[OSself.biases :  [ 0.07638215 -0.09396164  0.03727875 -0.01557712 -0.1381948   0.03864342
 -0.04691352 -0.00653407]
Epoch number and batch_no:  173 1
^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OSLoss before optimizing :  0.261728158429
^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[OS^[[1;3SLoss, accuracy and verification results :  0.261728158429 0.908100748809 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.84      0.86      0.85       305
          1       0.67      0.55      0.60        11
          2       0.92      0.99      0.95       289
          3       0.99      0.99      0.99        82
          5       0.98      1.00      0.99       475
          6       0.75      0.86      0.80       138
          7       0.96      0.63      0.76       169

avg / total       0.91      0.91      0.91      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       312
          1       1.00      1.00      1.00         9
          2       1.00      1.00      1.00       312
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       485
          6       1.00      1.00      1.00       157
          7       1.00      1.00      1.00       112

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07639317 -0.09391203  0.03712879 -0.01546038 -0.13819482  0.03861907
 -0.04742032 -0.00579229]
Epoch number and batch_no:  174 0
Loss before optimizing :  0.129113875719
Loss, accuracy and verification results :  0.129113875719 0.985059760956 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.97      0.95      0.96       152
          1       0.80      1.00      0.89         4
          2       1.00      1.00      1.00       224
          3       0.99      1.00      0.99        70
          5       0.98      1.00      0.99       381
          6       0.99      0.96      0.98        82
          7       0.99      0.95      0.97        91

avg / total       0.99      0.99      0.98      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       149
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       225
          3       1.00      1.00      1.00        71
          5       1.00      1.00      1.00       387
          6       1.00      1.00      1.00        80
          7       1.00      1.00      1.00        87

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07652325 -0.09392646  0.03693497 -0.01526614 -0.13819484  0.03847394
 -0.0479668  -0.00481994]
Epoch number and batch_no:  174 1
Loss before optimizing :  0.23356417114
Loss, accuracy and verification results :  0.23356417114 0.926480599047 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.87      0.87      0.87       305
          1       0.83      0.45      0.59        11
          2       0.97      0.99      0.98       289
          3       0.95      0.99      0.97        82
          5       0.99      1.00      0.99       475
          6       0.94      0.64      0.76       138
          7       0.79      0.96      0.87       169

avg / total       0.93      0.93      0.92      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       303
          1       1.00      1.00      1.00         6
          2       1.00      1.00      1.00       294
          3       1.00      1.00      1.00        85
          5       1.00      1.00      1.00       479
          6       1.00      1.00      1.00        95
          7       1.00      1.00      1.00       207

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07665184 -0.09366125  0.03679061 -0.01514966 -0.13819486  0.03835348
 -0.04743454 -0.00508407]
Epoch number and batch_no:  175 0
Loss before optimizing :  0.135720230153
Loss, accuracy and verification results :  0.135720230153 0.982071713147 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.95      0.96      0.95       152
          1       1.00      1.00      1.00         4
          2       1.00      0.99      0.99       224
          3       0.99      1.00      0.99        70
          5       1.00      0.99      1.00       381
          6       0.99      0.95      0.97        82
          7       0.92      0.98      0.95        91

avg / total       0.98      0.98      0.98      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       154
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       221
          3       1.00      1.00      1.00        71
          5       1.00      1.00      1.00       378
          6       1.00      1.00      1.00        79
          7       1.00      1.00      1.00        97

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07655846 -0.09344668  0.03686442 -0.01487594 -0.13819487  0.03839307
 -0.04704836 -0.00581898]
Epoch number and batch_no:  175 1
Loss before optimizing :  0.205139928123
Loss, accuracy and verification results :  0.205139928123 0.941456773315 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.90      0.88      0.89       305
          1       0.75      0.82      0.78        11
          2       0.98      0.99      0.99       289
          3       0.98      1.00      0.99        82
          5       0.99      0.99      0.99       475
          6       0.81      0.85      0.83       138
          7       0.91      0.88      0.90       169

avg / total       0.94      0.94      0.94      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       297
          1       1.00      1.00      1.00        12
          2       1.00      1.00      1.00       293
          3       1.00      1.00      1.00        84
          5       1.00      1.00      1.00       476
          6       1.00      1.00      1.00       144
          7       1.00      1.00      1.00       163

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07662853 -0.09348744  0.03693909 -0.01483399 -0.13819489  0.03848965
 -0.04728401 -0.00613934]
Epoch number and batch_no:  176 0
Loss before optimizing :  0.122603157197
Loss, accuracy and verification results :  0.122603157197 0.987051792829 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.98      0.97      0.97       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       0.99      1.00      0.99        70
          5       0.99      1.00      1.00       381
          6       0.96      0.98      0.97        82
          7       0.98      0.93      0.96        91

avg / total       0.99      0.99      0.99      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       150
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       225
          3       1.00      1.00      1.00        71
          5       1.00      1.00      1.00       384
          6       1.00      1.00      1.00        83
          7       1.00      1.00      1.00        87

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0767527  -0.09372657  0.03695233 -0.0147455  -0.1381949   0.03851263
 -0.04780599 -0.00591557]
Epoch number and batch_no:  176 1
Loss before optimizing :  0.204682835155
Loss, accuracy and verification results :  0.204682835155 0.940095302927 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.86      0.91      0.89       305
          1       0.89      0.73      0.80        11
          2       0.98      1.00      0.99       289
          3       0.96      0.99      0.98        82
          5       0.97      1.00      0.98       475
          6       0.91      0.76      0.83       138
          7       0.94      0.86      0.90       169

avg / total       0.94      0.94      0.94      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       323
          1       1.00      1.00      1.00         9
          2       1.00      1.00      1.00       295
          3       1.00      1.00      1.00        84
          5       1.00      1.00      1.00       489
          6       1.00      1.00      1.00       115
          7       1.00      1.00      1.00       154

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07682549 -0.09406597  0.03686698 -0.01473907 -0.13819492  0.03837221
 -0.04796285 -0.00522116]
Epoch number and batch_no:  177 0
Loss before optimizing :  0.118394022752
Loss, accuracy and verification results :  0.118394022752 0.98406374502 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.97      0.97      0.97       152
          1       1.00      1.00      1.00         4
          2       0.99      0.99      0.99       224
          3       0.96      1.00      0.98        70
          5       1.00      1.00      1.00       381
          6       0.99      0.90      0.94        82
          7       0.95      0.99      0.97        91

avg / total       0.98      0.98      0.98      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       151
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        73
          5       1.00      1.00      1.00       382
          6       1.00      1.00      1.00        75
          7       1.00      1.00      1.00        95

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07693058 -0.09431764  0.03682456 -0.01491452 -0.13819493  0.03824171
 -0.04772251 -0.00497625]
Epoch number and batch_no:  177 1
Loss before optimizing :  0.199268453217
Loss, accuracy and verification results :  0.199268453217 0.943498978897 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.91      0.86      0.89       305
          1       1.00      0.64      0.78        11
          2       0.99      0.98      0.98       289
          3       0.95      1.00      0.98        82
          5       1.00      0.99      0.99       475
          6       0.86      0.85      0.85       138
          7       0.84      0.96      0.90       169

avg / total       0.94      0.94      0.94      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       288
          1       1.00      1.00      1.00         7
          2       1.00      1.00      1.00       286
          3       1.00      1.00      1.00        86
          5       1.00      1.00      1.00       473
          6       1.00      1.00      1.00       136
          7       1.00      1.00      1.00       193

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07729883 -0.09425978  0.03695012 -0.01522275 -0.13819495  0.03825502
 -0.04771663 -0.00567786]
Epoch number and batch_no:  178 0
Loss before optimizing :  0.117468838861
Loss, accuracy and verification results :  0.117468838861 0.983067729084 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.93      0.98      0.96       152
          1       1.00      0.75      0.86         4
          2       1.00      1.00      1.00       224
          3       0.99      1.00      0.99        70
          5       1.00      0.99      1.00       381
          6       0.96      0.95      0.96        82
          7       0.98      0.93      0.96        91

avg / total       0.98      0.98      0.98      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       160
          1       1.00      1.00      1.00         3
          2       1.00      1.00      1.00       223
          3       1.00      1.00      1.00        71
          5       1.00      1.00      1.00       379
          6       1.00      1.00      1.00        81
          7       1.00      1.00      1.00        87

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07731376 -0.09410719  0.0370865  -0.01544955 -0.13819496  0.03832878
 -0.04774853 -0.00604183]
Epoch number and batch_no:  178 1
Loss before optimizing :  0.183752225157
Loss, accuracy and verification results :  0.183752225157 0.954390742001 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.88      0.94      0.91       305
          1       1.00      0.73      0.84        11
          2       0.98      1.00      0.99       289
          3       1.00      0.96      0.98        82
          5       0.99      1.00      0.99       475
          6       0.93      0.83      0.88       138
          7       0.96      0.88      0.92       169

avg / total       0.96      0.95      0.95      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       328
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       295
          3       1.00      1.00      1.00        79
          5       1.00      1.00      1.00       481
          6       1.00      1.00      1.00       122
          7       1.00      1.00      1.00       156

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0769728  -0.09388863  0.03706152 -0.01539445 -0.13819497  0.03832677
 -0.04777485 -0.00551501]
Epoch number and batch_no:  179 0
Loss before optimizing :  0.113963203666
Loss, accuracy and verification results :  0.113963203666 0.980079681275 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.96      0.94      0.95       152
          1       1.00      1.00      1.00         4
          2       0.99      0.99      0.99       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.96      0.91      0.94        82
          7       0.93      0.98      0.95        91

avg / total       0.98      0.98      0.98      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       149
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       225
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       382
          6       1.00      1.00      1.00        78
          7       1.00      1.00      1.00        96

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07689813 -0.0937226   0.0369915  -0.01538746 -0.13819499  0.03827004
 -0.04768543 -0.00524151]
Epoch number and batch_no:  179 1
Loss before optimizing :  0.177665321931
Loss, accuracy and verification results :  0.177665321931 0.958475153165 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.95      0.89      0.92       305
          1       0.83      0.91      0.87        11
          2       0.99      0.99      0.99       289
          3       1.00      0.98      0.99        82
          5       0.99      1.00      0.99       475
          6       0.87      0.88      0.88       138
          7       0.90      0.96      0.93       169

avg / total       0.96      0.96      0.96      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       285
          1       1.00      1.00      1.00        12
          2       1.00      1.00      1.00       291
          3       1.00      1.00      1.00        80
          5       1.00      1.00      1.00       480
          6       1.00      1.00      1.00       140
          7       1.00      1.00      1.00       181

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07739896 -0.09373504  0.03696529 -0.01513677 -0.138195    0.03810801
 -0.04803762 -0.00539555]
Epoch number and batch_no:  180 0
Loss before optimizing :  0.100915689035
Loss, accuracy and verification results :  0.100915689035 0.989043824701 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.96      0.99      0.97       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.99      0.91      0.95        82
          7       0.97      0.99      0.98        91

avg / total       0.99      0.99      0.99      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       156
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        76
          7       1.00      1.00      1.00        93

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07768196 -0.09379933  0.03702349 -0.01491047 -0.13819501  0.03796629
 -0.04811927 -0.00564523]
Epoch number and batch_no:  180 1
Loss before optimizing :  0.17182034886
Loss, accuracy and verification results :  0.17182034886 0.957113682777 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.88      0.96      0.92       305
          1       1.00      0.82      0.90        11
          2       0.99      0.99      0.99       289
          3       0.95      1.00      0.98        82
          5       1.00      0.99      0.99       475
          6       0.95      0.80      0.87       138
          7       0.95      0.91      0.93       169

avg / total       0.96      0.96      0.96      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       334
          1       1.00      1.00      1.00         9
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        86
          5       1.00      1.00      1.00       473
          6       1.00      1.00      1.00       117
          7       1.00      1.00      1.00       161

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07734111 -0.09392695  0.03708564 -0.01484904 -0.13819502  0.03790917
 -0.04778915 -0.00537599]
Epoch number and batch_no:  181 0
Loss before optimizing :  0.0949829639818
Loss, accuracy and verification results :  0.0949829639818 0.99203187251 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.99      0.97      0.98       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.96      0.96      0.96        82
          7       0.98      0.99      0.98        91

avg / total       0.99      0.99      0.99      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       150
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       225
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        92

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07714438 -0.0940702   0.03706692 -0.01477454 -0.13819503  0.03792895
 -0.04778515 -0.00508038]
Epoch number and batch_no:  181 1
Loss before optimizing :  0.163910661998
Loss, accuracy and verification results :  0.163910661998 0.960517358747 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.95      0.89      0.92       305
          1       0.88      0.64      0.74        11
          2       0.99      0.99      0.99       289
          3       0.96      1.00      0.98        82
          5       1.00      1.00      1.00       475
          6       0.87      0.91      0.89       138
          7       0.91      0.98      0.94       169

avg / total       0.96      0.96      0.96      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       285
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       288
          3       1.00      1.00      1.00        85
          5       1.00      1.00      1.00       477
          6       1.00      1.00      1.00       144
          7       1.00      1.00      1.00       182

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07753372 -0.09416072  0.03707826 -0.01497337 -0.13819505  0.03798992
 -0.04830388 -0.00524179]
Epoch number and batch_no:  182 0
Loss before optimizing :  0.0900709074156
Loss, accuracy and verification results :  0.0900709074156 0.994023904382 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.97      0.99      0.98       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.99      0.95      0.97        82
          7       1.00      0.99      0.99        91

avg / total       0.99      0.99      0.99      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       156
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        79
          7       1.00      1.00      1.00        90

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07776004 -0.0942106   0.03710541 -0.01503308 -0.13819506  0.03802479
 -0.04861885 -0.00540217]
Epoch number and batch_no:  182 1
Loss before optimizing :  0.160013786768
Loss, accuracy and verification results :  0.160013786768 0.964601769912 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.88      0.99      0.93       305
          1       1.00      0.73      0.84        11
          2       1.00      1.00      1.00       289
          3       0.98      1.00      0.99        82
          5       0.99      1.00      1.00       475
          6       0.98      0.78      0.87       138
          7       0.97      0.92      0.95       169

avg / total       0.97      0.96      0.96      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       341
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       288
          3       1.00      1.00      1.00        84
          5       1.00      1.00      1.00       477
          6       1.00      1.00      1.00       110
          7       1.00      1.00      1.00       161

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07733297 -0.09422259  0.03721699 -0.01510516 -0.13819507  0.03800447
 -0.04815189 -0.00526847]
Epoch number and batch_no:  183 0
Loss before optimizing :  0.0940013322612
Loss, accuracy and verification results :  0.0940013322612 0.990039840637 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.99      0.96      0.97       152
          1       1.00      1.00      1.00         4
          2       0.99      1.00      0.99       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.96      0.98      0.97        82
          7       0.99      0.99      0.99        91

avg / total       0.99      0.99      0.99      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       148
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       226
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       382
          6       1.00      1.00      1.00        83
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0772151  -0.09429333  0.03722926 -0.01513877 -0.13819508  0.03791333
 -0.04790942 -0.0050384 ]
Epoch number and batch_no:  183 1
Loss before optimizing :  0.156487176363
Loss, accuracy and verification results :  0.156487176363 0.970728386658 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.99      0.90      0.94       305
          1       0.85      1.00      0.92        11
          2       0.99      1.00      0.99       289
          3       0.99      0.99      0.99        82
          5       1.00      1.00      1.00       475
          6       0.89      0.96      0.93       138
          7       0.92      0.98      0.95       169

avg / total       0.97      0.97      0.97      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       278
          1       1.00      1.00      1.00        13
          2       1.00      1.00      1.00       292
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       476
          6       1.00      1.00      1.00       149
          7       1.00      1.00      1.00       179

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07779051 -0.09443524  0.03720169 -0.01526077 -0.13819509  0.03784455
 -0.04838633 -0.00514781]
Epoch number and batch_no:  184 0
Loss before optimizing :  0.0867407886077
Loss, accuracy and verification results :  0.0867407886077 0.995019920319 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.98      0.99      0.99       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.99      0.98      0.98        82
          7       0.99      0.99      0.99        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       154
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       223
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        81
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07808862 -0.09452943  0.03721352 -0.015276   -0.1381951   0.03781991
 -0.04856172 -0.00538208]
Epoch number and batch_no:  184 1
Loss before optimizing :  0.161100029045
Loss, accuracy and verification results :  0.161100029045 0.959836623553 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.88      0.98      0.92       305
          1       1.00      0.82      0.90        11
          2       0.99      0.99      0.99       289
          3       1.00      0.98      0.99        82
          5       0.99      0.99      0.99       475
          6       0.97      0.77      0.86       138
          7       0.96      0.95      0.95       169

avg / total       0.96      0.96      0.96      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       340
          1       1.00      1.00      1.00         9
          2       1.00      1.00      1.00       290
          3       1.00      1.00      1.00        80
          5       1.00      1.00      1.00       474
          6       1.00      1.00      1.00       109
          7       1.00      1.00      1.00       167

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07758511 -0.09446231  0.03726306 -0.01516813 -0.13819512  0.03787147
 -0.04786025 -0.00556789]
Epoch number and batch_no:  185 0
Loss before optimizing :  0.0774997226746
Loss, accuracy and verification results :  0.0774997226746 0.997011952191 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.99      0.99      0.99       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.99      1.00      0.99        82
          7       1.00      0.98      0.99        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       382
          6       1.00      1.00      1.00        83
          7       1.00      1.00      1.00        89

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07716038 -0.09445546  0.03733754 -0.01514476 -0.13819513  0.03790149
 -0.0474591  -0.00550886]
Epoch number and batch_no:  185 1
Loss before optimizing :  0.164029775666
Loss, accuracy and verification results :  0.164029775666 0.957113682777 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.98      0.86      0.92       305
          1       1.00      0.91      0.95        11
          2       0.98      0.99      0.99       289
          3       0.98      1.00      0.99        82
          5       0.98      1.00      0.99       475
          6       0.80      0.98      0.88       138
          7       0.95      0.92      0.93       169

avg / total       0.96      0.96      0.96      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       268
          1       1.00      1.00      1.00        10
          2       1.00      1.00      1.00       293
          3       1.00      1.00      1.00        84
          5       1.00      1.00      1.00       482
          6       1.00      1.00      1.00       169
          7       1.00      1.00      1.00       163

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07762424 -0.09438263  0.03735427 -0.0153629  -0.13819514  0.03787752
 -0.04843866 -0.00516861]
Epoch number and batch_no:  186 0
Loss before optimizing :  0.0760823333622
Loss, accuracy and verification results :  0.0760823333622 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.0779996  -0.09437069  0.03738596 -0.01557018 -0.13819515  0.03783699
 -0.04911086 -0.00495146]
Epoch number and batch_no:  186 1
Loss before optimizing :  0.150913205479
Loss, accuracy and verification results :  0.150913205479 0.963240299523 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.90      0.97      0.94       305
          1       1.00      0.73      0.84        11
          2       0.99      0.99      0.99       289
          3       0.99      1.00      0.99        82
          5       0.99      1.00      0.99       475
          6       0.99      0.75      0.85       138
          7       0.94      0.97      0.95       169

avg / total       0.97      0.96      0.96      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       330
          1       1.00      1.00      1.00         8
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        83
          5       1.00      1.00      1.00       480
          6       1.00      1.00      1.00       104
          7       1.00      1.00      1.00       175

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07785279 -0.09425559  0.03741592 -0.01560079 -0.13819516  0.03774436
 -0.04857247 -0.00505382]
Epoch number and batch_no:  187 0
Loss before optimizing :  0.0743406238009
Loss, accuracy and verification results :  0.0743406238009 0.998007968127 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      0.99      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.99      0.99      0.99        82
          7       0.99      1.00      0.99        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       151
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        92

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07755715 -0.09414616  0.03752325 -0.01554628 -0.13819517  0.03769943
 -0.04793456 -0.00535141]
Epoch number and batch_no:  187 1
Loss before optimizing :  0.134084245375
Loss, accuracy and verification results :  0.134084245375 0.974812797822 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.98      0.93      0.95       305
          1       1.00      1.00      1.00        11
          2       0.99      1.00      1.00       289
          3       0.99      0.99      0.99        82
          5       1.00      1.00      1.00       475
          6       0.86      0.97      0.91       138
          7       0.96      0.95      0.96       169

avg / total       0.98      0.97      0.98      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       289
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       291
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       155
          7       1.00      1.00      1.00       166

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07780314 -0.09413734  0.03756704 -0.0155281  -0.13819517  0.03768546
 -0.04831165 -0.00545627]
Epoch number and batch_no:  188 0
Loss before optimizing :  0.0711395827983
Loss, accuracy and verification results :  0.0711395827983 0.998007968127 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      0.99      0.99       152
          1       1.00      1.00      1.00         4
          2       0.99      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       150
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       226
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07806492 -0.09423234  0.03759483 -0.01550713 -0.13819518  0.03766894
 -0.0487852  -0.00542309]
Epoch number and batch_no:  188 1
Loss before optimizing :  0.126674582856
Loss, accuracy and verification results :  0.126674582856 0.974812797822 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.93      0.98      0.96       305
          1       1.00      1.00      1.00        11
          2       0.99      0.99      0.99       289
          3       0.98      1.00      0.99        82
          5       1.00      1.00      1.00       475
          6       0.95      0.85      0.90       138
          7       0.97      0.96      0.97       169

avg / total       0.97      0.97      0.97      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       320
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        84
          5       1.00      1.00      1.00       474
          6       1.00      1.00      1.00       123
          7       1.00      1.00      1.00       168

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07800331 -0.09440879  0.03755321 -0.01554172 -0.13819519  0.03767018
 -0.04879905 -0.00514726]
Epoch number and batch_no:  189 0
Loss before optimizing :  0.0637282287079
Loss, accuracy and verification results :  0.0637282287079 0.999003984064 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       0.99      1.00      0.99        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       223
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        92

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07784881 -0.09457228  0.03757686 -0.01555028 -0.1381952   0.03765575
 -0.04868556 -0.00495341]
Epoch number and batch_no:  189 1
Loss before optimizing :  0.12205289466
Loss, accuracy and verification results :  0.12205289466 0.980258679374 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.97      0.96      0.97       305
          1       1.00      0.91      0.95        11
          2       1.00      0.99      0.99       289
          3       1.00      1.00      1.00        82
          5       0.99      1.00      1.00       475
          6       0.95      0.92      0.94       138
          7       0.94      0.98      0.96       169

avg / total       0.98      0.98      0.98      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       301
          1       1.00      1.00      1.00        10
          2       1.00      1.00      1.00       288
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       478
          6       1.00      1.00      1.00       133
          7       1.00      1.00      1.00       177

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07789046 -0.09452816  0.03761676 -0.01555048 -0.13819521  0.03760711
 -0.04857196 -0.00509323]
Epoch number and batch_no:  190 0
Loss before optimizing :  0.0591053715641
Loss, accuracy and verification results :  0.0591053715641 0.999003984064 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      0.99      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       0.99      1.00      0.99        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       151
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        83
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07807276 -0.09445814  0.03767751 -0.01548549 -0.13819522  0.03754718
 -0.04871736 -0.00527836]
Epoch number and batch_no:  190 1
Loss before optimizing :  0.110082326897
Loss, accuracy and verification results :  0.110082326897 0.989108236896 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.97      0.99      0.98       305
          1       1.00      0.82      0.90        11
          2       0.99      1.00      0.99       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       0.98      0.96      0.97       138
          7       0.99      0.98      0.98       169

avg / total       0.99      0.99      0.99      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       309
          1       1.00      1.00      1.00         9
          2       1.00      1.00      1.00       292
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       474
          6       1.00      1.00      1.00       136
          7       1.00      1.00      1.00       167

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07814472 -0.09436519  0.0377045  -0.01534365 -0.13819523  0.03748108
 -0.04882396 -0.00529605]
Epoch number and batch_no:  191 0
Loss before optimizing :  0.0581255025282
Loss, accuracy and verification results :  0.0581255025282 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07817818 -0.09426511  0.0376751  -0.01522561 -0.13819523  0.03741682
 -0.04887782 -0.00517873]
Epoch number and batch_no:  191 1
Loss before optimizing :  0.103044130443
Loss, accuracy and verification results :  0.103044130443 0.988427501702 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.98      0.98      0.98       305
          1       1.00      0.91      0.95        11
          2       0.99      1.00      0.99       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       0.98      0.96      0.97       138
          7       0.99      0.98      0.99       169

avg / total       0.99      0.99      0.99      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       305
          1       1.00      1.00      1.00        10
          2       1.00      1.00      1.00       293
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       136
          7       1.00      1.00      1.00       168

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07815495 -0.09428301  0.03759376 -0.01516155 -0.13819524  0.03736038
 -0.04875833 -0.00497663]
Epoch number and batch_no:  192 0
Loss before optimizing :  0.0535071592875
Loss, accuracy and verification results :  0.0535071592875 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07816912 -0.09432479  0.03753689 -0.01512738 -0.13819525  0.03733531
 -0.04871319 -0.00487035]
Epoch number and batch_no:  192 1
Loss before optimizing :  0.096514350472
Loss, accuracy and verification results :  0.096514350472 0.989108236896 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.99      0.97      0.98       305
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       0.96      0.96      0.96       138
          7       0.97      0.99      0.98       169

avg / total       0.99      0.99      0.99      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       301
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       476
          6       1.00      1.00      1.00       138
          7       1.00      1.00      1.00       172

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07838958 -0.09443392  0.03755122 -0.0151255  -0.13819526  0.03733091
 -0.04887886 -0.00505401]
Epoch number and batch_no:  193 0
Loss before optimizing :  0.0519433300241
Loss, accuracy and verification results :  0.0519433300241 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07842327 -0.09457458  0.03762789 -0.01506915 -0.13819527  0.03734439
 -0.04896603 -0.00520959]
Epoch number and batch_no:  193 1
Loss before optimizing :  0.0885581190136
Loss, accuracy and verification results :  0.0885581190136 0.993873383254 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.98      1.00      0.99       305
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       0.99      0.95      0.97       138
          7       1.00      0.99      1.00       169

avg / total       0.99      0.99      0.99      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       311
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       476
          6       1.00      1.00      1.00       132
          7       1.00      1.00      1.00       168

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07828863 -0.09466274  0.03770674 -0.01498996 -0.13819528  0.03735789
 -0.04899464 -0.00518366]
Epoch number and batch_no:  194 0
Loss before optimizing :  0.0464120817983
Loss, accuracy and verification results :  0.0464120817983 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07818797 -0.09474443  0.0377689  -0.01489688 -0.13819528  0.03734911
 -0.04909573 -0.00506095]
Epoch number and batch_no:  194 1
Loss before optimizing :  0.082592843964
Loss, accuracy and verification results :  0.082592843964 0.997957794418 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.99      1.00      1.00       305
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       0.99      0.99      0.99       138
          7       1.00      0.99      1.00       169

avg / total       1.00      1.00      1.00      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       307
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       137
          7       1.00      1.00      1.00       168

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07826842 -0.0947732   0.03778959 -0.01485492 -0.13819529  0.03731632
 -0.04930892 -0.00495936]
Epoch number and batch_no:  195 0
Loss before optimizing :  0.0448778281275
Loss, accuracy and verification results :  0.0448778281275 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07842076 -0.09479057  0.03779779 -0.01486642 -0.1381953   0.03727224
 -0.04949171 -0.00491552]
Epoch number and batch_no:  195 1
Loss before optimizing :  0.0775864637972
Loss, accuracy and verification results :  0.0775864637972 0.997957794418 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.99      1.00      1.00       305
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      0.98      0.99       138
          7       1.00      1.00      1.00       169

avg / total       1.00      1.00      1.00      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       308
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       135
          7       1.00      1.00      1.00       169

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07846517 -0.09474683  0.03779793 -0.01491852 -0.1381953   0.03724506
 -0.04940775 -0.00497361]
Epoch number and batch_no:  196 0
Loss before optimizing :  0.0411378697106
Loss, accuracy and verification results :  0.0411378697106 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07847112 -0.09472405  0.03779389 -0.0149122  -0.13819531  0.03723048
 -0.04929631 -0.0050546 ]
Epoch number and batch_no:  196 1
Loss before optimizing :  0.0720967034094
Loss, accuracy and verification results :  0.0720967034094 0.999319264806 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       305
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      0.99      1.00       138
          7       1.00      1.00      1.00       169

avg / total       1.00      1.00      1.00      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       306
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       137
          7       1.00      1.00      1.00       169

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.0785075  -0.09472199  0.03779713 -0.01490948 -0.13819532  0.03721217
 -0.04919859 -0.00516745]
Epoch number and batch_no:  197 0
Loss before optimizing :  0.0386121032993
Loss, accuracy and verification results :  0.0386121032993 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07854829 -0.09472956  0.0378384  -0.01487887 -0.13819532  0.03719672
 -0.04925713 -0.00523762]
Epoch number and batch_no:  197 1
Loss before optimizing :  0.067386617267
Loss, accuracy and verification results :  0.067386617267 0.997957794418 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       305
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       0.99      0.99      0.99       138
          7       0.99      0.99      0.99       169

avg / total       1.00      1.00      1.00      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       306
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       137
          7       1.00      1.00      1.00       169

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07867422 -0.09478867  0.03787905 -0.01488045 -0.13819533  0.03718505
 -0.04949256 -0.00524675]
Epoch number and batch_no:  198 0
Loss before optimizing :  0.0362865626165
Loss, accuracy and verification results :  0.0362865626165 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07874833 -0.09482706  0.0378963  -0.01489369 -0.13819534  0.03718759
 -0.0496578  -0.00522888]
Epoch number and batch_no:  198 1
Loss before optimizing :  0.0646016673362
Loss, accuracy and verification results :  0.0646016673362 0.997957794418 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.99      1.00      1.00       305
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      0.98      0.99       138
          7       0.99      1.00      1.00       169

avg / total       1.00      1.00      1.00      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       307
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       135
          7       1.00      1.00      1.00       170

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07872429 -0.09486024  0.03789624 -0.01494663 -0.13819534  0.03719686
 -0.04962338 -0.00520056]
Epoch number and batch_no:  199 0
Loss before optimizing :  0.0342112408809
Loss, accuracy and verification results :  0.0342112408809 1.0 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       152
          1       1.00      1.00      1.00         4
          2       1.00      1.00      1.00       224
          3       1.00      1.00      1.00        70
          5       1.00      1.00      1.00       381
          6       1.00      1.00      1.00        82
          7       1.00      1.00      1.00        91

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.07865515 -0.09486866  0.03791126 -0.01499856 -0.13819535  0.03721815
 -0.04950984 -0.0052465 ]
Epoch number and batch_no:  199 1
Loss before optimizing :  0.0614189196371
Loss, accuracy and verification results :  0.0614189196371 0.998638529612 True
F1 score results : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       305
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       0.99      0.99      0.99       138
          7       0.99      1.00      1.00       169

avg / total       1.00      1.00      1.00      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       304
          1       1.00      1.00      1.00        11
          2       1.00      1.00      1.00       289
          3       1.00      1.00      1.00        82
          5       1.00      1.00      1.00       475
          6       1.00      1.00      1.00       138
          7       1.00      1.00      1.00       170

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.07870687 -0.09487638  0.03791807 -0.01505116 -0.13819535  0.03722954
 -0.04955715 -0.00529141]


"""






"""
Epoch number and batch_no:  0 0
Loss, accuracy :  910.271072368 275.0
Epoch number and batch_no:  0 1
Loss, accuracy :  batch_size1.51474569 291.0
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

"""
Epoch number and batch_no:  96 0
Loss before optimizing :  1.37624805636
Loss, accuracy and verification results :  1.37624805636 0.505976095618 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.79      0.14      0.24       152
          1       0.00      0.00      0.00         4
          2       0.88      0.30      0.45       224
          3       0.69      0.13      0.22        70
          5       0.49      1.00      0.66       381
          6       0.30      0.15      0.20        82
          7       0.22      0.18      0.20        91

avg / total       0.60      0.51      0.44      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        28
          2       1.00      1.00      1.00        77
          3       1.00      1.00      1.00        13
          5       1.00      1.00      1.00       774
          6       1.00      1.00      1.00        40
          7       1.00      1.00      1.00        72

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.01553228 -0.09186028  0.07326631 -0.01915726 -0.12206331  0.05017551
 -0.03588488 -0.00974813]
Epoch number and batch_no:  96 1
Loss before optimizing :  1.24284507325
Loss, accuracy and verification results :  1.24284507325 0.516678012253 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.65      0.24      0.35       305
          1       0.20      0.09      0.13        11
          2       0.71      0.52      0.60       289
          3       0.30      0.40      0.34        82
          5       0.67      0.81      0.73       475
          6       0.44      0.17      0.24       138
          7       0.23      0.55      0.33       169

avg / total       0.58      0.52      0.51      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       111
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       212
          3       1.00      1.00      1.00       110
          5       1.00      1.00      1.00       580
          6       1.00      1.00      1.00        52
          7       1.00      1.00      1.00       399

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.01764044 -0.09198615  0.07578766 -0.0206055  -0.12208426  0.04867134
 -0.03534396 -0.0128427 ]
Epoch number and batch_no:  97 0
Loss before optimizing :  1.32511782231
Loss, accuracy and verification results :  1.32511782231 0.497011952191 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.30      0.51      0.38       152
          1       0.00      0.00      0.00         4
          2       0.42      0.89      0.57       224
          3       0.66      0.41      0.51        70
          5       0.85      0.50      0.63       381
          6       0.60      0.04      0.07        82
          7       0.00      0.00      0.00        91

avg / total       0.56      0.50      0.46      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       256
          2       1.00      1.00      1.00       476
          3       1.00      1.00      1.00        44
          5       1.00      1.00      1.00       223
          6       1.00      1.00      1.00         5

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.01710549 -0.09214627  0.07546956 -0.0220802  -0.12210449  0.04961335
 -0.034228   -0.01425385]
Epoch number and batch_no:  97 1
Loss before optimizing :  1.34948807147
Loss, accuracy and verification results :  1.34948807147 0.495575221239 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.36      0.49      0.41       305
          1       0.00      0.00      0.00        11
          2       0.48      0.73      0.58       289
          3       1.00      0.01      0.02        82
          5       0.61      0.72      0.66       475
          6       0.44      0.17      0.25       138
          7       0.00      0.00      0.00       169

avg / total       0.46      0.50      0.44      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       412
          2       1.00      1.00      1.00       440
          3       1.00      1.00      1.00         1
          5       1.00      1.00      1.00       561
          6       1.00      1.00      1.00        55

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.01543615 -0.09200706  0.07428743 -0.02230088 -0.12212474  0.05062184
 -0.03360555 -0.012824  ]
Epoch number and batch_no:  98 0
Loss before optimizing :  1.12861186578
Loss, accuracy and verification results :  1.12861186578 0.581673306773 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.58      0.22      0.32       152
          1       0.00      0.00      0.00         4
          2       0.68      0.59      0.63       224
          3       0.65      0.21      0.32        70
          5       0.58      0.98      0.73       381
          6       0.35      0.37      0.36        82
          7       0.50      0.01      0.02        91

avg / total       0.58      0.58      0.52      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        57
          2       1.00      1.00      1.00       197
          3       1.00      1.00      1.00        23
          5       1.00      1.00      1.00       639
          6       1.00      1.00      1.00        86
          7       1.00      1.00      1.00         2

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.01480297 -0.09183728  0.0745125  -0.02124815 -0.12214543  0.04981957
 -0.03402408 -0.01013637]
Epoch number and batch_no:  98 1
Loss before optimizing :  1.29157145131
Loss, accuracy and verification results :  1.29157145131 0.515997277059 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.75      0.13      0.22       305
          1       0.40      0.18      0.25        11
          2       0.56      0.61      0.58       289
          3       0.71      0.06      0.11        82
          5       0.56      0.92      0.70       475
          6       0.31      0.35      0.33       138
          7       0.31      0.30      0.30       169

avg / total       0.56      0.52      0.46      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        53
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       315
          3       1.00      1.00      1.00         7
          5       1.00      1.00      1.00       774
          6       1.00      1.00      1.00       154
          7       1.00      1.00      1.00       161

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.01652751 -0.0917603   0.07504127 -0.01962564 -0.12216629  0.04836316
 -0.03560296 -0.00912291]
Epoch number and batch_no:  99 0
Loss before optimizing :  1.03699146106
Loss, accuracy and verification results :  1.03699146106 0.609561752988 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.63      0.34      0.44       152
          1       0.00      0.00      0.00         4
          2       0.56      0.85      0.67       224
          3       0.46      0.69      0.55        70
          5       0.86      0.76      0.81       381
          6       0.50      0.07      0.13        82
          7       0.21      0.30      0.25        91

avg / total       0.64      0.61      0.59      1004

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        81
          2       1.00      1.00      1.00       342
          3       1.00      1.00      1.00       105
          5       1.00      1.00      1.00       336
          6       1.00      1.00      1.00        12
          7       1.00      1.00      1.00       128

avg / total       1.00      1.00      1.00      1004

self.biases :  [ 0.01834752 -0.09179094  0.07472362 -0.01993384 -0.12218665  0.04826461
 -0.03622316 -0.01030711]
Epoch number and batch_no:  99 1
Loss before optimizing :  1.2505041845
Loss, accuracy and verification results :  1.2505041845 0.489448604493 True
F1 score results : 
              precision    recall  f1-score   support

          0       0.31      0.70      0.43       305
          1       0.40      0.18      0.25        11
          2       0.57      0.68      0.62       289
          3       0.38      0.18      0.25        82
          5       0.78      0.60      0.68       475
          6       0.00      0.00      0.00       138
          7       0.30      0.04      0.06       169

avg / total       0.49      0.49      0.45      1469

Predicted : 
              precision    recall  f1-score   support

          0       1.00      1.00      1.00       690
          1       1.00      1.00      1.00         5
          2       1.00      1.00      1.00       347
          3       1.00      1.00      1.00        40
          5       1.00      1.00      1.00       366
          6       1.00      1.00      1.00         1
          7       1.00      1.00      1.00        20

avg / total       1.00      1.00      1.00      1469

self.biases :  [ 0.01729476 -0.09207176  0.07439135 -0.02128846 -0.12220774  0.04928387
 -0.03523451 -0.01097222]
"""

























