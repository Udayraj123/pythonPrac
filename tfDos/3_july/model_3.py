import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from random import shuffle
import numpy as np
import time
from sklearn.metrics import classification_report as c_metric
import os
import sys
tf.logging.set_verbosity(tf.logging.ERROR)

def save_obj(obj,filename,overwrite=1):
  if(not overwrite and os.path.exists(filename)):
    return
  with open(filename,'wb') as f:
    pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    print("File saved to " + filename)

def load_obj(filename):
  with open(filename, 'rb') as f:
    obj = pickle.load(f)
    print("File loaded from " + filename)
    return obj

def get_data_train():
  file_path = './data/batch_wise_data_5.pkl'
  p=time.time()
  with open(file_path, 'rb') as file_ip:
    data_train = pickle.load(file_ip)
  print("Data has been loaded in %d seconds" % (time.time()-p) )
  return data_train

class BrnnForPsspModelOne:
  def __init__(self,model_path,load_model_filename,curr_model_filename,
    num_classes = 8,
    hidden_units = 100,
    batch_size = 5):
    print("Initializing model..")
    p=time.time()

    self.input_x = tf.placeholder(tf.float32, [ batch_size, 800, 122])
    self.input_y = tf.placeholder(tf.uint8, [ batch_size, 800]) # Int 8 will be sufficient for just 8 classes.
    self.input_msks = tf.placeholder(tf.float32, [ batch_size, 800])
    self.input_seq_len = tf.placeholder(tf.int32, [ batch_size])
    self.input_y_o = tf.one_hot(indices = self.input_y,
      depth = num_classes,
      on_value = 1.0,
      off_value = 0.0,
      axis = -1)

    self.hidden_units = tf.constant(hidden_units, dtype = tf.float32)
    # define weights and biases here (4 weights + 4 biases)
    self.weight_f_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_c = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_f_p_50 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_p_50 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_f_p_20 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_p_20 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_f_p_10 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_p_10 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_f_p_30 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    self.weight_b_p_30 = tf.Variable(0.01 * tf.random_uniform(shape=[hidden_units, num_classes], maxval=1, dtype=tf.float32), dtype=tf.float32) 
    # self.biases_f_c = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    # self.biases_b_c = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    # self.biases_f_p_50 = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    # self.biases_b_p_50 = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    # self.biases_f_p_20 = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    # self.biases_b_p_20 = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    # self.biases_f_p_10 = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    # self.biases_b_p_10 = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    # self.biases_f_p_30 = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    # self.biases_b_p_30 = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    self.biases = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), dtype=tf.float32)
    
    self.rnn_cell_f = rnn.GRUCell(num_units = hidden_units, 
      activation = tf.tanh)
    self.rnn_cell_b = rnn.GRUCell(num_units = hidden_units, 
      activation = tf.tanh)
    self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(
      cell_fw = self.rnn_cell_f,
      cell_bw = self.rnn_cell_b,
      inputs = self.input_x,
      sequence_length = self.input_seq_len,
      dtype = tf.float32,
      swap_memory = False)
    self.outputs_f = self.outputs[0]
    self.outputs_b = self.outputs[1]
    # self.outputs_f_p_50_l = []
    # self.outputs_b_p_50_l = []
    # self.outputs_f_p_20_l = []
    # self.outputs_b_p_20_l = []
    # # self.outputs_f_p_10_l = []
    # # self.outputs_b_p_10_l = []
    # # self.outputs_f_p_30_l = []
    # # self.outputs_b_p_30_l = []
    # for i in range(700):
    #   # 50 dummies + seq + 50 dummies
    #   # For forward maxpooling, index i will have maxpool from i-50:i 
    #   # Loss due to dummies will get maske completely 
    #   self.outputs_f_p_50_l.append(tf.reduce_max(self.outputs_f[: , i:i+50, :],
    #     axis = 1))
    #   self.outputs_b_p_50_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+101, :],
    #     axis = 1))
    #   self.outputs_f_p_20_l.append(tf.reduce_max(self.outputs_f[: , i+30:i+50, :],
    #     axis = 1))
    #   self.outputs_b_p_20_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+71, :],
    #     axis = 1))
    #   # self.outputs_f_p_10_l.append(tf.reduce_max(self.outputs_b[: , i+40:i+50, :],axis = 1))
    #   # self.outputs_b_p_10_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+61, :],axis = 1))
    #   # self.outputs_f_p_30_l.append(tf.reduce_max(self.outputs_b[: , i+20:i+50, :],axis = 1))
    #   # self.outputs_b_p_30_l.append(tf.reduce_max(self.outputs_b[: , i+51:i+81, :],axis = 1))
    # self.outputs_f_p_50 = tf.stack(self.outputs_f_p_50_l, axis = 1)
    # self.outputs_b_p_50 = tf.stack(self.outputs_b_p_50_l, axis = 1)
    # self.outputs_f_p_20 = tf.stack(self.outputs_f_p_20_l, axis = 1)
    # self.outputs_b_p_20 = tf.stack(self.outputs_b_p_20_l, axis = 1)
    # self.outputs_f_p_10 = tf.stack(self.outputs_f_p_10_l, axis = 1)
    # self.outputs_b_p_10 = tf.stack(self.outputs_b_p_10_l, axis = 1)
    # self.outputs_f_p_30 = tf.stack(self.outputs_f_p_30_l, axis = 1)
    # self.outputs_b_p_30 = tf.stack(self.outputs_b_p_30_l, axis = 1)
    
    self.outputs_f_p_50 = tf.reshape(
                            tf.nn.max_pool(
                              tf.reshape(self.outputs_f, [batch_size, 800, 100, 1]), 
                              ksize = [1, 50, 1, 1], 
                              strides = [1, 1, 1, 1], 
                              padding = 'VALID'),
                            [batch_size, 751, 100]
                          )[:, 0:700, :]
    self.outputs_b_p_50 = tf.reshape(
                            tf.nn.max_pool(
                              tf.reshape(self.outputs_b, [batch_size, 800, 100, 1]), 
                              ksize = [1, 50, 1, 1], 
                              strides = [1, 1, 1, 1], 
                              padding = 'VALID'),
                            [batch_size, 751, 100]
                          )[:, 51:751, :]
    self.outputs_f_p_20 = tf.reshape(
                            tf.nn.max_pool(
                              tf.reshape(self.outputs_f[:, 30:750, :], [batch_size, 720, 100, 1]), 
                              ksize = [1, 20, 1, 1], 
                              strides = [1, 1, 1, 1], 
                              padding = 'VALID'),
                            [batch_size, 701, 100]
                          )[:, 0:700, :]
    self.outputs_b_p_20 = tf.reshape(
                            tf.nn.max_pool(
                              tf.reshape(self.outputs_b[:, 50:770, :], [batch_size, 720, 100, 1]), 
                              ksize = [1, 20, 1, 1], 
                              strides = [1, 1, 1, 1], 
                              padding = 'VALID'),
                            [batch_size, 701, 100]
                          )[:, 1:701, :]
    
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
    self.get_equal = tf.multiply(tf.cast(self.get_equal_unmasked, tf.float32), self.input_msks_r)
    self.accuracy = ( tf.reduce_sum(tf.cast(self.get_equal, tf.float32)) / self.no_of_entries_unmasked)

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
    # 'Saver' op to save and restore all the variables
    self.saver = tf.train.Saver()

    # Restore model weights from previously saved model
    self.load_file_path = model_path+load_model_filename
    self.curr_file_path = model_path+curr_model_filename
    
    print("Model Initialized in %d seconds " % (time.time()-p))
    if os.path.exists(self.load_file_path):
      print("Restoring model...")
      p=time.time()
      self.sess.run(self.init)
      saver.restore(self.sess, self.load_file_path)
      print("Model restored from file: %s in %d seconds " % (save_path,time.time()-p))
    else:
      print("Load file DNE at "+load_model_filename+", Preparing new model...")
      #just make dir if DNE
      if not os.path.exists(model_path):
        print("created DIR "+model_path)
        os.makedirs(model_path)
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
    # f_c, b_c, f_p_50, b_p_50, f_p_20, b_p_20, 
    biases = self.sess.run([
      # self.biases_f_c,
      # self.biases_b_c,
      # self.biases_f_p_50,
      # self.biases_b_p_50,
      # self.biases_f_p_20,
      # self.biases_b_p_20,
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

#   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   ##
  model_path = "/tmp/LSTMmodels/"
  remake_chkpt=True
  args=sys.argv
  file_index=1
  if(len(args)>1):
    remake_chkpt = int(args[1])==0
    file_index= int(args[1])

  model_filenames_pkl = model_path+'model_filenames_pkl.pkl'
  epoch_wise_accs_pkl = model_path+'epoch_wise_accs_pkl.pkl'
  epoch_wise_loss_pkl = model_path+'epoch_wise_loss_pkl.pkl'
  start_time = time.strftime("%b%d_%H:%M%p") #by default takes current time
  curr_model_filename = "model_started_"+start_time+"_.ckpt"
  
  if(os.path.exists(model_filenames_pkl)):
    model_filenames = load_obj(model_filenames_pkl) #next time
  else:
    model_filenames=[curr_model_filename] #first time.

  if(remake_chkpt):
    print("Adding new checkpoint file")
    load_model_filename = curr_model_filename
  else:
    if( file_index > len(model_filenames) ):
      raise ValueError("Invalid file index. Avl checkpoints are : ",model_filenames)
    load_model_filename = model_filenames[-1* file_index]
    print("Loading model from file ",load_model_filename)
#   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   ##

  # Restore will happen from inside the class
  model = BrnnForPsspModelOne(model_path,load_model_filename,curr_model_filename)

  data_train = get_data_train()
  # for batch_no in range(43):
  model.get_shapes()
  n_epochs = 10
  num_batches= 1106
  
# Want = Accuracies of each epochs printed into a file.
  epoch_wise_accs = []
  epoch_wise_loss = []

  for epoch in range(n_epochs):
    for batch_no in range(num_batches):
      print("Epoch number and batch_no: ", epoch, batch_no)
      data = data_train[batch_no]
      x_inp = data[0]
      y_inp = data[1]
      m_inp = data[2]
      l_inp = data[3]
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
    
    epoch_wise_accs.append(accuracy)
    epoch_wise_loss.append(loss)
#   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   ##
    print('')
    # Save model weights to disk
    p=time.time()
    save_path = model.saver.save(model.sess, model.curr_file_path,global_step=epoch)
    model_filenames.append(save_path.split('/')[-1])
    print("Epoch %d : Model saved in file: %s in %d seconds " % (epoch, save_path,time.time()-p))
    save_obj(model_filenames,model_filenames_pkl,overwrite=1)
    save_obj(epoch_wise_accs,epoch_wise_accs_pkl,overwrite=1)
    save_obj(epoch_wise_loss,epoch_wise_loss_pkl,overwrite=1)
    print("Current saved checkpoints : ",model_filenames)
    print('')
#   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   #  #    #   #   ##
















