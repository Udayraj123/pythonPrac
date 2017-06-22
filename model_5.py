import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from random import shuffle
import numpy as np
from sklearn.metrics import classification_report as c_metric
tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.01
n_epochs = 10
batch_size = 1000
display_step = 1
hidden_units = 100
num_classes = 498

def get_data(min_no_of_seq = 200):
	file_path = './data/data_train_glove_vec' + str(min_no_of_seq) + '_pickle'
	file_ip = open(file_path, 'rb')
	data_train = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/data_cv_glove_vec' + str(min_no_of_seq) + '_pickle'
	file_ip = open(file_path, 'rb')
	data_cv = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/data_test_glove_vec' + str(min_no_of_seq) + '_pickle'
	file_ip = open(file_path, 'rb')
	data_test = pickle.load(file_ip)
	file_ip.close()

	# print(data_train[0], len(data_train))
	# debug = input()
	# print(data_cv[0], len(data_cv))
	# debug = input()
	# print(data_test[0], len(data_test))
	# debug = input()
	# dimensions matched

	return data_train, data_test, data_cv

class RnnForPfcModelFive:
	def __init__(self, 
		num_classes = 549, 
		hidden_units=100, 
		learning_rate=0.01):

		self.seq_length = tf.placeholder(tf.uint8, [None])
	
		self.x_input = tf.placeholder(tf.float32, [None, None, 100], name = 'x_ip')
		# batch_size * no_of_time_steps * 100
		self.y_input = tf.placeholder(tf.uint8, [None], name = 'y_ip')
		self.y_input_o = tf.one_hot(indices = self.y_input, 
									depth = num_classes+1,
									on_value = 1.0,
									off_value = 0.0,
									axis = -1)
		self.weights_f = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes+1], maxval=1))
		self.weights_p = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes+1], maxval=1))
		self.biases_f = tf.Variable(tf.random_uniform(shape=[num_classes+1]))
		self.biases_p = tf.Variable(tf.random_uniform(shape=[num_classes+1]))
		self.rnn_fcell = rnn.BasicLSTMCell(num_units = hidden_units, 
										   forget_bias = 1.0,
										   activation = tf.tanh)
		self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_fcell,
													  self.x_input,
													  sequence_length = self.seq_length,
													  dtype = tf.float32)
		
		self.outputs_f = tf.reshape(self.outputs[:, -1, :], [-1, hidden_units])
		self.outputs_maxpooled = tf.reduce_max(self.outputs, axis = 1)
		self.outputs_p = tf.reshape(self.outputs_maxpooled, [-1, hidden_units])
		self.y_predicted = (tf.matmul(self.outputs_f, self.weights_f) + self.biases_f 
   						   + tf.matmul(self.outputs_p, self.weights_p) + self.biases_p)
		self.loss = tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.y_input_o))

		# define optimizer and trainer
		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		self.trainer = self.optimizer.minimize(self.loss)

		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

		self.get_equal = tf.equal(tf.argmax(self.y_input_o, 1), tf.argmax(self.y_predicted, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.get_equal, tf.float32))

	def predict(self, x, y, seq_length):
		result = self.sess.run(self.y_predicted, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length})
		result = np.argmax(result, axis=1)
		result = np.reshape(result, [-1])
		return result

	def optimize(self, x, y, seq_length):
		result = self.sess.run(self.trainer, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length})

	def cross_validate(self, x, y, seq_length):
		result = self.sess.run(self.accuracy, feed_dict={self.x_input:x, self.y_input:y, self.seq_length:seq_length})
		return result

	def get_loss(self, x, y, seq_length):
		result = self.sess.run(self.loss, feed_dict={self.x_input:x, self.y_input:y, self.seq_length:seq_length})
		return result

data_train, data_test, data_cv = get_data(200)
print("Loaded the data files into memory")
print(len(data_train), (len(data_test)), (len(data_cv)))

model = RnnForPfcModelFive()

def train_on_train_data(epoch):
	no_of_batches = len(data_train) // batch_size
	shuffle(data_train)
	zero_100_list = [[0] * 100]
	for batch_no in range(no_of_batches):
		print("Iteration number, batch number : ", epoch, batch_no)
		x = []
		y = []
		max_length = 0
		seq_length = []
		for data in data_train[batch_no*batch_size: batch_no*batch_size+batch_size]:
			seq_length.append(len(data[0]))
			max_length = max(max_length, len(data[0]))
			x.append(data[0])
			y.append(data[1])
		x_n = [ row + (zero_100_list)*(max_length-len(row)) for row in x]
		x_padded = np.array(x_n)
		y = np.array(y)
		print("Here : ")
		model.optimize(x_padded, y, seq_length)
		if(batch_no%50==10):
			res_on_train_data()
			res_on_test_data()
		print("Training data accuracy : ", model.cross_validate(x_padded, y, seq_length))
		print("Training data loss     : ", model.get_loss(x_padded, y, seq_length))

def res_on_train_data():
	no_of_batches = len(data_train) // batch_size
	no_of_batches = 3
	shuffle(data_train)
	zero_100_list = [[0] * 100]
	correct = 0
	y_actual = []
	y_predicted = []

	for batch_no in range(no_of_batches):
		if(batch_no%5 == 0):
			print("res_on_train_data : ", batch_no)
		x = []
		y = []
		max_length = 0
		seq_length = []
		for data in data_train[batch_no*batch_size: batch_no*batch_size+batch_size]:
			seq_length.append(len(data[0]))
			max_length = max(max_length, len(data[0]))
			x.append(data[0])
			y.append(data[1])
		x_n = [ row + (zero_100_list)*(max_length-len(row)) for row in x]
		x_padded = np.array(x_n)
		y_actual.extend(y)
		y = np.array(y)
		accuracy_known = model.cross_validate(x_padded, y, seq_length)
		y_predicted_ = model.predict(x_padded, y, seq_length)
		y_predicted.extend(y_predicted_)
		correct += accuracy_known * batch_size
	accuracy = (correct * 100) / (batch_size * no_of_batches)
	correct_1 = len([i for i,j in zip(y_actual, y_predicted)])
	print("Accuracy on train data : ", accuracy, correct, correct_1)
	print (c_metric(y_actual, y_predicted))

def res_on_test_data():
	no_of_batches = len(data_test) // batch_size
	no_of_batches = 3
	shuffle(data_test)
	zero_100_list = [[0] * 100]
	correct = 0
	y_actual = []
	y_predicted = []

	for batch_no in range(no_of_batches):
		if(batch_no%5 == 0):
			print("res_on_test_data : ", batch_no)
		x = []
		y = []
		max_length = 0
		seq_length = []
		for data in data_test[batch_no*batch_size: batch_no*batch_size+batch_size]:
			seq_length.append(len(data[0]))
			max_length = max(max_length, len(data[0]))
			x.append(data[0])
			y.append(data[1])
		x_n = [ row + (zero_100_list)*(max_length-len(row)) for row in x]
		x_padded = np.array(x_n)
		y_actual.extend(y)
		y = np.array(y)
		accuracy_known = model.cross_validate(x_padded, y, seq_length)
		y_predicted_ = model.predict(x_padded, y, seq_length)
		y_predicted.extend(y_predicted_)
		correct += accuracy_known * batch_size
	accuracy = (correct * 100) / (batch_size * no_of_batches)
	correct_1 = len([i for i,j in zip(y_actual, y_predicted)])
	print("Accuracy on train data : ", accuracy, correct, correct_1)
	print (c_metric(y_actual, y_predicted))

def res_on_cv_data():
	no_of_batches = len(data_cv) // batch_size
	no_of_batches = 3
	shuffle(data_cv)
	zero_100_list = [[0] * 100]
	correct = 0
	y_actual = []
	y_predicted = []

	for batch_no in range(no_of_batches):
		if(batch_no%5 == 0):
			print("res_on_cv_data : ", batch_no)
		x = []
		y = []
		max_length = 0
		seq_length = []
		for data in data_cv[batch_no*batch_size: batch_no*batch_size+batch_size]:
			seq_length.append(len(data[0]))
			max_length = max(max_length, len(data[0]))
			x.append(data[0])
			y.append(data[1])
		x_n = [ row + (zero_100_list)*(max_length-len(row)) for row in x]
		x_padded = np.array(x_n)
		y_actual.extend(y)
		y = np.array(y)
		accuracy_known = model.cross_validate(x_padded, y, seq_length)
		y_predicted_ = model.predict(x_padded, y, seq_length)
		y_predicted.extend(y_predicted_)
		correct += accuracy_known * batch_size
	accuracy = (correct * 100) / (batch_size * no_of_batches)
	correct_1 = len([i for i,j in zip(y_actual, y_predicted)])
	print("Accuracy on train data : ", accuracy, correct, correct_1)
	print (c_metric(y_actual, y_predicted))

for epoch in range(n_epochs):
	train_on_train_data(epoch)
	res_on_train_data()
	res_on_test_data()
	res_on_cv_data()

model.close_summary_writer()
