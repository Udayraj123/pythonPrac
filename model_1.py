import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from random import shuffle
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.01
n_epochs = 80
batch_size = 1000
display_step = 1
hidden_units = 100
num_classes = 549

#@profile
def get_data(min_no_of_seq = 200):
	file_path = './data/db_' + str(min_no_of_seq) + '_pickle'
	# file_path = './data/db_100_pickle'
	# file_path = './data/db_50_pickle'
	file_ip = open(file_path, 'rb')
	data = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/amino_acid_map_pickle'
	file_ip = open(file_path, 'rb')
	amino_acid_map = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/families_map_pickle'
	file_ip = open(file_path, 'rb')
	families_map = pickle.load(file_ip)
	file_ip.close()

	data_train = []
	data_cv = []
	data_test = []

	# Process the data :
	# 1. Convert the data into integers.
	# 2. Now we have to divide the data
	#    into train-cv-test with 70-15-15.
	#    Do the processing class wise to 
	#    have stratification.
	# 3. Final output : 
	#    data_train
	#    data_test
	#    data_cv
	#    
	#    Each one is a list and every item in list
	#    is a mapping seq_int to fam_int

	total_no_of_seq = 0

	for k in data.keys():
		data_fam = data[k]
		shuffle(data_fam)
		fam_int = families_map[k]
		leave_the_family = False
		for entry_no in range(len(data_fam)):
			seq = data_fam[entry_no][1]
			if(len(seq) >= 1000):
				leave_the_family = True
		if(leave_the_family):
			continue
		total_no_of_seq += len(data_fam)
		for i in range(len(data_fam)):
			seq = data_fam[i][1]
			fam = data_fam[i][2]
			seq_int = []
			for j in range(len(seq)):
				seq_int.append(amino_acid_map[seq[j]])
			temp = [seq_int, fam_int]
			if(i >= 0.85*len(data_fam)):
				data_test.append(temp)
			elif(i >= 0.70*len(data_fam)):
				data_cv.append(temp)
			else:
				data_train.append(temp) 

	# for data in data_train:
	# 	print(data)
	# 	debug = input()
	print("Total, train, cv, test examples : ", total_no_of_seq,
		len(data_train), len(data_cv), len(data_test))

	return data_train, data_test, data_cv

class RnnForPfcModelOne:
#	@profile
	def __init__(self, 
		num_classes = 549, 
		hidden_units=100, 
		learning_rate=0.01):

		self.seq_length = tf.placeholder(tf.uint8, [None])
	
		self.x_input = tf.placeholder(tf.uint8, [None, None], name = 'x_ip')
		# batch_size * no_of_time_steps
		self.x_input_o = tf.one_hot(indices = self.x_input, 
			depth = 21,
			on_value = 1.0,
			off_value = 0.0,
			axis = -1)
		# batch_size * no_of_time_steps * 21
		self.y_input = tf.placeholder(tf.uint8, [None], name = 'y_ip')
		self.y_input_o = tf.one_hot(indices = self.y_input, 
									depth = num_classes+1,
									on_value = 1.0,
									off_value = 0.0,
									axis = -1)
		self.weights = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes+1], maxval=1))
		self.biases = tf.Variable(tf.random_uniform(shape=[num_classes+1]))
		self.rnn_fcell = rnn.BasicLSTMCell(num_units = hidden_units, 
										   forget_bias = 1.0,
										   activation = tf.tanh)
		self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_fcell,
													  self.x_input_o,
													  sequence_length = self.seq_length,
													  dtype = tf.float32)
		self.outputs_t = tf.reshape(self.outputs[:, -1, :], [-1, hidden_units])
		self.y_predicted = tf.matmul(self.outputs_t, self.weights) + self.biases
		self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.y_input_o)

		# define optimizer and trainer
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.trainer = self.optimizer.minimize(self.loss)

		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

		self.get_equal = tf.equal(tf.argmax(self.y_input_o, 1), tf.argmax(self.y_predicted, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.get_equal, tf.float32))
		self.summary_writer = tf.summary.FileWriter('./data/graph/', graph = self.sess.graph)
	
#	@profile
	def predict(self, x, y, seq_length):
		result = self.sess.run(self.y_predicted, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length})
		return result

#	@profile
	def optimize(self, x, y, seq_length):
		self.sess.run(self.trainer, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length})

#	@profile
	def cross_validate(self, x, y, seq_length):
		result = self.sess.run(self.accuracy, feed_dict={self.x_input:x, self.y_input:y, self.seq_length:seq_length})
		return result

#	@profile
	def close_summary_writer(self):
		self.summary_writer.close()

data_train, data_test, data_cv = get_data(200)
# print(len(data_train), (len(data_test)), (len(data_cv)))

model = RnnForPfcModelOne()

#@profile
def caller():
	for epoch in range(n_epochs):
		no_of_batches = len(data_train) // batch_size
		print(no_of_batches)
		shuffle(data_train)
		for batch_no in range(no_of_batches):
			x = []
			y = []
			max_length = 0
			seq_length = []
			for data in data_train[batch_no*batch_size: batch_no*batch_size+batch_size]:
				seq_length.append(len(data[0]))
				max_length = max(max_length, len(data[0]))
				x.append(data[0])
				y.append(data[1])
			x_padded = np.array([ row + [-1]*(max_length-len(row)) for row in x])
			y = np.array(y)
			model.optimize(x_padded, y, seq_length)
			accuracy_known = model.cross_validate(x_padded, y, seq_length)
			print("Iteration number, batch number : ", epoch, batch_no," Training data accuracy : ", accuracy_known)
			del accuracy_known
			del x
			del y
			del seq_length
			del x_padded
	

	no_of_batches = len(data_cv) // batch_size
	for batch_no in range(no_of_batches):
		x = []
		y = []
		max_length = 0
		seq_length = []
		for data in data_cv[batch_no*batch_size: batch_no*batch_size+batch_size]:
			seq_length.append(len(data[0]))
			max_length = max(max_length, len(data[0]))
			x.append(data[0])
			y.append(data[1])
		x_padded = np.array([ row + [-1]*(max_length-len(row)) for row in x])
		y = np.array(y)
		print("CV data accuracy : ", model.cross_validate(x_padded, y, seq_length))
		del x
		del y
		del seq_length
		del x_padded
	# x = []
	# y = []
	# max_length = 0
	# seq_length = []
	# for data in data_test:
	# 	seq_length.append(len(data[0]))
	# 	max_length = max(max_length, len(data[0]))
	# 	x.append(data[0])
	# 	y.append(data[1])
	# x_padded = np.array([ row + [-1]*(max_length-len(row)) for row in x])
	# y = np.array(y)
	# print("Test data accuracy : ", model.cross_validate(x_padded, y, seq_length))

caller()

model.close_summary_writer()
