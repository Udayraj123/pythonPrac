import tensorflow as tf

class Model:
	def weight_variable(self, shape):
	  initial = tf.truncated_normal(shape, stddev=0.1, name='weights')
	  return tf.Variable(initial)

	def bias_variable(self, shape):
	  initial = tf.constant(0.1, shape=shape, name='bias')
	  return tf.Variable(initial)

	def __init__(self, num_hidden, hidden_size, window_size, emb_size, vocabulary_size, n_classes, inputs, labels, keep_prob, learning_rate, embedding):
		self.num_hidden = num_hidden
		self.hidden_size = hidden_size
		self.window_size = window_size
		self.emb_size = emb_size
		self.vocabulary_size = vocabulary_size
		self.n_classes = n_classes
		self.inputs = inputs
		self.labels = labels
		self.keep_prob = keep_prob
		self.learning_rate = learning_rate

		#initialize embedding table
		with tf.variable_scope('distance_embedding'):
			#embeddings = tf.get_variable(name="embedding", shape=embedding.shape, initializer=tf.constant_initializer(embedding), trainable=False)
			embeddings = tf.constant(embedding, name="embedding", shape=embedding.shape, dtype='float32')
		
		#embedding lookup for input window
		with tf.variable_scope('lookup'):
			embed = tf.nn.embedding_lookup(embeddings, self.inputs)

		#initial setup for input
		prev_size = self.emb_size * self.window_size
		hidden_dropout = tf.reshape(embed, [tf.shape(embed)[0], prev_size])

		#create the hidden layers
		for i in xrange(num_hidden):
			with tf.variable_scope('W_h'+str(i+1)):
				weights = self.weight_variable([prev_size, hidden_size[i]])
				bias = self.bias_variable([hidden_size[i]])
				hidden = tf.nn.relu(tf.matmul(hidden_dropout, weights) + bias)
				hidden_dropout = tf.nn.dropout(hidden, keep_prob)
				prev_size = hidden_size[i]

		#create output layer
		with tf.variable_scope('W_out'):
			out_weights = self.weight_variable([prev_size, n_classes])
			out_bias = self.bias_variable([n_classes])
			self.out = tf.matmul(hidden, out_weights) + out_bias


		self.prediction = tf.nn.softmax(self.out)				#softmax for just output values
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.out))		#loss calculation
		self.optimize = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)		#optimizer
		mistakes = tf.equal(self.labels, tf.argmax(self.prediction, 1))
		self.correct = tf.reduce_mean(tf.cast(mistakes, tf.float32))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))