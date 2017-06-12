import tensorflow as tf
import numpy as np

 
import functools

def lazyproperty(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Modelcnn:
	def __init__(self,target,words,pos_tags,dis,pos_voc,pos_embed_size,dis_voc,dis_embed_size, embs, word_embed_size, filter_sizes,num_filters, partitions):
		
		self.words = words
		self.pos_tags = tf.cast(pos_tags,tf.int32)
		self.dis = tf.cast(dis,tf.int32)
		self.target = target
		self.out_size = 5
		self.pos_voc = pos_voc
		self.dis_voc = dis_voc
		self.dis_embed_size = dis_embed_size
		self.pos_embed_size = pos_embed_size
		self.word_embed_size = word_embed_size
		self.embs = embs

		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.max_length = 100
		self.num_partitions = 2
		self.partitions = partitions
	
		self.prediction
		# self.loss 
		# self.optimize 
		# self.correct 
		# self.accuracy



	@lazyproperty
	def prediction(self):

		# embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, 200], -1.0, 1.0))
		embeddings = tf.constant(self.embs, tf.float32)
		embed = tf.nn.embedding_lookup(embeddings, self.words)

		# dis_embeddings = tf.Variable(tf.random_uniform([self.dis_voc, self.dis_embed_size], -1.0, 1.0))
		# dis_embed = tf.nn.embedding_lookup(dis_embeddings, self.dis)

		# pos_embeddings = tf.Variable(tf.random_uniform([self.pos_voc, self.pos_embed_size], -1.0, 1.0))
		# pos_embed = tf.nn.embedding_lookup(pos_embeddings, self.pos_tags)

		# print(self.words.get_shape())
		# print(pos_embed.get_shape())
		# print(dis_embed.get_shape())

		# last = tf.concat([self.words , pos_embed , dis_embed], 2)
		last = embed
		print(last.get_shape())
		last_expanded = tf.expand_dims(last, -1)

		print(last_expanded.get_shape())
		# emb_size = 200 + self.pos_embed_size + self.dis_embed_size
		emb_size = 200		
		# self.count = (self.count + 1)%353
		# print(self.count)


		pooled_outputs = []
		for i, filter_size in enumerate(self.filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):

				filter_shape = [filter_size, self.word_embed_size, 1, self.num_filters]
        		
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
        		
				conv = tf.nn.conv2d(last_expanded,W,strides=[1, 1, 1, 1],padding="VALID",name="conv")

				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				print (h.get_shape())

				hnew = tf.reshape(h, [-1, self.max_length - filter_size + 1, self.num_filters])
				hnew = tf.transpose(hnew,[1, 0, 2])
				print (hnew.get_shape())
				split = tf.dynamic_partition(hnew,self.partitions[i],2)
				print (split[0].get_shape())
				[split0,split1] = [tf.transpose(sp, [1, 0, 2]) for sp in split]

				pool1 = tf.reduce_max(split0, 1)
				pool2 = tf.reduce_max(split1, 1)
				# pool3 = tf.reduce_max(split2, 1)
				print(pool2.get_shape())
				pooled = tf.stack([pool1,pool2], 1)
				# print(pooled.get_shape())
				pooled_outputs.append(pooled)
				# print(p)
 

		num_filters_total = self.num_filters * len(self.filter_sizes) * 2
		
		h_pool = tf.concat(pooled_outputs, 2)
		# print(h_pool.get_shape())
		h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
		print(h_pool_flat.get_shape())
		# h_pool_flat = h_pool
		

		out_size = self.out_size

		with tf.variable_scope('first_layer_weights'):
			weight = tf.Variable(tf.truncated_normal([num_filters_total, 100], stddev=0.1))

		with tf.variable_scope('first_layer_Bias'):
			bias = tf.Variable(tf.constant(0.1, shape=[100]))

		hidden = tf.nn.relu(tf.matmul(h_pool_flat, weight) + bias)

		with tf.variable_scope('second_layer_weights'):
			weight2 = tf.Variable(tf.truncated_normal([100, out_size], stddev=0.1))

		with tf.variable_scope('Bias'):
			bias2 = tf.Variable(tf.constant(0.1, shape=[out_size]))
		
		self.prediction = tf.sigmoid(tf.matmul(hidden, weight2) + bias2)
		# print self.prediction.shape
		return self.prediction

	@lazyproperty
	def loss(self):
		#self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.target))
		#self.loss = tf.reduce_mean(tf.contrib.losses.hinge_loss(logits=self.prediction, labels=self.target))
		print ("Prediction: " + str(tf.shape(self.prediction)) + " Target: " + str(tf.shape(self.target)))
		self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=self.prediction, targets=self.target,pos_weight=2))
		return self.loss

	@lazyproperty
	def optimize(self):
		#optimizer = tf.train.AdadeltaOptimizer(learning_rate =0.1)
		self.optimize = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)
		#optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
		#self.optimize = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

		return self.optimize
	@lazyproperty
	def correct(self):
		mistakes = tf.equal(tf.argmax(self.target,1), tf.argmax(self.prediction,1))
		self.correct = tf.reduce_mean(tf.cast(mistakes, tf.float32))
		return self.correct

	@lazyproperty
	def accuracy(self):	
		self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
		return self.accuracy
