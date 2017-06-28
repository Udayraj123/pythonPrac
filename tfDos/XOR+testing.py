
# coding: utf-8

# In[1]:


"""
TODO :
test XOR 
    with N layers
    with ReLU & Sigmoid & Softmax
        see gradient gets dead if bias is way too negative
    with Adam vs GradientDescent

"""


# In[2]:


import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import sys

tf.logging.set_verbosity(tf.logging.INFO)

# In[3]:


X = np.array(map(lambda x : map(int,list(x)),("00 01 10 11".split(' '))))
Y = np.reshape(map(int,list("0110")),(4,1))


# In[4]:
from datetime import datetime


class TwoLayerModel():
    def __init__(self,learning_rate=0.01,
		 facW1 = 0.1,		 facW2 = 1,
                 logs_path='/tmp/tensorflow/logs',
                 optimiser='grad'):
        with tf.name_scope('Alpha_'+str(learning_rate)+'_'+optimiser+'_'):
            with tf.name_scope('I/O'):
                self.x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
                self.y_ = tf.placeholder(tf.uint8, shape=[4,1], name="y-input")
           
            with tf.name_scope('Theta'):
                Theta1 = tf.Variable(tf.random_uniform([2,2],-facW1,facW1),name='Theta1')
                Theta2 = tf.Variable(tf.random_uniform([2,1],-facW2,facW2),name='Theta2')
            with tf.name_scope('Bias'):
                Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
                Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")
            with tf.name_scope('MLP'):
	            hidden_out = tf.sigmoid(tf.matmul(self.x_, Theta1) + Bias1)
	            net_h = tf.matmul(hidden_out, Theta2) + Bias2
	            Hypothesis = tf.sigmoid(net_h,name="Hypothesis")
	            self.y_pred = tf.round(Hypothesis)
	            self.y_float = tf.cast(self.y_,tf.float32)
#	            self.y_pred = tf.cond(Hypothesis > tf.constant([0.5]),lambda: tf.constant([1]),lambda: tf.constant([0]))
		#element wise multiplication will happen here between h and y_-
            self.cost = tf.reduce_mean(-1 * ( (self.y_float * tf.log(Hypothesis)) + ((1 - self.y_float) * tf.log(1.0 - Hypothesis)) ) )

            if (optimiser=='grad'):
                self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
            else:
                self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


            self.accuracy = tf.reduce_mean(
                tf.cast(
               tf.equal(self.y_float,self.y_pred),	
                tf.float32)
                ,name="Accuracy")

            self.summary_writer = tf.summary.FileWriter(logs_path+"/"+datetime.now().strftime("%Y%m%d-%H%M%S"), graph=tf.get_default_graph())
            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", self.cost)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", self.accuracy)
            # Merge all summaries into a single op - to pass into sess.run
            self.merged_summary_op = tf.summary.merge_all()



        
    
    def train(self,sess,x,y,index):
        _,c,acc,summary = sess.run([self.train_step,self.cost,self.accuracy,self.merged_summary_op],feed_dict={self.x_:x,self.y_:y})
        self.summary_writer.add_summary(summary,index)
        return c,acc



def printbuf(x):
	sys.stdout.write(x)
	sys.stdout.write('\r')
with tf.variable_scope('model1'):
	model1=TwoLayerModel(1,optimiser="grad")  
	
with tf.variable_scope('model2'):
	model2=TwoLayerModel(0.1,optimiser="grad")


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	iters = 1000000
	min_count=100
	count=0
	print('')
	for i in range(iters):
		c,acc =model1.train(sess,X,Y,i)
		
		c2,acc2 =model2.train(sess,X,Y,i)
		print('Iteration : ' +str(i) + ' Cost : ' +str(c) + ' Accuracy : ' +str(acc) + 
		' \t Iteration : ' +str(i) + ' Cost : ' +str(c2) + ' Accuracy : ' +str(acc2))
		if(acc==1):
			count+=1
			if( count==min_count):
				break	
		else:
			count=0
		#print(model2.train(X,Y))
	print('')

	# In[ ]:
      
