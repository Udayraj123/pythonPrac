import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)
# Enter an interactive TensorFlow Session.
sess = tf.InteractiveSession()

#CNN needs a lot of weights and biases.
# noise in weights for symmetry breaking, and to prevent 0 gradients
#Also, ReLU requires slight positive bias to avoid dead neurons
#These functions will be handy to avoid repetitions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #SAME means zero padding

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
# 2nd & 3rd dimensions to image width and height, and the final dimension corresponding to the number of color channels.

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#None indicates unknown size (no of images is variable)

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# First need to initialize these global vars
sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# tf's each err function will have its own optimisations in the below functions
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    train_batch = mnist.train.next_batch(100) # m=100 in this iter.
#     train
    train_step.run(feed_dict={x: train_batch[0], y_: train_batch[1]})
#     feed_dict can replace ANY TENSOR in the CG, not just placeholders.(like W)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# above acc of train data

print('train',accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
print('test',accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

