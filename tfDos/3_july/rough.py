import tensorflow as tf

y = tf.placeholder(tf.int64, [2,3])
y_o = tf.one_hot(indices = y,
	depth = 4,
	axis = -1)
temp = [[1, 2, -1], [3, -1, 1]]
y_dict = {y:temp}
weights = tf.Variable(
	tf.random_uniform(shape=[2, 6, 4], 
		maxval=1, 
		dtype=tf.float64))
weights_p_l = [] 
for i in range(2):
	weights_p_l.append(tf.reduce_max(weights[:,i:i+5,:], axis = 1))
weights_p = tf.stack(weights_p_l, axis = 1)
weights_c = tf.slice(weights, [0, 1, 0], [2, 4, 4])
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

print(sess.run(y, feed_dict=y_dict)) 
print(sess.run(y_o, feed_dict=y_dict)) 
w, w_p, w_c = (sess.run([weights, weights_p, weights_c], feed_dict=y_dict)) 
print(w, "\n")
print(w_p, "\n")
print(w_c, "\n")

