import memory_util
memory_util.vlog(1)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.Session()
# OOM
a = tf.random_uniform((1000,))
b = tf.random_uniform((1000,))
c = a + b
with memory_util.capture_stderr() as stderr:
    sess.run(c.op)
memory_util.print_memory_timeline(stderr, ignore_less_than_bytes=1000)