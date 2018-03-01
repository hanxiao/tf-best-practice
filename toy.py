import numpy as np
import tensorflow as tf
from tensorflow.python.ops.distributions.categorical import Categorical

a = tf.constant(np.random.rand(10, 5))
b = tf.argmax(a, axis=1)
c = Categorical(logits=a,
                name='categorical_dist',
                allow_nan_stats=False,
                dtype=tf.int32)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(tf.expand_dims(b, 1)))
    print(sess.run(c.sample()))
