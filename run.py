import numpy as np
import tensorflow as tf
from tensorflow.python.ops.distributions.categorical import Categorical

num_char = 92
output_size = 10

output_layer_info = {
    'units': num_char,  # this is the size of vocabulary
    'name': 'out_to_character',
    # linear 'activation': tf.nn.softmax
}

with tf.variable_scope('Shared_Dense', reuse=False) as dense_layer_scope:
    # this will be replaced by the cell_output later
    zeros_placeholder = tf.zeros([1, output_size])
    tf.layers.dense(zeros_placeholder, **output_layer_info)


def get_logits(cell_out):
    # cell_out should be BxH
    # dense layer is HxNUM_CHAR
    # logit output is BxNUM_CHAR
    with tf.variable_scope(dense_layer_scope, reuse=True):
        return tf.layers.dense(cell_out, **output_layer_info)


def get_dist(cell_out):
    return Categorical(logits=get_logits(cell_out),
                       name='categorical_dist',
                       allow_nan_stats=False,
                       dtype=tf.int32)


def get_prob(cell_out, obs):
    # get_dist output is BxNUM_CHAR
    return get_dist(cell_out).prob(obs)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    a = tf.random_normal([100, output_size])
    b = np.random.randint(low=0, high=92, size=100)
    c = [1] * 100
    print(sess.run(tf.shape(a)))
    print(sess.run(get_logits(a)))
    print(sess.run(tf.shape(get_logits(a))))
    print(sess.run(get_dist(a).sample()))
    print(sess.run(get_dist(a).prob(c)))
