import numpy as np
import tensorflow as tf

from utils.reader import load_dataset

a = load_dataset()

sent_ds = tf.data.Dataset.from_tensor_slices(a['data_sent'])
room_ds = tf.data.Dataset.from_tensor_slices(a['data_room'])
user_ds = tf.data.Dataset.from_tensor_slices(a['data_user'])

dataset = tf.data.Dataset.zip((sent_ds, room_ds, user_ds))

data1 = np.transpose(np.tile(list(range(10000)), [2, 1]))
data2 = -data1

D1 = tf.placeholder(tf.int32, shape=[None, None])
D2 = tf.placeholder(tf.int32, shape=[None, None])
dataset1 = tf.data.Dataset.from_tensor_slices(D1)  # type: tf.data.Dataset
dataset2 = tf.data.Dataset.from_tensor_slices(D2)
dataset = tf.data.Dataset.zip((dataset1, dataset2))
dataset = dataset.repeat()  # type: tf.data.Dataset
dataset = dataset.batch(32)  # type: tf.data.Dataset

iterator = dataset.make_initializable_iterator()

(ne1, ne2) = iterator.get_next()

training_init_op = iterator.make_initializer(dataset)

sess = tf.Session()

# Initialize an iterator over a dataset with 10 elements.
sess.run(training_init_op, feed_dict={D1: data1, D2: data2})
for _ in range(10):
    value = sess.run(ne1)
    print(value)

# Initialize the same iterator over a dataset with 100 elements.
sess.run(training_init_op, feed_dict={D1: data2, D2: data1})
for _ in range(100):
    value = sess.run(ne2)
    print(value)
