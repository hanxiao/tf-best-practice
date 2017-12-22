import tensorflow as tf
from tensorflow.python.data import Dataset


# print(db.from_sequence(range(1000000)).repartition(1).to_delayed().pop(0))
#
# exit()
# def gen():
#     b = db.from_sequence([j for j in range(1000)]).to_delayed()
#     for i in itertools.count(0):
#         print(b.pop(i).compute())
#         yield b.pop(i).compute()


def gen():
    for i in range(5):
        yield [i, i + 1] * i


def gen1():
    for i in range(5):
        yield i


def gen2():
    for i in range(30, 40):
        yield i * 3


ds0 = tf.data.Dataset.from_generator(gen, output_types=tf.int32, output_shapes=[None])  # type: Dataset
ds1 = tf.data.Dataset.from_generator(gen1, output_types=tf.int32, output_shapes=[])  # type: Dataset
ds2 = tf.data.Dataset.from_generator(gen2, output_types=tf.int32, output_shapes=[])  # type: Dataset

ds = tf.data.Dataset.zip((ds0, ds1, ds2)).shuffle(buffer_size=3).repeat().padded_batch(2, padded_shapes=(
    [None], [], []))  # type: Dataset

v1, v2, v3 = ds.make_one_shot_iterator().get_next()

sess = tf.Session()
for j in range(5):
    print(sess.run([v1, v2, v3]))
