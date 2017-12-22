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


gen = lambda: (j for j in [range(10000)])
ds = tf.data.Dataset.from_generator(gen, output_types=tf.int32, output_shapes=tf.TensorShape([None]))  # type: Dataset
ds = ds.batch(10)

value = ds.make_one_shot_iterator().get_next()

sess = tf.Session()
for j in range(1):
    print(sess.run(value))
