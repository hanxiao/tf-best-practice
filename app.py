import tensorflow as tf

from model.nade import NADE
from utils.reader import InputData

if __name__ == "__main__":
    model = NADE(InputData())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(model.logp_loss))
    print(sess.run(model.xentropy_loss))
    print(sess.run(model.X_sampled))
