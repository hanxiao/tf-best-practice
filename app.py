import tensorflow as tf

from model.nade import NADE
from utils.reader import InputData

if __name__ == "__main__":
    model = NADE(InputData())
    sess = tf.Session()
    print(sess.run(model.logp_loss))
