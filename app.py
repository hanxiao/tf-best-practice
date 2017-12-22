import tensorflow as tf

from model.nade import NADE
from utils.reader import InputData

if __name__ == "__main__":
    data = InputData()
    model = NADE(data)
    sess = tf.Session()
    print(sess.run(model.logp_loss))
