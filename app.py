import tensorflow as tf

from utils.reader import InputData

if __name__ == "__main__":
    data = InputData()
    sess = tf.Session()
    for _ in range(11):
        print(sess.run([data.X_s, data.X_r, data.X_u]))
    pass
