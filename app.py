import tensorflow as tf

from utils.reader import DataReader

if __name__ == "__main__":
    data = DataReader()
    sess = tf.Session()
    data.init_train_data_op(sess)
    for _ in range(11):
        print(sess.run([data.X_s, data.X_r, data.X_u]))
    pass
