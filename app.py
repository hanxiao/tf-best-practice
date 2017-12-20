import tensorflow as tf

from utils.reader import DataReader

if __name__ == "__main__":
    data = DataReader()
    sess = tf.Session()
    data.init_train_data_op(sess)
    for _ in range(10):
        print(sess.run(data.X_r))
    pass
