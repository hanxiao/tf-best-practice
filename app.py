import tensorflow as tf

from config import MODEL_CONFIG
from model import nade_estimator
from utils.reader import InputData

if __name__ == "__main__":
    input_data = InputData()
    MODEL_CONFIG.add_hparam('num_char', input_data.num_char)
    model = tf.estimator.Estimator(model_fn=nade_estimator.model_fn, params=MODEL_CONFIG)
    model.train(input_data.train_input_fn)
    # model = NADE(InputData())
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(model.logp_loss))
    # print(sess.run(model.xentropy_loss))
    # print(sess.run(model.X_sampled))
