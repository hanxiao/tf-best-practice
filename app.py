import tensorflow as tf

from config import MODEL_CONFIG
from model import nade
from utils.reader import InputData

tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
    input_data = InputData()
    MODEL_CONFIG.add_hparam('num_char', input_data.num_char)
    model = tf.estimator.Estimator(model_fn=nade.model_fn, params=MODEL_CONFIG)

    tensors_to_log = {"xtropy_loss": "Output/xtropy_loss"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=100)
    model.train(input_data.train_input_fn, hooks=[logging_hook])


if __name__ == "__main__":
    tf.app.run()
