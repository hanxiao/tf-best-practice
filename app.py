import tensorflow as tf

from config import MODEL_PARAM
from model import nade
from utils.reader import InputData

tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
    input_data = InputData()
    model = tf.estimator.Estimator(model_fn=nade.model_fn, params=MODEL_PARAM)
    model.train(input_data.train_input_fn)


if __name__ == "__main__":
    tf.app.run()
