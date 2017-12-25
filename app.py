import tensorflow as tf

from config import MODEL_PARAM
from model import nade
from utils.reader import InputData

tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
    input_data = InputData()
    model = tf.estimator.Estimator(model_fn=nade.model_fn, params=MODEL_PARAM)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_data.input_fn('train'))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_data.input_fn('eval'))
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    model.predict()


if __name__ == "__main__":
    tf.app.run()
