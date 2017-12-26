import itertools

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from config import MODEL_PARAM, LOGGER
from model import nade
from utils.reader import InputData

tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
    input_data = InputData()
    model = tf.estimator.Estimator(model_fn=nade.model_fn, params=MODEL_PARAM)
    # train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_data.input_fn(ModeKeys.TRAIN))
    # eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_data.input_fn(ModeKeys.EVAL))
    # tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    while True:
        model.train(input_fn=lambda: input_data.input_fn(ModeKeys.TRAIN), steps=1000)
        results_gen = model.predict(input_fn=lambda: input_data.input_fn(ModeKeys.INFER))
        LOGGER.info(input_data.decode(list(itertools.islice(results_gen, MODEL_PARAM.infer_batch_size))))


if __name__ == "__main__":
    tf.app.run()
