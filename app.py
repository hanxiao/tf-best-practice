import itertools

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from model import nade
from utils.parameter import AppConfig, ModelParams
from utils.reader import InputData

tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
    config = AppConfig('settings/config.yaml', argv[1])
    params = ModelParams('settings/params.yaml', argv[2])
    input_data = InputData(config, params)
    model = tf.estimator.Estimator(model_fn=nade.model_fn, params=params)
    while True:
        model.train(input_fn=lambda: input_data.input_fn(ModeKeys.TRAIN), steps=1000)
        results_gen = model.predict(input_fn=lambda: input_data.input_fn(ModeKeys.INFER))
        config.logger.info(input_data.decode(list(itertools.islice(results_gen, params.infer_batch_size))))
        # train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_data.input_fn(ModeKeys.TRAIN))
        # eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_data.input_fn(ModeKeys.EVAL))
        # tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == "__main__":
    tf.app.run()
