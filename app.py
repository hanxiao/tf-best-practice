import itertools

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from model import nade
from model.reader import DataIO
from utils.logger import JobContext
from utils.parameter import AppConfig, ModelParams

tf.logging.set_verbosity(tf.logging.INFO)


def generate(model, data_io: DataIO, out_fn, lang, max_infer_line):
    cur_ln = 0
    eof = False
    while not eof and cur_ln < max_infer_line:
        results_gen = model.predict(
            input_fn=lambda: data_io.output_fn(cur_ln, out_fn, lang))
        infer_line, eof = data_io.decode(list(itertools.islice(results_gen, 1)))
        with open(out_fn, 'a') as fp:
            fp.write(infer_line)
        cur_ln += 1


def main(argv):
    config = AppConfig('settings/config.yaml', argv[1])
    params = ModelParams('settings/params.yaml', argv[2])
    data_io = DataIO(config, params)
    model = tf.estimator.Estimator(model_fn=nade.model_fn, params=params, model_dir=config.model_dir)
    global_step = 0
    while True:
        model.train(input_fn=lambda: data_io.input_fn(ModeKeys.TRAIN), steps=params.train_step)
        global_step += params.train_step
        with JobContext('generating code at step %d...' % global_step, config.logger):
            generate(model, data_io,
                     config.output_path + '-%d.txt' % global_step,
                     'py', params.max_infer_line)


if __name__ == "__main__":
    tf.app.run()
