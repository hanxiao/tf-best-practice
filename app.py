import itertools

import tensorflow as tf
from ruamel.yaml import YAML
from tensorflow.contrib.learn import ModeKeys, Estimator

import shared
from bilevel import seq2seq
from bilevel.dataio import DataIO
from utils.helper import touch
from utils.logger import JobContext
from utils.parameter import AppConfig, ModelParams

tf.logging.set_verbosity(tf.logging.INFO)


def generate(model: Estimator, data_io: DataIO, out_fn, lang, max_infer_line):
    cur_ln = 0
    eof = False
    touch(out_fn)
    while not eof and cur_ln < max_infer_line:
        results_gen = model.predict(
            input_fn=lambda: data_io.output_fn(cur_ln, out_fn, lang))
        infer_line, eof = data_io.decode(list(itertools.islice(results_gen, 1)))
        with open(out_fn, 'a') as fp:
            fp.write(infer_line)
        cur_ln += 1


def parse_arg(argv):
    if len(argv) > 3:
        config = AppConfig(argv[3] + '/config.yaml', argv[1], argv[3].split('/')[-1])
        params = ModelParams(argv[3] + '/params.yaml', argv[2])
        yaml = YAML(typ='unsafe', pure=True)
        yaml.register_class(DataIO)
        with open(argv[3] + '/dataio.yaml') as fp:
            data_io = yaml.load(fp)  # type: DataIO
            data_io.after_init(config, params)
        shared.logger.info('recovered from %s' % argv[3])
    else:
        config = AppConfig('settings/config.yaml', argv[1])
        params = ModelParams('settings/params.yaml', argv[2])
        data_io = DataIO(config, params)

    shared.logger.info('configuration loaded!')
    return config, params, data_io


def main(argv):
    config, params, data_io = parse_arg(argv)
    model = tf.estimator.Estimator(model_fn=seq2seq.model_fn, params=params, model_dir=config.model_dir)

    global_step = 0
    while True:
        model.train(input_fn=lambda: data_io.input_fn(ModeKeys.TRAIN), steps=params.train_step)
        global_step += params.train_step
        with JobContext('generating code at step %d...' % global_step):
            generate(model, data_io,
                     config.output_path + '-%d.txt' % global_step,
                     'py', params.max_infer_line)


if __name__ == "__main__":
    tf.app.run()
