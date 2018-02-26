import re
from glob import glob
from string import punctuation

import dask.bag as db
import tensorflow as tf
from ruamel.yaml import YAML
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.data import Dataset

import shared
from utils.helper import flatten
from utils.logger import JobContext
from utils.parameter import AppConfig, ModelParams


class DataIO:

    def __init__(self, config: AppConfig, params: ModelParams):
        # copy yaml to model dir
        config.copyto(config.model_dir)
        params.copyto(config.model_dir)

        logger = shared.logger

        self.batch_size = params.batch_size
        self.num_epoch = params.num_epoch
        self.pattern = re.compile(r'(\s+|[{}])'.format(re.escape(punctuation)))
        self.unknown_char_idx = params.reserved_char.unknown
        self.start_line_idx = params.reserved_char.start
        self.eof_idx = params.reserved_char.end
        self.unknown_lang_idx = params.reserved_char.unknown

        with JobContext('indexing all codes...'):
            b = db.read_text([config.data_dir + '*.' + v for v in config.all_langs.values().values()])
            tokens = b.map(lambda x: self.tokenize(x)).flatten()

            # get frequent tokens with length > 1
            freq_tokens = tokens.frequencies().filter(
                lambda x: len(x[0]) > 1).topk(params.freqword_as_char, lambda x: x[1]).map(lambda x: x[0]).compute()

            # get all characters
            all_chars = b.flatten().distinct().filter(lambda x: x).compute()

            self.char2int = {c: idx for idx, c in
                             enumerate(all_chars + freq_tokens, start=len(params.reserved_char.values()))}
            self.lang2int = {c: idx for idx, c in
                             enumerate(config.all_langs.values(), start=len(params.reserved_lang.values()))}
            self.end_line_idx = self.char2int['\n']

        with JobContext('computing some statistics...'):
            self.num_line = b.count().compute()
            self.num_char = max(self.char2int.values()) + 1
            self.num_lang = max(self.lang2int.values()) + 1
            logger.info('# lines: %d' % self.num_line)
            logger.info('# chars: %d (# reserved: %d)' % (self.num_char, len(params.reserved_char.values())))
            logger.info('# langs: %d (# reserved: %d)' % (self.num_lang, len(params.reserved_lang.values())))
            logger.info('linebreak idx: %d' % self.end_line_idx)

        self.dump(config.dataio_path)
        self.after_init(config, params)

    def after_init(self, config: AppConfig, params: ModelParams):
        self.int2char = {i: c for c, i in self.char2int.items()}
        self.int2lang = {i: c for c, i in self.lang2int.items()}

        params.add_hparam('num_char', self.num_char)
        params.add_hparam('num_lang', self.num_lang)
        params.add_hparam('start_line_idx', self.start_line_idx)

        with JobContext('building data generator...'):
            self.build_infer_fn(config, params)
            self.build_train_fn(config, params)

    def input_fn(self, mode: ModeKeys):
        return {
                   ModeKeys.TRAIN:
                       lambda: self.train_ds.repeat(self.num_epoch).padded_batch(self.batch_size,
                                                                                 padded_shapes=self.output_shapes),
                   ModeKeys.EVAL:
                       lambda: self.eval_ds.padded_batch(self.batch_size, padded_shapes=self.output_shapes),
                   ModeKeys.INFER: lambda: Dataset.range(1)
               }[mode]().make_one_shot_iterator().get_next(), None

    def output_fn(self, cur_ln, out_fn, lang):
        self.infer_cur_ln = cur_ln
        self.infer_out_fn = out_fn
        if lang in self.lang2int:
            self.infer_lang = lang
        else:
            raise ValueError('inference %s is not supported!' % lang)

        return self.infer_ds.padded_batch(1,
                                          padded_shapes=self.infer_output_shapes).make_one_shot_iterator().get_next(), None

    def decode(self, predictions):
        r = []
        eof = False
        for j in predictions[0]:
            if j == self.eof_idx:
                eof = True
                break
            elif j == self.end_line_idx:
                break
            elif j == self.start_line_idx:
                r.append('<START>')
            elif j == self.unknown_char_idx:
                r.append('<UNKNOWN>')
            else:
                r.append(self.int2char[j])
        if r and r[-1] != self.end_line_idx:
            r.append('\n')
        return ''.join(r), eof

    def build_train_fn(self, config: AppConfig, params: ModelParams):
        def gen():
            file_list = [(w, v) for v in config.all_langs.values() for w in
                         glob(config.data_dir + '*.' + v, recursive=True)]
            for f, lang in file_list:
                with open(f) as fp:
                    linebreak = list(range(params.context_lines))
                    context = [[self.start_line_idx]] * params.context_lines
                    all_lines = fp.readlines()
                    for line in all_lines:
                        next_line = self.tokenize_by_keywords(line)
                        context_line = flatten(context)
                        yield context_line, next_line, \
                              len(context_line), len(next_line), \
                              self.lang2int.get(lang, self.unknown_lang_idx), \
                              flatten(linebreak)

                        context.append(next_line)
                        context.pop(0)
                        linebreak.append(linebreak[-1] + len(next_line))
                        linebreak = [v - linebreak[0] - 1 for v in linebreak]
                        linebreak.pop(0)

                    context_line = flatten(context)
                    yield context_line, [self.eof_idx], \
                          len(context_line), 1, \
                          self.lang2int.get(lang, self.unknown_lang_idx), \
                          linebreak

        self.output_shapes = ([None], [None], [], [], [], [None])
        self.output_types = (tf.int32,) * len(self.output_shapes)
        ds = Dataset.from_generator(generator=gen,
                                    output_types=self.output_types,
                                    output_shapes=self.output_shapes).shuffle(
            buffer_size=params.batch_size * 100)  # type: Dataset
        self.eval_ds = ds.take(params.num_eval)
        self.train_ds = ds.skip(params.num_eval)

    def build_infer_fn(self, config: AppConfig, params: ModelParams):
        def gen():
            linebreaks = list(range(params.context_lines))
            context = [[self.start_line_idx]] * params.context_lines

            with open(self.infer_out_fn) as fp:
                all_lines = [self.tokenize_by_keywords(v) for v in fp.readlines()]
                if len(all_lines) > 0:
                    context += all_lines
                    linebreaks = []
                    for v in context:
                        if linebreaks:
                            linebreaks.append(len(v) + linebreaks[-1])
                        else:
                            linebreaks.append(0)
                    context = context[-params.context_lines:]
                    linebreaks = [v - linebreaks[-params.context_lines - 1] - 1 for v in linebreaks]
                    linebreaks = linebreaks[-params.context_lines:]

                context_line = flatten(context)

                shared.logger.info('context: %s' % context)

                yield context_line, len(context_line), \
                      self.lang2int.get(self.infer_lang, self.unknown_lang_idx), \
                      linebreaks

        self.infer_output_shapes = ([None], [], [], [None])
        self.infer_output_types = (tf.int32,) * len(self.infer_output_shapes)
        self.infer_ds = Dataset.from_generator(generator=gen,
                                               output_types=self.infer_output_types,
                                               output_shapes=self.infer_output_shapes)

    def tokenize(self, line):
        return [p for p in self.pattern.split(line) if p]

    def tokenize_by_keywords(self, line):
        tokens = self.tokenize(line)
        kw = self.char2int
        return flatten([kw.get(t, [kw.get(c, self.unknown_char_idx) for c in t]) for t in tokens])

    def dump(self, fn):
        yaml = YAML(typ='unsafe', pure=True)
        yaml.register_class(DataIO)
        with open(fn, 'w') as fp:
            yaml.dump(self, fp)
        shared.logger.info('DataIO is dumped to %s' % fn)