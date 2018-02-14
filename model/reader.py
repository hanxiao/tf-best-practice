import re
from glob import glob
from string import punctuation
from typing import Dict

import dask.bag as db
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.data import Dataset

from utils.helper import flatten
from utils.logger import JobContext
from utils.parameter import AppConfig, ModelParams


class DataIO:
    def __init__(self, config: AppConfig, params: ModelParams):
        logger = config.logger
        self.pattern = re.compile(r'(\s+|[{}])'.format(re.escape(punctuation)))
        self.unknown_char_idx = params.reserved_char['unknown']
        self.start_char_idx = params.reserved_char['start']
        self.end_char_idx = params.reserved_char['end']
        self.unknown_lang_idx = params.reserved_char['unknown']
        self.int2reserved_char = {params.reserved_char[k]: k for k in params.reserved_char.keys()}

        with JobContext('indexing all codes...', logger):
            b = db.read_text([config.data_dir + '*.' + v for v in config.all_langs.values()])
            tokens = b.map(lambda x: self.tokenize(x)).flatten()

            # get frequent tokens with length > 1
            freq_tokens = tokens.frequencies().filter(
                lambda x: len(x[0]) > 1).topk(params.freqword_as_char, lambda x: x[1]).map(lambda x: x[0]).compute()

            # get all characters
            all_chars = b.flatten().distinct().filter(lambda x: x).compute()

            char2int_map = {c: idx for idx, c in enumerate(all_chars + freq_tokens, start=len(params.reserved_char))}
            lang2int_map = {c: idx for idx, c in enumerate(config.all_langs.values(), start=len(params.reserved_lang))}

        with JobContext('computing some statistics...', logger):
            num_line = b.count().compute()
            num_char = max(char2int_map.values()) + 1
            num_lang = max(lang2int_map.values()) + 1
            logger.info('# lines: %d' % num_line)
            logger.info('# chars: %d (# reserved: %d)' % (num_char, len(params.reserved_char)))
            logger.info('# langs: %d (# reserved: %d)' % (num_lang, len(params.reserved_lang)))

        with JobContext('building data generator...', logger):
            def infer():
                context = [[self.start_char_idx]] * params.context_lines
                if self.infer_cur_ln > 0:
                    with open(self.infer_out_fn) as fp:
                        all_lines = context + [self.tokenize_by_keywords(v, char2int_map) for v in fp.readlines()[
                                                                                                   :self.infer_cur_ln]]
                        context = all_lines[-params.context_lines:]
                logger.info('context: %s' % context)
                context_line = flatten(context)
                yield context_line, \
                      lang2int_map.get(self.infer_lang, self.unknown_lang_idx)

            self.infer_output_shapes = ([None], [])
            self.infer_output_types = (tf.int32,) * len(self.infer_output_shapes)
            self.infer_ds = Dataset.from_generator(generator=infer,
                                                   output_types=self.infer_output_types,
                                                   output_shapes=self.infer_output_shapes)

            def gen():
                file_list = [(w, v) for v in config.all_langs.values() for w in
                             glob(config.data_dir + '*.' + v, recursive=True)]
                for f, lang in file_list:
                    with open(f) as fp:
                        context = [[self.start_char_idx]] * params.context_lines
                        all_lines = fp.readlines()
                        for line in all_lines:
                            next_line = self.tokenize_by_keywords(line, char2int_map)
                            context_line = flatten(context)
                            yield context_line, next_line, \
                                  len(context_line), len(next_line), \
                                  lang2int_map.get(lang, self.unknown_lang_idx)
                            context.append(next_line)
                            context.pop(0)
                        context_line = flatten(context)
                        yield context_line, [self.end_char_idx], \
                              len(context_line), 1, \
                              lang2int_map.get(lang, self.unknown_lang_idx)

            self.output_shapes = ([None], [None], [], [], [])
            self.output_types = (tf.int32,) * len(self.output_shapes)
            ds = Dataset.from_generator(generator=gen,
                                        output_types=self.output_types,
                                        output_shapes=self.output_shapes).shuffle(
                buffer_size=params.batch_size * 100)  # type: Dataset
            self.eval_ds = ds.take(params.num_eval)
            self.train_ds = ds.skip(params.num_eval)

        self.num_char = num_char
        self.num_line = num_line
        self.num_lang = num_lang
        self.char2int = char2int_map
        self.lang2int = lang2int_map
        self.int2char = {i: c for c, i in char2int_map.items()}
        self.int2lang = {i: c for c, i in lang2int_map.items()}
        params.add_hparam('num_char', num_char)
        params.add_hparam('num_lang', num_lang)
        self.params = params

        logger.info('data loading finished!')

    def input_fn(self, mode: ModeKeys):
        return {
                   ModeKeys.TRAIN:
                       lambda: self.train_ds.repeat(self.params.num_epoch).padded_batch(self.params.batch_size,
                                                                                        padded_shapes=self.output_shapes),
                   ModeKeys.EVAL:
                       lambda: self.eval_ds.padded_batch(self.params.batch_size, padded_shapes=self.output_shapes),
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
            if j == self.end_char_idx:
                eof = True
                break
            elif j == self.start_char_idx:
                r.append('<START>')
            elif j == self.unknown_char_idx:
                r.append('<UNKNOWN>')
            else:
                r.append(self.int2char[j])
        return ''.join(r), eof

    def tokenize(self, line):
        return [p for p in self.pattern.split(line) if p]

    def tokenize_by_keywords(self, line, keywords: Dict[str, int]):
        tokens = self.tokenize(line)
        return flatten([keywords.get(t, [keywords.get(c, self.unknown_char_idx) for c in t]) for t in tokens])
