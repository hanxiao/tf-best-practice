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


class InputData:
    def __init__(self, config: AppConfig, params: ModelParams):
        logger = config.logger

        logger.info('maximum length of training sent: %d' % params.len_threshold)

        self.pattern = re.compile(r'(\s+|[{}])'.format(re.escape(punctuation)))
        self.unknown_char_idx = params.special_char['unknown']
        self.start_char_idx = params.special_char['start']
        self.end_char_idx = params.special_char['end']
        self.unknown_lang_idx = params.special_char['unknown']

        with JobContext('indexing all codes...', logger):
            b = db.read_text([config.data_dir + '*.' + v for v in config.all_langs.values()])
            tokens = b.map(lambda x: self.tokenize(x)).flatten()

            # get frequent tokens with length > 1
            freq_tokens = tokens.frequencies().filter(
                lambda x: len(x[0]) > 1).topk(params.freqword_as_char, lambda x: x[1]).map(lambda x: x[0]).compute()

            # get all characters
            all_chars = b.flatten().distinct().filter(lambda x: x).compute()

            char2int_map = {c: idx for idx, c in enumerate(all_chars + freq_tokens, start=len(params.special_char))}
            lang2int_map = {c: idx for idx, c in enumerate(config.all_langs.values(), start=len(params.special_lang))}

        with JobContext('computing some statistics...', logger):
            num_line = b.count().compute()
            num_char = len(char2int_map) + 1
            num_lang = len(lang2int_map) + 1
            logger.info('# lines: %d' % num_line)
            logger.info('# chars: %d' % num_char)
            logger.info('# langs: %d' % num_lang)

        with JobContext('building data generator...', logger):
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

    def decode(self, predictions):
        results = []
        for p in predictions:
            r = []
            for j in p:
                if j == 0:
                    break
                else:
                    r.append(self.int2char[j])
            results.append(''.join(r))
        return results

    def tokenize(self, line):
        return [p for p in self.pattern.split(line) if p]

    def tokenize_by_keywords(self, line, keywords: Dict[str, int]):
        tokens = self.tokenize(line)
        return flatten([keywords.get(t, [keywords.get(c, self.unknown_char_idx) for c in t]) for t in tokens])
