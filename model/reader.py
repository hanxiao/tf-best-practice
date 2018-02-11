import re
from glob import glob
from string import punctuation

import dask.bag as db
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.data import Dataset

from utils.logger import JobContext
from utils.parameter import AppConfig, ModelParams


class InputData:
    def __init__(self, config: AppConfig, params: ModelParams):
        logger = config.logger

        logger.info('maximum length of training sent: %d' % params.len_threshold)

        self.pattern = re.compile(r'(\s+|[{}])'.format(re.escape(punctuation)))
        self.char_based = params.char_based

        with JobContext('indexing all codes...', logger):
            b = db.read_text([config.data_dir + '*.' + v for v in config.all_langs.values()])

            all_chars = b.map(lambda x: self.tokenize(x)).flatten().distinct().filter(
                lambda x: x).compute()
            unknown_char_idx = 0
            char2int_map = {c: idx for idx, c in enumerate(all_chars, start=1)}

            unknown_lang_idx = 0
            lang2int_map = {c: idx for idx, c in enumerate(config.all_langs.values(), start=1)}

        with JobContext('computing some statistics...', logger):
            num_line = b.count().compute()
            num_char = len(char2int_map) + 1
            num_lang = len(config.all_langs) + 1
            logger.info('# lines: %d' % num_line)
            logger.info('# chars: %d' % num_char)
            logger.info('# langs: %d' % num_lang)

        with JobContext('building data generator...', logger):
            def gen():
                file_list = [(w, v) for v in config.all_langs.values() for w in
                             glob(config.data_dir + '*.' + v, recursive=True)]
                for f, lang in file_list:
                    with open(f) as fp:
                        all_lines = fp.readlines()
                        total_lines = len(all_lines)
                        for ln, line in enumerate(all_lines):
                            c = line
                            window_len = 1
                            while (ln + window_len) < total_lines and len(
                                            c + all_lines[ln + window_len]) < params.len_threshold:
                                c += all_lines[ln + window_len]
                                window_len += 1

                            c = self.tokenize(c)
                            yield [char2int_map.get(cc, unknown_char_idx) for cc in c], \
                                  len(c), \
                                  lang2int_map.get(lang, unknown_lang_idx)

            self.output_shapes = ([None], [], [])
            ds = Dataset.from_generator(generator=gen, output_types=(tf.int32, tf.int32, tf.int32),
                                        output_shapes=self.output_shapes).shuffle(buffer_size=10000)  # type: Dataset
            self.eval_ds = ds.take(params.num_eval)
            self.train_ds = ds.skip(params.num_eval)

        self.num_char = num_char
        self.num_line = num_line
        self.num_lang = num_lang
        self.char2int = char2int_map
        self.lang2int = lang2int_map
        self.int2char = {i: c for c, i in char2int_map.items()}
        self.int2lang = {i: c for c, i in lang2int_map.items()}
        self.unknown_char_idx = unknown_char_idx
        self.unknown_lang_idx = unknown_lang_idx

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
        return [p for p in self.pattern.split(line) if p] if not self.char_based else line
