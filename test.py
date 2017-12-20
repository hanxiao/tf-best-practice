import json

import dask.bag as db
import numpy as np

from app import LOGGER
from utils.helper import JobContext

with JobContext('indexing all chars...', LOGGER):
    b = db.read_text('data/*/db.json.2017-*-*.gz').map(json.loads)
    text_stream = b.pluck('text', '').filter(lambda x: x).distinct()
    all_chars = sorted(text_stream.flatten().filter(lambda x: x).distinct())
    char2int_map = {c: idx + 1 for idx, c in enumerate(all_chars)}

max_len = text_stream.map(len).max().compute()
num_sent = text_stream.count().compute()
num_char = len(char2int_map)
set_max_len = 20

LOGGER.info('num unique sentences: %d' % num_sent)
LOGGER.info('num unique chars: %d' % num_char)
LOGGER.info('max sequence length: %d' % max_len)

len_hist = db.from_sequence(sorted(text_stream.map(len).frequencies())) \
    .accumulate(lambda x, y: (y[0], (x[1] * num_sent + y[1]) / num_sent), (0, 0)) \
    .map(lambda x: '%d: %.4f%%' % (x[0], x[1] * 100)).compute()

LOGGER.info('histogram of sent length: %s' % len_hist)
LOGGER.info('maximum length of training sent: %d' % set_max_len)

d = b.pluck('text', '').filter(lambda x: len(x) <= set_max_len) \
    .map(lambda x: [char2int_map.get(c, 0) for c in x] + [0] * (set_max_len - len(x)))
print(np.array(list(d)))
