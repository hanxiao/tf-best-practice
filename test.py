import json

import dask.bag as db
import tensorflow as tf

from config import LOGGER
from utils.helper import JobContext

with JobContext('indexing all chars/chatrooms/users...', LOGGER):
    b = db.read_text('data/*/db.json.2017-*-*.gz').map(json.loads)
    msg_stream = b.filter(lambda x: x['msgType'] == 'Text')
    text_stream = msg_stream.pluck('text').distinct()
    chatroom_stream = msg_stream.pluck('chatroomName').distinct()
    user_stream = msg_stream.pluck('fromUser').distinct()

    all_chars = sorted(text_stream.flatten().filter(lambda x: x).distinct())
    all_rooms = sorted(chatroom_stream.compute())
    all_users = sorted(user_stream.compute())

    char_unknown_int = 0
    preserved_ints = 1
    char2int_map = {c: idx + preserved_ints for idx, c in enumerate(all_chars)}

    room_unknown_int = len(char2int_map)
    preserved_ints += len(char2int_map) + 1
    room2int_map = {c: idx + preserved_ints for idx, c in enumerate(all_rooms)}

    user_unknown_int = len(room2int_map)
    preserved_ints += len(room2int_map) + 1
    user2int_map = {c: idx + preserved_ints for idx, c in enumerate(all_users)}

    preserved_ints += len(user2int_map)

max_len = text_stream.map(len).max().compute()
num_sent = text_stream.count().compute()
num_room = chatroom_stream.count().compute()
num_user = user_stream.count().compute()
num_char = len(char2int_map)

LOGGER.info('unique sentences: %d' % num_sent)
LOGGER.info('unique chars: %d' % num_char)
LOGGER.info('unique rooms: %d' % num_room)
LOGGER.info('unique users: %d' % num_user)
LOGGER.info('max sequence length: %d' % max_len)
LOGGER.info('vocabulary size: %d' % preserved_ints)

len_hist = db.from_sequence(sorted(text_stream.map(len).frequencies())) \
    .accumulate(lambda x, y: (y[0], (x[1] * num_sent + y[1]) / num_sent), (0, 0)) \
    .map(lambda x: '%d: %.4f%%' % (x[0], x[1] * 100)).compute()

LOGGER.info('histogram of sent length: %s' % len_hist)

len_threshold = 20
LOGGER.info('maximum length of training sent: %d' % len_threshold)

d = (
    msg_stream.filter(lambda x: len(x['text']) <= len_threshold).map(
        lambda x: ([char2int_map.get(c, char_unknown_int) for c in x['text']] + [0] * (len_threshold - len(x)),
                   room2int_map.get(x['chatroomName'], room_unknown_int),
                   user2int_map.get(x['fromUser'], user_unknown_int))))

sent_ph = tf.placeholder(tf.int32, )

print(d.take(10))
print(d.pluck(0).take(10))
print(d.pluck(1).take(10))
print(d.pluck(2).take(10))
