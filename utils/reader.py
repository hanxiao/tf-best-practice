import itertools
import json

import dask.bag as db
import tensorflow as tf
from tensorflow.python.data import Dataset

from config import APP_CONFIG, LOGGER, MODEL_CONFIG
from .logger import JobContext


class InputData:
    def __init__(self):
        with JobContext('indexing all chars/chatrooms/users...', LOGGER):
            b = db.read_text(APP_CONFIG.data_file).map(json.loads)
            msg_stream = b.filter(lambda x: x['msgType'] == 'Text')
            text_stream = msg_stream.pluck('text').distinct()
            chatroom_stream = msg_stream.pluck('chatroomName').distinct()
            user_stream = msg_stream.pluck('fromUser').distinct()

            all_chars = sorted(text_stream.flatten().filter(lambda x: x).distinct())
            all_rooms = sorted(chatroom_stream.compute())
            all_users = sorted(user_stream.compute())

            unknown_char_idx = 0
            reserved_chars = 1
            char2int_map = {c: idx + reserved_chars for idx, c in enumerate(all_chars)}

            unknown_room_idx = len(char2int_map)
            reserved_chars += len(char2int_map) + 1
            room2int_map = {c: idx + reserved_chars for idx, c in enumerate(all_rooms)}

            unknown_user_idx = len(room2int_map)
            reserved_chars += len(room2int_map) + 1
            user2int_map = {c: idx + reserved_chars for idx, c in enumerate(all_users)}

            reserved_chars += len(user2int_map)

        with JobContext('computing some statistics...', LOGGER):
            max_len = text_stream.map(len).max().compute()
            num_sent = text_stream.count().compute()
            num_room = chatroom_stream.count().compute()
            num_user = user_stream.count().compute()
            num_char = len(char2int_map)
            len_hist = db.from_sequence(sorted(text_stream.map(len).frequencies())) \
                .accumulate(lambda x, y: (y[0], (x[1] * num_sent + y[1]) / num_sent), (0, 0)) \
                .map(lambda x: '%d: %.4f%%' % (x[0], x[1] * 100)).compute()

            LOGGER.info('unique sentences: %d' % num_sent)
            LOGGER.info('unique chars: %d' % num_char)
            LOGGER.info('unique rooms: %d' % num_room)
            LOGGER.info('unique users: %d' % num_user)
            LOGGER.info('max sequence length: %d' % max_len)
            LOGGER.info('vocabulary size: %d' % reserved_chars)
            LOGGER.info('histogram of sent length: %s' % len_hist)
            LOGGER.info('maximum length of training sent: %d' % MODEL_CONFIG.len_threshold)

        with JobContext('building dataset...', LOGGER):
            d = (
                msg_stream.filter(lambda x: len(x['text']) <= MODEL_CONFIG.len_threshold).map(
                    lambda x: (
                        [char2int_map.get(c, unknown_char_idx) for c in x['text']],
                        room2int_map.get(x['chatroomName'], unknown_room_idx),
                        user2int_map.get(x['fromUser'], unknown_user_idx))))

            X = d.pluck(0).compute(), d.pluck(1).compute(), d.pluck(2).compute()

            def gen():
                for i in itertools.count(0):
                    yield X[0][i], X[1][i], X[2][i]

            self.ds = Dataset.from_generator(generator=gen, output_types=(tf.int32, tf.int32, tf.int32),
                                             output_shapes=([None], [], []))  # type: Dataset

        self.num_char = num_char
        self.num_reserved_char = reserved_chars
        self.num_sent = num_sent
        self.num_room = num_room
        self.num_user = num_user
        self.max_len = max_len
        self.char2int = char2int_map
        self.user2int = user2int_map
        self.room2int = room2int_map
        self.unknown_char_idx = unknown_char_idx
        self.unknown_user_idx = unknown_user_idx
        self.unknown_room_idx = unknown_room_idx

        LOGGER.info('data loading finished!')

    def train_input_fn(self, num_epoch=5):
        return (self.ds.shuffle(buffer_size=10000)
                .repeat(num_epoch)  # first do repeat
                .padded_batch(MODEL_CONFIG.batch_size, padded_shapes=([None], [], []))
                ).make_one_shot_iterator().get_next(), None
