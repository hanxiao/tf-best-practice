import json

import dask.bag as db

from config import APP_CONFIG, LOGGER
from .logger import JobContext


def load_dataset(len_threshold: int = 100):
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
        preserved_ints = 1
        char2int_map = {c: idx + preserved_ints for idx, c in enumerate(all_chars)}

        unknown_room_idx = len(char2int_map)
        preserved_ints += len(char2int_map) + 1
        room2int_map = {c: idx + preserved_ints for idx, c in enumerate(all_rooms)}

        unknown_user_idx = len(room2int_map)
        preserved_ints += len(room2int_map) + 1
        user2int_map = {c: idx + preserved_ints for idx, c in enumerate(all_users)}

        preserved_ints += len(user2int_map)

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
        LOGGER.info('vocabulary size: %d' % preserved_ints)
        LOGGER.info('histogram of sent length: %s' % len_hist)
        LOGGER.info('maximum length of training sent: %d' % len_threshold)

    with JobContext('building dataset...', LOGGER):
        d = (
            msg_stream.filter(lambda x: len(x['text']) <= len_threshold).map(
                lambda x: ([char2int_map.get(c, unknown_char_idx) for c in x['text']] + [0] * (len_threshold - len(x)),
                           room2int_map.get(x['chatroomName'], unknown_room_idx),
                           user2int_map.get(x['fromUser'], unknown_user_idx))))

        return {
            'num_char': num_char,
            'num_user': num_user,
            'num_room': num_room,
            'num_sent': num_sent,
            'max_len': max_len,
            'len_threshold': len_threshold,
            'char2int': char2int_map,
            'unknown_char_idx': unknown_char_idx,
            'unknown_user_idx': unknown_user_idx,
            'unknown_room_idx': unknown_room_idx,
            'data_sent': list(d.pluck(0)),
            'data_room': list(d.pluck(1)),
            'data_user': list(d.pluck(2))
        }
