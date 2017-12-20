import time
from logging import Logger


def rotator(source, dest):
    import gzip, os
    with open(source, "rb") as sf:
        data = sf.read()
        compressed = gzip.compress(data)
        with open(dest, "wb") as df:
            df.write(compressed)
    os.remove(source)


def get_logger(name: str, log_path: str, log_format) -> Logger:
    import logging.handlers
    from utils.helper import touch
    touch(log_path, create_dirs=True)
    l = logging.getLogger(name)
    l.setLevel(logging.DEBUG)
    # add rotator to the logger. it's lazy in the sense that it wont rotate unless there are new logs
    fh = logging.handlers.TimedRotatingFileHandler(log_path, interval=24)
    fh.setLevel(logging.INFO)
    fh.rotator = rotator
    fh.namer = lambda x: x + ".gz"

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_format, datefmt='%H:%M:%S'))
    ch.setFormatter(logging.Formatter(log_format, datefmt='%H:%M:%S'))
    l.addHandler(fh)
    l.addHandler(ch)
    return l


class JobContext:
    def __init__(self, msg, logger: Logger = None):
        self._msg = msg
        self._logger = logger

    def __enter__(self):
        self.start = time.clock()
        if not self._logger:
            print(self._msg, end='')
        else:
            self._logger.info('☐ %s' % self._msg)

    def __exit__(self, typ, value, traceback):
        self.duration = time.clock() - self.start
        if not self._logger:
            print('    [%.3f secs]\n' % self.duration)
        else:
            self._logger.info('☑ %s    [%.3f secs]' % (self._msg, self.duration))
