def namer(name):
    return name + ".gz"


def rotator(source, dest):
    import gzip, os
    with open(source, "rb") as sf:
        data = sf.read()
        compressed = gzip.compress(data)
        with open(dest, "wb") as df:
            df.write(compressed)
    os.remove(source)


def get_logger(name: str, log_path: str, log_format):
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
