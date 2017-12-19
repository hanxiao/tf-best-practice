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

def get_logger(name: str):
    LOG_FORMAT = '%(asctime)s:%(filename)s:%(funcName)s:[%(levelname)s] %(message)s'
    LOG_PATH = ''
    import logging, logging.handlers
    l = logging.getLogger(name)
    l.setLevel(logging.DEBUG)
    # add rotator to the logger. it's lazy in the sense that it wont rotate unless there are new logs
    fh = logging.handlers.TimedRotatingFileHandler(LOG_PATH, interval=3)
    fh.setLevel(logging.INFO)
    fh.rotator = rotator
    fh.namer = namer

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt='%H:%M:%S'))
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt='%H:%M:%S'))
    l.addHandler(fh)
    l.addHandler(ch)
    return l
