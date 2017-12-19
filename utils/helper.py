import subprocess
import time
from threading import Thread, Event

import tensorflow as tf

from configs import *


def now_int():
    epoch = datetime.utcfromtimestamp(0)
    return (datetime.now() - epoch).total_seconds()


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def entry_stop_gradients(target, mask):
    mask_h = tf.logical_not(mask)

    mask = tf.cast(mask, dtype=target.dtype)
    mask_h = tf.cast(mask_h, dtype=target.dtype)

    return tf.stop_gradient(mask_h * target) + mask * target


# Adapeted from
# https://gist.github.com/kukuruza/03731dc494603ceab0c5#gistcomment-1879326
def on_grid(kernel, grid_side, pad=1):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
        kernel:    tensor of shape [Y, X, NumChannels, NumKernels]
        grid_side: side of the grid. Require: NumKernels == grid_side**2
        pad:       number of black pixels around each filter (between them)
    Returns:
        An image Tensor with shape [(Y+2*pad)*grid_side, (X+2*pad)*grid_side, NumChannels, 1].
    """

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(
        kernel1,
        tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]),
        mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2,
                    tf.stack(
                        values=[grid_side, Y * grid_side, X, channels],
                        axis=0))  # 3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4,
                    tf.stack(
                        values=[1, X * grid_side, Y * grid_side, channels],
                        axis=0))  # 3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


class JobContext(object):
    def __init__(self, msg):
        self._msg = msg

    def __enter__(self):
        self.start = time.clock()

    def __exit__(self, typ, value, traceback):
        self.duration = time.clock() - self.start
        LOGGER.info("{1}    [{0:.3f} secs]".format(self.duration, self._msg))


class UploadS3Thread(Thread):
    def __init__(self, event: Event):
        super().__init__()
        self.stopped = event

    def run(self):
        while not self.stopped.wait(SYNC_INTERVAL):
            upload_result_s3()


def upload_result_s3():
    LOGGER.info("Syncing data to S3...")
    with open(LOG_PATH, 'a', 1) as logfile:
        proc = subprocess.Popen(
            "bash %s %s %s" % (SYNC_SCRIPT_PATH, LOG_PATH, LOG_DIR),
            shell=True,
            stdin=subprocess.PIPE,
            stdout=logfile,
            stderr=logfile,
            cwd=ROOT_DIR,
            env=os.environ)

        # we have to wait until the training data is downloaded
        try:
            outs, errs = proc.communicate(timeout=SYNC_TIMEOUT)
            if outs:
                LOGGER.info(outs)
            if errs:
                LOGGER.error(errs)
        except subprocess.TimeoutExpired:
            proc.kill()


def download_s3_data():
    LOGGER.info("loading data from S3...")
    with open(LOG_PATH, 'a', 1) as logfile:
        proc = subprocess.Popen(
            "aws s3 sync s3://query2fdna-data/ %s --region=eu-central-1" % DATA_DIR,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=logfile,
            stderr=logfile,
            cwd=ROOT_DIR,
            env=os.environ)

        # we have to wait until the training data is downloaded
        try:
            outs, errs = proc.communicate(timeout=SYNC_TIMEOUT)
            if outs:
                LOGGER.info(outs)
            if errs:
                LOGGER.error(errs)
        except subprocess.TimeoutExpired:
            proc.kill()


def get_last_output(output, sequence_length, name):
    """Get the last value of the returned output of an RNN.
    http://disq.us/p/1gjkgdr
    output: [batch x number of steps x ... ] Output of the dynamic lstm.
    sequence_length: [batch] Length of each of the sequence.
    """
    rng = tf.range(0, tf.shape(sequence_length)[0])
    indexes = tf.stack([rng, sequence_length - 1], 1)
    return tf.gather_nd(output, indexes, name)
