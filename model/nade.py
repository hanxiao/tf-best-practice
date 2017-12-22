import tensorflow as tf
from tensorflow.python.ops.distributions.categorical import Categorical
from tensorflow.python.ops.rnn import _transpose_batch_time
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple, LSTMCell

from config import MODEL_CONFIG
from utils.reader import InputData
from utils.slstm import BasicSLSTMCell


class NADE:
    def __init__(self, input_data: InputData):
        self.is_training = tf.placeholder_with_default(True, shape=None, name='is_training')

        cur_batch_B = tf.shape(input_data.X_s)[0]
        cur_batch_T = tf.shape(input_data.X_s)[1]
        cur_batch_D = input_data.num_char

        Xs_embd = tf.one_hot(input_data.X_s, cur_batch_D)
        X_ta = tf.TensorArray(size=cur_batch_T, dtype=tf.float32).unstack(
            _transpose_batch_time(Xs_embd), 'TBD_Formatted_X')

        acell = {
            'lstm': lambda: LSTMCell(MODEL_CONFIG.num_hidden),
            'sru': lambda: BasicSLSTMCell(MODEL_CONFIG.num_hidden)
        }[MODEL_CONFIG.cell]()

        output_layer_info = {
            'units': cur_batch_D,  # this is the size of vocabulary
            'name': 'out_to_character',
            # linear 'activation': tf.nn.softmax
        }

        with tf.variable_scope('Shared_Dense', reuse=False) as dense_layer_scope:
            # this will be replaced by the cell_output later
            zeros_placeholder = tf.zeros([1, acell.output_size])
            tf.layers.dense(zeros_placeholder, **output_layer_info)

        def get_logits(cell_out):
            # useful when measuring the cross-entropy loss
            with tf.variable_scope(dense_layer_scope, reuse=True):
                return tf.layers.dense(cell_out, **output_layer_info)

        def get_dist(cell_out):
            return Categorical(logits=get_logits(cell_out), name='categorical_dist', allow_nan_stats=False,
                               dtype=tf.int32)

        def get_sample(cell_out):
            return tf.one_hot(get_dist(cell_out).sample(), cur_batch_D)

        def get_prob(cell_out, obs):
            # the observation is in
            return get_dist(cell_out).prob(obs)

        with tf.variable_scope('Initial_State'):
            h_init = tf.tile(tf.get_variable('init_state_h', [1, MODEL_CONFIG.num_hidden],
                                             initializer=tf.random_uniform_initializer(0)),
                             [cur_batch_B, 1])
            c_init = tf.tile(tf.get_variable('init_state_c', [1, MODEL_CONFIG.num_hidden],
                                             initializer=tf.random_uniform_initializer(0)),
                             [cur_batch_B, 1])
            cell_init_state = LSTMStateTuple(c_init, h_init)

            first_step = tf.zeros(shape=[cur_batch_B, cur_batch_D], dtype=tf.float32, name='first_character')

        with tf.name_scope('NADE'):
            output_ta = tf.TensorArray(size=cur_batch_T, dtype=tf.float32)

            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output  # == None for time == 0

                if cell_output is None:
                    # time == 0
                    # everything here will be used for initialization only
                    # no cell is invoked yet!
                    # this will be initial state
                    next_cell_state = cell_init_state

                    # this will be initial pixel for time 0, the output is expected to be the first pixel
                    next_step = first_step
                    next_loop_state = output_ta
                else:  # pass the last state to the next
                    next_cell_state = cell_state

                    # time 0, input: initial state (zero/var), expected output: 0th pixel
                    # time 1, input: 0th pixel, expected output: 1st pixel
                    # time 2, input: 1st pixel, expected output: 2nd pixel
                    # expected_output of current time is the next time input
                    next_step = tf.stop_gradient(tf.cond(self.is_training,
                                                         lambda: X_ta.read(time - 1),
                                                         lambda: get_sample(cell_output)))

                    next_loop_state = loop_state.write(time - 1, next_step)

                elements_finished = (time >= cur_batch_T)

                return elements_finished, next_step, next_cell_state, emit_output, next_loop_state

            output_ta, _, loop_state_ta = tf.nn.raw_rnn(acell, loop_fn)

        with tf.name_scope('Output'):
            outputs = _transpose_batch_time(output_ta.stack())
            logits = get_logits(outputs)
            self.X_sampled = _transpose_batch_time(loop_state_ta.stack())

            with tf.name_scope('Decoder_Loss'):
                self.logp_loss = -tf.reduce_mean(tf.log(1e-6 + get_prob(outputs, input_data.X_s)))
                self.xentropy_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=Xs_embd, logits=logits))
