import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.ops.distributions.categorical import Categorical
from tensorflow.python.ops.rnn import _transpose_batch_time
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple, LSTMCell

from utils.sru import SRUCell


def model_fn(features, labels, mode, params, config):
    cur_batch_D = params.dim_embed

    char_embd = tf.get_variable('char_ebd',
                                [params.num_char, params.dim_embed],
                                initializer=tf.random_uniform_initializer(0))

    hinit_embed = tf.get_variable('hinit_ebd',
                                  [params.num_lang, params.num_hidden],
                                  initializer=tf.random_uniform_initializer(0))

    cinit_embed = tf.get_variable('cinit_ebd',
                                  [params.num_lang, params.num_hidden],
                                  initializer=tf.random_uniform_initializer(0))

    first_embed = tf.get_variable('first_ebd',
                                  [params.num_lang, cur_batch_D],
                                  initializer=tf.random_uniform_initializer(0))

    if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
        X_s, X_l, X_r = features
        cur_batch_T = tf.shape(X_s)[1]

        Xs_embd = tf.nn.embedding_lookup(char_embd, X_s, name='ebd_query_seq')
        X_ta = tf.TensorArray(size=cur_batch_T, dtype=tf.float32).unstack(
            _transpose_batch_time(Xs_embd), 'TBD_Formatted_X')
    else:
        X_r = [1] * params.infer_batch_size
        cur_batch_T = params.infer_seq_length

    acell = {
        'lstm': lambda: LSTMCell(params.num_hidden),
        'sru': lambda: SRUCell(params.num_hidden)
    }[params.cell]()

    output_layer_info = {
        'units': params.num_char,  # this is the size of vocabulary
        'name': 'out_to_character',
        # linear 'activation': tf.nn.softmax
    }

    with tf.variable_scope('Shared_Dense', reuse=False) as dense_layer_scope:
        # this will be replaced by the cell_output later
        zeros_placeholder = tf.zeros([1, acell.output_size])
        tf.layers.dense(zeros_placeholder, **output_layer_info)

    def get_logits(cell_out):
        # cell_out should be BxH
        # dense layer is HxNUM_CHAR
        # logit output is BxNUM_CHAR
        with tf.variable_scope(dense_layer_scope, reuse=True):
            return tf.layers.dense(cell_out, **output_layer_info)

    def get_dist(cell_out):
        return Categorical(logits=get_logits(cell_out),
                           name='categorical_dist',
                           allow_nan_stats=False,
                           dtype=tf.int32)

    def get_prob(cell_out, obs):
        # get_dist output is BxNUM_CHAR
        return get_dist(cell_out).prob(obs)

    with tf.variable_scope('Initial_State'):
        h_init = tf.nn.embedding_lookup(hinit_embed, X_r)
        c_init = tf.nn.embedding_lookup(cinit_embed, X_r)
        cell_init_state = {
            'lstm': lambda: LSTMStateTuple(c_init, h_init),
            'sru': lambda: h_init
        }[params.cell]()

        first_step = tf.nn.embedding_lookup(first_embed, X_r)

    with tf.name_scope('NADE'):
        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            output_ta = tf.TensorArray(size=cur_batch_T, dtype=tf.float32)
        else:
            output_ta = tf.TensorArray(size=cur_batch_T, dtype=tf.int32)

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0

            if cell_output is None:
                next_cell_state = cell_init_state
                next_step = first_step
                next_loop_state = output_ta
            else:  # pass the last state to the next
                next_cell_state = cell_state
                if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
                    next_step = X_ta.read(time - 1)
                    next_loop_state = loop_state.write(time - 1, next_step)
                else:
                    next_symbol = get_dist(cell_output).sample()
                    next_step = tf.nn.embedding_lookup(char_embd, next_symbol)
                    next_loop_state = loop_state.write(time - 1, next_symbol)

            if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
                elements_finished = (time >= X_l)
            else:
                elements_finished = (time >= cur_batch_T)

            return elements_finished, next_step, next_cell_state, emit_output, next_loop_state

        decoder_emit_ta, _, loop_state_ta = tf.nn.raw_rnn(acell, loop_fn)

    with tf.name_scope('Output'):
        outputs = _transpose_batch_time(decoder_emit_ta.stack())
        logits = get_logits(outputs)

    if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
        logp_loss = -tf.reduce_mean(tf.log(1e-6 + get_prob(outputs, X_s)))
        xentropy_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.one_hot(X_s, params.num_char),
                                                    logits=logits), name='xtropy_loss')

        train_op = tf.train.RMSPropOptimizer(learning_rate=params.learning_rate).minimize(
            loss=logp_loss, global_step=tf.train.get_global_step())

        logging_hook = tf.train.LoggingTensorHook(tensors={"xtropy_loss": "xtropy_loss"},
                                                  every_n_iter=100)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=logp_loss,
            train_op=train_op,
            training_chief_hooks=[logging_hook])
    else:
        X_sampled = _transpose_batch_time(loop_state_ta.stack())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=X_sampled)
