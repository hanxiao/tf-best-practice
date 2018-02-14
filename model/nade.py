import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.ops.distributions.categorical import Categorical
from tensorflow.python.ops.rnn import _transpose_batch_time
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple, LSTMCell, GRUCell

from utils.sru import SRUCell


def make_var(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(0))


def model_fn(features, labels, mode, params, config):
    cur_batch_D = params.dim_embed

    hinit_embed = make_var('hinit_ebd', [params.num_lang, params.num_hidden])
    cinit_embed = make_var('cinit_ebd', [params.num_lang, params.num_hidden])
    zero_embed = make_var('zero_embed', [params.num_lang, cur_batch_D])
    char_embd = make_var('char_ebd', [params.num_char, params.dim_embed])

    if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
        X_c, X_s, L_c, L_s, T = features
        cur_batch_T = tf.shape(X_s)[1]
        Xs_embd = tf.nn.embedding_lookup(char_embd, X_s, name='ebd_nextline_seq')
        Xs_ta = tf.TensorArray(size=cur_batch_T, dtype=tf.float32).unstack(
            _transpose_batch_time(Xs_embd), 'TBD_Formatted_Xs')
    else:
        X_c, L_c, T = features  # only give the context info
        cur_batch_T = params.infer_seq_length

    make_cell = {
        'lstm': lambda x: LSTMCell(params.num_hidden, name=x, reuse=False),
        'sru': lambda x: SRUCell(params.num_hidden, name=x, reuse=False),
        'gru': lambda x: GRUCell(params.num_hidden, name=x, reuse=False)
    }[params.cell]

    with tf.variable_scope('InitState'):
        h_init = tf.nn.embedding_lookup(hinit_embed, T)
        c_init = tf.nn.embedding_lookup(cinit_embed, T)
        cell_init_state = {
            'lstm': lambda: LSTMStateTuple(c_init, h_init),
            'sru': lambda: h_init,
            'gru': lambda: h_init,
        }[params.cell]()

    with tf.variable_scope('Encoder'):
        # make a list of cells for dilated encoder
        _, last_enc_state = \
            tf.nn.dynamic_rnn(make_cell('encoder_cell'),
                              tf.nn.embedding_lookup(char_embd, X_c, name='ebd_context_seq'),
                              sequence_length=L_c,
                              initial_state=cell_init_state,
                              dtype=tf.float32)

    with tf.variable_scope('Decoder'):
        # make a new cell for decoder
        decoder_cell = make_cell('decoder_cell')
        zero_step = tf.nn.embedding_lookup(zero_embed, T)

        output_layer_info = {
            'units': params.num_char,  # this is the size of vocabulary
            'name': 'out_to_character',
            # linear 'activation': tf.nn.softmax
        }

        with tf.variable_scope('Shared_Dense', reuse=False) as dense_layer_scope:
            # this will be replaced by the cell_output later
            zeros_placeholder = tf.zeros([1, decoder_cell.output_size])
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

        with tf.name_scope('NADE'):
            # use this TA to store sampled character in inference time
            # and embed vector in training and evaluation time
            output_ta = tf.TensorArray(size=cur_batch_T,
                                       dtype=tf.int32 if mode == ModeKeys.INFER else tf.float32)

            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output  # == None for time == 0

                if cell_output is None:
                    next_cell_state = last_enc_state
                    next_step = zero_step
                    next_loop_state = output_ta
                else:  # pass the last state to the next
                    next_cell_state = cell_state
                    if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
                        next_step = Xs_ta.read(time - 1)
                        next_loop_state = loop_state.write(time - 1, next_step)
                    else:
                        next_symbol = get_dist(cell_output).sample()
                        next_step = tf.nn.embedding_lookup(char_embd, next_symbol)
                        next_loop_state = loop_state.write(time - 1, next_symbol)

                if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
                    elements_finished = (time >= L_s)
                else:
                    elements_finished = (time >= cur_batch_T)

                return elements_finished, next_step, next_cell_state, emit_output, next_loop_state

            decoder_emit_ta, _, loop_state_ta = tf.nn.raw_rnn(decoder_cell, loop_fn)

    if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
        out_logits = _transpose_batch_time(decoder_emit_ta.stack())
        logp_loss = -tf.reduce_mean(tf.log(1e-6 + get_prob(out_logits, X_s)))
        train_op = tf.train.RMSPropOptimizer(learning_rate=params.learning_rate).minimize(
            loss=logp_loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=logp_loss,
            train_op=train_op)
    else:
        X_sampled = _transpose_batch_time(loop_state_ta.stack())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=X_sampled)
