import copy

import tensorflow as tf

from configs import LOGGER


def dRNN(cell, inputs, rate, init_state, scope='default'):
    n_steps = len(inputs)
    if rate < 0 or rate >= n_steps:
        raise ValueError('The \'rate\' variable needs to be adjusted.')
    LOGGER.info("Building layer: %s, input length: %d, dilation rate: %d, input dim: %d." % (
        scope, n_steps, rate, inputs[0].get_shape()[1]))

    # make the length of inputs divide 'rate', by using zero-padding
    if n_steps % rate:
        # Create a tensor in shape (batch_size, input_dims), which all elements are zero.
        # This is used for zero padding
        zero_tensor = tf.zeros_like(inputs[0])
        dilated_n_steps = n_steps // rate + 1
        LOGGER.info("%d time points need to be padded. " % (
            dilated_n_steps * rate - n_steps))

        for _ in range(dilated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)
    else:
        dilated_n_steps = n_steps // rate

    LOGGER.info("input length for sub-RNN: %d" % dilated_n_steps)

    dilated_inputs = [tf.concat(inputs[i * rate:(i + 1) * rate],
                                axis=0) for i in range(dilated_n_steps)]

    # now batch is **rate** times larger than the original, but **rate** times shorter

    # building a dilated RNN with reformated (dilated) inputs
    dilated_outputs, _ = tf.nn.static_rnn(cell=cell,
                                          inputs=dilated_inputs,
                                          initial_state=init_state,
                                          dtype=tf.float32, scope=scope)

    splitted_outputs = [tf.split(output, rate, axis=0) for output in dilated_outputs]
    unrolled_outputs = [output for sublist in splitted_outputs for output in sublist]
    # remove padded zeros
    outputs = unrolled_outputs[:n_steps]

    return outputs


def multi_dRNN(cells, inputs, dilations, init_cell_states):
    assert (len(cells) == len(dilations))
    x = copy.copy(inputs)
    for cell, dilation, init_state in zip(cells, dilations, init_cell_states):
        scope_name = "multi_dRNN_dilation_%d" % dilation
        x = dRNN(cell, x, dilation, init_state, scope=scope_name)
    return x


def get_last_output_dRNN(input, cells, dilations, init_cell_states):
    # error checking
    assert (len(cells) == len(dilations) == len(init_cell_states))

    # define dRNN structures
    layer_outputs = multi_dRNN(cells, input, dilations, init_cell_states)

    if dilations[0] == 1:
        return layer_outputs[-1]
    else:
        # dilation starts not at 1, needs to fuse the output
        # concat hidden_outputs
        for idx, i in enumerate(range(-dilations[0], 0, 1)):
            if idx == 0:
                hidden_outputs_ = layer_outputs[i]
            else:
                hidden_outputs_ = tf.concat([hidden_outputs_, layer_outputs[i]], axis=1)
        return hidden_outputs_
