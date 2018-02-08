"""Implement https://arxiv.org/abs/1709.02755
Copy from BasicLSTMCell, and make it functionally correct with minimum code change
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl


class SRUCell(rnn_cell_impl._LayerRNNCell):
    """SRU, Simple Recurrent Unit
       Implementation based on
       Training RNNs as Fast as CNNs (cf. https://arxiv.org/abs/1709.02755).
       This variation of RNN cell is characterized by the simplified data
       dependence
       between hidden states of two consecutive time steps. Traditionally, hidden
       states from a cell at time step t-1 needs to be multiplied with a matrix
       W_hh before being fed into the ensuing cell at time step t.
       This flavor of RNN replaces the matrix multiplication between h_{t-1}
       and W_hh with a pointwise multiplication, resulting in performance
       gain.
    Args:
      num_units: int, The number of units in the SRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: (optional) String, the name of the layer. Layers with the same name
        will share weights, but to avoid mistakes we require reuse=True in such
        cases.
    """

    def __init__(self, num_units, activation=None, reuse=None, name=None):
        super(SRUCell, self).__init__(_reuse=reuse, name=name)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh

        # Restrict inputs to be 2-dimensional matrices
        self.input_spec = base_layer.InputSpec(ndim=2)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[1].value

        self._kernel = self.add_variable(
            rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
            shape=[input_depth, 4 * self._num_units])

        self._bias = self.add_variable(
            rnn_cell_impl._BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=init_ops.constant_initializer(0.0, dtype=self.dtype))

        self._built = True

    def call(self, inputs, state):
        """Simple recurrent unit (SRU) with num_units cells."""

        U = math_ops.matmul(inputs, self._kernel)
        x_bar, f_intermediate, r_intermediate, x_tx = array_ops.split(
            value=U, num_or_size_splits=4, axis=1)

        f_r = math_ops.sigmoid(
            nn_ops.bias_add(
                array_ops.concat([f_intermediate, r_intermediate], 1), self._bias))
        f, r = array_ops.split(value=f_r, num_or_size_splits=2, axis=1)

        c = f * state + (1.0 - f) * x_bar
        h = r * self._activation(c) + (1.0 - r) * x_tx

        return h, c
