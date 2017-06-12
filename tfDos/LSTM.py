
# coding: utf-8

# In[1]:

# Trying to reproduce results from
# paper :    arXiv : 1412.7828v2 [q-bio.QM] 4 Jan 2015
# Experiment name : Protein secondary structure prediction using
# LSTM networks.

# Model :
#  -- Standard stacked bidirectional LSTM with 3 layers.
#  -- (300 or 500) LSTM units in each layer
#  -- There is a FFN between h_rec and h with a skip connection. h_rec = ffn(h) + h
#  -- FFN is a two layer ReLU network with 300 or 500 units,
#  -- Introduce a FFN to combine output from forward and backward RNN
#  -- Has a ReLU with 200 or 80o0 hidden units.
#  -- The concatenation is regularized with 50% dropout.

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pandas as pd


import collections
import contextlib
import hashlib
import math
import numbers

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops.rnn_cell_impl import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope, _linear
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *


from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

print(tf.__version__)

data = np.load('./data/cullpdb+profile_6133.npy.gz')
print(data.shape)
data = np.reshape(data, [6133, 700, 57])
print(data.shape)

# print(data.info())
train_data = data[:5600, :]
cv_data = data[5600:5877, :]
test_data = data[5877:6133, :]

print(train_data.shape)
print(cv_data.shape)
print(test_data.shape)

"""
Source : arXiv:1403.1347v1  [q-bio.QM]  6 Mar 2014
:Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction

The resulting training data including both feature and la-
bels has 57 channels (22 for PSSM, 22 for sequence, 2 for
terminals,  8  for  secondary  structure  labels,  2  for  solvent
accessibility  labels),  and  the  overall  channel  size  is  700.
"""

"""
Source : http://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt
It is currently in numpy format as a (N protein x k features) matrix. You can reshape it to (N protein x 700 amino acids x 57 features) first.

The 57 features are:
"[0,22): amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F',
'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq'"
"[22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H',
'S', 'T','NoSeq'"
"[31,33): N- and C- terminals;"
"[33,35): relative and absolute solvent accessibility, used only for training.
(absolute accessibility is thresholded at 15; relative accessibility is normalized by the largest accessibility
value in a protein and thresholded at 0.15; original solvent accessibility is computed by DSSP)"
"[35,57): sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and
it is different from the order for amino acid residues"

The last feature of both amino acid residues and secondary structure labels just mark end of the protein sequence.
"[22,31) and [33,35) are hidden during testing."


"The dataset division for the first ""cullpdb+profile_6133.npy.gz"" dataset is"
"[0,5600) training"
"[5605,5877) test "
"[5877,6133) validation"
"""
# Split the train data
train_data_residues = train_data[:, :,  0:21]
train_data_secstruc = train_data[:, :, 22:30]
train_data_nctermin = train_data[:, :, 31:33]
train_data_rlabsolv = train_data[:, :, 33:35]
train_data_sequepro = train_data[:, :, 35:57]

# Checking shapes
print("Train data residues shape : ", train_data_residues.shape)
print("Train data secondary structue : ",train_data_secstruc.shape)
print("Train data n and c terminals : ", train_data_nctermin.shape)
print("Train data relative and absolute solvability : ", train_data_rlabsolv.shape)
print("Train data sequence profile : ", train_data_sequepro.shape)

train_data_input = train_data[:, :, np.r_[0:21, 36:57]]
train_data_otput = train_data[:, :, 23:31]
test_data_input = test_data[:, :, np.r_[0:21, 36:57]]
test_data_otput = test_data[:, :, 23:31]
# Checking shapes
# print("Train data input  shape : ", train_data_input.shape)
# print("Train data output shape : ", train_data_otput.shape)


# In[ ]:

learning_rate = 0.0000001
n_epochs = 100000
num_classes = 8
hidden_units = 30
_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))

# 
class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if not c.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

# 
# 


class LSTMCell(RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on:

    http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

  The peephole implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  """

  def __init__(self, num_units, input_size=None,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=tanh, reuse=None):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      input_size: Deprecated and unused.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._reuse = reuse

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
      scope: VariableScope for the created subgraph; defaults to "lstm_cell".

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    with _checked_scope(self, scope or "lstm_cell",
                        initializer=self._initializer,
                        reuse=self._reuse) as unit_scope:
      if self._num_unit_shards is not None:
        unit_scope.set_partitioner(
            partitioned_variables.fixed_size_partitioner(
                self._num_unit_shards))
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      lstm_matrix = _linear([inputs, m_prev], 4 * self._num_units, bias=True)
      i, j, f, o = array_ops.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)
      # Diagonal connections
      if self._use_peepholes:
        with vs.variable_scope(unit_scope) as projection_scope:
          if self._num_unit_shards is not None:
            projection_scope.set_partitioner(None)
          w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
             sigmoid(i + w_i_diag * c_prev) * self._activation(j))
      else:
        c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
             self._activation(j))

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * self._activation(c)
      else:
        m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        with vs.variable_scope("projection") as proj_scope:
          if self._num_proj_shards is not None:
            proj_scope.set_partitioner(
                partitioned_variables.fixed_size_partitioner(
                    self._num_proj_shards))
          m = _linear(m, self._num_proj, bias=False)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state

# 

with tf.device('/cpu:0'):
  class BrnnForPssp():

      def __init__(self, learning_rate, num_classes, hidden_units):

          # Initialize data and variables
          self.weights = tf.Variable(tf.random_uniform([hidden_units*2, num_classes], minval=-0.5, maxval=0.5))
          self.biases  = tf.Variable(tf.random_uniform([num_classes]))
          self.x = tf.placeholder("float", [None, 700, 42])
          self.y = tf.placeholder("float", [None, 700, 8])

          # Do the prediction
          self.fw_rnn_cell1 = LSTMCell(hidden_units, forget_bias=1.0)
          self.fw_rnn_cell2 = LSTMCell(hidden_units, forget_bias=1.0)
          self.fw_rnn_cell3 = LSTMCell(hidden_units, forget_bias=1.0)
          self.bw_rnn_cell1 = LSTMCell(hidden_units, forget_bias=1.0)
          self.bw_rnn_cell2 = LSTMCell(hidden_units, forget_bias=1.0)
          self.bw_rnn_cell3 = LSTMCell(hidden_units, forget_bias=1.0)
          self.fw_rnn_cells = [self.fw_rnn_cell1, self.fw_rnn_cell2, self.fw_rnn_cell3]
          self.bw_rnn_cells = [self.bw_rnn_cell1, self.bw_rnn_cell2, self.bw_rnn_cell3]
          self.outputs, self.states_fw, self.states_bw = rnn.stack_bidirectional_dynamic_rnn(
                                                              self.fw_rnn_cells,
                                                              self.bw_rnn_cells,
                                                              self.x,
                                                              dtype=tf.float32)
          # self.output.shape is (?, 700, 600)
          self.outputs_reshaped = tf.reshape(self.outputs, [-1, 2*hidden_units])
          self.y_reshaped = tf.reshape(self.y, [-1, num_classes])
          # check importantFunctions.py : line-40 to see how it works
          # reference link  is :
          # https://stackoverflow.com/questions/38051143/no-broadcasting-for-tf-matmul-in-tensorflow
          # self.y_predicted = tf.nn.softmax(tf.matmul(self.outputs_reshaped, self.weights) + self.biases)
          self.y_predicted = tf.matmul(self.outputs_reshaped, self.weights) + self.biases

          # Define the loss function
          self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.y_reshaped)

          # Define the trainer and optimizer
          self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
          self.trainer = self.optimizer.minimize(self.loss)

          # creating session and initializing variables
          self.sess = tf.Session()
          self.init = tf.global_variables_initializer()
          self.sess.run(self.init)

          # get accuracy
          self.get_equal = tf.equal(tf.argmax(self.y_reshaped, 1), tf.argmax(self.y_predicted, 1))
          self.accuracy = tf.reduce_mean(tf.cast(self.get_equal, tf.float32))

      def predict(self, x, y):
          result = self.sess.run(self.y_predicted, feed_dict={self.x : x, self.y : y})
          return result

      def optimize(self, x, y):
          result = self.sess.run(self.trainer, feed_dict={self.x : x, self.y : y})

      def cross_validate(self, x, y):
          result = self.sess.run(self.accuracy, feed_dict={self.x : x, self.y : y})
          return result

      def build_graph(self, x, y):
          writer = tf.summary.FileWriter('./graphs/lstmForPSSP',self.sess.graph)


  # In[ ]:

  model = BrnnForPssp(learning_rate=learning_rate, num_classes=8, hidden_units=hidden_units)
  print("Successfully created the model")

  results = list()
  for i in range(n_epochs):
      j = i%10
      x = train_data_input[j*80o0 : j*80o0+80o0, :]
      y = train_data_otput[j*80o0 : j*80o0+80o0, :]
      print(i, model.cross_validate(x=x, y=y))
      model.optimize(x=x, y=y)
      print(i, model.cross_validate(x=x, y=y))
      if i % 10 == 0:
          x = test_data_input
          y = test_data_otput
          print(i, model.cross_validate(x=x, y=y))
          # results.append(["Iteration, test accuracy : ", i, model.cross_validate(x=x, y=y)])

  for i in range(len(results)):
    print(results[i])

  # In[ ]:



    