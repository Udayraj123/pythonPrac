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
#  -- Has a ReLU with 200 or 400 hidden units.
#  -- The concatenation is regularized with 50% dropout.


import tensorflow as tf
import tensorflow
import numpy as np
from tensorflow.contrib import rnn
import pandas as pd
from tensorflow.contrib.rnn import * 
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import *
# from tensorflow.python.ops.rnn_cell_impl import _linear

model = RNNCell()
print(tf.__version__)

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def _concat(prefix, suffix, static=False):
  """Concat that enables int, Tensor, or TensorShape values.
  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).
  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.
  Returns:
    shape: the concatenation of prefix and suffix.
  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s"
                       % (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape

def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""
  def get_state_shape(s):
    """Combine s with batch_size to get a proper tensor shape."""
    c = _concat(batch_size, s)
    c_static = _concat(batch_size, s, static=True)
    size = array_ops.zeros(c, dtype=dtype)
    size.set_shape(c_static)
    return size
  return nest.map_structure(get_state_shape, state_size)


class RNNCell(base_layer.Layer):
  """Abstract object representing an RNN cell.
  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.  The optional
  third input argument, `scope`, is allowed for backwards compatibility
  purposes; but should be left off for new subclasses.
  This definition of cell differs from the definition used in the literature.
  In the literature, 'cell' refers to an object with a single scalar output.
  This definition refers to a horizontal array of such units.
  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  (possibly nested tuple of) TensorShape object(s), then it should return a
  matching structure of Tensors having shape `[batch_size].concatenate(s)`
  for each `s` in `self.batch_size`.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.
    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size x self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size x s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.
    Returns:
      A pair containing:
      - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    if scope is not None:
      with vs.variable_scope(scope,
                             custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      with vs.variable_scope(vs.get_variable_scope(),
                             custom_getter=self._rnn_get_variable):
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    trainable = (variable in tf_variables.trainable_variables() or
                 (isinstance(variable, tf_variables.PartitionedVariable) and
                  list(variable)[0] in tf_variables.trainable_variables()))
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.
    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def build(self, _):
    # This tells the parent Layer object that it's OK to call
    # self.add_variable() inside the call() method.
    pass

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size x s]` for each s in `state_size`.
    """
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      state_size = self.state_size
      return _zero_state_tensors(state_size, batch_size, dtype)

def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)

class newLSTMCell(RNNCell):
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

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell
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
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(LSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
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
    self._activation = activation or math_ops.tanh

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

  def call(self, inputs, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
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
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:
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

"""
cb513+profile_split1.npy.gz
cullpdb+profile_6133.npy.gz
cullpdb+profile_6133_filtered.npy.gz
"""

data = np.load('./data/cullpdb+profile_6133.npy.gz')
print(data.shape)
data = np.reshape(data, [6133, 700, 57])
print(data.shape)

# print(data.info())
cv_data = data[5600:5877, :]
train_data = np.load('./data/cullpdb+profile_6133_filtered.npy')
test_data = np.load('./data/cb513+profile_split1.npy')
train_data = np.reshape(train_data, [-1, 700, 57])
test_data = np.reshape(test_data, [-1, 700, 57])
print(train_data.shape)
print(test_data.shape)

print(train_data.shape)
# print(cv_data.shape)
print(test_data.shape)
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

learning_rate = 0.1
n_epochs = 1000
num_classes = 8
hidden_units = 300

class BrnnForPssp():

    def __init__(self, learning_rate, num_classes, hidden_units):

        # Initialize data and variables
        self.weights = tf.Variable(tf.random_uniform([hidden_units*2, num_classes], minval=-0.5, maxval=0.5))
        self.biases  = tf.Variable(tf.zeros([num_classes]))
        self.x = tf.placeholder("float", [None, 700, 42])
        self.y = tf.placeholder("float", [None, 700, 8])

        # Do the prediction

        # Remember to change activation to ReLU
        self.fw_rnn_cell1 = newLSTMCell(hidden_units, forget_bias=1.0)
        self.fw_rnn_cell2 = newLSTMCell(hidden_units, forget_bias=1.0)
        self.fw_rnn_cell3 = newLSTMCell(hidden_units, forget_bias=1.0)
        self.bw_rnn_cell1 = newLSTMCell(hidden_units, forget_bias=1.0)
        self.bw_rnn_cell2 = newLSTMCell(hidden_units, forget_bias=1.0)
        self.bw_rnn_cell3 = newLSTMCell(hidden_units, forget_bias=1.0)
        # self.fw_rnn_cells = [self.fw_rnn_cell1, self.fw_rnn_cell2, self.fw_rnn_cell3]
        # self.bw_rnn_cells = [self.bw_rnn_cell1, self.bw_rnn_cell2, self.bw_rnn_cell3]
        self.fw_rnn_cells = [self.fw_rnn_cell1]
        self.bw_rnn_cells = [self.bw_rnn_cell1]
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
        self.y_predicted = tf.nn.softmax(tf.matmul(self.outputs_reshaped, self.weights) + self.biases)

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
        result = self.sess.run(self.y_predicted, feed_dict={self.x: x, self.y: y})
        return result

    def optimize(self, x, y):
        print("Accuracy on next-to-be-trained batch : ", self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y}))
        result = self.sess.run(self.trainer, feed_dict={self.x: x, self.y: y})
        print("Accuracy obtained after training     : ", self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y}))

    def cross_validate(self, x, y):
        result = self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y})
        return result

    def build_graph(self, x, y):
        writer = tf.summary.FileWriter('./graphs/lstmForPSSP',self.sess.graph)



model = BrnnForPssp(learning_rate=learning_rate, num_classes=8, hidden_units=hidden_units)
print("Successfully created the model")

for i in range(n_epochs):
    if i % 10 == 0:
        x = test_data_input
        y = test_data_otput
        print(i, "td - ", model.cross_validate(x=x, y=y))
    j = i%50 + 1
    x = train_data_input[j*100:j*100+100, :]
    y = train_data_otput[j*100:j*100+100, :]
    print("Iteration number & batch number: ", i, j)
    model.optimize(x=x, y=y)




