from builtins import range
from builtins import object
import numpy as np

from cs6353.layers import *
from cs6353.layer_utils import *


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deterministic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    prev_dim = input_dim
    hidden_dims.append(num_classes)
    for i in range(self.num_layers):
      self.params['W'+str(i+1)] = np.random.normal(loc = 0, scale = weight_scale, size = (prev_dim, hidden_dims[i]))
      self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
      prev_dim = hidden_dims[i]

    if self.use_batchnorm:
      for i in range(self.num_layers-1):
        self.params['gamma' + str(i+1)] = np.ones(hidden_dims[i])
        self.params['beta' + str(i+1)] = np.zeros(hidden_dims[i])

    self.dropout_param = {}
    if self.use_dropout:
      ###############################################################################
      # TODO: When using dropout we need to pass a dropout_param dictionary to each #
      # dropout layer so that the layer knows the dropout probability and the mode  #
      # (train / test). You can pass the same dropout_param to each dropout layer.  #
      ###############################################################################
      
      self.dropout_param = {'p': dropout, 'mode': 'train'}
      if seed is not None:
        self.dropout_param['seed'] = seed
      
      ###############################################################################
      #                             END OF YOUR CODE                                #
      ###############################################################################


    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    if self.dropout_param is not None:
      ###############################################################################
      # TODO: Set train/test mode for dropout param since it behaves differently    #
      # during training and testing.                                                #
      ###############################################################################
      
      self.dropout_param['mode'] = mode
      
      ###############################################################################
      #                             END OF YOUR CODE                                #
      ###############################################################################
      
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    scores = None

    prev_layer = X
    cache = []
    layer = []
    for i in range(self.num_layers):
      W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]
      this_layer, this_cache = affine_forward(prev_layer, W, b)
      prev_layer = this_layer
      layer.append(this_layer)
      cache.append(this_cache)

      if i != self.num_layers - 1:
        if self.use_batchnorm:
          gamma, beta = self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)]
          this_layer, this_cache = batchnorm_forward(prev_layer, gamma, beta, self.bn_params[i])
          prev_layer = this_layer
          layer.append(this_layer)
          cache.append(this_cache)

        this_layer, this_cache = relu_forward(prev_layer)
        prev_layer = this_layer
        layer.append(this_layer)
        cache.append(this_cache)

        if self.use_dropout:
          ############################################################################
          # TODO: When using dropout, you'll need to pass self.dropout_param to each #
          # dropout forward pass.                                                    #
          ############################################################################

          prev_layer, dropout_cache = dropout_forward(prev_layer, self.dropout_param)
          cache.append(dropout_cache)
          
          ############################################################################
          #                             END OF YOUR CODE                             #
          ############################################################################


    scores = layer[-1]



    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    loss, prev_layer_grad = softmax_loss(layer[-1], y)

    for i in range(self.num_layers):
      W = self.params['W' + str(i + 1)]
      loss += 0.5 * self.reg * np.sum(np.multiply(W, W))

    for i in reversed(range(self.num_layers)):
      W = self.params['W' + str(i + 1)]

      if i != self.num_layers - 1:
          if self.use_dropout:
            prev_layer_grad = dropout_backward(prev_layer_grad, cache[-1])
            cache = cache[:-1]

          prev_layer_grad = relu_backward(prev_layer_grad, cache[-1])
          cache = cache[:-1]


          if self.use_batchnorm:
            prev_layer_grad, grads['gamma' + str(i + 1)], grads['beta' + str(i + 1)] = batchnorm_backward(prev_layer_grad, cache[-1])
            cache = cache[:-1]

      prev_layer_grad, grads['W' + str(i + 1)], grads['b' + str(i + 1)] = affine_backward(prev_layer_grad, cache[-1])
      cache = cache[:-1]
      grads['W' + str(i + 1)] += self.reg * W

    return loss, grads