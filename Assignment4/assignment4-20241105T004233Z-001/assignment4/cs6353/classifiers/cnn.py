import numpy as np

from cs6353.layers import *
from cs6353.fast_layers import *
from cs6353.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    # 2x2 max pool
    pool_size, stride = 2, 2
    HP = 1 + (H - pool_size) // stride
    WP = 1 + (W - pool_size) // stride

    W1_init = np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['W1'] = W1_init * weight_scale
    self.params['b1'] = np.zeros(num_filters)

    W2_init = np.random.randn(num_filters * HP * WP, hidden_dim)
    self.params['W2'] = W2_init * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)

    W3_init = np.random.randn(hidden_dim, num_classes)
    self.params['W3'] = W3_init * weight_scale
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

    A2, affine_relu_cache = affine_relu_forward(conv_out, W2, b2)

    scores, affine_cache = affine_forward(A2, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, softmax_gradient = softmax_loss(scores, y)

    for W in [W1, W2, W3]:
      loss += 0.5 * self.reg * np.sum(W * W)

    dout, grads['W3'], grads['b3'] = affine_backward(softmax_gradient, affine_cache)

    dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, affine_relu_cache)

    dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, conv_cache)

    for W, name in zip([W1, W2, W3], ['W1', 'W2', 'W3']):
      grads[name] += self.reg * W

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class CustomConvNet(ThreeLayerConvNet):
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
      super(CustomConvNet, self).__init__(input_dim, num_filters, filter_size,
                                    hidden_dim, num_classes, weight_scale, reg)
