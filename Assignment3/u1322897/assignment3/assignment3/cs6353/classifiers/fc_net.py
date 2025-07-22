from builtins import range
from builtins import object
import numpy as np

from cs6353.layers import *
from cs6353.layer_utils import *


class TwoLayerNet(object):
    '''
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    '''

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        '''
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        '''
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        
        W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['W1'] = W1

        b1 = np.zeros(hidden_dim)
        self.params['b1'] = b1

        W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['W2'] = W2

        b2 = np.zeros(num_classes)
        self.params['b2'] = b2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        '''
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        '''
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        L1_score, cache_l1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache_l2 = affine_forward(L1_score, self.params['W2'], self.params['b2'])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        reg_loss_W1 = np.sum(self.params['W1'] ** 2)
        reg_loss_W2 = np.sum(self.params['W2'] ** 2)
        reg_term = 0.5 * self.reg * (reg_loss_W1 + reg_loss_W2)
        loss = loss + reg_term

        dL1, dW2, db2 = affine_backward(dscores, cache_l2)
        dX, dW1, db1 = affine_relu_backward(dL1, cache_l1)
        dW2 += self.reg * self.params['W2'] 
        dW1 += self.reg * self.params['W1'] 

        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    '''
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    batch/layer normalization as an option. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu} x (L - 1) - affine - softmax

    where batch/layer normalization is optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    '''

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                  normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        '''
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.

        '''
        self.normalization = normalization
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        combined_dims = [input_dim] + hidden_dims + [num_classes]

        for layer_num in range(self.num_layers):
          index = str(layer_num + 1)
          dim_current = combined_dims[layer_num]
          dim_next = combined_dims[layer_num + 1]
          Wt = np.random.randn(dim_current, dim_next)
          Wt *= weight_scale

          bias = np.zeros(dim_next)
          self.params['W' + index] = Wt
          self.params['b' + index] = bias

        if self.normalization:
          for layer_num in range(self.num_layers - 1):
            index = str(layer_num + 1)

            self.params['gamma' + index] = np.ones((combined_dims[layer_num + 1]))
            self.params['beta' + index] = np.zeros((combined_dims[layer_num + 1]))
          
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        '''
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        '''
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params param since it
        # behaves differently during training and testing.
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        caches = {}

        for layer_num in range(self.num_layers - 1):
          index = str(layer_num + 1)
          Wt = self.params['W' + index]
          bias = self.params['b' + index]


          if self.normalization:
            X, caches[layer_num + 1] = affine_norm_relu_forward(X, Wt, bias, 
            self.params['gamma' + index], 
            self.params['beta' + index], 
            self.bn_params[layer_num], 
            self.normalization)
        
          else:
            X, caches[layer_num + 1] = affine_relu_forward(X, Wt, bias)
          
        scores, caches[self.num_layers] = affine_forward(X, self.params['W' + str(self.num_layers)],
                            self.params['b' + str(self.num_layers)])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        for layer_num in range(self.num_layers):
          index = str(layer_num + 1)
          reg_loss = np.sum(self.params['W' + index] ** 2)
          reg_term = 0.5 * self.reg * reg_loss
          loss += reg_term

        # grads = {}
        # final_index = str(self.num_layers)
        # dscores, dW, db = affine_backward(dscores, caches[self.num_layers])
        # grads['W' + final_index] = dW + self.reg * self.params['W' + final_index]
        # grads['b' + final_index] = db

        # for layer_num in range(self.num_layers - 1, 0, -1):
        #   index = str(layer_num)

        #   if self.normalization:
        #     dscores, dw, db, grads['gamma' + index], grads['beta' + index] =  affine_norm_relu_backward(dscores, caches[layer_num], self.normalization)
        #   else:
        #     dscores, dW, db = affine_relu_backward(dscores, caches[layer_num])


        #   grads['W' + index] = dW + self.reg * self.params['W' + index]
        #   grads['b' + index] = db

        # Backward pass
        grads = {}
        final_index = str(self.num_layers)
        dscores, dW, db = affine_backward(dscores, caches[self.num_layers])
        grads['W' + final_index] = dW + self.reg * self.params['W' + final_index]
        grads['b' + final_index] = db

        for layer_num in range(self.num_layers - 1, 0, -1):
            index = str(layer_num)

            if self.normalization:
                dscores, dW, db, grads['gamma' + index], grads['beta' + index] = affine_norm_relu_backward(
                    dscores, caches[layer_num], self.normalization
                )
            else:
                dscores, dW, db = affine_relu_backward(dscores, caches[layer_num])

            grads['W' + index] = dW + self.reg * self.params['W' + index]
            grads['b' + index] = db        
        

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def affine_norm_relu_forward(x, w, b, gamma, beta, bn_params, normalization):
  fc_cache, bn_cache, relu_cache = None, None, None
  out, fc_cache = affine_forward(x, w, b)

  if normalization == 'batchnorm':
      out, bn_cache = batchnorm_forward(out, gamma, beta, bn_params)
  elif normalization == 'layernorm':
      out, bn_cache = layernorm_forward(out, gamma, beta, bn_params)
  else:
      bn_cache = None 

  out, relu_cache = relu_forward(out)

  return out, (fc_cache, bn_cache, relu_cache)

def affine_norm_relu_backward(dout, cache, normalization):
  fc_cache, bn_cache, relu_cache = cache

  dout = relu_backward(dout, relu_cache)

  dgamma, dbeta = None, None
  if normalization == 'batchnorm':
      dout, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
  elif normalization == 'layernorm':
      dout, dgamma, dbeta = layernorm_backward(dout, bn_cache)

  dx, dw, db = affine_backward(dout, fc_cache)
  
  return dx, dw, db, dgamma, dbeta