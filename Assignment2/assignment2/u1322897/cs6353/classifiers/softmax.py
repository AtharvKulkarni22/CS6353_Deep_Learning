import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################  
  for i in range(X.shape[0]):
      scores = X[i].dot(W)

      scores -= scores.max()

      scores_exp = np.exp(scores)

      probabilties = scores_exp/np.sum(scores_exp)

      loss += -np.log(probabilties[y[i]])

      error = probabilties.reshape(1,-1)
      
      error[:, y[i]] -= 1

      column_vector = X[i].reshape(X[i].shape[0], 1)
      dW += np.dot(column_vector, error)

  loss /= X.shape[0]
  dW /= X.shape[0]

  loss += reg * np.sum(W * W)

  dW += 2 * reg * W    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  indices = np.array(range(num_train)) 

  scores = X.dot(W)
  
  scores -= scores.max(axis = 1, keepdims = True)

  scores_exp = np.exp(scores)

  probabilties = scores_exp/np.sum(scores_exp, axis = 1, keepdims = True)

  loss = -np.log(probabilties[indices, y])

  loss = np.sum(loss)

  error = probabilties.reshape(num_train, -1)

  error[indices, y] -= 1

  X_reshaped = X.T.reshape(X.shape[1], num_train)

  dW = np.dot(X_reshaped, error)

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)

  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

