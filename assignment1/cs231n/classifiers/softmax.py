import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_samples = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_samples):
    correct_class = y[i]
    
    # multiply by weight matrix
    f_i = X[i].dot(W) 
  
    f_i -= np.max(f_i) # http://cs231n.github.io/linear-classify/#softmax  - normalization ttrick o stop overflow

    e_i = np.exp(f_i)                 # calc exp of f_i  
    e_sum = np.sum(e_i)               # calc sum of exponentials

    norm = e_i[correct_class]/e_sum   # normalize by div exp of real class / sum of exponents
    
    loss += -np.log(norm)

    for cl in range(num_classes):  
      p_cl      = e_i[cl] / e_sum
      dW[:, cl]  += (p_cl - (cl == y[i])) * X[i] 


  loss /= num_samples
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_samples
  dW += reg*W

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
  num_samples = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  fx = X.dot(W)
  fx -= np.max(fx, axis=1, keepdims=True) # normalise and keep shape

  e_Fx = np.exp(fx) # exponentiate
  e_sum = np.sum(e_Fx, axis=1) #sum each row and store in shape
  e_y = e_Fx[np.arange(num_samples), y] # get each actual y

  loss = np.sum(-np.log((e_y / e_sum)))


  loss /= num_samples
  loss += 0.5 * reg * np.sum(W*W)
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

