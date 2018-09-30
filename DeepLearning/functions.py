""" All math functions and ops used by a model

This module provides the foundation of functions and operations
to build a network and optimize models.

It contains only the units that would used within a model.
Other functions, for training or data processing, can be found in `utils.py`

Module components
=================
Function : base class for functions

atomic functions : elementary functions and factors
    Log, Square, Exp, Power, Sqrt

connection functions : functions using learnable variables
    Bias, Matmul, Linear

ReductionFunction : functions that reduce dimensionality
    Sum, Mean, Prod, Max, Min

activation functions : nonlinearities
    ReLU, ELU, SeLU, Sigmoid, Tanh, Softmax, Swish

loss functions : objectives for gradient descent
    LogisticCrossEntropy, SoftmaxCrossEntropy

"""
import code
from functools import wraps
from pprint import PrettyPrinter
import numpy as np

import utils
from utils import TODO, NOTIMPLEMENTED, INSPECT

pretty_printer = PrettyPrinter()
pprint = lambda x: pretty_printer.pprint(x)

""" submodule imports
utils :
    `TODO` : decorator func
        serves as comment and call safety

    `NOTIMPLEMENTED` : decorator func
        raises NotImplementedErrorforces if class func has not been overridden

    `INSPECT` : decorator func
        interrupts computation and enters interactive shell,
        where the user can evaluate the input and output to func
"""



#------------------------------------------------------------------------------
# Helpers and handy decorators
#------------------------------------------------------------------------------

class AttrDict(dict):
    """ dict accessed/mutated by attribute instead of index """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def restore_axis_shape(x, ax, d):
    """ Restores an axis 'ax' that was reduced from 'x'
    to it's original shape 'd'

    Just a wrapper for the tedious
    broadcast_to(expand_dims(...)...) op
    """

    # Assumes:
    # x is at least 1D
    # only on dim is being restored through ax
    assert x.ndim >= 1 and isinstance(ax, int)

    # Restore shape
    bcast_shape = x.shape[:ax] + (d,) + x.shape[ax:]
    return np.broadcast_to(np.expand_dims(x, ax), bcast_shape)



#==============================================================================
#                              Function
#==============================================================================

# Base function class
#------------------------------------------------------------------------------

# Function
# --------
# inherits :
# derives  : ReductionFunction
class Function:
    """ Function parent class for various, mostly mathematical ops

    Function defines the basic structure and interface to most of
    the functions used in network ops

    In addition to ops and attributes described below, most functions
    inherit the following, unchanged, from Function:
     # __init__
     # __str__
     # __repr__
     # __call__
     # Function.cache

    Function ops
    ------------
    forward : f(X)
        the function
    backward : f'(X)
        the derivative of the function (wrt chained gradient)

    Attributes
    ----------
    name : str
        name of the function (based on function class name)
    cache : object
        cache is used for whatever data the functions store for
        retrieval during backpropagation (eg, inputs/outputs or
        reduction axes)
        WARNING: cache is automatically cleared after access
    """
    def __init__(self, *args, **kwargs):
        self.name = self.__class__.__name__
        self._cache = None
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    def __str__(self,):
        # str : $classname
        #     eg 'MatMul'
        return self.name

    def __repr__(self):
        # repr : functions.$classname
        #    eg "functions.MatMul"
        return "functions.{}".format(self.name)

    @property
    def cache(self):
        """ NOTE: 'clears' cache upon access
        (since it is always reset in backprop)
        """
        cache_objs = self._cache
        self._cache = None
        return cache_objs

    @cache.setter
    def cache(self, *fvars):
        self._cache = fvars if len(fvars) > 1 else fvars[0]

    def forward(self, X, *args):
        """ Forward serves as an interface to a staticmethod for most Functions
        These methods are purely functional, and perform the function
        represented by the class.

        Function, as the base class, provides a simple implementation
        of this behavior.

        If the Function class uses a staticmethod as such,
        the methods are generally the lower-case version of the class name,
        and the derivative functions have '_prime' concatenated.

        For example, the following Functions have staticmethods:
            Tanh : tanh, tanh_prime
            Sqrt : sqrt, sqrt_prime
            MatMul : matmul, matmul_prime

        However, many Functions have forward or backward processes
        that have more significant side-effects or other constraints
        that would not be complete by the pure functions alone.
          - LogisticCrossEntropy, for instance, cannot use its
            functions alone (and, incidentally, does not have
            conforming function names)

        For any function where this is the case, they will override the
        forward and backward methods to meet their constraints
        """
        # in this case, self.name = 'Function'
        fn_name = self.name.lower()
        if hasattr(self, fn_name): # is there a 'self.function' ?
            fn = getattr(self, fn_name)
            return fn(X, *args)
        else:
            raise NotImplementedError

    def backward(self, gY, *args):
        fn_name = self.name.lower() + '_prime'
        if hasattr(self, fn_name):
            fn = getattr(self, fn_name)
            return fn(gY, *args)
        else:
            raise NotImplementedError

    def __call__(self, *args, backprop=False):
        func = self.backward if backprop else self.forward
        return func(*args)


#==============================================================================
#                             Math Functions
#==============================================================================

#------------------------------------------------------------------------------
#  atomic functions
#------------------------------------------------------------------------------

class Exp(Function): #
    """ Exponential function """
    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def exp_prime(x):
        return np.exp(x)

    def forward(self, X):
        Y = self.exp(X)
        self.cache = Y
        return Y

    def backward(self, gY):
        Y = self.cache
        gX = Y * gY
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Log(Function): #
    """ Natural logarithm function """
    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def log_prime(x):
        return 1.0 / x

    def forward(self, X):
        self.cache = X
        Y = self.log(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.log_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Square(Function):#
    """ Square """
    @staticmethod
    def square(x):
        return np.square(x)

    @staticmethod
    def square_prime(x):
        return 2.0 * x

    def forward(self, X):
        self.cache = X
        Y = self.square(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.square_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Sqrt(Function): #
    """ Square root """
    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def sqrt_prime(y):
        return 1.0 / (2 * y)

    def forward(self, X):
        Y = self.sqrt(X)
        self.cache = Y
        return Y

    def backward(self, gY):
        Y = self.cache
        gX = gY * self.sqrt_prime(Y)
        return gX


#------------------------------------------------------------------------------
# Connection functions :
#     Functions using learnable variables
#------------------------------------------------------------------------------

class Bias(Function): #
    """ Adds to matrix X a bias vector B

    Params
    ------
    X : ndarray.float32, (N, D)
        input matrix data
    B : ndarray.float32, (D,)
        bias array
    """
    @staticmethod
    def bias(x, b):
        return x + b

    @staticmethod
    def bias_prime(y, b):
        gx = np.copy(y)
        gb = y.sum(0)
        return gx, gb

    def forward(self, X, B):
        Y = self.bias(X, B)
        return Y

    def backward(self, gY, B):
        gX, gB = self.bias_prime(gY, B)
        return gX, gB

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class MatMul(Function): #
    """ Matrix multiplication function defined on 2D matrices

    Assumes: matrices A, B are properly ordered as input args
        In other words, A.shape[-1] == B.shape[0]

    Params
    ------
    A : ndarray
        2D matrix with shape (N, K), N an arbitrary int
    B : ndarray
        2D matrix with shape (K, P), P an arbitrary int

    Returns
    -------
    Y : ndarray
        2D matrix product of AB, with shape (N, P)
    """
    @staticmethod
    def matmul(a, b):
        return np.matmul(a, b)

    @staticmethod
    def matmul_prime(y, a, b):
        da = np.matmul(y, b.T)
        db = np.matmul(a.T, y)
        return da, db

    def forward(self, A, B):
        self.cache = np.copy(A)
        Y = self.matmul(A, B)
        return Y

    def backward(self, gY, B):
        A = self.cache
        gA, gB = self.matmul_prime(gY, A, B)
        return gA, gB

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Linear(Function): #
    """ Affine transformation function, Y = X.W + B

    See also: MatMul and Bias, which Linear composes

    Params
    ------
    X : ndarray.float32, (N, D)
        2D data matrix, N samples, D features
    W : ndarray.float32, (D, K)
        2D weights matrix mapped to features
    B : ndarray.float32, (K,)
        1D bias vector

    Returns
    -------
    Y : ndarray.float32, (N, K)
        transformation on data X
    """
    @staticmethod
    def linear(x, w, b):
        y = MatMul.matmul(x, w)
        return Bias.bias(y, b)

    @staticmethod
    def linear_prime(y, x, w, b):
        dx, dw = MatMul.matmul_prime(np.copy(y), x, w)
        _, db = Bias.bias_prime(y, b) # bias just identity func on gradient
        return dx, dw, db

    def forward(self, X, W, B):
        self.cache = np.copy(X)
        Y = self.linear(X, W, B)
        return Y

    def backward(self, gY, W, B):
        X = self.cache
        gX, gW, gB = self.linear_prime(gY, X, W, B)
        return gX, gW, gB

#==============================================================================
#                          Activation Functions
#==============================================================================

class Sigmoid(Function): #
    """ Logistic sigmoid activation """
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(y):
        """ ASSUMES y == sigmoidx(x) """
        return y * (1 - y)

    def forward(self, X):
        Y = self.sigmoid(X)
        self.cache = Y
        return Y

    def backward(self, gY):
        Y = self.cache
        gX = gY * self.sigmoid_prime(Y)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Tanh(Function): #
    """ Hyperbolic tangent activation """
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_prime(y):
        return 1.0 - np.square(y)

    def forward(self, X):
        Y = self.tanh(X)
        self.cache = Y
        return Y

    def backward(self, gY):
        Y = self.cache
        gX = gY * self.tanh_prime(Y)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Softmax(Function): #
    """ Softmax activation """
    @staticmethod
    def softmax(x):
        kw = {'axis':1, 'keepdims':True}
        exp_x = np.exp(x - x.max(**kw))
        return exp_x / np.sum(exp_x, **kw)

    @staticmethod
    def softmax_prime(y):
        kw = {'axis':1, 'keepdims':True}
        sqr_sum_y = np.square(y).sum(**kw)
        return y - sqr_sum_y

    def forward(self, X):
        Y = self.softmax(X)
        self.cache = Y
        return Y

    def backward(self, gY):
        Y = self.cache
        gX = gY * self.softmax_prime(Y)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class Swish(Function): #
    """ Self-gated activation function
    Can be viewed as a smooth function where the nonlinearity
    interpolates between the linear function (x/2), and the
    ReLU function.
    Intended to improve/replace the ReLU masterrace

    "The best discovered activation function",
    - Authors
    See: https://arxiv.org/abs/1710.05941

     Params
     ------
     X : ndarray
        the usual
     b : ndarray
        scaling parameter. Authors say it can be a
        constant scalar (1.0), but always immediately follow it up
        saying it's best as a channel-wise trainable param.

        So it's going to be always be treated as an array with the same
        shape as X.shape[1:] (exlude batch dim)
    """
    @staticmethod
    def swish(x, b):
        return x * Sigmoid.sigmoid(x * b)

    @staticmethod
    def swish_prime(x, b):
        sig_xb = Sigmoid.sigmoid(x*b)
        y = x * sig_xb
        gb = y * (x - y)
        gx = sig_xb + b * (y * (1 - sig_xb))
        return gx, gb

    def forward(self, X, b):
        self.cache = np.copy(X)
        Y = self.swish(X, b)
        return Y

    def backward(self, gY, B):
        X = self.cache
        gx, gb = self.swish_prime(X, B)
        gX = gY * gx
        gB = np.sum(gY * gb, axis=0)
        return gX, gB



#==============================================================================
#                          Loss Functions
#==============================================================================


#------------------------------------------------------------------------------
# Cross entropy
#------------------------------------------------------------------------------

class LogisticCrossEntropy(Function): #
    """ Logistic cross-entropy loss defined on sigmoid activation

    Truth labels are converted to one-hot encoding to reduce
    the duplicate select and reduction ops in forward and backprop

    """
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def logistic_cross_entropy(x, t):
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        lhs = -t * np.log(x)
        rhs = (1 - t) * np.log(1 - x)
        return np.mean(lhs - rhs)

    @staticmethod
    def logistic_cross_entropy_prime(x, t):
        return (x - t) / x.size


    def forward(self, X, t_vec):
        """
        Params
        ------
        X : ndarray.float32, (N, D)
            linear output of network's final layer
        t_vec : ndarray.int32, (N,) ----> (N, D)
            truth labels on each sample, converted from a 1D
            vector of vals within [0, D) to 2D 1-hot (with binary vals)

        Returns
        -------
        Y : float, (1,)
            average cross entropy error over all samples
        p : ndarray.int32, (N,D)
            Network approximations on class labels (for accuracy metrics)
        """
        # Check dimensional integrity
        assert X.ndim == 2 and t_vec.shape[0] == X.shape[0]
        print('Logistic forward')


        # Convert labels to 1-hot
        t = utils.to_one_hot(np.copy(t_vec), X.shape[-1]) # (N,D)

        # Sigmoid activation
        #-------------------
        p = self.sigmoid(np.copy(X))
        self.cache = p, t

        # Average cross-entropy
        #----------------------
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        Y = self.logistic_cross_entropy(np.copy(p), t)
        return Y, p


    def backward(self, *args):
        """ Initial gradient to be chained through the network
        during backprop

        Params
        ------
        p : ndarray.float32, (N,D)
            sigmoid activation on network forward output
        t : ndarray.int32, (N,D), 1-hot
            ground truth labels for this sample set

        Returns
        -------
        gX : ndarray.float32, (N, D)
            derivative of X (network output) wrt the logistic loss

        """
        # Retrieve vars
        p, t = self.cache

        # Get grad
        #---------
        gX = self.logistic_cross_entropy_prime(p, t)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SoftmaxCrossEntropy(Function): #
    """ Cross entropy loss defined on softmax activation

    Truth labels are converted to one-hot encoding to reduce
    the duplicate select and reduction ops in forward and backprop

    """
    @staticmethod
    def softmax(x):
        return Softmax.softmax(x)

    @staticmethod
    def softmax_cross_entropy(x, t):
        tlog_x = -(t * np.log(x))
        return np.mean(np.sum(tlog_x, axis=1))

    @staticmethod
    def softmax_cross_entropy_prime(x, t):
        return (x - t) / x.shape[0]


    def forward(self, X, t_vec):
        """ Cross entropy loss function defined on a
        softmax activation

        Params
        ------
        X : ndarray float32, (N, D)
            output of network's final layer with no activation. *2D assumed*
        t_vec : ndarray int32, (N,) ----> (N, D)
            truth labels for each sample, where int values range [0, D)
            converted to (N, D) one-hot encoding

        Returns
        -------
        Y : float, (1,)
            average cross entropy error over all samples
        p : ndarray.int32, (N,D)
            Network approximations on class labels (for accuracy metrics)
        """

        # Check dimensional integrity
        assert X.ndim == 2 and t_vec.shape[0] == X.shape[0]

        # Convert labels to 1-hot
        t = utils.to_one_hot(np.copy(t_vec)) # (N,D)

        # Softmax activation
        #-------------------
        p = self.softmax(X)
        self.cache = p, t

        # Average cross entropy
        #----------------------
        Y = self.softmax_cross_entropy(np.copy(p), t)
        return Y, p


    def backward(self, *args):
        """ Initial backprop gradient grad on X wrt loss

        Params
        ------
        p : ndarray.float32, (N,D)
            sigmoid activation on network forward output
        t : ndarray.int32, (N,D), 1-hot
            ground truth labels for this sample set

        Returns
        -------
        gX : ndarray, (N, D)
            derivative of X (network prediction) wrt the cross entropy loss
        """

        # Retrieve data
        p, t = self.cache

        # Calculate grad
        #---------------
        gX = self.softmax_cross_entropy_prime(p, t)
        return gX

#==============================================================================
#                        Regularization and Normalization Functions
#==============================================================================

@NOTIMPLEMENTED
class Dropout(Function):
    """ Randomly drops input elements with the probability given by ratio

    Dropout will zero out elements of the input at random with
    probability ratio p, and then scale the remaining elements by
    by 1 / (1 - p).

    Dropout is a well established technique commonly used in feedforward
    networks in order to reduce overfitting on a training set. The idea
    being that network connections, instead of learning features by the
    detection and context of others, the connections are instead encouraged
    to learn more robust detection of features.

    """
    def __init__(self, drop_ratio=0.5):
        self.drop_ratio = drop_ratio

    def get_mask(self, X):
        rands = np.random.rand(*X.shape)
        drop  = self.drop_ratio
        scale = 1 / (1 - drop)
        mask = (rands >= drop) * scale
        return mask.astype(np.float32)

    @staticmethod
    def dropout(x, mask):
        return x * mask

    @staticmethod
    def dropout_prime(y, mask):
        return y * mask

    def forward(self, X):
        mask = self.get_mask(X)
        self.cache = mask
        Y = self.dropout(np.copy(X), mask)
        return Y

    def backward(self, gY):
        mask = self.cache
        gX = self.dropout_prime(gY, mask)
        return gX




#==============================================================================
#------------------------------------------------------------------------------
#                            FUNCTION COLLECTIONS
#------------------------------------------------------------------------------
#==============================================================================

MATH = {}

ACTIVATIONS = {'sigmoid': Sigmoid,
                'tanh': Tanh,
                'softmax': Softmax,
                'swish': Swish,
                }

CONNECTIONS = {'bias': Bias,
               'matmul': MatMul,
               'linear': Linear,
               }

OBJECTIVES = {'logistic_cross_entropy': LogisticCrossEntropy,
              'softmax_cross_entropy':  SoftmaxCrossEntropy,
              }
