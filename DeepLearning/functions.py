""" All math functions and ops used by a model

This module provides the foundation of functions and operations
to build a network and optimize models.

It contains only the units that would used within a model.
Other functions, for training or data processing, can be found in `utils.py`

Module components
=================
Function : base class for functions

atomic functions: elementary functions and factors
    Log, Square, Exp, Power, Bias, Matmul, Sqrt

composite functions : functions that combine other Functions
    Linear

ReductionFunction : functions that reduce dimensionality
    Sum, Mean, Prod, Max, Min

activation functions: nonlinearities
    ReLU, ELU, SeLU, Sigmoid, Tanh, Softmax

loss functions : objectives for gradient descent
    SoftmaxCrossEntropy

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


def preserve_default(method):
    """ always saves method inputs to fn_vars """
    @wraps(method)
    def preserve_args(self, *args, **kwargs):
        self.fn_vars = args
        return method(self, *args, **kwargs)
    return preserve_args


def preserve(inputs=True, outputs=True):
    """ Preserve inputs and outputs to fn_vars
    Also provides kwargs for individual cases
     - sometimes you only need inputs, or outputs
       this decorator allows you to choose which
       parts of the function you want to preserve
    """
    def inner_preserve(method):
        """ outer preserve is like a decorator to this
        decorator
        """
        @wraps(method)
        def preserve_args(self, *args, **kwargs):
            my_args = ()
            if inputs:
                my_args += args
            ret = method(self, *args, **kwargs)
            if outputs:
                my_args += (ret,)
            self.fn_vars = my_args
            return ret
        return preserve_args
    return inner_preserve




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

    def __repr__(self):
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

    @NOTIMPLEMENTED
    def forward(self, X, *args):
        pass

    @NOTIMPLEMENTED
    def backward(self, gY, *args): pass

    def __call__(self, *args, backprop=False):
        func = self.backward if backprop else self.forward
        return func(*args)


#==============================================================================
#                             Math Functions
#==============================================================================

#------------------------------------------------------------------------------
#  atomic functions
#------------------------------------------------------------------------------

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


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
        """ ASSUMES x == sigmoidx(x) """
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

#==============================================================================
#                          Loss Functions
#==============================================================================


#------------------------------------------------------------------------------
# Cross entropy
#------------------------------------------------------------------------------

class LogisticCrossEntropy(Function):
    """ Logistic cross-entropy loss defined on sigmoid activation

    Truth labels are converted to one-hot encoding to reduce
    the duplicate select and reduction ops in forward and backprop

    """
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def log_cross_entropy(x, t):
        lhs = -t * np.log(x)
        rhs = (1 - t) * np.log(1 - x)
        return np.mean(lhs - rhs)

    @staticmethod
    def log_cross_entropy_prime(x, t):
        return x - t

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

        p : ndarray.int32, (N,)
            Network approximations on class labels (for accuracy metrics)
        """
        # Check dimensional integrity
        assert X.ndim == 2 and t_vec.shape[0] == X.shape[0]

        # Convert labels to 1-hot
        t = utils.to_one_hot(np.copy(t_vec)) # (N,D)

        # Sigmoid activation
        #-------------------
        p = self.sigmoid(X)
        self.cache = p, t

        # Average cross-entropy
        #----------------------
        Y = self.log_cross_entropy(np.copy(p), t)

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
        gX = self.log_cross_entropy_prime(np.copy(p), t) / p.size
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SoftmaxCrossEntropy(Function):
    """ Cross entropy loss defined on softmax activation

    Notes
    -----
    : Assumes input had no activation applied
    : *Assumes network input is 2D*

    Attributes
    ----------
    softmax : Softmax :obj:
        instance of Softmax function
    """

    softmax = Softmax()

    def forward(self, X, t):
        """ Cross entropy loss function defined on a
        softmax activation

        Params
        ------
        X : ndarray float32, (N, D)
            output of network's final layer with no activation. *2D assumed*

        t : ndarray int32, (N,)
            truth labels for each sample, where int values range [0, D)

        Returns
        -------
        loss : float, (1,)
            the cross entropy error between the network's predicted
            class labels and ground truth labels
        """
        assert X.ndim == 2 and t.shape[0] == X.shape[0]

        N = t.shape[0]

        Y = self.softmax(X)
        #self.fn_vars = Y, t # preserve vars for backward
        self.cache = Y, t
        p = -np.log(Y[np.arange(N), t])
        loss = np.sum(p, keepdims=True) / float(N)
        return loss

    def backward(self, gLoss):
        """ gradient function for cross entropy loss

        Params
        ------
        gLoss : ndarray, (1,)
            cross entropy error (output of self.forward)

        Returns
        -------
        gX : ndarray, (N, D)
            derivative of X (network prediction) wrt the cross entropy loss
        """
        #gX, t = self.get_fn_vars() # (Y, t)
        gX, t = self.cache
        N = t.shape[0]
        gX[np.arange(N), t] -= 1
        gX = gLoss * (gX / float(N))
        return gX
