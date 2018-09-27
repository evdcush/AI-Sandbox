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
    composite functions : functions that combine other functions
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
#------------------------------------------------------------------------------
#                              Functions
#------------------------------------------------------------------------------
#==============================================================================



# Base function class
#------------------------------------------------------------------------------

# Function
# ------------
# inherits :
# derives  : ReductionFunction
class Function:
    """ Function for various, mostly mathematical ops

    Designed to abstract as much functionality from
    child classes as possible to reduce boilerplate.

    In practice, this works well with the "caching"
    operations for functions, and in some cases,
    the forward ops

    Function ops
    -------------
        forward : f(X)
            the function

        backward : f'(X)
            the derivative of the function

    Attributes
    ----------
    name : str
        name of the function (based on function class name)

    cache : object
        cache is whatever the functions need to store to reduce
        redundant computation and restore crucial data during
        backpropagation
            WARNING: cache is automatically cleared when getted

    function : object (most likely a numpy function)
        The function the class is defined by. All Functions use
        numpy ops, and for the more atomic functions, like
        exponential, square-root, etc., storing it as a class variable
        works okay

    function_kwargs : dict | None
        some functions take kwargs. Most don't. This allows, again,
        for some Functions to exploit the parent class (Function)
        super forward

    """
    name : str
    #_cache = None
    function = lambda *args: print("DID NOT OVERRIDE cls.function ATTR")
    function_kwargs = {}

    def __init__(self, *args, **kwargs):
        _cache = None
        self.name = self.__class__.__name__
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    def __repr__(self):
        return "functions.{}".format(self.name)

    @property
    def cache(self):
        """ NOTE: 'clears' cache upon access
        (since it is always reset in backprop)
        """
        #print(repr(self))
        cache_objs = self._cache
        #self._cache = None
        return cache_objs

    @cache.setter
    def cache(self, *fvars):
        #print(repr(self))
        self._cache = fvars if len(fvars) > 1 else fvars[0]

    def forward(self, X, *args): pass
        #func = self.function
        #if len(self.function_kwargs) > 0:
        #    Y = func(X, *args, **self.function_kwargs)
        #else:
        #    Y = func(X, *args)
        #self.cache = X, *args, Y
        #return Y

    @NOTIMPLEMENTED
    def backward(self, gY, *args): pass

    def __call__(self, *args, backprop=False):
        func = self.backward if backprop else self.forward
        return func(*args)


class Sigmoid(Function):
    """ Logistic sigmoid activation """
    #function = lambda X: 1 / (1 + np.exp(-X))

    def forward(self, X):
        Y = 1 / (1 + np.exp(-X))
        #self.fn_vars = Y
        self.cache = Y
        return Y

    def backward(self, gY):
        Y = self.cache
        gX = gY * Y * (1 - Y)
        return gX


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
