""" All math functions and ops used by a model

This module provides the foundation of functions and operations
to build a network and optimize models.

It contains only the units that would used within a model.
Other functions, for training or data processing, can be found in `utils.py`

Module components
=================
Function : base class for all functions
    MathFunction : base class for math ops
        AtomicFunction : elementary functions and factors
            : Log, Square, Exp, Power, Scale, Bias, Matmul, Sqrt, Abs,
              Clip, Scale
        CompositeFunction :
        ReductionFunction :
    ManipulationFunctions :
        : Where, Reshape, ExpandDims, Concat
    NoiseInjections :
        Dropout, GumbelSoftmax, Gaussian
    Normalization :
        LayerNorm
    Pooling :
        : AveragePooling, Unpooling
    Activation :
        Relu, ClippedRelu, Crelu, Elu, sigmid, Leaky Relu,
        Log_softmax, prelu, selu, sigm, tanh, swish, softplus,



#
Initializers:
    - Constant, Zero, One, (HeNormal), Glorot, Uniform


# Components of the module
#-------------------------
Functions : a collection of base functions
    Mostly activation and mathematic ops

Layers : the primary architectural feature of a network
    Layers use a set of hyperparameters (like weights), Functions,
    and data to produce an output

Network : manages data flow through a series of layers
    The relation of Networks to Layers is analogous to Layers and Functions
    However, a Network also manages the data flow for the forward and backward
    stages of data

Model : an interface to a Network
    Models are typically composed of a single Network, and are more
    task-specific, such as a "discriminative" Model or "generative," though
    any given task would also require structural changes down the hierarchy.

Optimizer : optimizes the model hyperparameters
    Optimizers receive the model output, and the error from a loss function
    and adjusts model hyperparameters to improve model fn_varsuracy


"""


import os
import sys
import code
from functools import wraps
from pprint import PrettyPrinter as ppr

import numpy as np

from utils import TODO, NOTIMPLEMENTED, INSPECT

""" submodule imports
utils :
    `TODO` : decorator
        serves as comment and call safety

    `NOTIMPLEMENTED` : decorator
        raises NotImplementedErrorforces if class func has not been overridden

    `INSPECT` : decorator
        interrupts computation and enters interactive shell,
        where the user can evaluate the input and output to func
"""


#==============================================================================
#------------------------------------------------------------------------------
#                              Network ops
#------------------------------------------------------------------------------
#==============================================================================

#==============================================================================
# Layers
#==============================================================================
@TODO
class NetworkLayer:
    def __init__(self):
        pass

    def __call__(self, x):
        # init weights if None
        pass
@TODO
class Dense(NetworkLayer):
    """ Fully connected linear layer
    """
    def __init__(self,):
        pass

    def __call__(self, x):
        pass

''' # TODO
batch-norm
layernorm

'''


''' # IF TIME:
- LSTM
- Conv2D
dropuout
'''







"""
###############################################################################
#                                                                             #
#  888888888888        ,ad8888ba,        88888888ba,          ,ad8888ba,      #
#       88            d8"'    `"8b       88      `"8b        d8"'    `"8b     #
#       88           d8'        `8b      88        `8b      d8'        `8b    #
#       88           88          88      88         88      88          88    #
#       88           88          88      88         88      88          88    #
#       88           Y8,        ,8P      88         8P      Y8,        ,8P    #
#       88            Y8a.    .a8P       88      .a8P        Y8a.    .a8P     #
#       88             `"Y8888Y"'        88888888Y"'          `"Y8888Y"'      #

- All of the function "interfaces" or initializers for the Functions
  - eg: for Sum, we need to make it's consituent `sum` that calls Sum


"""
###############################################################################





#==============================================================================
# Globals
#==============================================================================
#------------------------------------------------------------------------------
# Shift invariant ops
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Shift invariant nodes
# ========================================

#==== Thing

# Thing
#-----------


#==============================================================================
#------------------------------------------------------------------------------
#                              Functions
#------------------------------------------------------------------------------
#==============================================================================


#==============================================================================
# Base Function classes : Function, MathFunction, ReductionFunction
#==============================================================================

# Function
# --------
# inherits :
# derives : MathFunction
class Function:
    """
    Base class for a function, must be overridden

    # All functions take an input, do something, and return an output.

    Instance methods are all expected to be overridden. However,
      Function provides a useful initialization function
      for assigning an arbitrary number of attributes that most
      Function child instances will utilize

    """
    def __init__(self, **kwargs):
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    @NOTIMPLEMENTED
    def __call__(self,):
        """ Dispatch to forward or backward """
        pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# MathFunction
# ------------
# inherits : Function
# derives  : ReductionFunction
class MathFunction(Function):
    """
    Function for various mathematical ops

    Function ops
    -------------
    MathFunctions have two parts:
        forward : function
            the typical form of any given function
        backward : gradient function
            the derivative of the function

    Attributes
    ----------
    fn_vars : (no defined type)
        the variables stored by functions for use in backprop

    Callee dispatch
    ----------------
    Callers of MathFunction instances will only
    use MathFunction.__call__

    MathFunction.__call__ will then dispatch to
    forward or backward based on the caller-specified
    'backprop' kwarg.

    """
    _fn_vars = None

    @property
    def fn_vars(self,):
        return _fn_vars

    def set_fn_vars(self, fvars):
        self._fn_vars = fvars

    def reset_fn_vars(self,):
        self._fn_vars = None

    @NOTIMPLEMENTED
    def forward(self):
        pass

    @NOTIMPLEMENTED
    def backward(self):
        pass

    def __call__(self, *args, backprop=False, **kwargs):
        """ Dispatch to forward or backward """
        func = self.backward if backprop else self.forward
        return func(*args, **kwargs)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ReductionFunction
# -----------------
# inherits : MathFunction
# derives  : Sum, Mean, Prod, Max Min
class ReductionFunction(MathFunction):
    """
    MathFunction for dimensionality reduction ops
    like sum or mean

    # Attributes
    #-----------
    dims : list (int)
        the original shape of the input before reduction
    axes : list (int)
        the axes or axis that were reduced

    """
    _dims = []
    _axes = []

    def get_reduction_vars(self):
        return _dims, _axes

    def set_reduction_vars(self, dims, axes):
        self._dims = list(dims)
        self._axes = list(axes)

    def reset_reduction_vars(self,):
        self._dims = []
        self._axes = []

    @staticmethod
    def apply(fn, L):
        """ a map function with no return """
        for i in L:
            fn(i)

    def restore_shape(self, Y):
        """ Restores a variable Y to the original
            shape of the input var X

        Restore_shape is called in self.backward, during backpropagation

        Restoration steps
        -----------------
        1 - The dimensions (shape) of the original input X and the
            axes used to reduce Y are retrieved

        2 - The current dimensions of Y have any missing axes
            restored
            For example:
                if X.shape = (8, 32, 32, 3) and
                Y = a_reduction_function(X, axis=2)
                    (so Y.shape = (8, 32, 3))
                Y would first have axis 2 restored,
                so that Y.shape --> (8, 32, 1, 3)

        3 - Y has the full original shape of the input X restored by
            broadcasting
        """
        # Dimension vars
        dims_x, axes = self.get_reduction_vars()
        dims_y = y.shape

        # Get reshape dims
        #-----------------
        # reshape_dims will have a 1 for whatever axes were reduced
        #   (will be all 1s if no axes were given)
        reshape_dims = list(dims_x) if dims_y else [1]*len(dims_x)
        apply(lambda i: reshape_dims.__setitem__(i, 1), list(axes))

        # Restore the dimensionality of y
        y = np.broadcast_to(y.reshape(reshape_dims), dims_input)
        return y



#==============================================================================
# Derived functions
#==============================================================================
#------------------------------------------------------------------------------
# Atomic math functions
#------------------------------------------------------------------------------


class Sum(ReductionFunction):
    """ """
    fn_vars = None

    def rebroadcast_dims(self, x):
        x

    def forward(self, x, axis=None):
        self.fn_vars = axis

        return h

    def backward(self, gY):
        """ exp'(x) = exp(x) """
        gX = self.fn_vars * gY
        self.reset_stored_data()
        return gX

@TODO
class Max(MathFunction):
    """ Elementwise exponential function """
    fn_vars = None

    def forward(self, x):
        h = np.exp(x)
        self.fn_vars = h
        return h

    def backward(self, gY):
        """ exp'(x) = exp(x) """
        gexp_x = self.fn_vars * gY
        self.reset_stored_data()
        return gexp_x

@TODO
class Power(MathFunction):
    """ Elementwise exponential function """
    fn_vars = None

    def forward(self, x):
        h = np.exp(x)
        self.fn_vars = h
        return h

    def backward(self, gY):
        """ exp'(x) = exp(x) """
        gexp_x = self.fn_vars * gY
        self.reset_stored_data()
        return gexp_x

@TODO
class Log(MathFunction):
    """ Elementwise exponential function """
    fn_vars = None

    def forward(self, x):
        h = np.exp(x)
        self.fn_vars = h
        return h

    def backward(self, gY):
        """ exp'(x) = exp(x) """
        gexp_x = self.fn_vars * gY
        self.reset_stored_data()
        return gexp_x
@TODO
class Where(MathFunction):
    """ Elementwise exponential function """
    fn_vars = None

    def forward(self, x):
        h = np.exp(x)
        self.fn_vars = h
        return h

    def backward(self, gY):
        """ exp'(x) = exp(x) """
        gexp_x = self.fn_vars * gY
        self.reset_stored_data()
        return gexp_x

#------------------------------------------------------------------------------
# Atomic math functions
#------------------------------------------------------------------------------
class Exp(MathFunction):
    """ Elementwise exponential function """
    fn_vars = None

    def forward(self, x):
        h = np.exp(x)
        self.fn_vars = h
        return h

    def backward(self, gY):
        """ exp'(x) = exp(x) """
        gexp_x = self.fn_vars * gY
        self.reset_stored_data()
        return gexp_x


class Bias(MathFunction):
    """ Adds bias B to some data"""
    def forward(self, X, B):
        return X + B

    def backward(self, gY):
        gB = gY.sum(0)
        return gB


class MatMul(MathFunction):
    """ Performs matrix multiplication between
        two matrices x, w

    # Matmul assumptions
    #-------------------
    X : external data sample
        X is the gradient chained through the network
        shape : (N, m)
            N : int
                number of samples
            m : int
                arbitrary numberf of current features

    W : weight matrix
        shape : (m, k)
            m : int
                input dims
            k : int
                output dims
    """
    fn_vars = None

    def matmul(self, X, W):
        return np.matmul(X, W)

    def forward(self, X, W):
        """ matmul on x,w, assumes x.shape[-1] == w.shape[0] """
        self.fn_vars = (X, W)
        Y = self.matmul(X, W)
        return Y

    def backward(self, gY):
        """  """
        X_in, W_in = self.fn_vars

        # The derivative of X is what will continue being propagated
        gX = self.matmul(W_in, gY)

        # The derivative of W is what will be used for opt updates
        gW = self.matmul(X_in, gY)
        self.reset_stored_data()
        return gX, gW


#------------------------------------------------------------------------------
# Composite math functions
#------------------------------------------------------------------------------
class Linear(MathFunction):
    """ Performs linear transformation (function) using the
    the MatMul and Bias functions:
        y = XW + b
    Please reference MatMul and Bias for greater detail

    # Class variables
    #----------------
    Linear must keep an instance of MatMul to insure any
      any data stored during the forward pass will still be available
      during backpropagation. Bias is kept simply for completeness/neatness.

    _matmul : instance of MatMul
        matmul will store the X, W passed to Linear
    _bias : instance of Bias
        bias does not store anything. It's just neater to have it here ;]

    """
    _matmul = MatMul()
    _bias   = Bias()

    @property
    def matmul(self,):
        return self._matmul

    @property
    def bias(self,):
        return self._bias

    def reset(self,):
        self._matmul.reset_fn_vars()
        self._bias.reset_fn_vars()

    def forward(self, X, W, b):
        """ Computes Y = X.W + b, eg Y = bias(MatMul(X,W), b) """
        XW = self.matmul(X, W)
        B  = self.bias(XW, b)
        Y = XW + B
        return Y

    def backward(self, gY):
        """ backprop through linear func

        Parameters:
        -----------
        gY : ndarray
            current backprop gradient from loss

        Returns:
        --------
        gX : ndarray
            chained backprop gradient
        gW : ndarray
            gradient of weight var W for update
        gb : ndarray
            gradient of bias var b for update

        """
        gX, gW = self.matmul(gY, backprop=True)
        gb = self.bias(gY, backprop=True)

        return gX, (gW, gb)


#==============================================================================
# Activation functions
#==============================================================================

# Base Activation function class
# ========================================
class ActivationFunction(MathFunction):
    """ Just a nonlinear MathFunction """

''' # Necessary atomic funcs:
- sum
- max
- power
- log
- where?
'''


class AttrDict(dict):
    """ simply a dict fn_varsessed/mutated by attribute instead of index """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


#------------------------------------------------------------------------------
# ReLU family
#------------------------------------------------------------------------------
class RelU(ActivationFunction):
    pass

class LeakyRelu(ReLU):
    pass

class PReLU(ReLU):
    pass

class RRelU(ReLU):
    pass

class SeLU(ReLU):
    pass





class RRelU(ReLU):
    pass

class Softplus(ActivationFunction):
    pass



class Sigmoid(ActivationFunction):
    pass


#==============================================================================
# Matrix manipulation functions
#==============================================================================

class Where(Function):
    pass


