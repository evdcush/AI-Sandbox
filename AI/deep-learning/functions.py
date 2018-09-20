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
        Dropout, GumbelSoftmax
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
    - HUGE POTENTIAL ISSUE: since you aren't returning a Variable,
        or Tensor, how is the returned thing going to retain
        gradients of Functions?
- maybe make a reset_fn_vars decorator for the MathFunction backward

## concrete stuff:
- docstrings minmax class
- test min and max
- tets atomic funcs




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

#------------------------------------------------------------------------------
# Handy decorators
#------------------------------------------------------------------------------

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
    return actual_decorator

#==============================================================================
# Base Function classes :
#  Function, MathFunction, ReductionFunction
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
    def fn_vars(self):
        fvars = self._fn_vars
        return fvars

    @fn_vars.setter
    def fn_vars(self, *fvars):
        self._fn_vars = fvars if len(fvars) > 1 else fvars[0]

    def get_fn_vars(self,reset=False):
        fvars = self.fn_vars
        if reset:
            self.reset_fn_vars()
        return fvars

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
    fn_vars : (dims, axes)
        dims : list (int)
            the original shape of the input before reduction
        axes : list (int)
            the axes or axis that were reduced
    """
    @NOTIMPLEMENTED
    def forward(self, X, axis=None, keepdims=False):
        pass

    @staticmethod
    def apply(fn, L):
        """ a map function with no return """
        for i in L:
            fn(i)

    def format_axes(self, axes):
        ax = []
        if axes is None: return ax
        if type(axes) == int:
            ax.append(axes)
            return ax
        return list(axes)

    def restore_shape(self, Y, reset=False):
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
        dims_X, axes = self.get_fn_vars(reset=reset)
        dims_X = list(dims_X); axes = self.format(axes)
        dims_Y = Y.shape

        # Get reshape dims
        #-----------------
        # reshape_dims will have a 1 for whatever axes were reduced
        #   (will be all 1s if no axes were given)
        reshape_dims = list(dims_X) if dims_Y else [1]*len(dims_X)
        self.apply(lambda i: reshape_dims.__setitem__(i, 1), list(axes))

        # Restore the dimensionality of y
        Y = np.broadcast_to(Y.reshape(reshape_dims), dims_X)
        return Y



#==============================================================================
# Derived functions
#==============================================================================
#------------------------------------------------------------------------------
# Atomic math functions :
#  Exp, Log, Power, Square, Sqrt, Scale, Bias, MatMul, Clip
#------------------------------------------------------------------------------
class Exp(MathFunction):
    """ Elementwise exponential function """
    def forward(self, X):
        Y = self.fn_vars = np.exp(x)
        return Y

    def backward(self, gY):
        Y = self.get_fn_vars(reset=True)
        gX = Y * gY
        return gX


class Bias(MathFunction):
    """ Adds bias vector B to a matrix X

    The bias B is a vector, or 1D-matrix, whose size
    matches the last dimension of matrix X
      eg: X.shape[-1] == B.shape[0]

    """
    def forward(self, X, B):
        return X + B

    def backward(self, gY):
        gX = gY
        # Must reduce gY if gY.ndim > 2
        ax = 0 if gY.ndim <= 2 else tuple(range(gY.ndim - 1))
        gB = gY.sum(axis=ax)
        return gX, gB


class MatMul(MathFunction):
    """ Performs matrix multiplication between
        a matrix X and a weight matrix W

    Note - MatMul assumes X, W are matrices whose shapes have the following
        properties:
    Params
    ------
    X : ndarray
        Has arbitrary number of dimensions in shape (s0, s1, ..., m),
        but the final column of shape m are the features

    W : ndarray, weight matrix
        Of shape (m, k)
    """
    def forward(self, X, W):
        """ matmul on X, W assumes X.shape[-1] == W.shape[0] """
        self.fn_vars = X, W
        Y = np.matmul(X, W)
        return Y

    def backward(self, gY):
        """ While the forward matmul was fairly simple,
        as numpy can interpret the dimensionality
        of the matmul of X wrt to the smaller W, the matrices
        X and gY must be flattened to 2D to get the proper
        gW shape
        """
        # retrieve inputs
        X, W = self.get_fn_vars(reset=True)
        m, k = W.shape

        # get grads
        gX = np.matmul(gY, W.T)

        # need to reshape X.T, gY if ndims > 2, to match W shape
        gW = np.matmul(X.T.reshape(m, -1), gY.reshape(-1, k))
        return gX, gW


# >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><
# TODO
class Log(MathFunction):
    """ """
    def forward(self, X):
        pass

    def backward(self, gY):
        pass

class Power(MathFunction):
    """ """
    def forward(self, X, p):
        pass

    def backward(self, gY):
        pass

class Square(MathFunction):
    """ """
    def forward(self, X):
        pass

    def backward(self, gY):
        pass

class Sqrt(MathFunction):
    """ """
    def forward(self, X):
        pass

    def backward(self, gY):
        pass

class Scale(MathFunction):
    """ Elementwise multiplication between matrices """
    def forward(self, X, Z):
        pass

    def backward(self, gZ):
        pass

class Clip(MathFunction):
    """ """
    def forward(self, X, lhs=None, rhs=None):
        pass

    def backward(self, gY):
        pass

# TODO
# >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><


#------------------------------------------------------------------------------
# Reduction functions :
#  Sum, Mean, Prod, Max, Min
#------------------------------------------------------------------------------
class Sum(ReductionFunction):
    """ Compute sum along axis or axes """

    def forward(self, X, axis=None, keepdims=False):
        self.fn_vars = X.shape, axis
        Y = np.sum(X, axis=axis, keepdims=keepdims)
        return Y

    def backward(self, gY):
        gX = self.restore_shape(gY, reset=True)
        return gX


class Mean(ReductionFunction):
    """ Compute mean along axis or axes """

    def forward(self, X, axis=None, keepdims=False):
        self.fn_vars = X.shape, axis
        Y = np.mean(X, axis=axis, keepdims=keepdims)
        return Y

    def backward(self, gY):
        shape_in, axes = self.fn_vars
        axes = self.format_axes(axes)

        # Recover number of elements averaged out
        if axes:
            num_elements_avgd = np.prod([shape_in[i] for i in axes])
        else:
            # if axes is empty, mean was taken over all elements
            num_elements_avgd = np.prod(shape_in)

        gX = self.restore_shape(gY, reset=True) / num_elements_avgd
        return gX


class Prod(ReductionFunction):
    """ Compute product along axis or axes """

    def forward(self, X, axis=None, keepdims=False):
        self.fn_vars = X.shape, axis
        Y = np.prod(X, axis=axis, keepdims=keepdims)
        self.X = X
        self.Y = Y
        return Y

    def reset_fn_vars(self):
        super().reset_fn_vars()
        self.X = self.Y = None

    def backward(self, gY):
        X = self.X
        Y = self.Y
        gX = self.restore_shape(gY*Y) / X
        self.reset_fn_vars()
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class MaxMin(ReductionFunction):
    """ Base class for max, min funcs """
    MM_func  = None
    cmp_func = None

    def reset_fn_vars(self):
        super().reset_fn_vars()
        self.X = self.Y = None

    def forward(self, X, axis=None, keepdims=False):
        self.fn_vars = X.shape, axis
        Y = self.MM_func(X, axis=axis, keepdims=keepdims)
        self.X = X
        self.Y = Y
        return Y

    def backward(self, gY):
        X  = self.X
        Y  = self.restore_shape(self.Y)
        gY = self.restore_shape(gY)
        gX = np.where(self.cmp_func(X, Y), gY, 0)
        self.reset_fn_vars()
        return gX


class Max(MaxMin):
    """ Computes max along axis """
    MM_func  = np.max
    cmp_func = lambda x, y: x >= y

class Min(MaxMin):
    """ Computes min along axis """
    MM_func  = np.min
    cmp_func = lambda x, y: x < y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



#------------------------------------------------------------------------------
# Composite math functions :
#  Linear
#------------------------------------------------------------------------------
class Linear(MathFunction):
    """ Performs linear transformation using the
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
        self.matmul.reset_fn_vars()
        self.bias.reset_fn_vars()

    def forward(self, X, W, b):
        """ Computes Y = X.W + b, eg Y = bias(MatMul(X,W), b) """
        Y = self.matmul(X, W)
        B = self.bias(Y, b)
        return Y + B

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
        gB : ndarray
            gradient of bias var b for update

        """
        gX, gW = self.matmul(gY, backprop=True)
         _, gB =   self.bias(gY, backprop=True)
        self.reset()
        return gX, (gW, gB)


#==============================================================================
# Activation functions
#==============================================================================

class ReLU(MathFunction):
    """ standard ReLU activation
    zeroes out any negative elements within matrix
    """
    def forward(self, X):
        Y = self.fn_vars = X.clip(min=0)
        return Y

    def backward(self, gY):
        """ Since Y was clipped at 0, it's elements
            are either 0 or a positive number.
        We can exploit that property to use Y
          directly for indexing the gradient
         """
        Y = self.get_fn_vars(reset=True)
        gX = np.where(Y, gY, 0)
        return gX


class PReLU(ReLU):
    pass

class RRelU(ReLU):
    pass

class ELU(MathFunction):
    """ Exponential Linear Unit """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, X):
        x = self.fn_vars = np.copy(X)
        Y = np.where(x < 0, self.alpha*(np.exp(x)-1), x)
        return Y

    def backward(self, gY):
        X = self.get_fn_vars(reset=True)
        gX = np.where(X < 0, self.alpha * np.exp(X), gY)
        return gX


class SeLU(ELU):
    """ Scaled Exponential Linear Units
    ELU, but with a scalar and finetuning

    SeLU is the activation function featured in
    "Self-Normalizing Neural Networks," and they
    are designed to implicitly normalize feed-forward
    networks.

    Reference
    ---------
    Paper : https://arxiv.org/abs/1706.02515
        explains the properties and derivations for SeLU
        parameters, but the precision values below from their
        project code @ github.com/bioinf-jku/SNNs/

    Parameters : alpha, scale
        github.com/bioinf-jku/SNNs/blob/master/getSELUparameters.ipynb
        github.com/bioinf-jku/SNNs/blob/master/selu.py
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    elu = ELU(alpha=alpha)

    def __init__(self, *args, **kwargs): pass # insure alpha integrity

    def forward(self, X):
        assert self.elu.alpha == self.alpha # sanity check
        Y = self.scale * self.elu.forward(X)
        return Y

    def backward(self, gY):
        gX = self.scale * self.elu.backward(gY)
        return gX


class Sigmoid(MathFunction):
    """ Logistic sigmoid activation """
    def forward(self, X):
        Y = 1 / (1 + np.exp(-X))
        self.fn_vars = Y
        return Y

    def backward(self, gY):
        Y = self.get_fn_vars(reset=True)
        gX = gY * Y * (1 - Y)
        return gX


class Tanh(MathFunction):
    """ Hyperbolic tangent activation """
    def forward(self, X):
        Y = self.fn_vars = np.tanh(X)
        return Y

    def backward(self, gY):
        Y = self.get_fn_vars(reset=True)
        gX = gY * (1 - np.square(Y))
        return gX




class Softmax(MathFunction):
    """ Softmax activation """
    # reduction kwargs
    kw = {'axis':1, 'keepdims':True}

    @preserve(inputs=False)
    def forward(self, X):
        """ Since softmax is translation invariant
        (eg, softmax(x) == softmax(x+c), where c is some constant),
        it's common to first subtract the max from x before
        input, to avoid numerical instabilities risked with
        very large positive values

        Params
        ------
        X : ndarray, (N, K)
            input assumed to be 2D, N = num samples, K = num features

        Returns
        -------
        Y : ndarray (N, K)
            prob. distribution over K features, sums to 1
        """
        kw = self.kw # (axis=1, keepdims=True)
        x_exp = np.exp(X - X.max(**kw))
        Y = x_exp / np.sum(x_exp, **kw)
        return Y

    def backward(self, gY):
        kw = self.kw
        Y = self.get_fn_vars(reset=True)
        gY *= Y
        gsum = np.sum(gY, **kw)
        gX = gY - (Y * gsum)
        return gX


#==============================================================================
# Loss Functions
#==============================================================================

@TODO
class SoftmaxCrossEntropy(MathFunction):
    """ Cross entropy error on pre-softmax activations

    Assumes input to func did not already have softmax act.
    """
    kw = {'axis':1, 'keepdims'=True}
    def softmax(self, X):
        kw = self.kw
        x_exp = np.exp(X - X.max(**kw))
        Y = x_exp / x_exp.sum(**kw)
        return Y

    def forward(self, X, t):
        kw = self.kw
        probs = self.softmax(X)
        likelihood = -np.log(probs[np.arange(x.shape[0]), t])
        Y = np.sum(likelihood, **kw) / X.shape[0]
        return Y


@TODO
class LogisticCrossEntropy(MathFunction):
    """ log loss function """
    pass

@TODO
class MeanSquaredError(MathFunction):
    """ MSE """
    pass
