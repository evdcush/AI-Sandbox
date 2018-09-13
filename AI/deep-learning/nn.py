""" This module contains all the operations related to
the network and model.

# Module structure
# ========================================
The module is structured in the following manner:
  Functions < Layers < Network < Model < Optimizer

- Where X < Y means Y is composed of X, or Y interfaces X
  - So Layers are composed of functions, Networks of Layers, etc.

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
    and adjusts model hyperparameters to improve model accuracy


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
    TODO : decorator
        serves as comment and call safety
    NOTIMPLEMENTED : decorator
        raises NotImplementedErrorforces if class func has not been overridden
    INSPECT : decorator
        interrupts computation and enters interactive shell,
        where the user can evaluate the input and output to func
"""


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
""" Notes:
- Should Function treat the input or output as a class variable?
  - inputs are almost always saved for the backprop process
  - LEAVING OUT FOR NOW, may include

"""

# Base Function class
# ========================================
class Function:
    """ Base class for a function, must be overridden

    All functions take an input, do something, and return an output.
    The only shared property of a Function is __call__,
     though Function provides a simple initialization function
     for any number of attributes passed as kwargs

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

#==============================================================================
# Mathematical functions
#==============================================================================

# Base Math Function class
# ========================================
class MathFunction(Function):
    """ Function for various mathematical ops
    # Function ops
    #-------------
    MathFunctions have two parts:
        forward : function
            the typical form of any given function
        backward : gradient function
            the derivative of the function

    It's likely most MathFunction subclasses will also store
    their inputs for use during the backward process

    # Callee dispatch
    #----------------
    MathFunction callers should ideally only __call__
    MathFunctions dispatch, ideally on some kwarg,
      eg Y = self.my_function(X, fwd=True):

    __call__ : *args, **kwargs
        forward  : *args **kwargs
        backward : *args **kwargs

    """
    acc = None

    def reset_stored_data(self,):
        self.acc = None

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

#------------------------------------------------------------------------------
# Atomic math functions
#------------------------------------------------------------------------------
''' # Necessary atomic funcs:
- sum
- max
- power
- log
- where?
'''

class ReductionFunction(MathFunction):
    """ Reduction functions are functions like
    sum or mean that reduces the dimensionality of
    a matrix.

    They constitute their own subclass of
    MathFunction because of attributes
    that require nontrivial changes to how
    a normal MathFunction performs `backward`s

    # Attributes
    #-----------
    dims : list (int)
        the original shape of the input before reduction
    axis : list (int)
        the axis or axes that were reduced

    """
    dims = []
    axis = []

    def reset_stored_data(self,):
        self.dims = []
        self.axis = []

    def restore_shape(self, x):
        """ Restores a variable x to the original shape of
        this function's input.

        This feature is the primary difference between a
        RotationFunction and a normal MathFunction.

        """
        dims, ax = self.dims, self.ax
        if len(dims) == len(ax):


    @NOTIMPLEMENTED
    def forward(self, axis=None):
        pass

    @NOTIMPLEMENTED
    def backward(self):
        pass

    def __call__(self, *args, backprop=False, **kwargs):
        """ Dispatch to forward or backward """
        func = self.backward if backprop else self.forward
        return func(*args, **kwargs)



class Sum(MathFunction):
    """ Elementwise exponential function """
    acc = None

    def rebroadcast_dims(self, x):
        x

    def forward(self, x, axis=None):
        self.acc = axis

        return h

    def backward(self, grad):
        """ exp'(x) = exp(x) """
        dL_wrt_exp_x = self.acc * grad
        self.reset_stored_data()
        return dL_wrt_exp_x

@TODO
class Max(MathFunction):
    """ Elementwise exponential function """
    acc = None

    def forward(self, x):
        h = np.exp(x)
        self.acc = h
        return h

    def backward(self, grad):
        """ exp'(x) = exp(x) """
        dL_wrt_exp_x = self.acc * grad
        self.reset_stored_data()
        return dL_wrt_exp_x

@TODO
class Power(MathFunction):
    """ Elementwise exponential function """
    acc = None

    def forward(self, x):
        h = np.exp(x)
        self.acc = h
        return h

    def backward(self, grad):
        """ exp'(x) = exp(x) """
        dL_wrt_exp_x = self.acc * grad
        self.reset_stored_data()
        return dL_wrt_exp_x

@TODO
class Log(MathFunction):
    """ Elementwise exponential function """
    acc = None

    def forward(self, x):
        h = np.exp(x)
        self.acc = h
        return h

    def backward(self, grad):
        """ exp'(x) = exp(x) """
        dL_wrt_exp_x = self.acc * grad
        self.reset_stored_data()
        return dL_wrt_exp_x
@TODO
class Where(MathFunction):
    """ Elementwise exponential function """
    acc = None

    def forward(self, x):
        h = np.exp(x)
        self.acc = h
        return h

    def backward(self, grad):
        """ exp'(x) = exp(x) """
        dL_wrt_exp_x = self.acc * grad
        self.reset_stored_data()
        return dL_wrt_exp_x

class Exp(MathFunction):
    """ Elementwise exponential function """
    acc = None

    def forward(self, x):
        h = np.exp(x)
        self.acc = h
        return h

    def backward(self, grad):
        """ exp'(x) = exp(x) """
        dL_wrt_exp_x = self.acc * grad
        self.reset_stored_data()
        return dL_wrt_exp_x


class Bias(MathFunction):
    """ Adds bias b to some data"""
    def forward(self, x, b):
        return x + b

    def backward(self, grad):
        dL_wrt_b = grad.sum(0)
        return dL_dexp



class MatMul(MathFunction):
    """ Performs matrix multiplication between
        two matrices x, w

    # Matmul assumptions
    #-------------------
    x : external data sample
        x is to be saved for use in backprop
      shape : (N, m)
          N : int
              number of samples
          m : int
              arbitrary numberf of current features
    w : weight matrix
        w is the target for optimization update during backprop
      shape : (m, k)
          m : int
              input dims
          k : int
              output dims
    """
    acc = None

    def matmul(self, a, b):
        return np.matmul(a, b)

    def forward(self, x, w):
        """ matmul on x,w, assumes x.shape[-1] == w.shape[0] """
        self.acc = (x, w)
        h = self.matmul(x, w)
        return h

    def backward(self, grad):
        """ dw_dx =  x """
        x_in, w_in = self.acc

        # The derivative of x is what will continue being propagated
        dL_wrt_mmul_x = self.matmul(w_in, grad)

        # The derivative of w is what will be used for opt updates
        dL_wrt_mmul_w = self.matmul(x_in, grad)
        self.reset_stored_data()
        return dL_wrt_mmul_x, dL_wrt_mmul_w


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

    def forward(self, X, W, b):
        """ Computes h = X.W + b, eg h = bias(MatMul(X,W), b) """
        h_mul  = self.matmul(X, W)
        h = self.bias(h_mul, b)
        return h

    def backward(self, grad):
        """ backprop through linear func

        Parameters:
        -----------
        grad : ndarray
            current backprop gradient from loss

        Returns:
        --------
        dL_wrt_X : ndarray
            updated backprop gradient
        dL_wrt_W : ndarray
            gradient of weight var W for update
        dL_wrt_b : ndarray
            gradient of bias var b for update

        """
        dL_wrt_X, dL_wrt_W = self.matmul(grad, backprop=True)
        dL_wrt_b = self.bias(grad, backprop=True)

        return dL_wrt_X, (dL_wrt_W, dL_wrt_b)


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
    """ simply a dict accessed/mutated by attribute instead of index """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

#https://docs.chainer.org/en/latest/reference/functions.html#activation-functions

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
    pass #https://docs.chainer.org/en/latest/reference/generated/chainer.functions.selu.html#chainer.functions.selu


from chainer.functions.activation import elu


def selu(x,
         alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946):
    """Scaled Exponential Linear Unit function.
    For parameters :math:`\\alpha` and :math:`\\lambda`, it is expressed as
    .. math::
        f(x) = \\lambda \\left \\{ \\begin{array}{ll}
        x & {\\rm if}~ x \\ge 0 \\\\
        \\alpha (\\exp(x) - 1) & {\\rm if}~ x < 0,
        \\end{array} \\right.
    See: https://arxiv.org/abs/1706.02515
    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        alpha (float): Parameter :math:`\\alpha`.
        scale (float): Parameter :math:`\\lambda`.
    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.
    """
    return scale * elu.elu(x, alpha=alpha)


class RRelU(ReLU):
    pass

class Softplus(ActivationFunction):
    pass


'''
sigm
hard_sigm
tanh https://docs.chainer.org/en/latest/reference/generated/chainer.functions.tanh.html#chainer.functions.tanh
softmax https://docs.chainer.org/en/latest/reference/generated/chainer.functions.softmax.html#chainer.functions.softmax
softplus https://docs.chainer.org/en/latest/reference/generated/chainer.functions.softplus.html#chainer.functions.softplus
swish https://docs.chainer.org/en/latest/reference/generated/chainer.functions.swish.html#chainer.functions.swish
'''




class Sigmoid(ActivationFunction):
    """ Sigmoid blah blah

    NOTE: backprop assumes it is receiving the sigmoid activations from fwd

    """


#==============================================================================
# Matrix manipulation functions
#==============================================================================

class Where(Function):
    pass



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


#==============================================================================
# Globals
#==============================================================================


#==============================================================================
#------------------------------------------------------------------------------
#                               Models
#------------------------------------------------------------------------------
#==============================================================================
class Model:
    pass


class Classifier(Model):
    pass

class IrisClassifier(Classifier):
    pass




#==============================================================================
#------------------------------------------------------------------------------
#                             Loss functions
#------------------------------------------------------------------------------
#==============================================================================
class Loss:
    pass

class ClassificationLoss(Loss):
    pass

'''
- squared error
- sigm cross entropy ?
- softmax cross?
'''

#==============================================================================
# Loss functions
#==============================================================================






#==============================================================================
#------------------------------------------------------------------------------
#                               Optimizers
#------------------------------------------------------------------------------
#==============================================================================

class Optimizer:
    pass

class MomentumOptimizer(Optimizer):
    pass


class StochasticGradientDescent(VanillaGradientDescent):
    pass

'''
- adam
- SGD
 - MomentumSGD
'''

