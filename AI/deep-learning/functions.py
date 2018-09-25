""" All math functions and ops used by a model

This module provides the foundation of functions and operations
to build a network and optimize models.

It contains only the units that would used within a model.
Other functions, for training or data processing, can be found in `utils.py`

Module components
=================
Function : base class for all functions
    MathFunction : base class for math ops
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




#==============================================================================
#------------------------------------------------------------------------------
#                              Functions
#------------------------------------------------------------------------------
#==============================================================================

#------------------------------------------------------------------------------
# Helpers and handy decorators
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
    return inner_preserve



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

    def get_fn_vars(self,reset=True):
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
        print('Mathfunction call func: {}'.format(str(func)))
        if backprop:
            pass
            #import IPython
            #IPython.embed()
            #poo = func(*args, **kwargs)
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        return func(*args, **kwargs)

'''
def backward(self, gY, *args, **kwargs):
        gX = self.function(gY, *args, backprop=True, **kwargs)
        return gX

    def __call__(self, *args, backprop=False, **kwargs):
        func = self.backward if backprop else self.forward
        return func(*args, **kwargs)
'''
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
        dims_X = list(dims_X)
        axes = self.format(axes)
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
#  Exp, Log, Power, Square, Sqrt, Bias, MatMul
#------------------------------------------------------------------------------
class Exp(MathFunction):
    """ Elementwise exponential function """
    def forward(self, X):
        Y = self.fn_vars = np.exp(x)
        return Y

    def backward(self, gY):
        Y = self.get_fn_vars()
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
        X, W = self.get_fn_vars()
        m, k = W.shape

        # get grads
        gX = np.matmul(gY, W.T)

        # need to reshape X.T, gY if ndims > 2, to match W shape
        gW = np.matmul(X.T.reshape(m, -1), gY.reshape(-1, k))
        return gX, gW


# >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><
# TODO

@TODO
class Log(MathFunction):
    """ """
    eps = 1e-4
    def forward(self, X):
        self.fn_vars = X
        Y = np.log(X + self.eps)
        return Y

    def backward(self, gY):
        X = self.get_fn_vars()
        gX = gY / (X + self.eps)
        return gX

@TODO
class Power(MathFunction):
    """ """
    def forward(self, X, p):
        self.fn_vars = X, p
        Y = np.power(X, p)
        return Y

    def backward(self, gY):
        X, p = self.get_fn_vars()
        gX = gY * p * np.power(X, p - 1.0)
        return gX

@TODO
class Square(MathFunction):
    """ squares an array"""
    def forward(self, X):
        self.fn_vars = X
        Y = np.square(X)
        return Y

    def backward(self, gY):
        X = self.get_fn_vars()
        gX = gY * 2.0 * X
        return gX

@TODO
class Sqrt(MathFunction):
    """ """
    def forward(self, X):
        Y = np.sqrt(X)
        self.fn_vars = Y
        return Y

    def backward(self, gY):
        Y = self.get_fn_vars()
        gX = gY / (2 * Y)
        return gX



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
        _,  gB =   self.bias(gY, backprop=True)
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
        Y = self.get_fn_vars()
        gX = np.where(Y, gY, 0)
        return gX


class ELU(MathFunction):
    """ Exponential Linear Unit """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, X):
        x = self.fn_vars = np.copy(X)
        Y = np.where(x < 0, self.alpha*(np.exp(x)-1), x)
        return Y

    def backward(self, gY):
        X = self.get_fn_vars()
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

@TODO
class PReLU(ReLU):
    pass

@TODO
class RRelU(ReLU):
    pass

@TODO
class Swish(MathFunction):
    """ Self-gated activation function
    Can be viewed as a smooth function where the nonlinearity
    interpolates between the linear function (x/2), and the
    ReLU function.

    'The best discovered activation function',
    intended to improve/replace the ReLU masterrace

     - See https://arxiv.org/abs/1710.05941

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
    _sigmoid = lambda x: 1 / (1 + np.exp(-x)) # just use Sigmoid??

    @property
    def sigmoid(self):
        return self._sigmoid

    def forward(self, X, b):
        """ """
        sig_bX = self.sigmoid(b * X)
        Y = X * sig_bX
        self.fn_vars = X, b, sig_bX, Y
        return Y

    def backward(self, gY):
        X, b, sig_bX, Y = self.get_fn_vars()
        bY = b * Y
        gF = bY + sig_bX * (1 - bY) # gradient forward func
        gB = gY * Y * (X - Y) # gradient wrt beta
        gX = gY * gF # gradient wrt X
        return gX, gB



class Sigmoid(MathFunction):
    """ Logistic sigmoid activation """
    def forward(self, X):
        Y = 1 / (1 + np.exp(-X))
        self.fn_vars = Y
        return Y

    def backward(self, gY, *args, **kwargs):
        Y = self.get_fn_vars()
        gX = gY * Y * (1 - Y)
        return gX


class Tanh(MathFunction):
    """ Hyperbolic tangent activation """
    def forward(self, X):
        Y = self.fn_vars = np.tanh(X)
        return Y

    def backward(self, gY):
        Y = self.get_fn_vars()
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
        eX = np.exp(X - X.max(**kw))
        Y = eX / np.sum(eX, **kw)
        return Y

    def backward(self, gY):
        Y = self.get_fn_vars()
        gY *= Y
        Y *= np.sum(gY, **self.kw)
        gX = gY - Y
        return gX


#==============================================================================
# Loss Functions
#==============================================================================

class SoftmaxCrossEntropy(MathFunction):
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
        self.fn_vars = Y, t # preserve vars for backward
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


        gX, t = self.get_fn_vars() # (Y, t)
        N = t.shape[0]
        #gX = Y
        gX[np.arange(N), t] -= 1
        gX = gLoss * (gX / float(N))
        return gX
