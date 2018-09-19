""" This module contains all optimization routines
available to the model.

Currently only SGD and Adam are available.

As such, there is no need for a "base" optimizer class,
 instead just having two separate classes, SGD and Adam,
 since they share very little, but the idea is that
 optimizers share in the same functionality, as they all
 perform optimization using gradient descent, and that
 when more optimizers are implemented, more base optimizer
 classes will be created, and the module will have more
 robust cohesion


# Module components
#------------------
Optimizer : base class for gradient-based optimizers
    Receives a parameter, the gradient wrt to that parameter
    via a stochastic objective function, and updates the
    parameter through some routine

AdaptiveOptimizer : base class for momentum and accelerated optimizers
    Typically has a weight vector for each learnable/updatable parameter
    based on that parameter's gradient history

SGD : Optimizer, vanilla stochastic gradient descent algorithm
    Optimizes a parameter based on the gradients of a small
    subset (minibatch) of training data and a learning rate

Adam: Optimizer, Adaptive moment estimation algorithm
    Optimizes a parameter adaptively based on estimates
    from its previous, decaying, average gradients
    squared v, and and average gradients m

"""

import numpy as np

from utils import TODO, NOTIMPLEMENTED, INSPECT

""" module imports
utils :
    `TODO` : decorator
        serves as comment and call safety

    `NOTIMPLEMENTED` : decorator
        raises NotImplementedErrorforces if class func has not been overridden

    `INSPECT` : decorator
        interrupts computation and enters interactive shell,
        where the user can evaluate the input and output to func
"""


class Optimizer:
    """ Base optimizer

    Params
    ------
    lr : float
        learning-rate for optimizer, typically a small
        float value within [0.1, 1e-4], serves as the
        "step-size" during gradient descent in terms of
        how far we want to move towards the opposite of
        gradient. The greater the value, the greater
        the updates to a param.
    """
    def __init__(self, lr, *args, **kwargs):
        self.lr = lr

    def __call__(self, P, gP):
        """ Adjust parameter based on gradient from objective

        Params
        ------
        P : ndarray
            parameter to receive update
        gP : ndarray
            gradient of objective/loss function wrt the
            parameter P.
        """
        updated_P = P - self.lr * gP
        return updated_P


# call example
def update(self, gW, gB, opt):
        grads = {self.W_key: gW, self.B_key: gB}
        self.params = opt(self.params, grads)

class AdaptiveOptimizer(Optimizer):
    """ Base adaptive optimizer

    Parameters updated by an adaptive optimizer are
    also optimized by some parameter-specific variable,
    often a "momentum" vector, based on that parameter's
    past gradient values

    Params
    ------
    moments : dict (AttrDict (ndarray))
        moments is collection of moment vectors keyed
        to a param in the model

    momentum : float
        the decay rate on past gradient values

    moments_init : dtype(moments)
        the pretrained or saved moments from another model
        to be restored
    """
    moments = {}

    def __init__(self, lr, *args, momentum=0.9, moments_init=None, **kwargs):
        self.lr = lr
        self.momentum = momentum
        if moments_init is not None:
            self.restore_moments(moments_init)


    def restore_moments(self, moments):
        self.moments = moments # UNTESTED

    def initialize_moment(self, P):

    def moments(self, ):


    def __call__(self, P, gP):



class SGD(Optimizer):
    """ Vanilla stochastic gradient descent

    Already fully implemented by base class
    """
    pass

class Adam(AdaptiveOptimizer):
    """
    """
