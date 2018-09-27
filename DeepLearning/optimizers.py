""" This module contains all optimization routines
available to the model.

Currently only SGD and Adam are available.


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
import code
import numpy as np

from utils import TODO, NOTIMPLEMENTED, INSPECT




class OptSGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, P, gP):
        return P - self.lr * gP

    def __call__(self, params):
        """ params = {'var': ndarray, 'grad: ndarray}"""
        updated_params = {}
        for p_key, p_set in params.items():
            var  = p_set['var']
            grad = p_set['grad']

            # update var
            updated_var = self.update(var, grad)
            updated_p_set = {'var': updated_var, 'grad':None}

            # add to updated
            updated_params[p_key] = updated_p_set

        return updated_params

