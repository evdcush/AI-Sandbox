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




#==============================================================================
# Base Optimizers
#==============================================================================

# Optimizer
# ---------
# inherits :
# derives : SGD, AdaptiveOptimizer
class Optimizer:
    """ Base optimizer
    Uses gradient descent to move towards (local) minimas in the
    space of an objective function

    Params
    ------
    lr : float
        learning-rate for optimizer, typically a small
        float value within [0.1, 1e-4], serves as the
        "step-size" during gradient descent in terms of
        how far we want to move away from the
        gradient. The greater the value, the greater
        the updates to a param.

    """
    def __init__(self, lr=0.01 **kwargs):
        self.lr = lr
        for key, val in kwargs.items():
            setattr(self, key, val)

    @NOTIMPLEMENTED
    def update(self, P, P_grad, P_key):
        """ Update parameter P with it's gradient """
        pass

    def __call__(self, params):
        """ Adjust parameters based on gradients from an objective

        Params
        ------
        params : dict(str: tuple(ndarray, ndarray))
            Mapping of parameter keys to their respective parameters
            and gradients
            eg, if the 2nd dense layer was updating, it might
                call optimizer with the following params:
            {'Dense2_W': (W, W_grad), 'Dense2_B': (B, B_grad)}

        Returns
        -------
        updated_params : dict(str: ndarray)
            The updated parameter(s).
            Note: gradients are not returned.
        """
        updated_params = {}
        for param_key, param_vars in params.items():
            # Get parameter variables from tuple pair
            P, P_grad = param_vars

            # Update parameters
            updated_params[p_key] = self.update(P, P_grad, param_key)
        return updated_params

#------------------------------------------------------------------------------

class SGD(Optimizer):
    """ Vanilla stochastic gradient descent algorithm

    Adjusts parameters based on the difference between the
    variable and it's learning-rate scaled gradient
    """
    def update(self, P, P_grad, *args):
        return P - self.lr * P_grad




