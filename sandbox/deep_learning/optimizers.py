""" This module contains all optimization routines
available to the model.

Currently only SGD and Adam are available.


# Module components
#------------------
Optimizer : base class for gradient-based optimizers
    Receives a parameter, the gradient wrt to that parameter
    via a stochastic objective function, and updates the
    parameter through some routine

SGD : Optimizer, vanilla stochastic gradient descent algorithm
    Optimizes a parameter based on the gradients of a small
    subset (minibatch) of training data and a learning rate

Adam: Optimizer, Adaptive moment estimation algorithm
    Optimizes a parameter adaptively based on estimates
    from its previous, decaying, average gradients
    squared v, and and average gradients m

"""
import numpy as np

#==============================================================================
# Base Optimizer
#==============================================================================

# Optimizer
# ---------
# inherits :
# derives : SGD, Adam
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
    def __init__(self, lr=0.01, **kwargs):
        self.lr = lr
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        name = self.__class__.__name__
        return name

    def update(self, P, P_grad, P_key):
        """ Update parameter P with it's gradient """
        raise NotImplementedError

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
            updated_params[param_key] = self.update(P, P_grad, param_key)
        return updated_params

#==============================================================================
# Optimizers
#==============================================================================

class SGD(Optimizer):
    """ Vanilla stochastic gradient descent algorithm

    Adjusts parameters based on the difference between the
    variable and it's learning-rate scaled gradient
    """
    def update(self, P, P_grad, *args):
        return P - self.lr * P_grad

#------------------------------------------------------------------------------

class Adam(Optimizer):
    """ Adaptive optimization algorithm for gradient descent

    Adam uses adaptive estimates of lower-order moments,
    mean and variance ('m' and 'v', respectively) to optimize
    an objective

    Attributes
    ----------
    moments : dict(str : dict(str : ndarray))
        Collection of moment vectors keyed to a param in the model.
        For each parameter, there is a dict with two ndarrays,
        'm' and 'v', which are the moments for that param

    t : int
        timestep corresponding to number of updates made
        (ie, number of epochs or iterations completed thus far),
        used adapting stepsize (learning rate) for each update

    Params
    ------
    alpha : float
        stepsize
    beta1 : float
        exponential decay rate for first-order moment ('m')
    beta2 : float
        exponential decay rate for second-order moment ('v')
    eps : float
        arbitrarily small value used to prevent division by zero

    """
    moments = {} # eg, moments['layer2_W1'] = {'m': ndarray, 'v': ndarray}
    t = 0 # timestep

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 moments_init=None):
        """ suggested default values (by authors) """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps

        # restore pretrained moments
        if moments_init is not None:
            self.moments = moments_init

    def init_moments(self, P, P_key):
        """ initialize Adam moment estimates m, v for param P"""
        m = np.zeros_like(P).astype(np.float32)
        v = np.zeros_like(P).astype(np.float32)
        self.update_moments(P_key, m, v)
        return self.moments[P_key]

    def update_moments(self, P_key, m, v):
        """ Update Adam moment estimates m, v """
        updated_moments = {'m': m, 'v': v}
        self.moments[P_key] = updated_moments

    def get_moments(self, P, P_key):
        """ get moment estimates from collection
        If moments for parameter P do not exist, first initialize
        """
        if P_key not in self.moments:
            moments = self.init_moments(P, P_key)
        else:
            moments = self.moments[P_key]
        return moments

    @property
    def step(self):
        """ calculate current stepsize based on bias-corrected
        decay rates and timestep (Performed outside the update
        body for efficiency)
        """
        # Get params
        #-----------
        alpha = self.alpha
        beta1 = self.beta1
        beta2 = self.beta2
        t = self.t

        # bias correction func
        correct = lambda beta: 1 - np.power(beta, t)

        # Bias corrections
        #-----------------
        b1 = correct(beta1)
        b2 = correct(beta2)

        # Step at time t
        #---------------
        step = alpha * np.sqrt(b2) / b1
        return step


    def update(self, P, P_grad, P_key):
        """ Update parameter P with gradient P_grad """

        # update timestep
        self.t += 1

        # Get Adam update params
        #-----------------------
        step  = self.step
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps

        # Get moments
        #------------
        P_moments = self.get_moments(P, P_key)
        m = P_moments['m']
        v = P_moments['v']

        # Update moments
        #---------------
        m = beta1 * m + (1 - beta1) * P_grad
        v = beta2 * v + (1 - beta2) * np.square(P_grad)
        self.update_moments(P_key, m, v)

        # Update param P
        #---------------
        P_update = P - step * m / (np.sqrt(v) + eps)
        return P_update




#==============================================================================
# module utils
#==============================================================================

OPTIMIZERS = {'sgd': SGD, 'adam': Adam}

def get_optimizer(name):
    if name not in OPTIMIZERS:
        raise ValueError('there is no optimizer for {}'.format(name))
    else:
        return OPTIMIZERS[name]

