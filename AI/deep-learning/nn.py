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


class StochasticgradientDescent(VanillagradientDescent):
    pass

'''
- adam
- SGD
 - MomentumSGD
'''

