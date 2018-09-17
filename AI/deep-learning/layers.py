"""
This module contains the main architectural units of the network:
Blocks, and Layers

Any given model has the following hierarchy:

Model
    Network
        Layer 1
            Block 1
                Dense : functions.Linear
            Block 2
                Sigmoid : functions.Sigmoid
            ...
        Layer 2
            Block 1
            Block 2
            ...
        Layer 3
        ...
        Layer N

The reasoning for having two levels of abstraction (Blocks and Layers)
 between Network and Functions is that it makes the network more
 extensible.

With only one level of abstraction between Functions and Network, the user
 would have to consider more edge cases and have to adjust either functions
 or network operations on a per-task or per-architecture basis.

If we only had "Layer" between Network and Function, then we have to
 consider Layers that have activations vs. those that do not,
 whether a Layer has updateable params, or whether there is a pooling or
 normalization op, and how the optimizer would pass through all of those.

Additionally, instead of a clean, readable network architecture you might
expect for a simple feedforward, eg:
MLP:
    - Hidden1
    - Hidden2
    - Output

You would have
MLP:
    - Dense1
    - Activation1
    - Normalization1
    - Dense2
    - Activation2
    - Normalization2
    - Dense3

"""

import functions as F
from utils import Initializers
from nn import SGD, Adam



#==============================================================================
#------------------------------------------------------------------------------
#                              Blocks
#------------------------------------------------------------------------------
#==============================================================================

#==============================================================================
# Base Block classes:
#  Block, FunctionBlock
#==============================================================================


# Block
# -----
# inherits :
# derives  : FunctionBlock
class Block:
    """ Base class for a Block, which wraps various network ops
    (Currently, only FunctionBlocks are used, but this class
    is kept as base to allow support for different types of blocks)

    Variables
    ---------
    block_label : str
        name of block
    label_format : str
        labeling format for all blocks
    updates : bool
        whether the block has updateable parameters

    """
    block_label = 'Block'
    label_format = '{}_{}-{}' # 'Layer_Block-ID'
    updates = False

    def __init__(self, layer_label, ID, *args, **kwargs):
        self.ID = ID
        self.label = self.format_label(layer_label)
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    def format_label(self, layer_label):
        label_format = self.label_format
        block_label = self.block_label
        ID = self.ID
        label = label_format.format(layer_label, block_label, ID)
        return label

#------------------------------------------------------------------------------

# FunctionBlock
# -------------
# inherits : Block
# derives  : DenseBlock, activation blocks
class FunctionBlock(Block):
    """ Wraps Functions """
    block_label = 'FunctionBlock'
    function = None

    def __init__(self, layer_label, ID, kdim, *args, **kwargs):
        super().__init__(layer_label, ID)
        self.kdim = kdim

    def forward(self, X, *args, **kwargs):
        Y = self.function(X, *args, **kwargs)
        return Y

    def backward(self, gY, *args, **kwargs):
        gX = self.function(gY, *args, backprop=True, **kwargs)
        return gX

    def __call__(self, *args, backprop=False, **kwargs):
        func = self.backward if backprop else self.forward
        return func(*args, **kwargs)

#==============================================================================
# Derived Blocks
#==============================================================================
#------------------------------------------------------------------------------
# Updateable Blocks :
#  DenseBlock
#------------------------------------------------------------------------------
class DenseBlock(FunctionBlock):
    """ Fully-connected block that performs a
    linear transformation on an input X against its
    parameters

    Parameters
    ----------
    W : ndarray
        weight matrix. First dim matches the last in X
    B : ndarray, 1D
        bias vector, with size matching output dims
    """
    block_label = 'DenseBlock'
    function = F.Linear()
    update = True

    def __init__(self, layer_label, ID, kdim, init_W=None, init_B=None):
        super().__init__(layer_label, ID, kdim)
        self.W = init_W
        self.B = init_B
        self.W_key = '{}_{}'.format(self.label, 'W')
        self.B_key = '{}_{}'.format(self.label, 'B')


    def initialize(self, initializer=Intitializers.HeNorm):
        k_in, k_out = self.kdim
        if self.W is None:
            self.W = initializer(k_in, k_out)
        if self.B is None:
            self.B = np.ones(k_out,) * 1e-6 #initializer(k_out,)

    def forward(self, X):
        Y = self.function(X, self.W, self.B)
        return Y

    def backward(self, gY, opt):
        gX, params = self.function(gY, backprop=True)
        gW, gB = params
        self.update(gW, gB, opt)
        return gX

    def update(self, gW, gB, opt):
        W, B = self.W, self.B
        self.W = opt(W, gW, self.W_tag)
        self.B = opt(B, gB, self.B_tag)

    def __call__(self, *args, backprop=False, **kwargs):
        func = self.backward if backprop else self.forward
        return func(*args, **kwargs)

#------------------------------------------------------------------------------
# Activation Blocks :
#  SigmoidBlock, TanhBlock, ReluBlock, ELUBlock, SeluBlock
#------------------------------------------------------------------------------

class SigmoidBlock(FunctionBlock):
    """ Sigmoid activation """
    block_label = 'SigmoidBlock'
    function = F.Sigmoid()

class TanhBlock(FunctionBlock):
    """ Tanh activation """
    block_label = 'TanhBlock'
    function = F.Tanh()


class ReluBlock(FunctionBlock):
    """ ReLU activation """
    block_label = 'ReluBlock'
    function = F.ReLU()

class ELUBlock(FunctionBlock):
    """ ELU activation """
    block_label = 'ELUBlock'
    function = F.ELU()

class SeluBlock(FunctionBlock):
    """ SeLU activation """
    block_label = 'SeluBlock'
    function = F.SeLU()



#==============================================================================
#------------------------------------------------------------------------------
#                              Layers
#------------------------------------------------------------------------------
#==============================================================================

# Available Blocks
# ================
OPS = {'dense': DenseBlock
           '' : None}
ACTIVATIONS = {'sigmoid' : SigmoidBlock,
                  'tanh' : TanhBlock,
                  'relu' : ReluBlock,
                   'elu' : ELUBlock,
                  'selu' : SeluBlock
                      '' : None}

class Layer:
    layer_label = 'L'
    label_format = layer_label + '{}'

    def __init__(self, ID, kdim, op='dense', act='sigmoid', *args, **kwargs):
        self.ID = ID
        self.kdim =  kdim
        self.label = self.format_label(layer_label)
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

        # Initialize blocks
        self.op = OPS[op](self.label, 1, kdim, **kwargs)
        if act != '':
            self.activation = ACTIVATIONS[act](self.label, 2, kdim, **kwargs)

    def format_label(self, layer_label):
        label_format = self.label_format
        ID = self.ID
        label = label_format.format(ID)
        return label

    def forward(self, X, *args, **kwargs):
        pass

    def backward(self, gY, opt, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

#==============================================================================
#------------------------------------------------------------------------------
#                              Network
#------------------------------------------------------------------------------
#==============================================================================

class NeuralNet:
    pass

class FeedForwardNet:
    pass




