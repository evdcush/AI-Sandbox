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
from utils import HeNormal, Zeros
from nn import SGD, Adam


#==============================================================================
#------------------------------------------------------------------------------
#                              Blocks
#------------------------------------------------------------------------------
#==============================================================================
""" Blocks are the interface to Functions.

They maintain any parameters used by Functions, and provide easy access to
higher levels for updating and serializing those parameters.

Blocks allow the network to have greater modularity when designing
layers, as gradient chaining isn't coupled to a layer, instead being
self-contained within blocks and can be treated as black-boxes
"""


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
    params = {}
    update = True

    def __init__(self, layer_label, ID, kdim, init_W=HeNormal, init_B=Zeros):
        # super inits: self.ID, self.label, self.kdim
        super().__init__(layer_label, ID, kdim)
        self.format_params_labels()
        self.initialize_params(init_W, init_B)

    def format_params_labels(self):
        """ Creates final, unique str label used for identifying param

        These labels are used by the optimizer and serializer for keeping
        track of the unique parameter values keyed to this label.

        Example
        -------
        layer_label : str
            This label essentially represents a path from the top level
            callers in training, to the current block and parameter:
            '<model-label>_<network-label>_<layer-label>_<block-label>'

        This function completes the full label by concatenating the
        parameter tags, eg a full example:
        W : 'experimental_feed-forward-net_DenseLayer2_DenseBlock-1_W'
        B : 'experimental_feed-forward-net_DenseLayer2_DenseBlock-1_B'

        """
        fields = '{}_{}'
        self.W_key = fields.format('W')
        self.B_key = fields.format('B')

    @property
    def W(self):
        return self.params[self.W_key]

    @property
    def B(self):
        return self.params[self.B_key]

    @W.setter
    def W(self, val):
        self.params[self.W_key] = val

    @B.setter
    def B(self, val):
        self.params[self.B_key] = val


    def initialize_params(self, init_W, init_B):
        if init_W.__name__ == 'dict':
            # Then params are being restored, rather than init
            self.params = init_W
        else:
            # both are array creation routines
            self.W = init_W(self.kdim)
            self.B = init_B(self.kdim) + 1e-6 # near zero


    def forward(self, X):
        Y = self.function(X, self.W, self.B)
        return Y

    def backward(self, gY, opt):
        gX, params = self.function(gY, backprop=True)
        gW, gB = params
        self.update(gW, gB, opt)
        return gX

    def update(self, gW, gB, opt):
        grads = {self.W_key: gW, self.B_key: gB}
        self.params = opt(self.params, grads)

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

BLOCKS = {**OPS, **ACTIVATIONS}

#==============================================================================
# Base Layer class
#==============================================================================

# Layer
# -----
# inherits :
# derives : DenseLayer
class Layer:
    """ Base layer class composed of blocks

    Unlike most other 'base' classes in the module, which are mostly abstract,
    Layer can be used as a concretized instance if
    sufficiently specified

    Params
    ------
    layer_label : str
        The label specifies what position it is within the network,
        wrt to other layers, and is used by its constituent blocks
        to get a unique parameter key for all learnable parameters,
        which is then used for the optimizer and serialization

    """
    layer_label = 'L{}' # must be formatted!

    def __init__(self, ID, kdim, blocks=('dense', 'sigmoid'), *args, **kwargs):
        """ Layer initializes itself, as well as its blocks

        Params
        ------
        ID : int
            position number in network (eg, input layer would be 0)

        kdim : tuple (int)
            (k_in, k_out) input and output channels

        blocks : tuple (str)
            Arbitrarily long tuple of block "keys" for this layer.
                Each key is the slugified (all lower) and truncated name
                of a block, eg "sigmoid" keys to value SigmoidBlock
            ORDER matters in tuple: blocks[0] processes before blocks[1]

        """
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
        ID = self.ID
        label = self.layer_label.format(ID)
        return label

    def initialize_blocks(self, op, act, )

    def forward(self, X, *args, **kwargs):
        pass

    def backward(self, gY, opt, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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




