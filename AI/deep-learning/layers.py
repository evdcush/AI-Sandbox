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

Blocks are the interface to functions, maintaining all parameters used
 by a single Function instance, while Layers provide higher level access
 to the set of all Block parameters for updating.

"""

import functions as F
from initializers import HeNormal, Zeros, Ones
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

    def __init__(self, ID, layer_label, *args, **kwargs):
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

    def __init__(self, kdim, ID, layer_label, *args, **kwargs):
        super().__init__(ID, layer_label)
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
    updates = True
    def __init__(self, kdim, ID, layer_label, init_W=HeNormal, init_B=Zeros):
        # super inits: self.ID, self.label, self.kdim
        super().__init__(kdim, ID, layer_label)
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
        self.W_key = fields.format(self.label, 'W')
        self.B_key = fields.format(self.label, 'B')

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
            self.B = init_B(self.kdim) + 1e-7 # near zero

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
#  SigmoidBlock, SoftmaxBlock, TanhBlock, ReluBlock, ELUBlock, SeluBlock
#------------------------------------------------------------------------------

class SigmoidBlock(FunctionBlock):
    """ Sigmoid activation """
    block_label = 'SigmoidBlock'
    function = F.Sigmoid()

class Softmax(FunctionBlock):
    """ Softmax activation """
    block_label = 'SoftmaxBlock'
    function = F.Softmax()

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

class SwishBlock(FunctionBlock):
    """ Swish activation """
    block_label = 'SwishBlock'
    function = F.Swish()
    params = {}
    updates = True
    def __init__(self, *args, **kwargs):
        # super inits: self.ID, self.label, self.kdim
        super().__init__(*args, **kwargs)
        self.B_key ='{}_{}'.format(self.label, 'B')
        self.B = Ones(self.kdim[-1:])

    @property
    def B(self):
        return self.params[self.B_key]

    @B.setter
    def B(self, val):
        self.params[self.B_key] = val

    def forward(self, X):
        Y = self.function(X, self.B)
        return Y

    def backward(self, gY, opt):
        gX, gB = self.function(gY, backprop=True)
        self.update(gB, opt)
        return gX

    def update(self, gB, opt):
        grads = {self.B_key: gB}
        self.params = opt(self.params, grads)



#==============================================================================
#------------------------------------------------------------------------------
#                              Layers
#------------------------------------------------------------------------------
#==============================================================================

# Available Blocks
# ================
OPS = {'dense': DenseBlock,}

ACTIVATIONS = {'sigmoid' : SigmoidBlock,
                  'tanh' : TanhBlock,
                  'relu' : ReluBlock,
                   'elu' : ELUBlock,
                  'selu' : SeluBlock,
                  'swish': SwishBlock,
                  }

BLOCKS = {**OPS, **ACTIVATIONS}

def get_all_blocks():
    return BLOCKS

#==============================================================================
# Base Layer class
#==============================================================================

# Layer
# -----
# inherits :
# derives : FullyConnectedLayer
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

    def __init__(self, kdim, ID, network_label, *args,
                 blocks=('dense', 'sigmoid'), initializer=None,
                 no_act=False, **kwargs):
        """ Layer initializes itself, as well as its blocks

        Params
        ------
        kdim : tuple (int)
            (k_in, k_out) input and output channels

        ID : int
            position number in network (eg, input layer would be 0)

        network_label : str
            caller label, to be concatenated to with layer_label

        blocks : collection (str)
            Arbitrarily long collection of block "keys" for this layer.
                Each key is the slugified (all lower, dashes for spaces)
                and truncated name of a block, eg "sigmoid" keys
                to value SigmoidBlock
            ORDER matters in collection: blocks[0] processes before blocks[1]

        """
        self.ID = ID
        self.kdim = kdim
        self.label  = self.format_label(network_label)
        self.no_activation = no_act
        self.blocks = self.initialize_blocks(blocks, initializer)

        for attribute, value in kwargs.items():
            setattr(self, attribute, value)


    def format_label(self, caller_label):
        fields = '{}_{}'
        layer_ID = self.layer_label.format(self.ID)
        label = fields.format(caller_label, layer_ID)
        return label

    def initialize_blocks(self, block_keys, initializer):
        blocks = {} # {int : Block}

        # Check whether block_keys valid
          # all blocks available within module
        all_blocks = OPS if self.no_activation else get_all_blocks()
        assert all([key in all_blocks for key in block_keys])

        # initialize each block
        for ID, key in block_keys:
            block = all_blocks[key]
            args_block = (self.label, ID, self.kdim)
            if initializer is not None:
                args_block += (initializer,)
            blocks[ID] = block(*args_block)
        return blocks


    def forward(self, X, *args, **kwargs):
        blocks = self.blocks
        Y = np.copy(X)
        for ID in range(len(blocks)):
            block = blocks[ID]
            Y = block(Y)
        return Y

    def backward(self, gY, opt, *args, **kwargs):
        blocks = self.blocks
        gX = np.copy(gY)
        for ID in reversed(range(len(blocks))):
            block = blocks[ID]
            gX = block(gX, opt, *args,  backprop=True, **kwargs)
        return gX

    def __call__(self, *args, backprop=False, **kwargs):
        func = self.backward if backprop else self.forward
        return func(*args, **kwargs)


#==============================================================================
# Derived Layers
#  FullyConnectedLayer
#==============================================================================

class FullyConnectedLayer(Layer):
    """ Simple feed-forward layer, where every feature is connected
    to a neuron
    """
    layer_label = 'FC-Layer{}'
    def __init__(self, *args, blocks=['dense', 'sigmoid'],
                 no_act=False, **kwargs):
        super().__init__(*args, blocks=blocks, no_act=no_act, **kwargs)


#==============================================================================
#------------------------------------------------------------------------------
#                              Network
#------------------------------------------------------------------------------
#==============================================================================
""" Neural networks are composed of Layers

For our usage, we can think of the network both as an algorithm
and as a sort of ordered "data-structure," like a list or array,
and Layers being the elements of that list.

# Forward
  - To generate a prediction, we iterate through the "list" in order,
    propagating an initial input through each element.

# Backward
  - To optimize the network's predictions, we iterate through the list
    in reverse order, propagating the gradient of an objective
    function through each element.

"""

#==============================================================================
# NeuralNetwork class
#==============================================================================
""" Since most architectural features and functionalities have been
implemented at the callee levels below NeuralNetwork, we only really need
one network class.

"""
class NeuralNetwork:
    """ Base Neural Network compsed of Layers

    Control Flow
    ------------
    1 - Network instance initialized with a list of channels
        and Layer-types


    2 - Network receives external data input X, and propagates
        it through it's Layers

    3 - A final output Layer returns a prediction

      - (network prediction accuracy/error evaluated by an objective)

    4 - If training: the network receives the gradient of a loss
        function and backpropagates the gradient through it's
        Layers (propagates in reverse order of Layers)

    """
    network_label = 'NN'
    def __init__(self, channels, model_label, *args,
                 layer_blocks=['dense', 'sigmoid'], output_activation=False,
                 initializer=None, **kwargs):
        # Network config
        self.initializer = initializer
        self.label = self.format_label(model_label)

        # Dimensionality of network
        self.kdims = list(zip(channels, channels[1:]))
        self.num_layers = len(self.kdims)

        # Network layers
        #self.layer_type = layer
        self.layer_blocks = layer_blocks
        self.output_activation = output_activation
        self.layers = self.initialize_layers()


    def format_label(self, caller_label):
        label = '{}_{}'.format(model_label, self.network_label)
        return label

    def initialize_layers(self):
        """
        Layer init args:
        - kdim, ID, network_label, *args, blocks=('dense', 'sigmoid'),
          initializer=None, no_act=False, **kwargs
        """
        # Layer
        layer_blocks = self.layer_blocks
        layers = []

        # Layer init args
        label = self.label
        L_init = self.initializer
        act = lambda i: self.output_activation and i == self.num_layers - 1

        # Initialize all layers
        for ID, kdim in enumerate(self.kdims):
            layer = Layer(kdim, ID, label, blocks=layer_blocks, no_act=act(ID),
                         initializer=L_init)
            layers.append(layer)

        return layers


    def forward(self, X):
        """ Propagates input X through network layers """
        Y = np.copy(X) # more for convenience than mutation safety
        for layer in self.layers:
            Y = layer(Y)
        return Y

    def backward(self, gY, opt):
        """ Backpropagation through layers

        Params
        ------
        gY : ndarray
            gradient of loss function wrt to network output Y (or Y_hat)

        opt : Optimizer
            optimizer for updating parameters in layers

        Returns
        -------
        None : gX is gradient of loss wrt to input, which isn't "updated"
          - All optimizable params should be in Network, nowhere else
        """
        gX = np.copy(gY)
        for layer in reversed(self.layers):
            gX = layer(gX, opt)
        return

    def __call__(self, *args, backprop=False, **kwargs):
        func = self.backward if backprop else self.forward
        return func(*args, **kwargs)




