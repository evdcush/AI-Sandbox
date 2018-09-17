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




class Block:
    """ Base class for a Block, which wraps various network ops

    Variables
    ---------
    label : str
        name of block
    ID : int

    """
    ID = -1
    label = 'Block'
    update = False

    def __init__(self, label, ID, *args, **kwargs):
        self.ID = ID
        self.label = label

class FunctionBlock(Block):
    """ Wraps Functions
    """
    def __init__(self, func, label, ID, update):
        self.func = func


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
    linear = F.Linear()
    update = True

    def __init__(self, ID, kdim, label='Dense', init_W=None, init_B=None):
        self.ID = ID
        self.kdim = kdim
        self.label = label
        self.W = init_W
        self.B = init_B
        self.W_tag = '{}_{}-{}'.format(label, ID, 'W')
        self.B_tag = '{}_{}-{}'.format(label, ID, 'B')

    def initialize(self, initializer=Intitializers.HeNorm):
        k_in, k_out = self.kdim
        if self.W is None:
            self.W = initializer(k_in, k_out)
        if self.B is None:
            self.B = np.ones(k_out,) * 1e-6 #initializer(k_out,)

    def forward(self, X, evaluation=False):
        Y = self.linear(X, self.W, self.B, evaluation=evaluation)
        return Y

    def backward(self, gY, opt):
        gX, params = self.linear(gY, backprop=True)
        gW, gB = params
        self.update(gW, gB, opt)
        return gX

    def update(self, gW, gB, opt):
        self.W = opt(gW, self.W_tag)
        self.B = opt(gB, self.B_tag)







class Dense:
    """ Fully-connected layer that computes
    an output as the linear transformation
    of its input X against some weights W
    """
    linear = F.Linear()initializer(k_out,)

    def __init__(self, activation=F.sigmoid, init_W=None, init_B=None):
        """ initializes a Dense layer

        Params
        ------
        activation : activation function
            defaults to standard logistic sigmoid, but may be None
            for output layer

        init_W : initial weight matrix W
            if not None, then init_W is a pretrained weight matrix

        init_B : initial bias vector B
            if not None, then it is pretrained
        """
        self.activation = activation()
        self.W = init_W
        self.B = init_B

    def get_params(self):
        return self.W, self.B

    def initialize_params(self, kdims, initializer=Initializers.HeNorm):
        """ initialize layer params if uninitialized

        W : initialized from a random distribution
        B : initialized to be near zero

        Params
        ------
        kdims : tuple (int)
            kdims[0] is the feature input size
            kdims[1] is the output size

        initializer : Initializer
            initializes weights from a random distribution
        """
        K_in, K_out = Kdims
        if self.W is None:
            self.W = initializer(K_in, K_out)
            self.B = np.ones(K_out,).astype(np.float32) * 1e-6

    def forward(self, X):
        Y = self.linear(X, W, B)
        if self.activation:
            Y = self.activation(Y)
        return Y

    def backward(self, gY):
        if self.activation:
            gY = self.activation(gY, backprop=True)
        gX, params = self.linear(gY, backprop=True)
        gW, gB = params
        return gX

    def update(self, opt):



class NeuralNetwork:
    """ Base class for any network

    All networks have the following properties

    Properties
    ----------
    kdims : list (int)
        channel sizes for each layer in network
        longer the list --> deeper (more layers) the network
    Layers : list
        list of layers or modules per layer in network
    Initializer :
    Optimizer :


    All networks have a set of layers, and
    """
    def __init__(self, kdims, layers, initializer=HeNorm, optimizer=Adam):

class FeedForwardNetwork:
    pass
