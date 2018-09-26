"""
The layers that compose the neural network are in this module.

Layers serve as the interface to Functions for the Network.
They manage any parameters used by the functions, and provide high
level abstractions for backprop and updates.

There are (currently) two kinds of layers: parametric and static

# Layer types
#------------
Parametric layers : [Dense, Swish, ]
    these layers use variables that receive updates from an optimizer

Static layers : [Sigmoid, Tanh, Softmax, ReLU, ELU, SELU, ]
    these layers do not utilize variables and do not
    need updates from optimizer

"""
import code
import numpy as np
import functions as F
from initializers import HeNormal, Zeros, Ones

#==============================================================================
#------------------------------------------------------------------------------
#                          Parametric Layers
#------------------------------------------------------------------------------
#==============================================================================
""" These layers all have parameters that are updated through gradient descent
"""

class Dense:
    """ Vanilla fully-connected hidden layer
    The Dense layer is defined by the linear transformation function:

         f(X) = X.W + b
    Where
    : W is a weight matrix
    : b is a bias vector,
    and both are learnable parameters optimized through gradient descent.

    Note: the bias parameter is OPTIONAL in this implementation.
          (The weights matrix, of course, is always used)

    Params
    ------
    name : str
        layer name
    updates : bool
        whether the layer has learnable parameters
    matmul : Function
        matrix multiplication function (the X.W part in the example above)
    W_key : str
        name of the weight matrix parameter
    B_key : str
        name of the bias parameter
    params : dict
        the collection containing the layer's parameters, keyed by the
        "*_key" attributes
        params is structured in the following manner:
            {'W_key' : {'var': ndarray, 'grad': ndarray|None},
             'B_key' ...}
    """
    name = 'dense_layer'
    updates = True
    matmul = F.MatMul()
    W_key  = 'W'
    B_key  = 'B'
    params = {} # nested as {'W_key': {'var':array, 'grad': None}}
    def __init__(self, kdims, ID, init_W=HeNormal, init_B=Zeros,
                 nobias=False, restore=None):
        self.ID = ID            # int : Layer's position in the parent network
        self.kdims = kdims      # tuple(int) : channel-sizes
        self.init_W  = init_W() # Initializer : for weights
        self.init_B  = init_B() # Initializer : for bias
        self.nobias  = nobias   # bool : whether Dense instance uses a bias

        # Format param keys
        self.format_keys()

        # Initialize params
        if restore is not None:
            self.restore_params(restore)
        else:
            self.initialize_params()

    # Parameter access through properties
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ===== W
    @property
    def W(self):
        return self.params[self.W_key]['var']

    @W.setter
    def W(self, var):
        self.params.[self.W_key]['var'] = var

    @property
    def gW(self):
        return self.params[self.W_key]['grad']

    @gW.setter
    def gW(self, grad):
        .params.[self.W_key]['grad'] = grad

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ===== B
    @property
    def B(self):
        return self.params[self.B_key]['var']

    @B.setter
    def B(self, var):
        self.params.[self.B_key]['var'] = var

    @property
    def gB(self):
        return self.params[self.B_key]['grad']

    @gB.setter
    def gB(self, grad):
        self.params.[self.B_key]['grad'] = grad


    # Layer initialization
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def format_keys(self,):
        # format key for params; eg W_key : 'dense_layer2_W'
        ID = self.ID
        layer_name = self.name
        params_key = '{}{}_{{}}'.format(layer_name, ID)
        self.W_key = params_key.format(self.W_key)
        if not self.nobias:
            self.B_key = params_key.format(self.B_key)

    def restore_params(self, pmap):
        """ init layer params with pretrained parameters
        pmap is a dict with the same structure and keys as self.params
        """
        #==== Integrity check: keys ----> are they for the same layer?
        assert self.params.keys() == pmap.keys()

        #==== Integrity check: channels ----> do the weights have same dims?
        assert self.kdims == pmap[self.W_key]['var'].shape

        #==== Keys match, channels match, safe to restore
        self.params = pmap

        # init bias func if used
        if not self.nobias:
            self.bias = F.Bias()

    @staticmethod
    def init_param(init_fn, kdims):
        # initializes param from Initializer, into params sub-dict form
        var = init_fn(kdims)
        param = {'var': var, 'grad': None}
        return param

    def initialize_params(self, restore=restore):
        """ initializes params as a dictionary mapped to variables
        self.params has form :
            {var_key: {'var': ndarray, 'grad': ndarray OR None}}

        """
        # Initializing weight W
        W_param = self.init_param(self.init_W, self.kdims)
        self.params[self.W_key] = W_param

        # Initializing bias B
        if not self.nobias:
            self.bias = F.Bias()
            B_dims = self.kdims[-1:]
            B_param = self.init_param(self.init_B, B_dims)
            self.params[self.B_key] = B_param

    # Layer computation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def update(self, opt):
        assert all([v['grad'] is not None for v in self.params.values()])
        self.params = opt(self.params)

    def forward(self, X):
        W = self.W
        Y = self.matmul(X, W)
        if not self.nobias:
            Y += self.bias(Y, self.B)
        return Y

    def backward(self, gY):
        gX, gW = self.matmul(np.copy(gY), backprop=True)
        self.gW = gW
        if not self.nobias:
            _,  gB = self.bias(np.copy(gY), backprop=True)
            self.gB = gB
        return gX

#==============================================================================



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
        label = '{}_{}'.format(caller_label, self.network_label)
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
            gX = layer(gX, opt, backprop=True)
        return

    def __call__(self, *args, backprop=False, **kwargs):
        func = self.backward if backprop else self.forward
        return func(*args, **kwargs)




