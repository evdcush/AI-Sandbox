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
import code
import numpy as np
import functions as F
from initializers import HeNormal, Zeros, Ones

#==============================================================================
#------------------------------------------------------------------------------
#                              Layers
#------------------------------------------------------------------------------
#==============================================================================


class Dense:
    """ Fully-connected, feed-forward layer

    Params
    ------


    """
    name = 'dense_layer'
    updates = True
    matmul = F.MatMul()
    W_key  = 'W'
    B_key  = 'B'
    params = {} # nested as {'W_key': {'var':array, 'grad': None}}
    cache  = {}
    def __init__(self, kdims, ID, nobias=False, restore=None,
                 init_W=HeNormal, init_B=Zeros):
        self.ID = ID
        self.kdims  = kdims
        self.nobias = nobias
        self.restor = restore
        self.init_W = init_W()
        self.init_B = init_B()
        self.initialize_params()


    """ Note: why have all these property funcs?
    why not just init each param as an instance var,
    then when it comes time to update, you can just put
    everything into a dict for opt, then unpack the
    return??

    I don't know, I liek the idea of having a native 'params'
    collection as a class variable that is invariant to
    it's class.

    """
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


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @staticmethod
    def init_param(init_fn, kdims):
        var = init_fn(kdims)
        param = {'var': var, 'grad': None}
        return param

    def format_key(self, var_key):
        layer_name = self.name
        ID = self.ID
        params_key = '{}_{}_{}'.format(layer_name, ID, var_key)
        return params_key

    def initialize_params(self):
        """ initializes params as a dictionary mapped to variables
        self.params has form :
            {var_key: {'var': ndarray, 'grad': ndarray OR None}}

        """
        # Restore parameters if need be
        if self.restore is not None:
            self.params = self.restore
            return

        # Initializing weight W
        self.W_key = self.format_key(self.W_key)
        W_param = self.init_param(self.init_W, self.kdims)
        self.params[self.W_key] = W_param

        # Initializing bias B
        if not self.nobias:
            self.bias = F.Bias()
            B_dims = self.kdims[-1:]
            self.B_key = self.format_key(self.B_key)
            B_param = self.init_param(self.init_B, B_dims)
            self.params[self.B_key] = B_param

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




