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

#class ParametricLayer:
#    """ Abstracts some of the boilerplate from most parametric layers """
#    __slots__ = ('name', 'func', 'updates', 'params', key)


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
    matmul : Function
        matrix multiplication function (the X.W part in the example above)
    updates : bool
        whether the layer has learnable parameters
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
    matmul = F.MatMul()
    updates = True
    W_key  = 'W'
    B_key  = 'B'
    params = {} # nested as {'W_key': {'var':array, 'grad': None}}
    def __init__(self, ID, kdims, init_W=HeNormal, init_B=Zeros,
                 nobias=False, restore=None):
        self.ID = ID            # int : Layer's position in the parent network
        self.kdims = kdims      # tuple(int) : channel-sizes
        self.init_W = init_W() # Initializer : for weights
        self.init_B = init_B() # Initializer : for bias
        self.nobias = nobias   # bool : whether Dense instance uses a bias

        # Format name and param keys
        self.name = '{}{}'.format(self.name, self.ID)
        self.format_keys()

        # Initialize params
        if restore is not None:
            self.restore_params(restore)
        else:
            self.initialize_params()

    def __repr__(self):
        # eval repr format:
        rep = "{}('{}, {}, init_W={}, init_B={}, nobias={}')"

        # Instance vars
        cls_name = self.__class__.__name__
        init_W_name = self.init_W.__class__.__name__
        init_W_name = self.init_B.__class__.__name__
        ID = self.ID
        kdims = self.kdims

        # Format eval repr and ret
        rep_args = (cls_name, ID, kdims, init_W_name, init_B_name, self.nobias)
        return rep.format(*rep_args)

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
        self.params.[self.W_key]['grad'] = grad

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
        form_key = lambda key: '{}_{}'.format(self.name, key)
        self.W_key = form_key(self.W_key)
        if not self.nobias:
            self.B_key = form_key(self.B_key)

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

    def initialize_params(self):
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

class SwishActivation:
    """ Self-gated activation function

    Through the beta variable, swish's nonlinearity properties interpolate
    between a linear function and ReLU

    Params
    ------
    name : str
        layer name
    swish : Function
        swish activation function
    updates : bool
        whether the layer has learnable parameters
    B_key : str
        name of the beta variable
    params : dict
        collection of the layer's parameters (just beta)
    """
    name = 'swish_layer'
    swish = F.Swish()
    updates = True
    B_key  = 'beta'
    params = {} # nested as {'B_key': {'var':array, 'grad': None}}
    def __init__(self, ID, kdims, init_B=Ones, restore=None):
        self.ID = ID            # int : Layer's position in the parent network
        self.kdims = kdims[-1:] # tuple(int) : 1D shape of beta
        self.init_B = init_B()  # Initializer : for beta

        # Format name and param keys
        self.name = '{}{}'.format(self.name, self.ID)
        self.format_keys()

        # Initialize params
        if restore is not None:
            self.restore_params(restore)
        else:
            self.initialize_params()

    # Parameter access through properties
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ===== Beta
    @property
    def Beta(self):
        return self.params[self.B_key]['var']

    @Beta.setter
    def Beta(self, var):
        self.params.[self.B_key]['var'] = var

    @property
    def gBeta(self):
        return self.params[self.B_key]['grad']

    @gBeta.setter
    def gBeta(self, grad):
        self.params.[self.B_key]['grad'] = grad


    # Layer initialization
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def format_keys(self,):
        # format key for params; eg B_key : 'dense_layer1_beta'
        self.B_key = '{}_{}'.format(self.name, self.B_key)

    def restore_params(self, pmap):
        """ init layer params with pretrained parameters
        pmap is a dict with the same structure and keys as self.params
        """
        #==== Integrity check: keys ----> are they for the same layer?
        assert self.params.keys() == pmap.keys()

        #==== Integrity check: channels ----> do the weights have same dims?
        assert self.kdims == pmap[self.B_key]['var'].shape

        #==== Keys match, channels match, safe to restore
        self.params = pmap

    @staticmethod
    def init_param(init_fn, kdims):
        # initializes param from Initializer, into params sub-dict form
        var = init_fn(kdims)
        param = {'var': var, 'grad': None}
        return param

    def initialize_params(self):
        """ initializes params as a dictionary mapped to variables
        self.params has form :
            {var_key: {'var': ndarray, 'grad': ndarray OR None}}
        """
        # Initialize Beta
        B_param = self.init_param(self.init_B, self.kdims)
        self.params[self.B_key] = B_param


    # Layer computation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def update(self, opt):
        assert all([v['grad'] is not None for v in self.params.values()])
        self.params = opt(self.params) # clears grads after update

    def forward(self, X):
        Y = self.swish(X, self.B)
        return Y

    def backward(self, gY):
        gX, gBeta = self.swish(gY, backprop=True)
        self.gBeta = gBeta
        return gX


#==============================================================================
#------------------------------------------------------------------------------
#                             Static Layers
#------------------------------------------------------------------------------
#==============================================================================
""" Layers without updating variables, mostly activations """

class StaticLayer:
    """ Parent class covering most ops performed by static layers """
    updates = False
    __slots__ = ('name', 'func', 'ID')
    def __init__(self, ID, *args, **kwargs):
        self.ID = ID
        self.name = self.name + str(ID)

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.ID)

    def forward(self, X):
        return self.func(X)

    def backward(self, gY):
        return self.func(gY, backprop=True)


# Activations
#------------------------------------------------------------------------------
class SigmoidActivation(StaticLayer):
    __slots__ = ()
    name = 'sigmoid_layer'
    func = F.Sigmoid()

class SoftmaxActivation(StaticLayer):
    __slots__ = ()
    name = 'softmax_layer'
    func = F.Softmax()

class TanhActivation(StaticLayer):
    __slots__ = ()
    name = 'tanh_layer'
    func = F.Tanh()

class ReLUActivation(StaticLayer):
    __slots__ = ()
    name = 'relu_layer'
    func = F.ReLU()

class ELUActivation(StaticLayer):
    __slots__ = ()
    name = 'elu_layer'
    func = F.ELU()

class SeLUActivation(StaticLayer):
    __slots__ = ()
    name = 'selu_layer'
    func = F.SeLU()


#==============================================================================
#==============================================================================

# Available Layers
#-----------------
OPS = {'dense_layer': Dense,}

ACTIVATIONS = {'sigmoid_layer' : SigmoidActivation,
               'softmax_layer' : SoftmaxActivation,
                  'tanh_layer' : TanhActivation,
                  'relu_layer' : ReLUActivation,
                   'elu_layer' : ELUActivation,
                  'selu_layer' : SeLUActivation,
                  'swish_layer': SwishActivation,
                  }

LAYERS = {**OPS, **ACTIVATIONS}

def get_all_layers():
    return LAYERS
