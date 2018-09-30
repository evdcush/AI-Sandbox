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
import functions
from initializers import HeNormal, Zeros, Ones

#==============================================================================
#------------------------------------------------------------------------------
#                          Parametric Layers
#------------------------------------------------------------------------------
#==============================================================================
""" These layers all have parameters that are updated through gradient descent
"""

class ParametricLayer:
    """ Layers with learnable variables updated through gradient descent """
    updates = True
    def __init__(self, ID, kdims, **kwargs):
        self.name = '{}{}'.format(self.__class__.__name__, ID)
        self.ID = ID
        self.kdims = kdims

    def __str__(self,):
        # str : Dense$ID
        #     eg 'Dense3'
        return self.name

    def __repr__(self):
        # repr : Dense($ID, $kdims)
        #     eg "Dense('3, (32, 16)')"
        ID = self.ID
        kdims = self.kdims
        cls_name = self.__class__.__name__

        # Formal repr to eval form
        rep = "{}('{}, {}')".format(cls_name, ID, kdims)
        return rep

    def initialize_params(self, layer_vars):
        """ Initializes optimizable layer_vars for a ParametricLayer instance

        Params
        ------
        layer_vars : list(dict)
            Each element in layer_vars is a dictionary specifying
            the features of a different variable that needs to be
            initialized.
                each dict in layer_vars has the following form:
                {'tag': 'W', 'dims': tuple(int), 'init': Initializer}
        """
        key_val = '{}_{{}}'.format(self.name)
        for layer_var in layer_vars:
            # Unpack var attributes
            #-----------------------------
            tag = layer_var['tag']
            dims = layer_var['dims']
            initializer = layer_var['init']

            # Initialize instance attrs
            #------------------------------
            # var key
            setattr(self, '{}_Key'.format(tag)) = key_val.format(tag)

            # variable
            setattr(self, tag) = initializer(dims)

            # variable grad placeholder
            setattr(self, '{}_grad'format(tag)) = None



class Dense:
    """ Vanilla fully-connected hidden layer
    The Dense layer is defined by the linear transformation function:

         f(X) = X.W + B
    Where
    X : input matrix
    W : weight matrix
    B : bias vector
    and both are learnable parameters optimized through gradient descent.

    Attributes
    ----------
    updates : bool
        whether the layer has learnable parameters
    name : str
        layer name, as class-name + ID
    ID : int
        Dense layer instance's position in the network
    kdims : tuple(int)
        channel sizes (determines dimensions of params)
    linear : Function.Linear
        linear function instance, which performs the Y = X.W + B function
    W_Key : str
        unique string identifier for W
    B_key : str
        unique string identifier for B

    Params
    ------
    W : ndarray.float32, of shape kdims
        weight matrix
    B : ndarray.float32, of shape kdims[-1]
        bias vector

    """
    updates = True
    def __init__(self, ID, kdims, init_W=HeNormal, init_B=Zeros):
        super().__init__(ID, kdims, **kwargs)
        #self.name = '{}{}'.format(self.__class__.__name__, ID)
        #self.ID = ID
        #self.kdims = kdims
        self.linear = functions.Linear()
        self.initialize(init_W, init_B)

    def initialize(self, init_W, init_B):
        var_features = [{'tag': 'W', 'dims': self.kdims, 'init': init_W},
                        {'tag': 'B', 'dims': self.kdims[-1:], 'init': init_B},]
        super().initialize_params(var_features)




    # Layer optimization
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def update(self, opt):
        """ Update weights and bias with gradients from backprop
        Params
        ------
        opt : Optimizer instance
            Optimizing algorithm, updates through __call__
        """
        # Make sure grads exist
        assert self.W_grad is not None and self.B_grad is not None

        # Group vars with their grads, keyed to their respective keys
        params = {}
        params[self.W_Key] = (self.W, self.W_grad)
        params[self.B_Key] = (self.B, self.B_grad)

        # Get updates
        updated_params = opt(params)
        self.W = updated_params[self.W_key]
        self.B = updated_params[self.B_key]

        # Reset grads
        self.W_grad = None
        self.B_grad = None

    # Layer network ops
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def forward(self, X):
        # Outputs
        W = self.W
        B = self.B
        Y = self.linear(X, W, B)
        return Y

    def backward(self, gY):
        # Grads
        W = self.W
        B = self.B
        gX, gW, gB = self.linear(gY, W, B, backprop=True)

        # Assign var grads and chain gX to next layer
        self.W_grad = gW
        self.B_grad = gB
        return gX

    def __call__(self, *args, backprop=False):
        func = self.backward if backprop else forward
        return func(*args)


#==============================================================================
#------------------------------------------------------------------------------
#                             Static Layer
#------------------------------------------------------------------------------
#==============================================================================
""" Layers without updating variables, mostly activations """

class StaticLayer:
    """ Static layer parent class """
    updates = False
    def __init__(self, ID, func, *args):
        self.ID = ID
        self.function = func()
        self.name = '{}Layer{}'.format(str(self.function), ID)

    def __str__(self):
        # simple instance name
        #   eg: 'SigmoidLayer2'
        return self.name

    def __repr__(self):
        # eval form
        #   eg: "StaticLayer('2, functions.Sigmoid')"
        cls_name = self.__class__.__name__
        ID = self.ID
        func_repr  = repr(self.function)
        layer_repr = "{}('{}, {}')".format(cls_name, ID, func_repr)
        return layer_repr

    def __call__(self, *args, backprop=False):
        return self.function(*args, backprop=backprop)


def activation_layer(ID, func, *args):
    """ factory for StaticLayer instances with activation funcs """
    acts = functions.ACTIVATIONS
    if func in acts:
        # then func is key to Function class
        func_cls = acts[func]
        return StaticLayer(ID, func_cls, *args)
    elif func in acts.values():
        # func is actual Function class
        return StaticLayer(ID, func, *args)
    else:
        raise ValueError('Invalid activation function argument')




#==============================================================================

# Available Layers
#-----------------
CONNECTIONS = {'dense': Dense,}
ACTIVATIONS = functions.ACTIVATIONS
LAYERS = {**CONNECTIONS, **ACTIVATIONS}


'''
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
'''
