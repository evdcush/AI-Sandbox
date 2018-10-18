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


@REMOVED : if there are no parameters, there is no reason you cannot
           just use the Function as-is
Static layers : [Sigmoid, Tanh, Softmax, ReLU, ELU, SELU, ]
    these layers do not utilize variables and do not
    need updates from optimizer


"""
import functions
import initializers

#==== Initializers
HeNormal = initializers.HeNormal
Zeros    = initializers.Zeros
Ones     = initializers.Ones

#==============================================================================
#------------------------------------------------------------------------------
#                          Parametric Layers
#------------------------------------------------------------------------------
#==============================================================================
""" These layers all have parameters that are updated through gradient descent
"""

class ParametricLayer:
    """ Layers with learnable variables updated through gradient descent

    Attributes
    ----------
    updates : bool
        whether the layer has learnable parameters
    name : str
        layer name, as class-name + ID
    ID : int
        layer instance position in the network
    kdims : tuple(int)
        channel sizes (determines dimensions of params)
    """
    updates = True
    def __init__(self, ID, kdims, **kwargs):
        self.name = '{}-{}'.format(self.__class__.__name__, ID)
        self.ID = ID
        self.kdims = kdims

    def __str__(self,):
        # str : Dense-$ID
        #     eg 'Dense-3'
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
                {'tag': str, 'dims': tuple(int), 'init': Initializer}
        """
        key_val = '{}_{{}}'.format(self.name)
        for layer_var in layer_vars:
            # Unpack var attributes
            #-----------------------------
            tag = layer_var['tag']
            dims = layer_var['dims']
            initializer = layer_var['init']()

            # Initialize instance attrs
            #------------------------------
            # var key
            setattr(self, '{}_key'.format(tag), key_val.format(tag))

            # variable
            setattr(self, tag, initializer(dims))# = initializer(dims)

            # variable grad placeholder
            setattr(self, '{}_grad'.format(tag), None) #= None


    # Layer optimization
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def update(self, opt):
        """ Update weights and bias with gradients from backprop
        Params
        ------
        opt : Optimizer instance
            Optimizing algorithm, updates through __call__
        """
        # WIP
        raise NotImplementedError
        '''
        # Make sure grads exist
        assert self.W_grad is not None and self.B_grad is not None

        # Group vars with their grads, keyed to their respective keys
        params = {}
        params[self.W_key] = (self.W, self.W_grad)
        params[self.B_key] = (self.B, self.B_grad)

        # Get updates
        updated_params = opt(params)
        self.W = updated_params[self.W_key]
        self.B = updated_params[self.B_key]

        # Reset gradsinitialize
        self.W_grad = None
        self.B_grad = None
        '''

    def __call__(self, *args, backprop=False, test=False):
        func = self.backward if backprop else self.forward
        return func(*args, test=False)


#==============================================================================

class Dense(ParametricLayer):
    """ Vanilla fully-connected hidden layer
    Dense essentially provides the layer interface to functions.Linear,
    maintaining the connection variables W, and B, in the function:
        f(X) = X.W + B
    Where
      X : input matrix
      W : weight matrix
      B : bias vector
    and both are learnable parameters optimized through gradient descent.

    Attributes
    ----------
    linear : functions.Linear
        linear function instance, which performs the Y = X.W + B function
    W_key : str
        unique string identifier for W
    B_key : str
        unique string identifier for B
    """
    def __init__(self, ID, kdims, init_W=HeNormal, init_B=Zeros):
        super().__init__(ID, kdims)
        self.linear = functions.Linear()
        self.initialize_vars(init_W, init_B)

    def initialize_vars(self, init_W, init_B):
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
        params[self.W_key] = (self.W, self.W_grad)
        params[self.B_key] = (self.B, self.B_grad)

        # Get updates
        updated_params = opt(params)
        self.W = updated_params[self.W_key]
        self.B = updated_params[self.B_key]

        # Reset grads
        self.W_grad = None
        self.B_grad = None

    # Layer network ops
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def forward(self, X, **kwargs):
        # Outputs
        W = self.W
        B = self.B
        Y = self.linear(X, W, B, **kwargs)
        return Y

    def backward(self, gY, **kwargs):
        # Grads
        W = self.W
        B = self.B
        gX, gW, gB = self.linear(gY, W, B, backprop=True)

        # Assign var grads and chain gX to next layer
        self.W_grad = gW
        self.B_grad = gB
        return gX

#==============================================================================

class Swish(ParametricLayer):
    """ Swish activation layer """
    updates = True
    def __init__(self, ID, kdims, init_B=Ones, **kwargs):
        super().__init__(ID, kdims, **kwargs)
        self.swish = functions.Swish()
        self.initialize_vars(init_B)

    def initialize_vars(self, init_B):
        var_features = [{'tag': 'B', 'dims': self.kdims[-1:], 'init': init_B},]
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
        assert self.B_grad is not None

        # Group vars with their grads, keyed to their respective keys
        params = {}
        params[self.B_key] = (self.B, self.B_grad)

        # Get updates
        updated_params = opt(params)
        self.B = updated_params[self.B_key]
        self.B_grad = None

    # Layer network ops
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def forward(self, X, **kwargs):
        # Outputs
        B = self.B
        Y = self.swish(X, B, **kwargs)
        return Y

    def backward(self, gY, **kwargs):
        # Grads
        B = self.B
        gX, gB = self.swish(gY, B, backprop=True)

        # Assign var grads and chain gX to next layer
        self.B_grad = gB
        return gX


#==============================================================================

# Available Layers
#-----------------
CONNECTIONS = {'dense': Dense,}

# Special cases
#-----------------
PARAMETRIC_FUNCTIONS = {'swish': Swish}
LAYERS = {**CONNECTIONS, **PARAMETRIC_FUNCTIONS}


#==============================================================================
