"""
Neural network implementations are located in this module.


# Neural networks:
#---------------------------
Neural networks are composed of Layers
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

# Available implementations
#----------------------------
NeuralNetwork : vanilla MLP
    The most basic neural network (though of arbitrary depth). It is
    composed of fully-connected layers, which perform
    linear-transformations on input data, followed by nonlinearities
    (activations).

"""
import sys
import code
import numpy as np
import layers as L


class NeuralNetwork:
    """ Base Neural Network compsed of Layers

    Control Flow
    ------------
    1 - Network instance initialized with a list of channels
        and Layer-types

    2 - Network receives external data input X, and propagates
        it through it's Layers

    3 - A final output Layer returns a prediction

      - (error between network prediction and truth is evaluated by an objective)

    4 - If training: the network receives the gradient of a loss
        function and backpropagates the gradient through its
        layers (in reverse order)
        - If the layer has learnable parameters, such as a Dense layer,
          it will store the gradients for its variables

    5 - An optimizer then updates all learnable variables in the network's
        layers

    * This process is repeated indefinitely, typically until it has converged
      upon the best local minimum, or whenever the specified number of epochs
      or iterations has been reached

    """
    def __init__(self, channels,
                 connection_tag='dense',
                 activation_tag='sigmoid',
                 final_activation=False, **kwargs):
        """ Initializes a neural network that has depth len(channels)
        Params
        ------
        channels : list(int)
            channel sizes for each layer (determines the dimensionality
            of the layer transformations)
        connection_tag : str
            keyword used to retrieve it's corresponding Layer that uses
            connections (eg, has learnable weights)
        activation_tag : str
            keyword used to generate an activation layer based from
            its constituent Function
        final_activation : bool
            whether there is an activation on the final layer's output
        kwargs :
            While not explicit, if the model is reinitialized from
            pretrained weights, they would be passed through **kwargs
        """
        self.channels = list(zip(channels, channels[1:])) # tuple (k_in, k_out)
        self.register_layers(connection_tag, activation_tag)

        self.layers = []
        for ID, kdims in enumerate(self.channels):
            connection = self.connection(ID, kdims, **kwargs)
            activation =



        self.initialize_layers(**kwargs)

    def __repr__(self):
        # eval-style repr
        rep = ("{}('{}, connection_tag={}, activation_tag={}, \
            final_activation={}')")

        # Get instance vars
        name  = self.__class__.__name__
        chans = self.channels
        conn  = self.connection_tag
        act   = self.activation_tag
        final_act = self.final_activation

        # format repr and return
        rep = rep.format(name, chans, conn, act, final_act)
        return rep

    def add_activation(self, ID, kdim, act, **kwargs):
        last_layer = len(self.channels) - 1

        if not last_layer:
            activation = act(ID, kdim, **kwargs)
            self.layers.append(activation)
        else:
            if self.final_activation:
                activation = act(ID, kdim, **kwargs)
                self.layers.append(activation)

    def register_layers(self, connection_tag, activation_tag):
        """ Registers layers to attributes if tags are valid

        Note: self.connection is of type Layer, but
              self.activation is actually a Function, and will be
              instantiated as a 'layers.StaticLayer' when initialized
        """
        assert (activation_tag in L.ACTIVATIONS and
                connection_tag in L.CONNECTIONS)
        # Assign tags as attributes
        self.connection_tag = connection_tag
        self.activation_tag = activation_tag

        # Register layers
        self.connection = L.CONNECTIONS[connection_tag] # layers


    def initialize_layers(self, **kwargs):
        """ """
        # Get layers
        layers = L.LAYERS

        # Network specs
        kdims = self.channels
        conn_label = self.connection_layer
        act_label  = self.activation_layer
        act_output = self.final_activation

        # Integrity check on specified layers
        assert conn_label in layers and act_label in layers

        # Get network layers and initialize
        connection_layer = layers[conn_label]
        activation_layer = layers[act_label]

        for ID, kdim in enumerate(kdims):
            connection = connection_layer(ID, kdim, )#**kwargs)
            self.layers.append(connection)
            #print('{}, {} ---- id(params) = {}'.format(ID, kdim, id(connection.params)))
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            self.add_activation(ID, kdim, activation_layer, )#**kwargs)

    def forward(self, X):
        """ Propagates input X through network layers """
        Y = np.copy(X)
        for layer in self.layers:
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            Y = layer.forward(Y)
        return Y

    def backward(self, gY):
        """ Backpropagation through layers

        Params
        ------
        gY : ndarray
            gradient of loss function wrt to network output Y (or Y_hat)
        """
        gX = np.copy(gY)
        for layer in reversed(self.layers):
            #print('BACKWARD LOOP, NN, DEBUGGING LAYER INIT AND FORWARD, EXITING NOW')
            #sys.exit()
            #print(repr(layer))
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            gX = layer.backward(gX)

    def update(self, opt):
        """ Pass optimizer through layers to update layers with params """
        for layer in self.layers:
            if layer.updates:
                layer.update(opt)


"""
ID: 0
kdim: 4, 16

>>> id(connection)
140466774141696

>>> id(connection.params)
140466779102808

>>> connection.params.keys()
dict_keys(['dense_layer0_W', 'dense_layer0_B'])

--------------
ID: 1
kdim: 16, 3

>>> id(connection)
140466774246736

>>> id(connection.params)
140466779102808 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# SAME PARAMS OBJECT SHARED

>>> connection.params.keys()
dict_keys(['dense_layer0_W', 'dense_layer0_B', 'dense_layer1_W', 'dense_layer1_B'])


# REMOVE PARAMS FROM BEING A CLASS VARIABLE

"""
