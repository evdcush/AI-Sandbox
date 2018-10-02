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

import sys
import code
import numpy as np
import layers as L
import functions


class NeuralNetwork:
    """ Fully-connected, feed-forward neural network
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
        self.final_activation = final_activation
        self.register_layers(connection_tag, activation_tag)

        # Initialize layers
        #------------------
        self.layers = []
        for ID, kdims in enumerate(self.channels):
            # Connection
            #------------------
            connection = self.connection_layer(ID, kdims, **kwargs)
            self.layers.append(connection)

            # Activation
            #------------------
            # check whether last-layer activation
            if self.can_add_activation(ID):
                activation = self.activation_layer(ID, kdims, **kwargs)
                self.layers.append(activation)

    def __str__(self):
        name  = self.__class__.__name__
        return name

    def __repr__(self):
        # eval-style repr
        rep = ("{}('{}, connection_tag={}, activation_tag={}, final_activation={}')")
        # Get instance vars
        name  = self.__class__.__name__
        chans = self.channels
        conn  = self.connection_tag
        act   = self.activation_tag
        final_act = self.final_activation

        # format repr and return
        rep = rep.format(name, chans, conn, act, final_act)
        return rep

    def can_add_activation(self, ID):
        is_final_layer_id = (ID == len(self.channels) - 1)
        return not is_final_layer_id or self.final_activation

    def register_layers(self, connection_tag, activation_tag):
        """ Registers layers to attributes if tags are valid

        Note: self.connection_layer is of type Layer, but
              self.activation is actually a Function, and will be
              instantiated as a 'layers.StaticLayer' when initialized
        """
        assert (activation_tag in functions.ACTIVATIONS and
                connection_tag in L.CONNECTIONS)
        # Assign tags as attributes
        self.connection_tag = connection_tag
        self.activation_tag = activation_tag

        # Register layers
        self.connection_layer = L.CONNECTIONS[connection_tag] # layers
        self.activation_layer = L.ActivationLayer(activation_tag)

    def forward(self, X):
        """ Propagates input X through network layers """
        Y = np.copy(X)
        for layer in self.layers:
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            Y = layer(Y)
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
            gX = layer(gX, backprop=True)

    def update(self, opt):
        """ Pass optimizer through layers to update layers with params """
        for layer in self.layers:
            if layer.updates:
                layer.update(opt)
