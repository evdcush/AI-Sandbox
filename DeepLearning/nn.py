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
    layers = []
    def __init__(self, channels,
                 connection_layer='dense_layer',
                 activation_layer='sigmoid_layer',
                 final_activation=False, restore=None, initializers=None):
        self.channels = channels
        self.connection_layer = connection_layer
        self.activation_layer = activation_layer
        self.final_activation = final_activation

        self.initialize_layers(restore=restore, initializers=initializers)

    def __repr__(self):
        # eval-style repr
        rep = ("{}('{}, connection_layer={}, activation_layer={}, \
        final_activation={}')")

        # Get instance vars
        name = self.__class__.__name__
        chans = self.channels
        conn = self.connection_layer
        act  = self.activation_layer
        final_act = self.final_activation

        # format repr and return
        rep = rep.format(name, chans, conn, act, final_act)
        return rep

    def add_activation(self, ID, kdim, act, **kwargs):
        last_layer = len(self.channels) - 1

        if not last_layer:
            activation = self.act(ID, kdim, **kwargs)
            self.layers.append(activation)
        else:
            if self.final_act:
                activation = self.act(ID, kdim, **kwargs)
                self.layers.append(activation)


    def initialize_layers(self, **kwargs):
        """ """
        # Get layers
        layers = L.get_all_layers()

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
            connection = connection_layer(ID, kdim, **kwargs)
            self.layers.append(connection)
            self.add_activation(ID, kdim, activation_layer, **kwargs)

    def forward(self, X):
        """ Propagates input X through network layers """
        Y = np.copy(X)
        for layer in self.layers:
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
                opt(layer)


