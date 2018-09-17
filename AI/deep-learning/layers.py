import functions as F
from utils import Initializers
from nn import SGD, Adam



class Dense:
    """ Fully-connected layer that computes
    an output as the linear transformation
    of its input X against some weights W
    """
    linear = F.Linear()

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
