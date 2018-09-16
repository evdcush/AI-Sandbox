import functions as F
from utils import Initializers


class Dense:
    """ Fully-connected layer that computes
    an output as the linear transformation
    of its input X against some weights W
    """
    linear = F.Linear()
    W = None
    B = None
    opt = None

    def __init__(self, activation=F.sigmoid):
        self.activation = activation()

    def initialize_params(self, Kdims):
        K_in, K_out = Kdims
        self.W = Initializers.henorm(K_in, K_out)
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


