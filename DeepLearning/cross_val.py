"""
Cross validation for parameter search script
"""
import subprocess
import numpy as np

import utils
import layers
import functions as F
import optimizers as Opt

import utils
from network import NeuralNetwork
import argparse

class AttrDict(dict):
    """ simply a dict accessed/mutated by attribute instead of index
    Warning: cannot be pickled like normal dict/object
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

parser = argparse.ArgumentParser()
adg = parser.add_argument
adg('-s', type=int, required=True) # seed index
adg('-c', type=int, required=True) # channels index
adg('-a', type=int, required=True) # activation index
sess_args = AttrDict(vars(parser.parse_args()))
cidx = sess_args.c
sidx = sess_args.s
aidx = sess_args.a

# Select args for this session
channels = list(np.load('./TESTING/GEN_700_CHANNELS.npy'))[cidx]
seed_params = [3310, 99467, 27189, 77771][sidx]
activation = [F.Sigmoid, F.SeLU, layers.Swish][aidx]

optimizer = Opt.Adam
objective_fn = F.SoftmaxCrossEntropy


# Training config
#------------------
NUM_ITERS = 1000
batch_size = 6

# Split dataset
seed_dataset = round(seed_params / 7)
_X_train = np.copy(np.load(utils.IRIS_TRAIN))
X_train, X_test = utils.IrisDataset.split_dataset(_X_train, split_size=0.8, seed=seed_dataset)
num_test_samples = X_test.shape[0]


# Model initialization
#==============================================================================

# Instantiate model
#------------------
np.random.seed(seed_params)
model = NeuralNetwork(channels, activation=activation)
objective = objective_fn()
opt = optimizer()


lh_train = np.zeros((NUM_ITERS, 2), np.float32)
lh_test  = np.zeros((num_test_samples, 2), np.float32)

#plt.ion()
#==============================================================================
# Train
#==============================================================================

for step in range(NUM_ITERS):
    # batch data
    #------------------
    x, y = utils.IrisDataset.get_batch(X_train, step, batch_size)

    # forward pass
    #------------------
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, y)
    lh_train[step] = [error, accuracy]

    # backprop and update
    #------------------
    grad_loss = objective(error, backprop=True)
    model.backward(grad_loss)
    model.update(opt)


# Finished training
#------------------------------------------------------------------------------

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

#==============================================================================
# Validation
#==============================================================================

#------------------
for i in range(num_test_samples):
    x, y = utils.IrisDataset.get_batch(X_test, i, test=True)

    # forward pass
    #------------------
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, y)
    lh_test[i] = [error, accuracy]


# Finished test
#------------------------------------------------------------------------------

cv_lh_train = np.load('CV_lh_train.npy')
cv_lh_train[cidx, aidx, sidx] = lh_train
np.save('CV_lh_train.npy', cv_lh_train)

cv_lh_test = np.load('CV_lh_test.npy')
cv_lh_test[cidx, aidx, sidx] = lh_test
np.save('CV_lh_test.npy', cv_lh_test)
