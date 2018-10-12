"""
Cross validation for parameter search script
"""
import os
import sys
import time
import code
import numpy as np
import matplotlib.pyplot as plt
#import pylab as plt
#plt.style.use('ggplot')
#plt.style.use('bmh')
#plt.ion()

import utils
import layers
import functions as F
import optimizers as Opt
#from network import NeuralNetwork
"""
kernels = np.array([  31,  41,  32, 151,  16,  53,  78, 157,  47, 149, 144, 121,  11,
                     256,  13, 163,  55, 103, 128, 124,   4,  59, 131,  97,   8,  61,
                      29, 101,  83,  17, 113, 167, 127, 101, 130,  37, 139,   7,  64,
                      79, 109,  71,  43,  80,  67, 101,  19,  23,  74,  73, 107,  89,
                      173, 137])

dataset = generate_dataset()

trainer = utils.Trainer([4,128,13,3], Opt.SGD, F.SoftmaxCrossEntropy,
                        layers.Swish, dataset=dataset, steps=2000,
                        batch_size=batch_size, rng_seed=77771)
"""

#==============================================================================
# CV Setup
#==============================================================================
# Model config
#------------------
#CV_SEEDS = {'3310':3310, '99467': 99467, '27189': 27189, '77771': 77771}
SEEDS = [3310, 99467, 27189, 77771]
optimizer   = [Opt.SGD]
objective   = [F.SoftmaxCrossEntropy]
activations = [F.Sigmoid, F.SeLU, layers.Swish]

# Training config
#------------------
NUM_ITERS = 2000

# Dataset
#-------------------
"""
80/20 train/test split gives us an even 5-fold for cross val
Per fold:
 * 24 test samples
 * 96 train samples
"""
_X_train = np.load(utils.IRIS_TRAIN)
split_size = .8 # already default in dataset
batch_size = 6

# Target variable: channels
#--------------------------------------------------------------
K_IN  = 4
K_OUT = 3
MAX_DEPTH = 8 # 9 layers total, including input-out
MAX_SIZE = 650 # limit on sum kernels for any given channels sample
CHANNELS = list(np.load('GEN_500_CHANNELS.npy'))


# Helpers
#--------------------------------------------------------------
def generate_dataset():
    x_copy = np.copy(_X_train)
    split_seed = np.random.choice(15000)
    _dataset = utils.IrisDataset(x_copy, split_size, split_seed)
    return _dataset

def init_trainer(chan, act, dset, seed):
    return utils.Trainer(chan, Opt.SGD, F.SoftmaxCrossEntropy,
                         act, dataset=dset, steps=NUM_ITERS,
                         batch_size=batch_size, rng_seed=seed)


# Loss tracker
#--------------------------------------------------------------
cv_dims = (len(SEEDS), len(activations), len(CHANNELS), NUM_ITERS, 2)
cv_test_dims = cv_dims[:-2] + (24, 2)
#==== collections
CV_TRAIN_LOSS = np.zeros(cv_dims,      np.float32)
CV_TEST_LOSS  = np.zeros(cv_test_dims, np.float32)


# Train
#--------------------------------------------------------------
for idx_seed, seed in enumerate(SEEDS):
    for idx_act, act in enumerate(activations):
        dataset = generate_dataset()
        for idx_chan, channels in enumerate(CHANNELS):
            # copy data for safety
            dataset.X_train = np.copy(dataset.X_train)
            dataset.X_test  = np.copy(dataset.X_test)

            # make trainer and train
            trainer = init_trainer(channels, act, dataset, seed)
            trainer()

            # save results
            lh_train, lh_test = trainer.get_loss_histories()
            CV_TRAIN_LOSS[idx_seed, idx_act, idx_chan] = lh_train
            CV_TEST_LOSS[ idx_seed, idx_act, idx_chan] = lh_test

np.save('CV_train_loss', CV_TRAIN_LOSS)
np.save('CV_test_loss',  CV_TEST_LOSS)
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
