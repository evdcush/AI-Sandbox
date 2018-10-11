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
import functions as F
import optimizers as Opt
#from network import NeuralNetwork


#==============================================================================
# CV Setup
#==============================================================================
# Model config
#------------------
#CV_SEEDS = {'3310':3310, '99467': 99467, '27189': 27189, '77771': 77771}
SEEDS = [3310, 99467, 27189, 77771]
optimizer   = [Opt.SGD]
objective   = [F.SoftmaxCrossEntropy]
activations = [F.Sigmoid, F.SeLU, F.Swish]

# Target variable: channels
#--------------------------------------------------------------
K_IN  = 4
K_OUT = 3
MAX_DEPTH = 8 # 9 layers total, including input-out
MAX_SIZE = 650 # limit on sum kernels for any given channels sample

kernels = np.array([  31,  41,  32, 151,  16,  53,  78, 157,  47, 149, 144, 121,  11,
                     256,  13, 163,  55, 103, 128, 124,   4,  59, 131,  97,   8,  61,
                      29, 101,  83,  17, 113, 167, 127, 101, 130,  37, 139,   7,  64,
                      79, 109,  71,  43,  80,  67, 101,  19,  23,  74,  73, 107,  89,
                      173, 137])

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

def generate_dataset():
    x_copy = np.copy(_X_train)
    split_seed = np.random.choice(15000)
    _dataset = utils.IrisDataset(x_copy, split_size, split_seed)
    return _dataset

dataset = generate_dataset()

trainer = utils.Trainer([4,128,13,3], Opt.SGD, F.SoftmaxCrossEntropy,
                        F.Sigmoid, dataset=dataset, steps=2000,
                        batch_size=batch_size, rng_seed=77771)

trainer()
lh_train, lh_test = self.get_loss_histories()

code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
