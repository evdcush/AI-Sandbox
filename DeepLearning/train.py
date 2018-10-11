""" Training script for iris model
"""
import time
import sys
import code
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
#import pylab as plt
#plt.style.use('ggplot')
#plt.style.use('bmh')
#plt.ion()

import utils
from utils import SessionStatus, classification_accuracy
from network import NeuralNetwork

# args parser
#------------------
# config contains all model and session setup options
arg_parser = utils.Parser()
config = arg_parser.parse_args()
arg_parser.print_args()

# Load data
#------------------
iris_data = utils.IrisDataset()
num_test_samples = iris_data.X_test.shape[0]


#==============================================================================
# Model, session config
#==============================================================================
# Model config
#------------------
channels   = config.channels
activation = config.activation
optimizer  = config.optimizer
objective  = config.objective
dropout    = config.dropout
learning_rate = config.learn_rate # 0.01, omitted, opt defaults well configured

# Session config
#------------------
num_iters  = config.num_iters
batch_size = config.batch_size
verbose = config.verbose


# Model initialization
#==============================================================================

# Instantiate model
#------------------
np.random.seed(utils.RNG_SEED_PARAMS)
model = NeuralNetwork(channels, activation=activation, use_dropout=dropout)
objective = objective()
opt = optimizer()

# Model status reporter
#------------------
sess_status = SessionStatus(model, opt, objective, num_iters, num_test_samples)
loss_tracker = np.zeros((num_iters, 2))


#plt.ion()
#==============================================================================
# Train
#==============================================================================
t_start = time.time()
np.random.seed(utils.RNG_SEED_DATA)
prev_x = None

for step in range(num_iters):
    plt.clf()
    # batch data
    #------------------
    x, y = iris_data.get_batch(step, batch_size)
    # SANITY CHECK
    #if step == 0:
    #    prev_x = np.copy(x)
    #else:
    #    assert not np.all(x == prev_x)
    #    prev_x = np.copy(x)
#  ______   _   _   _____    ______   _____     _    _   ______   _____    ______   #
# |  ____| | \ | | |  __ \  |  ____| |  __ \   | |  | | |  ____| |  __ \  |  ____|  #
# | |__    |  \| | | |  | | | |__    | |  | |  | |__| | | |__    | |__) | | |__     #
# |  __|   | . ` | | |  | | |  __|   | |  | |  |  __  | |  __|   |  _  /  |  __|    #
# | |____  | |\  | | |__| | | |____  | |__| |  | |  | | | |____  | | \ \  | |____   #
# |______| |_| \_| |_____/  |______| |_____/   |_|  |_| |______| |_|  \_\ |______|  #
"""
Been getting really good numbers ever since the Dataset class and
session status class were made. There have been a lot of changes,
all over the package code, and v2 is just that much more consistent I guess
 - At any rate, needed to do some sanity checks, checking
   integrity of batching (making sure they are all different).
   Also made a separate loss tracker to keep track of statistics
   in case the wildly sprawling super-class SessionStatus (REALLY
   NEED TO RELOCATE SOME OF ITS FUNCTIONALITIES INTO SEPARATE CLASSES),
   but that was fine too.

* Originally planned on doing the CV thing in the notebook, where
  it's easiest to follow progress, eyeball current results and debug.
  But it would probably be better just to either use the NORMAL
  train script (this one), or a specialized CV script
    The options
    ===========
      * Notebook, fat, complex, nested for-loops (about 4~5)
      * Specialized script:
         - nest for-loops like notebook
         - basic setup for single-run
>>---> * Shell script or python script, with ALL permutations of
        network configs for CV, that runs another script for each one
        (instead of a bunch of for loops).
          > Leaning towards this option. Also makes it easier to
            serialize and organize the CV configurations to files,
            or a file, as well as write out good, formatted, summary
            statistics to a text file, or npy or dict or
            whatever makes sense.
"""
    # forward pass
    #------------------
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y)
    accuracy = classification_accuracy(class_scores, y)
    sess_status(step, error, accuracy, show=verbose)
    loss_tracker[step] = [error, accuracy]

    # backprop and update
    #------------------
    grad_loss = objective(error, backprop=True)
    model.backward(grad_loss)
    model.update(opt)


# Finished training
#------------------------------------------------------------------------------
# Summary info
t_finish = time.time()
elapsed_time = (t_finish - t_start)
#sess_status.print_results(t=elapsed_time)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

#==============================================================================
# Validation
#==============================================================================
#print('# Start testing\n#{}'.format('-'*78))
px = None
# Test
#------------------
for i in range(num_test_samples):
    x, y = iris_data.get_batch(i, test=True)
    # SANITY CHECK
    #if step == 0:
    #    px = np.copy(x)
    #else:
    #    assert not np.all(x == px)
    #    px = np.copy(x)

    # forward pass
    #------------------
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, y)
    sess_status(i, error, accuracy, show=verbose, test=True)


# Finished test
#------------------------------------------------------------------------------
# Summary info

# Print training summary
#print('\n# Finished Testing\n#{}'.format('-'*78))
sess_status.summarize_model(True, True)


## SANITY CHECK: making sure results are what they seem
#trunc = int(num_iters * .4)
#lh_trunc = loss_tracker[trunc:] # stats from latter 60% of training
#avg = np.mean(  np.copy(lh_trunc), axis=0)
#q50 = np.median(np.copy(lh_trunc), axis=0)
#print('SEPARATE TRACKER COLLECTION DATA:')
#print('            Error  |  Accuracy')
#print('* Average: {:.5f} | {:.5f}'.format(avg[0], avg[1]))
#print('*  Median: {:.5f} | {:.5f}'.format(q50[0], q50[1]))
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

