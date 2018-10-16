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
X_train, X_test = iris_data.X_train, iris_data.X_test
num_test_samples = X_test.shape[0]


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
    # batch data
    #------------------
    x, y = iris_data.get_batch(X_train, step, batch_size)

    # forward pass
    #------------------
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y)
    accuracy = classification_accuracy(class_scores, y)
    sess_status(step, error, accuracy, show=verbose)
    #loss_tracker[step] = [error, accuracy]

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


#==============================================================================
# Validation
#==============================================================================
#print('# Start testing\n#{}'.format('-'*78))
px = None
# Test
#------------------
for i in range(num_test_samples):
    x, y = iris_data.get_batch(X_test, i, test=True)

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

