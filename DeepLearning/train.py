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
X = utils.load_iris()
X_train, X_test = utils.split_dataset(X)
X = None

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
sess_status = SessionStatus(model, opt, objective, num_iters, X_test.shape[0])
loss_tracker = np.zeros((num_iters, 2))


#plt.ion()
#==============================================================================
# Train
#==============================================================================
t_start = time.time()
np.random.seed(utils.RNG_SEED_DATA)


for step in range(num_iters):
    plt.clf()
    # batch data
    #------------------
    x, y = utils.get_batch(X_train, step, batch_size)

    # forward pass
    #------------------
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y)
    accuracy = classification_accuracy(class_scores, y)
    loss_tracker[step] = error, accuracy
    sess_status(step, error, accuracy, show=True)

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
sess_status.print_results(t=elapsed_time)

code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

#==============================================================================
# Validation
#==============================================================================
'''
# Instantiate model test results collections
#------------------
num_test_samples = X_test.shape[0]
print('# Start testing\n#{}'.format('-'*78))
# Test
#------------------
for i in range(num_test_samples):
    x, y = utils.get_batch(X_test, i, test=True)

    # forward pass
    #------------------
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, y)
    sess_status(i, error, accuracy, train=False)


# Finished test
#------------------------------------------------------------------------------
# Summary info

# Print training summary
print('\n# Finished Testing\n#{}'.format('-'*78))
sess_status.summarize_model(True, True)

#trl, tel = sess_status.get_loss_history()
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
'''
