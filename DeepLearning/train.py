""" Training script for iris model
"""
import time
import sys
import code
import numpy as np

import nn
import utils
import layers
import initializers
from optimizers import Adam, SGD, get_optimizer
from functions import SoftmaxCrossEntropy, LogisticCrossEntropy


# args parser
#------------------
arg_parser = utils.Parser()
config = arg_parser.parse_args()
arg_parser.print_args()

# Load data
#------------------
X = utils.load_iris()
X_train, X_test = utils.split_dataset(X)
X = None


# Model, session config
#==============================================================================

# Model config
#------------------
channels = config.channels
activation = config.activation
layer_types = [config.layer_connection, config.layer_activation] # ['dense', 'sigmoid']
learning_rate = config.learn_rate # 0.01

# Session config
#------------------
num_iters  = config.num_iters
batch_size = config.batch_size


# Model initialization
#==============================================================================

# Instantiate model
#------------------
np.random.seed(utils.RNG_SEED_PARAMS)
model = nn.NeuralNetwork(channels, activation=activation)
opt = get_optimizer(config.optimizer)()
objective = SoftmaxCrossEntropy()
#objective = LogisticCrossEntropy()

# Model status reporter
#------------------
sess_status = utils.SessionStatus(model, opt, objective, num_iters, X_test.shape[0])


# Train
#==============================================================================
t_start = time.time()
np.random.seed(utils.RNG_SEED_DATA)

for step in range(num_iters):
    # batch data
    #------------------
    x, y = utils.get_batch(X_train, step, batch_size)

    # forward pass
    #------------------
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, np.squeeze(y))
    sess_status(step, error, accuracy)


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

# Print training summary
print('# Finished training\n#{}'.format('-'*78))
print(' * Elapsed time: {}s'.format(elapsed_time))
sess_status.print_results()



# Validation
#==============================================================================

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
sess_status.print_results(train=False)

sess_status.summarize_model(True, True)

trl, tel = sess_status.get_loss_history()
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
