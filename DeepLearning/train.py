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
import optimizers
from functions import SoftmaxCrossEntropy, LogisticCrossEntropy

# Data setup
arg_parser = utils.Parser()
config = arg_parser.parse_args()
arg_parser.print_args()

# Load data
X = utils.load_iris()
X_train, X_test = utils.split_dataset(X)
X = None


# Model, session config
#==============================================================================

# Session config
#------------------
num_iters  = config.num_iters
batch_size = config.batch_size
checkpoint = config.checkpoint

# Model config
#------------------
num_classes = len(utils.IRIS['classes'])
learning_rate = config.learn_rate
layer_types = [config.layer_op, config.layer_act] # ['dense', 'sigmoid']
#channels = [4, 16, 64, 32, 8, num_classes] # config.channels
#channels = [4, 16, num_classes] # config.channels
channels = [4, 64, num_classes] # config.channels



# Model initialization
#==============================================================================

# Instantiate model
#------------------
np.random.seed(utils.RNG_SEED_PARAMS)
model = nn.NeuralNetwork(channels, activation_tag='sigmoid', final_activation=False)
#objective = SoftmaxCrossEntropy()
objective = LogisticCrossEntropy()
#opt = Adam()
opt = optimizers.SGD(lr=learning_rate)

# Instantiate model results collections
#------------------
train_loss_history = np.zeros((num_iters,2))


# Train
#==============================================================================
def print_train_status(step, err, acc):
    pformat = '{:>3}: {:.5f}  |  {:.4f}'.format(step+1, float(err), float(acc))
    print(pformat)

t_start = time.time()
np.random.seed(utils.RNG_SEED_DATA)

for step in range(num_iters):
    # batch data
    #------------------
    x, y = utils.get_training_batch(X_train, batch_size, step)

    # forward pass
    #------------------
    #y_hat = model(x)
    #print('step {}: x_in =\n {}'.format(step+1, x))
    y_hat = model.forward(x)
    if (step+1) % 50 == 0:
        print('step: {}'.format(step+1))

    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, y)
    train_loss_history[step] = error, accuracy
    print_train_status(step, error, accuracy)

    # backprop and update
    #------------------
    grad_loss = objective(error, backprop=True)
    model.backward(grad_loss)
    model.update(opt)

# Finished training
#------------------------------------------------------------------------------
t_finish = time.time()

# Summary info
elapsed_time = (t_finish - t_start) / 60.
percent = .2
idx = int(num_iters * (1 - percent))
avg_final_error = np.mean(train_loss_history[idx:], axis=0)
median_final_error = np.median(train_loss_history[idx:], axis=0)

# Print training summary
print('# Finished training\n#{}'.format('-'*78))
print(' * Elapsed time: {} minutes'.format(elapsed_time))
print(' * Average error, last 20% iterations: {:.6f} |  {:.6f}'.format(avg_final_error[0], avg_final_error[1]))
print(' * Median error,  last 20% iterations: {:.6f} |  {:.6f}'.format(median_final_error[0], median_final_error[1]))
'''
# Validation
#==============================================================================

# Instantiate model test results collections
#------------------
num_test_samples = X_test.shape[0]
test_loss_history = np.zeros((num_test_samples))

# Test
#------------------
for i in range(num_test_samples):
    x, y = np.split(np.copy(X_test[i:i+1]), [-1], axis=1)

    # forward pass
    #------------------
    #y_hat = model(x)
    y_hat = model.forward(x)
    error = objective(y_hat, y)
    test_loss_history[i] = error


# Finished test
#------------------------------------------------------------------------------
# Summary info
avg_test_error = np.mean(test_loss_history)
median_test_error = np.median(test_loss_history)

# Print training summary
print('\n# Finished Testing\n#{}'.format('-'*78))
print(' * Average error : {:.6f} '.format(avg_test_error))
print(' * Median error  : {:.6f} '.format(median_test_error))
'''
