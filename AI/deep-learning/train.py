""" Training script for iris model
"""
import time
import code
import numpy as np

import utils
import layers
import initializers
from optimizers import SGD, Adam
from functions import SoftmaxCrossEntropy

# Data setup

argv_parser = utils.Parser()
config = argv_parser.parse_args()


X = np.load(utils.IRIS_DATA_PATH)
X_train, X_test = utils.split_dataset(X)


# Model, session config
#==============================================================================

# Session config
#------------------
num_iters  = config.num_iters
batch_size = config.batch_size
checkpoint = config.checkpoint

# Model config
#------------------
learning_rate = config.learn_rate
blocks = [config.block_op, config.block_act] # ['dense', 'sigmoid']
channels = [4, 16, 64, 32, 8, 1] # config.channels



# Model initialization
#==============================================================================

# Instantiate model
#------------------
np.random.seed(utils.RNG_SEED_PARAMS)
model = layers.NeuralNetwork(channels, 'M', layer_blocks=blocks)
objective = SoftmaxCrossEntropy()
opt = Adam()

# Instantiate model results collections
#------------------
train_loss_history = np.zeros((num_iters,))

# Train
#==============================================================================
t_start = time.time()
np.random.seed(utils.RNG_SEED_DATA)

for step in range(num_iters):
    # batch data
    #------------------
    x, y = utils.get_training_batch(X, batch_size, step)

    # forward pass
    #------------------
    y_hat = model(x)
    error = objective(y_hat, y)
    train_loss_history[step] = error


    # backprop and update
    #------------------
    grad_loss = objective(error, backprop=True)
    model(grad_loss, opt, backprop=True)

# Finished training
#------------------------------------------------------------------------------
t_finish = time.time()

# Summary info
elapsed_time = (t_finish - t_start) / 60.
avg_final_error = np.mean(train_loss_history[-50:])
median_final_error = np.median(train_loss_history[-50:])

# Print training summary
print('# Finished training\n#{}'.format('-'*78))
print(' * Elapsed time: {} minutes'.format(elapsed_time))
print(' * Average error, final 50 iterations: {:.6f} '.format(avg_final_error))
print(' * Median error,  final 50 iterations: {:.6f} '.format(median_final_error))

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
    y_hat = model(x)
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
