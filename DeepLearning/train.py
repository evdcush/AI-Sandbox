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
num_classes = len(utils.IRIS['classes']) # 3
learning_rate = config.learn_rate # 0.01
layer_types = [config.layer_connection, config.layer_activation] # ['dense', 'sigmoid']
activation = config.layer_activation
#channels = [4, 16, 64, 32, 8, num_classes]
#channels = [4, 16, num_classes]
channels = [4, 64, num_classes] # config.channels
#channels = config.channels

# Session config
#------------------
num_iters  = config.num_iters
batch_size = config.batch_size


# Model initialization
#==============================================================================

# Instantiate model
#------------------
np.random.seed(utils.RNG_SEED_PARAMS)
model = nn.NeuralNetwork(channels, activation_tag=activation, final_activation=False)
#objective = SoftmaxCrossEntropy()
objective = LogisticCrossEntropy()
#opt = Adam()
opt = optimizers.SGD(lr=learning_rate)

# Instantiate model results collections
#------------------
train_loss_history = np.zeros((num_iters,2))
code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

# Train
#==============================================================================
def print_status(step, err, acc):
    pformat = '{:>3}: {:.5f}  |  {:.4f}'.format(step+1, float(err), float(acc))
    print(pformat)

t_start = time.time()
np.random.seed(utils.RNG_SEED_DATA)

for step in range(num_iters):
    # batch data
    #------------------
    x, y = utils.get_batch(X_train, step, batch_size)

    # NOTE: SQUEEZE(y) (n, 1) --> (N,) MADE A HUGE DIFFERENCE TO CLASSIFICATION
    #       ACCURACY

    # forward pass
    #------------------
    y_hat = model.forward(x)

    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, np.squeeze(y))
    train_loss_history[step] = error, accuracy
    if (step+1) % 50 == 0:
        print_status(step, error, accuracy)
    #print_train_status(step, error, accuracy)

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
print('\nModel layers: {}\n'.format(model.layers))
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use



# Validation
#==============================================================================

# Instantiate model test results collections
#------------------
num_test_samples = X_test.shape[0]
test_loss_history = np.zeros((num_test_samples, 2))
print('# Start testing\n#{}'.format('-'*78))
# Test
#------------------
for i in range(num_test_samples):
    #x, y = np.split(np.copy(X_test[i*batch_size:(i+1)*batch_size]), [-1], axis=1)
    #y = y.astype(np.int32)
    x, y = utils.get_batch(X_test, i, test=True)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use


    # forward pass
    #------------------
    y_hat = model.forward(x)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, y)
    test_loss_history[i] = error, accuracy
    print_status(i, error, accuracy)


# Finished test
#------------------------------------------------------------------------------
# Summary info
avg_test_error = np.mean(test_loss_history, axis=0)
median_test_error = np.median(test_loss_history, axis=0)

# Print training summary
print('\n# Finished Testing\n#{}'.format('-'*78))
print(' * Average error : {:.6f}  |  {:.6f} '.format(avg_test_error[0], avg_test_error[1]))
print(' * Median error  : {:.6f}  |  {:.6f} '.format(median_test_error[0], median_test_error[1]))

#x_test, y_test = np.split(np.copy(X_test), [-1], axis=1)
#y_test = np.squeeze(y_test.astype(np.int32))
#y_hat_test = model.forward(x_test)
#error, class_scores = objective(y_hat_test, y_test)
#accuracy = utils.classification_accuracy(class_scores, np.squeeze(y_test))

#test_loss_history[i] = error, accuracy
#print('\n# Finished Testing\n#{}'.format('-'*78))
#print_status(0, error, accuracy)
#np.set_printoptions(precision=4, suppress=True)
#scores = np.argmax(class_scores, axis=1)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
