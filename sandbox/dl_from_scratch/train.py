""" Train script for classifier network
"""
import numpy as np

import utils
import functions as F
import optimizers
from network import NeuralNetwork

#-----------------------------------------------------------------------------#
#                                    SETUP                                    #
#-----------------------------------------------------------------------------#

args = utils.parse_args()

# Data
# ====
dataset = utils.DATASETS[args.dataset]()
assert hasattr(dataset, 'target_names')  # only support classification for now
dataset.split_dataset() # splits dataset into train, validation, test sets
num_features = dataset.x_train.shape[-1]
num_classes  = len(dataset.target_names)

# model conf
# ==========
channels = [num_features, 128, num_classes]
activation = F.Selu
learn_rate = 0.001

# training conf
# =============
num_iters  = args.iters
batch_size = args.batch_size
num_test = dataset.x_test.shape[0]
#num_val  = dataset.x_validation.shape[0]


# Initialize model
# ================
np.random.seed(args.rand)
model = NeuralNetwork(channels, activation=activation)
optimizer = optimizers.Adam(alpha=learn_rate)
objective = F.SoftmaxCrossEntropy()
#objective = F.LogisticCrossEntropy()


#-----------------------------------------------------------------------------#
#                                    TRAIN                                    #
#-----------------------------------------------------------------------------#
training_error = np.zeros((num_iters, 2))

np.random.seed(0)
for step in range(num_iters):
    # training batch
    x, y = dataset.get_batch(batch_size)

    # forward pass
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y, num_classes)
    accuracy = utils.classification_accuracy(class_scores, y)
    training_error[step] = [error, accuracy]
    pred = np.argmax(class_scores, 1)
    #print(f'{step:>5}: {pred} | {y}')

    # backprop and update
    grad_loss = objective(backprop=True)
    model.backward(grad_loss)
    model.update(optimizer)

utils.print_results(training_error[:,0], title='loss')
utils.print_results(training_error[:,1], title='accuracy')


#-----------------------------------------------------------------------------#
#                                    TEST                                     #
#-----------------------------------------------------------------------------#
test_error = np.zeros((num_test, 2))
print('BEGIN TEST')
import sys
from code import interact

'''
for i in range(num_test):
    # data sample
    j = i + 1
    x, y = dataset.x_test[i:j], dataset.y_test[i:j]

    # forward pass
    y_hat = model.forward(x, test=True)
    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, y)
    test_error[i] = [error, accuracy]
    print(f'{i:3>}: {error:.4f} {accuracy:.4f}')
    interact(local=dict(globals(), **locals()))
'''

xtest, ytest = dataset.x_test, dataset.y_test
y_hat = model.forward(xtest, test=True)
error, class_scores = objective(y_hat, ytest, num_classes)
accuracy = utils.classification_accuracy(class_scores, ytest)
test_error[:] = [error, accuracy]
pred = np.argmax(class_scores, 1)
print(pred)
print(ytest)


print('FINISH TEST')
utils.print_results(test_error[:,0], title='loss')
utils.print_results(test_error[:,1], title='accuracy')
