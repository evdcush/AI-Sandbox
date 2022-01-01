import os
import sys
import code
import argparse

import numpy as np

fpath = os.path.abspath(os.path.dirname(__file__))
sys.path.append('/'.join(fpath.split('/')[:-1]))
import utilities

#-----------------------------------------------------------------------------#
#                                   config                                    #
#-----------------------------------------------------------------------------#
""" Default config settings """

# Seeds
DATA_SEED   = 9959  # rng seed for splitting dataset
PARAMS_SEED = 123   # rng seed for variable initialization

# Model params
LEARNING_RATE = 0.01
CHANNELS = [64, 128] # hidden layer sizes


#-----------------------------------------------------------------------------#
#                                    Data                                     #
#-----------------------------------------------------------------------------#
DATASETS = utilities.DATASETS
datasets_available = list(DATASETS.keys())

#-----------------------------------------------------------------------------#
#                                   Parser                                    #
#-----------------------------------------------------------------------------#

CLI = utilities.CLI
adg = CLI.add_argument
adg('-i', '--iters', type=int, default=2000,
    help='number of training iterations')
adg('-b', '--batch_size', type=int, default=5, metavar='B',
    help='training minibatch size')
adg('-d', '--dataset', type=str, default='iris', choices=datasets_available,
    help='dataset choice')
adg('-r', '--rand', type=int, default=PARAMS_SEED, metavar='seed',
    help='random seed for parameter initialization')

def parse_args():
    args = CLI.parse_args()
    if args.batch_size < 1 or args.iters < 1:
        CLI.error('Only positive integers are permitted')
    return args



#-----------------------------------------------------------------------------#
#                                   Logger                                    #
#-----------------------------------------------------------------------------#

def print_results(error, axis=None, title=''):
    avg = np.mean(error,   axis=axis)
    q50 = np.median(error, axis=axis)
    print(f'ERROR {title}:\n\taverage :{avg:.5f}\n\t median : {q50:.5f}\n')



#-----------------------------------------------------------------------------#
#                                 model utils                                 #
#-----------------------------------------------------------------------------#
# need to make a model class, classifier subclass, and these funcs shhould
# be instance methods

def get_predictions(Y_hat):
    """ Select the highest valued class labels in prediction from
    network output distribution
    We can approximate a single label prediction from a distribution
    of prediction values over the different classes by selecting
    the largest value (value the model is most confident in)

    Params
    ------
    Y_hat : ndarray.float32, (N, D)
        network output, "predictions" or scores on the D classes
    Returns
    -------
    Y_pred : ndarray.int32, (N,)
        the maximal class score, by index, per sample
    """
    Y_pred = np.argmax(Y_hat, axis=-1)
    return Y_pred


def indifferent_scores(scores):
    """ Check whether class scores are all close to eachother

    If class scores are all nearly same, it means model
    is indifferent to class and makes equally distributed
    values on each.

    This means that, even in the case where the model was
    unable to learn, it would still get 1/3 accuracy by default

    This function attempts preserve integrity of predictions
    """
    N, D = scores.shape
    if N > 1:
        mu = np.copy(scores).mean(axis=1, keepdims=True)
        mu = np.broadcast_to(mu, scores.shape)
    else:
        mu = np.full(scores.shape, np.copy(scores).mean())
    return np.allclose(scores, mu, rtol=1e-2, atol=1e-2)



def classification_accuracy(Y_hat, Y_truth, strict=False):
    """ Computes classification accuracy over different classes

    Params
    ------
    Y_pred : ndarray.float32, (N, D)
        raw class "scores" from network output for the D classes
    Y_truth : ndarray.int32, (N,)
        ground truth

    Returns
    -------
    accuracy : float
        the averaged percentage of matching predictions between
        Y_hat and Y_truth
    """
    if indifferent_scores(Y_hat):
        return 0.0

    if not strict:
        # Reduce Y_hat to highest scores
        Y_pred = get_predictions(Y_hat)

        # Take average match
        accuracy = np.mean(Y_pred == Y_truth)

    else:
        # Show raw class score
        Y = to_one_hot(Y_truth)
        scores = np.amax(Y * Y_hat, axis=1)
        accuracy = np.mean(scores)

    return accuracy
