import os
import sys
import code
import argparse

import numpy as np


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

def load_dataset(dset='iris'):
    """ Loads a Dataset from data.dataset

    Available datasets:
        classification : iris, wine, digits, breast_cancer
        regression : boston, diabetes, linnerud

    if invalid dset arg passed, iris dataset will be loaded

    """
    assert isinstance(dset, str) and dset
    #=== pathing
    fpath = os.path.abspath(os.path.dirname(__file__))
    dpath = '/'.join(fpath.split('/')[:-1]) + 'data'
    if dpath not in sys.path:
        sys.path.append(dpath)
    #=== import
    from data.dataset import DATASETS
    if dset not in DATASETS:
        return DATASETS.iris()
    return DATASETS[dset]


def to_one_hot(Y, num_classes):
    """ make one-hot encoding for truth labels

    Encodes a 1D vector of integer class labels into
    a sparse binary 2D array where every sample has
    length num_classes, with a 1 at the index of its
    constituent label and zeros elsewhere.
    """
    # Dimensional integrity check on Y
    #  handles both ndim = 0 and ndim > 1 cases
    if Y.ndim != 1:
        Y = np.squeeze(Y)
        assert Y.ndim == 1

    # dims for one-hot
    n = Y.shape[0]
    d = num_classes

    # make one-hot
    one_hot = np.zeros((n, d))
    one_hot[np.arange(n), Y] = 1
    return one_hot.astype(np.int32)


#-----------------------------------------------------------------------------#
#                                   Parser                                    #
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#                                   Logger                                    #
#-----------------------------------------------------------------------------#
