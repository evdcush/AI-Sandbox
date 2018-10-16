""" Utilities for working with datasets

# Dataset class
#--------------
Given that the iris dataset is currently the only available dataset,
the IrisDataset is the most atomic or base dataset to work with.

When more datasets are added, if there is need for a standard interface
or too much boilerplate, a base class may be added.
"""
import os
import sys

import numpy


#------------------------------------------------------------------------------
#                               Constants
#------------------------------------------------------------------------------
# dir paths
# =========
PATH_DATA_DIR = os.path.abspath(sys.path[0])
PATH_IRIS_DIR = PATH_DATA_DIR + '/Iris/'

# file paths
# ==========
PATH_IRIS_FILE  = PATH_IRIS_DIR + 'iris.npy'
PATH_IRIS_TRAIN = PATH_IRIS_DIR + 'iris_train.npy'
PATH_IRIS_TEST  = PATH_IRIS_DIR + 'iris_test.npy'

# Seeds
# =====
RNG_SEED_IRIS = 9959  # for even class distribution in train/test


#==============================================================================
#                               Datasets
#==============================================================================


#------------------------------------------------------------------------------
#  Functions
#------------------------------------------------------------------------------
def load_dataset(dpath): # assumes npy serialization
    assert os.path.exists(dpath) and dpath[-4:] == '.npy'
    return numpy.load(dpath)



class IrisDataset:
    """ Maintains all info and utils related to the Iris dataset
    Attributes are simply extracted from IRIS
    """
    path = PATH_IRIS_DIR
    feature_split_idx = 4
    classes = {0 : 'Iris-setosa', 1 : 'Iris-versicolor', 2 : 'Iris-virginica'}
    def __init__(self, X=None, split_size=.8, seed=RNG_SEED_IRIS):
        self.split_size = split_size
        self.split_seed = seed
        self.process_input_data(X)

    # Load datasets
    #-----------------------------
    @staticmethod
    def load_full_dataset_iris():
        return load_dataset(PATH_IRIS_FILE)

    @staticmethod
    def load_train_eval_dataset_iris():
        return load_dataset(PATH_IRIS_TRAIN), load_dataset(PATH_IRIS_TEST)

    def process_input_data(self, X_in):
        if X_in is not None:
            # sometimes resplit original
            x_train, x_test = self.split_dataset(X_in)
        else:
            # otherwise just load train/test sets
            x_train = load_dataset(PATH_IRIS_TRAIN)
            x_test  = load_dataset(PATH_IRIS_TEST)
        self.x_train = x_train
        self.x_test  = x_test


    # Split into Train/Test sets
    #-----------------------------
    def split_dataset(self, X):
        """ Splits a dataset X into training and testing sets.
        Note: RNG_SEED_IRIS value was chosen because it evenly distributes
              the number of Iris classes between train and test sets,
              at split_size=.8
              (eg, for both train and test, the number of samples for
               num_setosa == num_versicolor == num_virginica)
        Params
        ------
        X : ndarray
            Primary dataset. split indices are instantiated
            on the first axis of X.shape, which is assumed to be
            the number of samples.
        split_size : float
            split percentage where N * split_size total samples are used
            for training, and the remaining for testing.
        split_seed : int
            used for seeding numpy's random module for consistent splits
            on X to produce training and testing sets

        Returns
        -------
        X_train, X_test : ndarray,
            train and test shapes, respectively are
            (N * split_size, D), (N * (1 - split_size), D)
            eg, for .8 split on 150-sample set:
                (120, D), (30, D)
        """
        # Get split vars
        #--------------------------
        split_size = self.split_size
        split_seed = self.split_seed
        assert 0.0 < split_size < 1.0

        # Spliting index
        #--------------------------
        N = X.shape[0]
        split_idx = [int(N * split_size)]

        # Seed, permute, split
        #--------------------------
        numpy.random.seed(split_seed)
        X_shuffled = numpy.random.permutation(X)
        X_train, X_test = numpy.split(X_shuffled, split_idx)
        return X_train, X_test

    # Batching on dataset
    #-----------------------------
    def get_batch(self, step, batch_size=1, test=False):
        """ Batches samples from dataset X
        ASSUMED: batch_size is a factor of the number of samples in X

        #==== Training
        - Batches are selected randomly *and without replacement* wrt
          the previous batches.
          This is done based on current step:
            When current step has become a multiple of the number
            of samples in X, the samples positions in X are
            randomly shuffled.
        #==== Testing
        - Batches are selected sequentially, with no randomness

        Params
        ------
        X : ndarray, (features & labels)
            X is an array containing both the features (x) and the labels
            or classes (y). split(X) ---> x, y
        step : int
            current training iteration
        batch_size : int
            number of samples in minibatch
        test : bool
            whether X is a test set. If test, no random ops.

        Returns
        -------
        x : ndarray, (batch_size, ...)
            batch data features
        y : ndarray.int32, (batch_size,)
            batch ground truth labels (or class)
        """
        assert batch_size > 0 and isinstance(batch_size, int)
        X = self.X_train if not test else self.X_test

        # Get dimensions and indices
        #-------------------------------
        N = X.shape[0]
        b = batch_size if batch_size <= N else batch_size % N
        #==== Batching indices
        i = (b * step) % N  # start index [inclusive]
        j = i + b           # end   index (exclusive)

        # Check if need for reshuffle
        #-------------------------------
        if i == 0 and not test: # test never shuffled
            numpy.random.shuffle(X)

        # Get batch and split
        #-------------------------------
        batch = numpy.copy(X[i:j])
        x, y = numpy.split(batch, [self.feature_split_idx], axis=1)
        y = y.astype(numpy.int32)
        #==== Squeeze Y to 1D
        y = numpy.squeeze(y) if b > 1 else y[:,0]
        return x, y
