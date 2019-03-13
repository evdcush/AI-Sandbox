""" Utilities for working with datasets

This module serves mostly as a superset of sklearn's toy datasets,
using scikit's default class structure, and adding some additional
utilities such as train/validation/test splitting, normalization,
and batching.

"""
import os
import sys
import code
import sklearn.datasets
import numpy as np

class Dataset:
    """ Wrapper class for toy sklearn.datasets
    Dataset is an interface to sklearn.utils.Bunch,
    the return type of the sklearn toy sets, and provide
    several convenient funcs for splitting and batching
    """
    seed = 0
    load = None  # sklearn.datasets.load_* func
    def __init__(self, split=True): # **kwargs):  # for now, ignore kwargs
        bunch = self.load()
        for key in dir(bunch):
            val = getattr(bunch, key)
            setattr(self, key, val)
        self.shape = self.data.shape

    def desc(self):
        print(self.DESCR)

    def split_dataset(self, num_test, num_val=0, shuffle=True, seed=None):
        """ split dataset in train, test, and optionally validation sets
        Params
        ------
        num_test : int; (0, N)
            number of test samples to split from data

        num_val : int; [0, N - num_test)
            number of validation samples, a value of 0 == no validation set

        shuffle : bool
            shuffle data before split
        """
        N, D = self.shape
        # arg checks
        assert isinstance(num_test, int) and 0 < num_test < N
        assert isinstance(num_val,  int) and 0 <= num_val < N - num_test

        # split data
        # ==========
        # indexing
        indices = np.arange(N)
        if shuffle:
            seed = seed if seed else self.seed
            np.random.seed(seed)
            np.random.shuffle(indices)

        # split sections
        v = [num_val] if num_val else []
        sections = v + [num_test + num_val]
        x_sets = np.split(self.data,   sections, axis=0)[::-1]
        y_sets = np.split(self.target, sections, axis=0)[::-1]
        self.x_train, self.x_test = x_sets[:2]
        self.y_train, self.y_test = y_sets[:2]
        if num_val:
            self.x_validation = x_sets[-1]
            self.y_validation = y_sets[-1]


    def get_batch(self, batch_size):
        """ get training batch x, y """
        if not hasattr(self, 'x_train'): # then dataset was not split
            X, Y = self.data, self.target
        else:
            X, Y = self.x_train, self.y_train
        n = X.shape[0]
        idx = np.random.choice(n, batch_size, replace=False)
        x = np.copy(X[idx])
        y = np.copy(Y[idx])
        return x, y

#-----------------------------------------------------------------------------#
#                                  Datasets                                   #
#-----------------------------------------------------------------------------#

# Regression
# ==========
class Boston(Dataset): # need to omit 'B' col (-2); the transform fn is nonsensical
    load = staticmethod(sklearn.datasets.load_boston)

class Diabetes(Dataset):
    load = staticmethod(sklearn.datasets.load_diabetes)

class Linnerud(Dataset): # multivariate
    load = staticmethod(sklearn.datasets.load_linnerud)

# Classification
# ==============
class Iris(Dataset):
    load = staticmethod(sklearn.datasets.load_iris)

class Wine(Dataset):
    load = staticmethod(sklearn.datasets.load_wine)

class Digits(Dataset):
    load = staticmethod(sklearn.datasets.load_digits)

class BreastCancer(Dataset):
    load = staticmethod(sklearn.datasets.load_breast_cancer)


