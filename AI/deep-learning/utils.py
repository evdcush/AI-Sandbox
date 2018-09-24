""" All essential utilities related to data and setup

AVAILABLE DATASETS :
  Iris

Module components
=================
# Dataset acquisition
#--------------------
 - Small (file-size) datasets can be stored in the project 'data' directory,
   but larger datasets will need to be sourced by user
 - Some basic utilties (like `sub_wget_data`) can help acquire a
   dataset from source

# Dataset management
#-------------------
- Dataset : base class for managing a dataset
    How data is read, processed, serialized, split, normalized, etc is
    most likely going to be per-dataset, per-task process.
    Dataset is a simple class that can be extended for the unique
    constraints of your dataset.
    - It has the following functionalities :
      - loads dataset or a subset of a dataset
      - shuffles samples within dataset
      - split train/test sets (if not already split)

# Data processing
#----------------

# Trainer

# CV
"""

import os
import sys
import code
import shutil
import subprocess
from functools import wraps

import numpy as np


#==============================================================================
#------------------------------------------------------------------------------
#                        Task-invariant utils
#------------------------------------------------------------------------------
#==============================================================================


#------------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------------

class AttrDict(dict):
    """ simply a dict accessed/mutated by attribute instead of index
    Warning: cannot be pickled like normal dict/object
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

# Getters
# ========================================
def sub_wget_data(url, fname, out_dir=DATA_DIR):
    """ Gets desired data from a url using wget """
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    try:
        subprocess.check_output(["wget", "-T", "120", url])
        shutil.move(fname, out_dir)
    except:
        print('Error in retrieving dataset')


# Decorators
# ========================================
def TODO(f):
    """ Serves as a convenient, clear flag for developers and insures
        wrapee func will not be called
    """
    @wraps(f)
    def not_finished(*args, **kwargs):
        print('\n  {} IS INCOMPLETE'.format(f.__name__))
    return not_finished

def NOTIMPLEMENTED(f):
    """ Like TODO, but for functions in a class
        raises error when wrappee is called
    """
    @wraps(f)
    def not_implemented(*args, **kwargs):
        func_class = args[0]
        f_class_name = func_class.get_class_name()
        f_name = f.__name__
        msg = '\n  Class: {}, function: {} has not been implemented!\n'
        print(msg.format(f_class_name, f_name))
        raise NotImplementedError
    return not_implemented


def INSPECT(f):
    """ interupts computation and opens interactive python shell
    for user to evaluate wrapee function input/output

    NB: simply placing the `code.interact` line directly in the code
        location you would like to evaluate/debug is often more
        effective
    """
    @wraps(f)
    def inspector(*args, **kwargs):
        print('\n Inspecting function: <{}>'.format(f.__name__))
        x = args
        y = kwargs
        z = f(*args, **kwargs)
        code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        return z
    return inspector


#==============================================================================
#------------------------------------------------------------------------------
#                               Settings
#------------------------------------------------------------------------------
#==============================================================================


# Pathing
#------------------------------------------------------------------------------

# Data pathing
# ========================================
DATA_PATH_ROOT = './data/'
IRIS_DATA_PATH = DATA_PATH_ROOT + 'Iris/iris.npy'

# Model pathing
# ========================================
# Model directory paths
MODEL_NAME_BASE = '{}_{}' # {data_label}_{layer_type}
MODEL_SAVE_PATH_ROOT = './Models/{}/' # Models/model_name/...
MODEL_PARAMS_PATH  = MODEL_SAVE_PATH_ROOT + 'Parameters/'
MODEL_RESULTS_PATH = MODEL_SAVE_PATH_ROOT + 'Results/'
MODEL_CHECKPT_PATH = MODEL_PARAMS_PATH    + 'checkpoints/' # save checkpoints
PYFILE_SAVE_PATH   = MODEL_SAVE_PATH_ROOT + 'original_files/' # overwrites

# Model results write path
RESULTS_BASE_NAME = 'X_{}'
PREDICTION_FNAME  = RESULTS_BASE_NAME.format('pred')
GROUNDTRUTH_FNAME = RESULTS_BASE_NAME.format('true')

# Model loss statistics
LOSS_PLOT_FNAME  = RESULTS_BASE_NAME.format('error') # visualization
LOSS_TRAIN_FNAME = RESULTS_BASE_NAME.format('train_error')
LOSS_TEST_FNAME  = RESULTS_BASE_NAME.format('test_error')


# Parameters
#------------------------------------------------------------------------------

# Random seeds
# ========================================
RNG_SEED_DATA   = 98765 # for shuffling data
RNG_SEED_PARAMS = 12345 # seeding parameter inits



#==============================================================================
#------------------------------------------------------------------------------
#                              Dataset
#------------------------------------------------------------------------------
#==============================================================================

# Dataset features format
# ========================================
"""
MY_DATASET_LABEL = {'label' : str
                        name of dataset (ideally short)

                    'path'  : str
                        path to your dataset file(s)

                    'num_samples' : int
                        the number of samples within the dataset
                        (note, split train/test not yet supported)

                    'features_per_sample' : int
                        the number of features per sample
                        If images, this would be H*W*color_channels

                    'feature_split_idx' : int, None
                        specifies the splitting idx between dataset
                        features and class labels
                        None, if data is structured differently,
                        or no ordinality

                    'classes' : dict
                        however classes are represented or
                        encoded for your task
                    }
"""

# Datasets
# ========================================
""" NOTE:
Per-dataset specs like this will likely be abstracted to their
respective dataset file directories
"""
IRIS = {'label' : 'iris',
        'path'  : DATA_PATH_ROOT.format('Iris/iris.npy'),
        'num_samples' : 150,
        'features_per_sample' : 4,
        'feature_split_idx' : 3,
        'classes' : {0 : 'Iris-setosa',
                     1 : 'Iris-versicolor',
                     2 : 'Iris-virginica'},
        }


#==============================================================================
# Dataset class
#==============================================================================

@TODO
class Dataset:
    """ Contains generic attributes and functions for a dataset """
    data_dir = DATA_PATH_ROOT # constant defined by user

    def __init__(self, label, fname, data_subdirs='', url=None, **kwargs):
        """ Instance vars here are limited to what would likely be used
            by any dataset.
        Potentially many more instance vars would be set through
        kwargs

        Parameters
        ----------
        label : str
            name of the dataset, eg. "Iris"
        fname : str
            name of the dataset file
        data_subdirs : str
            additional subdirectories within root data directory
        url : str
            the url to the source dataset online

        """
        #self.label = label
        #self.fname = fname
        #self.data_path = self.data_dir + data_subdirs + fname
        #self.url = url
        ## init other possible class instance vars
        #for attribute, value in kwargs.items():
        #    setattr(self, attribute, value)
        pass

    def download_dataset_from_source(self,):
        pass


#==============================================================================
# Data processing functions
#==============================================================================

#------------------------------------------------------------------------------
# Split into Train/Test sets
#------------------------------------------------------------------------------
@TODO
def load_dataset(data_path, XY_split=False, train_test_split=False):
    """   """
    pass

def split_dataset(X, Y=None, split_size=.8, seed=RNG_SEED_DATA):
    """ Splits a dataset (or array) into training and testing sets.
    (this is only called if train and test sets have NOT been
    serialized as separate files)

    Indices are permuted rather than shuffling X. This guarantees
    that we get consistent train/test splits in the case that
    the test set has not been serialized.

    Params
    ------
    X : ndarray
        primary features dataset, split indices are instantiated
        on the first axis of X.shape, which is assumed to be
        the number of samples
    Y : ndarray, None
        associated ground-truth vector or array containing the
        classes for each feature in X. Assumed to be another
        dimension to X unless specified
    split_size : float
        split percentage where N * split_size total samples are used
        for training, and the remaining for testing.
        NB: it's best to select a split size that evenly splits your
            data.
    seed : int
        used for seeding numpy's random module for consistent splits
        on X to produce training and testing sets

    Returns
    -------
    X_train, X_test : ndarray,
        train and test shapes, respectively are
        (N * split_size, D), (N * (1 - split_size), D)

    """
    assert 0.0 < split_size < 1.0

    # Get shape and indices
    N = X.shape[0]
    indices = np.arange(N)
    split_idx = [int(N * split_size)]

    # Seed, permute, and split
    np.random.seed(seed)
    rand_idx = np.random.permutation(indices)
    X_shuffled = X[rand_idx]
    X_train, X_test = np.split(X_shuffled, split_idx)
    return X_train, X_test


def one_hot(Y):
    """ make one-hot encoding for truth labels

    Assumptions
    -----------
    - Y is a column vector of ints, representing classes,
      with the same length as the number of samples

    Example
    -------
    Y = [3, 1, 1, 0, 2, 3, 2, 2, 2, 1]
    Y.shape = (10,)

    one_hot(Y) = [[0, 0, 0, 1], # 3
                  [0, 1, 0, 0], # 1
                  [0, 1, 0, 0], # 1
                  [1, 0, 0, 0], # 0
                  [0, 0, 1, 0], # 2
                  [0, 0, 0, 1], # 3
                  [0, 1, 0, 0], # 2
                  [0, 1, 0, 0], # 2
                  [0, 1, 0, 0], # 2
                  [1, 0, 0, 0], # 1
                 ]
    """
    # dims for one-hot
    n = Y.shape[0]
    d = np.max(Y) + 1

    # make one-hot
    Y1h = np.zeros((n, d))
    Y1h[np.arange(n), Y] = 1
    return Y1h


#==============================================================================
# Data preprocessing for training/test
#==============================================================================

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

def get_training_batch(X, batch_size, step, split_idx=-1):
    """ Get training batch

    Batches are selected randomly, and without replacement wrt
    the previous batches.

    This is done based on current step. When current
    step has become a multiple of the number of samples in X,
    X is reshuffled at random.

    ASSUMED: batch_size is a factor of the number of samples in X

    Params
    ------
    X : ndarray, (N,...,K)
        Full training dataset
    batch_size : int
        number of samples in minibatch
    step : int
        current training iteration
    split_idx : int
        the index upon which X is split into features and labels

    Returns
    -------
    x : ndarray, (batch_size, ...)
        the training minibatch of features
    y : ndarray(int), (batch_size,)
        the training minibatch ground truth labels
    """
    # Dims
    N = X.shape[0]
    b = batch_size

    # Subset indices
    i = (b * step) % N
    j = i + b

    # Check if we need to reshuffle
    if step != 0 and i == 0:
        np.random.shuffle(X)

    # Batch and split data
    batch = np.copy(X[i:j])
    x, y = np.split(batch, [split_idx], axis=1)

    # Format y from float --> int
    y = y.astype(np.int32)

    return x, y



#==============================================================================
#------------------------------------------------------------------------------
#                              Parser
#------------------------------------------------------------------------------
#==============================================================================

class Parser:
    """ Wrapper for argparse parser
    """
    P = argparse.ArgumentParser()
    # argparse does not like type=bool; this is a workaround
    p_bool = {'type':int, 'default':0, 'choices':[0,1]}

    def __init__(self):
        adg = self.P.add_argument
        # ==== Data variables
        adg('--data_path',  '-d', type=str, default=DATA_PATH_ROOT,)
        adg('--seed',       '-s', type=int, default=RNG_SEED_PARAMS,)
        adg('--model_name', '-m', type=str, default=MODEL_NAME_BASE,)
        adg('--name_suffix','-n', type=str, default='')

        # ==== Model parameter variables
        adg('--block_op',   '-o', type=str, default='dense')
        adg('--block_act',  '-a', type=str, default='sigmoid')
        adg('--channels',   '-c', type=int, default=[4, 32, 1], nargs='+')
        adg('--learn_rate', '-l', type=float, default=LEARNING_RATE)

        # ==== Training variables
        adg('--num_iters',  '-i', type=int, default=500)
        adg('--batch_size', '-b', type=int, default=6)
        adg('--restore',    '-r', **self.p_bool)
        adg('--checkpoint', '-h', type=int, default=50)
        self.parse_args()

    def parse_args(self):
        parsed = AttrDict(vars(self.P.parse_args()))
        parsed.restore = parsed.restore == 1
        self.args = parsed
        return parsed

    def print_args(self):
        print('SESSION CONFIG\n{}'.format('='*79))
        margin = len(max(self.args, key=len)) + 1
        for k,v in self.args.items():
            print('{:>{margin}}: {}'.format(k,v, margin=margin))


#==============================================================================
#------------------------------------------------------------------------------
#                              Trainer
#------------------------------------------------------------------------------
#==============================================================================


@TODO
class Trainer:
    pass
