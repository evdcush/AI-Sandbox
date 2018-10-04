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

"""

import os
import sys
import time
import code
import shutil
import argparse
import subprocess
from functools import wraps

import numpy as np

import layers
import functions
import optimizers

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
def sub_wget_data(url, fname, out_dir):
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
RNG_SEED_DATA   = 98765 # for shuffling data
RNG_SEED_PARAMS = 12345 # seeding parameter inits

# Hyperparameters
LEARNING_RATE = 0.01

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
# Dataset utils
#==============================================================================


# Loading dataset from disk
#------------------------------------------------------------------------------
def load_dataset(path):
    """ Loads dataset located at path """
    dataset = np.load(path)
    return dataset

def load_iris(path=IRIS_DATA_PATH):
    """ Load full Iris dataset """
    iris_dataset = load_dataset(path)
    return iris_dataset


# Split into Train/Test sets
#------------------------------------------------------------------------------
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



#==============================================================================
# Training utils
#==============================================================================


# Batching
#------------------------------------------------------------------------------
def get_batch(X, step, batch_size=1, test=False, split_idx=-1):
    """ Batches samples from dataset X

    ASSUMED: batch_size is a factor of the number of samples in X

    #==== Training
    - Batches are selected randomly *and without replacement* wrt
      the previous batches. This is done based on current step:
      - When current step has become a multiple of the number
        of samples in X, the samples positions in X are
        randomly shuffled.

    #==== Testing
    - Batches are selected in order, without any stochasticity.
    - Batch size is flexible with testing, though the number
      should still be a factor of the number of samples in X
      Depending on your memory constraints:
      - You can send the entire test set to your model
        if you select a batch_size = number of samples
      - Or you can simply feed the model one sample
        (batch_size=1) at a time

    Params
    ------
    X : ndarray, (N,...,K)
        Full dataset (training or testing)
    batch_size : int
        number of samples in minibatch
    step : int
        current training iteration
    split_idx : int
        the index upon which X is split into features and labels

    Returns
    -------
    x : ndarray, (batch_size, ...)
        batch data features
    y : ndarray.int32, (batch_size,)
        batch ground truth labels (or class)
    """
    assert batch_size > 0 and isinstance(batch_size, int)
    # Dims
    N = X.shape[0]
    b = batch_size if batch_size <= N else batch_size % N

    # Subset indices
    i = (b * step) % N  # start index [inclusive]
    j = i + b           # end   index (exclusive)

    # Check if we need to reshuffle (train only)
    if i == 0 and not test:
        np.random.shuffle(X)

    # Batch and split data
    batch = np.copy(X[i:j])
    x, y = np.split(batch, [split_idx], axis=1)
    y = y.astype(np.int32)

    # Squeeze Y to 1D
    y = np.squeeze(y) if b > 1 else y[:,0]
    return x, y


def to_one_hot(Y, num_classes=len(IRIS['classes'])):
    """ make one-hot encoding for truth labels

    Encodes a 1D vector of integer class labels into
    a sparse binary 2D array where every sample has
    length num_classes, with a 1 at the index of its
    constituent label and zeros elsewhere.

    Example
    -------
    Y = [3, 1, 1, 0, 2, 3, 2, 2, 2, 1]
    Y.shape = (10,)
    num_classes = 4
    one_hot shape == (10, 4)

    to_one_hot(Y) = [[0, 0, 0, 1], # 3
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


#==============================================================================
#------------------------------------------------------------------------------
#                              Parser
#------------------------------------------------------------------------------
#==============================================================================

class Parser:
    """ Wrapper for argparse parser

    In addition to parsing STDIN args, Parser is a significant
    part of setup and configuration for training, as it specifies the
    majority of settings to use (through `default`)

    # NB:
    #----------
    If you would like to use these values the same as in train.py,
    but in a different environment, such as a notebook or perhaps IDE,
    you can still get the values from by simply calling the parse_args()
    method with a Parser instance.
    eg:
    echo_parser = utils.Parser()
    config = echo_parser.parse_args()

    """
    P = argparse.ArgumentParser()
    # argparse does not like type=bool; this is a workaround
    p_bool = {'type':int, 'choices':[0,1]}

    def __init__(self):
        adg = self.P.add_argument
        chans = [4, 64, len(IRIS['classes'])]
        # ==== Data variables
        adg('--data_path',  '-p', type=str, default=DATA_PATH_ROOT,)
        adg('--seed',       '-s', type=int, default=RNG_SEED_PARAMS,)
        adg('--model_name', '-m', type=str, default=MODEL_NAME_BASE,)
        adg('--name_suffix','-n', type=str, default='')

        # ==== Model parameter variables
        #adg('--connection', '-k', type=str, default='dense') # only is dense..
        adg('--activation', '-a', type=str, default='sigmoid')
        adg('--dropout',    '-d', **self.p_bool, default=0)
        adg('--optimizer',  '-o', type=str, default='sgd')
        adg('--objective',  '-j', type=str, default='logistic_cross_entropy')
        adg('--channels',   '-c', type=int, default=chans, nargs='+')
        adg('--learn_rate', '-l', type=float, default=LEARNING_RATE)

        # ==== Training variables
        adg('--num_iters',  '-i', type=int, default=2000)
        adg('--batch_size', '-b', type=int, default=6)
        #adg('--restore',    '-r', **self.p_bool) # later
        #adg('--checkpoint', '-p', type=int, default=100) # later
        self.parse_args()

    def parse_args(self):
        parsed = AttrDict(vars(self.P.parse_args()))
        #parsed.restore = bool(parsed.restore)
        self.args = self.interpret_args(parsed)
        return parsed

    def interpret_args(self, parsed):
        # Dropout
        #-----------------------------
        parsed.dropout = parsed.dropout == 1

        # Activation
        #----------------------------
        pact = parsed.activation
        #==== Check if parameterized activation (Swish)
        activation = layers.PARAMETRIC_FUNCTIONS.get(pact, None)
        if activation is None:
            #==== Check if valid activation Function
            activation = functions.ACTIVATIONS.get(pact, None)
            if activation is None:
                # if None again, parsed activation arg is undefined in domain
                raise NotImplementedError('{} is undefined'.format(pact))
        # assign proper activation class
        parsed.activation = activation

        # Optimizer
        #----------------------------
        popt = parsed.optimizer
        opt = optimizers.get_optimizer(popt) # raises ValueError if not defined
        parsed.optimizer = opt

        # Objective
        #----------------------------
        pobj = parsed.objective
        objective = functions.OBJECTIVES.get(pobj, None)
        if objective is None: # then objective not in functions
            raise NotImplementedError('{} is undefined'.format(pobj))
        parsed.objective = objective
        return parsed

    def print_args(self):
        print('SESSION CONFIG\n{}'.format('='*79))
        margin = len(max(self.args, key=len)) + 1
        for k,v in self.args.items():
            print('{:>{margin}}: {}'.format(k,v, margin=margin))
        print('\n')


#==============================================================================
#------------------------------------------------------------------------------
#                      Training stats and other info
#------------------------------------------------------------------------------
#==============================================================================

# Status and results
#------------------------------------------------------------------------------
class SessionStatus:
    """ Provides information to user on current status and results of model

    In addition to helpful status updates, SessionStatus maintains
    the loss history collections for training or validation.
    """
    div1 = '#' + ('=' * 78)
    div2 = '#' + ('-' * 30)
    def __init__(self, model, opt, obj, num_iters, num_test):
        self.model_name = str(model)
        self.opt_name = str(opt)
        self.obj_name = str(obj)
        self.num_iters = num_iters
        self.num_test = num_test
        self.init_loss_history()
        self._get_network_arch(model)
        self.status_call_count = 0

    def init_loss_history(self):
        """ Initialize loss history containers for training and testing """
        # Each entry of form: (error, accuracy)
        self.train_history = np.zeros((self.num_iters, 2)).astype(np.float32)
        self.test_history  = np.zeros((self.num_test, 2)).astype(np.float32)

    def _get_network_arch(self, model):
        """ Formatted string of network architecture """

        # Header
        #-----------------
        arch = '{}\n  Layers: \n'.format(str(model)) # 'NeuralNetwork'

        # Layer body
        #-----------------
        line_layer    = '    {:>2} : {:<5} {}\n' # ParametricLayer, eg 'Dense'
        line_function = '          : {}\n'       # Function, eg 'Tanh'

        # Traverse layers
        #-----------------
        for unit in model.layers:
            unit_name   = str(unit)
            unit_class  = unit.__class__
            unit_module = unit.__module__

            #==== Parametric unit
            if unit_module == 'layers':
                kdims = unit.kdims
                layer_name, layer_num = unit_name.split('-')
                #==== Function case
                if unit_class in layers.PARAMETRIC_FUNCTIONS.values():
                    # non-connection functions are not considered discrete
                    # layers within the network like Dense
                    layer_num = ' '
                arch += line_layer.format(layer_num, layer_name, kdims)
            #==== function.Function
            else:
                arch += line_function.format(unit_name)
        self.network_arch = arch

    def print_results(self, train=True, t=None):
        d2 = self.div2
        num_tr = self.num_iters
        num_test = self.num_test
        header = '\n# {} results, {} {}\n' + d2
        # Format header based on training or test
        if train:
            header = header.format('Training', num_tr, 'iterations')
            f20 = int(num_tr * .8)
            loss_hist = self.train_history[f20:]
        else:
            header = header.format('Test', num_test, 'samples')
            loss_hist = self.test_history

        # Get stats on loss hist
        avg = np.mean(  np.copy(loss_hist), axis=0)
        q50 = np.median(np.copy(loss_hist), axis=0)

        # Print results
        print(header)
        if t is not None:
            print('Elapsed time: {}'.format(t))
        print('            Error  |  Accuracy')
        print('* Average: {:.5f} | {:.5f}'.format(avg[0], avg[1]))
        print('*  Median: {:.5f} | {:.5f}'.format(q50[0], q50[1]))
        print(d2)

    def summarize_model(self, train_results=False, test_results=False):
        arch = self.network_arch
        opt = self.opt_name
        obj = self.obj_name
        d1  = self.div1 #=====
        d2  = self.div2 #-----
        print(d1)
        print('# Model Summary: \n')
        print(arch)
        print('- OPTIMIZER : {}'.format(opt))
        print('- OBJECTIVE : {}'.format(obj))
        if train_results:
            self.print_results(train=True)
        if test_results:
            self.print_results(train=False)
        print(d1)

    def get_loss_history(self):
        print('Returning: train_history, test_history')
        return self.train_history, self.test_history

    def print_status(self, step, err, acc):
        title = '{:<5}:  {:^7}   |  {:^7}'.format('STEP', 'ERROR', 'ACCURACY')
        body  = '{:>5}:  {:.5f}   |   {:.4f}'
        i = step + 1
        e, a = float(err), float(acc)
        status = body.format(i, e, a)
        if self.status_call_count == 0:
            #d = '-' * len(title)
            #print('\n\n{}\n{}'.format(title, d))
            print('\n\n{}'.format(title))
            #print('{:<5}: {:^7}  |  {:^7}'.format('Step', 'Error', 'Accuracy'))
        #status = '{:>5}: {:.5f}  |  {:.4f}'.format(i, e, a)
        print(status)
        self.status_call_count += 1

    @staticmethod
    def print_status_inline(step, err, acc):
        """ WIP """
        return
        i = step + 1
        e, a = float(err), float(acc)
        sys.stdout.write('\r')
        sys.stdout.write('{:>5}: {:.5f}  |  {:.4f}'.format(i, e, a))
        sys.stdout.flush()

    def __call__(self, step, err, acc,
                 train=True, pstatus=True, pfreq=100):
        loss_hist = self.train_history if train else self.test_history
        loss_hist[step] = err, acc
        if not train or (pstatus and ((step+1) % pfreq == 0)):
            if not train and step == 0:
                self.status_call_count = 0
            self.print_status(step, err, acc)


# Classification eval
#------------------------------------------------------------------------------
def get_predictions(Y_hat):
    """ Select the highest valued class labels in prediction from
    network output distribution

    Y_hat is assumed to of shape (N, D), where
      N is the number of independent samples in the prediction, and
      D is the number of classes

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
