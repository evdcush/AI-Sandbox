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
from matplotlib import pyplot as plt

import layers
import network
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
    __delattr__ = dict.__delitem__


#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
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
DATA_DIR = '../data/'
#==== Iris dataset
IRIS_DIR = f'{DATA_DIR}Iris/'
IRIS_DATASET_PATH = f'{IRIS_DIR}iris.npy'
#---- Iris train/test files
IRIS_TRAIN = f'{IRIS_DIR}iris_train.npy'
IRIS_TEST  = f'{IRIS_DIR}iris_test.npy'




####################### Not yet supported #####################################
"""
# Model pathing
# ========================================
#('m', 'model_name', MODEL_NAME_BASE, 'name to which model params saved'),
#('n', 'name_suff',  '', 'label or tag suffixing model name'),
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
"""

# Dataset features
#------------------------------------------------------------------------------
# Iris dataset
# ========================================
IRIS = {'label' : 'iris',
        'path'  : IRIS_DATASET_PATH,
        'num_samples' : 150,
        'features_per_sample' : 4,
        'feature_split_idx' : 4,
        'classes' : {0 : 'Iris-setosa',
                     1 : 'Iris-versicolor',
                     2 : 'Iris-virginica'},
        }

# Parameters
#------------------------------------------------------------------------------
# Random seeds
RNG_SEED_DATA   = 9959  # for shuffling data into train/test evenly
RNG_SEED_PARAMS = 12345 # seeding parameter inits

# Hyperparameters
LEARNING_RATE = 0.01 # unused, opt defaults are good
CHANNELS = [IRIS['features_per_sample'], 64, len(IRIS['classes'])]



# Model/session configuration
#------------------------------------------------------------------------------

DEFAULT_CONFIGURATION = [
# ==== Data variables
('p', 'data_path',  DATA_DIR,  'relative path to dataset file'),
('s', 'seed',       RNG_SEED_PARAMS, 'int used for seeding random state'),
# ==== Model variables
('a', 'activation', 'sigmoid', '(lower-cased) activation func name'),
('d', 'dropout',    False, 'Whether to use dropout'),
('o', 'optimizer',  'sgd', '(lower-cased) optimizer name'),
('j', 'objective',  'softmax_cross_entropy', '(lower-cased) loss func name'),
('c', 'channels',   CHANNELS, 'list(int) layer sizes; more channels-->deeper'),
('l', 'learn_rate', LEARNING_RATE, 'optimizer learning rate'),
# ==== Training/session variables
('i', 'num_iters',  2000, 'number of training iterations'),
('b', 'batch_size', 6, 'training batch size: how many samples per iter'),
('v', 'verbose',    False, 'print model error while actively training'),
('_', 'dummy', False, 'dummy var workaround for notebook error'),
]


#==============================================================================
#------------------------------------------------------------------------------
#                                Data
#------------------------------------------------------------------------------
#==============================================================================

# One-hot encoding
#------------------------------------------------------------------------------
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

# Loading dataset from disk
#------------------------------------------------------------------------------
def load_dataset(path):
    """ Loads dataset located at path """
    dataset = np.load(path)
    return dataset

#==============================================================================
# Dataset utils
#==============================================================================

class IrisDataset:
    """ Maintains all info and utils related to the Iris dataset
    Attributes are simply extracted from IRIS
    """
    label = 'iris'
    path = IRIS_DATASET_PATH
    features_per_sample = 4
    feature_split_idx = 4
    classes = {0 : 'Iris-setosa', 1 : 'Iris-versicolor', 2 : 'Iris-virginica'}
    def __init__(self, X=None, split_size=.8, seed=RNG_SEED_DATA):
        X = X if X is not None else load_dataset(self.path)
        self.X_train, self.X_test = self.split_dataset(X, split_size, seed)


    # Split into Train/Test sets
    #-----------------------------
    @staticmethod
    def split_dataset(X, split_size=.8, seed=RNG_SEED_DATA):
        """ Splits a dataset X into training and testing sets.

        Note: RNG_SEED_DATA value was chosen because it evenly distributes
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
        seed : int
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
        assert 0.0 < split_size < 1.0
        # Spliting index
        #---------------------------
        N = X.shape[0]
        split_idx = [int(N * split_size)]

        # Seed, permute, split
        #--------------------------
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        np.random.seed(seed)
        X_shuffled = np.random.permutation(X)
        X_train, X_test = np.split(X_shuffled, split_idx)
        return X_train, X_test

    # Batching on dataset
    #-----------------------------
    @staticmethod
    def get_batch(X, step, batch_size=1, test=False, feature_split_idx=4):
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
            np.random.shuffle(X)

        # Get batch and split
        #-------------------------------
        batch = np.copy(X[i:j])
        x, y = np.split(batch, [feature_split_idx], axis=1)
        y = y.astype(np.int32)
        #==== Squeeze Y to 1D
        y = np.squeeze(y) if b > 1 else y[:,0]
        return x, y

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
    """
    P = argparse.ArgumentParser()
    def __init__(self):
        self.add_args(DEFAULT_CONFIGURATION)

    def add_args(self, defaults_config):
        """ formats default config to argparse readable form
        Argparse has a well defined standard library that has a
          lot of powerful utilities, but it is rather complex
          and there are certain constraints on the
          formatting of arguments added to the parser
        Example
        -------
        IN:  ('o', 'optimizer',  'sgd', '(lower-cased) optimizer name')

        OUT: ('-o', '--optimizer', type=str, default='sgd',
                    help='(lower-cased) optimizer name')

        The two current edge cases for parser args:
          - bools : require action='store_true' flag
          - lists : require nargs='+' flag
        """
        for k, name, d, h in defaults_config:
            #==== args: name
            args = ('-{}'.format(k), '--{}'.format(name))
            #==== kwargs: type, default, help, **
            if isinstance(d, bool):
                kw = {'action': 'store_true'}
            elif isinstance(d, list): # only have int lists
                kw = {'type':int, 'default':d, 'nargs':'+'}
            else:
                dtype = type(d)
                kw = {'type': dtype, 'default':d}
            kwargs = {**kw, **{'help': h}}
            #==== Add args to parser
            self.P.add_argument(*args, **kwargs)

    def parse_args_from_jupyter_notebook(self, inputs=None):
        """ workaround for parse_args call in notebook, ipython fine though"""
        inputs = '-_' if inputs is None else inputs
        return self.parse_args(inputs)

    def parse_args(self, inputs=None):
        """ parse_args, by default, reads from STDIN
        but can be directly called with the same format args
            eg: parser.parse_args('-i 3000 -a swish -b 10'.split(' '))
            or  parser.parse_args(['-i', '3000', '-a', 'swish, '-b', '10'])
        """
        if inputs is not None:
            ipt = inputs.split(' ') if isinstance(inputs, str) else inputs
            parsed = AttrDict(vars(self.P.parse_args(ipt)))
        else:
            parsed = AttrDict(vars(self.P.parse_args()))
        self.args = self.interpret_args(parsed)
        return parsed

    def interpret_args(self, parsed):
        # Remove jupyter-hack dummy var
        #----------------------------
        del parsed.dummy

        # Integrity input/output channels
        #----------------------------
        # input/output MUST adhere to dataset dims
        parsed.channels[0]  = CHANNELS[0]
        parsed.channels[-1] = CHANNELS[-1]

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
        # strings
        #-----------------
        arch = '{}\n  Layers: \n'.format(str(model)) # 'NeuralNetwork'
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
            #f20 = int(num_tr * .8)
            f20 = int(num_tr * .4)
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
        #print('Returning: train_history, test_history')
        return self.train_history, self.test_history

    def print_status(self, step, err, acc):
        title = '{:<5}:  {:^7}   |  {:^7}'.format('STEP', 'ERROR', 'ACCURACY')
        body  = '{:>5}:  {:.5f}   |   {:.4f}'
        i = step + 1
        e, a = float(err), float(acc)
        status = body.format(i, e, a)
        #==== Only print title on first call
        if self.status_call_count == 0:
            print('\n\n{}'.format(title))
        print(status)
        self.status_call_count += 1

    def plot_results(self, train=True, test=False):
        # Get plot info and subplots
        #------------------------------
        nplots = train + test # either 1 or 2
        assert nplots in [1,2]
        fig, axes = plt.subplots(nplots, 2)


        # Train and Test
        if nplots == 2:
            x = self.train_history
            y = self.test_history
            #==== Training plot
            axes[0, 0].plot(x[:,0], label=self.obj_name + ' Error')
            axes[0, 1].plot(x[:,1], label='Accuracy')
            #==== Test histogram
            axes[1, 0].hist(y[:,0], label=self.obj_name + ' Error')
            axes[1, 1].hist(y[:,1], label='Accuracy')

        # Train or Test
        else:
            if train:
                x = self.train_history
                num_samp = x.shape[0]

                # Curve smoothing
                #------------------------
                err_min, acc_min = x.min(axis=0)
                err_max, acc_max = x.max(axis=0)
                lsp_err = np.linspace(err_min, err_max, num_samp)
                lsp_acc = np.linspace(acc_min, acc_max, num_samp)
                #==== fit polynomial
                poly = np.poly1d(np.polyfit(np.arange(num_samp), x, 3))

                #====
            #   _____     ___     ___      ___     #
            #  |_   _|   / _ \   |   \    / _ \    #
            #    | |    | (_) |  | |) |  | (_) |   #
            #    |_|     \___/   |___/    \___/    #

            #y = self.test_history
            #fig, (train_ax, test_ax) = plt.subplots(2, 2)
            fig, (eax, aax) = plt.subplots(1,2)
            #==== Training plot
            eax.plot(x[:,0], label=self.obj_name + ' Error')
            aax.plot(x[:,1], label=self.obj_name + ' Accuracy')
            ##==== Test histogram
            #test_ax[1, 0].hist(y[:,0], label=self.obj_name + ' Error')
            #test_ax[1, 1].hist(y[:,1], label=self.obj_name + ' Accuracy')
        plt.show()

    def __call__(self, step, err, acc, show=True, freq=100, test=False):
        loss_hist = self.test_history if test else self.train_history
        loss_hist[step] = err, acc
        if show and (test or ((step+1) % freq == 0)):
            if test and step == 0:
                self.status_call_count = 0
            self.print_status(step, err, acc)


#==============================================================================
# Classification functions
#==============================================================================

# Classification eval
#------------------------------------------------------------------------------
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


#==============================================================================
#------------------------------------------------------------------------------
#                               Trainer
#------------------------------------------------------------------------------
#==============================================================================


class Trainer:
    """ Manages the training for a model

    Trainer accepts a model, along with an optimizer
    and objective function, and trains it for the
    specified number of steps using a DataSet. # X=None, split_size=.8, seed=RNG_SEED_DATA
    """
    def __init__(self, channels, opt, obj, activation,
                 dataset=None, steps=1000, batch_size=6, dropout=False,
                 verbose=False, rng_seed=RNG_SEED_PARAMS):
        self.channels = channels
        self.opt = opt()
        self.obj = obj()
        self.activation = activation
        self.steps = steps
        self.batch_size = batch_size
        self.verbose = verbose
        self.rng_seed = rng_seed
        self.init_network(activation, dropout)
        self.init_dataset(dataset)
        self.init_session_status()

    def init_network(self, act, use_dropout=False):
        """ seed and instantiate network """
        np.random.seed(self.rng_seed)
        self.model = network.NeuralNetwork(self.channels, act, use_dropout)

    def init_dataset(self, dataset):
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = IrisDataset()
        self.num_test = self.dataset.X_test.shape[0]

    def init_session_status(self):
        model = self.model
        opt = self.opt
        obj = self.obj
        steps = self.steps
        #num_test = self.dataset.X_test.shape[0]
        num_test = self.num_test
        self.session_status = SessionStatus(model, opt, obj, steps, num_test)

    def summarize_results(self):
        seed = self.rng_seed
        d2 = '-'*60
        num_tr = self.steps
        num_test = self.num_test
        header = '\n# Model: {}\n#  Seed: {}\n' + d2
        header = header.format(self.channels, self.rng_seed)

        # Get stats on train history
        #  skip first 100 iters
        avg = np.mean(  np.copy(self.train_history[100:]), axis=0)
        q50 = np.median(np.copy(self.train_history[100:]), axis=0)

        tavg = np.mean(np.copy(self.test_history), axis=0)
        tq50 = np.median(np.copy(self.test_history), axis=0)

        # Print results
        print(header)
        print('   TRAIN    Error  |  Accuracy')
        print('* Average: {:.5f} | {:.5f}'.format(avg[0], avg[1]))
        print('*  Median: {:.5f} | {:.5f}'.format(q50[0], q50[1]))
        print(d2)
        print('    TEST    Error  |  Accuracy')
        print('* Average: {:.5f} | {:.5f}'.format(tavg[0], tavg[1]))
        print('*  Median: {:.5f} | {:.5f}'.format(tq50[0], tq50[1]))
        print(d2)
        print('\n\n')

    def get_loss_histories(self):
        lhtr, lhte =  self.train_history, self.test_history
        return lhtr, lhte


    def train(self):
        v = self.verbose
        for step in range(self.steps):
            # batch data
            #------------------
            x, y = self.dataset.get_batch(step, self.batch_size)

            # forward pass
            #------------------
            y_hat = self.model.forward(x)
            error, class_scores = self.obj(y_hat, y)
            accuracy = classification_accuracy(class_scores, y)
            self.session_status(step, error, accuracy, show=v)

            # backprop and update
            #------------------
            grad_loss = self.obj(error, backprop=True)
            self.model.backward(grad_loss)
            self.model.update(self.opt)

    def evaluate(self):
        num_steps = self.dataset.X_test.shape[0]
        v = self.verbose
        for i in range(num_steps):
            x, y = self.dataset.get_batch(i, test=True)

            # forward pass
            #------------------
            y_hat = self.model.forward(x)
            error, class_scores = self.obj(y_hat, y)
            accuracy = classification_accuracy(class_scores, y)
            self.session_status(i, error, accuracy, show=v, test=True)

    def __call__(self):
        self.train()
        self.evaluate()
        #self.session_status.summarize_model(True, True)
        self.train_history, self.test_history = self.session_status.get_loss_history()
        self.summarize_results()

#==============================================================================
#------------------------------------------------------------------------------
#                             Visualization
#------------------------------------------------------------------------------
#==============================================================================



def plot_curve(y, step, c='b', accuracy=True, fsize=(12,6), title=None):
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Iterations')
    ylabel = 'Accuracy' if accuracy else 'Error'
    ax.set_ylabel(ylabel)

    cur = str(cur_iter)
    plt.grid(True)
    ax.plot(y, c=c)
    if title is None:
        title = 'Iteration: {}, loss: {.4f}'.format(step, y[-1])
    ax.set_title(title)
    return fig


def save_loss_curves(save_path, lh, mname, val=False):
    plt.close('all')
    plt.figure(figsize=(16,8))
    plt.grid()
    if val:
        pstart = 0
        color = 'r'
        title = '{}: {}'.format(mname, 'Validation Error')
        label = 'median: {}'.format(np.median(lh))
        spath = save_path + '_plot_validation'
    else:
        pstart = 200
        color = 'b'
        title = '{}: {}'.format(mname, 'Training Error')
        label = 'median: {}'.format(np.median(lh[-150:]))
        spath = save_path + '_plot_train'
    plt.title(title)
    plt.plot(lh[pstart:], c=color, label=label)
    plt.legend()
    plt.savefig(spath, bbox_inches='tight')
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    plt.close('all')