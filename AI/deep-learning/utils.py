""" utils contains all functions related to the data,
such as file IO, dataset structure and processing,
data preprocessing steps (such as batching) and many other
nifty utilities.

While the model and associated utils and layer ops
can be adapted to other datasets, the only dataset
evaluated so far is the Iris dataset.

# Iris dataset
# ========================================
Iris dataset from UCI Machine Learning Repository

URL: https://archive.ics.uci.edu/ml/datasets/iris/bezdekIris.data
Access Date: Accessed 2018-09-09
Citation: @misc{Dua:2017 ,
           author = "Dheeru, Dua and Karra Taniskidou, Efi",
           year = "2017",
           title = "{UCI} Machine Learning Repository",
           url = "http://archive.ics.uci.edu/ml",
           institution = "University of California, Irvine, School of
                          Information and Computer Sciences" }

# Dataset description
#--------------------
The Iris dataset consists of 150 samples of Iris flower features.
There are 50-samples for each of the three Iris species represented in the set:
 - Iris-setosa
 - Iris-versicolor
 - Iris-virginica

Each sample or record in the dataset has 5 attributes:
- The Iris class or type (as described above)
- 4 features of the flower:
  - sepal-length
  - sepal-width
  - petal-length
  - petal-width

And each sample is ordered in this manner:
[sepal-length, sepal-width, petal-length, petal-width, Iris-species]
 - With corresponding data-types of [float, float, float, float, string]

# Example records
* `5.1,3.8,1.6,0.2,Iris-setosa`
* `5.0,2.0,3.5,1.0,Iris-versicolor`
* `6.9,3.1,5.4,2.1,Iris-virginica`

"""

import os
import sys
import code
import shutil
import subprocess
from functools import wraps

import numpy as np


#==============================================================================
# Constants
#==============================================================================
# Data const
DATA_DIR = './data/'
RNG_SEED_DATA   = 98765
RNG_SEED_PARAMS = 12345



# Iris dataset source url
url_iris = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/'
FNAME_IRIS_DATASET = 'bezdekIris.data'
URL_IRIS_DATASET = url_iris + FNAME_IRIS_DATASET


#==============================================================================
# General utility funcs
#==============================================================================
"""
Insert the `code.interact(..)` at any line in the code you
would like to evaluate. It serves as a breakpoint,
interrupting computation at the line placed, and launches
an interactive python shell at that point.

# code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
"""

class AttrDict(dict):
    """ simply a dict accessed/mutated by attribute instead of index """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

#------------------------------------------------------------------------------
# Handy decorators
#------------------------------------------------------------------------------
def TODO(f):
    """ Serves as a convenient, clear flag for developers and insures
        wrapee func will not be called """
    @wraps(f)
    def not_finished(*args, **kwargs):
        print('\n  {} IS INCOMPLETE'.format(f.__name__))
    return not_finished

def NOTIMPLEMENTED(f):
    """ Like TODO, but for functions in a class
        raises error when wrappee is called """
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
    @wraps(f)
    def inspector(*args, **kwargs):
        print('\n Inspecting function: <{}>'.format(f.__name__))
        x = args
        y = kwargs
        z = f(*args, **kwargs)
        code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        return z
    return inspector




# Iris classes

#==============================================================================
# Dataset class
#==============================================================================
class Dataset:
    """ Contains generic attributes and functions for a dataset """
    data_dir = ROOT_DATA_DIR # constant defined by user

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
        self.label = label
        self.fname = fname
        self.data_path = self.data_dir + data_subdirs + fname
        self.url = url
        # init other possible class instance vars
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    def download_dataset_from_source(self,):
        pass

    def



#------------------------------------------------------------------------------
# Available datasets
#------------------------------------------------------------------------------
# Iris dataset
# ========================================
IRIS_DATASET = 'iris'
IRIS_CLASS_LABELS = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
IRIS_CLASS_MAP = {iris: idx for idx, iris in enumerate(IRIS_CLASS_LABELS)}


DATASETS = [IRIS_DATASET]

#==============================================================================
# General utility funcs
#==============================================================================
class AttrDict(dict):
    """ simply a dict accessed/mutated by attribe instead of index """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

# code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

#------------------------------------------------------------------------------
# Handy decorators
#------------------------------------------------------------------------------
def TODO(f):
    """ Serves as a convenient, clear flag for developers and insures
        wrapee func will not be called """
    @wraps(f)
    def not_finished(*args, **kwargs):
        print('\n  FUNCTION IS INCOMPLETE: <{}>'.format(f.__name__))
    return not_finished


def INSPECT(f):
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
# Retrieiving the iris dataset
#==============================================================================
#------------------------------------------------------------------------------
# Getting the dataset from source
#------------------------------------------------------------------------------
def sub_wget_data(url, fname, out_dir=DATA_DIR):
    """ Gets desired data from a url using wget """
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    commands = ["wget", url]
    try:
        subprocess.check_output(["wget", "-T", "120", url])
        shutil.move(fname, out_dir)
    except:
        print('Error in retrieving dataset')


def download_source_data(dataset=IRIS_DATASET, out_dir=DATA_DIR):
    """ Function to get a dataset from a source url """
    if
    print("Retrieving iris dataset from source...")
    sub_wget_data(URL_IRIS_DATASET, FNAME_IRIS_DATASET, out_dir)
    print("Iris dataset successfully downloaded!")


#------------------------------------------------------------------------------
# Load local dataset
#------------------------------------------------------------------------------
def process_raw_iris_dataset(fname):
    """ loads dataset from original format into numpy array """

    # helper func for converting lines
    def _process(record):
        """ convert entries in record from string """
        features = record[:-1]
        label = record[-1]
        processed = list(map(float, features))
        processed.append(IRIS_CLASS_MAP[label])
        return processed

    records = []
    with open(fname) as f:
        for line in f:
            record = line.strip().split(',') # splits into 5 attributes
            if len(record) != 5: continue
            record = _process(record) # converts attrib to respective dtype
            records.append(record)
    return np.array(records)


def load_dataset_iris(data_dir=DATA_DIR):
    """ loads the iris dataset as a numpy array
    If the iris dataset is not found locally, then
      it will download the dataset from the source

    Returns
    -------
    X : (150, 5)-shape numpy ndarray
        The dataset is randomly shuffled
    """
    fpath = out_dir + FNAME_IRIS_DATASET

    # If dataset has not been downloaded, or is not visible, download
    if not os.path.exists(fpath):
        download_source_iris_data(fpath)

    # Get numpy array and shuffle it
    X = process_raw_iris_dataset(fpath) # numpy array
    np.random.shuffle(X)
    return X

def load_dataset(dataset='iris', data_dir=DATA_DIR):



#==============================================================================
# Dataset processing for training
#==============================================================================
#------------------------------------------------------------------------------
# Split into Train/Test sets
#------------------------------------------------------------------------------
def load_dataset(data_dir=DATA_DIR):
    """
    """





def one_hot(y):
    """ make one-hot encoding for truth labels
    """
    n, d = (y.shape[0], 3)
    y1h = np.zeros((n, d))
    y1h[np.arange(n), y] = 1
    return y1h
