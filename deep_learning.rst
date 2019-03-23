#############
Deep learning
#############
Simple and (mostly) clean implementations of various neural network stuff.

Current deep learning implementations are a FFN.

Structure
=========
The project codebase, located in the |deep_learning directory|_.

:functions: Contains all functions related to the computation of the network. Activations, transformations, loss functions, regularizers--they are all located in `functions.py`_.
:layers: A broadly used term, layers are the essential component or architectural feature of neural networks. Layers, or "hidden layers," in this project refer specifically to the units or modules that *perform transformations on the data with a learnable or optimizable set of weights.* These are also referred to as "connections" in docstrings and comments within the code. All layers are are located in `layers.py`_.

  NB: Some functions, such as the ``Swish`` activation, have optimizable parameters, and corresponding ``Layer`` wrappers in `layers.py`_.

:network: The core component of our classifier model, all neural networks are implemented in the `network.py`_ module. Currently, the only available network implementation is a fully-connected feed-forward network with ``Dense`` layers.
:optimizer: The optimization routines for the neural network are implemented in this module. Available optimizers are ``SGD``--vanilla Stochastic Gradient Descent, and ``Adam``, a popular and powerful adaptive optimization algorithm based on moment estimation. Optimizers are located in `optimizers.py`_.

:train: The training routine is essentially an interface to all the modules of the project. It instantiates the network, and its layers by extension, the optimizer, and objective function, and loads the dataset through ``utils``. That being said, the majority of heavy lifting, such parameter initialization, file R/W, network ops, session information, etc. have been extracted to their constituent modules. Training especially relies on the utilities found in `utils.py`_. There is also a Trainer class within `utils.py`_ that performs the main operations as the train script, but has fewer options.

    Note that testing, or evalation *also* takes place within the `train.py`_ script.

:initializers: All learnable network parameters are initialized by the various initialization functions located in `initializers.py`_. Glorot normal, He normal, uniform, and constant initializations are available to choose from.
:utils: `utils.py`_ contains all functions not strictly related to network ops or computation. Something of a do-all script, utils contains the `Parser` class, and specifies the default network configuration (and updates it based on parsed command line args), maintains all information and statistics about training/testing error and accuracy, and provides various specialied preprocessing operations on data, such as one-hot encodings and the accuracy function.


Learning tasks
--------------
The deep learning project currently supports classification tasks on scikit-learn's `toy datasets <https://scikit-learn.org/stable/datasets/#toy-datasets>`_.


Setup
=====

Requirements
------------
- Python 3.7
- NumPy 1.15 (Earlier versions not tested)
- sklearn

Python packages can be installed via ``pip install pkg_name``.

Environment
...........
All project code has been developed on Linux *(Ubuntu 16.04, 18.04)*, but as long as you have your python environment setup with NumPy and CLI access, it should work on your machine. I would also suggest using a virtualenv manager like pyenv_.

****

Running the model
=================
First, clone this repo:
    ``git clone --depth=1 https://github.com/evdcush/AI-Sandbox.git``
Navigate to the deep_learning folder:
    ``cd sandbox/deep_learning``
Run the model via ``train.py``:
    ``python train.py``

Default model settings are configured as follows:

:Training iterations: 2000
:Batch size: 4
:Channels: [4, 163, 3]
:Activation: Selu
:Optimizer: ``Adam``
:Objective function: Softmax Cross Entropy
:Dataset: Iris


****



.. Substitutions:

.. PROJECT FILES:
.. _deep_learning directory: sandbox/deep_learning
.. |deep_learning directory| replace:: deep_learning directory
.. _functions.py: sandbox/deep_learning/functions.py
.. _layers.py: sandbox/deep_learning/layers.py
.. _network.py: sandbox/deep_learning/network.py
.. _initializers.py: sandbox/deep_learning/initializers.py
.. _optimizers.py: sandbox/deep_learning/optimizers.py
.. _utils.py: sandbox/deep_learning/utils.py
.. _train.py: sandbox/deep_learning/train.py

.. LOCAL FILES:
.. _LGPL: LICENSE

.. OTHER:
.. _pyenv: https://github.com/pyenv/pyenv
.. |pyenv| replace:: pyenv
