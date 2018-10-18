AI Sandbox
##########
Simple and clean NumPy implementations of various methods and models within AI.

Well, that's the goal. But, as the project name implies, this is a space for me to **build** and **explore** the core structures and algorithms involved in AI. Things could get *messy*.



Project Contents
----------------
Deep learning
=============
The current project content features deep learning work on a basic feed-foward network and, while it has some unfinished work, the implementations are fairly clean and I hope it is easy to use and easy to understand.

Structure
=========
The project codebase for the neural network is split neatly into different modules.

:functions: Contains all functions related to the computation of the network. Activations, transformations, loss functions, regularizers--they are all located in `functions.py`_.
:layers: A broadly used term, layers are the essential component or architectural feature of neural networks. Layers, or "hidden layers," in this project refer specifically to the units or modules that **perform transformations on the data with a learnable or optimizable set of weights**. These are also referred to as "connections" in docstrings and comments within the code.

  NB: Some functions, such as the ``Swish`` activation, have optimizable parameters, and corresponding ``ParametricLayer`` wrappers in `layers.py`_, but are not considered "layers"). All layers are are located in `layers.py`_.

:network: The core component of our classifier model, all neural networks are implemented in the `network.py`_ module. Currently, the only available network implementation is a fully-connected feed-forward network with ``Dense`` layers.
:optimizer: The optimization routines for the neural network are implemented in this module. Available optimizers are ``SGD``--vanilla Stochastic Gradient Descent, and ``Adam``, a popular and powerful adaptive optimization algorithm based on moment estimation.

  While the default optimizer is set for ``SGD``, I would recommend using ``Adam``, as it is *significantly* more powerful. Both implementations, and all those to come, are located in `optimizers.py`_.

:train: The training routine is essentially an interface to all the modules of the project. It instantiates the network, and its layers by extension, the optimizer, and objective function, and loads the dataset through ``utils``. That being said, the majority of heavy lifting, such parameter initialization, file R/W, network ops, session information, etc. have been extracted to their constituent modules. Training especially relies on the utilities found in `utils.py`_. There is also a Trainer class within `utils.py`_ that performs the same operations as the train script.

    Note that testing, or evalation *also* takes place within the `train.py`_ script.

:initializers: All learnable network parameters are initialized by the various initialization functions located in `initializers.py`_. Glorot normal, He normal, uniform, and constant initializations are available to choose from.
:utils: `utils.py`_ contains all functions not strictly related to network ops or computation. Something of a do-all script, utils loads datasets (and performs all related operations on them, such as train/test splits, batching, shuffuling, etc.), specifies the default network configuration (and updates it based on parsed command line args), maintains all information and statistics about training/testing error and accuracy, and provides various specialied preprocessing operations on data, such as one-hot encodings and the accuracy function.


Learning tasks
..............
The deep learning project is defined on the `Iris dataset`_, which is available as a serialized numpy array within the deep learning `data directory`_. There are assumptions made in the implementations of the deep learning code base, especially with respect to the dimensionality of data, but the code is generalized enough to work on other tasks.


Setup
-----

Requirements
============
- Python 3.2+
- NumPy 1.15 (Earlier versions not tested)

NumPy can be installed via pip: ``pip install numpy``

Environment
...........
All project code has been developed on Linux *(Ubuntu 16.04, 18.04)*, but as long as you have your python environment setup with NumPy and CLI access, it should work on your machine. I would also suggest using a virtualenv manager like pyenv_


Running the model
-----------------
First, clone this repo:
    ``git clone --depth=1 https://github.com/evdcush/AI-Sandbox.git``
Navigate to the deep_learning folder:
    ``cd sandbox/deep_learning``
Run the model via ``train.py``:
    ``python train.py``

You should see the training and test results for the classifier model:

Something like::

    # Model Summary:

    NeuralNetwork
      Layers:
         1 : Dense (4, 64)
              : Sigmoid
         2 : Dense (64, 3)
    - OPTIMIZER : SGD
    - OBJECTIVE : SoftmaxCrossEntropy

    # Training results, 1500 iterations
    #------------------------------
                Error  |  Accuracy
    * Average: 0.77293 | 0.77333
    *  Median: 0.78027 | 0.83333
    #------------------------------

    # Test results, 30 samples
    #------------------------------
                Error  |  Accuracy
    * Average: 0.55916 | 0.86667
    *  Median: 0.58926 | 1.00000
    #------------------------------



The default train settings are configured as follows:

:Training iterations: 1500
:Batch size: 6
:Channels: [4, 64, 3]
:Activation: Logistic sigmoid
:Optimizer: ``SGD``
:Objective function: Softmax Cross Entropy


Model Options
-------------
The model, as defined on this dataset, can be configured for other settings that can be specified in ``train.py`` or simply passed as arguments through STDIN, for example, the following line:

``python train.py -i 500 -o adam -a tanh -c 4 32 16 3``


Will train the model for 500 iterations, using hyperbolic-tangent activations, the Adam optimizer, and channels [4, 32, 16, 3]. While the ``SGD`` optimizer can be sensitive to network configuration (notably with channels), ``adam`` is robust and can converge with almost any network config.

|

There are many different settings that can be specified through the CLI, and you can review them all in ``utils.Parser``.

Training options quick-reference
================================

-i int, --num_iters  Number of training iterations
-b int, --batch_size  Training mini-batch sizes.

              This defines how many samples are passed to the model in one training iteration.

-a ACTIVATION, --activation
              Activation function used in the network.

              Available activations: ``relu, elu, selu, softplus, sigmoid, tanh, swish, softmax``

-o OPTIMIZER, --optimizer  Model optimizer.

    Available optimizers: ``sgd, adam``


Known issues
------------
None...yet. Please let me know if you have any issues with the code!

The model performs as expected on the Iris dataset, but there are some intra-module inconsistencies, missing features, and cleanup required.

The most notable lacking feature currently is the inability to serialize or save the model parameters. A lot of that plumbing is in place, such as how parameters are stored and accessed in layers, and the model pathing and constants in utils, but it has not been implemented yet.


License
-------
Except where noted otherwise, this project is licensed under the `BSD-3-Clause-Clear`_.


.. Substitutions:

.. PROJECT FILES:
.. _functions.py: sandbox/deep_learning/functions.py
.. _layers.py: sandbox/deep_learning/layers.py
.. _network.py: sandbox/deep_learning/network.py
.. _initializers.py: sandbox/deep_learning/initializers.py
.. _optimizers.py: sandbox/deep_learning/optimizers.py
.. _utils.py: sandbox/deep_learning/utils.py
.. _train.py: sandbox/deep_learning/train.py

.. LOCAL FILES:
.. _BSD-3-Clause-Clear: LICENSE
.. _Iris dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set

.. _|Iris dataset| replace :: `Iris dataset`
.. _data directory: sandbox/data/Iris

.. OTHER:
.. _pyenv: https://github.com/pyenv/pyenv
.. |pyenv| replace :: pyenv
