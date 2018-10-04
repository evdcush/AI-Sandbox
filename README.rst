AI Sandbox
##########
Simple and clean NumPy implementations of various methods and models within AI.

Well, that's the goal.

But, as the project name implies, this is a space for me to explore and evaluate the core structures and algorithms involved in AI. Things could get *messy*.

Current project content, which features deep learning work on a basic feed-foward network is fairly clean and, while it has some unfinished work, the codebase is well-structured and I hope others find it easy to use and easy to understand.



Setup
=====


Requirements
------------
- Python 3.2+

  + NumPy 1.15 (Earlier versions not tested)
  + Jupyter (optional)


Both packages can be installed via pip:
    ``pip install numpy jupyter``

All project code has been developed on Linux *(Ubuntu 16.04)*, but as long as you have your python environment setup with NumPy and CLI access, it should work on your machine. I would also suggest using a virtualenv manager like pyenv_


Running the model
^^^^^^^^^^^^^^^^^
First, clone this repo:
    ``git clone --depth=1 https://github.com/evdcush/AI-Sandbox.git``
Navigate to the DeepLearning folder:
    ``cd DeepLearning``
Run the model via ``train.py``:
    ``python train.py``

You should have seen the training results for the Iris classifier model, trained for 2000 iterations.

The default train settings are configured as follows:

- Training iterations: 2000
- Batch size: 6
- Channels (network depth): [4, 64, 3]
- Optimizer: SGD
- Objective function: Logistic Cross Entropy


Model Options
-------------
The model, as defined on this dataset, can be configured for other settings that can be specified in ``train.py`` or simply passed as arguments through STDIN, for example, the following line:

``python train.py -i 500 -o adam -a tanh -c 4 32 16 3``


Will train the model for 500 iterations, using hyperbolic-tangent activations, the Adam optimizer, and channels [4, 32, 16, 3].

There are many different settings that can be specified through the CLI, and you can review them all in ``utils.Parser``. Here is a quick reference:

-i, --num_iters, default=2000  
    Number of training iterations
--batch_size, -b, default=6    
    Training mini-batch size. This defines how many samples are passed to the model in one training iteration
--activation, -a, default=sigmoid, choices=[relu, elu, selu, sigmoid, tanh, swish, softmax]    
    Available activation functions.



License
-------
Except where noted otherwise, the content of this project is licensed under the `clear BSD-3`_.

.. _clear BSD-3: LICENSE







.. Substitutions:

.. _pyenv: https://github.com/pyenv/pyenv
.. |pyenv| replace :: pyenv
