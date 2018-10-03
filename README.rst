AI Sandbox
##########
Simple and clean NumPy implementations of various methods and models within AI.

The current project content only covers a basic feed-forward neural network, for training on the Iris dataset (available within the DeepLearning directory), but I look forward to expanding content within that domain and others.


Requirements
============
- Python 3.2+ (2.7 should work too)

  + NumPy 1.15 (Earlier versions not tested)
  + Jupyter (optional)
  

Both packages can be installed via pip:
    ``pip install numpy jupyter``

All project code has been developed on Linux *(Ubuntu 16.04)*, but as long as you have your python environment setup with NumPy and CLI access, it should work on your machine. I would also suggest using a virtualenv manager like pyenv_


Getting Started
^^^^^^^^^^^^^^^
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
- Optimizer: Adam
- Objective function: Logistic Cross Entropy

|

However, the model, as defined on this dataset, can be configured for other settings that can be specified in ``train.py`` or simply passed as arguments through STDIN, for example, the following line:

``python train.py -i 5000 -o sgd -a tanh -c 4 32 16 3``


Will train the model for 5000 iterations, using hyperbolic-tangent activations, with the vanilla stochastic gradient descent algorithm, and channels [4, 32, 16, 3].


License
-------
Except where noted otherwise, the content of this project is licensed under the `clear BSD-3`_.

.. _clear BSD-3: LICENSE







.. Substitutions:

.. _pyenv: https://github.com/pyenv/pyenv
.. |pyenv| replace :: pyenv