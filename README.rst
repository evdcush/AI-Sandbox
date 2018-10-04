AI Sandbox
##########
Simple and clean NumPy implementations of various methods and models within AI.

Well, that's the goal.

But, as the project name implies, this is a space for me to explore and evaluate the core structures and algorithms involved in AI. Things could get *messy*.

Current project content, which features deep learning work on a basic feed-foward network is fairly clean and, while it has some unfinished work, the codebase is well-structured and I hope others find it easy to use and easy to understand.

|

In order, the project goals are:

- Clarity

    + No "one-liners," no nested comprehensions, no excessive function compositions--the ever-tedious ``np.reshape(np.broadcast_to(np.expand_dims(...)))`` aside-- and no gross boilerplate.

- Simplicity

    + Closely related to clarity, the goal is to factor out complex or confusing subjects within AI. For example, the Dense layer in the deep learning module makes use of the functions.Linear function instead of simply managing the matrix multiplication + bias ops at the same convenient callee level as its parameters. Linear, in turn, uses the functions.MatMul and functions.Bias functions, instead of performing those ops at *its* own level, for caching convenience and efficiency.

        * But wait, there's more. Instead of MatMul and Bias performing their own ops at their respective ``__call__`` or ``forward`` functions, they process inputs/ouputs, and send **those** to a purely functional static method.

    The reason for this heierarchy is so the reader can see how each element of a model is composed, and can easily trace the flow of computation and gradient chaining through a network.

- Extensibility

    + While I would not use any of this code in place of an actual library, the idea is to have that level of flexibility and invariance wrt individual tasks.



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
