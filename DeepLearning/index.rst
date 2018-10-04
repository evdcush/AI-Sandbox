.. toctree::
    :maxdepth: 2
    :caption: Module Overview

    Functions
    Layers
    Optimizers
    Network
    Initializers
    Train
    Utils


Functions
=========

Module structure
^^^^^^^^^^^^^^^^

All functions inherit from Function, a base function class that provides the basic structure that nearly all functions use. The base Function class, ``functions.Function`` provides two essential components of the functions module:

1. It provides the structure or template of a function that nearly all functions within the module follow. Namely:

     a) Two instance methods: ``forward`` and ``backward``, and two constituent static methods for those instance methods, named after the class, such as

        ``sigmoid`` for forward
        ``sigmoid_prime`` for backward

    Static methods are the *purely functional* correlates to the mathematical function they represent. They have no state and no side-effects. These are the textbook definition functions.

    All methods receive an input, maybe do some kind of preprocessing, and send it to its corresponding static method.

    Aside from a clean, readable template and organization for each function, this sort of structure is inefficient compared to my original, "v1", approach, which was to only have the forward and backward methods.

    Being purely functional, and constraining all functions in the module to adhere to this call structure as close as possible, there is a lot of redundant computation. For example, nearly all implementations of the ``sigmoid`` function you see elsewhere save the result of the activation for easy backprop.

    While all my function implementations do cache inputs, or outputs where need be, I made sure that every derivative static method was not the "shortcut" or side-effect-optimized derivative function. They all expect the true input.

    The motivation for this was to keep it clear how the actual derivative functions work. The other motivation is that, without any assumptions on intra-class side-effects, *other functions can call on other functions' static methods* and expect the correct result.

Math
----

Parametric
----------

Activation
----------

Loss
----

Layers
======

Connections
-----------

Parametric
----------

Optimizers
==========

SGD
---

Adam
----

Network
=======

FullyConnected
--------------

Initializers
============

Uniform
-------

Normal
------

Constant
--------

Train
=====

Utils
=====

Dataset
-------

Session
-------

Parser
------

Accuracy
--------

Data
====

Iris
----
