"""
###############################################################################
#                                                                             #
#  888888888888        ,ad8888ba,        88888888ba,          ,ad8888ba,      #
#       88            d8"'    `"8b       88      `"8b        d8"'    `"8b     #
#       88           d8'        `8b      88        `8b      d8'        `8b    #
#       88           88          88      88         88      88          88    #
#       88           88          88      88         88      88          88    #
#       88           Y8,        ,8P      88         8P      Y8,        ,8P    #
#       88            Y8a.    .a8P       88      .a8P        Y8a.    .a8P     #
#       88             `"Y8888Y"'        88888888Y"'          `"Y8888Y"'      #


#==============================================================================
# General
#==============================================================================

# Project structure
- Review module organization in package.
  - Does it make sense? Is it intuitive?
  - Should this be it's own module?
  - Should this module be split, or should it be consolidated?
  - DON'T organize/overengineer stuff just to be "logical" or "oop"
    - intuitive structure ALWAYS comes first. Can easily refactor
      into OOP mess if you want

# Nice to haves
- Some kind of class that is able to interpret namespace/scope
  and can return a list or references to available functions
  and classes, instead of hardcoding into a dict or whatever
   - ie: FunctionCollection.print_all_available():
     - 'log', 'sum', 'square', ...

# Error/exception:
Probably won't do this. Project not meant to be a robust, production
 package, but simply a repository of AI algorithms made from scratch and
 implementations. Still needs basic robustness, but **code needs to be clear**,
 and `try...except`s and assertions and check-expects in every function
 and class is antithetical to the spirit of the project.
- Type-checks
- Arg checks
- Exceptions

#------------------------------------------------------------------------------
# Next-steps
#  (when finished with original task)
#------------------------------------------------------------------------------

# Functionality/Models
# ====================

# Basic
## Normalization/Noise
- batch/layer norm
- dropout

## Pooling
- avg pooling
- max pooling
- unpooling
- upsampling

# Extended:

# Recurrent
- vanilla RNN layers
- LSTM / peephole / GRU

# Conv
- Conv2D layer/functions
- ConvLSTM

# Graph
- nearest neighbors
  - in your own numpy, not sklearn
- graph conv


#------------------------------------------------------------------------------
# Cool stuff
After you've done most of the basics, can add support for
more topical stuff

## Classic models
- Alexnet
- imagenet
- resnet
- highway
- inception

## Generative models
- GAN, and it's billion variants

## neuro models
- basal ganglia stuff
- visual cortex
- data structures
  - episodic memory

## Ensembled/bootstrapped

## transfer/multi-task/generalized
- Neural fabric
- progressive nets / pathnet




#==============================================================================
# Functions
#==============================================================================

#------------------------------------------------------------------------------
# Functions

## Meta
- add option for `no_backprop`, for testing
  - Can already run forward-only, but it saves values expecting
    future backward call. Needless memory overhead
- Many functions repeat the same code/signatures with only
  tiny variations. Think of solution to reduce boilerplate
  - decorators?
  - more interpretation in parent classes?
  - define unique funcs as class variables?
  - Problem: while much IS duplicate, most functions have their
             unique save parameters
    - solution?: always save inputs and outputs, and give
       function kwargs for not saving,
       eg `def forward(inputs, save_in=True, save_out=True)`

## Implement
Math :
    sign, hadamard, trig?
    Composite :
        2D conv
Noise :
    Dropout, GumSoft?
Normalization :
    Layernorm, (batchnorm?)
Pooling :
    Averagepooling, unpooling, (up/down samp?)



## Concrete
- Some stuff needs testing: atomic funcs, minmax, activations
- DOCUMENTATION: better class docstrings for all, and method stuff

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



#------------------------------------------------------------------------------


"""
