import numpy as np
from utils import NOTIMPLEMENTED



#==============================================================================
#------------------------------------------------------------------------------
#                              Initializers
#------------------------------------------------------------------------------
#==============================================================================

#==============================================================================
# Base initialization functions:
#  Initializer, Normal, Uniform, Constant
#==============================================================================

# Initializer
# -----------
# inherits :
# derives : Normal, Uniform, Constant
class Initializer:
    """ Base class for initializer """

    dtype_ = np.float32
    def __init__(self, **kwargs):
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    @NOTIMPLEMENTED
    def __call__(self, kdims):
        """ initialize an array with values drawn from
        a distribution
        """
        pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Normal
# ------
# inherits : Initializer
# derives : HeNormal, GlorotNormal
class Normal(Initializer):
    """ Initializes an array with a normal distribution

    Values are drawn from a Gaussian distribution with
    mean 0 and stdv 'scale'
    """
    normal = np.random.normal

    def __init__(self, scale=0.05, **kwargs):
        super().__init__(**kwargs)
        self.scale = 0.05

    def __call__(self, kdims):
        scale = self.scale
        size = kdims
        param_array = self.normal(scale=scale, size=size).astype(self.dtype_)
        return param_array

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Uniform
# -------
# inherits : Initializer
# derives : GlorotUniform
class Uniform(Initializer):
    """ Initializes an array with a uniform distribution

    Values are drawn uniformly from the interval
    [-scale, scale)

    """
    uniform = np.random.uniform

    def __init__(self, scale=0.05, **kwargs):
        super().__init__(**kwargs)
        self.scale = 0.05

    def __call__(self, kdims):
        scale = self.scale
        low, high = -scale, scale
        dtype = self.dtype_
        size = kdims
        param_array = self.uniform(low=low, high=high, size=kdims).astype(dtype)
        return param_array

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Constant
# --------
# inherits : Initializer
# derives : Zeros
class Constant(Initializer):
    """ Initializes array with repeated constants """
    fill_value = 1.0

    def __call__(self, kdims, f_value=None):
        fill_val = self.fill_value if f_value is None else f_value
        param_array = np.full(kdims, fill_val, dtype=self.dtype_)
        return param_array

#==============================================================================
# Initializers
#==============================================================================

#------------------------------------------------------------------------------
# Normal :
#  HeNormal, GlorotNormal
#------------------------------------------------------------------------------

class HeNormal(Normal):
    """ Inititalizes array from scaled Gaussian

    Values drawn independently from a Gaussian distrib.
    with mean 0, and stddev:
      scale * sqrt(2/fan_in)

    Where 'fan_in' = number of input units (input channels)

    """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(scale, **kwargs)

    def __call__(self, kdims):
        scale = self.scale
        fan_in = kdims[0]
        stdv = scale * np.sqrt(2 / fan_in)
        param_array = self.normal(scale=stdv, size=kdims).astype(self.dtype_)
        return param_array


class GlorotNormal(Normal):
    """ Inititalizes array from scaled Gaussian

    Values drawn independently from a Gaussian distrib.
    with mean 0, and stddev:
      scale * sqrt(2/(fan_in + fan_out))

    Where
    'fan_in'  = number of input units (input channels)
    'fan_out' = number of output units (output channels)

    """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(scale, **kwargs)

    def __call__(self, kdims):
        scale = self.scale
        fan = sum(kdims)
        stdv = scale * np.sqrt(2 / fan)
        param_array = self.normal(scale=stdv, size=kdims).astype(self.dtype_)
        return param_array


#------------------------------------------------------------------------------
# Uniform :
#  GlorotUniform
#------------------------------------------------------------------------------

class GlorotUniform(Uniform):
    """ Initializes an array with a uniform distribution

    Values are drawn uniformly from the interval
    [-m, m)

    Where
    m = scale * sqrt(6/(fan_in + fan_out))
    'fan_in'  = number of input units (input channels)
    'fan_out' = number of output units (output channels)

    """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(scale, **kwargs)

    def __call__(self, kdims):
        scale = self.scale
        fan = sum(kdims)
        m = scale * np.sqrt(6 / fan)
        param_array = self.uniform(-m, m, size=kdims).astype(self.dtype_)
        return param_array

#------------------------------------------------------------------------------
# Constant :
#  Zeros, Ones
#------------------------------------------------------------------------------

class Zeros(Constant):
    """ Initializes array with all zeros """
    fill_value = 0.0

class Ones(Constant):
    """ Initializes array with all ones """
    fill_value = 1.0
