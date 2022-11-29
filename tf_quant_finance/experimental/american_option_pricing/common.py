"""Helper functions for calculating American option prices."""
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util

from tf_quant_finance import types

_SQRT_2 = np.sqrt(2.0, dtype=np.float64)


def standard_normal_cdf(a):
  # Based on the SciPy implementation:
  # https://github.com/scipy/scipy/blob/ac729b8f96a018b9156a0e5679e18b5d6e2e70a7/scipy/special/cephes/ndtr.c#L201
  x = a / _SQRT_2
  z = tf.cast(tf.math.abs(x), dtype=tf.float64)
  y1 = 0.5 + 0.5 * tf.math.erf(x)
  y2 = 0.5 * tf.math.erfc(z)
  y3 = 1 - y2
  return tf.where(z < (_SQRT_2 / 2), y1, tf.where(x > 0, y3, y2))


def d_plus(tau: types.FloatTensor, z: types.FloatTensor, r: types.FloatTensor,
           q: types.FloatTensor, sigma: types.FloatTensor) -> types.FloatTensor:
  dtype = tau.dtype
  epsilon = machine_eps(dtype)
  return (tf.math.log(tf.math.maximum(z, epsilon)) +
          (r - q + 0.5 * sigma**2) * tau) / (
              sigma * tf.math.sqrt(tau) + epsilon)


def d_minus(tau: types.FloatTensor, z: types.FloatTensor, r: types.FloatTensor,
            q: types.FloatTensor,
            sigma: types.FloatTensor) -> types.FloatTensor:
  return d_plus(tau, z, r, q, sigma) - sigma * tf.math.sqrt(tau)


def divide_with_positive_denominator(a, b):
  """Safely divides by a denominator which is mathematically always positive, but numerically can be zero."""
  dtype = b.dtype
  epsilon = machine_eps(dtype)
  b_positive = tf.debugging.assert_greater_equal(b,
                                                 tf.constant(0.0, dtype=dtype))
  with tf.control_dependencies([b_positive]):
    return a / (b + epsilon)


def machine_eps(dtype):
  """Returns the machine epsilon for the supplied dtype."""
  dtype = dtype_util.as_numpy_dtype(tf.as_dtype(dtype))
  return np.finfo(dtype).eps
