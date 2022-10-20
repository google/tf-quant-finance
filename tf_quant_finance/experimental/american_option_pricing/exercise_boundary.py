"""Calculating the exercise boundary function of an American option."""
from typing import Callable, Optional
import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.math.integration import gauss_legendre


_SQRT_2 = np.sqrt(2.0, dtype=np.float64)


def _standard_normal_cdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


def _d_plus(tau: types.FloatTensor, z: types.FloatTensor, r: types.FloatTensor,
            q: types.FloatTensor,
            sigma: types.FloatTensor) -> types.FloatTensor:
  return tf.math.divide_no_nan(
      tf.math.log(z) + (r - q + 0.5 * sigma**2) * tau,
      sigma * tf.math.sqrt(tau))


def _d_minus(tau: types.FloatTensor, z: types.FloatTensor, r: types.FloatTensor,
             q: types.FloatTensor,
             sigma: types.FloatTensor) -> types.FloatTensor:
  return _d_plus(tau, z, r, q, sigma) - sigma * tf.math.sqrt(tau)


def boundary_numerator(
    tau: types.FloatTensor,
    b: Callable[[types.FloatTensor, types.FloatTensor, types.FloatTensor,
                 types.FloatTensor], types.FloatTensor],
    k: types.FloatTensor,
    r: types.FloatTensor,
    q: types.FloatTensor,
    sigma: types.FloatTensor,
    integration_num_points: int = 32,
    dtype: Optional[tf.DType] = None) -> types.FloatTensor:
  """Subcalculation to get the exercise boundary function of an American option.

  Evaluates the numerator part of the calculation required to get the exercise
  boundary function of an American option. This corresponds to `N` in formula
  (3.7) in the paper [1].

  #### References
  [1] Leif Andersen, Mark Lake and Dimitri Offengenden. High-performance
  American option pricing. 2015
  https://engineering.nyu.edu/sites/default/files/2019-03/Carr-adjusting-exponential-levy-models.pdf#page=54

  #### Example
  ```python
    def b_0(tau, k, r, q):
      return k * tf.math.minimum(1, r/q)
    tau = tf.constant([1, 2], dtype=tf.float64)
    k = tf.constant([100, 100], dtype=tf.float64)
    r = tf.constant([0.01, 0.02], dtype=tf.float64)
    q = tf.constant([0.01, 0.02], dtype=tf.float64)
    sigma = tf.constant([0.1, 0.15], dtype=tf.float64)
    integration_num_points = 32
    boundary_numerator(tau, b_0, k, r, q, sigma, integration_num_points)
    # tf.constant([0.4849528 , 0.47702501], dtype=tf.float64)
  ```

  Args:
    tau: Grid of values of shape `batch_shape` indicating the time left until
      option maturity.
    b: Represents the exercise boundary function for the option.
    k: Same shape and dtype as `tau` representing the strike price of the
      option.
    r: Same shape and dtype as `tau` representing the annualized risk-free
      interest rate, continuously compounded.
    q: Same shape and dtype as `tau` representing the dividend rate.
    sigma: Same shape and dtype as `tau` representing the volatility of the
      option's returns.
    integration_num_points: The number of points used in the integration
      approximation method.
      Default value: 32.
    dtype: If supplied, the dtype for all input tensors. Result will have the
      same dtype.
      Default value: None which maps to dtype of `tau`.

  Returns:
    `Tensor` of shape `batch_shape`, containing a partial result for calculating
    the exercise boundary.
  """
  with tf.name_scope('calculate_N'):
    tau = tf.convert_to_tensor(tau, dtype=dtype)
    dtype = tau.dtype
    k = tf.convert_to_tensor(k, dtype=dtype)
    r = tf.convert_to_tensor(r, dtype=dtype)
    q = tf.convert_to_tensor(q, dtype=dtype)
    sigma = tf.convert_to_tensor(sigma, dtype=dtype)
    term1 = _standard_normal_cdf(_d_minus(tau, b(tau, k, r, q)/k, r, q, sigma))
    def func(u):
      tau_exp = tf.expand_dims(tau, axis=-1)
      k_exp = tf.expand_dims(k, axis=-1)
      r_exp = tf.expand_dims(r, axis=-1)
      q_exp = tf.expand_dims(q, axis=-1)
      sigma_exp = tf.expand_dims(sigma, axis=-1)
      ratio = b(tau_exp, k_exp, r_exp, q_exp) / b(u, k_exp, r_exp, q_exp)
      norm = _standard_normal_cdf(
          _d_minus(tau_exp - u, ratio, r_exp, q_exp, sigma_exp))
      return tf.math.exp(r_exp * u) * norm

    term2 = r * gauss_legendre(
        func=func,
        lower=tf.zeros_like(tau),
        upper=tau,
        num_points=integration_num_points,
        dtype=dtype)
    return term1 + term2


def boundary_denominator(
    tau: types.FloatTensor,
    b: Callable[[types.FloatTensor, types.FloatTensor, types.FloatTensor,
                 types.FloatTensor], types.FloatTensor],
    k: types.FloatTensor,
    r: types.FloatTensor,
    q: types.FloatTensor,
    sigma: types.FloatTensor,
    integration_num_points: int = 32,
    dtype: Optional[tf.DType] = None) -> types.FloatTensor:
  """Subcalculation to get the exercise boundary function of an American option.

  Evaluates the denominator part of the calculation required to get the exercise
  boundary function of an American option. This corresponds to `D` in formula
  (3.8) in the paper [1].

  #### References
  [1] Leif Andersen, Mark Lake and Dimitri Offengenden. High-performance
  American option pricing. 2015
  https://engineering.nyu.edu/sites/default/files/2019-03/Carr-adjusting-exponential-levy-models.pdf#page=54

  #### Example
  ```python
    def b_0(tau, k, r, q):
      return k * tf.math.minimum(1, r/q)
    tau = tf.constant([1, 2], dtype=tf.float64)
    k = tf.constant([100, 100], dtype=tf.float64)
    r = tf.constant([0.01, 0.02], dtype=tf.float64)
    q = tf.constant([0.01, 0.02], dtype=tf.float64)
    sigma = tf.constant([0.1, 0.15], dtype=tf.float64)
    integration_num_points = 32
    boundary_denominator(tau, b_0, k, r, q, sigma, integration_num_points)
    # tf.constant([0.52509737, 0.56378576], dtype=tf.float64)
  ```

  Args:
    tau: Grid of values of shape `batch_shape` indicating the time left until
      option maturity.
    b: Represents the exercise boundary function for the option.
    k: Same shape and dtype as `tau` representing the strike price of the
      option.
    r: Same shape and dtype as `tau` representing the annualized risk-free
      interest rate, continuously compounded.
    q: Same shape and dtype as `tau` representing the dividend rate.
    sigma: Same shape and dtype as `tau` representing the volatility of the
      option's returns.
    integration_num_points: The number of points used in the integration
      approximation method.
      Default value: 32.
    dtype: If supplied, the dtype for all input tensors. Result will have the
      same dtype.
      Default value: None which maps to dtype of `tau`.

  Returns:
    `Tensor` of shape `batch_shape`, containing a partial result for calculating
    the exercise boundary.
  """
  with tf.name_scope('calculate_D'):
    tau = tf.convert_to_tensor(tau, dtype=dtype)
    dtype = tau.dtype
    k = tf.convert_to_tensor(k, dtype=dtype)
    r = tf.convert_to_tensor(r, dtype=dtype)
    q = tf.convert_to_tensor(q, dtype=dtype)
    sigma = tf.convert_to_tensor(sigma, dtype=dtype)
    term1 = _standard_normal_cdf(_d_plus(tau, b(tau, k, r, q)/k, r, q, sigma))
    def func(u):
      tau_exp = tf.expand_dims(tau, axis=-1)
      k_exp = tf.expand_dims(k, axis=-1)
      r_exp = tf.expand_dims(r, axis=-1)
      q_exp = tf.expand_dims(q, axis=-1)
      sigma_exp = tf.expand_dims(sigma, axis=-1)
      ratio = b(tau_exp, k_exp, r_exp, q_exp) / b(u, k_exp, r_exp, q_exp)
      norm = _standard_normal_cdf(
          _d_plus(tau_exp - u, ratio, r_exp, q_exp, sigma_exp))
      return tf.math.exp(q_exp * u) * norm

    term2 = q * gauss_legendre(
        func=func,
        lower=tf.zeros_like(tau),
        upper=tau,
        num_points=integration_num_points,
        dtype=dtype)
    return term1 + term2
