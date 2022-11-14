"""Adaptive Gauss-Kronrod quadrature algorithm for numeric integration."""
from typing import Callable, Optional
import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.math.integration import adaptive_update
from tf_quant_finance.math.integration import gauss_constants


def _non_adaptive_gauss_kronrod(
    func: Callable[[types.FloatTensor], types.FloatTensor],
    lower: types.FloatTensor,
    upper: types.FloatTensor,
    num_points: int = 15,
    dtype: Optional[tf.DType] = None,
    name: Optional[str] = None) -> (types.FloatTensor, types.FloatTensor):
  """Evaluates definite integral using non-adaptive Gauss-Kronrod quadrature.

  Integrates `func` using non-adaptive Gauss-Kronrod quadrature [1].

  Applies change of variables to the function to obtain the [-1,1] integration
  interval.
  Takes the sum of values obtained from evaluating the new function at points
  given by the roots of the Legendre polynomial of degree `(num_points-1)//2`
  and the roots of the Stieltjes polynomial of degree `(num_points+1)//2`,
  multiplied with corresponding precalculated coefficients.

  #### References
  [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula

  #### Example
  ```python
    f = lambda x: x*x
    a = tf.constant([0.0])
    b = tf.constant([3.0])
    num_points = 21
    _non_adaptive_gauss_kronrod(f, a, b, num_points) # [9.0]
  ```

  Args:
    func: Represents a function to be integrated. It must be a callable of a
      single `Tensor` parameter and return a `Tensor` of the same shape and
      dtype as its input. It will be called with a `Tensor` of shape
      `lower.shape + [n]` (where n is integer number of points) and of the same
      `dtype` as `lower`.
    lower: Represents the lower limits of integration. `func` will be integrated
      between each pair of points defined by `lower` and `upper`.
    upper: Same shape and dtype as `lower` representing the upper limits of
      intergation.
    num_points: Number of points at which the function `func` will be evaluated.
      Implemented for 15,21,31.
      Default value: 15.
    dtype: If supplied, the dtype for the `lower` and `upper`. Result will have
      the same dtype. Default value: None which maps to dtype of `lower`.
    name: The name to give to the ops created by this function. Default value:
      None which maps to 'non_adaptive_gauss_kronrod'.

  Returns:
    A tuple:
      * `Tensor` of shape `batch_shape`, containing value of the definite
      integral,
      * `Tensor` of shape `batch_shape + [legendre_num_points]`, containing
      values of the function evaluated at the Legendre polynomial root points.
  """
  with tf.name_scope(name=name or 'non_adaptive_gauss_kronrod'):
    # Shape batch_shape
    lower = tf.convert_to_tensor(lower, dtype=dtype, name='lower')
    dtype = lower.dtype
    upper = tf.convert_to_tensor(upper, dtype=dtype, name='upper')
    legendre_num_points = (num_points - 1) // 2
    # Shape [legendre_num_points]
    legendre_roots = gauss_constants.legendre_roots.get(legendre_num_points,
                                                        None)
    # Shape [num_points - legendre_num_points]
    stieltjes_roots = gauss_constants.stieltjes_roots.get(num_points, None)
    if legendre_roots is None:
      raise ValueError(f'Unsupported value for `num_points`: {num_points}')
    if stieltjes_roots is None:
      raise ValueError(f'Unsupported value for `num_points`: {num_points}')
    # Shape batch_shape + [1]
    lower = tf.expand_dims(lower, -1)
    upper = tf.expand_dims(upper, -1)
    # Shape [num_points]
    roots = legendre_roots + stieltjes_roots
    roots = tf.constant(roots, dtype=dtype)
    # Shape batch_shape + [num_points]
    grid = ((upper - lower) * roots + upper + lower) / 2
    func_results = func(grid)
    # Shape [num_points]
    weights = gauss_constants.kronrod_weights.get(num_points, None)
    # Shape batch_shape
    result = tf.reduce_sum(
        func_results * (upper - lower) * weights / 2, axis=-1)
    # Shapes batch_shape, batch_shape + [num_points]
    return result, func_results


def gauss_kronrod(func: Callable[[types.FloatTensor], types.FloatTensor],
                  lower: types.FloatTensor,
                  upper: types.FloatTensor,
                  tolerance: float,
                  num_points: int = 21,
                  max_depth: int = 20,
                  dtype: Optional[tf.DType] = None,
                  name: Optional[str] = None) -> types.FloatTensor:
  """Evaluates definite integral using adaptive Gauss-Kronrod quadrature.

  Integrates `func` using adaptive Gauss-Kronrod quadrature [1].

  Applies change of variables to the function to obtain the [-1,1] integration
  interval.
  Takes the sum of values obtained from evaluating the new function at points
  given by the roots of the Legendre polynomial of degree `(num_points-1)//2`
  and the roots of the Stieltjes polynomial of degree `(num_points+1)//2`,
  multiplied with corresponding precalculated coefficients.
  Repeats procedure if not accurate enough by halving the intervals and dividing
  these into the same number of subintervals.

  #### References
  [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula

  #### Example
  ```python
    f = lambda x: x*x
    a = tf.constant([0.0])
    b = tf.constant([3.0])
    tol = 1e-5
    num_points = 21
    max_depth = 10
    gauss_kronrod(f, a, b, tol, num_points, max_depth) # [9.0]
  ```

  Args:
    func: Represents a function to be integrated. It must be a callable of a
      single `Tensor` parameter and return a `Tensor` of the same shape and
      dtype as its input. It will be called with a `Tensor` of shape
      `lower.shape + [n,  num_points]` (where `n` is defined by the algorithm
      and represents the number of subintervals) and of the same `dtype` as
      `lower`.
    lower: Represents the lower limits of integration. `func` will be integrated
      between each pair of points defined by `lower` and `upper`. Must be a
      1-dimensional tensor of shape `[batch_dim]`.
    upper: Same shape and dtype as `lower` representing the upper limits of
      intergation.
    tolerance: Represents the tolerance for the estimated error of the integral
      estimation, at which to stop further dividing the intervals.
    num_points: Number of points at which the function `func` will be evaluated.
      Implemented for 15,21,31. Default value: 21.
    max_depth: Maximum number of times to divide intervals into two parts and
      recalculate Gauss-Kronrod on them. Default value: 20.
    dtype: If supplied, the dtype for the `lower` and `upper`. Result will have
      the same dtype. Default value: None which maps to dtype of `lower`.
    name: The name to give to the ops created by this function. Default value:
      None which maps to 'gauss_kronrod'.

  Returns:
    `Tensor` of shape `[batch_dim]`, containing value of the definite integral.
  """
  with tf.name_scope(name=name or 'gauss_kronrod'):
    # Shape [batch_dim]
    lower = tf.convert_to_tensor(lower, dtype=dtype, name='lower')
    dtype = lower.dtype
    upper = tf.convert_to_tensor(upper, dtype=dtype, name='upper')
    legendre_num_points = (num_points - 1) // 2

    def cond(lower, upper, sum_estimates):
      del upper, sum_estimates
      return tf.size(lower) > 0

    def body(lower, upper, sum_estimates):
      # Shapes [batch_dim, n],
      # [batch_dim, n, num_points]
      kronrod_result, func_results = _non_adaptive_gauss_kronrod(
          func, lower, upper, num_points, dtype, name)
      # Shape [batch_dim, n, legendre_num_points]
      legendre_func_results = func_results[..., :legendre_num_points]
      # Shape [legendre_num_points]
      legendre_weights = tf.constant(
          gauss_constants.legendre_weights[legendre_num_points], dtype=dtype)
      # Shape [batch_dim, n, 1]
      lower_exp = tf.expand_dims(lower, -1)
      upper_exp = tf.expand_dims(upper, -1)
      # Shape [batch_dim, n]
      legendre_result = tf.reduce_sum(
          legendre_func_results * (upper_exp - lower_exp) *
          legendre_weights / 2, axis=-1)
      error = tf.abs(kronrod_result - legendre_result)
      # Shapes [batch_dim, n], [batch_dim, n], [batch_dim]
      new_lower, new_upper, sum_good_estimates = adaptive_update.update(
          lower, upper, kronrod_result, error, tolerance, dtype)
      # Shape [batch_dim]
      sum_estimates += sum_good_estimates
      # Shapes [batch_dim, n], [batch_dim, n], [batch_dim]
      return new_lower, new_upper, sum_estimates

    sum_estimates = tf.zeros_like(lower, dtype=dtype)
    # n = 1
    # Shape [batch_dim, n]
    lower = tf.expand_dims(lower, -1)
    upper = tf.expand_dims(upper, -1)
    loop_vars = (lower, upper, sum_estimates)
    # Ensure that the lower and upper have the same batch shape
    lower, upper = utils.broadcast_tensors(lower, upper)
    # Extract the batch shape
    batch_shape = lower.shape[:-1]
    _, _, estimate_result = tf.while_loop(
        cond=cond, body=body, loop_vars=loop_vars,
        maximum_iterations=max_depth,
        shape_invariants=(tf.TensorShape(batch_shape + [None]),
                          tf.TensorShape(batch_shape + [None]),
                          tf.TensorShape(batch_shape)))
    # Shape [batch_dim]
    return estimate_result
