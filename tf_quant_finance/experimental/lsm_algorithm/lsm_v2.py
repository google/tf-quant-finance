# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the regression MC of Longstaff and Schwartz.

This version of the algorithm is XLA-compatible but might be slower in some
cases than the previous version without XLA compilation. The following major
differences with the previous version are:

1. All tensors inside `_continuation_value_fn`, `_expected_exercise_fn` and
  `_updated_cashflow` are now of static shape. This increases the memory
  consumption of the algorithm but avoids dynamically shaped tensors.
2. `make_polynomial_basis` outputs the function that has an additional
  `time_index` argument. This is necessary to avoid dynamically shaped tensors.
"""

import collections
import tensorflow.compat.v2 as tf


LsmLoopVars = collections.namedtuple(
    "LsmLoopVars",
    [
        "exercise_index",  # int. The LSM algorithm iterates backwards over
        # times where an option can be exercised, this tracks progress.
        "cashflow"  # (N, V, K) shaped tensor. Tracks the optimal cashflow
        # of each sample path for each payoff dimension at each
        # exercise time.
    ])


def make_polynomial_basis(degree):
  """Produces a callable from samples to polynomial basis for use in regression.

  The output callable accepts a scalar `Tensor` `t` and a `Tensor` `X` of
  shape `[num_samples, dim]`, computes a centered value
  `Y = X - mean(X, axis=0)` and outputs a `Tensor` of shape
  `[degree * dim, num_samples]`, where
  ```
  Z[i*j, k] = X[k, j]**(degree - i) * X[k, j]**i, 0<=i<degree - 1, 0<=j<dim
  ```
  For example, if `degree` and `dim` are both equal to 2, the polynomial basis
  is `1, X, X**2, Y, Y**2, X * Y, X**2 * Y, X * Y**2`, where `X` and `Y` are
  the spatial axes.

  #### Example
  ```python
  basis = tff.experimental.lsm_algorithm.make_polynomial_basis_v2(2)
  x = [[1.0], [2.0], [3.0], [4.0]]
  x = tf.expand_dims(x, axis=-1)
  basis(x, tf.constant(0, dtype=np.int32))
  # Expected result:
  [[ 1.  ,  1.  ,  1.  ,  1.  ], [-1.5 , -0.5 ,  0.5 ,  1.5 ],
  [ 2.25,  0.25,  0.25,  2.25]]
  ```

  Args:
    degree: An `int32` scalar `Tensor`. The degree of the desired polynomial
      basis.

  Returns:
    A callable from `Tensor`s of shape `[batch_size, num_samples, dim]` to
    `Tensor`s of shape `[batch_size, (degree + 1)**dim, num_samples]`.

  Raises:
    ValueError: If `degree` is less than `1`.
  """
  tf.debugging.assert_greater_equal(
      degree, 0,
      message="Degree of polynomial basis can not be negative.")
  def basis(sample_paths, time_index):
    """Computes polynomial basis expansion at the given sample points.

    Args:
      sample_paths: A `Tensor` of either `flaot32` or `float64` dtype and of
        either shape `[num_samples, num_times, dim]` or
        `[batch_size, num_samples, num_times, dim]`.
      time_index: An integer scalar `Tensor` that corresponds to the time
        coordinate at which the basis function is computed.

    Returns:
      A `Tensor`s of shape `[batch_size, (degree + 1)**dim, num_samples]`.
    """
    sample_paths = tf.convert_to_tensor(sample_paths,
                                        name="sample_paths")
    if sample_paths.shape.rank == 3:
      sample_paths = tf.expand_dims(sample_paths, axis=0)
    shape = tf.shape(sample_paths)
    num_samples = shape[1]
    batch_size = shape[0]
    dim = sample_paths.shape[-1]  # Dimension should statically known
    # Shape [batch_size, num_samples, 1, dim]
    slice_samples = tf.slice(sample_paths, [0, 0, time_index, 0],
                             [batch_size, num_samples, 1, dim])
    # Shape [batch_size, num_samples, 1, dim]
    samples_centered = slice_samples - tf.math.reduce_mean(
        slice_samples, axis=1, keepdims=True)
    grid = tf.range(degree + 1, dtype=samples_centered.dtype)
    # Creates a grid of 'power' expansions, i.e., a `Tensor` of shape
    # [(degree + 1)**dim, dim] with entries [k_1, .., k_dim] where
    ## 0 <= k_i <= dim.
    grid = tf.meshgrid(*(dim * [grid]))
    # Shape [(degree + 1)**3, dim]
    grid = tf.reshape(tf.stack(grid, -1), [-1, dim])
    # `samples_centered` has shape [batch_size, num_samples, 1, dim],
    # `samples_centered**grid` has shape
    # `[batch_size, num_samples, (degree + 1)**dim, dim]`
    # so that the output shape is `[batch_size, num_samples, (degree + 1)**dim]`
    basis_expansion = tf.reduce_prod(samples_centered**grid, axis=-1)
    return tf.transpose(basis_expansion, [0, 2, 1])
  return basis


def least_square_mc(sample_paths,
                    exercise_times,
                    payoff_fn,
                    basis_fn,
                    discount_factors=None,
                    num_calibration_samples=None,
                    dtype=None,
                    name=None):
  """Values Amercian style options using the LSM algorithm.

  The Least-Squares Monte-Carlo (LSM) algorithm is a Monte-Carlo approach to
  valuation of American style options. Using the sample paths of underlying
  assets, and a user supplied payoff function it attempts to find the optimal
  exercise point along each sample path. With optimal exercise points known,
  the option is valued as the average payoff assuming optimal exercise
  discounted to present value.

  #### Example. American put option price through Monte Carlo
  ```python
  # Let the underlying model be a Black-Scholes process
  # dS_t / S_t = rate dt + sigma**2 dW_t, S_0 = 1.0
  # with `rate = 0.1`, and volatility `sigma = 1.0`.
  # Define drift and volatility functions for log(S_t)
  rate = 0.1
  def drift_fn(_, x):
    return rate - tf.ones_like(x) / 2.
  def vol_fn(_, x):
    return tf.expand_dims(tf.ones_like(x), axis=-1)
  # Use Euler scheme to propagate 100000 paths for 1 year into the future
  times = np.linspace(0., 1, num=50)
  num_samples = 100000
  log_paths = tf.function(tff.models.euler_sampling.sample)(
          dim=1,
          drift_fn=drift_fn, volatility_fn=vol_fn,
          random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
          times=times, num_samples=num_samples, seed=42, time_step=0.01)
  # Compute exponent to get samples of `S_t`
  paths = tf.math.exp(log_paths)
  # American put option price for strike 1.1 and expiry 1 (assuming actual day
  # count convention and no settlement adjustment)
  strike = [1.1]
  exercise_times = tf.range(times.shape[-1])
  discount_factors = tf.exp(-rate * times)
  payoff_fn = make_basket_put_payoff(strike)
  basis_fn = make_polynomial_basis(10)
  least_square_mc(paths, exercise_times, payoff_fn, basis_fn,
                  discount_factors=discount_factors)
  # Expected value: [0.397]
  # European put option price
  tff.black_scholes.option_price(volatilities=[1], strikes=strikes,
                                 expiries=[1], spots=[1.],
                                 discount_factors=discount_factors[-1],
                                 is_call_options=False,
                                 dtype=tf.float64)
  # Expected value: [0.379]
  ```
  #### References

  [1] Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
  simulation: a simple least-squares approach. The review of financial studies,
  14(1), pp.113-147.

  Args:
    sample_paths: A `Tensor` of either shape `[num_samples, num_times, dim]` or
      `[batch_size, num_samples, num_times, dim]`, the sample paths of the
      underlying ito process of dimension `dim` at `num_times` different points.
      The `batch_size` allows multiple options to be valued in parallel.
    exercise_times: An `int32` `Tensor` of shape `[num_exercise_times]`.
      Contents must be a subset of the integers `[0,...,num_times - 1]`,
      representing the time indices at which the option may be exercised.
    payoff_fn: Callable from a `Tensor` of shape `[num_samples, S, dim]`
      (where S <= num_times) to a `Tensor` of shape `[num_samples, batch_size]`
      of the same dtype as `samples`. The output represents the payout resulting
      from exercising the option at time `S`. The `batch_size` allows multiple
      options to be valued in parallel.
    basis_fn: Callable from a `Tensor` of the same shape and `dtype` as
      `sample_paths` and a positive integer `Tenor` (representing a current
      time index) to a `Tensor` of shape `[batch_size, basis_size, num_samples]`
      of the same dtype as `sample_paths`. The result being the design matrix
      used in regression of the continuation value of options.
    discount_factors: A `Tensor` of shape `[num_exercise_times]` or of rank 3
      and compatible with `[num_samples, batch_size, num_exercise_times]`.
      The `dtype` should be the same as of `samples`.
      Default value: `None` which maps to a one-`Tensor` of the same `dtype`
        as `samples` and shape `[num_exercise_times]`.
    num_calibration_samples: An optional integer less or equal to `num_samples`.
      The number of sampled trajectories used for the LSM regression step.
      Note that only the last`num_samples - num_calibration_samples` of the
      sampled paths are used to determine the price of the option.
      Default value: `None`, which means that all samples are used for
        regression and option pricing.
    dtype: Optional `dtype`. Either `tf.float32` or `tf.float64`. The `dtype`
      If supplied, represents the `dtype` for the input and output `Tensor`s.
      Default value: `None`, which means that the `dtype` inferred by TensorFlow
      is used.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` which is mapped to the default name
      'least_square_mc'.

  Returns:
    A `Tensor` of shape `[num_samples, batch_size]` of the same dtype as
    `samples`.
  """
  name = name or "least_square_mc"
  with tf.name_scope(name):
    # Conversion of the inputs to tensors
    sample_paths = tf.convert_to_tensor(sample_paths,
                                        dtype=dtype, name="sample_paths")
    dtype = sample_paths.dtype
    exercise_times = tf.convert_to_tensor(exercise_times, name="exercise_times")
    num_times = exercise_times.shape.as_list()[-1]
    if discount_factors is None:
      discount_factors = tf.ones(shape=exercise_times.shape,
                                 dtype=dtype,
                                 name="discount_factors")
    else:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name="discount_factors")
    if discount_factors.shape.rank == 1:
      discount_factors = tf.reshape(discount_factors, [1, 1, -1])

    rank_discount_factors = discount_factors.shape.rank
    discount_factors = tf.pad(
        discount_factors, (rank_discount_factors - 1) * [[0, 0]] + [[1, 0]],
        constant_values=1)
    # Initialise cashflow as the payoff at final sample.
    time_index = exercise_times[num_times - 1]
    # Calculate the payoff of each path if exercised now. Shape
    # [num_samples, batch_size]
    exercise_value = payoff_fn(sample_paths, time_index)
    zeros = tf.zeros(exercise_value.shape + [num_times - 1],
                     dtype=dtype)
    exercise_value = tf.expand_dims(exercise_value, axis=-1)

    # Shape [num_samples, batch_size, num_exercise]
    cashflow = tf.concat([zeros, exercise_value], axis=-1)
    calibration_indices = None
    if num_calibration_samples is not None:
      # Calibration indices to perform regression
      calibration_indices = tf.range(num_calibration_samples)
    # Starting state for loop iteration.
    lsm_loop_vars = LsmLoopVars(exercise_index=num_times - 1, cashflow=cashflow)
    def loop_body(exercise_index, cashflow):
      return _lsm_loop_body(
          sample_paths=sample_paths,
          exercise_times=exercise_times,
          discount_factors=discount_factors,
          payoff_fn=payoff_fn,
          basis_fn=basis_fn,
          num_times=num_times,
          exercise_index=exercise_index,
          cashflow=cashflow,
          calibration_indices=calibration_indices)

    max_iterations = tf.shape(exercise_times)[-1]
    loop_value = tf.while_loop(_lsm_loop_cond, loop_body, lsm_loop_vars,
                               maximum_iterations=max_iterations)
    present_values = _continuation_value_fn(
        loop_value.cashflow, discount_factors, 0)
    if num_calibration_samples is not None:
      # Skip num_calibration_samples to reduce bias
      present_values = present_values[num_calibration_samples:]
    return tf.math.reduce_mean(present_values, axis=0)


def _lsm_loop_cond(exercise_index, cashflow):
  """Condition to exit a countdown loop when the exercise date hits zero."""
  del cashflow
  return exercise_index > 0


def _continuation_value_fn(cashflow, discount_factors, exercise_index):
  """Returns the discounted value of the right hand part of the cashflow tensor.

  Args:
    cashflow: A real `Tensor` of shape
      `[num_samples, batch_size, num_exercise]`. Tracks the optimal cashflow of
       each sample path for each payoff dimension at each exercise time.
    discount_factors: A `Tensor` of shape `[num_exercise_times]` or of rank 3
      and compatible with `[num_samples, batch_size, num_exercise_times]`.
      The `dtype` should be the same as of `samples`
    exercise_index: An integer scalar `Tensor` representing the index of the
      exercise time of interest. Should be less than `num_exercise_times`.

  Returns:
    A `[num_samples, batch_size]` `Tensor` whose entries represent the sum of
    those elements to the right of `exercise_index` in `cashflow`, discounted to
    the time indexed by `exercise_index`. When `exercise_index` is zero, the
    return represents the sum of the cashflow discounted to present value for
    each sample path.
  """
  _, _, num_cashflow = cashflow.shape.as_list()
  disc_factors_are_used = tf.range(num_cashflow + 1) >= exercise_index + 1
  discount_factors_slice = tf.transpose(
      tf.transpose(discount_factors)[exercise_index])
  total_discount_factors = (
      tf.where(disc_factors_are_used, discount_factors, 0)
      / tf.expand_dims(discount_factors_slice, axis=-1))
  cashflows_are_used = tf.range(num_cashflow) >= exercise_index
  cashflow_masked = tf.where(cashflows_are_used, cashflow, 0)
  total_discount_factors_slice = tf.transpose(
      tf.transpose(total_discount_factors)[1:])
  return tf.math.reduce_sum(
      cashflow_masked * total_discount_factors_slice, axis=2)


def _expected_exercise_fn(
    design, calibration_indices, continuation_value, exercise_value):
  """Returns the expected continuation value for each path.

  Args:
    design: A real `Tensor` of shape `[batch_size, basis_size, num_samples]`.
    calibration_indices: A rank 1 integer `Tensor` denoting indices of samples
      used for regression.
    continuation_value: A `Tensor` of shape `[num_samples, batch_size]` and of
      the same dtype as `design`. The optimal value of the option conditional on
      not exercising now or earlier, taking future information into account.
    exercise_value: A `Tensor` of the same shape and dtype as
      `continuation_value`. Value of the option if exercised immideately at
      the current time

  Returns:
    A `Tensor` of the same shape and dtype as `continuation_value` whose
    `(n, v)`-th entry represents the expected continuation value of sample path
    `n` under the `v`-th payoff scheme.
  """
  # Zero out contributions from samples we'd never exercise at this point (i.e.,
  # these extra observations do not change the regression coefficients).
  mask = exercise_value > 0
  # Shape [batch_size, num_samples, basis_size]
  design_t = tf.transpose(design, [0, 2, 1])
  masked = tf.where(
      tf.expand_dims(tf.transpose(mask), axis=-1),
      design_t, tf.zeros_like(design_t))
  if calibration_indices is None:
    submask = masked
    mask_cont_value = continuation_value
  else:
    # Shape [batch_size, num_calibration_samples, basis_size]
    submask = tf.gather(masked, calibration_indices, axis=1)
    # Shape [num_calibration_samples, batch_size]
    mask_cont_value = tf.gather(continuation_value, calibration_indices)
  # For design matrix X and response y, the coefficients beta of the best linear
  # unbiased estimate are contained in the equation X'X beta = X'y. Here `lhs`
  # is X'X and `rhs` is X'y, or rather a tensor of such left hand and right hand
  # sides, one for each payoff dimension.
  # Shape [batch_size, basis_size, basis_size]
  lhs = tf.matmul(submask, submask, transpose_a=True)
  # Use pseudo inverse for the regression matrix to ensure stability of the
  # algorithm.
  lhs_pinv = tf.linalg.pinv(lhs)
  # Shape [batch_size, basis_size, 1]
  rhs = tf.matmul(
      submask,
      tf.expand_dims(tf.transpose(mask_cont_value), axis=-1),
      transpose_a=True)
  # Shape [batch_size, basis_size, 1]
  beta = tf.matmul(lhs_pinv, rhs)
  # Shape [batch_size, num_samples, 1]
  continuation = tf.matmul(design_t, beta)
  # Shape [num_samples, batch_size]
  return tf.nn.relu(tf.transpose(tf.squeeze(continuation, axis=-1)))


def _updated_cashflow(num_times, exercise_index, exercise_value,
                      expected_continuation, cashflow):
  """Revises the cashflow tensor where options will be exercised earlier."""
  # Shape [num_samples, batch_size]
  do_exercise_bool = exercise_value > expected_continuation
  # Shape [num_samples, batch_size]
  scaled_do_exercise = tf.where(do_exercise_bool, exercise_value,
                                0)
  # This picks out the samples where we now wish to exercise.
  # Shape [num_samples, batch_size, 1]
  new_samp_masked = tf.expand_dims(scaled_do_exercise, axis=2)
  # This is an array with nonzero entries showing newly exercised payoffs.
  # Has shape [num_samples, batch_size, 1]
  cashflow_update = tf.where(tf.expand_dims(do_exercise_bool, axis=-1),
                             tf.constant(0, dtype=cashflow.dtype),
                             cashflow)
  new_mask = tf.range(0, num_times) >= exercise_index
  # Shape [num_samples, batch_size, num_times]
  return tf.where(new_mask, cashflow_update, new_samp_masked)


def _lsm_loop_body(
    sample_paths, exercise_times, discount_factors, payoff_fn,
    basis_fn, num_times, exercise_index, cashflow, calibration_indices):
  """Finds the optimal exercise point and updates `cashflow`."""

  # Index of the sample path that the exercise index maps to.
  time_index = exercise_times[exercise_index - 1]
  # Calculate the payoff of each path if exercised now.
  # Shape [num_samples, batch_size]
  exercise_value = payoff_fn(sample_paths, time_index)
  # Present value of hanging on to the options (using future information).
  # Shape `[num_samples, batch_size]`
  continuation_value = _continuation_value_fn(cashflow, discount_factors,
                                              exercise_index)
  # Create a design matrix for regression based on the sample paths.
  # Shape `[batch_size, num_samples, basis_size]`
  design = basis_fn(sample_paths, time_index)

  # Expected present value of hanging on the options.
  expected_continuation = _expected_exercise_fn(
      design, calibration_indices, continuation_value, exercise_value)
  # Update the cashflow matrix to reflect where earlier exercise is optimal.
  # Shape `[num_samples, batch_size, num_exercise]`.
  rev_cash = _updated_cashflow(num_times, exercise_index, exercise_value,
                               expected_continuation,
                               cashflow)
  return LsmLoopVars(exercise_index=exercise_index - 1, cashflow=rev_cash)
