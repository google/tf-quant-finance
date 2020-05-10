# Lint as: python3
# Copyright 2019 Google LLC
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

"""Implementation of the regression MC algorithm of Longstaff and Schwartz."""

import collections
import tensorflow.compat.v2 as tf


LsmLoopVars = collections.namedtuple(
    'LsmLoopVars',
    [
        # int32. The LSM algorithm iterates backwards over times where an option
        # can be exercised, this tracks progress.
        'exercise_index',
        # [num_samples, num_exercise_times, payoff_dim] shaped tensor. Tracks
        # the optimal cashflow of each sample path for each payoff dimension at
        # each exercise time.
        'cashflow'
    ])


def make_polynomial_basis(degree):
  """Produces a callable from samples to polynomial basis for use in regression.

  The output callable accepts a `Tensor` `X` of shape `[num_samples, dim]`,
  computes a centered value `Y = X - mean(X, axis=0)` and outputs a `Tensor`
  of shape `[degree * dim, num_samples]`, where
  ```
  Z[i*j, k] = X[k, j]**(degree - i) * X[k, j]**i, 0<=i<degree - 1, 0<=j<dim
  ```
  For example, if `degree` and `dim` are both equal to 2, the polynomial basis
  is `1, X, X**2, Y, Y**2, X * Y, X**2 * Y, X * Y**2`, where `X` and `Y` are
  the spatial axes.

  #### Example
  ```python
  basis = make_polynomial_basis(2)
  x = [1.0, 2.0, 3.0, 4.0]
  x = tf.expand_dims(x, -1)
  basis(x)
  # Expected result:
  [[ 1.0, 1.0, 1.0, 1.0], [-1.5, -0.5, 0.5, 1.5]]
  ```

  Args:
    degree: An `int32` scalar `Tensor`. The degree of the desired polynomial
      basis.

  Returns:
    A callable from `Tensor`s of shape `[num_samples, dim]` to `Tensor`s of
    shape `[degree * dim, num_samples]`.

  Raises:
    ValueError: If `degree` is less than `1`.
  """
  tf.debugging.assert_greater_equal(
      degree, 0,
      message='Degree of polynomial basis can not be negative.')
  def basis(sample_paths):
    """Computes polynomial basis expansion at the given sample points.

    Args:
      sample_paths: A `Tensor`s of either `flot32` or `float64` dtype and of
        shape `[num_samples, dim]` where `dim` has to be statically known.

    Returns:
      A `Tensor`s of shape `[degree * dim, num_samples]`.
    """
    samples = tf.convert_to_tensor(sample_paths)
    dim = samples.shape.as_list()[-1]
    grid = tf.range(0, degree + 1, dtype=samples.dtype)

    samples_centered = samples - tf.math.reduce_mean(samples, axis=0)
    samples_centered = tf.expand_dims(samples_centered, -2)
    grid = tf.meshgrid(*(dim * [grid]))
    grid = tf.reshape(tf.stack(grid, -1), [-1, dim])
    # Shape [num_samples, degree * dim]
    basis_expansion = tf.reduce_prod(samples_centered**grid, -1)
    return  tf.transpose(basis_expansion)
  return basis


def least_square_mc(sample_paths,
                    exercise_times,
                    payoff_fn,
                    basis_fn,
                    discount_factors=None,
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
    return tf.expand_dims(tf.ones_like(x), -1)
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
  lsm_price(paths, exercise_times, payoff_fn, basis_fn,
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
    sample_paths: A `Tensor` of shape `[num_samples, num_times, dim]`, the
      sample paths of the underlying ito process of dimension `dim` at
      `num_times` different points.
    exercise_times: An `int32` `Tensor` of shape `[num_exercise_times]`.
      Contents must be a subset of the integers `[0,...,num_times - 1]`,
      representing the ticks at which the option may be exercised.
    payoff_fn: Callable from a `Tensor` of shape `[num_samples, num_times, dim]`
      and an integer scalar positive `Tensor` (representing the current time
      index) to a `Tensor` of shape `[num_samples, payoff_dim]`
      of the same dtype as `samples`. The output represents the payout resulting
      from exercising the option at time `S`. The `payoff_dim` allows multiple
      options on the same underlying asset (i.e., `samples`) to be valued in
      parallel.
    basis_fn: Callable from a `Tensor` of shape `[num_samples, dim]` to a
      `Tensor` of shape `[basis_size, num_samples]` of the same dtype as
      `samples`. The result being the design matrix used in regression of the
      continuation value of options.
    discount_factors: A `Tensor` of shape `[num_exercise_times]` and the same
      `dtype` as `samples`, the k-th element of which represents the discount
      factor at time tick `k`.
      Default value: `None` which maps to a one-`Tensor` of the same `dtype`
        as `samples` and shape `[num_exercise_times]`.
    dtype: Optional `dtype`. Either `tf.float32` or `tf.float64`. The `dtype`
      If supplied, represents the `dtype` for the input and output `Tensor`s.
      Default value: `None`, which means that the `dtype` inferred by TensorFlow
      is used.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` which is mapped to the default name
      'least_square_mc'.
  Returns:
    A `Tensor` of shape `[num_samples, payoff_dim]` of the same dtype as
    `samples`.
  """
  with tf.compat.v1.name_scope(name, default_name='least_square_mc',
                               values=[sample_paths, exercise_times]):
    # Conversion of the inputs to tensors
    sample_paths = tf.convert_to_tensor(sample_paths,
                                        dtype=dtype, name='sample_paths')
    exercise_times = tf.convert_to_tensor(exercise_times, name='exercise_times')
    num_times = exercise_times.shape.as_list()[-1]
    if discount_factors is None:
      discount_factors = tf.ones(shape=exercise_times.shape,
                                 dtype=sample_paths.dtype,
                                 name='discount_factors')
    else:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')
    discount_factors = tf.concat([[1], discount_factors], -1)
    # Initialise cashflow as the payoff at final sample.
    tick = exercise_times[num_times - 1]
    # Calculate the payoff of each path if exercised now. Shape
    # [num_samples, payoff_dim]
    exercise_value = payoff_fn(sample_paths, tick)
    zeros = tf.zeros(exercise_value.shape + [num_times - 1],
                     dtype=exercise_value.dtype)
    exercise_value = tf.expand_dims(exercise_value, -1)

    # Shape [num_samples, payoff_dim, num_exercise]
    cashflow = tf.concat([zeros, exercise_value], -1)
    # Starting state for loop iteration.
    lsm_loop_vars = LsmLoopVars(exercise_index=num_times - 1, cashflow=cashflow)
    def loop_body(exercise_index, cashflow):
      return _lsm_loop_body(sample_paths, exercise_times, discount_factors,
                            payoff_fn, basis_fn,
                            num_times, exercise_index, cashflow)

    loop_value = tf.while_loop(lsm_loop_cond, loop_body, lsm_loop_vars,
                               maximum_iterations=num_times)
    present_values = continuation_value_fn(
        loop_value.cashflow, discount_factors, 0)
    return tf.math.reduce_mean(present_values, axis=0)


def lsm_loop_cond(exercise_index, cashflow):
  """Condition to exit a countdown loop when the exercise date hits zero."""
  del cashflow
  return exercise_index > 0


def continuation_value_fn(cashflow, discount_factors, exercise_index):
  """Returns the discounted value of the right hand part of the cashflow tensor.

  Args:
    cashflow: A real `Tensor` of shape
      `[num_samples, payoff_dim, num_exercise]`. Tracks the optimal cashflow of
       each sample path for each payoff dimension at each exercise time.
    discount_factors: A `Tensor` of shape `[num_exercise_times + 1]` and the
      same `dtype` as `samples`, the `k`-th element of which represents the
      discount factor at time tick `k + 1`. `discount_factors[0]` is `1` which
      is the discount factor at time `0`.
    exercise_index: An integer scalar `Tensor` representing the index of the
      exercise time of interest. Should be less than `num_exercise_times`.

  Returns:
    A `[num_samples, payoff_dim]` `Tensor` whose entries represent the sum of
    those elements to the right of `exercise_index` in `cashflow`, discounted to
    the time indexed by `exercise_index`. When `exercise_index` is zero, the
    return represents the sum of the cashflow discounted to present value for
    each sample path.
  """
  total_discount_factors = (discount_factors[exercise_index + 1:]
                            / discount_factors[exercise_index])
  return tf.math.reduce_sum(
      cashflow[..., exercise_index:] * total_discount_factors, axis=2)


def expected_exercise_fn(design, continuation_value, exercise_value):
  """Returns the expected continuation value for each path.

  Args:
    design: A real `Tensor` of shape `[basis_size, num_samples]`.
    continuation_value: A `Tensor` of shape `[num_samples, payoff_dim]` and of
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
  # We wish to value each option under different payoffs, expressed through a
  # multidimensional payoff function. While the basis calculated from the sample
  # paths is the same for each payoff, the LSM algorithm requires us to fit a
  # regression model only on the in-the-money paths, which are payoff dependent,
  # hence we create multiple copies of the regression design (basis) matrix and
  # zero out rows for out of the money paths under each payoff.
  batch_design = tf.broadcast_to(
      tf.expand_dims(design, -1), design.shape + [continuation_value.shape[-1]])
  mask = tf.cast(exercise_value > 0, design.dtype)
  # Zero out contributions from samples we'd never exercise at this point (i.e.,
  # these extra observations do not change the regression coefficients).
  masked = tf.transpose(batch_design * mask, perm=(2, 1, 0))
  # For design matrix X and response y, the coefficients beta of the best linear
  # unbiased estimate are contained in the equation X'X beta = X'y. Here `lhs`
  # is X'X and `rhs` is X'y, or rather a tensor of such left hand and right hand
  # sides, one for each payoff dimension.
  lhs = tf.matmul(masked, masked, transpose_a=True)
  # Use pseudo inverse for the regression matrix to ensure stability of the
  # algorithm.
  lhs_pinv = tf.linalg.pinv(lhs)
  rhs = tf.matmul(
      masked,
      tf.expand_dims(tf.transpose(continuation_value), -1),
      transpose_a=True)
  beta = tf.linalg.matmul(lhs_pinv, rhs)
  continuation = tf.matmul(tf.transpose(batch_design, perm=(2, 1, 0)), beta)
  return tf.maximum(tf.transpose(tf.squeeze(continuation, -1)), 0.0)


def _updated_cashflow(num_times, exercise_index, exercise_value,
                      expected_continuation, cashflow):
  """Revises the cashflow tensor where options will be exercised earlier."""
  do_exercise_bool = exercise_value > expected_continuation
  do_exercise = tf.cast(do_exercise_bool, exercise_value.dtype)
  # Shape [num_samples, payoff_dim]
  scaled_do_exercise = tf.where(do_exercise_bool, exercise_value,
                                tf.zeros_like(exercise_value))
  # This picks out the samples where we now wish to exercise.
  # Shape [num_samples, payoff_dim, 1]
  new_samp_masked = tf.expand_dims(scaled_do_exercise, 2)
  # This should be one on the current time step and zero otherwise.
  # This is an array with nonzero entries showing newly exercised payoffs.
  pad_shape = scaled_do_exercise.shape.as_list()
  zeros_before = tf.zeros(pad_shape + [exercise_index - 1],
                          dtype=scaled_do_exercise.dtype)
  zeros_after = tf.zeros(pad_shape + [num_times - exercise_index],
                         dtype=scaled_do_exercise.dtype)
  new_cash = tf.concat([zeros_before, new_samp_masked, zeros_after], -1)

  # Has shape [num_samples, payoff_dim, 1]
  old_samp_masker = tf.expand_dims(1 - do_exercise, 2)
  # Broadcast to shape [num_samples, payoff_dim, num_times - exercise_index]
  old_samp_masker_after = tf.broadcast_to(
      old_samp_masker, pad_shape + [num_times - exercise_index])
  # Has shape `[num_samples, payoff_dim, exercise_index]`
  zeros_before = tf.zeros(pad_shape + [exercise_index],
                          dtype=scaled_do_exercise.dtype)
  # Shape [num_samples, payoff_dim, num_times]
  old_mask = tf.concat([zeros_before,
                        old_samp_masker_after], -1)
  # Shape [num_samples, payoff_dim, num_times]
  old_cash = old_mask * cashflow
  return new_cash + old_cash


def _lsm_loop_body(sample_paths, exercise_times, discount_factors, payoff_fn,
                   basis_fn, num_times, exercise_index, cashflow):
  """Finds the optimal exercise point and updates `cashflow`."""

  # Index of the sample path that the exercise index maps to.
  tick = exercise_times[exercise_index - 1]
  # Calculate the payoff of each path if exercised now.
  # Shape [num_samples, payoff_dim]
  exercise_value = payoff_fn(sample_paths, tick)
  # Present value of hanging on to the options (using future information).
  # Shape `[num_samples, payoff_dim]`
  continuation_value = continuation_value_fn(cashflow, discount_factors,
                                             exercise_index)
  # Create a design matrix for regression based on the sample paths.
  # Shape `[num_samples, basis_size]`
  design = basis_fn(sample_paths[:, tick, :])
  # Expected present value of hanging on the options.
  # Shape `[num_samples, payoff_dim]`
  expected_continuation = expected_exercise_fn(design, continuation_value,
                                               exercise_value)
  # Update the cashflow matrix to reflect where earlier exercise is optimal.
  # Shape `[num_samples, payoff_dim, num_exercise]`
  rev_cash = _updated_cashflow(num_times, exercise_index, exercise_value,
                               expected_continuation,
                               cashflow)
  rev_cash.set_shape(cashflow.shape)
  return LsmLoopVars(exercise_index=exercise_index - 1, cashflow=rev_cash)
