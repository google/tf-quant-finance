# Lint as: python3
# Copyright 2021 Google LLC
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

"""Root search functions."""


import collections

import tensorflow.compat.v2 as tf

from tf_quant_finance.math.root_search import utils

# TODO(b/179451420): Refactor BrentResults as RootSearchResults and return it
# for newton method as well.
BrentResults = collections.namedtuple(
    "BrentResults",
    [
        # A tensor containing the best estimate. If the search was successful,
        # this estimate is a root of the objective function.
        "estimated_root",
        # A tensor containing the value of the objective function at the best
        # estimate. If the search was successful, then this is close to 0.
        "objective_at_estimated_root",
        # A tensor containing number of iterations performed for each pair of
        # starting points.
        "num_iterations",
        # Scalar boolean tensor indicating whether the best estimate is a root
        # within the tolerance specified for the search.
        "converged",
    ])

# Values which remain fixed across all root searches (except for tensor dtypes
# and shapes).
_BrentSearchConstants = collections.namedtuple("_BrentSearchConstants", [
    "false",
    "zero",
    "zero_value",
])

# Values which are updated during the root search.
_BrentSearchState = collections.namedtuple("_BrentSearchState", [
    "best_estimate",
    "value_at_best_estimate",
    "last_estimate",
    "value_at_last_estimate",
    "contrapoint",
    "value_at_contrapoint",
    "step_to_best_estimate",
    "step_to_last_estimate",
    "num_iterations",
    "finished",
])

# Values which remain fixed for a given root search.
_BrentSearchParams = collections.namedtuple("_BrentSearchParams", [
    "objective_fn",
    "max_iterations",
    "absolute_root_tolerance",
    "relative_root_tolerance",
    "function_tolerance",
    "stopping_policy_fn",
])


def _swap_where(condition, x, y):
  """Swaps the elements of `x` and `y` based on `condition`.

  Args:
    condition: A `Tensor` of dtype bool.
    x: A `Tensor` with the same shape as `condition`.
    y: A `Tensor` with the same shape and dtype as `x`.

  Returns:
    Two `Tensors` with the same shape as `x` and `y`.
  """
  return tf.where(condition, y, x), tf.where(condition, x, y)


def _secant_step(x1, x2, y1, y2):
  """Returns the step size at the current position if using the secant method.

  This function is meant for exclusive use by the `_brent_loop_body` function:
  - It does not guard against divisions by zero, and instead assumes that `y1`
    is distinct from `y2`. The `_brent_loop_body` function guarantees this
    property.
  - It does not guard against overflows which may occur if the difference
    between `y1` and `y2` is small while that between `x1` and `x2` is not.
    In this case, the resulting step size will be larger than `bisection_step`
    and thus ignored by the `_brent_loop_body` function.

  Args:
    x1: `Tensor` containing the current position.
    x2: `Tensor` containing the previous position.
    y1: `Tensor` containing the value of `objective_fn` at `x1`.
    y2: `Tensor` containing the value of `objective_fn` at `x2`.

  Returns:
    A `Tensor` with the same shape and dtype as `current`.
  """
  x_difference = x1 - x2
  y_difference = y1 - y2
  return -y1 * x_difference / y_difference


def _quadratic_interpolation_step(x1, x2, x3, y1, y2, y3):
  """Returns the step size to use when using quadratic interpolation.

  This function is meant for exclusive use by the `_brent_loop_body` function.
  It does not guard against divisions by zero, and instead assumes that `y1` is
  distinct from `y2` and `y3`. The `_brent_loop_body` function guarantees this
  property.

  Args:
    x1: `Tensor` of any shape and real dtype containing the first position used
      for extrapolation.
    x2: `Tensor` of the same shape and dtype as `x1` containing the second
      position used for extrapolation.
    x3: `Tensor` of the same shape and dtype as `x1` containing the third
      position used for extrapolation.
    y1: `Tensor` containing the value of the interpolated function at `x1`.
    y2: `Tensor` containing the value of interpolated function at `x2`.
    y3: `Tensor` containing the value of interpolated function at `x3`.

  Returns:
    A `Tensor` with the same shape and dtype as `x1`.
  """
  r2 = (x2 - x1) / (y2 - y1)
  r3 = (x3 - x1) / (y3 - y1)
  return -x1 * tf.math.divide_no_nan(x3 * r3 - x2 * r2, r3 * r2 * (x3 - x2))


def _should_stop(state, stopping_policy_fn):
  """Indicates whether the overall Brent search should continue.

  Args:
    state: A Python `_BrentSearchState` namedtuple.
    stopping_policy_fn: Python `callable` controlling the algorithm termination.

  Returns:
    A boolean value indicating whether the overall search should continue.
  """
  return tf.convert_to_tensor(
      stopping_policy_fn(state.finished), name="should_stop", dtype=tf.bool)


# This is a direct translation of the Brent root-finding method.
# Each operation is guarded by a call to `tf.where` to avoid performing
# unnecessary calculations.
def _brent_loop_body(state, params, constants):
  """Performs one iteration of the Brent root-finding algorithm.

  Args:
    state: A Python `_BrentSearchState` namedtuple.
    params: A Python `_BrentSearchParams` namedtuple.
    constants: A Python `_BrentSearchConstants` namedtuple.

  Returns:
    The `Tensor`s to use for the next iteration of the algorithm.
  """

  best_estimate = state.best_estimate
  last_estimate = state.last_estimate
  contrapoint = state.contrapoint
  value_at_best_estimate = state.value_at_best_estimate
  value_at_last_estimate = state.value_at_last_estimate
  value_at_contrapoint = state.value_at_contrapoint
  step_to_best_estimate = state.step_to_best_estimate
  step_to_last_estimate = state.step_to_last_estimate
  num_iterations = state.num_iterations
  finished = state.finished

  # If the root is between the last two estimates, use the worst of the two
  # as new contrapoint. Adjust step sizes accordingly.
  replace_contrapoint = ~finished & (
      value_at_last_estimate * value_at_best_estimate < constants.zero_value)

  contrapoint = tf.where(replace_contrapoint, last_estimate, contrapoint)
  value_at_contrapoint = tf.where(replace_contrapoint, value_at_last_estimate,
                                  value_at_contrapoint)

  step_to_last_estimate = tf.where(replace_contrapoint,
                                   best_estimate - last_estimate,
                                   step_to_last_estimate)
  step_to_best_estimate = tf.where(replace_contrapoint, step_to_last_estimate,
                                   step_to_best_estimate)

  # If the contrapoint is a better guess than the current root estimate, swap
  # them. Also, replace the worst of the two with the current contrapoint.
  replace_best_estimate = tf.where(
      finished, constants.false,
      tf.math.abs(value_at_contrapoint) < tf.math.abs(value_at_best_estimate))

  last_estimate = tf.where(replace_best_estimate, best_estimate, last_estimate)
  best_estimate = tf.where(replace_best_estimate, contrapoint, best_estimate)
  contrapoint = tf.where(replace_best_estimate, last_estimate, contrapoint)

  value_at_last_estimate = tf.where(replace_best_estimate,
                                    value_at_best_estimate,
                                    value_at_last_estimate)
  value_at_best_estimate = tf.where(replace_best_estimate, value_at_contrapoint,
                                    value_at_best_estimate)
  value_at_contrapoint = tf.where(replace_best_estimate, value_at_last_estimate,
                                  value_at_contrapoint)

  # Compute the tolerance used to control root search at the current position
  # and the step size corresponding to the bisection method.
  root_tolerance = 0.5 * (
      params.absolute_root_tolerance +
      params.relative_root_tolerance * tf.math.abs(best_estimate))
  bisection_step = 0.5 * (contrapoint - best_estimate)

  # Mark the search as finished if either:
  # 1. the maximum number of iterations has been reached;
  # 2. the desired tolerance has been reached (even if no root was found);
  # 3. the current root estimate is good enough.
  # Using zero as `function_tolerance` will check for exact roots and match
  # both Brent's original algorithm and the SciPy implementation.
  finished |= (num_iterations >= params.max_iterations) | (
      tf.math.abs(bisection_step) <
      root_tolerance) | (~tf.math.is_finite(value_at_best_estimate)) | (
          tf.math.abs(value_at_best_estimate) <= params.function_tolerance)

  # Determine whether interpolation or extrapolation are worth performing at
  # the current position.
  compute_short_step = tf.where(
      finished, constants.false,
      (root_tolerance < tf.math.abs(step_to_last_estimate)) &
      (tf.math.abs(value_at_best_estimate) <
       tf.math.abs(value_at_last_estimate)))

  short_step = tf.where(
      compute_short_step,
      tf.where(
          # The contrapoint cannot be equal to the current root estimate since
          # they have opposite signs. However, it may be equal to the previous
          # estimate.
          tf.equal(last_estimate, contrapoint),
          # If so, use the secant method to avoid a division by zero which
          # would occur if using extrapolation.
          _secant_step(best_estimate, last_estimate, value_at_best_estimate,
                       value_at_last_estimate),
          # Pass values of the objective function as x values, and root
          # estimates as y values in order to perform *inverse* extrapolation.
          _quadratic_interpolation_step(value_at_best_estimate,
                                        value_at_last_estimate,
                                        value_at_contrapoint, best_estimate,
                                        last_estimate, contrapoint)),
      # Default to zero if using bisection.
      constants.zero)

  # Use the step calculated above if both:
  # 1. step size < |previous step size|
  # 2. step size < 3/4 * |contrapoint - current root estimate|
  # Ensure that `short_step` was calculated by guarding the calculation with
  # `compute_short_step`.
  use_short_step = tf.where(
      compute_short_step, 2 * tf.math.abs(short_step) < tf.minimum(
          3 * tf.math.abs(bisection_step) - root_tolerance,
          tf.math.abs(step_to_last_estimate)), constants.false)

  # Revert to bisection when not using `short_step`.
  step_to_last_estimate = tf.where(use_short_step, step_to_best_estimate,
                                   bisection_step)
  step_to_best_estimate = tf.where(
      finished, constants.zero,
      tf.where(use_short_step, short_step, bisection_step))

  # Update the previous and current root estimates.
  last_estimate = tf.where(finished, last_estimate, best_estimate)
  best_estimate += tf.where(
      finished, constants.zero,
      tf.where(root_tolerance < tf.math.abs(step_to_best_estimate),
               step_to_best_estimate,
               tf.where(bisection_step > 0, root_tolerance, -root_tolerance)))

  value_at_last_estimate = tf.where(finished, value_at_last_estimate,
                                    value_at_best_estimate)
  value_at_best_estimate = tf.where(finished, value_at_best_estimate,
                                    params.objective_fn(best_estimate))

  num_iterations = tf.where(finished, num_iterations, num_iterations + 1)

  return [
      _BrentSearchState(
          best_estimate=best_estimate,
          last_estimate=last_estimate,
          contrapoint=contrapoint,
          value_at_best_estimate=value_at_best_estimate,
          value_at_last_estimate=value_at_last_estimate,
          value_at_contrapoint=value_at_contrapoint,
          step_to_best_estimate=step_to_best_estimate,
          step_to_last_estimate=step_to_last_estimate,
          num_iterations=num_iterations,
          finished=finished)
  ]


def _prepare_brent_args(objective_fn,
                        left_bracket,
                        right_bracket,
                        value_at_left_bracket,
                        value_at_right_bracket,
                        absolute_root_tolerance=2e-7,
                        relative_root_tolerance=None,
                        function_tolerance=2e-7,
                        max_iterations=100,
                        stopping_policy_fn=None):
  r"""Prepares arguments for root search using Brent's method.

  Args:
    objective_fn: Python callable for which roots are searched. It must be a
      callable of a single `Tensor` parameter and return a `Tensor` of the same
      shape and dtype as `left_bracket`.
    left_bracket: `Tensor` or Python float representing the first starting
      points. The function will search for roots between each pair of points
      defined by `left_bracket` and `right_bracket`. The shape of `left_bracket`
      should match that of the input to `objective_fn`.
    right_bracket: `Tensor` of the same shape and dtype as `left_bracket` or
      Python float representing the second starting points. The function will
      search for roots between each pair of points defined by `left_bracket` and
      `right_bracket`. This argument must have the same shape as `left_bracket`.
    value_at_left_bracket: Optional `Tensor` or Python float representing the
      value of `objective_fn` at `left_bracket`. If specified, this argument
      must have the same shape as `left_bracket`. If not specified, the value
      will be evaluated during the search.
      Default value: None.
    value_at_right_bracket: Optional `Tensor` or Python float representing the
      value of `objective_fn` at `right_bracket`. If specified, this argument
      must have the same shape as `right_bracket`. If not specified, the value
      will be evaluated during the search.
      Default value: None.
    absolute_root_tolerance: Optional `Tensor` representing the absolute
      tolerance for estimated roots, with the total tolerance being calculated
      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If
      specified, this argument must be positive, broadcast with the shape of
      `left_bracket` and have the same dtype.
      Default value: `2e-7`.
    relative_root_tolerance: Optional `Tensor` representing the relative
      tolerance for estimated roots, with the total tolerance being calculated
      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If
      specified, this argument must be positive, broadcast with the shape of
      `left_bracket` and have the same dtype.
      Default value: `None` which translates to `4 *
        numpy.finfo(left_bracket.dtype.as_numpy_dtype).eps`.
    function_tolerance: Optional `Tensor` representing the tolerance used to
      check for roots. If the absolute value of `objective_fn` is smaller than
      or equal to `function_tolerance` at a given estimate, then that estimate
      is considered a root for the function. If specified, this argument must
      broadcast with the shape of `left_bracket` and have the same dtype. Set to
      zero to match Brent's original algorithm and to continue the search until
      an exact root is found.
      Default value: `2e-7`.
    max_iterations: Optional `Tensor` of an integral dtype or Python integer
      specifying the maximum number of steps to perform for each initial point.
      Must broadcast with the shape of `left_bracket`. If an element is set to
      zero, the function will not search for any root for the corresponding
      points in `left_bracket` and `right_bracket`. Instead, it will return the
      best estimate from the inputs.
      Default value: `100`.
    stopping_policy_fn: Python `callable` controlling the algorithm termination.
      It must be a callable accepting a `Tensor` of booleans with the shape of
      `left_bracket` (each denoting whether the search is finished for each
      starting point), and returning a scalar boolean `Tensor` (indicating
      whether the overall search should stop). Typical values are
      `tf.reduce_all` (which returns only when the search is finished for all
      pairs of points), and `tf.reduce_any` (which returns as soon as the search
      is finished for any pair of points).
      Default value: `None` which translates to `tf.reduce_all`.

  Returns:
    A tuple of 3 Python objects containing the state, parameters, and constants
    to use for the search.
  """
  stopping_policy_fn = stopping_policy_fn or tf.reduce_all
  if not callable(stopping_policy_fn):
    raise ValueError("stopping_policy_fn must be callable")

  left_bracket = tf.convert_to_tensor(left_bracket, name="left_bracket")
  right_bracket = tf.convert_to_tensor(
      right_bracket, name="right_bracket", dtype=left_bracket.dtype)

  if value_at_left_bracket is None:
    value_at_left_bracket = objective_fn(left_bracket)
  if value_at_right_bracket is None:
    value_at_right_bracket = objective_fn(right_bracket)

  value_at_left_bracket = tf.convert_to_tensor(
      value_at_left_bracket,
      name="value_at_left_bracket",
      dtype=left_bracket.dtype.base_dtype)
  value_at_right_bracket = tf.convert_to_tensor(
      value_at_right_bracket,
      name="value_at_right_bracket",
      dtype=left_bracket.dtype.base_dtype)

  if relative_root_tolerance is None:
    relative_root_tolerance = utils.default_relative_root_tolerance(
        left_bracket.dtype.base_dtype)

  absolute_root_tolerance = tf.convert_to_tensor(
      absolute_root_tolerance,
      name="absolute_root_tolerance",
      dtype=left_bracket.dtype)
  relative_root_tolerance = tf.convert_to_tensor(
      relative_root_tolerance,
      name="relative_root_tolerance",
      dtype=left_bracket.dtype)
  function_tolerance = tf.convert_to_tensor(
      function_tolerance, name="function_tolerance", dtype=left_bracket.dtype)

  max_iterations = tf.broadcast_to(
      tf.convert_to_tensor(max_iterations),
      name="max_iterations",
      shape=left_bracket.shape)
  num_iterations = tf.zeros_like(max_iterations)

  false = tf.constant(False, shape=left_bracket.shape)

  zero = tf.zeros_like(left_bracket)
  contrapoint = zero
  step_to_last_estimate = zero
  step_to_best_estimate = zero

  zero_value = tf.zeros_like(value_at_left_bracket)
  value_at_contrapoint = zero_value

  # Select the best root estimates from the inputs.
  # If no search is performed (e.g. `max_iterations` is `zero`), the estimate
  # computed this way will be returned. This differs slightly from the SciPy
  # implementation which always returns the `right_bracket`.
  swap_positions = tf.math.abs(value_at_left_bracket) < tf.math.abs(
      value_at_right_bracket)
  best_estimate, last_estimate = _swap_where(swap_positions, right_bracket,
                                             left_bracket)
  value_at_best_estimate, value_at_last_estimate = _swap_where(
      swap_positions, value_at_right_bracket, value_at_left_bracket)

  # Check if the current root estimate is good enough.
  # Using zero as `function_tolerance` will check for exact roots and match both
  # Brent's original algorithm and the SciPy implementation.
  finished = (num_iterations >=
              max_iterations) | (~tf.math.is_finite(value_at_last_estimate)) | (
                  ~tf.math.is_finite(value_at_best_estimate)) | (
                      tf.math.abs(value_at_best_estimate) <= function_tolerance)

  return (_BrentSearchState(
      best_estimate=best_estimate,
      last_estimate=last_estimate,
      contrapoint=contrapoint,
      value_at_best_estimate=value_at_best_estimate,
      value_at_last_estimate=value_at_last_estimate,
      value_at_contrapoint=value_at_contrapoint,
      step_to_best_estimate=step_to_best_estimate,
      step_to_last_estimate=step_to_last_estimate,
      num_iterations=num_iterations,
      finished=finished),
          _BrentSearchParams(
              objective_fn=objective_fn,
              max_iterations=max_iterations,
              absolute_root_tolerance=absolute_root_tolerance,
              relative_root_tolerance=relative_root_tolerance,
              function_tolerance=function_tolerance,
              stopping_policy_fn=stopping_policy_fn),
          _BrentSearchConstants(false=false, zero=zero, zero_value=zero_value))


# `_brent` currently only support inverse quadratic extrapolation.
# This will be fixed when adding the `brenth` variant.
def _brent(objective_fn,
           left_bracket,
           right_bracket,
           value_at_left_bracket=None,
           value_at_right_bracket=None,
           absolute_root_tolerance=2e-7,
           relative_root_tolerance=None,
           function_tolerance=2e-7,
           max_iterations=100,
           stopping_policy_fn=None,
           validate_args=False,
           name=None):
  r"""Finds root(s) of a function of a single variable using Brent's method.

  [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method) is a
  root-finding algorithm combining the bisection method, the secant method and
  extrapolation. Like bisection it is guaranteed to converge towards a root if
  one exists, but that convergence is superlinear and on par with less reliable
  methods.

  This implementation is a translation of the algorithm described in the
  [original article](https://academic.oup.com/comjnl/article/14/4/422/325237).

  Args:
    objective_fn: Python callable for which roots are searched. It must be a
      callable of a single `Tensor` parameter and return a `Tensor` of the same
      shape and dtype as `left_bracket`.
    left_bracket: `Tensor` or Python float representing the first starting
      points. The function will search for roots between each pair of points
      defined by `left_bracket` and `right_bracket`. The shape of `left_bracket`
      should match that of the input to `objective_fn`.
    right_bracket: `Tensor` of the same shape and dtype as `left_bracket` or
      Python float representing the second starting points. The function will
      search for roots between each pair of points defined by `left_bracket` and
      `right_bracket`. This argument must have the same shape as `left_bracket`.
    value_at_left_bracket: Optional `Tensor` or Python float representing the
      value of `objective_fn` at `left_bracket`. If specified, this argument
      must have the same shape as `left_bracket`. If not specified, the value
      will be evaluated during the search.
      Default value: None.
    value_at_right_bracket: Optional `Tensor` or Python float representing the
      value of `objective_fn` at `right_bracket`. If specified, this argument
      must have the same shape as `right_bracket`. If not specified, the value
      will be evaluated during the search.
      Default value: None.
    absolute_root_tolerance: Optional `Tensor` representing the absolute
      tolerance for estimated roots, with the total tolerance being calculated
      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If
      specified, this argument must be positive, broadcast with the shape of
      `left_bracket` and have the same dtype.
      Default value: `2e-7`.
    relative_root_tolerance: Optional `Tensor` representing the relative
      tolerance for estimated roots, with the total tolerance being calculated
      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If
      specified, this argument must be positive, broadcast with the shape of
      `left_bracket` and have the same dtype.
      Default value: `None` which translates to `4 *
        numpy.finfo(left_bracket.dtype.as_numpy_dtype).eps`.
    function_tolerance: Optional `Tensor` representing the tolerance used to
      check for roots. If the absolute value of `objective_fn` is smaller than
      or equal to `function_tolerance` at a given estimate, then that estimate
      is considered a root for the function. If specified, this argument must
      broadcast with the shape of `left_bracket` and have the same dtype. Set to
      zero to match Brent's original algorithm and to continue the search until
      an exact root is found.
      Default value: `2e-7`.
    max_iterations: Optional `Tensor` of an integral dtype or Python integer
      specifying the maximum number of steps to perform for each initial point.
      Must broadcast with the shape of `left_bracket`. If an element is set to
      zero, the function will not search for any root for the corresponding
      points in `left_bracket` and `right_bracket`. Instead, it will return the
      best estimate from the inputs.
      Default value: `100`.
    stopping_policy_fn: Python `callable` controlling the algorithm termination.
      It must be a callable accepting a `Tensor` of booleans with the shape of
      `left_bracket` (each denoting whether the search is finished for each
      starting point), and returning a scalar boolean `Tensor` (indicating
      whether the overall search should stop). Typical values are
      `tf.reduce_all` (which returns only when the search is finished for all
      pairs of points), and `tf.reduce_any` (which returns as soon as the search
      is finished for any pair of points).
      Default value: `None` which translates to `tf.reduce_all`.
    validate_args: Python `bool` indicating whether to validate arguments such
      as `left_bracket`, `right_bracket`, `absolute_root_tolerance`,
      `relative_root_tolerance`, `function_tolerance`, and `max_iterations`.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.

  Returns:
    brent_results: A Python object containing the following attributes:
      estimated_root: `Tensor` containing the best estimate explored. If the
        search was successful within the specified tolerance, this estimate is
        a root of the objective function.
      objective_at_estimated_root: `Tensor` containing the value of the
        objective function at `estimated_root`. If the search was successful
        within the specified tolerance, then this is close to 0. It has the
        same dtype and shape as `estimated_root`.
      num_iterations: `Tensor` containing the number of iterations performed.
        It has the same dtype as `max_iterations` and shape as `estimated_root`.
      converged: Scalar boolean `Tensor` indicating whether `estimated_root` is
        a root within the tolerance specified for the search. It has the same
        shape as `estimated_root`.

  Raises:
    ValueError: if the `stopping_policy_fn` is not callable.
  """

  with tf.compat.v1.name_scope(name, default_name="brent_root", values=[
      left_bracket, right_bracket, value_at_left_bracket,
      value_at_right_bracket, max_iterations
  ]):

    state, params, constants = _prepare_brent_args(
        objective_fn, left_bracket, right_bracket, value_at_left_bracket,
        value_at_right_bracket, absolute_root_tolerance,
        relative_root_tolerance, function_tolerance, max_iterations,
        stopping_policy_fn)

    assertions = []
    if validate_args:
      assertions += [
          tf.Assert(
              tf.reduce_all(
                  state.value_at_last_estimate *
                  state.value_at_best_estimate <= constants.zero_value),
              [state.value_at_last_estimate, state.value_at_best_estimate]),
          tf.Assert(
              tf.reduce_all(params.absolute_root_tolerance > constants.zero),
              [params.absolute_root_tolerance]),
          tf.Assert(
              tf.reduce_all(params.relative_root_tolerance > constants.zero),
              [params.relative_root_tolerance]),
          tf.Assert(
              tf.reduce_all(params.function_tolerance >= constants.zero),
              [params.function_tolerance]),
          tf.Assert(
              tf.reduce_all(params.max_iterations >= state.num_iterations),
              [params.max_iterations]),
      ]

    with tf.compat.v1.control_dependencies(assertions):
      result = tf.while_loop(
          # Negate `_should_stop` to determine if the search should continue.
          # This means, in particular, that tf.reduce_*all* will return only
          # when the search is finished for *all* starting points.
          lambda loop_vars: ~_should_stop(loop_vars, params.stopping_policy_fn),
          lambda state: _brent_loop_body(state, params, constants),
          loop_vars=[state],
          maximum_iterations=max_iterations)

  state = result[0]
  converged = tf.math.abs(state.value_at_best_estimate) <= function_tolerance

  return BrentResults(
      estimated_root=state.best_estimate,
      objective_at_estimated_root=state.value_at_best_estimate,
      num_iterations=state.num_iterations,
      converged=converged)


def brentq(objective_fn,
           left_bracket,
           right_bracket,
           value_at_left_bracket=None,
           value_at_right_bracket=None,
           absolute_root_tolerance=2e-7,
           relative_root_tolerance=None,
           function_tolerance=2e-7,
           max_iterations=100,
           stopping_policy_fn=None,
           validate_args=False,
           name=None):
  r"""Finds root(s) of a function of single variable using Brent's method.

  [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method) is a
  root-finding algorithm combining the bisection method, the secant method and
  extrapolation. Like bisection it is guaranteed to converge towards a root if
  one exists, but that convergence is superlinear and on par with less reliable
  methods.

  This implementation is a translation of the algorithm described in the
  [original article](https://academic.oup.com/comjnl/article/14/4/422/325237).

  Args:
    objective_fn: Python callable for which roots are searched. It must be a
      callable of a single `Tensor` parameter and return a `Tensor` of the same
      shape and dtype as `left_bracket`.
    left_bracket: `Tensor` or Python float representing the first starting
      points. The function will search for roots between each pair of points
      defined by `left_bracket` and `right_bracket`. The shape of `left_bracket`
      should match that of the input to `objective_fn`.
    right_bracket: `Tensor` of the same shape and dtype as `left_bracket` or
      Python float representing the second starting points. The function will
      search for roots between each pair of points defined by `left_bracket` and
      `right_bracket`. This argument must have the same shape as `left_bracket`.
    value_at_left_bracket: Optional `Tensor` or Python float representing the
      value of `objective_fn` at `left_bracket`. If specified, this argument
      must have the same shape as `left_bracket`. If not specified, the value
      will be evaluated during the search.
      Default value: None.
    value_at_right_bracket: Optional `Tensor` or Python float representing the
      value of `objective_fn` at `right_bracket`. If specified, this argument
      must have the same shape as `right_bracket`. If not specified, the value
      will be evaluated during the search.
      Default value: None.
    absolute_root_tolerance: Optional `Tensor` representing the absolute
      tolerance for estimated roots, with the total tolerance being calculated
      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If
      specified, this argument must be positive, broadcast with the shape of
      `left_bracket` and have the same dtype.
      Default value: `2e-7`.
    relative_root_tolerance: Optional `Tensor` representing the relative
      tolerance for estimated roots, with the total tolerance being calculated
      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If
      specified, this argument must be positive, broadcast with the shape of
      `left_bracket` and have the same dtype.
      Default value: `None` which translates to `4 *
        numpy.finfo(left_bracket.dtype.as_numpy_dtype).eps`.
    function_tolerance: Optional `Tensor` representing the tolerance used to
      check for roots. If the absolute value of `objective_fn` is smaller than
      or equal to `function_tolerance` at a given estimate, then that estimate
      is considered a root for the function. If specified, this argument must
      broadcast with the shape of `left_bracket` and have the same dtype. Set to
      zero to match Brent's original algorithm and to continue the search until
      an exact root is found.
      Default value: `2e-7`.
    max_iterations: Optional `Tensor` of an integral dtype or Python integer
      specifying the maximum number of steps to perform for each initial point.
      Must broadcast with the shape of `left_bracket`. If an element is set to
      zero, the function will not search for any root for the corresponding
      points in `left_bracket` and `right_bracket`. Instead, it will return the
      best estimate from the inputs.
      Default value: `100`.
    stopping_policy_fn: Python `callable` controlling the algorithm termination.
      It must be a callable accepting a `Tensor` of booleans with the shape of
      `left_bracket` (each denoting whether the search is finished for each
      starting point), and returning a scalar boolean `Tensor` (indicating
      whether the overall search should stop). Typical values are
      `tf.reduce_all` (which returns only when the search is finished for all
      pairs of points), and `tf.reduce_any` (which returns as soon as the search
      is finished for any pair of points).
      Default value: `None` which translates to `tf.reduce_all`.
    validate_args: Python `bool` indicating whether to validate arguments such
      as `left_bracket`, `right_bracket`, `absolute_root_tolerance`,
      `relative_root_tolerance`, `function_tolerance`, and `max_iterations`.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.

  Returns:
    brent_results: A Python object containing the following attributes:
      estimated_root: `Tensor` containing the best estimate explored. If the
        search was successful within the specified tolerance, this estimate is
        a root of the objective function.
      objective_at_estimated_root: `Tensor` containing the value of the
        objective function at `estimated_root`. If the search was successful
        within the specified tolerance, then this is close to 0. It has the
        same dtype and shape as `estimated_root`.
      num_iterations: `Tensor` containing the number of iterations performed.
        It has the same dtype as `max_iterations` and shape as `estimated_root`.
      converged: Scalar boolean `Tensor` indicating whether `estimated_root` is
        a root within the tolerance specified for the search. It has the same
        shape as `estimated_root`.

  Raises:
    ValueError: if the `stopping_policy_fn` is not callable.

  #### Examples

  ```python
  import tensorflow.compat.v2 as tf
  tf.enable_eager_execution()

  # Example 1: Roots of a single function for two pairs of starting points.

  f = lambda x: 63 * x**5 - 70 * x**3 + 15 * x + 2
  x1 = tf.constant([-10, 1], dtype=tf.float64)
  x2 = tf.constant([10, -1], dtype=tf.float64)

  tf.math.brentq(objective_fn=f, left_bracket=x1, right_bracket=x2)
  # ==> BrentResults(
  #    estimated_root=array([-0.14823253, -0.14823253]),
  #    objective_at_estimated_root=array([3.27515792e-15, 0.]),
  #    num_iterations=array([11, 6]),
  #    converged=array([True, True]))

  tf.math.brentq(objective_fn=f,
                 left_bracket=x1,
                 right_bracket=x2,
                 stopping_policy_fn=tf.reduce_any)
  # ==> BrentResults(
  #    estimated_root=array([-2.60718234, -0.14823253]),
  #    objective_at_estimated_root=array([-6.38579115e+03, 2.39763764e-11]),
  #    num_iterations=array([7, 6]),
  #    converged=array([False, True]))

  # Example 2: Roots of a multiplex function for one pair of starting points.

  def f(x):
    return tf.constant([0., 63.], dtype=tf.float64) * x**5 \
        + tf.constant([5., -70.], dtype=tf.float64) * x**3 \
        + tf.constant([-3., 15.], dtype=tf.float64) * x \
        + 2

  x1 = tf.constant([-5, -5], dtype=tf.float64)
  x2 = tf.constant([5, 5], dtype=tf.float64)

  tf.math.brentq(objective_fn=f, left_bracket=x1, right_bracket=x2)
  # ==> BrentResults(
  #    estimated_root=array([-1., -0.14823253]),
  #    objective_at_estimated_root=array([0., 2.08721929e-14]),
  #    num_iterations=array([13, 11]),
  #    converged=array([True, True]))

  # Example 3: Roots of a multiplex function for two pairs of starting points.

  def f(x):
    return tf.constant([0., 63.], dtype=tf.float64) * x**5 \
        + tf.constant([5., -70.], dtype=tf.float64) * x**3 \
        + tf.constant([-3., 15.], dtype=tf.float64) * x \
        + 2

  x1 = tf.constant([[-5, -5], [10, 10]], dtype=tf.float64)
  x2 = tf.constant([[5, 5], [-10, -10]], dtype=tf.float64)

  tf.math.brentq(objective_fn=f, left_bracket=x1, right_bracket=x2)
  # ==> BrentResults(
  #    estimated_root=array([
  #        [-1, -0.14823253],
  #        [-1, -0.14823253]]),
  #    objective_at_estimated_root=array([
  #        [0., 2.08721929e-14],
  #        [0., 2.08721929e-14]]),
  #    num_iterations=array([
  #        [13, 11],
  #        [15, 11]]),
  #    converged=array([
  #        [True, True],
  #        [True, True]]))
  ```
  """

  return _brent(
      objective_fn,
      left_bracket,
      right_bracket,
      value_at_left_bracket=value_at_left_bracket,
      value_at_right_bracket=value_at_right_bracket,
      absolute_root_tolerance=absolute_root_tolerance,
      relative_root_tolerance=relative_root_tolerance,
      function_tolerance=function_tolerance,
      max_iterations=max_iterations,
      stopping_policy_fn=stopping_policy_fn,
      validate_args=validate_args,
      name=name)
