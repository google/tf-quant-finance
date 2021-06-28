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

"""The Conjugate Gradient optimization algorithm.

References:
[HZ2006] Hager, William W., and Hongchao Zhang. "Algorithm 851: CG_DESCENT,
  a conjugate gradient method with guaranteed descent."
  http://users.clas.ufl.edu/hager/papers/CG/cg_compare.pdf
[HZ2013] W. W. Hager and H. Zhang (2013) The limited memory conjugate gradient
  method.
  https://pdfs.semanticscholar.org/8769/69f3911777e0ff0663f21b67dff30518726b.pdf
[JuliaLineSearches] Line search methods in Julia.
  https://github.com/JuliaNLSolvers/LineSearches.jl
"""

import collections

from  typing import Callable, Tuple

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.optimizer import converged_all
from tensorflow_probability.python.optimizer import linesearch
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils


__all__ = [
    'OptimizerResult',
    'ConjugateGradientParams',
    'minimize',
]


@tff_utils.dataclass
class OptimizerResult:
  """Optimization results.

  Attributes:
    converged: A boolean `Tensor` indicating whether the minimum as found within
      tolerance.
    failed: A boolean `Tensor` indicating whether a line search step failed to
      find a suitable step size satisfying Wolfe conditions. In the absence of
      any constraints on the number of objective evaluations permitted, this
      value will be the complement of `converged`. However, if there is a
      constraint and the search stopped due to available evaluations being
      exhausted, both `failed` and `converged` will be simultaneously False.
    num_iterations: The number of iterations
    num_objective_evaluations: The total number of objective evaluations
      performed.
    position: A real `Tensor` containing the last argument value found during
      the search. If the search converged, then this value is the argmin of the
      objective function (within some tolerance).
    objective_value: A real `Tensor` containing the value of the objective
      function at the `position`. If the search converged, then this is the
      (local) minimum of the objective function.
    objective_gradient: A real `Tensor` containing the gradient of the
      objective function at the `final_position`. If the search converged the
      max-norm of this tensor should be below the tolerance.
  """
  converged: types.BoolTensor
  failed: types.BoolTensor
  num_iterations: types.IntTensor
  num_objective_evaluations: types.IntTensor
  position: types.RealTensor
  objective_value: types.RealTensor
  objective_gradient: types.RealTensor


@tff_utils.dataclass
class _OptimizerState:
  """Internal state of optimizer."""
  # Fields from OptimizerResult.
  converged: types.BoolTensor
  failed: types.BoolTensor
  num_iterations: types.IntTensor
  num_objective_evaluations: types.IntTensor
  # Position (x_k in [HZ2006]).
  position: types.RealTensor
  # Objective (f_k in [HZ2006]).
  objective_value: types.RealTensor
  # Gradient (g_k in [HZ2006]).
  objective_gradient: types.RealTensor
  # Direction, along which to go at the next step (d_k in [HZ2006]).
  direction: types.RealTensor
  # Previous step length (a_{k-1} in [HZ2006]).
  prev_step: types.RealTensor


# TODO(b/191755220): Use dataclass instead once Hager-Zhang `val_where` utility
# works with dataclasses.
ValueAndGradient = collections.namedtuple(
    'ValueAndGradient',
    [
        # Point at which line function is evaluated.
        'x',
        # Value of function.
        'f',
        # Directional derivative.
        'df',
        # Full gradient evaluated at that point.
        'full_gradient'
    ])


@tff_utils.dataclass
class ConjugateGradientParams(object):
  """Adjustable parameters of conjugate gradient algorithm."""
  # Real number. Sufficient decrease parameter for Wolfe conditions.
  # Corresponds to `delta` in [HZ2006].
  # Range (0, 0.5). Defaults to 0.1.
  sufficient_decrease_param: types.RealTensor = 0.1
  # Real number. Curvature parameter for Wolfe conditions.
  # Corresponds to 'sigma' in [HZ2006].
  # Range [`delta`, 1). Defaults to 0.9.
  curvature_param: types.RealTensor = 0.9
  # Real number. Used to estimate the threshold at which the line search
  # switches to approximate Wolfe conditions.
  # Corresponds to 'epsilon' in [HZ2006].
  # Range (0, inf). Defaults to 1e-6.
  threshold_use_approximate_wolfe_condition: types.RealTensor = 1e-6
  # Real number. Shrinkage parameter for line search.
  # Corresponds to 'gamma' in [HZ2006].
  # Range (0, 1). Defaults to 0.66.
  shrinkage_param: types.RealTensor = 0.66
  # Real number. Parameter used in to calculate lower bound for coefficient
  # 'beta_k', used to calculate next direction.
  # Corresponds to 'eta' in [HZ2013].
  # Range (0, inf). Defaults to 0.4.
  direction_update_param: types.RealTensor = 0.4
  # Real number. Used in line search to expand the initial interval in case it
  # does not bracket a minimum.
  # Corresponds to 'rho' in [HZ2006].
  # Range (1.0, inf). Defaults to 5.0.
  expansion_param: types.RealTensor = 5.0
  # Real scalar `Tensor`. Factor used in initial guess for line search to
  # multiply previous step to get right point for quadratic interpolation.
  # Corresponds to 'psi_1' in [HZ2006].
  # Range (0, 1). Defaults to 0.2.
  initial_guess_small_factor: types.RealTensor = 0.2
  # Real number. Factor used in initial guess for line search to multiply
  # previous step if qudratic interpolation failed.
  # Corresponds to 'psi_2' in [HZ2006].
  # Range (1, inf). Defaults to 2.0.
  initial_guess_step_multiplier: types.RealTensor = 2.0
  # Boolean. Whether to try quadratic interpolation when finding initial step
  # for line search.
  # Corresponds to 'QuadStep' in [HZ2006].
  # Defaults  to `True`.
  quad_step: bool = True


def minimize(
    value_and_gradients_function: Callable[
        [types.RealTensor], Tuple[types.RealTensor, types.RealTensor]],
    initial_position: types.RealTensor,
    tolerance: types.RealTensor = 1e-8,
    x_tolerance: types.RealTensor = 0,
    f_relative_tolerance: types.RealTensor = 0,
    max_iterations: types.IntTensor = 50,
    parallel_iterations: types.IntTensor = 1,
    stopping_condition: Callable[[types.BoolTensor, types.BoolTensor],
                                 types.BoolTensor] = None,
    params: ConjugateGradientParams = None,
    name: str = None) -> OptimizerResult:
  """Minimizes a differentiable function.

  Implementation of algorithm described in [HZ2006]. Updated formula for next
  search direction were taken from [HZ2013].

  Supports batches with 1-dimensional batch shape.

  #### References:
  [HZ2006] Hager, William W., and Hongchao Zhang. "Algorithm 851: CG_DESCENT,
    a conjugate gradient method with guaranteed descent."
    http://users.clas.ufl.edu/hager/papers/CG/cg_compare.pdf
  [HZ2013] W. W. Hager and H. Zhang (2013) The limited memory conjugate gradient
    method.
    https://pdfs.semanticscholar.org/8769/69f3911777e0ff0663f21b67dff30518726b.pdf

  ### Usage:
  The following example demonstrates this optimizer attempting to find the
  minimum for a simple two dimensional quadratic objective function.

  ```python
    minimum = np.array([1.0, 1.0])  # The center of the quadratic bowl.
    scales = np.array([2.0, 3.0])  # The scales along the two axes.

    # The objective function and the gradient.
    def quadratic(x):
      value = tf.reduce_sum(scales * (x - minimum) ** 2)
      return value, tf.gradients(value, x)[0]

    start = tf.constant([0.6, 0.8])  # Starting point for the search.
    optim_results = conjugate_gradient.minimize(
        quadratic, initial_position=start, tolerance=1e-8)

    with tf.Session() as session:
      results = session.run(optim_results)
      # Check that the search converged
      assert(results.converged)
      # Check that the argmin is close to the actual value.
      np.testing.assert_allclose(results.position, minimum)
  ```

  Args:
    value_and_gradients_function:  A Python callable that accepts a point as a
      real `Tensor` and returns a tuple of `Tensor`s of real dtype containing
      the value of the function and its gradient at that point. The function to
      be minimized. The input should be of shape `[..., n]`, where `n` is the
      size of the domain of input points, and all others are batching
      dimensions. The first component of the return value should be a real
      `Tensor` of matching shape `[...]`. The second component (the gradient)
      should also be of shape `[..., n]` like the input value to the function.
    initial_position: Real `Tensor` of shape `[..., n]`. The starting point, or
      points when using batching dimensions, of the search procedure. At these
      points the function value and the gradient norm should be finite.
    tolerance: Scalar `Tensor` of real dtype. Specifies the gradient tolerance
      for the procedure. If the supremum norm of the gradient vector is below
      this number, the algorithm is stopped.
    x_tolerance: Scalar `Tensor` of real dtype. If the absolute change in the
      position between one iteration and the next is smaller than this number,
      the algorithm is stopped.
    f_relative_tolerance: Scalar `Tensor` of real dtype. If the relative change
      in the objective value between one iteration and the next is smaller than
      this value, the algorithm is stopped.
    max_iterations: Scalar positive int32 `Tensor`. The maximum number of
      iterations.
    parallel_iterations: Positive integer. The number of iterations allowed to
      run in parallel.
    stopping_condition: (Optional) A Python function that takes as input two
      Boolean tensors of shape `[...]`, and returns a Boolean scalar tensor. The
      input tensors are `converged` and `failed`, indicating the current status
      of each respective batch member; the return value states whether the
      algorithm should stop. The default is tfp.optimizer.converged_all which
      only stops when all batch members have either converged or failed. An
      alternative is tfp.optimizer.converged_any which stops as soon as one
      batch member has converged, or when all have failed.
    params: ConjugateGradientParams object with adjustable parameters of the
      algorithm. If not supplied, default parameters will be used.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'minimize' is used.

  Returns:
    optimizer_results: A namedtuple containing the following items:
      converged: boolean tensor of shape `[...]` indicating for each batch
        member whether the minimum was found within tolerance.
      failed:  boolean tensor of shape `[...]` indicating for each batch
        member whether a line search step failed to find a suitable step size
        satisfying Wolfe conditions. In the absence of any constraints on the
        number of objective evaluations permitted, this value will
        be the complement of `converged`. However, if there is
        a constraint and the search stopped due to available
        evaluations being exhausted, both `failed` and `converged`
        will be simultaneously False.
      num_objective_evaluations: The total number of objective
        evaluations performed.
      position: A tensor of shape `[..., n]` containing the last argument value
        found during the search from each starting point. If the search
        converged, then this value is the argmin of the objective function.
      objective_value: A tensor of shape `[...]` with the value of the
        objective function at the `position`. If the search converged, then
        this is the (local) minimum of the objective function.
      objective_gradient: A tensor of shape `[..., n]` containing the gradient
        of the objective function at the `position`. If the search converged
        the max-norm of this tensor should be below the tolerance.

  """
  with tf.compat.v1.name_scope(name, 'minimize', [initial_position, tolerance]):
    if params is None:
      params = ConjugateGradientParams()

    initial_position = tf.convert_to_tensor(
        value=initial_position, name='initial_position')
    dtype = initial_position.dtype
    tolerance = tf.convert_to_tensor(
        value=tolerance, dtype=dtype, name='grad_tolerance')
    f_relative_tolerance = tf.convert_to_tensor(
        value=f_relative_tolerance, dtype=dtype, name='f_relative_tolerance')
    x_tolerance = tf.convert_to_tensor(
        value=x_tolerance, dtype=dtype, name='x_tolerance')
    max_iterations = tf.convert_to_tensor(
        value=max_iterations, name='max_iterations')
    stopping_condition = stopping_condition or converged_all
    delta = tf.convert_to_tensor(
        params.sufficient_decrease_param, dtype=dtype, name='delta')
    sigma = tf.convert_to_tensor(
        params.curvature_param, dtype=dtype, name='sigma')
    eps = tf.convert_to_tensor(
        params.threshold_use_approximate_wolfe_condition,
        dtype=dtype,
        name='sigma')
    eta = tf.convert_to_tensor(
        params.direction_update_param, dtype=dtype, name='eta')
    psi_1 = tf.convert_to_tensor(
        params.initial_guess_small_factor, dtype=dtype, name='psi_1')
    psi_2 = tf.convert_to_tensor(
        params.initial_guess_step_multiplier, dtype=dtype, name='psi_2')

    f0, df0 = value_and_gradients_function(initial_position)
    converged = _norm(df0) < tolerance

    initial_state = _OptimizerState(
        converged=converged,
        failed=tf.zeros_like(converged),  # All false.
        num_iterations=tf.convert_to_tensor(value=0),
        num_objective_evaluations=tf.convert_to_tensor(value=1),
        position=initial_position,
        objective_value=f0,
        objective_gradient=df0,
        direction=-df0,
        prev_step=tf.ones_like(f0),
    )

    def _cond(state):
      """Continue if iterations remain and stopping condition is not met."""
      return (
          (state.num_iterations < max_iterations)
          & tf.logical_not(stopping_condition(state.converged, state.failed)))

    def _body(state):
      """Main optimization loop."""
      # We use notation of [HZ2006] for brevity.
      x_k = state.position
      d_k = state.direction
      f_k = state.objective_value
      g_k = state.objective_gradient
      a_km1 = state.prev_step  # Means a_{k-1}.

      # Define scalar function, which is objective restricted to direction.
      def ls_func(alpha):
        pt = x_k + tf.expand_dims(alpha, axis=-1) * d_k
        objective_value, gradient = value_and_gradients_function(pt)
        return ValueAndGradient(
            x=alpha,
            f=objective_value,
            df=_dot(gradient, d_k),
            full_gradient=gradient)

      # Generate initial guess for line search.
      # [HZ2006] suggests to generate first initial guess separately, but
      # [JuliaLineSearches] generates it as if previous step length was 1, and
      # we do the same.
      phi_0 = f_k
      dphi_0 = _dot(g_k, d_k)
      ls_val_0 = ValueAndGradient(
          x=tf.zeros_like(phi_0), f=phi_0, df=dphi_0, full_gradient=g_k)
      step_guess_result = _init_step(ls_val_0, a_km1, ls_func, psi_1, psi_2,
                                     params.quad_step)
      init_step = step_guess_result.step

      # Check if initial step size already satisfies Wolfe condition, and in
      # that case don't perform line search.
      c = init_step.x
      phi_lim = phi_0 + eps * tf.abs(phi_0)
      phi_c = init_step.f
      dphi_c = init_step.df
      # Original Wolfe conditions, T1 in [HZ2006].
      suff_decrease_1 = delta * dphi_0 >= tf.math.divide_no_nan(
          phi_c - phi_0, c)
      curvature = dphi_c >= sigma * dphi_0
      wolfe1 = suff_decrease_1 & curvature
      # Approximate Wolfe conditions, T2 in [HZ2006].
      suff_decrease_2 = (2 * delta - 1) * dphi_0 >= dphi_c
      curvature = dphi_c >= sigma * dphi_0
      wolfe2 = suff_decrease_2 & curvature & (phi_c <= phi_lim)
      wolfe = wolfe1 | wolfe2
      skip_line_search = (step_guess_result.may_terminate
                          & wolfe) | state.failed | state.converged

      # Call Hager-Zhang line search (L0-L3 in [HZ2006]).
      # Parameter theta from [HZ2006] is not adjustable, it's always 0.5.
      ls_result = linesearch.hager_zhang(
          ls_func,
          value_at_zero=ls_val_0,
          converged=skip_line_search,
          initial_step_size=init_step.x,
          value_at_initial_step=init_step,
          shrinkage_param=params.shrinkage_param,
          expansion_param=params.expansion_param,
          sufficient_decrease_param=delta,
          curvature_param=sigma,
          threshold_use_approximate_wolfe_condition=eps)

      # Moving to the next point, using step length from line search.
      # If line search was skipped, take step length from initial guess.
      # To save objective evaluation, use objective value and gradient returned
      # by line search or initial guess.
      a_k = tf.compat.v1.where(
          skip_line_search, init_step.x, ls_result.left.x)
      x_kp1 = state.position + tf.expand_dims(a_k, -1) * d_k
      f_kp1 = tf.compat.v1.where(
          skip_line_search, init_step.f, ls_result.left.f)
      g_kp1 = tf.compat.v1.where(skip_line_search, init_step.full_gradient,
                                 ls_result.left.full_gradient)

      # Evaluate next direction.
      # Use formulas (2.7)-(2.11) from [HZ2013] with P_k=I.
      y_k = g_kp1 - g_k
      d_dot_y = _dot(d_k, y_k)
      b_k = tf.math.divide_no_nan(
          _dot(y_k, g_kp1)
          - tf.math.divide_no_nan(_norm_sq(y_k) * _dot(g_kp1, d_k), d_dot_y),
          d_dot_y)
      eta_k = tf.math.divide_no_nan(eta * _dot(d_k, g_k), _norm_sq(d_k))
      b_k = tf.maximum(b_k, eta_k)
      d_kp1 = -g_kp1 + tf.expand_dims(b_k, -1) * d_k

      # Check convergence criteria.
      grad_converged = _norm_inf(g_kp1) <= tolerance
      x_converged = (_norm_inf(x_kp1 - x_k) <= x_tolerance)
      f_converged = (
          tf.math.abs(f_kp1 - f_k) <= f_relative_tolerance * tf.math.abs(f_k))
      converged = ls_result.converged & (grad_converged
                                         | x_converged | f_converged)
      failed = ls_result.failed
      # Construct new state for next iteration.
      new_state = _OptimizerState(
          converged=converged,
          failed=failed,
          num_iterations=state.num_iterations + 1,
          num_objective_evaluations=state.num_objective_evaluations +
          step_guess_result.func_evals + ls_result.func_evals,
          position=tf.compat.v1.where(state.converged, x_k, x_kp1),
          objective_value=tf.compat.v1.where(state.converged, f_k, f_kp1),
          objective_gradient=tf.compat.v1.where(state.converged, g_k, g_kp1),
          direction=d_kp1,
          prev_step=a_k)
      return (new_state,)

    final_state = tf.while_loop(
        _cond, _body, (initial_state,),
        parallel_iterations=parallel_iterations)[0]
    return OptimizerResult(
        converged=final_state.converged,
        failed=final_state.failed,
        num_iterations=final_state.num_iterations,
        num_objective_evaluations=final_state.num_objective_evaluations,
        position=final_state.position,
        objective_value=final_state.objective_value,
        objective_gradient=final_state.objective_gradient)


@tff_utils.dataclass
class _StepGuessResult:
  """A namedtuple with result of guessing initial step."""
  # ValueAndGradient describing the initial guess.
  step: types.RealTensor
  # Whether initial guess is "good enogh" to use. Used internally by
  # _init_step, must have all components `True` when returned.
  can_take: types.BoolTensor
  # If true, means that before performing line search we have to check
  # whether Wolfe conditions are already satisfied, and in that case don't
  # perform line search.
  # Set to true if step was obtained by quandratic interpolation.
  may_terminate: types.BoolTensor
  # Number of function calls made to determine initial step.
  func_evals: types.IntTensor


def _init_step(pos, prev_step, func, psi_1, psi_2, quad_step):
  """Finds initial step size for line seacrh at given point.

  Corresponds to I1-I2 in [HZ2006].

  Args:
    pos: ValueAndGradient for current point.
    prev_step: Step size at previous iteration.
    func: Callable taking real `Tensor` and returning ValueAndGradient,
      describes scalar function for line search.
    psi_1: Real scalar `Tensor`. Factor to multiply previous step to get right
      point for quadratic interpolation.
    psi_2: Real scalar `Tesnor`. Factor to multiply previous step if qudratic
      interpolation failed.
    quad_step: Boolean. Whether to try quadratic interpolation.

  Returns:
    _StepGuessResult namedtuple containing initial guess and additional data.
  """
  phi_0 = pos.f
  derphi_0 = pos.df
  step = func(psi_1 * prev_step)
  can_take = step.f > phi_0
  result = _StepGuessResult(
      step=step,
      func_evals=1,
      can_take=can_take,
      may_terminate=tf.zeros_like(can_take))

  # Try to approximate function with a parabola and take its minimum as initial
  # guess.
  if quad_step:
    # Quadratic coefficient of parabola. If it's positive, parabola is convex
    # and has minimum.
    q_koef = step.f - phi_0 - step.x * derphi_0
    quad_step_success = tf.logical_and(step.f <= phi_0, q_koef > 0.0)

    def update_result_1():
      new_x = tf.compat.v1.where(
          quad_step_success,
          -0.5 * tf.math.divide_no_nan(derphi_0 * step.x**2, q_koef),
          result.step.x)
      return _StepGuessResult(
          step=func(new_x),
          func_evals=result.func_evals + 1,
          can_take=tf.math.logical_or(result.can_take, quad_step_success),
          may_terminate=tf.math.logical_or(result.may_terminate,
                                           quad_step_success))

    result = tf.cond(
        tf.math.reduce_any(quad_step_success), update_result_1, lambda: result)

  def update_result_2():
    new_x = tf.compat.v1.where(can_take, result.step.x, psi_2 * prev_step)
    return _StepGuessResult(
        step=func(new_x),
        func_evals=result.func_evals + 1,
        can_take=tf.ones_like(can_take),
        may_terminate=result.may_terminate)

  # According to [HZ2006] we should fall back to psi_2*prev_step when quadratic
  # interpolation failed. However, [JuliaLineSearches] retains guess
  # psi_1*prev_step if func(psi_1 * prev_step) > func(0), because then local
  # minimum is within (0, psi_1*prev_step).
  result = tf.cond(
      tf.math.reduce_all(result.can_take), lambda: result, update_result_2)

  return result


def _dot(x, y):
  """Evaluates scalar product."""
  return tf.math.reduce_sum(x * y, axis=-1)


def _norm(x):
  """Evaluates L2 norm."""
  return tf.linalg.norm(x, axis=-1)


def _norm_sq(x):
  """Evaluates L2 norm squared."""
  return tf.math.reduce_sum(tf.square(x), axis=-1)


def _norm_inf(x):
  """Evaluates inf-norm."""
  return tf.reduce_max(tf.abs(x), axis=-1)
