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

# Lint as: python2, python3
"""Extrapolation time marching scheme for parabolic PDEs."""

from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.implicit import implicit_scheme
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.parabolic_equation_stepper import parabolic_equation_step


def extrapolation_step(
    time,
    next_time,
    coord_grid,
    value_grid,
    boundary_conditions,
    second_order_coeff_fn,
    first_order_coeff_fn,
    zeroth_order_coeff_fn,
    num_steps_performed,
    dtype=None,
    name=None):
  """Performs one step of the parabolic PDE solver using extrapolation scheme.

  For a given solution (given by the `value_grid`) of a parabolic PDE at a given
  `time` on a given `coord_grid` computes an approximate solution at the
  `next_time` on the same coordinate grid using extrapolation implicit-explicit
  time marching scheme. (see, e.g., [1]). The parabolic PDE is of the form:

  ```none
   V_{t} + a(t, x) * V_{xx} + b(t, x) * V_{x} + c(t, x) * V = 0
  ```

  Here `V = V(t, x)` is a solution to the 2-dimensional PDE. `V_{t}` is the
  derivative over time and `V_{x}` and `V_{xx}` are the first and second
  derivatives over the space component. For a solution to be well-defined, it is
  required for `a` to be positive on its domain.

  See `fd_solvers.step_back` for an example use case.

  ### References:
  [1]: D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods
  for Parabolic Partial Differential Equations. I. 1978
  SIAM Journal on Numerical Analysis. 15. 1212-1224.
  https://epubs.siam.org/doi/abs/10.1137/0715082

  Args:
    time: Real positive scalar `Tensor`. The start time of the grid.
      Corresponds to time `t0` above.
    next_time: Real scalar `Tensor` smaller than the `start_time` and greater
      than zero. The time to step back to. Corresponds to time `t1` above.
    coord_grid: List of `n` rank 1 real `Tensor`s. `n` is the dimension of the
      domain. The i-th `Tensor` has shape, `[d_i]` where `d_i` is the size of
      the grid along axis `i`. The coordinates of the grid points. Corresponds
      to the spatial grid `G` above.
    value_grid: Real `Tensor` containing the function values at time
      `start_time` which have to be stepped back to time `end_time`. The shape
      of the `Tensor` must broadcast with `[K, d_1, d_2, ..., d_n]`. The first
      axis of size `K` is the values batch dimension and allows multiple
      functions (with potentially different boundary/final conditions) to be
      stepped back simultaneously.
    boundary_conditions: The boundary conditions. Only rectangular boundary
      conditions are supported. A list of tuples of size 1. The list element is
      a tuple that consists of two callables representing the
      boundary conditions at the minimum and maximum values of the spatial
      variable indexed by the position in the list. `boundary_conditions[0][0]`
      describes the boundary at `x_min`, and `boundary_conditions[0][1]` the
      boundary at `x_max`. The boundary conditions are accepted in the form
      `alpha(t) V + beta(t) V_n = gamma(t)`, where `V_n` is the derivative
      with respect to the exterior normal to the boundary.
      Each callable receives the current time `t` and the `coord_grid` at the
      current time, and should return a tuple of `alpha`, `beta`, and `gamma`.
      Each can be a number, a zero-rank `Tensor` or a `Tensor` of the batch
      shape.
      For example, for a grid of shape `(b, n)`, where `b` is the batch size,
      `boundary_conditions[0][0]` should return a tuple of either numbers,
      zero-rank tensors or tensors of shape `(b, n)`.
      `alpha` and `beta` can also be `None` in case of Neumann and
      Dirichlet conditions, respectively.
    second_order_coeff_fn: See the spec in fd_solvers.step_back.
    first_order_coeff_fn: See the spec in fd_solvers.step_back.
    zeroth_order_coeff_fn: See the spec in fd_solvers.step_back.
    num_steps_performed: Python `int`. Number of steps performed so far.
    dtype: The dtype to use.
    name: The name to give to the ops.
      Default value: None which means `extrapolation_step` is used.

  Returns:
    A sequence of two `Tensor`s. The first one is a `Tensor` of the same
    `dtype` and `shape` as `coord_grid` and represents a new coordinate grid
    after one iteration. The second `Tensor` is of the same shape and `dtype`
    as`values_grid` and represents an approximate solution of the equation after
    one iteration.
  """
  del num_steps_performed
  name = name or 'extrapolation_step'
  return parabolic_equation_step(time,
                                 next_time,
                                 coord_grid,
                                 value_grid,
                                 boundary_conditions,
                                 second_order_coeff_fn,
                                 first_order_coeff_fn,
                                 zeroth_order_coeff_fn,
                                 time_marching_scheme=extrapolation_scheme,
                                 dtype=dtype,
                                 name=name)


def extrapolation_scheme(
    value_grid,
    t1,
    t2,
    equation_params_fn,
    backwards):
  """Constructs extrapolation implicit-explicit scheme.

  Performs two implicit half-steps, one full implicit step, and combines them
  with such coefficients that ensure second-order errors. More computationally
  expensive than Crank-Nicolson scheme, but provides a better approximation for
  high-wavenumber components, which results in absence of oscillations typical
  for Crank-Nicolson scheme in case of non-smooth initial conditions. See [1]
  for details.

  ### References:
  [1]: D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods
  for Parabolic Partial Differential Equations. I. 1978
  SIAM Journal on Numerical Analysis. 15. 1212-1224.
  https://epubs.siam.org/doi/abs/10.1137/0715082

  Args:
    value_grid: A `Tensor` of real dtype. Grid of solution values at the
      current time.
    t1: Lesser of the two times defining the step.
    t2: Greater of the two times defining the step.
    equation_params_fn: A callable that takes a scalar `Tensor` argument
      representing time and constructs the tridiagonal matrix `A`
      (a tuple of three `Tensor`s, main, upper, and lower diagonals)
      and the inhomogeneous term `b`. All of the `Tensor`s are of the same
      `dtype` as `inner_value_grid` and of the shape broadcastable with the
      shape of `inner_value_grid`.
    backwards: A Python bool. Whether we're making a step backwards in time.

  Returns:
    A `Tensor` of the same shape and `dtype` a
    `values_grid` and represents an approximate solution `u(t2)`.
  """
  if backwards:
    first_half_times = (t1 + t2) / 2, t2
    second_half_times = t1, (t1 + t2) / 2
  else:
    first_half_times = t1, (t1 + t2) / 2
    second_half_times = (t1 + t2) / 2, t2

  first_half_step = implicit_scheme(
      value_grid, first_half_times[0], first_half_times[1],
      equation_params_fn, backwards)
  two_half_steps = implicit_scheme(
      first_half_step, second_half_times[0], second_half_times[1],
      equation_params_fn, backwards)

  full_step = implicit_scheme(value_grid, t1, t2, equation_params_fn,
                              backwards)
  return 2 * two_half_steps - full_step
