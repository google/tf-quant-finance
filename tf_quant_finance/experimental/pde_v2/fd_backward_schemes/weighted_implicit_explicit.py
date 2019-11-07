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
"""Weighted implicit-explicit time marching scheme for parabolic PDEs."""

import tensorflow as tf

from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.parabolic_equation_stepper import parabolic_equation_step


def weighted_implicit_explicit_step(theta):
  """Constructs parabolic PDE solver using weighted implicit-explicit scheme.

  Creates a `parabolic_equation_step` function for solving a parabolic
  differential equation form:

  ```none
   V_{t} + a(t, x) * V_{xx} + b(t, x) * V_{x} + c(t, x) * V = 0
  ```
  using  weighted implicit-explicit time marching scheme.
  Here `V = V(t, x)` is a solution to the 2-dimensional PDE. `V_{t}` is the
  derivative over time and `V_{x}` and `V_{xx}` are the first and second
  derivatives over the space component.

  ### References:
  [1] I.V. Puzynin, A.V. Selin, S.I. Vinitsky, A high-order accuracy method for
  numerical solving of the time-dependent Schrodinger equation, Comput. Phys.
  Commun. 123 (1999), 1.
  https://www.sciencedirect.com/science/article/pii/S0010465599002246

  Args:
    theta: A float in range `[0, 1]`. A parameter used to mix implicit and
      explicit schemes together. Value of `0.0` corresponds to the fully
      implicit scheme, `1.0` to the fully explicit, and `0.5` to the
      Crank-Nicolson scheme. See, e.g., [1].

  Returns:
    `parabolic_equation_step` callable with weighted implicit-explicit time
    marching scheme.
  """
  scheme = weighted_implicit_explicit_scheme(theta)
  def step_fn(
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
    """Constructs parabolic_equation_stepper."""
    del num_steps_performed
    name = name or 'weighted_implicit_explicit_scheme'
    return parabolic_equation_step(time,
                                   next_time,
                                   coord_grid,
                                   value_grid,
                                   boundary_conditions,
                                   second_order_coeff_fn,
                                   first_order_coeff_fn,
                                   zeroth_order_coeff_fn,
                                   time_marching_scheme=scheme,
                                   dtype=dtype,
                                   name=name)
  return step_fn


def weighted_implicit_explicit_scheme(theta):
  """Constructs weighted implicit-explicit scheme.

  Approximates the exponent in the solution `du/dt = A(t) u(t) + b(t)` as
  `u(t2) = (1 - (1 - theta) dt A)^(-1) * (1 + theta dt A) u(t1) + dt b`.
  Here `dt = t2 - t1`, `A = A((t1 + t2)/2)`, `b = b((t1 + t2)/2)`, and `theta`
  is a float between `0` and `1`.
  Includes as particular cases the fully explicit scheme (`theta = 1`), the
  fully implicit scheme (`theta = 0`) and the Crank-Nicolson scheme
  (`theta = 0.5`).
  The scheme is first-order accurate in `t2 - t1` if `theta != 0.5` and
  second-order accurate if `theta = 0.5` (the Crank-Nicolson scheme).
  Note that while traditionally in the Crank-Nicolson scheme `A(t)` and `b(t)`
  are evaluated at `t1` and `t2` for the explicit and implicit substeps,
  respectively, we evaluate them at midpoint `t = (t1 + t2)/2`. This is also
  accurate to the second order (see e.g. [1], the paragraph after Eq. (14)), but
  more efficient.

  ### References:
  [1] I.V. Puzynin, A.V. Selin, S.I. Vinitsky, A high-order accuracy method for
  numerical solving of the time-dependent Schrodinger equation, Comput. Phys.
  Commun. 123 (1999), 1.
  https://www.sciencedirect.com/science/article/pii/S0010465599002246

  Args:
    theta: A float in range `[0, 1]`. A parameter used to mix implicit and
      explicit schemes together. Value of `0.0` corresponds to the fully
      implicit scheme, `1.0` to the fully explicit, and `0.5` to the
      Crank-Nicolson scheme. See, e.g., [1].

  Returns:
    A callable consumes the following arguments by keyword:
      1. inner_value_grid: Grid of solution values at the current time of
        the same `dtype` as `value_grid` and shape of `value_grid[..., 1:-1]`.
      2. t1: Lesser of the two times defining the step.
      3. t2: Greater of the two times defining the step.
      4. equation_params_fn: A callable that takes a scalar `Tensor` argument
        representing time, and constructs the tridiagonal matrix `A`
        (a tuple of three `Tensor`s, main, upper, and lower diagonals)
        and the inhomogeneous term `b`. All of the `Tensor`s are of the same
        `dtype` as `inner_value_grid` and of the shape broadcastable with the
        shape of `inner_value_grid`.
      5. backwards: A Python bool. Whether we're making a step backwards in
        time.
    The callable returns a `Tensor` of the same shape and `dtype` a
    `values_grid` and represents an approximate solution `u(t2)`.
  """
  if theta < 0 or theta > 1:
    raise ValueError(
        '`theta` should be in [0, 1]. Supplied: {}'.format(theta))

  def _marching_scheme(value_grid, t1, t2, equation_params_fn, backwards):
    """Constructs the time marching scheme."""
    (diag, superdiag, subdiag), inhomog_term = equation_params_fn(
        (t1 + t2) / 2)

    if theta == 0:  # fully implicit scheme
      rhs = value_grid
    else:
      rhs = _weighted_scheme_explicit_part(value_grid,
                                           diag, superdiag, subdiag,
                                           theta, t1, t2, backwards)

    if inhomog_term is not None:
      rhs += inhomog_term * (t2 - t1) * (-1 if backwards else 1)
    if theta < 1:
      # Note that if theta is `0`, `rhs` equals to the `value_grid`, so that the
      # fully implicit step is performed.
      return _weighted_scheme_implicit_part(rhs, diag, superdiag, subdiag,
                                            theta, t1, t2, backwards)
    return rhs

  return _marching_scheme


def _weighted_scheme_explicit_part(vec, diag, upper, lower, theta, t1, t2,
                                   backwards):
  """Explicit step of the weighted implicit-explicit scheme.

  Args:
    vec: A real dtype `Tensor` of shape `[num_equations, num_grid_points - 2]`.
      Represents the multiplied vector. "- 2" accounts for the boundary points,
      which the time-marching schemes do not touch.
    diag: A real dtype `Tensor` of the shape
      `[num_equations, num_grid_points - 2]`. Represents the main diagonal of
      a 3-diagonal matrix of the discretized PDE.
    upper: A real dtype `Tensor` of the shape
      `[num_equations, num_grid_points - 2]`. Represents the upper diagonal of
      a 3-diagonal matrix of the discretized PDE.
    lower:  A real dtype `Tensor` of the shape
      `[num_equations, num_grid_points - 2]`. Represents the lower diagonal of
      a 3-diagonal matrix of the discretized PDE.
    theta: A Python float between 0 and 1.
    t1: Smaller of the two times defining the step.
    t2: Greater of the two times defining the step.
    backwards: A Python bool. Whether we're making a step backwards in time.

  Returns:
    A tensor of the same shape and dtype as `vec`.
  """
  multiplier = theta * (t2 - t1) * (-1 if backwards else 1)
  diag = 1 + multiplier * diag
  upper = multiplier * upper
  lower = multiplier * lower

  # Multiply the tridiagonal matrix by the vector.
  diag_part = diag * vec
  zeros = tf.zeros_like(lower[..., :1])
  lower_part = tf.concat((zeros, lower[..., 1:] * vec[..., :-1]), axis=-1)
  upper_part = tf.concat((upper[..., :-1] * vec[..., 1:], zeros), axis=-1)
  return lower_part + diag_part + upper_part


def _weighted_scheme_implicit_part(vec, diag, upper, lower, theta, t1, t2,
                                   backwards):
  """Implicit step of the weighted implicit-explicit scheme.

  Args:
    vec: A real dtype `Tensor` of shape `[num_equations, num_grid_points - 2]`.
      Represents the multiplied vector. "- 2" accounts for the boundary points,
      which the time-marching schemes do not touch.
    diag: A real dtype `Tensor` of the shape
      `[num_equations, num_grid_points - 2]`. Represents the main diagonal of
      a 3-diagonal matrix of the discretized PDE.
    upper: A real dtype `Tensor` of the shape
      `[num_equations, num_grid_points - 2]`. Represents the upper diagonal of
      a 3-diagonal matrix of the discretized PDE.
    lower:  A real dtype `Tensor` of the shape
      `[num_equations, num_grid_points - 2]`. Represents the lower diagonal of
      a 3-diagonal matrix of the discretized PDE.
    theta: A Python float between 0 and 1.
    t1: Smaller of the two times defining the step.
    t2: Greater of the two times defining the step.
    backwards: A Python bool. Whether we're making a step backwards in time.

  Returns:
    A tensor of the same shape and dtype as `vec`.
  """
  multiplier = (1 - theta) * (t2 - t1) * (1 if backwards else -1)
  diag = 1 + multiplier * diag
  upper = multiplier * upper
  lower = multiplier * lower
  return tf.linalg.tridiagonal_solve([upper, diag, lower],
                                     vec,
                                     diagonals_format='sequence',
                                     transpose_rhs=True,
                                     partial_pivoting=False)
