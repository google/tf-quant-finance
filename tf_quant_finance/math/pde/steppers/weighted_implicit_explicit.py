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
"""Weighted implicit-explicit time marching scheme for parabolic PDEs."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math.pde.steppers.parabolic_equation_stepper import parabolic_equation_step


def weighted_implicit_explicit_step(theta):
  """Creates a stepper function with weighted implicit-explicit scheme.

  Given a space-discretized equation

  ```
  du/dt = A(t) u(t) + b(t)
  ```
  (here `u` is a value vector, `A` and `b` are the matrix and the vector defined
  by the PDE), the scheme approximates the right-hand side as a weighted average
  of values taken before and after a time step:

  ```
  (u(t2) - u(t1)) / (t2 - t1) = theta * (A(t1) u(t1) + b(t1))
     + (1 - theta) (A(t2) u(t2) + b(t2)).
  ```

  Includes as particular cases the implicit (`theta = 0`), explicit
  (`theta = 1`), and Crank-Nicolson (`theta = 0.5`) schemes.

  The scheme is stable for `theta >= 0.5`, is second order accurate if
  `theta = 0.5` (i.e. in Crank-Nicolson case), and first order accurate
  otherwise.

  More details can be found in `weighted_implicit_explicit_scheme` below.

  Args:
    theta: A float in range `[0, 1]`. A parameter used to mix implicit and
      explicit schemes together. Value of `0.0` corresponds to the fully
      implicit scheme, `1.0` to the fully explicit, and `0.5` to the
      Crank-Nicolson scheme.

  Returns:
    Callable to be used in finite-difference PDE solvers (see fd_solvers.py).
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
      inner_second_order_coeff_fn,
      inner_first_order_coeff_fn,
      num_steps_performed,
      dtype=None,
      name=None):
    """Performs the step."""
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
                                   inner_second_order_coeff_fn,
                                   inner_first_order_coeff_fn,
                                   time_marching_scheme=scheme,
                                   dtype=dtype,
                                   name=name)
  return step_fn


def weighted_implicit_explicit_scheme(theta):
  """Constructs weighted implicit-explicit scheme.

  Approximates the space-discretized equation of `du/dt = A(t) u(t) + b(t)` as
  ```
  (u(t2) - u(t1)) / (t2 - t1) = theta * (A u(t1) + b)
     + (1 - theta) (A u(t2) + b),
  ```
  where `A = A((t1 + t2)/2)`, `b = b((t1 + t2)/2)`, and `theta` is a float
  between `0` and `1`.

  Note that typically `A` and `b` are evaluated at `t1` and `t2` in
  the explicit and implicit terms respectively (the two terms of the right-hand
  side). Instead, we evaluate them at the midpoint `(t1 + t2)/2`, which saves
  some computation. One can check that evaluating at midpoint doesn't change the
  order of accuracy of the scheme: it is still second order accurate in
  `t2 - t1` if `theta = 0.5` and first order accurate otherwise.

  The solution is the following:
  `u(t2) = (1 - (1 - theta) dt A)^(-1) * (1 + theta dt A) u(t1) + dt b`.

  The main bottleneck here is inverting the matrix `(1 - (1 - theta) dt A)`.
  This matrix is tridiagonal (each point is influenced by the two neighbouring
  points), and thus the inversion can be efficiently performed using
  `tf.linalg.tridiagonal_solve`.

  #### References:
  [1] I.V. Puzynin, A.V. Selin, S.I. Vinitsky, A high-order accuracy method for
  numerical solving of the time-dependent Schrodinger equation, Comput. Phys.
  Commun. 123 (1999), 1.
  https://www.sciencedirect.com/science/article/pii/S0010465599002246

  Args:
    theta: A float in range `[0, 1]`. A parameter used to mix implicit and
      explicit schemes together. Value of `0.0` corresponds to the fully
      implicit scheme, `1.0` to the fully explicit, and `0.5` to the
      Crank-Nicolson scheme.

  Returns:
    A callable that consumes the following arguments by keyword:
      1. value_grid: Grid of values at time `t1`, i.e. `u(t1)`.
      2. t1: Time before the step.
      3. t2: Time after the step.
      4. equation_params_fn: A callable that takes a scalar `Tensor` argument
        representing time, and constructs the tridiagonal matrix `A`
        (a tuple of three `Tensor`s, main, upper, and lower diagonals)
        and the inhomogeneous term `b`. All of the `Tensor`s are of the same
        `dtype` as `value_grid` and of the shape broadcastable with the
        shape of `value_grid`.
    The callable returns a `Tensor` of the same shape and `dtype` as
    `value_grid` and represents an approximate solution `u(t2)`.
  """
  if theta < 0 or theta > 1:
    raise ValueError(
        '`theta` should be in [0, 1]. Supplied: {}'.format(theta))

  def _marching_scheme(value_grid, t1, t2, equation_params_fn):
    """Constructs the time marching scheme."""
    (diag, superdiag, subdiag), inhomog_term = equation_params_fn(
        (t1 + t2) / 2)

    if theta == 0:  # fully implicit scheme
      rhs = value_grid
    else:
      rhs = _weighted_scheme_explicit_part(value_grid, diag, superdiag, subdiag,
                                           theta, t1, t2)

    if inhomog_term is not None:
      rhs += inhomog_term * (t2 - t1)
    if theta < 1:
      # Note that if theta is `0`, `rhs` equals to the `value_grid`, so that the
      # fully implicit step is performed.
      return _weighted_scheme_implicit_part(rhs, diag, superdiag, subdiag,
                                            theta, t1, t2)
    return rhs

  return _marching_scheme


def _weighted_scheme_explicit_part(vec, diag, upper, lower, theta, t1, t2):
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

  Returns:
    A tensor of the same shape and dtype as `vec`.
  """
  multiplier = theta * (t2 - t1)
  diag = 1 + multiplier * diag
  upper = multiplier * upper
  lower = multiplier * lower

  # Multiply the tridiagonal matrix by the vector.
  diag_part = diag * vec
  zeros = tf.zeros_like(lower[..., :1])
  lower_part = tf.concat((zeros, lower[..., 1:] * vec[..., :-1]), axis=-1)
  upper_part = tf.concat((upper[..., :-1] * vec[..., 1:], zeros), axis=-1)
  return lower_part + diag_part + upper_part


def _weighted_scheme_implicit_part(vec, diag, upper, lower, theta, t1, t2):
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

  Returns:
    A tensor of the same shape and dtype as `vec`.
  """
  multiplier = (1 - theta) * (t1 - t2)
  diag = 1 + multiplier * diag
  upper = multiplier * upper
  lower = multiplier * lower
  return tf.linalg.tridiagonal_solve([upper, diag, lower],
                                     vec,
                                     diagonals_format='sequence',
                                     transpose_rhs=True,
                                     partial_pivoting=False)


__all__ = [
    'weighted_implicit_explicit_scheme',
    'weighted_implicit_explicit_step',
]
