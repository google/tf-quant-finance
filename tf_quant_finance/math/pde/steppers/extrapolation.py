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
"""Extrapolation time marching scheme for parabolic PDEs."""

from tf_quant_finance.math.pde.steppers.implicit import implicit_scheme
from tf_quant_finance.math.pde.steppers.parabolic_equation_stepper import parabolic_equation_step


def extrapolation_step():
  """Creates a stepper function with Extrapolation time marching scheme.

  Extrapolation scheme combines two half-steps and the full time step to obtain
  desirable properties. See more details below in `extrapolation_scheme`.

  It is slower than Crank-Nicolson scheme, but deals better with value grids
  that have discontinuities. Consider also `oscillation_damped_crank_nicolson`,
  an efficient combination of Crank-Nicolson and Extrapolation schemes.

  Returns:
    Callable to be used in finite-difference PDE solvers (see fd_solvers.py).
  """
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
    name = name or 'extrapolation_step'
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
                                   time_marching_scheme=extrapolation_scheme,
                                   dtype=dtype,
                                   name=name)
  return step_fn


def extrapolation_scheme(value_grid, t1, t2, equation_params_fn):
  """Constructs extrapolation implicit-explicit scheme.

  Performs two implicit half-steps, one full implicit step, and combines them
  with such coefficients that ensure second-order errors. More computationally
  expensive than Crank-Nicolson scheme, but provides a better approximation for
  high-wavenumber components, which results in absence of oscillations typical
  for Crank-Nicolson scheme in case of non-smooth initial conditions. See [1]
  for details.

  #### References:
  [1]: D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods
  for Parabolic Partial Differential Equations. I. 1978
  SIAM Journal on Numerical Analysis. 15. 1212-1224.
  https://epubs.siam.org/doi/abs/10.1137/0715082

  Args:
    value_grid: A `Tensor` of real dtype. Grid of solution values at the current
      time.
    t1: Time before the step.
    t2: Time after the step.
    equation_params_fn: A callable that takes a scalar `Tensor` argument
      representing time and constructs the tridiagonal matrix `A`
      (a tuple of three `Tensor`s, main, upper, and lower diagonals)
      and the inhomogeneous term `b`. All of the `Tensor`s are of the same
      `dtype` as `inner_value_grid` and of the shape broadcastable with the
      shape of `inner_value_grid`.

  Returns:
    A `Tensor` of the same shape and `dtype` a
    `values_grid` and represents an approximate solution `u(t2)`.
  """
  first_half_step = implicit_scheme(value_grid, t1, (t1 + t2) / 2,
                                    equation_params_fn)
  two_half_steps = implicit_scheme(first_half_step, (t1 + t2) / 2, t2,
                                   equation_params_fn)

  full_step = implicit_scheme(value_grid, t1, t2, equation_params_fn)
  return 2 * two_half_steps - full_step


__all__ = ['extrapolation_scheme', 'extrapolation_step']
