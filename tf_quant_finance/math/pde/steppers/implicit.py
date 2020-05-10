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
"""Implicit time marching scheme for parabolic PDEs."""

from tf_quant_finance.math.pde.steppers.parabolic_equation_stepper import parabolic_equation_step
from tf_quant_finance.math.pde.steppers.weighted_implicit_explicit import weighted_implicit_explicit_scheme


def implicit_step():
  """Creates a stepper function with implicit time marching scheme.

  Given a space-discretized equation

  ```
  du/dt = A(t) u(t) + b(t)
  ```
  (here `u` is a value vector, `A` and `b` are the matrix and the vector defined
  by the PDE), the implicit time marching scheme approximates the right-hand
  side with its value after the time step:

  ```
  (u(t2) - u(t1)) / (t2 - t1) = A(t2) u(t2) + b(t2)
  ```
  This scheme is stable, but is only first order accurate.
  Usually, Crank-Nicolson or Extrapolation schemes are preferable.

  More details can be found in `weighted_implicit_explicit.py` describing the
  weighted implicit-explicit scheme - implicit scheme is a special case
  with `theta = 0`.

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
    name = name or 'implicit_step'
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
                                   time_marching_scheme=implicit_scheme,
                                   dtype=dtype,
                                   name=name)
  return step_fn

implicit_scheme = weighted_implicit_explicit_scheme(theta=0)


__all__ = ['implicit_scheme', 'implicit_step']
