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
"""Crank-Nicolson time marching scheme for parabolic PDEs."""

from tf_quant_finance.math.pde.steppers.parabolic_equation_stepper import parabolic_equation_step
from tf_quant_finance.math.pde.steppers.weighted_implicit_explicit import weighted_implicit_explicit_scheme


def crank_nicolson_step():
  """Creates a stepper function with Crank-Nicolson time marching scheme.

  Crank-Nicolson time marching scheme is one of the the most widely used schemes
  for 1D PDEs. Given a space-discretized equation

  ```
  du/dt = A(t) u(t) + b(t)
  ```
  (here `u` is a value vector, `A` and `b` are the matrix and the vector defined
  by the PDE), it approximates the right-hand side as an average of values taken
  before and after the time step:

  ```
  (u(t2) - u(t1)) / (t2 - t1) = (A(t1) u(t1) + b(t1) + A(t2) u(t2) + b(t2)) / 2.
  ```

  Crank-Nicolson has second order accuracy and is stable.

  More details can be found in `weighted_implicit_explicit.py` describing the
  weighted implicit-explicit scheme - Crank-Nicolson scheme is a special case
  with `theta = 0.5`.

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
    name = name or 'crank_nicolson_step'
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
                                   time_marching_scheme=crank_nicolson_scheme,
                                   dtype=dtype,
                                   name=name)
  return step_fn


crank_nicolson_scheme = weighted_implicit_explicit_scheme(theta=0.5)


__all__ = ['crank_nicolson_step', 'crank_nicolson_scheme']
