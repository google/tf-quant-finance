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
"""Implicit time marching scheme for parabolic PDEs."""

from tf_quant_finance.experimental.pde_v2.steppers.parabolic_equation_stepper import parabolic_equation_step
from tf_quant_finance.experimental.pde_v2.steppers.weighted_implicit_explicit import weighted_implicit_explicit_scheme


def implicit_step(
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
  """Performs one step of the parabolic PDE solver using implicit scheme.

  For a given solution (given by the `value_grid`) of a parabolic PDE at a given
  `time` on a given `coord_grid` computes an approximate solution at the
  `next_time` on the same coordinate grid using implicit time marching scheme.
  (see, e.g., [1]). The parabolic PDE is of the form:

  ```none
   V_{t} + a(t, x) * V_{xx} + b(t, x) * V_{x} + c(t, x) * V = 0
  ```

  Here `V = V(t, x)` is a solution to the 2-dimensional PDE. `V_{t}` is the
  derivative over time and `V_{x}` and `V_{xx}` are the first and second
  derivatives over the space component. For a solution to be well-defined, it is
  required for `a` to be positive on its domain.

  See `fd_solvers.solve` for an example use case.

  ### References:
  [1]: P.A. Forsyth, K.R. Vetzal. Quadratic Convergence for Valuing American
    Options Using A Penalty Method. Journal on Scientific Computing, 2002.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.9066&rep=rep1&type=pdf

  Args:
    time: Real scalar `Tensor`. The time before the step.
    next_time: Real scalar `Tensor`. The time after the step.
    coord_grid: List of `n` rank 1 real `Tensor`s. `n` is the dimension of the
      domain. The i-th `Tensor` has shape, `[d_i]` where `d_i` is the size of
      the grid along axis `i`. The coordinates of the grid points. Corresponds
      to the spatial grid `G` above.
    value_grid: Real `Tensor` containing the function values at time
      `time` which have to be evolved to time `next_time`. The shape of the
      `Tensor` must broadcast with `B + [d_1, d_2, ..., d_n]`. `B` is the batch
      dimensions (one or more), which allow multiple functions (with potentially
      different boundary/final conditions and PDE coefficients) to be evolved
      simultaneously.
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
    second_order_coeff_fn: See the spec in fd_solvers.solve.
    first_order_coeff_fn: See the spec in fd_solvers.solve.
    zeroth_order_coeff_fn: See the spec in fd_solvers.solve.
    inner_second_order_coeff_fn: See the spec in fd_solvers.solve.
    inner_first_order_coeff_fn: See the spec in fd_solvers.solve.
    num_steps_performed: Python `int`. Number of steps performed so far.
    dtype: The dtype to use.
    name: The name to give to the ops.
      Default value: None which means `implicit_step` is used.

  Returns:
    A sequence of two `Tensor`s. The first one is a `Tensor` of the same
    `dtype` and `shape` as `coord_grid` and represents a new coordinate grid
    after one iteration. The second `Tensor` is of the same shape and `dtype`
    as`values_grid` and represents an approximate solution of the equation after
    one iteration.
  """
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

implicit_scheme = weighted_implicit_explicit_scheme(theta=0)
