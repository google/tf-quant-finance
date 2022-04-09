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
"""Helper functions to construct boundary conditions of PDEs."""

import functools


def dirichlet(boundary_values_fn):
  """Wrapper for Dirichlet boundary conditions to be used in PDE solvers.

  Example: the boundary value is 1 on both boundaries.

  ```python
  def lower_boundary_fn(t, location_grid):
    return 1

  def upper_boundary_fn(t, location_grid):
    return 0

  solver = fd_solvers.solve_forward(...,
      boundary_conditions = [(dirichlet(lower_boundary_fn),
                              dirichlet(upper_boundary_fn))],
      ...)
  ```

  Also can be used as a decorator:

  ```python
  @dirichlet
  def lower_boundary_fn(t, location_grid):
    return 1

  @dirichlet
  def upper_boundary_fn(t, location_grid):
    return 0

  solver = fd_solvers.solve_forward(...,
      boundary_conditions = [(lower_boundary_fn, upper_boundary_fn)],
      ...)
  ```

  Args:
    boundary_values_fn: Callable returning the boundary values at given time.
      Accepts two arguments - the moment of time and the current coordinate
      grid.
      Returns a number, a zero-rank Tensor or a Tensor of shape
      `batch_shape + grid_shape'`, where `grid_shape'` is grid_shape excluding
      the axis orthogonal to the boundary. For example, in 3D the value grid
      shape is `batch_shape + (z_size, y_size, x_size)`, and the boundary
      tensors on the planes `y = y_min` and `y = y_max` should be either scalars
      or have shape `batch_shape + (z_size, x_size)`. In 1D case this reduces
      to just `batch_shape`.

  Returns:
    Callable suitable for PDE solvers.
  """
  @functools.wraps(boundary_values_fn)
  def fn(t, x):
    # The boundary condition has the form alpha V + beta V_n = gamma, and we
    # should return a tuple (alpha, beta, gamma). In this case alpha = 1 and
    # beta = 0.
    return 1, None, boundary_values_fn(t, x)

  return fn


def neumann(boundary_normal_derivative_fn):
  """Wrapper for Neumann boundary condition to be used in PDE solvers.

  Example: the normal boundary derivative is 1 on both boundaries (i.e.
  `dV/dx = 1` on upper boundary, `dV/dx = -1` on lower boundary).

  ```python
  def lower_boundary_fn(t, location_grid):
    return 1

  def upper_boundary_fn(t, location_grid):
    return 1

  solver = fd_solvers.step_back(...,
      boundary_conditions = [(neumann(lower_boundary_fn),
                              neumann(upper_boundary_fn))],
      ...)
  ```

  Also can be used as a decorator:

  ```python
  @neumann
  def lower_boundary_fn(t, location_grid):
    return 1

  @neumann
  def upper_boundary_fn(t, location_grid):
    return 1

  solver = fd_solvers.solve_forward(...,
      boundary_conditions = [(lower_boundary_fn, upper_boundary_fn)],
      ...)
  ```

  Args:
    boundary_normal_derivative_fn: Callable returning the values of the
      derivative with respect to the exterior normal to the boundary at the
      given time.
      Accepts two arguments - the moment of time and the current coordinate
      grid.
      Returns a number, a zero-rank Tensor or a Tensor of shape
      `batch_shape + grid_shape'`, where `grid_shape'` is grid_shape excluding
      the axis orthogonal to the boundary. For example, in 3D the value grid
      shape is `batch_shape + (z_size, y_size, x_size)`, and the boundary
      tensors on the planes `y = y_min` and `y = y_max` should be either scalars
      or have shape `batch_shape + (z_size, x_size)`. In 1D case this reduces
      to just `batch_shape`.

  Returns:
    Callable suitable for PDE solvers.
  """
  @functools.wraps(boundary_normal_derivative_fn)
  def fn(t, x):
    # The boundary condition has the form alpha V_n + beta V_n = gamma, and we
    # should return a tuple (alpha, beta, gamma). In this case alpha = 0 and
    # beta = 1.
    return None, 1, boundary_normal_derivative_fn(t, x)

  return fn

__all__ = ["dirichlet", "neumann"]
