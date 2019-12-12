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
"""Stepper for multidimensional parabolic PDE solving."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def multidim_parabolic_equation_step(
    time,
    next_time,
    coord_grid,
    value_grid,
    boundary_conditions,
    time_marching_scheme,
    second_order_coeff_fn=None,
    first_order_coeff_fn=None,
    zeroth_order_coeff_fn=None,
    dtype=None,
    name=None):
  """Performs one step in time to solve a multidimensional PDE.

  The PDE is of the form

  ```none
   V_{t} + sum_{ij} a_{ij}(t, r) * V_{ij} + sum_{i} b_{i}(t, r) * V_{i}
   + c(t, r) * V = 0,
  ```

  Here `V` is the unknown function, `V_{...}` denotes partial derivatives
  w.r.t. dimensions specified in curly brackets, `i` and `j` denote spatial
  dimensions, `r` is the spatial radius-vector.

  `a_{ij}`, `b_{i}`, and `c` are referred to as second order, first order and
  zeroth order coefficients, respectively. `a_{ij}` must be a positive-definite
  symmetrical matrix for all `r` and `t` (the properties of the matrix are not
  explicitly checked, however; in particular `a_{ij}` with `i > j` is never
  queried and assumed to be equal `a_{ji}`).

  For example, let's solve a 2d diffusion-convection-reaction equation with
  anisotropic diffusion:

  ```
  V_{t} + D_xx u_{xx} +  D_yy u_{yy} + 2 D_xy u_{xy} + mu_x u_{x} + mu_y u_{y}
  + nu u = 0
  ```

  ```python
  grid = grids.uniform_grid(
      minimums=[-10, -10],
      maximums=[10, 10],
      sizes=[200, 200],
      dtype=tf.float32)

  diff_coeff_xx = 0.4   # D_xx
  diff_coeff_yy = 0.25  # D_yy
  diff_coeff_xy = 0.1   # D_xy
  drift_x = 0.1         # mu_x
  drift_y = 0.3         # mu_y
  nu = 1
  time_step = 0.1
  final_t = 1
  final_variance = 1

  @dirichlet
  def zero_boundary(t, location_grid):
    return 0.0  # Let's set this simple boundary condition on all boundaries.


  def second_order_coeff_fn(t, locations_grid):
    del t, locations_grid  # Not used, because D_xx, D_yy, D_xy are constant.
    return [[diff_coeff_y, diff_coeff_xy], [diff_coeff_xy, diff_coeff_x]]

  def first_order_coeff_fn(t, locations_grid, dim):
    del t, locations_grid  # Not used, because mu_x, mu_y are constant.
    return [drift_y, drift_x]

  def zeroth_order_coeff_fn(t, locations_grid):
    del t, locations_grid  # Not used, because nu is constant.
    return nu

  # Final values for the PDE
  def _gaussian(xs, variance):
    return (np.exp(-np.square(xs) / (2 * variance))
            / np.sqrt(2 * np.pi * variance))

  final_values = tf.expand_dims(
      tf.constant(
          np.outer(
              _gaussian(ys, final_variance), _gaussian(xs, final_variance)),
          dtype=tf.float32),
      axis=0)

  step_fn = douglas_adi_scheme(theta=0.5)
  result = fd_solvers.solve(
      start_time=final_t,
      end_time=0,
      coord_grid=grid,
      values_grid=final_values,
      time_step=time_step,
      one_step_fn=step_fn,
      boundary_conditions=bound_cond,
      second_order_coeff_fn=second_order_coeff_fn,
      first_order_coeff_fn=first_order_coeff_fn,
      zeroth_order_coeff_fn=zeroth_order_coeff_fn,
      dtype=grid[0].dtype)
  ```
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
      conditions are supported.
      A list of tuples of size `n` (space dimension
      of the PDE). Each tuple consists of two callables representing the
      boundary conditions at the minimum and maximum values of the spatial
      variable indexed by the position in the list. E.g. for `n=2`, the length
      of `boundary_conditions` should be 2, `boundary_conditions[0][0]`
      describes the boundary `(y_min, x)`, and `boundary_conditions[1][0]`- the
      boundary `(y, x_min)`. The boundary conditions are accepted in the form
      `alpha(t, x) V + beta(t, x) V_n = gamma(t, x)`, where `V_n` is the
      derivative with respect to the exterior normal to the boundary.
      Each callable receives the current time `t` and the `coord_grid` at the
      current time, and should return a tuple of `alpha`, `beta`, and `gamma`.
      Each can be a number, a zero-rank `Tensor` or a `Rensor` whose shape is
      the grid shape with the corresponding dimension removed.
      For example, for a two-dimensional grid of shape `(b, ny, nx)`, where `b`
      is the batch size, `boundary_conditions[0][0]` should return a tuple of
      either numbers, zero-rank tensors or tensors of shape `(b, nx)`. Similarly
      for `boundary_conditions[1][0]`, except the tensor shape should be
      `(b, ny)`. `alpha` and `beta` can also be `None` in case of Neumann and
      Dirichlet conditions, respectively.
    time_marching_scheme: A callable which represents the time marching scheme
      for solving the PDE equation. If `u(t)` is space-discretized vector of the
      solution of a PDE, a time marching scheme approximately solves the
      equation `du/dt = A(t) u(t) + b(t)` for `u(t2)` given `u(t1)`, or vice
      versa if going backwards in time. Here `A` is a banded matrix containing
      contributions from the current and neighboring points in space, `b` is an
      arbitrary vector (inhomogeneous term).
      Multidimensional time marching schemes are usually based on the idea of
      ADI (alternating direction implicit) method: the time step is split into
      substeps, and in each substep only one dimension is treated "implicitly",
      while all the others are treated "explicitly". This way one has to solve
      only tridiagonal systems of equations, but not more complicated banded
      ones. A few examples of time marching schemes (Douglas, Craig-Sneyd, etc.)
      can be found in [1].
      The callable consumes the following arguments by keyword:
        1. inner_value_grid: Grid of solution values at the current time of
          the same `dtype` as `value_grid` and shape of `value_grid[..., 1:-1]`.
        2. t1: Lesser of the two times defining the step.
        3. t2: Greater of the two times defining the step.
        4. equation_params_fn: A callable that takes a scalar `Tensor` argument
          representing time and returns a tuple of two elements.
          The first one represents `A`. The length must be the number of
          dimensions (`n_dims`), and A[i] must have length `n_dims - i`.
          `A[i][0]` is a tridiagonal matrix representing influence of the
          neighboring points along the dimension `i`. It is a tuple of
          superdiagonal, diagonal, and subdiagonal parts of the tridiagonal
          matrix. The shape of these tensors must be same as of `value_grid`.
          superdiagonal[..., -1] and subdiagonal[..., 0] are ignored.
          `A[i][j]` with `i < j < n_dims` are Tensors with same shape of
          `value_grid` representing the influence of points placed diagonally
          from the given point in the plane of dimensions `i` and `j`.
          Contributions from all 4 diagonal directions in the plane are assumed
          equal up to a sign. This is the case when they come from the mixed
          second derivative terms, and the grid is evenly spaced. The
          contribution of mixed term to
          `(A u)_{k, l}` is `A[i][j-i]_{k, l} (u_{k+1, l+1} - u_{k+1, l-1} -
          u_{k-1, l+1} + u_{k-1, l-1})`, where `k` and `l` are indices in the
          plain of dimensions `i` and `j`, and the other indices are omitted.
          The second element in the tuple is a list of contributions to `b(t)`
          associated with each dimension. E.g. if `b(t)` comes from boundary
          conditions, then it is split correspondingly. Each element in the list
          is a Tensor with the shape of `value_grid`.
          For example a 2D problem with `value_grid.shape = (B, ny, nx)`, where
          `B` is the batch size. The elements `Aij` are non-zero if `i = j` or
          `i` is a neighbor of `j` in the x-y plane. Depict these non-zero
          elements on the grid as follows:
          ```
          a_xy    a_y-   -a_xy
          a_x-    a_0    a_x+
          -a_xy   a_y+   a_xy
          ```
          The callable should return
          ```
          ([[(a_y-, a_0y, a_y+), a_xy], [a_xy, (a_x-, a_0x, a_x+)]],
          [b_y, b_x])
          ```
          where `a_0x + a_0y = a_0` (the splitting is arbitrary). The second
          `a_xy` is ignored and can be replaced with `None`.
          All the elements `a_...` may be different for each point in the grid,
          so they are `Tensors` of shape `(B, ny, nx)`. `b_y` and `b_x` are also
          `Tensors` of that shape.
        5. n_dims: A Python integer, the spatial dimension of the PDE.
      The callable should return a `Tensor` of the same shape and `dtype` as
      `values_grid` that represents an approximate solution of the
      space-discretized PDE.
    second_order_coeff_fn: Callable returning the second order coefficient
      `a_{ij}(t, r)` evaluated at given time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `locations_grid`: a `Tensor` representing a grid of locations `r` at
          which the coefficient should be evaluated.
      Returns an object `A` such that `A[i][j]` is defined and
      `A[i][j]=a_{ij}(r, t)`, where `0 <= i < n_dims` and `i <= j < n_dims`.
      For example, the object may be a list of lists or a rank 2 Tensor.
      Only the elements with `j >= i` will be used, and it is assumed that
      `a_{ji} = a_{ij}`, so `A[i][j] with `j < i` may return `None`.
      Each `A[i][j]` should be a Number, a `Tensor` broadcastable to the
      shape of the grid represented by `locations_grid`, or `None` if
      corresponding term is absent in the equation. Also, the callable itself
      may be None, meaning there are no second-order derivatives in the
      equation.
      For example, for `n_dims=2`, the callable may return either
      `[[a_yy, a_xy], [a_xy, a_xx]]` or `[[a_yy, a_xy], [None, a_xx]]`.
    first_order_coeff_fn: Callable returning the first order coefficients
      `b_{i}(t, r)` evaluated at given time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `locations_grid`: a `Tensor` representing a grid of locations `r` at
          which the coefficient should be evaluated.
      Returns a list or an 1D `Tensor`, `i`-th element of which represents
      `b_{i}(t, r)`. Each element should be a Number, a `Tensor` broadcastable
       to the shape of of the grid represented by `locations_grid`, or None if
       corresponding term is absent in the equation. The callable itself may be
       None, meaning there are no first-order derivatives in the equation.
    zeroth_order_coeff_fn: Callable returning the zeroth order coefficient
      `c(t, r)` evaluated at given time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `locations_grid`: a `Tensor` representing a grid of locations `r` at
          which the coefficient should be evaluated.
      Should return a Number or a `Tensor` broadcastable to the shape of
      the grid represented by `locations_grid`. May also return None or be None
      if the shift term is absent in the equation.
    dtype: The dtype to use.
    name: The name to give to the ops.
      Default value: None which means `parabolic_equation_step` is used.

  Returns:
    A sequence of two `Tensor`s. The first one is a `Tensor` of the same
    `dtype` and `shape` as `coord_grid` and represents a new coordinate grid
    after one iteration. The second `Tensor` is of the same shape and `dtype`
    as`values_grid` and represents an approximate solution of the equation after
    one iteration.

  ### References:
  [1] Tinne Haentjens, Karek J. in't Hout. ADI finite difference schemes
  for the Heston-Hull-White PDE. https://arxiv.org/abs/1111.4087
  """
  with tf.compat.v1.name_scope(
      name, 'multidim_parabolic_equation_step',
      values=[time, next_time, coord_grid, value_grid]):

    time = tf.convert_to_tensor(time, dtype=dtype, name='time')
    next_time = tf.convert_to_tensor(next_time, dtype=dtype, name='next_time')
    coord_grid = [tf.convert_to_tensor(x, dtype=dtype,
                                       name='coord_grid_axis_{}'.format(ind))
                  for ind, x in enumerate(coord_grid)]
    value_grid = tf.convert_to_tensor(value_grid, dtype=dtype,
                                      name='value_grid')

    n_dims = len(coord_grid)

    second_order_coeff_fn = (second_order_coeff_fn or
                             (lambda *args: [[0.0] * n_dims] * n_dims))
    first_order_coeff_fn = (first_order_coeff_fn or
                            (lambda *args: [0.0] * n_dims))
    zeroth_order_coeff_fn = zeroth_order_coeff_fn or (lambda *args: 0.0)

    batch_rank = len(value_grid.shape.as_list()) - len(coord_grid)
    def equation_params_fn(t):
      return _construct_discretized_equation_params(coord_grid,
                                                    value_grid,
                                                    boundary_conditions,
                                                    second_order_coeff_fn,
                                                    first_order_coeff_fn,
                                                    zeroth_order_coeff_fn,
                                                    batch_rank,
                                                    t)
    inner_grid_in = _trim_boundaries(value_grid, batch_rank)

    inner_grid_out = time_marching_scheme(
        value_grid=inner_grid_in,
        t1=time,
        t2=next_time,
        equation_params_fn=equation_params_fn,
        n_dims=n_dims)

    updated_value_grid = _apply_boundary_conditions_after_step(
        inner_grid_out, coord_grid, boundary_conditions, batch_rank, next_time)

    return coord_grid, updated_value_grid


def _construct_discretized_equation_params(
    coord_grid,
    value_grid,
    boundary_conditions,
    second_order_coeff_fn,
    first_order_coeff_fn,
    zeroth_order_coeff_fn,
    batch_rank,
    t):
  """Constructs parameters of discretized equation."""
  second_order_coeffs = second_order_coeff_fn(t, coord_grid)
  first_order_coeffs = first_order_coeff_fn(t, coord_grid)
  zeroth_order_coeffs = zeroth_order_coeff_fn(t, coord_grid)

  matrix_params = []
  inhomog_terms = []

  zeroth_order_coeffs = _prepare_pde_coeff(zeroth_order_coeffs, value_grid,
                                           batch_rank)

  n_dims = len(coord_grid)
  for dim in range(n_dims):
    # 1. Construct contributions of dV/dx_dim and d^2V/dx_dim^2. This yields
    # a tridiagonal matrix.
    delta = _get_grid_delta(coord_grid, dim)  # Non-uniform grids not supported.

    if second_order_coeffs is not None:
      second_order_coeff = second_order_coeffs[dim][dim]
    else:
      second_order_coeff = 0

    if first_order_coeffs is not None:
      first_order_coeff = first_order_coeffs[dim]
    else:
      first_order_coeff = 0

    superdiag, diag, subdiag = (
        _construct_tridiagonal_matrix(
            value_grid, second_order_coeff, first_order_coeff, delta,
            batch_rank))

    # 2. Account for boundary conditions on boundaries orthogonal to dim.
    # This modifies the first and last row of the tridiagonal matrix and also
    # yields a contribution to the inhomogeneous term
    (superdiag, diag, subdiag), inhomog_term_contribution = (
        _apply_boundary_conditions_to_tridiagonal_and_inhomog_terms(
            dim, batch_rank, boundary_conditions, coord_grid,
            superdiag, diag, subdiag, delta, t))

    # 3. Evenly distribute shift term among tridiagonal matrices of each
    # dimension. The minus sign is because we move the shift term to rhs.
    if zeroth_order_coeffs is not None:
      diag += -zeroth_order_coeffs / n_dims

    matrix_params_row = [None]*dim + [(superdiag, diag, subdiag)]

    # 4. Construct contributions of mixed terms, d^2V/(dx_dim dx_dim2).
    for dim2 in range(dim + 1, n_dims):
      if second_order_coeffs is not None:
        mixed_coeff = second_order_coeffs[dim][dim2]
      else:
        mixed_coeff = 0
      mixed_term_contrib = (
          _construct_contribution_of_mixed_term(
              mixed_coeff, coord_grid, value_grid, dim, dim2, batch_rank))
      matrix_params_row.append(mixed_term_contrib)

    matrix_params.append(matrix_params_row)
    inhomog_terms.append(inhomog_term_contribution)

  return matrix_params, inhomog_terms


def _construct_tridiagonal_matrix(value_grid, second_order_coeff,
                                  first_order_coeff, delta, batch_rank):
  """Constructs contributions of first and non-mixed second order terms."""
  second_order_coeff = _prepare_pde_coeff(second_order_coeff, value_grid,
                                          batch_rank)
  first_order_coeff = _prepare_pde_coeff(first_order_coeff, value_grid,
                                         batch_rank)

  dxdx_contrib = second_order_coeff / (delta * delta)
  dx_contrib = first_order_coeff / (2 * delta)

  superdiag = -dx_contrib - dxdx_contrib
  subdiag = dx_contrib - dxdx_contrib
  diag = 2 * dxdx_contrib
  return superdiag, diag, subdiag


def _construct_contribution_of_mixed_term(
    coeff, coord_grid, value_grid, dim1, dim2, batch_rank):
  """Constructs contribution of a mixed derivative term."""
  delta_dim1 = _get_grid_delta(coord_grid, dim1)
  delta_dim2 = _get_grid_delta(coord_grid, dim2)

  mixed_coeff = _prepare_pde_coeff(coeff, value_grid, batch_rank)
  mixed_coeff *= -1  # Move to rhs
  # The contribution of d2V/dx_dim1 dx_dim2 is
  # mixed_coeff / (4 * delta_dim1 * delta_dim2), but there is also
  # d2V/dx_dim2 dx_dim1, so the contribution is doubled.
  return mixed_coeff / (2 * delta_dim1 * delta_dim2)


def _apply_boundary_conditions_to_tridiagonal_and_inhomog_terms(
    dim, batch_rank, boundary_conditions, coord_grid, superdiag, diag, subdiag,
    delta, t):
  """Updates contributions according to boundary conditions."""
  # This is analogous to _apply_boundary_conditions_to_discretized_equation in
  # pde_kernels.py. The difference is that we work with the given spatial
  # dimension. In particular, in all the tensor slices we have to slice
  # into the dimension `batch_rank + dim` instead of the last dimension.

  # Retrieve the boundary conditions in the form alpha V + beta V' = gamma.

  alpha_l, beta_l, gamma_l = boundary_conditions[dim][0](t, coord_grid)
  alpha_u, beta_u, gamma_u = boundary_conditions[dim][1](t, coord_grid)

  def reshape_fn(bound_coeff):
    """Reshapes boundary coefficient."""
    # Say the grid shape is (b, nz, ny, nx), and dim = 1.
    # The boundary condition coefficients are expected to have shape
    # (b, nz, nx). We need to:
    # - Trim the boundaries: nz -> nz-2, nx -> nx-2, because we work with
    # the inner part of the grid here.
    # - Expand dimension batch_rank+dim=2, because broadcasting won't always
    # do this correctly in subsequent computations: if a has shape (5, 1) and
    # b has shape (5,) then a*b has shape (5, 5)!
    # Thus this function turns (b, nz, nx) into (b, nz-2, 1, nx-2).
    return _reshape_boundary_conds(
        bound_coeff, trim_from=batch_rank, expand_dim_at=batch_rank + dim)

  alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = map(
      reshape_fn, (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u))

  dim += batch_rank
  subdiag_first = _slice(subdiag, dim, 0, 1)
  superdiag_last = _slice(superdiag, dim, -1, 0)
  diag_inner = _slice(diag, dim, 1, -1)

  if beta_l is None and beta_u is None:
    # Dirichlet conditions on both boundaries. In this case there are no
    # corrections to the tridiagonal matrix, so we can take a shortcut.
    first_inhomog_element = subdiag_first * gamma_l / alpha_l
    last_inhomog_element = superdiag_last * gamma_u / alpha_u
    inhomog_term = tf.concat(
        (first_inhomog_element, tf.zeros_like(diag_inner),
         last_inhomog_element), dim)
    return (superdiag, diag, subdiag), inhomog_term

  # A few more slices we're going to need.
  subdiag_last = _slice(subdiag, dim, -1, 0)
  subdiag_except_last = _slice(subdiag, dim, 0, -1)
  superdiag_first = _slice(superdiag, dim, 0, 1)
  superdiag_except_first = _slice(superdiag, dim, 1, 0)
  diag_first = _slice(diag, dim, 0, 1)
  diag_last = _slice(diag, dim, -1, 0)

  # Convert the boundary conditions into the form v0 = xi1 v1 + xi2 v2 + eta,
  # and calculate corrections to the tridiagonal matrix and the inhomogeneous
  # term.
  xi1, xi2, eta = _discretize_boundary_conditions(delta, delta, alpha_l,
                                                  beta_l, gamma_l)
  diag_first_correction = subdiag_first * xi1
  superdiag_correction = subdiag_first * xi2
  first_inhomog_element = subdiag_first * eta
  xi1, xi2, eta = _discretize_boundary_conditions(delta, delta, alpha_u,
                                                  beta_u, gamma_u)
  diag_last_correction = superdiag_last * xi1
  subdiag_correction = superdiag_last * xi2
  last_inhomog_element = superdiag_last * eta
  diag = tf.concat((diag_first + diag_first_correction, diag_inner,
                    diag_last + diag_last_correction), dim)
  superdiag = tf.concat(
      (superdiag_first + superdiag_correction, superdiag_except_first), dim)
  subdiag = tf.concat(
      (subdiag_except_last, subdiag_last + subdiag_correction), dim)
  inhomog_term = tf.concat((first_inhomog_element, tf.zeros_like(diag_inner),
                            last_inhomog_element), dim)
  return (superdiag, diag, subdiag), inhomog_term


def _apply_boundary_conditions_after_step(inner_grid_out, coord_grid,
                                          boundary_conditions,
                                          batch_rank, t):
  """Calculates and appends boundary values after making a step."""
  # After we've updated the values in the inner part of the grid according to
  # the PDE, we append the boundary values calculated using the boundary
  # conditions.
  # This is done using the discretized form of the boundary conditions,
  # v0 = xi1 v1 + xi2 v2 + eta.
  # This is analogous to _apply_boundary_conditions_after_step in
  # pde_kernels.py, except we have to restore the boundaries in each
  # dimension. For example, for n_dims=2, inner_grid_out has dimensions
  # (b, ny-2, nx-2), which then becomes (b, ny, nx-2) and finally (b, ny, nx).
  grid = inner_grid_out
  for dim in range(len(coord_grid)):
    grid = _apply_boundary_conditions_after_step_to_dim(
        dim, batch_rank, boundary_conditions, coord_grid, grid, t)

  return grid


def _apply_boundary_conditions_after_step_to_dim(
    dim, batch_rank, boundary_conditions, coord_grid, value_grid, t):
  """Calculates and appends boundaries orthogonal to `dim`."""
  # E.g. for n_dims = 3, and dim = 1, the expected input grid shape is
  # (b, nx, ny-2, nz-2), and the output shape is (b, nx, ny, nz-2).
  lower_value_first = _slice(value_grid, batch_rank + dim, 0, 1)
  lower_value_second = _slice(value_grid, batch_rank + dim, 1, 2)
  upper_value_first = _slice(value_grid, batch_rank + dim, -1, 0)
  upper_value_second = _slice(value_grid, batch_rank + dim, -2, -1)

  alpha_l, beta_l, gamma_l = boundary_conditions[dim][0](t, coord_grid)
  alpha_u, beta_u, gamma_u = boundary_conditions[dim][1](t, coord_grid)

  def reshape_fn(bound_coeff):
    # Say the grid shape is (b, nz, ny-2, nx-2), and dim = 1: we have already
    # restored the z-boundaries and now are restoring the y-boundaries.
    # The boundary condition coefficients are expected to have the shape
    # (b, nz, nx). We need to:
    # - Trim the boundaries which we haven't yet restored: nx -> nx-2.
    # - Expand dimension batch_rank+dim=2, because broadcasting won't always
    # do this correctly in subsequent computations.
    # Thus this functions turns (b, nz, nx) into (b, nz, 1, nx-2).
    return _reshape_boundary_conds(
        bound_coeff,
        trim_from=batch_rank + dim,
        expand_dim_at=batch_rank + dim)

  alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = map(
      reshape_fn, (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u))

  delta = _get_grid_delta(coord_grid, dim)
  xi1, xi2, eta = _discretize_boundary_conditions(delta, delta, alpha_l,
                                                  beta_l, gamma_l)
  first_value = (xi1 * lower_value_first + xi2 * lower_value_second + eta)
  xi1, xi2, eta = _discretize_boundary_conditions(delta, delta, alpha_u,
                                                  beta_u, gamma_u)
  last_value = (xi1 * upper_value_first + xi2 * upper_value_second + eta)
  return tf.concat((first_value, value_grid, last_value), batch_rank + dim)


def _get_grid_delta(coord_grid, dim):
  # Retrieves delta along given dimension, assuming the grid is uniform.
  return coord_grid[dim][1] - coord_grid[dim][0]


def _prepare_pde_coeff(raw_coeff, value_grid, batch_rank):
  # Converts values received from second_order_coeff_fn and similar Callables
  # into a format usable further down in the pipeline.
  if raw_coeff is None:
    raw_coeff = 0
  dtype = value_grid.dtype
  coeff = tf.convert_to_tensor(raw_coeff, dtype=dtype)
  coeff = tf.broadcast_to(coeff, tf.shape(value_grid))
  coeff = _trim_boundaries(coeff, batch_rank)
  return coeff


def _discretize_boundary_conditions(dx0, dx1, alpha, beta, gamma):
  """Discretizes boundary conditions."""
  # Converts a boundary condition given as alpha V + beta V_n = gamma,
  # where V_n is the derivative w.r.t. the normal to the boundary into
  # v0 = xi1 v1 + xi2 v2 + eta,
  # where v0 is the value on the boundary point of the grid, v1 and v2 - values
  # on the next two points on the grid.
  # The expressions are exactly the same for both boundaries.

  if beta is None:
    # Dirichlet condition.
    if alpha is None:
      raise ValueError(
          "Invalid boundary conditions: alpha and beta can't both be None.")
    zeros = tf.zeros_like(gamma)
    return zeros, zeros, gamma / alpha

  denom = beta * dx1 * (2 * dx0 + dx1)
  if alpha is not None:
    denom += alpha * dx0 * dx1 * (dx0 + dx1)
  xi1 = beta * (dx0 + dx1) * (dx0 + dx1) / denom
  xi2 = -beta * dx0 * dx0 / denom
  eta = gamma * dx0 * dx1 * (dx0 + dx1) / denom
  return xi1, xi2, eta


def _reshape_boundary_conds(raw_coeff, trim_from, expand_dim_at):
  """Reshapes boundary condition coefficients."""
  # If the coefficient is None, a number or a rank-0 tensor, return as-is.
  if (not tf.is_tensor(raw_coeff)
      or len(raw_coeff.shape.as_list()) == 0):  # pylint: disable=g-explicit-length-test
    return raw_coeff
  # See explanation why we trim boundaries and expand dims in places where this
  # function is used.
  coeff = _trim_boundaries(raw_coeff, trim_from)
  coeff = tf.expand_dims(coeff, expand_dim_at)
  return coeff


def _slice(tensor, dim, start, end):
  """Slices the tensor along given dimension."""
  # Performs a slice along the dimension dim. E.g. for tensor t of rank 3,
  # _slice(t, 1, 3, 5) is same as t[:, 3:5].
  # For a slice unbounded to the right, set end=0: _slice(t, 1, -3, 0) is same
  # as t[:, -3:].
  rank = len(tensor.shape.as_list())
  if start < 0:
    start += tf.dimension_value(tensor.shape.as_list()[dim])
  if end <= 0:
    end += tf.dimension_value(tensor.shape.as_list()[dim])
  slice_begin = np.zeros(rank, dtype=np.int32)
  slice_begin[dim] = start
  slice_size = -np.ones(rank, dtype=np.int32)
  slice_size[dim] = end - start
  return tf.slice(tensor, slice_begin, slice_size)


def _trim_boundaries(tensor, from_dim):
  """Trims tensor boundaries starting from given dimension."""
  # For example, if tensor has shape (a, b, c, d) and from_dim=1, then the
  # output tensor has shape (a, b-2, c-2, d-2).
  rank = len(tensor.shape.as_list())
  slice_begin = np.zeros(rank, dtype=np.int32)
  slice_size = np.zeros(rank, dtype=np.int32)
  for i in range(from_dim):
    slice_size[i] = tf.dimension_value(tensor.shape.as_list()[i])
  for i in range(from_dim, rank):
    slice_begin[i] = 1
    slice_size[i] = tf.dimension_value(tensor.shape.as_list()[i]) - 2
  return tf.slice(tensor, slice_begin, slice_size)
