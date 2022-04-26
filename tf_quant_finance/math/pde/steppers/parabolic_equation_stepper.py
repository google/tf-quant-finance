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
"""Stepper for parabolic PDEs solving."""

import tensorflow.compat.v2 as tf

from tf_quant_finance import utils


def parabolic_equation_step(
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
    time_marching_scheme,
    dtype=None,
    name=None):
  """Performs one step of the one dimensional parabolic PDE solver.

  Typically one doesn't need to call this function directly, unless they have
  a custom time marching scheme. A simple stepper function for one-dimensional
  PDEs can be found in `crank_nicolson.py`.

  For a given solution (given by the `value_grid`) of a parabolic PDE at a given
  `time` on a given `coord_grid` computes an approximate solution at the
  `next_time` on the same coordinate grid. The parabolic differential equation
  is of the form:

  ```none
   dV/dt + a * d2(A * V)/dx2 + b * d(B * V)/dx + c * V = 0
  ```
  Here `a`, `A`, `b`, `B`, and `c` are known coefficients which may depend on
  `x` and `t`; `V = V(t, x)` is the solution to be found.

  Args:
    time: Real scalar `Tensor`. The time before the step.
    next_time: Real scalar `Tensor`. The time after the step.
    coord_grid: List of size 1 that contains a rank 1 real `Tensor` of
      has shape `[d]` or `B + [d]` where `d` is the size of the grid and `B` is
      a batch shape. Represents the coordinates of the grid points.
    value_grid: Real `Tensor` containing the function values at time
      `time` which have to be evolved to time `next_time`. The shape of the
      `Tensor` must broadcast with `B + [d]`. `B` is the batch
      dimensions (one or more), which allow multiple functions (with potentially
      different boundary/final conditions and PDE coefficients) to be evolved
      simultaneously.
    boundary_conditions: The boundary conditions. Only rectangular
      boundary conditions are supported. A list of tuples of size 1. The list
      element is a tuple that consists of two callables or `None`s representing
      the boundary conditions at the minimum and maximum values of the spatial
      variable indexed by the position in the list. `boundary_conditions[0][0]`
      describes the boundary at `x_min`, and `boundary_conditions[0][1]` the
      boundary at `x_max`. `None` values mean that the second order term on the
      boundary is assumed to be zero, i.e.,
      'dV/dt + b * d(B * V)/dx + c * V = 0'. This condition is appropriate for
      PDEs where the second order term disappears on the boundary. For not
      `None` values, the boundary conditions are accepted in the form
      `alpha(t) V + beta(t) V_n = gamma(t)`,
      where `V_n` is the derivative with respect to the exterior normal to the
      boundary. Each callable receives the current time `t` and the `coord_grid`
      at the current time, and should return a tuple of `alpha`, `beta`, and
      `gamma`. Each can be a number, a zero-rank `Tensor` or a `Tensor` of the
      batch shape.
      For example, for a grid of shape `(b, n)`, where `b` is the batch size,
      `boundary_conditions[0][0]` should return a tuple of either numbers,
      zero-rank tensors or tensors of shape `(b, n)`.
      `alpha` and `beta` can also be `None` in case of Neumann and
      Dirichlet conditions, respectively.
    second_order_coeff_fn: Callable returning the second order coefficient
      `a(t, r)` evaluated at given time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `locations_grid`: a `Tensor` representing a grid of locations `r` at
          which the coefficient should be evaluated.
      Returns an object `A` such that `A[0][0]` is defined and equals
      `a(r, t)`. `A[0][0]` should be a Number, a `Tensor` broadcastable to the
      shape of the grid represented by `locations_grid`, or `None` if
      corresponding term is absent in the equation. Also, the callable itself
      may be None, meaning there are no second-order derivatives in the
      equation.
    first_order_coeff_fn: Callable returning the first order coefficient
      `b(t, r)` evaluated at given time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `locations_grid`: a `Tensor` representing a grid of locations `r` at
          which the coefficient should be evaluated.
      Returns a list or an 1D `Tensor`, `0`-th element of which represents
      `b(t, r)`. This element should be a Number, a `Tensor` broadcastable
       to the shape of the grid represented by `locations_grid`, or None if
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
    inner_second_order_coeff_fn: Callable returning the coefficients under the
      second derivatives (i.e. `A(t, x)` above) at given time `t`. The
      requirements are the same as for `second_order_coeff_fn`.
    inner_first_order_coeff_fn: Callable returning the coefficients under the
      first derivatives (i.e. `B(t, x)` above) at given time `t`. The
      requirements are the same as for `first_order_coeff_fn`.
    time_marching_scheme: A callable which represents the time marching scheme
      for solving the PDE equation. If `u(t)` is space-discretized vector of the
      solution of the PDE, this callable approximately solves the equation
      `du/dt = A(t) u(t)` for `u(t1)` given `u(t2)`. Here `A` is a tridiagonal
      matrix. The callable consumes the following arguments by keyword:
        1. inner_value_grid: Grid of solution values at the current time of
          the same `dtype` as `value_grid` and shape of `value_grid[..., 1:-1]`.
        2. t1: Time before the step.
        3. t2: Time after the step.
        4. equation_params_fn: A callable that takes a scalar `Tensor` argument
          representing time, and constructs the tridiagonal matrix `A`
          (a tuple of three `Tensor`s, main, upper, and lower diagonals)
          and the inhomogeneous term `b`. All of the `Tensor`s are of the same
          `dtype` as `values_inner_value_grid` and of the shape
          broadcastable with the shape of `inner_value_grid`.
      The callable should return a `Tensor` of the same shape and `dtype` a
      `value_grid` and represents an approximate solution of the PDE after one
      iteraton.
    dtype: The dtype to use.
    name: The name to give to the ops.
      Default value: None which means `parabolic_equation_step` is used.

  Returns:
    A sequence of two `Tensor`s. The first one is a `Tensor` of the same
    `dtype` and `shape` as `coord_grid` and represents a new coordinate grid
    after one iteration. The second `Tensor` is of the same shape and `dtype`
    as`value_grid` and represents an approximate solution of the equation after
    one iteration.
  """
  with tf.compat.v1.name_scope(name, 'parabolic_equation_step',
                               [time, next_time, coord_grid, value_grid]):
    time = tf.convert_to_tensor(time, dtype=dtype, name='time')
    next_time = tf.convert_to_tensor(next_time, dtype=dtype, name='next_time')
    coord_grid = [tf.convert_to_tensor(x, dtype=dtype,
                                       name='coord_grid_axis_{}'.format(ind))
                  for ind, x in enumerate(coord_grid)]
    value_grid = tf.convert_to_tensor(value_grid, dtype=dtype,
                                      name='value_grid')

    if boundary_conditions[0][0] is None:
      # lower_index is used to build an inner grid on which boundary conditions
      # are imposed. For the default BC, no need for the value grid truncation.
      has_default_lower_boundary = True
      lower_index = 0
    else:
      has_default_lower_boundary = False
      lower_index = 1
    if boundary_conditions[0][1] is None:
      # For the default BC, no need for the the value grid truncation.
      upper_index = None
      has_default_upper_boundary = True
    else:
      upper_index = -1
      has_default_upper_boundary = False
    # Extract inner grid
    inner_grid_in = value_grid[..., lower_index:upper_index]
    coord_grid_deltas = coord_grid[0][..., 1:] - coord_grid[0][..., :-1]

    def equation_params_fn(t):
      return _construct_space_discretized_eqn_params(
          coord_grid, coord_grid_deltas, value_grid, boundary_conditions,
          has_default_lower_boundary, has_default_upper_boundary,
          second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn,
          inner_second_order_coeff_fn, inner_first_order_coeff_fn, t)

    inner_grid_out = time_marching_scheme(
        value_grid=inner_grid_in,
        t1=time,
        t2=next_time,
        equation_params_fn=equation_params_fn)

    updated_value_grid = _apply_boundary_conditions_after_step(
        inner_grid_out, boundary_conditions,
        has_default_lower_boundary, has_default_upper_boundary,
        coord_grid, coord_grid_deltas, next_time)
    return coord_grid, updated_value_grid


def _construct_space_discretized_eqn_params(
    coord_grid, coord_grid_deltas, value_grid,
    boundary_conditions, has_default_lower_boundary, has_default_upper_boundary,
    second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn,
    inner_second_order_coeff_fn, inner_first_order_coeff_fn, t):
  """Constructs the tridiagonal matrix and the inhomogeneous term."""
  # The space-discretized PDE has the form dv/dt = A(t) v(t) + b(t), where
  # v(t) is V(t, x) discretized by x, A(t) is a tridiagonal matrix and b(t) is
  # a vector. A(t) and b(t) depend on the PDE coefficients and the boundary
  # conditions. This function constructs A(t) and b(t). See construction of
  # A(t) e.g. in [Forsyth, Vetzal][1] (we denote `beta` and `gamma` from the
  # paper as `dx_coef` and `dxdx_coef`).

  # Get forward, backward and total differences.
  forward_deltas = coord_grid_deltas[..., 1:]
  backward_deltas = coord_grid_deltas[..., :-1]
  # Note that sum_deltas = 2 * central_deltas.
  sum_deltas = forward_deltas + backward_deltas

  # 3-diagonal matrix construction. See matrix `M` in [Forsyth, Vetzal][1].
  #  The `tridiagonal` matrix is of shape
  # `[value_dim, 3, num_grid_points]`.

  # Get the PDE coefficients and broadcast them to the shape of value grid.
  second_order_coeff_fn = second_order_coeff_fn or (lambda *args: [[None]])
  first_order_coeff_fn = first_order_coeff_fn or (lambda *args: [None])
  zeroth_order_coeff_fn = zeroth_order_coeff_fn or (lambda *args: None)
  inner_second_order_coeff_fn = inner_second_order_coeff_fn or (
      lambda *args: [[None]])
  inner_first_order_coeff_fn = inner_first_order_coeff_fn or (
      lambda *args: [None])

  second_order_coeff = _prepare_pde_coeffs(
      second_order_coeff_fn(t, coord_grid)[0][0], value_grid)
  first_order_coeff = _prepare_pde_coeffs(
      first_order_coeff_fn(t, coord_grid)[0], value_grid)
  zeroth_order_coeff = _prepare_pde_coeffs(
      zeroth_order_coeff_fn(t, coord_grid), value_grid)
  inner_second_order_coeff = _prepare_pde_coeffs(
      inner_second_order_coeff_fn(t, coord_grid)[0][0], value_grid)
  inner_first_order_coeff = _prepare_pde_coeffs(
      inner_first_order_coeff_fn(t, coord_grid)[0], value_grid)

  zeros = tf.zeros_like(value_grid[..., 1:-1])

  # Discretize zeroth-order term.
  if zeroth_order_coeff is None:
    diag_zeroth_order = zeros
  else:
    # Minus is due to moving to rhs.
    diag_zeroth_order = -zeroth_order_coeff[..., 1:-1]

  # Discretize first-order term.
  if first_order_coeff is None and inner_first_order_coeff is None:
    # No first-order term.
    superdiag_first_order = zeros
    diag_first_order = zeros
    subdiag_first_order = zeros
  else:
    superdiag_first_order = -backward_deltas / (sum_deltas * forward_deltas)
    subdiag_first_order = forward_deltas / (sum_deltas * backward_deltas)
    diag_first_order = -superdiag_first_order - subdiag_first_order
    if first_order_coeff is not None:
      superdiag_first_order *= first_order_coeff[..., 1:-1]
      subdiag_first_order *= first_order_coeff[..., 1:-1]
      diag_first_order *= first_order_coeff[..., 1:-1]
    if inner_first_order_coeff is not None:
      superdiag_first_order *= inner_first_order_coeff[..., 2:]
      subdiag_first_order *= inner_first_order_coeff[..., :-2]
      diag_first_order *= inner_first_order_coeff[..., 1:-1]

  # Discretize second-order term.
  if second_order_coeff is None and inner_second_order_coeff is None:
    # No second-order term.
    superdiag_second_order = zeros
    diag_second_order = zeros
    subdiag_second_order = zeros
  else:
    superdiag_second_order = -2 / (sum_deltas * forward_deltas)
    subdiag_second_order = -2 / (sum_deltas * backward_deltas)
    diag_second_order = -superdiag_second_order - subdiag_second_order
    if second_order_coeff is not None:
      superdiag_second_order *= second_order_coeff[..., 1:-1]
      subdiag_second_order *= second_order_coeff[..., 1:-1]
      diag_second_order *= second_order_coeff[..., 1:-1]
    if inner_second_order_coeff is not None:
      superdiag_second_order *= inner_second_order_coeff[..., 2:]
      subdiag_second_order *= inner_second_order_coeff[..., :-2]
      diag_second_order *= inner_second_order_coeff[..., 1:-1]

  superdiag = superdiag_first_order + superdiag_second_order
  subdiag = subdiag_first_order + subdiag_second_order
  diag = diag_zeroth_order + diag_first_order + diag_second_order
  # Apply default BC, if needed. This adds extra points to the diagonal terms
  # coming from discretization of 'V_t + b * d(B * V)/dx + c * V = 0'.
  (
      subdiag, diag, superdiag
  ) = _apply_default_boundary(subdiag, diag, superdiag,
                              zeroth_order_coeff,
                              inner_first_order_coeff,
                              first_order_coeff,
                              forward_deltas,
                              backward_deltas,
                              has_default_lower_boundary,
                              has_default_upper_boundary)
  # Apply Robin boundary conditions
  return _apply_robin_boundary_conditions(
      value_grid, boundary_conditions,
      has_default_lower_boundary, has_default_upper_boundary,
      coord_grid, coord_grid_deltas, diag, superdiag, subdiag, t)


def _apply_default_boundary(subdiag, diag, superdiag,
                            zeroth_order_coeff,
                            inner_first_order_coeff,
                            first_order_coeff,
                            forward_deltas,
                            backward_deltas,
                            has_default_lower_boundary,
                            has_default_upper_boundary):
  """Update discretization matrix for default boundary conditions."""
  # For default BC, we need to add spatial discretizations of
  # 'b * d(B * V)/dx + c * V' to the boundaries

  # Extract batch shape
  batch_shape = utils.get_shape(diag)[:-1]
  # Set zero coeff if it is None
  if zeroth_order_coeff is None:
    zeroth_order_coeff = tf.zeros([1], dtype=diag.dtype)
  # Updates for lower BC
  if has_default_lower_boundary:
    (
        subdiag, diag, superdiag
    ) = _apply_default_lower_boundary(subdiag, diag, superdiag,
                                      zeroth_order_coeff,
                                      inner_first_order_coeff,
                                      first_order_coeff,
                                      forward_deltas,
                                      batch_shape)

  # Updates for upper BC
  if has_default_upper_boundary:
    (
        subdiag, diag, superdiag
    ) = _apply_default_upper_boundary(subdiag, diag, superdiag,
                                      zeroth_order_coeff,
                                      inner_first_order_coeff,
                                      first_order_coeff,
                                      backward_deltas,
                                      batch_shape)
  return subdiag, diag, superdiag


def _apply_default_lower_boundary(subdiag, diag, superdiag,
                                  zeroth_order_coeff,
                                  inner_first_order_coeff,
                                  first_order_coeff,
                                  forward_deltas,
                                  batch_shape):
  """Update discretization matrix for default lower boundary conditions."""
  # TODO(b/185337444): Use second order discretization for the boundary points
  # Here we append '-b(t, x_min) * (B(t, x_min) * V(t, x_min)) / delta' to
  # diagonal and
  # 'b(t, x_min) * (B(t, x_min + delta) * V(t, x_min + delta)) / delta' to
  # superdiagonal
  # Update superdiag
  if inner_first_order_coeff is None:
    # Set to ones, if inner coefficient is not supplied
    inner_coeff = tf.constant([1, 1], dtype=diag.dtype)
  else:
    inner_coeff = inner_first_order_coeff
  if first_order_coeff is None:
    if inner_first_order_coeff is None:
      # Corresponds to B(t, x_min)
      extra_first_order_coeff = tf.zeros(batch_shape, dtype=diag.dtype)
    else:
      extra_first_order_coeff = tf.ones(batch_shape, dtype=diag.dtype)
  else:
    extra_first_order_coeff = first_order_coeff[..., 0]
  extra_superdiag_coeff = (inner_coeff[..., 1] * extra_first_order_coeff
                           / forward_deltas[..., 0])
  # Minus is due to moving to rhs.
  superdiag = _append_first(-extra_superdiag_coeff, superdiag)
  # Update diagonal
  extra_diag_coeff = (-inner_coeff[..., 0] * extra_first_order_coeff
                      / forward_deltas[..., 0]
                      + zeroth_order_coeff[..., 0])
  # Minus is due to moving to rhs.
  diag = _append_first(-extra_diag_coeff, diag)
  # Update subdiagonal
  subdiag = _append_first(tf.zeros_like(extra_diag_coeff), subdiag)
  return subdiag, diag, superdiag


def _apply_default_upper_boundary(subdiag, diag, superdiag,
                                  zeroth_order_coeff,
                                  inner_first_order_coeff,
                                  first_order_coeff,
                                  backward_deltas,
                                  batch_shape):
  """Update discretization matrix for default upper boundary conditions."""
  # TODO(b/185337444): Use second order discretization for the boundary points
  # Here we append '-b(t, x_min) * (B(t, x_max) * V(t, x_max)) / delta' to
  # diagonal and
  # 'b(t, x_max) * (B(t, x_max - delta) * V(t, x_max - delta)) / delta' to
  # subdiagonal
  # Update diagonal
  if inner_first_order_coeff is None:
    inner_coeff = tf.constant([1, 1], dtype=diag.dtype)
  else:
    inner_coeff = inner_first_order_coeff
  if first_order_coeff is None:
    if inner_first_order_coeff is None:
      # Corresponds to B(t, x_max)
      extra_first_order_coeff = tf.zeros(batch_shape, dtype=diag.dtype)
    else:
      extra_first_order_coeff = tf.ones(batch_shape, dtype=diag.dtype)
  else:
    extra_first_order_coeff = first_order_coeff[..., -1]
  extra_diag_coeff = (inner_coeff[..., -1] * extra_first_order_coeff
                      / backward_deltas[..., -1]
                      + zeroth_order_coeff[..., -1])
  # Minus is due to moving to rhs.
  diag = _append_last(diag, -extra_diag_coeff)
  # Update subdiagonal
  extra_sub_coeff = (-inner_coeff[..., -2] * extra_first_order_coeff
                     / backward_deltas[..., -1])
  # Minus is due to moving to rhs.
  subdiag = _append_last(subdiag, -extra_sub_coeff)
  # Update superdiag
  superdiag = _append_last(superdiag, -tf.zeros_like(extra_diag_coeff))
  return subdiag, diag, superdiag


def _apply_robin_boundary_conditions(
    value_grid,
    boundary_conditions,
    has_default_lower_boundary,
    has_default_upper_boundary,
    coord_grid, coord_grid_deltas,
    diagonal,
    upper_diagonal,
    lower_diagonal, t):
  """Updates space-discretized equation according to boundary conditions."""
  # Without taking into account the boundary conditions, the space-discretized
  # PDE has the form dv/dt = A(t) v(t), where v(t) is V(t, x) discretized by
  # x, and A is the tridiagonal matrix defined by coefficients of the PDE.
  # Boundary conditions change the first and the last row of A and introduce
  # the inhomogeneous term, so the equation becomes dv/dt = A'(t) v(t) + b(t),
  # where A' is the modified matrix, and b is a vector.
  # This function receives A and returns A' and b.
  # We do not update the rows of A where the boundary condition is default

  # If both boundaries are default, there is no need to update the
  # space-discretization matrix
  if (has_default_lower_boundary and
      has_default_upper_boundary):
    return (diagonal, upper_diagonal, lower_diagonal), tf.zeros_like(diagonal)

  batch_shape = utils.get_shape(value_grid)[:-1]
  # Retrieve the boundary conditions in the form alpha V + beta V' = gamma.
  if has_default_lower_boundary:
    # No need for the BC as default BC was applied
    alpha_l, beta_l, gamma_l = None, None, None
  else:
    alpha_l, beta_l, gamma_l = boundary_conditions[0][0](t, coord_grid)
  if has_default_upper_boundary:
    # No need for the BC as default BC was applied
    alpha_u, beta_u, gamma_u = None, None, None
  else:
    alpha_u, beta_u, gamma_u = boundary_conditions[0][1](t, coord_grid)

  alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = (
      _prepare_boundary_conditions(b, value_grid)
      for b in (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u))

  if beta_l is None and beta_u is None:
    # Dirichlet or default conditions on both boundaries. In this case there are
    # no corrections to the tridiagonal matrix, so we can take a shortcut.
    if has_default_lower_boundary:
      # Inhomogeneous term is zero for default BC
      first_inhomog_element = tf.zeros(batch_shape, dtype=value_grid.dtype)
    else:
      first_inhomog_element = lower_diagonal[..., 0] * gamma_l / alpha_l
    if has_default_upper_boundary:
      # Inhomogeneous term is zero for default BC
      last_inhomog_element = tf.zeros(batch_shape, dtype=value_grid.dtype)
    else:
      last_inhomog_element = upper_diagonal[..., -1] * gamma_u / alpha_u
    inhomog_term = _append_first_and_last(first_inhomog_element,
                                          tf.zeros_like(diagonal[..., 1:-1]),
                                          last_inhomog_element)
    return (diagonal, upper_diagonal, lower_diagonal), inhomog_term

  # Convert the boundary conditions into the form v0 = xi1 v1 + xi2 v2 + eta,
  # and calculate corrections to the tridiagonal matrix and the inhomogeneous
  # term.
  if has_default_lower_boundary:
    # No update for the default BC
    first_inhomog_element = tf.zeros(batch_shape, dtype=value_grid.dtype)
    diag_first_correction = 0
    upper_diag_correction = 0
  else:
    # Robin BC case for the lower bound
    xi1, xi2, eta = _discretize_boundary_conditions(coord_grid_deltas[0],
                                                    coord_grid_deltas[1],
                                                    alpha_l,
                                                    beta_l, gamma_l)
    diag_first_correction = lower_diagonal[..., 0] * xi1
    upper_diag_correction = lower_diagonal[..., 0] * xi2
    first_inhomog_element = lower_diagonal[..., 0] * eta

  if has_default_upper_boundary:
    # No update for the default BC
    last_inhomog_element = tf.zeros(batch_shape, dtype=value_grid.dtype)
    diag_last_correction = 0
    lower_diag_correction = 0
  else:
    # Robin BC case for the upper bound
    xi1, xi2, eta = _discretize_boundary_conditions(coord_grid_deltas[-1],
                                                    coord_grid_deltas[-2],
                                                    alpha_u,
                                                    beta_u, gamma_u)
    diag_last_correction = upper_diagonal[..., -1] * xi1
    lower_diag_correction = upper_diagonal[..., -1] * xi2
    last_inhomog_element = upper_diagonal[..., -1] * eta

  # Update spatial discretization matrix, where appropriate
  diagonal = _append_first_and_last(diagonal[..., 0] + diag_first_correction,
                                    diagonal[..., 1:-1],
                                    diagonal[..., -1] + diag_last_correction)
  upper_diagonal = _append_first(
      upper_diagonal[..., 0] + upper_diag_correction, upper_diagonal[..., 1:])
  lower_diagonal = _append_last(
      lower_diagonal[..., :-1],
      lower_diagonal[..., -1] + lower_diag_correction)
  inhomog_term = _append_first_and_last(first_inhomog_element,
                                        tf.zeros_like(diagonal[..., 1:-1]),
                                        last_inhomog_element)
  return (diagonal, upper_diagonal, lower_diagonal), inhomog_term


def _apply_boundary_conditions_after_step(
    inner_grid_out,
    boundary_conditions,
    has_default_lower_boundary,
    has_default_upper_boundary,
    coord_grid, coord_grid_deltas,
    time_after_step):
  """Calculates and appends boundary values after making a step."""
  # After we've updated the values in the inner part of the grid according to
  # the PDE, we append the boundary values calculated using the boundary
  # conditions.
  # This is done using the discretized form of the boundary conditions,
  # v0 = xi1 v1 + xi2 v2 + eta.
  # We do not change the rows of the spatial discretization matrix where the
  # boundary condition is default

  if has_default_lower_boundary:
    # No update for the default BC
    first_value = None
  else:
    # Robin BC case
    alpha, beta, gamma = boundary_conditions[0][0](time_after_step,
                                                   coord_grid)
    alpha, beta, gamma = (
        _prepare_boundary_conditions(b, inner_grid_out)
        for b in (alpha, beta, gamma))
    xi1, xi2, eta = _discretize_boundary_conditions(coord_grid_deltas[0],
                                                    coord_grid_deltas[1],
                                                    alpha, beta, gamma)
    first_value = (
        xi1 * inner_grid_out[..., 0] + xi2 * inner_grid_out[..., 1] + eta)

  if has_default_upper_boundary:
    # No update for the default BC
    last_value = None
  else:
    # Robin BC case
    alpha, beta, gamma = boundary_conditions[0][1](time_after_step,
                                                   coord_grid)
    alpha, beta, gamma = (
        _prepare_boundary_conditions(b, inner_grid_out)
        for b in (alpha, beta, gamma))
    xi1, xi2, eta = _discretize_boundary_conditions(coord_grid_deltas[-1],
                                                    coord_grid_deltas[-2],
                                                    alpha, beta, gamma)
    last_value = (
        xi1 * inner_grid_out[..., -1] + xi2 * inner_grid_out[..., -2] + eta)

  return _append_first_and_last(first_value, inner_grid_out, last_value)


def _prepare_pde_coeffs(raw_coeffs, value_grid):
  """Prepares values received from second_order_coeff_fn and similar."""
  if raw_coeffs is None:
    return None
  dtype = value_grid.dtype
  coeffs = tf.convert_to_tensor(raw_coeffs, dtype=dtype)
  broadcast_shape = utils.get_shape(value_grid)
  coeffs = tf.broadcast_to(coeffs, broadcast_shape)
  return coeffs


def _prepare_boundary_conditions(boundary_tensor, value_grid):
  """Prepares values received from boundary_condition callables."""
  if boundary_tensor is None:
    return None
  boundary_tensor = tf.convert_to_tensor(boundary_tensor, value_grid.dtype)
  # Broadcast to batch dimensions.
  broadcast_shape = utils.get_shape(value_grid)[:-1]
  return tf.broadcast_to(boundary_tensor, broadcast_shape)


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


def _append_first_and_last(first, inner, last):
  if first is None:
    return _append_last(inner, last)
  if last is None:
    return _append_first(first, inner)
  return tf.concat((tf.expand_dims(first, axis=-1),
                    inner,
                    tf.expand_dims(last, axis=-1)), axis=-1)


def _append_first(first, rest):
  if first is None:
    return rest
  return tf.concat((tf.expand_dims(first, axis=-1), rest), axis=-1)


def _append_last(rest, last):
  if last is None:
    return rest
  return tf.concat((rest, tf.expand_dims(last, axis=-1)), axis=-1)


__all__ = ['parabolic_equation_step']
