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
"""Douglas ADI method for solving multidimensional parabolic PDEs."""

import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.math.pde.steppers.multidim_parabolic_equation_stepper import multidim_parabolic_equation_step


def douglas_adi_step(theta=0.5):
  """Creates a stepper function with Crank-Nicolson time marching scheme.

  Douglas ADI scheme is the simplest time marching scheme for solving parabolic
  PDEs with multiple spatial dimensions. The time step consists of several
  substeps: the first one is fully explicit, and the following `N` steps are
  implicit with respect to contributions of one of the `N` axes (hence "ADI" -
  alternating direction implicit). See `douglas_adi_scheme` below for more
  details.

  Args:
    theta: positive Number. `theta = 0` corresponds to fully explicit scheme.
    The larger `theta` the stronger are the corrections by the implicit
    substeps. The recommended value is `theta = 0.5`, because the scheme is
    second order accurate in that case, unless mixed second derivative terms are
    present in the PDE.
  Returns:
    Callable to be used in finite-difference PDE solvers (see fd_solvers.py).
  """
  def _step_fn(
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
    name = name or 'douglas_adi_step'
    return multidim_parabolic_equation_step(time,
                                            next_time,
                                            coord_grid,
                                            value_grid,
                                            boundary_conditions,
                                            douglas_adi_scheme(theta),
                                            second_order_coeff_fn,
                                            first_order_coeff_fn,
                                            zeroth_order_coeff_fn,
                                            inner_second_order_coeff_fn,
                                            inner_first_order_coeff_fn,
                                            dtype=dtype,
                                            name=name)
  return _step_fn


def douglas_adi_scheme(theta):
  """Applies Douglas time marching scheme (see [1] and Eq. 3.1 in [2]).

  Time marching schemes solve the space-discretized equation
  `du_inner/dt = A(t) u_inner(t) + A_mixed u(t) + b(t)`,
  where `u`, `u_inner` and `b` are vectors and `A`, `A_mixed` are matrices.
  `u_inner` is `u` with all boundaries having Robin boundary conditions
  trimmed and `A_mixed` are contributions of mixed derivative terms.
  See more details in multidim_parabolic_equation_stepper.py.

  In Douglas scheme (as well as other ADI schemes), the matrix `A` is
  represented as sum `A = sum_i A_i`. `A_i` is the contribution of
  terms with partial derivatives w.r.t. dimension `i`. The shift term is split
  evenly between `A_i`. Similarly, inhomogeneous term is represented as sum
  `b = sum_i b_i`, where `b_i` comes from boundary conditions on boundary
  orthogonal to dimension `i`.

  Given the current values vector u(t1), the step is defined as follows
  (using the notation of Eq. 3.1 in [2]):
  `Y_0 = (1 + (A(t1) + A_mixed(t1)) dt) U_{n-1} + b(t1) dt`,
  `Y_j = Y_{j-1} + theta dt (A_j(t2) Y_j - A_j(t1) U_{n-1} + b_j(t2) - b_j(t1))`
  for each spatial dimension `j`, and
  `U_n = Y_{n_dims-1}`.

  Here the parameter `theta` is a non-negative number, `U_{n-1} = u(t1)`,
  `U_n = u(t2)`, and `dt = t2 - t1`.

  Note: Douglas scheme is only first-order accurate if mixed terms are
  present. More advanced schemes, such as Craig-Sneyd scheme, are needed to
  achieve the second-order accuracy.

  #### References:
  [1] Douglas Jr., Jim (1962), "Alternating direction methods for three space
    variables", Numerische Mathematik, 4 (1): 41-63
  [2] Tinne Haentjens, Karek J. in't Hout. ADI finite difference schemes for
    the Heston-Hull-White PDE. https://arxiv.org/abs/1111.4087

  Args:
    theta: Number between 0 and 1 (see the step definition above). `theta = 0`
      corresponds to fully-explicit scheme.

  Returns:
    A callable consumes the following arguments by keyword:
      1. inner_value_grid: Grid of solution values at the current time of
        the same `dtype` as `value_grid` and shape of
        `batch_shape` + `[d_1 - 2 + n_def_i , ..., d_n -2 + n_def_i]`
        where `d_i` is the number of space discretization points along dimension
        `i` and `n_def_i` is the number of default boundaries along that
        dimension. `n_def_i` takes values 0, 1, 2 (default boundary),
      2. t1: Time before the step.
      3. t2: Time after the step.
      4. equation_params_fn: A callable that takes a scalar `Tensor` argument
        representing time, and returns a tuple of two objects:
          * First object is a nested list `L` such that `L[i][i]` is a tuple of
          three `Tensor`s, main, upper, and lower diagonal of the tridiagonal
          matrix `A` in a direction `i`. Each element `L[i][j]` corresponds
          to the mixed terms and is either None (meaning there are no mixed
          terms present) or a tuple of `Tensor`s representing contributions of
          mixed terms in directions (i + 1, j + 1), (i + 1, j - 1),
          (i - 1, j + 1), and (i - 1, j - 1).
          * The second object is a tuple of inhomogeneous terms for each
          dimension.
        All of the `Tensor`s are of the same `dtype` as `inner_value_grid` and
        of the shape broadcastable with the shape of `inner_value_grid`.
      5. A callable that accepts a `Tensor` of shape `inner_value_grid` and
        appends boundaries according to the boundary conditions, i.e. transforms
        `u_inner` to `u`.
      6. n_dims: A Python integer, the spatial dimension of the PDE.
      7. has_default_lower_boundary: A Python list of booleans of length
        `n_dims`. List indices enumerate the dimensions with `True` values
        marking default lower boundary condition along corresponding dimensions,
        and  `False` values indicating Robin boundary conditions.
      8. has_default_upper_boundary: Similar to has_default_lower_boundary, but
        for upper boundaries.
    The callable returns a `Tensor` of the same shape and `dtype` a
    `values_grid` and represents an approximate solution `u(t2)`.
  """

  if theta < 0 or theta > 1:
    raise ValueError('Theta should be in the interval [0, 1].')

  def _marching_scheme(
      value_grid, t1, t2, equation_params_fn, append_boundaries_fn, n_dims,
      has_default_lower_boundary, has_default_upper_boundary):
    """Constructs the Douglas ADI time marching scheme."""
    current_grid = value_grid
    matrix_params_t1, inhomog_terms_t1 = equation_params_fn(t1)
    matrix_params_t2, inhomog_terms_t2 = equation_params_fn(t2)

    # Explicit substep: Y_0 = (1 + A(t1) dt) U_{n-1} + b(t1) dt,
    # where dt = t2 - t1
    value_grid_with_boundaries = append_boundaries_fn(value_grid)
    for i in range(n_dims - 1):
      for j in range(i + 1, n_dims):
        mixed_term = matrix_params_t1[i][j]
        if mixed_term is not None:
          current_grid += _apply_mixed_term_explicitly(
              value_grid_with_boundaries, mixed_term, t2 - t1, i, j,
              has_default_lower_boundary, has_default_upper_boundary, n_dims)

    # These are A_i(t1) * U_{n-1} * dt; caching them because they appear again
    # later in the correction substeps.
    explicit_contributions = []

    for i in range(n_dims):
      superdiag, diag, subdiag = (matrix_params_t1[i][i][d] for d in range(3))
      contribution = _apply_tridiag_matrix_explicitly(
          value_grid, superdiag, diag, subdiag, i, n_dims) * (t2 - t1)
      explicit_contributions.append(contribution)
      current_grid += contribution

    for inhomog_term in inhomog_terms_t1:
      current_grid += inhomog_term * (t2 - t1)

    # Correction substeps. For each dimension i:
    # Y_i = (1 - theta * A_i(t2) * dt)^(-1) *
    #      (Y_{i-1} - theta * dt * A_i(t1) * U_{n-1} + dt * (b_i(t2) - b_i(t1)))
    if theta == 0:
      return current_grid

    for i in range(n_dims):
      inhomog_term_delta = (inhomog_terms_t2[i] - inhomog_terms_t1[i])
      superdiag, diag, subdiag = (matrix_params_t2[i][i][d] for d in range(3))
      current_grid = _apply_correction(theta, current_grid,
                                       explicit_contributions[i],
                                       superdiag, diag, subdiag,
                                       inhomog_term_delta, t1, t2, i, n_dims)

    return current_grid
  return _marching_scheme


def _apply_mixed_term_explicitly(
    values_with_boundaries, mixed_term, delta_t, dim1, dim2,
    has_default_lower_boundary, has_default_upper_boundary, n_dims):
  """Applies mixed term explicitly."""
  (
      mixed_term_pp, mixed_term_pm, mixed_term_mp, mixed_term_mm
  ) = mixed_term

  batch_rank = values_with_boundaries.shape.rank - n_dims

  # Below we multiply the mixed terms by inner value grid "shifted" diagonally.
  # With Robin boundary conditions, this shift is done by restoring the
  # boundaries (values_with_boundaries already have them restored) and then
  # slicing. E.g. in 2D, values_inner = values_with_boundaries[1:-1, 1:-1],
  # and an example of a diagonally-shifted slice is
  # values_with_boundaries[:-2, 2:], which gets multiplied by mixed_term_mp.
  # However, with default boundaries, there's no boundary to restore, and the
  # diagonal shift go out of bounds. Since the mixed terms on the default
  # boundaries are zero, the "out-of-bounds" values are irrelevant. However, to
  # avoid going out of bounds and get the shapes right, we need to pad these
  # boundaries.
  paddings = batch_rank * [[0, 0]]
  for dim in range(n_dims):
    lower = 1 if has_default_lower_boundary[dim] else 0
    upper = 1 if has_default_upper_boundary[dim] else 0
    paddings += [[lower, upper]]

  # Pad defualt boundaries with zeros
  values_with_boundaries = tf.pad(values_with_boundaries, paddings=paddings)

  def create_trimming_shifts(dim1_shift, dim2_shift):
    # See _trim_boundaries. We need to apply shifts to dimensions dim1 and dim2.
    shifts = [0] * n_dims
    shifts[dim1] = dim1_shift
    shifts[dim2] = dim2_shift
    return shifts

  values_pp = _trim_boundaries(values_with_boundaries, from_dim=batch_rank,
                               shifts=create_trimming_shifts(1, 1))
  values_mm = _trim_boundaries(values_with_boundaries, from_dim=batch_rank,
                               shifts=create_trimming_shifts(-1, -1))
  values_pm = _trim_boundaries(values_with_boundaries, from_dim=batch_rank,
                               shifts=create_trimming_shifts(1, -1))
  values_mp = _trim_boundaries(values_with_boundaries, from_dim=batch_rank,
                               shifts=create_trimming_shifts(-1, 1))

  return (mixed_term_mm * values_mm +
          mixed_term_mp * values_mp +
          mixed_term_pm * values_pm +
          mixed_term_pp * values_pp) * delta_t


def _apply_tridiag_matrix_explicitly(values, superdiag, diag, subdiag,
                                     dim, n_dims):
  """Applies tridiagonal matrix explicitly."""
  perm = _get_permutation(values, n_dims, dim)

  # Make the given dimension the last one in the tensors, treat all the
  # other spatial dimensions as batch dimensions.
  if perm is not None:
    values = tf.transpose(values, perm)
    superdiag, diag, subdiag = (
        tf.transpose(c, perm) for c in (superdiag, diag, subdiag))

  values = tf.squeeze(
      tf.linalg.tridiagonal_matmul((superdiag, diag, subdiag),
                                   tf.expand_dims(values, -1),
                                   diagonals_format='sequence'), -1)

  # Transpose back to how it was.
  if perm is not None:
    values = tf.transpose(values, perm)
  return values


def _apply_correction(theta, values, explicit_contribution, superdiag, diag,
                      subdiag, inhomog_term_delta, t1, t2, dim, n_dims):
  """Applies correction for the given dimension."""
  rhs = (
      values - theta * explicit_contribution +
      theta * inhomog_term_delta * (t2 - t1))

  # Make the given dimension the last one in the tensors, treat all the
  # other spatial dimensions as batch dimensions.
  perm = _get_permutation(values, n_dims, dim)
  if perm is not None:
    rhs = tf.transpose(rhs, perm)
    superdiag, diag, subdiag = (
        tf.transpose(c, perm) for c in (superdiag, diag, subdiag))

  subdiag = -theta * subdiag * (t2 - t1)
  diag = 1 - theta * diag * (t2 - t1)
  superdiag = -theta * superdiag * (t2 - t1)
  result = tf.linalg.tridiagonal_solve((superdiag, diag, subdiag),
                                       rhs,
                                       diagonals_format='sequence',
                                       partial_pivoting=False)

  # Transpose back to how it was.
  if perm is not None:
    result = tf.transpose(result, perm)
  return result


def _get_permutation(tensor, n_dims, active_dim):
  """Returns the permutation that swaps the active and the last dimensions.

  Args:
    tensor: `Tensor` having a statically known rank.
    n_dims: Number of spatial dimensions.
    active_dim: The active spatial dimension.

  Returns:
    A list representing the permutation, or `None` if no permutation needed.

  For example, with 'tensor` having rank 5, `n_dims = 3` and `active_dim = 1`
  yields [0, 1, 2, 4, 3]. Explanation: we start with [0, 1, 2, 3, 4], where the
  last n_dims=3 dimensions are spatial dimensions, and the first two are batch
  dimensions. Among the spatial dimensions, we take the one at index 1, which
  is "3", and swap it with the last dimension "4".
  """
  if not tensor.shape:
    raise ValueError("Tensor's rank should be static")
  rank = len(tensor.shape)
  batch_rank = rank - n_dims
  if active_dim == n_dims - 1:
    return None
  perm = np.arange(rank)
  perm[rank - 1] = batch_rank + active_dim
  perm[batch_rank + active_dim] = rank - 1
  return perm


def _trim_boundaries(tensor, from_dim, shifts=None):
  """Trims tensor boundaries starting from given dimension."""
  # For example, if tensor has shape (a, b, c, d) and from_dim=1, then the
  # output tensor has shape (a, b-2, c-2, d-2).
  # For example _trim_boundaries(t, 1) with a rank-4
  # tensor t yields t[:, 1:-1, 1:-1, 1:-1].
  #
  # If shifts is specified, the slices applied are shifted. shifts is an array
  # of length rank(tensor) - from_dim, with values -1, 0, or 1, meaning slices
  # [:-2], [-1, 1], and [2:], respectively.
  # For example _trim_boundaries(t, 1, (1, 0, -1)) with a rank-4
  #  tensor t yields t[:, 2:, 1:-1, :-2].
  rank = tensor.shape.rank
  slices = rank * [slice(None)]
  for i in range(from_dim, rank):
    slice_begin = 1
    slice_end = -1
    # Apply shifts
    if shifts is not None:
      shift = shifts[i - from_dim]
      slice_begin += shift
      slice_end += shift
    # If slice_end is `0` integer, do not trim that upper dimension
    if isinstance(slice_end, int) and slice_end == 0:
      slice_end = None
    slices[i] = slice(slice_begin, slice_end)
  res = tensor[slices]
  return res


_all__ = ['douglas_adi_step', 'douglas_adi_scheme']
