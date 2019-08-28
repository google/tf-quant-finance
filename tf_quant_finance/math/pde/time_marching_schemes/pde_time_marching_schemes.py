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
"""Time marching schemes for finite difference methods for parabolic PDEs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_quant_finance.math.pde.internal import pde_time_marching_schemes_internal as internal_schemes
from tf_quant_finance.math.pde.time_marching_schemes.pde_time_marching_scheme import TimeMarchingScheme


class WeightedImplicitExplicitScheme(TimeMarchingScheme):
  """Weighted implicit-explicit scheme.

  Approximates the exponent in the solution `du/dt = A(t) u(t)` as
  `u(t2) = (1 - (1 - theta) dt A)^(-1) * (1 + theta dt A) u(t1)`.
   Here `dt = t2 - t1`, `A = A((t1 + t2)/2)` and `theta` is a float between `0`
   and `1`.
  Includes as particular cases the fully explicit scheme (`theta = 1`), the
  fully implicit scheme (`theta = 0`) and the Crank-Nicolson scheme
  (`theta = 0.5`).
  The scheme is first-order accurate in `t2 - t1` if `theta != 0.5` and
  second-order accurate if `theta = 0.5` (the Crank-Nicolson scheme).
  Note that evaluating `A(t)` at midpoint `t = (t1 + t2)/2` is important for
  maintaining second-order accuracy of the Crank-Nicolson scheme: see e.g. [1],
  the paragraph after Eq. (14).

  ### References:
  [1] I.V. Puzynin, A.V. Selin, S.I. Vinitsky, A high-order accuracy method for
  numerical solving of the time-dependent Schrodinger equation, Comput. Phys.
  Commun. 123 (1999), 1.
  https://www.sciencedirect.com/science/article/pii/S0010465599002246
  """

  def __init__(self, theta):
    """Initializes the finite-difference scheme.

    Args:
      theta: A float in range `[0, 1]`. A parameter used to mix implicit and
        explicit schemes together. Value of `0.0` corresponds to the fully
        implicit scheme, `1.0` to the fully explicit, and `0.5` to the
        Crank-Nicolson scheme. See, e.g., [1].  #### References
    [1]: P.A. Forsyth, K.R. Vetzal. Quadratic Convergence for Valuing American
      Options Using A Penalty Method. Journal on Scientific Computing, 2002.
      http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.9066&rep=rep1&type=pdf
    """
    if theta < 0 or theta > 1:
      raise ValueError(
          '`theta` should be in [0, 1]. Supplied: {}'.format(theta))
    self.theta = theta

  def apply(self,
            value_grid,
            t1,
            t2,
            num_steps_performed,
            matrix_constructor,
            lower_boundary_fn,
            upper_boundary_fn,
            backwards=False):
    diag, superdiag, subdiag = matrix_constructor((t1 + t2) / 2)

    boundary_vals = (tf.convert_to_tensor(lower_boundary_fn(t1)),
                     tf.convert_to_tensor(upper_boundary_fn(t1)))

    if self.theta == 0:  # fully implicit scheme
      rhs = value_grid[..., 1:-1]
    else:
      rhs = _weighted_scheme_explicit_part(value_grid, diag, superdiag, subdiag,
                                           self.theta, t1, t2, backwards)

    # Correction for the boundary term
    zeros = tf.zeros_like(rhs[..., 1:-1])
    lower_correction = tf.expand_dims(
        (1 - self.theta) * subdiag[..., 0] * boundary_vals[0], -1)
    upper_correction = tf.expand_dims(
        (1 - self.theta) * superdiag[..., -1] * boundary_vals[-1], -1)

    rhs -= (
        tf.concat([lower_correction, zeros, upper_correction], -1) * (t2 - t1))
    if self.theta < 1:
      # Note that if theta is `0`, `rhs` equals to the `value_grid`, so that the
      # fully implicit step is performed.
      next_step = _weighted_scheme_implicit_part(rhs, diag, superdiag, subdiag,
                                                 self.theta, t1, t2, backwards)
    else:  # fully explicit scheme
      next_step = rhs
    return tf.concat([
        tf.expand_dims(boundary_vals[0], -1), next_step,
        tf.expand_dims(boundary_vals[1], -1)
    ], -1)


def implicit_scheme():
  """Fully implicit scheme."""
  return WeightedImplicitExplicitScheme(theta=0)


def explicit_scheme():
  """Fully explicit scheme."""
  return WeightedImplicitExplicitScheme(theta=1)


def crank_nicolson_scheme():
  """Crank-Nicolson scheme."""
  return WeightedImplicitExplicitScheme(theta=0.5)


class ExtrapolationMarchingScheme(TimeMarchingScheme):
  """Extrapolation scheme.

  Performs two implicit half-steps, one full implicit step, and combines them
  with such coefficients that ensure second-order errors. More computationally
  expensive than Crank-Nicolson scheme, but provides a better approximation for
  high-wavenumber components, which results in absence of oscillations typical
  for Crank-Nicolson scheme in case of non-smooth initial conditions. See [1]
  for details.

  ### References:
  [1]: D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods
  for Parabolic Partial Differential Equations. I. 1978
  SIAM Journal on Numerical Analysis. 15. 1212-1224.
  https://epubs.siam.org/doi/abs/10.1137/0715082
  """

  def __init__(self):
    self.implicit_scheme = implicit_scheme()

  def apply(self,
            value_grid,
            t1,
            t2,
            num_steps_performed,
            matrix_constructor,
            lower_boundary_fn,
            upper_boundary_fn,
            backwards=False):
    if backwards:
      first_half_times = (t1 + t2) / 2, t2
      second_half_times = t1, (t1 + t2) / 2
    else:
      first_half_times = t1, (t1 + t2) / 2
      second_half_times = (t1 + t2) / 2, t2

    first_half_step = self.implicit_scheme.apply(
        value_grid, first_half_times[0], first_half_times[1],
        num_steps_performed, matrix_constructor, lower_boundary_fn,
        upper_boundary_fn, backwards)
    two_half_steps = self.implicit_scheme.apply(
        first_half_step, second_half_times[0], second_half_times[1],
        num_steps_performed, matrix_constructor, lower_boundary_fn,
        upper_boundary_fn, backwards)

    full_step = self.implicit_scheme.apply(value_grid, t1, t2,
                                           num_steps_performed,
                                           matrix_constructor,
                                           lower_boundary_fn, upper_boundary_fn,
                                           backwards)
    return 2 * two_half_steps - full_step


def crank_nicolson_with_oscillation_damping(extrapolation_steps=1):
  """Scheme similar to Crank-Nicolson, but ensuring damping of oscillations.

  Performs first (or first few) steps with Extrapolation scheme, then proceeds
  with Crank-Nicolson scheme. This combines absence of oscillations by virtue
  of Extrapolation scheme with lower computational cost of Crank-Nicolson
  scheme.

  See [1], [2] ([2] mostly discusses using fully implicit scheme on the first
  step, but mentions using extrapolation scheme for better accuracy in the end).

  Args:
    extrapolation_steps: number of first steps to which to apply the
      Extrapolation scheme. Defaults to `1`.

  Returns:
    `TimeMarchingScheme` with described properties.
  ### References:
  [1]: D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods for
    Parabolic Partial Differential Equations. I. 1978 SIAM Journal on Numerical
    Analysis. 15. 1212-1224.
    https://epubs.siam.org/doi/abs/10.1137/0715082
  [2]: B. Giles, Michael & Carter, Rebecca. Convergence analysis of
    Crank-Nicolson and Rannacher time-marching. J. Comput. Finance. 9. 2005.
    https://core.ac.uk/download/pdf/1633712.pdf
  """

  return internal_schemes.CompositeTimeMarchingScheme(
      extrapolation_steps, ExtrapolationMarchingScheme(),
      crank_nicolson_scheme())


def _weighted_scheme_explicit_part(vec, diag, upper, lower, theta, t1, t2,
                                   backwards):
  """Explicit step of the weighted implicit-explicit scheme.

  Args:
    vec: A real dtype `Tensor` of shape `[num_equations, num_grid_points]`.
      Represents the multiplied vector.
    diag: A real dtype `Tensor` of the shape `[num_equations, num_grid_points -
      2]`. Represents the main diagonal of a 3-diagonal matrix for the PDE
      scheme with Dirichlet boundary conditions.
    upper: A real dtype `Tensor` of the shape `[num_equations, num_grid_points -
      2]`. Represents the upper diagonal of a 3-diagonal matrix for the PDE
      scheme with Dirichlet boundary conditions.
    lower:  A real dtype `Tensor` of the shape `[num_equations, num_grid_points
      - 2]`. Represents the lower diagonal of a 3-diagonal matrix for the PDE
      scheme with Dirichlet boundary conditions.
    theta: A Python float between 0 and 1.
    t1: Smaller of the two times defining the step.
    t2: Greater of the two times defining the step.
    backwards: whether we're making a step backwards in time.

  Returns:
    A tensor of the same shape and dtype as `vec`.
  """
  multiplier = theta * (-1 if backwards else 1) * (t2 - t1)
  diag = 1 + multiplier * diag
  upper = multiplier * upper
  lower = multiplier * lower
  return lower * vec[..., :-2] + diag * vec[..., 1:-1] + upper * vec[..., 2:]


def _weighted_scheme_implicit_part(vec, diag, upper, lower, theta, t1, t2,
                                   backwards):
  """Implicit step of the weighted implicit-explicit scheme.

  Args:
    vec: A real dtype `Tensor` of shape `[num_equations, num_grid_points]`.
      Represents the multiplied vector.
    diag: A real dtype `Tensor` of the shape `[num_equations, num_grid_points -
      2]`. Represents the main diagonal of a 3-diagonal matrix for the PDE
      scheme with Dirichlet boundary conditions.
    upper: A real dtype `Tensor` of the shape `[num_equations, num_grid_points -
      2]`. Represents the upper diagonal of a 3-diagonal matrix for the PDE
      scheme with Dirichlet boundary conditions.
    lower:  A real dtype `Tensor` of the shape `[num_equations, num_grid_points
      - 2]`. Represents the lower diagonal of a 3-diagonal matrix for the PDE
      scheme with Dirichlet boundary conditions.
    theta: A Python float between 0 and 1.
    t1: Smaller of the two times defining the step.
    t2: Greater of the two times defining the step.
    backwards: whether we're making a step backwards in time.

  Returns:
    A tensor of the same shape and dtype as `vec`.
  """
  multiplier = (1 - theta) * (1 if backwards else -1) * (t2 - t1)
  diag = 1 + multiplier * diag
  upper = multiplier * upper
  lower = multiplier * lower
  return tf.linalg.tridiagonal_solve([upper, diag, lower],
                                     vec,
                                     diagonals_format='sequence',
                                     transpose_rhs=True,
                                     partial_pivoting=False)
