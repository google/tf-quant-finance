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
"""Partial differential equation kernels to be consumed by grid_stepper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_quant_finance.math.pde import time_marching_schemes


class ParabolicDifferentialEquationSolver(object):
  """Solver for parabolic differential equations.

  Creates a solver (see, e.g.,  [Forsyth, Vetzal(2002)][3]) for a
  parabolic differential equation of the form

  ```none
   V_{t} + 1 / 2 * a(t, x) * V_{xx} + b(t, x) * V_{x} - c(t, x) * V = 0
  ```

  Here `V = V(t, x)` is a solution to the 2-dimensional PDE. `V_{t}` is the
  derivative over time and `V_{x}` and `V_{xx}` are the first and second
  derivatives over the space component. For a solution to be well-defined, it is
  required for `a` to be positive on its domain. Henceforth,
  `a(t, x)`, `b(t, x)`, and `c(t, x)` are referred to as quadratic, linear and
  shift coefficients, respectively. Note that the backward Kolmogorov equation
  is of the above form with a(t, x) as the drift, b(t, x) the diffusion
  constant, and c = 0. See, e.g., [1] or [Oskendal(2010)][2].

  #### References
  [1]: Kolmogorov_backward_equations. Wikipedia/
    https://en.wikipedia.org/wiki/Kolmogorov_backward_equations_(diffusion)
  [2]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
    Applications. Springer. 2010.
  [3]: P.A. Forsyth, K.R. Vetzal. Quadratic Convergence for Valuing American
    Options Using A Penalty Method. Journal on Scientific Computing, 2002.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.9066&rep=rep1&type=pdf
  """

  def __init__(self,
               quadratic_coeff_fn,
               lower_boundary_fn,
               upper_boundary_fn,
               linear_coeff_fn=None,
               shift_coeff_fn=None,
               time_marching_scheme=None,
               name=None):
    """Initializes the solver.

    Args:
      quadratic_coeff_fn: A Python callable returning the quadratic coefficient
        `a(t, x)` at the given time `t`. The callable accepts the time `t` and a
        grid of values `x` at which the coefficient is to be computed. Should
        return a `Tensor` of float dtype and of shape broadcastable with
        `[num_equations, num_grid_points]`, where `num_equations` is the number
        of PDE equations being solved and `num_grid_points` is the number of
        grid points over the space coordinate `x`.
      lower_boundary_fn: A Python callable returning a `Tensor` of the same
        `dtype` as the output of `quadratic_coeff_fn` and of shape
        `[num_equations]`. Represents Dirichlet boundary condition for the lower
        value of `x`, i.e., `V(t, x_min)`.
      upper_boundary_fn: A Python callable returning a `Tensor` of the same
        `dtype` as the output of `quadratic_coeff_fn` and of shape
        `[num_equations]`. Represents Dirichlet boundary condition for the upper
        value of `x`, i.e., `V(t, x_max)`.
      linear_coeff_fn:  Python callable returning the linear coefficient `b(t,
        x)` at the given time `t`. The callable accepts the time `t` and a grid
        of values `x` at which the coefficient is to be computed. Should return
        a `Tensor` of the same dtype as the output of `quadratic_coeff_fn` and
        of shape broadcastable with `[num_equations, num_grid_points]`. Defaults
        to `linear_coeff_fn(t, x) = 0.0`.
      shift_coeff_fn:  Python callable returning the shift coefficient `c(t, x)`
        at the given time `t`. The callable accepts the time `t` and a grid of
        values `x` at which the coefficient is to be computed. Should return a
        `Tensor` of the same dtype as the output of `quadratic_coeff_fn` and of
        shape broadcastable with `[num_equations, num_grid_points]`. Defaults to
        `shift_coeff_fn(t, x) = 0.0`.
      time_marching_scheme: an implementation of TimeMarchingScheme (see
        pde_time_marching_schemes.py) defining a time marching scheme used:
          implicit, explicit, Crank-Nicolson, etc. By default, Crank-Nicolson
          scheme is used for all steps except the first. The first step is done
          using the extrapolation scheme (see [2]) to damp the oscillations
          typical for Crank-Nicolson solution with non-smooth initial
          conditions.
      name: Python str. The name prefixed to the ops created by this class. If
        not supplied, the default name 'CrankNicolson' is used.
    ### References:
    [1]: P.A. Forsyth, K.R. Vetzal. Quadratic Convergence for Valuing American
      Options Using A Penalty Method. Journal on Scientific Computing, 2002.
      http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.9066&rep=rep1&type=pdf
    [2]: D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods
      for Parabolic Partial Differential Equations. I. 1978 SIAM Journal on
      Numerical Analysis. 15. 1212-1224.
      https://epubs.siam.org/doi/abs/10.1137/0715082
    """
    self._name = name
    self._quadratic_coeff_fn = quadratic_coeff_fn

    self._lower_boundary_fn = lower_boundary_fn
    self._upper_boundary_fn = upper_boundary_fn

    self._linear_coeff_fn = (linear_coeff_fn or (lambda *args: 0.0))
    self._shift_coeff_fn = (shift_coeff_fn or (lambda *args: 0.0))
    self.time_marching_scheme = (
        time_marching_scheme or
        time_marching_schemes.crank_nicolson_with_oscillation_damping())

  @property
  def quadratic_coeff_fn(self):
    return self._quadratic_coeff_fn

  @property
  def lower_boundary_fn(self):
    return self._lower_boundary_fn

  @property
  def upper_boundary_fn(self):
    return self._upper_boundary_fn

  @property
  def linear_coeff_fn(self):
    return self._linear_coeff_fn

  @property
  def shift_coeff_fn(self):
    return self._shift_coeff_fn

  @property
  def name(self):
    return self._name

  def one_step(self, state):
    """Performs one step backwards in time.

    Given a space discretization grid `{x_1,.., x_N}`, time step `dt` the method
    updates values of the current estimates for the PDE solution
    {V^k(x_1),.., V^k(x_N)} using the explicit constraint timestep method
    (see (5.3) in [Forsyth, Vetzal][1]). Note, the method can propagate a batch
    of PDE's at the same time.

    ### Example. European call option pricing.
    ```python
    num_equations = 2  # Number of PDE
    num_grid_points = 1024  # Number of grid points
    dtype = np.float64
    # Build a log-uniform grid
    s_min = 0.01
    s_max = 300.
    grid = grids.uniform_grid(minimums=[s_min], maximums=[s_max],
                              sizes=[num_grid_points],
                              dtype=dtype)
    # Specify volatilities and interest rates for the options
    volatility = tf.constant([[0.3], [0.15]], dtype)
    rate = tf.constant([[0.01], [0.03]], dtype)
    expiry = 1.0
    strike = tf.constant([[50], [100]], dtype)
    # Define parabolic equation coefficients. In this case the coefficients
    # can be computed exactly but the same functions as below can be used to
    # get approximate values for general case.
    def quadratic_coeff_fn(t, x):
      del t
      return tf.square(volatility) * tf.square(x)

    def linear_coeff_fn(t, x):
      del t
      return rate * x

    def shift_coeff_fn(t, x):
      del t, x
      return rate

    def lower_boundary_fn(t):
      del t
      return dtype([0.0, 0.0])

    def upper_boundary_fn(t):
      return tf.squeeze(s_max - strike * tf.exp(-rate * (expiry - t)))

    solver_kernel = pde_kernels.ParabolicDifferentialEquationSolver(
        quadratic_coeff_fn,
        lower_boundary_fn,
        upper_boundary_fn,
        linear_coeff_fn, shift_coeff_fn)

    def time_step_fn(state):
      del state
      return tf.constant(0.01, dtype=dtype)
    def payoff_fn(state):
      option_bound = tf.nn.relu(state.coordinate_grid.locations[0] - strike)
      # Broadcast to the shape of value dimension, if necessary.
      option_bound = option_bound + tf.zeros_like(state.value_grid)
      return option_bound

    bgs = grid_stepper.BackwardGridStepper(
        expiry, solver_kernel.one_step, grid,
        time_step_fn=time_step_fn,
        value_dim=num_equations,
        dtype=dtype)
    bgs.transform_values(payoff_fn)
    bgs.step_back_to_time(0.0)
    estimate = bgs.state()
    grid_locations = estimate.coordinate_grid.locations[0]
    # Estimated prices of the option at the grid locations
    value_grid_first_option = estimate.value_grid[0, :]
    value_grid_second_option = estimate.value_grid[1, :]

    ```

    ### References:
    [1]: P.A. Forsyth, K.R. Vetzal. Quadratic Convergence for Valuing American
      Options Using A Penalty Method. Journal on Scientific Computing, 2002.
      http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.9066&rep=rep1&type=pdf

    Args:
      state: An instance of `GridStepperState` or a namedtuple containing the
        following attributes. `time`.  A scalar `Tensor` of float dtype. The
        current time. `time_step`. A positive scalar `Tensor` of float dtype.
        The next time step to take. `value_grid`. The function values on the
        grid. A tensor of shape `[num_equations, num_grid_points]`, where
        `num_grid_points` is the number of points on the grid and
        `num_equations` is the number of PDE equations to be propagated.
        `coordinate_grid`. Grid discretization along the space coordinate. An
        instance of `GridSpec` or a namedtuple containing the following
        attributes. `deltas`. The difference between consecutive elements of the
        grid. Represented as an array of `Tensor`s. Each `Tensor` is of a shape
        that will broadcast with a `Tensor` of shape broadcastable with
        `[num_equations, num_grid_points]`, where `num_grid_points` is the
        number of points on the grid and `num_equations` is the number of PDE
        equations to be propagated. `locations`. The full grid of coordinates.
        The grid is a single real `Tensor` of shape broadcastable with
        `[num_equations, num_grid_points]`, where `num_grid_points` is the
        number of points on the grid and `num_equations` is the number of PDE
        equations to be propagated.

    Returns:
      A `Tensor` of the same shape and dtype as value_grid. Corresponds to the
      grid values after one iteration of the solver.
    """
    with tf.compat.v1.name_scope(
        self.name, 'ParabolicDifferentialEquationSolver_one_step', [state]):
      matrix_constructor = (
          lambda t: self._construct_tridiagonal_matrix(state, t))

      return self.time_marching_scheme.apply(
          state.value_grid,
          state.time - state.time_step,
          state.time,
          state.num_steps_performed,
          matrix_constructor,
          self.lower_boundary_fn,
          self.upper_boundary_fn,
          backwards=True)

  def _construct_tridiagonal_matrix(self, state, t):
    """Constructs tridiagonal matrix to pass to the time marching scheme."""
    state_grid = state.coordinate_grid.locations[0][..., 1:-1]

    # Extract broadcasted grid deltas.
    diff = tf.broadcast_to(
        state.coordinate_grid.deltas[0],
        tf.shape(state.coordinate_grid.locations[0][..., 1:]))[..., 1:-1]

    # Get forward, backward and total differences.

    # Create forward paddings [[0, 0], [0, 0], .. [0, 1]]
    # Forward differences
    forward_paddings = tf.concat([
        tf.zeros([tf.rank(diff) - 1, 2], dtype=tf.int32),
        tf.constant([[0, 1]], dtype=tf.int32)
    ], 0)
    forward_diff = tf.pad(
        diff, forward_paddings, constant_values=tf.reduce_min(diff[..., -1]))
    # Create backward paddings [[0, 0], [0, 0], .. [1, 0]]
    backward_paddings = tf.concat([
        tf.zeros([tf.rank(diff) - 1, 2], dtype=tf.int32),
        tf.constant([[1, 0]], dtype=tf.int32)
    ], 0)
    # Backward differences
    backward_diff = tf.pad(
        diff, backward_paddings, constant_values=tf.reduce_min(diff[..., 0]))
    # Note that the total difference = 2 * central difference.
    total_diff = forward_diff + backward_diff

    # 3-diagonal matrix construction. See matrix `M` in [Forsyth, Vetzal][1].
    #  The `tridiagonal` matrix is of shape
    # `[value_dim, 3, num_grid_points]`.

    # Get the PDE coefficients and broadcast them to the shape of value grid.
    broadcast_shape = tf.shape(state.value_grid[..., 1:-1])

    quadratic_coeff = tf.convert_to_tensor(
        self._quadratic_coeff_fn(t, state_grid), dtype=state_grid.dtype)
    quadratic_coeff = tf.broadcast_to(quadratic_coeff, broadcast_shape)
    linear_coeff = tf.convert_to_tensor(
        self._linear_coeff_fn(t, state_grid), dtype=state_grid.dtype)
    linear_coeff = tf.broadcast_to(linear_coeff, broadcast_shape)
    shift_coeff = tf.convert_to_tensor(
        self._shift_coeff_fn(t, state_grid), dtype=state_grid.dtype)
    shift_coeff = tf.broadcast_to(shift_coeff, broadcast_shape)

    # The 3-diagonal matrix involves coefficients `gamma` and `beta` which
    # are referred to as `dxdx_coef` and `dx_coef`, respectively. This is done
    # to reflect that `dxdx_coef` is coming from the 2-nd order discretization
    # of `V_{xx}` and `dx_coef` is from 1-st order discretization of `V_{x}`,
    # where `V` is a solution to the PDE.

    temp = quadratic_coeff / total_diff
    dxdx_coef_1 = temp / forward_diff
    dxdx_coef_2 = temp / backward_diff

    dx_coef = linear_coeff / total_diff

    # The 3 main diagonals are constructed below. Note that all the diagonals
    # are of the same length
    upper_diagonal = (-dx_coef - dxdx_coef_1)

    lower_diagonal = (dx_coef - dxdx_coef_2)

    diagonal = shift_coeff - upper_diagonal - lower_diagonal

    return diagonal, upper_diagonal, lower_diagonal
