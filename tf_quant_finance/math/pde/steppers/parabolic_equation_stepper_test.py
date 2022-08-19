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
"""Tests for 1-D parabolic PDE solvers."""

import math

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

fd_solvers = tff.math.pde.fd_solvers
dirichlet = tff.math.pde.boundary_conditions.dirichlet
neumann = tff.math.pde.boundary_conditions.neumann
grids = tff.math.pde.grids
crank_nicolson_step = tff.math.pde.steppers.crank_nicolson.crank_nicolson_step
explicit_step = tff.math.pde.steppers.explicit.explicit_step
extrapolation_step = tff.math.pde.steppers.extrapolation.extrapolation_step
implicit_step = tff.math.pde.steppers.implicit.implicit_step
crank_nicolson_with_oscillation_damping_step = tff.math.pde.steppers.oscillation_damped_crank_nicolson.oscillation_damped_crank_nicolson_step
weighted_implicit_explicit_step = tff.math.pde.steppers.weighted_implicit_explicit.weighted_implicit_explicit_step


@test_util.run_all_in_graph_and_eager_modes
class ParabolicEquationStepperTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'Implicit',
          'one_step_fn': implicit_step(),
          'time_step': 0.001
      }, {
          'testcase_name': 'Explicit',
          'one_step_fn': explicit_step(),
          'time_step': 0.001
      }, {
          'testcase_name': 'Weighted',
          'one_step_fn': weighted_implicit_explicit_step(theta=0.3),
          'time_step': 0.001
      }, {
          'testcase_name': 'CrankNicolson',
          'one_step_fn': crank_nicolson_step(),
          'time_step': 0.01
      }, {
          'testcase_name': 'Extrapolation',
          'one_step_fn': extrapolation_step(),
          'time_step': 0.01
      }, {
          'testcase_name': 'CrankNicolsonWithOscillationDamping',
          'one_step_fn': crank_nicolson_with_oscillation_damping_step(),
          'time_step': 0.01
      })
  def testHeatEquationWithVariousSchemes(self, one_step_fn, time_step):
    """Test solving heat equation with various time marching schemes.

    Tests solving heat equation with the boundary conditions
    `u(x, t=1) = e * sin(x)`, `u(-2 pi n - pi / 2, t) = -e^t`, and
    `u(2 pi n + pi / 2, t) = -e^t` with some integer `n` for `u(x, t=0)`.

    The exact solution is `u(x, t=0) = sin(x)`.

    All time marching schemes should yield reasonable results given small enough
    time steps. First-order accurate schemes (explicit, implicit, weighted with
    theta != 0.5) require smaller time step than second-order accurate ones
    (Crank-Nicolson, Extrapolation).

    Args:
      one_step_fn: one_step_fn representing a time marching scheme to use.
      time_step: time step for given scheme.
    """
    def final_cond_fn(x):
      return math.e * math.sin(x)

    def expected_result_fn(x):
      return tf.sin(x)

    @dirichlet
    def lower_boundary_fn(t, x):
      del x
      return -tf.math.exp(t)

    @dirichlet
    def upper_boundary_fn(t, x):
      del x
      return tf.math.exp(t)

    grid = grids.uniform_grid(
        minimums=[-10.5 * math.pi],
        maximums=[10.5 * math.pi],
        sizes=[1000],
        dtype=np.float32)
    self._testHeatEquation(
        grid=grid,
        final_t=1,
        time_step=time_step,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=one_step_fn,
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-3)

  def testHeatEquation_WithNeumannBoundaryConditions(self):
    """Test for Neumann boundary conditions.

    Tests solving heat equation with the following boundary conditions:
    `u(x, t=1) = e * sin(x)`, `u_x(0, t) = e^t`, and
    `u_x(2 pi n + pi/2, t) = 0`, where `n` is some integer.

    The exact solution `u(x, t=0) = e^t sin(x)`.
    """

    def final_cond_fn(x):
      return math.e * math.sin(x)

    def expected_result_fn(x):
      return tf.sin(x)

    @neumann
    def lower_boundary_fn(t, x):
      del x
      return -tf.math.exp(t)

    @neumann
    def upper_boundary_fn(t, x):
      del t, x
      return 0

    grid = grids.uniform_grid(
        minimums=[0.0], maximums=[10.5 * math.pi], sizes=[1000],
        dtype=np.float32)
    self._testHeatEquation(
        grid,
        final_t=1,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=crank_nicolson_step(),
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-3)

  def testHeatEquation_WithMixedBoundaryConditions(self):
    """Test for mixed boundary conditions.

    Tests solving heat equation with the following boundary conditions:
    `u(x, t=1) = e * sin(x)`, `u_x(0, t) = e^t`, and
    `u(2 pi n + pi/2, t) = e^t`, where `n` is some integer.

    The exact solution `u(x, t=0) = e^t sin(x)`.
    """

    def final_cond_fn(x):
      return math.e * math.sin(x)

    def expected_result_fn(x):
      return tf.sin(x)

    @neumann
    def lower_boundary_fn(t, x):
      del x
      return -tf.math.exp(t)

    @dirichlet
    def upper_boundary_fn(t, x):
      del x
      return tf.math.exp(t)

    grid = grids.uniform_grid(minimums=[0], maximums=[10.5 * math.pi],
                              sizes=[1000], dtype=np.float32)
    self._testHeatEquation(
        grid,
        final_t=1,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=crank_nicolson_step(),
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-3)

  def testHeatEquation_WithRobinBoundaryConditions(self):
    """Test for Robin boundary conditions.

    Tests solving heat equation with the following boundary conditions:
    `u(x, t=1) = e * sin(x)`, `u_x(0, t) + 2u(0, t) = e^t`, and
    `2u(x_max, t) + u_x(x_max, t) = 2*e^t`, where `x_max = 2 pi n + pi/2` with
    some integer `n`.

    The exact solution `u(x, t=0) = e^t sin(x)`.
    """

    def final_cond_fn(x):
      return math.e * math.sin(x)

    def expected_result_fn(x):
      return tf.sin(x)

    def lower_boundary_fn(t, x):
      del x
      return 2, -1, tf.math.exp(t)

    def upper_boundary_fn(t, x):
      del x
      return 2, 1, 2 * tf.math.exp(t)

    grid = grids.uniform_grid(minimums=[0], maximums=[4.5 * math.pi],
                              sizes=[1000], dtype=np.float64)
    self._testHeatEquation(
        grid,
        final_t=1,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=crank_nicolson_step(),
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-2)

  def testHeatEquation_WithRobinBoundaryConditions_AndLogUniformGrid(self):
    """Same as previous, but with log-uniform grid."""

    def final_cond_fn(x):
      return math.e * math.sin(x)

    def expected_result_fn(x):
      return tf.sin(x)

    def lower_boundary_fn(t, x):
      del x
      return 2, -1, tf.math.exp(t)

    def upper_boundary_fn(t, x):
      del x
      return 2, 1, 2 * tf.math.exp(t)

    grid = grids.log_uniform_grid(
        minimums=[2 * math.pi],
        maximums=[4.5 * math.pi],
        sizes=[1000],
        dtype=np.float64)
    self._testHeatEquation(
        grid=grid,
        final_t=1,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=crank_nicolson_step(),
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-2)

  def testHeatEquation_WithRobinBoundaryConditions_AndExtraPointInGrid(self):
    """Same as previous, but with grid with an extra point.

    We add an extra point in a uniform grid so that grid[1]-grid[0] and
    grid[2]-grid[1] are significantly different. This is important for testing
    the discretization of boundary conditions: both deltas participate there.
    """

    def final_cond_fn(x):
      return math.e * math.sin(x)

    def expected_result_fn(x):
      return tf.sin(x)

    def lower_boundary_fn(t, x):
      del x
      return 2, -1, tf.math.exp(t)

    def upper_boundary_fn(t, x):
      del x
      return 2, 1, 2 * tf.math.exp(t)

    x_min = 0
    x_max = 4.5 * math.pi
    num_points = 1001
    locations = np.linspace(x_min, x_max, num=num_points - 1)
    delta = locations[1] - locations[0]
    locations = np.insert(locations, 1, locations[0] + delta / 3)

    grid = [tf.constant(locations)]

    self._testHeatEquation(
        grid=grid,
        final_t=1,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=crank_nicolson_step(),
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-2)

  def testCrankNicolsonOscillationDamping(self):
    """Tests the Crank-Nicolson oscillation damping.

    Oscillations arise in Crank-Nicolson scheme when the initial (or final)
    conditions have discontinuities. We use Heaviside step function as initial
    conditions. The exact solution of the heat equation with unbounded x is
    ```None
    u(x, t) = (1 + erf(x/2sqrt(t))/2
    ```
    We take large enough x_min, x_max to be able to use this as a reference
    solution.

    CrankNicolsonWithOscillationDamping produces much smaller error than
    the usual crank_nicolson_scheme.
    """

    final_t = 1
    x_min = -10
    x_max = 10
    dtype = np.float32

    def final_cond_fn(x):
      return 0.0 if x < 0 else 1.0

    def expected_result_fn(x):
      return 1 / 2 + tf.math.erf(x / (2 * tf.sqrt(dtype(final_t)))) / 2

    @dirichlet
    def lower_boundary_fn(t, x):
      del t, x
      return 0

    @dirichlet
    def upper_boundary_fn(t, x):
      del t, x
      return 1

    grid = grids.uniform_grid(
        minimums=[x_min], maximums=[x_max], sizes=[1000], dtype=dtype)

    self._testHeatEquation(
        grid=grid,
        final_t=final_t,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=crank_nicolson_with_oscillation_damping_step(),
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-3)

  @parameterized.named_parameters(
      {
          'testcase_name': 'DefaultBC',
          'lower_bc_type': 'Default',
          'upper_bc_type': 'Default',
      }, {
          'testcase_name': 'DefaultNeumanBC',
          'lower_bc_type': 'Default',
          'upper_bc_type': 'Neumann',
      }, {
          'testcase_name': 'NeumanDefaultBC',
          'lower_bc_type': 'Neumann',
          'upper_bc_type': 'Default',
      })
  def testHeatEquation_WithDefaultBoundaryCondtion(self,
                                                   lower_bc_type,
                                                   upper_bc_type):
    """Test for Default boundary conditions.

    Tests solving heat equation with the following boundary conditions involving
    default boundary `u_xx(0, t) = 0` or `u_xx(5 pi, t) = 0`.

    The exact solution `u(x, t=0) = e^t sin(x)`.
    Args:
      lower_bc_type: Lower boundary condition type.
      upper_bc_type: Upper boundary condition type.
    """

    def final_cond_fn(x):
      return math.e * math.sin(x)

    def expected_result_fn(x):
      return tf.sin(x)

    @neumann
    def boundary_fn(t, x):
      del x
      return -tf.exp(t)

    lower_boundary_fn = boundary_fn if lower_bc_type == 'Neumann' else None
    upper_boundary_fn = boundary_fn if upper_bc_type == 'Neumann' else None

    grid = grids.uniform_grid(
        minimums=[0.0], maximums=[5 * math.pi], sizes=[1000],
        dtype=np.float32)
    self._testHeatEquation(
        grid,
        final_t=1,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=crank_nicolson_step(),
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-3)

  def _testHeatEquation(self,
                        grid,
                        final_t,
                        time_step,
                        final_cond_fn,
                        expected_result_fn,
                        one_step_fn,
                        lower_boundary_fn,
                        upper_boundary_fn,
                        error_tolerance=1e-3):
    """Helper function with details of testing heat equation solving."""
    # Define coefficients for a PDE V_{t} + V_{XX} = 0.
    def second_order_coeff_fn(t, x):
      del t, x
      return [[1]]

    xs = self.evaluate(grid)[0]
    final_values = tf.constant([final_cond_fn(x) for x in xs],
                               dtype=grid[0].dtype)

    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        num_steps=None,
        start_step_count=0,
        time_step=time_step,
        one_step_fn=one_step_fn,
        boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)],
        values_transform_fn=None,
        second_order_coeff_fn=second_order_coeff_fn,
        dtype=grid[0].dtype)

    actual = self.evaluate(result[0])
    expected = self.evaluate(expected_result_fn(xs))
    self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

  @parameterized.named_parameters(
      {
          'testcase_name': 'DirichletBC',
          'bc_type': 'Dirichlet',
          'batch_grid': False,
      }, {
          'testcase_name': 'DefaultBC',
          'bc_type': 'Default',
          'batch_grid': False,
      }, {
          'testcase_name': 'DirichletBC_BatchGrid',
          'bc_type': 'Dirichlet',
          'batch_grid': True,
      }, {
          'testcase_name': 'DefaultBC_BatchGrid',
          'bc_type': 'Default',
          'batch_grid': True,
      })
  def testDocStringExample(self, bc_type, batch_grid):
    """Tests that the European Call option price is computed correctly."""
    num_equations = 2  # Number of PDE
    num_grid_points = 1024  # Number of grid points
    dtype = np.float64
    # Build a uniform grid
    if batch_grid:
      s_min = [0.01, 0.05]
      s_max = [200., 220]
      sizes = [num_grid_points, num_grid_points]
    else:
      s_min = [0.01]
      s_max = [200.0]
      sizes = [num_grid_points]

    grid = grids.uniform_grid(minimums=s_min,
                              maximums=s_max,
                              sizes=sizes,
                              dtype=dtype)
    # Shape [[batch_dim, num_grid_points]]
    grid = [tf.stack(grid, axis=0)]
    # Specify volatilities and interest rates for the options
    volatility = np.array([0.3, 0.15], dtype=dtype).reshape([-1, 1])
    rate = np.array([0.01, 0.03], dtype=dtype).reshape([-1, 1])
    expiry = 1.0
    strike = np.array([50, 100], dtype=dtype).reshape([-1, 1])

    def second_order_coeff_fn(t, location_grid):
      del t
      return [[tf.square(volatility) * tf.square(location_grid[0]) / 2]]

    def first_order_coeff_fn(t, location_grid):
      del t
      return [rate * location_grid[0]]

    def zeroth_order_coeff_fn(t, location_grid):
      del t, location_grid
      return -rate

    @dirichlet
    def lower_boundary_fn(t, location_grid):
      del t, location_grid
      return 0

    @dirichlet
    def upper_boundary_fn(t, location_grid):
      return (location_grid[0][..., -1]
              + tf.squeeze(-strike * tf.math.exp(-rate * (expiry - t))))

    final_values = tf.nn.relu(grid[0] - strike)
    # Broadcast to the shape of value dimension, if necessary.
    final_values += tf.zeros([num_equations, num_grid_points],
                             dtype=dtype)
    if bc_type == 'Default':
      boundary_conditions = [(None, upper_boundary_fn)]
    else:
      boundary_conditions = [(lower_boundary_fn, upper_boundary_fn)]
    # Estimate European call option price
    estimate = fd_solvers.solve_backward(
        start_time=expiry,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        num_steps=None,
        start_step_count=0,
        time_step=0.001,
        one_step_fn=crank_nicolson_step(),
        boundary_conditions=boundary_conditions,
        values_transform_fn=None,
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn,
        dtype=dtype)[0]
    estimate = self.evaluate(estimate)
    # Extract estimates for some of the grid locations and compare to the
    # true option price
    value_grid_first_option = estimate[0, :]
    value_grid_second_option = estimate[1, :]
    # Get two grid locations (correspond to spot 51.9537332 and 106.25407758,
    # respectively).
    loc_1 = 256
    loc_2 = 512
    # True call option price (obtained using black_scholes_price function)
    if batch_grid:
      spots = tf.stack([grid[0][0][loc_1], grid[0][-1][loc_2]])
    else:
      spots = tf.stack([grid[0][0][loc_1], grid[0][0][loc_2]])
    call_price = tff.black_scholes.option_price(
        volatilities=volatility[..., 0],
        strikes=strike[..., 0],
        expiries=expiry,
        discount_rates=rate[..., 0],
        spots=spots)
    self.assertAllClose(
        call_price, [value_grid_first_option[loc_1],
                     value_grid_second_option[loc_2]],
        rtol=1e-03, atol=1e-03)

  def testEuropeanCallDynamicVol(self):
    """Price for the European Call option with time-dependent volatility."""
    num_equations = 1  # Number of PDE
    num_grid_points = 1024  # Number of grid points
    dtype = np.float64
    # Build a log-uniform grid
    s_max = 300.
    grid = grids.log_uniform_grid(minimums=[0.01], maximums=[s_max],
                                  sizes=[num_grid_points],
                                  dtype=dtype)
    # Specify volatilities and interest rates for the options
    expiry = 1.0
    strike = 50.0

    # Volatility is of the form  `sigma**2(t) = 1 / 6 + 1 / 2 * t**2`.
    def second_order_coeff_fn(t, location_grid):
      return [[(1. / 6 + t**2 / 2) * tf.square(location_grid[0]) / 2]]

    @dirichlet
    def lower_boundary_fn(t, location_grid):
      del t, location_grid
      return 0

    @dirichlet
    def upper_boundary_fn(t, location_grid):
      del t
      return location_grid[0][-1] - strike

    final_values = tf.nn.relu(grid[0] - strike)
    # Broadcast to the shape of value dimension, if necessary.
    final_values += tf.zeros([num_equations, num_grid_points],
                             dtype=dtype)
    # Estimate European call option price
    estimate = fd_solvers.solve_backward(
        start_time=expiry,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        num_steps=None,
        start_step_count=0,
        time_step=tf.constant(0.01, dtype=dtype),
        one_step_fn=crank_nicolson_step(),
        boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)],
        values_transform_fn=None,
        second_order_coeff_fn=second_order_coeff_fn,
        dtype=dtype)[0]

    value_grid = self.evaluate(estimate)[0, :]
    # Get two grid locations (correspond to spot 51.9537332 and 106.25407758,
    # respectively).
    loc_1 = 849
    # True call option price (obtained using black_scholes_price function)
    call_price = 12.582092
    self.assertAllClose(call_price, value_grid[loc_1], rtol=1e-02, atol=1e-02)

  def testHeatEquation_InForwardDirection(self):
    """Test solving heat equation with various time marching schemes.

    Tests solving heat equation with the boundary conditions
    `u(x, t=1) = e * sin(x)`, `u(-2 pi n - pi / 2, t) = -e^t`, and
    `u(2 pi n + pi / 2, t) = -e^t` with some integer `n` for `u(x, t=0)`.

    The exact solution is `u(x, t=0) = sin(x)`.

    All time marching schemes should yield reasonable results given small enough
    time steps. First-order accurate schemes (explicit, implicit, weighted with
    theta != 0.5) require smaller time step than second-order accurate ones
    (Crank-Nicolson, Extrapolation).
    """
    final_time = 1.0

    def initial_cond_fn(x):
      return tf.sin(x)

    def expected_result_fn(x):
      return np.exp(-final_time) * tf.sin(x)

    @dirichlet
    def lower_boundary_fn(t, x):
      del x
      return -tf.math.exp(-t)

    @dirichlet
    def upper_boundary_fn(t, x):
      del x
      return tf.math.exp(-t)

    grid = grids.uniform_grid(
        minimums=[-10.5 * math.pi],
        maximums=[10.5 * math.pi],
        sizes=[1000],
        dtype=np.float32)

    def second_order_coeff_fn(t, x):
      del t, x
      return [[-1]]

    final_values = initial_cond_fn(grid[0])

    result = fd_solvers.solve_forward(
        start_time=0.0,
        end_time=final_time,
        coord_grid=grid,
        values_grid=final_values,
        time_step=0.01,
        boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)],
        second_order_coeff_fn=second_order_coeff_fn)[0]

    actual = self.evaluate(result)
    expected = self.evaluate(expected_result_fn(grid[0]))
    self.assertLess(np.max(np.abs(actual - expected)), 1e-3)

  def testReferenceEquation(self):
    """Tests the equation used as reference for a few further tests.

    We solve the diffusion equation `u_t = u_xx` on x = [0...1] with boundary
    conditions `u(x<=1/2, t=0) = x`, `u(x>1/2, t=0) = 1 - x`,
    `u(x=0, t) = u(x=1, t) = 0`.

    The exact solution of the diffusion equation with zero-Dirichlet boundaries
    is:
    `u(x, t) = sum_{n=1..inf} b_n sin(pi n x) exp(-n^2 pi^2 t)`,
    `b_n = 2 integral_{0..1} sin(pi n x) u(x, t=0) dx.`

    The initial conditions are taken so that the integral easily calculates, and
    the sum can be approximated by a few first terms (given large enough `t`).
    See the result in _reference_heat_equation_solution.

    Using this solution helps to simplify the tests, as we don't have to
    maintain complicated boundary conditions in each test or tweak the
    parameters to keep the "support" of the function far from boundaries.
    """
    grid = grids.uniform_grid(
        minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
    xs = grid[0]

    final_t = 0.1
    time_step = 0.001

    def second_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return [[-1]]

    initial = _reference_pde_initial_cond(xs)
    expected = _reference_pde_solution(xs, final_t)
    actual = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial,
        time_step=time_step,
        second_order_coeff_fn=second_order_coeff_fn)[0]

    self.assertAllClose(expected, actual, atol=1e-3, rtol=1e-3)

  def testReference_WithExponentMultiplier(self):
    """Tests solving diffusion equation with an exponent multiplier.

    Take the heat equation `v_{t} - v_{xx} = 0` and substitute `v = exp(x) u`.
    This yields `u_{t} - u_{xx} - 2u_{x} - u = 0`. The test compares numerical
    solution of this equation to the exact one, which is the diffusion equation
    solution times `exp(-x)`.
    """
    grid = grids.uniform_grid(
        minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
    xs = grid[0]

    final_t = 0.1
    time_step = 0.001

    def second_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return [[-1]]

    def first_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return [-2]

    def zeroth_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return -1

    initial = tf.math.exp(-xs) * _reference_pde_initial_cond(xs)
    expected = tf.math.exp(-xs) * _reference_pde_solution(xs, final_t)

    actual = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial,
        time_step=time_step,
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]

    self.assertAllClose(expected, actual, atol=1e-3, rtol=1e-3)

  def testInnerSecondOrderCoeff(self):
    """Tests handling inner_second_order_coeff.

    As in previous test, take the diffusion equation `v_{t} - v_{xx} = 0` and
    substitute `v = exp(x) u`, but this time keep exponent under the derivative:
    `u_{t} - exp(-x)[exp(x)u]_{xx} = 0`. Expect the same solution as in
    previous test.
    """
    grid = grids.uniform_grid(
        minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
    xs = grid[0]

    final_t = 0.1
    time_step = 0.001

    def second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[-tf.math.exp(-x)]]

    def inner_second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[tf.math.exp(x)]]

    initial = tf.math.exp(-xs) * _reference_pde_initial_cond(xs)
    expected = tf.math.exp(-xs) * _reference_pde_solution(xs, final_t)

    actual = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial,
        time_step=time_step,
        second_order_coeff_fn=second_order_coeff_fn,
        inner_second_order_coeff_fn=inner_second_order_coeff_fn)[0]

    self.assertAllClose(expected, actual, atol=1e-3, rtol=1e-3)

  def testInnerFirstAndSecondOrderCoeff(self):
    """Tests handling both inner_first_order_coeff and inner_second_order_coeff.

    We saw previously that the solution of `u_{t} - u_{xx} - 2u_{x} - u = 0` is
    `u = exp(-x) v`, where v solves the diffusion equation. Substitute now
    `u = exp(-x) v` without expanding the derivatives:
    `v_{t} - exp(x)[exp(-x)v]_{xx} - 2exp(x)[exp(-x)v]_{x} - v = 0`.
    Solve this equation and expect the solution of the diffusion equation.
    """
    grid = grids.uniform_grid(
        minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
    xs = grid[0]

    final_t = 0.1
    time_step = 0.001

    def second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[-tf.math.exp(x)]]

    def inner_second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[tf.math.exp(-x)]]

    def first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [-2 * tf.math.exp(x)]

    def inner_first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [tf.math.exp(-x)]

    def zeroth_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return -1

    initial = _reference_pde_initial_cond(xs)
    expected = _reference_pde_solution(xs, final_t)

    actual = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial,
        time_step=time_step,
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn,
        inner_second_order_coeff_fn=inner_second_order_coeff_fn,
        inner_first_order_coeff_fn=inner_first_order_coeff_fn)[0]

    self.assertAllClose(expected, actual, atol=1e-3, rtol=1e-3)

  def testCompareExpandedAndNotExpandedPdes(self):
    """Tests comparing PDEs with expanded derivatives and without.

    Take equation `u_{t} - [x^2 u]_{xx} + [x u]_{x} = 0`.
    Expanding the derivatives yields `u_{t} - x^2 u_{xx} - 3x u_{x} - u = 0`.
    Solve both equations and expect the results to be equal.
    """
    grid = grids.uniform_grid(
        minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
    xs = grid[0]

    final_t = 0.1
    time_step = 0.001

    initial = _reference_pde_initial_cond(xs)  # arbitrary

    def inner_second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[-tf.square(x)]]

    def inner_first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [x]

    result_not_expanded = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial,
        time_step=time_step,
        inner_second_order_coeff_fn=inner_second_order_coeff_fn,
        inner_first_order_coeff_fn=inner_first_order_coeff_fn)[0]

    def second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[-tf.square(x)]]

    def first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [-3 * x]

    def zeroth_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return -1

    result_expanded = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial,
        time_step=time_step,
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]

    self.assertAllClose(
        result_not_expanded, result_expanded, atol=1e-3, rtol=1e-3)

  def testDefaultBoundaryConditions(self):
    """Test for PDE with default boundary condition and no inner term.

    Take equation `u_{t} - x u_{xx} + (x - 1) u_{x} = 0` with boundary
    conditions `u_{t} + (x - 1) u_{x} = 0` at x = 0 and `u(t, 1) = exp(t + 1)`
    with an initial condition `u(0, x) = exp(x)`.

    Solve this equation and compare the result to `u(t, x) = exp(t + x)`.
    """
    @dirichlet
    def upper_boundary_fn(t, x):
      del x
      return tf.math.exp(t + 1)

    def second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[-x]]

    def first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [x - 1]

    grid = self.evaluate(grids.uniform_grid(
        minimums=[0],
        maximums=[1],
        sizes=[1000],
        dtype=np.float64))

    initial = tf.math.exp(grid[0])  # Initial condition
    time_step = 0.01
    final_t = 0.5

    est_values = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial,
        time_step=time_step,
        one_step_fn=crank_nicolson_step(),
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        boundary_conditions=[(None, upper_boundary_fn)])[0]

    true_values = tf.math.exp(final_t + grid[0])
    self.assertAllClose(
        est_values, true_values, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'LeftDefault',
          'default_bc': 'left',
      }, {
          'testcase_name': 'RightDefault',
          'default_bc': 'right',
      }, {
          'testcase_name': 'BothDefault',
          'default_bc': 'both',
      })
  def testDefaultBoundaryConditionsWithInnerTerm(self, default_bc):
    """Test for PDE with default boundary condition with inner term.

    Take equation
    `u_{t} - (x - x**3)[u]_{xx} + (1 + x) * [(1 - x**2) u]_{x}
     + (2 * x**2 - 1 + 2 *x - (1 - x**2))u = 0` with
    boundary conditions `u_{t} + (x - 1) u_{x} = 0` at x = 0
    and `u(t, 1) = exp(t + 1)`, and an initial condition `u(0, x) = exp(x)`.

    Solve this equation and compare the result to `u(t, x) = exp(t + x)`.

    Args:
      default_bc: A string to indicate which boundary condition is 'default'.
        Can be either 'left', 'right', or 'both'.
    """
    def second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[-(-x**3 + x)]]

    def first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [1 + x]

    def inner_first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [-x**2 + 1]

    @dirichlet
    def lower_boundary_fn(t, x):
      del x
      return tf.math.exp(t)

    @dirichlet
    def upper_boundary_fn(t, x):
      del x
      return tf.math.exp(1.0 + t)

    def zeroth_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return 2 * x**2 - 1 + 2 *x - (1 - x**2)

    grid = self.evaluate(grids.uniform_grid(
        minimums=[0],
        maximums=[1],
        sizes=[100],
        dtype=np.float64))

    initial_values = tf.math.exp(grid[0])  # Initial condition
    time_step = 0.001
    final_t = 0.1
    if default_bc == 'left':
      boundary_conditions = [(None, upper_boundary_fn)]
    elif default_bc == 'right':
      boundary_conditions = [(lower_boundary_fn, None)]
    else:
      boundary_conditions = [(None, None)]
    est_values = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial_values,
        time_step=time_step,
        boundary_conditions=boundary_conditions,
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        inner_first_order_coeff_fn=inner_first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]

    true_values = tf.math.exp(final_t + grid[0])
    self.assertAllClose(
        est_values, true_values, atol=1e-2, rtol=1e-2)

  def testDefaultBoundaryConditionsInnerTermNoOuterLower(self):
    """Test for PDE with default boundary condition with inner term.

    Take equation
    `u_{t} - (x - x**3)[u]_{xx} + [(x - x**3) u]_{x} + (3 * x**2 - 2)u = 0` with
    boundary conditions `u_{t} + (x - 1) u_{x} = 0` at x = 0
    and `u(t, 1) = exp(t + 1)`, and an initial condition `u(0, x) = exp(x)`.

    Solve this equation and compare the result to `u(t, x) = exp(t + x)`.
    """
    def second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[-(x - x**3)]]

    def first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [tf.ones_like(x)]

    def inner_first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [x - x**3]

    @dirichlet
    def upper_boundary_fn(t, x):
      del x
      return tf.math.exp(1 + t)

    def zeroth_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return 3 * x**2 - 2

    grid = self.evaluate(grids.uniform_grid(
        minimums=[0],
        maximums=[1],
        sizes=[100],
        dtype=np.float64))

    initial_values = tf.expand_dims(tf.math.exp(grid[0]), axis=0)
    final_t = 0.1
    time_step = 0.001

    est_values = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial_values,
        time_step=time_step,
        boundary_conditions=[(None, upper_boundary_fn)],
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        inner_first_order_coeff_fn=inner_first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]

    true_values = tf.expand_dims(tf.math.exp(final_t + grid[0]), axis=0)
    self.assertAllClose(
        est_values, true_values, atol=1e-2, rtol=1e-2)

  def testDefaultBoundaryConditionsInnerTermNoOuterUpper(self):
    """Test for PDE with default boundary condition with inner term.

    Take equation
    `u_{t} - (1 - x)[u]_{xx} + [(1 - x) u]_{x} = 0` with
    boundary conditions `u_{t} + (x - 1) u_{x} = 0` at x = 0
    and `u(t, 1) = exp(t + 1)`, and an initial condition `u(0, x) = exp(x)`.

    Solve this equation and compare the result to `u(t, x) = exp(t + x)`.
    """
    def second_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [[-(1 - x)]]

    def inner_first_order_coeff_fn(t, coord_grid):
      del t
      x = coord_grid[0]
      return [1 - x]

    @dirichlet
    def lower_boundary_fn(t, x):
      del x
      return tf.math.exp(t)

    grid = self.evaluate(grids.uniform_grid(
        minimums=[0],
        maximums=[1],
        sizes=[100],
        dtype=np.float64))

    initial_values = tf.expand_dims(tf.math.exp(grid[0]), axis=0)
    final_t = 0.1
    time_step = 0.001

    est_values = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial_values,
        time_step=time_step,
        boundary_conditions=[(lower_boundary_fn, None)],
        second_order_coeff_fn=second_order_coeff_fn,
        inner_first_order_coeff_fn=inner_first_order_coeff_fn)[0]

    true_values = tf.expand_dims(tf.math.exp(final_t + grid[0]), axis=0)
    self.assertAllClose(
        est_values, true_values, atol=1e-2, rtol=1e-2)


def _reference_pde_initial_cond(xs):
  """Initial conditions for the reference diffusion equation."""
  return -tf.math.abs(xs - 0.5) + 0.5


def _reference_pde_solution(xs, t, num_terms=5):
  """Solution for the reference diffusion equation."""
  u = tf.zeros_like(xs)
  for k in range(num_terms):
    n = 2 * k + 1
    term = tf.math.sin(np.pi * n * xs) * tf.math.exp(-n**2 * np.pi**2 * t)
    term *= 4 / (np.pi**2 * n**2)
    if k % 2 == 1:
      term *= -1
    u += term
  return u


if __name__ == '__main__':
  tf.test.main()
