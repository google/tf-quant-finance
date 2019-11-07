# Lint as: python2, python3
"""Tests for 1-D parabolic PDE solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tf_quant_finance.experimental.pde_v2 import fd_solvers
from tf_quant_finance.experimental.pde_v2.boundary_conditions import dirichlet
from tf_quant_finance.experimental.pde_v2.boundary_conditions import neumann
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.crank_nicolson import crank_nicolson_step
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.explicit import explicit_step
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.extrapolation import extrapolation_step
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.implicit import implicit_step
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.oscillation_damped_crank_nicolson import crank_nicolson_with_oscillation_damping_step
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.weighted_implicit_explicit import weighted_implicit_explicit_step
from tf_quant_finance.experimental.pde_v2.grids import grids

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class ParabolicEquationStepperTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'Implicit',
          'one_step_fn': implicit_step,
          'time_step': 0.001
      }, {
          'testcase_name': 'Explicit',
          'one_step_fn': explicit_step,
          'time_step': 0.001
      }, {
          'testcase_name': 'Weighted',
          'one_step_fn': weighted_implicit_explicit_step(theta=0.3),
          'time_step': 0.001
      }, {
          'testcase_name': 'CrankNicolson',
          'one_step_fn': crank_nicolson_step,
          'time_step': 0.01
      }, {
          'testcase_name': 'Extrapolation',
          'one_step_fn': extrapolation_step,
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
      return -tf.exp(t)

    @dirichlet
    def upper_boundary_fn(t, x):
      del x
      return tf.exp(t)

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
      return -tf.exp(t)

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
        one_step_fn=crank_nicolson_step,
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
      return -tf.exp(t)

    @dirichlet
    def upper_boundary_fn(t, x):
      del x
      return tf.exp(t)

    grid = grids.uniform_grid(minimums=[0], maximums=[10.5 * math.pi],
                              sizes=[1000], dtype=np.float32)
    self._testHeatEquation(
        grid,
        final_t=1,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=crank_nicolson_step,
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
      return 2, -1, tf.exp(t)

    def upper_boundary_fn(t, x):
      del x
      return 2, 1, 2 * tf.exp(t)

    grid = grids.uniform_grid(minimums=[0], maximums=[4.5 * math.pi],
                              sizes=[1000], dtype=np.float64)
    self._testHeatEquation(
        grid,
        final_t=1,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        one_step_fn=crank_nicolson_step,
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
      return 2, -1, tf.exp(t)

    def upper_boundary_fn(t, x):
      del x
      return 2, 1, 2 * tf.exp(t)

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
        one_step_fn=crank_nicolson_step,
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
      return 2, -1, tf.exp(t)

    def upper_boundary_fn(t, x):
      del x
      return 2, 1, 2 * tf.exp(t)

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
        one_step_fn=crank_nicolson_step,
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
      return 0.0

    @dirichlet
    def upper_boundary_fn(t, x):
      del t, x
      return 1.0

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

    result = fd_solvers.step_back(
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

  def testDocStringExample(self):
    """Tests that the European Call option price is computed correctly."""
    num_equations = 2  # Number of PDE
    num_grid_points = 1024  # Number of grid points
    dtype = np.float64
    # Build a log-uniform grid
    s_max = 300.
    grid = grids.log_uniform_grid(minimums=[0.01], maximums=[s_max],
                                  sizes=[num_grid_points],
                                  dtype=dtype)
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
      return dtype([0.0, 0.0])

    @dirichlet
    def upper_boundary_fn(t, location_grid):
      return tf.squeeze(location_grid[0][-1]
                        - strike * tf.exp(-rate * (expiry - t)))

    final_values = tf.nn.relu(grid[0] - strike)
    # Broadcast to the shape of value dimension, if necessary.
    final_values += tf.zeros([num_equations, num_grid_points],
                             dtype=dtype)
    # Estimate European call option price
    estimate = fd_solvers.step_back(
        start_time=expiry,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        num_steps=None,
        start_step_count=0,
        time_step=tf.constant(0.01, dtype=dtype),
        one_step_fn=crank_nicolson_step,
        boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)],
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
    loc_1 = 849
    loc_2 = 920
    # True call option price (obtained using black_scholes_price function)
    call_price = [7.35192484, 11.75642136]

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
      return dtype([0.0])

    @dirichlet
    def upper_boundary_fn(t, location_grid):
      del t
      return location_grid[0][-1] - strike

    final_values = tf.nn.relu(grid[0] - strike)
    # Broadcast to the shape of value dimension, if necessary.
    final_values += tf.zeros([num_equations, num_grid_points],
                             dtype=dtype)
    # Estimate European call option price
    estimate = fd_solvers.step_back(
        start_time=expiry,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        num_steps=None,
        start_step_count=0,
        time_step=tf.constant(0.01, dtype=dtype),
        one_step_fn=crank_nicolson_step,
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


if __name__ == '__main__':
  tf.test.main()
