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
"""Tests for `ParabolicDifferentialEquationSolver`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tf_quant_finance.math import pde
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

# Time marching schemes
schemes = pde.time_marching_schemes


@test_util.run_all_in_graph_and_eager_modes
class PdeKernelsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'Implicit',
          'scheme': schemes.implicit_scheme(),
          'time_step': 0.001
      }, {
          'testcase_name': 'Explicit',
          'scheme': schemes.explicit_scheme(),
          'time_step': 0.001
      }, {
          'testcase_name': 'Weighted',
          'scheme': schemes.WeightedImplicitExplicitScheme(theta=0.3),
          'time_step': 0.001
      }, {
          'testcase_name': 'CrankNicolson',
          'scheme': schemes.crank_nicolson_scheme(),
          'time_step': 0.01
      }, {
          'testcase_name': 'Extrapolation',
          'scheme': schemes.ExtrapolationMarchingScheme(),
          'time_step': 0.01
      }, {
          'testcase_name': 'CrankNicolsonWithOscillationDamping',
          'scheme': schemes.crank_nicolson_with_oscillation_damping(),
          'time_step': 0.01
      })
  def testHeatEquationWithVariousSchemes(self, scheme, time_step):
    """Test solving heat equation with various time marching schemes.

    Tests solving heat equation with the following final conditions
    `u(x, t=1) = e * sin(x)` for `u(x, t=0)`.

    Then, the exact solution for unbounded x is `u(x, t=0) = sin(x)`.
    ```

    We take x_min and x_max as (n + 1/2)*pi with some integers n, in which case
    the solution above is also correct for von Neumann boundary conditions.

    All time marching schemes should yield reasonable results given small enough
    time steps. First-order accurate schemes (explicit, implicit, weighted with
    theta != 0.5) require smaller time step than second-order accurate ones
    (Crank-Nicolson, Extrapolation).

    Args:
      scheme: time marching scheme used.
      time_step: time step for given scheme.
    """
    dtype = np.float64

    def final_cond_fn(x):
      return math.e * math.sin(x)

    def expected_result_fn(x):
      return tf.sin(x)

    def lower_boundary_fn(t):
      return [-tf.exp(t)]

    def upper_boundary_fn(t):
      return [tf.exp(t)]

    self._testHeatEquation(
        x_min=-10.5 * math.pi,
        x_max=10.5 * math.pi,
        final_t=1,
        time_step=time_step,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        time_marching_scheme=scheme,
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-3,
        dtype=dtype)

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
    dtype = np.float64

    def final_cond_fn(x):
      return 0.0 if x < 0 else 1.0

    def expected_result_fn(x):
      return 1 / 2 + tf.math.erf(x / (2 * tf.sqrt(dtype(final_t)))) / 2

    def lower_boundary_fn(t):
      del t
      return dtype([0.0])

    def upper_boundary_fn(t):
      del t
      return dtype([1.0])

    time_marching_scheme = schemes.crank_nicolson_with_oscillation_damping()

    self._testHeatEquation(
        x_min=x_min,
        x_max=x_max,
        final_t=final_t,
        time_step=0.01,
        final_cond_fn=final_cond_fn,
        expected_result_fn=expected_result_fn,
        time_marching_scheme=time_marching_scheme,
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_fn,
        error_tolerance=1e-3,
        dtype=dtype)

  def _testHeatEquation(self,
                        x_min,
                        x_max,
                        final_t,
                        time_step,
                        final_cond_fn,
                        expected_result_fn,
                        time_marching_scheme,
                        lower_boundary_fn,
                        upper_boundary_fn,
                        error_tolerance=1e-3,
                        dtype=np.float64):
    """Helper function with details of testing heat equation solving."""
    num_grid_points = 1000
    grid = pde.grids.uniform_grid(
        minimums=[x_min],
        maximums=[x_max],
        sizes=[num_grid_points],
        dtype=dtype)
    xs = self.evaluate(grid.locations[0])

    # Define coefficients for a PDE V_{t} + V_{XX} = 0.
    def quadratic_term_fn(t, x):
      del t, x
      return 2

    solver_kernel = pde.ParabolicDifferentialEquationSolver(
        quadratic_term_fn,
        lower_boundary_fn,
        upper_boundary_fn,
        time_marching_scheme=time_marching_scheme)

    def time_step_fn(state):
      del state
      return tf.constant(time_step, dtype=dtype)

    def final_conditions(_):
      return tf.expand_dims(
          tf.constant([final_cond_fn(x) for x in xs], dtype=dtype), axis=0)

    bgs = pde.BackwardGridStepper(
        final_t,
        solver_kernel.one_step,
        grid,
        time_step_fn=time_step_fn,
        value_dim=1,
        dtype=dtype)
    bgs.transform_values(final_conditions)

    # Run the solver.
    bgs.step_back_to_time(0.0)
    result_state = bgs.state()

    values = self.evaluate(result_state.value_grid[0, :])
    expected = self.evaluate(expected_result_fn(xs))
    self.assertLess(np.max(np.abs(values - expected)), error_tolerance)

  def testDocStringExample(self):
    """Tests that the European Call option price is computed correctly."""
    num_equations = 2  # Number of PDE
    num_grid_points = 1024  # Number of grid points
    dtype = np.float64
    # Build a log-uniform grid
    s_max = 300.
    grid = pde.grids.log_uniform_grid(
        minimums=[0.01], maximums=[s_max], sizes=[num_grid_points], dtype=dtype)
    # Specify volatilities and interest rates for the options
    volatility = np.array([0.3, 0.15], dtype=dtype).reshape([-1, 1])
    rate = np.array([0.01, 0.03], dtype=dtype).reshape([-1, 1])
    expiry = 1.0
    strike = np.array([50, 100], dtype=dtype).reshape([-1, 1])

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

    solver_kernel = pde.ParabolicDifferentialEquationSolver(
        quadratic_coeff_fn, lower_boundary_fn, upper_boundary_fn,
        linear_coeff_fn, shift_coeff_fn)

    def time_step_fn(state):
      del state
      return tf.constant(0.01, dtype=dtype)

    def payoff_fn(state):
      option_bound = tf.nn.relu(state.coordinate_grid.locations[0] - strike)
      # Broadcast to the shape of value dimension, if necessary.
      option_bound = option_bound + tf.zeros_like(state.value_grid)
      return option_bound

    bgs = pde.BackwardGridStepper(
        expiry,
        solver_kernel.one_step,
        grid,
        time_step_fn=time_step_fn,
        value_dim=num_equations,
        dtype=dtype)
    bgs.transform_values(payoff_fn)
    bgs.step_back_to_time(0.0)
    # Estimate European call option price
    estimate = self.evaluate(bgs.state())
    # Extract estimates for some of the grid locations and compare to the
    # True option price
    value_grid_first_option = estimate.value_grid[0, :]
    value_grid_second_option = estimate.value_grid[1, :]
    # Get two grid locations (correspond to spot 51.9537332 and 106.25407758,
    # respectively).
    loc_1 = 849
    loc_2 = 920
    # True call option price (obtained using black_scholes_price function)
    call_price = [7.35192484, 11.75642136]

    self.assertAllClose(
        call_price,
        [value_grid_first_option[loc_1], value_grid_second_option[loc_2]],
        rtol=1e-03,
        atol=1e-03)

  def testEuropeanCallDynamicVol(self):
    """Price for the European Call option with time-dependent volatility."""
    num_equations = 1  # Number of PDE
    num_grid_points = 1024  # Number of grid points
    dtype = np.float64
    # Build a log-uniform grid
    s_max = 300.
    grid = pde.grids.log_uniform_grid(
        minimums=[0.01], maximums=[s_max], sizes=[num_grid_points], dtype=dtype)
    # Specify expiry and strike. Rate is assumed to be zero
    expiry = 1.0
    strike = 50.

    # Volatility is of the form
    # `sigma**2(t) = 1 / 6 + 1 / 2 * t**2`.
    def quadratic_coeff_fn(t, x):
      return (1. / 6 + t**2 / 2) * tf.square(x)

    def lower_boundary_fn(t):
      del t
      return dtype([0.0])

    def upper_boundary_fn(t):
      del t
      return dtype([s_max - strike])

    solver_kernel = pde.ParabolicDifferentialEquationSolver(
        quadratic_coeff_fn, lower_boundary_fn, upper_boundary_fn)

    def time_step_fn(state):
      del state
      return tf.constant(0.001, dtype=dtype)

    def payoff_fn(state):
      option_bound = tf.nn.relu(state.coordinate_grid.locations[0] - strike)
      # Broadcast to the shape of value dimension, if necessary.
      option_bound = option_bound + tf.zeros_like(state.value_grid)
      return option_bound

    bgs = pde.BackwardGridStepper(
        expiry,
        solver_kernel.one_step,
        grid,
        time_step_fn=time_step_fn,
        value_dim=num_equations,
        dtype=dtype)
    bgs.transform_values(payoff_fn)
    bgs.step_back_to_time(0.0)
    # Estimate European call option price.
    estimate = self.evaluate(bgs.state())
    # Extract estimates for value grid locations.
    value_grid = estimate.value_grid[0, :]
    # Get two grid location that correspond to spot 51.9537332.
    loc_1 = 849
    # True call option price
    call_price = 12.582092
    self.assertAllClose(call_price, value_grid[loc_1], rtol=1e-02, atol=1e-02)


if __name__ == '__main__':
  tf.test.main()
