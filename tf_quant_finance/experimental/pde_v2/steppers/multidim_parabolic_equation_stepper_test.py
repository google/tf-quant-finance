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
"""Tests for multidimensional parabolic PDE solvers."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_quant_finance.experimental.pde_v2 import fd_solvers
from tf_quant_finance.experimental.pde_v2.boundary_conditions import dirichlet
from tf_quant_finance.experimental.pde_v2.boundary_conditions import neumann
from tf_quant_finance.experimental.pde_v2.grids import grids
from tf_quant_finance.experimental.pde_v2.steppers.douglas_adi import douglas_adi_step
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

_SQRT2 = np.sqrt(2)


@test_util.run_all_in_graph_and_eager_modes
class MultidimParabolicEquationStepperTest(tf.test.TestCase):

  def testAnisotropicDiffusion(self):
    """Tests solving 2d diffusion equation.

    The equation is `u_{t} + Dx u_{xx} + Dy u_{yy} = 0`.
    The final condition is a gaussian centered at (0, 0) with variance sigma.
    The variance along each dimension should evolve as
    `sigma + 2 Dx (t_final - t)` and `sigma + 2 Dy (t_final - t)`.
    """
    grid = grids.uniform_grid(
        minimums=[-10, -20],
        maximums=[10, 20],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    diff_coeff_x = 0.4  # Dx
    diff_coeff_y = 0.25  # Dy
    time_step = 0.1
    final_t = 1
    final_variance = 1

    def quadratic_coeff_fn(t, location_grid):
      del t, location_grid
      u_xx = diff_coeff_x
      u_yy = diff_coeff_y
      u_xy = None
      return [[u_yy, u_xy], [u_xy, u_xx]]

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(
                _gaussian(ys, final_variance), _gaussian(xs, final_variance)),
            dtype=tf.float32),
        axis=0)
    bound_cond = [(_zero_boundary, _zero_boundary),
                  (_zero_boundary, _zero_boundary)]
    step_fn = douglas_adi_step(theta=0.5)
    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=step_fn,
        boundary_conditions=bound_cond,
        second_order_coeff_fn=quadratic_coeff_fn,
        dtype=grid[0].dtype)

    variance_x = final_variance + 2 * diff_coeff_x * final_t
    variance_y = final_variance + 2 * diff_coeff_y * final_t
    expected = np.outer(_gaussian(ys, variance_y), _gaussian(xs, variance_x))

    self._assertClose(expected, result)

  def testSimpleDrift(self):
    """Tests solving 2d drift equation.

    The equation is `u_{t} + vx u_{x} + vy u_{y} = 0`.
    The final condition is a gaussian centered at (0, 0) with variance sigma.
    The gaussian should drift with velocity `[vx, vy]`.
    """
    grid = grids.uniform_grid(
        minimums=[-10, -20],
        maximums=[10, 20],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    time_step = 0.01
    final_t = 3
    variance = 1
    vx = 0.1
    vy = 0.3

    def first_order_coeff_fn(t, location_grid):
      del t, location_grid
      return [vy, vx]

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(_gaussian(ys, variance), _gaussian(xs, variance)),
            dtype=tf.float32),
        axis=0)

    bound_cond = [(_zero_boundary, _zero_boundary),
                  (_zero_boundary, _zero_boundary)]

    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=douglas_adi_step(theta=0.5),
        boundary_conditions=bound_cond,
        first_order_coeff_fn=first_order_coeff_fn,
        dtype=grid[0].dtype)

    expected = np.outer(
        _gaussian(ys + vy * final_t, variance),
        _gaussian(xs + vx * final_t, variance))

    self._assertClose(expected, result)

  # These four tests below run _testAnisotropicDiffusion with different formats
  # of quadratic term.
  def testAnisotropicDiffusion_TwoDimList(self):
    def pack_second_order_coeff_fn(u_yy, u_xy, u_xx):
      return [[u_yy, u_xy], [u_xy, u_xx]]
    self._testDiffusionInDiagonalDirection(pack_second_order_coeff_fn)

  def testAnisotropicDiffusion_TwoDimList_WithoutRedundantElement(self):
    def pack_second_order_coeff_fn(u_yy, u_xy, u_xx):
      return [[u_yy, u_xy], [None, u_xx]]
    self._testDiffusionInDiagonalDirection(pack_second_order_coeff_fn)

  def testAnisotropicDiffusion_ListOfTensors(self):
    def pack_second_order_coeff_fn(u_yy, u_xy, u_xx):
      return [tf.constant([u_yy, u_xy], dtype=tf.float32),
              tf.constant([u_xy, u_xx], dtype=tf.float32)]
    self._testDiffusionInDiagonalDirection(pack_second_order_coeff_fn)

  def testAnisotropicDiffusion_2DTensor(self):
    def pack_second_order_coeff_fn(u_yy, u_xy, u_xx):
      return tf.convert_to_tensor([[u_yy, u_xy], [u_xy, u_xx]],
                                  dtype=tf.float32)
    self._testDiffusionInDiagonalDirection(pack_second_order_coeff_fn)

  # pylint: disable=g-doc-args
  def _testDiffusionInDiagonalDirection(self, pack_second_order_coeff_fn):
    """Tests solving 2d diffusion equation involving mixed terms.

    The equation is `u_{t} + D u_{xx} / 2 +  D u_{yy} / 2 + D u_{xy} = 0`.
    The final condition is a gaussian centered at (0, 0) with variance sigma.

    The equation can be rewritten as `u_{t} + D u_{zz} = 0`, where
    `z = (x + y) / sqrt(2)`.

    Thus variance should evolve as `sigma + 2D(t_final - t)` along z dimension
    and stay unchanged in the orthogonal dimension:
    `u(x, y, t) = gaussian((x + y)/sqrt(2), sigma + 2D(t_final - t)) *
    gaussian((x - y)/sqrt(2), sigma)`.
    """
    dtype = tf.float32

    grid = grids.uniform_grid(
        minimums=[-10, -20], maximums=[10, 20], sizes=[201, 301], dtype=dtype)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    diff_coeff = 1  # D
    time_step = 0.1
    final_t = 3
    final_variance = 1

    def second_order_coeff_fn(t, location_grid):
      del t, location_grid
      return pack_second_order_coeff_fn(diff_coeff / 2, diff_coeff / 2,
                                        diff_coeff / 2)

    variance_along_diagonal = final_variance + 2 * diff_coeff * final_t

    def expected_fn(x, y):
      return (_gaussian((x + y) / _SQRT2, variance_along_diagonal) * _gaussian(
          (x - y) / _SQRT2, final_variance))

    expected = np.array([[expected_fn(x, y) for x in xs] for y in ys])

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(
                _gaussian(ys, final_variance), _gaussian(xs, final_variance)),
            dtype=dtype),
        axis=0)
    bound_cond = [(_zero_boundary, _zero_boundary),
                  (_zero_boundary, _zero_boundary)]
    step_fn = douglas_adi_step(theta=0.5)
    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=step_fn,
        boundary_conditions=bound_cond,
        second_order_coeff_fn=second_order_coeff_fn,
        dtype=grid[0].dtype)

    self._assertClose(expected, result)

  def testShiftTerm(self):
    """Simple test for the shift term.

    The equation is `u_{t} + a u = 0`, the solution is
    `u(x, y, t) = exp(-a(t - t_final)) u(x, y, t_final)`
    """
    grid = grids.uniform_grid(
        minimums=[-10, -20],
        maximums=[10, 20],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    time_step = 0.1
    final_t = 1
    variance = 1
    a = 2

    def zeroth_order_coeff_fn(t, location_grid):
      del t, location_grid
      return a

    expected = (
        np.outer(_gaussian(ys, variance), _gaussian(xs, variance)) *
        np.exp(a * final_t))

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(_gaussian(ys, variance), _gaussian(xs, variance)),
            dtype=tf.float32),
        axis=0)
    bound_cond = [(_zero_boundary, _zero_boundary),
                  (_zero_boundary, _zero_boundary)]
    step_fn = douglas_adi_step(theta=0.5)
    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=step_fn,
        boundary_conditions=bound_cond,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn,
        dtype=grid[0].dtype)

    self._assertClose(expected, result)

  def testNoTimeDependence(self):
    """Test for the case where all terms (quadratic, linear, shift) are null."""
    grid = grids.uniform_grid(
        minimums=[-10, -20],
        maximums=[10, 20],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    time_step = 0.1
    final_t = 1
    variance = 1

    final_cond = np.outer(_gaussian(ys, variance), _gaussian(xs, variance))
    final_values = tf.expand_dims(tf.constant(final_cond, dtype=tf.float32),
                                  axis=0)
    bound_cond = [(_zero_boundary, _zero_boundary),
                  (_zero_boundary, _zero_boundary)]
    step_fn = douglas_adi_step(theta=0.5)
    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=step_fn,
        boundary_conditions=bound_cond,
        dtype=grid[0].dtype)
    expected = final_cond  # No time dependence.
    self._assertClose(expected, result)

  def testAnisotropicDiffusion_WithDirichletBoundaries(self):
    """Tests solving 2d diffusion equation with Dirichlet boundary conditions.

    The equation is `u_{t} + u_{xx} + 2 u_{yy} = 0`.
    The final condition is `u(t=1, x, y) = e * sin(x/sqrt(2)) * cos(y / 2)`.
    The following function satisfies this PDE and final condition:
    `u(t, x, y) = exp(t) * sin(x / sqrt(2)) * cos(y / 2)`.
    We impose Dirichlet boundary conditions using this function:
    `u(t, x_min, y) = exp(t) * sin(x_min / sqrt(2)) * cos(y / 2)`, etc.
    The other tests below are similar, but with other types of boundary
    conditions.
    """
    time_step = 0.01
    final_t = 1
    x_min = -20
    x_max = 20
    y_min = -10
    y_max = 10

    grid = grids.uniform_grid(
        minimums=[y_min, x_min],
        maximums=[y_max, x_max],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    def second_order_coeff_fn(t, location_grid):
      del t, location_grid
      return [[2, None], [None, 1]]

    @dirichlet
    def lower_bound_x(t, location_grid):
      del location_grid
      f = tf.exp(t) * np.sin(x_min / _SQRT2) * tf.sin(ys / 2)
      return tf.expand_dims(f, 0)

    @dirichlet
    def upper_bound_x(t, location_grid):
      del location_grid
      f = tf.exp(t) * np.sin(x_max / _SQRT2) * tf.sin(ys / 2)
      return tf.expand_dims(f, 0)

    @dirichlet
    def lower_bound_y(t, location_grid):
      del location_grid
      f = tf.exp(t) * tf.sin(xs / _SQRT2) * np.sin(y_min / 2)
      return tf.expand_dims(f, 0)

    @dirichlet
    def upper_bound_y(t, location_grid):
      del location_grid
      f = tf.exp(t) * tf.sin(xs / _SQRT2) * np.sin(y_max / 2)
      return tf.expand_dims(f, 0)

    expected = np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2))

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2)) * np.exp(final_t),
            dtype=tf.float32),
        axis=0)
    bound_cond = [(lower_bound_y, upper_bound_y),
                  (lower_bound_x, upper_bound_x)]
    step_fn = douglas_adi_step(theta=0.5)
    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=step_fn,
        boundary_conditions=bound_cond,
        second_order_coeff_fn=second_order_coeff_fn,
        dtype=grid[0].dtype)

    self._assertClose(expected, result)

  def testAnisotropicDiffusion_WithNeumannBoundaries(self):
    """Tests solving 2d diffusion equation with Neumann boundary conditions."""
    time_step = 0.01
    final_t = 1
    x_min = -20
    x_max = 20
    y_min = -10
    y_max = 10

    grid = grids.uniform_grid(
        minimums=[y_min, x_min],
        maximums=[y_max, x_max],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    def second_order_coeff_fn(t, location_grid):
      del t, location_grid
      return [[2, None], [None, 1]]

    @neumann
    def lower_bound_x(t, location_grid):
      del location_grid
      f = -tf.exp(t) * np.cos(x_min / _SQRT2) * tf.sin(ys / 2) / _SQRT2
      return tf.expand_dims(f, 0)

    @neumann
    def upper_bound_x(t, location_grid):
      del location_grid
      f = tf.exp(t) * np.cos(x_max / _SQRT2) * tf.sin(ys / 2) / _SQRT2
      return tf.expand_dims(f, 0)

    @neumann
    def lower_bound_y(t, location_grid):
      del location_grid
      f = -tf.exp(t) * tf.sin(xs / _SQRT2) * np.cos(y_min / 2) / 2
      return tf.expand_dims(f, 0)

    @neumann
    def upper_bound_y(t, location_grid):
      del location_grid
      f = tf.exp(t) * tf.sin(xs / _SQRT2) * np.cos(y_max / 2) / 2
      return tf.expand_dims(f, 0)

    expected = np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2))

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2)) * np.exp(final_t),
            dtype=tf.float32),
        axis=0)
    bound_cond = [(lower_bound_y, upper_bound_y),
                  (lower_bound_x, upper_bound_x)]
    step_fn = douglas_adi_step(theta=0.5)
    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=step_fn,
        boundary_conditions=bound_cond,
        second_order_coeff_fn=second_order_coeff_fn,
        dtype=grid[0].dtype)

    self._assertClose(expected, result)

  def testAnisotropicDiffusion_WithMixedBoundaries(self):
    """Tests solving 2d diffusion equation with mixed boundary conditions."""
    time_step = 0.01
    final_t = 1
    x_min = -20
    x_max = 20
    y_min = -10
    y_max = 10

    grid = grids.uniform_grid(
        minimums=[y_min, x_min],
        maximums=[y_max, x_max],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    def second_order_coeff_fn(t, location_grid):
      del t, location_grid
      return [[2, None], [None, 1]]

    @dirichlet
    def lower_bound_x(t, location_grid):
      del location_grid
      f = tf.exp(t) * np.sin(x_min / _SQRT2) * tf.sin(ys / 2)
      return tf.expand_dims(f, 0)

    @neumann
    def upper_bound_x(t, location_grid):
      del location_grid
      f = tf.exp(t) * np.cos(x_max / _SQRT2) * tf.sin(ys / 2) / _SQRT2
      return tf.expand_dims(f, 0)

    @neumann
    def lower_bound_y(t, location_grid):
      del location_grid
      f = -tf.exp(t) * tf.sin(xs / _SQRT2) * np.cos(y_min / 2) / 2
      return tf.expand_dims(f, 0)

    @dirichlet
    def upper_bound_y(t, location_grid):
      del location_grid
      f = tf.exp(t) * tf.sin(xs / _SQRT2) * np.sin(y_max / 2)
      return tf.expand_dims(f, 0)

    expected = np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2))

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2)) * np.exp(final_t),
            dtype=tf.float32),
        axis=0)
    bound_cond = [(lower_bound_y, upper_bound_y),
                  (lower_bound_x, upper_bound_x)]
    step_fn = douglas_adi_step(theta=0.5)
    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=step_fn,
        boundary_conditions=bound_cond,
        second_order_coeff_fn=second_order_coeff_fn,
        dtype=grid[0].dtype)

    self._assertClose(expected, result)

  def testAnisotropicDiffusion_WithRobinBoundaries(self):
    """Tests solving 2d diffusion equation with Robin boundary conditions."""
    time_step = 0.01
    final_t = 1
    x_min = -20
    x_max = 20
    y_min = -10
    y_max = 10

    grid = grids.uniform_grid(
        minimums=[y_min, x_min],
        maximums=[y_max, x_max],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    def second_order_coeff_fn(t, location_grid):
      del t, location_grid
      return [[2, None], [None, 1]]

    def lower_bound_x(t, location_grid):
      del location_grid
      f = tf.exp(t) * tf.sin(ys / 2) * (
          np.sin(x_min / _SQRT2) - np.cos(x_min / _SQRT2) / _SQRT2)
      return 1, 1, tf.expand_dims(f, 0)

    def upper_bound_x(t, location_grid):
      del location_grid
      f = tf.exp(t) * tf.sin(ys / 2) * (
          np.sin(x_max / _SQRT2) + 2 * np.cos(x_max / _SQRT2) / _SQRT2)
      return 1, 2, tf.expand_dims(f, 0)

    def lower_bound_y(t, location_grid):
      del location_grid
      f = tf.exp(t) * tf.sin(xs / _SQRT2) * (
          np.sin(y_min / 2) - 3 * np.cos(y_min / 2) / 2)
      return 1, 3, tf.expand_dims(f, 0)

    def upper_bound_y(t, location_grid):
      del location_grid
      f = tf.exp(t) * tf.sin(
          xs / _SQRT2) * (2 * np.sin(y_max / 2) + 3 * np.cos(y_max / 2) / 2)
      return 2, 3, tf.expand_dims(f, 0)

    expected = np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2))

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2)) * np.exp(final_t),
            dtype=tf.float32),
        axis=0)
    bound_cond = [(lower_bound_y, upper_bound_y),
                  (lower_bound_x, upper_bound_x)]
    step_fn = douglas_adi_step(theta=0.5)
    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=step_fn,
        boundary_conditions=bound_cond,
        second_order_coeff_fn=second_order_coeff_fn,
        dtype=grid[0].dtype)

    self._assertClose(expected, result)

  def _assertClose(self, expected, stepper_result):
    actual = self.evaluate(stepper_result[0])
    self.assertLess(np.max(np.abs(actual - expected)) / np.max(expected), 0.01)

  def testAnisotropicDiffusion_InForwardDirection(self):
    """Tests solving 2d diffusion equation in forward direction.

    The equation is `u_{t} - Dx u_{xx} - Dy u_{yy} = 0`.
    The initial condition is a gaussian centered at (0, 0) with variance sigma.
    The variance along each dimension should evolve as `sigma + 2 Dx (t - t_0)`
    and `sigma + 2 Dy (t - t_0)`.
    """
    grid = grids.uniform_grid(
        minimums=[-10, -20],
        maximums=[10, 20],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    diff_coeff_x = 0.4  # Dx
    diff_coeff_y = 0.25  # Dy
    time_step = 0.1
    final_t = 1.0
    initial_variance = 1

    def quadratic_coeff_fn(t, location_grid):
      del t, location_grid
      u_xx = -diff_coeff_x
      u_yy = -diff_coeff_y
      u_xy = None
      return [[u_yy, u_xy], [u_xy, u_xx]]

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(
                _gaussian(ys, initial_variance),
                _gaussian(xs, initial_variance)),
            dtype=tf.float32),
        axis=0)
    bound_cond = [(_zero_boundary, _zero_boundary),
                  (_zero_boundary, _zero_boundary)]
    step_fn = douglas_adi_step(theta=0.5)
    result = fd_solvers.solve_forward(
        start_time=0.0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=final_values,
        time_step=time_step,
        one_step_fn=step_fn,
        boundary_conditions=bound_cond,
        second_order_coeff_fn=quadratic_coeff_fn,
        dtype=grid[0].dtype)

    variance_x = initial_variance + 2 * diff_coeff_x * final_t
    variance_y = initial_variance + 2 * diff_coeff_y * final_t
    expected = np.outer(_gaussian(ys, variance_y), _gaussian(xs, variance_x))

    self._assertClose(expected, result)

  def testReferenceEquation(self):
    """Tests the equation used as reference for a few further tests.

    We solve the heat equation `u_t = u_xx + u_yy` on x = [0...1], y = [0...1]
    with boundary conditions `u(x, y, t=0) = (1/2 - |x-1/2|)(1/2-|y-1/2|), and
    zero Dirichlet on all spatial boundaries.

    The exact solution of the diffusion equation with zero-Dirichlet rectangular
    boundaries is `u(x, y, t) = u(x, t) * u(y, t)`,
    `u(z, t) = sum_{n=1..inf} b_n sin(pi n z) exp(-n^2 pi^2 t)`,
    `b_n = 2 integral_{0..1} sin(pi n z) u(z, t=0) dz.`

    The initial conditions are taken so that the integral easily calculates, and
    the sum can be approximated by a few first terms (given large enough `t`).
    See the result in _reference_heat_equation_solution.

    Using this solution helps to simplify the tests, as we don't have to
    maintain complicated boundary conditions in each test or tweak the
    parameters to keep the "support" of the function far from boundaries.
    """
    grid = grids.uniform_grid(
        minimums=[0, 0], maximums=[1, 1], sizes=[201, 301], dtype=tf.float32)
    ys, xs = grid

    final_t = 0.1
    time_step = 0.002

    def second_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return [[-1, None], [None, -1]]

    initial = _reference_2d_pde_initial_cond(xs, ys)
    expected = _reference_2d_pde_solution(xs, ys, final_t)
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

    Take the heat equation `v_{t} - v_{xx} - v_{yy} = 0` and substitute
    `v = exp(x + 2y) u`.
    This yields `u_{t} - u_{xx} - u_{yy} - 2u_{x} - 4u_{y} - 5u = 0`. The test
    compares numerical solution of this equation to the exact one, which is the
    diffusion equation solution times `exp(-x-2y)`.
    """
    grid = grids.uniform_grid(
        minimums=[0, 0], maximums=[1, 1], sizes=[201, 301], dtype=tf.float32)
    ys, xs = grid

    final_t = 0.1
    time_step = 0.002

    def second_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return [[-1, None], [None, -1]]

    def first_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return [-4, -2]

    def zeroth_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return -5

    exp = _dir_prod(tf.exp(-2 * ys), tf.exp(-xs))
    initial = exp * _reference_2d_pde_initial_cond(xs, ys)
    expected = exp * _reference_2d_pde_solution(xs, ys, final_t)

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

    As in previous test, take the diffusion equation
    `v_{t} - v_{xx} - v_{yy} = 0` and substitute `v = exp(x + 2y) u`, but this
    time keep exponent under the derivative:
    `u_{t} - exp(-x)[exp(x)u]_{xx} - exp(-2y)[exp(2y)u]_{yy} = 0`.
    Expect the same solution as in previous test.
    """
    grid = grids.uniform_grid(
        minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)
    ys, xs = grid

    final_t = 0.1
    time_step = 0.002

    def second_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [[-tf.exp(-2 * y), None], [None, -tf.exp(-x)]]

    def inner_second_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [[tf.exp(2 * y), None], [None, tf.exp(x)]]

    exp = _dir_prod(tf.exp(-2 * ys), tf.exp(-xs))
    initial = exp * _reference_2d_pde_initial_cond(xs, ys)
    expected = exp * _reference_2d_pde_solution(xs, ys, final_t)

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

    We saw previously that the solution of
    `u_{t} - u_{xx} - u_{yy} - 2u_{x} - 4u_{y} - 5u = 0` is
    `u = exp(-x-2y) v`, where `v` solves the diffusion equation. Substitute now
    `u = exp(-x-2y) v` without expanding the derivatives:
    `v_{t} - exp(x)[exp(-x)v]_{xx} - exp(2y)[exp(-2y)v]_{yy} -
      2exp(x)[exp(-x)v]_{x} - 4exp(2y)[exp(-2y)v]_{y} - 5v = 0`.
    Solve this equation and expect the solution of the diffusion equation.
    """
    grid = grids.uniform_grid(
        minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)
    ys, xs = grid

    final_t = 0.1
    time_step = 0.002

    def second_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [[-tf.exp(2 * y), None], [None, -tf.exp(x)]]

    def inner_second_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [[tf.exp(-2 * y), None], [None, tf.exp(-x)]]

    def first_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [-4 * tf.exp(2 * y), -2 * tf.exp(x)]

    def inner_first_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [tf.exp(-2 * y), tf.exp(-x)]

    def zeroth_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return -5

    initial = _reference_2d_pde_initial_cond(xs, ys)
    expected = _reference_2d_pde_solution(xs, ys, final_t)

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

  def testReferenceEquation_WithTransformationYieldingMixedTerm(self):
    """Tests an equation with mixed terms against exact solution.

    Take the reference equation `v_{t} = v_{xx} + v_{yy}` and substitute
    `v(x, y, t) = u(x, 2y - x, t)`. This yields
    `u_{t} = u_{xx} + 5u_{zz} - 2u_{xz}`, where `z = 2y - x`.
    Having `u(x, z, t) = v(x, (x+z)/2, t)` where `v(x, y, t)` is the known
    solution of the reference equation, we derive the boundary conditions
    and the expected solution for `u(x, y, t)`.
    """
    grid = grids.uniform_grid(
        minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)

    final_t = 0.1
    time_step = 0.002

    def second_order_coeff_fn(t, coord_grid):
      del t, coord_grid
      return [[-5, 1], [None, -1]]

    @dirichlet
    def boundary_lower_z(t, coord_grid):
      x = coord_grid[1]
      return _reference_pde_solution(x, t) * _reference_pde_solution(x / 2, t)

    @dirichlet
    def boundary_upper_z(t, coord_grid):
      x = coord_grid[1]
      return _reference_pde_solution(x, t) * _reference_pde_solution(
          (x + 1) / 2, t)

    z_mesh, x_mesh = tf.meshgrid(grid[0], grid[1], indexing='ij')
    initial = (
        _reference_pde_initial_cond(x_mesh) * _reference_pde_initial_cond(
            (x_mesh + z_mesh) / 2))
    expected = (
        _reference_pde_solution(x_mesh, final_t) * _reference_pde_solution(
            (x_mesh + z_mesh) / 2, final_t))

    actual = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial,
        time_step=time_step,
        second_order_coeff_fn=second_order_coeff_fn,
        boundary_conditions=[(boundary_lower_z, boundary_upper_z),
                             (_zero_boundary, _zero_boundary)])[0]

    self.assertAllClose(expected, actual, atol=1e-3, rtol=1e-3)

  def testInnerMixedSecondOrderCoeffs(self):
    """Tests handling coefficients under the mixed second derivative.

    Take the equation from the previous test,
    `u_{t} = u_{xx} + 5u_{zz} - 2u_{xz}` and substitute `u = exp(xz) w`,
    leaving the exponent under the derivatives:
    `w_{t} = exp(-xz) [exp(xz) u]_{xx} + 5 exp(-xz) [exp(xz) u]_{zz}
    - 2 exp(-xz) [exp(xz) u]_{xz}`.
    We now have a coefficient under the mixed derivative. Test that the solution
    is `w = exp(-xz) u`, where u is from the previous test.
    """
    grid = grids.uniform_grid(
        minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)

    final_t = 0.1
    time_step = 0.002

    def second_order_coeff_fn(t, coord_grid):
      del t,
      z, x = tf.meshgrid(*coord_grid, indexing='ij')
      exp = tf.math.exp(-z * x)
      return [[-5 * exp, exp], [None, -exp]]

    def inner_second_order_coeff_fn(t, coord_grid):
      del t,
      z, x = tf.meshgrid(*coord_grid, indexing='ij')
      exp = tf.math.exp(z * x)
      return [[exp, exp], [None, exp]]

    @dirichlet
    def boundary_lower_z(t, coord_grid):
      x = coord_grid[1]
      return _reference_pde_solution(x, t) * _reference_pde_solution(x / 2, t)

    @dirichlet
    def boundary_upper_z(t, coord_grid):
      x = coord_grid[1]
      return tf.exp(-x) * _reference_pde_solution(
          x, t) * _reference_pde_solution((x + 1) / 2, t)

    z, x = tf.meshgrid(*grid, indexing='ij')
    exp = tf.math.exp(-z * x)
    initial = exp * (
        _reference_pde_initial_cond(x) * _reference_pde_initial_cond(
            (x + z) / 2))
    expected = exp * (
        _reference_pde_solution(x, final_t) * _reference_pde_solution(
            (x + z) / 2, final_t))

    actual = fd_solvers.solve_forward(
        start_time=0,
        end_time=final_t,
        coord_grid=grid,
        values_grid=initial,
        time_step=time_step,
        second_order_coeff_fn=second_order_coeff_fn,
        inner_second_order_coeff_fn=inner_second_order_coeff_fn,
        boundary_conditions=[(boundary_lower_z, boundary_upper_z),
                             (_zero_boundary, _zero_boundary)])[0]

    self.assertAllClose(expected, actual, atol=1e-3, rtol=1e-3)

  def testCompareExpandedAndNotExpandedPdes(self):
    """Tests comparing PDEs with expanded derivatives and without.

    The equation is
    `u_{t} + [x u]_{x} + [y^2 u]_{y} - [sin(x) u]_{xx} - [cos(y) u]_yy
     + [x^3 y^2 u]_{xy} = 0`.
    Solve the equation, expand the derivatives and solve the equation again.
    Expect the results to be equal.
    """
    grid = grids.uniform_grid(
        minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)

    final_t = 0.1
    time_step = 0.002
    y, x = grid

    initial = _reference_2d_pde_initial_cond(x, y)  # arbitrary

    def inner_second_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [[-tf.math.cos(y), x**3 * y**2 / 2], [None, -tf.math.sin(x)]]

    def inner_first_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [y**2, x]

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
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [[-tf.math.cos(y), x**3 * y**2 / 2], [None, -tf.math.sin(x)]]

    def first_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return [y**2 * (1 + 3 * x**2) + 2 * tf.math.sin(y),
              x * (1 + 2 * x**2 * y) - 2 * tf.math.cos(x)]

    def zeroth_order_coeff_fn(t, coord_grid):
      del t
      y, x = tf.meshgrid(*coord_grid, indexing='ij')
      return 1 + 2 * y + tf.math.sin(x) + tf.math.cos(x) + 6 * x**2 * y

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


def _gaussian(xs, variance):
  return np.exp(-np.square(xs) / (2 * variance)) / np.sqrt(2 * np.pi * variance)


@dirichlet
def _zero_boundary(t, locations):
  del t, locations
  return 0.0


def _reference_2d_pde_initial_cond(xs, ys):
  """Initial conditions for the reference 2d diffusion equation."""
  return _dir_prod(
      _reference_pde_initial_cond(ys), _reference_pde_initial_cond(xs))


def _reference_2d_pde_solution(xs, ys, t, num_terms=5):
  return _dir_prod(
      _reference_pde_solution(ys, t, num_terms),
      _reference_pde_solution(xs, t, num_terms))


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


def _dir_prod(a, b):
  """Calculates the direct product of two Tensors."""
  return tf.tensordot(a, b, ([], []))


if __name__ == '__main__':
  tf.test.main()
