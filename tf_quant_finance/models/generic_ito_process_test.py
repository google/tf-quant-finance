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
"""Tests for `sample_paths` of `ItoProcess`."""

from unittest import mock  # pylint: disable=g-importing-member
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

GenericItoProcess = tff.models.GenericItoProcess
grids = tff.math.pde.grids


@test_util.run_all_in_graph_and_eager_modes
class GenericItoProcessTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "no_xla",
      "use_xla": False,
  }, {
      "testcase_name": "xla",
      "use_xla": True,
  })
  def test_sample_paths_wiener(self, use_xla):
    """Tests paths properties for Wiener process (dX = dW)."""

    def drift_fn(_, x):
      return tf.zeros_like(x)

    def vol_fn(_, x):
      return tf.expand_dims(tf.ones_like(x), -1)

    process = GenericItoProcess(dim=1, drift_fn=drift_fn, volatility_fn=vol_fn)
    times = np.array([0.1, 0.2, 0.3])
    num_samples = 10000

    @tf.function
    def fn():
      return process.sample_paths(
          times=times, num_samples=num_samples, seed=42, time_step=0.01)

    if use_xla:
      paths = self.evaluate(tf.xla.experimental.compile(fn))[0]
    else:
      paths = self.evaluate(fn())

    means = np.mean(paths, axis=0).reshape([-1])
    covars = np.cov(paths.reshape([num_samples, -1]), rowvar=False)
    expected_means = np.zeros((3,))
    expected_covars = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))
    with self.subTest(name="Means"):
      self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)
    with self.subTest(name="Covar"):
      self.assertAllClose(covars, expected_covars, rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters({
      "testcase_name": "NoGridNoDraws",
      "use_time_grid": False,
      "supply_normal_draws": False,
  }, {
      "testcase_name": "WithGridWithDraws",
      "use_time_grid": True,
      "supply_normal_draws": True,
  })
  def test_sample_paths_2d(self, use_time_grid, supply_normal_draws):
    """Tests path properties for 2-dimentional Ito process.

    We construct the following Ito processes.

    dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2
    dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2

    mu_1, mu_2 are constants.
    s_ij = a_ij t + b_ij

    For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.

    Args:
      use_time_grid: A boolean to indicate whther `times_grid` is supplied.
      supply_normal_draws: A boolean to indicate whther `normal_draws` is
        supplied.
    """
    dtype = tf.float64

    mu = np.array([0.2, 0.7])
    a = np.array([[0.4, 0.1], [0.3, 0.2]])
    b = np.array([[0.33, -0.03], [0.21, 0.5]])

    def drift_fn(t, x):
      return mu * tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)

    def vol_fn(t, x):
      del x
      return (a * t + b) * tf.ones([2, 2], dtype=t.dtype)

    process = GenericItoProcess(dim=2, drift_fn=drift_fn, volatility_fn=vol_fn)
    times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
    x0 = np.array([0.1, -1.1])
    if use_time_grid:
      times_grid = tf.linspace(tf.constant(0.0, dtype=dtype), 0.55, 56)
      time_step = None
    else:
      times_grid = None
      time_step = 0.01
    if supply_normal_draws:
      num_samples = 1
      # Use antithetic sampling
      normal_draws = tf.random.normal(
          shape=[5000, times_grid.shape[0] - 1, 2],
          dtype=dtype)
      normal_draws = tf.concat([normal_draws, -normal_draws], axis=0)
    else:
      num_samples = 10000
      normal_draws = None
    paths = self.evaluate(
        process.sample_paths(
            times,
            num_samples=num_samples,
            initial_state=x0,
            time_step=time_step,
            times_grid=times_grid,
            normal_draws=normal_draws,
            seed=12134))

    # The correct number of samples
    num_samples = 10000
    self.assertAllClose(paths.shape, (num_samples, 5, 2), atol=0)
    means = np.mean(paths, axis=0)
    times = np.reshape(times, [-1, 1])
    expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)
    self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters({
      "testcase_name": "1DBatch",
      "batch_rank": 1,
  }, {
      "testcase_name": "2DBatch",
      "batch_rank": 2,
  })
  def test_batch_sample_paths_2d(self, batch_rank):
    """Tests path properties for a batch of 2-dimentional Ito process.

    We construct the following Ito processes.

    dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2
    dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2

    mu_1, mu_2 are constants.
    s_ij = a_ij t + b_ij

    For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.

    Args:
      batch_rank: The rank of the batch of processes being simulated.
    """
    dtype = tf.float64

    mu = np.array([0.2, 0.7])
    a = np.array([[0.4, 0.1], [0.3, 0.2]])
    b = np.array([[0.33, -0.03], [0.21, 0.5]])

    def drift_fn(t, x):
      return mu * tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)

    def vol_fn(t, x):
      del x
      return (a * t + b) * tf.ones([2]*batch_rank + [1, 2, 2], dtype=t.dtype)

    process = GenericItoProcess(dim=2, drift_fn=drift_fn, volatility_fn=vol_fn,
                                dtype=dtype)
    times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
    x0 = np.array([0.1, -1.1]) * np.ones([2]*batch_rank + [1, 2])

    times_grid = None
    time_step = 0.01

    num_samples = 10000
    normal_draws = None
    paths = self.evaluate(
        process.sample_paths(
            times,
            num_samples=num_samples,
            initial_state=x0,
            time_step=time_step,
            times_grid=times_grid,
            normal_draws=normal_draws,
            seed=12134))

    # The correct number of samples
    num_samples = 10000
    self.assertAllClose(list(paths.shape),
                        [2]*batch_rank + [num_samples, 5, 2], atol=0)
    means = np.mean(paths, axis=batch_rank)
    times = np.reshape(times, [1]*batch_rank + [-1, 1])
    expected_means = np.reshape(
        x0, [2]*batch_rank + [1, 2]) + (2.0 / 3.0) * mu * np.power(times, 1.5)
    self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)

  def test_sample_paths_dtypes(self):
    """Sampled paths have the expected dtypes."""
    for dtype in [np.float32, np.float64]:
      drift_fn = lambda t, x: tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)
      vol_fn = lambda t, x: t * tf.ones([1, 1], dtype=t.dtype)
      process = GenericItoProcess(
          dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=dtype)

      paths = self.evaluate(
          process.sample_paths(
              times=[0.1, 0.2],
              num_samples=10,
              initial_state=[0.1],
              time_step=0.01,
              seed=123))
      self.assertEqual(paths.dtype, dtype)

  # Several tests below are unit tests for GenericItoProcess.fd_solver_backward:
  # they mock out the pde solver and check only the conversion of SDE to PDE,
  # but not PDE solving. There are also integration tests further below.
  def test_backward_pde_coeffs_with_constant_params_1d(self):
    vol = 2
    drift = 1
    discounting = 3

    # batch_shape = (1, ), dim = 1
    # vol_fn(...).shape = batch_shape + (dim, dim) = (1, 1, 1)
    vol_fn = lambda t, x: tf.constant([[[vol]]], dtype=tf.float32)
    # drift_fn(...).shape = batch_shape + (dim,) = (1, 1)
    drift_fn = lambda t, x: tf.constant([[drift]], dtype=tf.float32)
    # discounting_fn(...).shape = batch_shape = (1, )
    discounting_fn = lambda t, x: tf.constant([discounting], dtype=tf.float32)

    process = GenericItoProcess(
        dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=tf.float32)

    pde_solver_fn = mock.Mock()
    coord_grid = [tf.constant([0])]
    process.fd_solver_backward(
        start_time=0,
        end_time=0,
        coord_grid=coord_grid,
        values_grid=tf.constant([0]),
        discounting=discounting_fn,
        pde_solver_fn=pde_solver_fn)
    kwargs = pde_solver_fn.call_args[1]
    second_order_coeff = self.evaluate(kwargs["second_order_coeff_fn"](
        0, coord_grid))
    self.assertAllClose(np.array([[[vol**2 / 2]]]), second_order_coeff)
    first_order_coeff = self.evaluate(kwargs["first_order_coeff_fn"](
        0, coord_grid))
    self.assertAllClose(np.array([[drift]]), first_order_coeff)
    zeroth_order_coeff = self.evaluate(kwargs["zeroth_order_coeff_fn"](
        0, coord_grid))
    self.assertAllClose(np.array([-discounting]), zeroth_order_coeff)

  def test_backward_pde_coeffs_with_nonconstant_params_1d(self):
    # vol = 2 * x**2
    # drift = 3 * x
    # discounting = 6 / x
    # x = [1, 2, 3]

    # batch_shape = (3, ), dim = 1
    # vol_fn(...).shape = batch_shape + (dim, dim) = (3, 1, 1)
    def vol_fn(t, x):
      del t
      x = x[:, 0]  # x.shape was (3, 1), now it's (3, )
      return tf.reshape(2 * x**2, (-1, 1, 1))

    # drift_fn(...).shape = batch_shape + (dim,) = (3, 1)
    def drift_fn(t, x):
      del t
      x = x[:, 0]
      return tf.reshape(3 * x, (-1, 1))

    # discounting_fn(...).shape = batch_shape = (3, )
    def discounting_fn(t, x):
      del t
      x = x[:, 0]
      return 6 / x

    coord_grid = [tf.constant([1, 2, 3])]

    process = GenericItoProcess(
        dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=tf.float32)

    pde_solver_fn = mock.Mock()
    process.fd_solver_backward(
        start_time=0,
        end_time=0,
        coord_grid=coord_grid,
        values_grid=tf.constant([0]),
        discounting=discounting_fn,
        pde_solver_fn=pde_solver_fn)
    kwargs = pde_solver_fn.call_args[1]
    second_order_coeff = self.evaluate(kwargs["second_order_coeff_fn"](
        0, coord_grid))
    self.assertAllClose(
        np.array([[[2, 8**2 / 2, 18**2 / 2]]]), second_order_coeff)
    first_order_coeff = self.evaluate(kwargs["first_order_coeff_fn"](
        0, coord_grid))
    self.assertAllClose(np.array([[3, 6, 9]]), first_order_coeff)
    zeroth_order_coeff = self.evaluate(kwargs["zeroth_order_coeff_fn"](
        0, coord_grid))
    self.assertAllClose(np.array([-6, -3, -2]), zeroth_order_coeff)

  def test_backward_pde_coeffs_with_constant_params_2d(self):
    vol = [[1, 2], [3, 4]]
    drift = [1, 2]
    discounting = 3

    # batch_shape = (1, ), dim = 2
    # vol_fn(...).shape = batch_shape + (dim, dim) = (1, 2, 2)
    def vol_fn(t, x):
      del t, x
      return tf.expand_dims(tf.constant(vol, dtype=tf.float32), 0)

    # drift_fn(...).shape = batch_shape + (dim,) = (1, 2)
    def drift_fn(t, x):
      del t, x
      return tf.expand_dims(tf.constant(drift, dtype=tf.float32), 0)
    # discounting_fn(...).shape = batch_shape = (1, )
    discounting_fn = lambda t, x: tf.constant([discounting], dtype=tf.float32)

    process = GenericItoProcess(
        dim=2, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=tf.float32)

    pde_solver_fn = mock.Mock()
    coord_grid = [tf.constant([0])]
    process.fd_solver_backward(
        start_time=0,
        end_time=0,
        coord_grid=coord_grid,
        values_grid=tf.constant([0]),
        discounting=discounting_fn,
        pde_solver_fn=pde_solver_fn)
    kwargs = pde_solver_fn.call_args[1]
    second_order_coeff = self.evaluate(kwargs["second_order_coeff_fn"](
        0, coord_grid))

    # second_order_coeff.shape = (dim, dim) + grid_shape = (2, 2, 1)
    self.assertAllClose(
        np.array([[[2.5], [5.5]], [[5.5], [12.5]]]), second_order_coeff)

    # first_order_coeff.shape = (dim) + grid_shape = (2, 1)
    first_order_coeff = self.evaluate(kwargs["first_order_coeff_fn"](
        0, coord_grid))
    self.assertAllClose(np.array([[1], [2]]), first_order_coeff)

    # first_order_coeff.shape = grid_shape = (1)
    zeroth_order_coeff = self.evaluate(kwargs["zeroth_order_coeff_fn"](
        0, coord_grid))
    self.assertAllClose(np.array([-3]), zeroth_order_coeff)

  def test_backward_pde_coeffs_with_nonconstant_params_2d(self):
    # vol = [[x * y, 2 * x], [x + y, 3]]
    # drift = [x + y, x - y]
    # discounting = 3 * x * y
    # x = [0, 1, 2], y = [-1, 0, 1]

    # batch_shape = (3, 3 ), dim = 2
    # vol_fn(...).shape = batch_shape + (dim, dim) = (3, 3, 2, 2)
    def vol_fn(t, grid):
      del t
      self.assertEqual((3, 3, 2), grid.shape)
      ys = grid[..., 0]
      xs = grid[..., 1]
      vol_yy = xs * ys
      vol_yx = 2 * xs
      vol_xy = xs + ys
      vol_xx = 3 * tf.ones_like(xs)
      vol = tf.stack((tf.stack(
          (vol_yy, vol_xy), axis=-1), tf.stack((vol_yx, vol_xx), axis=-1)),
                     axis=-1)
      self.assertEqual((3, 3, 2, 2), vol.shape)
      return vol

    # drift_fn(...).shape = batch_shape + (dim, ) = (3, 3, 2)
    def drift_fn(t, grid):
      del t
      self.assertEqual((3, 3, 2), grid.shape)
      ys = grid[..., 0]
      xs = grid[..., 1]
      drift_y = xs + ys
      drift_x = xs - ys
      drift = tf.stack((drift_y, drift_x), axis=-1)
      self.assertEqual((3, 3, 2), drift.shape)
      return drift

    # drift_fn(...).shape = batch_shape = (3, 3)
    def discounting_fn(t, grid):
      del t
      self.assertEqual((3, 3, 2), grid.shape)
      ys = grid[..., 0]
      xs = grid[..., 1]
      return 3 * xs * ys

    process = GenericItoProcess(
        dim=2, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=tf.float32)

    pde_solver_fn = mock.Mock()
    coord_grid = [tf.constant([-1, 0, 1]), tf.constant([0, 1, 2])]
    process.fd_solver_backward(
        start_time=0,
        end_time=0,
        coord_grid=coord_grid,
        values_grid=tf.zeros((3, 3)),
        discounting=discounting_fn,
        pde_solver_fn=pde_solver_fn)
    kwargs = pde_solver_fn.call_args[1]
    second_order_coeff = self.evaluate(kwargs["second_order_coeff_fn"](
        0, coord_grid))

    expected_coeff_yy = np.array([[0, 2.5, 10], [0, 2, 8], [0, 2.5, 10]])
    self.assertAllClose(expected_coeff_yy, second_order_coeff[0][0])

    expected_coeff_xy = np.array([[0, 3, 5], [0, 3, 6], [0, 4, 9]])
    self.assertAllClose(expected_coeff_xy, second_order_coeff[0][1])
    self.assertAllClose(expected_coeff_xy, second_order_coeff[1][0])

    expected_coeff_xx = np.array([[5, 4.5, 5], [4.5, 5, 6.5], [5, 6.5, 9]])
    self.assertAllClose(expected_coeff_xx, second_order_coeff[1][1])

    # first_order_coeff.shape = (dim) + grid_shape = (2, 1)
    first_order_coeff = self.evaluate(kwargs["first_order_coeff_fn"](
        0, coord_grid))

    expected_coeff_y = np.array([[-1, 0, 1], [0, 1, 2], [1, 2, 3]])
    self.assertAllClose(expected_coeff_y, first_order_coeff[0])
    expected_coeff_x = np.array([[1, 2, 3], [0, 1, 2], [-1, 0, 1]])
    self.assertAllClose(expected_coeff_x, first_order_coeff[1])

    # first_order_coeff.shape = grid_shape = (1)
    zeroth_order_coeff = self.evaluate(kwargs["zeroth_order_coeff_fn"](
        0, coord_grid))
    self.assertAllClose(
        np.array([[0, 3, 6], [0, 0, 0], [0, -3, -6]]), zeroth_order_coeff)

  def test_solving_backward_pde_for_sde_with_const_coeffs(self):
    # Integration test for converting 2d SDE with constant coeffs to a
    # backward Kolmogorov PDE and solving it.
    # The SDE is:
    # dS_x = (dW_1 + dW_2) / sqrt(2)
    # dS_y = (dW_1 + dW_2) / sqrt(2)
    # It is of course trivial, but we'll solve it the hard way for the sake of
    # testing.
    # The Kolmogorov backwards PDE is:
    # u_{t} + D u_{xx} / 2 +  D u_{yy} / 2 + D u_{xy} = 0
    # The equation can be rewritten as `u_{t} + D u_{zz} = 0`, where
    # z = (x + y) / sqrt(2).
    #  If the final condition is a gaussian centered at (0, 0) with variance
    #  sigma, then the solution is:
    # `u(x, y, t) = gaussian((x + y)/sqrt(2), sigma + 2D(t_final - t)) *
    # gaussian((x - y)/sqrt(2), sigma)`.

    def vol_fn(t, grid):
      del t
      xs = grid[..., 1]
      vol_elem = tf.ones_like(xs) / np.sqrt(2)  # all 4 elements are equal.
      return tf.stack((tf.stack(
          (vol_elem, vol_elem), axis=-1), tf.stack(
              (vol_elem, vol_elem), axis=-1)),
                      axis=-1)

    drift_fn = lambda t, grid: tf.zeros(grid.shape)

    process = GenericItoProcess(
        dim=2, volatility_fn=vol_fn, drift_fn=drift_fn, dtype=tf.float32)

    grid = grids.uniform_grid(
        minimums=[-10, -20],
        maximums=[10, 20],
        sizes=[201, 301],
        dtype=tf.float32)
    ys = self.evaluate(grid[0])
    xs = self.evaluate(grid[1])

    diff_coeff = 1
    time_step = 0.1
    final_t = 3
    final_variance = 1
    variance_along_diagonal = final_variance + 2 * diff_coeff * final_t

    def expected_fn(x, y):
      return (_gaussian(
          (x + y) / np.sqrt(2), variance_along_diagonal) * _gaussian(
              (x - y) / np.sqrt(2), final_variance))

    expected = np.array([[expected_fn(x, y) for x in xs] for y in ys])

    final_values = tf.expand_dims(
        tf.constant(
            np.outer(
                _gaussian(ys, final_variance), _gaussian(xs, final_variance)),
            dtype=tf.float32),
        axis=0)

    result = self.evaluate(
        process.fd_solver_backward(
            start_time=final_t,
            end_time=0,
            coord_grid=grid,
            values_grid=final_values,
            time_step=time_step,
            dtype=tf.float32)[0])

    self.assertLess(np.max(np.abs(result - expected)) / np.max(expected), 0.01)


def _gaussian(xs, variance):
  return np.exp(-np.square(xs) / (2 * variance)) / np.sqrt(2 * np.pi * variance)


if __name__ == "__main__":
  tf.test.main()
