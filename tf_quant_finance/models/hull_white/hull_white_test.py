# Lint as: python3
# Copyright 2020 Google LLC
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

"""Tests for Hull White Module."""

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class HullWhiteTest(tf.test.TestCase):

  def setUp(self):
    self.mean_reversion = [0.1, 0.05]
    self.volatility = [0.1, 0.2]
    self.instant_forward_rate_1d_fn = lambda *args: [0.01]
    self.instant_forward_rate_2d_fn = lambda *args: [0.01, 0.01]
    self.initial_state = [0.1, 0.2]
    # See D. Brigo, F. Mercurio. Interest Rate Models. 2007.
    def _true_mean(t):
      dtype = np.float64
      a = dtype(self.mean_reversion)
      sigma = dtype(self.volatility)
      initial_state = dtype(self.initial_state)
      return (dtype(self.instant_forward_rate_2d_fn(t))
              + (sigma * sigma / 2 / a**2)
              * (1.0 - np.exp(-a * t))**2
              - self.instant_forward_rate_2d_fn(0) * np.exp(-a * t)
              + initial_state *  np.exp(-a * t))
    self.true_mean = _true_mean
    def _true_var(t):
      dtype = np.float64
      a = dtype(self.mean_reversion)
      sigma = dtype(self.volatility)
      return (sigma * sigma / 2 / a) * (1.0 - np.exp(-2 * a * t))
    self.true_var = _true_var
    super(HullWhiteTest, self).setUp()

  def test_mean_and_variance_1d(self):
    """Tests model with piecewise constant parameters in 1 dimension."""
    for dtype in [tf.float32, tf.float64]:
      mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
          [0.1, 2.0], values=3 * [self.mean_reversion[0]], dtype=dtype)
      volatility = tff.math.piecewise.PiecewiseConstantFunc(
          [0.1, 2.0], values=3 * [self.volatility[0]], dtype=dtype)
      process = tff.models.hull_white.HullWhiteModel1F(
          mean_reversion=mean_reversion,
          volatility=volatility,
          instant_forward_rate_fn=self.instant_forward_rate_1d_fn,
          dtype=dtype)
      paths = process.sample_paths(
          [0.1, 0.5, 1.0],
          num_samples=50000,
          initial_state=self.initial_state[0],
          random_type=tff.math.random.RandomType.HALTON,
          skip=1000000)
      self.assertEqual(paths.dtype, dtype)
      self.assertAllEqual(paths.shape, [50000, 3, 1])
      paths = self.evaluate(paths)
      paths = paths[:, -1, :]  # Extract paths values for the terminal time
      mean = np.mean(paths, axis=0)
      variance = np.var(paths, axis=0)
      self.assertAllClose(mean, [self.true_mean(1.0)[0]], rtol=1e-4, atol=1e-4)
      self.assertAllClose(variance,
                          [self.true_var(1.0)[0]], rtol=1e-4, atol=1e-4)

  def test_mean_variance_correlation_piecewise_2d(self):
    """Tests model with piecewise constant parameters in 2 dimensions."""
    for dtype in [tf.float32, tf.float64]:
      # Mean reversion without batch dimesnion
      mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
          [0.1, 2.0], values=3 * [self.mean_reversion], dtype=dtype)
      # Volatility with batch dimesnion
      volatility = tff.math.piecewise.PiecewiseConstantFunc(
          [[0.1, 0.2, 0.5], [0.1, 2.0, 3.0]],
          values=[4 * [self.volatility[0]],
                  4 * [self.volatility[1]]], dtype=dtype)
      expected_corr_matrix = [[1., 0.5], [0.5, 1.]]
      corr_matrix = tff.math.piecewise.PiecewiseConstantFunc(
          [0.5, 1.0], values=3 * [expected_corr_matrix], dtype=dtype)
      process = tff.models.hull_white.VectorHullWhiteModel(
          dim=2,
          mean_reversion=mean_reversion,
          volatility=volatility,
          corr_matrix=corr_matrix,
          instant_forward_rate_fn=self.instant_forward_rate_2d_fn,
          dtype=dtype)
      paths = process.sample_paths(
          [0.1, 0.5, 1.0],
          num_samples=50000,
          initial_state=self.initial_state,
          random_type=tff.math.random.RandomType.SOBOL,
          skip=1000000)
      self.assertEqual(paths.dtype, dtype)
      self.assertAllEqual(paths.shape, [50000, 3, 2])
      paths = self.evaluate(paths)
      paths = paths[:, -1, :]  # Extract paths values for the terminal time
      mean = np.mean(paths, axis=0)
      estimated_corr_matrix = np.corrcoef(paths[:, 0], paths[:, 1])
      variance = np.var(paths, axis=0)
      self.assertAllClose(mean, self.true_mean(1.0), rtol=1e-4, atol=1e-4)
      self.assertAllClose(variance,
                          self.true_var(1.0), rtol=1e-4, atol=1e-4)
      self.assertAllClose(estimated_corr_matrix, expected_corr_matrix,
                          rtol=1e-4, atol=1e-4)

  def test_mean_variance_correlation_constant_piecewise_2d(self):
    """Tests model with piecewise constant or constant parameters in 2."""
    for dtype in [tf.float32, tf.float64]:
      tf.random.set_seed(10)  # Fix global random seed
      mean_reversion = self.mean_reversion
      volatility = tff.math.piecewise.PiecewiseConstantFunc(
          [0.2, 1.0], values=3 * [self.volatility], dtype=dtype)
      expected_corr_matrix = [[1., 0.5], [0.5, 1.]]
      corr_matrix = tff.math.piecewise.PiecewiseConstantFunc(
          [0.5, 2.0], values=[expected_corr_matrix,
                              [[1., 0.6], [0.6, 1.]],
                              [[1., 0.9], [0.9, 1.]]], dtype=dtype)
      process = tff.models.hull_white.VectorHullWhiteModel(
          dim=2,
          mean_reversion=mean_reversion,
          volatility=volatility,
          corr_matrix=corr_matrix,
          instant_forward_rate_fn=self.instant_forward_rate_2d_fn,
          dtype=dtype)
      paths = process.sample_paths(
          [0.1, 0.5, 1.0],
          num_samples=500000,
          initial_state=self.initial_state,
          random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
          seed=42)
      self.assertEqual(paths.dtype, dtype)
      self.assertAllEqual(paths.shape, [500000, 3, 2])
      paths = self.evaluate(paths)
      estimated_corr_matrix = np.corrcoef(paths[:, 1, 0], paths[:, 1, 1])
      paths = paths[:, -1, :]  # Extract paths values for the terminal time
      mean = np.mean(paths, axis=0)
      variance = np.var(paths, axis=0)
      self.assertAllClose(mean, self.true_mean(1.0), rtol=1e-3, atol=1e-3)
      self.assertAllClose(variance,
                          self.true_var(1.0), rtol=1e-3, atol=1e-3)
      self.assertAllClose(estimated_corr_matrix, expected_corr_matrix,
                          rtol=1e-2, atol=1e-2)

  def test_mean_variance_correlation_generic_2d(self):
    """Tests model with generic parameters in 2 dimensions."""
    for dtype in [tf.float32, tf.float64]:
      # Mean reversion without batch dimesnion
      mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
          [0.1, 2.0], values=3 * [self.mean_reversion], dtype=dtype)
      # Volatility with batch dimesnion
      volatility = tff.math.piecewise.PiecewiseConstantFunc(
          [[0.1, 0.2, 0.5], [0.1, 2.0, 3.0]],
          values=[4 * [self.volatility[0]],
                  4 * [self.volatility[1]]], dtype=dtype)
      def corr_matrix(t):
        one = tf.ones_like(t)
        row1 = tf.stack([one, 0.5 * t], axis=-1)
        row2 = tf.reverse(row1, [0])
        corr_matrix = tf.stack([row1, row2], axis=-1)
        return corr_matrix
      process = tff.models.hull_white.VectorHullWhiteModel(
          dim=2,
          mean_reversion=mean_reversion,
          volatility=volatility,
          corr_matrix=corr_matrix,
          instant_forward_rate_fn=self.instant_forward_rate_2d_fn,
          dtype=dtype)
      times = [0.1, 0.5]
      paths = process.sample_paths(
          times,
          num_samples=50000,
          initial_state=self.initial_state,
          random_type=tff.math.random.RandomType.SOBOL,
          skip=100000,
          time_step=0.01)
      self.assertEqual(paths.dtype, dtype)
      self.assertAllEqual(paths.shape, [50000, 2, 2])
      paths = self.evaluate(paths)
      paths = paths[:, -1, :]  # Extract paths values for the terminal time
      mean = np.mean(paths, axis=0)
      variance = np.var(paths, axis=0)
      self.assertAllClose(mean, self.true_mean(times[-1]), rtol=1e-3, atol=1e-3)
      self.assertAllClose(variance,
                          self.true_var(times[-1]), rtol=1e-3, atol=1e-3)

  def test_invalid_batch_size_piecewise(self):
    """Tests that the batch dimension should be [2] if it is not empty."""
    dtype = tf.float64
    # Batch shape is [1]. Should be [] or [2]
    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        [[0.1, 2.0]], values=[3 * [self.mean_reversion]], dtype=dtype)
    # Volatility with batch dimesnion
    volatility = self.volatility
    with self.assertRaises(ValueError):
      tff.models.hull_white.VectorHullWhiteModel(
          dim=2,
          mean_reversion=mean_reversion,
          volatility=volatility,
          corr_matrix=None,
          instant_forward_rate_fn=self.instant_forward_rate_2d_fn,
          dtype=dtype)

  def test_invalid_batch_rank_piecewise(self):
    """Tests that the batch rank should be 1 if it is not empty."""
    dtype = tf.float64
    # Batch rank is 2
    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        [[[0.1, 2.0]]], values=[[3 * [self.mean_reversion]]], dtype=dtype)
    # Volatility with batch dimesnion
    volatility = self.volatility
    with self.assertRaises(ValueError):
      tff.models.hull_white.VectorHullWhiteModel(
          dim=2,
          mean_reversion=mean_reversion,
          volatility=volatility,
          corr_matrix=None,
          instant_forward_rate_fn=self.instant_forward_rate_2d_fn,
          dtype=dtype)

  def test_time_step_not_supplied(self):
    """Tests that the `time_step` should be supplied if Euler scheme is used."""
    dtype = tf.float64
    def volatility_fn(t):
      del t
      return self.volatility
    process = tff.models.hull_white.VectorHullWhiteModel(
        dim=2,
        mean_reversion=self.mean_reversion,
        volatility=volatility_fn,
        instant_forward_rate_fn=self.instant_forward_rate_2d_fn,
        dtype=dtype)
    with self.assertRaises(ValueError):
      process.sample_paths(
          [0.1, 2.0],
          num_samples=100,
          initial_state=self.initial_state)

  def test_times_wrong_rank(self):
    """Tests that the `times` should be a rank 1 `Tensor`."""
    dtype = tf.float64
    process = tff.models.hull_white.VectorHullWhiteModel(
        dim=2,
        mean_reversion=self.mean_reversion,
        volatility=self.volatility,
        instant_forward_rate_fn=self.instant_forward_rate_2d_fn,
        dtype=dtype)
    with self.assertRaises(ValueError):
      process.sample_paths(
          [[0.1, 2.0]],
          num_samples=100,
          initial_state=self.initial_state)

if __name__ == '__main__':
  tf.test.main()
