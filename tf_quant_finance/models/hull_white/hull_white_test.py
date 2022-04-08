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

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class HullWhiteTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self.mean_reversion = [0.1, 0.05]
    self.volatility = [0.01, 0.02]
    self.volatility_time_dep_1d = [0.01, 0.02, 0.01]
    def instant_forward_rate_1d_fn(t):
      return 0.01 * tf.ones_like(t)
    self.instant_forward_rate_1d_fn = instant_forward_rate_1d_fn
    def instant_forward_rate_2d_fn(t):
      return 0.01 * tf.ones(t.shape.as_list() + [2], dtype=t.dtype)
    self.instant_forward_rate_2d_fn = instant_forward_rate_2d_fn
    self.initial_state = [0.01, 0.01]
    # See D. Brigo, F. Mercurio. Interest Rate Models. 2007.
    def _true_mean(t):
      dtype = np.float64
      a = dtype(self.mean_reversion)
      sigma = dtype(self.volatility)
      initial_state = dtype(self.initial_state)
      return (0.01
              + (sigma * sigma / 2 / a**2)
              * (1.0 - np.exp(-a * t))**2
              - 0.01 * np.exp(-a * t)
              + initial_state *  np.exp(-a * t))
    self.true_mean = _true_mean
    def _true_var(t):
      dtype = np.float64
      a = dtype(self.mean_reversion)
      sigma = dtype(self.volatility)
      return (sigma * sigma / 2 / a) * (1.0 - np.exp(-2 * a * t))
    self.true_var = _true_var

    def _true_std_time_dep(t, intervals, vol, k):
      res = np.zeros_like(t, dtype=np.float64)
      for i, tt in enumerate(t):
        var = 0.0
        for j in range(len(intervals) - 1):
          if tt >= intervals[j] and tt < intervals[j + 1]:
            var = var + vol[j]**2 / 2 / k * (
                np.exp(2 * k * tt) - np.exp(2 * k * intervals[j]))
            break
          else:
            var = var + vol[j]**2 / 2 / k * (
                np.exp(2 * k * intervals[j + 1]) - np.exp(2 * k * intervals[j]))
        else:
          var = var + vol[-1]**2/2/k *(np.exp(2*k*tt)-np.exp(2*k*intervals[-1]))
        res[i] = np.exp(-k*tt) * np.sqrt(var)

      return res
    self.true_std_time_dep = _true_std_time_dep

    def _true_zcb_std(t, tau, v, k):
      e_tau = np.exp(-k*tau)
      et = np.exp(k*t)
      val = v/k * (1. - e_tau*et) * np.sqrt((1.-1./et/et)/k/2)
      return val
    self.true_zcb_std = _true_zcb_std

    super(HullWhiteTest, self).setUp()

  @parameterized.named_parameters(
      {
          'testcase_name': 'STATELESS',
          'random_type': tff.math.random.RandomType.STATELESS_ANTITHETIC,
          'seed': [1, 2],
          'dtype': None,
      }, {
          'testcase_name': 'HALTON',
          'random_type': tff.math.random.RandomType.HALTON,
          'seed': None,
          'dtype': tf.float64,
      })
  def test_mean_and_variance_1d(self, random_type, seed, dtype):
    """Tests model with piecewise constant parameters in 1 dimension."""
    # exact discretization is not supported for time-dependent specification
    # of mean reversion rate.
    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        [], values=[self.mean_reversion[0]], dtype=dtype)
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        [0.1, 2.0], values=3 * [self.volatility[0]], dtype=dtype)
    process = tff.models.hull_white.HullWhiteModel1F(
        mean_reversion=mean_reversion,
        volatility=volatility,
        initial_discount_rate_fn=self.instant_forward_rate_1d_fn,
        dtype=dtype)
    paths = process.sample_paths(
        [0.1, 0.5, 1.0],
        num_samples=50000,
        random_type=random_type,
        seed=seed,
        skip=1000000)
    if dtype is not None:
      with self.subTest('Dytpe'):
        self.assertEqual(paths.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(paths.shape, [50000, 3, 1])
    paths = self.evaluate(paths)
    paths = paths[:, -1, :]  # Extract paths values for the terminal time
    mean = np.mean(paths, axis=0)
    variance = np.var(paths, axis=0)
    with self.subTest('Mean'):
      self.assertAllClose(mean, [self.true_mean(1.0)[0]], rtol=1e-4, atol=1e-4)
    with self.subTest('Variance'):
      self.assertAllClose(variance,
                          [self.true_var(1.0)[0]], rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'STATELESS_ANTITHETIC',
          'random_type': tff.math.random.RandomType.STATELESS_ANTITHETIC,
          'seed': [4, 2],
      }, {
          'testcase_name': 'HALTON',
          'random_type': tff.math.random.RandomType.HALTON,
          'seed': None,
      })
  def test_variance_zcb_1d(self, random_type, seed):
    """Tests discount bond variance in 1 dimension."""
    dtype = tf.float64
    def discount_fn(x):
      return 0.01 * tf.expand_dims(tf.ones_like(x), axis=-1)
    mr = 0.03
    vol = 0.015
    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        [], values=[mr], dtype=dtype)
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        [0.1, 2.0], values=3 * [vol], dtype=dtype)
    process = tff.models.hull_white.HullWhiteModel1F(
        mean_reversion=mean_reversion,
        volatility=volatility,
        initial_discount_rate_fn=discount_fn,
        dtype=dtype)
    curve_times = np.array([0.0, 1.0, 2.0, 3.0])
    paths, _ = process.sample_discount_curve_paths(
        [0.1, 1.0, 2.0],
        curve_times,
        num_samples=500000,
        random_type=random_type,
        seed=seed,
        skip=1000000)
    with self.subTest('Dtype'):
      self.assertEqual(paths.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(paths.shape, [500000, 4, 3, 1])
    paths = self.evaluate(tf.math.log(paths))
    std_zcb = np.std(paths, axis=0)[:, 2, 0]
    expected_std = self.true_zcb_std(2.0, 2.0 + curve_times, vol, mr)
    with self.subTest('VarianceAbsTol'):
      self.assertAllClose(std_zcb, expected_std, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'STATELESS',
          'random_type': tff.math.random.RandomType.STATELESS_ANTITHETIC,
          'seed': [1, 2],
      }, {
          'testcase_name': 'HALTON',
          'random_type': tff.math.random.RandomType.HALTON,
          'seed': None,
      })
  def test_time_dependent_1d(self, random_type, seed):
    """Tests model with time dependent vol in 1 dimension."""
    for dtype in [tf.float32, tf.float64]:
      def discount_fn(x):
        return 0.01 * tf.ones_like(x, dtype=dtype)  # pylint: disable=cell-var-from-loop
      volatility = tff.math.piecewise.PiecewiseConstantFunc(
          [0.1, 2.0], values=self.volatility_time_dep_1d, dtype=dtype)
      process = tff.models.hull_white.HullWhiteModel1F(
          mean_reversion=self.mean_reversion[0],
          volatility=volatility,
          initial_discount_rate_fn=discount_fn,
          dtype=dtype)
      times = np.array([0.1, 1.0, 2.0, 3.0])
      paths = process.sample_paths(
          times,
          num_samples=500000,
          random_type=random_type,
          seed=seed,
          skip=1000000)
      self.assertEqual(paths.dtype, dtype)
      self.assertAllEqual(paths.shape, [500000, 4, 1])
      paths = self.evaluate(paths)
      r_std = np.squeeze(np.std(paths, axis=0))
      expected_std = self.true_std_time_dep(
          times, np.array([0.0, 0.1, 2.0]),
          np.array(self.volatility_time_dep_1d), 0.1)
      self.assertAllClose(r_std, expected_std, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters({
      'testcase_name': 'no_xla',
      'use_xla': False,
  }, {
      'testcase_name': 'xla',
      'use_xla': True,
  })
  def test_mean_variance_correlation_piecewise_2d(self, use_xla):
    """Tests model with piecewise constant parameters in 2 dimensions."""
    dtype = np.float64
    # Mean reversion without batch dimesnion
    mean_reversion = self.mean_reversion
    # Volatility with batch dimesnion
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        [[0.1, 0.2, 0.5], [0.1, 2.0, 3.0]],
        values=[4 * [self.volatility[0]],
                4 * [self.volatility[1]]], dtype=dtype)
    expected_corr_matrix = [[1., 0.5], [0.5, 1.]]
    num_samples = 50000
    process = tff.models.hull_white.VectorHullWhiteModel(
        dim=2,
        mean_reversion=mean_reversion,
        volatility=volatility,
        corr_matrix=expected_corr_matrix,
        initial_discount_rate_fn=self.instant_forward_rate_2d_fn,
        dtype=dtype)
    @tf.function
    def fn():
      return process.sample_paths(
          [0.1, 0.5, 1.0],
          num_samples=num_samples,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[1, 2])
    if use_xla:
      paths = self.evaluate(tf.function(fn, jit_compile=True)())
    else:
      paths = self.evaluate(fn())
    with self.subTest('Dtype'):
      self.assertEqual(paths.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(paths.shape, [num_samples, 3, 2])
    paths = paths[:, -1, :]  # Extract paths values for the terminal time
    mean = np.mean(paths, axis=0)
    estimated_corr_matrix = np.corrcoef(paths[:, 0], paths[:, 1])
    variance = np.var(paths, axis=0)
    with self.subTest('Mean'):
      self.assertAllClose(mean, self.true_mean(1.0), rtol=1e-4, atol=1e-4)
    with self.subTest('Variance'):
      self.assertAllClose(variance,
                          self.true_var(1.0), rtol=1e-4, atol=1e-4)
    with self.subTest('CorrMatrix'):
      self.assertAllClose(estimated_corr_matrix, expected_corr_matrix,
                          rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters({
      'testcase_name': 'GenerateDraws',
      'supply_draws': False,
      'supply_grid': False,
  }, {
      'testcase_name': 'SupplyDraws',
      'supply_draws': True,
      'supply_grid': False,
  }, {
      'testcase_name': 'SupplyDrawsSupplyGrid',
      'supply_draws': True,
      'supply_grid': True,
  })
  def test_mean_variance_correlation_piecewise_constant_2d(
      self, supply_draws, supply_grid):
    """Tests model with piecewise constant or constant parameters in 2 dim."""
    dtype = tf.float64
    mean_reversion = self.mean_reversion
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        2 * [[0.2, 1.0]], values=np.array(3 * [self.volatility]).transpose(),
        dtype=dtype)
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
        initial_discount_rate_fn=self.instant_forward_rate_2d_fn,
        dtype=dtype)
    num_samples = 100000
    test_num_samples = 100000
    if supply_draws:
      normal_draws = tf.random.stateless_normal(
          [num_samples // 2, 9, 2], seed=[4, 2], dtype=dtype)
      normal_draws = tf.concat([normal_draws, -normal_draws], axis=0)
      test_num_samples = 1
    else:
      normal_draws = None
    times_grid = None
    if supply_grid:
      times_grid = [0.0, 0.1, 0.2, 0.5, 1.0]
      if supply_draws:
        normal_draws = tf.random.stateless_normal(
            [num_samples // 2, 4, 2], seed=[4, 2], dtype=dtype)
        normal_draws = tf.concat([normal_draws, -normal_draws], axis=0)
    paths = process.sample_paths(
        [0.1, 0.5, 1.0],
        num_samples=test_num_samples,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[4, 2],
        normal_draws=normal_draws,
        times_grid=times_grid)
    with self.subTest('Dtype'):
      self.assertEqual(paths.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(paths.shape, [num_samples, 3, 2])
    paths = self.evaluate(paths)
    estimated_corr_matrix = np.corrcoef(paths[:, 1, 0], paths[:, 1, 1])
    paths = paths[:, -1, :]  # Extract paths values for the terminal time
    mean = np.mean(paths, axis=0)
    variance = np.var(paths, axis=0)
    with self.subTest('Mean'):
      self.assertAllClose(mean, self.true_mean(1.0), rtol=1e-3, atol=1e-3)
    with self.subTest('Variance'):
      self.assertAllClose(variance,
                          self.true_var(1.0), rtol=1e-3, atol=1e-3)
    with self.subTest('CorrMatrix'):
      self.assertAllClose(estimated_corr_matrix, expected_corr_matrix,
                          rtol=1e-2, atol=1e-2)

  def test_mean_variance_correlation_generic_2d(self):
    """Tests model with generic parameters in 2 dimensions."""
    for dtype in [tf.float32, tf.float64]:
      mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
          2 * [[0.1, 2.0]], values=[3 * [self.mean_reversion[0]],
                                    3 * [self.mean_reversion[0]]], dtype=dtype)
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
          initial_discount_rate_fn=self.instant_forward_rate_2d_fn,
          dtype=dtype)
      times = [0.1, 0.5]
      paths = process.sample_paths(
          times,
          num_samples=50000,
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
    # Batch shape is [1]. Should be [2]
    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        [[0.1, 2.0]], values=[3 * [self.mean_reversion[0]]], dtype=dtype)
    # Volatility with batch dimesnion
    volatility = self.volatility
    with self.assertRaises(ValueError):
      tff.models.hull_white.VectorHullWhiteModel(
          dim=2,
          mean_reversion=mean_reversion,
          volatility=volatility,
          corr_matrix=None,
          initial_discount_rate_fn=self.instant_forward_rate_2d_fn,
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
          initial_discount_rate_fn=self.instant_forward_rate_2d_fn,
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
        initial_discount_rate_fn=self.instant_forward_rate_2d_fn,
        dtype=dtype)
    with self.assertRaises(ValueError):
      process.sample_paths(
          [0.1, 2.0],
          num_samples=100)

  def test_times_wrong_rank(self):
    """Tests that the `times` should be a rank 1 `Tensor`."""
    dtype = tf.float64
    process = tff.models.hull_white.VectorHullWhiteModel(
        dim=2,
        mean_reversion=self.mean_reversion,
        volatility=self.volatility,
        initial_discount_rate_fn=self.instant_forward_rate_2d_fn,
        dtype=dtype)
    with self.assertRaises(ValueError):
      process.sample_paths(
          [[0.1, 2.0]],
          num_samples=100)

  def test_time_dependent_mr(self):
    """Tests that time depemdent mr uses generic sampling."""
    dtype = tf.float64
    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        [1.0], values=[0.03, 0.04], dtype=dtype)

    process = tff.models.hull_white.VectorHullWhiteModel(
        dim=1,
        mean_reversion=mean_reversion,
        volatility=[0.015],
        corr_matrix=None,
        initial_discount_rate_fn=self.instant_forward_rate_1d_fn,
        dtype=dtype)
    self.assertTrue(process._sample_with_generic)

  def test_discount_bond_price_fn(self):
    """Tests implementation of P(t,T)|r(t)."""
    dtype = tf.float64
    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        [], values=[self.mean_reversion[0]], dtype=dtype)
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        [], values=[self.volatility[0]], dtype=dtype)
    process = tff.models.hull_white.HullWhiteModel1F(
        mean_reversion=mean_reversion,
        volatility=volatility,
        initial_discount_rate_fn=self.instant_forward_rate_1d_fn,
        dtype=dtype)
    bond_prices = process.discount_bond_price(
        [[0.011], [0.01]],
        [1.0, 2.0],
        [2.0, 3.5])
    self.assertEqual(bond_prices.dtype, dtype)
    self.assertAllEqual(bond_prices.shape, [2, 1])
    bond_prices = self.evaluate(bond_prices)
    expected = [0.98906753, 0.98495442]
    self.assertAllClose(np.squeeze(bond_prices), expected, atol=1e-12)


if __name__ == '__main__':
  tf.test.main()
