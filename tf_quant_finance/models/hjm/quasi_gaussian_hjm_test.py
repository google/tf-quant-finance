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

"""Tests for HJM module."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class HJMModelTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self.mean_reversion_1_factor = [0.03]
    self.volatility_1_factor = [0.01]
    self.mean_reversion_batch_1_factor = [[0.03], [0.04]]
    self.mean_reversion_batch_2_factor = [[0.03, 0.03], [0.04, 0.04]]
    self.volatility_batch_1_factor = [[0.01], [0.015]]
    self.volatility_batch_2_factor = [[0.005, 0.006], [0.004, 0.008]]
    self.mean_reversion_4_factor = [0.03, 0.02, 0.01, 0.005]
    self.volatility_4_factor = [0.01, 0.011, 0.015, 0.008]
    self.volatility_time_dep_1_factor = [0.01, 0.02, 0.01]
    self.instant_forward_rate = lambda *args: [0.01]
    def _instant_forward_rate_batch(t):
      ones = tf.transpose(tf.expand_dims(tf.ones_like(t), axis=0))
      return tf.transpose(tf.constant([0.01, 0.02], dtype=t.dtype) * ones)
    self.instant_forward_rate_batch = _instant_forward_rate_batch

    self.initial_state = [0.01, 0.01]
    self.initial_state_batch = [[[0.01]], [[0.02]]]
    # See D. Brigo, F. Mercurio. Interest Rate Models. 2007.
    def _true_mean(t, mr, vol, istate, f_0_t):
      dtype = np.float64
      a = dtype(mr)
      sigma = dtype(vol)
      initial_state = dtype(istate)
      return (dtype(f_0_t)
              + (sigma * sigma / 2 / a**2)
              * (1.0 - np.exp(-a * t))**2
              - f_0_t * np.exp(-a * t)
              + initial_state *  np.exp(-a * t))
    self.true_mean = _true_mean
    def _true_var(t, mr, vol):
      dtype = np.float64
      a = dtype(mr)
      sigma = dtype(vol)
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

    super(HJMModelTest, self).setUp()

  @parameterized.named_parameters({
      'testcase_name': 'no_xla_time_step',
      'use_xla': False,
      'time_step': 0.1,
      'num_time_steps': None,
  }, {
      'testcase_name': 'xla',
      'use_xla': True,
      'time_step': 0.1,
      'num_time_steps': None,
  }, {
      'testcase_name': 'no_xla_num_time_steps',
      'use_xla': False,
      'time_step': None,
      'num_time_steps': 11,
  }, {
      'testcase_name': 'xla_num_time_steps',
      'use_xla': True,
      'time_step': None,
      'num_time_steps': 11,
  })
  def test_mean_and_variance_1d(self, use_xla, time_step, num_time_steps):
    """Tests 1-Factor model with constant parameters."""
    dtype = tf.float64
    # exact discretization is not supported for time-dependent specification
    # of mean reversion rate.
    process = tff.models.hjm.QuasiGaussianHJM(
        dim=1,
        mean_reversion=self.mean_reversion_1_factor,
        volatility=self.volatility_1_factor,
        initial_discount_rate_fn=self.instant_forward_rate,
        dtype=dtype)

    num_samples = 10_000

    def _fn():
      paths, _, _, _ = process.sample_paths(
          [0.1, 0.5, 1.0],
          num_samples=num_samples,
          time_step=time_step,
          num_time_steps=num_time_steps,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[1, 2],
          skip=1000000)
      return paths

    if use_xla:
      paths = self.evaluate(tf.function(_fn, experimental_compile=True)())
    else:
      paths = self.evaluate(_fn())
    self.assertAllEqual(paths.shape, [num_samples, 3])
    paths = paths[:, -1]  # Extract paths values for the terminal time
    mean = np.mean(paths, axis=0)
    variance = np.var(paths, axis=0)
    self.assertAllClose(mean, self.true_mean(
        1.0, self.mean_reversion_1_factor, self.volatility_1_factor,
        self.initial_state, self.instant_forward_rate(1.0))[0],
                        rtol=1e-4, atol=1e-4)
    self.assertAllClose(variance, self.true_var(
        1.0, self.mean_reversion_1_factor, self.volatility_1_factor)[0],
                        rtol=1e-4, atol=1e-4)

  def test_zcb_variance_1_factor(self):
    """Tests 1-Factor model with constant parameters."""
    num_samples = 10_000
    for dtype in [tf.float64]:
      curve_times = np.array([0., 0.5, 1.0, 5.0, 10.0])
      times = np.array([0.1, 0.5, 1.0, 3])
      process = tff.models.hjm.QuasiGaussianHJM(
          dim=1,
          mean_reversion=self.mean_reversion_1_factor,
          volatility=self.volatility_1_factor,
          initial_discount_rate_fn=self.instant_forward_rate,
          dtype=dtype)
      # generate zero coupon paths
      paths, _, _ = process.sample_discount_curve_paths(
          times,
          curve_times=curve_times,
          num_samples=num_samples,
          time_step=0.1,
          random_type=tff.math.random.RandomType.SOBOL,
          seed=[1, 2],
          skip=1000000)
      self.assertEqual(paths.dtype, dtype)
      paths = self.evaluate(paths)
      self.assertAllEqual(paths.shape, [num_samples, 5, 4])
      sampled_std = tf.math.reduce_std(tf.math.log(paths), axis=0)
      for tidx in range(4):
        true_std = self.true_zcb_std(times[tidx], curve_times + times[tidx],
                                     self.volatility_1_factor[0],
                                     self.mean_reversion_1_factor[0])
        self.assertAllClose(
            sampled_std[:, tidx], true_std, rtol=1e-3, atol=1e-3)

  @parameterized.named_parameters({
      'testcase_name': '1d',
      'dim': 1,
      'corr_matrix': None,
      'dtype': None,
  }, {
      'testcase_name': '2d',
      'dim': 2,
      'corr_matrix': None,
      'dtype': tf.float32,
  }, {
      'testcase_name': '2d_with_corr',
      'dim': 2,
      'corr_matrix': [[[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.7], [0.7, 1.0]]],
      'dtype': tf.float64,
  })
  def test_mean_and_variance_batch(self, dim, corr_matrix, dtype):
    """Tests batch of 1-Factor model with constant parameters."""
    if dim == 1:
      mr = self.mean_reversion_batch_1_factor
      vol = self.volatility_batch_1_factor
    else:
      mr = self.mean_reversion_batch_2_factor
      vol = self.volatility_batch_2_factor

    process = tff.models.hjm.QuasiGaussianHJM(
        dim=dim,
        mean_reversion=mr,
        volatility=vol,
        initial_discount_rate_fn=self.instant_forward_rate_batch,
        corr_matrix=corr_matrix,
        dtype=dtype)

    paths, _, _, _ = process.sample_paths(
        [0.1, 0.5, 1.0],
        num_samples=20000,
        time_step=0.1,
        num_time_steps=None,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        skip=1000000)

    paths = self.evaluate(paths)
    self.assertAllEqual(paths.shape, [2, 20000, 3])
    paths = paths[:, :, -1]  # Extract paths values for the terminal time
    mean = np.mean(paths, axis=-1)
    variance = np.var(paths, axis=-1)
    f_0_t = [0.01, 0.02]
    for i in range(2):
      if dim == 1:
        eff_vol = vol[i][0]
      else:
        if corr_matrix is None:
          c = 0.0
        else:
          c = corr_matrix[i][1][0]
        eff_vol = np.sqrt(vol[i][0]**2 + vol[i][1]**2 + 2*c*vol[i][0]*vol[i][1])
      with self.subTest('CloseMean'):
        self.assertAllClose(
            mean[i], self.true_mean(
                1.0, mr[i][0], eff_vol, self.initial_state_batch[i][0][0],
                f_0_t[i]), rtol=1e-4, atol=1e-4)
      with self.subTest('CloseStd'):
        self.assertAllClose(
            variance[i], self.true_var(1.0, mr[i][0], eff_vol),
            rtol=1e-4, atol=1e-4)

  def test_zcb_variance_batch_1_factor(self):
    """Tests batch of 1-Factor model with constant parameters."""
    num_samples = 100000
    for dtype in [tf.float64]:
      curve_times = np.array([0., 0.5, 1.0, 5.0, 10.0])
      times = np.array([0.1, 0.5, 1.0, 3])
      process = tff.models.hjm.QuasiGaussianHJM(
          dim=1,
          mean_reversion=self.mean_reversion_batch_1_factor,
          volatility=self.volatility_batch_1_factor,
          initial_discount_rate_fn=self.instant_forward_rate_batch,
          dtype=dtype)
      # generate zero coupon paths
      paths, _, _ = process.sample_discount_curve_paths(
          times,
          curve_times=curve_times,
          num_samples=num_samples,
          time_step=0.1,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[1, 2],
          skip=1000000)
      self.assertEqual(paths.dtype, dtype)
      paths = self.evaluate(paths)
      self.assertAllEqual(paths.shape, [2, num_samples, 5, 4])
      sampled_std = tf.math.reduce_std(tf.math.log(paths), axis=1)
      for i in range(2):
        for tidx in range(4):
          true_std = self.true_zcb_std(times[tidx], curve_times + times[tidx],
                                       self.volatility_batch_1_factor[i][0],
                                       self.mean_reversion_batch_1_factor[i][0])
          with self.subTest('Batch_{}_time_index{}'.format(i, tidx)):
            self.assertAllClose(
                sampled_std[i, :, tidx], true_std, rtol=5e-4, atol=5e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'float32',
          'dtype': np.float32,
      }, {
          'testcase_name': 'float64',
          'dtype': np.float64,
      })
  def test_time_dependent_1d(self, dtype):
    """Tests 1-factor model with time dependent vol."""
    num_samples = 100000
    def discount_fn(x):
      return 0.01 * tf.ones_like(x, dtype=dtype)  # pylint: disable=cell-var-from-loop
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        [0.1, 2.0], values=self.volatility_time_dep_1_factor, dtype=dtype)
    def _vol_fn(t, r):
      del r
      return volatility([t])
    process = tff.models.hjm.QuasiGaussianHJM(
        dim=1,
        mean_reversion=self.mean_reversion_1_factor,
        volatility=_vol_fn,
        initial_discount_rate_fn=discount_fn,
        dtype=dtype)
    times = np.array([0.1, 1.0, 2.0, 3.0])
    paths, _, _, _ = process.sample_paths(
        times,
        num_samples=num_samples,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        skip=1000000)
    self.assertEqual(paths.dtype, dtype)
    paths = self.evaluate(paths)
    self.assertAllEqual(paths.shape, [num_samples, 4])
    r_std = np.squeeze(np.std(paths, axis=0))
    expected_std = self.true_std_time_dep(
        times, np.array([0.0, 0.1, 2.0]),
        np.array(self.volatility_time_dep_1_factor),
        self.mean_reversion_1_factor[0])
    self.assertAllClose(r_std, expected_std, rtol=1.75e-4, atol=1.75e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'float64',
          'dtype': np.float64,
      })
  def test_state_dependent_vol_1_factor(self, dtype):
    """Tests 1-factor model with state dependent vol."""
    num_samples = 100000
    def discount_fn(x):
      return 0.01 * tf.ones_like(x, dtype=dtype)  # pylint: disable=cell-var-from-loop
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        [], values=self.volatility_1_factor, dtype=dtype)
    def _vol_fn(t, r):
      return volatility([t]) * tf.math.abs(r)**0.5
    process = tff.models.hjm.QuasiGaussianHJM(
        dim=1,
        mean_reversion=self.mean_reversion_1_factor,
        volatility=_vol_fn,
        initial_discount_rate_fn=discount_fn,
        dtype=dtype)
    times = np.array([0.1, 1.0, 2.0, 3.0])
    _, discount_paths, _, _ = process.sample_paths(
        times,
        num_samples=num_samples,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        skip=1000000)
    self.assertEqual(discount_paths.dtype, dtype)
    discount_paths = self.evaluate(discount_paths)
    self.assertAllEqual(discount_paths.shape, [num_samples, 4])
    discount_mean = np.mean(discount_paths, axis=0)
    expected_mean = np.exp(-0.01 * times)
    self.assertAllClose(discount_mean, expected_mean, rtol=2e-4, atol=2e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'float64',
          'dtype': np.float64,
      })
  def test_correctness_4_factor(self, dtype):
    """Tests 4-factor model with constant vol."""
    num_samples = 100000
    def discount_fn(x):
      return 0.01 * tf.ones_like(x, dtype=dtype)  # pylint: disable=cell-var-from-loop

    process = tff.models.hjm.QuasiGaussianHJM(
        dim=4,
        mean_reversion=self.mean_reversion_4_factor,
        volatility=self.volatility_4_factor,
        initial_discount_rate_fn=discount_fn,
        dtype=dtype)
    times = np.array([0.1, 1.0, 2.0, 3.0])
    _, discount_paths, _, _ = process.sample_paths(
        times,
        num_samples=num_samples,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        skip=1000000)
    self.assertEqual(discount_paths.dtype, dtype)
    discount_paths = self.evaluate(discount_paths)
    self.assertAllEqual(discount_paths.shape, [num_samples, 4])
    discount_mean = np.mean(discount_paths, axis=0)
    expected_mean = np.exp(-0.01 * times)
    self.assertAllClose(discount_mean, expected_mean, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'float64',
          'dtype': np.float64,
      })
  def test_correctness_2_factor_with_correlation(self, dtype):
    """Tests 2-factor correlated model with constant vol."""
    num_samples = 100000
    def discount_fn(x):
      return 0.01 * tf.ones_like(x, dtype=dtype)  # pylint: disable=cell-var-from-loop

    process = tff.models.hjm.QuasiGaussianHJM(
        dim=2,
        mean_reversion=self.mean_reversion_4_factor[:2],
        volatility=self.volatility_4_factor[:2],
        corr_matrix=[[1.0, 0.5], [0.5, 1.0]],
        initial_discount_rate_fn=discount_fn,
        dtype=dtype)
    times = np.array([0.1, 1.0, 2.0, 3.0])
    _, discount_paths, _, _ = process.sample_paths(
        times,
        num_samples=num_samples,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        skip=1000000)
    self.assertEqual(discount_paths.dtype, dtype)
    discount_paths = self.evaluate(discount_paths)
    self.assertAllEqual(discount_paths.shape, [num_samples, 4])
    discount_mean = np.mean(discount_paths, axis=0)
    expected_mean = np.exp(-0.01 * times)
    self.assertAllClose(discount_mean, expected_mean, rtol=1e-4, atol=1e-4)

  def test_zcb_variance_2_factor(self):
    """Tests ZCB for sims 2-Factor correlated model."""
    num_samples = 100000
    for dtype in [tf.float64]:
      curve_times = np.array([0., 0.5, 1.0, 2.0, 5.0])
      times = np.array([0.1, 0.5, 1.0, 3])
      process = tff.models.hjm.QuasiGaussianHJM(
          dim=2,
          mean_reversion=[0.03, 0.03],
          volatility=[0.005, 0.005],
          corr_matrix=[[1.0, 0.5], [0.5, 1.0]],
          initial_discount_rate_fn=self.instant_forward_rate,
          dtype=dtype)
      # generate zero coupon paths
      paths, _, _ = process.sample_discount_curve_paths(
          times,
          curve_times=curve_times,
          num_samples=num_samples,
          time_step=0.1,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[1, 2],
          skip=1000000)
      self.assertEqual(paths.dtype, dtype)
      paths = self.evaluate(paths)
      self.assertAllEqual(paths.shape, [num_samples, 5, 4])
      sampled_std = tf.math.reduce_std(tf.math.log(paths), axis=0)
      for tidx in range(4):
        true_std = self.true_zcb_std(times[tidx], curve_times + times[tidx],
                                     0.005, 0.03)
        self.assertAllClose(
            sampled_std[:, tidx], np.sqrt(3) * true_std, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
