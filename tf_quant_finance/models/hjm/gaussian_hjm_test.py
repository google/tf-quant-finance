# Copyright 2021 Google LLC
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

"""Tests for Gaussian HJM module."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class GaussianHJMModelTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self.instant_forward_rate = lambda *args: [0.01]
    # See D. Brigo, F. Mercurio. Interest Rate Models. 2007.
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

    super(GaussianHJMModelTest, self).setUp()

  @parameterized.named_parameters(
      {
          'testcase_name': '1f_constant',
          'dim': 1,
          'mr': [0.03],
          'vol': [0.01],
          'corr': None,
          'vol_jumps': None,
          'vol_values': None,
          'num_time_steps': None,
          'dtype': tf.float32,
      },
      {
          'testcase_name': '1f_constant_num_time_steps',
          'dim': 1,
          'mr': [0.03],
          'vol': [0.01],
          'corr': None,
          'vol_jumps': None,
          'vol_values': None,
          'num_time_steps': 21,
          'dtype': tf.float64,
      },
      {
          'testcase_name': '1f_time_dep',
          'dim': 1,
          'mr': [0.03],
          'vol': None,
          'corr': None,
          'vol_jumps': [[0.5, 1.0]],
          'vol_values': [[0.01, 0.02, 0.01]],
          'num_time_steps': None,
          'dtype': None,
      },
      {
          'testcase_name': '2f_constant',
          'dim': 2,
          'mr': [0.03, 0.1],
          'vol': [0.005, 0.012],
          'corr': None,
          'vol_jumps': None,
          'vol_values': None,
          'num_time_steps': None,
          'dtype': tf.float64,
      },
      {
          'testcase_name': '2f_constant_with_corr',
          'dim': 2,
          'mr': [0.03, 0.1],
          'vol': [0.005, 0.012],
          'corr': [[1.0, 0.5], [0.5, 1.0]],
          'vol_jumps': None,
          'vol_values': None,
          'num_time_steps': None,
          'dtype': tf.float64,
      },
      {
          'testcase_name': '2f_time_dep',
          'dim': 2,
          'mr': [0.03, 0.1],
          'vol': None,
          'corr': None,
          'vol_jumps': [[0.5, 1.0], [0.5, 1.0]],
          'vol_values': [[0.005, 0.008, 0.005], [0.005, 0.008, 0.005]],
          'num_time_steps': None,
          'dtype': tf.float64,
      }
      )
  def test_correctness_rate_df_sims(self, dim, mr, vol, corr, vol_jumps,
                                    vol_values, num_time_steps, dtype):
    """Tests short rate and discount factor simulations."""
    if vol is None:
      vol = tff.math.piecewise.PiecewiseConstantFunc(vol_jumps, vol_values,
                                                     dtype=dtype)
    time_step = None if num_time_steps else 0.1
    num_samples = 100000
    process = tff.models.hjm.GaussianHJM(
        dim=dim,
        mean_reversion=mr,
        volatility=vol,
        initial_discount_rate_fn=self.instant_forward_rate,
        corr_matrix=corr,
        dtype=dtype)
    times = np.array([0.1, 0.5, 1.0, 2.0])
    paths, df, _, _ = process.sample_paths(
        times,
        num_samples=num_samples,
        time_step=time_step,
        num_time_steps=num_time_steps,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        skip=1000000)
    if dtype is not None:
      with self.subTest('Dtype'):
        self.assertEqual(paths.dtype, dtype)
    paths = self.evaluate(paths)
    df = self.evaluate(df)
    with self.subTest('ShapePaths'):
      self.assertAllEqual(paths.shape, [num_samples, 4])
    with self.subTest('ShapeDiscountFactors'):
      self.assertAllEqual(df.shape, [num_samples, 4])
    discount_mean = np.mean(df, axis=0)
    expected_mean = np.exp(-0.01 * times)
    with self.subTest('DiscountMean'):
      self.assertAllClose(discount_mean, expected_mean, rtol=1e-3, atol=1e-3)

  @parameterized.named_parameters(
      {
          'testcase_name': '1f_constant',
          'dim': 1,
          'mr': [0.03],
          'vol': [0.005],
          'corr': None,
          'factor': 1.0,
      },
      {
          'testcase_name': '2f_constant',
          'dim': 2,
          'mr': [0.03, 0.03],
          'vol': [0.005, 0.005],
          'corr': None,
          'factor': np.sqrt(2.0),
      },
      {
          'testcase_name': '2f_constant_with_corr',
          'dim': 2,
          'mr': [0.03, 0.03],
          'vol': [0.005, 0.005],
          'corr': [[1.0, 0.5], [0.5, 1.0]],
          'factor': np.sqrt(3.0),
      }
      )
  def test_correctness_zcb_sims(self, dim, mr, vol, corr, factor):
    """Tests discount bond simulations."""
    dtype = np.float64
    num_samples = 100000
    process = tff.models.hjm.GaussianHJM(
        dim=dim,
        mean_reversion=mr,
        volatility=vol,
        initial_discount_rate_fn=self.instant_forward_rate,
        corr_matrix=corr,
        dtype=dtype)
    times = np.array([0.1, 0.5, 1.0, 2.0])
    curve_times = np.array([0., 0.5, 1.0, 2.0, 5.0])
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
          sampled_std[:, tidx], factor * true_std, rtol=1e-3, atol=1e-3)

  @parameterized.named_parameters(
      {
          'testcase_name': '1f_single_time',
          'dim': 1,
          'mr': [0.03],
          'vol': [0.005],
          'corr': None,
          'times': [1.0],
          'expected': [0.9803327113840525],
      },
      {
          'testcase_name': '1f_many_times',
          'dim': 1,
          'mr': [0.03],
          'vol': [0.005],
          'corr': None,
          'times': [1.0, 2.0, 3.0],
          'expected': [0.9803327113840525,
                       0.9803218405347454,
                       0.9803116028646381],
      },
      {
          'testcase_name': '2f_single_time',
          'dim': 2,
          'mr': [0.03, 0.03],
          'vol': [0.005, 0.005],
          'corr': None,
          'times': [1.0],
          'expected': [0.9707109604475661],
      },
      {
          'testcase_name': '2f_many_times',
          'dim': 2,
          'mr': [0.03, 0.03],
          'vol': [0.005, 0.005],
          'corr': None,
          'times': [1.0, 2.0, 3.0],
          'expected': [0.9707109604475661,
                       0.9706894322583266,
                       0.9706691582097785]
      }
      )
  def test_correctness_discount_bond_price(self, dim, mr, vol, corr, times,
                                           expected):
    """Tests discount bond price computation."""
    dtype = np.float64
    process = tff.models.hjm.GaussianHJM(
        dim=dim,
        mean_reversion=mr,
        volatility=vol,
        initial_discount_rate_fn=self.instant_forward_rate,
        corr_matrix=corr,
        dtype=dtype)
    x_t = 0.01 * np.ones(shape=(len(times), dim))
    times = np.array(times)
    bond_prices = self.evaluate(
        process.discount_bond_price(x_t, times, times + 1.0))
    self.assertAllEqual(bond_prices.shape, times.shape)
    self.assertAllClose(expected, bond_prices, 1e-8, 1e-8)


if __name__ == '__main__':
  tf.test.main()
