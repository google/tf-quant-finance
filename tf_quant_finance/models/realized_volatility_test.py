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
"""Tests for models.realized_volatility."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class RealizedVolatilityTest(parameterized.TestCase, tf.test.TestCase):

  def test_log_vol_calculation(self):
    """Tests the basic calculation of log realized volatility."""
    dtype = tf.float64
    num_series = 200
    num_times = 100
    seed = (1, 2)
    draws = tf.random.stateless_normal((num_series, num_times),
                                       seed=seed,
                                       dtype=dtype)
    sample_paths = tf.math.exp(tf.math.cumsum(draws, axis=-1))
    volatilities = tff.models.realized_volatility(sample_paths)
    expected_volatilities = tf.math.sqrt(
        tf.math.reduce_sum(draws[:, 1:]**2, axis=1))
    self.assertAllClose(volatilities, expected_volatilities, 1e-6)

  def test_log_vol_scaling_factor(self):
    """Tests use of the scaling factor in log volatility calculation."""
    dtype = tf.float64
    num_series = 200
    num_times = 100
    seed = (1, 2)
    draws = tf.random.stateless_normal((num_series, num_times),
                                       seed=seed,
                                       dtype=dtype)
    sample_paths = tf.math.exp(tf.math.cumsum(draws, axis=-1))
    volatilities = tff.models.realized_volatility(
        sample_paths, scaling_factors=np.sqrt(num_times), dtype=dtype)
    expected_volatilities = tf.math.sqrt(
        tf.math.reduce_sum(draws[:, 1:]**2, axis=1) * num_times)
    self.assertAllClose(volatilities, expected_volatilities, 1e-6)

  def test_log_vol_log_scale_sample(self):
    """Tests the treatment of log scale samples in log volatility calc."""
    dtype = tf.float64
    num_series = 200
    num_times = 100
    seed = (1, 2)
    draws = tf.random.stateless_normal((num_series, num_times),
                                       seed=seed,
                                       dtype=dtype)
    sample_paths = tf.math.cumsum(draws, axis=-1)
    volatilities = tff.models.realized_volatility(
        sample_paths, path_scale=tff.models.PathScale.LOG)
    expected_volatilities = tf.math.sqrt(
        tf.math.reduce_sum(draws[:, 1:]**2, axis=1))
    self.assertAllClose(volatilities, expected_volatilities, 1e-6)

  def test_log_vol_non_default_times(self):
    """Tests use of non-uniform sampling times in the volatility calculation."""
    dtype = tf.float64
    num_series = 500
    num_times = 100
    seed = (1, 2)
    time_deltas = tf.random.stateless_uniform((num_series, num_times),
                                              seed=seed,
                                              dtype=dtype)
    draws = tf.random.stateless_normal((num_series, num_times),
                                       seed=seed,
                                       dtype=dtype)
    sample_paths = tf.math.exp(
        tf.math.cumsum(tf.math.sqrt(time_deltas) * draws, axis=-1))
    volatilities = tff.models.realized_volatility(
        sample_paths, times=tf.math.cumsum(time_deltas, axis=1))
    expected_volatilities = tf.math.sqrt(
        tf.math.reduce_sum(draws[:, 1:]**2, axis=1))
    self.assertAllClose(volatilities, expected_volatilities, 1e-6)

  def test_abs_volatility_calculation(self):
    """Tests the basic calculation of abs realized volatility."""
    dtype = tf.float64
    num_series = 200
    num_times = 100
    seed = (1, 2)
    draws = tf.random.stateless_normal((num_series, num_times),
                                       seed=seed,
                                       dtype=dtype)
    sample_paths = tf.math.exp(tf.math.cumsum(draws, axis=-1))
    volatilities = tff.models.realized_volatility(
        sample_paths, returns_type=tff.models.ReturnsType.ABS)
    diffs = tf.math.abs(tff.math.diff(sample_paths, exclusive=True))
    expected_volatilities = tf.reduce_sum(diffs / sample_paths[:, :-1], axis=1)
    self.assertAllClose(volatilities, expected_volatilities, 1e-6)

  def test_abs_volatility_scaling(self):
    """Tests abs realized volatility calculation with a scaling factor."""
    dtype = tf.float64
    num_series = 200
    num_times = 100
    seed = (1, 2)
    draws = tf.random.stateless_normal((num_series, num_times),
                                       seed=seed,
                                       dtype=dtype)
    sample_paths = tf.math.exp(tf.math.cumsum(draws, axis=-1))
    scaling = 100 * np.sqrt(np.pi / (2 * num_times))
    volatilities = tff.models.realized_volatility(
        sample_paths,
        scaling_factors=scaling,
        returns_type=tff.models.ReturnsType.ABS)
    diffs = tf.math.abs(tff.math.diff(sample_paths, exclusive=True))
    expected_volatilities = tf.reduce_sum(diffs / sample_paths[:, :-1], axis=1)
    self.assertAllClose(volatilities, scaling * expected_volatilities, 1e-6)

  def test_abs_volatility_logspace_samples(self):
    """Tests abs realized volatility for logspace sample paths."""
    dtype = tf.float64
    num_series = 200
    num_times = 100
    seed = (1, 2)
    draws = tf.random.stateless_normal((num_series, num_times),
                                       seed=seed,
                                       dtype=dtype)
    logspace_paths = tf.math.cumsum(draws, axis=-1)
    sample_paths = tf.math.exp(logspace_paths)
    volatilities = tff.models.realized_volatility(
        logspace_paths,
        path_scale=tff.models.PathScale.LOG,
        returns_type=tff.models.ReturnsType.ABS)
    diffs = tf.math.abs(tff.math.diff(sample_paths, exclusive=True))
    expected_volatilities = tf.reduce_sum(diffs / sample_paths[:, :-1], axis=1)
    self.assertAllClose(volatilities, expected_volatilities, 1e-6)

  def test_abs_volatility_non_default_times(self):
    """Tests abs realized volatiltity with non-default times."""
    dtype = tf.float64
    num_series = 200
    num_times = 100
    seed = (1, 2)
    draws = tf.random.stateless_normal((num_series, num_times),
                                       seed=seed,
                                       dtype=dtype)
    time_deltas = tf.random.stateless_uniform((num_series, num_times),
                                              seed=seed,
                                              dtype=dtype)
    sample_paths = tf.math.exp(tf.math.cumsum(draws, axis=-1))
    volatilities = tff.models.realized_volatility(
        sample_paths,
        times=tf.math.cumsum(time_deltas, axis=1),
        returns_type=tff.models.ReturnsType.ABS)
    numer = tf.math.abs(tff.math.diff(sample_paths, exclusive=True))
    denom = sample_paths[:, :-1] * time_deltas[:, 1:]
    expected_volatilities = tf.math.reduce_sum(numer / denom, axis=1)
    self.assertAllClose(volatilities, expected_volatilities, 1e-6)

  @parameterized.named_parameters(
      ('Abs', tff.models.ReturnsType.ABS),
      ('Log', tff.models.ReturnsType.LOG)
      )
  def test_non_default_axis(self, returns_type):
    """Tests realized volatility works with non default axis."""
    dtype = tf.float64
    num_series = 200
    num_times = 100
    seed = (1, 2)
    draws = tf.random.stateless_normal((num_series, num_times),
                                       seed=seed,
                                       dtype=dtype)
    sample_paths = tf.math.exp(tf.math.cumsum(draws, axis=-1))

    volatilities = tff.models.realized_volatility(
        tf.transpose(sample_paths),
        returns_type=returns_type,
        axis=0)

    if returns_type == tff.models.ReturnsType.ABS:
      diffs = tf.math.abs(tff.math.diff(sample_paths, exclusive=True))
      expected_volatilities = tf.reduce_sum(
          diffs / sample_paths[:, :-1], axis=1)
    elif returns_type == tff.models.ReturnsType.LOG:
      expected_volatilities = tf.math.sqrt(
          tf.math.reduce_sum(draws[:, 1:]**2, axis=1))

    self.assertAllClose(volatilities, expected_volatilities, 1e-6)


if __name__ == '__main__':
  tf.test.main()
