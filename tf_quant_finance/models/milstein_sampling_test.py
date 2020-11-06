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
"""Tests for methods in `milstein_sampling`."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

milstein_sampling = tff.models.milstein_sampling
euler_sampling = tff.models.euler_sampling
random = tff.math.random


@test_util.run_all_in_graph_and_eager_modes
class MilsteinSamplingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'CustomForLoopWithTimeStep',
          'watch_params': True,
          'use_time_step': True,
      }, {
          'testcase_name': 'WhileLoopWithTimeStep',
          'watch_params': False,
          'use_time_step': True,
      },
      {
          'testcase_name': 'CustomForLoopWithNumSteps',
          'watch_params': True,
          'use_time_step': False,
      }, {
          'testcase_name': 'WhileLoopWithNumSteps',
          'watch_params': False,
          'use_time_step': False,
      })
  def test_sample_paths_wiener(self, watch_params, use_time_step):
    """Tests paths properties for Wiener process (dX = dW)."""

    def drift_fn(_, x):
      return tf.zeros_like(x)

    def vol_fn(_, x):
      return tf.expand_dims(tf.ones_like(x), -1)

    times = np.array([0.1, 0.2, 0.3])
    num_samples = 10000
    if watch_params:
      watch_params = []
    else:
      watch_params = None
    if use_time_step:
      time_step = 0.01
      num_time_steps = None
    else:
      time_step = None
      num_time_steps = 30
    paths = milstein_sampling.sample(
        dim=1,
        drift_fn=drift_fn,
        volatility_fn=vol_fn,
        times=times,
        num_samples=num_samples,
        seed=42,
        time_step=time_step,
        num_time_steps=num_time_steps,
        watch_params=watch_params)
    self.assertAllEqual(paths.shape.as_list(), [num_samples, 3, 1])
    paths = self.evaluate(paths)
    means = np.mean(paths, axis=0).reshape([-1])
    covars = np.cov(paths.reshape([num_samples, -1]), rowvar=False)
    expected_means = np.zeros((3,))
    expected_covars = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))
    self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)
    self.assertAllClose(covars, expected_covars, rtol=1e-2, atol=1e-2)

  def test_sample_paths_1d(self):
    """Tests path properties for 1-dimentional Ito process.

    We construct the following Ito process.

    ````
    dX = mu * sqrt(t) * dt + (a * t + b) dW
    ````

    For this process expected value at time t is x_0 + 2/3 * mu * t^1.5 .
    """
    mu = 0.2
    a = 0.4
    b = 0.33

    def drift_fn(t, x):
      return mu * tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)

    def vol_fn(t, x):
      del x
      return (a * t + b) * tf.ones([1, 1], dtype=t.dtype)

    times = np.array([0.0, 0.1, 0.21, 0.32, 0.43, 0.55])
    num_samples = 10000
    x0 = np.array([0.1])
    paths = self.evaluate(
        milstein_sampling.sample(
            dim=1,
            drift_fn=drift_fn,
            volatility_fn=vol_fn,
            times=times,
            num_samples=num_samples,
            initial_state=x0,
            random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
            time_step=0.01,
            seed=[1, 42]))
    paths_no_zero = self.evaluate(
        milstein_sampling.sample(
            dim=1,
            drift_fn=drift_fn,
            volatility_fn=vol_fn,
            times=times[1:],
            num_samples=num_samples,
            initial_state=x0,
            random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
            time_step=0.01,
            seed=[1, 42]))

    with self.subTest('CorrectShape'):
      self.assertAllClose(paths.shape, (num_samples, 6, 1), atol=0)
    means = np.mean(paths, axis=0).reshape(-1)
    expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)
    with self.subTest('ExpectedResult'):
      self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)
    with self.subTest('IncludeInitialState'):
      self.assertAllClose(paths[:, 1:, :], paths_no_zero)

  def test_sample_bsm(self):
    r"""Tests path properties for 1-dimensional Black Scholes Merton.

    We construct the following Ito process.

    ````
    dX = r * X * dt + \sigma * X * dW
    ````

    Note, that we're not testing in log space.
    """
    r = 0.5
    sigma = 0.5

    def drift_fn(t, x):
      del t
      return r * x

    def vol_fn(t, x):
      del t
      return sigma * tf.expand_dims(x, -1)

    times = np.array([0.0, 0.1, 0.21, 0.32, 0.43, 0.55])
    num_samples = 10000
    x0 = np.array([0.1])
    paths = self.evaluate(
        milstein_sampling.sample(
            dim=1,
            drift_fn=drift_fn,
            volatility_fn=vol_fn,
            times=times,
            num_samples=num_samples,
            initial_state=x0,
            random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
            time_step=0.01,
            seed=[1, 42]))

    euler_paths = self.evaluate(
        euler_sampling.sample(
            dim=1,
            drift_fn=drift_fn,
            volatility_fn=vol_fn,
            times=times,
            num_samples=num_samples,
            initial_state=x0,
            random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
            time_step=0.01,
            seed=[1, 42]))

    mean = np.average(paths)
    stddev = np.std(paths)
    euler_mean = np.average(euler_paths)
    euler_stddev = np.std(euler_paths)
    self.assertAllClose((mean, stddev), (euler_mean, euler_stddev),
                        rtol=1e-3,
                        atol=1e-3)

  def test_sample_paths_dtypes(self):
    """Tests that sampled paths have the expected dtypes."""
    r = 0.5
    sigma = 0.5

    def drift_fn(t, x):
      del t
      return r * x

    def vol_fn(t, x):
      del t
      return sigma * tf.expand_dims(x, -1)

    for dtype in [np.float32, np.float64]:
      paths = self.evaluate(
          milstein_sampling.sample(
              dim=1,
              drift_fn=drift_fn, volatility_fn=vol_fn,
              times=[0.1, 0.2],
              num_samples=10,
              initial_state=[0.1],
              time_step=0.01,
              seed=123,
              dtype=dtype))

      self.assertEqual(paths.dtype, dtype)


if __name__ == '__main__':
  tf.test.main()
