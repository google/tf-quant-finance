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
"""Tests for methods in `euler_sampling`."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

euler_sampling = tff.models.euler_sampling
random = tff.math.random


@test_util.run_all_in_graph_and_eager_modes
class EulerSamplingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'CustomForLoopWithTimeStep',
          'watch_params': True,
          'use_time_step': True,
          'use_time_grid': False,
          'supply_normal_draws': False,
      }, {
          'testcase_name': 'WhileLoopWithTimeStep',
          'watch_params': False,
          'use_time_step': True,
          'use_time_grid': False,
          'supply_normal_draws': False,
      },
      {
          'testcase_name': 'CustomForLoopWithNumSteps',
          'watch_params': True,
          'use_time_step': False,
          'use_time_grid': False,
          'supply_normal_draws': False,
      }, {
          'testcase_name': 'WhileLoopWithNumSteps',
          'watch_params': False,
          'use_time_step': False,
          'use_time_grid': False,
          'supply_normal_draws': False,
      }, {
          'testcase_name': 'WhileLoopWithGrid',
          'watch_params': False,
          'use_time_step': False,
          'use_time_grid': True,
          'supply_normal_draws': False,
      }, {
          'testcase_name': 'WhileLoopWithGridAndDraws',
          'watch_params': False,
          'use_time_step': False,
          'use_time_grid': True,
          'supply_normal_draws': True,
      })
  def test_sample_paths_wiener(self, watch_params, use_time_step,
                               use_time_grid, supply_normal_draws):
    """Tests paths properties for Wiener process (dX = dW)."""
    dtype = tf.float64

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
    if use_time_grid:
      time_step = None
      times_grid = tf.linspace(tf.constant(0.0, dtype=dtype), 0.3, 31)
    else:
      times_grid = None
    if supply_normal_draws:
      num_samples = 1
      # Use antithetic sampling
      normal_draws = tf.random.stateless_normal(
          shape=[5000, 30, 1],
          seed=[1, 42],
          dtype=dtype)
      normal_draws = tf.concat([normal_draws, -normal_draws], axis=0)
    else:
      normal_draws = None
    paths = euler_sampling.sample(
        dim=1, drift_fn=drift_fn, volatility_fn=vol_fn,
        times=times,
        num_samples=num_samples,
        time_step=time_step,
        num_time_steps=num_time_steps,
        watch_params=watch_params,
        normal_draws=normal_draws,
        times_grid=times_grid,
        random_type=random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 42])

    # The correct number of samples
    num_samples = 10000
    with self.subTest('Shape'):
      self.assertAllEqual(paths.shape.as_list(), [num_samples, 3, 1])
    paths = self.evaluate(paths)
    means = np.mean(paths, axis=0).reshape([-1])
    covars = np.cov(paths.reshape([num_samples, -1]), rowvar=False)
    expected_means = np.zeros((3,))
    expected_covars = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))
    with self.subTest('Means'):
      self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)
    with self.subTest('Covariance'):
      self.assertAllClose(covars, expected_covars, rtol=1e-2, atol=1e-2)

  def test_times_grid_long(self):
    """Tests paths properties for Wiener process (dX = dW)."""
    dtype = tf.float64

    def drift_fn(_, x):
      return tf.zeros_like(x)

    def vol_fn(_, x):
      return tf.expand_dims(tf.ones_like(x), -1)

    times = np.array([0.1, 0.2, 0.3])
    num_samples = 10000
    times_grid = tf.linspace(tf.constant(0.0, dtype=dtype), 0.32, 33)
    # Use antithetic sampling
    normal_draws = tf.random.stateless_normal(
        shape=[5000, 32, 1],
        seed=[1, 42],
        dtype=dtype)
    normal_draws = tf.concat([normal_draws, -normal_draws], axis=0)

    paths = euler_sampling.sample(
        dim=1, drift_fn=drift_fn, volatility_fn=vol_fn,
        times=times,
        num_samples=num_samples,
        normal_draws=normal_draws,
        times_grid=times_grid,
        seed=[1, 42])
    with self.subTest('Shape'):
      self.assertAllEqual(paths.shape.as_list(), [num_samples, 3, 1])
    paths = self.evaluate(paths)
    means = np.mean(paths, axis=0).reshape([-1])
    covars = np.cov(paths.reshape([num_samples, -1]), rowvar=False)
    expected_means = np.zeros((3,))
    expected_covars = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))
    with self.subTest('Means'):
      self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)
    with self.subTest('Covariance'):
      self.assertAllClose(covars, expected_covars, rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NonBatch',
          'use_batch': False,
          'watch_params': None,
          'supply_normal_draws': False,
          'random_type': tff.math.random.RandomType.STATELESS,
      }, {
          'testcase_name': 'Batch',
          'use_batch': True,
          'watch_params': None,
          'supply_normal_draws': False,
          'random_type': tff.math.random.RandomType.STATELESS,
      }, {
          'testcase_name': 'BatchAntithetic',
          'use_batch': True,
          'watch_params': None,
          'supply_normal_draws': False,
          'random_type': tff.math.random.RandomType.STATELESS_ANTITHETIC,
      }, {
          'testcase_name': 'BatchWithCustomForLoop',
          'use_batch': True,
          'watch_params': [],
          'supply_normal_draws': False,
          'random_type': tff.math.random.RandomType.STATELESS,
      }, {
          'testcase_name': 'BatchWithNormalDraws',
          'use_batch': True,
          'watch_params': None,
          'supply_normal_draws': True,
          'random_type': tff.math.random.RandomType.STATELESS,
      })
  def test_sample_paths_1d(self, use_batch, watch_params, supply_normal_draws,
                           random_type):
    """Tests path properties for 1-dimentional Ito process.

    We construct the following Ito process.

    ````
    dX = mu * sqrt(t) * dt + (a * t + b) dW
    ````

    For this process expected value at time t is x_0 + 2/3 * mu * t^1.5 .
    Args:
      use_batch: Test parameter to specify if we are testing the batch of Euler
        sampling.
      watch_params: Triggers custom for loop.
      supply_normal_draws: Supply normal draws.
      random_type: `RandomType` of the sampled normal draws.
    """
    dtype = tf.float64
    mu = 0.2
    a = 0.4
    b = 0.33

    def drift_fn(t, x):
      drift = mu * tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)
      return drift

    def vol_fn(t, x):
      del x
      if not use_batch:
        return (a * t + b) * tf.ones([1, 1], dtype=t.dtype)
      else:
        return (a * t + b) * tf.ones([2, 1, 1, 1], dtype=t.dtype)

    times = np.array([0.0, 0.1, 0.21, 0.32, 0.43, 0.55])
    num_samples = 10000

    if supply_normal_draws:
      # Use antithetic sampling
      normal_draws = tf.random.stateless_normal(
          shape=[2, 5000, 55, 1],
          seed=[1, 42],
          dtype=dtype)
      normal_draws = tf.concat([normal_draws, -normal_draws], axis=1)
    else:
      normal_draws = None

    if use_batch:
      # x0.shape = [2, 1, 1]
      x0 = np.array([[[0.1]], [[0.1]]])
    else:
      x0 = np.array([0.1])
    paths = self.evaluate(
        euler_sampling.sample(
            dim=1,
            drift_fn=drift_fn, volatility_fn=vol_fn,
            times=times, num_samples=num_samples, initial_state=x0,
            random_type=random_type,
            normal_draws=normal_draws,
            watch_params=watch_params,
            time_step=0.01,
            seed=[1, 42],
            dtype=dtype))
    paths_no_zero = self.evaluate(
        euler_sampling.sample(
            dim=1,
            drift_fn=drift_fn, volatility_fn=vol_fn,
            times=times[1:], num_samples=num_samples, initial_state=x0,
            random_type=random_type,
            normal_draws=normal_draws,
            time_step=0.01,
            seed=[1, 42],
            dtype=dtype))

    with self.subTest('CorrectShape'):
      if not use_batch:
        self.assertAllClose(paths.shape, (num_samples, 6, 1), atol=0)
      else:
        self.assertAllClose(paths.shape, (2, num_samples, 6, 1), atol=0)
    if not use_batch:
      means = np.mean(paths, axis=0).reshape(-1)
    else:
      means = np.mean(paths, axis=1).reshape([2, 1, 6])
    expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)
    with self.subTest('ExpectedResult'):
      self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)
    if not use_batch:
      with self.subTest('IncludeInitialState'):
        self.assertAllClose(paths[:, 1:, :], paths_no_zero)

  @parameterized.named_parameters(
      {
          'testcase_name': 'PSEUDO',
          'random_type': tff.math.random.RandomType.PSEUDO,
          'seed': 12134,
      }, {
          'testcase_name': 'STATELESS',
          'random_type': tff.math.random.RandomType.STATELESS,
          'seed': [1, 2],
      }, {
          'testcase_name': 'SOBOL',
          'random_type': tff.math.random.RandomType.SOBOL,
          'seed': None,
      }, {
          'testcase_name': 'HALTON_RANDOMIZED',
          'random_type': tff.math.random.RandomType.HALTON_RANDOMIZED,
          'seed': 12134,
      })
  def test_sample_paths_2d(self, random_type, seed):
    """Tests path properties for 2-dimentional Ito process.

    We construct the following Ito processes.

    dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2
    dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2

    mu_1, mu_2 are constants.
    s_ij = a_ij t + b_ij

    For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.

    Args:
      random_type: Random number type defined by tff.math.random.RandomType
        enum.
      seed: Random seed.
    """
    mu = np.array([0.2, 0.7])
    a = np.array([[0.4, 0.1], [0.3, 0.2]])
    b = np.array([[0.33, -0.03], [0.21, 0.5]])

    def drift_fn(t, x):
      return mu * tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)

    def vol_fn(t, x):
      del x
      return (a * t + b) * tf.ones([2, 2], dtype=t.dtype)

    num_samples = 10000
    times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
    x0 = np.array([0.1, -1.1])
    paths = self.evaluate(
        euler_sampling.sample(
            dim=2,
            drift_fn=drift_fn, volatility_fn=vol_fn,
            times=times,
            num_samples=num_samples,
            initial_state=x0,
            time_step=0.01,
            random_type=random_type,
            seed=seed))

    self.assertAllClose(paths.shape, (num_samples, 5, 2), atol=0)
    means = np.mean(paths, axis=0)
    times = np.reshape(times, [-1, 1])
    expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)
    self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'CustomForLoop',
          'watch_params': True,
      }, {
          'testcase_name': 'WhileLoop',
          'watch_params': False,
      })
  def test_halton_sample_paths_2d(self, watch_params):
    """Tests path properties for 2-dimentional Ito process."""
    # We construct the following Ito processes.
    # dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2
    # dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2
    # mu_1, mu_2 are constants.
    # s_ij = a_ij t + b_ij
    # For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.
    dtype = tf.float64
    num_samples = 10000
    times = np.array([0.1, 0.21, 0.32])
    x0 = np.array([0.1, -1.1])
    mu = np.array([0.2, 0.7])
    a = np.array([[0.4, 0.1], [0.3, 0.2]])
    b = np.array([[0.33, -0.03], [0.21, 0.5]])
    def sample_fn(mu, a, b):
      mu = tf.convert_to_tensor(mu, dtype=dtype)
      a = tf.convert_to_tensor(a, dtype=dtype)
      b = tf.convert_to_tensor(b, dtype=dtype)
      def drift_fn(t, x):
        return mu * tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)

      def vol_fn(t, x):
        del x
        return (a * t + b) * tf.ones([2, 2], dtype=t.dtype)
      if watch_params:
        watch_params_tf = [a, b]
      else:
        watch_params_tf = None
      return euler_sampling.sample(
          dim=2,
          drift_fn=drift_fn, volatility_fn=vol_fn,
          times=times,
          num_samples=num_samples,
          initial_state=x0,
          random_type=tff.math.random.RandomType.HALTON,
          time_step=0.01,
          seed=12134,
          skip=100,
          watch_params=watch_params_tf,
          dtype=dtype)

    paths = self.evaluate(tf.function(sample_fn)(mu, a, b))

    self.assertAllClose(paths.shape, (num_samples, 3, 2), atol=0)
    means = np.mean(paths, axis=0)
    times = np.reshape(times, [-1, 1])
    expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)
    self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'PSEUDO_ANTITHETIC',
          'random_type': tff.math.random.RandomType.PSEUDO,
          'seed': 12134,
      }, {
          'testcase_name': 'STATELESS_ANTITHETIC',
          'random_type': tff.math.random.RandomType.STATELESS,
          'seed': [0, 12134],
      })
  def test_antithetic_sample_paths_mean_2d(self, random_type, seed):
    """Tests path properties for 2-dimentional anthithetic variates method.

    The same test as above but with `PSEUDO_ANTITHETIC` random type.
    We construct the following Ito processes.

    dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2
    dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2

    mu_1, mu_2 are constants.
    s_ij = a_ij t + b_ij

    For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.

    Args:
      random_type: Random number type defined by tff.math.random.RandomType
        enum.
      seed: Random seed.
    """
    mu = np.array([0.2, 0.7])
    a = np.array([[0.4, 0.1], [0.3, 0.2]])
    b = np.array([[0.33, -0.03], [0.21, 0.5]])

    def drift_fn(t, x):
      del x
      return mu * tf.sqrt(t)

    def vol_fn(t, x):
      del x
      return (a * t + b) * tf.ones([2, 2], dtype=t.dtype)

    times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
    num_samples = 5000
    x0 = np.array([0.1, -1.1])
    paths = self.evaluate(
        euler_sampling.sample(
            dim=2,
            drift_fn=drift_fn, volatility_fn=vol_fn,
            times=times,
            num_samples=num_samples,
            initial_state=x0,
            random_type=random_type,
            time_step=0.01,
            seed=seed))

    self.assertAllClose(paths.shape, (num_samples, 5, 2), atol=0)
    means = np.mean(paths, axis=0)
    times = np.reshape(times, [-1, 1])
    expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)
    # Antithetic variates method produces better estimate than the
    # estimate with the `PSEUDO` random type
    self.assertAllClose(means, expected_means, rtol=5e-3, atol=5e-3)

  def test_sample_paths_dtypes(self):
    """Sampled paths have the expected dtypes."""
    for dtype in [np.float32, np.float64]:
      drift_fn = lambda t, x: tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)
      vol_fn = lambda t, x: t * tf.ones([1, 1], dtype=t.dtype)

      paths = self.evaluate(
          euler_sampling.sample(
              dim=1,
              drift_fn=drift_fn, volatility_fn=vol_fn,
              times=[0.1, 0.2],
              num_samples=10,
              initial_state=[0.1],
              time_step=0.01,
              seed=123,
              dtype=dtype))

      self.assertEqual(paths.dtype, dtype)

  def test_wrong_times(self):
    """Error is raised if `times` or `times_grid` is not increasing."""
    dtype = tf.float64
    def drift_fn(_, x):
      return tf.zeros_like(x)

    def vol_fn(_, x):
      return tf.expand_dims(tf.ones_like(x), -1)

    with self.subTest('WrongTimes'):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        times_grid = [0.1, 0.5, 1.0]
        normal_draws = tf.random.stateless_normal(
            shape=[100, 5, 1], seed=[1, 1], dtype=dtype)
        self.evaluate(
            euler_sampling.sample(
                dim=1, drift_fn=drift_fn, volatility_fn=vol_fn,
                times=[0.1, 0.5, 1.0],
                normal_draws=normal_draws,
                times_grid=times_grid,
                seed=42,
                validate_args=True,
                dtype=dtype))
    with self.subTest('WrongTimesGrid'):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        times_grid = [0.1, 0.5, 1.0]
        normal_draws = tf.random.stateless_normal(
            shape=[100, 5, 1], seed=[1, 1], dtype=dtype)
        self.evaluate(
            euler_sampling.sample(
                dim=1, drift_fn=drift_fn, volatility_fn=vol_fn,
                times=[0.1, 0.5, 1.0],
                normal_draws=normal_draws,
                times_grid=times_grid,
                seed=42,
                validate_args=True,
                dtype=dtype))

  def test_sample_shape_mismatch(self):
    """Error is raised if `dim` is mismatched with the one from normal_draws."""
    dtype = tf.float64
    def drift_fn(_, x):
      return tf.zeros_like(x)

    def vol_fn(_, x):
      return tf.expand_dims(tf.ones_like(x), -1)

    with self.subTest('WrongTimes'):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            euler_sampling.sample(
                dim=1, drift_fn=drift_fn, volatility_fn=vol_fn,
                times=[0.1, 0.5, 2.0, 1.0],
                time_step=0.01,
                seed=42,
                validate_args=True,
                dtype=dtype))

    with self.subTest('WrongTimesGrid'):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        times_grid = [0.1, 0.5, 1.0, 1.0]
        self.evaluate(euler_sampling.sample(
            dim=1, drift_fn=drift_fn, volatility_fn=vol_fn,
            times=[0.1, 0.5, 1.0],
            times_grid=times_grid,
            seed=42,
            validate_args=True,
            dtype=dtype))

if __name__ == '__main__':
  tf.test.main()
