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
"""Tests for `sample_paths` of `ItoProcess`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_quant_finance.experimental.models_v2 import generic_ito_process
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class GenericItoProcessTest(tf.test.TestCase):

  def test_sample_paths_wiener(self):
    """Tests paths properties for Wiener process (dX = dW)."""

    def drift_fn(_, x):
      return tf.zeros_like(x)

    def vol_fn(_, x):
      return tf.expand_dims(tf.ones_like(x), -1)

    process = generic_ito_process.GenericItoProcess(
        dim=1, drift_fn=drift_fn, volatility_fn=vol_fn)
    times = np.array([0.1, 0.2, 0.3])
    num_samples = 10000

    paths = self.evaluate(
        process.sample_paths(
            times=times, num_samples=num_samples, seed=42, time_step=0.005))

    means = np.mean(paths, axis=0).reshape([-1])
    covars = np.cov(paths.reshape([num_samples, -1]), rowvar=False)
    expected_means = np.zeros((3,))
    expected_covars = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))
    self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)
    self.assertAllClose(covars, expected_covars, rtol=1e-2, atol=1e-2)

  def test_sample_paths_2d(self):
    """Tests path properties for 2-dimentional Ito process.

    We construct the following Ito processes.

    dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2
    dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2

    mu_1, mu_2 are constants.
    s_ij = a_ij t + b_ij

    For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.
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
    process = generic_ito_process.GenericItoProcess(
        dim=2, drift_fn=drift_fn, volatility_fn=vol_fn)
    times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
    x0 = np.array([0.1, -1.1])
    paths = self.evaluate(
        process.sample_paths(
            times,
            num_samples=num_samples,
            initial_state=x0,
            time_step=0.01,
            seed=12134))

    self.assertAllClose(paths.shape, (num_samples, 5, 2), atol=0)
    means = np.mean(paths, axis=0)
    times = np.reshape(times, [-1, 1])
    expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)
    self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)

  def test_sample_paths_dtypes(self):
    """Sampled paths have the expected dtypes."""
    for dtype in [np.float32, np.float64]:
      drift_fn = lambda t, x: tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)
      vol_fn = lambda t, x: t * tf.ones([1, 1], dtype=t.dtype)
      process = generic_ito_process.GenericItoProcess(
          dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=dtype)

      paths = self.evaluate(
          process.sample_paths(
              times=[0.1, 0.2],
              num_samples=10,
              initial_state=[0.1],
              time_step=0.01,
              seed=123))
      self.assertEqual(paths.dtype, dtype)


if __name__ == '__main__':
  tf.test.main()
