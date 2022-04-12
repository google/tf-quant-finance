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
"""Tests for Brownian Bridge method."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class BrownianBridgeTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for Brownian Bridge method."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_brownian_bridge_double(self, dtype):
    def brownian_bridge_numpy(x_start, x_end, upper_b, lower_b, variance,
                              n_cutoff):
      def f(k):
        a = np.exp(-2 * k * (upper_b - lower_b) * (
            k * (upper_b - lower_b) + (x_end - x_start)) / variance)
        b = np.exp(-2 * (k * (upper_b - lower_b) + x_start - upper_b) * (
            k * (upper_b - lower_b) + (x_end - upper_b)) / variance)
        return a - b

      return np.sum([f(k) for k in range(-n_cutoff, n_cutoff + 1)], axis=0)

    x_start = np.asarray([[1.0, 1.1, 1.1], [1.05, 1.11, 1.11]], dtype=dtype)
    x_end = np.asarray([[2.0, 2.1, 2.8], [2.05, 2.11, 2.11]], dtype=dtype)
    variance = np.asarray([1.0, 1.0, 1.1], dtype=dtype)
    n_cutoff = 3

    upper_barrier = 3.0
    lower_barrier = 0.5

    np_values = brownian_bridge_numpy(
        x_start,
        x_end,
        upper_barrier,
        lower_barrier,
        variance,
        n_cutoff=n_cutoff)

    tff_values = self.evaluate(
        tff.black_scholes.brownian_bridge_double(
            x_start=x_start,
            x_end=x_end,
            variance=variance,
            dtype=dtype,
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            n_cutoff=n_cutoff))

    self.assertEqual(tff_values.shape, np_values.shape)
    self.assertArrayNear(tff_values.flatten(), np_values.flatten(), 1e-7)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_brownian_bridge_single(self, dtype):
    def brownian_bridge_numpy(x_start, x_end, barrier, variance):
      return 1 - np.exp(-2 * (x_start - barrier) * (x_end - barrier) / variance)

    x_start = np.asarray([[1.0, 1.1, 1.1], [1.05, 1.11, 1.11]], dtype=dtype)
    x_end = np.asarray([[2.0, 2.1, 2.8], [2.05, 2.11, 2.11]], dtype=dtype)
    variance = np.asarray([1.0, 1.0, 1.1], dtype=dtype)

    barrier = 3.0

    np_values = brownian_bridge_numpy(
        x_start,
        x_end,
        barrier,
        variance)

    tff_values = self.evaluate(
        tff.black_scholes.brownian_bridge_single(
            x_start=x_start,
            x_end=x_end,
            variance=variance,
            dtype=dtype,
            barrier=barrier))

    self.assertEqual(tff_values.shape, np_values.shape)
    self.assertArrayNear(tff_values.flatten(), np_values.flatten(), 1e-7)

if __name__ == '__main__':
  tf.test.main()
