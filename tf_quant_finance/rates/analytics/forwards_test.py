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
"""Tests for rate forwards."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class ForwardRatesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_forward_rates(self, dtype):
    # Discount factors at start dates
    df_start_dates = [[0.95, 0.9, 0.75], [0.95, 0.99, 0.85]]
    # Discount factors at end dates
    df_end_dates = [[0.8, 0.6, 0.5], [0.8, 0.9, 0.5]]
    # Daycount fractions between the dates
    daycount_fractions = [[0.5, 1.0, 2], [0.6, 0.4, 4.0]]
    forward_rates = self.evaluate(
        tff.rates.analytics.forwards.forward_rates(
            df_start_dates, df_end_dates, daycount_fractions, dtype=dtype))
    expected_forward_rates = np.array(
        [[0.375, 0.5, 0.25], [0.3125, 0.25, 0.175]], dtype=dtype)
    np.testing.assert_allclose(
        forward_rates, expected_forward_rates, atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_forward_rates_from_yields(self, dtype):
    groups = np.array([0, 0, 0, 1, 1, 1, 1])
    times = np.array([0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 1.5], dtype=dtype)
    rates = np.array([0.04, 0.041, 0.044, 0.022, 0.025, 0.028, 0.036],
                     dtype=dtype)
    forward_rates = self.evaluate(
        tff.rates.analytics.forwards.forward_rates_from_yields(
            rates, times, groups=groups, dtype=dtype))
    expected_forward_rates = np.array(
        [0.04, 0.042, 0.047, 0.022, 0.028, 0.031, 0.052], dtype=dtype)
    np.testing.assert_allclose(
        forward_rates, expected_forward_rates, atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_forward_rates_from_yields_no_batches(self, dtype):
    times = np.array([0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5], dtype=dtype)
    rates = np.array([0.04, 0.041, 0.044, 0.046, 0.046, 0.047, 0.050],
                     dtype=dtype)
    forward_rates = self.evaluate(
        tff.rates.analytics.forwards.forward_rates_from_yields(
            rates, times, dtype=dtype))
    expected_forward_rates = np.array(
        [0.04, 0.042, 0.047, 0.054, 0.046, 0.05, 0.062], dtype=dtype)
    np.testing.assert_allclose(
        forward_rates, expected_forward_rates, atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_yields_from_forwards(self, dtype):
    groups = np.array([0, 0, 0, 1, 1, 1, 1])
    times = np.array([0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 1.5], dtype=dtype)
    forward_rates = np.array([0.04, 0.042, 0.047, 0.022, 0.028, 0.031, 0.052],
                             dtype=dtype)
    expected_rates = np.array(
        [0.04, 0.041, 0.044, 0.022, 0.025, 0.028, 0.036], dtype=dtype)
    actual_rates = self.evaluate(
        tff.rates.analytics.forwards.yields_from_forward_rates(
            forward_rates, times, groups=groups, dtype=dtype))
    np.testing.assert_allclose(actual_rates, expected_rates, atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_yields_from_forward_rates_no_batches(self, dtype):
    times = np.array([0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5], dtype=dtype)
    forward_rates = np.array([0.04, 0.042, 0.047, 0.054, 0.046, 0.05, 0.062],
                             dtype=dtype)
    expected_rates = np.array(
        [0.04, 0.041, 0.044, 0.046, 0.046, 0.047, 0.050], dtype=dtype)
    actual_rates = self.evaluate(
        tff.rates.analytics.forwards.yields_from_forward_rates(
            forward_rates, times, dtype=dtype))
    np.testing.assert_allclose(actual_rates, expected_rates, atol=1e-6)


if __name__ == '__main__':
  tf.test.main()
