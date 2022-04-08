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

"""Tests for monotone_convex module."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.rates.hagan_west import monotone_convex


class MonotoneConvexTest(tf.test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_interpolate_adjacent(self):
    dtype = tf.float64
    times = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    values = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    actual = self.evaluate(monotone_convex._interpolate_adjacent(times, values))
    expected = [0.75, 1.5, 2.5, 3.5, 4.5, 5.25]
    np.testing.assert_array_equal(actual, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_interpolation(self):
    dtype = tf.float64
    interval_times = tf.constant([0.25, 0.5, 1.0, 2.0, 3.0, 4.0], dtype=dtype)
    interval_values = tf.constant([0.05, 0.051, 0.052, 0.053, 0.055, 0.055],
                                  dtype=dtype)
    test_times = tf.constant([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 1.1], dtype=dtype)
    expected = [0.0505, 0.05133333, 0.05233333, 0.055, 0.055, 0.055, 0.05241]
    actual, integrated_actual = self.evaluate(
        monotone_convex.interpolate(test_times, interval_values,
                                    interval_times))
    np.testing.assert_allclose(actual, expected)
    integrated_expected = [0, 0, 0, 0, 0, 0.055, 0.005237]
    np.testing.assert_allclose(integrated_actual, integrated_expected)

  @test_util.run_in_graph_and_eager_modes
  def test_interpolation_two_intervals(self):
    dtype = tf.float64
    interval_times = tf.constant([0.25, 0.5], dtype=dtype)
    interval_values = tf.constant([0.05, 0.051], dtype=dtype)
    test_times = tf.constant([0.25, 0.5], dtype=dtype)
    actual, integrated_actual = self.evaluate(
        monotone_convex.interpolate(test_times, interval_values,
                                    interval_times))
    expected = [0.0505, 0.05125]
    np.testing.assert_allclose(actual, expected)
    integrated_expected = [0.0, 0.01275]
    np.testing.assert_allclose(integrated_actual, integrated_expected)

  @test_util.deprecated_graph_mode_only
  def test_interpolation_differentiable(self):
    dtype = tf.float64
    interval_times = tf.constant([0.25, 0.5, 1.0, 2.0, 3.0], dtype=dtype)
    knot_1y = tf.constant([0.052], dtype=dtype)
    interval_values = tf.concat([
        tf.constant([0.05, 0.051], dtype=dtype), knot_1y,
        tf.constant([0.053, 0.055], dtype=dtype)
    ],
                                axis=0)
    test_time = tf.constant([1.1, 2.7], dtype=dtype)
    interpolated, _ = monotone_convex.interpolate(test_time, interval_values,
                                                  interval_times)
    gradient_1y = self.evaluate(tf.convert_to_tensor(
        tf.gradients(interpolated[0], knot_1y)[0]))
    gradient_zero = self.evaluate(tf.convert_to_tensor(
        tf.gradients(interpolated[1], knot_1y)[0]))

    self.assertAlmostEqual(gradient_1y[0], 0.42)
    self.assertAlmostEqual(gradient_zero[0], 0.0)

  @test_util.run_in_graph_and_eager_modes
  def test_integrated_value(self):
    dtype = np.float64
    interval_times = tf.constant([0.25, 0.5, 1.0, 2.0, 3.0], dtype=dtype)
    interval_values = tf.constant([0.05, 0.051, 0.052, 0.053, 0.055],
                                  dtype=dtype)
    # Checks that the integral is correct by computing it using a Riemann sum.
    test_times = tf.constant(np.linspace(1.0, 1.1, num=51), dtype=dtype)
    dt = np.array(0.002, dtype=dtype)
    values, integrated = self.evaluate(
        monotone_convex.interpolate(test_times, interval_values,
                                    interval_times))
    actual = integrated[-1]
    expected = np.sum(dt * (values[1:] + values[:-1]) / 2)
    self.assertAlmostEqual(actual, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_flat_values(self):
    dtype = np.float64
    interval_times = np.array([0.3, 1.0, 1.43, 3.7, 9.2, 12.48], dtype=dtype)
    interval_values = np.array([8.0] * 6, dtype=dtype)
    test_times = tf.constant(
        [0.1, 1.1, 1.22, 0.45, 1.8, 3.8, 7.45, 7.73, 9.6, 11.7, 12.],
        dtype=dtype)
    actual, _ = self.evaluate(
        monotone_convex.interpolate(test_times, interval_values,
                                    interval_times))
    expected = np.zeros([11], dtype=dtype) + 8.
    np.testing.assert_allclose(actual, expected)

  def test_interpolated_forwards_with_discrete_forwards(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      reference_times = tf.constant([0.25, 0.5, 1.0, 2.0, 3.0], dtype=dtype)
      discrete_forwards = tf.constant([0.05, 0.051, 0.052, 0.053, 0.055],
                                      dtype=dtype)
      test_times = tf.constant([0.25, 0.5, 1.0, 2.0, 3.0, 1.1], dtype=dtype)
      expected = np.array(
          [0.0505, 0.05133333, 0.05233333, 0.054, 0.0555, 0.05241], dtype=dtype)
      actual = self.evaluate(
          monotone_convex.interpolate_forward_rate(
              test_times, reference_times, discrete_forwards=discrete_forwards))
      np.testing.assert_allclose(actual, expected, rtol=1e-5)

  def test_interpolated_forwards_with_yields(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      reference_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
      yields = np.array([2.75, 4.0, 4.75, 5.0, 4.75], dtype=dtype) / 100

      # Times for which the interpolated values are required.
      interpolation_times = np.array([0.3, 1.3, 2.1, 4.5], dtype=dtype)
      actual = self.evaluate(
          monotone_convex.interpolate_forward_rate(
              interpolation_times,
              reference_times=reference_times,
              yields=yields))
      expected = np.array([0.0229375, 0.05010625, 0.0609, 0.03625], dtype=dtype)
      np.testing.assert_allclose(actual, expected, rtol=1e-5)

  def test_interpolated_yields_with_discrete_forwards(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      reference_times = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=dtype)
      discrete_forwards = tf.constant([5, 4.5, 4.1, 5.5], dtype=dtype)
      test_times = tf.constant(
          [0.25, 0.5, 1.0, 2.0, 3.0, 1.1, 2.5, 2.9, 3.6, 4.0], dtype=dtype)
      expected = np.array([
          5.1171875, 5.09375, 5.0, 4.75, 4.533333, 4.9746, 4.624082, 4.535422,
          4.661777, 4.775
      ],
                          dtype=dtype)
      actual = self.evaluate(
          monotone_convex.interpolate_yields(
              test_times, reference_times, discrete_forwards=discrete_forwards))
      np.testing.assert_allclose(actual, expected, rtol=1e-5)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Example1',
          'interpolation_times': [0.25, 0.5, 1.0, 2.0, 3.0,
                                  1.1, 2.5, 2.9, 3.6, 4.0],
          'reference_times': [1.0, 2.0, 3.0, 4.0],
          'yields': [5.0, 4.75, 4.53333333, 4.775],
          'expected': [5.1171875, 5.09375, 5.0, 4.75, 4.533333, 4.9746,
                       4.624082, 4.535422, 4.661777, 4.775]
      }, {
          'testcase_name': 'Example2',
          'interpolation_times': [0.1, 0.2, 0.21],
          'reference_times': [0.1, 0.2, 0.21],
          'yields': [0.1, 0.2, 0.3],
          'expected': [0.1, 0.2, 0.3]
      })
  def test_interpolated_yields_with_yields(
      self, interpolation_times, reference_times, yields, expected):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      expected = np.array(expected, dtype=dtype)
      actual = self.evaluate(
          monotone_convex.interpolate_yields(
              interpolation_times, reference_times, yields=yields, dtype=dtype))
      np.testing.assert_allclose(actual, expected, rtol=1e-5)

  def test_interpolated_yields_flat_curve(self):
    """Checks the interpolation for flat curves."""
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      reference_times = np.array([0.3, 1.0, 1.43, 3.7, 9.2, 12.48], dtype=dtype)
      yields = np.array([8.0] * 6, dtype=dtype)

      # Times for which the interpolated values are required.
      interpolation_times = tf.constant(
          [0.1, 1.1, 1.22, 0.45, 1.8, 3.8, 7.45, 7.73, 9.6, 11.7, 12.],
          dtype=dtype)
      expected = np.array([8.0] * 11, dtype=dtype)
      actual = self.evaluate(
          monotone_convex.interpolate_yields(
              interpolation_times, reference_times, yields=yields))
      np.testing.assert_allclose(actual, expected, rtol=1e-5)

  def test_interpolated_yields_zero_time(self):
    """Checks the interpolation for yield curve is non-NaN for 0 time."""
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      reference_times = np.array([0.3, 1.0, 1.43, 3.7, 9.2, 12.48], dtype=dtype)
      yields = np.array([3.0, 3.1, 2.9, 4.1, 4.3, 5.1], dtype=dtype)

      # Times for which the interpolated values are required.
      interpolation_times = tf.constant([0.0], dtype=dtype)
      actual = self.evaluate(
          monotone_convex.interpolate_yields(
              interpolation_times, reference_times, yields=yields))
      np.testing.assert_allclose(actual, [0.0], rtol=1e-8)

  def test_interpolated_yields_consistency(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      reference_times = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
      yields = np.array([5.0, 4.75, 4.53333333, 4.775], dtype=dtype)

      # Times for which the interpolated values are required.
      interpolation_times_1 = tf.constant([0.25, 0.5, 1.0, 2.0, 3.0],
                                          dtype=dtype)
      interpolation_times_2 = tf.constant([1.1, 2.5, 2.9, 3.6, 4.0],
                                          dtype=dtype)
      expected = np.array([
          5.1171875, 5.09375, 5.0, 4.75, 4.533333, 4.9746, 4.624082, 4.535422,
          4.661777, 4.775
      ],
                          dtype=dtype)
      actual_1 = monotone_convex.interpolate_yields(
          interpolation_times_1, reference_times, yields=yields)
      actual_2 = monotone_convex.interpolate_yields(
          interpolation_times_2, reference_times, yields=yields)
      actual = self.evaluate(tf.concat([actual_1, actual_2], axis=0))
      np.testing.assert_allclose(actual, expected, rtol=1e-5)


if __name__ == '__main__':
  tf.test.main()
