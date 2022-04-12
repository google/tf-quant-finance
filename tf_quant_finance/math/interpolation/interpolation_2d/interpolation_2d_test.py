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

"""Tests for interpolation_2d."""

import numpy as np
import tensorflow.compat.v1 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
interpolation_2d = tff.math.interpolation.interpolation_2d


@test_util.run_all_in_graph_and_eager_modes
class Interpolation2DTest(tf.test.TestCase):

  def test_docstring_example(self):
    """Computes values of example in the docstring for function interpolate."""
    for dtype in [np.float32, np.float64]:
      times = tf.constant([2., 2.5, 3], dtype=dtype)
      strikes = tf.constant([16, 22, 35], dtype=dtype)

      times_data = tf.constant([1.5, 2.5, 3.5, 4.5, 5.5], dtype=dtype)
      sigma_square_data = tf.constant(
          [[0.15, 0.25, 0.35, 0.4, 0.45, 0.4],
           [0.2, 0.35, 0.55, 0.45, 0.4, 0.6],
           [0.3, 0.45, 0.25, 0.4, 0.5, 0.65],
           [0.25, 0.25, 0.45, 0.25, 0.5, 0.55],
           [0.35, 0.35, 0.25, 0.4, 0.55, 0.55]], dtype=dtype)
      total_variance = tf.expand_dims(times_data, -1) * sigma_square_data
      strike_data = tf.broadcast_to(
          tf.constant([15, 25, 35, 40, 50, 55], dtype=dtype), [5, 6])
      interpolator = interpolation_2d.Interpolation2D(
          times_data, strike_data, total_variance, dtype=dtype)
      interpolated_vols = interpolator.interpolate(times, strikes)
      with self.subTest("CorrectDtype"):
        self.assertEqual(interpolated_vols.dtype.as_numpy_dtype, dtype)
      with self.subTest("CorrectShape"):
        self.assertAllClose(interpolated_vols.shape.as_list(), [3])
      expected_vols = np.array([0.382399, 0.716694, 1.125])
      with self.subTest("CorrectResult"):
        self.assertAllClose(
            interpolated_vols, expected_vols, rtol=1e-04, atol=1e-04)

  def test_batch(self):
    """Test batching."""
    dtype = np.float64
    times = tf.constant([[2., 2.5, 3], [2., 2.5, 3]], dtype=dtype)
    strikes = tf.constant([[16, 22, 35], [16, 22, 35]], dtype=dtype)

    times_data = tf.constant([[1.5, 2.5, 3.5, 4.5, 5.5],
                              [1.2, 2.2, 3.5, 4.5, 5.5]], dtype=dtype)
    # Corresponding squared volatility values
    sigma_square_data = tf.constant(
        [[0.15, 0.25, 0.35, 0.4, 0.45, 0.4], [0.2, 0.35, 0.55, 0.45, 0.4, 0.6],
         [0.3, 0.45, 0.25, 0.4, 0.5, 0.65], [0.25, 0.25, 0.45, 0.25, 0.5, 0.55],
         [0.35, 0.35, 0.25, 0.4, 0.55, 0.65]],
        dtype=dtype)
    # Interpolation is done for the total variance
    total_variance = tf.expand_dims(times_data, -1) * sigma_square_data
    # Corresponding strike values. Notice we need to broadcast to the shape of
    # `sigma_square_data`
    strike_data = tf.broadcast_to(
        tf.constant([15, 25, 35, 40, 50, 55], dtype=dtype), [5, 6])
    sigma_square_data = tf.broadcast_to(sigma_square_data, [2, 5, 6])
    strike_data = tf.broadcast_to(strike_data, [2, 5, 6])
    # Interpolate total variance on the grid [times, strikes]
    interpolator = interpolation_2d.Interpolation2D(
        times_data, strike_data, total_variance, dtype=dtype)
    interpolated_values = interpolator.interpolate(times, strikes)

    with self.subTest("CorrectDtype"):
      self.assertEqual(interpolated_values.dtype.as_numpy_dtype, dtype)
    with self.subTest("CorrectShape"):
      self.assertAllClose(interpolated_values.shape.as_list(), [2, 3])
    expected_vols = np.array([[0.38239871, 0.71669375, 1.125],
                              [0.40785739, 0.85479298, 1.00384615]])
    with self.subTest("CorrectResult"):
      self.assertAllClose(
          interpolated_values, expected_vols, rtol=1e-04, atol=1e-04)

  def test_interpolation_in_y_direction(self):
    """Test interpolation in y direction (cubic interpolation)."""
    for dtype in [np.float32, np.float64]:
      x_values = tf.constant([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=dtype)
      y_values_list = [-1, 0, 1, 2, 2.5, 5, -1, 0, 1, 2, 2.5, 5]
      y_values = tf.constant(y_values_list, dtype=dtype)

      # Function at x=1.
      f1 = lambda x: x
      # Function as x=2
      f2 = lambda x: x**3

      # Interpolated cubic spline at x=1.
      def f1_interpolated(x):
        if x < 0:
          return 0
        elif x >= 0 and x <= 2:
          return x
        else:
          return 2

      # Interpolated cubic spline at x=2.
      # Since we use *natural* cubic interpolation the spline that we
      # get is not the same as the original cubic polynomial.
      def f2_interpolated(x):
        if x < 2:
          return 8
        elif x >= 2 and x <= 3:
          return 4.5 * x**3 - 27 * x**2 + 68.5 * x - 57
        elif x >= 3 and x <= 4:
          return -4.5 * x**3 + 54 * x**2 - 174.5 * x + 186
        else:
          return 64

      x_data = tf.constant([1, 2], dtype=dtype)
      y_data = tf.constant([[0, 1, 2], [2, 3, 4]], dtype=dtype)

      z_data = tf.constant(
          [[f1(0), f1(1), f1(2)], [f2(2), f2(3), f2(4)]], dtype=dtype)

      interpolator = interpolation_2d.Interpolation2D(
          x_data, y_data, z_data, dtype=dtype)
      result = interpolator.interpolate(x_values, y_values)

      with self.subTest("CorrectDtype"):
        self.assertEqual(result.dtype.as_numpy_dtype, dtype)
      with self.subTest("CorrectShape"):
        self.assertAllClose(result.shape.as_list(), [12])
      expected_result = np.array(
          [f1_interpolated(x) for x in y_values_list[:6]]
          + [f2_interpolated(x) for x in y_values_list[:6]])
      with self.subTest("CorrectResults"):
        self.assertAllClose(result, expected_result, rtol=1e-04, atol=1e-04)

  def test_interpolation_in_x_direction(self):
    """Test interpolation in x direction (linear interpolation)."""
    for dtype in [np.float32, np.float64]:
      x_values = tf.constant([0, 1.2, 3, 4, 5, 5], dtype=dtype)
      y_values_list = [-1, 0, 1, 2, 2.5, 5]
      y_values = tf.constant(y_values_list, dtype=dtype)

      # Function at x=1.
      f1 = lambda x: x
      # Function at x=2.
      f2 = lambda x: x**3

      # Interpolated cubic spline at x=1.
      def f1_interpolated(x):
        if x < 0:
          return 0
        elif x >= 0 and x <= 2:
          return x
        else:
          return 2

      # Interpolated cubic spline at x=2.
      # Since we use *natural* cubic interpolation the spline that we
      # get is not the same as the original cubic polynomial.
      def f2_interpolated(x):
        if x < 2:
          return 8
        elif x >= 2 and x <= 3:
          return 4.5 * x**3 - 27 * x**2 + 68.5 * x - 57
        elif x >= 3 and x <= 4:
          return -4.5 * x**3 + 54 * x**2 - 174.5 * x + 186
        else:
          return 64

      x_data = tf.constant([1, 2], dtype=dtype)
      y_data = tf.constant([[0, 1, 2], [2, 3, 4]], dtype=dtype)

      z_data = tf.constant(
          [[f1(0), f1(1), f1(2)], [f2(2), f2(3), f2(4)]], dtype=dtype)

      interpolator = interpolation_2d.Interpolation2D(
          x_data, y_data, z_data, dtype=dtype)
      result = interpolator.interpolate(x_values, y_values)

      with self.subTest("CorrectDtype"):
        self.assertEqual(result.dtype.as_numpy_dtype, dtype)
      with self.subTest("CorrectShape"):
        self.assertAllClose(result.shape.as_list(), [6])
      x0 = y_values_list[0]
      x1 = y_values_list[1]
      expected_result = np.array(
          [f1_interpolated(x0)]
          + [f1_interpolated(x1)
             + 0.2 * (f2_interpolated(x1) - f1_interpolated(x1))]
          +  [f2_interpolated(x) for x in y_values_list[2:]])
      with self.subTest("CorrectInterpolation"):
        self.assertAllClose(result, expected_result, rtol=1e-04, atol=1e-04)

  def test_grad(self):
    """Computes forward gradient values for the interpolate function."""
    dtype = np.float64
    times = tf.constant([2., 2.5, 3], dtype=dtype)
    strikes = tf.constant([16, 22, 35], dtype=dtype)

    times_data = tf.constant([1.5, 2.5, 3.5, 4.5, 5.5], dtype=dtype)
    sigma_square_data = tf.constant(
        [[0.15, 0.25, 0.35, 0.4, 0.45, 0.4],
         [0.2, 0.35, 0.55, 0.45, 0.4, 0.6],
         [0.3, 0.45, 0.25, 0.4, 0.5, 0.65],
         [0.25, 0.25, 0.45, 0.25, 0.5, 0.55],
         [0.35, 0.35, 0.25, 0.4, 0.55, 0.55]], dtype=dtype)
    total_variance = tf.expand_dims(times_data, -1) * sigma_square_data
    strike_data = tf.broadcast_to(
        tf.constant([15, 25, 35, 40, 50, 55], dtype=dtype), [5, 6])

    interpolator = interpolation_2d.Interpolation2D(
        times_data, strike_data, total_variance, dtype=dtype)
    def _times_fn(x):
      return interpolator.interpolate(x, strikes)
    def _strike_fn(x):
      return interpolator.interpolate(times, x)

    # Compute forward gradients with respect to `times` and `strikes`
    grad_times = tff.math.fwd_gradient(_times_fn, times)
    grad_strikes = tff.math.fwd_gradient(_strike_fn, strikes)

    # Expected values are computed with finite difference method
    expected_grad_times = np.array([0.284797, 0.386694, -0.5])
    expected_grad_strikes = np.array([0.02002702, 0.04353051, 0.01577941])
    self.assertAllClose(grad_times, expected_grad_times, rtol=1e-04, atol=1e-04)
    self.assertAllClose(
        grad_strikes, expected_grad_strikes, rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
  tf.test.main()
