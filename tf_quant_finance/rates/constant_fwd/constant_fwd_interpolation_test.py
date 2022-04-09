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
"""Tests for Constant Fwd Interpolation."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

from tf_quant_finance.rates.constant_fwd import constant_fwd_interpolation


@test_util.run_all_in_graph_and_eager_modes
class ConstantFwdInterpolationTest(tf.test.TestCase, parameterized.TestCase):

  def test_correctness(self):
    interpolation_times = [1., 3., 6., 7., 8., 15., 18., 25., 30.]

    reference_times = [0.0, 2.0, 6.0, 8.0, 18.0, 30.0]
    reference_yields = [0.01, 0.02, 0.015, 0.014, 0.02, 0.025]
    result = self.evaluate(
        constant_fwd_interpolation.interpolate(interpolation_times,
                                               reference_times,
                                               reference_yields))
    expected_result = np.array(
        [0.02, 0.0175, 0.015, 0.01442857, 0.014, 0.01904, 0.02, 0.0235, 0.025])
    np.testing.assert_allclose(result, expected_result, atol=1e-6)

  def test_extrapolation(self):
    interpolation_times = [0.5, 35.0]

    reference_times = [1.0, 2.0, 6.0, 8.0, 18.0, 30.0]
    reference_yields = [0.01, 0.02, 0.015, 0.014, 0.02, 0.025]
    result = self.evaluate(
        constant_fwd_interpolation.interpolate(interpolation_times,
                                               reference_times,
                                               reference_yields))
    expected_result = np.array([0.01, 0.025])
    np.testing.assert_allclose(result, expected_result, atol=1e-6)

  def test_batching(self):
    interpolation_times = [[1., 3., 6., 7.], [8., 15., 18., 25.]]

    reference_times = [[0.0, 2.0, 6.0, 8.0, 18.0, 30.0],
                       [0.0, 2.0, 6.0, 8.0, 18.0, 30.0]]
    reference_yields = [[0.01, 0.02, 0.015, 0.014, 0.02, 0.025],
                        [0.01, 0.02, 0.015, 0.014, 0.02, 0.025]]
    result = self.evaluate(
        constant_fwd_interpolation.interpolate(interpolation_times,
                                               reference_times,
                                               reference_yields))
    expected_result = np.array(
        [[0.02, 0.0175, 0.015, 0.01442857], [0.014, 0.01904, 0.02, 0.0235]])
    np.testing.assert_allclose(result, expected_result, atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'BroadcastInterpolationTimes',
          'interpolation_times': [1., 3., 6., 7.],
          'reference_times': [[0.0, 2.0, 6.0, 8.0, 18.0, 30.0],
                              [0.0, 2.5, 6.5, 8.5, 18.5, 30.5]],
          'reference_yields': [[0.01, 0.02, 0.015, 0.014, 0.02, 0.025]],
          'results': [[0.02, 0.0175, 0.015, 0.01442857],
                      [0.02, 0.01864583, 0.01526042, 0.01469643]],
      },
      {
          'testcase_name': 'BroadcastReferenceYields',
          'interpolation_times': [[1., 3., 6., 7.], [2., 4., 5., 7.]],
          'reference_times': [[0.0, 2.0, 6.0, 8.0, 18.0, 30.0],
                              [0.0, 2.5, 6.5, 8.5, 18.5, 30.5]],
          'reference_yields': [0.01, 0.02, 0.015, 0.014, 0.02, 0.025],
          'results': [[0.02, 0.0175, 0.015, 0.01442857],
                      [0.02, 0.01695312, 0.0159375, 0.01469643]],
      },
      {
          'testcase_name': 'BroadcastReferenceTimes',
          'interpolation_times': [[1., 3., 6., 7.], [2., 4., 5., 7.]],
          'reference_times': [0.0, 2.0, 6.0, 8.0, 18.0, 30.0],
          'reference_yields': [[0.01, 0.02, 0.015, 0.014, 0.02, 0.025],
                               [0.01, 0.02, 0.015, 0.014, 0.02, 0.025]],
          'results': [[0.02, 0.0175, 0.015, 0.01442857],
                      [0.02, 0.01625, 0.0155, 0.01442857]],})
  def test_batching_auto_broadcast(
      self, interpolation_times, reference_times, reference_yields, results):
    dtype = tf.float64
    result = self.evaluate(
        constant_fwd_interpolation.interpolate(
            interpolation_times, reference_times,
            reference_yields, dtype=dtype))
    expected_result = np.array(results)
    np.testing.assert_allclose(result, expected_result, atol=1e-6)

  def test_batching_and_extrapolation(self):
    interpolation_times = [[0.5, 1., 3., 6., 7.], [8., 15., 18., 25., 35.]]

    reference_times = [[1.0, 2.0, 6.0, 8.0, 18.0, 30.0],
                       [1.0, 2.0, 6.0, 8.0, 18.0, 30.0]]
    reference_yields = [[0.01, 0.02, 0.015, 0.014, 0.02, 0.025],
                        [0.005, 0.02, 0.015, 0.014, 0.02, 0.025]]
    result = self.evaluate(
        constant_fwd_interpolation.interpolate(interpolation_times,
                                               reference_times,
                                               reference_yields))
    expected_result = np.array(
        [[0.01, 0.01, 0.0175, 0.015, 0.01442857],
         [0.014, 0.01904, 0.02, 0.0235, 0.025]])
    np.testing.assert_allclose(result, expected_result, atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
