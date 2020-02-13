# Lint as: python3
"""Tests for Constant Fwd Interpolation."""

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance.rates.constant_fwd import constant_fwd_interpolation

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class ConstantFwdInterpolationTest(tf.test.TestCase):

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
