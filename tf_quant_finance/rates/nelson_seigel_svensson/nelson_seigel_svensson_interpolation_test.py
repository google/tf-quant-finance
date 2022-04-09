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
"""Tests for nelson_svensson_interpolation."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class NelsonSvenssonInterpolationTest(tf.test.TestCase,
                                      parameterized.TestCase):

  def test_all_beta_0(self):
    # Due to theoretical properties, all output values should be 0 if all betas
    # are 0
    interpolation_times = [5., 10., 15., 20.]
    s_p = tff.rates.nelson_seigel_svensson.SvenssonParameters(
        beta_0=0.0, beta_1=0.0, beta_2=0.0, beta_3=0.0, tau_1=100.0, tau_2=10.0)

    output = self.evaluate(
        tff.rates.nelson_seigel_svensson.interpolate(interpolation_times, s_p))

    expected_output = [0.0, 0.0, 0.0, 0.0]

    np.testing.assert_allclose(output, expected_output)

  def test_custom_input(self):
    # Expected results computed using `nelson-seigel-svensson` package
    interpolation_times = [5., 10., 15., 20.]
    s_p = tff.rates.nelson_seigel_svensson.SvenssonParameters(
        beta_0=0.05, beta_1=-0.01, beta_2=0.3, beta_3=0.02,
        tau_1=1.5, tau_2=20.0)

    output = self.evaluate(
        tff.rates.nelson_seigel_svensson.interpolate(interpolation_times, s_p))

    expected_output = [0.12531409, 0.09667101, 0.08360796, 0.0770343]

    np.testing.assert_allclose(output, expected_output, atol=1e-5, rtol=1e-5)

  def test_batch_input(self):
    # Expected results computed using `nelson-seigel-svensson` package
    interpolation_times = [[1., 2., 3., 4.], [5., 10., 15., 20.],
                           [30., 40., 50., 60.]]
    s_p = tff.rates.nelson_seigel_svensson.SvenssonParameters(
        beta_0=0.25, beta_1=-0.4, beta_2=-0.02, beta_3=0.05,
        tau_1=4.5, tau_2=10)

    output = self.evaluate(
        tff.rates.nelson_seigel_svensson.interpolate(interpolation_times, s_p))

    expected_output = [[-0.10825214, -0.07188015, -0.04012282, -0.0123332],
                       [0.01203921, 0.09686097, 0.14394756, 0.1716945],
                       [0.20045316, 0.21411455, 0.22179659, 0.22668882]]

    np.testing.assert_allclose(output, expected_output, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  tf.test.main()
