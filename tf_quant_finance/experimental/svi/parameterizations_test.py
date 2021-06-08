# Lint as: python3
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
"""Tests for svi.parameterizations."""

import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


# @test_util.run_all_in_graph_and_eager_modes
class SviParameterizationsTest(tf.test.TestCase):

  def test_volatility_from_raw_correctness(self):
    a = -0.02
    b = 0.15
    rho = 0.3
    m = 0.2
    sigma = 0.4

    parameters = tf.convert_to_tensor([a, b, rho, m, sigma], dtype=tf.float64)
    k = np.linspace(-1.25, 1.25, 11)

    actual = self.evaluate(
        tff.experimental.svi.total_variance_from_raw(parameters, k))
    expected = np.array([
        0.14037413, 0.11573666, 0.09186646, 0.06943387, 0.05006196, 0.03808204,
        0.04271693, 0.0685, 0.10676103, 0.15016408, 0.19579154
    ])
    self.assertArrayNear(actual, expected, 1e-8)


if __name__ == "__main__":
  tf.test.main()
