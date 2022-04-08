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
"""Payoff function tests."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

lsm_algorithm = tff.models.longstaff_schwartz

# See Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
# simulation: a simple least-squares approach.
_SAMPLES = [[1.0, 1.09, 1.08, 1.34],
            [1.0, 1.16, 1.26, 1.54],
            [1.0, 1.22, 1.07, 1.03],
            [1.0, 0.93, 0.97, 0.92],
            [1.0, 1.11, 1.56, 1.52],
            [1.0, 0.76, 0.77, 0.90],
            [1.0, 0.92, 0.84, 1.01],
            [1.0, 0.88, 1.22, 1.34]]


@test_util.run_all_in_graph_and_eager_modes
class PayoffTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_put_payoff_function(self, dtype):
    """Tests the put payoff function for a batch of strikes."""
    # Create payoff functions for 2 different strike values
    payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1, 1.2], dtype=dtype)
    sample_paths = tf.convert_to_tensor(_SAMPLES, dtype=dtype)
    sample_paths = tf.expand_dims(sample_paths, -1)
    # Actual payoff
    payoff = payoff_fn(sample_paths, 3)
    # Expected payoffs at the final time
    expected_payoff = [[0, 0],
                       [0, 0],
                       [0.07, 0.17],
                       [0.18, 0.28],
                       [0, 0],
                       [0.2, 0.3],
                       [0.09, 0.19],
                       [0, 0]]
    self.assertAllClose(expected_payoff, payoff, rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_put_payoff_function_batch(self, dtype):
    """Tests the put payoff function for a batch of samples."""
    # Create payoff functions for 2 different strike values
    payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1, 1.3], dtype=dtype)
    # A batch of sample paths
    sample_paths = tf.convert_to_tensor(_SAMPLES, dtype=dtype)
    sample_paths1 = tf.expand_dims(sample_paths, -1)
    sample_paths2 = sample_paths1 + 0.1
    sample_paths = tf.stack([sample_paths1, sample_paths2], axis=0)
    # Actual payoff
    payoff = payoff_fn(sample_paths, 3)
    # Expected payoffs at the final time
    expected_payoff = [[0, 0],
                       [0, 0],
                       [0.07, 0.17],
                       [0.18, 0.28],
                       [0, 0],
                       [0.2, 0.3],
                       [0.09, 0.19],
                       [0, 0]]
    self.assertAllClose(expected_payoff, payoff, rtol=1e-6, atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
