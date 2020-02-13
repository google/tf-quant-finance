# Lint as: python3
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
"""Tests for the `utils` module."""

import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tf_quant_finance.models import utils
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class UtilsTest(tf.test.TestCase):

  def test_sobol_numbers_generation(self):
    """Sobol random dtype results in the correct draws."""
    for dtype in (tf.float32, tf.float64):
      num_draws = tf.constant(2, dtype=tf.int32)
      steps_num = tf.constant(3, dtype=tf.int32)
      num_samples = tf.constant(4, dtype=tf.int32)
      random_type = tff.math.random.RandomType.SOBOL
      skip = 10
      samples = utils.generate_mc_normal_draws(
          num_normal_draws=num_draws, num_time_steps=steps_num,
          num_sample_paths=num_samples, random_type=random_type,
          dtype=dtype, skip=skip)
      expected_samples = [[[0.8871465, 0.48877636],
                           [-0.8871465, -0.48877636],
                           [0.48877636, 0.8871465],
                           [-0.15731068, 0.15731068]],
                          [[0.8871465, -1.5341204],
                           [1.5341204, -0.15731068],
                           [-0.15731068, 1.5341204],
                           [-0.8871465, 0.48877636]],
                          [[-0.15731068, 1.5341204],
                           [0.15731068, -0.48877636],
                           [-1.5341204, 0.8871465],
                           [0.8871465, -1.5341204]]]
      self.assertAllClose(samples, expected_samples, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  tf.test.main()
