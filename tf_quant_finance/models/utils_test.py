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

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.models import utils


@test_util.run_all_in_graph_and_eager_modes
class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('SinglePrecision', np.float32),
      ('DoublePrecision', np.float64),
  )
  def test_sobol_numbers_generation(self, dtype):
    """Sobol random dtype results in the correct draws."""
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

  @parameterized.named_parameters(
      ('SinglePrecision', np.float32),
      ('DoublePrecision', np.float64),
  )
  def test_maybe_update_along_axis(self, dtype):
    """Tests that the values are updated correctly."""
    tensor = tf.ones([5, 4, 3, 2], dtype=dtype)
    new_tensor = tf.zeros([5, 4, 1, 2], dtype=dtype)
    @tf.function
    def maybe_update_along_axis(do_update):
      return utils.maybe_update_along_axis(
          tensor=tensor, new_tensor=new_tensor, axis=1, ind=2,
          do_update=do_update)
    updated_tensor = maybe_update_along_axis(True)
    with self.subTest(name='Shape'):
      self.assertEqual(updated_tensor.shape, tensor.shape)
    with self.subTest(name='UpdatedVals'):
      self.assertAllEqual(updated_tensor[:, 2, :, :],
                          tf.zeros_like(updated_tensor[:, 2, :, :]))
    with self.subTest(name='NotUpdatedVals'):
      self.assertAllEqual(updated_tensor[:, 1, :, :],
                          tf.ones_like(updated_tensor[:, 2, :, :]))
    with self.subTest(name='DoNotUpdateVals'):
      not_updated_tensor = maybe_update_along_axis(False)
      self.assertAllEqual(not_updated_tensor, tensor)

  def test_block_diagonal_to_dense(self):
    matrices = [[[1.0, 0.1], [0.1, 1.0]],
                [[1.0, 0.3, 0.2],
                 [0.3, 1.0, 0.5],
                 [0.2, 0.5, 1.0]], [[1.0]]]
    dense = utils.block_diagonal_to_dense(*matrices)
    expected_result = [[1.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                       [0.1, 1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.3, 0.2, 0.0],
                       [0.0, 0.0, 0.3, 1.0, 0.5, 0.0],
                       [0.0, 0.0, 0.2, 0.5, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    self.assertAllClose(dense, expected_result, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ('SinglePrecision', np.float32),
      ('DoublePrecision', np.float64),
  )
  def test_prepare_grid_num_time_step(self, dtype):
    num_points = 100
    times = tf.linspace(tf.constant(0.02, dtype=dtype), 1.0, 50)
    time_step = times[-1] / num_points
    grid, _, time_indices = utils.prepare_grid(
        times=times, time_step=time_step, dtype=dtype,
        num_time_steps=num_points)
    expected_grid = np.linspace(0, 1, 101, dtype=dtype)
    expected_indices = np.array([2 * i for i in range(1, 51)])
    with self.subTest('Grid'):
      self.assertAllClose(grid, expected_grid, rtol=1e-6, atol=1e-6)
    with self.subTest('TimeIndex'):
      self.assertAllClose(time_indices, expected_indices, rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(
      ('SinglePrecision', np.float32),
      ('DoublePrecision', np.float64),
  )
  def test_prepare_grid_time_step(self, dtype):
    times = tf.constant([0.1, 0.5, 1, 2], dtype=dtype)
    time_step = tf.constant(0.1, dtype=tf.float64)
    grid, _, time_indices = utils.prepare_grid(
        times=times, time_step=time_step, dtype=dtype)
    expected_grid = np.linspace(0, 2, 21, dtype=dtype)
    recovered_times = tf.gather(grid, time_indices)
    with self.subTest('Grid'):
      self.assertAllClose(grid, expected_grid, rtol=1e-6, atol=1e-6)
    with self.subTest('Times'):
      self.assertAllClose(times, recovered_times, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  tf.test.main()
