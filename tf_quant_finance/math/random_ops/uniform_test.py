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

"""Tests for uniform sampling."""


import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tff_rnd = tff.math.random


# TODO(b/145134423): Increase test coverage.
@test_util.run_all_in_graph_and_eager_modes
class UniformTest(tf.test.TestCase):

  def test_uniform(self):
    """Tests uniform pseudo random numbers."""
    sample_shape = [2, 3, 5000]
    dim = 10
    sample = self.evaluate(
        tff_rnd.uniform(dim=dim,
                        sample_shape=sample_shape,
                        seed=101))
    np.testing.assert_array_equal(sample.shape, sample_shape + [dim])
    expected_mean = 0.5 * np.ones(sample_shape[:-1] + [dim])
    np.testing.assert_array_almost_equal(
        np.mean(sample, axis=2), expected_mean, decimal=2)

  def test_stateless(self):
    """Tests stateless pseudo random numbers."""
    sample_shape = [2, 3, 5000]
    dim = 10
    sample = self.evaluate(
        tff_rnd.uniform(dim=dim,
                        sample_shape=sample_shape,
                        random_type=tff_rnd.RandomType.STATELESS,
                        seed=[2, 2]))
    np.testing.assert_array_equal(sample.shape, sample_shape + [dim])
    expected_mean = 0.5 * np.ones(sample_shape[:-1] + [dim])
    np.testing.assert_array_almost_equal(
        np.mean(sample, axis=2), expected_mean, decimal=2)

  def test_sobol(self):
    """Tests Sobol samples."""
    # The number of initial points of the Sobol sequence to skip
    skip = 1000
    for dtype in [np.float32, np.float64]:
      sample = tff_rnd.uniform(dim=5,
                               sample_shape=[100],
                               random_type=tff_rnd.RandomType.SOBOL,
                               skip=skip,
                               dtype=dtype)
      expected_samples = tff_rnd.sobol.sample(dim=5,
                                              num_results=100,
                                              skip=skip,
                                              dtype=dtype)
      self.assertAllClose(sample, expected_samples)
      self.assertEqual(sample.dtype.as_numpy_dtype, dtype)

  def test_halton_randomization_params(self):
    """Tests samples for the randomized Halton sequence."""
    dtype = np.float32
    num_samples = 10000
    # Set up randomization parameters (this is an optional arg)
    randomization_params = tff.math.random.halton.sample(
        2, num_samples, randomized=True, seed=42)[1]
    sample = self.evaluate(
        tff_rnd.uniform(dim=2,
                        sample_shape=[num_samples],
                        random_type=tff_rnd.RandomType.HALTON_RANDOMIZED,
                        randomization_params=randomization_params,
                        dtype=dtype))
    np.testing.assert_array_equal(sample.shape, [num_samples, 2])
    self.assertArrayNear(np.mean(sample, axis=0), [0.5, 0.5], 1e-2)

  def test_halton(self):
    """Tests samples for halton random numbers."""
    sample_shape = [2, 3, 1000]  # Need less samples than for the uniform case
    dim = 10
    sample = self.evaluate(
        tff_rnd.uniform(dim=dim,
                        sample_shape=sample_shape,
                        random_type=tff_rnd.RandomType.HALTON))
    np.testing.assert_array_equal(sample.shape, sample_shape + [dim])
    expected_mean = 0.5 * np.ones([dim])
    np.testing.assert_array_almost_equal(
        np.mean(sample, axis=(0, 1, 2)), expected_mean, decimal=3)
    expected_mean = 0.5 * np.ones(sample_shape[:-1] + [dim])
    np.testing.assert_array_almost_equal(
        np.mean(sample, axis=2), expected_mean, decimal=2)


if __name__ == '__main__':
  tf.test.main()
