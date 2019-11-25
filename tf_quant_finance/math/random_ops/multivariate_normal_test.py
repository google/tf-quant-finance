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

# Lint as: python2, python3
"""Tests for random.multivariate_normal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tff_rnd = tff.math.random


@test_util.run_all_in_graph_and_eager_modes
class RandomTest(tf.test.TestCase):

  def test_shapes(self):
    """Tests the sample shapes."""
    sample_no_batch = self.evaluate(
        tff_rnd.mv_normal_sample([2, 4], mean=[0.2, 0.1]))
    self.assertEqual(sample_no_batch.shape, (2, 4, 2))
    sample_batch = self.evaluate(
        tff_rnd.mv_normal_sample(
            [2, 4], mean=[[0.2, 0.1], [0., -0.1], [0., 0.1]]))
    self.assertEqual(sample_batch.shape, (2, 4, 3, 2))

  def test_mean_default(self):
    """Tests that the default value of mean is 0."""
    covar = np.array([[1.0, 0.1], [0.1, 1.0]])
    sample = self.evaluate(
        tff_rnd.mv_normal_sample(
            [40000], covariance_matrix=covar, seed=1234))
    np.testing.assert_array_equal(sample.shape, [40000, 2])
    self.assertArrayNear(np.mean(sample, axis=0), [0.0, 0.0], 1e-2)
    self.assertArrayNear(
        np.cov(sample, rowvar=False).reshape([-1]), covar.reshape([-1]), 2e-2)

  def test_covariance_default(self):
    """Tests that the default value of the covariance matrix is identity."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0]])
    sample = self.evaluate(
        tff_rnd.mv_normal_sample([10000], mean=mean))

    np.testing.assert_array_equal(sample.shape, [10000, 2, 2])
    np.testing.assert_array_almost_equal(
        np.mean(sample, axis=0), mean, decimal=1)
    np.testing.assert_array_almost_equal(
        np.cov(sample[:, 0, :], rowvar=False), np.eye(2), decimal=1)
    np.testing.assert_array_almost_equal(
        np.cov(sample[:, 1, :], rowvar=False), np.eye(2), decimal=1)

  def test_general_mean_covariance(self):
    """Tests that the sample is correctly generated for general params."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0]])
    covar = np.array([
        [[0.9, -0.1], [-0.1, 1.0]],
        [[1.1, -0.3], [-0.3, 0.6]],
    ])
    size = 30000
    sample = self.evaluate(
        tff_rnd.mv_normal_sample(
            [size], mean=mean, covariance_matrix=covar, seed=4567))

    np.testing.assert_array_equal(sample.shape, [size, 2, 2])
    np.testing.assert_array_almost_equal(
        np.mean(sample, axis=0), mean, decimal=1)
    np.testing.assert_array_almost_equal(
        np.cov(sample[:, 0, :], rowvar=False), covar[0], decimal=1)
    np.testing.assert_array_almost_equal(
        np.cov(sample[:, 1, :], rowvar=False), covar[1], decimal=1)

  def test_mean_and_scale(self):
    """Tests sample for scale specification."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0]])
    scale = np.array([[0.4, -0.1], [0.22, 1.38]])

    covariance = np.matmul(scale, scale.transpose())
    size = 30000
    sample = self.evaluate(
        tff_rnd.mv_normal_sample(
            [size], mean=mean, scale_matrix=scale, seed=7534))

    np.testing.assert_array_equal(sample.shape, [size, 2, 2])
    np.testing.assert_array_almost_equal(
        np.mean(sample, axis=0), mean, decimal=1)
    np.testing.assert_array_almost_equal(
        np.cov(sample[:, 0, :], rowvar=False), covariance, decimal=1)
    np.testing.assert_array_almost_equal(
        np.cov(sample[:, 1, :], rowvar=False), covariance, decimal=1)

  def test_mean_default_sobol(self):
    """Tests that the default value of mean is 0."""
    covar = np.array([[1.0, 0.1], [0.1, 1.0]])
    # The number of initial points of the Sobol sequence to skip
    skip = 1000
    sample = self.evaluate(
        tff_rnd.mv_normal_sample(
            [10000], covariance_matrix=covar,
            random_type=tff_rnd.RandomType.SOBOL,
            skip=skip))
    np.testing.assert_array_equal(sample.shape, [10000, 2])
    self.assertArrayNear(np.mean(sample, axis=0), [0.0, 0.0], 1e-2)
    self.assertArrayNear(
        np.cov(sample, rowvar=False).reshape([-1]), covar.reshape([-1]), 2e-2)

  def test_mean_and_scale_sobol(self):
    """Tests sample for scale specification."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0], [2.0, 0.3], [0., 0.]])
    scale = np.array([[0.4, -0.1], [0.22, 1.38]])
    covariance = np.matmul(scale, scale.transpose())
    sample_shape = [2, 3, 5000]
    sample = self.evaluate(
        tff_rnd.mv_normal_sample(
            sample_shape, mean=mean, scale_matrix=scale,
            random_type=tff_rnd.RandomType.SOBOL))

    np.testing.assert_array_equal(sample.shape, sample_shape + [4, 2])
    np.testing.assert_array_almost_equal(
        np.mean(sample, axis=(0, 1, 2)), mean, decimal=1)
    for i in range(4):
      np.testing.assert_array_almost_equal(
          np.cov(sample[0, 1, :, i, :], rowvar=False), covariance, decimal=1)

  def test_mean_default_halton(self):
    """Tests that the default value of mean is 0."""
    covar = np.array([[1.0, 0.1], [0.1, 1.0]])
    # The number of initial points of the Sobol sequence to skip
    skip = 1000
    sample = self.evaluate(
        tff_rnd.mv_normal_sample(
            [10000], covariance_matrix=covar,
            random_type=tff_rnd.RandomType.HALTON_RANDOMIZED,
            skip=skip))
    np.testing.assert_array_equal(sample.shape, [10000, 2])
    self.assertArrayNear(np.mean(sample, axis=0), [0.0, 0.0], 1e-2)
    self.assertArrayNear(
        np.cov(sample, rowvar=False).reshape([-1]), covar.reshape([-1]), 2e-2)

  def test_mean_default_halton_randomization_params(self):
    """Tests that the default value of mean is 0."""
    dtype = np.float32
    covar = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=dtype)
    num_samples = 10000
    # Set up randomization parameters
    randomization_params = tff.math.random.halton.sample(
        2, num_samples, randomized=True, seed=42)[1]
    sample = self.evaluate(
        tff_rnd.mv_normal_sample(
            [num_samples], covariance_matrix=covar,
            random_type=tff_rnd.RandomType.HALTON_RANDOMIZED,
            randomization_params=randomization_params))
    np.testing.assert_array_equal(sample.shape, [num_samples, 2])
    self.assertArrayNear(np.mean(sample, axis=0), [0.0, 0.0], 1e-2)
    self.assertArrayNear(
        np.cov(sample, rowvar=False).reshape([-1]), covar.reshape([-1]), 2e-2)

  def test_mean_and_scale_halton(self):
    """Tests sample for scale specification."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0], [2.0, 0.3], [0., 0.]])
    scale = np.array([[0.4, -0.1], [0.22, 1.38]])
    covariance = np.matmul(scale, scale.transpose())
    sample_shape = [2, 3, 5000]
    sample = self.evaluate(
        tff_rnd.mv_normal_sample(
            sample_shape, mean=mean, scale_matrix=scale,
            random_type=tff_rnd.RandomType.HALTON))

    np.testing.assert_array_equal(sample.shape, sample_shape + [4, 2])
    np.testing.assert_array_almost_equal(
        np.mean(sample, axis=(0, 1, 2)), mean, decimal=1)
    for i in range(4):
      np.testing.assert_array_almost_equal(
          np.cov(sample[0, 2, :, i, :], rowvar=False), covariance, decimal=1)

  def test_mean_and_scale_antithetic(self):
    """Tests antithetic sampler for scale specification."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0]])
    scale = np.array([[0.4, -0.1], [0.22, 1.38]])

    covariance = np.matmul(scale, scale.transpose())
    size = 30000
    sample = self.evaluate(
        tff_rnd.mv_normal_sample(
            [size], mean=mean, scale_matrix=scale,
            random_type=tff_rnd.RandomType.PSEUDO_ANTITHETIC, seed=42))
    np.testing.assert_array_equal(sample.shape, [size, 2, 2])
    # Antithetic combination of samples should be equal to the `mean`
    antithetic_size = size // 2
    antithetic_combination = (sample[:antithetic_size, ...]
                              + sample[antithetic_size:, ...]) / 2
    np.testing.assert_allclose(
        antithetic_combination,
        mean + np.zeros([antithetic_size, 2, 2]), 1e-10, 1e-10)
    # Get the antithetic pairs and verify normality
    np.testing.assert_array_almost_equal(
        np.mean(sample[:antithetic_size, ...], axis=0), mean, decimal=1)
    np.testing.assert_array_almost_equal(
        np.cov(sample[:antithetic_size, 0, :], rowvar=False),
        covariance, decimal=1)
    np.testing.assert_array_almost_equal(
        np.cov(sample[:antithetic_size, 1, :], rowvar=False),
        covariance, decimal=1)

  def test_antithetic_sample_requires_even_dim(self):
    """Error is trigerred if the first dim of sample_shape is odd."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0]])
    scale = np.array([[0.4, -0.1], [0.22, 1.38]])
    sample_shape = [11, 100]
    # Should fail: The first dimension of `sample_shape` should be even.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          tff_rnd.mv_normal_sample(
              sample_shape, mean=mean, scale_matrix=scale,
              random_type=tff_rnd.RandomType.PSEUDO_ANTITHETIC))

if __name__ == '__main__':
  tf.test.main()
