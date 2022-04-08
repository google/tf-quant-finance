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
"""Tests for lattice rules."""

import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

qmc = tff.math.qmc


@test_util.run_all_in_graph_and_eager_modes
class LatticeRuleTest(tf.test.TestCase):

  # Generating vectors for a lattice rule with n=2^20 points in 20 dimensions.
  generating_vectors_values = [
      1, 387275, 314993, 50301, 174023, 354905, 303021, 486111, 286797, 463237,
      211171, 216757, 29831, 155061, 315509, 193933, 129563, 276501, 395079,
      139111
  ]

  def generating_vectors(self, dtype=tf.int32):
    return tf.constant(self.generating_vectors_values, dtype=dtype)

  def test_random_scrambling_vectors(self):
    dim = 20
    seed = (2, 3)

    actual = qmc.random_scrambling_vectors(dim, seed, validate_args=True)

    with self.subTest('Shape'):
      self.assertEqual(actual.shape, (dim,))
    with self.subTest('DType'):
      self.assertEqual(actual.dtype, tf.float32)
    with self.subTest('Min Value'):
      self.assertAllLess(actual, tf.ones(shape=(), dtype=tf.float32))
    with self.subTest('Max Value'):
      self.assertAllGreaterEqual(actual, tf.zeros(shape=(), dtype=tf.float32))

  def test_random_scrambling_vectors_with_dtype(self):
    dim = 20
    seed = (2, 3)

    for dtype in [tf.float32, tf.float64]:
      actual = qmc.random_scrambling_vectors(
          dim, seed, dtype=dtype, validate_args=True)

      with self.subTest('Shape'):
        self.assertEqual(actual.shape, (dim,))
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, dtype)
      with self.subTest('Min Value'):
        self.assertAllLess(actual, tf.ones(shape=(), dtype=dtype))
      with self.subTest('Max Value'):
        self.assertAllGreaterEqual(actual, tf.zeros(shape=(), dtype=dtype))

  def test_lattice_rule_sample(self):

    expected = tf.constant([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0625, 0.6875, 0.0625, 0.8125, 0.4375, 0.5625],
                            [0.1250, 0.3750, 0.1250, 0.6250, 0.8750, 0.1250],
                            [0.1875, 0.0625, 0.1875, 0.4375, 0.3125, 0.6875],
                            [0.2500, 0.7500, 0.2500, 0.2500, 0.7500, 0.2500],
                            [0.3125, 0.4375, 0.3125, 0.0625, 0.1875, 0.8125],
                            [0.3750, 0.1250, 0.3750, 0.8750, 0.6250, 0.3750],
                            [0.4375, 0.8125, 0.4375, 0.6875, 0.0625, 0.9375],
                            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                            [0.5625, 0.1875, 0.5625, 0.3125, 0.9375, 0.0625],
                            [0.6250, 0.8750, 0.6250, 0.1250, 0.3750, 0.6250],
                            [0.6875, 0.5625, 0.6875, 0.9375, 0.8125, 0.1875],
                            [0.7500, 0.2500, 0.7500, 0.7500, 0.2500, 0.7500],
                            [0.8125, 0.9375, 0.8125, 0.5625, 0.6875, 0.3125],
                            [0.8750, 0.6250, 0.8750, 0.3750, 0.1250, 0.8750],
                            [0.9375, 0.3125, 0.9375, 0.1875, 0.5625, 0.4375]],
                           dtype=tf.float32)

    for dtype in [tf.int32, tf.int64]:
      actual = qmc.lattice_rule_sample(
          self.generating_vectors(dtype=dtype), 6, 16, validate_args=True)

      with self.subTest('Values'):
        self.assertAllClose(
            self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, expected.dtype)

  def test_lattice_rule_sample_with_sequence_indices(self):
    indices = [2, 3, 6, 9, 11, 14]

    expected = tf.constant([[0.1250, 0.3750, 0.1250, 0.6250, 0.8750, 0.1250],
                            [0.1875, 0.0625, 0.1875, 0.4375, 0.3125, 0.6875],
                            [0.3750, 0.1250, 0.3750, 0.8750, 0.6250, 0.3750],
                            [0.5625, 0.1875, 0.5625, 0.3125, 0.9375, 0.0625],
                            [0.6875, 0.5625, 0.6875, 0.9375, 0.8125, 0.1875],
                            [0.8750, 0.6250, 0.8750, 0.3750, 0.1250, 0.8750]],
                           dtype=tf.float32)

    actual = qmc.lattice_rule_sample(
        self.generating_vectors(),
        6,
        16,
        sequence_indices=tf.constant(indices, dtype=tf.int32),
        validate_args=True)

    with self.subTest('Values'):
      self.assertAllClose(
          self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
    with self.subTest('DType'):
      self.assertEqual(actual.dtype, expected.dtype)

  def test_lattice_rule_sample_with_zero_additive_shift(self):
    generating_vectors = self.generating_vectors()

    expected = tf.constant([[0.000, 0.000, 0.000, 0.000, 0.000],
                            [0.125, 0.375, 0.125, 0.625, 0.875],
                            [0.250, 0.750, 0.250, 0.250, 0.750],
                            [0.375, 0.125, 0.375, 0.875, 0.625],
                            [0.500, 0.500, 0.500, 0.500, 0.500],
                            [0.625, 0.875, 0.625, 0.125, 0.375],
                            [0.750, 0.250, 0.750, 0.750, 0.250],
                            [0.875, 0.625, 0.875, 0.375, 0.125]],
                           dtype=tf.float32)

    for dtype in [tf.float32, tf.float64]:
      actual = qmc.lattice_rule_sample(
          generating_vectors,
          5,
          8,
          additive_shift=tf.zeros_like(generating_vectors, dtype=dtype),
          validate_args=True)

      with self.subTest('Values'):
        self.assertAllClose(
            self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, expected.dtype)

  def test_lattice_rule_sample_with_non_zero_additive_shift(self):
    generating_vectors = self.generating_vectors()

    additive_shift = [
        .00, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65,
        .70, .75, .80, .85, .90, .95
    ]

    expected = tf.constant([[0.000, 0.050, 0.100, 0.150, 0.200],
                            [0.125, 0.425, 0.225, 0.775, 0.075],
                            [0.250, 0.800, 0.350, 0.400, 0.950],
                            [0.375, 0.175, 0.475, 0.025, 0.825],
                            [0.500, 0.550, 0.600, 0.650, 0.700],
                            [0.625, 0.925, 0.725, 0.275, 0.575],
                            [0.750, 0.300, 0.850, 0.900, 0.450],
                            [0.875, 0.675, 0.975, 0.525, 0.325]],
                           dtype=tf.float32)

    for dtype in [tf.float32, tf.float64]:
      actual = qmc.lattice_rule_sample(
          generating_vectors,
          5,
          8,
          additive_shift=tf.constant(additive_shift, dtype=dtype),
          validate_args=True)

      with self.subTest('Values'):
        self.assertAllClose(
            self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, expected.dtype)

  def test_lattice_rule_sample_with_tent_transform(self):

    expected = tf.constant([[0.000, 0.000, 0.000, 0.000, 0.000],
                            [0.250, 0.750, 0.250, 0.750, 0.250],
                            [0.500, 0.500, 0.500, 0.500, 0.500],
                            [0.750, 0.250, 0.750, 0.250, 0.750],
                            [1.000, 1.000, 1.000, 1.000, 1.000],
                            [0.750, 0.250, 0.750, 0.250, 0.750],
                            [0.500, 0.500, 0.500, 0.500, 0.500],
                            [0.250, 0.750, 0.250, 0.750, 0.250]],
                           dtype=tf.float32)

    actual = qmc.lattice_rule_sample(
        self.generating_vectors(),
        5,
        8,
        apply_tent_transform=True,
        validate_args=True)

    with self.subTest('Values'):
      self.assertAllClose(
          self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
    with self.subTest('DType'):
      self.assertEqual(actual.dtype, expected.dtype)

  def test_lattice_rule_sample_with_dtype(self):
    generating_vectors = self.generating_vectors()

    for dtype in [tf.float32, tf.float64]:
      expected = tf.constant([[0.000, 0.000, 0.000, 0.000, 0.000],
                              [0.125, 0.375, 0.125, 0.625, 0.875],
                              [0.250, 0.750, 0.250, 0.250, 0.750],
                              [0.375, 0.125, 0.375, 0.875, 0.625],
                              [0.500, 0.500, 0.500, 0.500, 0.500],
                              [0.625, 0.875, 0.625, 0.125, 0.375],
                              [0.750, 0.250, 0.750, 0.750, 0.250],
                              [0.875, 0.625, 0.875, 0.375, 0.125]],
                             dtype=dtype)

      actual = qmc.lattice_rule_sample(
          generating_vectors, 5, 8, validate_args=True, dtype=dtype)

      with self.subTest('Values'):
        self.assertAllClose(
            self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, dtype)


if __name__ == '__main__':
  tf.test.main()
