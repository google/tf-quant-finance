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
"""Tests for random.stateless."""


import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tff_rnd = tff.math.random


class StatelessRandomOpsTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testOutputIsPermutation(self):
    """Checks that stateless_random_shuffle outputs a permutation."""
    for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
      identity_permutation = tf.range(10, dtype=dtype)
      random_shuffle_seed_1 = tff_rnd.stateless_random_shuffle(
          identity_permutation, seed=tf.constant((1, 42), tf.int64))
      random_shuffle_seed_2 = tff_rnd.stateless_random_shuffle(
          identity_permutation, seed=tf.constant((2, 42), tf.int64))
      # Check that the shuffles are of the correct dtype
      for shuffle in (random_shuffle_seed_1, random_shuffle_seed_2):
        np.testing.assert_equal(shuffle.dtype, dtype.as_numpy_dtype)
      random_shuffle_seed_1 = self.evaluate(random_shuffle_seed_1)
      random_shuffle_seed_2 = self.evaluate(random_shuffle_seed_2)
      identity_permutation = self.evaluate(identity_permutation)
      # Check that the shuffles are different
      self.assertTrue(
          np.abs(random_shuffle_seed_1 - random_shuffle_seed_2).max())
      # Check that the shuffles are indeed permutations
      for shuffle in (random_shuffle_seed_1, random_shuffle_seed_2):
        self.assertAllEqual(set(shuffle), set(identity_permutation))

  @test_util.run_in_graph_and_eager_modes
  def testOutputIsStateless(self):
    """Checks that stateless_random_shuffle is stateless."""
    random_permutation_next_call = None
    for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
      random_permutation = tff_rnd.stateless_random_shuffle(
          tf.range(10, dtype=dtype), seed=(100, 42))
      random_permutation_first_call = self.evaluate(random_permutation)
      if random_permutation_next_call is not None:
        # Checks that the values are the same across different dtypes
        np.testing.assert_array_equal(random_permutation_first_call,
                                      random_permutation_next_call)
      random_permutation_next_call = self.evaluate(random_permutation)
      np.testing.assert_array_equal(random_permutation_first_call,
                                    random_permutation_next_call)

  @test_util.run_in_graph_and_eager_modes
  def testOutputIsIndependentOfInputValues(self):
    """stateless_random_shuffle output is independent of input_tensor values."""
    # Generate sorted array of random numbers to control that the result
    # is independent of `input_tesnor` values
    np.random.seed(25)
    random_input = np.random.normal(size=[10])
    random_input.sort()
    for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
      # Permutation of a sequence [0, 1, .., 9]
      random_permutation = tff_rnd.stateless_random_shuffle(
          tf.range(10, dtype=dtype), seed=(100, 42))
      random_permutation = self.evaluate(random_permutation)
      # Shuffle `random_input` with the same seed
      random_shuffle_control = tff_rnd.stateless_random_shuffle(
          random_input, seed=(100, 42))
      random_shuffle_control = self.evaluate(random_shuffle_control)
      # Checks that the generated permutation does not depend on the underlying
      # values
      np.testing.assert_array_equal(
          np.argsort(random_permutation), np.argsort(random_shuffle_control))

  @test_util.run_v1_only("Sessions are not available in TF2.0")
  def testOutputIsStatelessSession(self):
    """Checks that stateless_random_shuffle is stateless across Sessions."""
    random_permutation_next_call = None
    for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
      random_permutation = tff_rnd.stateless_random_shuffle(
          tf.range(10, dtype=dtype), seed=tf.constant((100, 42), tf.int64))
      with tf.compat.v1.Session() as sess:
        random_permutation_first_call = sess.run(random_permutation)
      if random_permutation_next_call is not None:
        # Checks that the values are the same across different dtypes
        np.testing.assert_array_equal(random_permutation_first_call,
                                      random_permutation_next_call)
      with tf.compat.v1.Session() as sess:
        random_permutation_next_call = sess.run(random_permutation)
      np.testing.assert_array_equal(random_permutation_first_call,
                                    random_permutation_next_call)

  @test_util.run_in_graph_and_eager_modes
  def testMultiDimensionalShape(self):
    """Check that stateless_random_shuffle works with multi-dim shapes."""
    for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
      input_permutation = tf.constant([[[1], [2], [3]], [[4], [5], [6]]],
                                      dtype=dtype)
      random_shuffle = tff_rnd.stateless_random_shuffle(
          input_permutation, seed=(1, 42))
      random_permutation_first_call = self.evaluate(random_shuffle)
      random_permutation_next_call = self.evaluate(random_shuffle)
      input_permutation = self.evaluate(input_permutation)
      # Check that the dtype is correct
      np.testing.assert_equal(random_permutation_first_call.dtype,
                              dtype.as_numpy_dtype)
      # Check that the shuffles are the same
      np.testing.assert_array_equal(random_permutation_first_call,
                                    random_permutation_next_call)
      # Check that the output shape is correct
      np.testing.assert_equal(random_permutation_first_call.shape,
                              input_permutation.shape)


if __name__ == "__main__":
  tf.test.main()
