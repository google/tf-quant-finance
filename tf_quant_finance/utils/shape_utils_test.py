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
"""Tests for the shape utilitis."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class ShapeUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'BoolTensor',
          'x': [[True], [False]],
          'expected_shape': [2, 1]
      }, {
          'testcase_name': 'RealTensor',
          'x': [[1], [2]],
          'expected_shape': [2, 1]
      },
  )
  def test_prefer_static_shape(self, x, expected_shape):
    shape = tff.utils.get_shape(x)
    self.assertAllEqual(shape.as_list(), expected_shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'PartiallyKnown',
          'shape': [1, None],
      }, {
          'testcase_name': 'Unkown',
          'shape': None,
      },
  )
  def test_prefer_static_shape_dynamic(self, shape):
    x = tf.ones([1, 2], dtype=tf.float64)
    @tf.function(input_signature=[tf.TensorSpec(shape, dtype=x.dtype)])
    def fn(x):
      return tff.utils.get_shape(x)
    shape = self.evaluate(fn(x))

    self.assertAllEqual(shape, x.shape.as_list())

  def test_broadcast_tensors_shapes(self):
    args = [tf.ones([1, 2], dtype=tf.float64),
            tf.constant([[True], [False]]),
            tf.zeros([1], dtype=tf.float32)]
    @tf.function(input_signature=[
        tf.TensorSpec([1, None], dtype=tf.float64),
        tf.TensorSpec([2, 1], dtype=tf.bool),
        tf.TensorSpec(None, dtype=tf.float32)])
    def fn(x, y, z):
      return tff.utils.broadcast_tensors(x, y, z)
    x, y, z = self.evaluate(fn(*args))

    with self.subTest('Arg1Dtype'):
      self.assertAllEqual(x.dtype, tf.float64)
    with self.subTest('Arg1Value'):
      self.assertAllEqual(x, [[1, 1], [1, 1]])
    with self.subTest('Arg2Dtype'):
      self.assertAllEqual(y.dtype, tf.bool)
    with self.subTest('Arg2Value'):
      self.assertAllEqual(y, [[True, True], [False, False]])
    with self.subTest('Arg3Dtype'):
      self.assertAllEqual(z.dtype, tf.float32)
    with self.subTest('Arg3Value'):
      self.assertAllEqual(z, [[0, 0], [0, 0]])

  def test_broadcast_tensors_shapes_incompatible(self):
    args = [tf.ones([1, 2], dtype=tf.float64),
            tf.zeros([3, 3], dtype=tf.float32)]
    with self.assertRaises(ValueError):
      tff.utils.broadcast_tensors(*args)

  @parameterized.named_parameters(
      {
          'testcase_name': 'StaticShapedInputs',
          'dynamic': False,
          'input_signature': None
      }, {
          'testcase_name': 'DynamicShapedInputs1',
          'dynamic': True,
          'input_signature': [
              tf.TensorSpec([1, None], dtype=tf.float64),
              tf.TensorSpec([2, 1], dtype=tf.bool),
              tf.TensorSpec(None, dtype=tf.float32)]
      }, {
          'testcase_name': 'DynamicShapedInputs2',
          'dynamic': True,
          'input_signature': [
              tf.TensorSpec([1, None], dtype=tf.float64),
              tf.TensorSpec([2, 1], dtype=tf.bool),
              tf.TensorSpec([1], dtype=tf.float32)]
      },
  )
  def test_common_shape(self, dynamic, input_signature):
    args = [tf.ones([1, 2], dtype=tf.float64),
            tf.constant([[True], [False]]),
            tf.zeros([1], dtype=tf.float32)]
    def fn(x, y, z):
      return tff.utils.common_shape(x, y, z)
    if dynamic:
      fn = tf.function(fn, input_signature=input_signature)
    shape = fn(*args)
    self.assertAllEqual(shape, [2, 2])

  def test_common_shape_incompatible(self):
    args = [tf.ones([1, 2], dtype=tf.float64),
            tf.zeros([3, 3], dtype=tf.float32)]
    with self.assertRaises(ValueError):
      tff.utils.common_shape(*args)

  def test_broadcast_common_batch_shape(self):
    x = tf.zeros([3, 4])
    y = tf.zeros([2, 1, 3, 10])
    z = tf.zeros([])
    x, y, z = tff.utils.broadcast_common_batch_shape(x, y, z)
    with self.subTest('ShapeX'):
      self.assertAllEqual(x, np.zeros([2, 1, 3, 4]))
    with self.subTest('ShapeY'):
      self.assertAllEqual(y, np.zeros([2, 1, 3, 10]))
    with self.subTest('ShapeZ'):
      self.assertAllEqual(z, np.zeros([2, 1, 3]))

  def test_broadcast_common_batch_shape_different_ranks(self):
    x = tf.zeros([3, 4])
    y = tf.zeros([2, 1, 3, 2, 2])
    z = tf.zeros([])
    x, y, z = tff.utils.broadcast_common_batch_shape(x, y, z,
                                                     event_ranks=[1, 2, 0])
    with self.subTest('ShapeX'):
      self.assertAllEqual(x, np.zeros([2, 1, 3, 4]))
    with self.subTest('ShapeY'):
      self.assertAllEqual(y, np.zeros([2, 1, 3, 2, 2]))
    with self.subTest('ShapeZ'):
      self.assertAllEqual(z, np.zeros([2, 1, 3]))

  def test_broadcast_common_batch_shape_dynamic(self):
    """Test broadcasting with dynamic shapes."""
    @tf.function(input_signature=[tf.TensorSpec([2, 3, None]),
                                  tf.TensorSpec([None, 5])])
    def fn(x, y):
      return tff.utils.broadcast_common_batch_shape(x, y)

    x = tf.random.uniform(shape=[2, 3, 10])
    y = tf.random.uniform(shape=[1, 5])
    x_broadcasted, y_broadcasted = fn(x, y)
    with self.subTest('ShapeX'):
      x_eval = self.evaluate(x_broadcasted)
      self.assertAllEqual(x_eval.shape, [2, 3, 10])
    with self.subTest('ShapeY'):
      y_eval = self.evaluate(y_broadcasted)
      self.assertAllEqual(y_eval.shape, [2, 3, 5])

if __name__ == '__main__':
  tf.test.main()
