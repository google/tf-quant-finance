"""Tests for interpolation utilities."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.math.interpolation import utils


@test_util.run_all_in_graph_and_eager_modes
class UtilsTest(tf.test.TestCase):

  def test_broadcast_batch_shape(self):
    x = tf.zeros([1, 5, 3, 7])
    batch_shape = [2, 4, 5, 3]
    x = utils.broadcast_batch_shape(x, batch_shape)
    y = self.evaluate(x)
    self.assertAllEqual(y, np.zeros([2, 4, 5, 3, 7]))

  def test_expand_rank(self):
    x = tf.zeros([3, 2])
    x = utils.expand_to_rank(x, 4)
    y = self.evaluate(x)
    self.assertAllEqual(y, np.zeros([1, 1, 3, 2]))

  def test_broadcast_common_batch_shape(self):
    x = tf.zeros([3, 4])
    y = tf.zeros([2, 1, 3, 4])
    x, y = utils.broadcast_common_batch_shape(x, y)
    with self.subTest("ShapeX"):
      x_eval = self.evaluate(x)
      self.assertAllEqual(x_eval, np.zeros([2, 1, 3, 4]))
    with self.subTest("ShapeY"):
      y_eval = self.evaluate(y)
      self.assertAllEqual(y_eval, np.zeros([2, 1, 3, 4]))

  def test_broadcast_common_batch_shape_dynamic(self):
    """Test broadcasting with dynamic shapes."""
    @tf.function(input_signature=[tf.TensorSpec([2, 3, None]),
                                  tf.TensorSpec([None, 5])])
    def fn(x, y):
      return utils.broadcast_common_batch_shape(x, y)

    x = tf.random.uniform(shape=[2, 3, 10])
    y = tf.random.uniform(shape=[1, 5])
    x_broadcasted, y_broadcasted = fn(x, y)
    with self.subTest("ShapeX"):
      x_eval = self.evaluate(x_broadcasted)
      self.assertAllEqual(x_eval.shape, [2, 3, 10])
    with self.subTest("ShapeY"):
      y_eval = self.evaluate(y_broadcasted)
      self.assertAllEqual(y_eval.shape, [2, 3, 5])

if __name__ == "__main__":
  tf.test.main()
