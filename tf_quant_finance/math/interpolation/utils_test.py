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

  def test_prepare_indices(self):
    indices = tf.zeros([5, 3, 5, 8])
    index_matrix = utils.prepare_indices(indices)
    self.assertAllEqual(index_matrix.shape,
                        indices.shape + [indices.shape.rank - 1])

if __name__ == "__main__":
  tf.test.main()
