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
"""Tests for math.piecewise."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_quant_finance.math import piecewise
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class Piecewise(tf.test.TestCase):
  """Tests for methods in piecewise module."""

  def test_find_interval_index_correct_dtype(self):
    """Tests find_interval_index outputs the correct type."""
    result = self.evaluate(piecewise.find_interval_index([1.0], [0.0, 1.0]))
    self.assertIsInstance(result[0], np.int32)

  def test_find_interval_index_one_interval(self):
    """Tests find_interval_index is correct with one half-open interval."""
    result = self.evaluate(piecewise.find_interval_index([1.0], [1.0]))
    self.assertAllEqual(result, [0])

    result = self.evaluate(piecewise.find_interval_index([0.0], [1.0]))
    self.assertAllEqual(result, [-1])

    result = self.evaluate(piecewise.find_interval_index([2.0], [1.0]))
    self.assertAllEqual(result, [0])

  def test_find_interval_index(self):
    """Tests find_interval_index is correct in the general case."""
    interval_lower_xs = [0.25, 0.5, 1.0, 2.0, 3.0]
    query_xs = [0.25, 3.0, 5.0, 0.0, 0.5, 0.8]
    result = piecewise.find_interval_index(query_xs, interval_lower_xs)
    self.assertAllEqual(result, [0, 4, 4, -1, 1, 1])

  def test_find_interval_index_last_interval_is_closed(self):
    """Tests find_interval_index is correct in the general case."""
    result = piecewise.find_interval_index([3.0, 4.0], [2.0, 3.0],
                                           last_interval_is_closed=True)
    self.assertAllEqual(result, [0, 1])


if __name__ == '__main__':
  tf.test.main()
