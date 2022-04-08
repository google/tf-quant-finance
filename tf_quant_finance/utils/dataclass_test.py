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
"""Tests for the DataClass decorator."""

import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class DataclassTest(tf.test.TestCase):

  def test_docstring_example(self):
    @tff.utils.dataclass
    class Coords:
      x: tf.Tensor
      y: tf.Tensor

    @tf.function
    def fn(start_coords: Coords) -> Coords:
      def cond(it, _):
        return it < 10
      def body(it, coords):
        return it + 1, Coords(x=coords.x + 1, y=coords.y + 2)
      return tf.while_loop(cond, body, loop_vars=(0, start_coords))[1]

    start_coords = Coords(x=tf.constant(0), y=tf.constant(0))
    end_coords = fn(start_coords)
    with self.subTest('OutputType'):
      self.assertIsInstance(end_coords, Coords)
    end_coords_eval = self.evaluate(end_coords)
    with self.subTest('FirstValue'):
      self.assertEqual(end_coords_eval.x, 10)
    with self.subTest('SecondValue'):
      self.assertEqual(end_coords_eval.y, 20)

  def test_docstring_preservation(self):
    @tff.utils.dataclass
    class Coords:
      """A coordinate grid."""
      x: tf.Tensor
      y: tf.Tensor

    self.assertEqual(Coords.__doc__, 'A coordinate grid.')

if __name__ == '__main__':
  tf.test.main()
