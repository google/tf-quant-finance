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
"""Tests for spatial grids."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from tf_quant_finance.math.pde import grids
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class GridsTest(tf.test.TestCase):

  def test_uniform_grid_1d(self):
    dtype = np.float64
    min_x, max_x, size = dtype(0.0), dtype(1.0), 21
    np_grid = np.linspace(min_x, max_x, num=size)
    grid_spec = self.evaluate(
        grids.uniform_grid([min_x], [max_x], [size], dtype=dtype))
    self.assertEqual(grid_spec.dim, 1)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([21, 1]))
    self.assertArrayNear(np.squeeze(grid), np_grid, 1e-10)
    self.assertEqual(grid.dtype, dtype)

  def test_uniform_grid_2d(self):
    dtype = np.float64
    min_x, max_x, sizes = [0.0, 1.0], [3.0, 5.0], [11, 21]
    grid_spec = self.evaluate(
        grids.uniform_grid(min_x, max_x, sizes, dtype=dtype))
    self.assertEqual(grid_spec.dim, 2)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([11, 21, 2]))
    self.assertEqual(grid.dtype, dtype)
    grid_marks = []
    for i in range(2):
      grid_marks.append(np.linspace(min_x[i], max_x[i], num=sizes[i]))
    np_grid = np.stack(np.meshgrid(*grid_marks, indexing='ij'), axis=-1)
    self.assertArrayNear(grid.reshape([-1]), np_grid.reshape([-1]), 1e-10)
    locs = grid_spec.locations
    self.assertEqual(len(locs), 2)
    self.assertArrayNear(locs[0], grid_marks[0], 1e-10)
    self.assertArrayNear(locs[1], grid_marks[1], 1e-10)
    dxs = grid_spec.deltas
    self.assertEqual(len(dxs), 2)
    self.assertArrayNear(dxs[0].reshape([-1]), [0.3], 1e-10)
    self.assertArrayNear(dxs[1].reshape([-1]), [0.2], 1e-10)

  def test_log_uniform_grid_1d(self):
    dtype = np.float64
    min_x, max_x, size = dtype(0.1), dtype(1.0), 21
    log_np_grid = np.linspace(np.log(min_x), np.log(max_x), num=size)
    np_grid = np.exp(log_np_grid)
    np_deltas = np_grid[1:] - np_grid[:-1]
    grid_spec = self.evaluate(
        grids.log_uniform_grid([min_x], [max_x], [size], dtype=dtype))
    self.assertEqual(grid_spec.dim, 1)
    grid = grid_spec.grid
    deltas = grid_spec.deltas
    self.assertEqual(grid.shape, tuple([21, 1]))
    self.assertArrayNear(np.squeeze(grid), np_grid, 1e-10)
    self.assertArrayNear(np.squeeze(deltas), np_deltas, 1e-10)
    self.assertEqual(grid.dtype, dtype)

  def test_log_uniform_grid_2d(self):
    dtype = np.float64
    min_x, max_x, sizes = [0.1, 1.0], [3.0, 5.0], [11, 21]
    grid_spec = self.evaluate(
        grids.log_uniform_grid(min_x, max_x, sizes, dtype=dtype))
    self.assertEqual(grid_spec.dim, 2)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([11, 21, 2]))
    self.assertEqual(grid.dtype, dtype)
    grid_marks = []
    for i in range(2):
      grid_marks.append(
          np.exp(np.linspace(np.log(min_x[i]), np.log(max_x[i]), num=sizes[i])))
    np_grid = np.stack(np.meshgrid(*grid_marks, indexing='ij'), axis=-1)
    self.assertArrayNear(grid.reshape([-1]), np_grid.reshape([-1]), 1e-10)
    locs = grid_spec.locations
    self.assertEqual(len(locs), 2)
    self.assertArrayNear(locs[0], grid_marks[0], 1e-10)
    self.assertArrayNear(locs[1], grid_marks[1], 1e-10)
    deltas = grid_spec.deltas
    np_deltas = [marks[1:] - marks[:-1] for marks in grid_marks]
    self.assertEqual(len(deltas), 2)
    for d, np_d in zip(deltas, np_deltas):
      self.assertArrayNear(d, np_d, 1e-10)

  def test_rectangular_grid_1d(self):
    axis_locations = [[0.1, 0.4, 0.8, 1.0, 1.1]]
    grid_spec = self.evaluate(
        grids.rectangular_grid(axis_locations, dtype=np.float64))
    self.assertEqual(grid_spec.dim, 1)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([5, 1]))
    self.assertArrayNear(np.squeeze(grid), np.squeeze(axis_locations), 1e-10)
    self.assertEqual(grid.dtype, np.float64)
    self.assertArrayNear(grid_spec.maximums, [1.1], 1e-10)
    self.assertArrayNear(grid_spec.minimums, [0.1], 1e-10)
    self.assertArrayNear(grid_spec.sizes, [5], 1e-10)

  def test_rectangular_grid_2d(self):
    dtype = np.float64
    axis_locations = [[0.0, 1.0, 2.0, 4.0], [-2.0, -1.9, 1.1]]
    grid_spec = self.evaluate(
        grids.rectangular_grid(axis_locations, dtype=dtype))
    self.assertEqual(grid_spec.dim, 2)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([4, 3, 2]))
    self.assertEqual(grid.dtype, dtype)
    np_grid = np.stack(np.meshgrid(*axis_locations, indexing='ij'), axis=-1)
    self.assertArrayNear(grid.reshape([-1]), np_grid.reshape([-1]), 1e-10)
    locs = grid_spec.locations
    self.assertEqual(len(locs), 2)
    self.assertArrayNear(locs[0], axis_locations[0], 1e-10)
    self.assertArrayNear(locs[1], axis_locations[1], 1e-10)
    dxs = grid_spec.deltas
    self.assertEqual(len(dxs), 2)
    self.assertArrayNear(dxs[0].reshape([-1]), [1.0, 1.0, 2.0], 1e-10)
    self.assertArrayNear(dxs[1].reshape([-1]), [0.1, 3.0], 1e-10)


class GridsExtraLocationTest(tf.test.TestCase):

  def test_uniform_grid_1d(self):
    dtype = np.float64
    # Batch of extra locations
    extra_locations = [[1], [7]]
    min_x, max_x, size = dtype(0.0), dtype(10.0), 3
    np_grid = np.array([[[0.0], [1.0], [5.0], [10.0]],
                        [[0.0], [5.0], [7.0], [10.0]]])
    grid_spec = self.evaluate(
        grids.uniform_grid_with_extra_point([min_x], [max_x], [size],
                                            extra_grid_point=extra_locations,
                                            dtype=dtype))
    self.assertEqual(grid_spec.dim, 1)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([2, 4, 1]))
    np.testing.assert_almost_equal(grid, np_grid, 10)
    self.assertEqual(grid.dtype, dtype)

  def test_uniform_grid_2d(self):
    dtype = np.float64
    # Batch of extra locations
    extra_locations = tf.constant([[1, 2], [2, 3]], dtype=dtype)
    min_x, max_x, sizes = [0.0, 0.0], [10.0, 5.0], [3, 2]
    grid_spec = self.evaluate(
        grids.uniform_grid_with_extra_point(
            min_x, max_x, sizes, extra_grid_point=extra_locations, dtype=dtype))
    self.assertEqual(grid_spec.dim, 2)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([2, 4, 3, 2]))
    self.assertEqual(grid.dtype, dtype)
    np_grid = np.array([[[[0., 0.], [0., 2.], [0., 5.]],
                         [[1., 0.], [1., 2.], [1., 5.]],
                         [[5., 0.], [5., 2.], [5., 5.]],
                         [[10., 0.], [10., 2.], [10., 5.]]],
                        [[[0., 0.], [0., 3.], [0., 5.]],
                         [[2., 0.], [2., 3.], [2., 5.]],
                         [[5., 0.], [5., 3.], [5., 5.]],
                         [[10., 0.], [10., 3.], [10., 5.]]]])
    np.testing.assert_almost_equal(grid, np_grid, 10)

  def test_uniform_grid_batched_boundary_2d(self):
    dtype = np.float64
    # Batch of extra locations
    extra_locations = tf.constant([[1, 2], [2, 3]], dtype=dtype)
    min_x, max_x, sizes = [[0, 0], [0, 0]], [[10, 5], [100, 5]], [3, 2]
    grid_spec = self.evaluate(
        grids.uniform_grid_with_extra_point(
            min_x, max_x, sizes, extra_grid_point=extra_locations, dtype=dtype))
    self.assertEqual(grid_spec.dim, 2)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([2, 4, 3, 2]))
    self.assertEqual(grid.dtype, dtype)
    np_grid = np.array([[[[0., 0.], [0., 2.], [0., 5.]],
                         [[1., 0.], [1., 2.], [1., 5.]],
                         [[5., 0.], [5., 2.], [5., 5.]],
                         [[10., 0.], [10., 2.], [10., 5.]]],
                        [[[0., 0.], [0., 3.], [0., 5.]],
                         [[2., 0.], [2., 3.], [2., 5.]],
                         [[50., 0.], [50., 3.], [50., 5.]],
                         [[100., 0.], [100., 3.], [100., 5.]]]])
    np.testing.assert_almost_equal(grid, np_grid, 10)

  def test_log_uniform_grid_1d(self):
    dtype = np.float64
    # Batch of extra locations
    extra_locations = [[0.5], [7]]
    min_x, max_x, size = dtype(0.1), dtype(10.0), 3
    np_grid = np.array([[[0.1], [0.5], [1.0], [10.0]],
                        [[0.1], [1.0], [7.0], [10.0]]])
    grid_spec = self.evaluate(
        grids.log_uniform_grid_with_extra_point(
            [min_x], [max_x], [size],
            extra_grid_point=extra_locations,
            dtype=dtype))
    self.assertEqual(grid_spec.dim, 1)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([2, 4, 1]))
    np.testing.assert_almost_equal(grid, np_grid, 10)
    self.assertEqual(grid.dtype, dtype)

  def test_log_uniform_grid_2d(self):
    dtype = np.float64
    # Batch of extra locations
    extra_locations = tf.constant([[1, 2], [2, 3]], dtype=dtype)
    min_x, max_x, sizes = [0.1, 0.1], [10.0, 5.0], [3, 2]
    grid_spec = self.evaluate(
        grids.log_uniform_grid_with_extra_point(
            min_x, max_x, sizes, extra_grid_point=extra_locations, dtype=dtype))
    self.assertEqual(grid_spec.dim, 2)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([2, 4, 3, 2]))
    self.assertEqual(grid.dtype, dtype)
    np_grid = np.array([[[[0.1, 0.1], [0.1, 2.], [0.1, 5.]],
                         [[1., 0.1], [1., 2.], [1., 5.]],
                         [[1., 0.1], [1., 2.], [1., 5.]],
                         [[10., 0.1], [10., 2.], [10., 5.]]],
                        [[[0.1, 0.1], [0.1, 3.], [0.1, 5.]],
                         [[1., 0.1], [1., 3.], [1., 5.]],
                         [[2., 0.1], [2., 3.], [2., 5.]],
                         [[10., 0.1], [10., 3.], [10., 5.]]]])
    np.testing.assert_almost_equal(grid, np_grid, 10)

  def test_log_uniform_batched_boundary_grid_2d(self):
    dtype = np.float64
    # Batch of extra locations
    extra_locations = tf.constant([[1, 2], [2, 3]], dtype=dtype)
    min_x, max_x, sizes = [[0.1, 0.1], [0.01, 0.1]], [[10, 5], [100, 5]], [3, 2]
    grid_spec = self.evaluate(
        grids.log_uniform_grid_with_extra_point(
            min_x, max_x, sizes, extra_grid_point=extra_locations, dtype=dtype))
    self.assertEqual(grid_spec.dim, 2)
    grid = grid_spec.grid
    self.assertEqual(grid.shape, tuple([2, 4, 3, 2]))
    self.assertEqual(grid.dtype, dtype)
    np_grid = np.array([[[[0.1, 0.1], [0.1, 2.], [0.1, 5.]],
                         [[1., 0.1], [1., 2.], [1., 5.]],
                         [[1., 0.1], [1., 2.], [1., 5.]],
                         [[10., 0.1], [10., 2.], [10., 5.]]],
                        [[[0.01, 0.1], [0.01, 3.], [0.01, 5.]],
                         [[1., 0.1], [1., 3.], [1., 5.]],
                         [[2., 0.1], [2., 3.], [2., 5.]],
                         [[100., 0.1], [100., 3.], [100., 5.]]]])
    np.testing.assert_almost_equal(grid, np_grid, 10)


if __name__ == '__main__':
  tf.test.main()
