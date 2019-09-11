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
"""Tests for `BackwardGridStepper`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_quant_finance.math import pde
from tf_quant_finance.math.pde import grids
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class GridStepperTest(tf.test.TestCase):

  def test_grid_stepper_basic(self):
    dtype = np.float64
    min_x, max_x, size = dtype(0.0), dtype(1.0), 21
    np_grid = np.linspace(min_x, max_x, num=size)
    grid_spec = grids.uniform_grid([min_x], [max_x], [size], dtype=dtype)
    r = dtype(0.03)

    def kernel(state):
      return state.value_grid * tf.exp(-r * state.time_step)

    def time_step_fn(state):
      del state
      return tf.constant(0.1, dtype=dtype)

    def payoff_fn(state):
      return tf.maximum(state.coordinate_grid.grid, 0.5)

    bgs = pde.BackwardGridStepper(
        2.0, kernel, grid_spec, time_step_fn=time_step_fn, dtype=dtype)
    bgs.transform_values(payoff_fn)  # At time 2.
    bgs.step_back_to_time(dtype(0.13))
    intermediate_state = self.evaluate(bgs.state())
    bgs.step_back_to_time(dtype(0.0))
    initial_state = self.evaluate(bgs.state())
    expected_final_vals = np.maximum(np_grid, 0.5)
    expected_intermediate_values = expected_final_vals * np.exp(-r * (2 - 0.13))
    expected_initial_values = expected_final_vals * np.exp(-r * 2)
    self.assertArrayNear(expected_intermediate_values,
                         intermediate_state.value_grid, 1e-10)
    self.assertEqual(0.13, intermediate_state.time)
    self.assertArrayNear(expected_initial_values, initial_state.value_grid,
                         1e-10)
    self.assertEqual(0.0, initial_state.time)

  def test_nonconst_time_step(self):
    dtype = np.float64
    min_x, max_x, size = dtype(0.0), dtype(1.0), 21
    np_grid = np.linspace(min_x, max_x, num=size)
    grid_spec = grids.uniform_grid([min_x], [max_x], [size], dtype=dtype)
    r = dtype(0.03)

    def kernel(state):
      return state.value_grid * tf.exp(-r * state.time_step)

    def time_step_fn(state):
      """A non-constant time step."""
      # Returns the first element of the harmonic sequence which is smaller
      # than the current time floored by 0.01. If the current time is t,
      # then returns tf.max(1/tf.ceil(1/t), 0.01).
      return tf.maximum(dtype(0.01), 1. / tf.ceil(1. / state.time))

    def payoff_fn(state):
      return tf.maximum(state.coordinate_grid.grid, 0.5)

    bgs = pde.BackwardGridStepper(
        1.9, kernel, grid_spec, time_step_fn=time_step_fn, dtype=dtype)
    bgs.transform_values(payoff_fn)  # At time 1.9.
    bgs.step_back_to_time(dtype(0.13))
    intermediate_state = self.evaluate(bgs.state())
    bgs.step_back_to_time(dtype(0.0))
    initial_state = self.evaluate(bgs.state())
    expected_final_vals = np.maximum(np_grid, 0.5)
    expected_intermediate_values = expected_final_vals * np.exp(-r *
                                                                (1.9 - 0.13))
    expected_initial_values = expected_final_vals * np.exp(-r * 1.9)
    self.assertArrayNear(expected_intermediate_values,
                         intermediate_state.value_grid, 1e-10)
    self.assertEqual(0.13, intermediate_state.time)
    self.assertArrayNear(expected_initial_values, initial_state.value_grid,
                         1e-10)
    self.assertEqual(0.0, initial_state.time)

  def test_step_back(self):
    """Tests the step_back method."""
    dtype = np.float64
    min_x, max_x, size = dtype(0.0), dtype(1.0), 21
    grid_spec = grids.uniform_grid([min_x], [max_x], [size], dtype=dtype)
    r = dtype(0.03)

    def kernel(state):
      return state.value_grid * tf.exp(-r * state.time_step)

    def time_step_fn(state):
      """A non-constant time step."""
      # Returns the first element of the harmonic sequence which is smaller
      # than the current time floored by 0.01. If the current time is t,
      # then returns tf.max(1/tf.ceil(1/t), 0.01).
      return tf.maximum(dtype(0.01), 1. / tf.ceil(1. / state.time))

    def payoff_fn(state):
      return tf.maximum(state.coordinate_grid.grid, 0.5)

    bgs = pde.BackwardGridStepper(
        1.9, kernel, grid_spec, time_step_fn=time_step_fn, dtype=dtype)
    bgs.transform_values(payoff_fn)  # At time 1.9.

    # Calls step back and checks that the time of the grid changes to 0.9.
    bgs.step_back()
    state_1 = self.evaluate(bgs.state())
    self.assertTrue(state_1.time, 0.9)
    # Also check that the next time step is correctly populated. It should be
    # 0.5 now.
    self.assertTrue(state_1.time_step, 0.5)
    # Step back again and check that the time step is correctly applied.
    bgs.step_back()
    state_2 = self.evaluate(bgs.state())
    self.assertTrue(state_2.time, 0.4)

  def test_value_transform_fn(self):
    """Tests the value transform functionality of the step back method."""
    dtype = np.float64
    min_x, max_x, size = dtype(0.0), dtype(1.0), 21
    np_grid = np.linspace(min_x, max_x, num=size)
    grid_spec = grids.uniform_grid([min_x], [max_x], [size], dtype=dtype)
    r = dtype(0.03)

    multipliers = np.linspace(0.1, 1.1, num=size)

    def transform_fn(state):
      return state.value_grid * multipliers.reshape([-1, 1])

    def kernel(state):
      return state.value_grid * tf.exp(-r * state.time_step)

    def payoff_fn(state):
      return tf.maximum(state.coordinate_grid.grid, 0.5)

    bgs = pde.BackwardGridStepper(
        2.0, kernel, grid_spec, time_step=0.5, dtype=dtype)
    bgs.transform_values(payoff_fn)  # At time 2.
    expected_final_vals = np.maximum(np_grid, 0.5)
    expected_initial_values = (
        expected_final_vals * np.exp(-r * 2) * (multipliers**4))
    bgs.step_back_to_time(0.0, value_transform_fn=transform_fn)
    initial_state = self.evaluate(bgs.state())
    self.assertArrayNear(expected_initial_values, initial_state.value_grid,
                         1e-10)
    self.assertEqual(0.0, initial_state.time)

  def test_extra_time_points(self):
    """Tests the extra time points of the step back method."""
    dtype = np.float64
    min_x, max_x, size = dtype(0.0), dtype(1.0), 21
    np_grid = np.linspace(min_x, max_x, num=size)
    grid_spec = grids.uniform_grid([min_x], [max_x], [size], dtype=dtype)
    r = dtype(0.03)

    def transform_fn(state):
      return tf.where(state.time <= 1.0, state.value_grid, payoff_fn(state))

    def kernel(state):
      return state.value_grid * tf.exp(-r * state.time_step)

    def payoff_fn(state):
      return tf.maximum(state.coordinate_grid.grid, 0.5)

    bgs = pde.BackwardGridStepper(
        2.0, kernel, grid_spec, time_step=0.3, dtype=dtype)
    bgs.transform_values(payoff_fn)  # At time 2.
    expected_final_vals = np.maximum(np_grid, 0.5)
    expected_initial_values = expected_final_vals * np.exp(-r * 1)
    bgs.step_back_to_time(
        0.0,
        value_transform_fn=transform_fn,
        extra_time_points=[1.3, 1.5, 1.0, 3.0])
    initial_state = self.evaluate(bgs.state())
    self.assertArrayNear(expected_initial_values, initial_state.value_grid,
                         1e-10)
    self.assertEqual(0.0, initial_state.time)


if __name__ == '__main__':
  tf.test.main()
