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
"""Tests for `pde_time_marching_schemes_internal`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_quant_finance.math import pde
from tf_quant_finance.math.pde.internal.pde_time_marching_schemes_internal import CompositeTimeMarchingScheme

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

# Time marching schemes
schemes = pde.time_marching_schemes


@test_util.run_all_in_graph_and_eager_modes
class PdeTimeMarchingSchemesInternal(tf.test.TestCase):

  def testCompositeTimeMarchingScheme(self):
    """CompositeTimeMarchingScheme applies the two schemes correctly."""

    class FirstScheme(schemes.TimeMarchingScheme):

      def apply(self, value_grid, *args):
        return value_grid + 1

    class SecondScheme(schemes.TimeMarchingScheme):

      def apply(self, value_grid, *args):
        return value_grid + 2

    scheme = CompositeTimeMarchingScheme(1, FirstScheme(), SecondScheme())

    grid = pde.grids.uniform_grid(
        minimums=[0.0], maximums=[1.0], sizes=[3], dtype=tf.float32)

    solver_kernel = pde.ParabolicDifferentialEquationSolver(
        lambda *args: 1,
        lambda *args: 0,
        lambda *args: 0,
        time_marching_scheme=scheme)

    time_step = 1
    final_t = 3

    def time_step_fn(state):
      del state
      return tf.constant(time_step, dtype=tf.float32)

    bgs = pde.BackwardGridStepper(
        final_t,
        solver_kernel.one_step,
        grid,
        time_step_fn=time_step_fn,
        value_dim=1,
        dtype=tf.float32)
    bgs.step_back_to_time(0.0)
    result = self.evaluate(bgs.state().value_grid)[0]

    # Should apply FirstScheme once, and SecondScheme twice, yielding 1+2*2=5
    self.assertAllClose(result, [5, 5, 5])


if __name__ == '__main__':
  tf.test.main()
