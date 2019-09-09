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
"""Time marching schemes for finite difference methods for parabolic PDEs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_quant_finance.math.pde.time_marching_schemes.pde_time_marching_scheme import TimeMarchingScheme


class CompositeTimeMarchingScheme(TimeMarchingScheme):
  """Composes two time marching schemes.

  The first scheme is applied on the given number of first steps, and the second
  scheme - on all subsequent steps.

  Kept internal, because it currently exists for testability purposes only.

  # TODO(b/139954730): Improve documentation of the CompositeTimeMarchingScheme
  # as it is very scantly documented.
  """

  def __init__(self, first_scheme_steps, first_scheme, second_scheme):
    """Initializer.

    Args:
      first_scheme_steps: Number of steps to apply `first_scheme` on.
      first_scheme: First `TimeMarchingScheme`.
      second_scheme: Second `TimeMarchingScheme`.
    """
    self.steps = first_scheme_steps
    self.first_scheme = first_scheme
    self.second_scheme = second_scheme

  def apply(self,
            value_grid,
            t1,
            t2,
            num_steps_performed,
            matrix_constructor,
            lower_boundary_fn,
            upper_boundary_fn,
            backwards=False):

    def apply_first():
      return self.first_scheme.apply(value_grid, t1, t2, num_steps_performed,
                                     matrix_constructor, lower_boundary_fn,
                                     upper_boundary_fn, backwards)

    def apply_second():
      return self.second_scheme.apply(value_grid, t1, t2, num_steps_performed,
                                      matrix_constructor, lower_boundary_fn,
                                      upper_boundary_fn, backwards)

    return tf.cond(num_steps_performed < self.steps, apply_first, apply_second)
