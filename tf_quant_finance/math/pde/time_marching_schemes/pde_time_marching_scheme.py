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
"""Defines the interface for PDE time marching schemes to implement."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class TimeMarchingScheme(object):
  """Abstract class representing time marching schemes.

  # TODO(b/139954730): Add thorough documentation of the class.
  """

  @abc.abstractmethod
  def apply(self,
            value_grid,
            t1,
            t2,
            num_steps_performed,
            matrix_constructor,
            lower_boundary_fn,
            upper_boundary_fn,
            backwards=False):
    """Applies the time marching scheme to a time interval.

    If `u(t)` is space-discretized vector of the solution of a PDE, this method
    approximately solves the equation `du/dt = A(t) u(t)` for `u(t2)` given
    `u(t1)`, or vice versa if `backwards=True`. Here `A` is a tridiagonal
    matrix.

    Time marching schemes give approximate solutions to the above equation.
    For example, explicit scheme corresponds to
    `u(t2) = (1 + (t2 - t1) A(t1)) u(t1)`.

    Args:
      value_grid: Grid of solution values at the current time.
      t1: Lesser of the two times defining the step.
      t2: Greater of the two times defining the step.
      num_steps_performed: Number of steps performed so far.
      matrix_constructor: Callable that constructs a tridiagonal matrix `A` (see
        above). Accepts one argument: the time `t`. `t` can be any time between
          `t1` and `t2`. Returns three tensors representing diagonal,
          superdiagonal, and subdiagonal parts of the tridiagonal matrix. The
          shape of these tensors must be same as of `value_grid`.
          superdiagonal[..., -1] and subdiagonal[..., 0] are ignored.
      lower_boundary_fn: Callable returning a `Tensor` of the same
        `dtype` as `value_grid` and with shape `value_grid.shape[:-1]`.
          Represents Dirichlet boundary condition at the lower boundary.
      upper_boundary_fn: Callable similar to `lower_boundary_fn` representing
        Dirichlet boundary condition at the upper boundary.
      backwards: Whether time marching goes backwards in time (important for
        schemes that subdivide the steps).

    Returns:
      Grid of values with the same shape as value_grid, representing the
      result of applying the step.
    """
    raise NotImplementedError
