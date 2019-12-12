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
"""Composition of two time marching scheme."""

import tensorflow as tf

from tf_quant_finance.experimental.pde_v2.steppers.parabolic_equation_stepper import parabolic_equation_step


def composite_scheme_step(first_scheme_steps, first_scheme, second_scheme):
  """Composes two time marching schemes.

  Applies a step of parabolic PDE solver using `first_scheme` if number of
  performed steps is less than `first_scheme_steps`, and using `second_scheme`
  otherwise.

  Args:
    first_scheme_steps: A Python integer. Number of steps to apply
      `first_scheme` on.
    first_scheme: First time marching scheme (see `time_marching_scheme`
      argument of `parabolic_equation_step`).
    second_scheme: Second time marching scheme (see `time_marching_scheme`
      argument of `parabolic_equation_step`).

  Returns:
    Callable to use as `one_step_fn` in fd_solvers.
  """
  def step_fn(
      time,
      next_time,
      coord_grid,
      value_grid,
      boundary_conditions,
      second_order_coeff_fn,
      first_order_coeff_fn,
      zeroth_order_coeff_fn,
      num_steps_performed,
      dtype=None,
      name=None):
    """Applies `parabolic_equation_step` with one of the two schemes."""
    name = name or 'composite_scheme_step'

    def scheme(*args, **kwargs):
      return tf.cond(num_steps_performed < first_scheme_steps,
                     lambda: first_scheme(*args, **kwargs),
                     lambda: second_scheme(*args, **kwargs))
    return parabolic_equation_step(
        time,
        next_time,
        coord_grid,
        value_grid,
        boundary_conditions,
        second_order_coeff_fn,
        first_order_coeff_fn,
        zeroth_order_coeff_fn,
        time_marching_scheme=scheme,
        dtype=dtype,
        name=name)

  return step_fn
