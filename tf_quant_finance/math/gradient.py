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
"""Helper functions for computing gradients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow_probability.python.math import value_and_gradient


def gradients(f, xs, output_gradients=None, use_gradient_tape=False, name=None):
  """Computes the gradients of `f` wrt to `*xs`.

  Args:
    f: Python `callable` to be differentiated.
    xs: Python list of parameters of `f` for which to differentiate. (Can also
      be single `Tensor`.)
    output_gradients: A `Tensor` or list of `Tensor`s the same size as the
      result `ys = f(*xs)` and holding the gradients computed for each `y` in
      `ys`. This argument is forwarded to the underlying gradient implementation
      (i.e., either the `grad_ys` argument of `tf.gradients` or the
      `output_gradients` argument of `tf.GradientTape.gradient`).
    use_gradient_tape: Python `bool` indicating that `tf.GradientTape` should be
      used regardless of `tf.executing_eagerly()` status.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'gradients'`).

  Returns:
    A `Tensor` with the gradient of `y` wrt each of `xs`.
  """
  _, grad = value_and_gradient(
      f, xs, output_gradients=output_gradients,
      use_gradient_tape=use_gradient_tape, name=name or 'gradients')
  return grad


def make_val_and_grad_fn(value_fn):
  """Function decorator to compute both function value and gradient.

  For example:

  ```
  @tff.math.make_val_and_grad_fn
  def quadratic(x):
    return tf.reduce_sum(scales * (x - minimum) ** 2, axis=-1)
  ```

  Turns `quadratic` into a function that accepts a point as a `Tensor` as input
  and returns a tuple of two `Tensor`s with the value and the gradient of the
  defined quadratic function evaluated at the input point.

  This is useful for constucting functions to optimize with tff.math.optimizer
  methods.

  Args:
    value_fn: A python function to decorate.

  Returns:
    The decorated function.
  """
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return value_and_gradient(value_fn, x)

  return val_and_grad
