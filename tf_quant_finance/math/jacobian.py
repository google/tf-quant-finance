# Lint as: python3
# Copyright 2020 Google LLC
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
"""Helper functions for computing jacobian."""

import tensorflow.compat.v2 as tf


def jacobian(func, x, unconnected_gradients=None,
             parallel_iterations=None, experimental_use_pfor=True,
             name=None):
  """Computes the jacobian of `func` wrt to `x`.

  Args:
    func: Python callable accepting one `Tensor` of shape of `x` and returning
      a `Tensor` of any shape. The function whose jacobian is to be computed.
    x: A `Tensor` with respect to which the gradient is to be computed.
    unconnected_gradients: An enum `tf.UnconnectedGradients` which specifies
      the gradient value returned when the given input tensors are
      unconnected. Default value: `None`, which maps to
      `tf.UnconnectedGradients.NONE`.
    parallel_iterations: A knob to control how many iterations are dispatched
      in parallel. This knob can be used to control the total memory usage.
    experimental_use_pfor: If true, uses pfor for computing the Jacobian.
      Else uses a tf.while_loop.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., 'jacobian').

  Returns:
    A `Tensor` with the gradient of `y` wrt each of `x`.
  """
  unconnected_gradients = unconnected_gradients or tf.UnconnectedGradients.NONE
  x, is_x_batch_size = _prepare_args(x)
  with tf.name_scope(name or "jacobian"):
    if not callable(func):
      raise ValueError("`func` should be a callable in eager mode or "
                       "when `tf.GradientTape` is used.")
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = func(x)
    jac = tape.batch_jacobian(
        y, x,
        unconnected_gradients=unconnected_gradients,
        parallel_iterations=parallel_iterations,
        experimental_use_pfor=experimental_use_pfor)

    if is_x_batch_size:
      return jac

    return jac[0]


def value_and_jacobian(f, x, unconnected_gradients=None, name=None,
                       parallel_iterations=None, experimental_use_pfor=True):
  """Computes `f(x)` and its jacobian wrt to `x`.

  Args:
    f: Python `callable` to be differentiated. If `f` returns a scalar, this
      scalar will be differentiated. If `f` returns a tensor or list of
      tensors, by default a scalar will be computed by adding all their values
      to produce a single scalar. If desired, the tensors can be elementwise
      multiplied by the tensors passed as the `dy` keyword argument to the
      returned jacobian function.
    x: A `Tensor` with respect to which the gradient is to be computed.
    unconnected_gradients: An enum `tf.UnconnectedGradients` which specifies
      the gradient value returned when the given input tensors are
      unconnected. Default value: `None`, which maps to
      `tf.UnconnectedGradients.NONE`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'value_and_jacobian'`).
    parallel_iterations: A knob to control how many iterations are dispatched
      in parallel. This knob can be used to control the total memory usage.
    experimental_use_pfor: If true, uses pfor for computing the Jacobian.
      Else uses a tf.while_loop.

  Returns:
    A tuple of two elements. The first one is a `Tensor` representing the value
    of the function at `x` and the second one is a `Tensor` representing
    jacobian of `f(x)` wrt `x`.
    y: `y = f(x)`.
    dydx: Jacobian of `y` wrt `x_i`, where `x_i` is the i-th parameter in
    `x`.
  """
  unconnected_gradients = unconnected_gradients or tf.UnconnectedGradients.NONE
  x, is_x_batch_size = _prepare_args(x)
  with tf.name_scope(name or "value_and_jacobian"):
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = f(x)
    jac = tape.batch_jacobian(
        y, x, unconnected_gradients=unconnected_gradients,
        parallel_iterations=parallel_iterations,
        experimental_use_pfor=experimental_use_pfor)

    if is_x_batch_size:
      return y, jac

    return y[0], jac[0]


def _prepare_args(x):
  """Converst `x` to a batched dimension if necessary."""
  if len(x.shape) == 1:
    return tf.expand_dims(x, axis=0), False

  return x, True
