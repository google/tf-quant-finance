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
"""Helper functions for computing gradients (JAX native).

Converted from the TF GradientTape implementation to jax.jvp / jax.vjp.
"""

import functools

import jax
import jax.numpy as jnp


def fwd_gradient(func_or_y, x, input_gradients=None, use_gradient_tape=False,
                 unconnected_gradients=None, name=None):
  """Forward-mode directional derivative J(F).u of `func_or_y` at `x`.

  Args:
    func_or_y: callable accepting one tensor of `x`'s shape, returning a tensor.
    x: tensor at which to evaluate the derivative.
    input_gradients: direction `u` (defaults to ones_like(x)).
    name: unused (kept for signature compatibility).

  Returns:
    Tensor of the same shape as `func(x)` holding J(F).u.
  """
  del use_gradient_tape, unconnected_gradients, name
  f = _prepare_func(func_or_y)
  u = input_gradients if input_gradients is not None else jnp.ones_like(x)
  _, tangents = jax.jvp(f, (x,), (u,))
  return tangents


def gradients(func_or_y, xs, output_gradients=None, use_gradient_tape=False,
              unconnected_gradients=None, name=None):
  """Reverse-mode gradient of `func_or_y` wrt `*xs`.

  Args:
    func_or_y: callable accepting `*xs`.
    xs: tensor or list of tensors.
    output_gradients: cotangent `dy` (defaults to ones_like(y), i.e. sum(y)).
    name: unused.

  Returns:
    Tensor (if `xs` is a single tensor) or list of tensors (if `xs` is a list).
  """
  del use_gradient_tape, unconnected_gradients, name
  f = _prepare_func(func_or_y)
  xs_list, is_list = _prepare_args(xs)
  y, vjp = jax.vjp(f, *xs_list)
  if output_gradients is None:
    cotangent = jnp.ones_like(y)
  elif isinstance(output_gradients, (list, tuple)):
    # Multiple outputs folded into a single cotangent by summation of shaped ones.
    cotangent = sum(output_gradients)
  else:
    cotangent = output_gradients
  grad = vjp(cotangent)
  return list(grad) if is_list else grad[0]


def value_and_gradient(f, xs, output_gradients=None, use_gradient_tape=False,
                       unconnected_gradients=None, name=None):
  """Computes `f(*xs)` and its gradients wrt `*xs`.

  Args:
    f: callable to differentiate.
    xs: tensor or list of tensors.
    output_gradients: cotangent `dy` (defaults to ones_like(y)).
    name: unused.

  Returns:
    (y, grad) where grad is a tensor (single xs) or list (xs is a list).
  """
  del use_gradient_tape, unconnected_gradients, name
  xs_list, is_list = _prepare_args(xs)
  y, vjp = jax.vjp(f, *xs_list)
  if output_gradients is None:
    cotangent = jnp.ones_like(y)
  elif isinstance(output_gradients, (list, tuple)):
    cotangent = sum(output_gradients)
  else:
    cotangent = output_gradients
  grad = vjp(cotangent)
  return (y, list(grad)) if is_list else (y, grad[0])


def make_val_and_grad_fn(value_fn):
  """Decorator: turns `value_fn(x)` into `(value, grad)`."""
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return value_and_gradient(value_fn, x)
  return val_and_grad


def _prepare_func(func_or_y):
  """Creates a function out of the input callable or `Tensor`."""
  if callable(func_or_y):
    return func_or_y
  return lambda *args: func_or_y


def _prepare_args(xs):
  """Converts `xs` to a list if necessary."""
  if isinstance(xs, (list, tuple)):
    return list(xs), True
  return [xs], False
