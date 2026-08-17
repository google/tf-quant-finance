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
"""Helper functions for computing jacobian (JAX native).

Converted from tf.GradientTape.batch_jacobian to jax.vmap(jax.jacfwd).
"""

import jax
import jax.numpy as jnp


def jacobian(func, x, unconnected_gradients=None, parallel_iterations=None,
             experimental_use_pfor=True, name=None):
  """Computes the (batched) jacobian of `func` wrt `x`.

  Args:
    func: callable accepting one tensor, returning a tensor.
    x: tensor; if rank-1 it is treated as a single (unbatched) sample.
    parallel_iterations: unused (kept for signature compatibility).
    experimental_use_pfor: unused.
    name: unused.

  Returns:
    Tensor of shape `y.shape + x.shape` (per-sample when batched).
  """
  del unconnected_gradients, parallel_iterations, experimental_use_pfor, name
  if not callable(func):
    raise tf.errors.InvalidArgumentError("`func` should be a callable.")
  x, is_batch = _prepare_args(x)
  jac = jax.vmap(jax.jacfwd(func))(x)
  return jac if is_batch else jac[0]


def value_and_jacobian(f, x, unconnected_gradients=None, name=None,
                       parallel_iterations=None, experimental_use_pfor=True):
  """Computes `f(x)` and its (batched) jacobian wrt `x`."""
  del unconnected_gradients, name, parallel_iterations, experimental_use_pfor
  x, is_batch = _prepare_args(x)
  y = f(x)
  jac = jax.vmap(jax.jacfwd(f))(x)
  return (y, jac) if is_batch else (y[0], jac[0])


def _prepare_args(x):
  """If `x` is rank-1, add a batch dimension."""
  if len(x.shape) == 1:
    return jnp.expand_dims(x, axis=0), False
  return x, True
