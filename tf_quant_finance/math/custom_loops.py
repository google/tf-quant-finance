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
"""Differentiable for-loop (JAX native).

The TF implementation hand-rolled Jacobian accumulation across iterations
(because tf.while_loop's reverse-mode grad was memory-heavy). JAX's
`jax.lax.scan` is natively differentiable, so the entire custom-gradient /
Jacobian apparatus collapses to a thin scan wrapper.
"""

import jax
import jax.numpy as jnp

# A zero-shaped placeholder retained for backwards-compat with callers that
# inspect params (params now flow through body_fn's closure / the autodiff
# trace, so this argument is otherwise unused).
_PARAMS_UNUSED = None


def for_loop(body_fn, initial_state, params, num_iterations, name=None):
  """Differentiable for-loop over `body_fn(i, state) -> next_state`.

  Args:
    body_fn: callable `(i, state) -> next_state` with the same pytree structure.
    initial_state: sequence (pytree) of tensors sharing batch dims.
    params: kept for API compatibility; under JAX these flow through `body_fn`'s
      closure and are differentiated automatically by `jax.grad`/`lax.scan`.
    num_iterations: rank-0 int (run that many steps, return final state) or
      rank-1 int array of sorted iteration indices (return the state stacked at
      those iteration counts along a new leading axis).
    name: unused.

  Returns:
    A pytree matching `initial_state`. If `num_iterations` is rank-1, each leaf
    has an extra leading dimension of size `len(num_iterations)`.
  """
  del params, name
  num_iterations = jnp.asarray(num_iterations)
  if num_iterations.ndim == 0:
    length = int(num_iterations)
    final, _ = jax.lax.scan(_step(body_fn), initial_state, xs=jnp.arange(length))
    return final
  return _accumulating(body_fn, initial_state, num_iterations)


def _step(body_fn):
  def scan_body(carry, i):
    return body_fn(i, carry), None
  return scan_body


def _accumulating(body_fn, initial_state, num_iterations):
  """Rank-1 case: collect state at each requested iteration count."""
  n_max = int(jnp.max(num_iterations))
  # Scan n_max steps; record the state AFTER each step (ys[k] = state after k+1
  # iterations, i.e. iteration index k+1).
  def scan_body(carry, i):
    next_state = body_fn(i, carry)
    return next_state, next_state
  _, states = jax.lax.scan(scan_body, initial_state, xs=jnp.arange(n_max))
  idx = num_iterations - 1
  return jax.tree_util.tree_map(lambda s: s[idx], states)
