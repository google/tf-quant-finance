# Lint as: python3
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
"""Custom implementations of loops for improved performance."""

import tensorflow.compat.v2 as tf


def for_loop(body_fn, initial_state, params, num_iterations, name=None):
  """A for loop with a custom batched gradient.

  A for loop with a custom gradient that in certain cases outperforms the
  tf.while_loop gradient implementation.

  This is not a general replacement for tf.while_loop as imposes a number of
  restrictions on the inputs:
  - All tensors in loop state must have the same shape except for the last
  dimension.
  - All dimensions except for the last one are treated as batch dimensions, i.e.
  it is assumed that batch elements don't interact with each other inside the
  loop body.
  - The last dimensions and the number of parameters must be statically known
  and be reasonably small (so that the full Jacobian matrix with respect to them
  can be calculated efficiently, see below).
  - It requires an explicit list of parameters used in the loop body, with
  respect to which one wishes to calculate derivatives. This is different from
  tf.while_loop which automatically deals with tensors captured in the closure.
  - Parameters must be a sequence of zero-dimensional tensors.
  - Arbitrary nested structure of state is not supported, the state must be a
  flat sequence of tensors.

  The issue this implementation addresses is the additional while loops created
  by the gradient of `tf.while_loop`. To compute the backward gradient
  (more precisely, the vector-Jacobian product) of a while loop, one needs to
  run the loop "backwards". This implementation avoids creating a second loop by
  calculating the full (batched) Jacobian matrix in the forward pass. It is
  efficient when the non-batched part of the shape of the Jacobian is small.
  This part has size `nd * (nd + p)` where `nd` is the sum of last dimensions of
  tensors in the state and `p` is the number of parameters.

  This implementation is suitable for e.g. Monte-Carlo sampling, where the state
  represents a batch of independent paths.

  #### Example:

  ```python
  x = tf.constant([[3.0, 4.0], [30.0, 40.0]])
  y = tf.constant([[7.0, 8.0], [70.0, 80.0]])
  alpha = tf.constant(2.0)
  beta = tf.constant(1.0)

  with tf.GradientTape(persistent=True) as tape:
    tape.watch([alpha, beta])
    def body(i, state):
      x, y = state
      return [x * alpha - beta, y * beta + x]
    x_out, y_out = for_loop(body, [x, y], [alpha, beta], 3)

  grad = tape.gradient(y_out, beta)  # Returns tf.Tensor(783.0)
  ```

  Args:
    body_fn: A Callable. Accepts an iteration index as a 0-dimensional int32
      tensor and state - a tuple of Tensors of same shape as `initial_state`.
      Should return the output state with the same structure as the input state.
    initial_state: A sequence of Tensors with common batch shape. All dimensions
      except the last are treated as batch shape (i.e. not mixed in loop body).
    params: A list of zero-dimensional Tensors - tensors that `body_fn` uses,
      and with respect to which the differentiation is going to happen.
    num_iterations: A rank 0 or rank 1 integer tensor. If the rank is 1, the
      entries are expected to be unique and ordered and  the output will contain
      results obtained at each iteration number specified in `num_iterations`,
      stacked along the first dimension. E.g. if `initial_state` has shapes
      `(10, 20, 2)` and `(10, 20, 3)`, and `num_iterations = [2, 5, 7, 10]` the
      output is a list of tensors with shapes `(4, 10, 20, 2)` and
      `(4, 10, 20, 3)`.

    name: Python str. The name to give to the ops created by this function,
      'for_loop' by default.

  Returns:
   A list of Tensors of the same shape as `initial_state`, if `num_iterations`
   is a single integer, or with extra first dimension of size
   `len(num_iterations)` otherwise.
   The outputs are differentiable with respect to `initial_state` and `params`,
   but not any other tensors that are captured by `body_fn`. Differentiating
   with respect to an element of `initial_state` yields a tensor with the same
   shape as that element. Differentiating with respect to one of `params` yields
   a tensor of zero shape. If the output state doesn't depend on the given
   parameter, the tensor will be filled with zeros.
  """

  # Implementation explanation.
  #
  # Notation:
  # n - number of state tensors,
  # d - last dim of state (it can be different among the n state tensors, but
  # for illustration we'll assume it's the same),
  # p - number of parameters.
  #
  # The common batch dimensions are omitted below.
  #
  # The Jacobian has the form
  # | Js Jp |
  # | 0  I  |,
  # where
  # Js = d state / d state, shape = (nd, nd),
  # Js = d state / d params, shape = (nd, p),
  # 0 - zero matrix (p, nd),
  # I - unit matrix (p, p).
  #
  # Multiplying two Jacobians yields
  # | Js1 Js2    Js1 Jp2 + Jp1 |
  # |   0              I       |
  #
  # The custom gradient function receives output weights ws with shape = (nd,).
  # We turn them into row vectors ws' with shape(1, nd), multiply by Jacobians,
  # (ws' Js, ws' Jp), yielding shapes (1, nd) and (1, p), and finally squeeze
  # the dimensions to get the desired shapes (nd,), (p,).
  #
  # Js and Jp have block structure themselves:
  #
  # Js =  | Js_11 ... Js_1n|
  #       | ...   ... ...  |
  #       | Js_n1 ... Js_nn|
  #
  # Js =  | Jp_11 ... Jp_1p|
  #       | ...   ... ...  |
  #       | Jp_n1 ... Jp_np|
  #
  # where Js_ij have shape (d, d), Jp_ij have shape (d, 1).
  #
  # Js_ij and Jp_ij are Tensors, and the rest are nested lists. We multiply and
  # add the tensors with TF, and the nested lists - manually with Python loops.
  #
  # Note that we can't concatenate the parameters into a single Tensor to avoid
  # some Python loops, even though it's cheap. There will be no path from
  # tf.concat node to the body_fn output, because the user uses the original
  # (not concatenated) parameters in body_fn.

  num_iterations = tf.convert_to_tensor(num_iterations, dtype=tf.int32,
                                        name="num_iterations")
  num_iterations_shape = num_iterations.shape.as_list()
  if num_iterations_shape is None:
    raise ValueError("Rank of num_iterations must be statically known.")
  if len(num_iterations_shape) > 1:
    raise ValueError("Rank of num_iterations must be 0 or 1")
  if len(num_iterations_shape) == 1:
    return _accumulating_for_loop(body_fn, initial_state, params,
                                  num_iterations, name)

  with tf.name_scope(name or "for_loop"):
    initial_jac = _make_unit_jacobian(initial_state, params)
    n = len(initial_state)

    @tf.custom_gradient
    def inner(*args):
      initial_state, params = args[:n], args[n:]
      def while_cond(i, state, jac):
        del state, jac
        return i < num_iterations

      def while_body(i, state, jac):
        with tf.GradientTape(persistent=True) as tape:
          tape.watch(state)
          tape.watch(params)
          next_state = tuple(body_fn(i, state))
        step_jac = _compute_step_jacobian(state, next_state, params, tape)
        next_jac = _multiply_jacobians(step_jac, jac)
        return i + 1, next_state, next_jac

      loop_vars = (0, initial_state, initial_jac)

      _, state, jac = tf.compat.v2.while_loop(
          while_cond, while_body, loop_vars=loop_vars,
          maximum_iterations=num_iterations)

      def gradient(*ws):
        # tf.custom_gradient converts any structure of function outputs into a
        # flat tuple when calling the custom gradient.

        # Expand into (..., 1, d), so that we can matmul it with Jacobian.
        ws = [tf.expand_dims(w, axis=-2) for w in ws]
        ws = [ws]  # expand dims on block level as well.

        js, jp = jac
        ws_js, ws_jp = _block_matmul(ws, js), _block_matmul(ws, jp)

        # Now undo the expansions
        ws_js, ws_jp = ws_js[0], ws_jp[0]
        ws_js = [tf.squeeze(t, axis=-2) for t in ws_js]
        # These should be 0-dimensional
        ws_jp = [tf.reduce_sum(t) for t in ws_jp]

        # Flatten into a single tuple, so that it has the same structure as args
        # in inner().
        return ws_js + ws_jp

      return state, gradient

    # tf.custom_gradient can only handle a flat sequence of args.
    args = tuple(initial_state + params)
    return inner(*args)


def _make_unit_jacobian(initial_state, params):
  """Creates a unit Jacobian matrix."""
  n = len(initial_state)
  d = [initial_state[i].shape.as_list()[-1] for i in range(n)]
  if None in d:
    raise ValueError("Last dimensions of initial_state Tensors must be known.")
  p = len(params)
  dtype = initial_state[0].dtype

  def make_js_block(i, j):
    shape = initial_state[i].shape.concatenate((d[j],))
    if i != j:
      return tf.zeros(shape, dtype=dtype)
    eye = tf.eye(d[i], dtype=dtype)
    return tf.broadcast_to(eye, shape)

  def make_jp_block(i, j):
    del j
    shape = initial_state[i].shape.concatenate((1,))
    return tf.zeros(shape, dtype=dtype)

  js = [[make_js_block(i, j) for j in range(n)] for i in range(n)]
  jp = [[make_jp_block(i, j) for j in range(p)] for i in range(n)]
  return js, jp


def _compute_step_jacobian(state, next_state, params, tape):
  """Computes a Jacobian of a transformation next_state = f(state, params)."""
  n = len(state)
  p = len(params)
  js = [[_batch_jacobian(next_state[i], state[j], tape)
         for j in range(n)]
        for i in range(n)]
  jp = [[_jacobian_wrt_parameter(next_state[i], params[j], tape)
         for j in range(p)]
        for i in range(n)]
  return js, jp


def _batch_jacobian(y, x, tape):
  """Computes a Jacobian w.r.t. last dimensions of y and x."""
  # y and x must have the same batch dimensions.
  # For input shapes (b, dy), (b, dx) yields shape (b, dy, dx).
  d = y.shape.as_list()[-1]
  if d is None:
    raise ValueError("Last dimension of state Tensors must be known.")
  grads = []
  for i in range(d):
    w = tf.broadcast_to(tf.one_hot(i, d, dtype=y.dtype), y.shape)
    # We must use tf.UnconnectedGradients.ZERO here and below, because some
    # state components may legitimately not depend on each other or some of the
    # params.
    grad = tape.gradient(y, x, output_gradients=w,
                         unconnected_gradients=tf.UnconnectedGradients.ZERO)
    grads.append(grad)
  return tf.stack(grads, axis=-2)


def _jacobian_wrt_parameter(y, param, tape):
  """Computes a Jacobian w.r.t. a parameter."""
  # For input shapes (b, dy), yields shape (b, dy, 1) (1 is added for
  # convenience elsewhere).
  # To avoid having to broadcast param to y's shape, we need to take a forward
  # gradient.
  with tf.GradientTape() as w_tape:
    w = tf.zeros_like(y)
    w_tape.watch(w)
    vjp = tape.gradient(y, param, output_gradients=w)
  if vjp is None:  # Unconnected.
    return tf.expand_dims(tf.zeros_like(y), axis=-1)
  return tf.expand_dims(w_tape.gradient(vjp, w), axis=-1)


def _multiply_jacobians(jac1, jac2):
  """Multiplies two Jacobians."""
  js1, jp1 = jac1
  js2, jp2 = jac2
  return _block_matmul(js1, js2), _block_add(_block_matmul(js1, jp2), jp1)


def _block_matmul(m1, m2):
  """Multiplies block matrices represented as nested lists."""
  # Calls itself recursively to multiply blocks, until reaches the level of
  # tf.Tensors.
  if isinstance(m1, tf.Tensor):
    assert isinstance(m2, tf.Tensor)
    return tf.matmul(m1, m2)
  assert _is_nested_list(m1) and _is_nested_list(m2)

  i_max = len(m1)
  k_max = len(m2)
  j_max = 0 if k_max == 0 else len(m2[0])
  if i_max > 0:
    assert len(m1[0]) == k_max

  def row_by_column(i, j):
    return _block_add(*[_block_matmul(m1[i][k], m2[k][j])
                        for k in range(k_max)])
  return [[row_by_column(i, j) for j in range(j_max)] for i in range(i_max)]


def _block_add(*ms):
  """Adds block matrices represented as nested lists."""
  # Calls itself recursively to add blocks, until reaches the level of
  # tf.Tensors.
  if len(ms) == 1:
    return ms[0]
  if isinstance(ms[0], tf.Tensor):
    assert all(isinstance(m, tf.Tensor) for m in ms[1:])
    return tf.math.add_n(ms)
  assert all(_is_nested_list(m) for m in ms)
  for i in range(1, len(ms)):
    tf.nest.assert_same_structure(ms[0], ms[i])

  i_max = len(ms[0])
  j_max = 0 if i_max == 0 else len(ms[0][0])
  return [[_block_add(*[ms[k][i][j] for k in range(len(ms))])
           for j in range(j_max)]
          for i in range(i_max)]


def _is_nested_list(m):
  return isinstance(m, list) and (not m or isinstance(m[0], list))


def _accumulating_for_loop(body_fn, initial_state, params, num_iterations,
                           name=None):
  """Version of for_loop with multiple values of num_iterations."""
  # Every tensor in nested tensors (state and Jacobian) gets an extra
  # "accumulating" dimension in front. Functions _create_accumulators etc. below
  # help to work with this dimension.

  with tf.name_scope(name or "accumulating_for_loop"):
    max_iterations = tf.math.reduce_max(num_iterations)
    acc_size = num_iterations.shape[0]

    # num_iteration = [2, 5] -> mask = [0, 0, 1, 0, 0, 1]. Tells when we should
    # increment acc index before writing. Last element won't be used (i = 0..4).
    mask = tf.scatter_nd(indices=tf.expand_dims(num_iterations, axis=-1),
                         updates=tf.ones_like(num_iterations),
                         shape=(max_iterations + 1,))

    n = len(initial_state)

    @tf.custom_gradient
    def inner(*args):
      initial_state, params = args[:n], args[n:]
      def while_cond(i, acc_index, acc_state, acc_jac):
        del acc_index, acc_state, acc_jac
        return i < max_iterations

      def while_body(i, acc_index, acc_state, acc_jac):
        state = _read_from_accumulators(acc_state, acc_index)
        jac = _read_from_accumulators(acc_jac, acc_index)
        with tf.GradientTape(persistent=True) as tape:
          tape.watch(state)
          tape.watch(params)
          next_state = tuple(body_fn(i, state))
        step_jac = _compute_step_jacobian(state, next_state, params, tape)
        next_jac = _multiply_jacobians(step_jac, jac)
        acc_index += mask[i]
        acc_state = _write_to_accumulators(acc_state, next_state, acc_index)
        acc_jac = _write_to_accumulators(acc_jac, next_jac, acc_index)

        return i + 1, acc_index, acc_state, acc_jac

      initial_acc_state = _create_accumulators(initial_state, acc_size)
      initial_acc_state = _write_to_accumulators(initial_acc_state,
                                                 initial_state, 0)

      initial_jac = _make_unit_jacobian(initial_state, params)
      initial_acc_jac = _create_accumulators(initial_jac, acc_size)
      initial_acc_jac = _write_to_accumulators(initial_acc_jac, initial_jac, 0)

      loop_vars = (0, 0, initial_acc_state, initial_acc_jac)

      _, _, final_acc_state, final_acc_jac = tf.compat.v2.while_loop(
          while_cond, while_body, loop_vars=loop_vars,
          maximum_iterations=max_iterations)

      def gradient(*ws):
        # Same as in for_loop, except we need to sum over the accumulating
        # dimension. E.g. if x = for_loop(... num_iterations=[2, 5]) and
        # y = 2*x[0] + 3*x[1], then taking gradient of y will lead to ws having
        # coeffs 2 and 3 in the acc dimension, and we should sum over it.
        ws = [tf.expand_dims(w, axis=-2) for w in ws]
        ws = [ws]  # expand dims on block level as well.

        js, jp = final_acc_jac
        ws_js, ws_jp = _block_matmul(ws, js), _block_matmul(ws, jp)

        ws_js, ws_jp = ws_js[0], ws_jp[0]
        ws_js = [tf.squeeze(t, axis=-2) for t in ws_js]
        ws_jp = [tf.squeeze(t, axis=[-2, -1]) for t in ws_jp]

        # Sum over acc axis.
        ws_js = [tf.math.reduce_sum(t, axis=0) for t in ws_js]
        # ws_jp should be 0-dimensional
        ws_jp = [tf.math.reduce_sum(t) for t in ws_jp]

        return ws_js + ws_jp

      return final_acc_state, gradient

    # tf.custom_gradient can only handle a flat sequence of args.
    args = tuple(initial_state + params)
    return inner(*args)


def _create_accumulators(nested_tensor, size):
  if isinstance(nested_tensor, tf.Tensor):
    # Single tensor.
    return tf.zeros(shape=[size] + nested_tensor.shape.as_list(),
                    dtype=nested_tensor.dtype)
  return [_create_accumulators(t, size) for t in nested_tensor]


def _write_to_accumulators(nested_acc, nested_tensor, index):
  if isinstance(nested_tensor, tf.Tensor):
    assert isinstance(nested_acc, tf.Tensor)
    acc_size = nested_acc.shape.as_list()[0]
    one_hot = tf.one_hot(index, depth=acc_size)
    one_hot = tf.reshape(one_hot, [acc_size] + [1] * len(nested_tensor.shape))
    return tf.where(one_hot > 0, nested_tensor, nested_acc)

  return [_write_to_accumulators(acc, t, index)
          for acc, t in zip(nested_acc, nested_tensor)]


def _read_from_accumulators(nested_acc, index):
  if isinstance(nested_acc, tf.Tensor):
    return nested_acc[index]
  return [_read_from_accumulators(acc, index) for acc in nested_acc]


# We don't currently expose this module as a library API, but may use it
# internally, e.g. in Monte-Carlo sampling.
__all__ = []
