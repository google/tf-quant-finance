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
"""The Milstein sampling method for ito processes."""

import functools
import math
from typing import Callable, List, Optional

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
from tf_quant_finance.math import custom_loops
from tf_quant_finance.math import gradient
from tf_quant_finance.math import random_ops as random
from tf_quant_finance.models import utils


_PI = math.pi
_SQRT_2 = np.sqrt(2.)


def sample(
    *,
    dim: int,
    drift_fn: Callable[..., types.RealTensor],
    volatility_fn: Callable[..., types.RealTensor],
    times: types.RealTensor,
    time_step: Optional[types.RealTensor] = None,
    num_time_steps: Optional[types.IntTensor] = None,
    num_samples: types.IntTensor = 1,
    initial_state: Optional[types.RealTensor] = None,
    grad_volatility_fn: Optional[Callable[..., List[types.RealTensor]]] = None,
    random_type: Optional[random.RandomType] = None,
    seed: Optional[types.IntTensor] = None,
    swap_memory: bool = True,
    skip: types.IntTensor = 0,
    precompute_normal_draws: bool = True,
    watch_params: Optional[List[types.RealTensor]] = None,
    stratonovich_order: int = 5,
    dtype: Optional[tf.DType] = None,
    name: Optional[str] = None) -> types.RealTensor:
  r"""Returns a sample paths from the process using the Milstein method.

  For an Ito process,

  ```
    dX = a(t, X_t) dt + b(t, X_t) dW_t
  ```
  given drift `a`, volatility `b` and derivative of volatility `b'`, the
  Milstein method generates a
  sequence {Y_n} approximating X

  ```
  Y_{n+1} = Y_n + a(t_n, Y_n) dt + b(t_n, Y_n) dW_n + \frac{1}{2} b(t_n, Y_n)
  b'(t_n, Y_n) ((dW_n)^2 - dt)
  ```
  where `dt = t_{n+1} - t_n`, `dW_n = (N(0, t_{n+1}) - N(0, t_n))` and `N` is a
  sample from the Normal distribution.

  In higher dimensions, when `a(t, X_t)` is a d-dimensional vector valued
  function and `W_t` is a d-dimensional Wiener process, we have for the kth
  element of the expansion:

  ```
  Y_{n+1}[k] = Y_n[k] + a(t_n, Y_n)[k] dt + \sum_{j=1}^d b(t_n, Y_n)[k, j]
  dW_n[j] + \sum_{j_1=1}^d \sum_{j_2=1}^d L_{j_1} b(t_n, Y_n)[k, j_2] I(j_1,
  j_2)
  ```
  where `L_{j} = \sum_{i=1}^d b(t_n, Y_n)[i, j] \frac{\partial}{\partial x^i}`
  is an operator and `I(j_1, j_2) = \int_{t_n}^{t_{n+1}} \int_{t_n}^{s_1}
  dW_{s_2}[j_1] dW_{s_1}[j_2]` is a multiple Ito integral.


  See [1] and [2] for details.

  #### References
  [1]: Wikipedia. Milstein method:
  https://en.wikipedia.org/wiki/Milstein_method
  [2]: Peter E. Kloeden,  Eckhard Platen. Numerical Solution of Stochastic
    Differential Equations. Springer. 1992

  Args:
    dim: Python int greater than or equal to 1. The dimension of the Ito
      Process.
    drift_fn: A Python callable to compute the drift of the process. The
      callable should accept two real `Tensor` arguments of the same dtype. The
      first argument is the scalar time t, the second argument is the value of
      Ito process X - tensor of shape `batch_shape + [dim]`. The result is
      value of drift a(t, X). The return value of the callable is a real
      `Tensor` of the same dtype as the input arguments and of shape
      `batch_shape + [dim]`.
    volatility_fn: A Python callable to compute the volatility of the process.
      The callable should accept two real `Tensor` arguments of the same dtype
      as `times`. The first argument is the scalar time t, the second argument
      is the value of Ito process X - tensor of shape `batch_shape + [dim]`. The
      result is value of volatility b(t, X). The return value of the callable is
      a real `Tensor` of the same dtype as the input arguments and of shape
      `batch_shape + [dim, dim]`.
    times: Rank 1 `Tensor` of increasing positive real values. The times at
      which the path points are to be evaluated.
    time_step: An optional scalar real `Tensor` - maximal distance between
      points in grid in Milstein schema.
      Either this or `num_time_steps` should be supplied.
      Default value: `None`.
    num_time_steps: An optional Scalar integer `Tensor` - a total number of time
      steps performed by the algorithm. The maximal distance between points in
      grid is bounded by `times[-1] / (num_time_steps - times.shape[0])`.
      Either this or `time_step` should be supplied.
      Default value: `None`.
    num_samples: Positive scalar `int`. The number of paths to draw.
      Default value: 1.
    initial_state: `Tensor` of shape `[dim]`. The initial state of the
      process.
      Default value: None which maps to a zero initial state.
    grad_volatility_fn: An optional python callable to compute the gradient of
      `volatility_fn`. The callable should accept three real `Tensor` arguments
      of the same dtype as `times`. The first argument is the scalar time t. The
      second argument is the value of Ito process X - tensor of shape
      `batch_shape + [dim]`. The third argument is a tensor of input gradients
      of shape `batch_shape + [dim]` to pass to `gradient.fwd_gradient`. The
      result is a list of values corresponding to the forward gradient of
      volatility b(t, X) with respect to X. The return value of the callable is
      a list of size `dim` containing real `Tensor`s of the same dtype as the
      input arguments and of shape `batch_shape + [dim, dim]`. Each index of the
      list corresponds to a dimension of the state. If `None`, the gradient is
      computed from `volatility_fn` using forward differentiation.
    random_type: Enum value of `RandomType`. The type of (quasi)-random number
      generator to use to generate the paths.
      Default value: None which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of `[STATELESS, PSEUDO, HALTON_RANDOMIZED,
      PSEUDO_ANTITHETIC, STATELESS_ANTITHETIC]`. For `PSEUDO`,
      `PSEUDO_ANTITHETIC` and `HALTON_RANDOMIZED` the seed should be a Python
      integer. For `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as
      an integer `Tensor` of shape `[2]`.
      Default value: `None` which means no seed is set.
    swap_memory: A Python bool. Whether GPU-CPU memory swap is enabled for this
      op. See an equivalent flag in `tf.while_loop` documentation for more
      details. Useful when computing a gradient of the op since `tf.while_loop`
      is used to propagate stochastic process in time.
      Default value: True.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    precompute_normal_draws: Python bool. Indicates whether the noise increments
      `N(0, t_{n+1}) - N(0, t_n)` are precomputed. For `HALTON` and `SOBOL`
      random types the increments are always precomputed. While the resulting
      graph consumes more memory, the performance gains might be significant.
      Default value: `True`.
    watch_params: An optional list of zero-dimensional `Tensor`s of the same
      `dtype` as `initial_state`. If provided, specifies `Tensor`s with respect
      to which the differentiation of the sampling function will happen. A more
      efficient algorithm is used when `watch_params` are specified. Note the
      the function becomes differentiable only wrt to these `Tensor`s and the
      `initial_state`. The gradient wrt any other `Tensor` is set to be zero.
    stratonovich_order: A positive integer. The number of terms to use when
      calculating the approximate Stratonovich integrals in the multidimensional
      scheme. Stratonovich integrals are an alternative to Ito integrals, and
      can be used interchangeably when defining the higher order terms in the
      update equation. We use Stratonovich integrals here because they have a
      convenient approximation scheme for calculating cross terms involving
      different components of the Wiener process. See Eq. 8.10 in Section 5.8 of
      [2]. Default value: `5`.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which means that the dtype implied by `times` is used.
    name: Python string. The name to give this op.
      Default value: `None` which maps to `milstein_sample`.
  """
  name = name or 'milstein_sample'
  with tf.name_scope(name):
    if stratonovich_order <= 0:
      raise ValueError('`stratonovich_order` must be a positive integer.')
    times = tf.convert_to_tensor(times, dtype=dtype)
    if dtype is None:
      dtype = times.dtype
    if initial_state is None:
      initial_state = tf.zeros(dim, dtype=dtype)
    initial_state = tf.convert_to_tensor(
        initial_state, dtype=dtype, name='initial_state')
    num_requested_times = tff_utils.get_shape(times)[0]
    # Create a time grid for the Milstein scheme.
    if num_time_steps is not None and time_step is not None:
      raise ValueError('Only one of either `num_time_steps` or `time_step` '
                       'should be defined but not both')
    if time_step is None:
      if num_time_steps is None:
        raise ValueError('Either `num_time_steps` or `time_step` should be '
                         'defined.')
      num_time_steps = tf.convert_to_tensor(
          num_time_steps, dtype=tf.int32, name='num_time_steps')
      time_step = times[-1] / tf.cast(num_time_steps, dtype=dtype)
    else:
      time_step = tf.convert_to_tensor(time_step, dtype=dtype,
                                       name='time_step')
    times, keep_mask, time_indices = utils.prepare_grid(
        times=times, time_step=time_step, num_time_steps=num_time_steps,
        dtype=dtype)
    if watch_params is not None:
      watch_params = [
          tf.convert_to_tensor(param, dtype=dtype) for param in watch_params
      ]
    if grad_volatility_fn is None:

      def _grad_volatility_fn(current_time, current_state, input_gradients):
        return gradient.fwd_gradient(
            functools.partial(volatility_fn, current_time),
            current_state,
            input_gradients=input_gradients,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

      grad_volatility_fn = _grad_volatility_fn

    input_gradients = None
    if dim > 1:
      input_gradients = tf.unstack(tf.eye(dim, dtype=dtype))
      input_gradients = [
          tf.broadcast_to(start, [num_samples, dim])
          for start in input_gradients
      ]

    return _sample(
        dim=dim,
        drift_fn=drift_fn,
        volatility_fn=volatility_fn,
        grad_volatility_fn=grad_volatility_fn,
        times=times,
        time_step=time_step,
        keep_mask=keep_mask,
        num_requested_times=num_requested_times,
        num_samples=num_samples,
        initial_state=initial_state,
        random_type=random_type,
        seed=seed,
        swap_memory=swap_memory,
        skip=skip,
        precompute_normal_draws=precompute_normal_draws,
        watch_params=watch_params,
        time_indices=time_indices,
        input_gradients=input_gradients,
        stratonovich_order=stratonovich_order,
        dtype=dtype)


def _sample(*, dim, drift_fn, volatility_fn, grad_volatility_fn, times,
            time_step, keep_mask, num_requested_times, num_samples,
            initial_state, random_type, seed, swap_memory, skip,
            precompute_normal_draws, watch_params, time_indices,
            input_gradients, stratonovich_order, dtype):
  """Returns a sample of paths from the process using the Milstein method."""
  dt = times[1:] - times[:-1]
  sqrt_dt = tf.sqrt(dt)
  current_state = initial_state + tf.zeros([num_samples, dim],
                                           dtype=initial_state.dtype)
  if dt.shape.is_fully_defined():
    steps_num = dt.shape.as_list()[-1]
  else:
    steps_num = tf.shape(dt)[-1]
  # In order to use low-discrepancy random_type we need to generate the sequence
  # of independent random normals upfront. We also precompute random numbers
  # for stateless random type in order to ensure independent samples for
  # multiple function calls with different seeds.
  if precompute_normal_draws or random_type in (
      random.RandomType.SOBOL, random.RandomType.HALTON,
      random.RandomType.HALTON_RANDOMIZED, random.RandomType.STATELESS,
      random.RandomType.STATELESS_ANTITHETIC):
    # Process dimension plus auxiliary random variables for stratonovich
    # integral computation.
    all_normal_draws = utils.generate_mc_normal_draws(
        num_normal_draws=dim + 3 * dim * stratonovich_order,
        num_time_steps=steps_num,
        num_sample_paths=num_samples,
        random_type=random_type,
        dtype=dtype,
        seed=seed,
        skip=skip)
    normal_draws = all_normal_draws[:, :, :dim]
    wiener_mean = None
    # Auxiliary normal draws for use with the stratonovich integral
    # approximation.
    aux_normal_draws = []
    start = dim
    for _ in range(3):
      end = start + dim * stratonovich_order
      aux_normal_draws.append(all_normal_draws[:, :, start:end])
      start = end
  else:
    # If pseudo or anthithetic sampling is used, proceed with random sampling
    # at each step.
    wiener_mean = tf.zeros((dim,), dtype=dtype, name='wiener_mean')
    normal_draws = None
    aux_normal_draws = None
  if watch_params is None:
    # Use while_loop if `watch_params` is not passed
    return _while_loop(
        dim=dim,
        steps_num=steps_num,
        current_state=current_state,
        drift_fn=drift_fn,
        volatility_fn=volatility_fn,
        grad_volatility_fn=grad_volatility_fn,
        wiener_mean=wiener_mean,
        num_samples=num_samples,
        times=times,
        dt=dt,
        sqrt_dt=sqrt_dt,
        time_step=time_step,
        keep_mask=keep_mask,
        num_requested_times=num_requested_times,
        swap_memory=swap_memory,
        random_type=random_type,
        seed=seed,
        normal_draws=normal_draws,
        input_gradients=input_gradients,
        stratonovich_order=stratonovich_order,
        aux_normal_draws=aux_normal_draws,
        dtype=dtype)
  else:
    # Use custom for_loop if `watch_params` is specified
    return _for_loop(
        dim=dim,
        steps_num=steps_num,
        current_state=current_state,
        drift_fn=drift_fn,
        volatility_fn=volatility_fn,
        grad_volatility_fn=grad_volatility_fn,
        wiener_mean=wiener_mean,
        num_samples=num_samples,
        times=times,
        dt=dt,
        sqrt_dt=sqrt_dt,
        time_indices=time_indices,
        keep_mask=keep_mask,
        watch_params=watch_params,
        random_type=random_type,
        seed=seed,
        normal_draws=normal_draws,
        input_gradients=input_gradients,
        stratonovich_order=stratonovich_order,
        aux_normal_draws=aux_normal_draws)


def _while_loop(*, dim, steps_num, current_state, drift_fn, volatility_fn,
                grad_volatility_fn, wiener_mean, num_samples, times, dt,
                sqrt_dt, time_step, num_requested_times, keep_mask, swap_memory,
                random_type, seed, normal_draws, input_gradients,
                stratonovich_order, aux_normal_draws, dtype):
  """Sample paths using tf.while_loop."""
  written_count = 0
  if isinstance(num_requested_times, int) and num_requested_times == 1:
    record_samples = False
    result = current_state
  else:
    # If more than one sample has to be recorded, create a TensorArray
    record_samples = True
    element_shape = current_state.shape
    result = tf.TensorArray(dtype=dtype,
                            size=num_requested_times,
                            element_shape=element_shape,
                            clear_after_read=False)
    # Include initial state, if necessary
    result = result.write(written_count, current_state)
  written_count += tf.cast(keep_mask[0], dtype=tf.int32)
  # Define sampling while_loop body function
  def cond_fn(i, written_count, *args):
    # It can happen that `times_grid[-1] > times[-1]` in which case we have
    # to terminate when `written_count` reaches `num_requested_times`
    del args
    return tf.math.logical_and(i < steps_num,
                               written_count < num_requested_times)
  def step_fn(i, written_count, current_state, result):
    return _milstein_step(
        dim=dim,
        i=i,
        written_count=written_count,
        current_state=current_state,
        result=result,
        drift_fn=drift_fn,
        volatility_fn=volatility_fn,
        grad_volatility_fn=grad_volatility_fn,
        wiener_mean=wiener_mean,
        num_samples=num_samples,
        times=times,
        dt=dt,
        sqrt_dt=sqrt_dt,
        keep_mask=keep_mask,
        random_type=random_type,
        seed=seed,
        normal_draws=normal_draws,
        input_gradients=input_gradients,
        stratonovich_order=stratonovich_order,
        aux_normal_draws=aux_normal_draws,
        record_samples=record_samples)

  maximum_iterations = (
      tf.cast(1. / time_step, dtype=tf.int32) + tf.size(times))
  # Sample paths
  _, _, _, result = tf.while_loop(
      cond_fn,
      step_fn, (0, written_count, current_state, result),
      maximum_iterations=maximum_iterations,
      swap_memory=swap_memory)
  if not record_samples:
    # shape [num_samples, 1, dim]
    return tf.expand_dims(result, axis=-2)
  # Shape [num_time_points] + [num_samples, dim]
  result = result.stack()
  # transpose to shape [num_samples, num_time_points, dim]
  n = result.shape.rank
  perm = list(range(1, n-1)) + [0, n - 1]
  return tf.transpose(result, perm)


def _for_loop(*, dim, steps_num, current_state, drift_fn, volatility_fn,
              grad_volatility_fn, wiener_mean, watch_params, num_samples, times,
              dt, sqrt_dt, time_indices, keep_mask, random_type, seed,
              normal_draws, input_gradients, stratonovich_order,
              aux_normal_draws):
  """Sample paths using custom for_loop."""
  num_time_points = time_indices.shape.as_list()[-1]
  if num_time_points == 1:
    iter_nums = steps_num
  else:
    iter_nums = time_indices

  def step_fn(i, current_state):
    # Unpack current_state
    current_state = current_state[0]
    _, _, next_state, _ = _milstein_step(
        dim=dim,
        i=i,
        written_count=0,
        current_state=current_state,
        result=tf.expand_dims(current_state, axis=1),
        drift_fn=drift_fn,
        volatility_fn=volatility_fn,
        grad_volatility_fn=grad_volatility_fn,
        wiener_mean=wiener_mean,
        num_samples=num_samples,
        times=times,
        dt=dt,
        sqrt_dt=sqrt_dt,
        keep_mask=keep_mask,
        random_type=random_type,
        seed=seed,
        normal_draws=normal_draws,
        input_gradients=input_gradients,
        stratonovich_order=stratonovich_order,
        aux_normal_draws=aux_normal_draws,
        record_samples=False)
    return [next_state]

  result = custom_loops.for_loop(
      body_fn=step_fn,
      initial_state=[current_state],
      params=watch_params,
      num_iterations=iter_nums)[0]
  if num_time_points == 1:
    return tf.expand_dims(result, axis=1)
  return tf.transpose(result, (1, 0, 2))


def _outer_prod(v1, v2):
  """Computes the outer product of v1 and v2."""
  return tf.linalg.einsum('...i,...j->...ij', v1, v2)


def _stratonovich_integral(dim, dt, sqrt_dt, dw, stratonovich_draws, order):
  """Approximate Stratonovich integrals J(i, j).



  Args:
    dim: An integer. The dimension of the state.
    dt: A double. The time step.
    sqrt_dt: A double. The square root of dt.
    dw: A double. The Wiener increment.
    stratonovich_draws: A list of tensors corresponding to the independent
      N(0,1) random variables used in the approximation.
    order: An integer. The stratonovich_order.

  Returns:
    A Tensor of shape [dw.shape[0], dim, dim] corresponding to the Stratonovich
    integral for each pairwise component of the Wiener process. In other words,
    J(i,j) corresponds to an integral over W_i and W_j.
  """
  p = order - 1
  sqrt_rho_p = tf.sqrt(
      tf.constant(
          1 / 12 - sum([1 / r**2 for r in range(1, order + 1)]) / 2 / _PI**2,
          dtype=dw.dtype))
  mu = stratonovich_draws[0]
  # Move dimensions around to make computation easier later.
  zeta = tf.transpose(stratonovich_draws[1], [2, 0, 1])
  eta = tf.transpose(stratonovich_draws[2], [2, 0, 1])
  xi = dw / sqrt_dt
  r_i = tf.stack([
      tf.ones(zeta[0, ...].shape + [dim], dtype=zeta.dtype) / r
      for r in range(1, order + 1)
  ], 0)

  # See Eq 3.7 of section 10.3 in [2]
  # First term scaled by dt.
  value = dt * (
      _outer_prod(dw, dw) / 2 + sqrt_rho_p *
      (_outer_prod(mu[..., p], xi) - _outer_prod(xi, mu[..., p])))

  # Vectorized sum over r scaled by dt / 2 / pi.
  value += dt * tf.reduce_sum(
      tf.multiply((_outer_prod(zeta, _SQRT_2 * xi + eta) -
                   _outer_prod(_SQRT_2 * xi + eta, zeta)), r_i), 0) / (2 * _PI)
  return value


def _milstein_hot(dim, vol, grad_vol, dt, sqrt_dt, dw, stratonovich_draws,
                  stratonovich_order):
  """Higher order terms for Milstein update."""
  # Generate approximate Stratonovich integrals J(i,j) then replace the diagonal
  # with exact values.
  offdiag = _stratonovich_integral(
      dim=dim,
      dt=dt,
      sqrt_dt=sqrt_dt,
      dw=dw,
      stratonovich_draws=stratonovich_draws,
      order=stratonovich_order)
  stratonovich_integrals = tf.linalg.set_diag(offdiag, dw * dw / 2)

  # Compute L_bar^{j1} b^{k, j2} J(j1, j2)
  # See Eq 3.4 of section 10.3 in [2]
  stacked_grad_vol = []
  for state_ix in range(dim):
    stacked_grad_vol.append(
        tf.transpose(
            tf.stack([x[..., state_ix, :] for x in grad_vol], -1), [0, 2, 1]))
  stacked_grad_vol = tf.stack(stacked_grad_vol, 0)
  lbar = tf.matmul(stacked_grad_vol, vol)
  return tf.transpose(
      tf.reduce_sum(tf.multiply(lbar, stratonovich_integrals), [-2, -1]))


def _stratonovich_drift_update(num_samples, vol, grad_vol):
  """Updates drift function for use with stratonovich integrals."""
  # Compute 1/2 \sum_j^m L_bar^j b^{k,j}
  # See Eq 1.3 of section 10.1 in [2]
  vol = tf.reshape(vol, [num_samples, -1])
  grad_vol = tf.concat(grad_vol, 2)
  # A tensor of shape [num_samples, dim].
  return tf.linalg.matvec(grad_vol, vol)


def _milstein_1d(dw, dt, sqrt_dt, current_state, drift, vol, grad_vol):
  """Performs the milstein update in one dimension."""
  dw = dw * sqrt_dt
  dt_inc = dt * drift  # pylint: disable=not-callable
  dw_inc = tf.linalg.matvec(vol, dw)  # pylint: disable=not-callable
  # Higher order terms. For dim 1, the product here is elementwise.
  hot_vol = tf.squeeze(tf.multiply(vol, grad_vol), -1)
  hot_dw = dw * dw - dt
  hot_inc = tf.multiply(hot_vol, hot_dw) / 2
  return current_state + dt_inc + dw_inc + hot_inc


def _milstein_nd(dim, num_samples, dw, dt, sqrt_dt, current_state, drift, vol,
                 grad_vol, stratonovich_draws, stratonovich_order):
  """Performs the milstein update in multiple dimensions."""
  dw = dw * sqrt_dt
  drift_update = _stratonovich_drift_update(num_samples, vol, grad_vol)
  dt_inc = dt * (drift - drift_update)  # pylint: disable=not-callable
  dw_inc = tf.linalg.matvec(vol, dw)  # pylint: disable=not-callable
  hot_inc = _milstein_hot(
      dim=dim,
      vol=vol,
      grad_vol=grad_vol,
      dt=dt,
      sqrt_dt=sqrt_dt,
      dw=dw,
      stratonovich_draws=stratonovich_draws,
      stratonovich_order=stratonovich_order)
  return current_state + dt_inc + dw_inc + hot_inc


def _milstein_step(*, dim, i, written_count, current_state, result, drift_fn,
                   volatility_fn, grad_volatility_fn, wiener_mean, num_samples,
                   times, dt, sqrt_dt, keep_mask, random_type, seed,
                   normal_draws, input_gradients, stratonovich_order,
                   aux_normal_draws, record_samples):
  """Performs one step of Milstein scheme."""
  current_time = times[i + 1]
  written_count = tf.cast(written_count, tf.int32)
  if normal_draws is not None:
    dw = normal_draws[i]
  else:
    dw = random.mv_normal_sample((num_samples,),
                                 mean=wiener_mean,
                                 random_type=random_type,
                                 seed=seed)
  if aux_normal_draws is not None:
    stratonovich_draws = []
    for j in range(3):
      stratonovich_draws.append(
          tf.reshape(aux_normal_draws[j][i],
                     [num_samples, dim, stratonovich_order]))
  else:
    stratonovich_draws = []
    # Three sets of normal draws for stratonovich integrals.
    for j in range(3):
      stratonovich_draws.append(
          random.mv_normal_sample((num_samples,),
                                  mean=tf.zeros(
                                      (dim, stratonovich_order),
                                      dtype=current_state.dtype,
                                      name='stratonovich_draws_{}'.format(j)),
                                  random_type=random_type,
                                  seed=seed))

  if dim == 1:
    drift = drift_fn(current_time, current_state)
    vol = volatility_fn(current_time, current_state)
    grad_vol = grad_volatility_fn(current_time, current_state,
                                  tf.ones_like(current_state))
    next_state = _milstein_1d(
        dw=dw,
        dt=dt[i],
        sqrt_dt=sqrt_dt[i],
        current_state=current_state,
        drift=drift,
        vol=vol,
        grad_vol=grad_vol)
  else:
    drift = drift_fn(current_time, current_state)
    vol = volatility_fn(current_time, current_state)
    # This is a list of size equal to the dimension of the state space `dim`.
    # It contains tensors of shape [num_samples, dim, wiener_dim] representing
    # the gradient of the volatility function. In our case, the dimension of the
    # wiener process `wiener_dim` is equal to the state dimension `dim`.
    grad_vol = [
        grad_volatility_fn(current_time, current_state, start)
        for start in input_gradients
    ]
    next_state = _milstein_nd(
        dim=dim,
        num_samples=num_samples,
        dw=dw,
        dt=dt[i],
        sqrt_dt=sqrt_dt[i],
        current_state=current_state,
        drift=drift,
        vol=vol,
        grad_vol=grad_vol,
        stratonovich_draws=stratonovich_draws,
        stratonovich_order=stratonovich_order)
  if record_samples:
    result = result.write(written_count, next_state)
  else:
    result = next_state
  written_count += tf.cast(keep_mask[i + 1], dtype=tf.int32)

  return i + 1, written_count, next_state, result


__all__ = ['sample']
