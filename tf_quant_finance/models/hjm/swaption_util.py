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
"""Collection of utility functions for pricing swaptions."""

from typing import Callable, Tuple

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.models import utils

__all__ = [
    'discount_factors_and_bond_prices_from_samples'
]


def discount_factors_and_bond_prices_from_samples(
    expiries: types.RealTensor,
    payment_times: types.RealTensor,
    sample_discount_curve_paths_fn: Callable[..., Tuple[types.RealTensor,
                                                        types.RealTensor,
                                                        types.RealTensor]],
    num_samples: types.IntTensor,
    times: types.RealTensor = None,
    curve_times: types.RealTensor = None,
    dtype: tf.DType = None) -> Tuple[types.RealTensor, types.RealTensor]:
  """Utility function to compute the discount factors and the bond prices.

  Args:
    expiries: A real `Tensor` of any shape and dtype. The time to expiration of
      the swaptions. The shape of this input determines the number (and shape)
      of swaptions to be priced and the shape of the output - e.g. if there are
      two swaptions, and there are 11 payment dates for each swaption, then the
      shape of `expiries` is [2, 11], with entries repeated along the second
      axis.
    payment_times: A real `Tensor` of same dtype and compatible shape with
      `expiries` - e.g. if there are two swaptions, and there are 11 payment
      dates for each swaption, then the shape of `payment_times` should be [2,
      11]
    sample_discount_curve_paths_fn: Callable which takes the following args:
      1) times: Rank 1 `Tensor` of positive real values, specifying the times at
        which the path points are to be evaluated.
      2) curve_times: Rank 1 `Tensor` of positive real values, specifying the
        maturities at which the discount curve is to be computed at each
        simulation time.
      3) num_samples: Positive scalar integer specifying the number of paths to
        draw.
      Returns three `Tensor`s, the first being a N-D tensor of shape
        `model_batch_shape + [num_samples, m, k, d]` containing the simulated
        zero coupon bond curves, the second being a `Tensor` of shape
        `model_batch_shape + [num_samples, k, d]` containing the simulated
        short rate paths, the third `Tensor` of shape
        `model_batch_shape + [num_samples, k, d]` containing the simulated path
        discount factors. Here, m is the size of `curve_times`, k is the size
        of `times`, d is the dimensionality of the paths and
        `model_batch_shape` is shape of the batch of independent HJM models.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation.
    times: An optional rank 1 `Tensor` of increasing positive real values. The
      times at which Monte Carlo simulations are performed.
      Default value: `None`.
    curve_times: An optional rank 1 `Tensor` of positive real values. The
      maturities at which spot discount curve is computed during simulations.
      Default value: `None`.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.

  Returns:
    Two real tensors, `discount_factors` and `bond_prices`, both of shape
    [num_samples] + swaption_batch_shape + [dim], where `dim` is the dimension
    of each path (e.g for a Hull-White with two models, dim==2; while for HJM
    dim==1 always). `swaption_batch_shape` has the same rank as `expiries.shape`
    and its leading dimensions are broadcasted to `model_batch_shape`.
  """
  if times is not None:
    sim_times = tf.convert_to_tensor(times, dtype=dtype)
  else:
    # This might not be the most efficient if we have a batch of Models each
    # pricing swaptions with different expiries.
    sim_times = tf.reshape(expiries, shape=[-1])
    sim_times = tf.sort(sim_times, name='sort_sim_times')

  swaptionlet_shape = tf.shape(payment_times)
  tau = payment_times - expiries

  if curve_times is not None:
    curve_times = tf.convert_to_tensor(curve_times, dtype=dtype)
  else:
    # This might not be the most efficient if we have a batch of Models each
    # pricing swaptions with different expiries and payment times.
    curve_times = tf.reshape(tau, shape=[-1])
    curve_times, _ = tf.unique(curve_times)
    curve_times = tf.sort(curve_times, name='sort_curve_times')

  p_t_tau, r_t, discount_factors = sample_discount_curve_paths_fn(
      times=sim_times, curve_times=curve_times, num_samples=num_samples)

  dim = tf.shape(p_t_tau)[-1]
  model_batch_shape = tf.shape(p_t_tau)[:-4]
  model_batch_rank = p_t_tau.shape[:-4].rank
  instr_batch_shape = tf.shape(expiries)[model_batch_rank:]
  try:
    swaptionlet_shape = tf.concat(
        [model_batch_shape, instr_batch_shape], axis=0)
    expiries = tf.broadcast_to(expiries, swaptionlet_shape)
    tau = tf.broadcast_to(tau, swaptionlet_shape)
  except:
    raise ValueError('The leading dimensions of `expiries` of shape {} are not '
                     'compatible with the batch shape {} of the model.'.format(
                         expiries.shape.as_list(),
                         p_t_tau.shape.as_list()[:-4]))

  if discount_factors is None:
    dt = tf.concat(axis=0, values=[[0.0], sim_times[1:] - sim_times[:-1]])
    dt = tf.expand_dims(tf.expand_dims(dt, axis=-1), axis=0)

    # Transpose before (and after) because we want the cumprod along axis=1
    # but `cumsum_using_matvec` operates on the last axis. Also we use cumsum
    # and then exponentiate instead of taking cumprod of exponents for
    # efficiency.
    cumul_rdt = tf.transpose(
        utils.cumsum_using_matvec(tf.transpose(r_t * dt, perm=[0, 2, 1])),
        perm=[0, 2, 1])
    discount_factors = tf.math.exp(-cumul_rdt)

  # Make discount factors the same shape as `p_t_tau`. This involves adding
  # an extra dimension (corresponding to `curve_times`).
  discount_factors = tf.expand_dims(discount_factors, axis=model_batch_rank + 1)

  # tf.repeat is needed because we will use gather_nd later on this tensor.
  discount_factors_simulated = tf.repeat(
      discount_factors, tf.shape(p_t_tau)[model_batch_rank + 1],
      axis=model_batch_rank + 1)

  # `sim_times` and `curve_times` are sorted for simulation. We need to
  # select the indices corresponding to our input.
  new_shape = tf.concat([model_batch_shape, [-1]], axis=0)
  sim_time_index = tf.searchsorted(sim_times, tf.reshape(expiries, [-1]))
  curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))

  sim_time_index = tf.reshape(sim_time_index, new_shape)
  curve_time_index = tf.reshape(curve_time_index, new_shape)
  gather_index = tf.stack([curve_time_index, sim_time_index], axis=-1)

  # shape=[num_samples] + batch_shape + [len(sim_times_index), dim]
  discount_factors_simulated = _gather_tensor_at_swaption_payoff(
      discount_factors_simulated, gather_index)
  payoff_discount_factors = tf.reshape(
      discount_factors_simulated,
      tf.concat([[num_samples], swaptionlet_shape, [dim]], axis=0))

  # shape=[num_samples, len(sim_times_index), dim]
  p_t_tau = _gather_tensor_at_swaption_payoff(p_t_tau, gather_index)
  payoff_bond_price = tf.reshape(
      p_t_tau,
      tf.concat([[num_samples], swaptionlet_shape, [dim]], axis=0))

  return payoff_discount_factors, payoff_bond_price


def _gather_tensor_at_swaption_payoff(param, indices):
  """Returns the values of the input `Tensor` at Swaption payoff times.

  `Tensor`s such as simulated path discount factors and spot discount curves
  have shape `[batch_shape, num_samples, curve_times, sim_times, dim]`. In
  order to compute swaption payoffs at exercise times, we need to gather their
  values at `[batch_shape, num_samples, curve_times_idx, sim_times_idx, dim]`
  where `curve_times_idx` are the relevant indices corresponding to times at
  which spot discount curves are sampled and `sim_times_idx` are the relevant
  indices corresponding to simulation times.

  To achieve this task we first transpose the tensor to shape
  `[batch_shape, curve_times, sim_times, dim, num_samples]` and then use
  `tf.gather_nd`.

  Args:
    param: The `Tensor` from which values will be extracted. The shape of the
      `Tensor` is `[batch_shape, num_samples, curve_times, sim_times, dim]`.
    indices: A N-D `Tensor` of shape `batch_shape + [num_indices, 2]`. The first
      column contains the indices along the `curve_times` axis and the second
      column contains the indices along the `sim_times` axis.

  Returns:
    A `Tensor` of same dtype as `param` and shape
    `[num_samples, batch_shape, num_indices, dim]`.
  """
  batch_rank = param.shape[:-4].rank
  # Transpose to shape `[batch_shape, curve_times, sim_times, dim, num_samples]`
  perm = (list(range(batch_rank)) +
          [batch_rank + 1, batch_rank + 2, batch_rank + 3, batch_rank])
  param = tf.transpose(param, perm=perm)
  param = tf.gather_nd(param, indices, batch_dims=batch_rank)
  # Transpose to shape `[num_samples, batch_shape, num_indices, dim]`
  perm = [2 + batch_rank] + list(range(2 + batch_rank))
  param = tf.transpose(param, perm=perm)

  return param
