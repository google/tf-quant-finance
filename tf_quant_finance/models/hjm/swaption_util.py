# Lint as: python3
# Copyright 2020 Google LLC
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

import tensorflow.compat.v2 as tf

from tf_quant_finance.models import utils


def discount_factors_and_bond_prices_from_samples(
    expiries,
    payment_times,
    sample_discount_curve_paths_fn,
    num_samples,
    times=None,
    curve_times=None,
    dtype=None):
  """Utility function to compute the discount factors and the bond prices.

  Args:
    expiries: A real `Tensor` of any and dtype. The time to expiration of the
      swaptions. The shape of this input determines the number (and shape) of
      swaptions to be priced and the shape of the output - e.g. if there are two
      swaptions, and there are 11 payment dates for each swaption, then the
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
        draw.  Returns two `Tensor`s, the first being a Rank-4 tensor of shape
        [num_samples, m, k, d] containing the simulated zero coupon bond curves,
        and the second being a `Tensor` of shape [num_samples, k, d] containing
        the simulated short rate paths. Here, m is the size of `curve_times`, k
        is the size of `times`, and d is the dimensionality of the paths.
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
    [num_samples] + shape(payment_times) + [dim], where `dim` is the dimension
    of each path (e.g for a Hull-White with two models, dim==2; while for HJM
    dim==1 always.)
  """
  if times is not None:
    sim_times = tf.convert_to_tensor(times, dtype=dtype)
  else:
    sim_times = tf.reshape(expiries, shape=[-1])
    sim_times = tf.sort(sim_times, name='sort_sim_times')

  swaptionlet_shape = tf.shape(payment_times)
  tau = payment_times - expiries

  if curve_times is not None:
    curve_times = tf.convert_to_tensor(curve_times, dtype=dtype)
  else:
    curve_times = tf.reshape(tau, shape=[-1])
    curve_times, _ = tf.unique(curve_times)
    curve_times = tf.sort(curve_times, name='sort_curve_times')

  p_t_tau, r_t, discount_factors = sample_discount_curve_paths_fn(
      times=sim_times, curve_times=curve_times, num_samples=num_samples)
  dim = tf.shape(p_t_tau)[-1]

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
  # an extra dimenstion (corresponding to `curve_times`).
  discount_factors = tf.expand_dims(discount_factors, axis=1)

  # tf.repeat is needed because we will use gather_nd later on this tensor.
  discount_factors_simulated = tf.repeat(
      discount_factors, tf.shape(p_t_tau)[1], axis=1)

  # `sim_times` and `curve_times` are sorted for simulation. We need to
  # select the indices corresponding to our input.
  sim_time_index = tf.searchsorted(sim_times, tf.reshape(expiries, [-1]))
  curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))

  gather_index = _prepare_indices_ijjk(
      tf.range(0, num_samples), curve_time_index, sim_time_index,
      tf.range(0, dim))

  # The shape after `gather_nd` will be `(num_samples*num_swaptionlets*dim,)`
  payoff_discount_factors_builder = tf.gather_nd(discount_factors_simulated,
                                                 gather_index)
  # Reshape to `[num_samples] + swaptionlet.shape + [dim]`
  payoff_discount_factors = tf.reshape(
      payoff_discount_factors_builder,
      tf.concat([[num_samples], swaptionlet_shape, [dim]], axis=0))
  payoff_bond_price_builder = tf.gather_nd(p_t_tau, gather_index)
  payoff_bond_price = tf.reshape(
      payoff_bond_price_builder,
      tf.concat([[num_samples], swaptionlet_shape, [dim]], axis=0))

  return payoff_discount_factors, payoff_bond_price


def _prepare_indices_ijjk(idx0, idx1, idx2, idx3):
  """Prepares indices to get x[i, j, j, k]."""
  # For a 4-D `Tensor` x, creates indices for tf.gather_nd to retrieve
  # x[i, j, j, k].
  len0 = tf.shape(idx0)[0]
  len1 = tf.shape(idx1)[0]
  len3 = tf.shape(idx3)[0]
  idx0 = tf.repeat(idx0, len1 * len3)
  idx1 = tf.tile(tf.repeat(idx1, len3), [len0])
  idx2 = tf.tile(tf.repeat(idx2, len3), [len0])
  idx3 = tf.tile(idx3, [len0 * len1])

  return tf.stack([idx0, idx1, idx2, idx3], axis=-1)
