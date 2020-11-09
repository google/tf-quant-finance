"""Collection of utility functions for pricing options on zero coupon bonds."""
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

import tensorflow.compat.v2 as tf

from tf_quant_finance.models import utils


def options_price_from_samples(strikes,
                               expiries,
                               maturities,
                               is_call_options,
                               sample_discount_curve_paths_fn,
                               num_samples,
                               time_step,
                               dtype=None):
  """Computes the zero coupon bond options price from simulated discount curves.

  Args:
    strikes: A real `Tensor` of any shape and dtype. The strike price of the
      options. The shape of this input determines the number (and shape) of the
      options to be priced and the output.
    expiries: A real `Tensor` of the same dtype and compatible shape as
      `strikes`.  The time to expiry of each bond option.
    maturities: A real `Tensor` of the same dtype and compatible shape as
      `strikes`.  The time to maturity of the underlying zero coupon bonds.
    is_call_options: A boolean `Tensor` of a shape compatible with `strikes`.
      Indicates whether the option is a call (if True) or a put (if False).
    sample_discount_curve_paths_fn: Callable which takes the following args:

      1) times: Rank 1 `Tensor` of positive real values, specifying the times at
        which the path points are to be evaluated.
      2) curve_times: Rank 1 `Tensor` of positive real values, specifying the
        maturities at which the discount curve is to be computed at each
        simulation time.
      3) num_samples: Positive scalar integer specifying the number of paths to
        draw.

      and returns two `Tensor`s, the first being a Rank-4 tensor of shape
        [num_samples, m, k, d] containing the simulated zero coupon bond curves,
        and the second being a `Tensor` of shape [num_samples, k, d] containing
        the simulated short rate paths. Here, m is the size of `curve_times`, k
        is the size of `times`, and d is the dimensionality of the paths.

    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation.
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.

  Returns:
    A `Tensor` of real dtype and shape `strikes.shape + [dim]` containing the
    computed option prices.
  """
  sim_times, _ = tf.unique(tf.reshape(expiries, shape=[-1]))
  longest_expiry = tf.reduce_max(sim_times)
  sim_times, _ = tf.unique(
      tf.concat(
          [sim_times, tf.range(time_step, longest_expiry, time_step)], axis=0))
  sim_times = tf.sort(sim_times, name='sort_sim_times')
  tau = maturities - expiries
  curve_times_builder, _ = tf.unique(tf.reshape(tau, shape=[-1]))
  curve_times = tf.sort(curve_times_builder, name='sort_curve_times')

  p_t_tau, r_t = sample_discount_curve_paths_fn(
      times=sim_times, curve_times=curve_times, num_samples=num_samples)
  dim = p_t_tau.shape[-1]

  dt_builder = tf.concat(
      axis=0,
      values=[
          tf.convert_to_tensor([0.0], dtype=dtype),
          sim_times[1:] - sim_times[:-1]
      ])
  dt = tf.expand_dims(tf.expand_dims(dt_builder, axis=-1), axis=0)
  discount_factors_builder = tf.math.exp(-r_t * dt)
  # Transpose before (and after) because we want the cumprod along axis=1
  # and `matvec` operates on the last axis. The shape before and after would
  # be `(num_samples, len(times), dim)`
  discount_factors_builder = tf.transpose(
      utils.cumprod_using_matvec(
          tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])

  # make discount factors the same shape as `p_t_tau`. This involves adding
  # an extra dimenstion (corresponding to `curve_times`).
  discount_factors_builder = tf.expand_dims(discount_factors_builder, axis=1)
  discount_factors_simulated = tf.repeat(
      discount_factors_builder, p_t_tau.shape.as_list()[1], axis=1)

  # `sim_times` and `curve_times` are sorted for simulation. We need to
  # select the indices corresponding to our input.
  sim_time_index = tf.searchsorted(sim_times, tf.reshape(expiries, [-1]))
  curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))

  gather_index = _prepare_indices(
      tf.range(0, num_samples), curve_time_index, sim_time_index,
      tf.range(0, dim))

  # The shape after `gather_nd` would be (num_samples*num_strikes*dim,)
  payoff_discount_factors_builder = tf.gather_nd(discount_factors_simulated,
                                                 gather_index)
  # Reshape to `[num_samples] + strikes.shape + [dim]`
  payoff_discount_factors = tf.reshape(payoff_discount_factors_builder,
                                       [num_samples] + strikes.shape + [dim])
  payoff_bond_price_builder = tf.gather_nd(p_t_tau, gather_index)
  payoff_bond_price = tf.reshape(payoff_bond_price_builder,
                                 [num_samples] + strikes.shape + [dim])

  is_call_options = tf.reshape(
      tf.broadcast_to(is_call_options, strikes.shape),
      [1] + strikes.shape + [1])

  strikes = tf.reshape(strikes, [1] + strikes.shape + [1])
  payoff = tf.where(is_call_options,
                    tf.math.maximum(payoff_bond_price - strikes, 0.0),
                    tf.math.maximum(strikes - payoff_bond_price, 0.0))
  option_value = tf.math.reduce_mean(payoff_discount_factors * payoff, axis=0)

  return option_value


def _prepare_indices(idx0, idx1, idx2, idx3):
  """Prepare indices to get relevant slice from discount curve simulations."""
  len0 = idx0.shape.as_list()[0]
  len1 = idx1.shape.as_list()[0]
  len3 = idx3.shape.as_list()[0]
  idx0 = tf.repeat(idx0, len1 * len3)
  idx1 = tf.tile(tf.repeat(idx1, len3), [len0])
  idx2 = tf.tile(tf.repeat(idx2, len3), [len0])
  idx3 = tf.tile(idx3, [len0 * len1])

  return tf.stack([idx0, idx1, idx2, idx3], axis=-1)
