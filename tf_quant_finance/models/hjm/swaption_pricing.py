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
"""Pricing of the Interest rate Swaption using the HJM model."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.models.hjm import quasi_gaussian_hjm
from tf_quant_finance.models.hjm import swaption_util


def price(*,
          expiries,
          fixed_leg_payment_times,
          fixed_leg_daycount_fractions,
          fixed_leg_coupon,
          reference_rate_fn,
          num_hjm_factors,
          mean_reversion,
          volatility,
          time_step,
          notional=None,
          is_payer_swaption=None,
          num_samples=1,
          random_type=None,
          seed=None,
          skip=0,
          dtype=None,
          name=None):
  """Calculates the price of European swaptions using the HJM model.

  A European Swaption is a contract that gives the holder an option to enter a
  swap contract at a future date at a prespecified fixed rate. A swaption that
  grants the holder the right to pay fixed rate and receive floating rate is
  called a payer swaption while the swaption that grants the holder the right to
  receive fixed and pay floating payments is called the receiver swaption.
  Typically the start date (or the inception date) of the swap coincides with
  the expiry of the swaption. Mid-curve swaptions are currently not supported
  (b/160061740).

  This implementation uses the HJM model to numerically value the swaption via
  Monte-Carlo. For more information on the formulation of the HJM model, see
  quasi_gaussian_hjm.py.


  #### References:
    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.
    Second Edition. 2007. Section 6.7, page 237.

  Args:
    expiries: A real `Tensor` of any shape and dtype. The time to expiration of
      the swaptions. The shape of this input determines the number (and shape)
      of swaptions to be priced and the shape of the output.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as `expiries`.
      The payment times for each payment in the fixed leg. The shape of this
      input should be `expiries.shape + [n]` where `n` denotes the number of
      fixed payments in each leg. The `fixed_leg_payment_times` should be
      greater-than or equal-to the corresponding expiries.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions for
      each payment in the fixed leg.
    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape as
      `fixed_leg_payment_times`. The fixed rate for each payment in the fixed
      leg.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape +
      [num_hjm_factors]`. Returns the continuously compounded zero rate at the
      present time for the input expiry time.
    num_hjm_factors: A Python scalar which corresponds to the number of factors
      in the HJM model to be used for pricing.
    mean_reversion: A real positive `Tensor` of shape `[num_hjm_factors]`.
      Corresponds to the mean reversion rate of each factor.
    volatility: A real positive `Tensor` of the same `dtype` and shape as
      `mean_reversion` or a callable with the following properties: (a)  The
        callable should accept a scalar `Tensor` `t` and a 1-D `Tensor` `r(t)`
        of shape `[num_samples]` and returns a 2-D `Tensor` of shape
        `[num_samples, num_hjm_factors]`. The variable `t`  stands for time and
        `r(t)` is the short rate at time `t`.  The function returns the
        instantaneous volatility `sigma(t) = sigma(t, r(r))`. When `volatility`
        is specified as a real `Tensor`, each factor is assumed to have a
        constant instantaneous volatility  and the  model is effectively a
        Gaussian HJM model. Corresponds to the instantaneous volatility of each
        factor.
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation. This
      input is required.
    notional: An optional `Tensor` of same dtype and compatible shape as
      `strikes`specifying the notional amount for the underlying swaps.
       Default value: None in which case the notional is set to 1.
    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.
      Indicates whether the swaption is a payer (if True) or a receiver (if
      False) swaption. If not supplied, payer swaptions are assumed.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation. This input is ignored during analytic
      valuation.
      Default value: The default value is 1.
    random_type: Enum value of `RandomType`. The type of (quasi)-random number
      generator to use to generate the simulation paths. This input is relevant
      only for Monte-Carlo valuation and ignored during analytic valuation.
      Default value: `None` which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of `[STATELESS, PSEUDO, HALTON_RANDOMIZED,
      PSEUDO_ANTITHETIC, STATELESS_ANTITHETIC]`. For `PSEUDO`,
      `PSEUDO_ANTITHETIC` and `HALTON_RANDOMIZED` the seed should be an Python
      integer. For `STATELESS` and  `STATELESS_ANTITHETIC` must be supplied as
      an integer `Tensor` of shape `[2]`. This input is relevant only for
      Monte-Carlo valuation and ignored during analytic valuation.
      Default value: `None` which means no seed is set.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name `hjm_swaption_price`.

  Returns:
    A `Tensor` of real dtype and shape expiries.shape + [num_hjm_factors]
    containing the computed swaption prices. For swaptions that have reset in
    the past (expiries<0), the function sets the corresponding option prices to
    0.0.
  """
  if time_step is None:
    raise ValueError('`time_step` must be provided for simulation based '
                     'swaption valuation.')

  # TODO(b/160061740): Extend the functionality to support mid-curve swaptions.
  name = name or 'hjm_swaption_price'
  with tf.name_scope(name):
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    dtype = dtype or expiries.dtype
    fixed_leg_payment_times = tf.convert_to_tensor(
        fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
    fixed_leg_daycount_fractions = tf.convert_to_tensor(
        fixed_leg_daycount_fractions,
        dtype=dtype,
        name='fixed_leg_daycount_fractions')
    fixed_leg_coupon = tf.convert_to_tensor(
        fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    notional = tf.expand_dims(
        tf.broadcast_to(notional, expiries.shape), axis=-1)
    if is_payer_swaption is None:
      is_payer_swaption = True
    is_payer_swaption = tf.convert_to_tensor(
        is_payer_swaption, dtype=tf.bool, name='is_payer_swaption')

    output_shape = expiries.shape.as_list() + [1]
    # Add a dimension corresponding to multiple cashflows in a swap
    if expiries.shape.rank == fixed_leg_payment_times.shape.rank - 1:
      expiries = tf.expand_dims(expiries, axis=-1)
    elif expiries.shape.rank < fixed_leg_payment_times.shape.rank - 1:
      raise ValueError('Swaption expiries not specified for all swaptions '
                       'in the batch. Expected rank {} but received {}.'.format(
                           fixed_leg_payment_times.shape.rank - 1,
                           expiries.shape.rank))

    # Expected shape: batch_shape + [m], where m is the number of fixed leg
    # payments per underlying swap. This is the same as
    # fixed_leg_payment_times.shape
    #
    # We need to explicitly use tf.repeat because we need to price
    # batch_shape + [m] bond options with different strikes along the last
    # dimension.
    expiries = tf.repeat(
        expiries, tf.shape(fixed_leg_payment_times)[-1], axis=-1)

    # Monte-Carlo pricing
    model = quasi_gaussian_hjm.QuasiGaussianHJM(
        num_hjm_factors,
        mean_reversion=mean_reversion,
        volatility=volatility,
        initial_discount_rate_fn=reference_rate_fn,
        dtype=dtype)

    def _sample_discount_curve_path_fn(times, curve_times, num_samples):
      p_t_tau, r_t, _ = model.sample_discount_curve_paths(
          times=times,
          curve_times=curve_times,
          num_samples=num_samples,
          random_type=random_type,
          time_step=time_step,
          seed=seed,
          skip=skip)
      p_t_tau = tf.expand_dims(p_t_tau, axis=-1)
      r_t = tf.expand_dims(r_t, axis=-1)
      return p_t_tau, r_t

    payoff_discount_factors, payoff_bond_price = (
        swaption_util.discount_factors_and_bond_prices_from_samples(
            expiries=expiries,
            payment_times=fixed_leg_payment_times,
            sample_discount_curve_paths_fn=_sample_discount_curve_path_fn,
            num_samples=num_samples,
            time_step=time_step,
            dtype=dtype))

    # Add an axis corresponding to `dim`
    fixed_leg_pv = tf.expand_dims(
        fixed_leg_coupon * fixed_leg_daycount_fractions,
        axis=-1) * payoff_bond_price

    # Sum fixed coupon payments within each swap.
    # Here, axis=-2 is the payments axis - i.e. summing over all payments; and
    # the last axis is the `dim` axis, as explained in comment above
    # `fixed_leg_pv` (Note that for HJM the dim of this axis is 1 always).
    fixed_leg_pv = tf.math.reduce_sum(fixed_leg_pv, axis=-2)
    float_leg_pv = 1.0 - payoff_bond_price[..., -1, :]
    payoff_swap = payoff_discount_factors[..., -1, :] * (
        float_leg_pv - fixed_leg_pv)
    payoff_swap = tf.where(is_payer_swaption, payoff_swap, -1.0 * payoff_swap)
    payoff_swaption = tf.math.maximum(payoff_swap, 0.0)
    option_value = tf.reshape(
        tf.math.reduce_mean(payoff_swaption, axis=0), output_shape)

    return notional * option_value
