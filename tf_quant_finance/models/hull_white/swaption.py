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
"""Pricing of Interest rate Swaptions using the Hull-White model."""

import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.math import root_search
from tf_quant_finance.models.hull_white import vector_hull_white
from tf_quant_finance.models.hull_white import zero_coupon_bond_option as zcb

__all__ = ['swaption_price']


def _jamshidian_decomposition(hw_model,
                              expiries,
                              maturities,
                              coefficients,
                              dtype,
                              name=None):
  """Jamshidian decomposition for European swaption valuation.

  Jamshidian decomposition is a widely used technique for the valuation of
  European swaptions (and options on coupon bearing bonds) when the underlying
  models for the term structure are short rate models (such as Hull-White
  model). The method transforms the swaption valuation to the valuation of a
  portfolio of call (put) options on zero-coupon bonds.

  Consider the following swaption payoff(assuming unit notional) at the
  exipration time (under a single curve valuation):

  ```None
  payoff = max(1 - P(T0, TN, r(T0)) - sum_1^N tau_i * X_i * P(T0, Ti, r(T0)), 0)
         = max(1 - sum_0^N alpha_i * P(T0, Ti, r(T0)), 0)
  ```

  where `T0` denotes the swaption expiry, P(T0, Ti, r(T0)) denotes the price
  of the zero coupon bond at `T0` with maturity `Ti` and `r(T0)` is the short
  rate at time `T0`. If `r*` (or breakeven short rate) is the solution of the
  following equation:

  ```None
  1 - sum_0^N alpha_i * P(T0, Ti, r*) = 0            (1)
  ```

  Then the swaption payoff can be expressed as the following (Ref. [1]):

  ```None
  payoff = sum_1^N alpha_i max(P(T0, Ti, r*) - P(T0, Ti), 0)
  ```
  where in the above formulation the swaption payoff is the same as that of
  a portfolio of bond options with strikes `P(T0, Ti, r*)`.

  The function accepts relevant inputs for the above computation and returns
  the strikes of the bond options computed using the Jamshidian decomposition.

  #### References:
    [1]: Leif B. G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling.
    Volume II: Term Structure Models. Chapter 10.


  Args:
    hw_model: An instance of `VectorHullWhiteModel`. The model used for the
      valuation.
    expiries: A real `Tensor` of any shape and dtype. The the time to
      expiration of the swaptions.
    maturities: A real `Tensor` of same shape and dtype as `expiries`. The
      payment times for fixed payments of the underlying swaptions.
    coefficients: A real `Tensor` of shape `expiries.shape + [n]` where `n`
      denotes the number of payments in the fixed leg of the underlying swaps.
    dtype: The default dtype to use when converting values to `Tensor`s.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `jamshidian_decomposition`.

  Returns:
    A real `Tensor` of shape expiries.shape + [dim] containing the forward
    bond prices computed at the breakeven short rate using the Jamshidian
    decomposition. `dim` stands for the dimensionality of the Hull-White
    process.
  """

  name = name or 'jamshidian_decomposition'
  with tf.name_scope(name):
    dim = hw_model.dim()
    coefficients = tf.expand_dims(coefficients, axis=-1)

    def _zero_fun(x):
      # Get P(t0, t, r(t0)).
      p_t0_t = hw_model.discount_bond_price(x, expiries, maturities)
      # return_value.shape = batch_shape + [1] + [dim]
      return_value = tf.reduce_sum(
          coefficients * p_t0_t, axis=-2, keepdims=True) + [1.0]
      return return_value

    swap_shape = expiries.shape.as_list()[:-1] + [1] + [dim]
    lower_bound = -1 * tf.ones(swap_shape, dtype=dtype)
    upper_bound = 1 * tf.ones(swap_shape, dtype=dtype)
    # Solve Eq.(1)
    brent_results = root_search.brentq(_zero_fun, lower_bound, upper_bound)
    breakeven_short_rate = brent_results.estimated_root
    return hw_model.discount_bond_price(breakeven_short_rate, expiries,
                                        maturities)


def _prepare_swaption_indices(tensor_shape):
  """Indices for `gather_nd` for analytic valuation.

  For a `Tensor` x of shape `tensor_shape` = [n] + batch_shape + [n], this
  function returns indices for tf.gather_nd to get `x[i,...,i]`

  Args:
    tensor_shape: A list of length `k` representing shape of the `Tensor`.

  Returns:
    A `Tensor` of shape (num_elements, k) where num_elements= n * batch_size
    of dtype tf.int64.
  """

  tensor_shape = np.array(tensor_shape, dtype=np.int64)
  batch_shape = tensor_shape[1:-1]
  batch_size = np.prod(batch_shape)
  index_list = []
  for i in range(len(tensor_shape)):
    index = np.arange(0, tensor_shape[i], dtype=np.int64)
    if i == 0 or i == len(tensor_shape) - 1:
      index = tf.tile(index, [batch_size])
    else:
      index = tf.tile(
          tf.repeat(index, np.prod(tensor_shape[i+1:])),
          [np.prod(tensor_shape[1:i])])
    index_list.append(index)

  return tf.stack(index_list, axis=-1)


def _prepare_indices(idx0, idx1, idx2, idx3):
  """Prepares indices to get relevant slice from discount curve simulations."""
  # For a 4-D `Tensor` x, creates indices for tf.gather_nd to retrieve
  # x[i, j, j, k].
  len0 = idx0.shape.as_list()[0]
  len1 = idx1.shape.as_list()[0]
  len3 = idx3.shape.as_list()[0]
  idx0 = tf.repeat(idx0, len1 * len3)
  idx1 = tf.tile(tf.repeat(idx1, len3), [len0])
  idx2 = tf.tile(tf.repeat(idx2, len3), [len0])
  idx3 = tf.tile(idx3, [len0 * len1])

  return tf.stack([idx0, idx1, idx2, idx3], axis=-1)


def _cumprod_using_matvec(input_tensor):
  """Computes cumprod using matrix algebra."""
  dtype = input_tensor.dtype
  axis_length = input_tensor.shape.as_list()[-1]
  ones = tf.ones([axis_length, axis_length], dtype=dtype)
  lower_triangular = tf.linalg.band_part(ones, -1, 0)
  cumsum = tf.linalg.matvec(lower_triangular, tf.math.log(input_tensor))
  return tf.math.exp(cumsum)


def _analytic_valuation(expiries, floating_leg_start_times,
                        floating_leg_end_times, fixed_leg_payment_times,
                        fixed_leg_daycount_fractions, fixed_leg_coupon,
                        reference_rate_fn, dim, mean_reversion, volatility,
                        notional, is_payer_swaption, output_shape,
                        dtype, name):
  """Helper function for analytic valuation."""
  # The below inputs are needed for midcurve swaptions
  del floating_leg_start_times, floating_leg_end_times
  with tf.name_scope(name):
    is_call_options = tf.where(is_payer_swaption,
                               tf.convert_to_tensor(False, dtype=tf.bool),
                               tf.convert_to_tensor(True, dtype=tf.bool))

    model = vector_hull_white.VectorHullWhiteModel(
        dim,
        mean_reversion,
        volatility,
        initial_discount_rate_fn=reference_rate_fn,
        dtype=dtype)
    coefficients = fixed_leg_daycount_fractions * fixed_leg_coupon
    jamshidian_coefficients = tf.concat([
        -coefficients[..., :-1],
        tf.expand_dims(-1.0 - coefficients[..., -1], axis=-1)], axis=-1)

    breakeven_bond_option_strikes = _jamshidian_decomposition(
        model, expiries,
        fixed_leg_payment_times, jamshidian_coefficients, dtype,
        name=name + '_jamshidian_decomposition')

    bond_strike_rank = breakeven_bond_option_strikes.shape.rank
    perm = [bond_strike_rank-1] + [x for x in range(0, bond_strike_rank - 1)]
    breakeven_bond_option_strikes = tf.transpose(
        breakeven_bond_option_strikes, perm=perm)
    bond_option_prices = zcb.bond_option_price(
        strikes=breakeven_bond_option_strikes,
        expiries=expiries,
        maturities=fixed_leg_payment_times,
        discount_rate_fn=reference_rate_fn,
        dim=dim,
        mean_reversion=mean_reversion,
        volatility=volatility,
        is_call_options=is_call_options,
        use_analytic_pricing=True,
        dtype=dtype,
        name=name + '_bond_option')
    bond_option_prices = notional * bond_option_prices

    # Now compute P(T0, TN) + sum_i (c_i * tau_i * P(T0, Ti))
    # bond_option_prices.shape = [dim] + batch_shape + [m] + [dim], where `m`
    # denotes the number of fixed payments for the underlying swaps.
    swaption_values = (
        tf.reduce_sum(
            bond_option_prices * tf.expand_dims(coefficients, axis=-1),
            axis=-2) + bond_option_prices[..., -1, :])
    swaption_shape = swaption_values.shape
    gather_index = _prepare_swaption_indices(swaption_shape.as_list())
    swaption_values = tf.gather_nd(swaption_values, gather_index)
    return tf.reshape(swaption_values, output_shape)


def swaption_price(*,
                   expiries,
                   floating_leg_start_times,
                   floating_leg_end_times,
                   fixed_leg_payment_times,
                   floating_leg_daycount_fractions,
                   fixed_leg_daycount_fractions,
                   fixed_leg_coupon,
                   reference_rate_fn,
                   dim,
                   mean_reversion,
                   volatility,
                   notional=None,
                   is_payer_swaption=None,
                   use_analytic_pricing=True,
                   num_samples=1,
                   random_type=None,
                   seed=None,
                   skip=0,
                   time_step=None,
                   dtype=None,
                   name=None):
  """Calculates the price of European Swaptions using the Hull-White model.

  A European Swaption is a contract that gives the holder an option to enter a
  swap contract at a future date at a prespecified fixed rate. A swaption that
  grants the holder to pay fixed rate and receive floating rate is called a
  payer swaption while the swaption that grants the holder to receive fixed and
  pay floating payments is called the receiver swaption. Typically the start
  date (or the inception date) of the swap concides with the expiry of the
  swaption. Mid-curve swaptions are currently not supported (b/160061740).

  Analytic pricing of swaptions is performed using the Jamshidian decomposition
  [1].

  #### References:
    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.
    Second Edition. 2007.

  #### Example
  The example shows how value a batch of 1y x 1y and 1y x 2y swaptions using the
  Hull-White model.

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  expiries = [1.0, 1.0]
  float_leg_start_times = [[1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0],
                            [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]]
  float_leg_end_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                          [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
  fixed_leg_payment_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                          [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
  float_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
  fixed_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
  fixed_leg_coupon = [[0.011, 0.011, 0.011, 0.011, 0.0, 0.0, 0.0, 0.0],
                      [0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011]]
  zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
  price = tff.models.hull_white.swaption_price(
      expiries=expiries,
      floating_leg_start_times=float_leg_start_times,
      floating_leg_end_times=float_leg_end_times,
      fixed_leg_payment_times=fixed_leg_payment_times,
      floating_leg_daycount_fractions=float_leg_daycount_fractions,
      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
      fixed_leg_coupon=fixed_leg_coupon,
      reference_rate_fn=zero_rate_fn,
      notional=100.,
      dim=1,
      mean_reversion=[0.03],
      volatility=[0.02],
      dtype=dtype)
  # Expected value: [[0.7163243383624043], [1.4031415262337608]] # shape = (2,1)
  ````

  Args:
    expiries: A real `Tensor` of any shape and dtype. The time to
      expiration of the swaptions. The shape of this input determines the number
      (and shape) of swaptions to be priced and the shape of the output.
    floating_leg_start_times: A real `Tensor` of the same dtype as `expiries`.
      The times when accrual begins for each payment in the floating leg. The
      shape of this input should be `expiries.shape + [m]` where `m` denotes
      the number of floating payments in each leg.
    floating_leg_end_times: A real `Tensor` of the same dtype as `expiries`.
      The times when accrual ends for each payment in the floating leg. The
      shape of this input should be `expiries.shape + [m]` where `m` denotes
      the number of floating payments in each leg.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as `expiries`.
      The payment times for each payment in the fixed leg. The shape of this
      input should be `expiries.shape + [n]` where `n` denotes the number of
      fixed payments in each leg.
    floating_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `floating_leg_start_times`. The daycount fractions
      for each payment in the floating leg.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions
      for each payment in the fixed leg.
    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape
      as `fixed_leg_payment_times`. The fixed rate for each payment in the
      fixed leg.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape + [dim]`. Returns
      the continuously compounded zero rate at the present time for the input
      expiry time.
    dim: A Python scalar which corresponds to the number of Hull-White Models
      to be used for pricing.
    mean_reversion: A real positive `Tensor` of shape `[dim]` or a Python
      callable. The callable can be one of the following:
      (a) A left-continuous piecewise constant object (e.g.,
      `tff.math.piecewise.PiecewiseConstantFunc`) that has a property
      `is_piecewise_constant` set to `True`. In this case the object should
      have a method `jump_locations(self)` that returns a `Tensor` of shape
      `[dim, num_jumps]` or `[num_jumps]`. In the first case,
      `mean_reversion(t)` should return a `Tensor` of shape `[dim] + t.shape`,
      and in the second, `t.shape + [dim]`, where `t` is a rank 1 `Tensor` of
      the same `dtype` as the output. See example in the class docstring.
      (b) A callable that accepts scalars (stands for time `t`) and returns a
      `Tensor` of shape `[dim]`.
      Corresponds to the mean reversion rate.
    volatility: A real positive `Tensor` of the same `dtype` as
      `mean_reversion` or a callable with the same specs as above.
      Corresponds to the lond run price variance.
    notional: An optional `Tensor` of same dtype and compatible shape as
      `strikes`specifying the notional amount for the underlying swap.
       Default value: None in which case the notional is set to 1.
    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.
      Indicates whether the swaption is a payer (if True) or a receiver
      (if False) swaption. If not supplied, payer swaptions are assumed.
    use_analytic_pricing: A Python boolean specifying if analytic valuation
      should be performed. Analytic valuation is only supported for constant
      `mean_reversion` and piecewise constant `volatility`. If the input is
      `False`, then valuation using Monte-Carlo simulations is performed.
      Default value: The default value is `True`.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation. This input is ignored during analytic
      valuation.
      Default value: The default value is 1.
    random_type: Enum value of `RandomType`. The type of (quasi)-random
      number generator to use to generate the simulation paths. This input is
      relevant only for Monte-Carlo valuation and ignored during analytic
      valuation.
      Default value: `None` which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of
      `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
        STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
      `HALTON_RANDOMIZED` the seed should be an Python integer. For
      `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
      `Tensor` of shape `[2]`. This input is relevant only for Monte-Carlo
      valuation and ignored during analytic valuation.
      Default value: `None` which means no seed is set.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation. This
      input is ignored during analytic valuation.
      Default value: `None`.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
      TensorFlow are used.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `hw_swaption_price`.

  Returns:
    A `Tensor` of real dtype and shape  expiries.shape + [dim] containing the
    computed swaption prices. For swaptions that have. reset in the past
    (expiries<0), the function sets the corresponding option prices to 0.0.
  """
  # TODO(b/160061740): Extend the functionality to support mid-curve swaptions.
  name = name or 'hw_swaption_price'
  del floating_leg_daycount_fractions
  with tf.name_scope(name):
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    dtype = dtype or expiries.dtype
    float_leg_start_times = tf.convert_to_tensor(
        floating_leg_start_times, dtype=dtype, name='float_leg_start_times')
    float_leg_end_times = tf.convert_to_tensor(
        floating_leg_end_times, dtype=dtype, name='float_leg_end_times')
    fixed_leg_payment_times = tf.convert_to_tensor(
        fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
    fixed_leg_daycount_fractions = tf.convert_to_tensor(
        fixed_leg_daycount_fractions, dtype=dtype,
        name='fixed_leg_daycount_fractions')
    fixed_leg_coupon = tf.convert_to_tensor(
        fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    if is_payer_swaption is None:
      is_payer_swaption = True
    is_payer_swaption = tf.convert_to_tensor(
        is_payer_swaption, dtype=tf.bool, name='is_payer_swaption')

    output_shape = expiries.shape.as_list() + [dim]
    # Add a dimension corresponding to multiple cashflows in a swap
    if expiries.shape.rank == fixed_leg_payment_times.shape.rank - 1:
      expiries = tf.expand_dims(expiries, axis=-1)
    elif expiries.shape.rank < fixed_leg_payment_times.shape.rank - 1:
      raise ValueError('Swaption expiries not specified for all swaptions '
                       'in the batch. Expected rank {} but received {}.'.format(
                           fixed_leg_payment_times.shape.rank - 1,
                           expiries.shape.rank))

    # Expected shape: batch_shape + [m], same as fixed_leg_payment_times.shape
    # We need to explicitly use tf.repeat because we need to price
    # batch_shape + [m] bond options with different strikes along the last
    # dimension.
    expiries = tf.repeat(
        expiries, fixed_leg_payment_times.shape.as_list()[-1], axis=-1)

    if use_analytic_pricing:
      return _analytic_valuation(expiries, float_leg_start_times,
                                 float_leg_end_times, fixed_leg_payment_times,
                                 fixed_leg_daycount_fractions,
                                 fixed_leg_coupon, reference_rate_fn,
                                 dim, mean_reversion, volatility, notional,
                                 is_payer_swaption, output_shape, dtype,
                                 name + '_analytic_valyation')

    # Monte-Carlo pricing
    model = vector_hull_white.VectorHullWhiteModel(
        dim,
        mean_reversion,
        volatility,
        initial_discount_rate_fn=reference_rate_fn,
        dtype=dtype)

    if time_step is None:
      raise ValueError('`time_step` must be provided for simulation '
                       'based bond option valuation.')

    sim_times, _ = tf.unique(tf.reshape(expiries, shape=[-1]))
    longest_expiry = tf.reduce_max(sim_times)
    sim_times, _ = tf.unique(tf.concat([sim_times, tf.range(
        time_step, longest_expiry, time_step)], axis=0))
    sim_times = tf.sort(sim_times, name='sort_sim_times')

    maturities = fixed_leg_payment_times
    swaptionlet_shape = maturities.shape
    tau = maturities - expiries

    curve_times_builder, _ = tf.unique(tf.reshape(tau, shape=[-1]))
    curve_times = tf.sort(curve_times_builder, name='sort_curve_times')

    p_t_tau, r_t = model.sample_discount_curve_paths(
        times=sim_times,
        curve_times=curve_times,
        num_samples=num_samples,
        random_type=random_type,
        seed=seed,
        skip=skip)

    dt = tf.concat(
        [tf.convert_to_tensor([0.0], dtype=dtype),
         sim_times[1:] - sim_times[:-1]], axis=0)
    dt = tf.expand_dims(tf.expand_dims(dt, axis=-1), axis=0)
    discount_factors_builder = tf.math.exp(-r_t * dt)
    # Transpose before (and after) because we want the cumprod along axis=1
    # and `matvec` operates on the last axis.
    discount_factors_builder = tf.transpose(
        _cumprod_using_matvec(
            tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])

    # make discount factors the same shape as `p_t_tau`. This involves adding
    # an extra dimenstion (corresponding to `curve_times`).
    discount_factors_builder = tf.expand_dims(
        discount_factors_builder,
        axis=1)
    # tf.repeat is needed because we will use gather_nd later on this tensor.
    discount_factors_simulated = tf.repeat(
        discount_factors_builder, p_t_tau.shape.as_list()[1], axis=1)

    # `sim_times` and `curve_times` are sorted for simulation. We need to
    # select the indices corresponding to our input.
    sim_time_index = tf.searchsorted(sim_times, tf.reshape(expiries, [-1]))
    curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))

    gather_index = _prepare_indices(
        tf.range(0, num_samples), curve_time_index, sim_time_index,
        tf.range(0, dim))

    # The shape after `gather_nd` will be `(num_samples*num_swaptionlets*dim,)`
    payoff_discount_factors_builder = tf.gather_nd(
        discount_factors_simulated, gather_index)
    # Reshape to `[num_samples] + swaptionlet.shape + [dim]`
    payoff_discount_factors = tf.reshape(
        payoff_discount_factors_builder,
        [num_samples] + swaptionlet_shape + [dim])
    payoff_bond_price_builder = tf.gather_nd(p_t_tau, gather_index)
    payoff_bond_price = tf.reshape(
        payoff_bond_price_builder, [num_samples] + swaptionlet_shape + [dim])

    # Add an axis corresponding to `dim`
    fixed_leg_pv = tf.expand_dims(
        fixed_leg_coupon * fixed_leg_daycount_fractions,
        axis=-1) * payoff_bond_price
    # Sum fixed coupon payments within each swap
    fixed_leg_pv = tf.math.reduce_sum(fixed_leg_pv, axis=-2)
    float_leg_pv = 1.0 - payoff_bond_price[..., -1, :]
    payoff_swap = payoff_discount_factors[..., -1, :] * (
        float_leg_pv - fixed_leg_pv)
    payoff_swap = tf.where(is_payer_swaption, payoff_swap, -1.0 * payoff_swap)
    payoff_swaption = tf.math.maximum(payoff_swap, 0.0)
    option_value = notional * tf.math.reduce_mean(payoff_swaption, axis=0)

    return tf.reshape(option_value, output_shape)
