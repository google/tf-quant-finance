# Copyright 2021 Google LLC
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
"""Variance swap pricing using replicating portfolio approach."""

import tensorflow.compat.v2 as tf
from tf_quant_finance.black_scholes import vanilla_prices
from tf_quant_finance.math import diff_ops


def replicating_weights(ordered_strikes,
                        reference_strikes,
                        expiries,
                        validate_args=False,
                        dtype=None,
                        name=None):
  """Calculates the weights for options to recreate the variance swap payoff.

  This implements the approach in Appendix A of Demeterfi et al for calculating
  the weight of European options required to replicate the payoff of a variance
  swap given traded strikes. In particular this function calculates the weights
  for the put option part of the portfolio (when `ordered_strikes` is descending
  ) or for the call option part of the portfolio (when `ordered_strikes`
  is ascending). See the fair strike docstring for further details on variance
  swaps.

  #### Example

  ```python
  dtype = tf.float64
  ordered_put_strikes = [100, 95, 90, 85]
  reference_strikes = ordered_put_strikes[0]
  expiries = 0.25
  # Contains weights for put options at ordered_put_strikes[:-1]
  put_weights = variance_replicating_weights(
    ordered_put_strikes, reference_strikes, expiries, dtype=dtype)
  # [0.00206927, 0.00443828, 0.00494591]
  ```

  #### References

  [1] Demeterfi, K., Derman, E., Kamal, M. and Zou, J., 1999. More Than You Ever
    Wanted To Know About Volatility Swaps. Goldman Sachs Quantitative Strategies
    Research Notes.

  Args:
    ordered_strikes: A real `Tensor` of liquidly traded strikes of shape
      `batch_shape + [num_strikes]`. The last entry will not receive a weight in
      the portfolio. The values must be sorted ascending if the strikes are for
      calls, or descending if the strikes are for puts. The final value in
      `ordered_strikes` will not itself receive a weight.
    reference_strikes: A `Tensor` of the same dtype as `ordered_strikes` and of
      shape compatible with `batch_shape`. An arbitrarily chosen strike
      representing an at the money strike price.
    expiries: A `Tensor` of the same dtype as `ordered_strikes` and of shape
      compatible with `batch_shape`. Represents the time to maturity of the
      options.
    validate_args: Python `bool`. When `True`, input `Tensor`s are checked for
      validity. The checks verify that `ordered_strikes` is indeed ordered. When
      `False` invalid inputs may silently render incorrect outputs, yet runtime
      performance may be improved.
      Default value: False.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None leading to use of `ordered_strikes.dtype`.
    name: Python str. The name to give to the ops created by this function.
      Default value: `None` which maps to 'variance_replicating_weights'.

  Returns:
    A `Tensor` of shape `batch_shape + [num_strikes - 1]` representing the
    weight which should be given to each strike in the replicating portfolio,
    save for the final strike which is not represented.
  """
  with tf.name_scope(name or 'replicating_weights'):
    # Input conversion.
    ordered_strikes = tf.convert_to_tensor(
        ordered_strikes, dtype=dtype, name='ordered_strikes')
    dtype = dtype or ordered_strikes.dtype
    reference_strikes = tf.expand_dims(
        tf.convert_to_tensor(
            reference_strikes, dtype=dtype, name='reference_strikes'), -1)
    expiries = tf.expand_dims(
        tf.convert_to_tensor(expiries, dtype=dtype, name='expiries'), -1)
    # Descending is required for the formulae regardless of use as control dep.
    strike_diff = diff_ops.diff(ordered_strikes, order=1, exclusive=True)
    strikes_descending = tf.math.reduce_all(strike_diff < 0)
    control_dependencies = []
    if validate_args:
      strikes_ascending = tf.math.reduce_all(strike_diff > 0)
      control_dependencies.append(
          tf.compat.v1.debugging.Assert(
              tf.math.logical_or(strikes_descending, strikes_ascending),
              [strike_diff]))
    with tf.control_dependencies(control_dependencies):
      # Weights calculation
      term_lin = (ordered_strikes - reference_strikes) / reference_strikes
      term_log = tf.math.log(ordered_strikes) - tf.math.log(reference_strikes)
      payoff = (2.0 / expiries) * (term_lin - term_log)
      payoff_diff = diff_ops.diff(payoff, order=1, exclusive=True)
      r_vals = tf.math.divide_no_nan(payoff_diff, strike_diff)
      zero = tf.zeros(r_vals.shape[:-1] + [1], dtype=r_vals.dtype)
      r_vals_diff = diff_ops.diff(
          tf.concat([zero, r_vals], axis=-1), order=1, exclusive=True)
      # If the strikes were for puts we need to flip the sign before returning.
      return tf.where(strikes_descending, -r_vals_diff, r_vals_diff)


def fair_strike(put_strikes,
                put_volatilities,
                call_strikes,
                call_volatilities,
                expiries,
                discount_rates,
                spots,
                reference_strikes,
                validate_args=False,
                dtype=None,
                name=None):
  """Calculates the fair value strike for a variance swap contract.

  This implements the approach in Appendix A of Demeterfi et al (1999), where a
  variance swap is defined as a forward contract on the square of annualized
  realized volatility (though the approach assumes continuous sampling). The
  variance swap payoff is, then:

  `notional * (realized_volatility^2 - variance_strike)`

  The method calculates the weight of each European option required to
  approximately replicate such a payoff using the discrete range of strike
  prices and implied volatilities of European options traded on the market. The
  fair value `variance_strike` is that which is expected to produce zero payoff.

  #### Example

  ```python
  dtype = tf.float64
  call_strikes = tf.constant([[100, 105, 110, 115], [1000, 1100, 1200, 1300]],
    dtype=dtype)
  call_vols = 0.2 * tf.ones((2, 4), dtype=dtype)
  put_strikes = tf.constant([[100, 95, 90, 85], [1000, 900, 800, 700]],
    dtype=dtype)
  put_vols = 0.2 * tf.ones((2, 4), dtype=dtype)
  reference_strikes = tf.constant([100.0, 1000.0], dtype=dtype)
  expiries = tf.constant([0.25, 0.25], dtype=dtype)
  discount_rates = tf.constant([0.05, 0.05], dtype=dtype)
  variance_swap_price(
    put_strikes,
    put_vols,
    call_strikes,
    put_vols,
    expiries,
    discount_rates,
    reference_strikes,
    reference_strikes,
    dtype=tf.float64)
  # [0.03825004, 0.04659269]
  ```

  #### References

  [1] Demeterfi, K., Derman, E., Kamal, M. and Zou, J., 1999. More Than You Ever
    Wanted To Know About Volatility Swaps. Goldman Sachs Quantitative Strategies
    Research Notes.

  Args:
    put_strikes: A real `Tensor` of shape  `batch_shape + [num_put_strikes]`
      containing the strike values of traded puts. This must be supplied in
      **descending** order, and its elements should be less than or equal to the
      `reference_strike`.
    put_volatilities: A real `Tensor` of shape  `batch_shape +
      [num_put_strikes]` containing the market volatility for each strike in
      `put_strikes. The final value is unused.
    call_strikes: A real `Tensor` of shape  `batch_shape + [num_call_strikes]`
      containing the strike values of traded calls. This must be supplied in
      **ascending** order, and its elements should be greater than or equal to
      the `reference_strike`.
    call_volatilities: A real `Tensor` of shape  `batch_shape +
      [num_call_strikes]` containing the market volatility for each strike in
      `call_strikes`. The final value is unused.
    expiries: A real `Tensor` of shape compatible with `batch_shape` containing
      the time to expiries of the contracts.
    discount_rates: A real `Tensor` of shape compatible with `batch_shape`
      containing the discount rate to be applied.
    spots: A real `Tensor` of shape compatible with `batch_shape` containing the
      current spot price of the asset.
    reference_strikes: A real `Tensor` of shape compatible with `batch_shape`
      containing an arbitrary value demarcating the atm boundary between liquid
      calls and puts. Typically either the spot price or the (common) first
      value of `put_strikes` or `call_strikes`.
    validate_args: Python `bool`. When `True`, input `Tensor`s are checked for
      validity. The checks verify the the matching length of strikes and
      volatilties. When `False` invalid inputs may silently render incorrect
      outputs, yet runtime performance will be improved.
      Default value: False.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None, leading to the default value inferred by Tensorflow.
    name: Python str. The name to give to the ops created by this function.
      Default value: `None` which maps to 'variance_swap_price'.

  Returns:
    A `Tensor` of shape `batch_shape` containing the fair value of variance for
    each item in the batch. Note this is on the decimal rather than square
    percentage scale.
  """
  with tf.name_scope(name or 'variance_swap_price'):
    put_strikes = tf.convert_to_tensor(
        put_strikes, dtype=dtype, name='put_strikes')
    dtype = dtype or put_strikes.dtype
    put_volatilities = tf.convert_to_tensor(
        put_volatilities, dtype=dtype, name='put_volatilities')
    call_strikes = tf.convert_to_tensor(
        call_strikes, dtype=dtype, name='call_strikes')
    call_volatilities = tf.convert_to_tensor(
        call_volatilities, dtype=dtype, name='call_volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    discount_rates = tf.expand_dims(
        tf.convert_to_tensor(
            discount_rates, dtype=dtype, name='discount_rates'), -1)
    spots = tf.expand_dims(
        tf.convert_to_tensor(spots, dtype=dtype, name='spots'), -1)
    reference_strikes = tf.convert_to_tensor(
        reference_strikes, dtype=dtype, name='reference_strikes')

    # Check the inputs are consistent in length.
    control_dependencies = []
    if validate_args:
      control_dependencies.append(
          tf.math.reduce_all(
              tf.shape(put_strikes)[-1] == tf.shape(put_volatilities)[-1]))
      control_dependencies.append(
          tf.math.reduce_all(
              tf.shape(call_strikes)[-1] == tf.shape(call_volatilities)[-1]))

    with tf.control_dependencies(control_dependencies):
      # Shape is `batch_shape + [num_put_strikes - 1]`
      put_weights = replicating_weights(
          put_strikes, reference_strikes, expiries, validate_args=validate_args)
      # Shape is `batch_shape + [num_call_strikes - 1]`
      call_weights = replicating_weights(
          call_strikes,
          reference_strikes,
          expiries,
          validate_args=validate_args)

      expiries = tf.expand_dims(expiries, -1)
      reference_strikes = tf.expand_dims(reference_strikes, -1)

      put_prices = vanilla_prices.option_price(
          volatilities=put_volatilities[..., :-1],
          strikes=put_strikes[..., :-1],
          expiries=expiries,
          spots=spots,
          discount_rates=discount_rates,
          is_call_options=False,
      )
      call_prices = vanilla_prices.option_price(
          volatilities=call_volatilities[..., :-1],
          strikes=call_strikes[..., :-1],
          expiries=expiries,
          spots=spots,
          discount_rates=discount_rates,
          is_call_options=True,
      )

      effective_rate = expiries * discount_rates
      discount_factor = tf.math.exp(effective_rate)

      s_ratio = spots / reference_strikes
      centrality_term = (2.0 / expiries) * (
          effective_rate - discount_factor * s_ratio + 1 +
          tf.math.log(s_ratio))

      options_value = discount_factor * (
          tf.math.reduce_sum(put_weights * put_prices, axis=-1, keepdims=True) +
          tf.math.reduce_sum(
              call_weights * call_prices, axis=-1, keepdims=True))

      # Return values, undoing the dimension expansion introduced earlier.
      return tf.squeeze(options_value + centrality_term, axis=-1)
