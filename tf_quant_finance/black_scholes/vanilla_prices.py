# Lint as: python3
# Copyright 2019 Google LLC
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
"""Black Scholes prices of a batch of European options."""

import numpy as np
import tensorflow.compat.v2 as tf


def option_price(*,
                 volatilities,
                 strikes,
                 expiries,
                 spots=None,
                 forwards=None,
                 discount_rates=None,
                 continuous_dividends=None,
                 cost_of_carries=None,
                 discount_factors=None,
                 is_call_options=None,
                 dtype=None,
                 name=None):
  """Computes the Black Scholes price for a batch of call or put options.

  #### Example

  ```python
    # Price a batch of 5 vanilla call options.
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Strikes will automatically be broadcasted to shape [5].
    strikes = np.array([3.0])
    # Expiries will be broadcast to shape [5], i.e. each option has strike=3
    # and expiry = 1.
    expiries = 1.0
    computed_prices = tff.black_scholes.option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards)
  # Expected print output of computed prices:
  # [ 0.          2.          2.04806848  1.00020297  2.07303131]
  ```

  #### References:
  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
  [2] Wikipedia contributors. Black-Scholes model. Available at:
    https://en.wikipedia.org/w/index.php?title=Black%E2%80%93Scholes_model

  Args:
    volatilities: Real `Tensor` of any shape and dtype. The volatilities to
      expiry of the options to price.
    strikes: A real `Tensor` of the same dtype and compatible shape as
      `volatilities`. The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and compatible shape as
      `volatilities`. The expiry of each option. The units should be such that
      `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `volatilities`. The current spot price of the underlying. Either this
      argument or the `forwards` (but not both) must be supplied.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `volatilities`. The forwards to maturity. Either this argument or the
      `spots` must be supplied but both must not be supplied.
    discount_rates: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      If not `None`, discount factors are calculated as e^(-rT),
      where r are the discount rates, or risk free rates. At most one of
      discount_rates and discount_factors can be supplied.
      Default value: `None`, equivalent to r = 0 and discount factors = 1 when
      discount_factors also not given.
    continuous_dividends: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      If not `None`, `cost_of_carries` is calculated as r - q,
      where r are the `discount_rates` and q is `continuous_dividends`. Either
      this or `cost_of_carries` can be given.
      Default value: `None`, equivalent to q = 0.
    cost_of_carries: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      Cost of storing a physical commodity, the cost of interest paid when
      long, or the opportunity cost, or the cost of paying dividends when short.
      If not `None`, and `spots` is supplied, used to calculate forwards from
      `spots`: F = e^(bT) * S, where F is the forwards price, b is the cost of
      carries, T is expiries and S is the spot price. If `None`, value assumed
      to be equal to the `discount_rate` - `continuous_dividends`
      Default value: `None`, equivalent to b = r.
    discount_factors: An optional real `Tensor` of same dtype as the
      `volatilities`. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with discount_rate and cost_of_carry.
      If neither is given, no discounting is applied (i.e. the undiscounted
      option price is returned). If `spots` is supplied and `discount_factors`
      is not `None` then this is also used to compute the forwards to expiry.
      At most one of discount_rates and discount_factors can be supplied.
      Default value: `None`, which maps to -log(discount_factors) / expiries
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: `None` which maps to the default dtype inferred by
        TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: `None` which is mapped to the default name `option_price`.

  Returns:
    option_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
    ValueError: If both `discount_rates` and `discount_factors` is supplied.
    ValueError: If both `continuous_dividends` and `cost_of_carries` is
      supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  if (discount_rates is not None) and (discount_factors is not None):
    raise ValueError('At most one of discount_rates and discount_factors may '
                     'be supplied')
  if (continuous_dividends is not None) and (cost_of_carries is not None):
    raise ValueError('At most one of continuous_dividends and cost_of_carries '
                     'may be supplied')

  with tf.name_scope(name or 'option_price'):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')

    if discount_rates is not None:
      discount_rates = tf.convert_to_tensor(
          discount_rates, dtype=dtype, name='discount_rates')
    elif discount_factors is not None:
      discount_rates = -tf.math.log(discount_factors) / expiries
    else:
      discount_rates = tf.convert_to_tensor(
          0.0, dtype=dtype, name='discount_rates')

    if continuous_dividends is None:
      continuous_dividends = tf.convert_to_tensor(
          0.0, dtype=dtype, name='continuous_dividends')

    if cost_of_carries is not None:
      cost_of_carries = tf.convert_to_tensor(
          cost_of_carries, dtype=dtype, name='cost_of_carries')
    else:
      cost_of_carries = discount_rates - continuous_dividends

    if discount_factors is not None:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')
    else:
      discount_factors = tf.exp(-discount_rates * expiries)

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      forwards = spots * tf.exp(cost_of_carries * expiries)

    sqrt_var = volatilities * tf.math.sqrt(expiries)
    d1 = (tf.math.log(forwards / strikes) + sqrt_var * sqrt_var / 2) / sqrt_var
    d2 = d1 - sqrt_var
    undiscounted_calls = forwards * _ncdf(d1) - strikes * _ncdf(d2)
    if is_call_options is None:
      return discount_factors * undiscounted_calls
    undiscounted_forward = forwards - strikes
    undiscounted_puts = undiscounted_calls - undiscounted_forward
    predicate = tf.broadcast_to(is_call_options, tf.shape(undiscounted_calls))
    return discount_factors * tf.where(predicate, undiscounted_calls,
                                       undiscounted_puts)


def price_barrier_option(
        rate,
        asset_yield,
        asset_price,
        strike_price,
        barrier_price,
        rebate,
        asset_volatility,
        time_to_maturity,
        otype):
  """

  Function determines the approximate price for the barrier option. The
  approximation functions for each integrals are split into two matrix.
  The first matrix contains the algebraic terms and the second matrix
  contains the probability distribution terms. Masks are used to filter
  appropriate terms for calculating the integral. Then a dot product
  of each row calculates the approximate price of the barrier option.

  #### Examples

  ```python
  rate: [.08, .08]
  asset_yield: [.04, .04]
  asset_price: [100., 100.]
  strike_price: [90., 90.]
  barrier_price: [95. 95.]
  rebate: [3. 3.]
  asset_volatility: [.25, .25]
  time_to_maturity: [.5, .5]
  otype: [5, 1]

  price = price_barrier_option(
    rate, asset_yield, asset_price, strike_price,
    barrier_price, rebate, asset_volatility,
    time_to_maturity, otype)

  # Expected output
  #  `Tensor` with values [9.024, 7.7627]
  ````

  #### References

  # Technical Report
  [1]: Lee Clewlow, Javier Llanos, Chris Strickland, Caracas Venezuela
  Pricing Exotic Options in a Black-Scholes World, 1994
  https://warwick.ac.uk/fac/soc/wbs/subjects/finance/research/wpaperseries/1994/94-54.pdf

  # Textbook
  [2]: Espen Gaarder Haug, The complete guide to option pricing formulas,
  2nd Edition, 1997

  Args:
    rate: A real scalar or vector `Tensor` where each element represents rate
      for each option.
    asset_yield: A real scalar or vector `Tensor` that is the yield on asset.
    asset_price: A real scalar or vector `Tensor` that is the asset price for
      the underlying security.
    strike_price: A real scalar or vector `Tensor` that is the strike price
      for the option.
    barrier_price: A real scalar or vector `Tensor` that is the barrier price
      for the option to take effect.
    rebate: A real scalar or vector `Tensor` that is a rebate contingent upon
      reaching the barrier price.
    asset_volatility: A real scalar or vector `Tensor` that measure the
      volatility of the asset price.
    time_to_maturity: A real scalar or vector `Tensor` with time to maturity
      of option.
    otype: An real scalar or vector of `Int` that determines barrier option
      to approximate
      [
      0 -> down and in call,
      1 -> down and in put,
      2 -> up and in call,
      3 -> up and in put,
      4 -> down and out call,
      5 -> down and out put,
      6 -> up and out call,
      8 -> up and out put,
      ]
  Returns:
    A `Tensor` of same shape as input that is the approximate price of the
    barrier option.

  """
  down_and_in_call = 1
  down_and_in_put = 2
  up_and_in_call = 3
  up_and_in_put = 4
  down_and_out_call = 5
  down_and_out_put = 6
  up_and_out_call = 7
  up_and_out_put = 8
  with tf.name_scope("Price_Barrier_Option"):
    # Convert all to tensor and enforce float dtype where required
    rate = tf.convert_to_tensor(
        rate, dtype=tf.float32, name="rate")
    asset_yield = tf.convert_to_tensor(
        asset_yield, dtype=tf.float32, name="asset_yield")
    asset_price = tf.convert_to_tensor(
        asset_price, dtype=tf.float32, name="asset_yield")
    strike_price = tf.convert_to_tensor(
        strike_price, dtype=tf.float32, name="strike_price")
    barrier_price = tf.convert_to_tensor(
        barrier_price, dtype=tf.float32, name="barrier_price")
    rebate = tf.convert_to_tensor(
        rebate, dtype=tf.float32, name="rebate")
    asset_volatility = tf.convert_to_tensor(
        asset_volatility, dtype=tf.float32, name="asset_volatility")
    time_to_maturity = tf.convert_to_tensor(
        time_to_maturity, dtype=tf.float32, name="time_to_maturity")
    otype = tf.convert_to_tensor(
        otype, dtype=tf.int32, name="otype")

    # masks, Masks are used to sum appropriate integrals for approximation
    down_and_in_call_tensor_lower_strike = tf.convert_to_tensor(
        [1., 1., -1., -1., 0., 0., 1., 1., 1., 1., 0., 0.])
    down_and_in_call_tensor_greater_strike = tf.convert_to_tensor(
        [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.])
    down_and_in_put_tensor_lower_strike = tf.convert_to_tensor(
        [1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.])
    down_and_in_put_tensor_greater_strike = tf.convert_to_tensor(
        [0., 0., 1., 1., -1., -1., 1., 1., 0., 0., 1., 1.])
    up_and_in_call_tensor_lower_strike = tf.convert_to_tensor(
        [0., 0., 1., 1., -1., -1., 1., 1., 1., 1., 0., 0.])
    up_and_in_call_tensor_greater_strike = tf.convert_to_tensor(
        [1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.])
    up_and_in_put_tensor_lower_strike = tf.convert_to_tensor(
        [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.])
    up_and_in_put_tensor_greater_strike = tf.convert_to_tensor(
        [1., 1., -1., -1., 0., 0., 1., 1., 1., 1., 0., 0.])
    down_and_out_call_tensor_lower_strike = tf.convert_to_tensor(
        [0., 0., 1., 1., 0., 0., -1, -1., 0., 0., 1., 1.])
    down_and_out_call_tensor_greater_strike = tf.convert_to_tensor(
        [1., 1., 0., 0., -1., -1., 0., 0., 0., 0., 1., 1.])
    down_and_out_put_tensor_lower_strike = tf.convert_to_tensor(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.])
    down_and_out_put_tensor_greater_strike = tf.convert_to_tensor(
        [1., 1., -1., -1., 1., 1., -1., -1., 0., 0., 1., 1.])
    up_and_out_call_tensor_lower_strike = tf.convert_to_tensor(
        [1., 1., -1., -1., 1., 1., -1., -1., 0., 0., 1., 1.])
    up_and_out_call_tensor_greater_strike = tf.convert_to_tensor(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.])
    up_and_out_put_tensor_lower_strike = tf.convert_to_tensor(
        [1., 1., 0., 0., -1., -1., 0., 0., 0., 0., 1., 1.])
    up_and_out_put_tensor_greater_strike = tf.convert_to_tensor(
        [0., 0., 1., 1., 0., 0., -1., -1., 0., 0., 1., 1.])

    # Constructing Masks
    @tf.function
    def get_out_map(params):
      """
      Function maps params to option pricing masks
      Args:
      params: Tuple of tensors. (barrier price, strike_price, option type)
        params[0] = barrier_price
        params[1] = strike_price
        params[2] = otype
      Returns:
      returns mask used to price specified option
      """
      out_map = down_and_in_call_tensor_greater_strike # Default
      if tf.math.less(params[1], params[0]):
        if tf.equal(params[2], down_and_in_call):
          out_map = down_and_in_call_tensor_lower_strike
        if tf.equal(params[2], down_and_in_put):
          out_map = down_and_in_put_tensor_lower_strike
        if tf.equal(params[2], up_and_in_call):
          out_map = up_and_in_call_tensor_lower_strike
        if tf.equal(params[2], up_and_in_put):
          out_map = up_and_in_put_tensor_lower_strike
        if tf.equal(params[2], down_and_out_call):
          out_map = down_and_out_call_tensor_lower_strike
        if tf.equal(params[2], down_and_out_put):
          out_map = down_and_out_put_tensor_lower_strike
        if tf.equal(params[2], up_and_out_call):
          out_map = up_and_out_call_tensor_lower_strike
        if tf.equal(params[2], up_and_out_put):
          out_map = up_and_out_put_tensor_lower_strike
      else:
        if tf.equal(params[2], down_and_in_call):
          out_map = down_and_in_call_tensor_greater_strike
        if tf.equal(params[2], down_and_in_put):
          out_map = down_and_in_put_tensor_greater_strike
        if tf.equal(params[2], up_and_in_call):
          out_map = up_and_in_call_tensor_greater_strike
        if tf.equal(params[2], up_and_in_put):
          out_map = up_and_in_put_tensor_greater_strike
        if tf.equal(params[2], down_and_out_call):
          out_map = down_and_out_call_tensor_greater_strike
        if tf.equal(params[2], down_and_out_put):
          out_map = down_and_out_put_tensor_greater_strike
        if tf.equal(params[2], up_and_out_call):
          out_map = up_and_out_call_tensor_greater_strike
        if tf.equal(params[2], up_and_out_put):
          out_map = up_and_out_put_tensor_greater_strike
      return out_map
    
    @tf.function
    def get_below_or_above(x):
      if tf.math.less(x[0], x[1]):
        return -1.
      return 1.

    @tf.function
    def get_call_or_put(x):
      if tf.equal(tf.math.mod(x, 2), 0):
        return -1.
      return 1.

    # build call or put, below or above barrier price, and masks
    if otype.shape == ():
      below_or_above = get_below_or_above((asset_price, barrier_price))
      call_or_put = get_call_or_put(otype)
      masks = get_out_map((barrier_price, strike_price, otype))
    else:
      masks = tf.map_fn(
          get_out_map,
          (barrier_price, strike_price, otype), dtype=tf.float32)
      below_or_above = tf.map_fn(
          get_below_or_above,
          (asset_price, barrier_price), dtype=tf.float32)
      call_or_put = tf.map_fn(
          get_call_or_put,
          otype, dtype=tf.float32)

    # Calculate params for integrals
    time_asset_volatility = tf.math.multiply(
        asset_volatility, tf.math.sqrt(time_to_maturity),
        name="time_asset_volatility")
    mu = tf.math.subtract(
        tf.math.subtract(rate, asset_yield),
        tf.math.divide(tf.math.square(asset_volatility), 2), name="mu")
    lamda = tf.math.add(1.,
                        tf.math.divide(mu, tf.math.square(asset_volatility)),
                        name="lambda")
    num = tf.math.log(tf.math.divide(asset_price, strike_price))
    x = tf.math.add(tf.math.divide(num, time_asset_volatility),
                    tf.math.multiply(lamda, time_asset_volatility),
                    name="x")
    num = tf.math.log(tf.math.divide(asset_price, barrier_price))
    x1 = tf.math.add(tf.math.divide(num, time_asset_volatility),
                     tf.math.multiply(lamda, time_asset_volatility),
                     name="x1")
    num = tf.math.log(
        tf.math.divide(tf.math.square(barrier_price),
                       tf.math.multiply(asset_price, strike_price)))
    y = tf.math.add(tf.math.divide(num, time_asset_volatility),
                    tf.math.multiply(lamda, time_asset_volatility),
                    name="y")
    num = tf.math.log(tf.math.divide(barrier_price, asset_price))
    y1 = tf.math.add(tf.math.divide(num, time_asset_volatility),
                     tf.math.multiply(lamda, time_asset_volatility),
                     name="y1")
    num = tf.math.sqrt(tf.math.add(tf.math.square(mu),
                                   2.*tf.math.multiply(
                                       tf.math.square(asset_volatility), rate)))
    b = tf.math.divide(num, tf.math.square(asset_volatility), name="b")
    num = tf.math.log(tf.math.divide(barrier_price, asset_price))
    z = tf.math.add(tf.math.divide(num, time_asset_volatility),
                    tf.math.multiply(b, time_asset_volatility),
                    name="z")
    a = tf.math.divide(mu, tf.math.square(asset_volatility), name="a")


    # Other params used for integrals
    rate_exponent = tf.math.exp(-1.*tf.math.multiply(rate, time_to_maturity),
                                name="rate_exponent")
    asset_yield_exponent = tf.math.exp(-1.*tf.math.multiply(asset_yield,
                                                            time_to_maturity),
                                       name="asset_yield_exponent")
    barrier_ratio = tf.math.divide(barrier_price, asset_price,
                                   name="barrier_ratio")
    asset_price_term = tf.math.multiply(call_or_put,
                                        tf.math.multiply(asset_price,
                                                         asset_yield_exponent),
                                        name="asset_price_term")
    strike_price_term = tf.math.multiply(call_or_put,
                                         tf.math.multiply(strike_price,
                                                          rate_exponent),
                                         name="strike_price_term")

    # Constructing Matrix with first and second algebraic terms for each integral
    terms_mat = tf.stack((asset_price_term, -1.*strike_price_term,
                          asset_price_term, -1.*strike_price_term,
                          tf.math.multiply(asset_price_term,
                                           tf.math.pow(
                                               barrier_ratio, (2*lamda))),
                          -1.*tf.math.multiply(strike_price_term,
                                               tf.math.pow(barrier_ratio,
                                                           (2*lamda)-2)),
                          tf.math.multiply(asset_price_term,
                                           tf.math.pow(barrier_ratio,
                                                       (2*lamda))),
                          -1.*tf.math.multiply(strike_price_term,
                                               tf.math.pow(barrier_ratio,
                                                           (2*lamda)-2)),
                          tf.math.multiply(rebate, rate_exponent),
                          -1.*tf.math.multiply(tf.math.multiply(rebate,
                                                                rate_exponent),
                                               tf.math.pow(barrier_ratio,
                                                           (2*lamda)-2)),
                          tf.math.multiply(rebate,
                                           tf.math.pow(barrier_ratio,
                                                       tf.math.add(a, b))),
                          tf.math.multiply(rebate,
                                           tf.math.pow(barrier_ratio,
                                                       tf.math.subtract(a, b)),
                                           name="term_matrix")))

    # Constructing Matrix with first and second norm for each integral
    cdf_mat = tf.stack(
        (_ncdf(tf.math.multiply(call_or_put, x)),
         _ncdf(tf.math.multiply(call_or_put,
                                tf.math.subtract(x, time_asset_volatility))),
         _ncdf(tf.math.multiply(call_or_put, x1)),
         _ncdf(tf.math.multiply(call_or_put,
                                tf.math.subtract(x1, time_asset_volatility))),
         _ncdf(tf.math.multiply(below_or_above, y)),
         _ncdf(tf.math.multiply(below_or_above,
                                tf.math.subtract(y, time_asset_volatility))),
         _ncdf(tf.math.multiply(below_or_above, y1)),
         _ncdf(tf.math.multiply(below_or_above,
                                tf.math.subtract(y1, time_asset_volatility))),
         _ncdf(tf.math.multiply(below_or_above,
                                tf.math.subtract(x1, time_asset_volatility))),
         _ncdf(tf.math.multiply(below_or_above,
                                tf.math.subtract(y1, time_asset_volatility))),
         _ncdf(tf.math.multiply(below_or_above, z)),
         _ncdf(tf.math.multiply(below_or_above,
                 tf.math.subtract(
                     z, 2.*tf.math.multiply(b, time_asset_volatility))))),
        name="cdf_matrix")
    # Calculating and returning price for each option
    return tf.reduce_sum(
        tf.math.multiply(
            tf.math.multiply(
                tf.transpose(masks), terms_mat), cdf_mat), axis=0)


# TODO(b/154806390): Binary price signature should be the same as that of the
# vanilla price.
def binary_price(*,
                 volatilities,
                 strikes,
                 expiries,
                 spots=None,
                 forwards=None,
                 discount_factors=None,
                 is_call_options=None,
                 dtype=None,
                 name=None):
  """Computes the Black Scholes price for a batch of binary call or put options.

  The binary call (resp. put) option priced here is that which pays off a unit
  of cash if the underlying asset has a value greater (resp. smaller) than the
  strike price at expiry. Hence the binary option price is the discounted
  probability that the asset will end up higher (resp. lower) than the
  strike price at expiry.

  #### Example

  ```python
    # Price a batch of 5 binary call options.
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Strikes will automatically be broadcasted to shape [5].
    strikes = np.array([3.0])
    # Expiries will be broadcast to shape [5], i.e. each option has strike=3
    # and expiry = 1.
    expiries = 1.0
    computed_prices = tff.black_scholes.binary_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards)
  # Expected print output of prices:
  # [0.         0.         0.15865525 0.99764937 0.85927418]
  ```

  #### References:

  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
  [2] Wikipedia contributors. Binary option. Available at:
  https://en.wikipedia.org/w/index.php?title=Binary_option

  Args:
    volatilities: Real `Tensor` of any shape and dtype. The volatilities to
      expiry of the options to price.
    strikes: A real `Tensor` of the same dtype and compatible shape as
      `volatilities`. The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and compatible shape as
      `volatilities`. The expiry of each option. The units should be such that
      `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `volatilities`. The current spot price of the underlying. Either this
      argument or the `forwards` (but not both) must be supplied.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `volatilities`. The forwards to maturity. Either this argument or the
      `spots` must be supplied but both must not be supplied.
    discount_factors: An optional real `Tensor` of same dtype as the
      `volatilities`. If not None, these are the discount factors to expiry
      (i.e. e^(-rT)). If None, no discounting is applied (i.e. the undiscounted
      option price is returned). If `spots` is supplied and `discount_factors`
      is not None then this is also used to compute the forwards to expiry.
      Default value: None, equivalent to discount factors = 1.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by TensorFlow
        (float32).
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name `binary_price`.

  Returns:
    binary_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the binary options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')

  with tf.name_scope(name or 'binary_price'):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')

    if discount_factors is None:
      discount_factors = tf.convert_to_tensor(
          1.0, dtype=dtype, name='discount_factors')
    else:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      forwards = spots / discount_factors

    sqrt_var = volatilities * tf.math.sqrt(expiries)
    d1 = (tf.math.log(forwards / strikes) + sqrt_var * sqrt_var / 2) / sqrt_var
    d2 = d1 - sqrt_var
    undiscounted_calls = _ncdf(d2)
    if is_call_options is None:
      return discount_factors * undiscounted_calls
    is_call_options = tf.convert_to_tensor(is_call_options,
                                           dtype=tf.bool,
                                           name='is_call_options')
    undiscounted_puts = 1 - undiscounted_calls
    predicate = tf.broadcast_to(is_call_options, tf.shape(undiscounted_calls))
    return discount_factors * tf.where(predicate, undiscounted_calls,
                                       undiscounted_puts)


def _ncdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


_SQRT_2 = np.sqrt(2.0, dtype=np.float64)
