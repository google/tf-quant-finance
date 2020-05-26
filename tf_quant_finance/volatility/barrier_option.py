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
"""Pricing barrier options"""

import tensorflow as tf
import tensorflow_probability as tfp


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
    """
    @tf.function
    def get_out_map(params):
      
      Function maps params to option pricing masks
      Args:
      params: Tuple of tensors. (barrier price, strike_price, option type)

      Returns:
      returns mask used to price specified option
      
      barrier_price = params[0]
      strike_price = params[1]
      otype = params[2]
      # out_map = down_and_in_call_tensor_greater_strike
      if strike_price < barrier_price:
        if otype == down_and_in_call:
          return down_and_in_call_tensor_lower_strike
        elif otype == down_and_in_put:
          return down_and_in_put_tensor_lower_strike
        elif otype == up_and_in_call:
          return up_and_in_call_tensor_lower_strike
        elif otype == up_and_in_put:
          return up_and_in_put_tensor_lower_strike
        elif otype == down_and_out_call:
          return down_and_out_call_tensor_lower_strike
        elif otype == down_and_out_put:
          return down_and_out_put_tensor_lower_strike
        elif otype == up_and_out_call:
          return up_and_out_call_tensor_lower_strike
        elif otype == up_and_out_put:
          return up_and_out_put_tensor_lower_strike
      else:
        if otype == down_and_in_call:
          return down_and_in_call_tensor_greater_strike
        elif otype == down_and_in_put:
          return down_and_in_put_tensor_greater_strike
        elif otype == up_and_in_call:
          return up_and_in_call_tensor_greater_strike
        elif otype == up_and_in_put:
          return up_and_in_put_tensor_greater_strike
        elif otype == down_and_out_call:
          return down_and_out_call_tensor_greater_strike
        elif otype == down_and_out_put:
          return down_and_out_put_tensor_greater_strike
        elif otype == up_and_out_call:
          return up_and_out_call_tensor_greater_strike
        elif otype == up_and_out_put:
          return up_and_out_put_tensor_greater_strike
    """
    
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
    norm = tfp.distributions.Normal(loc=0, scale=1)
    cdf_mat = tf.stack(
        (norm.cdf(tf.math.multiply(call_or_put, x)),
         norm.cdf(
             tf.math.multiply(call_or_put,
                              tf.math.subtract(x, time_asset_volatility))),
         norm.cdf(tf.math.multiply(call_or_put, x1)),
         norm.cdf(
             tf.math.multiply(call_or_put,
                              tf.math.subtract(x1, time_asset_volatility))),
         norm.cdf(tf.math.multiply(below_or_above, y)),
         norm.cdf(
             tf.math.multiply(below_or_above,
                              tf.math.subtract(y, time_asset_volatility))),
         norm.cdf(tf.math.multiply(below_or_above, y1)),
         norm.cdf(
             tf.math.multiply(below_or_above,
                              tf.math.subtract(y1, time_asset_volatility))),
         norm.cdf(
             tf.math.multiply(below_or_above,
                              tf.math.subtract(x1, time_asset_volatility))),
         norm.cdf(
             tf.math.multiply(below_or_above,
                              tf.math.subtract(y1, time_asset_volatility))),
         norm.cdf(tf.math.multiply(below_or_above, z)),
         norm.cdf(
             tf.math.multiply(
                 below_or_above,
                 tf.math.subtract(
                     z, 2.*tf.math.multiply(b, time_asset_volatility))))),
        name="cdf_matrix")
    # Calculating and returning price for each option
    return tf.reduce_sum(
        tf.math.multiply(
            tf.math.multiply(
                tf.transpose(masks), terms_mat), cdf_mat), axis=0)
