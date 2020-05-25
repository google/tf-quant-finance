"""Pricing barrier options"""
import tensorflow as tf
import tensorflow_probability as tfp


def price_barrier_option(
        call_or_put,
        below_or_above,
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
  Function determines the approximate price for the barrier option.
  #### References
  [1] given
  [2] Options pricing forumulas

  Args:
  call_or_put: A real 1-2D Tensor where each element represents a call or put for the security.(1 for call, -1 for put)
  rate: A real 1-2D Tensor where each element represents rate for individual option.
  asset_yield: A real 1-2D tensor that is the yield on asset
  asset_price: A real 1-2D Tensor that is the asset price for the underlying security.
  strike_price: A real 1-2D tensor that is the strike price for the option.
  barrier_price: A real 1-2D tensor that is the barrier price for the option to take effect.
  rebate: A real 1-2D Tensor that is a rebate contingent upon reaching the barrier price.
  asset_volatility: A real 1-2D Tensor that measure the volatility of the asset price.
  time_to_maturity: A real 1-2D Tensor with time to maturity of option.
  otype: An Integer that tells which barrier option to approimate
  0 -> down in call
  1 -> down in put
  2 -> up in call
  3 -> up in put
  4 -> down out call
  5 -> down out put
  6 -> up out call
  8 -> up out put
  Returns:
  A Tensor of same shape as input that is the approximate price of the barrier option.
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
    # Convert all to tensor and enforce float dtyle
    call_or_put = tf.convert_to_tensor(
        call_or_put, dtype=tf.float32, name="call_or_put")
    below_or_above = tf.convert_to_tensor(
        below_or_above, dtype=tf.float32, name="below_or_above")
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

    # masks
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
    def get_out_map(params):
      """
      Function maps params to option pricing masks
      Args:
      params: Tuple of tensors. (barrier price, strike_price, option type)
      
      Returns:
      returns mask used to price specified option
      """
      barrier_price = params[0]
      strike_price = params[1]
      otype = params[2]
      if strike_price < barrier_price:
        if otype == down_and_in_call:
          out_map = down_and_in_call_tensor_lower_strike
        elif otype == down_and_in_put:
          out_map = down_and_in_put_tensor_lower_strike
        elif otype == up_and_in_call:
          out_map = up_and_in_call_tensor_lower_strike
        elif otype == up_and_in_put:
          out_map = up_and_in_put_tensor_lower_strike
        elif otype == down_and_out_call:
          out_map = down_and_out_call_tensor_lower_strike
        elif otype == down_and_out_put:
          out_map = down_and_out_put_tensor_lower_strike
        elif otype == up_and_out_call:
          out_map = up_and_out_call_tensor_lower_strike
        elif otype == up_and_out_put:
          out_map = up_and_out_put_tensor_lower_strike
      else:
        if otype == down_and_in_call:
          out_map = down_and_in_call_tensor_greater_strike
        elif otype == down_and_in_put:
          out_map = down_and_in_put_tensor_greater_strike
        elif otype == up_and_in_call:
          out_map = up_and_in_call_tensor_greater_strike
        elif otype == up_and_in_put:
          out_map = up_and_in_put_tensor_greater_strike
        elif otype == down_and_out_call:
          out_map = down_and_out_call_tensor_greater_strike
        elif otype == down_and_out_put:
          out_map = down_and_out_put_tensor_greater_strike
        elif otype == up_and_out_call:
          out_map = up_and_out_call_tensor_greater_strike
        elif otype == up_and_out_put:
          out_map = up_and_out_put_tensor_greater_strike
      return out_map

    # build masks
    if otype.shape == ():
      masks = get_out_map((barrier_price, strike_price, otype))
    else:
      masks = tf.map_fn(
          get_out_map,
          (barrier_price, strike_price, otype), dtype=tf.float32)

    # Calculate params for integrals
    time_asset_volatility = tf.math.multiply(
        asset_volatility, tf.math.sqrt(time_to_maturity),
        name="time_asset_volatility")
    mu = tf.math.subtract(
        tf.math.subtract(rate, asset_yield),
        tf.math.divide(tf.math.square(asset_volatility), 2), name="mu")
    lamda = tf.math.add(1,
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

    # Constructing Matix with first and second terms for each integral
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
        name="norm_matrix")

    # Calculating and returning price for each option
    return tf.reduce_sum(
        tf.math.multiply(
            tf.math.multiply(
                tf.transpose(masks), terms_mat), cdf_mat), axis=0)
