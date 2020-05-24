"""Pricing barrier options"""

import tensorflow as tf
import tensorflow_probability as tfp


def _construct_params(rate, asset_yield, asset_price, strike_price, barrier_price, volitility, time_to_maturity):
    """
    Function calculates parameters used in calculating integrals for pricing barrier options
    
    Inputs must be tensors
    args:
    rate: tensor, risk-free interest rate
    asset_yield: tensor, continuous yield on an asset
    asset_price: tensor, price of underyling asset.
    strike_price: tensor, strike prices for the options.
    barrier_price: tensor, price of underlying asset at barrier
    volitility: tensor, Volitility in the underyling asset.
    time_to_maturity: tensor, Number of years ot maturity of option.
    returns:
    array of parameters
    """
    with tf.name_scope("construct_params"):
        time_volitility = tf.math.multiply(volitility, tf.math.sqrt(time_to_maturity), name="time_volitility")
        mu = tf.math.subtract(tf.math.subtract(rate, asset_yield), tf.math.divide(tf.math.square(volitility), 2), name="mu")
        lamda = tf.math.add(1, tf.math.divide(mu, tf.math.square(volitility)), name="lambda")
        # Constructing x
        num = tf.math.log(tf.math.divide(asset_price, strike_price))
        x = tf.math.add(tf.math.divide(num, time_volitility), tf.math.multiply(lamda, time_volitility), name="x")
        num = tf.math.log(tf.math.divide(asset_price, barrier_price))
        x1 = tf.math.add(tf.math.divide(num, time_volitility), tf.math.multiply(lamda, time_volitility), name="x1")
        num = tf.math.log(tf.math.divide(tf.math.square(barrier_price), tf.math.multiply(asset_price, strike_price)))
        y = tf.math.add(tf.math.divide(num, time_volitility), tf.math.multiply(lamda, time_volitility), name="y")
        num = tf.math.log(tf.math.divide(barrier_price, asset_price))
        y1 = tf.math.add(tf.math.divide(num, time_volitility), tf.math.multiply(lamda, time_volitility, name="y1"))
        num = tf.math.sqrt(tf.math.square(mu) + 2.*tf.math.multiply(tf.math.square(volitility), rate))
        b = tf.math.divide(num, tf.math.square(volitility), name="b")
        num = tf.math.log(tf.math.divide(barrier_price, asset_price))
        z = tf.math.add(tf.math.divide(num, time_volitility), tf.math.multiply(b, time_volitility, name="z"))
        a = tf.math.divide(mu, tf.math.square(volitility), name="a")
        return [x, x1, y, y1, lamda, z, a, b, time_volitility]


def price_barrier_option(
        call_or_put,
        below_or_above,
        rate,
        asset_yield,
        asset_price,
        strike_price,
        barrier_price,
        rebate,
        asset_volitility,
        time_to_maturity,
        otype):
    """"
    Function determines the approximate price for the barrier option.

    #### References
    
    [1] given
    [2] Options pricing forumulas
    
    Args:
      call_or_put: A real 1-2D Tensor where each element represents a call or put for the security.
       (1 for call, -1 for put)
      rate: A real 1-2D Tensor where each element represents rate for individual option.
      asset_yield: A real 1-2D tensor that is the yield on asset
      asset_price: A real 1-2D Tensor that is the asset price for the underlying security.
      strike_price: A real 1-2D tensor that is the strike price for the option.
      barrier_price: A real 1-2D tensor that is the barrier price for the option to take effect.
      rebate: A real 1-2D Tensor that is a rebate contingent upon reaching the barrier price.
      asset_volitility: A real 1-2D Tensor that measure the volitility of the asset price. 
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
    DOWN_AND_IN_CALL = 1
    DOWN_AND_IN_PUT = 2
    UP_AND_IN_CALL = 3
    UP_AND_IN_PUT = 4
    DOWN_AND_OUT_CALL = 5
    DOWN_AND_OUT_PUT = 6
    UP_AND_OUT_CALL = 7
    UP_AND_OUT_PUT = 8
    with tf.name_scope("Price_Barrier_Option"):
        # Convert all to tensor and enforce float dtyle
        call_or_put = tf.convert_to_tensor(call_or_put, dtype=tf.float32, name="call_or_put")
        below_or_above = tf.convert_to_tensor(below_or_above, dtype=tf.float32, name="below_or_above")
        rate = tf.convert_to_tensor(rate, dtype=tf.float32, name="rate")
        asset_yield = tf.convert_to_tensor(asset_yield, dtype=tf.float32, name="asset_yield")
        asset_price = tf.convert_to_tensor(asset_price, dtype=tf.float32, name="asset_yield")
        strike_price = tf.convert_to_tensor(strike_price, dtype=tf.float32, name="strike_price")
        barrier_price = tf.convert_to_tensor(barrier_price, dtype=tf.float32, name="barrier_price")
        rebate = tf.convert_to_tensor(rebate, dtype=tf.float32, name="rebate")
        asset_volitility = tf.convert_to_tensor(asset_volitility, dtype=tf.float32, name="asset_volitility")
        time_to_maturity = tf.convert_to_tensor(time_to_maturity, dtype=tf.float32, name="time_to_maturity")
        
        # Construct params for integrals
        params = _construct_params(rate, asset_yield, asset_price, strike_price, barrier_price, asset_volitility, time_to_maturity)
        params = {
            "x": params[0],
            "x1": params[1],
            "y": params[2],
            "y1": params[3],
            "lambda": params[4],
            "z": params[5],
            "a": params[6],
            "b": params[7],
            "time_volitility": params[8]
        }
        
        if otype == DOWN_AND_IN_CALL:
            out_map = [[1.], [-1.], [0.], [1.], [1.], [0.]]
        elif otype == DOWN_AND_IN_PUT:
            out_map = [[1.], [0.], [0.], [0.], [1.], [0.]]
        elif otype == UP_AND_IN_CALL:
            out_map = [[0.], [1.], [-1.], [1.], [1.], [0.]]
        elif otype == UP_AND_IN_PUT:
            out_map = [[0.], [0.], [1.], [0.], [1.], [0.]]
        elif otype == DOWN_AND_OUT_CALL:
            out_map = [[0.], [1.], [0.], [-1.], [0.], [1.]]
        elif otype == DOWN_AND_OUT_PUT:
            out_map = [[0.], [0.], [0.], [0.], [0.], [1.]]
        elif otype == UP_AND_OUT_CALL:
            out_map = [[1.], [-1.], [1.], [-1.], [0.], [1.]]
        elif otype == UP_AND_OUT_PUT:
            out_map = [[1.], [0.], [-1.], [0.], [0.], [1.]]
        
        # Create a matrix
        rate_exponent = tf.math.exp(-1.*tf.math.multiply(rate, time_to_maturity), name="rate_exponent")
        asset_yield_exponent = tf.math.exp(-1.*tf.math.multiply(asset_yield, time_to_maturity), name="asset_yield_exponent")
        barrier_ratio = tf.math.divide(barrier_price, asset_price, name="barrier_ratio")
        asset_price_term = tf.math.multiply(call_or_put, tf.math.multiply(asset_price, asset_yield_exponent))
        strike_price_term = tf.math.multiply(call_or_put, tf.math.multiply(strike_price, rate_exponent))
        call_or_put_x = tf.math.multiply(call_or_put, params["x"])
        call_or_put_x1 = tf.math.multiply(call_or_put, params["x1"])
        below_or_above_x1 = tf.math.multiply(below_or_above, params["x1"])
        below_or_above_y = tf.math.multiply(below_or_above, params["y"])
        below_or_above_y1 = tf.math.multiply(below_or_above, params["y1"])
        below_or_above_z = tf.math.multiply(below_or_above, params["z"])
        out_map_dup = tf.tile([out_map], multiples=[2,1, call_or_put.shape[0]])
        
        # Construct Terms 
        I12_terms = tf.math.multiply(out_map_dup[:, :2, :], [[asset_price_term], [-1.*strike_price_term]])
        I34_terms = tf.math.multiply(out_map_dup[:, 2:4, :], [[tf.math.multiply(asset_price_term, tf.math.pow(barrier_ratio, 2*params["lambda"]))],
                                                              [-1.*tf.math.multiply(strike_price_term, tf.math.pow(barrier_ratio, (2*params["lambda"])-2))]])
        I5_terms = tf.math.multiply(out_map_dup[:, 4:5, :], [[tf.math.multiply([rebate], rate_exponent)],
                                                              [-1.*tf.math.multiply([rebate], tf.math.multiply(rate_exponent, tf.math.pow(barrier_ratio, (2*params["lambda"])-2)))]])
        I6_terms = tf.math.multiply(out_map_dup[:, 5:, :], [[tf.math.multiply([rebate], tf.math.pow(barrier_ratio, params["a"]+params["b"]))],
                                                             [tf.math.multiply([rebate], tf.math.pow(barrier_ratio, params["a"]-params["b"]))]])
        
        # Construct Distribution Matricies
        norm = tfp.distributions.Normal(loc=0, scale=1)
        I1_cdf = tf.math.multiply(out_map_dup[:, 0, :], [norm.cdf(call_or_put_x),
                                                         norm.cdf(tf.math.subtract(call_or_put_x, tf.math.multiply(call_or_put, params["time_volitility"])))])
        I2_cdf = tf.math.multiply(out_map_dup[:, 1, :], [norm.cdf(call_or_put_x1),
                                                         norm.cdf(tf.math.subtract(call_or_put_x1, tf.math.multiply(call_or_put, params["time_volitility"])))])
        I3_cdf = tf.math.multiply(out_map_dup[:, 2, :], [norm.cdf(below_or_above_y),
                                                         norm.cdf(tf.math.subtract(below_or_above_y, tf.math.multiply(below_or_above, params["time_volitility"])))])
        I4_cdf = tf.math.multiply(out_map_dup[:, 3, :], [norm.cdf(below_or_above_y1),
                                                         norm.cdf(tf.math.subtract(below_or_above_y1, tf.math.multiply(below_or_above, params["time_volitility"])))])
        I5_cdf = tf.math.multiply(out_map_dup[:, 4, :], [norm.cdf(tf.math.subtract(below_or_above_x1, tf.math.multiply(below_or_above, params["time_volitility"]))),
                                                         norm.cdf(tf.math.subtract(below_or_above_y1, tf.math.multiply(below_or_above, params["time_volitility"])))])
        I6_cdf = tf.math.multiply(out_map_dup[:, 5, :], [norm.cdf(below_or_above_z),
                                                        norm.cdf(tf.math.multiply(below_or_above, params["z"]-(2*params["b"]*params["time_volitility"])))])
        
        # Build Matricies
        cdf_matrix = tf.concat((I1_cdf, I2_cdf, I3_cdf, I4_cdf, I5_cdf, I6_cdf), axis=1, name="cdf_matrix")
        term_matrix = tf.squeeze(tf.concat((I12_terms, I34_terms, I5_terms, I6_terms), axis=1, name="term_matrix"))
        
        # Approximation is the sum of the diagonal of mat mul
        approx = tf.linalg.diag_part(tf.linalg.matmul(term_matrix, cdf_matrix, transpose_a=True, a_is_sparse=True, b_is_sparse=True))
        return tf.reduce_sum(tf.math.multiply(approx, tf.squeeze(out_map))).numpy()
