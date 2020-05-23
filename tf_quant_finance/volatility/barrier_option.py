"""Pricing barrier options"""

import tensorflow as tf
import tensorflow_probability as tfp


def _integral12(call_or_put, rate, asset_price, asset_yield, time_to_maturity, strike_price, params, x_param):
    """
    Function approximates the the first or second integral required to appoximate barrier options
    
    Inputs must be tensors
    args:
    rate: tensor, risk-free interets rate
    call_or_put: Tensor, float32 determines if appoximating a call or put option(1 for call, -1 for put).
    asset_price: Tensor, float32 price of underyling asset.
    asset_yield: Tensor, float32 Continous yield on an asset.
    time_to_maturity: Tensor, float32 Number of years to maturity for option.
    strike_price: Tensor, float32 strike prices for the options.
    params: Dictionary, Params required for calculating the integeral.
    x_param: String. Determines which x value to use, x for integral1 x1 for integral2
    returns:
    approxmation: the approximate value for integral1
    """
    with tf.name_scope("Integral12"):
        x = params[x_param]
        time_volitility = params["time_volitility"]
        norm  = tfp.distributions.Normal(loc=0, scale=1)
        exp1 = tf.math.exp(tf.math.multiply(-1*asset_yield, time_to_maturity))
        cdf1 = norm.cdf(tf.math.multiply(call_or_put, x))
        term1 = tf.math.multiply(call_or_put, tf.math.multiply(asset_price, tf.math.multiply(exp1, cdf1)))
        exp2 = tf.math.exp(tf.math.multiply(-1*rate, time_to_maturity))
        cdf2 = norm.cdf(tf.math.subtract(tf.math.multiply(call_or_put, x), tf.math.multiply(call_or_put, time_volitility)))
        term2 = tf.math.multiply(call_or_put, tf.math.multiply(strike_price, tf.math.multiply(exp2, cdf2)))
        return term1-term2


def _integral34(call_or_put, below_or_above, rate, asset_price, asset_yield, time_to_maturity, strike_price, barrier_price, params, y_param):
    """
    Function approximates the the third or fourth integral required to appoximate barrier options
    
    Inputs must be tensors
    args:
    call_or_put: Tensor, float32 determines if appoximating a call or put option(1 for call, -1 for put).
    below_or_above: Tensor, float32 indicates if the asset price starts from above or below the barrier. 1 for above, -1 for below
    rate: tensor, risk-free interets rate
    asset_price: Tensor, float32 price of underyling asset.
    asset_yield: Tensor, float32 Continous yield on an asset.
    time_to_maturity: Tensor, float32 Number of years to maturity for option.
    strike_price: Tensor, float32 strike prices for the options.
    barrier_price: Tensor, float32 price at which option becomes active.
    params: Dictionary, Params required for calculating the integeral.
    y_param: String. Determines which y value to use, y for integral1 y1 for integral2
    returns:
    approxmation: the approximate value for integral1
    """
    with tf.name_scope("Integral34"):
        y = params[y_param]
        time_volitility = params["time_volitility"]
        lamda = params["lambda"]
        norm  = tfp.distributions.Normal(loc=0, scale=1)
        exp1 = tf.math.exp(tf.math.multiply(-1*asset_yield, time_to_maturity))
        cdf1 = norm.cdf(tf.math.multiply(below_or_above, y))
        div = tf.math.divide(barrier_price, asset_price)
        term1 = tf.math.multiply(call_or_put, tf.math.multiply(asset_price, tf.math.multiply(exp1, tf.math.multiply(tf.math.pow(div, 2*lamda), cdf1))))        
        exp2 = tf.math.exp(tf.math.multiply(-1*rate, time_to_maturity))
        cdf2 = norm.cdf(tf.math.subtract(tf.math.multiply(below_or_above, y), tf.math.multiply(below_or_above, time_volitility)))
        term2 = tf.math.multiply(call_or_put, tf.math.multiply(strike_price, tf.math.multiply(exp2, tf.multiply(tf.math.pow(div, (2*lamda)-2), cdf2))))
        return term1-term2


def _integral5(below_or_above, pay_off, rate, asset_price, time_to_maturity, barrier_price, params):
    """"
    Function approximates the fifth integral for pricing options
    args:
    below_or_above: Tensor, float32 indicates if the asset price starts from above or below the barrier. 1 for above, -1 for below
    rate: tensor, risk-free interets rate
    asset_price: Tensor, float32 price of underyling asset.
    time_to_maturity: Tensor, float32 Number of years to maturity for option
    barrier_price: Tensor, float32 price at which option becomes active.
    params: Dictionary, Params required for calculating the integeral.
    returns:
    fifth integral aproximation
    """
    with tf.name_scope("Integral5"):
        x1 = params["x1"]
        y1 = params["y1"]
        lamda = params["lambda"]
        time_volitility = params["time_volitility"]
        norm = tfp.distributions.Normal(loc=0, scale=1)
        div = tf.math.divide(barrier_price, asset_price)
        cdf1 = norm.cdf(tf.math.subtract(tf.math.multiply(below_or_above, x1),
                                          tf.math.multiply(below_or_above, time_volitility)))
        cdf2 = norm.cdf(tf.math.subtract(tf.math.multiply(below_or_above, y1),
                                          tf.math.multiply(below_or_above, time_volitility)))
        diff = tf.math.subtract(cdf1, tf.math.multiply(
            tf.math.pow(div, (2*lamda)-2), cdf2))
        exp = tf.math.exp(tf.math.multiply(-1*rate, time_to_maturity))
        I5 = tf.math.multiply(pay_off, tf.math.multiply(
            exp, diff))
        return I5


def _integral6(below_or_above, pay_off, asset_price, barrier_price, params):
    """
    Function approximates the sixth integral for pricing barrier options
    args:
    below_or_above: Tensor, float32 indicates if the asset price starts from above or below the barrier. 1 for above, -1 for below
    pay_off: tensor, pay off for option
    rate: tensor, risk-free interets rate
    asset_price: Tensor, float32 price of underyling asset.
    barrier_price: Tensor, float32 price at which option becomes active.
    params: Dictionary, Params required for calculating the integeral.
    returns:
    fifth integral aproximation
    """
    with tf.name_scope("Integral6"):
        a = params["a"]
        b = params["b"]
        z = params["z"]
        time_volitility = params["time_volitility"]
        div = tf.math.divide(barrier_price, asset_price)
        norm = tfp.distributions.Normal(loc=0, scale=1)
        cdf1 = norm.cdf(tf.math.multiply(below_or_above, z))
        cdf2 = norm.cdf(tf.math.subtract(
            tf.math.multiply(below_or_above, z),
            2.*tf.math.multiply(below_or_above, tf.math.multiply(b, time_volitility))))
        div1 = tf.math.pow(div, a+b)
        div2 = tf.math.pow(div, a-b)
        diff = tf.math.subtract(tf.math.multiply(div1, cdf1),
                                tf.math.multiply(div2, cdf2))
        I6 = tf.math.multiply(pay_off, diff)
        return I6


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

def down_and_in_call(below_or_above, rate, asset_yield, asset_price, strike_price, barrier_price, rebate, time_to_maturity, params):
    with tf.name_scope("Down_and_in_Call"):
        def down_in_call_strike_greater():
            return _integral34(1., below_or_above, rate, asset_price, asset_yield, time_to_maturity, strike_price, barrier_price, params, "y")  + _integral5(below_or_above, rebate, rate, asset_price, time_to_maturity, barrier_price, params)
        def down_in_call_barrier_greater():
            return
    
        rtn_tensor = tf.where(strike_price > barrier_price, x=dic_strine_greater(), y=dic_strine_greater(), name="down_and_in_call")
        return rtn_tensor


def down_and_in_put(below_or_above, rate, asset_yield, asset_price, strike_price, barrier_price, rebate, volitility, time_to_maturity):
    with tf.name_scope("Down_and_in_put"):
        param = _construct_params(rate, asset_yield, asset_price, strike_price, barrier_price, volitility, time_to_maturity)
        params = {
            "x": param[0],
            "x1": param[1],
            "y": param[2],
            "y1": param[3],
            "lambda": param[4],
            "z": param[5],
            "a": param[6],
            "b": param[7],
            "time_volitility": param[8]
        }
        put = _integral12(-1., rate, asset_price, asset_yield, time_to_maturity, strike_price, params, "x1") - _integral34(-1., below_or_above, rate, asset_price, asset_yield, time_to_maturity, strike_price, barrier_price, params, "y") + _integral34(-1., below_or_above, rate, asset_price, asset_yield, time_to_maturity, strike_price, barrier_price, params, "y1") +  _integral5(below_or_above, rebate, rate, asset_price, time_to_maturity, barrier_price, params)
        return put


def up_and_in_call(below_or_above, rate, asset_yield, asset_price, strike_price, barrier_price, rebate, volitility, time_to_maturity):
    with tf.name_scope("Up_and_in_Call"):
        param = _construct_params(rate, asset_yield, asset_price, strike_price, barrier_price, volitility, time_to_maturity)
        params = {
            "x": param[0],
            "x1": param[1],
            "y": param[2],
            "y1": param[3],
            "lambda": param[4],
            "z": param[5],
            "a": param[6],
            "b": param[7],
            "time_volitility": param[8]
        }
        call = _integral12(1., rate, asset_price, asset_yield, time_to_maturity, strike_price, params, "x") + _integral5(below_or_above, rebate, rate, asset_price, time_to_maturity, barrier_price, params)
        return call


def up_and_in_put(below_or_above, rate, asset_yield, asset_price, strike_price, barrier_price, rebate, volitility, time_to_maturity):
    with tf.name_scope("Up_and_in_Put"):
        param = _construct_params(rate, asset_yield, asset_price, strike_price, barrier_price, volitility, time_to_maturity)
        params = {
            "x": param[0],
            "x1": param[1],
            "y": param[2],
            "y1": param[3],
            "lambda": param[4],
            "z": param[5],
            "a": param[6],
            "b": param[7],
            "time_volitility": param[8]
        }
        put = _integral12(-1., rate, asset_price, asset_yield, time_to_maturity, strike_price, params, "x") - _integral12(-1., rate, asset_price, asset_yield, time_to_maturity, strike_price, params, "x1") + _integral34(-1., below_or_above, rate, asset_price, asset_yield, time_to_maturity, strike_price, barrier_price, params, "y1") + _integral5(below_or_above, rebate, rate, asset_price, time_to_maturity, barrier_price, params)
        return put




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
        print("cp: ", call_or_put)
        print("ba: ", below_or_above)
        print("a: ", params["a"])
        print("b: ", params["b"])
        print("z: ", params["z"])
        print("tv: ", params["time_volitility"])

        
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
        print("otype: ", otype)
        
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
        print(approx)
        return tf.reduce_sum(tf.math.multiply(approx, tf.squeeze(out_map))).numpy()
