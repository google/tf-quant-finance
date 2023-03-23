# Copyright 2023 Google LLC
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
"""Analytical approximation for the spread-option price under Black-Scholes using 
    Kirk's approximation and WKB method.
"""

import numpy as np
import tensorflow.compat.v2 as tf
from typing import Optional
from tf_quant_finance import types

def spread_option_price(volatilities1: types.RealTensor,
                        volatilities2: types.RealTensor,
                        correlations: types.RealTensor,
                        strikes: types.RealTensor,
                        expiries: types.RealTensor,
                        spots1: Optional[types.RealTensor] = None,
                        spots2: Optional[types.RealTensor] = None,
                        forwards1: Optional[types.RealTensor] = None,
                        forwards2: Optional[types.RealTensor] = None,
                        discount_rates: Optional[types.RealTensor] = None,
                        dividend_rates1: Optional[types.RealTensor] = None,
                        dividend_rates2: Optional[types.RealTensor] = None,
                        discount_factors: Optional[types.RealTensor] = None,
                        is_call_options: Optional[types.BoolTensor] = None,#if not provided, assume call options
                        dtype: tf.DType =None,
                        name: str =None):
    """Computes the Black Scholes price for a batch of call or put spread options
    based on Kirk's approximation using WKB method.
    
    #### Example
    ```pythona
       # Price a batch of 1 call spread options
       volatilities1 = np.array([0.10])
       volatilities2 = np.array([0.15])
       correlations = np.array([0.3])
       strikes = np.array([5.0])
       expiries = 1.0
       spots1 = np.array([109.998])
       spots2 = np.array([100])
       computed_price = tff.black_scholes.spread_option_price(
        volatilities1=volatilities1,
        volatilities2=volatilities2,
        correlations=correlations,
        strikes=strikes,
        expiries=expiries,
        spots1=spots1,
        spots2=spots2,
        discount_rates=discount_rates,
        dividend_rates1=dividend_rates1,
        dividend_rates2=dividend_rates2,
    )
    # Expected print output of computed prices:
    # [ 8.36364059 ]
    ```
    #### References:
    [1] C. F. Lo, 2013. A simple derivation of Kirk's approximation for spread options. Applied Mathematical Letters.
    [2] D. Prathumwan & K. Trachoo, 2020. On the solution of two-dimensional fractional Black-Scholes Equation for 
    European put option. Advances in Difference Equations.
    Args:
    volatilities1: Real `Tensor` of any shape and dtype. The volatilities of the first asset
        to expiry of the options to price.
    volatilities2: Real `Tensor` of any shape and dtype. The volatilities of the second asset 
        to expiry of the options to price.
    correlations: Real `Tensor` of the same dtype and compatible shapre as the 
        volatilities. The correlations of the two underlying prices.
    strikes: A real `Tensor` of the same dtype and compatible shape as the
      volatilities. The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and compatible shape as
      `volatilities`. The expiry of each option. 
      #The units should be such that `expiry * volatility**2` is dimensionless.
    spots1: A real `Tensor` of any shape that broadcasts to the shape of the
      volatilities. The current spot price of the first underlying. Either this
      argument or the `forwards1` (but not both) must be supplied.
    spots2: A real `Tensor` of any shape that broadcasts to the shape of the
      volatilities. The current spot price of the second underlying. Either this
      argument or the `forwards2` (but not both) must be supplied.
    forwards1: A real `Tensor` of any shape that broadcasts to the shape of the
      volatilities. The forwards to maturity of the first underlying. Either this 
      argument or the `spots1` must be supplied but both must not be supplied.
    forwards1: A real `Tensor` of any shape that broadcasts to the shape of the
      volatilities. The forwards to maturity of the second underlying. Either this 
      argument or the `spots2` must be supplied but both must not be supplied.
    discount_rates: An optional real `Tensor` of same dtype as the
      volatilities and of the shape that broadcasts with volatilities.
      If not `None`, discount factors are calculated as e^(-rT),
      where r are the discount rates, or risk free rates. At most one of
      `discount_rates` and `discount_factors` can be supplied.
      Default value: `None`, equivalent to r = 0 and discount factors = 1 when
      `discount_factors` also not given.
    dividend_rates: An optional real `Tensor` of same dtype as the
      volatilities and of the shape that broadcasts with volatilities.
      Default value: `None`, equivalent to q = 0.
    discount_factors: An optional real `Tensor` of same dtype as the
      volatilities. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with `discount_rates`. If neither is
      given, no discounting is applied (i.e. the undiscounted option price is
      returned). If `spots` is supplied and `discount_factors` is not `None`
      then this is also used to compute the forwards to expiry. At most one of
      `discount_rates` and `discount_factors` can be supplied.
      Default value: `None`, which maps to e^(-rT) calculated from
      discount_rates.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: `None` which maps to the default dtype inferred by
      TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: `None` which is mapped to the default name `spread_option_price`.
    """
    if (spots1 is None) == (forwards1 is None):
        if (spots2 is None) == (forwards2 is None):
            raise ValueError('Either spots or forwards must be supplied but not both.')
        elif (spots2 is not None) or (forwards2 is not None):
            raise ValueError('Either spots or forwards for both assets must be supplied.')
    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may '
                        'be supplied')
    
    with tf.name_scope(name or 'spread_option_price'):

        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        dtype = strikes.dtype
        volatilities1 = tf.convert_to_tensor(
            volatilities1, dtype=dtype, name='volatilities1')
        volatilities2 = tf.convert_to_tensor(
            volatilities2, dtype=dtype, name='volatilities2')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        correlations = tf.convert_to_tensor(correlations, dtype=dtype, name='correlations')

        if discount_rates is not None:
            discount_rates = tf.convert_to_tensor(
                discount_rates, dtype=dtype, name='discount_rates')
            discount_factors = tf.exp(-discount_rates * expiries)
        elif discount_factors is not None:
            discount_factors = tf.convert_to_tensor(
                discount_factors, dtype=dtype, name='discount_factors')
            discount_rates = -tf.math.log(discount_factors) / expiries
        else:
            discount_rates = tf.convert_to_tensor(
                0.0, dtype=dtype, name='discount_rates')
            discount_factors = tf.convert_to_tensor(
                1.0, dtype=dtype, name='discount_factors')
        if dividend_rates1 is not None:
            dividend_rates1 = tf.convert_to_tensor(
                dividend_rates1, dtype=dtype, name='dividend_rates1')
        else:
            dividend_rates1 = tf.convert_to_tensor(
                0.0, dtype=dtype, name='dividend_rates1')
        if dividend_rates2 is not None:
            dividend_rates2 = tf.convert_to_tensor(
                dividend_rates2, dtype=dtype, name='dividend_rates2')
        else:
            dividend_rates2 = tf.convert_to_tensor(
                0.0, dtype=dtype, name='dividend_rates2')
        if forwards1 is not None and forwards2 is not None:
            forwards1 = tf.convert_to_tensor(forwards1, dtype=dtype, name='forwards1')
            forwards2 = tf.convert_to_tensor(forwards2, dtype=dtype, name='forwards2')
        else:
            spots1 = tf.convert_to_tensor(spots1, dtype=dtype, name='spots1')
            spots2 = tf.convert_to_tensor(spots2, dtype=dtype, name='spots2')
            forwards1 = spots1 * tf.exp((discount_rates - dividend_rates1) * expiries)
            forwards2 = spots2 * tf.exp((discount_rates - dividend_rates2) * expiries)
            
        sqrt_var_eff = volatilities2 * tf.math.divide_no_nan(forwards2, (forwards2 + strikes))
        sqrt_var_ = tf.math.sqrt(tf.math.square(volatilities1) - 2 * correlations * volatilities1 * sqrt_var_eff + tf.math.square(sqrt_var_eff))
        sqrt_var = sqrt_var_ * tf.math.sqrt(expiries)

        d1 = tf.math.divide_no_nan(tf.math.log(forwards1 / (forwards2 + strikes)), 
                                   sqrt_var) + sqrt_var / 2
        d2 = d1 - sqrt_var

        undiscounted_calls = tf.where(sqrt_var > 0,
                                        forwards1 * _ncdf(d1) - (forwards2 + strikes) * _ncdf(d2),
                                        tf.math.maximum(forwards1 - forwards2 - strikes, 0.0))#TODO
        if is_call_options is None:
            return discount_factors * undiscounted_calls
        
        undiscounted_puts = tf.where(sqrt_var > 0, 
                                    (forwards2 + strikes) * _ncdf(-d2) - forwards1 * _ncdf(-d1), 
                                    tf.math.maximum(forwards2 + strikes - forwards1, 0.0))

        return discount_factors * undiscounted_puts


def _ncdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


_SQRT_2 = np.sqrt(2.0, dtype=np.float64)
