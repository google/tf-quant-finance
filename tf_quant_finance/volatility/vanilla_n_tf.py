# Lint as: python3
# Copyright 2020 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

# straight fwd implementation of the Bachelier pricing
# there is a version with just one call to exp !!

    
def option_price(spots, 
                 strikes, 
                 volatilities, 
                 expiries,
                 rates,
                 is_call_options=None,
                 dtype = None,
                 name = None):
    """ Compute the Bachelier price for a batch of European options.
    
  ## References:
  [1] Kienitz, J. "Interest Rate Derivatives Explained I", Plagrave McMillan (2014) p.119
  [2] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3428994
    
    Parameters
    ----------
    spots : A real `Tensor` of any shape. The current spot prices to
      expiry.
    strikes : A real `Tensor` of the same shape and dtype as `spots`. The
      strikes of the options to be priced.
    volatilities : A real `Tensor` of same shape and dtype as `spots`. The
      volatility to expiry.
    expiries : A real `Tensor` of same shape and dtype as `spots`. The expiry
      for each option. The units should be such that `expiry * volatility**2` is
      dimensionless.
    rates : A real `Tensor` of the same shape and dtype as `spots`. The
      rates of the options to be priced.
    is_call_options: A boolean `Tensor` of a shape compatible with `forwards`.
      Indicates whether to compute the price of a call (if True) or a put (if
      False). If not supplied, it is assumed that every element is a call.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by TensorFlow
      (float32).
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name `option_price`.

    Returns
    -------
    option_prices: A `Tensor` of the same shape as `spots`. The Bachelier
    price of the options.
    
    #### Examples
    ```python
    spots = np.array([0.03, 0.02])
    strikes = np.array([.02, .02])
    volatilities = np.array([.004, .005])
    expiries = 2.0
    rates = [0.02, 0.01]
    computed_prices = option_price(
      spots,
      strikes,
      volatilities,
      expiries,
      rates,
      dtype=tf.float64)
    # Expected print output of computed prices:
    # <tf.Tensor: id=2482, shape=(2,), dtype=float32, numpy=array([0.01605039, 0.00720789], dtype=float32)>
    ```

    """   
    with tf.compat.v1.name_scope(
      name,
      default_name='option_price',
      values=[
          spots, strikes, volatilities, expiries, rates,
          is_call_options
      ]):
   
        spots = tf.convert_to_tensor(spots, dtype=tf.float64, name='forwards')
        strikes = tf.convert_to_tensor(strikes, dtype=tf.float64, name='strikes')
        volatilities = tf.convert_to_tensor(volatilities, tf.float64, name='volatilities')
        expiries = tf.convert_to_tensor(expiries, tf.float64, name='expiries')
        rates = tf.convert_to_tensor(rates, tf.float64, name='rates')
        
        z = tf.zeros_like(strikes)
        
        normal = tfp.distributions.Normal(
            loc=tf.zeros([], dtype=spots.dtype), scale=1)
        
        df = tf.math.exp(-rates*expiries)
        vt = volatilities * tf.math.sqrt(expiries)
        
        z = tf.where(rates == 0., (spots - strikes)/vt, 
                     (spots-strikes*df)/(volatilities 
                      * tf.math.sqrt(0.5*(1.-tf.math.exp(-2.*rates*expiries))/rates)))
                 
        n1 = normal.cdf(z)
        n2 = normal.prob(z)
        calls = tf.where(rates==0., (spots - strikes) * n1 + vt * n2,
                         (spots - strikes*df)*n1 
                         + volatilities*tf.math.sqrt(0.5*(1-tf.math.exp(-2*rates*expiries))/rates))
                         
            
        if is_call_options is None:
            return calls
        
        puts = calls - spots + strikes * tf.math.exp(-rates*expiries)
        
        return tf.where(is_call_options, calls, puts)
    
    
    
def vanilla_n_dawson_tf(forwards, 
                 strikes, 
                 volatilities, 
                 expiries,
                 discount_factors = None,
                 is_call_options=None,
                 dtype = None,
                 name = None):

    """Computes the Black Scholes price for a batch of European options.

        ## References:
        [1] Dawson, P., Blake, D., Cairns, A. J. G. and Dowd, K.: Options on normal under-
            lyings, CRIS Discussion Paper Series – 2007.VII, 2007.

        Args:
        forwards: A real `Tensor` of any shape. The current forward prices to
        expiry.
        strikes: A real `Tensor` of the same shape and dtype as `forwards`. The
            strikes of the options to be priced.
        volatilities: A real `Tensor` of same shape and dtype as `forwards`. The
            volatility to expiry.
        expiries: A real `Tensor` of same shape and dtype as `forwards`. The expiry
            for each option. The units should be such that `expiry * volatility**2` is
            dimensionless.
        discount_factors: A real `Tensor` of same shape and dtype as the `forwards`.
            The discount factors to expiry (i.e. e^(-rT)). If not specified, no
            discounting is applied (i.e. the undiscounted option price is returned).
            Default value: None, interpreted as discount factors = 1.
        is_call_options: A boolean `Tensor` of a shape compatible with `forwards`.
            Indicates whether to compute the price of a call (if True) or a put (if
            False). If not supplied, it is assumed that every element is a call.
        dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
            of any supplied non-`Tensor` arguments to `Tensor`.
            Default value: None which maps to the default dtype inferred by TensorFlow
            (float32).
        name: str. The name for the ops created by this function.
            Default value: None which is mapped to the default name `option_price`.

        Returns:
            option_prices: A `Tensor` of the same shape as `forwards`. The Bachelier
            price of the options.


         #### Examples
            ```python
            spots = np.array([0.03, 0.02])
            strikes = np.array([.02, .02])
            volatilities = np.array([.004, .005])
            expiries = 2.0
            expiries = 1.0
            computed_prices = option_price(
                forwards,
                strikes,
                volatilities,
                expiries,
                dtype=tf.float64)
        # Expected print output of computed prices:
        # <tf.Tensor: id=2527, shape=(2,), dtype=float32, numpy=array([0.01008754, 0.00282095], dtype=float32)>
        ```
    """  
    with tf.compat.v1.name_scope(
      name,
      default_name='option_price_dawson',
      values=[
          forwards, strikes, volatilities, expiries, discount_factors,
          is_call_options
      ]):
   
        forwards = tf.convert_to_tensor(forwards, dtype=None, name='forwards')
        strikes = tf.convert_to_tensor(strikes, dtype=None, name='strikes')
        volatilities = tf.convert_to_tensor(volatilities, dtype=None, name='volatilities')
        expiries = tf.convert_to_tensor(expiries, dtype=None, name='expiries')
        
        if discount_factors is None:
            discount_factors = 1.
        discount_factors = tf.convert_to_tensor(
            discount_factors, dtype=dtype, name='discount_factors')
        
        vt = volatilities * tf.math.sqrt(expiries)
        normal = tfp.distributions.Normal(
            loc=tf.zeros([], dtype=forwards.dtype), scale=1)
        
        z = (forwards - strikes) / vt
                 
        n1 = normal.cdf(z)
        n2 = normal.prob(z)
        undiscounted_calls = (forwards-strikes) * n1 + vt * n2
                         
        if is_call_options is None:
            return discount_factors * undiscounted_calls
        undiscounted_forward = forwards - strikes
        undiscounted_puts = undiscounted_calls - undiscounted_forward
        
        return discount_factors * tf.where(is_call_options, undiscounted_calls,
                                       undiscounted_puts)
    
    
    