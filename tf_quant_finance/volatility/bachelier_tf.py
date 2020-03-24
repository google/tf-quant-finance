"""
Created on Fri Nov 22 15:22:13 2019

# Copyright 2020 Joerg Kienitz

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

@author: Joerg Kienitz
"""

import tensorflow.compat.v2 as tf
import numpy as np


def _ncdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1.) / 2.0

def _npdf(x):
    return tf.math.exp(-x**2/2)/_SQRT_2/_SQRT_pi

_SQRT_2 = tf.math.sqrt(tf.constant(2.0,dtype=tf.float64)) #1.4142135623730951   
_SQRT_pi = tf.math.sqrt(tf.constant(np.pi,dtype=tf.float64))

# straight fwd implementation of the Bachelier pricing
# there is a version with just one call to exp !!

    
def bachelier_option_price(spots, 
                 strikes, 
                 volatilities, 
                 expiries,
                 discount_rates = None,
                 discount_factors = None,
                 is_call_options=None,
                 dtype = None,
                 name = None):
    """ computes the Bachelier price for a batch of European options.
    We assume a standard Brownian motion of the form
       dS = r dt + sigma dW
    for the underlying
       
  ## References:
  [1] Kienitz, J. "Interest Rate Derivatives Explained I", Palgrave McMillan (2014) p.119
      Link: https://www.palgrave.com/gp/book/9781137360069
  [2] Terakado, Satoshi: On the Option Pricing Formula Based on the Bachelier Model
      Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3428994
    
    #### Examples

        spots = np.array([0.03, 0.02])
        strikes = np.array([.02, .02])
        volatilities = np.array([.004, .005])
        expiries = 2.0
        discount_rates = [0.02, 0.01]
        computed_prices = bachelier_option_price(
                spots,
                strikes,
                volatilities,
                expiries,
                dtype=tf.float64)
    # Expected print output of computed prices:
    # <tf.Tensor: id=90474, shape=(2,), dtype=float64, numpy=array([0.01008754, 0.00199471])>
    
    Args:
    
    spots : A real `Tensor` of any shape. The current spot prices to
      expiry.
    strikes : A real `Tensor` of the same shape and dtype as `spots`. The
      strikes of the options to be priced.
    volatilities : A real `Tensor` of same shape and dtype as `spots`. The
      volatility to expiry.
    expiries : A real `Tensor` of same shape and dtype as `spots`. 
    discount_rates : rates from which discount factor via 
        discount factor = exp(-discount rate * T) are calculated
    discountr_factors : A real 'Tensor' of same shape and dtype as 'spots' The 
        discounting factor; discount_rates = -log(discount factor) * expiries
    is_call_options : A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    dtype: supplied dtype but converted to tf.float64
    name: name of the function

    Returns

    option_prices: A `Tensor` of the same shape as `spots`. The Bachelier
    price of the options.
    


    """   
    with tf.compat.v1.name_scope(
      name,
      default_name='bachelier_option_price',
      values=[
          spots, strikes, volatilities, expiries, discount_rates,
          discount_factors, is_call_options
      ]):            
      
        spots = tf.convert_to_tensor(spots, dtype=tf.float64, name='forwards')
        strikes = tf.convert_to_tensor(strikes, dtype=tf.float64, name='strikes')
        volatilities = tf.convert_to_tensor(volatilities, tf.float64, name='volatilities')
        expiries = tf.convert_to_tensor(expiries, tf.float64, name='expiries')
        if (discount_rates != None and discount_factors != None):
            raise ValueError('Either discount rates or discount factors have to be used.')
        
        if (discount_rates != None and discount_factors == None):
            rates = tf.convert_to_tensor(discount_rates, tf.float64, name='rates')
            df = tf.math.exp(-rates * expiries)
        elif (discount_factors != None and discount_rates == None):
            rates = -tf.math.log(tf.convert_to_tensor(discount_rates, tf.float64, name='rates')) / expiries
            df = discount_factors
        else:
            rates = 0.0
            df = tf.convert_to_tensor(rates, dtype=tf.float64, name='discount_rates')   
            
           
        z = tf.zeros_like(strikes)
        
        #normal = tfp.distributions.Normal(
        #    loc=tf.zeros([], dtype=spots.dtype), scale=1)
             
        vt = volatilities * tf.math.sqrt(expiries)
        
        z = tf.where(rates == 0., (spots - strikes)/vt, 
                     (spots - strikes * df) / (volatilities 
                      * tf.math.sqrt(0.5 * (1.-tf.math.exp(-2. * rates*expiries)) / rates)))
                 
        n1 = _ncdf(z)
        n2 = _npdf(z)
        calls = tf.where(rates==0., (spots - strikes) * n1 + vt * n2,
                         (spots - strikes * df) * n1 
                         + volatilities * tf.math.sqrt(0.5 * (1 - tf.math.exp(-2 * rates * expiries)) / rates))
                         
            
        if is_call_options is None:
            return calls
        
        puts = calls - spots + strikes * tf.math.exp(-rates * expiries)
        
        return tf.where(is_call_options, calls, puts)
    
    
    
def dawson_option_price(forwards, 
                 strikes, 
                 volatilities, 
                 expiries,
                 discount_rates = None,
                 discount_factors = None,
                 is_call_options=None,
                 dtype = None,
                 name = None):

    """Computes the Bachelier price for a batch of European options.
       We assume a standard Brownian motion of the form
           dS = r dt + sigma dW
       for the underlying
       
       ## References:
       [1] Dawson, P., Blake, D., Cairns, A. J. G. and Dowd, K.: Options on normal under-
            lyings, CRIS Discussion Paper Series – 2007.VII, 2007.
            
       #### Examples
            spots = np.array([0.03, 0.02])
            strikes = np.array([.02, .02])
            volatilities = np.array([.004, .005])
            expiries = 2.0
            expiries = 1.0
            computed_prices = dawson_option_price(
                forwards,
                strikes,
                volatilities,
                expiries,
                dtype=tf.float64)
        # Expected print output of computed prices:
        # <tf.Tensor: id=90474, shape=(2,), dtype=float64, numpy=array([0.01008754, 0.00199471])>
        
        Args:
            
        forwards: A real `Tensor` of any shape. The current forward prices to
        expiry.
        strikes: A real `Tensor` of the same shape and dtype as `forwards`. The
            strikes of the options to be priced.
        volatilities: A real `Tensor` of same shape and dtype as `forwards`. The
            volatility to expiry.
        expiries : A real `Tensor` of same shape and dtype as `spots`. 
        discount_rates : rates from which discount factor via 
            discount factor = exp(-discount rate * T) are calculated
        discount_factors : A real 'Tensor' of same shape and dtype as 'spots' The 
        discounting factor; discount_rates = -log(discount factor) * expiries
        is_call_options : A boolean `Tensor` of a shape compatible with
            `volatilities`. Indicates whether the option is a call (if True) or a put
            (if False). If not supplied, call options are assumed.
        dtype: supplied dtype but converted to tf.float64
        name: name of the function

        Returns:
            option_prices: A `Tensor` of the same shape as `forwards`. The Bachelier
            price of the options.


         
    """  
    with tf.compat.v1.name_scope(
      name,
      default_name='dawson_option_price',
      values=[
          forwards, strikes, volatilities, expiries, discount_factors,
          discount_rates, is_call_options
      ]):
   
        forwards = tf.convert_to_tensor(forwards, dtype=tf.float64, name='forwards')
        strikes = tf.convert_to_tensor(strikes, dtype=tf.float64, name='strikes')
        volatilities = tf.convert_to_tensor(volatilities, dtype=tf.float64, name='volatilities')
        expiries = tf.convert_to_tensor(expiries, dtype=tf.float64, name='expiries')
        
        # check if discount rates or discount factor version is used
        if (discount_rates != None and discount_factors != None):
             raise ValueError('Either discount rates or discount factors have to be used.')
        
        if (discount_rates != None and discount_factors == None):
             discount_factors = tf.math.exp(-tf.convert_to_tensor(discount_rates, tf.float64, name='discount factors')*expiries)
        else:
            if (discount_factors == None and discount_rates == None):
                discount_factors = 1.0
            discount_factors = tf.convert_to_tensor(discount_factors, dtype=tf.float64, name='discount_factors')   
              
        vt = volatilities * tf.math.sqrt(expiries)
        
        z = (forwards - strikes) / vt
        
        # calculate constituents of Bachelier formula         
        n1 = _ncdf(z)
        n2 = _npdf(z)
        undiscounted_calls = (forwards - strikes) * n1 + vt * n2   # Bachelier option price
        
        # check if calls or puts need to be considered                 
        if is_call_options is None:
            return discount_factors * undiscounted_calls
        undiscounted_forward = forwards - strikes
        undiscounted_puts = undiscounted_calls - undiscounted_forward
        
        # return call, resp. put prices
        return discount_factors * tf.where(is_call_options, undiscounted_calls,
                                       undiscounted_puts)
    
    
