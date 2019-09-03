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
"""Black Scholes prices of a batch of European options."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


def option_price(forwards,
                 strikes,
                 volatilities,
                 expiries,
                 discount_factors=None,
                 is_call_options=None,
                 dtype=None,
                 name=None):
  """Computes the Black Scholes price for a batch of European options.

  ## References:
  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
  [2] Wikipedia contributors. Black-Scholes model. Available at:
    https://en.wikipedia.org/w/index.php?title=Black%E2%80%93Scholes_model

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
    option_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the options.


  #### Examples
  ```python
  forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
  volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
  expiries = 1.0
  computed_prices = option_price(
      forwards,
      strikes,
      volatilities,
      expiries,
      dtype=tf.float64)
  # Expected print output of computed prices:
  # [ 0.          2.          2.04806848  1.00020297  2.07303131]
  ```
  """
  with tf.compat.v1.name_scope(
      name,
      default_name='option_price',
      values=[
          forwards, strikes, volatilities, expiries, discount_factors,
          is_call_options
      ]):
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    if discount_factors is None:
      discount_factors = 1
    discount_factors = tf.convert_to_tensor(
        discount_factors, dtype=dtype, name='discount_factors')
    normal = tfp.distributions.Normal(
        loc=tf.zeros([], dtype=forwards.dtype), scale=1)
    sqrt_var = volatilities * tf.math.sqrt(expiries)
    d1 = (tf.math.log(forwards / strikes) + sqrt_var * sqrt_var / 2) / sqrt_var
    d2 = d1 - sqrt_var
    undiscounted_calls = forwards * normal.cdf(d1) - strikes * normal.cdf(d2)
    if is_call_options is None:
      return discount_factors * undiscounted_calls
    undiscounted_forward = forwards - strikes
    undiscounted_puts = undiscounted_calls - undiscounted_forward
    return discount_factors * tf.where(is_call_options, undiscounted_calls,
                                       undiscounted_puts)


def binary_price(forwards,
                 strikes,
                 volatilities,
                 expiries,
                 discount_factors=None,
                 is_call_options=None,
                 dtype=None,
                 name=None):
  """Computes the Black Scholes price for a batch of European binary options.

  The binary call (resp. put) option priced here is that which pays off a unit
  of cash if the underlying asset has a value greater (resp. smaller) than the
  strike price at expiry. Hence the binary option price is the discounted
  probability that the asset will end up higher (resp. lower) than the
  strike price at expiry.

  ## References:
  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
  [2] Wikipedia contributors. Binary option. Available at:
  https://en.wikipedia.org/w/index.php?title=Binary_option

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
      Default value: None which is mapped to the default name `binary_price`.

  Returns:
    option_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the binary options with unit of cash payoff.

  #### Examples
  ```python
  forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
  volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
  expiries = 1.0
  prices = binary_price(forwards, strikes, volatilities, expiries,
                               dtype=tf.float64)
  # Expected print output of prices:
  # [0.         0.         0.15865525 0.99764937 0.85927418]
  ```
  """
  with tf.compat.v1.name_scope(
      name,
      default_name='binary_price',
      values=[
          forwards, strikes, volatilities, expiries, discount_factors,
          is_call_options
      ]):
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    if is_call_options is None:
      is_call_options = True
    if discount_factors is None:
      discount_factors = 1
    discount_factors = tf.convert_to_tensor(
        discount_factors, dtype=dtype, name='discount_factors')
    sqrt_var = volatilities * tf.math.sqrt(expiries)
    d2 = (tf.math.log(forwards / strikes) - sqrt_var * sqrt_var / 2) / sqrt_var
    one = tf.ones_like(forwards)
    d2_signs = tf.where(is_call_options, one, -one)
    normal = tfp.distributions.Normal(
        loc=tf.zeros([], dtype=forwards.dtype), scale=1)
    return discount_factors * normal.cdf(d2_signs * d2)
