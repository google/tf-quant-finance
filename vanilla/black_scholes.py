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

"""A tool to compute the Black Scholes prices of a batch of European options."""

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
                 name=None):
  """Computes the Black Scholes price for a batch of European options.

  For more details see:
  [Wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model).

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
    name: The name for the ops created by this function.

  Returns:
    option_prices: A `Tensor` of the same shape as `forwards`. The Black Scholes
      price of the options.


  #### Examples
  ```python
  forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
  volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
  expiries = 1.0
  computed_prices = black_scholes.black_scholes_price(
      forwards,
      strikes,
      volatilities,
      expiries)
  with tf.Session() as sess:
    print(sess.run(computed_prices))
  # Expected output:
  # [ 0.          2.          2.04806848  1.00020297  2.07303131]
  ```
  """
  with tf.name_scope(
      name,
      default_name='black_scholes_price',
      values=[
          forwards, strikes, volatilities, expiries, discount_factors,
          is_call_options
      ]):
    if discount_factors is None:
      discount_factors = 1
    forwards = tf.convert_to_tensor(forwards)
    dtype = forwards.dtype
    strikes = tf.convert_to_tensor(strikes, dtype=dtype)
    volatilities = tf.convert_to_tensor(volatilities, dtype=dtype)
    expiries = tf.convert_to_tensor(expiries, dtype=dtype)
    normal = tfp.distributions.Normal(loc=tf.zeros([], dtype=dtype), scale=1)
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
