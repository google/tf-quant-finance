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
"""Baron-Adesi Whaley approximation of the Black Scholes equation to calculate
 the price of a batch of American options."""


import tensorflow as tf
import tf_quant_finance as tff


def option_price(volatilities,
                 strikes,
                 expiries,
                 risk_free_rates,
                 cost_of_carries,
                 spots=None,
                 forwards=None,
                 is_call_options=None,
                 dtype=None,
                 name=None):
  """Computes the price for a batch of call or put options, using an approximate
  pricing formula, the Baron-Adesi Whaley approximation

  #### Example

  ```python
  forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
  volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
  expiries = 1.0
  computed_prices = option_price(
      volatilities,
      strikes,
      expiries,
      forwards=forwards,
      dtype=tf.float64)
  # Expected print output of computed prices:
  # [ 0.          2.          2.04806848  1.00020297  2.07303131]
  ```

  ## References:
  [1] Baron-Adesi, Whaley, Efficient Analytic Approximation of American Option
    Values, The Journal of Finance, Vol XLII, No. 2, June 1987

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
  risk_free_rates: An real `Tensor` of same dtype and compatible shape as
    `volatilities`. Discount factors are calculated using it: e^(-rT), where r
    is the risk free rate.
  cost_of_carries: An real `Tensor` of same dtype and compatible shape as
    `volatilities`. If `spots` is supplied, used to calculate forwards from
    spots: F = e^(bT) * S.
  is_call_options: A boolean `Tensor` of a shape compatible with
    `volatilities`. Indicates whether the option is a call (if True) or a put
    (if False). If not supplied, call options are assumed.
  dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
    of any supplied non-`Tensor` arguments to `Tensor`.
    Default value: None which maps to the default dtype inferred by TensorFlow
      (float32).
  name: str. The name for the ops created by this function.
    Default value: None which is mapped to the default name `option_price`.

  Returns:
    option_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  #TODO: ask: if b=r, should use European therefore no point in allowing it to
  # be not given, yes? then again, have to build that part anyway so may as well
  # allow for it

  #TODO: ask:  _ncdf better to duplicate or move to like math & use from there

  with tf.compat.v1.name_scope(
      name,
      default_name='option_price',
      values=[
          forwards, strikes, volatilities, expiries, risk_free_rates,
          cost_of_carries, is_call_options
      ]):
    
    # Check to see if not any b < r ---> return European options
    # Evaluate q2 for call
    # Evaluate S* for call
    # - Evaluate S* seed for call
    # - Evaluate S* for call
    # Evaluate A2
    # Evaluate european (S, T)
    # call_adjustment = where S < S*: european (S, T) + A2 * (S/S*) ** q2,
    # where S >= S*: S - X
    # is_call_options is None: ----> Evaluate european (S, T) with True
    #   for is_call_options, return price
    # Evaluate q1 for put
    # Evaluate S** for put
    # - Evaluate S** seed for put
    # - Evaluate S** for put
    # Evaluate A1
    # price_put = where S > S**: european (S, T) + A1 * (S/S**) ** q1,
    # where S <= S**: X - X
    # return where is_call_options(price_call, price_put)
    #TODO: can caluclate S* and S** fewer times if underlying is the same!
    #   (scale by X)
    # Check if can caluclate q1, q2, A1, A2 more efficiently together for
    #   put & call
    # Check if logical where's & tensorflow smart enough to minimise calculation
    #   when e.g. all put/ only calculate put field for fields where
    #   is_call_options is false

    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    risk_free_rates = tf.convert_to_tensor(
          risk_free_rates, dtype=dtype, name='risk_free_rates')
    cost_of_carries = tf.convert_to_tensor(
          cost_of_carries, dtype=dtype, name='cost_of_carries')
    discount_factors = tf.exp(-risk_free_rates * expiries)

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      forwards = spots * tf.exp(cost_of_carries * expiries)

    sqrt_var = volatilities * tf.math.sqrt(expiries)
    d1 = (tf.math.log(forwards / strikes) + cost_of_carries * expiries + sqrt_var * sqrt_var / 2) / sqrt_var
    d2 = d1 - sqrt_var
    undiscounted_calls = forwards * _ncdf(d1) - strikes * _ncdf(d2)
    if is_call_options is None:
      return discount_factors * undiscounted_calls
    undiscounted_forward = forwards - strikes
    undiscounted_puts = undiscounted_calls - undiscounted_forward
    predicate = tf.broadcast_to(is_call_options, tf.shape(undiscounted_calls))
    return discount_factors * tf.where(predicate, undiscounted_calls,
                                       undiscounted_puts)


def _ncdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


_SQRT_2 = 1.4142135623730951
