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

import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
import tf_quant_finance.black_scholes.implied_vol_newton_root as implied_vol_newton


def adesi_whaley(volatilities,
                 strikes,
                 expiries,
                 risk_free_rates,
                 cost_of_carries,
                 spots=None,
                 forwards=None,
                 is_call_options=None,
                 dtype=None,
                 name="american_price"):
  """Computes the price for a batch of call or put options, using an approximate
  pricing formula, the Baron-Adesi Whaley approximation

  #### Example

  ```python
  spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
  strikes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
  volatilities = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
  expiries = 0.25
  cost_of_carries = -0.04
  risk_free_rates = 0.08
  computed_prices = adesi_whaley(
      volatilities,
      strikes,
      expiries,
      risk_free_rates,
      cost_of_carries,
      spots=spots,
      dtype=tf.float64)
  # Expected print output of computed prices:
  # [0.03, 0.59, 3.52, 10.31, 20.0]
  ```
  The naming convention will align variables with the variables named in
  reference [1], but made lower case, and differentiating between put and
  call option values with the suffix _put and _call.

  ## References:
  [1] Baron-Adesi, Whaley, Efficient Analytic Approximation of American Option
    Values, The Journal of Finance, Vol XLII, No. 2, June 1987
    https://deriscope.com/docs/Barone_Adesi_Whaley_1987.pdf

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

  # if b>=r, return European only, if b<r, return American only, if mixed, split
  return tf.case(
    [(tf.reduce_all(cost_of_carries >= risk_free_rates),
      lambda: tff.black_scholes.option_price(
        volatilities, strikes, expiries, spots=spots, forwards=forwards,
        risk_free_rates=risk_free_rates, cost_of_carries=cost_of_carries,
        is_call_options=is_call_options, dtype=dtype, name=name)),
     (tf.reduce_all(cost_of_carries < risk_free_rates),
        lambda: _option_price(*_convert_to_tensors(
          volatilities, strikes, expiries, risk_free_rates, cost_of_carries,
          spots=spots, forwards=forwards, is_call_options=is_call_options,
          dtype=dtype, name=name)))],
    default=lambda: _split_out(
          volatilities, strikes, expiries, spots=spots, forwards=forwards,
          risk_free_rates=risk_free_rates, cost_of_carries=cost_of_carries,
          is_call_options=is_call_options, dtype=dtype, name=name),
    exclusive=True)


def _split_out(volatilities,
               strikes,
               expiries,
               risk_free_rates,
               cost_of_carries,
               spots=None,
               forwards=None,
               is_call_options=None,
               dtype=None,
               name="american_price"):
  """
  Splits prices out to European and American batches, then combines the result.
  """

  if forwards is not None:
    t = tf.broadcast_to(tf.convert_to_tensor(
      expiries, dtype=dtype), forwards.shape, name="expiries")
    s = tf.convert_to_tensor(
      forwards * tf.exp(- cost_of_carries * t),
      dtype=dtype,
      name="spots")
  else:
    t = tf.broadcast_to(tf.convert_to_tensor(
      expiries, dtype=dtype), spots.shape, name="expiries")
    s = tf.convert_to_tensor(spots, dtype=dtype, name="spots")

  b_less_r = risk_free_rates > cost_of_carries
  b_more_eq_r = risk_free_rates <= cost_of_carries
  am_indices = tf.where(b_less_r)
  eu_indices = tf.where(b_more_eq_r)
  shape = tf.shape(s, out_type=am_indices.dtype)

  strikes = tf.broadcast_to(tf.convert_to_tensor(
    strikes, dtype=dtype), shape, name="strikes")
  volatilities = tf.broadcast_to(tf.convert_to_tensor(
    volatilities, dtype=dtype), shape, name="volatilities")
  cost_of_carries = tf.broadcast_to(tf.convert_to_tensor(
    cost_of_carries, dtype=dtype), shape, name="cost_of_carries")
  risk_free_rates = tf.broadcast_to(tf.convert_to_tensor(
    risk_free_rates, dtype=dtype), shape, name="risk_free_rates")

  if is_call_options is not None:
    is_call_options = tf.broadcast_to(tf.convert_to_tensor(
      is_call_options), s.shape, name="risk_free_rates")
    is_call_eu = tf.gather_nd(is_call_options, eu_indices)
    is_call_am = tf.gather_nd(is_call_options, am_indices)
  else:
    is_call_eu, is_call_am = None, None

  x_eu = tf.gather_nd(strikes, eu_indices)
  sigma_eu = tf.gather_nd(volatilities, eu_indices)
  s_eu = tf.gather_nd(s, eu_indices)
  t_eu = tf.gather_nd(t, eu_indices)
  r_eu = tf.gather_nd(risk_free_rates, eu_indices)
  b_eu = tf.gather_nd(cost_of_carries, eu_indices)

  x_am = tf.gather_nd(strikes, am_indices)
  sigma_am = tf.gather_nd(volatilities, am_indices)
  s_am = tf.gather_nd(s, am_indices)
  t_am = tf.gather_nd(t, am_indices)
  r_am = tf.gather_nd(risk_free_rates, am_indices)
  b_am = tf.gather_nd(cost_of_carries, am_indices)

  eu_prices = tff.black_scholes.option_price(
    sigma_eu, x_eu, t_eu, spots=s_eu, risk_free_rates=r_eu,
    cost_of_carries=b_eu, is_call_options=is_call_eu, dtype=dtype, name=name)

  eu_prices_or_zero = tf.scatter_nd(eu_indices, eu_prices, shape)

  am_prices = _option_price(
      sigma_am, x_am, t_am, spots=s_am, risk_free_rates=r_am,
      cost_of_carries=b_am, is_call_options=is_call_am, dtype=dtype, name=name)

  return tf.where(b_more_eq_r, eu_prices_or_zero, am_prices)


def _convert_to_tensors(volatilities,
                        strikes,
                        expiries,
                        risk_free_rates,
                        cost_of_carries,
                        spots=None,
                        forwards=None,
                        is_call_options=None,
                        dtype=None,
                        name=None):
  """
  Converts to tensors the np.array inputs.
  """
  if forwards is not None:
    expiries = tf.broadcast_to(tf.convert_to_tensor(
      expiries, dtype=dtype), forwards.shape, name="expiries")
    spots = tf.convert_to_tensor(
      forwards * tf.exp(- cost_of_carries * expiries),
      dtype=dtype,
      name="spots")
  else:
    expiries = tf.broadcast_to(tf.convert_to_tensor(
      expiries, dtype=dtype), spots.shape, name="expiries")
    spots = tf.convert_to_tensor(spots, dtype=dtype, name="spots")

  strikes = tf.broadcast_to(tf.convert_to_tensor(
    strikes, dtype=dtype), spots.shape, name="strikes")
  volatilities = tf.broadcast_to(tf.convert_to_tensor(
    volatilities, dtype=dtype), spots.shape, name="volatilities")
  cost_of_carries = tf.broadcast_to(tf.convert_to_tensor(
    cost_of_carries, dtype=dtype), spots.shape, name="cost_of_carries")
  risk_free_rates = tf.broadcast_to(tf.convert_to_tensor(
    risk_free_rates, dtype=dtype), spots.shape, name="risk_free_rates")
  return (
    volatilities, strikes, expiries, risk_free_rates, cost_of_carries, spots,
    is_call_options, dtype, name)
  

def _option_price(volatilities,
                  strikes,
                  expiries,
                  risk_free_rates,
                  cost_of_carries,
                  spots=None,
                  is_call_options=None,
                  dtype=None,
                  name=None):
  """Computes the price for a batch of call or put options, using an approximate
    pricing formula, the Baron-Adesi Whaley approximation

    The naming convention will align variables with the variables named in
    reference [1], but made lower case, and differentiating between put and
    call option values with the suffix _put and _call.

    ## References:
    [1] Baron-Adesi, Whaley, Efficient Analytic Approximation of American Option
      Values, The Journal of Finance, Vol XLII, No. 2, June 1987;
      https://deriscope.com/docs/Barone_Adesi_Whaley_1987.pdf

    Returns:
      option_prices: A `Tensor` of the same shape as `forwards` or `spots`.
      The Baron-Adesi-Whaley approximation of the price of the american options.
    """

  with tf.name_scope(name):
    where_call = tf.broadcast_to(
      tf.constant(True), spots.shape) if is_call_options is None else is_call_options
    call_indices = tf.where(where_call)
    put_indices = tf.where(tf.logical_not(where_call))

    if is_call_options is None:
      sigma_call = volatilities
      x_call = strikes
      s_call = spots
      t_call = expiries
      r_call = risk_free_rates
      b_call = cost_of_carries

    else:
      x_call = tf.gather_nd(strikes, call_indices)
      sigma_call = tf.gather_nd(volatilities, call_indices)
      s_call = tf.gather_nd(spots, call_indices)
      t_call = tf.gather_nd(expiries, call_indices)
      r_call = tf.gather_nd(risk_free_rates, call_indices)
      b_call = tf.gather_nd(cost_of_carries, call_indices)

    q2, a2, s_crit_call = _option_price_put_or_call(
      sigma_call, x_call, t_call, r_call, b_call, s_call, True, dtype)

    # Calculate eu_call_prices only for spot prices that are less than s_crit
    is_less_than_s_crit = s_call < s_crit_call
    indices = tf.where(is_less_than_s_crit)
    shape = tf.shape(s_call, out_type=indices.dtype)
    s_crit_s_call = tf.gather_nd(s_call, indices)
    s_crit_t_call = tf.gather_nd(t_call, indices)
    s_crit_sigma_call = tf.gather_nd(sigma_call, indices)
    s_crit_x_call = tf.gather_nd(x_call, indices)
    s_crit_r_call = tf.gather_nd(r_call, indices)
    s_crit_b_call = tf.gather_nd(b_call, indices)

    eu_call_prices = tff.black_scholes.option_price(
      volatilities=s_crit_sigma_call, strikes=s_crit_x_call,
      expiries=s_crit_t_call, spots=s_crit_s_call,
      risk_free_rates=s_crit_r_call, cost_of_carries=s_crit_b_call,
      dtype=dtype, name=name)
    eu_call_prices_or_zero = tf.scatter_nd(indices, eu_call_prices, shape)

    american_call_prices = tf.where(
      is_less_than_s_crit,
      eu_call_prices_or_zero + a2 * (s_call / s_crit_call) ** q2,
      s_call - x_call)

    if is_call_options is None:
      return american_call_prices

    # Put option
    x_put = tf.gather_nd(strikes, put_indices)
    sigma_put = tf.gather_nd(volatilities, put_indices)
    s_put = tf.gather_nd(spots, put_indices)
    t_put = tf.gather_nd(expiries, put_indices)
    r_put = tf.gather_nd(risk_free_rates, put_indices)
    b_put = tf.gather_nd(cost_of_carries, put_indices)

    q1, a1, s_crit_put = _option_price_put_or_call(
      sigma_put, x_put, t_put, r_put, b_put, s_put, False, dtype)

    # Calculate eu_put_prices only for spot prices larger than s_crit
    is_more_than_s_crit = s_put > s_crit_put
    indices = tf.where(is_more_than_s_crit)
    shape = tf.shape(s_put, out_type=indices.dtype)
    s_crit_s_put = tf.gather_nd(s_put, indices)
    s_crit_t_put = tf.gather_nd(t_put, indices)
    s_crit_sigma_put = tf.gather_nd(sigma_put, indices)
    s_crit_x_put = tf.gather_nd(x_put, indices)
    s_crit_r_put = tf.gather_nd(r_put, indices)
    s_crit_b_put = tf.gather_nd(b_put, indices)

    eu_put_prices = tff.black_scholes.option_price(
      volatilities=s_crit_sigma_put, strikes=s_crit_x_put,
      expiries=s_crit_t_put, spots=s_crit_s_put,
      risk_free_rates=s_crit_r_put, cost_of_carries=s_crit_b_put,
      is_call_options=tf.constant(False), dtype=dtype, name=name)
    eu_put_prices_or_zero = tf.scatter_nd(indices, eu_put_prices, shape)

    american_put_prices = tf.where(
      is_more_than_s_crit,
      eu_put_prices_or_zero + a1 * (s_put / s_crit_put) ** q1,
      x_put - s_put)

    shape = tf.shape(spots, out_type=call_indices.dtype)

    call_or_zero = tf.scatter_nd(call_indices, american_call_prices, shape)
    put_or_zero = tf.scatter_nd(put_indices, american_put_prices, shape)
    return tf.where(is_call_options, call_or_zero, put_or_zero)


def _option_price_put_or_call(sigma, x, t, r, b, s, call, dtype):
  """ Calculates q, a for the american option price formula.
  Also calculates the critical spot price above and below which the american
  option price formula behaves differently.

  :param sigma:
  :param x:
  :param t:
  :param r:
  :param b:
  :param s:
  :param call:
  :param dtype:
  :return:
  """
  is_call_options = None if call else tf.constant(False)
  sign = 1 if call else -1

  M = 2 * r / sigma ** 2
  N = 2 * b / sigma ** 2
  K = 1 - tf.exp(- r * t)
  q_ = _calc_q(N, M, K, call)

  value = lambda s_crit: tff.black_scholes.option_price(
    volatilities=sigma, strikes=x, expiries=t, spots=s_crit, risk_free_rates=r,
    cost_of_carries=b, is_call_options=is_call_options, dtype=dtype) + sign * (
      1 - tf.exp((b - r) * t) * _ncdf(sign * _calc_d1(
    s_crit, x, sigma, b, t))) * s_crit / q_ - sign * (s_crit - x)

  value_and_gradient = lambda price: tff.math.value_and_gradient(value, price)

  # Calculate seed value for critical spot price
  q_inf = _calc_q(N, M, call=call)
  s_inf = x / (1 - 1 / q_inf)
  h = -(sign * b * t + 2 * sigma * tf.sqrt(t)) * sign * (x / (s_inf - x))
  if call:
    s_seed = x + (s_inf - x) * (1 - tf.exp(h))
  else:
    s_seed = s_inf + (x - s_inf) * tf.exp(h)

  s_crit, _, _ = implied_vol_newton.newton_root_finder(
    value_and_gradient, s_seed, dtype=dtype)

  a_ = (sign * s_crit / q_) * (1 - tf.exp((b - r) * t) * _ncdf(
    sign * _calc_d1(s_crit, x, sigma, b, t)))

  return q_, a_, s_crit


def _calc_d1(s, x, sigma, b, t):
  return (tf.math.log(s / x) + (b + sigma ** 2 / 2) * t) / (
      sigma * tf.math.sqrt(t))

def _calc_q(N, M, K=1, call=True):
  sign = 1 if call else -1
  return ((1 - N) + sign * tf.sqrt((N - 1) ** 2 + 4 * M / K)) / 2


def _ncdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


def _npdf(x):
  return tf.exp(-0.5 * x ** 2) / tf.math.sqrt(2 * np.pi)


_SQRT_2 = np.sqrt(2., dtype=np.float64)
