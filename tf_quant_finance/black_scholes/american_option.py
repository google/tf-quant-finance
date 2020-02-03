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
  # TODO: ask: if b=r, should use European therefore no point in allowing it to
  # be not given, yes? then again, have to build that part anyway so may as well
  # allow for it

  # Check to see if not any b < r ---> return European options
  return tf.compat.v1.cond(
      # Is this needed, or do is a simple if statement enough?
      tf.reduce_all(cost_of_carries >= risk_free_rates),
      lambda: tff.black_scholes.vanilla_prices.option_price(
          volatilities, strikes, expiries, spots=spots, forwards=forwards,
          risk_free_rates=risk_free_rates, cost_of_carries=cost_of_carries,
          is_call_options=is_call_options, dtype=dtype, name=name),
      lambda: _option_price(
          volatilities, strikes, expiries, risk_free_rates, cost_of_carries,
          spots=spots, forwards=forwards, is_call_options=is_call_options,
          dtype=dtype, name=name))


def _option_price(volatilities,
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

    The naming convention will align variables with the variables named in
    reference [1], but made lower case, and differentiating between put and
    call option values with the suffix _put and _call.

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

  #TODO: ask:  _ncdf better to duplicate or move to like math & use from there
  with tf.compat.v1.name_scope(
      name,
      default_name='option_price',
      values=[
        spots, strikes, volatilities, expiries, risk_free_rates,
        cost_of_carries, is_call_options
      ]):
    where_call = tf.constant(
      True) if is_call_options is None else is_call_options
    call_indices = tf.compat.v1.where_v2(where_call)
    put_indices = tf.compat.v1.where_v2(tf.logical_not(where_call))

    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name="strikes")
    volatilities = tf.convert_to_tensor(volatilities, dtype=dtype, name="volatilities")
    if forwards is not None:
      spots = tf.convert_to_tensor(
        forwards * tf.exp(- cost_of_carries * expiries),
        dtype=dtype,
        name="spots")
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name="spots")

    if is_call_options is None:
      x_call = strikes
      sigma_call = volatilities
      s_call = spots
      t_call = expiries
      r_call = risk_free_rates
      b_call = cost_of_carries

      x_put, sigma_put, s_put, t_put, r_put, b_put = None, None, None, None, \
                                                     None, None

    else:
      x_call = tf.gather_nd(strikes, call_indices)
      sigma_call = tf.gather_nd(volatilities, call_indices)
      s_call = tf.gather_nd(spots, call_indices)
      t_call = tf.gather_nd(expiries, call_indices)
      r_call = tf.gather_nd(risk_free_rates, call_indices)
      b_call = tf.gather_nd(cost_of_carries, call_indices)

      x_put = tf.gather_nd(strikes, put_indices)
      sigma_put = tf.gather_nd(volatilities, put_indices)
      s_put = tf.gather_nd(spots, put_indices)
      t_put = tf.gather_nd(expiries, put_indices)
      r_put = tf.gather_nd(risk_free_rates, put_indices)
      b_put = tf.gather_nd(cost_of_carries, put_indices)

    q2, a2, s_crit_call = _option_price_put_or_call(
      sigma_call, x_call, t_call, r_call, b_call, s_call, True, dtype)

    # Calculate eu_call_prices only for spot prices that are less than s_crit
    is_less_than_s_crit = s_call < s_crit_call
    shape = tf.shape(s_call)
    indices = tf.compat.v1.where_v2(is_less_than_s_crit)
    s_crit_s_call = tf.gather_nd(s_call, indices)
    s_crit_t_call = tf.gather_nd(t_call, indices)
    s_crit_sigma_call = tf.gather_nd(sigma_call, indices)
    s_crit_x_call = tf.gather_nd(x_call, indices)
    s_crit_r_call = tf.gather_nd(r_call, indices)
    s_crit_b_call = tf.gather_nd(b_call, indices)

    eu_call_prices = tff.black_scholes.vanilla_prices.option_price(
      volatilities=s_crit_sigma_call, strikes=s_crit_x_call,
      expiries=s_crit_t_call, spots=s_crit_s_call,
      risk_free_rates=s_crit_r_call, cost_of_carries=s_crit_b_call,
      dtype=dtype, name=name)
    eu_call_prices_or_zero = tf.scatter_nd(indices, eu_call_prices, shape)

    american_call_prices = tf.compat.v1.where_v2(
      is_less_than_s_crit,
      eu_call_prices_or_zero + a2 * (s_call / s_crit_call) ** q2,
      s_call - x_call)

    if is_call_options is None:
      return american_call_prices

    # Put option
    q1, a1, s_crit_put = _option_price_put_or_call(
      sigma_put, x_put, t_put, r_put, b_put, s_put, False, dtype)

    is_more_than_s_crit = s_put > s_crit_put
    shape = tf.shape(s_put)
    indices = tf.compat.v1.where_v2(is_more_than_s_crit)
    s_crit_s_put = tf.gather_nd(s_put, indices)
    s_crit_t_put = tf.gather_nd(t_put, indices)
    s_crit_sigma_put = tf.gather_nd(sigma_put, indices)
    s_crit_x_put = tf.gather_nd(x_put, indices)
    s_crit_r_put = tf.gather_nd(r_put, indices)
    s_crit_b_put = tf.gather_nd(b_put, indices)

    eu_put_prices = tff.black_scholes.vanilla_prices.option_price(
      volatilities=s_crit_sigma_put, strikes=s_crit_x_put,
      expiries=s_crit_t_put, spots=s_crit_s_put,
      risk_free_rates=s_crit_r_put, cost_of_carries=s_crit_b_put,
      is_call_options=tf.constant(False), dtype=dtype, name=name)
    eu_put_prices_or_zero = tf.scatter_nd(indices, eu_put_prices, shape)

    american_put_prices = tf.compat.v1.where_v2(
      is_more_than_s_crit,
      eu_put_prices_or_zero + a1 * (s_put / s_crit_put) ** q1,
      x_put - s_put)

    shape = tf.shape(spots)

    call_or_zero = tf.scatter_nd(call_indices, american_call_prices, shape)
    put_or_zero = tf.scatter_nd(put_indices, american_put_prices, shape)
    return tf.where_v2(is_call_options, call_or_zero, put_or_zero)


def _option_price_put_or_call(sigma, x, t, r, b, s, call, dtype):
  is_call_options = None if call else tf.constant(False)
  sign = 1 if call else -1

  M = 2 * r / sigma ** 2
  N = 2 * b / sigma ** 2
  K = 1 - tf.exp(- r * t)
  q_ = [(1 - N) + sign * tf.sqrt((N - 1) ** 2 + 4 * M / K)] / 2

  # Below is estimate starting point for S*; in future will calculate
  s_crit = tf.identity(s, name="S*" if call else "S**")
  LHS = s_crit - x
  RHS = tff.black_scholes.vanilla_prices.option_price(
    volatilities=sigma, strikes=x, expiries=t, spots=s_crit,
    risk_free_rates=r, cost_of_carries=b,
    is_call_options=is_call_options, dtype=dtype) + sign * (
      1 - tf.exp((b - r) * t) * _ncdf(
    sign * _calc_d1(s_crit, x, sigma, b, t))) * s_crit / q_

  s_crit, LHS, RHS = tf.compat.v1.while_loop(
    loop_vars=[s_crit, LHS, RHS],
    cond=lambda s, LHS, RHS: (tf.abs(LHS - RHS) / x) < 0.00001,
    body=lambda s, LHS, RHS: _update_s_crit(
      s, q_, sigma, x, t, r, b, is_call_options, dtype))

  a_ = (sign * s_crit / q_) * (1 - tf.exp((b - r) * t)) * _ncdf(
    sign * _calc_d1(s_crit, x, sigma, b, t))

  return q_, a_, s_crit


def _calc_d1(s, x, sigma, b, t):
  return (tf.math.log(s / x) + (b + sigma ** 2 / 2) * t) / (
      sigma * tf.math.sqrt(t))


def _update_s_crit(s_crit, q, sigma, x, t, r, b, is_call_options, dtype):
  sign = 1 if is_call_options is None else -1
  LHS = s_crit - x
  d1 = _calc_d1(s_crit, x, sigma, b, t)
  RHS = tff.black_scholes.vanilla_prices.option_price(
    volatilities=sigma, strikes=x, expiries=t, spots=s_crit, risk_free_rates=r,
    cost_of_carries=b, is_call_options=is_call_options, dtype=dtype) + sign * (
            1 - tf.exp((b - r) * t) * _ncdf(sign * d1)) * s_crit / q

  # TODO change bi below for put option
  bi = tf.exp((b - r) * t) * _ncdf(d1) * (1 - 1 / q) + (
      1 - tf.exp((b - r) * t) * _npdf(d1) / (sigma * tf.sqrt(t))) / q

  s_crit = (x + RHS - bi * s_crit) / (1 - bi)
  return s_crit, LHS, RHS


def _ncdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


def _npdf(x):
  return tf.exp(-0.5 * x ** 2) / tf.math.sqrt(2 * np.pi)


_SQRT_2 = 1.4142135623730951
