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

"""Local Volatility Model."""

import functools

import tensorflow.compat.v2 as tf

from tf_quant_finance import black_scholes
from tf_quant_finance import datetime
from tf_quant_finance import math
from tf_quant_finance.experimental.pricing_platform.framework.market_data import volatility_surface
from tf_quant_finance.models import generic_ito_process

interpolation_2d = math.interpolation.interpolation_2d


def _dupire_local_volatility(time, spot_price, initial_spot_price,
                             implied_volatility_surface, discount_factor_fn,
                             dividend_yield):
  """Constructs local volatility function using Dupire's formula.

  Args:
    time: A real `Tensor` of shape compatible with `spot_price` specifying the
      times at which local volatility function is computed.
    spot_price: A real `Tensor` specifying the underlying price at which local
      volatility function is computed.
    initial_spot_price: A real `Tensor` of shape compatible with `spot_price`
      specifying the underlying spot price at t=0.
    implied_volatility_surface: A Python callable which implements the
      interpolation of market implied volatilities. The callable should have the
      interface `implied_volatility_surface(strike, expiry_times)` which takes
      real `Tensor`s corresponding to option strikes and time to expiry and
      returns a real `Tensor` containing the corresponding market implied
      volatility. The shape of `strike` is `(n,dim)` where `dim` is the
      dimensionality of the local volatility process and `t` is a scalar tensor.
      The output from the callable is a `Tensor` of shape `(n,dim)` containing
      the interpolated implied volatilties.
    discount_factor_fn: A python callable accepting one real `Tensor` argument
      time t. It should return a `Tensor` specifying the discount factor to time
      t.
    dividend_yield: A real `Tensor` of shape compatible with `spot_price`
      specifying the (continuously compounded) dividend yield.

  Returns:
    A real `Tensor` of same shape as `spot_price` containing the local
    volatility computed at `(spot_price,time)` using the Dupire's
    construction of local volatility.
  """
  dtype = time.dtype

  risk_free_rate_fn = _get_risk_free_rate_from_discount_factor(
      discount_factor_fn)
  risk_free_rate = tf.convert_to_tensor(risk_free_rate_fn(time), dtype=dtype)

  def _option_price(expiry_time, strike):
    discount_factors = tf.convert_to_tensor(
        discount_factor_fn(expiry_time), dtype=dtype)
    vols = implied_volatility_surface(strike=strike, expiry_times=expiry_time)
    c_k_t = black_scholes.option_price(
        volatilities=vols,
        strikes=strike,
        expiries=expiry_time,
        spots=initial_spot_price,
        continuous_dividends=dividend_yield,
        discount_factors=discount_factors,
        dtype=dtype)
    return c_k_t

  dcdk_fn = lambda x: _option_price(time, x)
  dcdt_fn = lambda x: _option_price(x, spot_price)
  d2cdk2_fn = lambda x: math.fwd_gradient(dcdk_fn, x)

  # TODO(b/173568116): Replace gradients of call prices with imp vol gradients.
  numerator = (
      math.fwd_gradient(dcdt_fn, time) + (risk_free_rate - dividend_yield) *
      spot_price * math.fwd_gradient(dcdk_fn, spot_price) +
      dividend_yield * _option_price(time, spot_price))
  denominator = math.fwd_gradient(d2cdk2_fn, spot_price) * spot_price**2
  # we use relu for safety so that we do not take the square root of
  # negative real `Tensors`.
  local_volatility_squared = tf.nn.relu(
      2 * tf.math.divide_no_nan(numerator, denominator))
  return tf.math.sqrt(local_volatility_squared)


class LocalVolatilityModel(generic_ito_process.GenericItoProcess):
  r"""Local volatility model for smile modeling.

  Local volatility (LV) model specifies that the dynamics of an asset is
  governed by the following stochastic differential equation:

  ```None
    dS(t) / S(t) =  mu(t, S(t)) dt + sigma(t, S(t)) * dW(t)
  ```
  where `mu(t, S(t))` is the drift and `sigma(t, S(t))` is the instantaneous
  volatility. The local volatility function `sigma(t, S(t))` is state dependent
  and is computed by calibrating against a given implied volatility surface
  `sigma_iv(T, K)` using the Dupire's formula [1]:

  ```
  sigma(T,K)^2 = 2 * (dC(T,K)/dT + (r - q)K dC(T,K)/dK + qC(T,K)) /
                     (K^2 d2C(T,K)/dK2)
  ```
  where the derivatives above are the partial derivatives. The LV model provides
  a flexible framework to model any (arbitrage free) volatility surface.

  #### Example: Simulation of local volatility process.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64
  dim = 2
  year = dim * [[2021, 2022]]
  month = dim * [[1, 1]]
  day = dim * [[1, 1]]
  expiries = tff.datetime.dates_from_year_month_day(year, month, day)
  valuation_date = [(2020, 1, 1)]
  expiry_times = tff.datetime.daycount_actual_365_fixed(
      start_date=valuation_date, end_date=expiries, dtype=dtype)
  strikes = dim * [[[0.1, 0.9, 1.0, 1.1, 3], [0.1, 0.9, 1.0, 1.1, 3]]]
  iv = dim * [[[0.135, 0.13, 0.1, 0.11, 0.13],
               [0.135, 0.13, 0.1, 0.11, 0.13]]]
  spot = dim * [1.0]
  risk_free_rate = [0.02]
  r = tf.convert_to_tensor(risk_free_rate, dtype=dtype)
  df = lambda t: tf.math.exp(-r * t)

  lv = tff.models.LocalVolatilityModel.from_market_data(
      dim, val_date, expiries, strikes, iv, spot, df, [0.0], dtype=dtype)
  num_samples = 10000
  paths = lv.sample_paths(
      [1.0, 1.5, 2.0],
      num_samples=num_samples,
      initial_state=spot,
      time_step=0.1,
      random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
      seed=[1, 2])
  # paths.shape = (10000, 3, 2)

  #### References:
    [1]: Iain J. Clark. Foreign exchange option pricing - A Practitioner's
    guide. Chapter 5, Section 5.3.2. 2011.
  """

  def __init__(self,
               dim,
               risk_free_rate=None,
               dividend_yield=None,
               local_volatility_fn=None,
               corr_matrix=None,
               dtype=None,
               name=None):
    """Initializes the Local volatility model.

    Args:
      dim: A Python scalar which corresponds to the number of underlying assets
        comprising the model.
      risk_free_rate: One of the following: (a) An optional real `Tensor` of
        shape compatible with `[dim]` specifying the (continuously compounded)
        risk free interest rate. (b) A python callable accepting one real
        `Tensor` argument time t returning a `Tensor` of shape compatible with
        `[dim]`. If the underlying is an FX rate, then use this input to specify
        the domestic interest rate.
        Default value: `None` in which case the input is set to Zero.
      dividend_yield: A real `Tensor` of shape compatible with `spot_price`
        specifying the (continuosly compounded) dividend yield.
        If the underlying is an FX rate, then use this input to specify the
        foreign interest rate.
        Default value: `None` in which case the input is set to Zero.
      local_volatility_fn: A Python callable which returns instantaneous
        volatility as a function of state and time. The function must accept a
        scalar `Tensor` corresponding to time 't' and a real `Tensor` of shape
        `[num_samples, dim]` corresponding to the underlying price (S) as inputs
        and return a real `Tensor` of shape `[num_samples, dim]` containing the
        local volatility computed at (S,t).
      corr_matrix: A `Tensor` of shape `[dim, dim]` and the same `dtype` as
        `risk_free_rate`. Corresponds to the instantaneous correlation between
        the underlying assets.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
          `local_volatility_model`.
    """
    self._name = name or "local_volatility_model"
    self._local_volatility_fn = local_volatility_fn

    with tf.name_scope(self._name):
      self._dtype = dtype
      risk_free_rate = risk_free_rate or [0.0]
      dividend_yield = dividend_yield or [0.0]

      self._domestic_rate = _convert_to_tensor_fn(risk_free_rate, dtype,
                                                  "risk_free_rate")
      self._foreign_rate = _convert_to_tensor_fn(dividend_yield, dtype,
                                                 "dividend_yield")

      corr_matrix = corr_matrix or tf.eye(dim, dim, dtype=self._dtype)
      self._rho = tf.convert_to_tensor(
          corr_matrix, dtype=self._dtype, name="rho")
      self._sqrt_rho = tf.linalg.cholesky(self._rho)

      # TODO(b/173286140): Simulate using X(t)=log(S(t))
      def _vol_fn(t, state):
        """Volatility function of LV model."""
        lv = self._local_volatility_fn(t, state)
        diffusion = tf.expand_dims(state * lv, axis=-1)
        return diffusion * self._sqrt_rho

      # Drift function
      def _drift_fn(t, state):
        """Drift function of LV model."""
        domestic_rate = self._domestic_rate(t)

        foreign_rate = self._foreign_rate(t)
        return (domestic_rate - foreign_rate) * state

      super(LocalVolatilityModel, self).__init__(
          dim, _drift_fn, _vol_fn, dtype, name)

  def local_volatility_fn(self):
    """Local volatility function."""
    return self._local_volatility_fn

  @classmethod
  def from_market_data(cls,
                       dim,
                       valuation_date,
                       expiry_dates,
                       strikes,
                       implied_volatilities,
                       spot,
                       discount_factor_fn,
                       dividend_yield=None,
                       dtype=None,
                       name=None):
    """Creates a `LocalVolatilityModel` from market data.

    Args:
      dim: A Python scalar which corresponds to the number of underlying assets
        comprising the model.
      valuation_date: A `DateTensor` specifying the valuation (or settlement)
        date for the market data.
      expiry_dates: A `DateTensor` of shape `(dim, num_expiries)` containing the
        expiry dates on which the implied volatilities are specified.
      strikes: A `Tensor` of real dtype and shape `(dim, num_expiries,
        num_strikes)`specifying the strike prices at which implied volatilities
        are specified.
      implied_volatilities: A `Tensor` of real dtype and shape `(dim,
        num_expiries, num_strikes)` specifying the implied volatilities.
      spot: A real `Tensor` of shape `(dim,)` specifying the underlying spot
        price on the valuation date.
      discount_factor_fn: A python callable accepting one real `Tensor` argument
        time t. It should return a `Tensor` specifying the discount factor to
        time t.
      dividend_yield: A real `Tensor` of shape compatible with `spot_price`
        specifying the (continuosly compounded) dividend yield. If the
        underlying is an FX rate, then use this input to specify the foreign
        interest rate.
        Default value: `None` in which case the input is set to Zero.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name `from_market_data`.

    Returns:
      An instance of `LocalVolatilityModel` constructed using the input data.
    """
    name = name or "from_market_data"
    with tf.name_scope(name):
      spot = tf.convert_to_tensor(spot, dtype=dtype)
      dtype = dtype or spot.dtype
      dividend_yield = dividend_yield or [0.0]
      dividend_yield = tf.convert_to_tensor(dividend_yield, dtype=dtype)

      risk_free_rate_fn = _get_risk_free_rate_from_discount_factor(
          discount_factor_fn)

      valuation_date = datetime.convert_to_date_tensor(valuation_date)
      expiry_dates = datetime.convert_to_date_tensor(expiry_dates)
      expiry_times = (
          tf.cast(valuation_date.days_until(expiry_dates), dtype=dtype) / 365.0)
      strikes = tf.convert_to_tensor(strikes, dtype=dtype)
      implied_volatilities = tf.convert_to_tensor(
          implied_volatilities, dtype=dtype)

      def _log_forward_moneyness(times, strikes):
        # log_fwd_moneyness = log(strike/(spot*exp((r-d)*times)))
        risk_free_rate = tf.squeeze(risk_free_rate_fn(times))
        log_forward_moneyness = tf.math.log(
            tf.math.divide_no_nan(strikes, tf.reshape(
                spot, [dim, 1, 1]))) - tf.expand_dims(
                    (risk_free_rate - dividend_yield) * times, axis=-1)
        return log_forward_moneyness

      interpolator = interpolation_2d.Interpolation2D(
          expiry_times,
          _log_forward_moneyness(expiry_times, strikes),
          implied_volatilities,
          dtype=dtype)

      def _log_moneyness_2d_interpolator(times, strikes):
        risk_free_rate = risk_free_rate_fn(times)
        log_forward_moneyness = tf.math.log(
            strikes / spot) - (risk_free_rate - dividend_yield) * times
        moneyness_transposed = tf.transpose(log_forward_moneyness)
        times = tf.broadcast_to(times, moneyness_transposed.shape)
        return tf.transpose(
            interpolator.interpolate(times, moneyness_transposed))

      vs = volatility_surface.VolatilitySurface(
          valuation_date,
          expiry_dates,
          strikes,
          implied_volatilities,
          interpolator=_log_moneyness_2d_interpolator,
          dtype=dtype)

      local_volatility_fn = functools.partial(
          _dupire_local_volatility,
          initial_spot_price=spot,
          discount_factor_fn=discount_factor_fn,
          dividend_yield=dividend_yield,
          implied_volatility_surface=vs.volatility)

      return LocalVolatilityModel(
          dim,
          risk_free_rate=risk_free_rate_fn,
          dividend_yield=dividend_yield,
          local_volatility_fn=local_volatility_fn,
          dtype=dtype)

  @classmethod
  def from_volatility_surface(cls,
                              dim,
                              spot,
                              implied_volatility_surface,
                              discount_factor_fn,
                              dividend_yield=None,
                              dtype=None,
                              name=None):
    """Creates a `LocalVolatilityModel` from implied volatility data.

    Args:
      dim: A Python scalar which corresponds to the number of underlying assets
        comprising the model.
      spot: A real `Tensor` of shape `(dim,)` specifying the underlying spot
        price on the valuation date.
      implied_volatility_surface: Either an instance of
        `processed_market_data.VolatilitySurface` or a Python object containing
        the implied volatility market data. If the input is a Python object,
        then the object must implement a function `volatility(strike,
        expiry_times)` which takes real `Tensor`s corresponding to option
        strikes and time to expiry and returns a real `Tensor` containing the
        corresponding market implied volatility.
        The shape of `strike` is `(n,dim)` where `dim` is the dimensionality of
        the local volatility process and `t` is a scalar tensor. The output from
        the callable is a `Tensor` of shape `(n,dim)` containing the
        interpolated implied volatilties.
      discount_factor_fn: A python callable accepting one real `Tensor` argument
        time t. It should return a `Tensor` specifying the discount factor to
        time t.
      dividend_yield: A real `Tensor` of shape compatible with `spot_price`
        specifying the (continuosly compounded) dividend yield.
        If the underlying is an FX rate, then use this input to specify the
        foreign interest rate.
        Default value: `None` in which case the input is set to Zero.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
          `from_volatility_surface`.

    Returns:
      An instance of `LocalVolatilityModel` constructed using the input data.
    """
    name = name or "from_volatility_surface"
    with tf.name_scope(name):
      dividend_yield = dividend_yield or [0.0]
      dividend_yield = tf.convert_to_tensor(dividend_yield, dtype=dtype)

      risk_free_rate_fn = _get_risk_free_rate_from_discount_factor(
          discount_factor_fn)

      local_volatility_fn = functools.partial(
          _dupire_local_volatility,
          initial_spot_price=spot,
          discount_factor_fn=discount_factor_fn,
          dividend_yield=dividend_yield,
          implied_volatility_surface=implied_volatility_surface.volatility)

      return LocalVolatilityModel(
          dim,
          risk_free_rate=risk_free_rate_fn,
          dividend_yield=dividend_yield,
          local_volatility_fn=local_volatility_fn,
          dtype=dtype)


def _convert_to_tensor_fn(x, dtype, name):
  if callable(x):
    return x
  else:
    return lambda t: tf.convert_to_tensor(x, dtype, name=name)


def _get_risk_free_rate_from_discount_factor(discount_factor_fn):
  """Returns r(t) given a discount factor function."""

  def risk_free_rate_fn(t):
    logdf = lambda x: -tf.math.log(discount_factor_fn(x))
    return math.fwd_gradient(
        logdf, t, unconnected_gradients=tf.UnconnectedGradients.ZERO)

  return risk_free_rate_fn
