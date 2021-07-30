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

"""Local Stochastic Volatility process."""

import functools

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dates
from tf_quant_finance import math
from tf_quant_finance.experimental import local_volatility as lvm
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils
from tf_quant_finance.math import pde
from tf_quant_finance.math.interpolation import linear
from tf_quant_finance.models import generic_ito_process


class LocalStochasticVolatilityModel(generic_ito_process.GenericItoProcess):
  r"""Local stochastic volatility model.

  Local stochastic volatility (LSV) models assume that the spot price of an
  asset follows the following stochastic differential equation under the risk
  neutral measure [1]:

  ```None
    dS(t) / S(t) =  (r - d) dt + sqrt(v(t)) * L(t, S(t)) * dW_s(t)
    dv(t) = a(v(t)) dt + b(v(t)) dW_v(t)
    E[dW_s(t)dW_v(t)] = rho dt
  ```
  where `r` and `d` denote the risk free interest rate and dividend yield
  respectively. `S(t)` is the spot price, `v(t)` denotes the stochastic variance
  and the function `L(t, S(t))`  is the leverage function which is calibrated
  using the volatility smile data. The functions `a(v(t))` and `b(v(t))` denote
  the drift and volitility of the stochastic process for the variance and `rho`
  denotes the instantabeous correlation between the spot and the variance
  process. LSV models thus combine the local volatility dynamics with
  stochastic volatility.

  Using the relationship between the local volatility and the expectation of
  future instantaneous variance, leverage function can be computed as follows
  [2]:

  ```
  sigma(T,K)^2 = L(T,K)^2 * E[v(T)|S(T)=K]
  ```
  where the local volatility function `sigma(T,K)` can be computed using the
  Dupire's formula.

  The `LocalStochasticVolatilityModel` class contains a generic implementation
  of the LSV model with the flexibility to specify an arbitrary variance
  process. The default variance process is a Heston type process with
  mean-reverting variance (as in Ref. [1]):

  ```
  dv(t) = k(m - v(t)) dt + alpha*sqrt(v(t)) dW_v(t)
  ```

  #### References:
    [1]: Iain J. Clark. Foreign exchange option pricing - A Practitioner's
    guide. Chapter 5. 2011.
    [2]: I. Gyongy. Mimicking the one-dimensional marginal distributions of
    processes having an ito differential. Probability Theory and Related
    Fields, 71, 1986.
  """

  def __init__(self,
               leverage_fn,
               variance_process,
               risk_free_rate=None,
               dividend_yield=None,
               rho=None,
               dtype=None,
               name=None):
    """Initializes the Local stochastic volatility model.

    Args:
      leverage_fn: A Python callable which returns the Leverage function
        `L(t, S(t))` as a function of state and time. The function must accept
        a scalar `Tensor` corresponding to time 't' and a real `Tensor` of shape
        `[num_samples, 1]` corresponding to the underlying price (S) as
        inputs  and return a real `Tensor` containing the leverage function
        computed at (S,t).
      variance_process: An instance of `ItoProcess` specifying the
        dynamics of the variance process of the LSV model. The
        `variance_process` should implement a one-factor stochastic process.
        For the common version of Heston like variance model use
        `LSVVarianceModel`.
      risk_free_rate: An optional scalar real `Tensor` specifying the
        (continuously compounded) risk free interest rate. If the underlying is
        an FX rate, then use this input to specify the domestic interest rate.
        Note that the current implementation supports constant interest rates
        and dividend yield.
        Default value: `None` in which case the input is set to zero.
      dividend_yield: An optional real scalar `Tensor` specifying the
        (continuosly compounded) dividend yield. If the underlying is an FX
        rate, then use this input to specify the foreign interest rate.
        Note that the currect implementation supports constant interest rates
        and dividend yield.
        Default value: `None` in which case the input is set to zero.
      rho: A real scalar `Tensor` specifying the correlation between the
        underlying spot price and the variance process.
        Default value: `None` in which case cross correlations are assumed
        to be zero.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
        `local_stochastic_volatility_model`.
    """
    self._name = name or "local_stochastic_volatility_model"
    with tf.name_scope(self._name):
      if risk_free_rate is None:
        risk_free_rate = 0.0
      if dividend_yield is None:
        dividend_yield = 0.0
      self._risk_free_rate = tf.convert_to_tensor(risk_free_rate, dtype=dtype)
      self._dtype = dtype or self._domestic_rate.dtype
      self._dividend_yield = tf.convert_to_tensor(dividend_yield, dtype=dtype)
      self._leverage_fn = leverage_fn
      self._variance_process = variance_process
      dim = 1 + variance_process.dim()
      rho = rho or 0.0
      self._rho = _create_corr_matrix(rho, self._dtype)
      self._sqrt_rho = tf.linalg.cholesky(self._rho)

      def _vol_fn(t, state):
        """Volatility function of LSV model."""
        num_samples = state.shape.as_list()[0]
        broadcasted_t = tf.broadcast_to(t, [1, num_samples])
        spot_prices = state[:, 0]
        variance = state[:, 1:]
        level_fun = self._leverage_fn(
            broadcasted_t, tf.expand_dims(spot_prices, axis=0))
        spot_diffusion = tf.expand_dims(
            level_fun[0, :], axis=-1) * tf.expand_dims(
                spot_prices, axis=-1) * tf.math.sqrt(variance)
        variance_diffusion = self._variance_process.volatility_fn()(
            t, variance)
        diffusion = tf.concat([spot_diffusion, variance_diffusion], axis=1)
        diffusion = tf.expand_dims(diffusion, axis=-2)
        return diffusion * self._sqrt_rho

      # Drift function
      def _drift_fn(t, state):
        """Drift function of LSV model."""
        spot_drift = (
            self._risk_free_rate - self._dividend_yield) * state[:, :1]
        variance_drift = self._variance_process.drift_fn()(t, state[:, 1:])
        return tf.concat([spot_drift, variance_drift], axis=1)

      super(LocalStochasticVolatilityModel,
            self).__init__(dim, _drift_fn, _vol_fn, self._dtype, self._name)

  @classmethod
  def from_market_data(cls,
                       valuation_date,
                       expiry_dates,
                       strikes,
                       implied_volatilities,
                       variance_process,
                       initial_spot,
                       initial_variance,
                       rho=None,
                       risk_free_rate=None,
                       dividend_yield=None,
                       time_step=None,
                       num_grid_points=None,
                       grid_minimums=None,
                       grid_maximums=None,
                       dtype=None):
    """Creates a `LocalStochasticVolatilityModel` from market data.

    This function computes the leverage function for the LSV model by first
    computing the joint probability density function `p(t, X(t), v(t))` where
    `X(t)` is the log of the spot price and `v(t)` is the variance at time `t`.
    The joint probablity density is computed using the Fokker-Planck equation of
    the LSV model (see 6.8.2 in Ref [1]):

    ```None
    dp/dt = 1/2 d^2 [v L(t,X)^2 p]/dX^2 + 1/2 d^2 [b(v)^2 p]/dv^2 +
            rho d^2 [sqrt(v)L(t,X)b(v) p]/dXdv -
            d[(r - d - 1/2 v L(t,X)^2)p]/dX -
            d[a(v) p]/dv
    ```

    where `a(v)` and `b(v)` are the drift and diffusion functions for the
    variance process. Defining

    ```None
    I_n(k,t) = int v^n p(t, k, v) dv
    ```

    we can calculate the leverage function as follows:
    ```None
    L(k, t) = sigma(exp(k), t) sqrt(I_0(k, t)/I_1(k, t)).
    ```

    Note that the computation of `I_0` and `I_1` require the knowledge of
    leverage function and hence the computation of the leverage function is
    implicit in nature.

    Args:
      valuation_date: A scalar `DateTensor` specifying the valuation
        (or settlement) date for the market data.
      expiry_dates: A `DateTensor` of shape `(num_expiries,)` containing the
        expiry dates on which the implied volatilities are specified.
      strikes: A `Tensor` of real dtype and shape `(num_expiries,
        num_strikes)` specifying the strike prices at which implied volatilities
        are specified.
      implied_volatilities: A `Tensor` of real dtype and shape `(num_expiries,
        num_strikes)` specifying the implied volatilities.
      variance_process: An instance of `LSVVarianceModel` or
        `ItoProcess` specifying the dynamics of the variance process of
        the LSV model.
      initial_spot: A real scalar `Tensor` specifying the underlying spot price
        on the valuation date.
      initial_variance: A real scalar `Tensor` specifying the initial variance
        on the valuation date.
      rho: A real scalar `Tensor` specifying the correlation between spot price
        and the stochastic variance.
      risk_free_rate: A real scalar `Tensor` specifying the (continuosly
        compounded) risk free interest rate. If the underlying is an FX rate,
        then use this input to specify the domestic interest rate.
        Default value: `None` in which case the input is set to zero.
      dividend_yield: A real scalar `Tensor` specifying the (continuosly
        compounded) divident yield. If the underlying is an FX rate, then use
        this input to specify the foreign interest rate.
        Default value: `None` in which case the input is set to zero.
      time_step: A real scalar `Tensor` specifying the time step during the
        numerical solution of the Fokker-Planck PDE.
        Default value: None, in which case `time_step` corresponding to 100 time
          steps is used.
      num_grid_points: A scalar integer `Tensor` specifying the number of
        discretization points for each spatial dimension.
        Default value: None, in which case number of grid points is set to 100.
      grid_minimums: An optional `Tensor` of size 2 containing the minimum grid
        points for PDE spatial discretization. `grid_minimums[0]` correspond
        to the minimum spot price in the spatial grid and `grid_minimums[1]`
        correspond to the minimum variance value.
      grid_maximums: An optional `Tensor` of size 2 containing the maximum grid
        points for PDE spatial discretization. `grid_maximums[0]` correspond
        to the maximum spot price in the spatial grid and `grid_maximums[1]`
        correspond to the maximum variance value.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.

    Returns:
      An instance of `LocalStochasticVolatilityModel` constructed using the
      input data.
    """
    if risk_free_rate is None:
      discount_factor_fn = lambda t: tf.ones_like(t, dtype=dtype)
    else:
      r = tf.convert_to_tensor(risk_free_rate, dtype=dtype)
      discount_factor_fn = lambda t: tf.math.exp(-r * t)
    lv_model = lvm.LocalVolatilityModel.from_market_data(
        dim=1,
        valuation_date=valuation_date,
        expiry_dates=expiry_dates,
        strikes=strikes,
        implied_volatilities=implied_volatilities,
        spot=initial_spot,
        discount_factor_fn=discount_factor_fn,
        dividend_yield=dividend_yield,
        dtype=dtype)

    dtype = dtype or lv_model.dtype()
    max_time = tf.math.reduce_max(
        dates.daycount_actual_365_fixed(
            start_date=valuation_date, end_date=expiry_dates, dtype=dtype))
    if time_step is None:
      time_step = max_time / 100.0

    rho = rho or 0.0
    num_grid_points = num_grid_points or 100

    leverage_fn = _leverage_function_using_pde(
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        lv_model=lv_model,
        variance_model=variance_process,
        rho=[rho],
        initial_spot=initial_spot,
        initial_variance=initial_variance,
        time_step=time_step,
        max_time=max_time,
        num_grid_points=num_grid_points,
        grid_minimums=grid_minimums,
        grid_maximums=grid_maximums,
        dtype=dtype)
    return LocalStochasticVolatilityModel(
        leverage_fn,
        variance_process,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        rho=rho,
        dtype=dtype)

  @classmethod
  def from_volatility_surface(cls,
                              implied_volatility_surface,
                              variance_process,
                              initial_spot,
                              initial_variance,
                              rho=None,
                              risk_free_rate=None,
                              dividend_yield=None,
                              time_step=None,
                              num_grid_points=None,
                              grid_minimums=None,
                              grid_maximums=None,
                              dtype=None):
    """Creates a `LocalStochasticVolatilityModel` from volatility surface.

    This function computes the leverage function for the LSV model by first
    computing the joint probablity density function `p(t, X(t), v(t))` where
    `X(t)` is the log of the spot price and `v(t)` is the variance at time `t`.
    The joint probablity density is computed using the Fokker-Planck equation of
    the LSV model (see 6.8.2 in Ref [1]):
    ```None
    dp/dt = 1/2 d^2 [v L(t,X)^2 p]/dX^2 + 1/2 d^2 [b(v)^2 p]/dv^2 +
            rho d^2 [sqrt(v)L(t,X)b(v) p]/dXdv -
            d[(r - d - 1/2 v L(t,X)^2)p]/dX -
            d[a(v) p]/dv
    ```

    where `a(v)` and `b(v)` are the drift and diffusion functions for the
    variance process. Defining

    ```None
    I_n(k,t) = int v^n p(t, k, v) dv
    ```

    we can calculate the leverage function as follows:
    ```None
    L(k, t) = sigma(exp(k), t) sqrt(I_0(k, t)/I_1(k, t)).
    ```

    Args:
      implied_volatility_surface: Either an instance of
        `processed_market_data.VolatilitySurface` or a Python object containing
        the implied volatility market data. If the input is a Python object,
        then the object must implement a function `volatility(strike,
        expiry_times)` which takes real `Tensor`s corresponding to option
        strikes and time to expiry and returns a real `Tensor` containing the
        corresponding market implied volatility.
      variance_process: An instance of `LSVVarianceModel` or
        `ItoProcess`specifying the dynamics of the variance process of
        the LSV model.
      initial_spot: A real scalar `Tensor` specifying the underlying spot price
        on the valuation date.
      initial_variance: A real scalar `Tensor` specifying the initial variance
        on the valuation date.
      rho: A real scalar `Tensor` specifying the correlation between spot price
        and the stochastic variance.
      risk_free_rate: A real scalar `Tensor` specifying the (continuosly
        compounded) risk free interest rate. If the underlying is an FX rate,
        then use this input to specify the domestic interest rate.
        Default value: `None` in which case the input is set to zero.
      dividend_yield: A real scalar `Tensor` specifying the (continuosly
        compounded) divident yield. If the underlying is an FX rate, then use
        this input to specify the foreign interest rate.
        Default value: `None` in which case the input is set to zero.
      time_step: An optional real scalar `Tensor` specifying the time step
        during the numerical solution of the Fokker-Planck PDE.
        Default value: None, in which case `time_step` corresponding to 100 time
          steps is used.
      num_grid_points: A scalar integer `Tensor` specifying the number of
        discretization points for each spatial dimension.
        Default value: None, in which case number of grid points is set to 100.
      grid_minimums: An optional `Tensor` of size 2 containing the minimum grid
        points for PDE spatial discretization. `grid_minimums[0]` correspond
        to the minimum spot price in the spatial grid and `grid_minimums[1]`
        correspond to the minimum variance value.
      grid_maximums: An optional `Tensor` of size 2 containing the maximum grid
        points for PDE spatial discretization. `grid_maximums[0]` correspond
        to the maximum spot price in the spatial grid and `grid_maximums[1]`
        correspond to the maximum variance value.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.

    Returns:
      An instance of `LocalStochasticVolatilityModel` constructed using the
      input data.
    """

    if risk_free_rate is None:
      discount_factor_fn = lambda t: tf.ones_like(t, dtype=dtype)
    else:
      r = tf.convert_to_tensor(risk_free_rate, dtype=dtype)
      discount_factor_fn = lambda t: tf.math.exp(-r * t)
    lv_model = lvm.LocalVolatilityModel.from_volatility_surface(
        dim=1,
        spot=initial_spot,
        implied_volatility_surface=implied_volatility_surface,
        discount_factor_fn=discount_factor_fn,
        dividend_yield=dividend_yield,
        dtype=dtype)

    dtype = dtype or lv_model.dtype()
    day_count_fn = utils.get_daycount_fn(
        implied_volatility_surface.daycount_convention)
    max_time = tf.math.reduce_max(
        day_count_fn(
            start_date=implied_volatility_surface.settlement_date(),
            end_date=implied_volatility_surface.node_expiries()))
    if time_step is None:
      time_step = max_time / 100.0

    rho = rho or 0.0
    num_grid_points = num_grid_points or 100

    leverage_fn = _leverage_function_using_pde(
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        lv_model=lv_model,
        variance_model=variance_process,
        rho=[rho],
        initial_spot=initial_spot,
        initial_variance=initial_variance,
        time_step=time_step,
        max_time=max_time,
        num_grid_points=num_grid_points,
        grid_minimums=grid_minimums,
        grid_maximums=grid_maximums,
        dtype=dtype)
    return LocalStochasticVolatilityModel(
        leverage_fn,
        variance_process,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        rho=rho,
        dtype=dtype)


def _create_corr_matrix(rho, dtype):
  """Create correlation matrix with scalar `rho`."""
  one = tf.constant(1.0, dtype=dtype)
  m1 = tf.concat([one, rho], axis=0)
  m2 = tf.concat([rho, one], axis=0)
  return tf.stack([m1, m2])


def _machine_eps(dtype):
  """Returns the machine epsilon for the supplied dtype."""
  dtype = tf.as_dtype(dtype).as_numpy_dtype
  eps = 1e-6 if dtype == np.float32 else 1e-10
  return eps


def _two_d_integration(grid, value_grid):
  """Perform 2-D integration numerically."""
  log_spot_grid, variance_grid = tf.meshgrid(*grid)
  delta_v = variance_grid[1:, :] - variance_grid[:-1, :]
  delta_s = log_spot_grid[:, 1:] - log_spot_grid[:, :-1]
  integral = tf.math.reduce_sum(value_grid[0, :-1, :] * delta_v, axis=0)
  integral = tf.math.reduce_sum(integral[:-1] * delta_s[0, :])
  return integral


# TODO(b/175023506): Move to `grids` module
def _tavella_randell_nonuniform_grid(x_min, x_max, x_star, num_grid_points,
                                     alpha, dtype):
  """Creates non-uniform grid clustered around a specified point.

  Args:
    x_min: A real `Tensor` of shape `(dim,)` specifying the lower limit of the
      grid.
    x_max: A real `Tensor` of same shape and dtype as `x_min` specifying the
      upper limit of the grid.
    x_star: A real `Tensor` of same shape and dtype as `x_min` specifying the
      location on the grid around which higher grid density is desired.
    num_grid_points: A scalar integer `Tensor` specifying the number of points
      on the grid.
    alpha: A scalar parameter which controls the degree of non-uniformity of the
      grid. The smaller values of `alpha` correspond to greater degree of
      clustering around `x_star`.
    dtype: The default dtype to use when converting values to `Tensor`s.

  Returns:
    A real `Tensor` of shape `(dim, num_grid_points+1)` containing the
    non-uniform grid.
  """
  c1 = tf.math.asinh((x_min - x_star) / alpha)
  c2 = tf.math.asinh((x_max - x_star) / alpha)
  i = tf.expand_dims(tf.range(0, num_grid_points + 1, 1, dtype=dtype), axis=-1)
  grid = x_star + alpha * tf.math.sinh(c2 * i / num_grid_points + c1 *
                                       (1 - i / num_grid_points))
  # reshape from (num_grid_points+1, dim) to (dim, num_grid_points+1)
  return tf.transpose(grid)


def _conditional_expected_variance_from_pde_solution(grid, value_grid, dtype):
  """Computes E[variance|log_spot=k]."""
  # value_grid.shape = [1, num_x, num_y]
  log_spot_grid, variance_grid = tf.meshgrid(*grid)
  delta_s = variance_grid[1:, :] - variance_grid[:-1, :]
  # Calculate I(0)
  integral_0 = tf.math.reduce_sum(value_grid[0, :-1, :] * delta_s, axis=0)
  # Calculate I(1)
  integral_1 = tf.math.reduce_sum(
      variance_grid[:-1, :] * value_grid[0, :-1, :] * delta_s, axis=0)
  variance_given_logspot = tf.math.divide_no_nan(integral_1, integral_0)
  return functools.partial(
      linear.interpolate,
      x_data=log_spot_grid[0, :],
      y_data=variance_given_logspot,
      dtype=dtype)


def _leverage_function_using_pde(*, risk_free_rate, dividend_yield, lv_model,
                                 variance_model, rho, initial_spot,
                                 initial_variance, max_time, time_step,
                                 num_grid_points, grid_minimums,
                                 grid_maximums, dtype):
  """Computes Leverage function using Fokker-Planck PDE for joint density.

  This function computes the leverage function for the LSV model by first
  computing the joint probablity density function `p(t, X(t), v(t))` where
  `X(t)` is the log of the spot price and `v(t)` is the variance at time `t`.
  The joint probablity density is computed using the Fokker-Planck equation of
  the LSV model (see 6.8.2 in Ref [1]):
  ```None
  dp/dt = 1/2 d^2 [v L(t,X)^2 p]/dX^2 + 1/2 d^2 [b(v)^2 p]/dv^2 +
          rho d^2 [sqrt(v)L(t,X)b(v) p]/dXdv - d[(r - d - 1/2 v L(t,X)^2)p]/dX -
          d[a(v) p]/dv
  ```

  where `a(v)` and `b(v)` are the drift and diffusion functions for the
  variance process. Defining

  ```None
  I_n(k,t) = int v^n p(t, k, v) dv
  ```

  we can calculate the leverage function as follows:
  ```None
  L(k, t) = sigma(exp(k), t) sqrt(I_0(k, t)/I_1(k, t)).
  ```

  Args:
    risk_free_rate: A scalar real `Tensor` specifying the (continuosly
      compounded) risk free interest rate. If the underlying is an FX rate, then
      use this input to specify the domestic interest rate.
    dividend_yield: A real scalar `Tensor` specifying the (continuosly
      compounded) dividend yield. If the underlying is an FX rate, then use this
      input to specify the foreign interest rate.
    lv_model: An instance of `LocalVolatilityModel` specifying the local
      volatility for the spot price.
    variance_model: An instance of `LSVVarianceModel` specifying the dynamics of
      the variance process of the LSV model.
    rho: A real scalar `Tensor` specifying the correlation between spot price
      and the stochastic variance.
    initial_spot: A real scalar `Tensor` specifying the underlying spot price on
      the valuation date.
    initial_variance: A real scalar `Tensor` specifying the initial variance on
      the valuation date.
    max_time: A real scalar `Tensor` specifying the maximum time to which the
      Fokker-Planck PDE is evolved.
    time_step: A real scalar `Tensor` specifying the time step during the
      numerical solution of the Fokker-Planck PDE.
    num_grid_points: A scalar integer `Tensor` specifying the number of
      discretization points for each spatial dimension.
    grid_minimums: An optional `Tensor` of size 2 containing the minimum grid
      points for PDE spatial discretization. `grid_minimums[0]` correspond
      to the minimum spot price in the spatial grid and `grid_minimums[1]`
      correspond to the minimum variance value.
    grid_maximums: An optional `Tensor` of size 2 containing the maximum grid
      points for PDE spatial discretization. `grid_maximums[0]` correspond
      to the maximum spot price in the spatial grid and `grid_maximums[1]`
      correspond to the maximum variance value.
    dtype: The default dtype to use when converting values to `Tensor`s.

  Returns:
    A Python callable which computes the Leverage function `L(t, S(t))`. The
    function accepts a scalar `Tensor` corresponding to time 't' and a real
    `Tensor` of shape `[num_samples, 1]` corresponding to the spot price (S) as
    inputs  and return a real `Tensor` corresponding to the leverage function
    computed at (S,t).

  """
  if variance_model.dim() > 1:
    raise ValueError("The default model of Leverage function doesn\'t support "
                     "the variance process with more than 1 factor.")

  pde_grid_tol = _machine_eps(dtype)
  rho = tf.convert_to_tensor(rho, dtype=dtype)
  initial_spot = tf.convert_to_tensor(initial_spot, dtype=dtype)
  initial_log_spot = tf.math.log(
      tf.convert_to_tensor(initial_spot, dtype=dtype))
  initial_variance = tf.convert_to_tensor(initial_variance, dtype=dtype)
  risk_free_rate = tf.convert_to_tensor(risk_free_rate, dtype=dtype)
  dividend_yield = tf.convert_to_tensor(dividend_yield, dtype=dtype)
  rho = tf.convert_to_tensor(rho, dtype=dtype)

  x_scale = initial_log_spot
  y_scale = initial_variance
  # scaled log spot = log(spot/initial_spot)
  # scaled variance = variance / initial_variance
  scaled_initial_point = tf.convert_to_tensor([0.0, 1.0], dtype=dtype)

  # These are minimums and maximums for scaled log spot and scaled variance
  if grid_minimums is None:
    grid_minimums = [0.01, 0.0001]
  else:
    grid_minimums = tf.convert_to_tensor(grid_minimums, dtype=dtype)
    grid_minimums = [grid_minimums[0] / initial_spot,
                     grid_minimums[1] / initial_variance]
  if grid_maximums is None:
    grid_maximums = [10.0, 5.0]
  else:
    grid_maximums = tf.convert_to_tensor(grid_maximums, dtype=dtype)
    grid_maximums = [grid_maximums[0] / initial_spot,
                     grid_maximums[1] / initial_variance]

  log_spot_min = tf.math.log(
      tf.convert_to_tensor([grid_minimums[0]], dtype=dtype))
  log_spot_max = tf.math.log(
      tf.convert_to_tensor([grid_maximums[0]], dtype=dtype))
  variance_min = tf.convert_to_tensor([grid_minimums[1]], dtype=dtype)
  variance_max = tf.convert_to_tensor([grid_maximums[1]], dtype=dtype)

  grid_minimums = tf.concat([log_spot_min, variance_min], axis=0)
  grid_maximums = tf.concat([log_spot_max, variance_max], axis=0)

  grid = _tavella_randell_nonuniform_grid(grid_minimums, grid_maximums,
                                          scaled_initial_point, num_grid_points,
                                          0.3, dtype)
  grid = [tf.expand_dims(grid[0], axis=0), tf.expand_dims(grid[1], axis=0)]

  delta_x = tf.math.reduce_min(grid[0][0, 1:] - grid[0][0, :-1])
  delta_y = tf.math.reduce_min(grid[1][0, 1:] - grid[1][0, :-1])
  # Initialize leverage function L(t=0, S) = 1
  leverage_fn = functools.partial(
      linear.interpolate, x_data=[[0.0, 1.0]], y_data=[[1.0, 1.0]], dtype=dtype)

  def _initial_value():
    """Computes initial value as a delta function delta(log_spot(t), var(0))."""
    log_spot, variance = tf.meshgrid(*grid)

    init_value = tf.where(
        tf.math.logical_and(
            tf.math.abs(log_spot - scaled_initial_point[0]) <
            delta_x + pde_grid_tol,
            tf.math.abs(variance - scaled_initial_point[1]) <
            delta_y + pde_grid_tol), 1.0 / (delta_x * delta_y * 4), 0.0)
    # initial_value.shape = (1, num_grid_x, num_grid_y)
    return tf.expand_dims(init_value, axis=0)

  def _second_order_coeff_fn(t, grid):
    log_spot = grid[0] + x_scale
    variance = grid[1] * y_scale
    leverage_fn_t_x = leverage_fn(log_spot)
    val_xx = 0.5 * variance * leverage_fn_t_x**2
    val_xy = 0.5 * (rho * tf.math.sqrt(variance) * leverage_fn_t_x *
                    variance_model.volatility_fn()(t, variance)) / y_scale
    val_yx = val_xy
    val_yy = 0.5 * variance_model.volatility_fn()(t, variance)**2 / y_scale**2
    # return list of shape = (2,2). Each element has shape = grid.shape
    return [[-val_yy, -val_yx], [-val_xy, -val_xx]]

  def _first_order_coeff_fn(t, grid):
    log_spot = grid[0] + x_scale
    variance = grid[1] * y_scale
    leverage_fn_t_x = leverage_fn(log_spot)
    val_x = (risk_free_rate - dividend_yield -
             0.5 * variance * leverage_fn_t_x**2)
    val_y = variance_model.drift_fn()(t, variance)
    # return list of shape = (2,). Each element has shape = grid.shape
    return [val_y / y_scale, val_x]

  def _compute_leverage_fn(t, coord_grid, value_grid):
    log_spot = tf.expand_dims(coord_grid[0], axis=-1) + x_scale
    local_volatility_values = lv_model.local_volatility_fn()(
        t, tf.math.exp(log_spot))
    # TODO(b/176826650): Large values represent instability. Eventually this
    # should be addressed inside local vol model.
    local_volatility_values = tf.where(
        tf.math.abs(local_volatility_values) > 1e4, 0.0,
        local_volatility_values)
    # variance_given_logspot.shape = (num_grid_x, 1)
    variance_given_logspot = _conditional_expected_variance_from_pde_solution(
        [coord_grid[0] + x_scale, coord_grid[1] * y_scale], value_grid, dtype)(
            log_spot)

    leverage_fn_values = tf.math.divide_no_nan(
        local_volatility_values, tf.math.sqrt(variance_given_logspot))

    leverage_fn = functools.partial(
        linear.interpolate,
        x_data=grid[0] + x_scale,
        y_data=tf.transpose(leverage_fn_values),
        dtype=dtype)

    return leverage_fn

  @pde.boundary_conditions.neumann
  def _trivial_neumann_boundary(t, location_grid):
    del t, location_grid
    return 0.0

  leverage_fn_values = []
  leverage_fn_values.append(leverage_fn(grid[0][0])[0])
  # joint_density.shape = (1, num_grid_x, num_grid_y)
  joint_density = _initial_value()

  for tstart in np.arange(0.0, max_time, time_step):
    joint_density, coord_grid, _, _ = pde.fd_solvers.solve_forward(
        tstart,
        tstart + time_step,
        coord_grid=[grid[0][0], grid[1][0]],
        values_grid=joint_density,
        time_step=time_step / 10.0,
        values_transform_fn=None,
        inner_second_order_coeff_fn=_second_order_coeff_fn,
        inner_first_order_coeff_fn=_first_order_coeff_fn,
        zeroth_order_coeff_fn=None,
        boundary_conditions=[[
            _trivial_neumann_boundary, _trivial_neumann_boundary
        ], [_trivial_neumann_boundary, _trivial_neumann_boundary]],
        dtype=dtype)
    joint_density = tf.math.maximum(joint_density, 0.0)
    area_under_joint_density = _two_d_integration(
        [grid[0][0, :], grid[1][0, :]], joint_density)
    joint_density = joint_density / area_under_joint_density

    # TODO(b/176826743): Perform fixed point iteration instead of one step
    # update
    leverage_fn = _compute_leverage_fn(
        tf.convert_to_tensor(tstart + time_step), coord_grid, joint_density)
    leverage_fn_values.append(leverage_fn(grid[0][0, :] + x_scale)[0, :])

  # leverage_fn_values.shape = (num_pde_timesteps, num_grid_x,)
  leverage_fn_values = tf.convert_to_tensor(leverage_fn_values, dtype=dtype)
  times = tf.range(0.0, max_time + time_step, time_step, dtype=dtype)

  def _return_fn(t, spot):
    leverage_fn_interpolator = (
        math.interpolation.interpolation_2d.Interpolation2D(
            x_data=[times],
            y_data=tf.expand_dims(
                tf.repeat(grid[0] + x_scale, times.shape[0], axis=0), axis=0),
            z_data=tf.expand_dims(leverage_fn_values, axis=0),
            dtype=dtype))
    return leverage_fn_interpolator.interpolate(t, tf.math.log(spot))

  return _return_fn
