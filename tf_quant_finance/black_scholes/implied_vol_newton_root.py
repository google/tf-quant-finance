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
"""Calculation of the Black-Scholes implied volatility via Newton's method."""

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance.black_scholes import implied_vol_approximation as approx
from tf_quant_finance.black_scholes import implied_vol_utils as utils
from tf_quant_finance.math.root_search import newton

_SQRT_2 = np.sqrt(2., dtype=np.float64)
_SQRT_2_PI = np.sqrt(2 * np.pi, dtype=np.float64)
_NORM_PDF_AT_ZERO = 1. / _SQRT_2_PI


def _cdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


def _pdf(x):
  return tf.math.exp(-0.5 * x ** 2) / _SQRT_2_PI


def implied_vol(*,
                prices,
                strikes,
                expiries,
                spots=None,
                forwards=None,
                discount_factors=None,
                is_call_options=None,
                initial_volatilities=None,
                underlying_distribution=utils.UnderlyingDistribution.LOG_NORMAL,
                tolerance=1e-8,
                max_iterations=20,
                validate_args=False,
                dtype=None,
                name=None):
  """Computes implied volatilities from given call or put option prices.

  This method applies a Newton root search algorithm to back out the implied
  volatility given the price of either a put or a call option.

  The implementation assumes that each cell in the supplied tensors corresponds
  to an independent volatility to find.

  Args:
    prices: A real `Tensor` of any shape. The prices of the options whose
      implied vol is to be calculated.
    strikes: A real `Tensor` of the same dtype as `prices` and a shape that
      broadcasts with `prices`. The strikes of the options.
    expiries: A real `Tensor` of the same dtype as `prices` and a shape that
      broadcasts with `prices`. The expiry for each option. The units should be
      such that `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `prices`. The current spot price of the underlying. Either this argument
      or the `forwards` (but not both) must be supplied.
      Default value: None.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `prices`. The forwards to maturity. Either this argument or the `spots`
      must be supplied but both must not be supplied.
      Default value: None.
    discount_factors: An optional real `Tensor` of same dtype as the `prices`.
      If not None, these are the discount factors to expiry (i.e. e^(-rT)). If
      None, no discounting is applied (i.e. it is assumed that the undiscounted
      option prices are provided ). If `spots` is supplied and
      `discount_factors` is not None then this is also used to compute the
      forwards to expiry.
      Default value: None, equivalent to discount factors = 1.
    is_call_options: A boolean `Tensor` of a shape compatible with `prices`.
      Indicates whether the option is a call (if True) or a put (if False). If
      not supplied, call options are assumed.
      Default value: None.
    initial_volatilities: A real `Tensor` of the same shape and dtype as
      `forwards`. The starting positions for Newton's method.
      Default value: None. If not supplied, the starting point is chosen using
        the Stefanica-Radoicic scheme. See `polya_approx.implied_vol` for
        details.
      Default value: None.
    underlying_distribution: Enum value of ImpliedVolUnderlyingDistribution to
      select the distribution of the underlying.
      Default value: UnderlyingDistribution.LOG_NORMAL
    tolerance: `float`. The root finder will stop where this tolerance is
      crossed.
    max_iterations: `int`. The maximum number of iterations of Newton's method.
      Default value: 20.
    validate_args: A Python bool. If True, indicates that arguments should be
      checked for correctness before performing the computation. The checks
      performed are: (1) Forwards and strikes are positive. (2) The prices
        satisfy the arbitrage bounds (i.e. for call options, checks the
        inequality `max(F-K, 0) <= Price <= F` and for put options, checks that
        `max(K-F, 0) <= Price <= K`.). (3) Checks that the prices are not too
        close to the bounds. It is numerically unstable to compute the implied
        vols from options too far in the money or out of the money.
      Default value: False.
    dtype: `tf.Dtype` to use when converting arguments to `Tensor`s. If not
      supplied, the default TensorFlow conversion will take place. Note that
      this argument does not do any casting for `Tensor`s or numpy arrays.
      Default value: None.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'implied_vol' is used.
      Default value: None.

  Returns:
    A 3-tuple containing the following items in order:
       (a) implied_vols: A `Tensor` of the same dtype as `prices` and shape as
         the common broadcasted shape of
         `(prices, spots/forwards, strikes, expiries)`. The implied vols as
         inferred by the algorithm. It is possible that the search may not have
         converged or may have produced NaNs. This can be checked for using the
         following return values.
       (b) converged: A boolean `Tensor` of the same shape as `implied_vols`
         above. Indicates whether the corresponding vol has converged to within
         tolerance.
       (c) failed: A boolean `Tensor` of the same shape as `implied_vols` above.
         Indicates whether the corresponding vol is NaN or not a finite number.
         Note that converged being True implies that failed will be false.
         However, it may happen that converged is False but failed is not True.
         This indicates the search did not converge in the permitted number of
         iterations but may converge if the iterations are increased.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')

  with tf.compat.v1.name_scope(
      name,
      default_name='implied_vol',
      values=[
          prices, spots, forwards, strikes, expiries, discount_factors,
          is_call_options, initial_volatilities
      ]):
    prices = tf.convert_to_tensor(prices, dtype=dtype, name='prices')
    dtype = prices.dtype
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    if discount_factors is None:
      discount_factors = tf.convert_to_tensor(
          1.0, dtype=dtype, name='discount_factors')
    else:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      forwards = spots / discount_factors

    if initial_volatilities is None:
      if underlying_distribution is utils.UnderlyingDistribution.LOG_NORMAL:
        initial_volatilities = approx.implied_vol(
            prices=prices,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            discount_factors=discount_factors,
            is_call_options=is_call_options,
            validate_args=validate_args)
      elif underlying_distribution is utils.UnderlyingDistribution.NORMAL:
        initial_volatilities = prices / _NORM_PDF_AT_ZERO
    else:
      initial_volatilities = tf.convert_to_tensor(
          initial_volatilities, dtype=dtype, name='initial_volatilities')

    implied_vols, converged, failed = _newton_implied_vol(
        prices, strikes, expiries, forwards, discount_factors, is_call_options,
        initial_volatilities, underlying_distribution,
        tolerance, max_iterations)
    return implied_vols, converged, failed


def _newton_implied_vol(prices, strikes, expiries, forwards, discount_factors,
                        is_call_options, initial_volatilities,
                        underlying_distribution, tolerance, max_iterations):
  """Uses Newton's method to find Black Scholes implied volatilities of options.

  Finds the volatility implied under the Black Scholes option pricing scheme for
  a set of European options given observed market prices. The implied volatility
  is found via application of Newton's algorithm for locating the root of a
  differentiable function.

  The implementation assumes that each cell in the supplied tensors corresponds
  to an independent volatility to find.

  Args:
    prices: A real `Tensor` of any shape. The prices of the options whose
      implied vol is to be calculated.
    strikes: A real `Tensor` of the same dtype as `prices` and a shape that
      broadcasts with `prices`. The strikes of the options.
    expiries: A real `Tensor` of the same dtype as `prices` and a shape that
      broadcasts with `prices`. The expiry for each option. The units should be
      such that `expiry * volatility**2` is dimensionless.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `prices`. The forwards to maturity.
    discount_factors: An optional real `Tensor` of same dtype as the `prices`.
      If not None, these are the discount factors to expiry (i.e. e^(-rT)). If
      None, no discounting is applied (i.e. it is assumed that the undiscounted
      option prices are provided ).
    is_call_options: A boolean `Tensor` of a shape compatible with `prices`.
      Indicates whether the option is a call (if True) or a put (if False). If
      not supplied, call options are assumed.
    initial_volatilities: A real `Tensor` of the same shape and dtype as
      `forwards`. The starting positions for Newton's method.
    underlying_distribution: Enum value of ImpliedVolUnderlyingDistribution to
      select the distribution of the underlying.
    tolerance: `float`. The root finder will stop where this tolerance is
      crossed.
    max_iterations: `int`. The maximum number of iterations of Newton's method.

  Returns:
    A three tuple of `Tensor`s, each the same shape as `forwards`. It
    contains the implied volatilities (same dtype as `forwards`), a boolean
    `Tensor` indicating whether the corresponding implied volatility converged,
    and a boolean `Tensor` which is true where the corresponding implied
    volatility is not a finite real number.
  """
  if underlying_distribution is utils.UnderlyingDistribution.LOG_NORMAL:
    pricer = _make_black_lognormal_objective_and_vega_func(
        prices, forwards, strikes, expiries, is_call_options,
        discount_factors)
  elif underlying_distribution is utils.UnderlyingDistribution.NORMAL:
    pricer = _make_bachelier_objective_and_vega_func(
        prices, forwards, strikes, expiries, is_call_options,
        discount_factors)

  results = newton.root_finder(
      pricer,
      initial_volatilities,
      max_iterations=max_iterations,
      tolerance=tolerance)
  return results


def _get_normalizations(prices, forwards, strikes, discount_factors):
  """Returns the normalized prices, normalization factors, and discount_factors.

  The normalization factors is the larger of strikes and forwards.
  If `discount_factors` is not None, these are the discount factors to expiry.
  If None, no discounting is applied and 1's are returned.

  Args:
    prices: A real `Tensor` of any shape. The observed market prices of the
      assets.
    forwards: A real `Tensor` of the same shape and dtype as `prices`. The
      current forward prices to expiry.
    strikes: A real `Tensor` of the same shape and dtype as `prices`. The strike
      prices of the options.
    discount_factors: A real `Tensor` of same dtype as the `prices`.

  Returns:
    the normalized prices, normalization factors, and discount_factors.
  """
  strikes_abs = tf.abs(strikes)
  forwards_abs = tf.abs(forwards)
  # orientations will decide the normalization strategy.
  orientations = strikes_abs >= forwards_abs
  # normalization is the greater of strikes or forwards
  normalization = tf.where(orientations, strikes_abs, forwards_abs)
  normalization = tf.where(tf.equal(normalization, 0),
                           tf.ones_like(normalization), normalization)
  normalized_prices = prices / normalization
  if discount_factors is not None:
    normalized_prices /= discount_factors
  else:
    discount_factors = tf.ones_like(normalized_prices)

  return normalized_prices, normalization, discount_factors


def _make_black_lognormal_objective_and_vega_func(
    prices, forwards, strikes, expiries, is_call_options, discount_factors):
  """Produces an objective and vega function for the Black Scholes model.

  The returned function maps volatilities to a tuple of objective function
  values and their gradients with respect to the volatilities. The objective
  function is the difference between Black Scholes prices and observed market
  prices, whereas the gradient is called vega of the option. That is:

  ```
  g(s) = (f(s) - a, f'(s))
  ```

  Where `g` is the returned function taking volatility parameter `s`, `f` the
  Black Scholes price with all other variables curried and `f'` its derivative,
  and `a` the observed market prices of the options. Hence `g` calculates the
  information necessary for finding the volatility implied by observed market
  prices for options with given terms using first order methods.

  #### References
  [1] Hull, J., 2018. Options, Futures, and Other Derivatives. Harlow, England.
  Pearson. (p.358 - 361)

  Args:
    prices: A real `Tensor` of any shape. The observed market prices of the
      assets.
    forwards: A real `Tensor` of the same shape and dtype as `prices`. The
      current forward prices to expiry.
    strikes: A real `Tensor` of the same shape and dtype as `prices`. The strike
      prices of the options.
    expiries: A real `Tensor` of same shape and dtype as `forwards`. The expiry
      for each option. The units should be such that `expiry * volatility**2` is
      dimensionless.
    is_call_options: A boolean `Tensor` of same shape and dtype as `forwards`.
      Positive one where option is a call, negative one where option is a put.
    discount_factors: A real `Tensor` of the same shape and dtype as `forwards`.
      The total discount factors to apply.

  Returns:
    A function from volatilities to a Black Scholes objective and its
    derivative (which is coincident with Vega).
  """
  normalized_prices, normalization, discount_factors = _get_normalizations(
      prices, forwards, strikes, discount_factors)

  norm_forwards = forwards / normalization
  norm_strikes = strikes / normalization
  lnz = tf.math.log(forwards) - tf.math.log(strikes)
  sqrt_t = tf.sqrt(expiries)
  if is_call_options is not None:
    is_call_options = tf.convert_to_tensor(is_call_options,
                                           dtype=tf.bool,
                                           name='is_call_options')
  def _black_objective_and_vega(volatilities):
    """Calculate the Black Scholes price and vega for a given volatility.

    This method returns normalized results.

    Args:
      volatilities: A real `Tensor` of same shape and dtype as `forwards`. The
        volatility to expiry.

    Returns:
      A tuple containing (value, gradient) of the black scholes price, both of
        which are `Tensor`s of the same shape and dtype as `volatilities`.
    """
    vol_t = volatilities * sqrt_t
    d1 = (lnz / vol_t + vol_t / 2)
    d2 = d1 - vol_t
    implied_prices = norm_forwards * _cdf(d1) - norm_strikes * _cdf(d2)
    if is_call_options is not None:
      put_prices = implied_prices - norm_forwards + norm_strikes
      implied_prices = tf.where(
          tf.broadcast_to(is_call_options, tf.shape(put_prices)),
          implied_prices, put_prices)
    vega = norm_forwards * _pdf(d1) * sqrt_t / discount_factors
    return implied_prices - normalized_prices, vega

  return _black_objective_and_vega


def _make_bachelier_objective_and_vega_func(
    prices, forwards, strikes, expiries, is_call_options, discount_factors):
  """Produces an objective and vega function for the Bachelier model.

  The returned function maps volatilities to a tuple of objective function
  values and their gradients with respect to the volatilities. The objective
  function is the difference between model implied prices and observed market
  prices, whereas the gradient is called vega of the option. That is:

  ```
  g(s) = (f(s) - a, f'(s))
  ```

  Where `g` is the returned function taking volatility parameter `s`, `f` the
  Black Scholes price with all other variables curried and `f'` its derivative,
  and `a` the observed market prices of the options. Hence `g` calculates the
  information necessary for finding the volatility implied by observed market
  prices for options with given terms using first order methods.

  #### References
  [1] Wenqing H., 2013. Risk Measures with Normal Distributed Black Options
  Pricing Model. Mälardalen University, Sweden. (p. 10 - 17)

  Args:
    prices: A real `Tensor` of any shape. The observed market prices of the
      options.
    forwards: A real `Tensor` of the same shape and dtype as `prices`. The
      current forward prices to expiry.
    strikes: A real `Tensor` of the same shape and dtype as `prices`. The strike
      prices of the options.
    expiries: A real `Tensor` of same shape and dtype as `forwards`. The expiry
      for each option. The units should be such that `expiry * volatility**2` is
      dimensionless.
    is_call_options: A boolean `Tensor` of same shape and dtype as `forwards`.
      `True` for call options and `False` for put options.
    discount_factors: A real `Tensor` of the same shape and dtype as `forwards`.
      The total discount factors to option expiry.

  Returns:
    A function from volatilities to a Black Scholes objective and its
    derivative (which is coincident with Vega).
  """
  normalized_prices, normalization, discount_factors = _get_normalizations(
      prices, forwards, strikes, discount_factors)
  norm_forwards = forwards / normalization
  norm_strikes = strikes / normalization
  sqrt_t = tf.sqrt(expiries)
  if is_call_options is not None:
    is_call_options = tf.convert_to_tensor(is_call_options,
                                           dtype=tf.bool,
                                           name='is_call_options')
  def _objective_and_vega(volatilities):
    """Calculate the Bachelier price and vega for a given volatility.

    This method returns normalized results.

    Args:
      volatilities: A real `Tensor` of same shape and dtype as `forwards`. The
        volatility to expiry.

    Returns:
      A tuple containing (value, gradient) of the black scholes price, both of
        which are `Tensor`s of the same shape and dtype as `volatilities`.
    """
    vols = volatilities * sqrt_t / normalization
    d1 = (norm_forwards - norm_strikes) / vols
    implied_prices = ((norm_forwards - norm_strikes) * _cdf(d1)
                      +  vols * _pdf(d1))
    if is_call_options is not None:
      put_prices = implied_prices - norm_forwards + norm_strikes
      implied_prices = tf.where(is_call_options, implied_prices, put_prices)

    vega = _pdf(d1) * sqrt_t / discount_factors / normalization
    return implied_prices - normalized_prices, vega

  return _objective_and_vega
