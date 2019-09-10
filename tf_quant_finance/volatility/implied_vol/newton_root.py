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

# Lint as: python2, python3
"""Calculation of the Black-Scholes implied volatility via Newton's method."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_quant_finance.volatility.implied_vol import polya_approx


def implied_vol(forwards,
                strikes,
                expiries,
                discount_factors,
                prices,
                option_signs,
                initial_volatilities=None,
                validate_args=True,
                tolerance=1e-8,
                max_iterations=20,
                name=None,
                dtype=None):
  """Finds the implied volatilities of options under the Black Scholes model.

  This method first estimates, using the Radiocic-Polya approximation, the
  volatility of an option implied by observed market prices under the Black
  Scholes model. This estimate is then used to initialise Newton's algorithm for
  locating the root of a differentiable function.

  The implementation assumes that each cell in the supplied tensors corresponds
  to an independent volatility to find.

  #### Examples
  ```python
  forwards = np.array([1.0, 1.0, 1.0, 1.0])
  strikes = np.array([1.0, 2.0, 1.0, 0.5])
  expiries = np.array([1.0, 1.0, 1.0, 1.0])
  discount_factors = np.array([1.0, 1.0, 1.0, 1.0])
  option_signs = np.array([1.0, 1.0, -1.0, -1.0])
  volatilities = np.array([1.0, 1.0, 1.0, 1.0])
  prices = black_scholes.option_price(
      forwards,
      strikes,
      volatilities,
      expiries,
      discount_factors=discount_factors,
      is_call_options=is_call_options)
  implied_vols = newton_vol.implied_vol(forwards,
                                        strikes,
                                        expiries,
                                        discount_factors,
                                        prices,
                                        option_signs)
  with tf.Session() as session:
    print(session.run(implied_vols[0]))
  # Expected output:
  # [ 1.  1.  1.  1.]

  Args:
    forwards: A real `Tensor`. The current forward prices to expiry.
    strikes: A real `Tensor` of the same shape and dtype as `forwards`. The
      strikes of the options to be priced.
    expiries: A real `Tensor` of same shape and dtype as `forwards`. The expiry
      for each option. The units should be such that `expiry * volatility**2` is
      dimensionless.
    discount_factors: A real `Tensor` of the same shape and dtype as `forwards`.
      The total discount factors to apply.
    prices: A real `Tensor` of the same shape and dtype as `forwards`. The
      observed market prices to match.
    option_signs: A real `Tensor` of same shape and dtype as `forwards`.
      Positive one where option is a call, negative one where option is a put.
    initial_volatilities: A real `Tensor` of the same shape and dtype as
      `forwards`. The starting postions for Newton's method.
      Default value: None. If not supplied, the starting point is chosen using
        the Stefanica-Radoicic scheme. See `polya_approx.implied_vol` for
        details.
    validate_args: A Python bool. If True, indicates that arguments should be
      checked for correctness before performing the computation. The checks
      performed are: (1) Forwards and strikes are positive. (2) The prices
        satisfy the arbitrage bounds (i.e. for call options, checks the
        inequality `max(F-K, 0) <= Price <= F` and for put options, checks that
        `max(K-F, 0) <= Price <= K`.). (3) Checks that the prices are not too
        close to the bounds. It is numerically unstable to compute the implied
        vols from options too far in the money or out of the money.
    tolerance: `float`. The root finder will stop where this tolerance is
      crossed.
    max_iterations: `int`. The maximum number of iterations of Newton's method.
    name: `str`, default "implied_vol", to be prefixed to the name of TensorFlow
      ops created by this function.
    dtype: optional `tf.DType`. If supplied the `forwards`, `strikes`,
      `expiries`, `discounts`, `prices`, `initial_volatilities` and
      `option_signs` will be coerced to this type.

  Returns:
    A three tuple of `Tensor`s, each the same shape as `forwards`. It
    contains the implied volatilities (same dtype as `forwards`), a boolean
    `Tensor` indicating whether the corresponding implied volatility converged,
    and a boolean `Tensor` which is true where the corresponding implied
    volatility is not a finite real number.
  ```
  """
  with tf.compat.v1.name_scope(
      name, "implied_vol",
      [prices, forwards, strikes, expiries, discount_factors, option_signs]):
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name="forwards")
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name="strikes")
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name="expiries")
    discount_factors = tf.convert_to_tensor(
        discount_factors, dtype=dtype, name="discount_factors")
    prices = tf.convert_to_tensor(prices, dtype=dtype, name="prices")
    option_signs = tf.convert_to_tensor(
        option_signs, dtype=dtype, name="option_signs")
    is_call_options = tf.convert_to_tensor(
        option_signs > 0.0, name="is_call_options")
    if initial_volatilities is None:
      initial_volatilities = polya_approx.polya_implied_vol(
          prices,
          forwards,
          strikes,
          expiries,
          discount_factors=discount_factors,
          is_call_options=is_call_options,
          validate_args=validate_args)
    else:
      initial_volatilities = tf.convert_to_tensor(
          initial_volatilities, dtype=dtype, name="initial_volatilities")

    implied_vols, converged, failed = _newton_implied_vol(
        forwards,
        strikes,
        expiries,
        discount_factors,
        prices,
        initial_volatilities,
        option_signs,
        max_iterations=max_iterations,
        tolerance=tolerance)
    return implied_vols, converged, failed


# TODO(b/139566435): Extract the Newton root finder to a separate module.
def newton_root_finder(value_and_grad_func,
                       initial_values,
                       max_iterations=20,
                       tolerance=1e-8,
                       dtype=None,
                       name=None):
  """Uses Newton's method to find roots of scalar functions of scalar variables.

  This method uses Newton's algorithm to find values `x` such that `f(x)=0` for
  some real-valued differentiable function `f`. Given an initial value `x_0` the
  values are iteratively updated as:

    `x_{n+1} = x_n - f(x_n) / f'(x_n),`

  for further details on Newton's method, see [1]. The implementation accepts
  array-like arguments and assumes that each cell corresponds to an independent
  scalar model.

  #### Examples
  ```python
  # Set up the problem of finding the square roots of three numbers.
  constants = np.array([4.0, 9.0, 16.0])
  initial_values = np.ones(len(constants))
  def objective_and_gradient(values):
    objective = values**2 - constants
    gradient = 2.0 * values
    return objective, gradient

  # Obtain and evaluate a tensor containing the roots.
  roots = newton_vol.newton_newton_newton_root_finder(objective_and_gradient,
                                        initial_values)
  with tf.Session() as sess:
    root_values, converged, failed = sess.run(roots)
    print(root_values)  # Expected output: [ 2.  3.  4.]
    print(converged)  # Expected output: [ True  True  True]
    print(failed)  # Expected output: [False False False]
  ```

  #### References
  [1] Luenberger, D.G., 1984. 'Linear and Nonlinear Programming'. Reading, MA:
  Addison-Wesley.

  Args:
    value_and_grad_func: A python callable that takes a `Tensor` of the same
      shape and dtype as the `initial_values` and which returns a two-`tuple` of
      `Tensors`, namely the objective function and the gradient evaluated at the
      passed parameters.
    initial_values: A real `Tensor` of any shape. The initial values of the
      parameters to use (`x_0` in the notation above).
    max_iterations: positive `int`, default 100. The maximum number of
      iterations of Newton's method.
    tolerance: positive scalar `Tensor`, default 1e-8. The root finder will
      judge an element to have converged if `|f(x_n) - a|` is less than
      `tolerance` (using the notation above), or if `x_n` becomes `nan`. When an
      element is judged to have converged it will no longer be updated. If all
      elements converge before `max_iterations` is reached then the root finder
      will return early.
    dtype: optional `tf.DType`. If supplied the `initial_values` will be coerced
      to this data type.
    name: `str`, default "newton_root_finder", to be prefixed to the name of
      TensorFlow ops created by this function.

  Returns:
    A three tuple of `Tensor`s, each the same shape as `initial_values`. It
    contains the found roots (same dtype as `initial_values`), a boolean
    `Tensor` indicating whether the corresponding root results in an objective
    function value less than the tolerance, and a boolean `Tensor` which is true
    where the corresponding 'root' is not finite.
  """
  with tf.compat.v1.name_scope(
      name,
      default_name="newton_root_finder",
      values=[initial_values, tolerance]):

    initial_values = tf.convert_to_tensor(
        initial_values, dtype=dtype, name="initial_values")

    starting_position = (tf.constant(0, dtype=tf.int32), initial_values,
                         tf.zeros_like(initial_values, dtype=tf.bool),
                         tf.is_nan(initial_values))

    def _condition(counter, parameters, converged, failed):
      del parameters
      early_stop = tf.reduce_all(converged | failed)
      return ~((counter >= max_iterations) | early_stop)

    def _updater(counter, parameters, converged, failed):
      """Updates each parameter via Newton's method."""
      values, gradients = value_and_grad_func(parameters)
      converged = tf.abs(values) < tolerance
      # Used to zero out updates to cells that have converged.
      update_mask = tf.cast(~converged, dtype=parameters.dtype)
      increment = -update_mask * values / gradients
      updated_parameters = parameters + increment
      failed = ~tf.is_finite(updated_parameters)

      return counter + 1, updated_parameters, converged, failed

    return tf.while_loop(_condition, _updater, starting_position)[1:]


def _make_black_objective_and_vega_func(prices, forwards, strikes, expiries,
                                        option_signs, discount_factors):
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

  ### References
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
    option_signs: A real `Tensor` of same shape and dtype as `forwards`.
      Positive one where option is a call, negative one where option is a put.
    discount_factors: A real `Tensor` of the same shape and dtype as `forwards`.
      The total discount factors to apply.

  Returns:
    A function from volatilities to a Black Scholes objective and its
    derivative (which is coincident with Vega).
  """
  dtype = prices.dtype.base_dtype
  phi = tfp.distributions.Normal(
      loc=tf.zeros(1, dtype=dtype), scale=tf.ones(1, dtype=dtype))
  # orientations will decide the normalization strategy.
  orientations = strikes >= forwards
  # normalization is the greater of strikes or forwards
  normalization = tf.where(orientations, strikes, forwards)
  normalized_prices = prices / normalization
  units = tf.ones_like(forwards)
  # y is 1 when strikes >= forwards and strikes/forwards otherwise
  y = tf.where(orientations, units, strikes / forwards)
  # x is forwards/strikes when strikes >= forwards and 1 otherwise
  x = tf.where(orientations, forwards / strikes, units)
  lnz = tf.log(forwards) - tf.log(strikes)
  sqrt_t = tf.sqrt(expiries)

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
    v = volatilities * sqrt_t
    d1 = option_signs * (lnz / v + v / 2)
    d2 = option_signs * (option_signs * d1 - v)
    val = option_signs * discount_factors * (
        (x * phi.cdf(d1) - y * phi.cdf(d2))) - normalized_prices
    grad = option_signs * discount_factors * (
        y * phi.prob(d2) * (d1 / v) - x * phi.prob(d1) * (d2 / v))
    return val, grad

  return _black_objective_and_vega


def _newton_implied_vol(forwards,
                        strikes,
                        expiries,
                        discount_factors,
                        prices,
                        initial_volatilities,
                        option_signs,
                        max_iterations=100,
                        tolerance=1e-8,
                        name=None,
                        dtype=None):
  """Uses Newton's method to find Black Scholes implied volatilities of options.

  Finds the volatility implied under the Black Scholes option pricing scheme for
  a set of European options given observed market prices. The implied volatility
  is found via application of Newton's algorithm for locating the root of a
  differentiable function.

  The implmentation assumes that each cell in the supplied tensors corresponds
  to an independent volatility to find.

  #### Examples
  ```python
  forwards = np.array([1.0, 1.0, 1.0, 1.0])
  strikes = np.array([1.0, 2.0, 1.0, 0.5])
  expiries = np.array([1.0, 1.0, 1.0, 1.0])
  discounts = np.array([1.0, 1.0, 1.0, 1.0])
  initial_volatilities = np.array([2.0, 0.5, 2.0, 0.5])
  option_signs = np.array([1.0, 1.0, -1.0, -1.0])
  is_call_options = np.array([True, True, False, False])
  volatilities = np.array([1.0, 1.0, 1.0, 1.0])
  prices = black_scholes.option_price(
      forwards,
      strikes,
      volatilities,
      expiries,
      discount_factors=discounts,
      is_call_options=is_call_options)
  implied_vols, converged, failed = newton_vol.newton_implied_vol(
      forwards,
      strikes,
      expiries,
      discounts,
      prices,
      initial_volatilities,
      option_signs,
      max_iterations=100)
  with tf.Session() as session:
    print(session.run(implied_vols))
  # Expected output:
  # [ 1.  1.  1.  1.]

  Args:
    forwards: A real `Tensor`. The current forward prices to expiry.
    strikes: A real `Tensor` of the same shape and dtype as `forwards`. The
      strikes of the options to be priced.
    expiries: A real `Tensor` of same shape and dtype as `forwards`. The expiry
      for each option. The units should be such that `expiry * volatility**2` is
      dimensionless.
    discount_factors: A real `Tensor` of the same shape and dtype as `forwards`.
      The total discount factors to apply.
    prices: A real `Tensor` of the same shape and dtype as `forwards`. The
      observed market prices to match.
    initial_volatilities: A real `Tensor` of the same shape and dtype as
      `forwards`. The starting postions for Newton's method.
    option_signs: A real `Tensor` of same shape and dtype as `forwards`.
      Positive one where option is a call, negative one where option is a put.
    max_iterations: `int`. The maximum number of iterations of Newton's method.
    tolerance: `float`. The root finder will stop where this tolerance is
      crossed.
    name: `str`, default "newton_implied_vol", to be prefixed to the name of
      TensorFlow ops created by this function.
    dtype: optional `tf.DType`. If supplied the `forwards`, `strikes`,
      `expiries`, `discounts`, `prices`, `initial_volatilities` and
      `option_signs` will be coerced to this type.

  Returns:
    A three tuple of `Tensor`s, each the same shape as `forwards`. It
    contains the implied volatilities (same dtype as `forwards`), a boolean
    `Tensor` indicating whether the corresponding implied volatility converged,
    and a boolean `Tensor` which is true where the corresponding implied
    volatility is not a finite real number.
  ```
  """
  with tf.compat.v1.name_scope(name, "newton_implied_vol", [
      forwards, strikes, expiries, discount_factors, prices,
      initial_volatilities, option_signs, max_iterations, tolerance
  ]):

    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name="forwards")
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name="strikes")
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name="expiries")
    discount_factors = tf.convert_to_tensor(
        discount_factors, dtype=dtype, name="discount_factors")
    prices = tf.convert_to_tensor(prices, dtype=dtype, name="prices")
    # discounted_prices = discounts * prices
    initial_volatilities = tf.convert_to_tensor(
        initial_volatilities, dtype=dtype, name="initial_volatilities")
    option_signs = tf.convert_to_tensor(
        option_signs, dtype=dtype, name="option_signs")
    pricer = _make_black_objective_and_vega_func(prices, forwards, strikes,
                                                 expiries, option_signs,
                                                 discount_factors)
    results = newton_root_finder(
        pricer,
        initial_volatilities,
        max_iterations=max_iterations,
        tolerance=tolerance)
    return results
