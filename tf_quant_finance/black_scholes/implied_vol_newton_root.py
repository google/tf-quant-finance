# Lint as: python3
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

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_quant_finance.black_scholes import implied_vol_approximation as approx


def implied_vol(*,
                prices,
                strikes,
                expiries,
                spots=None,
                forwards=None,
                discount_factors=None,
                is_call_options=None,
                initial_volatilities=None,
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
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `prices`. The forwards to maturity. Either this argument or the `spots`
      must be supplied but both must not be supplied.
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
    initial_volatilities: A real `Tensor` of the same shape and dtype as
      `forwards`. The starting postions for Newton's method.
      Default value: None. If not supplied, the starting point is chosen using
        the Stefanica-Radoicic scheme. See `polya_approx.implied_vol` for
        details.
    tolerance: `float`. The root finder will stop where this tolerance is
      crossed.
    max_iterations: `int`. The maximum number of iterations of Newton's method.
    validate_args: A Python bool. If True, indicates that arguments should be
      checked for correctness before performing the computation. The checks
      performed are: (1) Forwards and strikes are positive. (2) The prices
        satisfy the arbitrage bounds (i.e. for call options, checks the
        inequality `max(F-K, 0) <= Price <= F` and for put options, checks that
        `max(K-F, 0) <= Price <= K`.). (3) Checks that the prices are not too
        close to the bounds. It is numerically unstable to compute the implied
        vols from options too far in the money or out of the money.
    dtype: `tf.Dtype` to use when converting arguments to `Tensor`s. If not
      supplied, the default TensorFlow conversion will take place. Note that
      this argument does not do any casting for `Tensor`s or numpy arrays.
      Default value: None.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'implied_vol' is used.
      Default value: None

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
      initial_volatilities = approx.implied_vol(
          prices=prices,
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          discount_factors=discount_factors,
          is_call_options=is_call_options,
          validate_args=validate_args)
    else:
      initial_volatilities = tf.convert_to_tensor(
          initial_volatilities, dtype=dtype, name='initial_volatilities')

    implied_vols, converged, failed = _newton_implied_vol(
        prices, strikes, expiries, forwards, discount_factors, is_call_options,
        initial_volatilities, tolerance, max_iterations)
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
      default_name='newton_root_finder',
      values=[initial_values, tolerance]):

    initial_values = tf.convert_to_tensor(
        initial_values, dtype=dtype, name='initial_values')

    starting_position = (tf.constant(0, dtype=tf.int32), initial_values,
                         tf.zeros_like(initial_values, dtype=tf.bool),
                         tf.math.is_nan(initial_values))

    def _condition(counter, parameters, converged, failed):
      del parameters
      early_stop = tf.reduce_all(converged | failed)
      return ~((counter >= max_iterations) | early_stop)

    def _updater(counter, parameters, converged, failed):
      """Updates each parameter via Newton's method."""
      values, gradients = value_and_grad_func(parameters)
      # values, _ = value_and_grad_func(parameters)
      # values_bump, _ = value_and_grad_func(parameters + 1e-6)
      # gradients = (values_bump - values) / 1e-6
      converged = tf.abs(values) < tolerance
      # Used to zero out updates to cells that have converged.
      update_mask = tf.cast(~converged, dtype=parameters.dtype)
      increment = -update_mask * values / gradients
      updated_parameters = parameters + increment
      failed = ~tf.math.is_finite(updated_parameters)

      return counter + 1, updated_parameters, converged, failed

    return tf.while_loop(_condition, _updater, starting_position)[1:]


def _newton_implied_vol(prices, strikes, expiries, forwards, discount_factors,
                        is_call_options, initial_volatilities, tolerance,
                        max_iterations):
  """Uses Newton's method to find Black Scholes implied volatilities of options.

  Finds the volatility implied under the Black Scholes option pricing scheme for
  a set of European options given observed market prices. The implied volatility
  is found via application of Newton's algorithm for locating the root of a
  differentiable function.

  The implmentation assumes that each cell in the supplied tensors corresponds
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
      `forwards`. The starting postions for Newton's method.
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
  pricer = _make_black_objective_and_vega_func(prices, forwards, strikes,
                                               expiries, is_call_options,
                                               discount_factors)
  results = newton_root_finder(
      pricer,
      initial_volatilities,
      max_iterations=max_iterations,
      tolerance=tolerance)
  return results


def _make_black_objective_and_vega_func(prices, forwards, strikes, expiries,
                                        is_call_options, discount_factors):
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
  dtype = prices.dtype
  phi = tfp.distributions.Normal(
      loc=tf.zeros(1, dtype=dtype), scale=tf.ones(1, dtype=dtype))
  # orientations will decide the normalization strategy.
  orientations = strikes >= forwards
  # normalization is the greater of strikes or forwards
  normalization = tf.where(orientations, strikes, forwards)
  normalized_prices = prices / normalization
  if discount_factors is not None:
    normalized_prices /= discount_factors
  else:
    discount_factors = tf.ones_like(normalized_prices)

  units = tf.ones_like(forwards)
  # y is 1 when strikes >= forwards and strikes/forwards otherwise
  y = tf.where(orientations, units, strikes / forwards)
  # x is forwards/strikes when strikes >= forwards and 1 otherwise
  x = tf.where(orientations, forwards / strikes, units)
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
    v = volatilities * sqrt_t
    d1 = (lnz / v + v / 2)
    d2 = d1 - v
    implied_prices = x * phi.cdf(d1) - y * phi.cdf(d2)
    if is_call_options is not None:
      put_prices = implied_prices - x + y
      implied_prices = tf.where(
          tf.broadcast_to(is_call_options, tf.shape(put_prices)),
          implied_prices, put_prices)
    vega = x * phi.prob(d1) * sqrt_t / discount_factors
    return implied_prices - normalized_prices, vega

  return _black_objective_and_vega
