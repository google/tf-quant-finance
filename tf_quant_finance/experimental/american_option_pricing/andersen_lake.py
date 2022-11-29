"""Calculating American option prices with Andersen-Lake approximation."""

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.black_scholes import vanilla_prices
from tf_quant_finance.experimental.american_option_pricing import common
from tf_quant_finance.experimental.american_option_pricing import exercise_boundary
from tf_quant_finance.math.integration import gauss_kronrod


calculate_exercise_boundary = exercise_boundary.exercise_boundary
standard_normal_cdf = common.standard_normal_cdf
d_plus = common.d_plus
d_minus = common.d_minus
machine_eps = common.machine_eps
divide_with_positive_denominator = common.divide_with_positive_denominator


def andersen_lake(
    *,
    volatilities: types.RealTensor,
    strikes: types.RealTensor,
    expiries: types.RealTensor,
    spots: types.RealTensor = None,
    forwards: types.RealTensor = None,
    discount_rates: types.RealTensor = None,
    discount_factors: types.RealTensor = None,
    dividend_rates: types.RealTensor = None,
    is_call_options: types.BoolTensor = None,
    grid_num_points: int = 10,
    integration_num_points_kronrod: int = 31,
    integration_num_points_legendre: int = 32,
    max_iterations_exercise_boundary: int = 30,
    max_depth_kronrod: int = 30,
    tolerance_exercise_boundary: types.RealTensor = 1e-8,
    tolerance_kronrod: types.RealTensor = 1e-8,
    dtype: tf.DType = None,
    name: str = None
) -> types.RealTensor:
  """Computes American option prices using the Andersen-Lake approximation.

  #### Example

  ```python
  volatilities = [0.1, 0.15]
  strikes = [3, 2]
  expiries = [1, 2]
  spots = [8.0, 9.0]
  discount_rates = [0.01, 0.02]
  dividend_rates = [0.01, 0.02]
  is_call_options = [True, False]
  grid_num_points = 40
  integration_num_points_kronrod = 31
  integration_num_points_legendre = 32
  max_iterations_exercise_boundary = 500
  max_depth_kronrod = 50
  tolerance_exercise_boundary = 1e-11
  tolerance_kronrod = 1e-11
  computed_prices = andersen_lake(
      volatilities=volatilities,
      strikes=strikes,
      expiries=expiries,
      spots=spots,
      discount_rates=discount_rates,
      dividend_rates=dividend_rates,
      is_call_options=is_call_options,
      grid_num_points=grid_num_points,
      integration_num_points_kronrod=integration_num_points_kronrod,
      integration_num_points_legendre=integration_num_points_legendre,
      max_iterations_exercise_boundary=max_iterations_exercise_boundary,
      max_depth_kronrod=max_depth_kronrod,
      tolerance_exercise_boundary=tolerance_exercise_boundary,
      tolerance_kronrod=tolerance_kronrod
      dtype=tf.float64)
  # Expected print output of computed prices:
  # [4.950249e+00, 7.768513e-14]
  ```

  #### References:
  [1] Leif Andersen, Mark Lake and Dimitri Offengenden. High-performance
  American option pricing. 2015
  https://engineering.nyu.edu/sites/default/files/2019-03/Carr-adjusting-exponential-levy-models.pdf#page=46

  Args:
    volatilities: Real `Tensor` of any real dtype and shape `[num_options]`.
      The volatilities to expiry of the options to price.
    strikes: A real `Tensor` of the same dtype and same shape as `volatilities`.
      The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and same shape as `volatilities`.
      The expiry of each option. The units should be such that
      `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of same shape as `volatilities`. The current spot
      price of the underlying. Either this argument or the `forwards` (but not
      both) must be supplied.
    forwards: A real `Tensor` of same shape as `volatilities`. The forwards to
      maturity. Either this argument or the `spots` must be supplied but both
      must not be supplied.
    discount_rates: An optional real `Tensor` of same shape and dtype as the
      `volatilities`. If not `None`, discount factors are calculated as e^(-rT),
      where r are the discount rates, or risk free rates.
      Default value: `None`, which maps to `-log(discount_factors) / expiries`
        if `discount_factors` is not `None`, or maps to `0` when
        `discount_factors` is also `None`.
    discount_factors: An optional real `Tensor` of same shape and dtype as the
      `volatilities`. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with `discount_rate`. If neither is
      given, no discounting is applied (i.e. the undiscounted option price is
      returned). If `spots` is supplied and `discount_factors` is not `None`
      then this is also used to compute the forwards to expiry.
      Default value: `None`.
    dividend_rates: An optional real `Tensor` of same shape and dtype as the
      `volatilities`. The continuous dividend rate on the underliers. May be
      negative (to indicate costs of holding the underlier).
      Default value: `None`, equivalent to zero dividends.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    grid_num_points: positive `int`. The number of equidistant points to divide
      the values given in `expiries` into in the grid of `tau_grid`.
      Default value: 10.
    integration_num_points_kronrod: positive `int`. The number of points used in
      the Gauss-Kronrod integration approximation method used for
      calculating the option prices.
      Default value: 31.
    integration_num_points_legendre: positive `int`. The number of points used
      in the Gauss-Legendre integration approximation method used for
      calculating the exercise boundary function used for pricing the options.
      Default value: 32.
    max_iterations_exercise_boundary: positive `int`. Maximum number of
      iterations for calculating the exercise boundary if it doesn't converge
      earlier.
      Default value: 30.
    max_depth_kronrod: positive `int`. Maximum number of iterations for
      calculating the Gauss-Kronrod integration approximation.
      Default value: 30.
    tolerance_exercise_boundary: Positive scalar `Tensor`. The tolerance for the
      convergence of calculating the exercise boundary function.
      Default value: 1e-8.
    tolerance_kronrod: Positive scalar `Tensor`. The tolerance for the
      convergence of calculating the Gauss-Kronrod integration approximation.
      Default value: 1e-8.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by
        TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name `andersen_lake`.

  Returns:
    `Tensor` of shape `[num_options]`, containing the calculated American option
    prices.

  Raises:
    ValueError:
      (a) If both `forwards` and `spots` are supplied or if neither is supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  if (discount_rates is not None) and (discount_factors is not None):
    raise ValueError('At most one of discount_rates and discount_factors may '
                     'be supplied')
  with tf.name_scope(name or 'andersen_lake'):
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    dtype = volatilities.dtype  # This dtype should be common for all inputs
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    if discount_rates is not None:
      discount_rates = tf.convert_to_tensor(
          discount_rates, dtype=dtype, name='discount_rates')
    elif discount_factors is not None:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')
      # For expiries == 0 discount_rates is irrelevant for the option price,
      # so we can set it to 0.
      discount_rates = tf.math.divide_no_nan(-tf.math.log(discount_factors),
                                             expiries)
    else:
      discount_rates = tf.constant([0.0], dtype=dtype, name='discount_rates')

    if dividend_rates is not None:
      dividend_rates = tf.convert_to_tensor(
          dividend_rates, dtype=dtype, name='dividend_rates')
    else:
      dividend_rates = tf.constant([0.0], dtype=dtype, name='dividend_rates')
    # Set forwards and spots
    if forwards is not None:
      spots = tf.convert_to_tensor(
          forwards * tf.exp(-(discount_rates - dividend_rates) * expiries),
          dtype=dtype,
          name='spots')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
    if is_call_options is not None:
      is_call_options = tf.convert_to_tensor(
          is_call_options, dtype=tf.bool, name='is_call_options')
    else:
      is_call_options = tf.constant(True, name='is_call_options')

    # Shape [num_options]
    am_prices = _andersen_lake(
        sigma=volatilities,
        k_temp=strikes,
        tau=expiries,
        s_temp=spots,
        r_temp=discount_rates,
        q_temp=dividend_rates,
        is_call_options=is_call_options,
        grid_num_points=grid_num_points,
        integration_num_points_kronrod=integration_num_points_kronrod,
        integration_num_points_legendre=integration_num_points_legendre,
        max_iterations_exercise_boundary=max_iterations_exercise_boundary,
        max_depth_kronrod=max_depth_kronrod,
        tolerance_exercise_boundary=tolerance_exercise_boundary,
        tolerance_kronrod=tolerance_kronrod,
        dtype=dtype)
    # Shape [num_options]
    return am_prices


def _andersen_lake(*, sigma, k_temp, tau, s_temp, r_temp, q_temp,
                   is_call_options, grid_num_points,
                   integration_num_points_kronrod,
                   integration_num_points_legendre,
                   max_iterations_exercise_boundary, max_depth_kronrod,
                   tolerance_exercise_boundary, tolerance_kronrod, dtype):
  """Computes American option prices using the Andersen-Lake formula."""
  # Shape [num_options]
  eu_prices = vanilla_prices.option_price(
      volatilities=sigma,
      strikes=k_temp,
      expiries=tau,
      spots=s_temp,
      discount_rates=r_temp,
      dividend_rates=q_temp,
      is_call_options=is_call_options,
      dtype=dtype)

  # Some alterations depeneding on whether it is a call or put option:
  # https://www.researchgate.net/publication/243657048_A_Parity_Result_for_American_Options
  # For call options spots (s) and strikes (k) need to be swapped
  k = tf.where(is_call_options, s_temp, k_temp)
  s = tf.where(is_call_options, k_temp, s_temp)
  # For call options discount rates (r) and dividend rates (q) need to be
  # swapped
  r = tf.where(is_call_options, q_temp, r_temp)
  q = tf.where(is_call_options, r_temp, q_temp)

  # Shape [num_options, grid_num_points]
  tau_grid = tf.linspace(tau / grid_num_points, tau, grid_num_points, axis=-1)
  epsilon = machine_eps(dtype)
  # Where r == q == 0 the exercise boundary should be infinite in theory, but in
  # practice it is numerically unstable, therefore we set r == q == 0.1 just for
  # calculating the exercise boundary, so that it converges.
  # The exercise boundary will be irrelevant for the final result in this case.
  r_e_b = tf.where(
      tf.math.abs(r) < epsilon,
      tf.where(tf.math.abs(q) < epsilon, tf.constant(0.1, dtype=dtype), r), r)
  q_e_b = tf.where(
      tf.math.abs(q) < epsilon,
      tf.where(tf.math.abs(r) < epsilon, tf.constant(0.1, dtype=dtype), q), q)
  exercise_boundary_fn = calculate_exercise_boundary(
      tau_grid, k, r_e_b, q_e_b, sigma, max_iterations_exercise_boundary,
      tolerance_exercise_boundary, integration_num_points_legendre, dtype)
  # Shape [num_options, 1, 1]
  k_exp = k[:, tf.newaxis, tf.newaxis]
  r_exp = r[:, tf.newaxis, tf.newaxis]
  q_exp = q[:, tf.newaxis, tf.newaxis]
  sigma_exp = sigma[:, tf.newaxis, tf.newaxis]
  s_exp = s[:, tf.newaxis, tf.newaxis]
  tau_exp = tau[:, tf.newaxis, tf.newaxis]

  def get_ratio(u):
    u_shape = utils.get_shape(u)
    # Shape [num_options, n * integration_num_points_kronrod]
    # Need reshaping because exercise_boundary_fn expects a 2-dimensional input.
    u_reshaped = tf.reshape(u, [u_shape[0], u_shape[1] * u_shape[2]])
    # Shape [num_options, n, integration_num_points_kronrod]
    return divide_with_positive_denominator(
        s_exp,
        tf.reshape(
            exercise_boundary_fn(u_reshaped),
            [u_shape[0], u_shape[1], u_shape[2]]))

  def func_1(u):
    # Shape [num_options, n, integration_num_points_kronrod]
    ratio = get_ratio(u)
    norm = standard_normal_cdf(
        -d_minus(tau_exp - u, ratio, r_exp, q_exp, sigma_exp))
    return r_exp * k_exp * tf.math.exp(-r_exp * (tau_exp - u)) * norm

  # Shape [num_options]
  term1 = gauss_kronrod(
      func=func_1,
      lower=tf.zeros_like(tau),
      upper=tau,
      tolerance=tolerance_kronrod,
      num_points=integration_num_points_kronrod,
      max_depth=max_depth_kronrod,
      dtype=dtype)

  def func_2(u):
    # Shape [num_options, n, integration_num_points_kronrod]
    ratio = get_ratio(u)
    norm = standard_normal_cdf(
        -d_plus(tau_exp - u, ratio, r_exp, q_exp, sigma_exp))
    return q_exp * s_exp * tf.math.exp(-q_exp * (tau_exp - u)) * norm

  # Shape [num_options]
  term2 = gauss_kronrod(
      func=func_2,
      lower=tf.zeros_like(tau),
      upper=tau,
      tolerance=tolerance_kronrod,
      num_points=integration_num_points_kronrod,
      max_depth=max_depth_kronrod,
      dtype=dtype)
  # Shape [num_options]
  return eu_prices + term1 - term2
