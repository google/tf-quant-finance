# Lint as: python3
# Copyright 2021 Google LLC
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
"""Calibrates the approximated SABR model parameters using option prices."""

from dataclasses import dataclass
import tensorflow.compat.v2 as tf

from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math import optimizer
from tf_quant_finance.models.sabr.approximations import european_options
from tf_quant_finance.models.sabr.approximations.implied_volatility import SabrApproximationType
from tf_quant_finance.models.sabr.approximations.implied_volatility import SabrImpliedVolatilityType


@dataclass
class CalibrationResult:
  """Collection of calibrated SABR parameters.

  For a review of the SABR model and the conventions used, please see the
  docstring for `implied_volatility`, or for `calibration` below.

  Attributes:
    alpha: Rank-1 `Tensor`, specifying the initial volatility levels.
    beta: Rank-1 `Tensor`, specifying the exponents.
    volvol: Rank-1 `Tensor`, specifying the vol-vol parameters.
    rho: Rank-1 `Tensor`, specifying the correlations between the forward and
      the stochastic volatility.
  """
  alpha: tf.Tensor
  beta: tf.Tensor
  volvol: tf.Tensor
  rho: tf.Tensor


def calibration(*,
                prices,
                strikes,
                expiries,
                forwards,
                is_call_options,
                beta,
                nu,
                rho,
                volatility_type=SabrImpliedVolatilityType.LOGNORMAL,
                approximation_type=SabrApproximationType.HAGAN,
                alpha=None,
                alpha_lower_bound=None,
                alpha_upper_bound=None,
                calibrate_beta=False,
                beta_lower_bound=0.0,
                beta_upper_bound=1.0,
                nu_lower_bound=0.0,
                nu_upper_bound=1.0,
                rho_lower_bound=-1.0,
                rho_upper_bound=1.0,
                optimizer_fn=None,
                tolerance=1e-6,
                maximum_iterations=100,
                validate_args=False,
                dtype=None,
                name=None):
  """Calibrates the SABR model using European option prices.

  The SABR model specifies the risk neutral dynamics of the underlying as the
  following set of stochastic differential equations:

  ```
    dF = sigma F^beta dW_1
    dsigma = nu sigma dW_2
    dW1 dW2 = rho dt

    F(0) = f
    sigma(0) = alpha
  ```
  where F(t) represents the value of the forward price as a function of time,
  and sigma(t) is the volatility.

  Given a set of European option prices, this function estimates the SABR model
  parameters which best describe the input data. Calibration is done using the
  closed-form approximations for European option pricing.

  If volatility-based calibration is not selected, then the calibration is
  performed by minimizing the mean-squared-loss of the *log1p* of the input
  and estimated European options prices.

  #### Example

  ```python
  import tf_quant_finance as tff
  import tensorflow.compat.v2 as tf

  dtype = np.float64

  # Set some market conditions.
  observed_prices = np.array(
      [[20.09689284, 10.91953054, 4.25012702, 1.11561839, 0.20815853],
       [3.34813209, 6.03578711, 10.2874194, 16.26824328, 23.73850935]],
      dtype=dtype)
  strikes = np.array(
      [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
      dtype=dtype)
  expiries = np.array([[0.5], [1.0]], dtype=dtype)
  forwards = 100.0
  is_call_options = np.array([[True], [False]])

  # Calibrate the model.
  # In this example, we are calibrating a SABR model using the lognormal
  # volatility approximation for implied volatility, and we explicitly fix the
  # betas ourselves.
  beta = np.array([0.5, 0.5], dtype=dtype)
  models, is_converged, _ = tff.models.sabr.approximations.calibration(
      prices=observed_prices,
      strikes=strikes,
      expiries=expiries,
      forwards=forwards,
      is_call_options=is_call_options,
      beta=beta,
      calibrate_beta=False,
      nu=np.array([1.0, 1.0], dtype=dtype),
      nu_lower_bound=0.0,
      nu_upper_bound=10.0,
      rho=np.array([0.0, 0.0], dtype=dtype),
      rho_lower_bound=-0.75,
      rho_upper_bound=0.75,
      maximum_iterations=1000)

  # This will return two `SabrModel`s, where:
  # Model 1 has alpha = 1.5, beta = 0.5, volvol = 0.33, and rho = 0.1
  # Model 2 has alpha = 2.5, beta = 0.5, volvol = 0.66, and rho = -0.1

  ```

  Args:
    prices: Real `Tensor` of shape [batch_size, num_strikes] specifying the
      observed options prices. Here, `batch_size` refers to the number of SABR
      models calibrated in this invocation.
    strikes: Real `Tensor` of shape [batch_size, num_strikes] specifying the
      strike prices of the options.
    expiries: Real `Tensor` of shape compatible with [batch_size, num_strikes]
      specifying the options expiries.
    forwards: Real `Tensor` of shape compatible with [batch_size, num_strikes]
      specifying the observed forward prices/rates.
    is_call_options: Boolean `Tensor` of shape compatible with [batch_size,
      num_strikes] specifying whether or not the prices correspond to a call
      option (=True) or a put option (=False).
    beta: Real `Tensor` of shape [batch_size], specifying the initial estimate
      of the model `beta`. Values must satisify 0 <= `beta` <= 1
    nu: Real `Tensor` of shape [batch_size], specifying the initial estimate of
      the vol-vol parameter. Values must satisfy 0 <= `nu`.
    rho: Real `Tensor` of shape [batch_size], specifying the initial estimate of
      the correlation between the forward price and the volatility. Values must
      satisfy -1 < `rho` < 1.
    volatility_type: Either SabrImpliedVolatility.NORMAL or LOGNORMAL.
      Default value: `LOGNORMAL`
    approximation_type: Instance of `SabrApproxmationScheme`.
      Default value: `HAGAN`.
    alpha: Real `Tensor` of shape [batch_size], specifying the initial estimate
      of initial level of the volatility. Values must be strictly positive. If
      this is not provided, then an initial value will be estimated, along with
      lower and upper bounds.
      Default value: `None`, indicating that the routine should try to find a
        reasonable initial estimate.
    alpha_lower_bound: Real `Tensor` compatible with that of `alpha`, specifying
      the lower bound for the calibrated value. This is ignored if `alpha` is
      `None`.
      Default value: `None`.
    alpha_upper_bound: Real `Tensor` compatible with that of `alpha`, specifying
      the upper bound for the calibrated value. This is ignored if `alpha` is
      `None`.
      Default value: `None`.
    calibrate_beta: Boolean value indicating whether or not the `beta`
      parameters should be calibrated. If `True`, then the `beta_lower_bound`
      and `beta_upper_bound` must be specified. If `False`, then the model will
      use the values specified in `beta`.
      Default value: `False`.
    beta_lower_bound: Only used if `calibrate_beta` is True. Real `Tensor`
      compatible with that of `beta`, specifying the lower bound for the
      calibrated value.
      Default value: 0.0.
    beta_upper_bound: Only used if `calibrate_beta` is True. Real `Tensor`
      compatible with that of `beta`, specifying the upper bound for the
      calibrated value.
      Defalut value: 1.0
    nu_lower_bound: Real `Tensor` compatible with that of `nu`, specifying the
      lower bound for the calibrated value.
      Default value: 0.0.
    nu_upper_bound: Real `Tensor` compatible with that of `nu`, specifying the
      lower bound for the calibrated value.
      Default value: 1.0.
    rho_lower_bound: Real `Tensor` compatible with that of `rho`, specifying the
      lower bound for the calibrated value.
      Default value: -1.0.
    rho_upper_bound: Real `Tensor` compatible with that of `rho`, specifying the
      upper bound for the calibrated value.
      Default value: 1.0.
    optimizer_fn: Optional Python callable which implements the algorithm used
      to minimize the objective function during calibration. It should have
      the following interface: result =
        optimizer_fn(value_and_gradients_function, initial_position, tolerance,
        max_iterations) `value_and_gradients_function` is a Python callable that
        accepts a point as a real `Tensor` and returns a tuple of `Tensor`s of
        real dtype containing the value of the function and its gradient at that
        point. 'initial_position' is a real `Tensor` containing the starting
        point of the optimization, 'tolerance' is a real scalar `Tensor` for
        stopping tolerance for the procedure and `max_iterations` specifies the
        maximum number of iterations.
      `optimizer_fn` should return a namedtuple containing the items: `position`
        (a tensor containing the optimal value), `converged` (a boolean
        indicating whether the optimize converged according the specified
        criteria), `failed` (a boolean indicating if the optimization resulted
        in a failure), `num_iterations` (the number of iterations used), and
        `objective_value` ( the value of the objective function at the optimal
        value). The default value for `optimizer_fn` is None and conjugate
        gradient algorithm is used.
      Default value: `None` - indicating conjugate gradient minimizer.
    tolerance: Scalar `Tensor` of real dtype. The absolute tolerance for
      terminating the iterations.
      Default value: 1e-6.
    maximum_iterations: Scalar positive integer `Tensor`. The maximum number of
      iterations during the optimization.
      Default value: 100.
    validate_args: Boolean value indicating whether or not to validate the shape
      and values of the input arguments, at the potential expense of performance
      degredation.
      Defalut value: False.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None`, which means that default dtypes inferred by
        TensorFlow are used.
    name: String. The name to give to the ops created by this function.
      Default value: `None`, which maps to the default name 'sabr_calibration'.

  Returns:
    A Tuple of three elements. The first is a `CalibrationResult` holding the
    calibrated alpha, beta, volvol, and rho, where alpha[i] corresponds to the
    calibrated `alpha` of the i-th batch, etc.
    The second and third elements contains the optimization status
    (whether the optimization algorithm succeeded in finding the optimal point
    based on the specified convergance criteria) and the number of iterations
    performed.
  """
  name = name or 'sabr_calibration'
  with tf.name_scope(name):
    prices = tf.convert_to_tensor(prices, dtype=dtype, name='prices')
    dtype = dtype or prices.dtype
    batch_size = tf.shape(prices)[0]

    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='expiries')
    is_call_options = tf.convert_to_tensor(
        is_call_options, name='is_call', dtype=tf.bool)

    if optimizer_fn is None:
      optimizer_fn = optimizer.conjugate_gradient_minimize

    def _price_transform(x):
      return tf.math.log1p(x)

    if alpha is None:
      # We set the initial value of alpha to be s.t. alpha * F^(beta - 1) is
      # on the order of 10%.
      initial_alpha_guess = tf.math.reduce_mean(forwards)
      alpha = tf.fill(dims=[batch_size], value=initial_alpha_guess)
      alpha = tf.pow(alpha, 1.0 - beta) * 0.1
      alpha_lower_bound = alpha * 0.1
      alpha_upper_bound = alpha * 10.0
    else:
      alpha_lower_bound = tf.convert_to_tensor(alpha_lower_bound, dtype=dtype)
      alpha_upper_bound = tf.convert_to_tensor(alpha_upper_bound, dtype=dtype)

    alpha = _assert_parameter_valid(
        validate_args,
        alpha,
        shape=[batch_size],
        lower_bound=alpha_lower_bound,
        upper_bound=alpha_upper_bound,
        message='`alpha` is invalid!')
    initial_alpha = _to_unconstrained(alpha, alpha_lower_bound,
                                      alpha_upper_bound)

    nu_lower_bound = tf.convert_to_tensor(nu_lower_bound, dtype=dtype)
    nu_upper_bound = tf.convert_to_tensor(nu_upper_bound, dtype=dtype)
    nu = _assert_parameter_valid(
        validate_args,
        nu,
        shape=[batch_size],
        lower_bound=nu_lower_bound,
        upper_bound=nu_upper_bound,
        message='`nu` is invalid!')
    initial_nu = _to_unconstrained(nu, nu_lower_bound, nu_upper_bound)

    rho_lower_bound = tf.convert_to_tensor(rho_lower_bound, dtype=dtype)
    rho_upper_bound = tf.convert_to_tensor(rho_upper_bound, dtype=dtype)
    rho = _assert_parameter_valid(
        validate_args,
        rho,
        shape=[batch_size],
        lower_bound=rho_lower_bound,
        upper_bound=rho_upper_bound,
        message='`rho` is invalid!')
    initial_rho = _to_unconstrained(rho, rho_lower_bound, rho_upper_bound)

    beta = tf.convert_to_tensor(beta, dtype=dtype)
    beta_lower_bound = tf.convert_to_tensor(beta_lower_bound, dtype=dtype)
    beta_upper_bound = tf.convert_to_tensor(beta_upper_bound, dtype=dtype)
    beta = _assert_parameter_valid(
        validate_args,
        beta,
        shape=[batch_size],
        lower_bound=beta_lower_bound,
        upper_bound=beta_upper_bound,
        message='`beta` is invalid!')
    if calibrate_beta:
      initial_beta = _to_unconstrained(beta, beta_lower_bound, beta_upper_bound)
      initial_x = tf.concat(
          [initial_alpha, initial_nu, initial_rho, initial_beta], axis=0)
    else:
      initial_x = tf.concat([initial_alpha, initial_nu, initial_rho], axis=0)

    def _get_alpha_nu_rho(packed_optimizer_params):
      a = _to_constrained(
          packed_optimizer_params[0 * batch_size:1 * batch_size],
          alpha_lower_bound, alpha_upper_bound)
      v = _to_constrained(
          packed_optimizer_params[1 * batch_size:2 * batch_size],
          nu_lower_bound, nu_upper_bound)
      r = _to_constrained(
          packed_optimizer_params[2 * batch_size:3 * batch_size],
          rho_lower_bound, rho_upper_bound)
      return (a, v, r)

    def _get_beta(packed_optimizer_params):
      if calibrate_beta:
        return _to_constrained(
            packed_optimizer_params[3 * batch_size:4 * batch_size],
            beta_lower_bound, beta_upper_bound)
      else:
        return beta

    scaled_target_values = _price_transform(prices)

    @make_val_and_grad_fn
    def loss_function(x):
      """Loss function for the optimization."""
      a, v, r = _get_alpha_nu_rho(x)
      b = _get_beta(x)
      values = european_options.option_price(
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call_options,
          alpha=tf.expand_dims(a, axis=-1),
          beta=tf.expand_dims(b, axis=-1),
          nu=tf.expand_dims(v, axis=-1),
          rho=tf.expand_dims(r, axis=-1),
          volatility_type=volatility_type,
          approximation_type=approximation_type,
          dtype=dtype)

      scaled_values = _price_transform(values)
      return tf.math.reduce_mean((scaled_values - scaled_target_values)**2)

    optimization_result = optimizer_fn(
        loss_function,
        initial_position=initial_x,
        tolerance=tolerance,
        max_iterations=maximum_iterations)

    calibration_parameters = optimization_result.position
    calibrated_alpha, calibrated_nu, calibrated_rho = _get_alpha_nu_rho(
        calibration_parameters)
    calibrated_beta = _get_beta(calibration_parameters)

    return (CalibrationResult(
        alpha=calibrated_alpha,
        beta=calibrated_beta,
        volvol=calibrated_nu,
        rho=calibrated_rho), optimization_result.converged,
            optimization_result.num_iterations)


def _scale(x, lb, ub):
  """Scales the values to be normalized to [lb, ub]."""
  return (x - lb) / (ub - lb)


def _to_unconstrained(x, lb, ub):
  """Scale and apply inverse-sigmoid."""
  x = _scale(x, lb, ub)
  return -tf.math.log((1.0 - x) / x)


def _to_constrained(x, lb, ub):
  """Sigmoid and unscale."""
  x = 1.0 / (1.0 + tf.math.exp(-x))
  return x * (ub - lb) + lb


def _assert_parameter_valid(validate_args, x, shape, lower_bound, upper_bound,
                            message):
  """Helper to check that the input parameter is valid."""
  if validate_args:
    with tf.control_dependencies([
        tf.debugging.assert_equal(tf.shape(x), shape, message=message),
        tf.debugging.assert_greater_equal(x, lower_bound, message=message),
        tf.debugging.assert_less_equal(x, upper_bound, message=message),
    ]):
      return tf.identity(x)
  else:
    return x
