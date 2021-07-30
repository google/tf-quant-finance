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

from typing import Callable, Tuple

import tensorflow.compat.v2 as tf

from tf_quant_finance import black_scholes
from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.black_scholes.implied_vol_utils import UnderlyingDistribution
from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math import optimizer
from tf_quant_finance.models.sabr import approximations
from tf_quant_finance.models.sabr.approximations.implied_volatility import SabrApproximationType
from tf_quant_finance.models.sabr.approximations.implied_volatility import SabrImpliedVolatilityType


__all__ = [
    'CalibrationResult',
    'calibration',
]


@utils.dataclass
class CalibrationResult:
  """Collection of calibrated SABR parameters.

  For a review of the SABR model and the conventions used, please see the
  docstring for `SABRModel`, or for `calibration` below.

  Attributes:
    alpha: Rank-1 `Tensor` specifying the initial volatility levels.
    beta: Rank-1 `Tensor` specifying the exponents.
    volvol: Rank-1 `Tensor` specifying the vol-vol parameters.
    rho: Rank-1 `Tensor` specifying the correlations between the forward and
      the stochastic volatility.
  """
  alpha: types.RealTensor
  beta: types.RealTensor
  volvol: types.RealTensor
  rho: types.RealTensor


def calibration(
    *,
    prices: types.RealTensor,
    strikes: types.RealTensor,
    expiries: types.RealTensor,
    forwards: types.RealTensor,
    is_call_options: types.BoolTensor,
    beta: types.RealTensor,
    volvol: types.RealTensor,
    rho: types.RealTensor,
    volatility_type: SabrImpliedVolatilityType = None,
    approximation_type: SabrApproximationType = None,
    volatility_based_calibration: bool = True,
    alpha: types.RealTensor = None,
    alpha_lower_bound: types.RealTensor = None,
    alpha_upper_bound: types.RealTensor = None,
    calibrate_beta: bool = False,
    beta_lower_bound: types.RealTensor = 0.0,
    beta_upper_bound: types.RealTensor = 1.0,
    volvol_lower_bound: types.RealTensor = 0.0,
    volvol_upper_bound: types.RealTensor = 1.0,
    rho_lower_bound: types.RealTensor = -1.0,
    rho_upper_bound: types.RealTensor = 1.0,
    optimizer_fn: Callable[..., types.RealTensor] = None,
    tolerance: types.RealTensor = 1e-6,
    maximum_iterations: types.RealTensor = 100,
    validate_args: bool = False,
    dtype: tf.DType = None,
    name: str = None) -> Tuple[CalibrationResult,
                               types.BoolTensor,
                               types.IntTensor]:
  """Calibrates the SABR model using European option prices.

  The SABR model specifies the risk neutral dynamics of the underlying as the
  following set of stochastic differential equations:

  ```
    dF = sigma F^beta dW_1
    dsigma = volvol sigma dW_2
    dW1 dW2 = rho dt

    F(0) = f
    sigma(0) = alpha
  ```
  where F(t) represents the value of the forward price as a function of time,
  and sigma(t) is the volatility.

  Given a set of European option prices, this function estimates the SABR model
  parameters which best describe the input data. Calibration is done using the
  closed-form approximations for European option pricing.

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
      volvol=np.array([1.0, 1.0], dtype=dtype),
      volvol_lower_bound=0.0,
      volvol_upper_bound=10.0,
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
      of the model `beta`. Values must satisfy 0 <= `beta` <= 1
    volvol: Real `Tensor` of shape [batch_size], specifying the initial estimate
      of the vol-vol parameter. Values must satisfy `0 <= volvol`.
    rho: Real `Tensor` of shape [batch_size], specifying the initial estimate of
      the correlation between the forward price and the volatility. Values must
      satisfy -1 < `rho` < 1.
    volatility_type: Either SabrImpliedVolatility.NORMAL or LOGNORMAL.
      Default value: `None` which maps to `LOGNORMAL`
    approximation_type: Instance of `SabrApproxmationScheme`.
      Default value: `None` which maps to `HAGAN`.
    volatility_based_calibration: Boolean. If `True`, then the options prices
      are first converted to implied volatilities, and the calibration is then
      performed by minimizing the difference between input implied volatilities
      and the model implied volatilities. Otherwise, the calibration is
      performed by minimizing the mean-squared-loss of the *log1p* of the input
      and estimated European options prices.
      Default value: True
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
      Default value: 1.0
    volvol_lower_bound: Real `Tensor` compatible with that of `volvol`,
      specifying the lower bound for the calibrated value.
      Default value: 0.0.
    volvol_upper_bound: Real `Tensor` compatible with that of `volvol`,
      specifying the lower bound for the calibrated value.
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
      Default value: `None` - indicating LBFGS minimizer.
    tolerance: Scalar `Tensor` of real dtype. The absolute tolerance for
      terminating the iterations.
      Default value: 1e-6.
    maximum_iterations: Scalar positive integer `Tensor`. The maximum number of
      iterations during the optimization.
      Default value: 100.
    validate_args: Boolean value indicating whether or not to validate the shape
      and values of the input arguments, at the potential expense of performance
      degredation.
      Default value: False.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None`, which means that default dtypes inferred by
        TensorFlow are used.
    name: String. The name to give to the ops created by this function.
      Default value: `None`, which maps to the default name 'sabr_calibration'.

  Returns:
    A Tuple of three elements:
    * The first is a `CalibrationResult` holding the calibrated alpha, beta,
      volvol, and rho, where alpha[i] corresponds to the calibrated `alpha` of
      the i-th batch, etc.
    * A `Tensor` of optimization status for each batch element (whether the
      optimization algorithm has found the optimal point based on the specified
      convergance criteria).
    * A `Tensor` containing the number of iterations performed by the
      optimization algorithm.
  """
  if approximation_type is None:
    approximation_type = SabrApproximationType.HAGAN
  if volatility_type is None:
    volatility_type = SabrImpliedVolatilityType.LOGNORMAL
  name = name or 'sabr_calibration'
  with tf.name_scope(name):
    prices = tf.convert_to_tensor(prices, dtype=dtype, name='prices')
    dtype = dtype or prices.dtype

    # Extract batch shape
    batch_shape = prices.shape.as_list()[:-1]
    if None in batch_shape:
      batch_shape = tf.shape(prices)[:-1]

    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    is_call_options = tf.convert_to_tensor(
        is_call_options, name='is_call', dtype=tf.bool)

    if optimizer_fn is None:
      optimizer_fn = optimizer.lbfgs_minimize

    beta = tf.convert_to_tensor(beta, dtype=dtype)
    beta_lower_bound = tf.convert_to_tensor(beta_lower_bound, dtype=dtype)
    beta_upper_bound = tf.convert_to_tensor(beta_upper_bound, dtype=dtype)
    beta = _assert_parameter_valid(
        validate_args,
        beta,
        lower_bound=beta_lower_bound,
        upper_bound=beta_upper_bound,
        message='`beta` is invalid!')
    # Broadcast beta to the correct batch shape
    beta = tf.broadcast_to(beta, batch_shape, name='broadcast_beta')

    if alpha is None:
      # We set the initial value of alpha to be s.t. alpha * F^(beta - 1) is
      # on the order of 10%.
      if forwards.shape.rank == 0:
        forwards = forwards[tf.newaxis]
      # Shape compatible with batch_shape
      alpha = tf.math.reduce_mean(forwards, axis=-1)
      alpha = tf.pow(alpha, 1.0 - beta) * 0.1
      alpha_lower_bound = alpha * 0.1
      alpha_upper_bound = alpha * 10.0
    else:
      alpha_lower_bound = tf.convert_to_tensor(alpha_lower_bound, dtype=dtype)
      alpha_upper_bound = tf.convert_to_tensor(alpha_upper_bound, dtype=dtype)

    alpha = _assert_parameter_valid(
        validate_args,
        alpha,
        lower_bound=alpha_lower_bound,
        upper_bound=alpha_upper_bound,
        message='`alpha` is invalid!')
    # Broadcast alpha to the correct batch shape
    # alpha += tf.zeros(batch_shape, dtype=alpha.dtype)
    initial_alpha = _to_unconstrained(alpha, alpha_lower_bound,
                                      alpha_upper_bound)

    volvol_lower_bound = tf.convert_to_tensor(volvol_lower_bound, dtype=dtype)
    volvol_upper_bound = tf.convert_to_tensor(volvol_upper_bound, dtype=dtype)
    volvol = _assert_parameter_valid(
        validate_args,
        volvol,
        lower_bound=volvol_lower_bound,
        upper_bound=volvol_upper_bound,
        message='`volvol` is invalid!')
    # Broadcast volvol to the correct batch shape
    volvol = tf.broadcast_to(volvol, batch_shape, name='broadcast_volvol')
    initial_volvol = _to_unconstrained(volvol, volvol_lower_bound,
                                       volvol_upper_bound)

    rho_lower_bound = tf.convert_to_tensor(rho_lower_bound, dtype=dtype)
    rho_upper_bound = tf.convert_to_tensor(rho_upper_bound, dtype=dtype)
    rho = _assert_parameter_valid(
        validate_args,
        rho,
        lower_bound=rho_lower_bound,
        upper_bound=rho_upper_bound,
        message='`rho` is invalid!')
    # Broadcast rho to the correct batch shape
    rho = tf.broadcast_to(rho, batch_shape, name='broadcast_rho')
    initial_rho = _to_unconstrained(rho, rho_lower_bound, rho_upper_bound)

    # Construct initial state for the optimizer
    if calibrate_beta:
      initial_beta = _to_unconstrained(beta, beta_lower_bound, beta_upper_bound)
      # Shape `batch_shape + [4]`
      initial_x = tf.stack(
          [initial_alpha, initial_volvol, initial_rho, initial_beta], axis=-1)
    else:
      # Shape `batch_shape + [3]`
      initial_x = tf.stack([initial_alpha, initial_volvol, initial_rho],
                           axis=-1)

    optimizer_arg_handler = _OptimizerArgHandler(
        alpha_lower_bound=alpha_lower_bound,
        alpha_upper_bound=alpha_upper_bound,
        volvol_lower_bound=volvol_lower_bound,
        volvol_upper_bound=volvol_upper_bound,
        rho_lower_bound=rho_lower_bound,
        rho_upper_bound=rho_upper_bound,
        calibrate_beta=calibrate_beta,
        beta=beta,
        beta_lower_bound=beta_lower_bound,
        beta_upper_bound=beta_upper_bound)

    if volatility_based_calibration:
      loss_function = _get_loss_for_volatility_based_calibration(
          prices=prices,
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call_options,
          volatility_type=volatility_type,
          approximation_type=approximation_type,
          dtype=dtype,
          optimizer_arg_handler=optimizer_arg_handler)

    else:  # Price based calibration.
      loss_function = _get_loss_for_price_based_calibration(
          prices=prices,
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call_options,
          volatility_type=volatility_type,
          approximation_type=approximation_type,
          dtype=dtype,
          optimizer_arg_handler=optimizer_arg_handler)
    optimization_result = optimizer_fn(
        loss_function,
        initial_position=initial_x,
        tolerance=tolerance,
        max_iterations=maximum_iterations)

    calibration_parameters = optimization_result.position
    calibrated_alpha = optimizer_arg_handler.get_alpha(calibration_parameters)
    calibrated_volvol = optimizer_arg_handler.get_volvol(calibration_parameters)
    calibrated_rho = optimizer_arg_handler.get_rho(calibration_parameters)
    calibrated_beta = optimizer_arg_handler.get_beta(calibration_parameters)

    return (CalibrationResult(
        alpha=calibrated_alpha,
        beta=calibrated_beta,
        volvol=calibrated_volvol,
        rho=calibrated_rho), optimization_result.converged,
            optimization_result.num_iterations)


def _get_loss_for_volatility_based_calibration(*, prices, strikes, expiries,
                                               forwards, is_call_options,
                                               volatility_type,
                                               approximation_type, dtype,
                                               optimizer_arg_handler):
  """Creates a loss function to be used in volatility-based calibration."""
  if volatility_type == SabrImpliedVolatilityType.LOGNORMAL:
    underlying_distribution = UnderlyingDistribution.LOG_NORMAL
  elif volatility_type == SabrImpliedVolatilityType.NORMAL:
    underlying_distribution = UnderlyingDistribution.NORMAL
  else:
    raise ValueError('Unsupported `volatility_type`!')
  target_implied_vol = black_scholes.implied_vol(
      prices=prices,
      strikes=strikes,
      expiries=expiries,
      forwards=forwards,
      is_call_options=is_call_options,
      underlying_distribution=underlying_distribution)

  @make_val_and_grad_fn
  def loss_function(x):
    """Loss function for vol-based optimization."""
    candidate_alpha = optimizer_arg_handler.get_alpha(x)
    candidate_volvol = optimizer_arg_handler.get_volvol(x)
    candidate_rho = optimizer_arg_handler.get_rho(x)
    candidate_beta = optimizer_arg_handler.get_beta(x)

    implied_vol = approximations.implied_volatility(
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        alpha=tf.expand_dims(candidate_alpha, axis=-1),
        beta=tf.expand_dims(candidate_beta, axis=-1),
        volvol=tf.expand_dims(candidate_volvol, axis=-1),
        rho=tf.expand_dims(candidate_rho, axis=-1),
        volatility_type=volatility_type,
        approximation_type=approximation_type,
        dtype=dtype)
    # shape `batch_shape`
    return tf.math.reduce_mean((target_implied_vol - implied_vol)**2, axis=-1)

  return loss_function


def _get_loss_for_price_based_calibration(*, prices, strikes, expiries,
                                          forwards, is_call_options,
                                          volatility_type, approximation_type,
                                          dtype, optimizer_arg_handler):
  """Creates a loss function to be used in volatility-based calibration."""

  def _price_transform(x):
    return tf.math.log1p(x)

  scaled_target_values = _price_transform(prices)

  @make_val_and_grad_fn
  def loss_function(x):
    """Loss function for the price-based optimization."""
    candidate_alpha = optimizer_arg_handler.get_alpha(x)
    candidate_volvol = optimizer_arg_handler.get_volvol(x)
    candidate_rho = optimizer_arg_handler.get_rho(x)
    candidate_beta = optimizer_arg_handler.get_beta(x)

    values = approximations.european_option_price(
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        alpha=tf.expand_dims(candidate_alpha, axis=-1),
        beta=tf.expand_dims(candidate_beta, axis=-1),
        volvol=tf.expand_dims(candidate_volvol, axis=-1),
        rho=tf.expand_dims(candidate_rho, axis=-1),
        volatility_type=volatility_type,
        approximation_type=approximation_type,
        dtype=dtype)

    scaled_values = _price_transform(values)
    return tf.math.reduce_mean((scaled_values - scaled_target_values)**2,
                               axis=-1)

  return loss_function


@utils.dataclass
class _OptimizerArgHandler:
  """Handles the packing/transformation of estimated parameters."""
  alpha_lower_bound: types.RealTensor
  alpha_upper_bound: types.RealTensor
  volvol_lower_bound: types.RealTensor
  volvol_upper_bound: types.RealTensor
  rho_lower_bound: types.RealTensor
  rho_upper_bound: types.RealTensor
  calibrate_beta: bool
  beta: types.RealTensor
  beta_lower_bound: types.RealTensor
  beta_upper_bound: types.RealTensor

  def get_alpha(self,
                packed_optimizer_args: types.RealTensor) -> types.RealTensor:
    """Unpack and return the alpha parameter."""
    alpha = packed_optimizer_args[..., 0]
    return _to_constrained(
        alpha,
        self.alpha_lower_bound, self.alpha_upper_bound)

  def get_volvol(self,
                 packed_optimizer_args: types.RealTensor) -> types.RealTensor:
    """Unpack and return the volvol parameter."""
    volvol = packed_optimizer_args[..., 1]
    return _to_constrained(
        volvol,
        self.volvol_lower_bound, self.volvol_upper_bound)

  def get_rho(self,
              packed_optimizer_args: types.RealTensor) -> types.RealTensor:
    """Unpack and return the rho parameter."""
    rho = packed_optimizer_args[..., 2]
    return _to_constrained(
        rho,
        self.rho_lower_bound, self.rho_upper_bound)

  def get_beta(self,
               packed_optimizer_args: types.RealTensor) -> types.RealTensor:
    """Unpack and return the beta parameter."""
    if self.calibrate_beta:
      beta = packed_optimizer_args[..., -1]
      return _to_constrained(
          beta,
          self.beta_lower_bound, self.beta_upper_bound)
    else:
      return self.beta


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


def _assert_parameter_valid(validate_args, x, lower_bound, upper_bound,
                            message):
  """Helper to check that the input parameter is valid."""
  if validate_args:
    with tf.control_dependencies([
        tf.debugging.assert_greater_equal(x, lower_bound, message=message),
        tf.debugging.assert_less_equal(x, upper_bound, message=message),
    ]):
      return tf.identity(x)
  else:
    return x
