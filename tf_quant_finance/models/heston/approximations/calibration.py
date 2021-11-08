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
"""Calibrates the approximated Heston model parameters using option prices."""

from typing import Callable, Tuple

import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math import optimizer
from tf_quant_finance.models.heston import approximations


__all__ = [
    'CalibrationResult',
    'calibration',
]


@utils.dataclass
class CalibrationResult:
  """Collection of calibrated Heston parameters.

  For a review of the Heston model and the conventions used, please see the
  docstring for `HestonModel`, or for `calibration` below.

  Attributes:
    initial_variance: Rank-1 `Tensor` specifying the initial volatility levels.
    mean_reversion: Rank-1 `Tensor` specifying the mean reversion rate.
    theta: Rank-1 `Tensor` specifying the long run price variance.
    volvol: Rank-1 `Tensor` specifying the vol-vol parameters.
    rho: Rank-1 `Tensor` specifying the correlations between the forward and
      the stochastic volatility.
  """
  initial_variance: types.RealTensor
  mean_reversion: types.RealTensor
  theta: types.RealTensor
  volvol: types.RealTensor
  rho: types.RealTensor


def calibration(
    *,
    prices: types.RealTensor,
    strikes: types.RealTensor,
    expiries: types.RealTensor,
    spots: types.RealTensor = None,
    forwards: types.RealTensor = None,
    is_call_options: types.BoolTensor,
    discount_rates: types.RealTensor = None,
    dividend_rates: types.RealTensor = None,
    discount_factors: types.RealTensor = None,
    mean_reversion: types.RealTensor,
    initial_variance: types.RealTensor,
    theta: types.RealTensor,
    volvol: types.RealTensor,
    rho: types.RealTensor,
    mean_reversion_lower_bound: types.RealTensor = 0.001,
    mean_reversion_upper_bound: types.RealTensor = 0.5,
    initial_variance_lower_bound: types.RealTensor = 0.0,
    initial_variance_upper_bound: types.RealTensor = 1.0,
    theta_lower_bound: types.RealTensor = 0.0,
    theta_upper_bound: types.RealTensor = 1.0,
    volvol_lower_bound: types.RealTensor = 0.0,
    volvol_upper_bound: types.RealTensor = 1.0,
    rho_lower_bound: types.RealTensor = -1.0,
    rho_upper_bound: types.RealTensor = 1.0,
    optimizer_fn: Callable[..., types.RealTensor] = None,
    tolerance: types.RealTensor = 1e-6,
    maximum_iterations: types.RealTensor = 100,
    validate_args: bool = False,
    dtype: tf.DType = None,
    name: str = None
    ) -> Tuple[CalibrationResult, types.BoolTensor, types.IntTensor]:

  """Calibrates the Heston model using European option prices.

  Represents the Ito process:

  ```None
    dX(t) = -V(t) / 2 * dt + sqrt(V(t)) * dW_{X}(t),
    dV(t) = mean_reversion(t) * (theta(t) - V(t)) * dt
            + volvol(t) * sqrt(V(t)) * dW_{V}(t)
  ```

  where `W_{X}` and `W_{V}` are 1D Brownian motions with a correlation `rho(t)`.
  `mean_reversion`, `theta`, `volvol`, and `rho` are positive piecewise constant
  functions of time. Here `V(t)` represents the process variance at time `t` and
  `X` represents logarithm of the spot price at time `t`.

  `mean_reversion` corresponds to the mean reversion rate, `theta` is the long
  run price variance, and `volvol` is the volatility of the volatility.

  #### Example

  ```python
  import tf_quant_finance as tff
  import tensorflow.compat.v2 as tf

  dtype = np.float64

  # Set some market conditions.
  observed_prices = np.array(
      [[29.33668202, 23.98724723, 19.54631658, 15.9022847, 12.93591534],
       [15.64785924, 21.05865247, 27.11907971, 33.74249536, 40.8485591]],
      dtype=dtype)
  strikes = np.array(
      [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
      dtype=dtype)
  expiries = np.array([[0.5], [1.0]], dtype=dtype)
  forwards = 100.0
  is_call_options = np.array([[True], [False]])

  # Calibrate the model.
  # In this example, we are calibrating a Heston model.
  models, is_converged, _ = tff.models.heston.calibration(
      prices=observed_prices,
      strikes=strikes,
      expiries=expiries,
      forwards=forwards,
      is_call_options=is_call_options,
      mean_reversion=np.array([0.3], dtype=dtype),
      initial_variance=np.array([0.8], dtype=dtype),
      theta=np.array([0.75], dtype=dtype),
      volvol=np.array([0.1], dtype=dtype),
      rho=np.array(0.0, dtype=dtype),
      optimizer_fn=tff.math.optimizer.bfgs_minimize,
      maximum_iterations=1000)

  # This will return two `HestonModel`s, where:
  # Model 1 has mean_reversion = 0.3, initial_variance = 0.473, volvol = 0.1,
  # theta = 0.724 and rho = 0.028
  # Model 2 has mean_reversion = 0.3, initial_variance = 0.45, volvol = 0.1,
  # theta = 0.691 and rho = -0.073

  ```

  Args:
    prices: Real `Tensor` of shape [batch_size, num_strikes] specifying the
      observed options prices. Here, `batch_size` refers to the number of Heston
      models calibrated in this invocation.
    strikes: Real `Tensor` of shape [batch_size, num_strikes] specifying the
      strike prices of the options.
    expiries: Real `Tensor` of shape compatible with [batch_size, num_strikes]
      specifying the options expiries.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `volatilities`. The current spot price of the underlying. Either this
      argument or the `forwards` (but not both) must be supplied.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `strikes`. The forwards to maturity. Either this argument or the
      `spots` must be supplied but both must not be supplied.
    is_call_options: A boolean `Tensor` of shape compatible with
      [batch_size, num_strikes] specifying whether or not the prices correspond
      to a call option (=True) or a put option (=False).
    discount_rates: An optional real `Tensor` of same dtype as the
      `strikes` and of the shape that broadcasts with `strikes`.
      If not `None`, discount factors are calculated as e^(-rT),
      where r are the discount rates, or risk free rates. At most one of
      discount_rates and discount_factors can be supplied.
      Default value: `None`, equivalent to r = 0 and discount factors = 1 when
      discount_factors also not given.
    dividend_rates: An optional real `Tensor` of same dtype as the
      `strikes` and of the shape that broadcasts with `volatilities`.
      Default value: `None`, equivalent to q = 0.
    discount_factors: An optional real `Tensor` of same dtype as the
      `strikes`. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with `discount_rates`. If neither is
      given, no discounting is applied (i.e. the undiscounted option price is
      returned). If `spots` is supplied and `discount_factors` is not `None`
      then this is also used to compute the forwards to expiry. At most one of
      `discount_rates` and `discount_factors` can be supplied.
      Default value: `None`, which maps to e^(-rT) calculated from
      discount_rates.
    mean_reversion: Real `Tensor` of shape [batch_size], specifying the initial
      estimate of the mean reversion parameter.
    initial_variance: Real `Tensor` of shape [batch_size], specifying the
      initial estimate of the variance parameter.
      Values must satisfy `0 <= initial_variance`.
    theta: Real `Tensor` of shape [batch_size], specifying the initial estimate
      of the long run variance parameter. Values must satisfy `0 <= theta`.
    volvol: Real `Tensor` of shape [batch_size], specifying the initial estimate
      of the vol-vol parameter. Values must satisfy `0 <= volvol`.
    rho: Real `Tensor` of shape [batch_size], specifying the initial estimate of
      the correlation between the forward price and the volatility. Values must
      satisfy -1 < `rho` < 1.
    mean_reversion_lower_bound: Real `Tensor` compatible with that of
      `mean_reversion`, specifying the lower bound for the calibrated value.
      Default value: 0.001.
    mean_reversion_upper_bound: Real `Tensor` compatible with that of
      `mean_reversion`, specifying the lower bound for the calibrated value.
      Default value: 0.5.
    initial_variance_lower_bound: Real `Tensor` compatible with that of
      `initial_variance`, specifying the lower bound for the calibrated value.
      Default value: 0.0.
    initial_variance_upper_bound: Real `Tensor` compatible with that of
      `initial_variance`, specifying the lower bound for the calibrated value.
      Default value: 1.0.
    theta_lower_bound: Real `Tensor` compatible with that of `theta`,
      specifying the lower bound for the calibrated value.
      Default value: 0.0.
    theta_upper_bound: Real `Tensor` compatible with that of `theta`,
      specifying the lower bound for the calibrated value.
      Default value: 1.0.
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
      Default value: `None`, which maps to the default name 'heston_calibration'

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

  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  if (discount_rates is not None) and (discount_factors is not None):
    raise ValueError('At most one of discount_rates and discount_factors may '
                     'be supplied')

  name = name or 'heston_calibration'
  with tf.name_scope(name):
    prices = tf.convert_to_tensor(prices, dtype=dtype, name='prices')
    dtype = dtype or prices.dtype

    # Extract batch shape
    batch_shape = prices.shape.as_list()[:-1]
    if None in batch_shape:
      batch_shape = tff.utils.get_shape(prices)[:-1]

    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')

    if discount_factors is not None:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')

    if discount_rates is not None:
      discount_rates = tf.convert_to_tensor(
          discount_rates, dtype=dtype, name='discount_rates')
    elif discount_factors is not None:
      discount_rates = -tf.math.log(discount_factors) / expiries
    else:
      discount_rates = tf.convert_to_tensor(
          0.0, dtype=dtype, name='discount_rates')

    if dividend_rates is None:
      dividend_rates = 0.0
    dividend_rates = tf.convert_to_tensor(
        dividend_rates, dtype=dtype, name='dividend_rates')

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      cost_of_carries = discount_rates - dividend_rates
      forwards = spots * tf.exp(cost_of_carries * expiries)

    is_call_options = tf.convert_to_tensor(
        is_call_options, name='is_call', dtype=tf.bool)

    if optimizer_fn is None:
      optimizer_fn = optimizer.lbfgs_minimize

    # Pre-processing mean_reversion
    mean_reversion = tf.convert_to_tensor(mean_reversion, dtype=dtype)
    mean_reversion_lower_bound = tf.convert_to_tensor(
        mean_reversion_lower_bound, dtype=dtype)
    mean_reversion_upper_bound = tf.convert_to_tensor(
        mean_reversion_upper_bound, dtype=dtype)
    mean_reversion = _assert_parameter_valid(
        validate_args,
        mean_reversion,
        lower_bound=mean_reversion_lower_bound,
        upper_bound=mean_reversion_upper_bound,
        message='`mean_reversion` is invalid!')
    # Broadcast mean_reversion to the correct batch shape
    mean_reversion = tf.broadcast_to(mean_reversion, batch_shape,
                                     name='broadcast_mean_reversion')
    initial_mean_reversion = _to_unconstrained(
        mean_reversion, mean_reversion_lower_bound, mean_reversion_upper_bound)

    # Pre-processing theta
    theta = tf.convert_to_tensor(theta, dtype=dtype)
    theta_lower_bound = tf.convert_to_tensor(theta_lower_bound, dtype=dtype)
    theta_upper_bound = tf.convert_to_tensor(theta_upper_bound, dtype=dtype)
    theta = _assert_parameter_valid(
        validate_args,
        theta,
        lower_bound=theta_lower_bound,
        upper_bound=theta_upper_bound,
        message='`theta` is invalid!')
    # Broadcast theta to the correct batch shape
    theta = tf.broadcast_to(theta, batch_shape, name='broadcast_theta')
    initial_theta = _to_unconstrained(theta, theta_lower_bound,
                                      theta_upper_bound)

    # Pre-processing volvol
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

    initial_variance_lower_bound = tf.convert_to_tensor(
        initial_variance_lower_bound, dtype=dtype)
    initial_variance_upper_bound = tf.convert_to_tensor(
        initial_variance_upper_bound, dtype=dtype)

    initial_variance = _assert_parameter_valid(
        validate_args, initial_variance,
        lower_bound=initial_variance_lower_bound,
        upper_bound=initial_variance_upper_bound,
        message='`initial_variance` is invalid!')
    # Broadcast initial_variance to the correct batch shape
    initial_variance = tf.broadcast_to(initial_variance, batch_shape,
                                       name='broadcast_initial_variance')
    initial_variance_unconstrained = _to_unconstrained(
        initial_variance, initial_variance_lower_bound,
        initial_variance_upper_bound)

    # Pre-processing rho
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
    # shape[batch_size, 5]
    initial_x = tf.stack(
        [initial_variance_unconstrained, initial_theta, initial_volvol,
         initial_mean_reversion, initial_rho], axis=-1)

    optimizer_arg_handler = _OptimizerArgHandler(
        volvol_lower_bound=volvol_lower_bound,
        volvol_upper_bound=volvol_upper_bound,
        rho_lower_bound=rho_lower_bound,
        rho_upper_bound=rho_upper_bound,
        mean_reversion_lower_bound=mean_reversion_lower_bound,
        mean_reversion_upper_bound=mean_reversion_upper_bound,
        theta_lower_bound=theta_lower_bound,
        theta_upper_bound=theta_upper_bound,
        initial_variance_lower_bound=initial_variance_lower_bound,
        initial_variance_upper_bound=initial_variance_upper_bound)

    loss_function = _get_loss_for_price_based_calibration(
        prices=prices,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        dtype=dtype,
        optimizer_arg_handler=optimizer_arg_handler)

    optimization_result = optimizer_fn(
        loss_function,
        initial_position=initial_x,
        tolerance=tolerance,
        max_iterations=maximum_iterations)

    calibration_parameters = optimization_result.position
    calibrated_theta = optimizer_arg_handler.get_theta(calibration_parameters)
    calibrated_volvol = optimizer_arg_handler.get_volvol(calibration_parameters)
    calibrated_rho = optimizer_arg_handler.get_rho(calibration_parameters)
    calibrated_initial_variance = optimizer_arg_handler.get_initial_variance(
        calibration_parameters)
    calibrated_mean_reversion = optimizer_arg_handler.get_mean_reversion(
        calibration_parameters)

    return (CalibrationResult(mean_reversion=calibrated_mean_reversion,
                              volvol=calibrated_volvol,
                              rho=calibrated_rho,
                              theta=calibrated_theta,
                              initial_variance=calibrated_initial_variance),
            optimization_result.converged,
            optimization_result.num_iterations)


def _get_loss_for_price_based_calibration(
    *, prices, strikes, expiries, forwards, is_call_options,
    optimizer_arg_handler, dtype):
  """Creates a loss function to be used in volatility-based calibration."""

  def _price_transform(x):
    return tf.math.log1p(x)

  scaled_target_values = _price_transform(prices)

  @make_val_and_grad_fn
  def loss_function(x):
    """Loss function for the price-based optimization."""
    candidate_initial_variance = optimizer_arg_handler.get_initial_variance(x)
    candidate_theta = optimizer_arg_handler.get_theta(x)
    candidate_volvol = optimizer_arg_handler.get_volvol(x)
    candidate_rho = optimizer_arg_handler.get_rho(x)
    candidate_mean_reversion = optimizer_arg_handler.get_mean_reversion(x)

    values = approximations.european_option_price(
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        variances=tf.expand_dims(candidate_initial_variance, axis=-1),
        mean_reversion=tf.expand_dims(candidate_mean_reversion, axis=-1),
        volvol=tf.expand_dims(candidate_volvol, axis=-1),
        rho=tf.expand_dims(candidate_rho, axis=-1),
        theta=tf.expand_dims(candidate_theta, axis=-1),
        dtype=dtype)

    scaled_values = _price_transform(values)
    return tf.math.reduce_mean(
        (scaled_values - scaled_target_values)**2, axis=-1)

  return loss_function


@utils.dataclass
class _OptimizerArgHandler:
  """Handles the packing/transformation of estimated parameters."""
  theta_lower_bound: types.RealTensor
  theta_upper_bound: types.RealTensor
  volvol_lower_bound: types.RealTensor
  volvol_upper_bound: types.RealTensor
  rho_lower_bound: types.RealTensor
  rho_upper_bound: types.RealTensor
  mean_reversion_lower_bound: types.RealTensor
  mean_reversion_upper_bound: types.RealTensor
  initial_variance_lower_bound: types.RealTensor
  initial_variance_upper_bound: types.RealTensor

  def get_initial_variance(
      self, packed_optimizer_args: types.RealTensor) -> types.RealTensor:
    """Unpack and return the rho parameter."""
    initial_variance = packed_optimizer_args[..., 0]
    return _to_constrained(
        initial_variance,
        self.initial_variance_lower_bound, self.initial_variance_upper_bound)

  def get_theta(self,
                packed_optimizer_args: types.RealTensor) -> types.RealTensor:
    """Unpack and return the volvol parameter."""
    theta = packed_optimizer_args[..., 1]
    return _to_constrained(
        theta,
        self.theta_lower_bound, self.theta_upper_bound)

  def get_volvol(self,
                 packed_optimizer_args: types.RealTensor) -> types.RealTensor:
    """Unpack and return the volvol parameter."""
    volvol = packed_optimizer_args[..., 2]
    return _to_constrained(
        volvol,
        self.volvol_lower_bound, self.volvol_upper_bound)

  def get_mean_reversion(
      self, packed_optimizer_args: types.RealTensor) -> types.RealTensor:
    """Unpack and return the mean_reversion parameter."""
    mean_reversion = packed_optimizer_args[..., 3]
    return _to_constrained(mean_reversion, self.mean_reversion_lower_bound,
                           self.mean_reversion_upper_bound)

  def get_rho(self,
              packed_optimizer_args: types.RealTensor) -> types.RealTensor:
    """Unpack and return the rho parameter."""
    rho = packed_optimizer_args[..., -1]
    return _to_constrained(
        rho,
        self.rho_lower_bound, self.rho_upper_bound)


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
