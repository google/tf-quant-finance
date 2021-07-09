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
"""Calibration methods for the SVI volatility model."""

from typing import Callable, Tuple
import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.experimental.svi import parameterizations
from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math import optimizer

__all__ = ['calibration']


def calibration(
    *,
    forwards: types.RealTensor,
    expiries: types.RealTensor,
    strikes: types.RealTensor,
    volatilities: types.RealTensor,
    weights: types.RealTensor = None,
    initial_position: types.RealTensor = None,
    optimizer_fn: Callable[..., types.RealTensor] = None,
    tolerance: types.RealTensor = 1e-6,
    x_tolerance: types.RealTensor = 0,
    f_relative_tolerance: types.RealTensor = 0,
    maximum_iterations: types.IntTensor = 100,
    dtype: tf.DType = None,
    name: str = None
) -> Tuple[types.RealTensor, types.BoolTensor, types.IntTensor]:
  """Calibrates the SVI model parameters for a batch of volatility skews.

  This function optimizes the SVI model parameters to fit the given volatilities
  at various strikes. The loss function is the L2 norm of the differences in the
  volatility space.

  Each volatility skew in the batch corresponds to a fixed expiry for options
  on some underlying assets. Optimization is done independently for each skew.

  TODO(b/189458981): add flexibility to accept higher rank tensors as inputs.

  #### Example
  This example shows how to calibrate a single skew, loosely based on market
  prices for GOOG210820C* (GOOG calls with 2021-08-20 expiry) as of 2021-05-27.
  https://finance.yahoo.com/quote/GOOG/options?p=GOOG&date=1629417600

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  forwards = np.array([2402.])
  expiries = np.array([0.23])
  strikes = np.array([[
      1700., 1800., 1900., 2000., 2050., 2100., 2200., 2250., 2350., 2400.,
      2450., 2500., 2550., 2600., 2650., 2700., 2750., 2800., 2850., 2900.,
      2950., 3000.
  ]])
  volatilities = np.array([[
      0.5335, 0.4882, 0.4389, 0.3937, 0.3749, 0.3569, 0.3259, 0.3135, 0.29,
      0.283, 0.2717, 0.2667, 0.2592, 0.2566, 0.2564, 0.2574, 0.2595, 0.2621,
      0.2669, 0.2732, 0.2826, 0.2967
  ]])

  (svi_params, converged, _) = tff.experimental.svi.calibration(
      forwards=forwards,
      expiries=expiries,
      strikes=strikes,
      volatilities=volatilities)

  # Expected results are tensors containing (up to numerical tolerance):
  # svi_params: [[-0.2978, 0.4212, 0.0415, 0.1282, 0.7436]]
  # converged: [True]
  ```

  Args:
    forwards: A rank 1 real `Tensor` of shape [batch_size]. The forward prices
      of the underlyig asset for each skew in the batch.
    expiries: A rank 1 real `Tensor` of shape [batch_size]. The option expiries
      for each skew in the batch.
    strikes: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
      strike prices of the options.
    volatilities: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
      market implied Black-Scholes volatilities to calibrate.
    weights: An optional rank 2 real `Tensor` of shape [batch_size,
      num_strikes]. Used to define the loss function as the weighted L2 norm of
      the residuals.
      Default value: None, in which case weights are set to 1.
    initial_position: A rank 2 real `Tensor` of shape [batch_size, 5]. Raw SVI
      parameter tuples `(a, b, rho, m, sigma)` to be used as the initial values
      for the optimization.
      Default value: None, in which case the initial values are estimated
        heuristically and may lead to slower convergence.
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
    tolerance: Scalar `Tensor` of real dtype. Specifies the gradient tolerance
      for the procedure. If the supremum norm of the gradient vector is below
      this number, the algorithm is stopped.
      Default value: 1e-6.
    x_tolerance: Scalar `Tensor` of real dtype. If the absolute change in the
      position between one iteration and the next is smaller than this number,
      the algorithm is stopped.
      Default value: 1e-6.
    f_relative_tolerance: Scalar `Tensor` of real dtype. If the relative change
      in the objective value between one iteration and the next is smaller than
      this value, the algorithm is stopped.
    maximum_iterations: Scalar positive int32 `Tensor`. The maximum number of
      iterations during the optimization.
      Default value: 200.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None`, uses the default dtypes inferred by TensorFlow.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None`, maps to the default name `svi_skew_calibration`.

  Returns:
    A Tuple of three elements: (parameters, status, iterations)
    - parameters: a tensor of shape [batch_size, 5] representing raw parameters
      for the SVI model calibrated with given input Black-Scholes volatilities.
    - status: boolean, whether the optimization algorithm succeeded in finding
      the optimal point based on the specified convergance criteria.
    - iterations: the number of iterations performed during the optimization.

  """
  name = name or 'svi_skew_calibration'
  with tf.name_scope(name):
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    dtype = dtype or volatilities.dtype
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')

    if weights is None:
      weights = tf.ones_like(strikes, dtype=dtype, name='weights')
    else:
      weights = tf.convert_to_tensor(weights, dtype=dtype, name='weights')

    # the standard notation for log moneyness in the literature is k:=log(K/F)
    log_moneyness = tf.math.log(strikes / forwards[:, None])

    # the target total variance to be approximated by the model
    total_variance = volatilities**2 * expiries[:, None]

    if optimizer_fn is None:
      optimizer_fn = optimizer.conjugate_gradient_minimize

    if initial_position is None:
      initial_position = _estimate_initial_position(log_moneyness,
                                                    total_variance)

    unconstrained_initial = _raw_svi_to_unconstrained(initial_position)
    # protect against NaNs, which may appear if the required constraints on SVI
    # parameters are not satisfied for the user-supplied `initial_position`
    unconstrained_initial = tf.where(
        tf.math.is_nan(unconstrained_initial),
        tf.zeros_like(unconstrained_initial, dtype), unconstrained_initial)

    @make_val_and_grad_fn
    def loss_function(unconstrained_params):
      """Loss function for the optimization."""
      parameters = _unconstrained_to_raw_svi(unconstrained_params)
      model_variance = parameterizations.total_variance_from_raw_svi_parameters(
          svi_parameters=parameters, log_moneyness=log_moneyness)

      model_vol = tf.math.sqrt(model_variance / expiries[:, None])

      weighted_squared_difference = weights * tf.math.squared_difference(
          model_vol, volatilities)
      loss = tf.math.reduce_sum(weighted_squared_difference, axis=1)
      return loss

    optimization_result = optimizer_fn(
        loss_function,
        initial_position=unconstrained_initial,
        tolerance=tolerance,
        x_tolerance=x_tolerance,
        f_relative_tolerance=f_relative_tolerance,
        max_iterations=maximum_iterations)
    calibrated_parameters = _unconstrained_to_raw_svi(
        optimization_result.position)

    return (calibrated_parameters, optimization_result.converged,
            optimization_result.num_iterations)


def _estimate_initial_position(log_moneyness, total_variance):
  """Provides a heuristic initial guess for the SVI parameters.

  The values for `rho` and `sigma` are predetermined.
  `rho = 0` enforces the symmetry of the initial skew.
  `sigma = 0.5` enforces certain smoothness at the vertex of the skew.

  The value for `m` estimated as the position of the vertex of the skew:
  `m` is the log-moneyness corresponding to the smallest input variance.

  The values for `a`, `b` are computed using a simple linear regression, using
  the input data and the above estimates for `rho`, `sigma` and `m`.

  Args:
    log_moneyness: A rank 2 real `Tensor` of shape [batch_size, num_strikes].
      The log-moneyness `k := log(K/F)` of the options.
    total_variance: A rank 2 real `Tensor` of shape [batch_size, num_strikes].
      The target total variance to be approximated by the SVI model.

  Returns:
  A rank 2 real `Tensor` of shape [batch_size, 5], representing an initial
  guess for the SVI parameter optimization.
  """
  dtype = total_variance.dtype

  # Estimate `m` as the log_moneyess with the smallest target variance
  minvol_index = tf.argmin(total_variance, axis=1)
  m = tf.gather(log_moneyness, minvol_index, axis=1, batch_dims=1)

  # The initial guess will be a reasonably smooth symmetric smile
  sigma = 0.5 * tf.ones_like(minvol_index, dtype=dtype)
  rho = tf.zeros_like(minvol_index, dtype=dtype)

  # At this point, the SVI equation is reduced to `y = a + b * x`, where
  y = total_variance
  x = tf.sqrt((log_moneyness - m[:, None])**2 + sigma[:, None]**2)

  # Solve the simple regression for `a` and `b`, using the standard formulas,
  # cf. https://en.wikipedia.org/wiki/Simple_linear_regression
  e_x = tf.math.reduce_mean(x, axis=1)
  e_y = tf.math.reduce_mean(y, axis=1)
  e_xy = tf.math.reduce_mean(x * y, axis=1)
  var_x = tf.math.reduce_variance(x, axis=1)
  b = (e_xy - e_x * e_y) / var_x
  a = e_y - b * e_x

  initial_position = tf.transpose([a, b, rho, m, sigma])
  return initial_position


def _raw_svi_to_unconstrained(parameters):
  """Converts raw SVI parameters to unconstrained ones for optimization.

  The raw SVI parameters are subject to constraints:
  `b > 0`, `|rho| < 1`, `sigma > 0`, `a + b * sqrt(1 - rho^2) > 0'
  In order to perform SVI model fitting using the standard optimizers, the
  model is reparameterized into an equivalent set of unconstrained parameters.

  Args:
    parameters: A rank 2 real `Tensor` of shape [batch_size, 5], representing
      SVI model's raw parameters.

  Returns:
    A rank 2 real `Tensor` of shape [batch_size, 5], representing the
  unconstrained parameters, used in internal optimization of the SVI model.
  """
  a = parameters[..., 0]
  b = parameters[..., 1]
  rho = parameters[..., 2]
  m = parameters[..., 3]
  sigma = parameters[..., 4]

  logminvar = tf.math.log(a + b * sigma * tf.math.sqrt(1 - rho**2))
  logb = tf.math.log(b)
  r = tf.math.log1p(rho) - tf.math.log1p(-rho)
  logsigma = tf.math.log(sigma)
  return tf.transpose([logminvar, logb, r, m, logsigma])


def _unconstrained_to_raw_svi(unconstrained_parameters):
  """Converts unconstrained optimizarion parameters to raw SVI ones.

  Performs the inverse transformation of the internal unconstrained model
  parameters into the standard raw SVI parameters `a, b, rho, m, sigma`.

  Args:
    unconstrained_parameters: A rank 2 real `Tensor` of shape [batch_size, 5],
      representing SVI model's raw parameters.

  Returns:
    A rank 2 real `Tensor` of shape [batch_size, 5], representing the
    unconstrained parameters, used in internal optimization of the SVI model.
  """
  b = tf.math.exp(unconstrained_parameters[..., 1])
  rho = 2 * tf.math.sigmoid(unconstrained_parameters[..., 2]) - 1
  m = unconstrained_parameters[..., 3]
  sigma = tf.math.exp(unconstrained_parameters[..., 4])
  a = tf.math.exp(
      unconstrained_parameters[..., 0]) - b * sigma * tf.math.sqrt(1 - rho**2)
  # Return shape: `[batch_size, 5]`
  return tf.transpose([a, b, rho, m, sigma])
