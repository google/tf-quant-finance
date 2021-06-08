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

import tensorflow.compat.v2 as tf

from tf_quant_finance.experimental.svi import parameterizations
from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math import optimizer


def calibrate(*,
              forwards,
              expiries,
              strikes,
              volatilities,
              initial_position=None,
              optimizer_fn=None,
              tolerance=1e-6,
              maximum_iterations=100,
              dtype=None,
              name=None):
  """Calibrates the SVI model parameters for a batch of volatility skews.

  This function optimizes the SVI model parameters to fit the given volatilities
  at various strikes. The loss function is the L2 norm of the differences in the
  volatility space.

  Each volatility skew in the batch corresponds to a fixed expiry for options
  on some underlying assets. Optimization is done independently for each skew.

  TODO(b/189458981): add flexibility to accept higher rank tensors as inputs.

  #### Example
  The example shows how to calibrate a single skew, loosely based on market
  prices for GOOG210820C* (GOOG calls with 2021-08-20 expiry) as of 2021-05-27.
  https://finance.yahoo.com/quote/GOOG/options?p=GOOG&date=1629417600

  ````python
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

  tolerance=1e-4
  (svi_params, converged, _) = tff.experimental.svi.calibrate(
      forwards=forwards,
      expiries=expiries,
      strikes=strikes,
      volatilities=volatilities)

  # Expected results are tensors containing (up to numerical tolerance):
  # svi_params: [[-0.2978, 0.4212, 0.0415, 0.1282, 0.7436]]
  # converged: [True]
  ````

  Args:
    forwards: A rank 1 real `Tensor` of shape [batch_size]. The forward prices
      of the underlyig asset for each skew in the batch.
    expiries: A rank 1 real `Tensor` of shape [batch_size]. The option expiries
      for each skew in the batch.
    strikes: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
      strike prices of the options.
    volatilities: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
      market implied Black-Scholes volatilities to calibrate.
    initial_position: A rank 2 real `Tensor` of shape [batch_size, 5]. The SVI
      parameters to use as the initial values for the optimization. The default
      value is None, in which case the initial values are guessed heuristically
      and may lead to slower convergence.
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
    tolerance: Scalar `Tensor` of real dtype. The absolute tolerance for
      terminating the iterations.
      Default value: 1e-6.
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

    # the standard notation for log moneyness in the literature is k:=log(K/F)
    log_moneyness = tf.math.log(strikes / forwards[:, None])

    if initial_position is None:
      minvol_index = tf.argmin(volatilities, axis=1)
      a0 = tf.gather(volatilities, minvol_index, axis=1, batch_dims=1)**2
      b0 = tf.zeros_like(forwards, dtype=dtype)
      rho0 = tf.zeros_like(forwards, dtype=dtype)
      sigma0 = 0.5 * tf.ones_like(forwards, dtype=dtype)
      m0 = tf.gather(log_moneyness, minvol_index, axis=1, batch_dims=1)
      initial_position = tf.transpose([a0, b0, rho0, m0, sigma0])

    if optimizer_fn is None:
      optimizer_fn = optimizer.conjugate_gradient_minimize

    @make_val_and_grad_fn
    def loss_function(parameters):
      """Loss function for the optimization."""
      total_variance = parameterizations.total_variance_from_raw(
          parameters, log_moneyness)

      model_vol = tf.where(total_variance < 0., tf.zeros_like(total_variance),
                           tf.sqrt(total_variance / expiries[:, None]))

      squared_difference = tf.where(
          total_variance < 0., volatilities**2 - total_variance,
          tf.math.squared_difference(model_vol, volatilities))

      loss = tf.math.reduce_sum(squared_difference, axis=1)
      return loss

    optimization_result = optimizer_fn(
        loss_function,
        initial_position=initial_position,
        tolerance=tolerance,
        max_iterations=maximum_iterations)

    # The optimizer may converge negative SVI sigma; to enforce the positivity
    # convention, we take sigma by absolute value, which yields the same model.
    calibrated_parameters = tf.concat([
        optimization_result.position[:, :-1],
        tf.math.abs(optimization_result.position[:, -1, None])
    ],
                                      axis=1)

    return (calibrated_parameters, optimization_result.converged,
            optimization_result.num_iterations)
