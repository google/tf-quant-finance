# Copyright 2021 Google LLC
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
"""Calibration methods for the HJM model."""

from typing import Callable, Tuple

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.black_scholes import implied_vol
from tf_quant_finance.black_scholes.implied_vol_utils import UnderlyingDistribution
from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math import optimizer
from tf_quant_finance.math import random
from tf_quant_finance.models import valuation_method as vm
from tf_quant_finance.models.hjm.swaption_pricing import price as swaption_price
from tf_quant_finance.rates.analytics import swap

__all__ = [
    'CalibrationResult',
    'calibration_from_swaptions'
]


@utils.dataclass
class CalibrationResult:
  """Collection of calibrated QuasiGaussianHJM parameters.

  For a review of the HJM model and the conventions used, please see the
  docstring for `QuasiGaussianHJM`, or for `calibration_from_swaptions` below.

  Attributes:
    mean_reversion: Rank-1 `Tensor` specifying the mean-reversion parameter.
    volatility: Rank-1 `Tensor` specifying the volatility parameter.
    corr_matrix: Rank-1 `Tensor` specifying the correlation matrix parameter.
  """
  mean_reversion: types.RealTensor
  volatility: types.RealTensor
  corr_matrix: types.RealTensor


# TODO(b/182392038): Move this to tff.math module.
def _correlation_matrix_using_hypersphere_decomposition(num_assets,
                                                        thetas,
                                                        dtype=None):
  """Rebonato-Jaeckel method to generate valid correlation matrix.

  The method returns a valid correlation matrix using the hypersphere
  decomposition approach described in Ref [1].

  #### References:
  [1]: Riccardo Rebonato, Peter Jaeckel, The Most General Methodology to Create
       a Valid Correlation Matrix for Risk Management and Option Pricing
       Purposes.
  https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID1969689_code1760921.pdf?abstractid=1969689&mirid=1

  Args:
    num_assets: An integer `Tensor` specifying the number of correlated assets.
    thetas: A real `Tensor` of shape `[num_assets * (num_assets - 1)]`
      specifying the angles used for the construction of the correlation matrix.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.

  Returns:
    A real `Tensor` of shape `(num_assets, num_assets)` containing the
    correlation matrix derived from `thetas` using the hypersphere
    decomposition method.
  """
  thetas = tf.convert_to_tensor(thetas, dtype=dtype)
  dtype = dtype or thetas.dtype
  thetas = tf.reshape(thetas, shape=[num_assets, num_assets - 1])
  cos_theta = tf.math.cos(thetas)
  cos_theta = tf.concat([cos_theta, tf.ones([num_assets, 1], dtype=dtype)],
                        axis=1)
  sin_cumprod = tf.math.cumprod(tf.math.sin(thetas), axis=-1)
  sin_cumprod = tf.concat([tf.ones([num_assets, 1], dtype=dtype), sin_cumprod],
                          axis=1)
  j = tf.expand_dims(tf.range(1, num_assets + 1), axis=0)
  corr_matrix = tf.where(j < num_assets, cos_theta * sin_cumprod, sin_cumprod)
  corr_matrix = tf.linalg.matmul(corr_matrix, corr_matrix, transpose_b=True)

  return corr_matrix


_THETA_UB = 2 * np.pi + 0.01


def calibration_from_swaptions(
    *,
    prices: types.RealTensor,
    expiries: types.RealTensor,
    floating_leg_start_times: types.RealTensor,
    floating_leg_end_times: types.RealTensor,
    fixed_leg_payment_times: types.RealTensor,
    floating_leg_daycount_fractions: types.RealTensor,
    fixed_leg_daycount_fractions: types.RealTensor,
    fixed_leg_coupon: types.RealTensor,
    reference_rate_fn: Callable[..., types.RealTensor],
    num_hjm_factors: types.RealTensor,
    mean_reversion: types.RealTensor,
    volatility: types.RealTensor,
    notional: types.RealTensor = None,
    is_payer_swaption: types.BoolTensor = None,
    swaption_valuation_method: vm.ValuationMethod = None,
    num_samples: types.IntTensor = 1,
    random_type: random.RandomType = None,
    seed: types.IntTensor = None,
    skip: types.IntTensor = 0,
    times: types.RealTensor = None,
    time_step: types.RealTensor = None,
    num_time_steps: types.IntTensor = None,
    curve_times: types.RealTensor = None,
    time_step_finite_difference: types.RealTensor = None,
    num_grid_points_finite_difference: types.IntTensor = 101,
    volatility_based_calibration: bool = True,
    calibrate_correlation: bool = True,
    optimizer_fn: Callable[..., types.RealTensor] = None,
    mean_reversion_lower_bound: types.RealTensor = 0.001,
    mean_reversion_upper_bound: types.RealTensor = 0.5,
    volatility_lower_bound: types.RealTensor = 0.00001,
    volatility_upper_bound: types.RealTensor = 0.1,
    tolerance: types.RealTensor = 1e-6,
    maximum_iterations: types.IntTensor = 50,
    dtype: tf.DType = None,
    name: str = None) -> Tuple[CalibrationResult,
                               types.BoolTensor,
                               types.IntTensor]:
  """Calibrates a batch of HJM models using European Swaption prices.

  This function estimates the mean-reversion rates, volatility and correlation
  parameters of a multi factor HJM model using a set of European swaption
  prices as the target. The calibration is performed using least-squares
  optimization where the loss function minimizes the squared error between the
  target swaption prices (or volatilities) and the model implied swaption
  prices (or volatilities). The current calibration supports constant mean
  reversion, volatility and correlation parameters.

  #### Example
  The example shows how to calibrate a Two factor HJM model with constant mean
  reversion rate and constant volatility.

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  expiries = np.array(
      [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 10., 10.])
  float_leg_start_times = np.array([
      [0.5, 1.0, 1.5, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],  # 6M x 2Y  swap
      [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],  # 6M x 5Y  swap
      [1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # 1Y x 2Y  swap
      [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],  # 1Y x 5Y  swap
      [2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # 2Y x 2Y  swap
      [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],  # 2Y x 5Y  swap
      [3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # 3Y x 2Y  swap
      [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],  # 3Y x 5Y  swap
      [4.0, 4.5, 5.0, 5.5, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # 4Y x 2Y  swap
      [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],  # 4Y x 5Y  swap
      [5.0, 5.5, 6.0, 6.5, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],  # 5Y x 2Y  swap
      [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],  # 5Y x 5Y  swap
      [10.0, 10.5, 11.0, 11.5, 12.0, 12.0, 12.0, 12.0, 12.0,
       12.0],  # 10Y x 2Y  swap
      [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
       14.5]  # 10Y x 5Y  swap
  ])
  float_leg_end_times = float_leg_start_times + 0.5
  max_maturities = np.array(
      [2.5, 5.5, 3.0, 6.0, 4., 7., 5., 8., 6., 9., 7., 10., 12., 15.])
  for i in range(float_leg_end_times.shape[0]):
    float_leg_end_times[i] = np.clip(
        float_leg_end_times[i], 0.0, max_maturities[i])

  fixed_leg_payment_times = float_leg_end_times
  float_leg_daycount_fractions = (
      float_leg_end_times - float_leg_start_times)
  fixed_leg_daycount_fractions = float_leg_daycount_fractions
  fixed_leg_coupon = 0.01 * np.ones_like(fixed_leg_payment_times)

  zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
  notional = 1.0
  prices = np.array([
      0.42919881, 0.98046542, 0.59045074, 1.34909391, 0.79491583,
      1.81768802, 0.93210461, 2.13625342, 1.05114573, 2.40921088,
      1.12941064, 2.58857507, 1.37029637, 3.15081683])

  (calibrated_mr, calibrated_vol, calibrated_corr), _, _ = (
  tff.models.hjm.calibration_from_swaptions(
      prices=prices,
      expiries=expiries,
      floating_leg_start_times=float_leg_start_times,
      floating_leg_end_times=float_leg_end_times,
      fixed_leg_payment_times=fixed_leg_payment_times,
      floating_leg_daycount_fractions=float_leg_daycount_fractions,
      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
      fixed_leg_coupon=fixed_leg_coupon,
      reference_rate_fn=zero_rate_fn,
      notional=100.,
      mean_reversion=[0.01, 0.01],  # Initial guess for mean reversion rate
      volatility=[0.005, 0.004],  # Initial guess for volatility
      volatility_based_calibration=True,
      calibrate_correlation=True,
      num_samples=2000,
      time_step=0.1,
      random_type=random.RandomType.STATELESS_ANTITHETIC,
      seed=[0,0],
      maximum_iterations=50,
      dtype=dtype))
  # Expected calibrated_mr: [0.00621303, 0.3601772]
  # Expected calibrated_vol: [0.00586125, 0.00384013]
  # Expected correlation: 0.65126492
  # Prices using calibrated model: [
      0.42939121, 0.95362327, 0.59186236, 1.32622752, 0.79575526,
      1.80457544, 0.93909176, 2.14336776, 1.04132595, 2.39385229,
      1.11770416, 2.58809336, 1.39557389, 3.29306317]
  ````

  Args:
    prices: An N-D real `Tensor` of shape `batch_shape + [k]`. `batch_shape` is
      the shape of the batch of models to calibrate and `k` is the number of
      swaptions per calibration. The input represents the prices of swaptions
      used for calibration.
    expiries: A real `Tensor` of same shape and dtype as `prices`. The time to
      expiration of the swaptions.
    floating_leg_start_times: A real `Tensor` of the same dtype as `prices`. The
      times when accrual begins for each payment in the floating leg. The shape
      of this input should be `expiries.shape + [m]` where `m` denotes the
      number of floating payments in each leg.
    floating_leg_end_times: A real `Tensor` of the same dtype as `prices`. The
      times when accrual ends for each payment in the floating leg. The shape of
      this input should be `expiries.shape + [m]` where `m` denotes the number
      of floating payments in each leg.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as `prices`. The
      payment times for each payment in the fixed leg. The shape of this input
      should be `expiries.shape + [n]` where `n` denotes the number of fixed
      payments in each leg.
    floating_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `floating_leg_start_times`. The daycount fractions for
      each payment in the floating leg.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions for
      each payment in the fixed leg.
    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape as
      `fixed_leg_payment_times`. The fixed rate for each payment in the fixed
      leg.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape`. Returns
      the continuously compounded zero rate at the present time for the input
      expiry time.
    num_hjm_factors: A Python scalar which corresponds to the number of factors
      in the batch of calibrated HJM models.
    mean_reversion: A real positive `Tensor` of same dtype as `prices` and shape
      `batch_shape  + [num_hjm_factors]`. Corresponds to the initial values of
      the mean reversion rates of the factors for calibration.
    volatility: A real positive `Tensor` of the same `dtype` and shape as
      `mean_reversion`. Corresponds to the initial values of the volatility of
      the factors for calibration.
    notional: An optional `Tensor` of same dtype and compatible shape as
      `strikes`specifying the notional amount for the underlying swap.
       Default value: None in which case the notional is set to 1.
    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.
      Indicates whether the prices correspond to payer (if True) or receiver (if
      False) swaption. If not supplied, payer swaptions are assumed.
    swaption_valuation_method: An enum of type
      `valuation_method.ValuationMethod` specifying the method to be used for
      swaption valuation during calibration. Currently the valuation is
      supported using `MONTE_CARLO` and `FINITE_DIFFERENCE` methods. Valuation
      using finite difference is only supported for Gaussian HJM models, i.e.
      for models with constant mean-reversion rate and time-dependent
      volatility.
      Default value: `valuation_method.ValuationMethod.MONTE_CARLO`, in which
      case swaption valuation is done using Monte Carlo simulations.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation of swaptions. This input is ignored
      during analytic valuation.
      Default value: The default value is 1.
    random_type: Enum value of `RandomType`. The type of (quasi)-random number
      generator to use to generate the simulation paths. This input is relevant
      only for Monte-Carlo valuation and ignored during analytic valuation.
      Default value: `None` which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of `[STATELESS, PSEUDO, HALTON_RANDOMIZED,
      PSEUDO_ANTITHETIC, STATELESS_ANTITHETIC]`. For `PSEUDO`,
      `PSEUDO_ANTITHETIC` and `HALTON_RANDOMIZED` the seed should be an Python
      integer. For `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as
      an integer `Tensor` of shape `[2]`. This input is relevant only for
      Monte-Carlo valuation and ignored during analytic valuation.
      Default value: `None` which means no seed is set.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    times: An optional rank 1 `Tensor` of increasing positive real values. The
      times at which Monte Carlo simulations are performed. Relevant when
      swaption valuation is done using Monte Calro simulations.
      Default value: `None` in which case simulation times are computed based
      on either `time_step` or `num_time_steps` inputs.
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation. This
      input is ignored during analytic valuation.
      Default value: `None`.
    num_time_steps: An optional scalar integer `Tensor` - a total number of
      time steps during Monte Carlo simulations. The maximal distance betwen
      points in grid is bounded by
      `times[-1] / (num_time_steps - times.shape[0])`.
      Either this or `time_step` should be supplied when the valuation method
      is Monte Carlo.
      Default value: `None`.
    curve_times: An optional rank 1 `Tensor` of positive real values. The
      maturities at which spot discount curve is computed during simulations.
      Default value: `None` in which case `curve_times` is computed based on
      swaption expities and `fixed_leg_payments_times` inputs.
    time_step_finite_difference: Scalar real `Tensor`. Spacing between time
      grid points in finite difference discretization. This input is only
      relevant for valuation using finite difference.
      Default value: `None`, in which case a `time_step` corresponding to 100
      discrete steps is used.
    num_grid_points_finite_difference: Scalar real `Tensor`. Number of spatial
      grid points for discretization. This input is only relevant for valuation
      using finite difference.
      Default value: 100.
    volatility_based_calibration: An optional Python boolean specifying whether
      calibration is performed using swaption implied volatilities. If the input
      is `True`, then the swaption prices are first converted to normal implied
      volatilities and calibration is performed by minimizing the error between
      input implied volatilities and model implied volatilities.
      Default value: True.
    calibrate_correlation: An optional Python boolean specifying if the
      correlation matrix between HJM factors should calibrated. If the input is
      `False`, then the model is calibrated assuming that the HJM factors are
      uncorrelated.
      Default value: True.
    optimizer_fn: Optional Python callable which implements the algorithm used
      to minimize the objective function during calibration. It should have
      the following interface:
      result = optimizer_fn(value_and_gradients_function, initial_position,
        tolerance, max_iterations)
      `value_and_gradients_function` is a Python callable that accepts a point
      as a real `Tensor` and returns a tuple of `Tensor`s of real dtype
      containing the value of the function and its gradient at that point.
      'initial_position' is a real `Tensor` containing the starting point of
      the optimization, 'tolerance' is a real scalar `Tensor` for stopping
      tolerance for the procedure and `max_iterations` specifies the maximum
      number of iterations.
      `optimizer_fn` should return a namedtuple containing the items: `position`
      (a tensor containing the optimal value), `converged` (a boolean
      indicating whether the optimize converged according the specified
      criteria), `failed` (a boolean indicating if the optimization resulted
      in a failure), `num_iterations` (the number of iterations used), and
      `objective_value` ( the value of the objective function at the optimal
      value). The default value for `optimizer_fn` is None and conjugate
      gradient algorithm is used.
    mean_reversion_lower_bound: An optional scalar `Tensor` specifying the lower
      limit of mean reversion rate during calibration.
      Default value: 0.001.
    mean_reversion_upper_bound: An optional scalar `Tensor` specifying the upper
      limit of mean reversion rate during calibration.
      Default value: 0.5.
    volatility_lower_bound: An optional scalar `Tensor` specifying the lower
      limit of volatility during calibration.
      Default value: 0.00001 (0.1 basis points).
    volatility_upper_bound: An optional scalar `Tensor` specifying the upper
      limit of volatility during calibration.
      Default value: 0.1.
    tolerance: Scalar `Tensor` of real dtype. The absolute tolerance for
      terminating the iterations.
      Default value: 1e-6.
    maximum_iterations: Scalar positive int32 `Tensor`. The maximum number of
      iterations during the optimization.
      Default value: 50.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
        `hjm_swaption_calibration`.

  Returns:
    A Tuple of three elements:
    * The first element is an instance of `CalibrationResult` whose parameters
      are calibrated to the input swaption prices.
    * A `Tensor` of optimization status for each batch element (whether the
      optimization algorithm has found the optimal point based on the specified
      convergance criteria).
    * A `Tensor` containing the number of iterations performed by the
      optimization algorithm.
  """
  del floating_leg_daycount_fractions
  name = name or 'hjm_swaption_calibration'
  with tf.name_scope(name):
    prices = tf.convert_to_tensor(prices, dtype=dtype, name='prices')
    dtype = dtype or prices.dtype
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    float_leg_start_times = tf.convert_to_tensor(
        floating_leg_start_times, dtype=dtype, name='float_leg_start_times')
    float_leg_end_times = tf.convert_to_tensor(
        floating_leg_end_times, dtype=dtype, name='float_leg_end_times')
    fixed_leg_payment_times = tf.convert_to_tensor(
        fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
    fixed_leg_daycount_fractions = tf.convert_to_tensor(
        fixed_leg_daycount_fractions,
        dtype=dtype,
        name='fixed_leg_daycount_fractions')
    fixed_leg_coupon = tf.convert_to_tensor(
        fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')

    if times is None:
      times, _ = tf.unique(tf.reshape(expiries, [-1]))
      times = tf.sort(times, name='sort_times')
    else:
      times = tf.convert_to_tensor(times, dtype=dtype)

    if curve_times is None:
      tau = fixed_leg_payment_times - tf.expand_dims(expiries, axis=-1)
      curve_times, _ = tf.unique(tf.reshape(tau, [-1]))
      curve_times = tf.sort(curve_times)
    else:
      curve_times = tf.convert_to_tensor(curve_times, dtype=dtype)

    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    vol_lb = tf.convert_to_tensor(volatility_lower_bound, dtype=dtype)
    vol_ub = tf.convert_to_tensor(volatility_upper_bound, dtype=dtype)
    mr_lb = tf.convert_to_tensor(mean_reversion_lower_bound, dtype=dtype)
    mr_ub = tf.convert_to_tensor(mean_reversion_upper_bound, dtype=dtype)
    theta_lb = tf.convert_to_tensor(0, dtype=dtype)
    theta_ub = tf.convert_to_tensor(_THETA_UB, dtype=dtype)

    mean_reversion = tf.convert_to_tensor(mean_reversion, dtype=dtype)
    volatility = tf.convert_to_tensor(volatility, dtype=dtype)

    swaption_valuation_method = (
        swaption_valuation_method or vm.ValuationMethod.MONTE_CARLO)

    if optimizer_fn is None:
      optimizer_fn = optimizer.conjugate_gradient_minimize

    def _price_to_normal_vol(x, swap_rate, annuity):
      vols = implied_vol(
          prices=x / annuity / notional,
          strikes=fixed_leg_coupon[..., 0],
          expiries=expiries,
          forwards=swap_rate,
          is_call_options=is_payer_swaption,
          underlying_distribution=UnderlyingDistribution.NORMAL,
          dtype=dtype)
      return vols

    if volatility_based_calibration:
      batch_shape = tf.shape(prices)[:-1]
      batch_size = tf.math.reduce_prod(batch_shape)
      num_instruments = tf.shape(prices)[-1]
      swap_rate, annuity = swap.ir_swap_par_rate_and_annuity(
          float_leg_start_times, float_leg_end_times, fixed_leg_payment_times,
          fixed_leg_daycount_fractions, reference_rate_fn)
      # Because we require `reference_rate_fn` to return a Tensor of shape
      # `[batch_shape] + t.shape`, we get cross product terms that we don't
      # need. The logic below takes `swap_rate` and `annuity` from shape
      # `[batch_shape, batch_shape, num_instruments]` to
      # `[batch_shape, num_instruments]`
      swap_rate = tf.reshape(
          swap_rate, [batch_size, batch_size, num_instruments])
      annuity = tf.reshape(
          annuity, [batch_size, batch_size, num_instruments])
      indices = tf.stack([tf.range(batch_size, dtype=tf.int32),
                          tf.range(batch_size, dtype=tf.int32)],
                         axis=-1)
      swap_rate = tf.gather_nd(swap_rate, indices)
      annuity = tf.gather_nd(annuity, indices)
      swap_rate = tf.reshape(swap_rate, tf.shape(prices))
      annuity = tf.reshape(annuity, tf.shape(prices))
      target_values = _price_to_normal_vol(prices, swap_rate, annuity)
    else:
      target_values = prices

    with tf.control_dependencies([target_values]):
      tf.debugging.assert_all_finite(
          target_values, 'Conversion to implied vols resulted in failure for '
          'input swaption prices.')

    target_lb = tf.constant(0.0, dtype=dtype)
    target_ub = tf.math.reduce_max(target_values)

    def _scale(x, lb, ub):
      return (x - lb) / (ub - lb)

    def _to_unconstrained(x, lb, ub):
      x = _scale(x, lb, ub)
      return -tf.math.log((1.0 - x) / x)

    def _to_constrained(x, lb, ub):
      x = tf.math.exp(x) / (1.0 + tf.math.exp(x))
      return x * (ub - lb) + lb

    if calibrate_correlation:
      num_thetas = num_hjm_factors * (num_hjm_factors - 1)
      init_corr = tf.range(0.1, num_thetas + 0.1, dtype=dtype) / num_thetas
    else:
      init_corr = []
      if mean_reversion.shape.rank > 1:
        init_corr = [[]] * mean_reversion.shape.rank

    initial_guess = tf.concat([
        _to_unconstrained(mean_reversion, mr_lb, mr_ub),
        _to_unconstrained(volatility, vol_lb, vol_ub),
        _to_unconstrained(init_corr, theta_lb, theta_ub)
    ], axis=-1)
    scaled_target = _scale(target_values, target_lb, target_ub)

    @make_val_and_grad_fn
    def loss_function(x):
      """Loss function for the optimization."""
      x_mr = _to_constrained(x[..., :num_hjm_factors], mr_lb, mr_ub)
      x_vol = _to_constrained(x[..., num_hjm_factors:2 * num_hjm_factors],
                              vol_lb, vol_ub)

      if calibrate_correlation:
        thetas = x[..., 2 * num_hjm_factors:]
        thetas = tfp.math.clip_by_value_preserve_gradient(thetas, -25.0, 25.0)
        x_corr = _correlation_matrix_using_hypersphere_decomposition(
            num_hjm_factors, _to_constrained(thetas, theta_lb, theta_ub))
      else:
        x_corr = None

      volatility_param = _make_hjm_volatility_fn(x_vol, dtype)

      # TODO(b/182663434): Use precomputed random draws.
      model_values = swaption_price(
          expiries=expiries,
          fixed_leg_payment_times=fixed_leg_payment_times,
          fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
          fixed_leg_coupon=fixed_leg_coupon,
          reference_rate_fn=reference_rate_fn,
          num_hjm_factors=num_hjm_factors,
          mean_reversion=x_mr,
          volatility=volatility_param,
          corr_matrix=x_corr,
          notional=notional,
          is_payer_swaption=is_payer_swaption,
          valuation_method=swaption_valuation_method,
          num_samples=num_samples,
          random_type=random_type,
          seed=seed,
          skip=skip,
          times=times,
          time_step=time_step,
          num_time_steps=num_time_steps,
          curve_times=curve_times,
          time_step_finite_difference=time_step_finite_difference,
          num_grid_points_finite_difference=num_grid_points_finite_difference,
          dtype=dtype)

      if volatility_based_calibration:
        model_values = _price_to_normal_vol(model_values, swap_rate, annuity)
        model_values = tf.where(
            tf.math.is_nan(model_values), tf.constant(1e-7, dtype=dtype),
            model_values)

      value = tf.math.reduce_sum(
          (_scale(model_values, target_lb, target_ub) - scaled_target)**2,
          axis=-1)
      return value

    optimization_result = optimizer_fn(
        loss_function,
        initial_position=initial_guess,
        tolerance=tolerance,
        max_iterations=maximum_iterations)
    calibrated_parameters = optimization_result.position
    mean_reversion_calibrated = _to_constrained(
        calibrated_parameters[..., :num_hjm_factors], mr_lb, mr_ub)
    volatility_calibrated = _to_constrained(
        calibrated_parameters[..., num_hjm_factors:2 * num_hjm_factors], vol_lb,
        vol_ub)

    if calibrate_correlation:
      correlation_calibrated = (
          _correlation_matrix_using_hypersphere_decomposition(
              num_hjm_factors,
              _to_constrained(
                  calibrated_parameters[..., 2 * num_hjm_factors:],
                  theta_lb, theta_ub)))
    else:
      correlation_calibrated = None

    return (CalibrationResult(mean_reversion=mean_reversion_calibrated,
                              volatility=volatility_calibrated,
                              corr_matrix=correlation_calibrated),
            optimization_result.converged,
            optimization_result.num_iterations)


def _make_hjm_volatility_fn(volatility, dtype):
  volatility = tf.convert_to_tensor(volatility, dtype=dtype)

  return volatility
