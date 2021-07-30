# Lint as: python3
# Copyright 2020 Google LLC
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
"""Calibration methods for the Hull-White model."""
from typing import Callable, Tuple, Union

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.black_scholes import implied_vol
from tf_quant_finance.black_scholes.implied_vol_utils import UnderlyingDistribution
from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math import optimizer
from tf_quant_finance.math import piecewise
from tf_quant_finance.math import random
from tf_quant_finance.models.hull_white import cap_floor
from tf_quant_finance.models.hull_white import swaption
from tf_quant_finance.rates.analytics import swap


__all__ = [
    'CalibrationResult',
    'calibration_from_swaptions',
    'calibration_from_cap_floors'
]


@utils.dataclass
class CalibrationResult:
  """Collection of calibrated one factor Hull-White parameters.

  For a review of the HullWhite model and the conventions used, please see the
  docstring for `HullWhiteModel1F`, or for `calibration_from_swaptions` below.

  Attributes:
    mean_reversion: An instance of `PiecewiseConstant` function specifying the
      mean-reversion parameter.
    volatility:  An instance of `PiecewiseConstant` specifying the volatility
      parameter.
  """
  mean_reversion: types.RealTensor
  volatility: types.RealTensor


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
    mean_reversion: Union[types.RealTensor, Callable[..., types.RealTensor]],
    volatility: Union[types.RealTensor, Callable[..., types.RealTensor]],
    notional: types.RealTensor = None,
    is_payer_swaption: types.BoolTensor = None,
    use_analytic_pricing: bool = True,
    num_samples: types.IntTensor = 1,
    random_type: random.RandomType = None,
    seed: types.IntTensor = None,
    skip: types.IntTensor = 0,
    time_step: types.RealTensor = None,
    volatility_based_calibration: bool = True,
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
  """Calibrates the Hull-White model using European Swaptions.

  This function estimates the mean-reversion rate and volatility parameters of
  a Hull-White 1-factor model using a set of European swaption prices as the
  target. The calibration is performed using least-squares optimization where
  the loss function minimizes the squared error between the target swaption
  prices and the model implied swaption prices.

  #### Example
  The example shows how to calibrate a Hull-White model with constant mean
  reversion rate and constant volatility.

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  mean_reversion = [0.03]
  volatility = [0.01]
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
  prices = tff.models.hull_white.swaption_price(
      expiries=expiries,
      floating_leg_start_times=float_leg_start_times,
      floating_leg_end_times=float_leg_end_times,
      fixed_leg_payment_times=fixed_leg_payment_times,
      floating_leg_daycount_fractions=float_leg_daycount_fractions,
      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
      fixed_leg_coupon=fixed_leg_coupon,
      reference_rate_fn=zero_rate_fn,
      notional=100.,
      dim=1,
      mean_reversion=mean_reversion,
      volatility=volatility,
      use_analytic_pricing=True,
      dtype=dtype)

  calibrated_parameters = tff.models.hull_white.calibration_from_swaptions(
      prices=prices[:, 0],
      expiries=expiries,
      floating_leg_start_times=float_leg_start_times,
      floating_leg_end_times=float_leg_end_times,
      fixed_leg_payment_times=fixed_leg_payment_times,
      floating_leg_daycount_fractions=float_leg_daycount_fractions,
      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
      fixed_leg_coupon=fixed_leg_coupon,
      reference_rate_fn=zero_rate_fn,
      notional=100.,
      mean_reversion=[0.01],  # Initial guess for mean reversion rate
      volatility=[0.005],  # Initial guess for volatility
      maximum_iterations=50,
      dtype=dtype)
  # Expected calibrated_parameters.mean_reversion.values(): [0.03]
  # Expected calibrated_parameters.volatility.values(): [0.01]
  ````

  Args:
    prices: A rank 1 real `Tensor`. The prices of swaptions used for
      calibration.
    expiries: A real `Tensor` of same shape and dtype as `prices`. The time to
      expiration of the swaptions.
    floating_leg_start_times: A real `Tensor` of the same dtype as `expiries`.
      The times when accrual begins for each payment in the floating leg. The
      shape of this input should be `expiries.shape + [m]` where `m` denotes
      the number of floating payments in each leg.
    floating_leg_end_times: A real `Tensor` of the same dtype as `expiries`.
      The times when accrual ends for each payment in the floating leg. The
      shape of this input should be `expiries.shape + [m]` where `m` denotes
      the number of floating payments in each leg.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as `expiries`.
      The payment times for each payment in the fixed leg. The shape of this
      input should be `expiries.shape + [n]` where `n` denotes the number of
      fixed payments in each leg.
    floating_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `floating_leg_start_times`. The daycount fractions
      for each payment in the floating leg.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions
      for each payment in the fixed leg.
    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape
      as `fixed_leg_payment_times`. The fixed rate for each payment in the
      fixed leg.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape either `input_shape` or
      `input_shape + [1]`. Returns the continuously compounded zero rate at the
      present time for the input expiry time.
    mean_reversion: A real positive scalar `Tensor` or an Python callable. The
      callable should satisfy the following:
      (a) A left-continuous piecewise constant object (e.g.,
      `tff.math.piecewise.PiecewiseConstantFunc`) that has a property
      `is_piecewise_constant` set to `True`. In this case the object should
      have a method `jump_locations(self)` that returns a `Tensor` of shape
      `[num_jumps]` and `values(self)` that returns a `Tensor` of shape
      `[num_jumps + 1]`. The callable, `mean_reversion(t)` should return a
      `Tensor` of shape `t.shape`, where `t` is a rank 1 `Tensor` of
      the same `dtype` as the output.
      Corresponds to the mean reversion rate to be calibrated. The input
      `Tensor` or the `Tensor` `mean_reversion.values()` is also used as the
      initial point for calibration.
    volatility: A real positive scalar `Tensor` of the same `dtype` as
      `mean_reversion` or a callable with the same specs as above.
      Corresponds to the Hull-White volatility parameter to be calibrated.
    notional: An optional `Tensor` of same dtype and compatible shape as
      `strikes`specifying the notional amount for the underlying swap.
       Default value: None in which case the notional is set to 1.
    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.
      Indicates whether the prices correspond to payer (if True) or receiver
      (if False) swaption. If not supplied, payer swaptions are assumed.
    use_analytic_pricing: A Python boolean specifying if swaption pricing is
      done analytically during calibration. Analytic valuation is only
      supported for constant `mean_reversion` and piecewise constant
      `volatility`. If the input is `False`, then valuation using Monte-Carlo
      simulations is performed.
      Default value: The default value is `True`.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation of swaptions. This input is ignored
      during analytic valuation.
      Default value: The default value is 1.
    random_type: Enum value of `RandomType`. The type of (quasi)-random
      number generator to use to generate the simulation paths. This input is
      relevant only for Monte-Carlo valuation and ignored during analytic
      valuation.
      Default value: `None` which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of
      `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
        STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
      `HALTON_RANDOMIZED` the seed should be an Python integer. For
      `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
      `Tensor` of shape `[2]`. This input is relevant only for Monte-Carlo
      valuation and ignored during analytic valuation.
      Default value: `None` which means no seed is set.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation. This
      input is ignored during analytic valuation.
      Default value: `None`.
    volatility_based_calibration: An optional Python boolean specifying whether
      calibration is performed using swaption implied volatilities. If the
      input is `True`, then the swaption prices are first converted to normal
      implied volatilities and calibration is performed by minimizing the
      error between input implied volatilities and model implied volatilities.
      Default value: True.
    optimizer_fn: Optional Python callable which implements the algorithm used
      to minimize the objective function during calibration. It should have
      the following interface:
      result = optimizer_fn(value_and_gradients_function, initial_position,
      tolerance, max_iterations)
      `value_and_gradients_function` is a Python callable that accepts a point
      as a real `Tensor` and returns a tuple of `Tensor`s of real dtype
      containing the value of the function and its gradient at that point.
      'initial_position' is a real `Tensor` containing the starting point of the
      optimization, 'tolerance' is a real scalar `Tensor` for stopping tolerance
      for the procedure and `max_iterations` specifies the maximum number of
      iterations.
      `optimizer_fn` should return a namedtuple containing the items: `position`
      (a tensor containing the optimal value), `converged` (a boolean indicating
      whether the optimize converged according the specified criteria),
      `failed` (a boolean indicating if the optimization resulted in a failure),
      `num_iterations` (the number of iterations used), and `objective_value` (
      the value of the objective function at the optimal value).
      The default value for `optimizer_fn` is None and conjugate gradient
      algorithm is used.
    mean_reversion_lower_bound: An optional scalar `Tensor` specifying the
      lower limit of mean reversion rate during calibration.
      Default value: 0.001.
    mean_reversion_upper_bound: An optional scalar `Tensor` specifying the
      upper limit of mean reversion rate during calibration.
      Default value: 0.5.
    volatility_lower_bound: An optional scalar `Tensor` specifying the
      lower limit of Hull White volatility during calibration.
      Default value: 0.00001 (0.1 basis points).
    volatility_upper_bound: An optional scalar `Tensor` specifying the
      upper limit of Hull White volatility during calibration.
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
      `hw_swaption_calibration`.

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
  name = name or 'hw_swaption_calibration'
  with tf.name_scope(name):
    prices = tf.convert_to_tensor(prices, dtype=dtype, name='prices')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    dtype = dtype or expiries.dtype
    float_leg_start_times = tf.convert_to_tensor(
        floating_leg_start_times, dtype=dtype, name='float_leg_start_times')
    float_leg_end_times = tf.convert_to_tensor(
        floating_leg_end_times, dtype=dtype, name='float_leg_end_times')
    fixed_leg_payment_times = tf.convert_to_tensor(
        fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
    fixed_leg_daycount_fractions = tf.convert_to_tensor(
        fixed_leg_daycount_fractions, dtype=dtype,
        name='fixed_leg_daycount_fractions')
    fixed_leg_coupon = tf.convert_to_tensor(
        fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    vol_lb = tf.convert_to_tensor(volatility_lower_bound, dtype=dtype)
    vol_ub = tf.convert_to_tensor(volatility_upper_bound, dtype=dtype)
    mr_lb = tf.convert_to_tensor(mean_reversion_lower_bound, dtype=dtype)
    mr_ub = tf.convert_to_tensor(mean_reversion_upper_bound, dtype=dtype)

    if not hasattr(mean_reversion, 'is_piecewise_constant'):
      mean_reversion = piecewise.PiecewiseConstantFunc(
          jump_locations=[], values=mean_reversion, dtype=dtype)
    if not hasattr(volatility, 'is_piecewise_constant'):
      volatility = piecewise.PiecewiseConstantFunc(
          jump_locations=[], values=volatility, dtype=dtype)

    if optimizer_fn is None:
      optimizer_fn = optimizer.conjugate_gradient_minimize

    if volatility_based_calibration:
      def reference_rate_squeeze_fn(t):
        r = reference_rate_fn(t)
        if r.shape.as_list()[-1] == 1:
          r = tf.squeeze(r, axis=-1)
        return r
      swap_rate, annuity = swap.ir_swap_par_rate_and_annuity(
          float_leg_start_times, float_leg_end_times, fixed_leg_payment_times,
          fixed_leg_daycount_fractions, reference_rate_squeeze_fn)
      target_values = implied_vol(
          prices=prices / annuity / notional,
          strikes=fixed_leg_coupon[..., 0],
          expiries=expiries,
          forwards=swap_rate,
          is_call_options=is_payer_swaption,
          underlying_distribution=UnderlyingDistribution.NORMAL,
          dtype=dtype)
    else:
      target_values = prices

    target_lb = tf.constant(0.0, dtype=dtype)
    target_ub = tf.math.reduce_max(target_values)

    initial_guess = tf.concat(
        [_to_unconstrained(mean_reversion.values(), mr_lb, mr_ub),
         _to_unconstrained(volatility.values(), vol_lb, vol_ub)], axis=0)
    num_mean_reversion = mean_reversion.values().shape.as_list()[0]
    scaled_target = _scale(target_values, target_lb, target_ub)

    @make_val_and_grad_fn
    def loss_function(x):
      """Loss function for the optimization."""
      x_mr = _to_constrained(x[:num_mean_reversion], mr_lb, mr_ub)
      x_vol = _to_constrained(x[num_mean_reversion:], vol_lb, vol_ub)

      mean_reversion_param = piecewise.PiecewiseConstantFunc(
          jump_locations=[], values=x_mr, dtype=dtype)
      volatility_param = piecewise.PiecewiseConstantFunc(
          jump_locations=volatility.jump_locations(),
          values=x_vol, dtype=dtype)

      model_values = swaption.swaption_price(
          expiries=expiries,
          floating_leg_start_times=float_leg_start_times,
          floating_leg_end_times=float_leg_end_times,
          fixed_leg_payment_times=fixed_leg_payment_times,
          floating_leg_daycount_fractions=floating_leg_daycount_fractions,
          fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
          fixed_leg_coupon=fixed_leg_coupon,
          reference_rate_fn=reference_rate_fn,
          dim=1,
          mean_reversion=mean_reversion_param,
          volatility=volatility_param,
          notional=notional,
          is_payer_swaption=is_payer_swaption,
          use_analytic_pricing=use_analytic_pricing,
          num_samples=num_samples,
          random_type=random_type,
          seed=seed,
          skip=skip,
          time_step=time_step,
          dtype=dtype)[:, 0]

      if volatility_based_calibration:
        model_values = implied_vol(
            prices=model_values / annuity / notional,
            strikes=fixed_leg_coupon[..., 0],
            expiries=expiries,
            forwards=swap_rate,
            is_call_options=is_payer_swaption,
            underlying_distribution=UnderlyingDistribution.NORMAL,
            dtype=dtype)
        model_values = tf.where(
            tf.math.is_nan(model_values), tf.zeros_like(model_values),
            model_values)

      value = tf.math.reduce_sum(
          (_scale(model_values, target_lb, target_ub) - scaled_target)**2)
      return value

    optimization_result = optimizer_fn(
        loss_function, initial_position=initial_guess, tolerance=tolerance,
        max_iterations=maximum_iterations)
    calibrated_parameters = optimization_result.position
    mean_reversion_calibrated = piecewise.PiecewiseConstantFunc(
        jump_locations=[],
        values=_to_constrained(
            calibrated_parameters[:num_mean_reversion], mr_lb, mr_ub),
        dtype=dtype)
    volatility_calibrated = piecewise.PiecewiseConstantFunc(
        jump_locations=volatility.jump_locations(),
        values=_to_constrained(
            calibrated_parameters[num_mean_reversion:], vol_lb, vol_ub),
        dtype=dtype)

    calibration_result = CalibrationResult(
        mean_reversion=mean_reversion_calibrated,
        volatility=volatility_calibrated)

    return (calibration_result, optimization_result.converged,
            optimization_result.num_iterations)


def calibration_from_cap_floors(
    *,
    prices: types.RealTensor,
    strikes: types.RealTensor,
    expiries: types.RealTensor,
    maturities: types.RealTensor,
    daycount_fractions: types.RealTensor,
    reference_rate_fn,
    mean_reversion: Union[types.RealTensor, Callable[..., types.RealTensor]],
    volatility: Union[types.RealTensor, Callable[..., types.RealTensor]],
    notional: types.RealTensor = None,
    # TODO(b/183418183) Allow for dim > 1
    dim: int = 1,
    is_cap: types.BoolTensor = True,
    use_analytic_pricing: bool = True,
    num_samples: types.IntTensor = 1,
    random_type: random.RandomType = None,
    seed: types.IntTensor = None,
    skip: types.IntTensor = 0,
    time_step: types.RealTensor = None,
    optimizer_fn=None,
    mean_reversion_lower_bound=0.001,
    mean_reversion_upper_bound=0.5,
    volatility_lower_bound: types.RealTensor = 0.00001,
    volatility_upper_bound: types.RealTensor = 0.1,
    tolerance: types.RealTensor = 1e-6,
    maximum_iterations: types.IntTensor = 50,
    dtype: tf.DType = None,
    name: str = None) -> Tuple[CalibrationResult,
                               types.BoolTensor,
                               types.IntTensor]:
  """Calibrates the Hull-White model using the observed Cap/Floor prices.

  This function estimates the mean-reversion rate and volatility parameters of
  a Hull-White 1-factor model using a set of Cap/Floor prices as the target.
  The calibration is performed using least-squares optimization where
  the loss function minimizes the squared error between the observed option
  prices and the model implied prices.

  #### Example
  The example shows how to calibrate a Hull-White model with constant mean
  reversion rate and constant volatility.

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  # In this example, we synthetically generate some prices. Then we use our
  # calibration to back out these prices.
  dtype = tf.float64

  daycount_fractions = np.array([
      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
      [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
  ])
  expiries = np.array([
      [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75],
      [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75],
  ])
  maturities = np.array([
      [0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0],
      [0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0],
  ])
  is_cap = np.array([True, False, True, False])
  strikes = 0.01 * np.ones_like(expiries)

  # Setup - generate some observed prices using the model.
  expected_mr = [0.4]
  expected_vol = [0.01]

  zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
  prices = tff.models.hull_white.cap_floor_price(
      strikes=strikes,
      expiries=expiries,
      maturities=maturities,
      daycount_fractions=daycount_fractions,
      reference_rate_fn=zero_rate_fn,
      notional=1.0,
      dim=1,
      mean_reversion=expected_mr,
      volatility=expected_vol,
      is_cap=tf.expand_dims(is_cap, axis=1),
      use_analytic_pricing=True,
      dtype=dtype)

  # Calibrate the model.
  calibrated_model, is_converged, _ = (
      tff.models.hull_white.calibration_from_cap_floors(
          prices=tf.squeeze(prices),
          strikes=strikes,
          expiries=expiries,
          maturities=maturities,
          daycount_fractions=daycount_fractions,
          reference_rate_fn=zero_rate_fn,
          mean_reversion=[0.3],
          volatility=[0.02],
          notional=1.0,
          dim=1,
          is_cap=tf.expand_dims(is_cap, axis=1),
          use_analytic_pricing=True,
          optimizer_fn=None,
          num_samples=1000,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[0, 0],
          time_step=0.1,
          maximum_iterations=200,
          dtype=dtype))

  calibrated_mr = calibrated_model.mean_reversion.values()
  calibrated_vol = calibrated_model.volatility.values()

  # Running this inside a unit test passes:
  #
  # calibrated_mr, calibrated_vol = self.evaluate(
  #     [calibrated_mr, calibrated_vol])
  # self.assertTrue(is_converged)
  # self.assertAllClose(calibrated_mr, expected_mr, atol=1e-3, rtol=1e-2)
  # self.assertAllClose(calibrated_vol, expected_vol, atol=1e-3, rtol=1e-2)

  ````

  Args:
    prices: A real `Tensor` of shape [num_capfloors], holding the prices of
      cap/floors used for calibration; e.g. `prices[i]` holds the price for the
      i-th cap/floor.
    strikes: A real `Tensor` of shape [num_capfloors, num_optionlets], where the
      second axis corresponds to the strikes of the caplets or floorlets
      contained within each option; e.g. `strikes[i, j]` holds the strike price
      for the j-th caplet/floorlet of the i-th cap/floor.
    expiries: A real `Tensor` of shape [num_capfloors, num_optionlets], where
      `expiries[i, j]` holds the reset time for the j-th caplet/floorlet of the
      i-th cap/floor.
    maturities: A real `Tensor` of shape [num_capfloors, num_optionlets], where
      `maturities[i, j]` holds the maturity time (aka the end of accrual) of the
      underlying forward rate for the j-th caplet/floorlet of the i-th
      cap/floor. The payment occurs on the maturity as well.
    daycount_fractions: A real `Tensor` of shape [num_capfloors,
      num_optionlets], where `daycount_fractions[i, j]` holds the daycount
      fractions associated with the underlying forward rate of the j-th
      caplet/floorlet of the i-th cap/floor.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of the same shape and dtype, representing
      the continuously compounded zero rate at the present time for the input
      expiry time.
    mean_reversion: A real positive `Tensor` of shape broadcastable to `[dim]`,
      or a Python callable. The callable can be one of the following:
      (a) A left-continuous piecewise constant object (e.g.,
      `tff.math.piecewise.PiecewiseConstantFunc`) that has a property
      `is_piecewise_constant` set to `True`. In this case the object should
      have a method `jump_locations(self)` that returns a `Tensor` of shape
      `[dim, num_jumps]` or `[num_jumps]`. In the first case,
      `mean_reversion(t)` should return a `Tensor` of shape `[dim] + t.shape`,
      and in the second, `t.shape + [dim]`, where `t` is a rank 1 `Tensor` of
      the same `dtype` as the output. See example in the class docstring.
      (b) A callable that accepts scalars (stands for time `t`) and returns a
      `Tensor` of shape `[dim]`.
      Corresponds to the *initial estimate* of the mean reversion rate for the
      calibration.
    volatility: A real positive `Tensor` of the same `dtype` as
      `mean_reversion` or a callable with the same specs as above.
      Corresponds to the *initial estimate* of the volatility for the
      calibration.
    notional: A real `Tensor` broadcast to [num_capfloors], such that
      `notional[i]` is the notional amount for the i-th cap/floor.
    dim: A Python scalar which corresponds to the number of Hull-White Models to
      be used for pricing.
      Default value: The default value is `1`.
      Currently, dim > 1 is not yet implemented.
    is_cap: A boolean tensor broadcastable to [num_capfloors], such that
      `is_cap[i]` represents whether or not the i-th instrument is a cap (True)
      or floor (False).
    use_analytic_pricing: A Python boolean specifying if cap/floor pricing is
      done analytically during calibration. Analytic valuation is only supported
      for constant `mean_reversion` and piecewise constant `volatility`. If the
      input is `False`, then valuation using Monte-Carlo simulations is
      performed.
      Default value: The default value is `True`.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation of cap/floors. This input is ignored
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
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation. This
      input is ignored during analytic valuation.
      Default value: `None`.
    optimizer_fn: Optional Python callable which implements the algorithm used
      to minimize the objective function during calibration. It should have
      the following interface:  result =
        optimizer_fn(value_and_gradients_function, initial_position, tolerance,
        max_iterations)  `value_and_gradients_function` is a Python callable
        that accepts a point as a real `Tensor` and returns a tuple of `Tensor`s
        of real dtype containing the value of the function and its gradient at
        that point. 'initial_position' is a real `Tensor` containing the
        starting point of the optimization, 'tolerance' is a real scalar
        `Tensor` for stopping tolerance for the procedure and `max_iterations`
        specifies the maximum number of iterations.
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
      limit of Hull White volatility during calibration.
      Default value: 0.00001 (0.1 basis points).
    volatility_upper_bound: An optional scalar `Tensor` specifying the upper
      limit of Hull White volatility during calibration.
      Default value: 0.1.
    tolerance: Scalar `Tensor` of real dtype. The absolute tolerance for
      terminating the iterations.
      Default value: 1e-6.
    maximum_iterations: Scalar positive int32 `Tensor`. The maximum number of
      iterations during the optimization.
      Default value: 50.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred from
      `prices` is used.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
        `hw_capfloor_calibration`.

  Returns:
    A Tuple of three elements:
    * The first element is an instance of `CalibrationResult` whose parameters
      are calibrated to the input cap/floor prices.
    * A `Tensor` of optimization status for each batch element (whether the
      optimization algorithm succeeded in finding the optimal point based on
      the specified convergance criteria).
    * A `Tensor` containing the number of iterations performed by the
      optimization algorithm.
  """
  name = name or 'hw_capfloor_calibration'
  with tf.name_scope(name):
    prices = tf.convert_to_tensor(prices, dtype=dtype, name='prices')
    dtype = dtype or prices.dtype

    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    maturities = tf.convert_to_tensor(
        maturities, dtype=dtype, name='maturities')
    daycount_fractions = tf.convert_to_tensor(
        daycount_fractions, dtype=dtype, name='daycount_fractions')
    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    is_cap = tf.convert_to_tensor(is_cap, name='is_cap', dtype=tf.bool)

    if not hasattr(mean_reversion, 'is_piecewise_constant'):
      mean_reversion = piecewise.PiecewiseConstantFunc(
          jump_locations=[], values=mean_reversion, dtype=dtype)
    if not hasattr(volatility, 'is_piecewise_constant'):
      volatility = piecewise.PiecewiseConstantFunc(
          jump_locations=[], values=volatility, dtype=dtype)

    if optimizer_fn is None:
      optimizer_fn = optimizer.conjugate_gradient_minimize

    target_values = prices
    target_lb = tf.constant(0.0, dtype=dtype)
    target_ub = tf.math.reduce_max(target_values)

    vol_lb = tf.convert_to_tensor(volatility_lower_bound, dtype=dtype)
    vol_ub = tf.convert_to_tensor(volatility_upper_bound, dtype=dtype)
    mr_lb = tf.convert_to_tensor(mean_reversion_lower_bound, dtype=dtype)
    mr_ub = tf.convert_to_tensor(mean_reversion_upper_bound, dtype=dtype)

    initial_guess = tf.concat([
        _to_unconstrained(mean_reversion.values(), mr_lb, mr_ub),
        _to_unconstrained(volatility.values(), vol_lb, vol_ub)
    ], axis=0)
    num_mean_reversion = mean_reversion.values().shape.as_list()[0]
    scaled_target = _scale(target_values, target_lb, target_ub)

    @make_val_and_grad_fn
    def loss_function(x):
      """Loss function for the optimization."""
      x_mr = _to_constrained(x[:num_mean_reversion], mr_lb, mr_ub)
      x_vol = _to_constrained(x[num_mean_reversion:], vol_lb, vol_ub)

      mean_reversion_param = piecewise.PiecewiseConstantFunc(
          jump_locations=[], values=x_mr, dtype=dtype)
      volatility_param = piecewise.PiecewiseConstantFunc(
          jump_locations=volatility.jump_locations(), values=x_vol, dtype=dtype)

      model_values = cap_floor.cap_floor_price(
          strikes=strikes,
          expiries=expiries,
          maturities=maturities,
          daycount_fractions=daycount_fractions,
          reference_rate_fn=reference_rate_fn,
          dim=dim,
          mean_reversion=mean_reversion_param,
          volatility=volatility_param,
          notional=notional,
          is_cap=is_cap,
          use_analytic_pricing=use_analytic_pricing,
          num_samples=num_samples,
          random_type=random_type,
          seed=seed,
          skip=skip,
          time_step=time_step,
          dtype=dtype)[:, 0]

      return tf.math.reduce_mean(
          (_scale(model_values, target_lb, target_ub) - scaled_target)**2)

    optimization_result = optimizer_fn(
        loss_function,
        initial_position=initial_guess,
        tolerance=tolerance,
        max_iterations=maximum_iterations)
    calibrated_parameters = optimization_result.position
    mean_reversion_calibrated = piecewise.PiecewiseConstantFunc(
        jump_locations=[],
        values=_to_constrained(calibrated_parameters[:num_mean_reversion],
                               mr_lb, mr_ub),
        dtype=dtype)
    volatility_calibrated = piecewise.PiecewiseConstantFunc(
        jump_locations=volatility.jump_locations(),
        values=_to_constrained(calibrated_parameters[num_mean_reversion:],
                               vol_lb, vol_ub),
        dtype=dtype)

    calibration_result = CalibrationResult(
        mean_reversion=mean_reversion_calibrated,
        volatility=volatility_calibrated)

    return (calibration_result, optimization_result.converged,
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
