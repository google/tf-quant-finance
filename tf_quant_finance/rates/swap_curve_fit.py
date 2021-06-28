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
"""Methods to construct a swap curve.

Building swap curves is a core problem in mathematical finance. Swap
curves are built using the available market data in liquidly traded fixed income
products. These include LIBOR rates, interest rate swaps, forward rate
agreements (FRAs) or exchange traded futures contracts. This module contains
methods to build swap curve from market data.

The algorithm implemented here uses conjugate gradient optimization to minimize
the weighted least squares error between the input present values of the
instruments and the present values computed using the constructed swap curve.

#### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 6. 2010.
"""

from typing import List, Callable, Any

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math import optimizer as optimizers
from tf_quant_finance.math.interpolation import linear
from tf_quant_finance.rates import swap_curve_common as scc


__all__ = [
    'swap_curve_fit',
]


def swap_curve_fit(
    float_leg_start_times: List[types.RealTensor],
    float_leg_end_times: List[types.RealTensor],
    float_leg_daycount_fractions: List[types.RealTensor],
    fixed_leg_start_times: List[types.RealTensor],
    fixed_leg_end_times: List[types.RealTensor],
    fixed_leg_daycount_fractions: List[types.RealTensor],
    fixed_leg_cashflows: List[types.RealTensor],
    present_values: List[types.RealTensor],
    initial_curve_rates: types.RealTensor,
    present_values_settlement_times: List[types.RealTensor] = None,
    float_leg_discount_rates: List[types.RealTensor] = None,
    float_leg_discount_times: List[types.RealTensor] = None,
    fixed_leg_discount_rates: List[types.RealTensor] = None,
    fixed_leg_discount_times: List[types.RealTensor] = None,
    optimizer: Callable[..., Any] = None,
    curve_interpolator: Callable[..., types.RealTensor] = None,
    instrument_weights: types.RealTensor = None,
    curve_tolerance: types.RealTensor = 1e-8,
    maximum_iterations: types.IntTensor = 50,
    dtype: tf.DType = None,
    name: str = None) -> scc.SwapCurveBuilderResult:
  """Constructs the zero swap curve using optimization.

  A zero swap curve is a function of time which gives the interest rate that
  can be used to project forward rates at arbitrary `t` for the valuation of
  interest rate securities.

  Suppose we have a set of `N` Interest Rate Swaps (IRS) `S_i` with increasing
  expiries whose market prices are known.
  Suppose also that the `i`th IRS issues cashflows at times `T_{ij}` where
  `1 <= j <= n_i` and `n_i` is the number of cashflows (including expiry)
  for the `i`th swap.
  Denote by `T_i` the time of final payment for the `i`th swap
  (hence `T_i = T_{i,n_i}`). This function estimates a set of rates `r(T_i)`
  such that when these rates are interpolated to all other cashflow times,
  the computed value of the swaps matches the market value of the swaps
  (within some tolerance). Rates at intermediate times are interpolated using
  the user specified interpolation method (the default interpolation method
  is linear interpolation on rates).

  #### Example:

  The following example illustrates the usage by building an implied swap curve
  from four vanilla (fixed to float) LIBOR swaps.

  ```python

  dtype = np.float64

  # Next we will set up LIBOR reset and payment times for four spot starting
  # swaps with maturities 1Y, 2Y, 3Y, 4Y. The LIBOR rate spans 6M.

  float_leg_start_times = [
            np.array([0., 0.5], dtype=dtype),
            np.array([0., 0.5, 1., 1.5], dtype=dtype),
            np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5], dtype=dtype),
            np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=dtype)
        ]

  float_leg_end_times = [
            np.array([0.5, 1.0], dtype=dtype),
            np.array([0.5, 1., 1.5, 2.0], dtype=dtype),
            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
        ]

  # Next we will set up start and end times for semi-annual fixed coupons.

  fixed_leg_start_times = [
            np.array([0., 0.5], dtype=dtype),
            np.array([0., 0.5, 1., 1.5], dtype=dtype),
            np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5], dtype=dtype),
            np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=dtype)
        ]

  fixed_leg_end_times = [
            np.array([0.5, 1.0], dtype=dtype),
            np.array([0.5, 1., 1.5, 2.0], dtype=dtype),
            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
        ]

  # Next setup a trivial daycount for floating and fixed legs.

  float_leg_daycount = [
            np.array([0.5, 0.5], dtype=dtype),
            np.array([0.5, 0.5, 0.5, 0.5], dtype=dtype),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype)
        ]

  fixed_leg_daycount = [
            np.array([0.5, 0.5], dtype=dtype),
            np.array([0.5, 0.5, 0.5, 0.5], dtype=dtype),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype)
        ]

  fixed_leg_cashflows = [
        # 1 year swap with 2.855% semi-annual fixed payments.
        np.array([-0.02855, -0.02855], dtype=dtype),
        # 2 year swap with 3.097% semi-annual fixed payments.
        np.array([-0.03097, -0.03097, -0.03097, -0.03097], dtype=dtype),
        # 3 year swap with 3.1% semi-annual fixed payments.
        np.array([-0.031, -0.031, -0.031, -0.031, -0.031, -0.031], dtype=dtype),
        # 4 year swap with 3.2% semi-annual fixed payments.
        np.array([-0.032, -0.032, -0.032, -0.032, -0.032, -0.032, -0.032,
        -0.032], dtype=dtype)
    ]

  # The present values of the above IRS.
  pvs = np.array([0., 0., 0., 0.], dtype=dtype)

  # Initial state of the curve.
  initial_curve_rates = np.array([0.01, 0.01, 0.01, 0.01], dtype=dtype)

  results = swap_curve_fit(float_leg_start_times, float_leg_end_times,
                           float_leg_daycount, fixed_leg_start_times,
                           fixed_leg_end_times, fixed_leg_cashflows,
                           fixed_leg_daycount, pvs, dtype=dtype,
                           initial_curve_rates=initial_curve_rates)

  #### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 6. 2010.

  Args:
    float_leg_start_times: List of `Tensor`s. Each `Tensor` must be either of
      shape `batch_shape  + [k_i]` or `[k_i]` and of the same real dtype. `k_i`
      may be of different sizes. Each `Tensor` represents the beginning of the
      accrual period for the forward rate which determines the floating payment.
      Each element in the list belong to a unique swap to be used to build the
      curve.
    float_leg_end_times: List of `Tensor`s of shapes and `dtype` compatible with
      `float_leg_start_times`. Each `Tensor` represents the end of the
      accrual period for the forward rate which determines the floating payment.
    float_leg_daycount_fractions: List of `Tensor`s of shapes and `dtype`
      compatible with `float_leg_start_times`. Each `Tensor` represents the
      daycount fraction of the forward rate which determines the floating
      payment.
    fixed_leg_start_times: List of `Tensor`s. Each `Tensor` must be either of
      shape `batch_shape  + [n_i]` or `[n_i]` and of the same real dtype.
      `n_i` may be of different sizes. All elements must have the same `dtype`
      as `float_leg_start_times`. Each `Tensor` represents the begining of the
      accrual period fixed coupon.
    fixed_leg_end_times: List of `Tensor`s of shapes and `dtype` compatible with
      `fixed_leg_start_times`. All elements must have the same `dtype` as
      `fixed_leg_start_times`. Each `Tensor` represents the
      end of the accrual period for the fixed coupon.
    fixed_leg_daycount_fractions: List of `Tensor`s of shapes and `dtype`
    compatible with
      `fixed_leg_start_times` Each `Tensor` represents the daycount fraction
      applicable for the fixed payment.
    fixed_leg_cashflows: List of `Tensor`s of shapes and `dtype` compatible with
      `fixed_leg_start_times`. The input contains fixed cashflows at each
      coupon payment time including notional (if any). The sign should be
      negative (positive) to indicate net outgoing (incoming) cashflow.
    present_values: List containing `Tensor`s of the same dtype as
      elements of `fixed_leg_cashflows` and of shapes compatible with
      `batch_shape`. The length of the list must be the same as the length of
      `fixed_leg_cashflows`. The input contains the market price of the
      underlying instruments.
    initial_curve_rates: A `Tensor` of the `dtype` as `present_values` and of
      shape `[batch_shape, num_instruments]` where `num_instruments` is the
      length of `float_leg_start_times`. The starting guess for the discount
      rates used to initialize the iterative procedure.
    present_values_settlement_times: Optional list of `Tensor`s with the shapes
      and `dtype` compatible with `present_values` The settlement times for the
      present values is the time from now when the instrument is traded to the
      time that the purchase price is actually delivered. If not supplied, then
      it is assumed that the settlement times are zero for every bond.
      Default value: `None` which is equivalent to zero settlement times.
    float_leg_discount_rates: Optional list of `Tensor`s with the shapes
      and `dtype` compatible with `present_values`. This input contains the
      continuously compounded discount rates the will be used
      to discount the floating cashflows. This allows the swap curve to
      constructed using an independent discount curve (e.g. OIS curve). By
      default the cashflows are discounted using the curve that is being
      constructed.
    float_leg_discount_times: Optional list of `Tensor`s with the shapes
      and `dtype` compatible with `present_values`. This input contains the
      times corresponding to the rates specified via
      the `float_leg_discount_rates`.
    fixed_leg_discount_rates: Optional list of `Tensor`s with the shapes
      and `dtype` compatible with `present_values`. This input contains the
      continuously compounded discount rates the will be used to discount the
      fixed cashflows. This allows the swap curve to constructed using an
      independent discount curve (e.g. OIS curve). By default the cashflows are
      discounted using the curve that is being constructed.
    fixed_leg_discount_times: Optional list of `Tensor`s with the shapes
      and `dtype` compatible with `present_values`. This input contains the
      times corresponding to the rates specified via the
      `fixed_leg_discount_rates`.
    optimizer: Optional Python callable which implements the algorithm used
      to minimize the objective function during calibration. It should have
      the following interface: result =
        optimizer(value_and_gradients_function, initial_position, tolerance,
        max_iterations) `value_and_gradients_function` is a Python callable that
        accepts a point as a real `Tensor` and returns a tuple of `Tensor`s of
        real dtype containing the value of the function and its gradient at that
        point. 'initial_position' is a real `Tensor` containing the starting
        point of the optimization, 'tolerance' is a real scalar `Tensor` for
        stopping tolerance for the procedure and `max_iterations` specifies the
        maximum number of iterations.
      `optimizer` should return a namedtuple containing the items: `position`
        (a tensor containing the optimal value), `converged` (a boolean
        indicating whether the optimize converged according the specified
        criteria), `failed` (a boolean indicating if the optimization resulted
        in a failure), `num_iterations` (the number of iterations used), and
        `objective_value` ( the value of the objective function at the optimal
        value). The default value for `optimizer` is None and conjugate
        gradient algorithm is used.
      Default value: `None` - indicating conjugate gradient minimizer.
    curve_interpolator: Optional Python callable used to interpolate the zero
      swap rates at cashflow times. It should have the following interface:
      yi = curve_interpolator(xi, x, y)
      `x`, `y`, 'xi', 'yi' are all `Tensors` of real dtype. `x` and `y` are the
      sample points and values (respectively) of the function to be
      interpolated. `xi` are the points at which the interpolation is
      desired and `yi` are the corresponding interpolated values returned by the
      function. The default value for `curve_interpolator` is None in which
      case linear interpolation is used.
      Default value: `None`. If not supplied, the yields to maturity for the
        bonds is used as the initial value.
    instrument_weights: Optional 'Tensor' of the same dtype and shape as
      `initial_curve_rates`. This input contains the weight of each instrument
      in computing the objective function for the conjugate gradient
      optimization. By default the weights are set to be the inverse of
      maturities.
    curve_tolerance: Optional positive scalar `Tensor` of same dtype as
      elements of `bond_cashflows`. The absolute tolerance for terminating the
      iterations used to fit the rate curve. The iterations are stopped when the
      estimated discounts at the expiry times of the bond_cashflows change by a
      amount smaller than `discount_tolerance` in an iteration.
      Default value: 1e-8.
    maximum_iterations: Optional positive integer `Tensor`. The maximum number
      of iterations permitted when fitting the curve.
      Default value: 50.
    dtype: `tf.Dtype`. If supplied the dtype for the (elements of)
      `float_leg_start_times`, and `fixed_leg_start_times`.
      Default value: None which maps to the default dtype inferred by
      TensorFlow.
    name: Python str. The name to give to the ops created by this function.
      Default value: `None` which maps to 'swap_curve'.

  Returns:
    curve_builder_result: An instance of `SwapCurveBuilderResult` containing the
      following attributes.
      times: Rank 1 real `Tensor`. Times for the computed discount rates. These
        are chosen to be the expiry times of the supplied cashflows.
      discount_rates: Rank 1 `Tensor` of the same dtype as `times`.
        The inferred discount rates.
      discount_factor: Rank 1 `Tensor` of the same dtype as `times`.
        The inferred discount factors.
      initial_discount_rates: Rank 1 `Tensor` of the same dtype as `times`. The
        initial guess for the discount rates.
      converged: Scalar boolean `Tensor`. Whether the procedure converged.
        The procedure is said to have converged when the maximum absolute
        difference in the discount factors from one iteration to the next falls
        below the `discount_tolerance`.
      failed: Scalar boolean `Tensor`. Whether the procedure failed. Procedure
        may fail either because a NaN value was encountered for the discount
        rates or the discount factors.
      iterations: Scalar int32 `Tensor`. Number of iterations performed.
      objective_value: Scalar real `Tensor`. The value of the ibjective function
        evaluated using the fitted swap curve.

  Raises:
    ValueError: If the initial state of the curve is not
    supplied to the function.

  """

  with tf.name_scope(name or 'swap_curve'):
    if optimizer is None:
      optimizer = optimizers.conjugate_gradient_minimize

    present_values = _convert_to_tensors(dtype, present_values,
                                         'present_values')
    dtype = present_values[0].dtype
    if present_values_settlement_times is None:
      pv_settlement_times = [tf.zeros([], dtype=dtype) for pv in present_values]
    else:
      pv_settlement_times = _convert_to_tensors(
          dtype, present_values_settlement_times, 'pv_settlement_times')

    float_leg_start_times = _convert_to_tensors(dtype, float_leg_start_times,
                                                'float_leg_start_times')
    float_leg_end_times = _convert_to_tensors(dtype, float_leg_end_times,
                                              'float_leg_end_times')
    float_leg_daycount_fractions = _convert_to_tensors(
        dtype, float_leg_daycount_fractions, 'float_leg_daycount_fractions')
    fixed_leg_start_times = _convert_to_tensors(dtype, fixed_leg_start_times,
                                                'fixed_leg_start_times')
    fixed_leg_end_times = _convert_to_tensors(dtype, fixed_leg_end_times,
                                              'fixed_leg_end_times')
    fixed_leg_daycount_fractions = _convert_to_tensors(
        dtype, fixed_leg_daycount_fractions, 'fixed_leg_daycount_fractions')
    fixed_leg_cashflows = _convert_to_tensors(dtype, fixed_leg_cashflows,
                                              'fixed_leg_cashflows')

    present_values = tf.stack(present_values, axis=-1)
    if instrument_weights is None:
      instrument_weights = _initialize_instrument_weights(float_leg_end_times,
                                                          fixed_leg_end_times,
                                                          dtype=dtype)
    else:
      instrument_weights = _convert_to_tensors(dtype, instrument_weights,
                                               'instrument_weights')

    if curve_interpolator is None:
      def default_interpolator(xi, x, y):
        return linear.interpolate(xi, x, y,
                                  dtype=dtype)
      curve_interpolator = default_interpolator
    self_discounting_float_leg = False
    self_discounting_fixed_leg = False
    # Determine how the floating and fixed leg will be discounted. If separate
    # discount curves for each leg are not specified, the curve will be self
    # discounted using the swap curve.
    if float_leg_discount_rates is None and fixed_leg_discount_rates is None:
      self_discounting_float_leg = True
      self_discounting_fixed_leg = True
      float_leg_discount_rates = [0.0]
      float_leg_discount_times = [0.]
      fixed_leg_discount_rates = [0.]
      fixed_leg_discount_times = [0.]
    elif fixed_leg_discount_rates is None:
      fixed_leg_discount_rates = float_leg_discount_rates
      fixed_leg_discount_times = float_leg_discount_times
    elif float_leg_discount_rates is None:
      self_discounting_float_leg = True
      float_leg_discount_rates = [0.]
      float_leg_discount_times = [0.]

    # Create tensors for discounting curves
    float_leg_discount_rates = _convert_to_tensors(dtype,
                                                   float_leg_discount_rates,
                                                   'float_disc_rates')
    float_leg_discount_rates = tf.stack(float_leg_discount_rates, axis=-1)
    float_leg_discount_times = _convert_to_tensors(dtype,
                                                   float_leg_discount_times,
                                                   'float_disc_times')
    float_leg_discount_times = tf.stack(float_leg_discount_times, axis=-1)
    fixed_leg_discount_rates = _convert_to_tensors(dtype,
                                                   fixed_leg_discount_rates,
                                                   'fixed_disc_rates')
    fixed_leg_discount_rates = tf.stack(fixed_leg_discount_rates, axis=-1)
    fixed_leg_discount_times = _convert_to_tensors(dtype,
                                                   fixed_leg_discount_times,
                                                   'fixed_disc_times')
    fixed_leg_discount_times = tf.stack(fixed_leg_discount_times, axis=-1)

    if initial_curve_rates is not None:
      initial_rates = tf.convert_to_tensor(
          initial_curve_rates, dtype=dtype, name='initial_rates')
    else:
      # TODO(b/144600429): Create a logic for a meaningful initial state of the
      # curve
      raise ValueError('Initial state of the curve is not specified.')

    return _build_swap_curve(float_leg_start_times,
                             float_leg_end_times,
                             float_leg_daycount_fractions,
                             fixed_leg_start_times,
                             fixed_leg_end_times,
                             fixed_leg_cashflows,
                             fixed_leg_daycount_fractions,
                             float_leg_discount_rates,
                             float_leg_discount_times,
                             fixed_leg_discount_rates,
                             fixed_leg_discount_times,
                             self_discounting_float_leg,
                             self_discounting_fixed_leg,
                             present_values,
                             pv_settlement_times,
                             optimizer,
                             curve_interpolator,
                             initial_rates,
                             instrument_weights,
                             curve_tolerance,
                             maximum_iterations)


def _build_swap_curve(float_leg_start_times, float_leg_end_times,
                      float_leg_daycount_fractions, fixed_leg_start_times,
                      fixed_leg_end_times, fixed_leg_cashflows,
                      fixed_leg_daycount_fractions, float_leg_discount_rates,
                      float_leg_discount_times, fixed_leg_discount_rates,
                      fixed_leg_discount_times, self_discounting_float_leg,
                      self_discounting_fixed_leg, present_values,
                      pv_settlement_times, optimizer, curve_interpolator,
                      initial_rates, instrument_weights, curve_tolerance,
                      maximum_iterations):
  """Build the zero swap curve."""
  # The procedure uses optimization to estimate the swap curve as follows:
  # 1. Start with an initial state of the swap curve.
  # 2. Define a loss function which measures the deviations between model prices
  #   of the IR swaps and their present values specified as input.
  # 3. Use numerical optimization (currently conjugate gradient optimization) to
  #   to build the swap curve such that the loss function is minimized.
  del fixed_leg_start_times, float_leg_daycount_fractions
  curve_tensors = _create_curve_building_tensors(
      float_leg_start_times, float_leg_end_times, fixed_leg_end_times,
      pv_settlement_times)
  expiry_times = curve_tensors.expiry_times
  calc_groups_float = curve_tensors.calc_groups_float
  calc_groups_fixed = curve_tensors.calc_groups_fixed
  settle_times_float = curve_tensors.settle_times_float
  settle_times_fixed = curve_tensors.settle_times_fixed

  float_leg_calc_times_start = tf.concat(float_leg_start_times, axis=-1)
  float_leg_calc_times_end = tf.concat(float_leg_end_times, axis=-1)
  calc_fixed_leg_cashflows = tf.concat(fixed_leg_cashflows, axis=-1)
  calc_fixed_leg_daycount = tf.concat(fixed_leg_daycount_fractions, axis=-1)
  fixed_leg_calc_times = tf.concat(fixed_leg_end_times, axis=-1)

  def _interpolate(x1, x_data, y_data):
    return curve_interpolator(x1, x_data, y_data)

  @make_val_and_grad_fn
  def loss_function(x):
    """Loss function for the optimization."""
    # Currently the loss function is a weighted root mean squared difference
    # between the model PV and market PV. The model PV is interest rate swaps is
    # computed as follows:

    # 1. Interpolate the swap curve at intermediate times required to compute
    #   forward rates for the computation of floating cashflows.
    # 2. Interpolate swap curve or the discount curve (if a separate discount
    #   curve is specified) at intermediate cashflow times.
    # 3. Compute the PV of the swap as the aggregate of floating and fixed legs.
    # 4. Compute the loss (which is being minized) as the weighted root mean
    #   squared difference between the model PV (computed above) and the market
    #   PV (specified as input).

    rates_start = _interpolate(float_leg_calc_times_start, expiry_times, x)
    rates_end = _interpolate(float_leg_calc_times_end, expiry_times, x)
    float_cashflows = (
        tf.math.exp(float_leg_calc_times_end * rates_end
                    - float_leg_calc_times_start * rates_start) - 1.)

    if self_discounting_float_leg:
      float_discount_rates = rates_end
      float_settle_rates = _interpolate(settle_times_float, expiry_times, x)
    else:
      float_discount_rates = _interpolate(float_leg_calc_times_end,
                                          float_leg_discount_times,
                                          float_leg_discount_rates)
      float_settle_rates = _interpolate(settle_times_float,
                                        float_leg_discount_times,
                                        float_leg_discount_rates)
    if self_discounting_fixed_leg:
      fixed_discount_rates = _interpolate(fixed_leg_calc_times, expiry_times, x)
      fixed_settle_rates = _interpolate(settle_times_fixed, expiry_times, x)
    else:
      fixed_discount_rates = _interpolate(fixed_leg_calc_times,
                                          fixed_leg_discount_times,
                                          fixed_leg_discount_rates)
      fixed_settle_rates = _interpolate(settle_times_fixed,
                                        fixed_leg_discount_times,
                                        fixed_leg_discount_rates)

    # exp(-r(t) * t) / exp(-r(t_s) * t_s)
    calc_discounts_float_leg = (
        tf.math.exp(-float_discount_rates * float_leg_calc_times_end +
                    float_settle_rates * settle_times_float))
    calc_discounts_fixed_leg = (
        tf.math.exp(-fixed_discount_rates * fixed_leg_calc_times +
                    fixed_settle_rates * settle_times_fixed))
    # Fast segment sum on `calc_discounts_float_leg` groups
    # TODO(b/178173737): use tf.math.segment_sum once a fast version is
    # supported
    float_pv = tf.linalg.matvec(
        calc_groups_float,
        float_cashflows * calc_discounts_float_leg)
    # Fast segment sum on `calc_fixed_leg_cashflows` groups
    # TODO(b/178173737): use tf.math.segment_sum once a fast version is
    # supported
    fixed_pv = tf.linalg.matvec(
        calc_groups_fixed,
        calc_fixed_leg_daycount * calc_fixed_leg_cashflows *
        calc_discounts_fixed_leg)
    swap_pv = float_pv + fixed_pv
    value = tf.math.reduce_sum(input_tensor=instrument_weights *
                               (swap_pv - present_values)**2, axis=-1)

    return value

  optimization_result = optimizer(
      loss_function, initial_position=initial_rates, tolerance=curve_tolerance,
      max_iterations=maximum_iterations)

  discount_rates = optimization_result.position
  discount_factors = tf.math.exp(-discount_rates * expiry_times)
  results = scc.SwapCurveBuilderResult(
      times=expiry_times,
      rates=discount_rates,
      discount_factors=discount_factors,
      initial_rates=initial_rates,
      converged=optimization_result.converged,
      failed=optimization_result.failed,
      iterations=optimization_result.num_iterations,
      objective_value=optimization_result.objective_value)
  return results


def _convert_to_tensors(dtype, input_array, name):
  """Converts the supplied list to a tensor."""

  output_tensor = [
      tf.convert_to_tensor(
          x, dtype=dtype, name=name + '_{}'.format(i))
      for i, x in enumerate(input_array)
  ]

  return output_tensor


def _initialize_instrument_weights(float_times, fixed_times, dtype):
  """Function to compute default initial weights for optimization."""
  weights = tf.ones(len(float_times), dtype=dtype)
  one = tf.ones([], dtype=dtype)
  float_times_last = tf.stack([times[-1] for times in float_times])
  fixed_times_last = tf.stack([times[-1] for times in fixed_times])
  weights = tf.maximum(
      tf.math.divide_no_nan(one, float_times_last),
      tf.math.divide_no_nan(one, fixed_times_last))
  weights = tf.minimum(one, weights)
  return tf.unstack(weights, name='instrument_weights')


@utils.dataclass
class CurveFittingVars:
  """Curve fitting variables."""
  # The `Tensor` of maturities at which the curve will be built.
  # Coorspond to maturities on the underlying instruments
  expiry_times: types.RealTensor
  # `Tensor` containing the instrument index of each floating cashflow
  calc_groups_float: types.IntTensor
  # `Tensor` containing the instrument index of each fixed cashflow
  calc_groups_fixed: types.IntTensor
  # `Tensor` containing the settlement time of each floating cashflow
  settle_times_float: types.RealTensor
  # `Tensor` containing the settlement time of each fixed cashflow
  settle_times_fixed: types.RealTensor


def _create_curve_building_tensors(float_leg_start_times,
                                   float_leg_end_times,
                                   fixed_leg_end_times,
                                   pv_settlement_times):
  """Helper function to create tensors needed for curve construction."""
  calc_groups_float = []
  calc_groups_fixed = []
  expiry_times = []
  settle_times_float = []
  settle_times_fixed = []
  num_instruments = len(float_leg_start_times)
  for i in range(num_instruments):
    expiry_times.append(
        tf.math.maximum(float_leg_end_times[i][-1], fixed_leg_end_times[i][-1]))

    calc_groups_float.append(
        tf.fill(tf.shape(float_leg_start_times[i]), i))
    calc_groups_fixed.append(tf.fill(tf.shape(fixed_leg_end_times[i]), i))
    settle_time = pv_settlement_times[i]
    if settle_time.shape.rank > 0:
      settle_time = tf.expand_dims(settle_time, axis=-1)
    # shape [batch_shape, float_leg_start_times[i].shape]
    stf = settle_time + tf.zeros_like(float_leg_start_times[i])
    settle_times_float.append(stf)
    # shape [batch_shape, fixed_leg_end_times[i].shape]
    stf = settle_time + tf.zeros_like(fixed_leg_end_times[i])
    settle_times_fixed.append(stf)

  expiry_times = tf.stack(expiry_times, axis=0)
  dtype = expiry_times.dtype
  # Float let groups
  num_groups_float = len(calc_groups_float)  # Number of instruments
  calc_groups_float = tf.concat(calc_groups_float, axis=-1)
  # Shape batch_shape + [num_groups_float, calc_groups_float.shape[0]]
  axis = calc_groups_float.shape.rank - 1
  calc_groups_float_mat = tf.one_hot(
      calc_groups_float, num_groups_float, axis=axis, dtype=dtype)
  # Fixed leg groups
  num_groups_fixed = len(calc_groups_fixed)
  calc_groups_fixed = tf.concat(calc_groups_fixed, axis=-1)
  # Shape batch_shape + [num_groups_fixed, calc_groups_fixed.shape[0]]
  axis = calc_groups_fixed.shape.rank - 1
  calc_groups_fixed_mat = tf.one_hot(
      calc_groups_fixed, num_groups_fixed, axis=axis, dtype=dtype)
  settle_times_float = tf.concat(settle_times_float, axis=-1)
  settle_times_fixed = tf.concat(settle_times_fixed, axis=-1)

  return CurveFittingVars(expiry_times=expiry_times,
                          calc_groups_float=calc_groups_float_mat,
                          calc_groups_fixed=calc_groups_fixed_mat,
                          settle_times_float=settle_times_float,
                          settle_times_fixed=settle_times_fixed)
