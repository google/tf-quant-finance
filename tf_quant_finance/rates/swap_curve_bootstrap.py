# Copyright 2020 Google LLC
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

The algorithm implemented here uses the bootstrap method to iteratively
construct the swap curve. It decouples interpolation from bootstrapping so that
any arbitrary interpolation scheme could be used to build the curve.

#### References:

[1]: Patrick Hagan & Graeme West. Interpolation Methods for Curve Construction.
  Applied Mathematical Finance. Vol 13, No. 2, pp 89-129. June 2006.
  https://www.researchgate.net/publication/24071726_Interpolation_Methods_for_Curve_Construction
"""

from typing import List, Callable

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.math.interpolation import linear
# TODO(b/148945638): Move common functionality for swap curve construction to a
# separate python module.
from tf_quant_finance.rates import swap_curve_common as scc


__all__ = [
    'swap_curve_bootstrap',
]


def swap_curve_bootstrap(
    float_leg_start_times: List[types.RealTensor],
    float_leg_end_times: List[types.RealTensor],
    fixed_leg_start_times: List[types.RealTensor],
    fixed_leg_end_times: List[types.RealTensor],
    fixed_leg_cashflows: List[types.RealTensor],
    present_values: List[types.RealTensor],
    present_values_settlement_times: List[types.RealTensor] = None,
    float_leg_daycount_fractions: List[types.RealTensor] = None,
    fixed_leg_daycount_fractions: List[types.RealTensor] = None,
    float_leg_discount_rates: List[types.RealTensor] = None,
    float_leg_discount_times: List[types.RealTensor] = None,
    fixed_leg_discount_rates: List[types.RealTensor] = None,
    fixed_leg_discount_times: List[types.RealTensor] = None,
    curve_interpolator: Callable[..., types.RealTensor] = None,
    initial_curve_rates: types.RealTensor = None,
    curve_tolerance: types.RealTensor = 1e-8,
    maximum_iterations: types.IntTensor = 50,
    dtype: tf.DType = None,
    name: str = None) -> scc.SwapCurveBuilderResult:
  """Constructs the zero swap curve using bootstrap method.

  A zero swap curve is a function of time which gives the interest rate that
  can be used to project forward rates at arbitrary `t` for the valuation of
  interest rate securities (e.g. FRAs, Interest rate futures, Swaps etc.).

  Suppose we have a set of `N` Interest Rate Swaps (IRS) `S_i` with increasing
  expiries whose market prices are known.
  Suppose also that the `i`th IRS issues cashflows at times `T_{ij}` where
  `1 <= j <= n_i` and `n_i` is the number of cashflows (including expiry)
  for the `i`th swap.
  Denote by `T_i` the time of final payment for the `i`th swap
  (hence `T_i = T_{i,n_i}`). This function estimates a set of rates `r(T_i)`
  such that when these rates are interpolated (using the user specified
  interpolation method) to all other cashflow times, the computed value of the
  swaps matches the market value of the swaps (within some tolerance).

  The algorithm implemented here uses the bootstrap method to iteratively
  construct the swap curve [1].

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

  results = swap_curve_bootstrap(float_leg_start_times, float_leg_end_times,
                                 float_leg_daycount, fixed_leg_start_times,
                                 fixed_leg_end_times, fixed_leg_cashflows,
                                 fixed_leg_daycount, pvs, dtype=dtype,
                                 initial_curve_rates=initial_curve_rates)

  #### References:

  [1]: Patrick Hagan & Graeme West. Interpolation Methods for Curve
    Construction. Applied Mathematical Finance. Vol 13, No. 2, pp 89-129.
    June 2006.
    https://www.researchgate.net/publication/24071726_Interpolation_Methods_for_Curve_Construction

  Args:
    float_leg_start_times: List of `Tensor`s. Each `Tensor` must be of rank 1
      and of the same real dtype. They may be of different sizes. Each `Tensor`
      represents the beginning of the accrual period for the forward rate which
      determines the floating payment. Each element in the list belong to a
      unique swap to be used to build the curve.
    float_leg_end_times: List of `Tensor`s. Each `Tensor` must be of rank 1 and
      and the same shape and of the same real dtype as the corresponding element
      in `float_leg_start_times`. Each `Tensor` represents the end of the
      accrual period for the forward rate which determines the floating payment.
    fixed_leg_start_times: List of `Tensor`s. Each `Tensor` must be of rank 1
      and of the same real dtype. They may be of different sizes. Each `Tensor`
      represents the beginning of the accrual period fixed coupon.
    fixed_leg_end_times: List of `Tensor`s. Each `Tensor` must be of the same
      shape and type as `fixed_leg_start_times`. Each `Tensor` represents the
      end of the accrual period for the fixed coupon.
    fixed_leg_cashflows: List of `Tensor`s. The list must be of the same length
      as the `fixed_leg_start_times`. Each `Tensor` must be of rank 1 and of the
      same dtype as the `Tensor`s in `fixed_leg_start_times`. The input contains
      fixed cashflows at each coupon payment time including notional (if any).
      The sign should be negative (positive) to indicate net outgoing (incoming)
      cashflow.
    present_values: List containing scalar `Tensor`s of the same dtype as
      elements of `fixed_leg_cashflows`. The length of the list must be the same
      as the length of `fixed_leg_cashflows`. The input contains the market
      price of the underlying instruments.
    present_values_settlement_times: List containing scalar `Tensor`s of the
      same dtype as elements of `present_values`. The length of the list must be
      the same as the length of `present_values`. The settlement times for the
      present values is the time from now when the instrument is traded to the
      time that the purchase price is actually delivered. If not supplied, then
      it is assumed that the settlement times are zero for every bond.
      Default value: `None`, which is equivalent to zero settlement times.
    float_leg_daycount_fractions: Optional list of `Tensor`s. Each `Tensor` must
      be of the same shape and type as `float_leg_start_times`. They may be of
      different sizes. Each `Tensor` represents the daycount fraction of the
      forward rate which determines the floating payment.
      Default value: `None`, If omitted the daycount fractions are computed as
      the difference between float_leg_end_times and float_leg_start_times.
    fixed_leg_daycount_fractions: Optional list of `Tensor`s. Each `Tensor` must
      be of the same shape and type as `fixed_leg_start_times`. Each `Tensor`
      represents the daycount fraction applicable for the fixed payment.
      Default value: `None`, If omitted the daycount fractions are computed as
      the difference between fixed_leg_end_times and fixed_leg_start_times.
    float_leg_discount_rates: Optional `Tensor` of the same dtype as
      `initial_discount_rates`. This input contains the continuously compounded
      discount rates the will be used to discount the floating cashflows. This
      allows the swap curve to constructed using an independent discount curve
      (e.g. OIS curve).
      Default value: `None`, in which case the cashflows are discounted using
      the curve that is being constructed.
    float_leg_discount_times: Optional `Tensor` of the same dtype and shape as
      `float_leg_discount_rates`. This input contains the times corresponding to
      the rates specified via the `float_leg_discount_rates`.
    fixed_leg_discount_rates: Optional `Tensor` of the same dtype as
      `initial_discount_rates`. This input contains the continuously compounded
      discount rates the will be used to discount the fixed cashflows. This
      allows the swap curve to constructed using an independent discount curve
      (e.g. OIS curve).
      Default value: `None`, in which case the cashflows are discounted using
      the curve that is being constructed.
    fixed_leg_discount_times: Optional `Tensor` of the same dtype and shape as
      `fixed_leg_discount_rates`. This input contains the times corresponding to
      the rates specified via the `fixed_leg_discount_rates`.
    curve_interpolator: Optional Python callable used to interpolate the zero
      swap rates at cashflow times. It should have the following interface:
      yi = curve_interpolator(xi, x, y)
      `x`, `y`, 'xi', 'yi' are all `Tensors` of real dtype. `x` and `y` are the
      sample points and values (respectively) of the function to be
      interpolated. `xi` are the points at which the interpolation is
      desired and `yi` are the corresponding interpolated values returned by the
      function.
      Default value: `None`, which maps to linear interpolation.
    initial_curve_rates: Optional `Tensor` of the same dtype and shape as
      `present_values`. The starting guess for the discount rates used to
      initialize the iterative procedure.
      Default value: `None`. If not supplied, the yields to maturity for the
        bonds is used as the initial value.
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
      times: Rank 1 real `Tensor`. Times for the computed rates. These
        are chosen to be the expiry times of the supplied instruments.
      rates: Rank 1 `Tensor` of the same dtype as `times`.
        The inferred zero rates.
      discount_factor: Rank 1 `Tensor` of the same dtype as `times`.
        The inferred discount factors.
      initial_rates: Rank 1 `Tensor` of the same dtype as `times`. The
        initial guess for the rates.
      converged: Scalar boolean `Tensor`. Whether the procedure converged.
        The procedure is said to have converged when the maximum absolute
        difference in the discount factors from one iteration to the next falls
        below the `discount_tolerance`.
      failed: Scalar boolean `Tensor`. Whether the procedure failed. Procedure
        may fail either because a NaN value was encountered for the discount
        rates or the discount factors.
      iterations: Scalar int32 `Tensor`. Number of iterations performed.

  Raises:
    ValueError: If the initial state of the curve is not
    supplied to the function.

  """

  name = name or 'swap_curve_bootstrap'
  with tf.name_scope(name):

    if curve_interpolator is None:
      def default_interpolator(xi, x, y):
        return linear.interpolate(xi, x, y, dtype=dtype)
      curve_interpolator = default_interpolator

    if present_values_settlement_times is None:
      pv_settle_times = [tf.zeros_like(pv) for pv in present_values]
    else:
      pv_settle_times = present_values_settlement_times

    if float_leg_daycount_fractions is None:
      float_leg_daycount_fractions = [
          y - x for x, y in zip(float_leg_start_times, float_leg_end_times)
      ]

    if fixed_leg_daycount_fractions is None:
      fixed_leg_daycount_fractions = [
          y - x for x, y in zip(fixed_leg_start_times, fixed_leg_end_times)
      ]

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
    present_values = _convert_to_tensors(dtype, present_values,
                                         'present_values')
    pv_settle_times = _convert_to_tensors(dtype, pv_settle_times,
                                          'pv_settle_times')

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
    float_leg_discount_times = _convert_to_tensors(dtype,
                                                   float_leg_discount_times,
                                                   'float_disc_times')
    fixed_leg_discount_rates = _convert_to_tensors(dtype,
                                                   fixed_leg_discount_rates,
                                                   'fixed_disc_rates')
    fixed_leg_discount_times = _convert_to_tensors(dtype,
                                                   fixed_leg_discount_times,
                                                   'fixed_disc_times')

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
                             pv_settle_times,
                             curve_interpolator,
                             initial_rates,
                             curve_tolerance,
                             maximum_iterations,
                             dtype)


def _build_swap_curve(float_leg_start_times, float_leg_end_times,
                      float_leg_daycount_fractions, fixed_leg_start_times,
                      fixed_leg_end_times, fixed_leg_cashflows,
                      fixed_leg_daycount_fractions, float_leg_discount_rates,
                      float_leg_discount_times, fixed_leg_discount_rates,
                      fixed_leg_discount_times, self_discounting_float_leg,
                      self_discounting_fixed_leg, present_values,
                      pv_settlement_times, curve_interpolator,
                      initial_rates, curve_tolerance, maximum_iterations,
                      dtype):
  """Build the zero swap curve using the bootstrap method."""

  # The procedure is recursive and as follows:
  # 1. Start with an initial state of the swap curve. Set this as the current
  #   swap curve.
  # 2. From the current swap curve, compute the relevant forward rates and
  #   discount factors (if cashflows are discounted using the swap curve). Use
  #   the specified interpolation method to compute rates at intermediate times
  #   as needed to calculate either the forward rate or the discount factors.
  # 3. Using the above and the input present values of bootstarpping
  #   instruments,compute the zero swap rate at expiry by inverting the swap
  #   pricing formula. The following illustrates the procedure:

  #   Assuming that a swap pays fixed payments c_i at times t_i (i=[1,...,n])
  #   and receives floating payments at times t_j (j=[1,...,m]), then the
  #   present value of the swap is given by

  #   ```None
  #   PV = sum_{j=1}^m (P_{j-1}/P_j - 1.) * P_j - sum_{i=1}^n a_i * c_i * P_i
  #                                                                        (A)
  #
  #   ```
  #   where P_i = exp(-r(t_i) * t_i) and a_i are the daycount fractions. We
  #   update the current estimate of the rate at curve node t_k = t_n = t_m
  #   by inverting the above equation:

  #   ```None
  #   P_k * (1 + a_i * c_i) = sum_{j=1}^{m - 1} (P_{j-1}/P_j - 1.) * P_j -
  #                           sum_{i=1}^{n - 1} a_i * c_i * P_i - PV       (B)

  #   ```
  #   From Eq. (B), we get r(t_k) = -log(P_k) / t_k.
  #   Using this as the next guess for the discount rates and we repeat the
  #   procedure from Step (2) until convergence.

  del fixed_leg_start_times, pv_settlement_times
  curve_tensors = _create_curve_building_tensors(float_leg_start_times,
                                                 float_leg_end_times,
                                                 float_leg_daycount_fractions,
                                                 fixed_leg_end_times,
                                                 fixed_leg_cashflows,
                                                 fixed_leg_daycount_fractions)

  float_leg_calc_times_start = curve_tensors.float_leg_times_start
  float_leg_calc_times_end = curve_tensors.float_leg_times_end
  calc_fixed_leg_cashflows = curve_tensors.fixed_leg_cashflows
  calc_fixed_leg_daycount = curve_tensors.fixed_leg_daycount
  fixed_leg_calc_times = curve_tensors.fixed_leg_calc_times
  calc_groups_float = curve_tensors.calc_groups_float
  calc_groups_fixed = curve_tensors.calc_groups_fixed
  last_float_leg_start_time = curve_tensors.last_float_leg_start_time
  last_float_leg_end_time = curve_tensors.last_float_leg_end_time
  last_fixed_leg_end_time = curve_tensors.last_fixed_leg_calc_time
  last_fixed_leg_daycount = curve_tensors.last_fixed_leg_daycount
  last_fixed_leg_cashflows = curve_tensors.last_fixed_leg_cashflows
  expiry_times = curve_tensors.expiry_times

  def _one_step(converged, failed, iteration, discount_factor):
    """One step of the bootstrap iteration."""

    x = -tf.math.log(discount_factor) / expiry_times
    rates_start = curve_interpolator(float_leg_calc_times_start, expiry_times,
                                     x)
    rates_end = curve_interpolator(float_leg_calc_times_end, expiry_times, x)
    rates_start_last = curve_interpolator(last_float_leg_start_time,
                                          expiry_times, x)

    float_cashflows = (
        tf.math.exp(float_leg_calc_times_end * rates_end) /
        tf.math.exp(float_leg_calc_times_start * rates_start) - 1.)

    if self_discounting_float_leg:
      float_discount_rates = rates_end
    else:
      float_discount_rates = curve_interpolator(float_leg_calc_times_end,
                                                float_leg_discount_times,
                                                float_leg_discount_rates)
      last_float_discount_rate = curve_interpolator(last_float_leg_end_time,
                                                    float_leg_discount_times,
                                                    float_leg_discount_rates)
      last_float_discount_factor = tf.math.exp(-last_float_discount_rate *
                                               last_float_leg_end_time)
    if self_discounting_fixed_leg:
      fixed_discount_rates = curve_interpolator(fixed_leg_calc_times,
                                                expiry_times, x)
    else:
      fixed_discount_rates = curve_interpolator(fixed_leg_calc_times,
                                                fixed_leg_discount_times,
                                                fixed_leg_discount_rates)
      last_fixed_discount_rate = curve_interpolator(last_fixed_leg_end_time,
                                                    fixed_leg_discount_times,
                                                    fixed_leg_discount_rates)
      last_fixed_discount_factor = tf.math.exp(-last_fixed_leg_end_time *
                                               last_fixed_discount_rate)
      last_fixed_leg_cashflow_pv = (
          last_fixed_leg_daycount * last_fixed_leg_cashflows *
          last_fixed_discount_factor)

    calc_discounts_float_leg = tf.math.exp(-float_discount_rates *
                                           float_leg_calc_times_end)
    calc_discounts_fixed_leg = tf.math.exp(-fixed_discount_rates *
                                           fixed_leg_calc_times)

    float_pv = tf.math.segment_sum(float_cashflows * calc_discounts_float_leg,
                                   calc_groups_float)
    fixed_pv = tf.math.segment_sum(
        calc_fixed_leg_daycount * calc_fixed_leg_cashflows *
        calc_discounts_fixed_leg, calc_groups_fixed)

    if self_discounting_float_leg and self_discounting_fixed_leg:
      p_n_minus_1 = tf.math.exp(-rates_start_last * last_float_leg_start_time)
      scale = last_fixed_leg_cashflows * last_fixed_leg_daycount - 1.
      next_discount = (present_values - float_pv - fixed_pv -
                       p_n_minus_1) / scale
    elif self_discounting_float_leg:
      p_n_minus_1 = tf.math.exp(-rates_start_last * last_float_leg_start_time)
      next_discount = (float_pv + (fixed_pv + last_fixed_leg_cashflow_pv) -
                       present_values + p_n_minus_1)
    else:
      p_n_minus_1 = tf.math.exp(-rates_start_last * last_float_leg_start_time)
      scale = present_values - float_pv - (
          fixed_pv + last_fixed_leg_cashflow_pv) + last_float_discount_factor
      next_discount = p_n_minus_1 * last_float_discount_factor / scale

    discount_diff = tf.math.abs(next_discount - discount_factor)
    converged = (~tf.math.reduce_any(tf.math.is_nan(discount_diff)) &
                 (tf.math.reduce_max(discount_diff) < curve_tolerance))

    return (converged, failed, iteration + 1, next_discount)

  def cond(converged, failed, iteration, x):
    # Note we do not need to check iteration count here because that
    # termination mode is imposed by the maximum_iterations parameter in the
    # while loop.
    del iteration, x
    return ~tf.math.logical_or(converged, failed)

  initial_vals = (False, False, 0, tf.math.exp(-initial_rates * expiry_times))
  bootstrap_result = tf.compat.v2.while_loop(
      cond, _one_step, initial_vals, maximum_iterations=maximum_iterations)

  discount_factors = bootstrap_result[-1]
  discount_rates = -tf.math.log(discount_factors) / expiry_times
  results = scc.SwapCurveBuilderResult(
      times=expiry_times,
      rates=discount_rates,
      discount_factors=discount_factors,
      initial_rates=initial_rates,
      converged=bootstrap_result[0],
      failed=bootstrap_result[1],
      iterations=bootstrap_result[2],
      objective_value=tf.constant(0, dtype=dtype))
  return results


def _convert_to_tensors(dtype, input_array, name):
  """Converts the supplied list to a tensor."""

  output_tensor = [
      tf.convert_to_tensor(
          x, dtype=dtype, name=name + '_{}'.format(i))
      for i, x in enumerate(input_array)
  ]

  return output_tensor


@utils.dataclass
class CurveFittingVars:
  """Curve fitting variables."""
  # The `Tensor` of maturities at which the curve will be built.
  # Coorspond to maturities on the underlying instruments
  expiry_times: types.RealTensor
  # `Tensor` containing fixed leg cashflows from all instruments
  # "flattened" (or concatenated)
  fixed_leg_cashflows: types.RealTensor
  # `Tensor` containing daycount associated with fixed leg cashflows
  fixed_leg_daycount: types.RealTensor
  # `Tensor` containing the times at which fixed cashflows are discouted
  fixed_leg_calc_times: types.RealTensor
  # `Tensor` containing the instrument settlement times expanded to match
  # the dimensions of fixed leg cashflows
  settle_times_fixed: types.RealTensor
  # `Tensor` containing the start times of time-periods corresponding to
  # floating cashflows for all instruments "flattened" (or concatenated)
  float_leg_times_start: types.RealTensor
  # `Tensor` containing the end times of time-periods corresponding to
  # floating cashflows for all instruments "flattened" (or concatenated)
  float_leg_times_end: types.RealTensor
  # `Tensor` containing daycount associated with floating leg cashflows
  float_leg_daycount: types.RealTensor
  # `Tensor` containing the instrument settlement times expanded to match
  # the dimensions of floating cashflows
  settle_times_float: types.RealTensor
  # `Tensor` containing the instrument index of each "flattened" `Tensor`
  # for floating cashflows
  calc_groups_float: types.IntTensor
  # `Tensor` containing the instrument index of each "flattened" `Tensor`
  # for fixed cashflows
  calc_groups_fixed: types.IntTensor
  # `Tensor` containing the start times of the final accrual periods of
  # each insrument. These will be used for bootstrap iterations.
  last_float_leg_start_time: types.RealTensor
  # `Tensor` containing the end times of the final accrual periods of
  # each insrument. These will be used for bootstrap iterations.
  last_float_leg_end_time: types.RealTensor
  # `Tensor` containing the daycount of the final accrual periods of
  # each insrument. These will be used for bootstrap iterations.
  last_float_leg_daycount: types.RealTensor
  # `Tensor` containing the discounting times of the final fixed cashflow
  # of each insrument. These will be used for bootstrap iterations.
  last_fixed_leg_calc_time: types.RealTensor
  # `Tensor` containing the daycount of the final fixed cashflow
  # of each insrument. These will be used for bootstrap iterations.
  last_fixed_leg_daycount: types.RealTensor
  # `Tensor` containing the the final fixed cashflow of each instrument.
  # These will be used for bootstrap iterations.
  last_fixed_leg_cashflows: types.RealTensor


def _create_curve_building_tensors(float_leg_start_times,
                                   float_leg_end_times,
                                   float_leg_daycount_fractions,
                                   fixed_leg_end_times,
                                   fixed_leg_cashflows,
                                   fixed_leg_daycount_fractions):
  """Helper function to create tensors needed for curve construction."""
  calc_float_leg_daycount = []
  float_leg_calc_times_start = []
  float_leg_calc_times_end = []
  calc_fixed_leg_cashflows = []
  calc_fixed_leg_daycount = []
  fixed_leg_calc_times = []
  calc_groups_float = []
  calc_groups_fixed = []
  last_float_leg_start_time = []
  last_float_leg_end_time = []
  last_float_leg_daycount = []
  last_fixed_leg_end_time = []
  last_fixed_leg_daycount = []
  last_fixed_leg_cashflows = []
  expiry_times = []
  num_instruments = len(float_leg_start_times)
  for i in range(num_instruments):
    calc_float_leg_daycount.append(float_leg_daycount_fractions[i][:-1])
    float_leg_calc_times_start.append(float_leg_start_times[i][:-1])
    float_leg_calc_times_end.append(float_leg_end_times[i][:-1])
    calc_fixed_leg_cashflows.append(fixed_leg_cashflows[i][:-1])
    calc_fixed_leg_daycount.append(fixed_leg_daycount_fractions[i][:-1])
    fixed_leg_calc_times.append(fixed_leg_end_times[i][:-1])
    last_float_leg_start_time.append(float_leg_start_times[i][-1])
    last_float_leg_end_time.append(float_leg_end_times[i][-1])
    last_float_leg_daycount.append(float_leg_daycount_fractions[i][-1])
    last_fixed_leg_end_time.append(fixed_leg_end_times[i][-1])
    last_fixed_leg_daycount.append(fixed_leg_daycount_fractions[i][-1])
    last_fixed_leg_cashflows.append(fixed_leg_cashflows[i][-1])
    expiry_times.append(
        tf.math.maximum(float_leg_end_times[i][-1], fixed_leg_end_times[i][-1]))

    calc_groups_float.append(
        tf.fill(tf.shape(float_leg_start_times[i][:-1]), i))
    calc_groups_fixed.append(tf.fill(tf.shape(fixed_leg_end_times[i][:-1]), i))

  output = CurveFittingVars(
      float_leg_daycount=tf.concat(calc_float_leg_daycount, axis=0),
      float_leg_times_start=tf.concat(float_leg_calc_times_start, axis=0),
      float_leg_times_end=tf.concat(float_leg_calc_times_end, axis=0),
      fixed_leg_cashflows=tf.concat(calc_fixed_leg_cashflows, axis=0),
      fixed_leg_daycount=tf.concat(calc_fixed_leg_daycount, axis=0),
      fixed_leg_calc_times=tf.concat(fixed_leg_calc_times, axis=0),
      settle_times_fixed=None,
      settle_times_float=None,
      expiry_times=tf.stack(expiry_times, axis=0),
      calc_groups_float=tf.concat(calc_groups_float, axis=0),
      calc_groups_fixed=tf.concat(calc_groups_fixed, axis=0),
      last_float_leg_start_time=tf.stack(last_float_leg_start_time, axis=0),
      last_float_leg_end_time=tf.stack(last_float_leg_end_time, axis=0),
      last_float_leg_daycount=tf.stack(last_float_leg_daycount, axis=0),
      last_fixed_leg_calc_time=tf.stack(last_fixed_leg_end_time, axis=0),
      last_fixed_leg_daycount=tf.stack(last_fixed_leg_daycount, axis=0),
      last_fixed_leg_cashflows=tf.stack(last_fixed_leg_cashflows, axis=0)
      )

  return output
