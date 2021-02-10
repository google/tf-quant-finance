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
"""Methods to construct a discount curve from bonds.

Building discount curves is a core problem in mathematical finance. Discount
curves are built using the available market data in liquidly traded rates
products. These include bonds, swaps, forward rate agreements (FRAs) or
eurodollar futures contracts. This module contains methods to build rate curve
from (potentially coupon bearing) bonds data.

A discount curve is a function of time which gives the interest rate that
applies to a unit of currency deposited today for a period of  time `t`.
The traded price of bonds implicitly contains the market view on the discount
rates. The purpose of discount curve construction is to extract this
information.

The algorithm implemented here is based on the Monotone Convex Interpolation
method described by Hagan and West in Ref [1, 2].

#### References:

[1]: Patrick Hagan & Graeme West. Interpolation Methods for Curve Construction.
  Applied Mathematical Finance. Vol 13, No. 2, pp 89-129. June 2006.
  https://www.researchgate.net/publication/24071726_Interpolation_Methods_for_Curve_Construction
[2]: Patrick Hagan & Graeme West. Methods for Constructing a Yield Curve.
  Wilmott Magazine, pp. 70-81. May 2008.
"""

import collections
import tensorflow.compat.v2 as tf

from tf_quant_finance.rates.analytics import cashflows
from tf_quant_finance.rates.hagan_west import monotone_convex

CurveBuilderResult = collections.namedtuple(
    'CurveBuilderResult',
    [
        # Rank 1 real `Tensor`. Times for the computed discount rates.
        'times',
        # Rank 1 `Tensor` of the same dtype as `times`.
        # The inferred discount rates.
        'discount_rates',
        # Rank 1 `Tensor` of the same dtype as `times`.
        # The inferred discount factors.
        'discount_factors',
        # Rank 1 `Tensor` of the same dtype as `times`. The
        # initial guess for the discount rates.
        'initial_discount_rates',
        # Scalar boolean `Tensor`. Whether the procedure converged.
        'converged',
        # Scalar boolean `Tensor`. Whether the procedure failed.
        'failed',
        # Scalar int32 `Tensor`. Number of iterations performed.
        'iterations'
    ])


def bond_curve(bond_cashflows,
               bond_cashflow_times,
               present_values,
               present_values_settlement_times=None,
               initial_discount_rates=None,
               discount_tolerance=1e-8,
               maximum_iterations=50,
               validate_args=False,
               dtype=None,
               name=None):
  """Constructs the bond discount rate curve using the Hagan-West algorithm.


  A discount curve is a function of time which gives the interest rate that
  applies to a unit of currency deposited today for a period of  time `t`.
  The traded price of bonds implicitly contains the market view on the discount
  rates. The purpose of discount curve construction is to extract this
  information.

  Suppose we have a set of `N` bonds `B_i` with increasing expiries whose market
  prices are known.
  Suppose also that the `i`th bond issues cashflows at times `T_{ij}` where
  `1 <= j <= n_i` and `n_i` is the number of cashflows (including expiry)
  for the `i`th bond.
  Denote by `T_i` the time of final payment for the `i`th bond
  (hence `T_i = T_{i,n_i}`). This function estimates a set of rates `r(T_i)`
  such that when these rates are interpolated to all other cashflow times using
  the Monotone Convex interpolation scheme (Ref [1, 2]), the computed value of
  the bonds matches the market value of the bonds (within some tolerance).

  The algorithm implemented here is based on the Monotone Convex Interpolation
  method described by Hagan and West in Ref [1, 2].


  ### Limitations

  The fitting algorithm suggested in Hagan and West has a few limitations that
  are worth keeping in mind.

    1. Non-convexity: The implicit loss function that is minimized by the
      procedure is non-convex. Practically this means that for a given level of
      tolerance, it is possible to find distinct values for the discount rates
      all of which price the given cashflows to within tolerance. Depending
      on the initial values chosen, the procedure of Hagan-West can converge to
      different minima.
    2. Stability: The procedure iterates by computing the rate to expiry of
      a bond given the approximate rates for the coupon dates. If the initial
      guess is widely off or even if it isn't but the rates are artificially
      large, it can happen that the discount factor estimated at an iteration
      step (see Eq. 14 in Ref. [2]) is negative. Hence no real discount rate
      can be found to continue the iterations. Additionally, it can be shown
      that the procedure diverges if the final cashflow is not larger than
      all the intermediate cashflows. While this situation does not arise in
      the case of bond cashflows, it is an important consideration from a
      mathematical perspective. For the details of the stability and
      convergence of the scheme see the associated technical note.
      TODO(b/139052353): Write the technical note and add a reference here.

  #### Example:

  The following example demonstrates the usage by building the implied curve
  from four coupon bearing bonds.

  ```python

  dtype=np.float64

  # These need to be sorted by expiry time.
  cashflow_times = [
      np.array([0.25, 0.5, 0.75, 1.0], dtype=dtype),
      np.array([0.5, 1.0, 1.5, 2.0], dtype=dtype),
      np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
      np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
  ]

  cashflows = [
      # 1 year bond with 5% three monthly coupon.
      np.array([12.5, 12.5, 12.5, 1012.5], dtype=dtype),
      # 2 year bond with 6% semi-annual coupon.
      np.array([30, 30, 30, 1030], dtype=dtype),
      # 3 year bond with 8% semi-annual coupon.
      np.array([40, 40, 40, 40, 40, 1040], dtype=dtype),
      # 4 year bond with 3% semi-annual coupon.
      np.array([15, 15, 15, 15, 15, 15, 15, 1015], dtype=dtype)
  ]

  # The present values of the above cashflows.
  pvs = np.array([
      999.68155223943393, 1022.322872470043, 1093.9894418810143,
      934.20885689015677
  ], dtype=dtype)

  results = bond_curve(cashflows, cashflow_times, pvs)

  # The curve times are the expiries of the supplied cashflows.
  np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 4.0])

  expected_discount_rates = np.array([5.0, 4.75, 4.53333333, 4.775],
                                     dtype=dtype) / 100

  np.testing.assert_allclose(results.discount_rates, expected_discount_rates,
                             atol=1e-6)
  ```

  #### References:

  [1]: Patrick Hagan & Graeme West. Interpolation Methods for Curve
    Construction. Applied Mathematical Finance. Vol 13, No. 2, pp 89-129.
    June 2006.
  https://www.researchgate.net/publication/24071726_Interpolation_Methods_for_Curve_Construction
  [2]: Patrick Hagan & Graeme West. Methods for Constructing a Yield Curve.
    Wilmott Magazine, pp. 70-81. May 2008.
  https://www.researchgate.net/profile/Patrick_Hagan3/publication/228463045_Methods_for_constructing_a_yield_curve/links/54db8cda0cf23fe133ad4d01.pdf

  Args:
    bond_cashflows: List of `Tensor`s. Each `Tensor` must be of rank 1 and of
      the same real dtype. They may be of different sizes. Each `Tensor`
      represents the bond cashflows defining a particular bond. The elements of
      the list are the bonds to be used to build the curve.
    bond_cashflow_times: List of `Tensor`s. The list must be of the same length
      as the `bond_cashflows` and each `Tensor` in the list must be of the same
      length as the `Tensor` at the same index in the `bond_cashflows` list.
      Each `Tensor` must be of rank 1 and of the same dtype as the `Tensor`s in
      `bond_cashflows` and contain strictly positive and increasing values. The
      times of the bond cashflows for the bonds must in an ascending order.
    present_values: List containing scalar `Tensor`s of the same dtype as
      elements of `bond_cashflows`. The length of the list must be the same as
      the length of `bond_cashflows`. The market price (i.e the all-in or dirty
      price) of the bond cashflows supplied in the `bond_cashflows`.
    present_values_settlement_times: List containing scalar `Tensor`s of the
      same dtype as elements of `bond_cashflows`. The length of the list must be
      the same as the length of `bond_cashflows`. The settlement times for the
      present values is the time from now when the bond is traded to the time
      that the purchase price is actually delivered. If not supplied, then it is
      assumed that the settlement times are zero for every bond.
      Default value: `None` which is equivalent to zero settlement times.
    initial_discount_rates: Optional `Tensor` of the same dtype and shape as
      `present_values`. The starting guess for the discount rates used to
      initialize the iterative procedure.
      Default value: `None`. If not supplied, the yields to maturity for the
        bonds is used as the initial value.
    discount_tolerance: Optional positive scalar `Tensor` of same dtype as
      elements of `bond_cashflows`. The absolute tolerance for terminating the
      iterations used to fit the rate curve. The iterations are stopped when the
      estimated discounts at the expiry times of the bond_cashflows change by a
      amount smaller than `discount_tolerance` in an iteration.
      Default value: 1e-8.
    maximum_iterations: Optional positive integer `Tensor`. The maximum number
      of iterations permitted when fitting the curve.
      Default value: 50.
    validate_args: Optional boolean flag to enable validation of the input
      arguments. The checks performed are: (1) There are no cashflows which
      expire before or at the corresponding settlement time (or at time 0 if
      settlement time is not provided). (2) Cashflow times for each bond form
      strictly increasing sequence. (3) Final cashflow for each bond is larger
      than any other cashflow for that bond.
      Default value: False.
    dtype: `tf.Dtype`. If supplied the dtype for the (elements of)
      `bond_cashflows`, `bond_cashflow_times` and `present_values`.
      Default value: None which maps to the default dtype inferred by TensorFlow
        (float32).
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'hagan_west'.

  Returns:
    curve_builder_result: An instance of `CurveBuilderResult` containing the
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

  Raises:
    ValueError: If the `cashflows` and `cashflow_times` are not all of the same
      length greater than or equal to two. Also raised if the
      `present_values_settlement_times` is not None and not of the same length
      as the `cashflows`.
    tf.errors.InvalidArgumentError: In case argument validation is requested and
      conditions explained in the corresponding section of Args comments are not
      met.
  """
  with tf.compat.v1.name_scope(
      name,
      default_name='bond_curve',
      values=[
          bond_cashflows, bond_cashflow_times, present_values,
          present_values_settlement_times
      ]):
    if present_values_settlement_times is None:
      pv_settle_times = [tf.zeros_like(pv) for pv in present_values]
    else:
      pv_settle_times = present_values_settlement_times

    args = _convert_to_tensors(dtype, bond_cashflows, bond_cashflow_times,
                               present_values, pv_settle_times)

    bond_cashflows, bond_cashflow_times, present_values, pv_settle_times = args

    # Always perform static validation.
    _perform_static_validation(bond_cashflows, bond_cashflow_times,
                               present_values, pv_settle_times)

    control_inputs = []
    if validate_args:
      control_inputs = _validate_args_control_deps(bond_cashflows,
                                                   bond_cashflow_times,
                                                   pv_settle_times)

    if initial_discount_rates is not None:
      initial_rates = tf.convert_to_tensor(
          initial_discount_rates, dtype=dtype, name='initial_rates')
    else:
      # Note that we ignore the pv settlement times for this computation.
      # This should be OK so long as the settlement times are not too large
      # compared to the bond expiry times. Ignoring the settlement times amounts
      # to overestimating the starting point if the true discount curve is
      # positive.
      initial_rates = _initial_discount_rates(
          bond_cashflows,
          bond_cashflow_times,
          present_values,
          name='initial_rates')

    with tf.compat.v1.control_dependencies(control_inputs):
      return _build_discount_curve(bond_cashflows, bond_cashflow_times,
                                   present_values, pv_settle_times,
                                   initial_rates, discount_tolerance,
                                   maximum_iterations)


def _build_discount_curve(bond_cashflows, bond_cashflow_times, present_values,
                          pv_settle_times, initial_discount_rates,
                          discount_tolerance, maximum_iterations):
  """Estimates the discount curve.

  The procedure is recursive and as follows:
  1. Assume some initial set of discount rates/discount factors.
    Set this as the current yield curve.
  2. From the current yield curve, interpolate to get the discount rates
    for each time at which bond_cashflows occur.
  3. Using these discounts and the known bond prices, compute the discount
    rate to expiry of each bond by inverting the bond pricing formula as
    follows. We know that the bond price satisfies (`P` is the present value,
    `r_i` is the discount rate to time `t_i`, `c_i` is the cashflow occurring at
    time `t_i`.):

    ```None
      P e^{-r_0 t_0} = c_1 e^{-r_1 t_1} + ... + c_n e^{-r_n t_n}        (A)

    ```
    Assuming we have estimated r_0, r_1, r_2, ..., r_{n-1}, we can invert the
    above equation to calculate r_n. We write this in a suggestive form
    suitable for the implementation below.

    ```None
      -c_n z_n = -P z_0 + c_1 z_1 + c_2 z_2 + ... + c_{n-1} z_{n-1}     (B)

    ```
    where

    ```None
      z_i = e^{-r_i t_i}      (C)

    ```
    The RHS of Eq. (B) looks like the PV of cashflows
    `[-P, c_1, c_2, ... c_{n-1}]` paid out at times `[t_0, t_1, ..., t_{n-1}]`.

    Concatenate these "synthetic" cashflow times for each bond:

    `Ts = [t1_0, t1_1, ... t1_{n1-1}] + [t2_0, t2_1, ... t2_{n2-1}] ...`

    Also concatenate the synthetic bond cashflows as:

    `Cs = [-P1, c1_1, ..., c1_{n1-1}] + [-P2, c2_1, ..., c2_{n2-1}] ...`

    Then compute `Rs = InterpolateRates[Ts], Zs = exp(-Rs * Ts)`

    Let `Zns = [z_n1, z_n2, ... ], Cns = [c1_n, c2_n, ...]` be the discount
    factors to expiry and the final cashflow of each bond.
    We can derive `Zns = - SegmentSum(Cs * Zs) / Cns`.

    From that, we get Rns = -log(Zns) / Tns.
    Using this as the next guess for the discount rates and we repeat the
    procedure from Step (1) until convergence.

  Args:
    bond_cashflows: List of `Tensor`s. Each `Tensor` must be of rank 1 and of
      the same real dtype. They may be of different sizes. Each `Tensor`
      represents the bond cashflows defining a particular bond. The elements of
      the list are the bonds to be used to build the curve.
    bond_cashflow_times: List of `Tensor`s. The list must be of the same length
      as the `bond_cashflows` and each `Tensor` in the list must be of the same
      length as the `Tensor` at the same index in the `bond_cashflows` list.
      Each `Tensor` must be of rank 1 and of the same dtype as the `Tensor`s in
      `bond_cashflows` and contain strictly positive and increasing values. The
      times of the bond cashflows for the bonds must in an ascending order.
    present_values: List containing scalar `Tensor`s of the same dtype as
      elements of `bond_cashflows`. The length of the list must be the same as
      the length of `bond_cashflows`. The market price (i.e the all-in or dirty
      price) of the bond cashflows supplied in the `bond_cashflows`.
    pv_settle_times:   List containing scalar `Tensor`s of the same dtype as
      elements of `bond_cashflows`. The length of the list must be the same as
      the length of `bond_cashflows`. The settlement times for the present
      values is the time from now when the bond is traded to the time that the
      purchase price is actually delivered.
    initial_discount_rates: Rank 1 `Tensor` of same shape and dtype as
      `pv_settle_times`. The initial guess for the discount rates to bond expiry
      times.
    discount_tolerance: Positive scalar `Tensor` of same dtype as
      `initial_discount_factors`. The absolute tolerance for terminating the
      iterations used to fit the rate curve. The iterations are stopped when the
      estimated discounts at the expiry times of the bond cashflows change by a
      amount smaller than `discount_tolerance` in an iteration.
    maximum_iterations: Positive scalar `tf.int32` `Tensor`. The maximum number
      of iterations permitted.

  Returns:
    curve_builder_result: An instance of `CurveBuilderResult` containing the
      following attributes.
      times: Rank 1 real `Tensor`. Times for the computed discount rates.
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
      iterations: Scalar `tf.int32` `Tensor`. Number of iterations performed.
  """
  calc_bond_cashflows = []  # Cs
  calc_times = []  # Ts
  expiry_times = []  # Tns
  expiry_bond_cashflows = []  # Cns
  calc_groups = []
  num_bonds = len(bond_cashflows)
  for i in range(num_bonds):
    calc_bond_cashflows.extend([[-present_values[i]], bond_cashflows[i][:-1]])
    calc_times.extend([[pv_settle_times[i]], bond_cashflow_times[i][:-1]])
    expiry_times.append(bond_cashflow_times[i][-1])
    expiry_bond_cashflows.append(bond_cashflows[i][-1])
    calc_groups.append(tf.fill(tf.shape(bond_cashflows[i]), i))

  calc_bond_cashflows = tf.concat(calc_bond_cashflows, axis=0)
  calc_times = tf.concat(calc_times, axis=0)
  expiry_times = tf.stack(expiry_times, axis=0)
  expiry_bond_cashflows = tf.stack(expiry_bond_cashflows, axis=0)
  calc_groups = tf.concat(calc_groups, axis=0)

  def one_step(converged, failed, iteration, expiry_discounts):
    """One step of the iteration."""
    expiry_rates = -tf.math.log(expiry_discounts) / expiry_times
    failed = tf.math.reduce_any(
        tf.math.is_nan(expiry_rates) | tf.math.is_nan(expiry_discounts))
    calc_rates = monotone_convex.interpolate_yields(
        calc_times, expiry_times, yields=expiry_rates)
    calc_discounts = tf.math.exp(-calc_rates * calc_times)
    next_expiry_discounts = -tf.math.segment_sum(
        calc_bond_cashflows * calc_discounts,
        calc_groups) / expiry_bond_cashflows
    discount_diff = tf.math.abs(next_expiry_discounts - expiry_discounts)
    converged = (~tf.math.reduce_any(tf.math.is_nan(discount_diff)) &
                 (tf.math.reduce_max(discount_diff) < discount_tolerance))
    return converged, failed, iteration + 1, next_expiry_discounts

  def cond(converged, failed, iteration, expiry_discounts):
    del expiry_discounts, iteration
    # Note we do not need to check iteration count here because that
    # termination mode is imposed by the maximum_iterations parameter in the
    # while loop.
    return ~tf.math.logical_or(converged, failed)

  initial_discount_factors = tf.math.exp(-initial_discount_rates * expiry_times)
  initial_vals = (False, False, 0, initial_discount_factors)
  loop_result = tf.while_loop(
      cond, one_step, initial_vals, maximum_iterations=maximum_iterations)
  discount_factors = loop_result[-1]
  discount_rates = -tf.math.log(discount_factors) / expiry_times
  results = CurveBuilderResult(
      times=expiry_times,
      discount_rates=discount_rates,
      discount_factors=discount_factors,
      initial_discount_rates=initial_discount_rates,
      converged=loop_result[0],
      failed=loop_result[1],
      iterations=loop_result[2])
  return results


def _initial_discount_rates(bond_cashflows,
                            bond_cashflow_times,
                            present_values,
                            name='initial_discount_rates'):
  """Constructs a guess for the initial rates as the yields to maturity."""
  n = len(bond_cashflows)
  groups = []
  for i in range(n):
    groups.append(tf.fill(tf.shape(bond_cashflows[i]), i))
  bond_cashflows = tf.concat(bond_cashflows, axis=0)
  bond_cashflow_times = tf.concat(bond_cashflow_times, axis=0)
  groups = tf.concat(groups, axis=0)
  return cashflows.yields_from_pv(
      bond_cashflows,
      bond_cashflow_times,
      present_values,
      groups=groups,
      name=name)


def _perform_static_validation(bond_cashflows, bond_cashflow_times,
                               present_values, pv_settle_times):
  """Performs static validation on the arguments."""
  if len(bond_cashflows) != len(bond_cashflow_times):
    raise ValueError(
        'Cashflow times and bond_cashflows must be of the same length.'
        'bond_cashflows are of size'
        ' {} and times of size {}'.format(
            len(bond_cashflows), len(bond_cashflow_times)))

  if len(bond_cashflows) != len(present_values):
    raise ValueError(
        'Present values and bond_cashflows must be of the same length.'
        'bond_cashflows are of size'
        ' {} and PVs of size {}'.format(
            len(bond_cashflows), len(present_values)))

  if len(present_values) != len(pv_settle_times):
    raise ValueError(
        'Present value settlement times and present values must be of'
        'the same length. Settlement times are of size'
        ' {} and PVs of size {}'.format(
            len(pv_settle_times), len(present_values)))

  if len(bond_cashflows) < 2:
    raise ValueError(
        'At least two bonds must be supplied to calibrate the curve.'
        'Found {}.'.format(len(bond_cashflows)))


def _validate_args_control_deps(bond_cashflows, bond_cashflow_times,
                                pv_settle_times):
  """Returns assertions for the validity of the arguments."""
  cashflows_are_strictly_increasing = []
  cashflow_after_settlement = []
  final_cashflow_is_the_largest = []
  for bond_index, bond_cashflow in enumerate(bond_cashflows):
    times = bond_cashflow_times[bond_index]
    time_difference = times[1:] - times[:-1]
    cashflows_are_strictly_increasing.append(
        tf.debugging.assert_positive(time_difference))
    cashflow_after_settlement.append(
        tf.debugging.assert_greater(times[0], pv_settle_times[bond_index]))
    final_cashflow_is_the_largest.append(
        tf.debugging.assert_greater(
            tf.fill(tf.shape(bond_cashflow[:-1]),
                    bond_cashflow[-1]), bond_cashflow[:-1]))
  return (cashflow_after_settlement + cashflows_are_strictly_increasing +
          final_cashflow_is_the_largest)


def _convert_to_tensors(dtype, bond_cashflows, bond_cashflow_times,
                        present_values, pv_settle_times):
  """Converts each element of the supplied lists to a tensor."""

  bond_cashflows = [
      tf.convert_to_tensor(
          cashflow, dtype=dtype, name='cashflows_bond_{}'.format(i))
      for i, cashflow in enumerate(bond_cashflows)
  ]
  bond_cashflow_times = [
      tf.convert_to_tensor(
          cashflow_times, dtype=dtype, name='cashflow_times_bond_{}'.format(i))
      for i, cashflow_times in enumerate(bond_cashflow_times)
  ]
  present_values = [
      tf.convert_to_tensor(pv, dtype=dtype, name='pv_bond_{}'.format(i))
      for i, pv in enumerate(present_values)
  ]
  pv_settle_times = [
      tf.convert_to_tensor(
          pv_time, dtype=dtype, name='pv_settle_time_bond_{}'.format(i))
      for i, pv_time in enumerate(pv_settle_times)
  ]

  return bond_cashflows, bond_cashflow_times, present_values, pv_settle_times
