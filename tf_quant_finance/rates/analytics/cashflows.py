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
"""Collection of functions to compute properties of cashflows."""

import tensorflow.compat.v2 as tf


def present_value(cashflows,
                  discount_factors,
                  dtype=None,
                  name=None):
  """Computes present value of a stream of cashflows given discount factors.


  ```python

    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    # Note that the first four entries in the cashflows are the cashflows of
    # the first bond (group=0) and the next six are the cashflows of the second
    # bond (group=1).
    cashflows = [[20, 20, 20, 1020, 0, 0],
                 [30, 30, 30, 30, 30, 1030]]

    # Corresponding discount factors for the cashflows
    discount_factors = [[0.96, 0.93, 0.9, 0.87, 1.0, 1.0],
                        [0.97, 0.95, 0.93, 0.9, 0.88, 0.86]]

    present_values = present_value(
        cashflows, discount_factors, dtype=np.float64)
    # Expected: [943.2, 1024.7]
  ```

  Args:
    cashflows: A real `Tensor` of shape `batch_shape + [n]`. The set of
      cashflows of underlyings. `n` is the number of cashflows per bond
      and `batch_shape` is the number of bonds. Bonds with different number
      of cashflows should be padded to a common number `n`.
    discount_factors: A `Tensor` of the same `dtype` as `cashflows` and of
      compatible shape. The set of discount factors corresponding to the
      cashflows.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `cashflows`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'present_value'.

  Returns:
    Real `Tensor` of shape `batch_shape`. The present values of the cashflows.
  """
  name = name or 'present_value'
  with tf.name_scope(name):
    cashflows = tf.convert_to_tensor(cashflows, dtype=dtype, name='cashflows')
    dtype = dtype or cashflows.dtype
    discount_factors = tf.convert_to_tensor(
        discount_factors, dtype=dtype, name='discount_factors')
    discounted = cashflows * discount_factors
    return tf.math.reduce_sum(discounted, axis=-1)


def pv_from_yields(cashflows,
                   times,
                   yields,
                   groups=None,
                   dtype=None,
                   name=None):
  """Computes present value of cashflows given yields.

  For a more complete description of the terminology as well as the mathematics
  of pricing bonds, see Ref [1]. In particular, note that `yields` here refers
  to the yield of the bond as defined in Section 4.4 of Ref [1]. This is
  sometimes also referred to as the internal rate of return of a bond.

  #### Example

  The following example demonstrates the present value computation for two
  bonds. Both bonds have 1000 face value with semi-annual coupons. The first
  bond has 4% coupon rate and 2 year expiry. The second has 6% coupon rate and
  3 year expiry. The yields to maturity (ytm) are 7% and 5% respectively.

  ```python
    dtype = np.float64

    # The first element is the ytm of the first bond and the second is the
    # yield of the second bond.
    yields_to_maturity = np.array([0.07, 0.05], dtype=dtype)

    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    # Note that the first four entries in the cashflows are the cashflows of
    # the first bond (group=0) and the next six are the cashflows of the second
    # bond (group=1).
    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                         dtype=dtype)

    # The times of the cashflows.
    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)

    # Group entries take values between 0 and 1 (inclusive) as there are two
    # bonds. One needs to assign each of the cashflow entries to one group or
    # the other.
    groups = np.array([0] * 4 + [1] * 6)

    # Produces [942.712, 1025.778] as the values of the two bonds.
    present_values = pv_from_yields(
        cashflows, times, yields_to_maturity, groups=groups, dtype=dtype)
  ```

  #### References:

  [1]: John C. Hull. Options, Futures and Other Derivatives. Ninth Edition.
    June 2006.

  Args:
    cashflows: Real rank 1 `Tensor` of size `n`. The set of cashflows underlying
      the bonds.
    times: Real positive rank 1 `Tensor` of size `n`. The set of times at which
      the corresponding cashflows occur quoted in years.
    yields: Real rank 1 `Tensor` of size `1` if `groups` is None or of size `k`
      if the maximum value in the `groups` is of `k-1`. The continuously
      compounded yields to maturity/internal rate of returns corresponding to
      each of the cashflow groups. The `i`th component is the yield to apply to
      all the cashflows with group label `i` if `groups` is not None. If
      `groups` is None, then this is a `Tensor` of size `[1]` and the only
      component is the yield that applies to all the cashflows.
    groups: Optional int `Tensor` of size `n` containing values between 0 and
      `k-1` where `k` is the number of related cashflows.
      Default value: None. This implies that all the cashflows are treated as a
        single group.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `cashflows`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'pv_from_yields'.

  Returns:
    Real rank 1 `Tensor` of size `k` if groups is not `None` else of size `[1]`.
      The present value of the cashflows. The `i`th component is the present
      value of the cashflows in group `i` or to the entirety of the cashflows
      if `groups` is None.
  """
  with tf.compat.v1.name_scope(
      name,
      default_name='pv_from_yields',
      values=[cashflows, times, yields, groups]):
    cashflows = tf.convert_to_tensor(cashflows, dtype=dtype, name='cashflows')
    times = tf.convert_to_tensor(times, dtype=dtype, name='times')
    yields = tf.convert_to_tensor(yields, dtype=dtype, name='yields')
    cashflow_yields = yields
    if groups is not None:
      groups = tf.convert_to_tensor(groups, name='groups')
      cashflow_yields = tf.gather(yields, groups)
    discounted = cashflows * tf.math.exp(-times * cashflow_yields)
    if groups is not None:
      return tf.math.segment_sum(discounted, groups)
    return tf.math.reduce_sum(discounted, keepdims=True)


def yields_from_pv(cashflows,
                   times,
                   present_values,
                   groups=None,
                   tolerance=1e-8,
                   max_iterations=10,
                   dtype=None,
                   name=None):
  """Computes yields to maturity from present values of cashflows.

  For a complete description of the terminology as well as the mathematics
  of computing bond yields, see Ref [1]. Note that `yields` here refers
  to the yield of the bond as defined in Section 4.4 of Ref [1]. This is
  sometimes also referred to as the internal rate of return of a bond.

  #### Example

  The following example demonstrates the yield computation for two
  bonds. Both bonds have 1000 face value with semi-annual coupons. The first
  bond has 4% coupon rate and 2 year expiry. The second has 6% coupon rate and
  3 year expiry. The true yields to maturity (ytm) are 7% and 5% respectively.

  ```python
    dtype = np.float64

    # The first element is the present value (PV) of the first bond and the
    # second is the PV of the second bond.
    present_values = np.array([942.71187528177757, 1025.7777300221542],
                              dtype=dtype)

    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    # Note that the first four entries in the cashflows are the cashflows of
    # the first bond (group=0) and the next six are the cashflows of the second
    # bond (group=1).
    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                         dtype=dtype)

    # The times of the cashflows.
    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)

    # Group entries take values between 0 and 1 (inclusive) as there are two
    # bonds. One needs to assign each of the cashflow entries to one group or
    # the other.
    groups = np.array([0] * 4 + [1] * 6)

    # Expected yields = [0.07, 0.05]
    yields = yields_from_pv(
        cashflows, times, present_values, groups=groups, dtype=dtype)
  ```

  #### References:

  [1]: John C. Hull. Options, Futures and Other Derivatives. Ninth Edition.
    June 2006.

  Args:
    cashflows: Real rank 1 `Tensor` of size `n`. The set of cashflows underlying
      the bonds.
    times: Real positive rank 1 `Tensor` of size `n`. The set of times at which
      the corresponding cashflows occur quoted in years.
    present_values: Real rank 1 `Tensor` of size `k` where `k-1` is the maximum
      value in the `groups` arg if supplied. If `groups` is not supplied, then
      this is a `Tensor` of size `1`. The present values corresponding to each
      of the cashflow groups. The `i`th component is the present value of all
      the cashflows with group label `i` (or the present value of all the
      cashflows if `groups=None`).
    groups: Optional int `Tensor` of size `n` containing values between 0 and
      `k-1` where `k` is the number of related cashflows.
      Default value: None. This implies that all the cashflows are treated as a
        single group.
    tolerance: Positive real scalar `Tensor`. The tolerance for the estimated
      yields. The yields are computed using a Newton root finder. The iterations
      stop when the inferred yields change by less than this tolerance or the
      maximum iterations are exhausted (whichever is earlier).
      Default value: 1e-8.
    max_iterations: Positive scalar int `Tensor`. The maximum number of
      iterations to use to compute the yields. The iterations stop when the max
      iterations is exhausted or the tolerance is reached (whichever is
      earlier). Supply `None` to remove the limit on the number of iterations.
      Default value: 10.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `cashflows`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'yields_from_pv'.

  Returns:
    Real rank 1 `Tensor` of size `k`. The yield to maturity of the cashflows.
      The `i`th component is the yield to maturity of the cashflows in group
      `i`.
  """
  with tf.compat.v1.name_scope(
      name,
      default_name='yields_from_pv',
      values=[
          cashflows, times, present_values, groups, tolerance, max_iterations
      ]):
    cashflows = tf.convert_to_tensor(cashflows, dtype=dtype, name='cashflows')
    times = tf.convert_to_tensor(times, dtype=dtype, name='times')
    present_values = tf.convert_to_tensor(
        present_values, dtype=dtype, name='present_values')
    if groups is None:
      groups = tf.zeros_like(cashflows, dtype=tf.int32, name='groups')
    else:
      groups = tf.convert_to_tensor(groups, name='groups')

    def pv_and_duration(yields):
      cashflow_yields = tf.gather(yields, groups)
      discounted = cashflows * tf.math.exp(-times * cashflow_yields)
      durations = tf.math.segment_sum(discounted * times, groups)
      pvs = tf.math.segment_sum(discounted, groups)
      return pvs, durations

    yields0 = tf.zeros_like(present_values)

    def _cond(should_stop, yields):
      del yields
      return tf.math.logical_not(should_stop)

    def _body(should_stop, yields):
      del should_stop
      pvs, durations = pv_and_duration(yields)
      delta_yields = (pvs - present_values) / durations
      next_should_stop = (tf.math.reduce_max(tf.abs(delta_yields)) <= tolerance)
      return (next_should_stop, yields + delta_yields)

    loop_vars = (tf.convert_to_tensor(False), yields0)
    _, estimated_yields = tf.while_loop(
        _cond,
        _body,
        loop_vars,
        shape_invariants=(tf.TensorShape([]), tf.TensorShape([None])),
        maximum_iterations=max_iterations,
        parallel_iterations=1)
    return estimated_yields

__all__ = ['present_value', 'pv_from_yields', 'yields_from_pv']
