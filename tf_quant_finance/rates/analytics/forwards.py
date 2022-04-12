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
"""Collection of functions to compute properties of forwards."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math import segment_ops


def forward_rates(df_start_dates,
                  df_end_dates,
                  daycount_fractions,
                  dtype=None,
                  name=None):
  """Computes forward rates from daycount fractions and discount factors.

  #### Example
  ```python
  # Discount factors at start dates
  df_start_dates = [[0.95, 0.9, 0.75], [0.95, 0.99, 0.85]]
  # Discount factors at end dates
  df_end_dates = [[0.8, 0.6, 0.5], [0.8, 0.9, 0.5]]
  # Daycount fractions between the dates
  daycount_fractions = [[0.5, 1.0, 2], [0.6, 0.4, 4.0]]
  # Expected:
  #  [[0.375 , 0.5   , 0.25  ],
  #   [0.3125, 0.25  , 0.175 ]]
  forward_rates(df_start_dates, df_end_dates, daycount_fractions,
                dtype=tf.float64)
  ```

  Args:
    df_start_dates: A real `Tensor` representing discount factors at the start
      dates.
    df_end_dates: A real `Tensor` representing discount factors at the end
      dates.
    daycount_fractions: A real `Tensor` representing  year fractions for the
      coupon accrual.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `df_start_dates`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'forward_rates'.

  Returns:

  """
  name = name or 'forward_rates'
  with tf.name_scope(name):
    df_start_dates = tf.convert_to_tensor(
        df_start_dates, dtype, name='df_start_dates')
    dtype = dtype or df_start_dates.dtype
    df_end_dates = tf.convert_to_tensor(
        df_end_dates, dtype, name='df_end_dates')
    daycount_fractions = tf.convert_to_tensor(
        daycount_fractions, dtype, name='daycount_fractions')
    return tf.math.divide_no_nan(
        tf.math.divide_no_nan(df_start_dates, df_end_dates) - 1,
        daycount_fractions)


def forward_rates_from_yields(yields,
                              times,
                              groups=None,
                              dtype=None,
                              name=None):
  """Computes forward rates given a set of zero rates.

  Denote the price of a zero coupon bond maturing at time `t` by `Z(t)`. Then
  the zero rate to time `t` is defined as

  ```None
    r(t) = - ln(Z(t)) / t       (1)

  ```

  This is the (continuously compounded) interest rate that applies between time
  `0` and time `t` as seen at time `0`. The forward rate between times `t1` and
  `t2` is defined as the interest rate that applies to the period `[t1, t2]`
  as seen from today. It is related to the zero coupon bond prices by

  ```None
    exp(-f(t1, t2)(t2-t1)) = Z(t2) / Z(t1)                 (2)
    f(t1, t2) = - (ln Z(t2) - ln Z(t1)) / (t2 - t1)        (3)
    f(t1, t2) = (t2 * r(t2) - t1 * r(t1)) / (t2 - t1)      (4)
  ```

  Given a sequence of increasing times `[t1, t2, ... tn]` and the zero rates
  for those times, this function computes the forward rates that apply to the
  consecutive time intervals i.e. `[0, t1], [t1, t2], ... [t_{n-1}, tn]` using
  Eq. (4) above. Note that for the interval `[0, t1]` the forward rate is the
  same as the zero rate.

  Additionally, this function supports this computation for a batch of such
  rates. Batching is made slightly complicated by the fact that different
  zero curves may have different numbers of tenors (the parameter `n` above).
  Instead of a batch as an extra dimension, we support the concept of groups
  (also see documentation for `tf.segment_sum` which uses the same concept).

  #### Example

  The following example illustrates this method along with the concept of
  groups. Assuming there are two sets of zero rates (e.g. for different
  currencies) whose implied forward rates are needed. The first set has a total
  of three marked tenors at `[0.25, 0.5, 1.0]`. The second set
  has four marked tenors at `[0.25, 0.5, 1.0, 1.5]`.
  Suppose, the zero rates for the first set are:
  `[0.04, 0.041, 0.044]` and the second are `[0.022, 0.025, 0.028, 0.036]`.
  Then this data is batched together as follows:
  Groups: [0,    0    0,   1,    1,   1    1  ]
  First three times for group 0, next four for group 1.
  Times:  [0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 1.5]
  First three rates for group 0, next four for group 1.
  Rates:  [0.04, 0.041, 0.044, 0.022, 0.025, 0.028, 0.036]


  ```python
    dtype = np.float64
    groups = np.array([0, 0, 0, 1, 1, 1, 1])
    times = np.array([0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 1.5], dtype=dtype)
    rates = np.array([0.04, 0.041, 0.044, 0.022, 0.025, 0.028, 0.036],
                     dtype=dtype)
    forward_rates = forward_rates_from_yields(
        rates, times, groups=groups, dtype=dtype)
  ```

  #### References:

  [1]: John C. Hull. Options, Futures and Other Derivatives. Ninth Edition.
    June 2006.

  Args:
    yields: Real rank 1 `Tensor` of size `n`. The discount/zero rates.
    times: Real positive rank 1 `Tensor` of size `n`. The set of times
      corresponding to the supplied zero rates. If no `groups` is supplied, then
      the whole array should be sorted in an increasing order. If `groups` are
      supplied, then the times within a group should be in an increasing order.
    groups: Optional int `Tensor` of size `n` containing values between 0 and
      `k-1` where `k` is the number of different curves.
      Default value: None. This implies that all the rates are treated as a
        single group.
    dtype: `tf.Dtype`. If supplied the dtype for the `yields` and `times`.
      Default value: None which maps to the default dtype inferred from
      `yields`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'forward_rates_from_yields'.

  Returns:
    Real rank 1 `Tensor` of size `n` containing the forward rate that applies
    for each successive time interval (within each group if groups are
    specified).
  """
  with tf.compat.v1.name_scope(
      name,
      default_name='forward_rates_from_yields',
      values=[yields, times, groups]):
    yields = tf.convert_to_tensor(yields, dtype=dtype, name='yields')
    dtype = dtype or yields.dtype
    times = tf.convert_to_tensor(times, dtype=dtype, name='times')
    if groups is not None:
      groups = tf.convert_to_tensor(groups, name='groups')
    # (t2 * r(t2) - t1 * r(t1)) / (t2 - t1)
    rate_times = yields * times
    diff_rate_times = segment_ops.segment_diff(
        rate_times, order=1, exclusive=False, segment_ids=groups)
    diff_times = segment_ops.segment_diff(
        times, order=1, exclusive=False, segment_ids=groups)
    return diff_rate_times / diff_times


def yields_from_forward_rates(discrete_forwards,
                              times,
                              groups=None,
                              dtype=None,
                              name=None):
  """Computes yield rates from discrete forward rates.

  Denote the price of a zero coupon bond maturing at time `t` by `Z(t)`. Then
  the zero rate to time `t` is defined as

  ```None
    r(t) = - ln(Z(t)) / t       (1)

  ```

  This is the (continuously compounded) interest rate that applies between time
  `0` and time `t` as seen at time `0`. The forward rate between times `t1` and
  `t2` is defined as the interest rate that applies to the period `[t1, t2]`
  as seen from today. It is related to the zero coupon bond prices by

  ```None
    exp(-f(t1, t2)(t2-t1)) = Z(t2) / Z(t1)                 (2)
    f(t1, t2) = - (ln Z(t2) - ln Z(t1)) / (t2 - t1)        (3)
    f(t1, t2) = (t2 * r(t2) - t1 * r(t1)) / (t2 - t1)      (4)
  ```

  Given a sequence of increasing times `[t1, t2, ... tn]` and the forward rates
  for the consecutive time intervals, i.e. `[0, t1]`, `[t1, t2]` to
  `[t_{n-1}, tn]`, this function computes the yields to maturity for maturities
  `[t1, t2, ... tn]` using Eq. (4) above.

  Additionally, this function supports this computation for a batch of such
  forward rates. Batching is made slightly complicated by the fact that
  different zero curves may have different numbers of tenors (the parameter `n`
  above). Instead of a batch as an extra dimension, we support the concept of
  groups (also see documentation for `tf.segment_sum` which uses the same
  concept).

  #### Example

  The following example illustrates this method along with the concept of
  groups. Assuming there are two sets of zero rates (e.g. for different
  currencies) whose implied forward rates are needed. The first set has a total
  of three marked tenors at `[0.25, 0.5, 1.0]`. The second set
  has four marked tenors at `[0.25, 0.5, 1.0, 1.5]`.
  Suppose, the forward rates for the first set are:
  `[0.04, 0.041, 0.044]` and the second are `[0.022, 0.025, 0.028, 0.036]`.
  Then this data is batched together as follows:
  Groups:   [0,    0    0,   1,    1,   1    1  ]
  First three times for group 0, next four for group 1.
  Times:    [0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 1.5]
  First three discrete forwards for group 0, next four for group 1.
  Forwards: [0.04, 0.042, 0.047, 0.022, 0.028, 0.031, 0.052]

  ```python
    dtype = np.float64
    groups = np.array([0, 0, 0, 1, 1, 1, 1])
    times = np.array([0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 1.5], dtype=dtype)
    discrete_forwards = np.array(
        [0.04, 0.042, 0.047, 0.022, 0.028, 0.031, 0.052], dtype=dtype)
    yields = yields_from_forward_rates(discrete_forwards, times,
                                       groups=groups, dtype=dtype)
    # Produces: [0.04, 0.041, 0.044, 0.022, 0.025, 0.028, 0.036]
  ```

  #### References:

  [1]: John C. Hull. Options, Futures and Other Derivatives. Ninth Edition.
    June 2006.

  Args:
    discrete_forwards: Real rank 1 `Tensor` of size `n`. The forward rates for
      the time periods specified. Note that the first value applies between `0`
      and time `times[0]`.
    times: Real positive rank 1 `Tensor` of size `n`. The set of times
      corresponding to the supplied zero rates. If no `groups` is supplied, then
      the whole array should be sorted in an increasing order. If `groups` are
      supplied, then the times within a group should be in an increasing order.
    groups: Optional int `Tensor` of size `n` containing values between 0 and
      `k-1` where `k` is the number of different curves.
      Default value: None. This implies that all the rates are treated as a
        single group.
    dtype: `tf.Dtype`. If supplied the dtype for the `discrete_forwards` and
      `times`.
      Default value: None which maps to the default dtype inferred from
      `discrete_forwards`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'yields_from_forward_rates'.

  Returns:
    yields: Real rank 1 `Tensor` of size `n` containing the zero coupon yields
    that for the supplied maturities (within each group if groups are
    specified).
  """
  with tf.compat.v1.name_scope(
      name,
      default_name='yields_from_forward_rates',
      values=[discrete_forwards, times, groups]):
    discrete_forwards = tf.convert_to_tensor(discrete_forwards, dtype=dtype,
                                             name='discrete_forwards')
    dtype = dtype or discrete_forwards.dtype
    times = tf.convert_to_tensor(times, dtype=dtype, name='times')
    if groups is not None:
      groups = tf.convert_to_tensor(groups, name='groups')
    # Strategy for solving this equation without loops.
    # Define x_i = f_i (t_i - t_{i-1}) where f are the forward rates and
    # t_{-1}=0. Also define y_i = r_i t_i
    # Then the relationship between the forward rate and the yield can be
    # written as: x_i = y_i - y_{i-1} which we need to solve for y.
    # Hence, y_i = x_0 + x_1 + ... x_i.
    intervals = segment_ops.segment_diff(
        times, order=1, exclusive=False, segment_ids=groups)
    x = intervals * discrete_forwards
    y = segment_ops.segment_cumsum(x, exclusive=False, segment_ids=groups)
    return tf.math.divide_no_nan(y, times)


__all__ = ['forward_rates_from_yields',
           'yields_from_forward_rates',
           'forward_rates']
