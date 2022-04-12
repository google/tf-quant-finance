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
"""Collection of functions to create interest rate and equity swaps."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.rates.analytics import cashflows

__all__ = [
    'swap_price',
    'equity_leg_cashflows',
    'rate_leg_cashflows',
    'ir_swap_price',
    'ir_swap_par_rate_and_annuity',
    'equity_swap_price'
]


def swap_price(pay_leg_cashflows,
               receive_leg_cashflows,
               pay_leg_discount_factors,
               receive_leg_discount_factors,
               dtype=None,
               name=None):
  """Computes prices of a batch of generic swaps.

  #### Example
  ```python
  pay_leg_cashflows = [[100, 100, 100], [200, 250, 300]]
  receive_leg_cashflows = [[200, 250, 300, 300], [100, 100, 100, 100]]
  pay_leg_discount_factors = [[0.95, 0.9, 0.8],
                              [0.9, 0.85, 0.8]]
  receive_leg_discount_factors = [[0.95, 0.9, 0.8, 0.75],
                                  [0.9, 0.85, 0.8, 0.75]]
  swap_price(pay_leg_cashflows=pay_leg_cashflows,
             receive_leg_cashflows=receive_leg_cashflows,
             pay_leg_discount_factors=pay_leg_discount_factors,
             receive_leg_discount_factors=receive_leg_discount_factors,
             dtype=tf.float64)
  # Expected: [615.0, -302.5]
  ```

  Args:
    pay_leg_cashflows: A real `Tensor` of shape
      `batch_shape + [num_pay_cashflows]`, where `num_pay_cashflows` is the
      number of cashflows for each batch element. Cashflows of the pay leg of
      the swaps.
    receive_leg_cashflows: A `Tensor` of the same `dtype` as `pay_leg_cashflows`
      and of shape `batch_shape + [num_receive_cashflows]` where
      `num_pay_cashflows` is the number of cashflows for each batch element.
      Cashflows of the receive leg of the swaps.
    pay_leg_discount_factors: A `Tensor` of the same `dtype` as
      `pay_leg_cashflows` and of compatible shape. Discount factors for each
      cashflow of the pay leg.
    receive_leg_discount_factors: A `Tensor` of the same `dtype` as
      `receive_leg_cashflows` and of compatible shape. Discount factors for each
      cashflow of the receive leg.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `pay_leg_cashflows`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'floating_coupons'.

  Returns:
    A `Tensor` of the same `dtype` as `coupon_rates` and of shape `batch_shape`.
    Present values of swaps from receiver perspective.
  """
  name = name or 'swap_price'
  with tf.name_scope(name):
    pay_leg_cashflows = tf.convert_to_tensor(
        pay_leg_cashflows, dtype=dtype, name='pay_leg_cashflows')
    dtype = dtype or pay_leg_cashflows.dtype
    receive_leg_cashflows = tf.convert_to_tensor(
        receive_leg_cashflows, dtype=dtype, name='receive_leg_cashflows')
    pay_leg_discount_factors = tf.convert_to_tensor(
        pay_leg_discount_factors, dtype=dtype, name='pay_leg_discount_factors')
    receive_leg_discount_factors = tf.convert_to_tensor(
        receive_leg_discount_factors, dtype=dtype,
        name='receive_leg_discount_factors')
    receive_leg_pv = cashflows.present_value(
        receive_leg_cashflows,
        receive_leg_discount_factors)
    pay_leg_pv = cashflows.present_value(
        pay_leg_cashflows,
        pay_leg_discount_factors)
    return receive_leg_pv - pay_leg_pv


def equity_leg_cashflows(
    forward_prices,
    spots,
    notional,
    dividends=None,
    dtype=None,
    name=None):
  """Computes cashflows for a batch of equity legs.

  Equity cashflows are defined as a total equity return between pay dates, say,
  `T_1, ..., T_n`. Let `S_i` represent the value of the equity at time `T_i` and
  `d_i` be a discrete dividend paid at this time. Then the the payment at time
  `T_i` is defined as `(S_i - S_{i - 1}) / S_{i-1} + d_i`. The value of
  the cashflow is then the discounted sum of the paments. See, e.g., [1] for the
  reference.

  #### Example
  ```python
  notional = 10000
  forward_prices = [[110, 120, 140], [210, 220, 240]]
  spots = [100, 200]
  dividends = [[1, 1, 1], [2, 2, 2]]
  equity_leg_cashflows(forward_prices, spots, notional, dividends,
                       dtype=tf.float64)
  # Expected:
  #  [[1000.01, 909.1, 1666.675],
  #   [ 500.01, 476.2, 909.1]]
  ```

  Args:
    forward_prices: A real `Tensor` of shape `batch_shape + [num_cashflows]`,
      where `num_cashflows` is the number of cashflows for each batch element.
      Equity forward prices at leg reset times.
    spots:  A `Tensor` of the same `dtype` as `forward_prices` and of
      shape compatible with `batch_shape`. Spot prices for each batch element
    notional: A `Tensor` of the same `dtype` as `forward_prices` and of
      compatible shape. Notional amount for each cashflow.
    dividends:  A `Tensor` of the same `dtype` as `forward_prices` and of
      compatible shape. Discrete dividends paid at the leg reset times.
      Default value: None which maps to zero dividend.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `forward_prices`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'equity_leg_cashflows'.

  Returns:
    A `Tensor` of the same `dtype` as `forward_prices` and of shape
    `batch_shape + [num_cashflows]`.

  #### References
  [1] Don M. Chance and Don R Rich,
    The Pricing of Equity Swaps and Swaptions, 1998
    https://jod.pm-research.com/content/5/4/19
  """
  name = name or 'equity_leg_cashflows'
  with tf.name_scope(name):
    forward_prices = tf.convert_to_tensor(
        forward_prices, dtype=dtype, name='forward_prices')
    dtype = dtype or forward_prices.dtype
    spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    dividends = 0 if dividends is None else dividends
    dividends = tf.convert_to_tensor(dividends, dtype=dtype, name='dividends')
    spots_expand = tf.expand_dims(spots, axis=-1)
    forward_prices = tf.concat([spots_expand, forward_prices], axis=-1)
    # Cashflows are equal to
    # (forward_{i+1} - forward_{i} + dividend_{i}) / forward_prices_{i}
    return tf.math.divide_no_nan(
        notional * (forward_prices[..., 1:] - forward_prices[..., :-1])
        + dividends,
        forward_prices[..., :-1])


def rate_leg_cashflows(
    coupon_rates,
    notional,
    daycount_fractions,
    dtype=None,
    name=None):
  """Computes cashflows for a batch or interest rate legs.

  #### Example
  ```python
  coupon_rates = [[0.1, 0.1, 0.1], [0.02, 0.12, 0.14]]
  notional = 1000
  daycount_fractions = [[1, 1, 1], [1, 2, 1]]
  rate_leg_cashflows(
      coupon_rates, notional, daycount_fractions, dtype=tf.float64)
  # Expected:
  #  [[100.0, 100.0, 100.0],
  #   [ 20.0, 240.0, 140.0]]
  ```

  Args:
    coupon_rates: A real `Tensor` of shape `batch_shape + [num_cashflows]`,
      where `num_cashflows` is the number of cashflows for each batch element.
      Coupon rates for each cashflow of the leg. Can be a scalar for a fixed
      leg or represent forward rates for a floating leg.
    notional: A `Tensor` of the same `dtype` as `coupon_rates` and of
      compatible shape. Notional amount for each cashflow.
    daycount_fractions: A `Tensor` of the same `dtype` as `coupon_rates` and of
      compatible shape. Year fractions for the coupon accrual.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `coupon_rates`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'rate_leg_cashflows'.

  Returns:
    A `Tensor` of the same `dtype` as `coupon_rates` and of shape
    `batch_shape + [num_cashflows]`.
  """
  name = name or 'rate_leg_cashflows'
  with tf.name_scope(name):
    coupon_rates = tf.convert_to_tensor(
        coupon_rates, dtype=dtype, name='coupon_rates')
    dtype = dtype or coupon_rates.dtype
    daycount_fractions = tf.convert_to_tensor(
        daycount_fractions, dtype=dtype, name='daycount_fractions')
    notional = tf.convert_to_tensor(
        notional, dtype=dtype, name='notional')
    return notional * daycount_fractions * coupon_rates


def ir_swap_price(
    pay_leg_coupon_rates,
    receive_leg_coupon_rates,
    pay_leg_notional,
    receive_leg_notional,
    pay_leg_daycount_fractions,
    receive_leg_daycount_fractions,
    pay_leg_discount_factors,
    receive_leg_discount_factors,
    dtype=None,
    name=None):
  """Computes prices of a batch of interest rate swaps.

  #### Example
  ```python
  pay_leg_coupon_rates = [[0.1], [0.15]]
  receive_leg_coupon_rates = [[0.1, 0.2, 0.05], [0.1, 0.05, 0.2]]
  notional = 1000
  pay_leg_daycount_fractions = 0.5
  receive_leg_daycount_fractions = [[0.5, 0.5, 0.5], [0.4, 0.5, 0.6]]
  discount_factors = [[0.95, 0.9, 0.85], [0.98, 0.92, 0.88]]

  ir_swap_price(
      pay_leg_coupon_rates=pay_leg_coupon_rates,
      receive_leg_coupon_rates=receive_leg_coupon_rates,
      pay_leg_notional=notional,
      receive_leg_notional=notional,
      pay_leg_daycount_fractions=pay_leg_daycount_fractions,
      receive_leg_daycount_fractions=receive_leg_daycount_fractions,
      pay_leg_discount_factors=discount_factors,
      receive_leg_discount_factors=discount_factors,
      dtype=tf.float64)
  # Expected: [23.75, -40.7]
  ```

  Args:
    pay_leg_coupon_rates: A real `Tensor` of shape
      `batch_shape + [num_pay_cashflows]`, where `num_pay_cashflows` is the
      number of cashflows for each batch element. Coupon rates for the paying
      leg.
    receive_leg_coupon_rates: A `Tensor` of the same `dtype` as
      `pay_leg_coupon_rates` and of shape
      `batch_shape + [num_receive_cashflows]`, where `num_receive_cashflows` is
      the number of cashflows for each batch element. Coupon rates the
      receiving leg.
    pay_leg_notional: A `Tensor` of the same `dtype` as `pay_leg_coupon_rates`
      and of compatible shape. Notional amount for each cashflow.
    receive_leg_notional: A `Tensor` of the same `dtype` as
      `receive_leg_coupon_rates` and of compatible shape. Notional amount for
      each cashflow.
    pay_leg_daycount_fractions: A `Tensor` of the same `dtype` as
      `pay_leg_coupon_rates` and of compatible shape.  Year fractions for the
      coupon accrual.
    receive_leg_daycount_fractions: A `Tensor` of the same `dtype` as
      `receive_leg_coupon_rates` and of compatible shape.  Year fractions for
      the coupon accrual.
    pay_leg_discount_factors: A `Tensor` of the same `dtype` as
      `pay_leg_coupon_rates` and of compatible shape. Discount factors for each
      cashflow of the pay leg.
    receive_leg_discount_factors: A `Tensor` of the same `dtype` as
      `pay_leg_coupon_rates` and of compatible shape. Discount factors for each
      cashflow of the receive leg.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `pay_leg_coupon_rates`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'equity_swap_price'.

  Returns:
    A `Tensor` of the same `dtype` as `pay_leg_coupon_rates` and of shape
    `batch_shape`. Present values of the interest rate swaps.

  """
  name = name or 'ir_swap_price'
  with tf.name_scope(name):
    pay_leg_coupon_rates = tf.convert_to_tensor(
        pay_leg_coupon_rates, dtype=dtype, name='pay_leg_coupon_rates')
    dtype = dtype or pay_leg_coupon_rates.dtype
    receive_leg_coupon_rates = tf.convert_to_tensor(
        receive_leg_coupon_rates, dtype=dtype, name='receive_leg_coupon_rates')
    pay_leg_notional = tf.convert_to_tensor(
        pay_leg_notional, dtype=dtype, name='pay_leg_notional')
    receive_leg_notional = tf.convert_to_tensor(
        receive_leg_notional, dtype=dtype,
        name='receive_leg_notional')
    pay_leg_daycount_fractions = tf.convert_to_tensor(
        pay_leg_daycount_fractions, dtype=dtype,
        name='pay_leg_daycount_fractions')
    receive_leg_daycount_fractions = tf.convert_to_tensor(
        receive_leg_daycount_fractions, dtype=dtype,
        name='receive_leg_daycount_fractions')
    pay_leg_discount_factors = tf.convert_to_tensor(
        pay_leg_discount_factors, dtype=dtype, name='pay_leg_discount_factors')
    receive_leg_discount_factors = tf.convert_to_tensor(
        receive_leg_discount_factors, dtype=dtype,
        name='receive_leg_discount_factors')

    pay_leg_cashflows = rate_leg_cashflows(
        pay_leg_notional,
        pay_leg_daycount_fractions,
        pay_leg_coupon_rates)
    receive_leg_cashflows = rate_leg_cashflows(
        receive_leg_notional,
        receive_leg_daycount_fractions,
        receive_leg_coupon_rates)
    return swap_price(pay_leg_cashflows,
                      receive_leg_cashflows,
                      pay_leg_discount_factors,
                      receive_leg_discount_factors)


def ir_swap_par_rate_and_annuity(floating_leg_start_times,
                                 floating_leg_end_times,
                                 fixed_leg_payment_times,
                                 fixed_leg_daycount_fractions,
                                 reference_rate_fn,
                                 dtype=None,
                                 name=None):
  """Utility function to compute par swap rate and annuity.

  Args:
    floating_leg_start_times: A real `Tensor` of the same dtype as `expiries`.
      The times when accrual begins for each payment in the floating leg. The
      shape of this input should be `expiries.shape + [m]` where `m` denotes the
      number of floating payments in each leg.
    floating_leg_end_times: A real `Tensor` of the same dtype as `expiries`. The
      times when accrual ends for each payment in the floating leg. The shape of
      this input should be `expiries.shape + [m]` where `m` denotes the number
      of floating payments in each leg.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as `expiries`.
      The payment times for each payment in the fixed leg. The shape of this
      input should be `expiries.shape + [n]` where `n` denotes the number of
      fixed payments in each leg.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions for
      each payment in the fixed leg.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape + [dim]`. Returns
      the continuously compounded zero rate at the present time for the input
      expiry time.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
        `floating_leg_start_times`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'ir_swap_par_rate_and_annuity'.

  Returns:
    A tuple with two elements containing par swap rate and swap annuities.
  """
  name = name or 'ir_swap_par_rate_and_annuity'
  with tf.name_scope(name):
    floating_leg_start_times = tf.convert_to_tensor(
        floating_leg_start_times, dtype=dtype)
    dtype = dtype or floating_leg_start_times.dtype
    floating_leg_end_times = tf.convert_to_tensor(
        floating_leg_end_times, dtype=dtype)
    fixed_leg_payment_times = tf.convert_to_tensor(
        fixed_leg_payment_times, dtype=dtype)
    fixed_leg_daycount_fractions = tf.convert_to_tensor(
        fixed_leg_daycount_fractions, dtype=dtype)

    floating_leg_start_df = tf.math.exp(
        -reference_rate_fn(floating_leg_start_times) * floating_leg_start_times)
    floating_leg_end_df = tf.math.exp(
        -reference_rate_fn(floating_leg_end_times) * floating_leg_end_times)
    fixed_leg_payment_df = tf.math.exp(
        -reference_rate_fn(fixed_leg_payment_times) * fixed_leg_payment_times)
    annuity = tf.math.reduce_sum(
        fixed_leg_payment_df * fixed_leg_daycount_fractions, axis=-1)
    swap_rate = tf.math.reduce_sum(
        floating_leg_start_df - floating_leg_end_df, axis=-1) / annuity
    return swap_rate, annuity


def equity_swap_price(
    rate_leg_coupon_rates,
    equity_leg_forward_prices,
    equity_leg_spots,
    rate_leg_notional,
    equity_leg_notional,
    rate_leg_daycount_fractions,
    rate_leg_discount_factors,
    equity_leg_discount_factors,
    equity_dividends=None,
    is_equity_receiver=None,
    dtype=None,
    name=None):
  """Computes prices of a batch of equity swaps.

  The swap consists of an equity and interest rate legs.

  #### Example
  ```python
  rate_leg_coupon_rates = [[0.1, 0.2, 0.05], [0.1, 0.05, 0.2]]
  # Two cashflows of 4 and 3 payments, respectively
  forward_prices = [[110, 120, 140, 150], [210, 220, 240, 0]]
  spots = [100, 200]
  notional = 1000
  pay_leg_daycount_fractions = 0.5
  rate_leg_daycount_fractions = [[0.5, 0.5, 0.5], [0.4, 0.5, 0.6]]
  rate_leg_discount_factors = [[0.95, 0.9, 0.85], [0.98, 0.92, 0.88]]
  equity_leg_discount_factors = [[0.95, 0.9, 0.85, 0.8],
                                 [0.98, 0.92, 0.88, 0.0]]

  equity_swap_price(
      rate_leg_coupon_rates=rate_leg_coupon_rates,
      equity_leg_forward_prices=forward_prices,
      equity_leg_spots=spots,
      rate_leg_notional=notional,
      equity_leg_notional=notional,
      rate_leg_daycount_fractions=rate_leg_daycount_fractions,
      rate_leg_discount_factors=rate_leg_discount_factors,
      equity_leg_discount_factors=equity_leg_discount_factors,
      is_equity_receiver=[True, False],
      dtype=tf.float64)
  # Expected: [216.87770563, -5.00952381]
  forward_rates(df_start_dates, df_end_dates, daycount_fractions,
                dtype=tf.float64)
  ```

  Args:
    rate_leg_coupon_rates: A real `Tensor` of shape
      `batch_shape + [num_rate_cashflows]`, where `num_rate_cashflows` is the
      number of cashflows for each batch element. Coupon rates for the
      interest rate leg.
    equity_leg_forward_prices: A `Tensor` of the same `dtype` as
      `rate_leg_coupon_rates` and of shape
      `batch_shape + [num_equity_cashflows]`, where `num_equity_cashflows` is
      the number of cashflows for each batch element. Equity leg forward
      prices.
    equity_leg_spots: A `Tensor` of the same `dtype` as
      `equity_leg_forward_prices` and of shape compatible with `batch_shape`.
      Spot prices for each batch element of the equity leg.
    rate_leg_notional: A `Tensor` of the same `dtype` as `rate_leg_coupon_rates`
      and of compatible shape. Notional amount for each cashflow.
    equity_leg_notional: A `Tensor` of the same `dtype` as
      `equity_leg_forward_prices` and of compatible shape.  Notional amount for
      each cashflow.
    rate_leg_daycount_fractions: A `Tensor` of the same `dtype` as
      `rate_leg_coupon_rates` and of compatible shape.  Year fractions for the
      coupon accrual.
    rate_leg_discount_factors: A `Tensor` of the same `dtype` as
      `rate_leg_coupon_rates` and of compatible shape. Discount factors for each
      cashflow of the rate leg.
    equity_leg_discount_factors: A `Tensor` of the same `dtype` as
      `equity_leg_forward_prices` and of compatible shape. Discount factors for
      each cashflow of the equity leg.
    equity_dividends: A `Tensor` of the same `dtype` as
      `equity_leg_forward_prices` and of compatible shape. Dividends paid at the
      leg reset times.
      Default value: None which maps to zero dividend.
    is_equity_receiver: A boolean `Tensor` of shape compatible with
      `batch_shape`. Indicates whether the swap holder is equity holder or
      receiver.
      Default value: None which means that all swaps are equity reiver swaps.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `rate_leg_coupon_rates`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'equity_swap_price'.

  Returns:
    A `Tensor` of the same `dtype` as `rate_leg_coupon_rates` and of shape
    `batch_shape`. Present values of the equity swaps.
  """
  name = name or 'equity_swap_price'
  with tf.name_scope(name):
    rate_leg_coupon_rates = tf.convert_to_tensor(
        rate_leg_coupon_rates, dtype=dtype, name='rate_leg_coupon_rates')
    dtype = dtype or rate_leg_coupon_rates.dtype
    equity_leg_forward_prices = tf.convert_to_tensor(
        equity_leg_forward_prices, dtype=dtype,
        name='equity_leg_forward_prices')
    equity_leg_spots = tf.convert_to_tensor(
        equity_leg_spots, dtype=dtype,
        name='equity_leg_spots')
    rate_leg_daycount_fractions = tf.convert_to_tensor(
        rate_leg_daycount_fractions, dtype=dtype,
        name='rate_leg_daycount_fractions')
    equity_dividends = equity_dividends or 0
    equity_dividends = tf.convert_to_tensor(
        equity_dividends, dtype=dtype,
        name='equity_dividends')
    rate_leg_notional = tf.convert_to_tensor(
        rate_leg_notional, dtype=dtype,
        name='rate_leg_notional')
    equity_leg_notional = tf.convert_to_tensor(
        equity_leg_notional, dtype=dtype,
        name='equity_leg_notional')
    rate_leg_discount_factors = tf.convert_to_tensor(
        rate_leg_discount_factors, dtype=dtype,
        name='rate_leg_discount_factors')
    equity_leg_discount_factors = tf.convert_to_tensor(
        equity_leg_discount_factors, dtype=dtype,
        name='equity_leg_discount_factors')
    if is_equity_receiver is None:
      is_equity_receiver = True
    is_equity_receiver = tf.convert_to_tensor(
        is_equity_receiver, dtype=tf.bool, name='is_equity_receiver')
    one = tf.ones([], dtype=dtype)
    equity_receiver = tf.where(is_equity_receiver, one, -one)
    equity_cashflows = equity_leg_cashflows(
        forward_prices=equity_leg_forward_prices,
        spots=equity_leg_spots,
        notional=equity_leg_notional,
        dividends=equity_dividends)
    rate_cashflows = rate_leg_cashflows(
        coupon_rates=rate_leg_coupon_rates,
        notional=rate_leg_notional,
        daycount_fractions=rate_leg_daycount_fractions)
    return equity_receiver * swap_price(
        rate_cashflows,
        equity_cashflows,
        rate_leg_discount_factors,
        equity_leg_discount_factors)
