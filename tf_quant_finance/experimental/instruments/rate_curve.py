# Lint as: python3
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

"""Interest rate curve defintion."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates
from tf_quant_finance.math.interpolation import linear


class RateCurve(object):
  """Represents an interest rate curve."""

  def __init__(self,
               maturity_dates,
               rates,
               valuation_date,
               compounding=None,
               interpolator=None,
               dtype=None,
               name=None):
    """Initializes the interest rate curve.

    Args:
      maturity_dates: A `DateTensor` containing the maturity dates on which the
        curve is specified.
      rates: A `Tensor` of real dtype specifying the rates (or yields)
        corresponding to the input maturities. The shape of this input should
        match the shape of `maturity_dates`.
      valuation_date: A scalar `DateTensor` specifying the valuation (or
        settlement) date for the curve.
      compounding: Optional scalar `Tensor` of dtype int32 specifying the
        componding frequency of the input rates. Use compounding=0 for
        continuously compounded rates. If compounding is different than 0, then
        rates are converted to continuously componded rates to perform
        interpolation.
        Default value: If omitted, the default value is 0.
      interpolator: Optional Python callable specifying the desired
        interpolation method. It should have the following interface: yi =
        interpolator(xi, x, y) `x`, `y`, 'xi', 'yi' are all `Tensors` of real
        dtype. `x` and `y` are the sample points and values (respectively) of
        the function to be interpolated. `xi` are the points at which the
        interpolation is desired and `yi` are the corresponding interpolated
        values returned by the function.
        Default value: None in which case linear interpolation is used.
      dtype: `tf.Dtype`. Optional input specifying the dtype of the `rates`
        input.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'rate_curve'.
    """
    self._name = name or 'rate_curve'
    with tf.compat.v1.name_scope(self._name):
      self._dtype = dtype
      if interpolator is None:
        def default_interpolator(xi, x, y):
          return linear.interpolate(xi, x, y, dtype=dtype)
        interpolator = default_interpolator

      if compounding is None:
        compounding = 0

      self._dates = dates.convert_to_date_tensor(maturity_dates)
      self._valuation_date = dates.convert_to_date_tensor(
          valuation_date)

      self._times = self._get_time(self._dates)
      self._rates = tf.convert_to_tensor(rates, dtype=self._dtype,
                                         name='curve_rates')

      if compounding > 0:
        self._rates = tf.where(
            self._times > 0.,
            tf.math.log(
                (1. + self._rates / compounding)**(compounding * self._rates)) /
            self._times, self._rates)
      self._interpolator = interpolator

  def get_rates(self, interpolation_dates):
    """Returns interpolated rates at `interpolation_dates`."""

    idates = dates.convert_to_date_tensor(interpolation_dates)
    times = self._get_time(idates)
    return self._interpolator(times, self._times, self._rates)

  def get_discount_factor(self, interpolation_dates):
    """Returns discount factors at `interpolation_dates`."""

    idates = dates.convert_to_date_tensor(interpolation_dates)
    times = self._get_time(idates)
    return tf.math.exp(-self.get_rates(idates) * times)

  def get_forward_rate(self, start_date, maturity_date, daycount_fraction=None):
    """Returns the simply accrued forward rate between [start_dt, maturity_dt].

    Args:
      start_date: A `DateTensor` specifying the start of the accrual period
        for the forward rate.
      maturity_date: A `DateTensor` specifying the end of the accrual period
        for the forward rate. The shape of `maturity_date` must be the same
        as the shape of the `DateTensor` `start_date`.
      daycount_fraction: An optional `Tensor` of real dtype specifying the
        time between `start_date` and `maturity_date` in years computed using
        the forward rate's day count basis. The shape of the input should be
        the same as that of `start_date` and `maturity_date`.
        Default value: `None`, in which case the daycount fraction is computed
        using `ACTUAL_365` convention.

    Returns:
      A real tensor of same shape as the inputs containing the simply compounded
      forward rate.
    """
    start_date = dates.convert_to_date_tensor(start_date)
    maturity_date = dates.convert_to_date_tensor(maturity_date)
    if daycount_fraction is None:
      daycount_fraction = dates.daycount_actual_365_fixed(
          start_date=start_date, end_date=maturity_date, dtype=self._dtype)
    else:
      daycount_fraction = tf.convert_to_tensor(daycount_fraction, self._dtype)
    dfstart = self.get_discount_factor(start_date)
    dfmaturity = self.get_discount_factor(maturity_date)
    return (dfstart / dfmaturity - 1.) / daycount_fraction

  @property
  def valuation_date(self):
    return self._valuation_date

  def _get_time(self, desired_dates):
    """Computes the year fraction from the curve's valuation date."""

    return dates.daycount_actual_365_fixed(
        start_date=self._valuation_date,
        end_date=desired_dates,
        dtype=self._dtype)


class RateCurveFromDiscountingFunction(RateCurve):
  """Implements `RateCurve` class using discounting function."""

  def __init__(self, maturity_dates, rates, valuation_date, discount_fn,
               dtype):
    super(RateCurveFromDiscountingFunction, self).__init__(
        maturity_dates, rates, valuation_date, dtype=dtype)
    self._discount_fn = discount_fn

  def get_discount_factor(self, interpolation_dates):
    return self._discount_fn(interpolation_dates)


def ratecurve_from_discounting_function(discount_fn, dtype=None):
  """Returns `RateCurve` object using the supplied function for discounting.

  Args:
    discount_fn: A python callable which takes a `DateTensor` as an input and
      returns the corresponding discount factor as an output.
    dtype: `tf.Dtype`. Optional input specifying the dtype of the real tensors
      and ops.

  Returns:
    An object of class `RateCurveFromDiscountingFunction` which uses the
    supplied function for discounting.
  """

  dtype = dtype or tf.constant(0.0).dtype
  pseudo_maturity_dates = dates.convert_to_date_tensor([(2020, 1, 1)])
  pseudo_rates = tf.convert_to_tensor([0.0], dtype=dtype)
  pseudo_valuation_date = dates.convert_to_date_tensor((2020, 1, 1))

  return RateCurveFromDiscountingFunction(
      pseudo_maturity_dates, pseudo_rates, pseudo_valuation_date,
      discount_fn, dtype)
