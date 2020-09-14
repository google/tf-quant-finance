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
"""Implementation of RateCurve object."""

from typing import Optional, Tuple

import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance import math
from tf_quant_finance import rates as rates_lib
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import interpolation_method
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils

_DayCountConventions = daycount_conventions.DayCountConventions
_InterpolationMethod = interpolation_method.InterpolationMethod

_DayCountConventionsProtoType = types.DayCountConventionsProtoType


class RateCurve(pmd.RateCurve):
  """Represents an interest rate curve."""

  def __init__(
      self,
      maturity_dates: types.DateTensor,
      discount_factors: tf.Tensor,
      valuation_date: types.DateTensor,
      interpolator: Optional[_InterpolationMethod] = None,
      interpolate_rates: Optional[bool] = True,
      daycount_convention: Optional[_DayCountConventionsProtoType] = None,
      curve_type: Optional[curve_types.CurveType] = None,
      dtype: Optional[tf.DType] = None,
      name: Optional[str] = None):
    """Initializes the interest rate curve.

    Args:
      maturity_dates: A `DateTensor` containing the maturity dates on which the
        curve is specified.
      discount_factors: A `Tensor` of real dtype specifying the discount factors
        corresponding to the input maturities. The shape of this input should
        match the shape of `maturity_dates`.
      valuation_date: A scalar `DateTensor` specifying the valuation (or
        settlement) date for the curve.
      interpolator: An instance of `InterpolationMethod`.
        Default value: `None` in which case cubic interpolation is used.
      interpolate_rates: A boolean specifying whether the interpolation should
        be done in discount rates or discount factors space.
        Default value: `True`, i.e., interpolation is done in the discount
        factors space.
      daycount_convention: `DayCountConventions` to use for the interpolation
        purpose.
        Default value: `None` which maps to actual/365 day count convention.
      curve_type: An instance of `CurveTypes` to mark the rate curve.
        Default value: `None` which means that the curve does not have the
          marker.
      dtype: `tf.Dtype`. Optional input specifying the dtype of the `rates`
        input.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'rate_curve'.
    """
    self._name = name or "rate_curve"
    with tf.compat.v1.name_scope(self._name):
      self._discount_factor_nodes = tf.convert_to_tensor(
          discount_factors, dtype=dtype,
          name="curve_discount_factors")
      self._dtype = dtype or self._discount_factor_nodes.dtype
      if interpolator is None or interpolator == _InterpolationMethod.CUBIC:
        def cubic_interpolator(xi, x, y):
          spline_coeffs = math.interpolation.cubic.build_spline(x, y)
          return math.interpolation.cubic.interpolate(xi, spline_coeffs,
                                                      dtype=dtype)
        interpolator = cubic_interpolator
        self._interpolation_method = _InterpolationMethod.CUBIC
      elif interpolator == _InterpolationMethod.LINEAR:
        def linear_interpolator(xi, x, y):
          return math.interpolation.linear.interpolate(xi, x, y,
                                                       dtype=dtype)
        interpolator = linear_interpolator
        self._interpolation_method = _InterpolationMethod.LINEAR
      elif interpolator == _InterpolationMethod.CONSTANT_FORWARD:
        def constant_fwd(xi, x, y):
          return rates_lib.constant_fwd.interpolate(xi, x, y, dtype=dtype)
        interpolator = constant_fwd
        self._interpolation_method = _InterpolationMethod.CONSTANT_FORWARD
      else:
        raise ValueError(f"Unknown interpolation method {interpolator}.")
      self._dates = dateslib.convert_to_date_tensor(maturity_dates)
      self._valuation_date = dateslib.convert_to_date_tensor(
          valuation_date)

      self._daycount_convention = (
          daycount_convention or _DayCountConventions.ACTUAL_365)
      self._day_count_fn = utils.get_daycount_fn(self._daycount_convention)
      self._times = self._get_time(self._dates)
      self._interpolator = interpolator
      self._interpolate_rates = interpolate_rates
      # Precompute discount rates:
      self._curve_type = curve_type

  @property
  def daycount_convention(self) -> types.DayCountConventionsProtoType:
    """Daycount convention."""
    return self._daycount_convention

  def daycount_fn(self):
    """Daycount function."""
    return self._day_count_fn

  @property
  def discount_factor_nodes(self) -> types.FloatTensor:
    """Discount factors at the interpolation nodes."""
    return self._discount_factor_nodes

  @property
  def node_dates(self) -> types.DateTensor:
    """Dates at which the discount factors and rates are specified."""
    return self._dates

  @property
  def discount_rate_nodes(self) -> types.FloatTensor:
    """Discount rates at the interpolation nodes."""
    discount_rates = tf.math.divide_no_nan(
        -tf.math.log(self.discount_factor_nodes), self._times,
        name="discount_rate_nodes")
    return discount_rates

  def set_discount_factor_nodes(self, values: types.FloatTensor):
    """Update discount factors at the interpolation nodes with new values."""
    values = tf.convert_to_tensor(values, dtype=self._dtype)
    values_shape = values.shape.as_list()
    nodes_shape = self.discount_factor_nodes.shape.as_list()
    if values_shape != nodes_shape:
      raise ValueError("New values should have shape {0} but are of "
                       "shape {1}".format(nodes_shape, values_shape))
    self._discount_factor_nodes = values

  def discount_rate(self,
                    interpolation_dates: types.DateTensor,
                    name: Optional[str] = None):
    """Returns interpolated rates at `interpolation_dates`."""

    interpolation_dates = dateslib.convert_to_date_tensor(
        interpolation_dates)
    times = self._get_time(interpolation_dates)
    rates = self._interpolator(times, self._times,
                               self.discount_rate_nodes)
    if self._interpolate_rates:
      rates = self._interpolator(times, self._times,
                                 self.discount_rate_nodes)
    else:
      discount_factor = self._interpolator(
          times, self._times, self.discount_factor_nodes)
      rates = -tf.math.divide_no_nan(
          tf.math.log(discount_factor), times)
    return tf.identity(rates, name=name or "discount_rate")

  def discount_factor(self,
                      interpolation_dates: types.DateTensor,
                      name: Optional[str] = None):
    """Returns discount factors at `interpolation_dates`."""

    interpolation_dates = dateslib.convert_to_date_tensor(
        interpolation_dates)
    times = self._get_time(interpolation_dates)
    if self._interpolate_rates:
      rates = self._interpolator(times, self._times,
                                 self.discount_rate_nodes)
      discount_factor = tf.math.exp(-rates * times)
    else:
      discount_factor = self._interpolator(
          times, self._times, self.discount_factor_nodes)
    return tf.identity(discount_factor, name=name or "discount_factor")

  def forward_rate(
      self,
      start_date: types.DateTensor,
      maturity_date: types.DateTensor,
      day_count_fraction: Optional[tf.Tensor] = None):
    """Returns the simply accrued forward rate between [start_dt, maturity_dt].

    Args:
      start_date: A `DateTensor` specifying the start of the accrual period
        for the forward rate.
      maturity_date: A `DateTensor` specifying the end of the accrual period
        for the forward rate. The shape of `maturity_date` must be the same
        as the shape of the `DateTensor` `start_date`.
      day_count_fraction: An optional `Tensor` of real dtype specifying the
        time between `start_date` and `maturity_date` in years computed using
        the forward rate's day count basis. The shape of the input should be
        the same as that of `start_date` and `maturity_date`.
        Default value: `None`, in which case the daycount fraction is computed
          using `daycount_convention`.

    Returns:
      A real `Tensor` of same shape as the inputs containing the simply
      compounded forward rate.
    """
    start_date = dateslib.convert_to_date_tensor(start_date)
    maturity_date = dateslib.convert_to_date_tensor(maturity_date)
    if day_count_fraction is None:
      day_count_fn = self._day_count_fn
      day_count_fraction = day_count_fn(
          start_date=start_date, end_date=maturity_date, dtype=self._dtype)
    else:
      day_count_fraction = tf.convert_to_tensor(day_count_fraction,
                                                self._dtype,
                                                name="day_count_fraction")
    dfstart = self.discount_factor(start_date)
    dfmaturity = self.discount_factor(maturity_date)
    return (dfstart / dfmaturity - 1.) / day_count_fraction

  @property
  def valuation_date(self) -> types.DateTensor:
    return self._valuation_date

  @property
  def interpolation_method(self) -> _InterpolationMethod:
    return self._interpolation_method

  def _get_time(self,
                dates: types.DateTensor) -> types.FloatTensor:
    """Computes the year fraction from the curve's valuation date."""
    return self._day_count_fn(start_date=self._valuation_date,
                              end_date=dates,
                              dtype=self._dtype)

  @property
  def curve_type(self) -> curve_types.CurveType:
    return self._curve_type

  def discount_factors_and_dates(self) -> Tuple[types.FloatTensor,
                                                types.DateTensor]:
    """Returns discount factors and dates at which the discount curve is fitted.
    """
    return (self._discount_factor_nodes, self._dates)

  @property
  def dtype(self) -> types.Dtype:
    return self._dtype

  @property
  def interpolate_rates(self) -> bool:
    """Returns `True` if the interpolation is on rates and not on discounts."""
    return self._interpolate_rates


__all__ = ["RateCurve"]
