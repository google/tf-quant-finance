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
import tensorflow.compat.v1 as tf
from tf_quant_finance.math.interpolation import linear


class RateCurve(object):
  """Represents an interest rate curve."""

  def __init__(self,
               maturity,
               rate,
               compounding=None,
               interpolator=None,
               dtype=None,
               name=None):
    """Initializes the interest rate curve.

    Args:
      maturity: A `Tensor` of real dtype specifying the time to maturities of
        the curve in years.
      rate: A `Tensor` of real dtype specifying the rates (or yields)
        corresponding to the input maturities. The shape of this input should
        match the shape of `maturity`.
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
      dtype: `tf.Dtype`. Optional input specifying the dtype of the input
        `Tensors`.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'rate_curve'.
    """
    self._name = name or 'rate_curve'
    with tf.compat.v1.name_scope(self._name,
                                 values=[maturity, rate, compounding,
                                         interpolator]):
      self._dtype = dtype
      if interpolator is None:
        def default_interpolator(xi, x, y):
          return linear.interpolate(xi, x, y, dtype=dtype)
        interpolator = default_interpolator

      if compounding is None:
        compounding = 0

      self._times = tf.convert_to_tensor(maturity, dtype, 'curve_times')
      self._rates = tf.convert_to_tensor(rate, dtype, 'curve_rates')

      if compounding > 0:
        self._rates = tf.where(
            self._times > 0.,
            tf.math.log(
                (1. + self._rates / compounding)**(compounding * self._rates)) /
            self._times, self._rates)
      self._interpolator = interpolator

  def get_rates(self, times):
    """Returns interpolated rates at input `times`."""

    times = tf.convert_to_tensor(times, self._dtype)
    return self._interpolator(times, self._times, self._rates)

  def get_discount(self, times):
    """Returns discount factors at input `times`."""

    times = tf.convert_to_tensor(times, self._dtype)
    return tf.math.exp(-self.get_rates(times) * times)

  def get_forward_rate(self, t_start, t_maturity, daycount_fraction):
    """Returns the simply accrued forward rate with span [t_start, t_maturity].

    Args:
      t_start: A `Tensor` of real dtype specifying the start of the accrual
        period for the forward rate.
      t_maturity: A `Tensor` of real dtype specifying the end of the accrual
        period for the forward rate. The shape of `t_maturity` must be the same
        as the shape of the `Tensor` `t_start`.
      daycount_fraction: A `Tensor` of real dtype specifying the a time between
        `t_start` and `t_maturity` in years computed using the forward rate's
        day count basis. The shape of the input should be the same as that if
        `t_start` and `t_maturity`.

    Returns:
      A real tensor of same shape as the inputs containing the simply compounded
      forward rate.
    """

    dfstart = self.get_discount(t_start)
    dfmaturity = self.get_discount(t_maturity)
    return (dfstart / dfmaturity - 1.) / daycount_fraction
