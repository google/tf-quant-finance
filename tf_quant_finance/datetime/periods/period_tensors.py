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
"""PeriodTensor definition."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.datetime import constants
from tf_quant_finance.datetime import tensor_wrapper


def day():
  return days(1)


def days(n):
  return PeriodTensor(n, constants.PeriodType.DAY)


def week():
  return weeks(1)


def weeks(n):
  return PeriodTensor(n, constants.PeriodType.WEEK)


def month():
  return months(1)


def months(n):
  return PeriodTensor(n, constants.PeriodType.MONTH)


def year():
  return years(1)


def years(n):
  return PeriodTensor(n, constants.PeriodType.YEAR)


class PeriodTensor(tensor_wrapper.TensorWrapper):
  """Represents a tensor of time periods."""

  def __init__(self, quantity, period_type):
    """Initializer.

    Args:
      quantity: A Tensor of type tf.int32, representing the quantities
        of period types (e.g. how many months). Can be both positive and
        negative.
      period_type: A PeriodType (a day, a month, etc). Currently only one
        PeriodType per instance of PeriodTensor is supported.

    Example:
    ```python
    two_weeks = PeriodTensor(2, PeriodType.WEEK)

    months = [3, 6, 9, 12]
    periods = PeriodTensor(months, PeriodType.MONTH)
    ```
    """
    self._quantity = tf.convert_to_tensor(quantity, dtype=tf.int32,
                                          name="pt_quantity")
    self._period_type = period_type

  def period_type(self):
    """Returns the PeriodType of this PeriodTensor."""
    return self._period_type

  def quantity(self):
    """Returns the quantities tensor of this PeriodTensor."""
    return self._quantity

  def __mul__(self, multiplier):
    """Multiplies PeriodTensor by a Tensor of ints."""
    multiplier = tf.convert_to_tensor(multiplier, tf.int32)
    return PeriodTensor(self._quantity * multiplier, self._period_type)

  def __add__(self, other):
    """Adds another PeriodTensor of the same type."""
    if other.period_type() != self._period_type:
      raise ValueError("Mixing different period types is not supported")

    return PeriodTensor(self._quantity + other.quantity(), self._period_type)

  def __sub__(self, other):
    """Subtracts another PeriodTensor of the same type."""
    if other.period_type() != self._period_type:
      raise ValueError("Mixing different period types is not supported")

    return PeriodTensor(self._quantity - other.quantity(), self._period_type)

  @property
  def shape(self):
    return self._quantity.shape

  @property
  def rank(self):
    return tf.rank(self._quantity)

  @classmethod
  def _apply_sequence_to_tensor_op(cls, op_fn, tensor_wrappers):
    q = op_fn([t.quantity() for t in tensor_wrappers])
    period_type = tensor_wrappers[0].period_type()
    if not all(t.period_type() == period_type for t in tensor_wrappers[1:]):
      raise ValueError("Combined PeriodTensors must have the same PeriodType")
    return PeriodTensor(q, period_type)

  def _apply_op(self, op_fn):
    q = op_fn(self._quantity)
    return PeriodTensor(q, self._period_type)

  def __repr__(self):
    output = "PeriodTensor: shape={}".format(self.shape)
    if tf.executing_eagerly():
      return output + ", quantities={}".format(repr(self._quantity.numpy()))
    return output


__all__ = [
    "day",
    "days",
    "month",
    "months",
    "week",
    "weeks",
    "year",
    "years",
    "PeriodTensor",
]
