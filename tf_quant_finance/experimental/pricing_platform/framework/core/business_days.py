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
"""Supported bank holidays and business day conventions."""

import enum
from tf_quant_finance.experimental.pricing_platform.instrument_protos import business_days_pb2

__all__ = ["BusinessDayConvention",
           "convention_from_proto_value",
           "BankHolidays",
           "holiday_from_proto_value"]


BusinessDayConvention = enum.Enum(
    "BusinessDayConvention",
    zip(business_days_pb2.BusinessDayConvention.keys(),
        business_days_pb2.BusinessDayConvention.keys()))
BusinessDayConvention.__doc__ = "Supported business day conventions."
BusinessDayConvention.__repr__ = lambda self: self.value
BusinessDayConvention.__call__ = lambda self: self.value


# Typing can't resolve type of generated enums so we use Method Resolution Order
# to infer the type.
BusinessDayConventionProtoType = BusinessDayConvention.mro()[0]


def convention_from_proto_value(value: int) -> BusinessDayConventionProtoType:
  """Creates BusinessDayConvention from a proto field value."""
  return BusinessDayConvention(
      business_days_pb2.BusinessDayConvention.Name(value))


BankHolidays = enum.Enum("BankHolidays",
                         zip(business_days_pb2.BankHolidays.keys(),
                             business_days_pb2.BankHolidays.keys()))
BankHolidays.__doc__ = "Supported bank holidays."
BankHolidays.__repr__ = lambda self: self.value
BankHolidays.__call__ = lambda self: self.value


# Typing can't resolve type of generated enums so we use Method Resolution Order
# to infer the type.
BankHolidaysProtoType = BankHolidays.mro()[0]


def holiday_from_proto_value(value: int) -> BankHolidaysProtoType:
  """Creates BankHolidays from a proto field value."""
  return BankHolidays(business_days_pb2.BankHolidays.Name(value))
