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
"""Supported day count conventions."""

import enum
from tf_quant_finance.experimental.pricing_platform.instrument_protos import daycount_conventions_pb2

DayCountConventions = enum.Enum(
    "DayCountConventions",
    zip(daycount_conventions_pb2.DayCountConvention.keys(),
        daycount_conventions_pb2.DayCountConvention.keys()))
DayCountConventions.__doc__ = "Supported day count conventions."
DayCountConventions.__repr__ = lambda self: self.value
DayCountConventions.__call__ = lambda self: self.value


# Typing can't resolve type of Currency so we use Method Resolution Order to
# infer the type.
DayCountConventionsProtoType = DayCountConventions.mro()[0]


def from_proto_value(value: int) -> DayCountConventionsProtoType:
  """Creates DayCountConventions from a proto field value."""
  return DayCountConventions(
      daycount_conventions_pb2.DayCountConvention.Name(value))


__all__ = ["DayCountConventions", "from_proto_value"]


