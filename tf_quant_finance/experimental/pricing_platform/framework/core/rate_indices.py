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
"""Rate indices."""

import enum
from tf_quant_finance.experimental.pricing_platform.instrument_protos import rate_indices_pb2


RateIndexType = enum.Enum("RateIndexType",
                          zip(rate_indices_pb2.RateIndexType.keys(),
                              rate_indices_pb2.RateIndexType.keys()))
RateIndexType.__doc__ = "Supported rate indices."
RateIndexType.__repr__ = lambda self: self.value
RateIndexType.__call__ = lambda self: self.value


def from_proto_value(value: int) -> RateIndexType.mro()[0]:
  """Creates RateIndexType from a proto field value."""
  return RateIndexType(rate_indices_pb2.RateIndexType.Name(value))


__all__ = ["RateIndexType", "from_proto_value"]
