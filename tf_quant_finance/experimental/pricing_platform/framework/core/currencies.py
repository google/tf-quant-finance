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
"""Supported currencies."""

import enum
from tf_quant_finance.experimental.pricing_platform.instrument_protos import currencies_pb2


Currency = enum.Enum("Currency",
                     zip(currencies_pb2.Currency.keys(),
                         currencies_pb2.Currency.keys()))
Currency.__doc__ = "Supported currencies."
Currency.__repr__ = lambda self: self.value
Currency.__call__ = lambda self: self.value


# Typing can't resolve type of generated enums so we use Method Resolution Order
# to infer the type.
CurrencyProtoType = Currency.mro()[0]


def from_proto_value(value: int) -> CurrencyProtoType:
  """Creates Currency from a proto field value."""
  return Currency(currencies_pb2.Currency.Name(value))

__all__ = ["Currency", "from_proto_value"]
