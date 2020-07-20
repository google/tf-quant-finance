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
"""Curve types."""

import enum
import dataclasses
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies


class Index(enum.Enum):
  """Supported rate curve indices.

  Index here is used to refer to a specific curve in a market data.
  Instrument configuration specifies which curve to use for pricing and, more
  specifically, how to map `RateIndexType` to `Index`.
  """
  OIS = "OIS"

  SOFR = "SOFR"  # USD
  SONIA = "SONIA"  # GBP
  ESTER = "ESTER"  # EUR
  SARON = "SARON"  # CHF
  LIBOR_OVERNIGHT = "LIBOR_OVERNIGHT"
  LIBOR_1W = "LIBOR_1W"
  LIBOR_1M = "LIBOR_1M"
  LIBOR_3M = "LIBOR_3M"
  LIBOR_6M = "LIBOR_6M"
  LIBOR_1Y = "LIBOR_1Y"
  EURIBOR_OVERNIGHT = "EURIBOR_OVERNIGHT"  # EUR
  EURIBOR_1W = "EURIBOR_1W"  # EUR
  EURIBOR_1M = "EURIBOR_1M"  # EUR
  EURIBOR_3M = "EURIBOR_3M"  # EUR
  EURIBOR_6M = "EURIBOR_6M"  # EUR
  EURIBOR_1Y = "EURIBOR_1Y"  # EUR
  STIBOR_OVERNIGHT = "STIBOR_OVERNIGHT"  # SEK
  STIBOR_1W = "STIBOR_1W"  # SEK
  STIBOR_1M = "STIBOR_1M"  # SEK
  STIBOR_3M = "STIBOR_3M"  # SEK
  STIBOR_6M = "STIBOR_6M"  # SEK


@dataclasses.dataclass(frozen=True)
class CurveType:
  """"Rate curve types."""
  currency: currencies.CurrencyProtoType
  index_type: Index


__all__ = ["Index", "CurveType"]
