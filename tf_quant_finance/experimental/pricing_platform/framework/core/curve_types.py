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

from typing import Union
import dataclasses
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import rate_indices


def _init_currency(
    currency: Union[currencies.CurrencyProtoType, str]
    ) -> currencies.CurrencyProtoType:
  """Converts input to a currency object."""
  if isinstance(currency, str):
    try:
      return getattr(currencies.Currency, currency)
    except KeyError:
      raise ValueError(f"{currency} is not a valid currency")
  return currency


@dataclasses.dataclass
class RiskFreeCurve:
  """Risk free curve description."""
  currency: Union[currencies.CurrencyProtoType, str]

  def __post_init__(self):
    self.currency = _init_currency(self.currency)

  def __hash__(self):
    return hash((self.currency,))


@dataclasses.dataclass
class RateIndexCurve:
  """Rate index curve description."""
  currency: currencies.CurrencyProtoType
  index: rate_indices.RateIndex

  def __post_init__(self):
    self.currency = _init_currency(self.currency)

  def __hash__(self):
    return hash((self.currency, self.index.type))


CurveType = Union[RiskFreeCurve, RateIndexCurve]


__all__ = ["CurveType",
           "RiskFreeCurve",
           "RateIndexCurve"]
