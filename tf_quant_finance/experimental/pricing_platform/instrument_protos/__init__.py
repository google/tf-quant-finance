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
"""Instrument protos module."""

from tf_quant_finance.experimental.pricing_platform.instrument_protos import all_instruments_pb2 as instruments
from tf_quant_finance.experimental.pricing_platform.instrument_protos import american_equity_option_pb2 as american_equity_option
from tf_quant_finance.experimental.pricing_platform.instrument_protos import business_days_pb2 as business_days
from tf_quant_finance.experimental.pricing_platform.instrument_protos import currencies_pb2 as currencies
from tf_quant_finance.experimental.pricing_platform.instrument_protos import date_pb2 as date
from tf_quant_finance.experimental.pricing_platform.instrument_protos import daycount_conventions_pb2 as daycount_conventions
from tf_quant_finance.experimental.pricing_platform.instrument_protos import decimal_pb2 as decimal
from tf_quant_finance.experimental.pricing_platform.instrument_protos import forward_rate_agreement_pb2 as forward_rate_agreement
from tf_quant_finance.experimental.pricing_platform.instrument_protos import interest_rate_swap_pb2 as interest_rate_swap
from tf_quant_finance.experimental.pricing_platform.instrument_protos import metadata_pb2 as metadata
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2 as period
from tf_quant_finance.experimental.pricing_platform.instrument_protos import rate_indices_pb2 as rate_indices
from tf_quant_finance.experimental.pricing_platform.instrument_protos import swaption_pb2 as swaption

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import


Instrument = instruments.Instrument

_allowed_symbols = [
    "Instrument",
    "american_equity_option",
    "business_days",
    "currencies",
    "date",
    "daycount_conventions",
    "decimal",
    "forward_rate_agreement",
    "interest_rate_swap",
    "metadata",
    "period",
    "rate_indices",
    "swaption",

]

remove_undocumented(__name__, _allowed_symbols)
