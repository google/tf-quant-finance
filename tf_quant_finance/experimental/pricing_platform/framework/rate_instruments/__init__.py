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
"""Rate instruments module."""

from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import cashflow_streams
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import coupon_specs
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.forward_rate_agreement import forward_rate_agreement
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.interest_rate_swap import interest_rate_swap

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    "cashflow_streams",
    "coupon_specs",
    "forward_rate_agreement",
    "interest_rate_swap",
    "utils",
]

remove_undocumented(__name__, _allowed_symbols)
