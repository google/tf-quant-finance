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
"""Collection of functions to compute properties of forwards."""

from tf_quant_finance.rates.analytics import forwards
# ponytail: was tf.deprecation.deprecated_alias(...) wrappers; dropped the
# deprecation indirection, kept the function aliases.
forward_rates_from_yields = forwards.forward_rates_from_yields
yields_from_forward_rates = forwards.yields_from_forward_rates

__all__ = ['forward_rates_from_yields', 'yields_from_forward_rates']
