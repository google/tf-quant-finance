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
"""TF Quant Finance — JAX backend (migration in progress).

Submodules are imported individually; any still mid-conversion is skipped so the
package stays importable under JAX. As modules are ported they import cleanly.
TF is no longer required.
"""
import importlib
import logging

__all__ = [
    "black_scholes",
    "datetime",
    "experimental",
    "math",
    "models",
    "rates",
    "types",
    "utils",
]

_log = logging.getLogger("tf_quant_finance")

for _name in __all__:
    try:
        globals()[_name] = importlib.import_module(f"tf_quant_finance.{_name}")
    except Exception as e:  # submodule not yet JAX-ready
        _log.debug("tf_quant_finance.%s not yet available: %s", _name, e)
