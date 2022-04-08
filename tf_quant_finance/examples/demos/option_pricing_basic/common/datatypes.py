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
"""Common data types needed across multiple containers in this project."""

import dataclasses
import numpy as np


@dataclasses.dataclass
class OptionBatch:
  """Represents a batch of options."""

  strike: np.ndarray  # float64 array.
  call_put_flag: np.ndarray  # Boolean array. True if Call, False otherwise.
  expiry_date: np.ndarray  # int array containing date ordinals.
  trade_id: np.ndarray  # int32 array.
  underlier_id: np.ndarray  # int32 array. The identifier for the underlying.


@dataclasses.dataclass
class OptionMarketData:
  """Represents market data to be used to price the batch of options."""

  underlier_id: np.ndarray  # int32 array. The identifier for an underlying.
  spot: np.ndarray  # double array. The spot price of the underlier.
  volatility: np.ndarray  # double array. The volatility of the underlier.
  rate: np.ndarray  # double array. The risk free rate.


@dataclasses.dataclass
class ComputeData:
  """Carries a compute request from the downloader to the calculator."""

  market_data_path: str  # Path to the market data file to use.
  portfolio_path: str  # Path to the portfolio file to compute.
