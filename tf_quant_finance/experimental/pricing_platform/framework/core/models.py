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
"""Supported pricing models."""

import enum


class InterestRateModelType(enum.Enum):
  """Models for pricing interest rate derivatives.

  LOGNORMAL_RATE: Lognormal model for the underlying rate.
  NORMAL_RATE: Normal model for the underlying rate
  LOGNORMAL_SMILE_CONSISTENT_REPLICATION: Smile consistent replication
    (lognormal vols).
  NORMAL_SMILE_CONSISTENT_REPLICATION: Smile consistent replication
    (normal vols).
  """
  LOGNORMAL_RATE = 1

  NORMAL_RATE = 2

  LOGNORMAL_SMILE_CONSISTENT_REPLICATION = 3

  NORMAL_SMILE_CONSISTENT_REPLICATION = 4


__all__ = ["InterestRateModelType"]
