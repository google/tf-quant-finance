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
"""Tests for market data utility functions."""

import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


framework = tff.experimental.pricing_platform.framework
DayCountConventions = framework.core.types.daycount_conventions.DayCountConventions
BusinessDayConvention = framework.core.types.business_days.BusinessDayConvention


@test_util.run_all_in_graph_and_eager_modes
class UtilsTest(tf.test.TestCase):

  def test_get_daycount_fn(self):
    for key in DayCountConventions:
      # Call function for all available daycount conventions
      if key != DayCountConventions.DAY_COUNT_CONVENTION_UNKNOWN:
        framework.market_data.utils.get_daycount_fn(key)

  def test_get_business_day_convention(self):
    for key in BusinessDayConvention:
      # Call function for all available daycount conventions
      if key != BusinessDayConvention.BUSINESS_DAY_CONVENTION_UNKNOWN:
        framework.market_data.utils.get_business_day_convention(key)


if __name__ == "__main__":
  tf.test.main()
