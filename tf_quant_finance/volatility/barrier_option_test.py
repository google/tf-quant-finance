# Lint as: python3
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
"""Tests for pricing barrier options"""

import tensorflow as tf
from tensorflow.python.framework import test_util

import tf_quant_finance as tff

price_barrier_option = tff.volatility.barrier_option.price_barrier_option


@test_util.run_all_in_graph_and_eager_modes
class BarrierOptionTest(tf.test.TestCase):
  """Class Tests Barrier Options Pricing"""
  def test_price_barrier_option_1d(self):
    """Function tests barrier option pricing for scalar input"""
    asset_price = 100.0
    rebate = 3.0
    time_to_maturity = 0.5
    rate = 0.08
    b = 0.04
    asset_yield = -(b-rate)
    strike_price, barrier_price, price_true, mp = self.get_test_vals("cdo")
    volitility = 0.25
    """
    1 -> cdi
    2 -> pdi
    3 -> cui
    4 -> pui
    5 -> cdo
    6 -> pdo
    7 -> cuo
    8 -> puo
    """
    price = price_barrier_option(
        rate, asset_yield, asset_price, strike_price,
        barrier_price, rebate, volitility, time_to_maturity, mp)
    self.assertAllClose(price, price_true, 10e-3)

  def get_test_vals(self, param):
    """Function returns testing vals for type of option"""
    if param == "cdo":
      return 90, 95, 9.0246, 5
    if param == "cdi":
      return 90, 95, 7.7627, 1
    if param == "cuo":
      return 90, 105, 2.6789, 7
    if param == "cui":
      return 90, 105, 14.1112, 3
    if param == "pdo":
      return 90, 95, 2.2798, 6
    if param == "puo":
      return 90, 105, 3.7760, 8
    if param == "pdi":
      return 90, 95, 2.9586, 2
    if param == "pui":
      return 90, 105, 1.4653, 4

  def test_price_barrier_option_2d(self):
    """Function tests barrier option pricing for vector inputs"""
    asset_price = [100., 100., 100., 100., 100., 100., 100., 100.]
    rebate = [3., 3., 3., 3., 3., 3., 3., 3.]
    time_to_maturity = [.5, .5, .5, .5, .5, .5, .5, .5]
    rate = [.08, .08, .08, .08, .08, .08, .08, .08]
    volitility = [.25, .25, .25, .25, .25, .25, .25, .25]
    strike_price = [90., 90., 90., 90., 90., 90., 90., 90.]
    barrier_price = [95., 95., 105., 105., 95., 105., 95., 105.]
    price_true = [
        9.024, 7.7627, 2.6789, 14.1112, 2.2798, 3.7760, 2.95586, 1.4653]
    mp = [5, 1, 7, 3, 6, 8, 2, 4]
    asset_yield = [.04, .04, .04, .04, .04, .04, .04, .04]
    price = price_barrier_option(
        rate, asset_yield, asset_price, strike_price,
        barrier_price, rebate, volitility, time_to_maturity, mp)
    self.assertAllClose(price, price_true, 10e-3)

if __name__ == '__main__':
  tf.test.main()
