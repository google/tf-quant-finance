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

# Lint as: python2, python3
"""Tests for cashflows module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_quant_finance.rates import cashflows as cashflows_lib
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class CashflowsTest(tf.test.TestCase):

  def test_pv_from_yields_no_group(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      yield_rate = 0.04
      coupon_rate = 0.04
      # Fifteen year bond with semi-annual coupons.
      cashflows = np.array(
          [coupon_rate * 500] * 29 + [1000 + coupon_rate * 500], dtype=dtype)
      times = np.linspace(0.5, 15, num=30).astype(dtype)
      expected_pv = 995.50315587
      actual_pv = self.evaluate(
          cashflows_lib.pv_from_yields(
              cashflows, times, [yield_rate], dtype=dtype))
      np.testing.assert_allclose(expected_pv, actual_pv)

  def test_pv_from_yields_grouped(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      yield_rates = [0.07, 0.05]
      # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
      cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                           dtype=dtype)
      times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
      groups = np.array([0] * 4 + [1] * 6)
      expected_pvs = np.array([942.71187528177757, 1025.7777300221542])
      actual_pvs = self.evaluate(
          cashflows_lib.pv_from_yields(
              cashflows, times, yield_rates, groups=groups, dtype=dtype))
      np.testing.assert_allclose(expected_pvs, actual_pvs)

  def test_pv_zero_yields(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      yield_rates = [0., 0.]
      # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
      cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                           dtype=dtype)
      times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
      groups = np.array([0] * 4 + [1] * 6)
      expected_pvs = np.array([1080., 1180.])
      actual_pvs = self.evaluate(
          cashflows_lib.pv_from_yields(
              cashflows, times, yield_rates, groups=groups, dtype=dtype))
      np.testing.assert_allclose(expected_pvs, actual_pvs)

  def test_pv_infinite_yields(self):
    """Tests in the limit of very large yields."""
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      yield_rates = [300., 300.]
      # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
      cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                           dtype=dtype)
      times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
      groups = np.array([0] * 4 + [1] * 6)
      expected_pvs = np.array([0., 0.])
      actual_pvs = self.evaluate(
          cashflows_lib.pv_from_yields(
              cashflows, times, yield_rates, groups=groups, dtype=dtype))
      np.testing.assert_allclose(expected_pvs, actual_pvs, atol=1e-9)

  def test_yields_from_pvs_no_group(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      coupon_rate = 0.04
      # Fifteen year bond with semi-annual coupons.
      cashflows = np.array(
          [coupon_rate * 500] * 29 + [1000 + coupon_rate * 500], dtype=dtype)
      pv = 995.50315587
      times = np.linspace(0.5, 15, num=30).astype(dtype)
      expected_yield_rate = 0.04
      actual_yield_rate = self.evaluate(
          cashflows_lib.yields_from_pv(cashflows, times, [pv], dtype=dtype))
      np.testing.assert_allclose(expected_yield_rate, actual_yield_rate)

  def test_yields_from_pv_grouped(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
      cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                           dtype=dtype)
      times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
      groups = np.array([0] * 4 + [1] * 6)
      pvs = np.array([942.71187528177757, 1025.7777300221542])
      expected_yield_rates = [0.07, 0.05]
      actual_yield_rates = self.evaluate(
          cashflows_lib.yields_from_pv(
              cashflows, times, pvs, groups=groups, dtype=dtype))
      np.testing.assert_allclose(
          expected_yield_rates, actual_yield_rates, atol=1e-7)

  def test_yield_saturated_pv(self):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
      cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                           dtype=dtype)
      times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
      groups = np.array([0] * 4 + [1] * 6)
      pvs = np.array([1080., 1180.])
      expected_yields = [0., 0.]
      actual_yields = self.evaluate(
          cashflows_lib.yields_from_pv(
              cashflows, times, pvs, groups=groups, dtype=dtype))
      np.testing.assert_allclose(expected_yields, actual_yields, atol=1e-9)

  def test_yield_small_pv(self):
    """Tests in the limit where implied yields are high."""
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
      cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                           dtype=dtype)
      times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
      groups = np.array([0] * 4 + [1] * 6)
      pvs = np.array([7.45333412e-05, 2.27476813e-08])
      expected_yields = [25.0, 42.0]
      actual_yields = self.evaluate(
          cashflows_lib.yields_from_pv(
              cashflows,
              times,
              pvs,
              groups=groups,
              dtype=dtype,
              max_iterations=30))
      np.testing.assert_allclose(expected_yields, actual_yields, atol=1e-9)


if __name__ == '__main__':
  tf.test.main()
