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
"""Tests for Heston Price method."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class HestonPriceTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for Heston Price method."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64,
      },
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32,
      })
  def test_docstring(self, dtype):
    prices = tff.models.heston.approximations.european_option_price(
        variances=0.11,
        strikes=102.0,
        expiries=1.2,
        forwards=100.0,
        is_call_options=True,
        mean_reversion=2.0,
        theta=0.5,
        volvol=0.15,
        rho=0.3,
        discount_factors=1.0,
        dtype=dtype)
    # Computed using scipy
    self.assertAllClose(prices, 24.822196, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      {
          'testcase_name': 'DoublePrecisionUseForwards',
          'use_forwards': True,
          'dtype': np.float64,
      },
      {
          'testcase_name': 'DoublePrecisionUseSpots',
          'use_forwards': False,
          'dtype': np.float64,
      },
      {
          'testcase_name': 'SinglePrecisionUseForwards',
          'use_forwards': True,
          'dtype': np.float32,
      })
  def test_heston_price(self, dtype, use_forwards):
    mean_reversion = np.array([0.1, 10.0], dtype=dtype)
    theta = np.array([0.1, 0.5], dtype=dtype)
    variances = np.array([0.1, 0.5], dtype=dtype)
    discount_factors = np.array([0.99, 0.98], dtype=dtype)
    expiries = np.array([1.0], dtype=dtype)
    forwards = np.array([10.0], dtype=dtype)
    if use_forwards:
      spots = None
    else:
      spots = forwards * discount_factors
      forwards = None
    volvol = np.array([1.0, 0.9], dtype=dtype)
    strikes = np.array([9.7, 10.0], dtype=dtype)

    rho = np.array([0.5, 0.1], dtype=dtype)

    tff_prices = self.evaluate(
        tff.models.heston.approximations.european_option_price(
            mean_reversion=mean_reversion,
            theta=theta,
            volvol=volvol,
            rho=rho,
            variances=variances,
            forwards=forwards,
            spots=spots,
            expiries=expiries,
            strikes=strikes,
            discount_factors=discount_factors,
            is_call_options=np.asarray([True, False], dtype=np.bool)))
    # Computed using scipy
    scipy_prices = [1.07475678, 2.708217]
    np.testing.assert_allclose(
        tff_prices,
        scipy_prices, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
  tf.test.main()
