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

"""Tests for swap_curve."""

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class SwapCurveTest(tf.test.TestCase):

  def test_correctness(self):
    dtypes = [np.float64, np.float32]
    mats = [1., 2., 3., 5., 7., 10., 30.]
    par_swap_rates = [2.855, 3.097, 3.134, 3.152, 3.181, 3.23, 3.27]
    for dtype in dtypes:
      float_leg_start_times = [np.arange(0., x, 0.25, dtype) for x in mats]

      float_leg_end_times = [
          np.arange(0.25, x + 0.1, 0.25, dtype) for x in mats
      ]

      float_leg_dc = [
          np.array(np.repeat(0.25, len(x)), dtype=dtype)
          for x in float_leg_start_times
      ]

      fixed_leg_start_times = [np.arange(0., x, 0.5, dtype) for x in mats]

      fixed_leg_end_times = [np.arange(0.5, x + 0.1, 0.5, dtype) for x in mats]

      fixed_leg_dc = [
          np.array(np.repeat(0.5, len(x)), dtype=dtype)
          for x in fixed_leg_start_times
      ]

      fixed_leg_cashflows = [
          np.array(np.repeat(-y / 100., len(x)), dtype=dtype)
          for x, y in zip(fixed_leg_start_times, par_swap_rates)
      ]

      pvs = np.array(np.repeat(0., len(mats)), dtype=dtype)

      initial_curve_rates = np.array(np.repeat(0.01, len(mats)), dtype=dtype)

      results = self.evaluate(
          tff.rates.swap_curve_fit(
              float_leg_start_times,
              float_leg_end_times,
              float_leg_dc,
              fixed_leg_start_times,
              fixed_leg_end_times,
              fixed_leg_cashflows,
              fixed_leg_dc,
              pvs,
              dtype=dtype,
              initial_curve_rates=initial_curve_rates))

      np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                                                 30.0])

      self.assertFalse(results.failed)
      expected_discount_rates = np.array([
          0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
          0.03213901, 0.03257991
      ],
                                         dtype=dtype)
      np.testing.assert_allclose(
          results.rates, expected_discount_rates, atol=1e-6)

  def test_OIS_discounting(self):
    """Test the discouting of cashflows using a separate discounting curve."""
    dtypes = [np.float64, np.float32]
    mats = [1., 2., 3., 5., 7., 10., 30.]
    par_swap_rates = [2.855, 3.097, 3.134, 3.152, 3.181, 3.23, 3.27]

    expected_discount_rates = np.array([
        0.02844861, 0.03084989, 0.03121727, 0.0313961, 0.0316839, 0.03217002,
        0.03256696
    ],
                                       dtype=np.float64)

    for dtype in dtypes:
      float_leg_start_times = [np.arange(0., x, 0.25, dtype) for x in mats]

      float_leg_end_times = [
          np.arange(0.25, x + 0.1, 0.25, dtype) for x in mats
      ]

      float_leg_dc = [
          np.array(np.repeat(0.25, len(x)), dtype=dtype)
          for x in float_leg_start_times
      ]

      fixed_leg_start_times = [np.arange(0., x, 0.5, dtype) for x in mats]

      fixed_leg_end_times = [np.arange(0.5, x + 0.1, 0.5, dtype) for x in mats]

      fixed_leg_dc = [
          np.array(np.repeat(0.5, len(x)), dtype=dtype)
          for x in fixed_leg_start_times
      ]

      fixed_leg_cashflows = [
          np.array(np.repeat(-y / 100., len(x)), dtype=dtype)
          for x, y in zip(fixed_leg_start_times, par_swap_rates)
      ]

      pvs = np.array(np.repeat(0., len(mats)), dtype=dtype)

      discount_rates = np.array([0.0, 0.0, 0.0, 0.0], dtype=dtype)
      discount_times = np.array([1.0, 5.0, 10.0, 30.0], dtype=dtype)

      initial_curve_rates = np.array(np.repeat(0.01, len(mats)), dtype=dtype)

      results = self.evaluate(
          tff.rates.swap_curve_fit(
              float_leg_start_times,
              float_leg_end_times,
              float_leg_dc,
              fixed_leg_start_times,
              fixed_leg_end_times,
              fixed_leg_cashflows,
              fixed_leg_dc,
              pvs,
              float_leg_discount_rates=discount_rates,
              float_leg_discount_times=discount_times,
              dtype=dtype,
              initial_curve_rates=initial_curve_rates))

      np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                                                 30.0])

      self.assertFalse(results.failed)
      np.testing.assert_allclose(
          results.rates, expected_discount_rates, atol=1e-6)

  def test_interpolation_const_fwd(self):
    dtypes = [np.float64, np.float32]
    mats = [1., 2., 3., 5., 7., 10., 30.]
    par_swap_rates = [2.855, 3.097, 3.134, 3.152, 3.181, 3.23, 3.27]
    for dtype in dtypes:
      float_leg_start_times = [np.arange(0., x, 0.25, dtype) for x in mats]

      float_leg_end_times = [
          np.arange(0.25, x + 0.1, 0.25, dtype) for x in mats
      ]

      float_leg_dc = [
          np.array(np.repeat(0.25, len(x)), dtype=dtype)
          for x in float_leg_start_times
      ]

      fixed_leg_start_times = [np.arange(0., x, 0.5, dtype) for x in mats]

      fixed_leg_end_times = [np.arange(0.5, x + 0.1, 0.5, dtype) for x in mats]

      fixed_leg_dc = [
          np.array(np.repeat(0.5, len(x)), dtype=dtype)
          for x in fixed_leg_start_times
      ]

      fixed_leg_cashflows = [
          np.array(np.repeat(-y / 100., len(x)), dtype=dtype)
          for x, y in zip(fixed_leg_start_times, par_swap_rates)
      ]

      pvs = np.array(np.repeat(0., len(mats)), dtype=dtype)

      initial_curve_rates = np.array(np.repeat(0.01, len(mats)), dtype=dtype)
      curve_interpolator = lambda xi, x, y: tff.rates.constant_fwd.interpolate(
          xi, x, y, dtype=dtype)

      results = self.evaluate(
          tff.rates.swap_curve_fit(
              float_leg_start_times,
              float_leg_end_times,
              float_leg_dc,
              fixed_leg_start_times,
              fixed_leg_end_times,
              fixed_leg_cashflows,
              fixed_leg_dc,
              pvs,
              dtype=dtype,
              curve_interpolator=curve_interpolator,
              initial_curve_rates=initial_curve_rates))

      np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                                                 30.0])

      self.assertFalse(results.failed)
      expected_discount_rates = np.array([
          0.02834814, 0.03076991, 0.03113377, 0.03130508, 0.03160601,
          0.03213445, 0.0325467
      ],
                                         dtype=dtype)

      np.testing.assert_allclose(
          results.rates, expected_discount_rates, atol=1e-6)

  def test_settlement_times(self):
    dtypes = [np.float64, np.float32]
    mats = [1., 2., 3., 5., 7., 10., 30.]
    par_swap_rates = [2.855, 3.097, 3.134, 3.152, 3.181, 3.23, 3.27]
    for dtype in dtypes:
      float_leg_start_times = [np.arange(0., x, 0.25, dtype) for x in mats]

      float_leg_end_times = [
          np.arange(0.25, x + 0.1, 0.25, dtype) for x in mats
      ]

      float_leg_dc = [
          np.array(np.repeat(0.25, len(x)), dtype=dtype)
          for x in float_leg_start_times
      ]

      fixed_leg_start_times = [np.arange(0., x, 0.5, dtype) for x in mats]

      fixed_leg_end_times = [np.arange(0.5, x + 0.1, 0.5, dtype) for x in mats]

      fixed_leg_dc = [
          np.array(np.repeat(0.5, len(x)), dtype=dtype)
          for x in fixed_leg_start_times
      ]

      fixed_leg_cashflows = [
          np.array(np.repeat(-y / 100., len(x)), dtype=dtype)
          for x, y in zip(fixed_leg_start_times, par_swap_rates)
      ]

      discount_rates_float = np.array([0.01, 0.01, 0.01, 0.01], dtype=dtype)
      discount_times_float = np.array([1.0, 5.0, 10.0, 30.0], dtype=dtype)
      discount_rates_fixed = np.array([0.02, 0.02, 0.02, 0.02], dtype=dtype)
      discount_times_fixed = np.array([1.0, 5.0, 10.0, 30.0], dtype=dtype)

      pvs = np.array(np.repeat(0., len(mats)), dtype=dtype)

      settle_times = np.array(np.repeat(3./365., len(mats)), dtype=dtype)

      initial_curve_rates = np.array(np.repeat(0.01, len(mats)), dtype=dtype)

      results = self.evaluate(
          tff.rates.swap_curve_fit(
              float_leg_start_times,
              float_leg_end_times,
              float_leg_dc,
              fixed_leg_start_times,
              fixed_leg_end_times,
              fixed_leg_cashflows,
              fixed_leg_dc,
              pvs,
              float_leg_discount_rates=discount_rates_float,
              float_leg_discount_times=discount_times_float,
              fixed_leg_discount_rates=discount_rates_fixed,
              fixed_leg_discount_times=discount_times_fixed,
              present_values_settlement_times=settle_times,
              dtype=dtype,
              initial_curve_rates=initial_curve_rates))

      np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                                                 30.0])

      self.assertFalse(results.failed)
      expected_discount_rates = np.array([
          0.02820418, 0.03044715, 0.0306564, 0.03052609, 0.030508, 0.03053637,
          0.02808225
      ],
                                         dtype=dtype)

      np.testing.assert_allclose(
          results.rates, expected_discount_rates, atol=1e-6)

  def test_correctness_bootstrap(self):
    dtypes = [np.float64, np.float32]
    mats = [1., 2., 3., 5., 7., 10., 30.]
    par_swap_rates = [2.855, 3.097, 3.134, 3.152, 3.181, 3.23, 3.27]
    for dtype in dtypes:
      float_leg_start_times = [np.arange(0., x, 0.25, dtype) for x in mats]

      float_leg_end_times = [
          np.arange(0.25, x + 0.1, 0.25, dtype) for x in mats
      ]

      float_leg_dc = [
          np.array(np.repeat(0.25, len(x)), dtype=dtype)
          for x in float_leg_start_times
      ]

      fixed_leg_start_times = [np.arange(0., x, 0.5, dtype) for x in mats]

      fixed_leg_end_times = [np.arange(0.5, x + 0.1, 0.5, dtype) for x in mats]

      fixed_leg_dc = [
          np.array(np.repeat(0.5, len(x)), dtype=dtype)
          for x in fixed_leg_start_times
      ]

      fixed_leg_cashflows = [
          np.array(np.repeat(-y / 100., len(x)), dtype=dtype)
          for x, y in zip(fixed_leg_start_times, par_swap_rates)
      ]

      pvs = np.array(np.repeat(0., len(mats)), dtype=dtype)

      initial_curve_rates = np.array(np.repeat(0.01, len(mats)), dtype=dtype)

      results = self.evaluate(
          tff.rates.swap_curve_bootstrap(
              float_leg_start_times,
              float_leg_end_times,
              fixed_leg_start_times,
              fixed_leg_end_times,
              fixed_leg_cashflows,
              pvs,
              float_leg_daycount_fractions=float_leg_dc,
              fixed_leg_daycount_fractions=fixed_leg_dc,
              dtype=dtype,
              initial_curve_rates=initial_curve_rates))

      np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                                                 30.0])

      self.assertFalse(results.failed)
      expected_discount_rates = np.array([
          0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
          0.03213901, 0.03257991
      ],
                                         dtype=dtype)
      np.testing.assert_allclose(
          results.rates, expected_discount_rates, atol=1e-6)

  def test_OIS_discounting_bootstrap(self):
    """Test the discouting of cashflows using a separate discounting curve."""
    dtypes = [np.float64, np.float32]
    mats = [1., 2., 3., 5., 7., 10., 30.]
    par_swap_rates = [2.855, 3.097, 3.134, 3.152, 3.181, 3.23, 3.27]

    expected_discount_rates = np.array([
        0.02844861, 0.03084989, 0.03121727, 0.0313961, 0.0316839, 0.03217002,
        0.03256696
    ], dtype=np.float64)

    for dtype in dtypes:
      float_leg_start_times = [np.arange(0., x, 0.25, dtype) for x in mats]

      float_leg_end_times = [
          np.arange(0.25, x + 0.1, 0.25, dtype) for x in mats
      ]

      float_leg_dc = [
          np.array(np.repeat(0.25, len(x)), dtype=dtype)
          for x in float_leg_start_times
      ]

      fixed_leg_start_times = [np.arange(0., x, 0.5, dtype) for x in mats]

      fixed_leg_end_times = [np.arange(0.5, x + 0.1, 0.5, dtype) for x in mats]

      fixed_leg_dc = [
          np.array(np.repeat(0.5, len(x)), dtype=dtype)
          for x in fixed_leg_start_times
      ]

      fixed_leg_cashflows = [
          np.array(np.repeat(-y / 100., len(x)), dtype=dtype)
          for x, y in zip(fixed_leg_start_times, par_swap_rates)
      ]

      pvs = np.array(np.repeat(0., len(mats)), dtype=dtype)

      discount_rates = np.array([0.0, 0.0, 0.0, 0.0], dtype=dtype)
      discount_times = np.array([1.0, 5.0, 10.0, 30.0], dtype=dtype)

      initial_curve_rates = np.array(np.repeat(0.01, len(mats)), dtype=dtype)

      results = self.evaluate(
          tff.rates.swap_curve_bootstrap(
              float_leg_start_times,
              float_leg_end_times,
              fixed_leg_start_times,
              fixed_leg_end_times,
              fixed_leg_cashflows,
              pvs,
              float_leg_daycount_fractions=float_leg_dc,
              fixed_leg_daycount_fractions=fixed_leg_dc,
              float_leg_discount_rates=discount_rates,
              float_leg_discount_times=discount_times,
              dtype=dtype,
              initial_curve_rates=initial_curve_rates))

      np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                                                 30.0])

      self.assertFalse(results.failed)
      np.testing.assert_allclose(
          results.rates, expected_discount_rates, atol=1e-6)

  def test_missing_daycount_bootstrap(self):
    dtypes = [np.float64, np.float32]
    mats = [1., 2., 3., 5., 7., 10., 30.]
    par_swap_rates = [2.855, 3.097, 3.134, 3.152, 3.181, 3.23, 3.27]
    for dtype in dtypes:
      float_leg_start_times = [np.arange(0., x, 0.25, dtype) for x in mats]

      float_leg_end_times = [
          np.arange(0.25, x + 0.1, 0.25, dtype) for x in mats
      ]

      fixed_leg_start_times = [np.arange(0., x, 0.5, dtype) for x in mats]

      fixed_leg_end_times = [np.arange(0.5, x + 0.1, 0.5, dtype) for x in mats]

      fixed_leg_cashflows = [
          np.array(np.repeat(-y / 100., len(x)), dtype=dtype)
          for x, y in zip(fixed_leg_start_times, par_swap_rates)
      ]

      pvs = np.array(np.repeat(0., len(mats)), dtype=dtype)

      initial_curve_rates = np.array(np.repeat(0.01, len(mats)), dtype=dtype)

      results = self.evaluate(
          tff.rates.swap_curve_bootstrap(
              float_leg_start_times,
              float_leg_end_times,
              fixed_leg_start_times,
              fixed_leg_end_times,
              fixed_leg_cashflows,
              pvs,
              dtype=dtype,
              initial_curve_rates=initial_curve_rates))

      np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                                                 30.0])

      self.assertFalse(results.failed)
      expected_discount_rates = np.array([
          0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
          0.03213901, 0.03257991
      ],
                                         dtype=dtype)
      np.testing.assert_allclose(
          results.rates, expected_discount_rates, atol=1e-6)


if __name__ == '__main__':
  tf.test.main()
