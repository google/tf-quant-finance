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
"""Tests for swap_curve."""

import numpy as np
import tensorflow as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class SwapCurveTest(tf.test.TestCase):

  def test_correctness(self):
    dtypes = [np.float64, np.float32]
    mats = [1., 2., 3., 5., 7., 10., 30.]
    par_swap_rates = [2.855, 3.097, 3.134, 3.152, 3.181, 3.23, 3.27]
    for dtype in dtypes:
      float_leg_start_times = list(
          map(lambda x: np.arange(0., x, 0.25, dtype), mats))

      float_leg_end_times = list(
          map(lambda x: np.arange(0.25, x + 0.1, 0.25, dtype), mats))

      float_leg_dc = list(
          map(lambda x: np.array(np.repeat(0.25, len(x)), dtype=dtype),
              float_leg_start_times))

      fixed_leg_start_times = list(
          map(lambda x: np.arange(0., x, 0.5, dtype), mats))

      fixed_leg_end_times = list(
          map(lambda x: np.arange(0.5, x + 0.1, 0.5, dtype), mats))

      fixed_leg_dc = list(
          map(lambda x: np.array(np.repeat(0.5, len(x)), dtype=dtype),
              fixed_leg_start_times))

      fixed_leg_cashflows = list(
          map(lambda x, y: np.array(np.repeat(-y / 100., len(x)), dtype=dtype),
              fixed_leg_start_times, par_swap_rates))

      pvs = np.array(np.repeat(0., len(mats)), dtype=dtype)

      initial_curve_rates = np.array(np.repeat(0.01, len(mats)), dtype=dtype)

      results = self.evaluate(
          tff.rates.hagan_west.swap_curve(
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
          results.discount_rates, expected_discount_rates, atol=1e-6)

  def test_OIS_discounting(self):
    """Test the discouting of cashflows using a separate discounting curve."""
    dtypes = [np.float64, np.float32]
    mats = [1., 2., 3., 5., 7., 10., 30.]
    par_swap_rates = [2.855, 3.097, 3.134, 3.152, 3.181, 3.23, 3.27]

    expected_discount_rates = np.array([
                0.02844861, 0.03084989, 0.03121727, 0.0313961, 0.0316839,
                0.03217002, 0.03256696
            ],
                     dtype=np.float64)

    for dtype in dtypes:
      float_leg_start_times = list(map(lambda x: np.arange(0., x, 0.25,
                                                           dtype), mats))

      float_leg_end_times = list(map(lambda x: np.arange(0.25, x+0.1, 0.25,
                                                         dtype), mats))

      float_leg_dc = list(
          map(lambda x: np.array(np.repeat(0.25, len(x)), dtype=dtype),
              float_leg_start_times))

      fixed_leg_start_times = list(map(lambda x: np.arange(0., x, 0.5,
                                                           dtype), mats))

      fixed_leg_end_times = list(map(lambda x: np.arange(0.5, x+0.1, 0.5,
                                                         dtype), mats))

      fixed_leg_dc = list(
          map(lambda x: np.array(np.repeat(0.5, len(x)), dtype=dtype),
              fixed_leg_start_times))

      fixed_leg_cashflows = list(
          map(lambda x, y: np.array(np.repeat(-y / 100., len(x)), dtype=dtype),
              fixed_leg_start_times, par_swap_rates))

      pvs = np.array(np.repeat(0., len(mats)), dtype=dtype)

      discount_rates = np.array([0.0, 0.0, 0.0, 0.0], dtype=dtype)
      discount_times = np.array([1.0, 5.0, 10.0, 30.0], dtype=dtype)

      initial_curve_rates = np.array(np.repeat(0.01, len(mats)), dtype=dtype)

      results = self.evaluate(
          tff.rates.hagan_west.swap_curve(
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
          results.discount_rates, expected_discount_rates, atol=1e-6)


if __name__ == '__main__':
  tf.test.main()
