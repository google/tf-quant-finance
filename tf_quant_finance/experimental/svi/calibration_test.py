# Lint as: python3
# Copyright 2021 Google LLC
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
"""Tests for calibration.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class SviVolatilityCalibrationTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'conjugate_gradient_optimizer',
          'optimizer_fn': tff.math.optimizer.conjugate_gradient_minimize,
      }, {
          'testcase_name': 'bfgs_optimizer',
          'optimizer_fn': tfp.optimizer.bfgs_minimize,
      })
  def test_calibration_correctness(self, optimizer_fn):
    true_parameters = np.array([[0.04, 0.15, 0.5, 0.3, 0.3],
                                [0.02, 0.25, 0.4, 0.25, 0.25]])

    forwards = np.array([5., 6.])
    expiries = np.array([1., 1.])

    strikes = np.array([
        [
            0.24893534, 0.35810619, 0.51515402, 0.74107533, 1.06607467,
            1.53360279, 2.20616584, 3.17368209, 4.56550358, 6.56770979,
            9.44798554, 13.59140914
        ],
        [
            0.8120117, 1.16812025, 1.68040056, 2.41734193, 3.47746967,
            5.00251751, 7.19637661, 10.35235484, 14.89239051, 21.4234634,
            30.8187449, 44.33433659
        ],
    ])

    volatilities = np.array([
        [
            0.53809037, 0.51236712, 0.48535488, 0.45686258, 0.42666483,
            0.39451975, 0.3602712, 0.32435877, 0.29085295, 0.28834027,
            0.3644639, 0.45468296
        ],
        [
            0.60080077, 0.55414627, 0.50349337, 0.44780265, 0.38588647,
            0.31868425, 0.27922126, 0.38249619, 0.51199324, 0.62087246,
            0.71471493, 0.7980863
        ],
    ])

    (model_parameters, converged, _) = self.evaluate(
        tff.experimental.svi.calibrate(
            forwards=forwards,
            expiries=expiries,
            strikes=strikes,
            volatilities=volatilities,
            optimizer_fn=optimizer_fn))

    # Assert model convergence to expected parameters.
    self.assertTrue(converged.all())
    self.assertAllClose(model_parameters, true_parameters, atol=1e-4, rtol=1e-3)

  def test_calibration_correctness_bulk(self):
    np.random.seed(321)
    batch_size = 7
    num_strikes = 30
    forwards = 4. + 5. * np.random.random(size=batch_size)
    expiries = 0.5 + np.ones_like(forwards)
    log_moneyness = np.random.normal(size=(batch_size, num_strikes))
    strikes = forwards[:, np.newaxis] * np.exp(log_moneyness)

    svi_a = 0.1 + 0.3 * np.random.random(size=batch_size)
    svi_b = 0.3 * np.random.random(size=batch_size)
    svi_rho = np.random.random(size=batch_size) - 0.5
    svi_m = 0.5 * np.random.normal(size=batch_size)
    svi_sigma = 0.1 + 0.5 * np.random.random(size=batch_size)
    true_parameters = np.transpose([svi_a, svi_b, svi_rho, svi_m, svi_sigma])

    target_variance = tff.experimental.svi.total_variance_from_raw(
        true_parameters, log_moneyness)
    target_volatilities = tf.sqrt(target_variance / expiries[:, None])

    (model_parameters, converged, _) = self.evaluate(
        tff.experimental.svi.calibrate(
            forwards=forwards,
            expiries=expiries,
            strikes=strikes,
            volatilities=target_volatilities,
            optimizer_fn=tfp.optimizer.bfgs_minimize))

    # Assert model convergence to expected parameters.
    self.assertTrue(converged.all())
    self.assertAllClose(model_parameters, true_parameters, atol=1e-3, rtol=1e-2)

  def test_real_market_data(self):
    # Loosely based on prices for GOOG call options (GOOG210820C*) provided by
    # https://finance.yahoo.com/quote/GOOG/options?date=1629417600

    forwards = np.array([2402.])
    expiries = np.array([0.23])
    strikes = np.array([[
        1700., 1800., 1900., 2000., 2050., 2100., 2200., 2250., 2350., 2400.,
        2450., 2500., 2550., 2600., 2650., 2700., 2750., 2800., 2850., 2900.,
        2950., 3000.
    ]])
    volatilities = np.array([[
        0.5335, 0.4882, 0.4389, 0.3937, 0.3749, 0.3569, 0.3259, 0.3135, 0.29,
        0.283, 0.2717, 0.2667, 0.2592, 0.2566, 0.2564, 0.2574, 0.2595, 0.2621,
        0.2669, 0.2732, 0.2826, 0.2967
    ]])

    tolerance = 1e-4
    (model_parameters, converged, _) = self.evaluate(
        tff.experimental.svi.calibrate(
            forwards=forwards,
            expiries=expiries,
            strikes=strikes,
            volatilities=volatilities,
            tolerance=tolerance))

    # Assert model convergence to expected parameters.
    expected_parameters = np.array([[-0.30, 0.42, 0.04, 0.12, 0.75]])
    self.assertTrue(converged.all())
    self.assertAllClose(
        model_parameters, expected_parameters, atol=0.01, rtol=0.3)


if __name__ == '__main__':
  tf.test.main()
