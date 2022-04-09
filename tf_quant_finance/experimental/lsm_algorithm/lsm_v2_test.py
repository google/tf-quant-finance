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

"""Tests for the regression Monte Carlo algorithm."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


lsm_algorithm = tff.experimental.lsm_algorithm

_SAMPLES = [[1.0, 1.09, 1.08, 1.34],
            [1.0, 1.16, 1.26, 1.54],
            [1.0, 1.22, 1.07, 1.03],
            [1.0, 0.93, 0.97, 0.92],
            [1.0, 1.11, 1.56, 1.52],
            [1.0, 0.76, 0.77, 0.90],
            [1.0, 0.92, 0.84, 1.01],
            [1.0, 0.88, 1.22, 1.34]]


@test_util.run_all_in_graph_and_eager_modes
class LsmTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Sets `samples` as in the Longstaff-Schwartz paper."""
    super(LsmTest, self).setUp()
    # See Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach.
    # Expand dims to reflect that `samples` represent sample paths of
    # a 1-dimensional process
    self.samples = np.expand_dims(_SAMPLES, -1)
    # Interest rates between exercise times
    interest_rates = [0.06, 0.06, 0.06]
    # Corresponding discount factors
    self.discount_factors = np.exp(-np.cumsum(interest_rates))

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_european_option_put(self, dtype):
    """Tests that LSM price of European put option is computed as expected."""
    # This is the same example as in Section 1 of
    # Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach. The review of financial
    # studies, 14(1), pp.113-147.
    basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
    payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1], dtype=dtype)
    # Option price
    european_put_price = lsm_algorithm.least_square_mc_v2(
        self.samples, [3], payoff_fn, basis_fn,
        discount_factors=[self.discount_factors[-1]], dtype=dtype)
    self.assertAllClose(european_put_price, [0.0564],
                        rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32,
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64,
      }, {
          'testcase_name': 'DoublePrecisionPassCalibrationSamples',
          'dtype': np.float64,
      })
  def test_american_option_put(self, dtype):
    """Tests that LSM price of American put option is computed as expected."""
    # This is the same example as in Section 1 of
    # Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach. The review of financial
    # studies, 14(1), pp.113-147.
    basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
    payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1], dtype=dtype)
    # Option price
    american_put_price = lsm_algorithm.least_square_mc_v2(
        self.samples, [1, 2, 3], payoff_fn, basis_fn,
        discount_factors=self.discount_factors,
        dtype=dtype)
    self.assertAllClose(american_put_price, [0.1144],
                        rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'num_calibration_samples': 4,
          'dtype': np.float32,
      }, {
          'testcase_name': 'DoublePrecision',
          'num_calibration_samples': 4,
          'dtype': np.float64,
      })
  def test_american_option_put_calibration(
      self, num_calibration_samples, dtype):
    """Tests that LSM price of American put option is computed as expected."""
    # This is the same example as in Section 1 of
    # Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach. The review of financial
    # studies, 14(1), pp.113-147.
    basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
    payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1], dtype=dtype)
    # Option price
    american_put_price = lsm_algorithm.least_square_mc_v2(
        self.samples, [1, 2, 3], payoff_fn, basis_fn,
        discount_factors=self.discount_factors,
        num_calibration_samples=num_calibration_samples,
        dtype=dtype)
    self.assertAllClose(american_put_price, [0.174226],
                        rtol=1e-4, atol=1e-4)

  def test_american_basket_option_put(self):
    """Tests the LSM price of American Basket put option."""
    # This is the same example as in Section 1 of
    # Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach. The review of financial
    # studies, 14(1), pp.113-147.
    # This is the minimum number of basis functions for the tests to pass.
    basis_fn = lsm_algorithm.make_polynomial_basis_v2(10)
    exercise_times = [1, 2, 3]
    dtype = np.float64
    payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1, 1.2, 1.3],
                                                     dtype=dtype)
    # Create a 2-d process which is simply follows the `samples` paths:
    samples = tf.convert_to_tensor(self.samples, dtype=dtype)
    samples_2d = tf.concat([samples, samples], -1)
    # Price American basket option
    american_basket_put_price = lsm_algorithm.least_square_mc_v2(
        samples_2d, exercise_times, payoff_fn, basis_fn,
        discount_factors=self.discount_factors, dtype=dtype)
    # Since the marginal processes of `samples_2d` are 100% correlated, the
    # price should be the same as of the American option computed for
    # `samples`
    american_put_price = lsm_algorithm.least_square_mc_v2(
        self.samples, exercise_times, payoff_fn, basis_fn,
        discount_factors=self.discount_factors, dtype=dtype)
    with self.subTest(name='Price'):
      self.assertAllClose(american_basket_put_price, american_put_price,
                          rtol=1e-4, atol=1e-4)
    with self.subTest(name='Shape'):
      self.assertAllEqual(american_basket_put_price.shape, [3])

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_american_option_put_batch_payoff(self, dtype):
    """Tests that LSM price of American put option is computed as expected."""
    # This is the same example as in Section 1 of
    # Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach. The review of financial
    # studies, 14(1), pp.113-147.
    basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
    payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1, 1.2], dtype=dtype)
    interest_rates = [[0.06, 0.06, 0.06],
                      [0.05, 0.05, 0.05]]
    discount_factors = np.exp(-np.cumsum(interest_rates, -1))
    discount_factors = np.expand_dims(discount_factors, 0)
    # Option price
    american_put_price = lsm_algorithm.least_square_mc_v2(
        self.samples, [1, 2, 3], payoff_fn, basis_fn,
        discount_factors=discount_factors, dtype=dtype)
    self.assertAllClose(american_put_price, [0.1144, 0.199],
                        rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_american_option_put_batch_samples(self, dtype):
    """Tests LSM price of a batch of American put options."""
    # This is the same example as in Section 1 of
    # Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach. The review of financial
    # studies, 14(1), pp.113-147.
    basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
    payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1, 1.2], dtype=dtype)
    interest_rates = [[0.06, 0.06, 0.06],
                      [0.05, 0.05, 0.05]]
    discount_factors = np.exp(-np.cumsum(interest_rates, -1))
    discount_factors = np.expand_dims(discount_factors, 0)
    # A batch of sample paths
    # Shape [num_samples, dum_times, dim]
    sample_paths1 = tf.convert_to_tensor(self.samples, dtype=dtype)
    # Shape [num_samples, dum_times, dim]
    sample_paths2 = sample_paths1 + 0.1
    # Shape [2, num_samples, dum_times, dim]
    sample_paths = tf.stack([sample_paths1, sample_paths2], axis=0)
    # Option price
    american_put_price = lsm_algorithm.least_square_mc_v2(
        sample_paths, [1, 2, 3], payoff_fn, basis_fn,
        discount_factors=discount_factors, dtype=dtype)
    self.assertAllClose(american_put_price, [0.1144, 0.1157],
                        rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
