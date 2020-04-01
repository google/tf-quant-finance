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
"""Tests for Heston Price method."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from scipy import integrate
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

import math
import tf_quant_finance as tff


def get_heston_prices(kappa=None,
                      theta=None,
                      sigma=None,
                      rho=None,
                      v0=None,
                      forward=None,
                      expiry=None,
                      strike=None,
                      discount_factor=None,
                      ):
  """ Calculates Heston call and put prices using Attari paper.

  ## References

  [1] Mukarram Attari, Option Pricing Using Fourier Transforms: A Numerically
  Efficient Simplification
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=520042
  """
  def char_fun(u):
    d = (rho * sigma * u * 1j - kappa) ** 2 - sigma ** 2 * (-u * 1j - u ** 2)
    d = np.sqrt(d)
    s = rho * sigma * u * 1j
    g = (kappa - s - d) / (kappa - rho * sigma * u * 1j + d)
    a = kappa * theta
    h = g * np.exp(-d * expiry)
    m = 2 * np.log((1 - h) / (1 - g))
    C = (a / sigma ** 2) * ((kappa - s - d) * expiry - m)
    D = (kappa - s - d) / sigma ** 2 * ((1 - np.exp(-d * expiry)) / (1 - h))

    return np.exp(C + D * v0)

  def integrand_function(u, k):
    char_fun_value = char_fun(u)
    a = (char_fun_value.real + char_fun_value.imag / u) * np.cos(u * k)
    b = (char_fun_value.imag - char_fun_value.real / u) * np.sin(u * k)

    return (a + b) / (1.0 + u * u)

  k = np.log(strike / forward)

  integral = integrate.quad(
    lambda u: integrand_function(u, k),
    0,
    float("inf")
  )[0]

  undiscount_call_price = (forward - strike * (0.5 + 1 / math.pi * integral))
  undiscount_put_price = undiscount_call_price - forward + strike
  call_price = undiscount_call_price * discount_factor
  put_price = undiscount_put_price * discount_factor

  return call_price, put_price


@test_util.run_all_in_graph_and_eager_modes
class HestonPriceTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for Heston Price method."""

  # TODO: Need to test single precision (test wouldn't pass with simpsons
  #  integration)
  @parameterized.named_parameters(
    {
      'testcase_name': 'DoublePrecision',
      'dtype': np.float64
    })
  def test_heston_price(self, dtype):
    kappas = np.asarray([0.1, 10.0], dtype=dtype)
    thetas = np.asarray([0.1, 0.5], dtype=dtype)
    v0s = np.asarray([0.1, 0.5], dtype=dtype)
    forwards = np.asarray([10.0], dtype=dtype)
    sigmas = np.asarray([1.0], dtype=dtype)
    strikes = np.asarray([9.7, 10.0, 10.3], dtype=dtype)
    expiries = np.asarray([1.0], dtype=dtype)
    discount_factors = np.asarray([0.99], dtype=dtype)

    # TODO: test rhos = [-0.5, 0, 0.5]
    #  (test wouldn't pass with simpsons integration)
    rhos = np.asarray([0], dtype=dtype)

    for kappa in kappas:
      for theta in thetas:
        for sigma in sigmas:
          for rho in rhos:
            for v0 in v0s:
              for forward in forwards:
                for expiriy in expiries:
                  for strike in strikes:
                    for discount_factor in discount_factors:
                      tff_prices = self.evaluate(
                        tff.models.heston.approximations.eu_option_price(
                          kappas=np.asarray([kappa]),
                          thetas=np.asarray([theta]),
                          sigmas=np.asarray([sigma]),
                          rhos=np.asarray([rho]),
                          variances=np.asarray([v0]),
                          forwards=np.asarray([forward]),
                          expiries=np.asarray([expiriy]),
                          strikes=np.asarray([strike]),
                          discount_factors=np.asarray([discount_factor]),
                          is_call_options=np.asarray([True, False], dtype=np.bool)
                        ))

                      params = {
                        "kappa": kappa,
                        "theta": theta,
                        "sigma": sigma,
                        "rho": rho,
                        "v0": v0,
                        "forward": forward,
                        "expiry": expiriy,
                        "strike": strike,
                        "discount_factor": discount_factor,
                      }

                      target_call_price, target_put_price = get_heston_prices(**params)

                      # Normalize error in basis point
                      call_error = abs(tff_prices[0] - target_call_price) / forward
                      msg = "Found error = {0}bp".format(call_error * 1e4)
                      self.assertLess(call_error, 1e-5, msg)

                      put_error = abs(tff_prices[1] - target_put_price) / forward
                      msg = "Found error = {0}bp".format(put_error * 1e4)
                      self.assertLess(put_error, 1e-5, msg)


if __name__ == '__main__':
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  tf.test.main()