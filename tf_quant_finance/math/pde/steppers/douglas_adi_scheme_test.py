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
"""Time marching schemes for finite difference methods for parabolic PDEs."""

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

douglas_adi_scheme = tff.math.pde.steppers.douglas_adi.douglas_adi_scheme


@test_util.run_all_in_graph_and_eager_modes
class DouglasAdiSchemeTest(tf.test.TestCase):

  def test_douglas_step_2d(self):
    u = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    d = np.arange(11, 27, dtype=np.float32).reshape(4, 4)
    dx = np.array([d, -3 * d, 2 * d])
    dy = np.array([2 * d, -6 * d, 4 * d])
    dxy = np.arange(-8, 8, dtype=np.float32).reshape(4, 4)
    bx = np.arange(2, 18, dtype=np.float32).reshape(4, 4)
    by = np.arange(5, 21, dtype=np.float32).reshape(4, 4)
    theta = 0.3

    def equation_params_fn(t):
      del t
      return ([[_tfconst(dy), _spread_mixed_term(_tfconst(dxy))],
               [None, _tfconst(dx)]],
              [_tfconst(by), _tfconst(bx)])

    scheme = douglas_adi_scheme(theta=theta)

    def pad_fn(t):
      paddings = tf.constant([[1, 1], [1, 1]])
      return tf.pad(t, paddings)

    actual = self.evaluate(
        scheme(value_grid=tf.constant(u, dtype=tf.float32), t1=0, t2=1,
               equation_params_fn=equation_params_fn,
               append_boundaries_fn=pad_fn,
               n_dims=2,
               has_default_lower_boundary=[False, False],
               has_default_upper_boundary=[False, False]))
    expected = self._simplified_douglas_step_2d(u, dx, dy, dxy, bx, by,
                                                0, 1, theta)
    self.assertLess(np.max(np.abs(expected - actual)), 0.01)

  def test_douglas_step_3d(self):
    u = np.arange(0, 80, dtype=np.float32).reshape(4, 4, 5)
    d = np.arange(10, 90, dtype=np.float32).reshape(4, 4, 5)
    dx = np.array([d, -3 * d, 2 * d])
    dy = 2 * dx
    dz = 3 * dx
    dxy = np.arange(-20, 60, dtype=np.float32).reshape(4, 4, 5)
    dyz = 2 * dxy
    dxz = 3 * dxy
    bx = np.arange(20, 100, dtype=np.float32).reshape(4, 4, 5)
    by = np.arange(30, 110, dtype=np.float32).reshape(4, 4, 5)
    bz = np.arange(40, 120, dtype=np.float32).reshape(4, 4, 5)
    theta = 0.3

    def equation_params_fn(t):
      del t
      dyz_spread = _spread_mixed_term(_tfconst(dyz))
      dxz_spread = _spread_mixed_term(_tfconst(dxz))
      dxy_spread = _spread_mixed_term(_tfconst(dxy))
      return ([[_tfconst(dz), dyz_spread, dxz_spread],
               [dyz_spread, _tfconst(dy), dxy_spread],
               [dxz_spread, dxy_spread, _tfconst(dx)]],
              [_tfconst(bz), _tfconst(by), _tfconst(bx)])

    scheme = douglas_adi_scheme(theta=theta)

    def pad_fn(t):
      paddings = tf.constant([[0, 1], [1, 0], [1, 0]])
      return tf.pad(t, paddings)

    actual = self.evaluate(
        scheme(value_grid=tf.constant(u, dtype=tf.float32), t1=0, t2=1,
               equation_params_fn=equation_params_fn,
               append_boundaries_fn=pad_fn, n_dims=3,
               # Tests default boundary as well
               has_default_lower_boundary=[True, False, False],
               has_default_upper_boundary=[False, True, True]))
    expected = self._simplified_douglas_step_3d(u, dx, dy, dz, dxy, dyz, dxz,
                                                bx, by, bz, 0, 1, theta)
    self.assertLess(np.max(np.abs(expected - actual)), 0.01)

  def _simplified_douglas_step_2d(self, u, dx, dy, dxy, bx, by, t1, t2,
                                  theta):
    # Simplified version of the step to test against: fixed number of
    # dimensions, np instead of tf, for loops, etc.
    dt = t2 - t1

    # u0 = (1 + A * dt) u + b * dt
    dx_contrib = (
        dx[0] * _np_shift(u, 1, -1) + dx[1] * u + dx[2] * _np_shift(u, 1, 1))
    dy_contrib = (
        dy[0] * _np_shift(u, 0, -1) + dy[1] * u + dy[2] * _np_shift(u, 0, 1))
    dxy_contrib = dxy * (
        _np_shift(_np_shift(u, 0, 1), 1, 1) - _np_shift(
            _np_shift(u, 0, 1), 1, -1) - _np_shift(_np_shift(u, 0, -1), 1, 1) +
        _np_shift(_np_shift(u, 0, -1), 1, -1))
    u0 = u + (dx_contrib + dy_contrib + dxy_contrib) * dt
    u0 += (bx + by) * dt

    # u1 = (1 - theta * dt * A_y)^(-1) (u0 - theta * dt * A_y u)
    theta *= dt  # Theta is always multiplied by dt.
    rhs = u0 - theta * (
        dy[0] * _np_shift(u, 0, -1) + dy[1] * u + dy[2] * _np_shift(u, 0, 1))

    u1 = np.zeros_like(u0)
    for i in range(u0.shape[1]):
      diags = np.array(
          [-theta * dy[0, :, i], 1 - theta * dy[1, :, i], -theta * dy[2, :, i]])
      u1[:, i] = self._np_tridiagonal_solve(diags, rhs[:, i])

    # u2 = (1 - theta * dt * A_x)^(-1) (u1 - theta * dt * A_x u)
    rhs = u1 - theta * (
        dx[0] * _np_shift(u, 1, -1) + dx[1] * u + dx[2] * _np_shift(u, 1, 1))
    u2 = np.zeros_like(u0)
    for i in range(u0.shape[0]):
      diags = np.array(
          [-theta * dx[0, i], 1 - theta * dx[1, i], -theta * dx[2, i]])
      u2[i] = self._np_tridiagonal_solve(diags, rhs[i])
    return u2

  def _np_tridiagonal_solve(self, diags, rhs):
    return self.evaluate(
        tf.linalg.tridiagonal_solve(
            tf.constant(diags, dtype=tf.float32),
            tf.constant(rhs, dtype=tf.float32)))

  def _simplified_douglas_step_3d(self, u, dx, dy, dz, dxy, dyz, dxz, bx,
                                  by, bz, t1, t2, theta):
    # Simplified version of the step to test against: fixed number of
    # dimensions, np instead of tf, for loops, etc.
    dt = t2 - t1

    # u0 = (1 + dt * A) u + b * dt
    def tridiag_contrib(tridiag_term, dim):
      return (tridiag_term[0] * _np_shift(u, dim, -1) + tridiag_term[1] * u +
              tridiag_term[2] * _np_shift(u, dim, 1)) * dt

    def mixed_term_contrib(mixed_term, dim1, dim2):
      return mixed_term * (_np_shift(_np_shift(u, dim1, 1), dim2, 1) -
                           _np_shift(_np_shift(u, dim1, 1), dim2, -1) -
                           _np_shift(_np_shift(u, dim1, -1), dim2, 1) +
                           _np_shift(_np_shift(u, dim1, -1), dim2, -1)) * dt

    u0 = (
        u + tridiag_contrib(dx, 2) + tridiag_contrib(dy, 1) +
        tridiag_contrib(dz, 0) + mixed_term_contrib(dxy, 2, 1) +
        mixed_term_contrib(dyz, 1, 0) + mixed_term_contrib(dxz, 2, 0))
    u0 += (bx + by + bz) * dt

    # u1 = (1 - theta * dt * A_z)^(-1) (u0 - theta * dt * A_z u)
    theta *= dt  # Theta is always multiplied by dt.
    rhs = u0 - theta * (
        dz[0] * _np_shift(u, 0, -1) + dz[1] * u + dz[2] * _np_shift(u, 0, 1))

    u1 = np.zeros_like(u0)
    for y in range(u0.shape[1]):
      for x in range(u0.shape[2]):
        diags = np.array([
            -theta * dz[0, :, y, x], 1 - theta * dz[1, :, y, x],
            -theta * dz[2, :, y, x]
        ])
        u1[:, y, x] = self._np_tridiagonal_solve(diags, rhs[:, y, x])

    # u2 = (1 - theta * dt * A_y)^(-1) (u1 - theta * dt * A_y u)
    rhs = u1 - theta * (
        dy[0] * _np_shift(u, 1, -1) + dy[1] * u + dy[2] * _np_shift(u, 1, 1))
    u2 = np.zeros_like(u0)
    for z in range(u0.shape[0]):
      for x in range(u0.shape[2]):
        diags = np.array([
            -theta * dy[0, z, :, x], 1 - theta * dy[1, z, :, x],
            -theta * dy[2, z, :, x]
        ])
        u2[z, :, x] = self._np_tridiagonal_solve(diags, rhs[z, :, x])

    # u3 = (1 - theta * dt * A_x)^(-1) (u2 - theta * dt * A_x u)
    rhs = u2 - theta * (
        dx[0] * _np_shift(u, 2, -1) + dx[1] * u + dx[2] * _np_shift(u, 2, 1))
    u3 = np.zeros_like(u0)
    for z in range(u0.shape[0]):
      for y in range(u0.shape[1]):
        diags = np.array([
            -theta * dx[0, z, y, :], 1 - theta * dx[1, z, y, :],
            -theta * dx[2, z, y, :]
        ])
        u3[z, y, :] = self._np_tridiagonal_solve(diags, rhs[z, y, :])

    return u3

  def test_douglas_step_with_batching(self):
    u = np.arange(0, 80, dtype=np.float32).reshape(4, 4, 5)
    d = np.arange(10, 90, dtype=np.float32).reshape(4, 4, 5)
    dx = np.array([d, -3 * d, 2 * d])
    dy = 2 * dx
    dxy = (np.arange(-20, 60, dtype=np.float32).reshape(4, 4, 5)) * 4
    theta = 0.3
    bx = np.arange(20, 100, dtype=np.float32).reshape(4, 4, 5)
    by = np.arange(30, 110, dtype=np.float32).reshape(4, 4, 5)

    def equation_params_fn(t):
      del t
      return ([[_tfconst(dy), _spread_mixed_term(_tfconst(dxy))],
               [None, _tfconst(dx)]],
              [_tfconst(by), _tfconst(bx)])

    scheme = douglas_adi_scheme(theta=theta)

    def pad_fn(t):
      paddings = tf.constant([[0, 0], [0, 1], [1, 0]])
      return tf.pad(t, paddings)

    actual = self.evaluate(
        scheme(value_grid=tf.constant(u, dtype=tf.float32), t1=0, t2=1,
               equation_params_fn=equation_params_fn,
               append_boundaries_fn=pad_fn, n_dims=2,
               # Tests default boundary as well
               has_default_lower_boundary=[True, False],
               has_default_upper_boundary=[False, True]))
    expected = np.zeros_like(u)
    for i in range(4):
      expected[i] = self._simplified_douglas_step_2d(u[i], dx[:, i], dy[:, i],
                                                     dxy[i], bx[i],
                                                     by[i], 0, 1, theta)

    self.assertLess(np.max(np.abs(expected - actual)), 0.01)


def _np_shift(values, axis, delta):
  values = np.roll(values, delta, axis)
  sl = [slice(None)] * values.ndim
  if delta > 0:
    sl[axis] = slice(None, delta)
  else:
    sl[axis] = slice(delta, None)
  values[tuple(sl)] = 0
  return values


def _tfconst(np_array):
  return tf.constant(np_array, dtype=tf.float32)


def _spread_mixed_term(term):
  # Turns non-diagonal element into a tuple of 4 elements representing 4
  # diagonal points.
  return term, -term, -term, term

if __name__ == '__main__':
  tf.test.main()
