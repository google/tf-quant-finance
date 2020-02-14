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

"""Implementations of finite difference methods."""

from autograd import elementwise_grad as egrad
import autograd.numpy as np
import pandas as pd
import sobol_seq


def rosenbrock(x):
  """Rosenbrock function: test function for evaluating algorithms."""
  x = np.array(x)
  x_curr, x_next = x[..., :-1], x[..., 1:]
  terms = 100 * np.square(x_next - np.square(x_curr)) + np.square(1 - x_curr)
  return np.sum(terms, axis=-1)


def grid(num, ndim, large=False):
  """Build a uniform grid with num points along each of ndim axes."""
  if not large:
    _check_not_too_large(np.power(num, ndim) * ndim)
  x = np.linspace(0, 1, num, dtype='float64')
  w = 1 / (num - 1)
  points = np.stack(
      np.meshgrid(*[x for _ in range(ndim)], indexing='ij'), axis=-1)
  return points, w


def non_uniform_grid(num, ndim, skip=42, large=False):
  """Build a non-uniform grid with num points of ndim dimensions."""
  if not large:
    _check_not_too_large(num * ndim)
  return sobol_seq.i4_sobol_generate(ndim, num, skip=skip)


def autograd(f, ds, points):
  """Evaluate derivatives of f on the given points."""
  df_ds = lambda *args: f(np.stack(args, axis=-1))
  for i in ds:
    df_ds = egrad(df_ds, i)
  ndim = points.shape[-1]
  return df_ds(*[points[..., i] for i in range(ndim)])


def central(f0, ds, w):
  """Apply central difference method to estimate derivatives."""
  f = lambda o: shift(f0, o)
  eye = np.eye(f0.ndim, dtype=int)
  offsets = [-eye[d] for d in ds]

  if not ds:
    return f0
  elif len(ds) == 1:  # First order derivatives.
    i = offsets[0]
    return (f(i) - f(-i)) / (2 * w)
  elif len(ds) == 2:  # Second order derivatives.
    i, j = offsets
    w2 = np.square(w)
    if ds[0] == ds[1]:  # d^2/dxdx
      return (f(i) - 2 * f0 + f(-i)) / w2
    else:  # d^2/dxdy
      return (f(i + j) - f(i - j) - f(j - i) + f(-i - j)) / (4 * w2)
  else:
    raise NotImplementedError(ds)


def triangular(n):
  """Compute the n-th triangular number."""
  return np.floor_divide(n * (n + 1), 2)


def derivative_names(ndim):
  """Iterate over derivative speficiations and their names."""
  # Note: len(list(derivative_names(ndim)) == triangular(ndim + 1).
  yield (), 'f'  # Function value.
  for i in range(ndim):
    yield (i,), 'df/d%i' % i  # First derivative along an axis.
  for i in range(ndim):
    yield (i, i), 'd^2f/d%i^2' % i  # Second derivative along an axis.
  for i, j in zip(*np.triu_indices(ndim, k=1)):
    # Second derivarive along mixed axes.
    yield (int(i), int(j)), 'd^2f/(d%i d%i)' % (i, j)


def taylor_approx(target, stencil, values):
  """Use taylor series to approximate up to second order derivatives.

  Args:
    target: An array of shape (..., n), a batch of n-dimensional points
      where one wants to approximate function value and derivatives.
    stencil: An array of shape broadcastable to (..., k, n), for each target
      point a set of k = triangle(n + 1) points to use on its approximation.
    values: An array of shape broadcastable to (..., k), the function value at
      each of the stencil points.

  Returns:
    An array of shape (..., k), for each target point the approximated
    function value, gradient and hessian evaluated at that point (flattened
    and in the same order as returned by derivative_names).
  """
  # Broadcast arrays to their required shape.
  batch_shape, ndim = target.shape[:-1], target.shape[-1]
  stencil = np.broadcast_to(stencil, batch_shape + (triangular(ndim + 1), ndim))
  values = np.broadcast_to(values, stencil.shape[:-1])

  # Subtract target from each stencil point.
  delta_x = stencil - np.expand_dims(target, axis=-2)
  delta_xy = np.matmul(
      np.expand_dims(delta_x, axis=-1), np.expand_dims(delta_x, axis=-2))
  i = np.arange(ndim)
  j, k = np.triu_indices(ndim, k=1)

  # Build coefficients for the Taylor series equations, namely:
  #   f(stencil) = coeffs @ [f(target), df/d0(target), ...]
  coeffs = np.concatenate([
      np.ones(delta_x.shape[:-1] + (1,)),  # f(target)
      delta_x,  # df/di(target)
      delta_xy[..., i, i] / 2,  # d^2f/di^2(target)
      delta_xy[..., j, k],  # d^2f/{dj dk}(target)
  ], axis=-1)

  # Then: [f(target), df/d0(target), ...] = coeffs^{-1} @ f(stencil)
  return np.squeeze(
      np.matmul(np.linalg.inv(coeffs), values[..., np.newaxis]), axis=-1)


def non_uniform_approx_nearest(points, values):
  """Approximate derivatives using nearest points in non-uniform grid."""
  ndim = points.shape[-1]
  k = triangular(ndim + 1)
  diffs = np.expand_dims(points, axis=0) - np.expand_dims(points, axis=1)
  norms = np.linalg.norm(diffs, axis=-1)
  nearest_k = np.argpartition(norms, k)[..., :k]
  return taylor_approx(points, points[nearest_k], values[nearest_k])


def central_errors(f, num, ndim, label=None):
  """Build DataFrame of approximation errors with central differences method."""
  points, w = grid(num, ndim)
  values = f(points)

  def name_errors():
    for ds, name in derivative_names(ndim):
      actual = autograd(f, ds, points)
      approx = central(values, ds, w)
      yield name, np.abs(actual - approx)

  return _build_errors_df(name_errors(), label)


def non_uniform_errors(f, num, ndim, label=None):
  """Build DataFrame of approximation errors with non uniform grid."""
  points = non_uniform_grid(np.power(num, ndim), ndim)
  values = f(points)
  approx_all = non_uniform_approx_nearest(points, values)

  def name_errors():
    for (ds, name), approx in zip(derivative_names(ndim), approx_all.T):
      actual = autograd(f, ds, points)
      yield name, np.abs(actual - approx)

  return _build_errors_df(name_errors(), label)


def _build_errors_df(name_errors, label):
  """Helper to build errors DataFrame."""
  series = []
  percentiles = np.linspace(0, 100, 21)
  index = percentiles / 100
  for name, errors in name_errors:
    series.append(pd.Series(
        np.nanpercentile(errors, q=percentiles), index=index, name=name))
  df = pd.concat(series, axis=1)
  df.columns.name = 'derivative'
  df.index.name = 'quantile'
  df = df.stack().rename('error').reset_index()
  with np.errstate(divide='ignore'):
    df['log(error)'] = np.log(df['error'])
  if label is not None:
    df['label'] = label
  return df


def shift(x, offsets):
  """Similar to np.roll, but fills with nan instead of rolling values over.

  Also shifts along multiple axes at the same time.

  Args:
    x: The input array to shift.
    offsets: How much to shift each axis, offsets[i] is the offset for the i-th
      axis.

  Returns:
    An array with same shape as the input, with specified shifts applied.
  """
  def to_slice(offset):
    return slice(offset, None) if offset >= 0 else slice(None, offset)

  out = np.empty_like(x)
  out.fill(np.nan)
  ind_src = tuple(to_slice(-o) for o in offsets)
  ind_dst = tuple(to_slice(o) for o in offsets)
  out[ind_dst] = x[ind_src]
  return out


def _check_not_too_large(num_values):
  if num_values > 10e6:
    raise ValueError('Attempting to create an array with more than 10M values')
