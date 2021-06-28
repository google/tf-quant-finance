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

"""Quasi Monte Carlo support: Sobol sequence.

A TensorFlow implementation of Sobol sequences, a type of quasi-random
low-discrepancy sequence: https://en.wikipedia.org/wiki/Sobol_sequence.
"""

import logging
import os

import numpy as np
from six.moves import range
import tensorflow.compat.v2 as tf

from tf_quant_finance import types


__all__ = [
    'sample'
]


_LN_2 = np.log(2.)


def sample(dim: int,
           num_results: types.IntTensor,
           skip: types.IntTensor = 0,
           validate_args: bool = False,
           dtype: tf.DType = None,
           name=None) -> types.RealTensor:
  """Returns num_results samples from the Sobol sequence of dimension dim.

  Uses the original ordering of points, not the more commonly used Gray code
  ordering. Derived from notes by Joe & Kuo[1].

  [1] describes bitwise operations on binary floats. The implementation below
  achieves this by transforming the floats into ints, being careful to align
  the digits so the bitwise operations are correct, then transforming back to
  floats.

  Args:
    dim: Positive Python `int` representing each sample's `event_size.`
    num_results: Positive scalar `Tensor` of dtype int32. The number of Sobol
      points to return in the output.
    skip: Positive scalar `Tensor` of dtype int32. The number of initial points
      of the Sobol sequence to skip.
    validate_args: Python `bool`. When `True`, input `Tensor's` are checked for
      validity despite possibly degrading runtime performance. The checks verify
      that `dim >= 1`, `num_results >= 1`, and `skip >= 0`. When `False` invalid
      inputs may silently render incorrect outputs.
      Default value: False.
    dtype: Optional `dtype`. The dtype of the output `Tensor` (either `float32`
      or `float64`).
      Default value: `None` which maps to the `float32`.
    name: Python `str` name prefixed to ops created by this function.

  Returns:
    `Tensor` of samples from Sobol sequence with `shape` [n, dim].

  #### References

  [1]: S. Joe and F. Y. Kuo. Notes on generating Sobol sequences. August 2008.
       https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf
  """
  with tf.compat.v1.name_scope(name, 'sobol_sample', [dim, num_results, skip]):
    num_results = tf.convert_to_tensor(
        num_results, dtype=tf.int32, name='num_results')
    skip = tf.convert_to_tensor(skip, dtype=tf.int32, name='skip')
    control_dependencies = []
    if validate_args:
      if dim < 1:
        raise ValueError(
            'Dimension must be greater than zero. Supplied {}'.format(dim))
      control_dependencies.append(tf.compat.v1.debugging.assert_greater(
          num_results, 0,
          message='Number of results `num_results` must be greater than zero.'))
      control_dependencies.append(tf.compat.v1.debugging.assert_greater(
          skip, 0,
          message='`skip` must be non-negative.'))

    with tf.compat.v1.control_dependencies(control_dependencies):
      if validate_args:
        # The following tf.maximum ensures that the graph can be executed
        # even with ignored control dependencies
        num_results = tf.maximum(num_results, 1, name='fix_num_results')
        skip = tf.maximum(skip, 0, name='fix_skip')
      direction_numbers = tf.convert_to_tensor(
          _compute_direction_numbers(dim), name='direction_numbers')
      # Number of digits actually needed for binary representation.
      num_digits = tf.cast(
          tf.math.ceil(
              tf.math.log(tf.cast(skip + num_results + 1, tf.float64)) / _LN_2),
          tf.int32)
    # Direction numbers, reshaped and with the digits shifted as needed for the
    # bitwise xor operations below. Note that here and elsewhere we use bit
    # shifts rather than powers of two because exponentiating integers is not
    # currently supported on GPUs.
    direction_numbers = tf.bitwise.left_shift(
        direction_numbers[:dim, :num_digits], tf.range(num_digits - 1, -1, -1))
    direction_numbers = tf.expand_dims(tf.transpose(a=direction_numbers), 1)

    # Build the binary matrix corresponding to the i values in Joe & Kuo[1]. It
    # is a matrix of the binary representation of the numbers (skip+1, skip+2,
    # ..., skip+num_results). For example, with skip=0 and num_results=6 the
    # binary matrix (before expanding the dimension) looks like this:
    # [[1 0 1 0 1 0]
    #  [0 1 1 0 0 1]
    #  [0 0 0 1 1 1]]
    irange = tf.range(skip + 1, skip + num_results + 1)
    dig_range = tf.expand_dims(tf.range(num_digits), 1)
    binary_matrix = tf.bitwise.bitwise_and(
        1, tf.bitwise.right_shift(irange, dig_range))
    binary_matrix = tf.expand_dims(binary_matrix, -1)

    # Multiply and bitwise-xor everything together. We use while_loop rather
    # than foldl(bitwise_xor(...)) because the latter is not currently supported
    # on GPUs.
    product = direction_numbers * binary_matrix

    def _cond(partial_result, i):
      del partial_result
      return i < num_digits

    def _body(partial_result, i):
      return tf.bitwise.bitwise_xor(partial_result, product[i, :, :]), i + 1

    result, _ = tf.while_loop(_cond, _body, (product[0, :, :], 1))
    # Shift back from integers to floats.
    dtype = dtype or tf.float32
    return (tf.cast(result, dtype)
            / tf.cast(tf.bitwise.left_shift(1, num_digits), dtype))


# TODO(b/135590027): Add option to store these instead of recomputing each time.
def _compute_direction_numbers(dim):
  """Returns array of direction numbers for dimension dim.

  These are the m_kj values in the Joe & Kuo notes[1], not the v_kj values. So
  these refer to the 'abuse of notation' mentioned in the notes -- it is a
  matrix of integers, not floats. The variable names below are intended to match
  the notation in the notes as closely as possible.

  Args:
    dim: int, dimension.

  Returns:
    `numpy.array` of direction numbers with `shape` [dim, 32].
  """
  m = np.empty((dim, 32), dtype=np.int32)
  m[0, :] = np.ones(32, dtype=np.int32)
  for k in range(dim - 1):
    a_k = _PRIMITIVE_POLYNOMIAL_COEFFICIENTS[k]
    deg = np.int32(np.floor(np.log2(a_k)))  # degree of polynomial
    m[k + 1, :deg] = _INITIAL_DIRECTION_NUMBERS[:deg, k]
    for j in range(deg, 32):
      m[k + 1, j] = m[k + 1, j - deg]
      for i in range(deg):
        if (a_k >> i) & 1:
          m[k + 1, j] = np.bitwise_xor(m[k + 1, j], m[k + 1, j - deg + i] <<
                                       (deg - i))
  return m


def _get_sobol_data_path():
  """Returns path of file 'new-joe-kuo-6.21201'.

     Location of file 'new-joe-kuo-6.21201' depends on the environment in
     which this code is executed. In Google internal environment file
     'new-joe-kuo-6.21201' is accessible using the
     'third_party/sobol_data/new-joe-kuo-6.21201' file path.

     However, this doesn't work in the pip package. In pip package the directory
     'third_party' is a subdirectory of directory 'tf_quant_finance' and in
     this case we construct a file path relative to the __file__ file path.

     If this library is installed in editable mode with `pip install -e .`, then
     the directory 'third_party` will be at the top level of the repository
     and need to search a level further up relative to __file__.
  """
  filename = 'new-joe-kuo-6.21201'
  path1 = os.path.join('third_party', 'sobol_data', filename)
  path2 = os.path.abspath(
      os.path.join(
          os.path.dirname(__file__), '..', '..', '..',
          'third_party', 'sobol_data', filename))
  path3 = os.path.abspath(
      os.path.join(
          os.path.dirname(__file__), '..', '..', '..', '..',
          'third_party', 'sobol_data', filename))

  paths = [path1, path2, path3]
  for path in paths:
    if os.path.exists(path):
      return path


def _load_sobol_data():
  """Parses file 'new-joe-kuo-6.21201'."""
  path = _get_sobol_data_path()
  if path is None:
    logging.warning('Unable to find path to sobol data file.')
    return NotImplemented, NotImplemented
  header_line = True
  # Primitive polynomial coefficients.
  polynomial_coefficients = np.zeros(shape=(21200,), dtype=np.int64)
  # Initial direction numbers.
  direction_numbers = np.zeros(shape=(18, 21200), dtype=np.int64)
  index = 0
  with open(path) as f:
    for line in f:
      # Skip first line (header).
      if header_line:
        header_line = False
        continue
      tokens = line.split()
      s, a = tokens[1:3]
      polynomial_coefficients[index] = 2**int(s) + 2 * int(a) + 1
      for i, m_i in enumerate(tokens[3:]):
        direction_numbers[i, index] = int(m_i)
      index += 1
  return polynomial_coefficients, direction_numbers


(_PRIMITIVE_POLYNOMIAL_COEFFICIENTS,
 _INITIAL_DIRECTION_NUMBERS) = _load_sobol_data()
