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

"""Quasi Monte Carlo support: Halton sequence."""

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.math.random_ops import stateless


__all__ = [
    'sample',
]

# The maximum dimension we support. This is limited by the number of primes
# in the _PRIMES array.
_MAX_DIMENSION = 1000

# The maximum sequence index we support, depending on data type.
_MAX_INDEX_BY_DTYPE = {tf.float32: 2**24 - 1, np.float32: 2**24 - 1,
                       tf.float64: 2**53 - 1, np.float64: 2**53 - 1}

# The number of coefficients we use to represent each Halton number when
# expressed in the (prime) base for an event dimension. In theory this should be
# infinite, but in practice it is useful to cap this based on data type.
_NUM_COEFFS_BY_DTYPE = {tf.float32: 24, np.float32: 24,
                        tf.float64: 54, np.float64: 54}


# Parameters that can be reused with subsequent calls to halton.sample().
@utils.dataclass
class HaltonParams:
  """Halton randomization parameters."""
  # Uniform iid sample from the space of permutations as
  # returned by _get_permutations() below. A tensor of shape
  # [_MAX_SIZES_BY_AXES[dtype], sum(_PRIMES)] and dtype
  # tf.float32 or tf.float64.
  perms: types.IntTensor
  # A scaled uniform random tensor of shape [dim] and  dtype tf.float32 or
  # tf.float64. Used to appropriately adjust the randomization described in Owen
  # (2017), see below for further details.
  zero_correction: types.RealTensor


def sample(dim: int,
           num_results: types.IntTensor = None,
           sequence_indices: types.IntTensor = None,
           randomized: bool = True,
           randomization_params=None,
           seed: types.IntTensor = None,
           validate_args: bool = False,
           dtype: tf.DType = None,
           name: str = None) -> types.RealTensor:
  r"""Returns a sample from the `dim` dimensional Halton sequence.

  Warning: The sequence elements take values only between 0 and 1. Care must be
  taken to appropriately transform the domain of a function if it differs from
  the unit cube before evaluating integrals using Halton samples. It is also
  important to remember that quasi-random numbers without randomization are not
  a replacement for pseudo-random numbers in every context. Quasi random numbers
  are completely deterministic and typically have significant negative
  autocorrelation unless randomization is used.

  Computes the members of the low discrepancy Halton sequence in dimension
  `dim`. The `dim`-dimensional sequence takes values in the unit hypercube in
  `dim` dimensions. Currently, only dimensions up to 1000 are supported. The
  prime base for the k-th axes is the k-th prime starting from 2. For example,
  if `dim` = 3, then the bases will be [2, 3, 5] respectively and the first
  element of the non-randomized sequence will be: [0.5, 0.333, 0.2]. For a more
  complete description of the Halton sequences see
  [here](https://en.wikipedia.org/wiki/Halton_sequence). For low discrepancy
  sequences and their applications see
  [here](https://en.wikipedia.org/wiki/Low-discrepancy_sequence).

  If `randomized` is true, this function produces a scrambled version of the
  Halton sequence introduced by [Owen (2017)][1]. For the advantages of
  randomization of low discrepancy sequences see [here](
  https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method#Randomization_of_quasi-Monte_Carlo).

  The number of samples produced is controlled by the `num_results` and
  `sequence_indices` parameters. The user must supply either `num_results` or
  `sequence_indices` but not both.
  The former is the number of samples to produce starting from the first
  element. If `sequence_indices` is given instead, the specified elements of
  the sequence are generated. For example, sequence_indices=tf.range(10) is
  equivalent to specifying n=10.

  #### Examples

  ```python
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp

  # Produce the first 1000 members of the Halton sequence in 3 dimensions.
  num_results = 1000
  dim = 3
  sample, params = qmc.halton.sample(
    dim,
    num_results=num_results,
    seed=127)

  # Evaluate the integral of x_1 * x_2^2 * x_3^3  over the three dimensional
  # hypercube.
  powers = tf.range(1.0, limit=dim + 1)
  integral = tf.reduce_mean(tf.reduce_prod(sample ** powers, axis=-1))
  true_value = 1.0 / tf.reduce_prod(powers + 1.0)
  with tf.Session() as session:
    values = session.run((integral, true_value))

  # Produces a relative absolute error of 1.7%.
  print ("Estimated: %f, True Value: %f" % values)

  # Now skip the first 1000 samples and recompute the integral with the next
  # thousand samples. The sequence_indices argument can be used to do this.


  sequence_indices = tf.range(start=1000, limit=1000 + num_results,
                              dtype=tf.int32)
  sample_leaped, _ = qmc.halton.sample(
      dim,
      sequence_indices=sequence_indices,
      randomization_params=params)

  integral_leaped = tf.reduce_mean(tf.reduce_prod(sample_leaped ** powers,
                                                  axis=-1))
  with tf.Session() as session:
    values = session.run((integral_leaped, true_value))
  # Now produces a relative absolute error of 0.05%.
  print ("Leaped Estimated: %f, True Value: %f" % values)
  ```

  Args:
    dim: Positive Python `int` representing each sample's `event_size.` Must not
      be greater than 1000.
    num_results: (Optional) Positive scalar `Tensor` of dtype int32. The number
      of samples to generate. Either this parameter or sequence_indices must be
      specified but not both. If this parameter is None, then the behaviour is
      determined by the `sequence_indices`.
      Default value: `None`.
    sequence_indices: (Optional) `Tensor` of dtype int32 and rank 1. The
      elements of the sequence to compute specified by their position in the
      sequence. The entries index into the Halton sequence starting with 0 and
      hence, must be whole numbers. For example, sequence_indices=[0, 5, 6] will
      produce the first, sixth and seventh elements of the sequence. If this
      parameter is None, then the `num_results` parameter must be specified
      which gives the number of desired samples starting from the first sample.
      Default value: `None`.
    randomized: (Optional) bool indicating whether to produce a randomized
      Halton sequence. If True, applies the randomization described in [Owen
      (2017)][1]. If True, either seed or randomization_params must be
      specified. This is because the randomization uses stateless random number
      generation which requires an explicitly specified seed.
      Default value: `True`.
    randomization_params: (Optional) Instance of `HaltonParams` that fully
      describes the randomization behavior. If provided and randomized is True,
      seed will be ignored and these will be used instead of computing them from
      scratch. If randomized is False, this parameter has no effect.
      Default value: `None`. In this case with randomized = True, the necessary
        randomization parameters will be computed from scratch.
    seed: (Optional) Python integer to seed the random number generator. Must be
      specified if `randomized` is True and randomization_params is not
      specified. Ignored if randomized is False or randomization_params is
      specified.
      Default value: `None`.
    validate_args: If True, checks that maximum index is not exceeded and that
      the dimension `dim` is less than 1 or greater than 1000.
      Default value: `False`.
    dtype: Optional `dtype`. The dtype of the output `Tensor` (either `float32`
    or `float64`).
      Default value: `None` which maps to the `float32`.
    name:  (Optional) Python `str` describing ops managed by this function. If
      not supplied the name of this function is used.
      Default value: "halton_sample".

  Returns:
    halton_elements: Elements of the Halton sequence. `Tensor` of supplied dtype
      and `shape` `[num_results, dim]` if `num_results` was specified or shape
      `[s, dim]` where s is the size of `sequence_indices` if `sequence_indices`
      were specified.
    randomization_params: None if randomized is False. If randomized is True
      and randomization_params was supplied as an argument, returns that.
      Otherwise returns the computed randomization_params, an instance of
      `HaltonParams` that fully describes the randomization behavior.

  Raises:
    ValueError: if both `sequence_indices` and `num_results` were specified.
    ValueError: if `randomization` is True but `seed` is not specified.
    InvalidArgumentError: if `validate_args` is True and the maximum supported
      sequence index is exceeded.

  #### References

  [1]: Art B. Owen. A randomized Halton algorithm in R. _arXiv preprint
       arXiv:1706.02808_, 2017. https://arxiv.org/abs/1706.02808
  """
  if (num_results is None) == (sequence_indices is None):
    raise ValueError('Either `num_results` or `sequence_indices` must be'
                     ' specified but not both.')
  dtype = dtype or tf.float32

  with tf.compat.v1.name_scope(
      name, 'halton_sample', values=[num_results, sequence_indices]):
    # Here and in the following, the shape layout is as follows:
    # [sample dimension, event dimension, coefficient dimension].
    # The coefficient dimension is an intermediate axes which will hold the
    # weights of the starting integer when expressed in the (prime) base for
    # an event dimension.
    if num_results is not None:
      num_results = tf.convert_to_tensor(value=num_results,
                                         dtype=tf.int32,
                                         name='name_results')
    if sequence_indices is not None:
      sequence_indices = tf.convert_to_tensor(value=sequence_indices,
                                              dtype=tf.int32,
                                              name='sequence_indices')
    indices = _get_indices(num_results, sequence_indices, dtype)

    runtime_assertions = []
    if validate_args:
      runtime_assertions.append(
          tf.compat.v1.assert_less_equal(
              tf.reduce_max(indices),
              tf.constant(_MAX_INDEX_BY_DTYPE[dtype], dtype=dtype),
              message=(
                  'Maximum sequence index exceeded. Maximum index for dtype %s '
                  'is %d.' % (dtype, _MAX_INDEX_BY_DTYPE[dtype]))))
      runtime_assertions.append(
          tf.compat.v1.assert_greater_equal(
              dim, 1, message='`dim` should be greater than 1'))
      runtime_assertions.append(
          tf.compat.v1.assert_less_equal(
              dim, _MAX_DIMENSION,
              message='`dim` should be less or equal than %d' % _MAX_DIMENSION))

    with tf.compat.v1.control_dependencies(runtime_assertions):
      radixes = tf.convert_to_tensor(_PRIMES, dtype=dtype, name='radixes')
      radixes = tf.reshape(radixes[0:dim], shape=[dim, 1])

      max_sizes_by_axes = tf.convert_to_tensor(
          _MAX_SIZES_BY_AXES[dtype],
          dtype=dtype,
          name='max_sizes_by_axes')[:dim]
      max_size = tf.reduce_max(max_sizes_by_axes)

      # The powers of the radixes that we will need. Note that there is a bit
      # of an excess here. Suppose we need the place value coefficients of 7
      # in base 2 and 3. For 2, we will have 3 digits but we only need 2 digits
      # for base 3. However, we can only create rectangular tensors so we
      # store both expansions in a [2, 3] tensor. This leads to the problem that
      # we might end up attempting to raise large numbers to large powers. For
      # example, base 2 expansion of 1024 has 10 digits. If we were in 10
      # dimensions, then the 10th prime (29) we will end up computing 29^10 even
      # though we don't need it. We avoid this by setting the exponents for each
      # axes to 0 beyond the maximum value needed for that dimension.
      exponents_by_axes = tf.tile([tf.range(max_size, dtype=dtype)], [dim, 1])

      # The mask is true for those coefficients that are irrelevant.
      weight_mask = exponents_by_axes >= max_sizes_by_axes
      capped_exponents = tf.where(weight_mask, tf.zeros_like(exponents_by_axes),
                                  exponents_by_axes)
      # Round to nearest integer to correct small errors when applying floating-
      # point pow(radixes, capped_exponents).
      weights = tf.compat.v1.round(radixes**capped_exponents)
      # The following computes the base b expansion of the indices. Suppose,
      # x = a0 + a1*b + a2*b^2 + ... Then, performing a floor div of x with
      # the vector (1, b, b^2, b^3, ...) will produce
      # (a0 + s1 * b, a1 + s2 * b, ...) where s_i are coefficients we don't care
      # about. Noting that all a_i < b by definition of place value expansion,
      # we see that taking the elements mod b of the above vector produces the
      # place value expansion coefficients.
      coeffs = tf.compat.v1.floor_div(indices, weights)
      coeffs *= 1. - tf.cast(weight_mask, dtype)
      coeffs %= radixes
      if not randomized:
        coeffs /= radixes
        return tf.reduce_sum(input_tensor=coeffs / weights, axis=-1), None

      if randomization_params is None:
        perms, zero_correction = None, None
      else:
        perms, zero_correction = randomization_params
      coeffs, perms = _randomize(coeffs, radixes, seed, perms=perms)
      # Remove the contribution from randomizing the trailing zero for the
      # axes where max_size_by_axes < max_size. This will be accounted
      # for separately below (using zero_correction).
      coeffs *= 1. - tf.cast(weight_mask, dtype)
      coeffs /= radixes
      base_values = tf.reduce_sum(input_tensor=coeffs / weights, axis=-1)

      # The randomization used in Owen (2017) does not leave 0 invariant. While
      # we have accounted for the randomization of the first `max_size_by_axes`
      # coefficients, we still need to correct for the trailing zeros. Luckily,
      # this is equivalent to adding a uniform random value scaled so the first
      # `max_size_by_axes` coefficients are zero. The following statements
      # perform this correction.
      if zero_correction is None:
        if seed is None:
          zero_correction = tf.random.uniform([dim, 1], dtype=dtype)
        else:
          zero_correction = tf.random.stateless_uniform([dim, 1],
                                                        seed=(seed, seed),
                                                        dtype=dtype)
        zero_correction /= radixes**max_sizes_by_axes
        zero_correction = tf.reshape(zero_correction, [-1])

      return base_values + zero_correction, HaltonParams(perms, zero_correction)


def _randomize(coeffs, radixes, seed, perms=None):
  """Applies the Owen (2017) randomization to the coefficients."""
  given_dtype = coeffs.dtype
  coeffs = tf.cast(coeffs, dtype=tf.int32)
  num_coeffs = _NUM_COEFFS_BY_DTYPE[given_dtype]
  radixes = tf.reshape(tf.cast(radixes, dtype=tf.int32), shape=[-1])
  if perms is None:
    perms = _get_permutations(num_coeffs, radixes, seed)
    perms = tf.reshape(perms, shape=[-1])
  radix_sum = tf.reduce_sum(input_tensor=radixes)
  radix_offsets = tf.reshape(tf.cumsum(radixes, exclusive=True), shape=[-1, 1])
  offsets = radix_offsets + tf.range(num_coeffs) * radix_sum
  permuted_coeffs = tf.gather(perms, coeffs + offsets)
  return tf.cast(permuted_coeffs, dtype=given_dtype), perms


def _get_permutations(num_results, dims, seed):
  """Uniform iid sample from the space of permutations.

  Draws a sample of size `num_results` from the group of permutations of degrees
  specified by the `dims` tensor. These are packed together into one tensor
  such that each row is one sample from each of the dimensions in `dims`. For
  example, if dims = [2,3] and num_results = 2, the result is a tensor of shape
  [2, 2 + 3] and the first row of the result might look like:
  [1, 0, 2, 0, 1]. The first two elements are a permutation over 2 elements
  while the next three are a permutation over 3 elements.

  Args:
    num_results: A positive scalar `Tensor` of integer type. The number of
      draws from the discrete uniform distribution over the permutation groups.
    dims: A 1D `Tensor` of the same dtype as `num_results`. The degree of the
      permutation groups from which to sample.
    seed: (Optional) Python integer to seed the random number generator.

  Returns:
    permutations: A `Tensor` of shape `[num_results, sum(dims)]` and the same
    dtype as `dims`.
  """
  sample_range = tf.range(num_results)

  def generate_one(d):

    def fn(i):
      if seed is None:
        return tf.random.shuffle(tf.range(d))
      else:
        return stateless.stateless_random_shuffle(tf.range(d),
                                                  seed=(seed + i, d))

    return tf.map_fn(
        fn, sample_range, parallel_iterations=1 if seed is not None else 10)

  return tf.concat([generate_one(d) for d in tf.unstack(dims)], axis=-1)


def _get_indices(num_results, sequence_indices, dtype, name=None):
  """Generates starting points for the Halton sequence procedure.

  The k'th element of the sequence is generated starting from a positive integer
  which must be distinct for each `k`. It is conventional to choose the starting
  point as `k` itself (or `k+1` if k is zero based). This function generates
  the starting integers for the required elements and reshapes the result for
  later use.

  Args:
    num_results: Positive scalar `Tensor` of dtype int32. The number of samples
      to generate. If this parameter is supplied, then `sequence_indices` should
      be None.
    sequence_indices: `Tensor` of dtype int32 and rank 1. The entries index into
      the Halton sequence starting with 0 and hence, must be whole numbers. For
      example, sequence_indices=[0, 5, 6] will produce the first, sixth and
      seventh elements of the sequence. If this parameter is not None then `n`
      must be None.
    dtype: The dtype of the sample. One of `float32` or `float64`. Default is
      `float32`.
    name: Python `str` name which describes ops created by this function.

  Returns:
    indices: `Tensor` of dtype `dtype` and shape = `[n, 1, 1]`.
  """
  with tf.compat.v1.name_scope(name, '_get_indices',
                               [num_results, sequence_indices]):
    if sequence_indices is None:
      num_results = tf.cast(num_results, dtype=dtype)
      sequence_indices = tf.range(num_results, dtype=dtype)
    else:
      sequence_indices = tf.cast(sequence_indices, dtype)

    # Shift the indices so they are 1 based.
    indices = sequence_indices + 1

    # Reshape to make space for the event dimension and the place value
    # coefficients.
    return tf.reshape(indices, [-1, 1, 1])


def _base_expansion_size(num, bases):
  """Computes the number of terms in the place value expansion.

  Let num = a0 + a1 b + a2 b^2 + ... ak b^k be the place value expansion of
  `num` in base b (ak <> 0). This function computes and returns `k+1` for each
  base `b` specified in `bases`.

  This can be inferred from the base `b` logarithm of `num` as follows:
    $$k = Floor(log_b (num)) + 1  = Floor( log(num) / log(b)) + 1$$

  Args:
    num: Scalar numpy array of dtype either `float32` or `float64`. The number
      to compute the base expansion size of.
    bases: Numpy array of the same dtype as num. The bases to compute the size
      against.

  Returns:
    Tensor of same dtype and shape as `bases` containing the size of num when
    written in that base.
  """
  return np.floor(np.log(num) / np.log(bases)) + 1


# First 1000 prime numbers.
_PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
    613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
    709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
    821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
    919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
    1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
    1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181,
    1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277,
    1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361,
    1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451,
    1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531,
    1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609,
    1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699,
    1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789,
    1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889,
    1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997,
    1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083,
    2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161,
    2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273,
    2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357,
    2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441,
    2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551,
    2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663,
    2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729,
    2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819,
    2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917,
    2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023,
    3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137,
    3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251,
    3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331,
    3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449,
    3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533,
    3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617,
    3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709,
    3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821,
    3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917,
    3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013,
    4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111,
    4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219,
    4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297,
    4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423,
    4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519,
    4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639,
    4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729,
    4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831,
    4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951,
    4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023,
    5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147,
    5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261,
    5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387,
    5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471,
    5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563,
    5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659,
    5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779,
    5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857,
    5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981,
    5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089,
    6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199,
    6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287,
    6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367,
    6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491,
    6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607,
    6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709,
    6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827,
    6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917,
    6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013,
    7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129,
    7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243,
    7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369,
    7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499,
    7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577,
    7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681,
    7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789,
    7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901,
    7907, 7919
],
                   dtype=np.int32)

# For each supported data type, we store the maximum number of digits we might
# need for each dimension. See _base_expansion_size() for more details.
_MAX_SIZES_BY_AXES = {
    dtype: _base_expansion_size(_MAX_INDEX_BY_DTYPE[dtype],
                                np.expand_dims(_PRIMES, 1))
    for dtype in _MAX_INDEX_BY_DTYPE
}

assert len(_PRIMES) == _MAX_DIMENSION
