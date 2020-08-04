"""Common utilities for interpolation."""

import numpy as np
import tensorflow.compat.v2 as tf


def broadcast_batch_shape(x, batch_shape):
  """Broadcasts batch shape of `x`."""
  return tf.broadcast_to(x, tf.TensorShape(batch_shape) + x.shape[-1])


def expand_to_rank(x, rank):
  """Expands zero dimension of `x` to match the `rank`."""
  rank_x = x.shape.rank
  if rank_x < rank:
    # Output shape with extra dimensions
    output_shape = tf.concat([(rank - rank_x) * [1], tf.shape(x)], axis=-1)
    x = tf.reshape(x, output_shape)
  return x


def broadcast_common_batch_shape(x, y):
  """Broadcasts batch shapes of `x` and `y`."""
  rank = max(x.shape.rank, y.shape.rank)
  x = expand_to_rank(x, rank)
  y = expand_to_rank(y, rank)
  if x.shape.as_list()[:-1] != y.shape.as_list()[:-1]:
    try:
      x = broadcast_batch_shape(x, y.shape[:-1])
    except (tf.errors.InvalidArgumentError, ValueError):
      try:
        y = broadcast_batch_shape(y, x.shape[:-1])
      except (tf.errors.InvalidArgumentError, ValueError):
        raise ValueError(
            "Can not broadcast batch shapes {0} and {1}".format(
                x.shape.as_list()[:-1], y.shape.as_list()[:-1]))
  return x, y


def prepare_indices(indices):
  """Prepares `tf.searchsorted` output for index argument of `tf.gather_nd`.

  Creates an index matrix that can be used along with `tf.gather_nd`.

  #### Example
  indices = tf.constant([[[1, 2], [2, 3]]])
  index_matrix = utils.prepare_indices(indices)
  # Outputs a tensor of shape [1, 2, 3, 2]
  # [[[[0, 0], [0, 0], [0, 0]], [[0, 1], [0, 1], [0, 1]]]]
  # The index matrix can be concatenated with the indices in order to obtain
  # gather_nd selection matrix
  tf.concat([index_matrix, tf.expand_dims(indices, axis=-1)], axis=-1)
  # Outputs
  # [[[[0, 0, 1], [0, 0, 2], [0, 0, 3]],
  #   [[0, 1, 2], [0, 1, 3], [0, 1, 4]]]]

  Args:
    indices: A `Tensor` of any shape and dtype.

  Returns:
    A `Tensor` of the same dtype as `indices` and shape
    `indices.shape + [indices.shape.rank - 1]`.
  """
  batch_shape_reverse = indices.shape.as_list()[:-1]
  batch_shape_reverse.reverse()
  # Shape batch_shape + [batch_rank]
  index_matrix = tf.constant(
      np.flip(np.transpose(np.indices(batch_shape_reverse)), -1),
      dtype=indices.dtype)
  # Broadcast index matrix to
  # `batch_shape + [num_points] + [batch_rank]`
  index_matrix = (tf.expand_dims(index_matrix, -2)
                  + tf.zeros_like(tf.expand_dims(indices, -1)))
  return index_matrix
