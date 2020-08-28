"""Common utilities for interpolation."""

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
