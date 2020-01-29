# Lint as: python3
"""Payoff functions."""

import tensorflow.compat.v2 as tf


def make_basket_put_payoff(strike_price, dtype=None, name=None):
  """Produces a callable from samples to payoff of a simple basket put option.

  Args:
    strike_price: A `Tensor` of `dtype` consistent with `samples` and shape
      `[num_samples, num_strikes]`.
    dtype: Optional `dtype`. Either `tf.float32` or `tf.float64`. The `dtype`
      If supplied, represents the `dtype` for the 'strike_price' as well as
      for the input argument of the ouput payoff callable.
      Default value: `None`, which means that the `dtype` inferred by TensorFlow
      is used.
    name: Python `str` name prefixed to Ops created by the callable created
      by this function.
      Default value: `None` which is mapped to the default name 'put_valuer'
  Returns:
    A callable from `Tensor` of shape `[num_samples, num_exercise_times, dim]`
    to `Tensor` of shape `[num_samples, num_strikes]`.
  """
  strike_price = tf.convert_to_tensor(strike_price, dtype=dtype,
                                      name='strike_price')
  def put_valuer(sample_paths):
    with tf.compat.v1.name_scope(name, default_name='put_valuer',
                                 values=[sample_paths, strike_price]):
      sample_paths = tf.convert_to_tensor(sample_paths, dtype=dtype,
                                          name='sample_paths')
      average = tf.expand_dims(
          tf.math.reduce_mean(sample_paths[:, -1, :], axis=-1), axis=-1)
      return tf.nn.relu(strike_price - average)

  return put_valuer
