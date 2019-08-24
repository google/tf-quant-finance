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

# Lint as: python2, python3
"""Tests for quasirandom.halton that are particularly long."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow as tf
import tensorflow_probability as tfp

from tf_quant_finance.math.random import halton
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

# TODO: Remove dependency on contrib, which is being deprecated.
tfb = tf.contrib.bayesflow
tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class HaltonSequenceTest(tf.test.TestCase):

  def test_batch_randomized_converges_at_expected_rate(self):
    num_batches = 10
    batch_sizes = np.array([1000] * num_batches)
    dim = 2
    seed = 92251
    # The exact integral of this function is dim/3. To mimic a real problem more
    # closely, below we don't use the exact solution but instead compute the
    # variance over the set of estimated means for each replica.
    my_func = lambda x: tf.reduce_sum(x * x, axis=1)
    num_replicas = 10

    # Variance of the replica means for each batch.
    variance = np.empty(num_batches)
    # Cumulative (over the batches) mean for each replica for the current batch.
    current_means = np.zeros(num_replicas)

    for batch_index, batch_size in enumerate(batch_sizes):
      # We consider the batches cumulatively. For a given replica, we generate
      # 5000 Halton points per batch, and we consider all the batches generated
      # so far for that replica when updating the mean. The batch_weight is used
      # to update the current cumulative mean from the current batch of points.
      batch_weight = batch_size / batch_sizes[:(batch_index + 1)].sum()
      seq_min = batch_sizes[:batch_index].sum()
      seq_max = seq_min + batch_size
      sequence_indices = np.arange(seq_min, seq_max, dtype=np.int32)
      for replica_index in range(num_replicas):
        sample, _ = halton.sample(
            dim, sequence_indices=sequence_indices, seed=(seed + replica_index))
        batch_mean = self.evaluate(tf.reduce_mean(my_func(sample)))
        current_means[replica_index] = (
            (1 - batch_weight) * current_means[replica_index] +
            batch_weight * batch_mean)
      variance[batch_index] = np.var(current_means)

    logx = np.log(np.cumsum(batch_sizes))
    logy = np.log(np.sqrt(variance))
    slope = np.polyfit(logx, logy, 1)[0]
    # We expect the error to be close to O(1/n) for a Quasi-Monte Carlo method,
    # which corresponds to a slope of -1 on a log scale. In comparison,
    # conventional Monte Carlo produces an error of O(1/sqrt(N)) or a slope of
    # -0.5 on a log scale.
    self.assertAllClose(slope, -1, rtol=1e-1)


if __name__ == "__main__":
  tf.test.main()
