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
"""Tests for Levenberg Marquardt algorithm."""


from functools import partial

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import (
    test_util,
)  # pylint: disable=g-direct-tensorflow-import


# Test functions.
def _linear(p, x):
    a = p[0]
    b = p[1]
    return a * x + b


@test_util.run_all_in_graph_and_eager_modes
class LevenbergMarquardtTest(tf.test.TestCase):
    def _check_algorithm(
        self,
        func=None,
        x_data=None,
        y_data=None,
        start_point=None,
        xtol=1e-4,
        ftol=1e-4,
        gtol=1e-4,
        expected_position=None,
    ):
        """Runs algorithm on given test case and verifies result."""
        val_grad_func = lambda p, x: tff.math.value_and_gradient(partial(func, x=x))
        x_data = tf.constant(x_data, dtype=tf.float64)
        y_data = tf.constant(y_data, dtype=tf.float64)
        start_point = tf.constant(start_point, dtype=tf.float64)
        expected_position = np.array(expected_position, dtype=np.float64)

        f_call_ctr = tf.Variable(0, dtype=tf.int32)

        def val_grad_func_with_counter(x):
            with tf.compat.v1.control_dependencies(
                [tf.compat.v1.assign_add(f_call_ctr, 1)]
            ):
                return val_grad_func

        result = tff.math.experimental.optimizer.levenberg_marquardt_fit(
            val_grad_func_with_counter,
            x_data=x_data,
            y_data=y_data,
            initial_position=start_point,
            x_tolerance=xtol,
            f_tolerance=ftol,
            g_tolerance=gtol,
            max_iterations=1,
        )  # TODO: Max iterations
        self.evaluate(tf.compat.v1.global_variables_initializer())
        result = self.evaluate(result)
        f_call_ctr = self.evaluate(f_call_ctr)

    def test_linear(self):
        self._check_algorithm(
            func=_linear,
            x_data=[-1.0, 0.0, 1.0],
            y_data=[0.0, 1.0, 2.0],
            start_point=[100.0],
            expected_position=[1.0, 1.0],
        )


if __name__ == "__main__":
    tf.test.main()
