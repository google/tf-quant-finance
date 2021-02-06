# Lint as: python3
# Copyright 2021 Google LLC
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
"""Util common functions for brent and newton methods."""

import numpy as np
import tensorflow.compat.v2 as tf


def default_relative_root_tolerance(dtype):
  """Returns the default relative root tolerance used for a TensorFlow dtype."""
  if dtype is None:
    dtype = tf.float64
  return 4 * np.finfo(dtype.as_numpy_dtype(0)).eps
