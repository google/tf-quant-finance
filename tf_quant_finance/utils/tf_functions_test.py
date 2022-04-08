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
"""Tests for tf_functions."""

from typing import Dict
import dataclasses
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@dataclasses.dataclass
class TestDataclass:
  simple_field: int
  dict_field: Dict[str, int]


class TfFunctionsTest(tf.test.TestCase):

  def _assert_keys_values(self,
                          iterator,
                          expected_keys=None,
                          expected_values=None):
    key_lists, values = zip(*list(iterator))
    keys = ['_'.join(k) for k in key_lists]
    if expected_keys is not None and expected_values is not None:
      result_dict = {k: v for k, v in zip(keys, values)}
      expected_dict = {k: v for k, v in zip(expected_keys, expected_values)}
      self.assertDictEqual(result_dict, expected_dict)
      return

    if expected_keys is not None:
      self.assertSameElements(keys, expected_keys)
      return

    if expected_values is not None:
      self.assertSameElements(values, expected_values)

  def test_non_nested(self):
    d = {'a': 1, 'b': 2}
    iterator = tff.utils.iterate_nested(d)
    self._assert_keys_values(
        iterator, expected_keys=['a', 'b'], expected_values=[1, 2])

  def test_empty(self):
    items = []
    for item in tff.utils.iterate_nested({}):
      items.append(item)
    self.assertEmpty(items)

  def test_array_values(self):
    d = {'a': [1, 2, 3], 'b': {'c': [4, 5]}}
    self._assert_keys_values(
        tff.utils.iterate_nested(d),
        expected_keys=['a', 'b_c'],
        expected_values=[[1, 2, 3], [4, 5]])

  def test_nested(self):
    nested_dict = {'a': 1, 'b': [2, 3, 4], 'c': {'d': 8}}
    self._assert_keys_values(
        tff.utils.iterate_nested(nested_dict),
        expected_keys=['a', 'b', 'c_d'],
        expected_values=[1, [2, 3, 4], 8])

  def test_dataclass(self):
    d = {
        'a': {
            'b': {
                'c': 5.1
            }
        },
        'data': TestDataclass(simple_field=42, dict_field={
            'a': 6,
            'z': 10
        })
    }
    expected_keys = [
        'a_b_c', 'data_simple_field', 'data_dict_field_a', 'data_dict_field_z'
    ]
    expected_vals = [5.1, 42, 6, 10]
    self._assert_keys_values(
        tff.utils.iterate_nested(d),
        expected_keys=expected_keys,
        expected_values=expected_vals)


if __name__ == '__main__':
  tf.test.main()
