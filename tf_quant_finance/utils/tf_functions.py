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
"""Functions to deal with tf.function objects."""

import dataclasses
from typing import Any, Dict, Tuple, Union, Iterator, List, Optional

# Dataclasses do not have a specific base type but do have a common interface
# to convert them to dictionaries. We treat them as non-opaque objects whose
# fields will be traversed.
DataClassType = Any
# Anything which is not a dataclass or a dictionary is treated as a leaf node
# and terminates the traversal.
LeafType = Any

# A nested dictionary with string keys and dictionary or dataclass values which
# will also be traversed.
NestedDict = Dict[str, Union[Any, DataClassType, 'NestedDict']]

# Ordered collections of keys starting from the outermost key to the innermost.
KeyList = List[str]


def iterate_nested(
    nd: NestedDict,
    previous_keys: Optional[KeyList] = None
) -> Iterator[Tuple[KeyList, LeafType]]:
  """Creates an iterator over every leaf value in depth first order.

  Iterates over a nested dictionary in depth first order. The order in which
  the peer keys are traversed is not guaranteed (same as for the keys of a
  dictionary).

  ```Example
  nested_dict = {'a': 1, 'b': [2, 3, 4], 'c': {'d': 8}}
  for k, v in iterate_nested(nested_dict):
    print('_'.join(k), v)
  # Prints out:
  # a: 1
  # b: [2, 3, 4]
  # c_d: 8
  ```

  Args:
    nd: The dictionary to be traversed.
    previous_keys: If supplied, the computed key list will be a join of the
      previous_keys and the current keys.

  Yields:
    A tuple of the key path and the value for each leaf node.
  """
  if previous_keys is None:
    previous_keys = []
  for k, v in nd.items():
    keys = previous_keys + [k]
    if not _is_nested(v):
      yield keys, v
    else:  # It is nested.
      as_dict = dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
      for val in iterate_nested(as_dict, keys):
        yield val


def _is_nested(x: Any) -> bool:
  """Returns whether a value is nested."""
  return isinstance(x, dict) or dataclasses.is_dataclass(x)
