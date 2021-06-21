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
"""Data class objects that works with tf.function."""

from typing import Type, Any
import attr

__all__ = ['dataclass']


def dataclass(cls: Type[Any]) -> Type[Any]:
  """Creates a data class object compatible with `tf.function`.

  Modifies dunder methods of an input class with typed attributes to work as an
  input/output to `tf.function`, as well as a loop variable of
  `tf.while_loop`.

  An intended use case for this decorator is on top of a simple class definition
  with type annotated arguments like in the example below. It is not guaranteed
  that this decorator works with an arbitrary class.


  #### Examples

  ```python
  import tensorflow as tf
  import tf_quant_finance as tff

  @tff.utils.dataclass
  class Coords:
    x: tf.Tensor
    y: tf.Tensor

  @tf.function
  def fn(start_coords: Coords) -> Coords:
    def cond(it, _):
      return it < 10
    def body(it, coords):
      return it + 1, Coords(x=coords.x + 1, y=coords.y + 2)
    return tf.while_loop(cond, body, loop_vars=(0, start_coords))[1]

  start_coords = Coords(x=tf.constant(0), y=tf.constant(0))
  fn(start_coords)
  # Expected Coords(a=10, b=20)
  ```

  Args:
    cls: Input class object with type annotated arguments. The class should not
      have an init method defined. Class fields are treated as ordered in the
      same order as they appear in the class definition.


  Returns:
    Modified class that can be used as a `tf.function` input/output as well
    as a loop variable of `tf.function`. All typed arguments of the original
    class are treated as ordered in the same order as they appear in the class
    definition. All untyped arguments are ignored. Modified class modifies
    `len` and `iter` methods defined for the  class instances such that `len`
    returns the number of arguments, and `iter`  creates an iterator for the
    ordered argument values.
  """
  # Wrap the class with attr.s to ensure that the class can be an input/output
  # to a `tf.function`
  cls = attr.s(cls, auto_attribs=True)

  # Define __iter__ and __len__ method to ensure tf.while_loop compatibility
  def __iter__(self):  # pylint: disable=invalid-name
    # Note that self.__attrs_attrs__ is a tuple so the iteration order is fixed
    for item in self.__attrs_attrs__:
      name = item.name
      yield getattr(self, name)

  def __len__(self):  # pylint: disable=invalid-name
    return len(self.__attrs_attrs__)

  cls.__len__ = __len__
  cls.__iter__ = __iter__
  return cls
