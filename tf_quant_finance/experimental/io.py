# Lint as: python3
"""Utilities to serialize and deserialize dictionaries of numpy arrays.

This module defines a data reader and writer to export collections of numpy
arrays to files. It is based on `TFRecords` format. The main difference is
that instead of directly storing elements in `FloatList` (or `IntList` etc)
protos, it first serializes them to bytes and then stores them as a single
element `BytesList`. This is necessary to improve deserialization performance
when we have large arrays.
"""

from typing import Dict, Callable, Optional

import numpy as np
import tensorflow.compat.v2 as tf


__all__ = [
    'encode_array',
    'decode_array',
    'ArrayDictReader',
    'ArrayDictWriter'
]


# Needed for decoding serialized arrays.
_CLS = type(tf.make_tensor_proto([0]))

ArrayEncoderFn = Callable[[np.ndarray], bytes]
ArrayDecoderFn = Callable[[bytes], np.ndarray]


def encode_array(x: np.ndarray) -> bytes:
  """Encodes a numpy array using `TensorProto` protocol buffer."""
  return tf.make_tensor_proto(x).SerializeToString()


def decode_array(bytestring: bytes) -> np.ndarray:
  """Decodes a bytestring into a numpy array.

  The bytestring should be a serialized `TensorProto` instance. For more details
  see `tf.make_tensor_proto`.

  Args:
    bytestring: The serialized `TensorProto`.

  Returns:
    A numpy array.
  """
  return tf.make_ndarray(_CLS.FromString(bytestring))


class ArrayDictWriter:
  """Writer to write dictionaries of numpy arrays in binary format.

  Writes dictionaries with string keys and numpy array values as records into
  a [tfrecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) file.
  The usage of tfrecords should be treated as an implementation detail. To
  read the file, use the `ArrayDictReader` class.

  Notes and Limitations:

  1. Any values which are not numpy arrays will be first converted to
    an array before serializing.
  2. Serializing strings or arrays of strings is complicated because numpy
    doesn't support variable length strings. By default, a python string
    converted to a numpy array will be converted to a fixed size dtype (of the
    form 'Un' where n is the size of the largest string). During serialization
    this information is lost and the deserialization produces an object array
    with bytes elements. These need to be manually converted back
    to a unicode string using `object.astype('U?') where ? is the length of the
    largest string in the array.

  #### Example
  ```python
    options_data = {
        'instrument_type': 'EuropeanOption',
        'strikes': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'is_call': np.array([True, True, False]),
        'expiries': np.array([0.4, 1.3, 2.3], dtype=np.float64)
    }
    barriers_data = {
        'instrument_type': 'BarrierOption',
        'strikes': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'is_call': np.array([True, True, False]),
        'expiries': np.array([0.4, 1.3, 2.3], dtype=np.float64),
        'barrier': np.array([1.4, 2.5, 2.5], dtype=np.float64),
        'is_knockout': np.array([True, True, False])
    }
    with ArrayDictWriter('datafile.bin') as writer:
      writer.write(options_data)
      writer.write(barriers_data)
  ```
  """

  def __init__(self, path):
    self._writer = tf.io.TFRecordWriter(path)

  def __enter__(self) -> 'ArrayDictWriter':
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    """Exits a `with` block and closes the file."""
    del unused_type, unused_value, unused_traceback
    self.close()

  def write(self, array_dict: Dict[str, np.ndarray]) -> None:
    """Writes a dictionary of arrays to the file.

    Args:
      array_dict: A record to write. Should be a dictionary with string keys and
        numpy array values.
    """
    self._writer.write(_make_example(array_dict))

  def flush(self):
    """Flushes the file."""
    self._writer.flush()

  def close(self):
    """Close the file."""
    self._writer.close()


class ArrayDictReader():
  """Iterator over the data serialized by `ArrayDictWriter`.

  The reader counterpart of the `ArrayDictWriter`. It deserializes the binary
  `tfrecords` files written by the writer.

  Provides an iterable interface.

  #### Example
  ```python
    filename = '...'
    reader = ArrayDictReader(filename)
    # Read one record.
    first_record = next(reader)
    for record in reader:
      print(record)  # Print the rest of the records.
  ```
  """

  def __init__(self, path: str):
    self._iter = tf.data.TFRecordDataset([path]).as_numpy_iterator()

  def __iter__(self):
    return self

  def __next__(self) -> Dict[str, np.ndarray]:
    # Pull an element out of the dataset and parse it.
    raw_data = self._iter.next()
    example = tf.train.Example()
    example.ParseFromString(raw_data)
    feature = example.features.feature
    output = {
        key: decode_array(value.bytes_list.value[0])
        for key, value in feature.items()
    }
    return output

  def next(self):
    """Returns the next record if there is one or raises `StopIteration`."""
    return next(self)


def _make_feature(
    arr: np.ndarray,
    array_encoder: Optional[ArrayEncoderFn] = None) -> tf.train.Feature:
  """Encodes the array and wraps it into a `tf.train.Feature`."""
  if array_encoder is None:
    array_encoder = encode_array
  # This is important to do to ensure that values not actually
  # wrapped already in a numpy array are encoded and decoded predictably.
  if not isinstance(arr, np.ndarray):
    arr = np.array(arr)
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[encode_array(arr)]))


def _make_example(d: Dict[str, np.ndarray],
                  array_encoder: Optional[ArrayEncoderFn] = None) -> bytes:
  """Serializes a dictionary of arrays using an `tf.train.Example` proto."""
  features_dict = {
      key: _make_feature(value, array_encoder) for key, value in d.items()
  }
  return tf.train.Example(features=tf.train.Features(
      feature=features_dict)).SerializeToString()

