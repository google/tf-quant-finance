# Lint as: python3
"""Tests for io module."""

from os import path

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_quant_finance.experimental import io
from tf_quant_finance.experimental.pricing_platform import instrument_protos
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

InterestRateSwap = instrument_protos.interest_rate_swap.InterestRateSwap
InstrumentMetadata = instrument_protos.metadata.InstrumentMetadata


class IoTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "float32",
          "input_array": np.array([1, 2, 3], dtype=np.float32),
      }, {
          "testcase_name": "float64",
          "input_array": np.array([1, 2, 3], dtype=np.float64),
      }, {
          "testcase_name": "int32",
          "input_array": np.array([1, 2, 3], dtype=np.int32),
      }, {
          "testcase_name": "int64",
          "input_array": np.array([1, 2, 3], dtype=np.int64),
      }, {
          "testcase_name": "bool",
          "input_array": np.array([True, False, True], dtype=np.int32),
      })
  def test_array_encoder_decoder(self, input_array):
    as_bytes = io.encode_array(input_array)
    recovered = io.decode_array(as_bytes)
    np.testing.assert_array_equal(input_array, recovered)

  def test_array_encoder_decoder_string(self):
    input_array = np.array(["centaur", "satyr", "harpy"])
    as_bytes = io.encode_array(input_array)
    recovered = io.decode_array(as_bytes).astype("U")
    np.testing.assert_array_equal(input_array, recovered)

  def test_read_write(self):
    options_data = {
        "instrument_type": "EuropeanOption",
        "strikes": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "is_call": np.array([True, True, False]),
        "expiries": np.array([0.4, 1.3, 2.3], dtype=np.float64)
    }
    barriers_data = {
        "instrument_type": "BarrierOption",
        "strikes": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "is_call": np.array([True, True, False]),
        "expiries": np.array([0.4, 1.3, 2.3], dtype=np.float64),
        "barrier": np.array([1.4, 2.5, 2.5], dtype=np.float64),
        "is_knockout": np.array([True, True, False])
    }
    temp_dir = self.create_tempdir()
    temp_file = path.join(temp_dir.full_path, "datafile.bin")
    with io.ArrayDictWriter(temp_file) as writer:
      writer.write(options_data)
      writer.write(barriers_data)

    self.assertTrue(path.exists(temp_file))

    # Check that we can read the file.
    reader = io.ArrayDictReader(temp_file)
    first_record = reader.next()
    self.assertEqual(options_data.keys(), first_record.keys())
    for key, value in options_data.items():
      extracted = first_record[key]
      if key == "instrument_type":
        # Strings need to be explicitly coerced to unicode.
        extracted = extracted.astype("U")
      np.testing.assert_array_equal(value, extracted)

    remaining = []
    for record in reader:
      remaining.append(record)

    self.assertLen(remaining, 1)
    self.assertEqual(barriers_data.keys(), remaining[0].keys())

  def test_reader_and_writer(self):
    instr_1 = instrument_protos.Instrument(
        interest_rate_swap=InterestRateSwap(
            metadata=InstrumentMetadata(id="instr_1")))
    instr_2 = instrument_protos.Instrument(
        interest_rate_swap=InterestRateSwap(
            metadata=InstrumentMetadata(id="instr_2")))
    temp_dir = self.create_tempdir()
    temp_file = path.join(temp_dir.full_path, "datafile.bin")
    with io.record_writer(
        temp_file, io.CompressionType.GZIP) as writer:
      writer.write(instr_1.SerializeToString())
      writer.write(instr_2.SerializeToString())

    reader = io.record_iterator(temp_file, io.CompressionType.GZIP)
    instr_list = []
    for el in reader:
      instr = instrument_protos.Instrument()
      instr.ParseFromString(el)
      instr_list.append(instr)

    # Check that the protos are the same
    self.assertProtoEquals(instr_list[0], instr_1)
    self.assertProtoEquals(instr_list[1], instr_2)

if __name__ == "__main__":
  tf.test.main()
