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
"""Tests for framework utility functions."""

import hashlib
import importlib.util
import json
import pathlib
import unittest


_UTILS_PATH = pathlib.Path(__file__).with_name("utils.py")
_SPEC = importlib.util.spec_from_file_location("framework_utils", _UTILS_PATH)
utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(utils)


class UtilsTest(unittest.TestCase):

  def test_hasher_uses_sha256(self):
    payload = {
        "bank_holidays": ["2026-01-01", "2026-12-25"],
        "business_day_convention": "MODIFIED_FOLLOWING",
        "currency": "USD",
    }

    expected = hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    self.assertEqual(expected, utils.hasher(payload))
    self.assertEqual(64, len(utils.hasher(payload)))


if __name__ == "__main__":
  unittest.main()
