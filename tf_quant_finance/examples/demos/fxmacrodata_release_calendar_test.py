"""Offline tests for the HTTP/pagination helpers; no tensor operations needed."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
import traceback


def load_demo():
    spec = importlib.util.spec_from_file_location(
        "fxmacrodata_demo", Path(__file__).with_name("fxmacrodata_release_calendar.py")
    )
    module = importlib.util.module_from_spec(spec)
    # The import-only placeholder keeps unrelated TensorFlow initialization out
    # of these standard-library transport tests. Tensor helpers are not tested.
    names = ("tensorflow", "tensorflow.compat", "tensorflow.compat.v2")
    with patch.dict(sys.modules, {name: types.ModuleType(name) for name in names}):
        spec.loader.exec_module(module)
    return module


demo = load_demo()


class TransportTests(unittest.TestCase):
    def fetch(self, payload):
        response = io.BytesIO(json.dumps(payload).encode())
        with patch.object(demo._OPENER, "open", return_value=response):
            result = demo.fetch_fxmacrodata("calendar/usd", api_key="test-key")
        self.assertTrue(response.closed)
        return result

    def test_valid_response(self):
        self.assertEqual(
            self.fetch({"data": [{"release": "cpi"}]}), {"data": [{"release": "cpi"}]}
        )

    def test_error_envelope_with_data_rejected(self):
        with self.assertRaises(ValueError):
            self.fetch({"error": "failed", "data": []})

    def test_detail_envelope_with_data_rejected(self):
        with self.assertRaises(ValueError):
            self.fetch({"detail": "failed", "data": []})

    def test_transport_failure_sanitized(self):
        with patch.object(
            demo._OPENER, "open", side_effect=URLError("sensitive upstream text")
        ):
            try:
                demo.fetch_fxmacrodata("calendar/usd")
            except ValueError as error:
                self.assertNotIn(
                    "sensitive upstream text",
                    "".join(traceback.format_exception(error)),
                )
            else:
                self.fail("Expected sanitized transport error")

    def test_http_failure_has_no_private_context(self):
        error = HTTPError(
            "https://example.invalid/", 403, "sensitive upstream text", {}, io.BytesIO()
        )
        with patch.object(demo._OPENER, "open", side_effect=error):
            with self.assertRaises(ValueError) as caught:
                demo.fetch_fxmacrodata("calendar/usd")
        self.assertTrue(caught.exception.__suppress_context__)

    def test_invalid_json_sanitized_and_closed(self):
        response = io.BytesIO(b"invalid")
        with patch.object(demo._OPENER, "open", return_value=response):
            with self.assertRaises(ValueError) as caught:
                demo.fetch_fxmacrodata("calendar/usd")
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertTrue(response.closed)

    def test_redirects_not_followed(self):
        self.assertIsNone(
            demo._NoRedirect().redirect_request(
                None, None, 302, "Found", {}, "https://other.invalid/"
            )
        )

    def test_control_character_key_rejected(self):
        with self.assertRaises(ValueError):
            demo.fetch_fxmacrodata("calendar/usd", api_key="test\x00key")


class PaginationTests(unittest.TestCase):
    def pages(self, payloads, limit=None):
        with patch.object(demo, "fetch_fxmacrodata", side_effect=payloads) as fetch:
            rows = demo.fetch_fxmacrodata_rows("announcements/usd/cpi", limit=limit)
        return rows, fetch

    def test_two_pages(self):
        rows, fetch = self.pages(
            [
                {
                    "data": [{"value": 1}],
                    "pagination": {"has_more": True, "next_offset": 1},
                },
                {"data": [{"value": 2}], "pagination": {"has_more": False}},
            ]
        )
        self.assertEqual(rows, [{"value": 1}, {"value": 2}])
        self.assertEqual(fetch.call_args_list[1].args[1]["offset"], 1)

    def test_requested_limit_bounds_rows(self):
        rows, fetch = self.pages([{"data": [{"value": 1}, {"value": 2}]}], limit=1)
        self.assertEqual(rows, [{"value": 1}])
        self.assertEqual(fetch.call_count, 1)

    def test_repeated_offset_rejected(self):
        with self.assertRaises(ValueError):
            self.pages(
                [{"data": [{}], "pagination": {"has_more": True, "next_offset": 0}}]
            )

    def test_negative_offset_rejected(self):
        with self.assertRaises(ValueError):
            self.pages(
                [{"data": [{}], "pagination": {"has_more": True, "next_offset": -1}}]
            )

    def test_string_offset_rejected(self):
        with self.assertRaises(ValueError):
            self.pages(
                [{"data": [{}], "pagination": {"has_more": True, "next_offset": "1"}}]
            )

    def test_boolean_offset_rejected(self):
        with self.assertRaises(ValueError):
            self.pages(
                [{"data": [{}], "pagination": {"has_more": True, "next_offset": True}}]
            )

    def test_empty_more_page_rejected(self):
        with self.assertRaises(ValueError):
            self.pages(
                [{"data": [], "pagination": {"has_more": True, "next_offset": 1}}]
            )

    def test_page_cap_does_not_return_partial_rows(self):
        with patch.object(demo, "FXMACRODATA_MAX_PAGES", 1):
            with self.assertRaises(ValueError):
                self.pages(
                    [{"data": [{}], "pagination": {"has_more": True, "next_offset": 1}}]
                )

    def test_missing_offset_advances_by_page_length(self):
        rows, fetch = self.pages(
            [{"data": [{}], "pagination": {"has_more": True}}, {"data": []}]
        )
        self.assertEqual(rows, [{}])
        self.assertEqual(fetch.call_args_list[1].args[1]["offset"], 1)

    def test_missing_rows_rejected(self):
        with self.assertRaises(ValueError):
            demo.payload_rows({"currency": "USD"})

    def test_invalid_rows_rejected(self):
        with self.assertRaises(ValueError):
            demo.payload_rows({"data": [{"value": 1}, "invalid"]})

    def test_dictionary_catalogue_preserved(self):
        self.assertEqual(
            demo.payload_rows({"data": {"cpi": {"name": "CPI"}}}),
            [{"indicator": "cpi", "name": "CPI"}],
        )

    def test_bad_pagination_rejected(self):
        with self.assertRaises(ValueError):
            self.pages([{"data": [{}], "pagination": "invalid"}])

    def test_string_has_more_rejected(self):
        with self.assertRaises(ValueError):
            self.pages([{"data": [{}], "pagination": {"has_more": "false"}}])

    def test_explicit_last_page(self):
        rows, _ = self.pages([{"data": [{}], "pagination": {"has_more": False}}])
        self.assertEqual(rows, [{}])


if __name__ == "__main__":
    unittest.main()
