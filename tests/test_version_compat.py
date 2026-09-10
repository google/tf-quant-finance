import importlib.util
import unittest
from pathlib import Path


module_path = Path(__file__).parents[1] / "tf_quant_finance" / "_version.py"
module_spec = importlib.util.spec_from_file_location("tff_version", module_path)
version_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(version_module)
version_tuple = version_module.version_tuple


class VersionCompatibilityTests(unittest.TestCase):
    def test_release_versions(self):
        self.assertEqual(version_tuple("2.3.0"), (2, 3, 0))
        self.assertEqual(version_tuple("2.15.1"), (2, 15, 1))

    def test_prerelease_versions(self):
        self.assertEqual(version_tuple("2.3.0-dev20240101"), (2, 3, 0))

    def test_invalid_versions_are_rejected(self):
        with self.assertRaises(ValueError):
            version_tuple("tensorflow")


if __name__ == "__main__":
    unittest.main()
