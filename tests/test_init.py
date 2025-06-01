from importlib.metadata import PackageNotFoundError
from unittest.mock import patch


class TestPackageVersion:
    def test_version_fallback(self):
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            import importlib

            importlib.reload(importlib.import_module("ttfemesh"))
            from ttfemesh import __version__

            assert __version__ == "unknown"
