import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fitten2api.credentials import FittenCredentials, load_credentials, save_credentials


class CredentialTests(unittest.TestCase):
    def test_load_from_env_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "credentials.json"
            with patch.dict(
                os.environ,
                {
                    "FITTEN_ACCESS_TOKEN": "a",
                    "FITTEN_REFRESH_TOKEN": "r",
                    "FITTEN_USER_ID": "u",
                },
                clear=False,
            ):
                creds = load_credentials(path)
            self.assertEqual(creds, FittenCredentials("a", "r", "u"))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["access_token"], "a")

    def test_load_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "credentials.json"
            save_credentials(FittenCredentials("a", "r", "u"), path)
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_credentials(path, export=False).user_id, "u")


if __name__ == "__main__":
    unittest.main()
