import tempfile
import unittest
from pathlib import Path

from fitten2api.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_server_and_fitten_options_from_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                """
[server]
host = "0.0.0.0"
port = 9000
api_key = "secret"

[fitten]
chat_endpoint = "/codeapi/chat_auth"
models_base_url = "https://api.example.test"
models_endpoint = "/codeapi/chat/models"
fetch_models = false
""",
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config.server.api_key, "secret")
        self.assertEqual(config.server.port, 9000)
        self.assertEqual(config.fitten.models_base_url, "https://api.example.test")
        self.assertFalse(config.fitten.fetch_models)
        self.assertEqual([model.id for model in config.models], ["Default", "DeepSeek V3", "DeepSeek R1", "Default (Agent)"])


if __name__ == "__main__":
    unittest.main()
