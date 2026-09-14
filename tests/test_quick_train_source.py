"""Cheap source-level checks for the quick-start path; no model/data downloads required."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class QuickTrainSourceTests(unittest.TestCase):
    def test_quick_trainer_parses(self):
        ast.parse((ROOT / 'scripts/train_quick.py').read_text())

    def test_model_only_download_flag_exists(self):
        source = (ROOT / 'scripts/download.py').read_text()
        self.assertIn("'--model-only'", source)
        self.assertIn('if not args.model_only:', source)

    def test_quick_checkpoint_is_chat_compatible(self):
        source = (ROOT / 'scripts/train_quick.py').read_text()
        for required in [
            "'schema_version': 2",
            "'adapter_sha256': sha256(checkpoint)",
            "'graph_sha256': sha256(GRAPH / 'manifest.json')",
            "'system_text': cfg.SYSTEM_TEXT",
            "'interface': cfg.INTERFACE",
        ]:
            self.assertIn(required, source)


if __name__ == '__main__':
    unittest.main()
