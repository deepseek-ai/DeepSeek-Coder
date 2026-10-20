"""CPU-only tests: python -m unittest discover -s tests -p test_eval_jsonl_encoding.py -v."""

import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SUITES = ("HumanEval", "MBPP", "LeetCode")


class JsonlEncodingTests(unittest.TestCase):
    def load_data(self, suite):
        path = ROOT / "Evaluation" / suite / "human_eval" / "data.py"
        spec = importlib.util.spec_from_file_location("eval_data_" + suite, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_utf8_gzip_is_independent_of_default_text_encoding(self):
        original_wrapper = io.TextIOWrapper

        def ascii_default(buffer, encoding=None, errors=None, newline=None):
            selected = "ascii" if encoding in (None, "locale") else encoding
            return original_wrapper(buffer, encoding=selected, errors=errors, newline=newline)

        for suite in SUITES:
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "samples.jsonl.gz"
                rows = [{"task_id": "unicode/0", "prompt": "\u4f60\u597d \U0001f600"}]
                with gzip.open(path, "wt", encoding="utf-8") as stream:
                    stream.write(json.dumps(rows[0], ensure_ascii=False) + "\n\n")
                module = self.load_data(suite)
                # Emulate a non-UTF-8 locale at the standard-library I/O boundary.
                # Explicit encodings are honored; the actual reader remains unchanged.
                with patch.object(gzip.io, "TextIOWrapper", ascii_default):
                    self.assertEqual(list(module.stream_jsonl(str(path))), rows)

    def test_plain_utf8_jsonl(self):
        for suite in SUITES:
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "samples.jsonl"
                rows = [{"task_id": "unicode/0", "prompt": "\u4f60\u597d \U0001f600"}]
                path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n\n", encoding="utf-8")
                self.assertEqual(list(self.load_data(suite).stream_jsonl(str(path))), rows)

    def test_gzip_write_and_append_round_trip(self):
        for suite in SUITES:
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as temp:
                path = str(Path(temp) / "samples.jsonl.gz")
                module = self.load_data(suite)
                first = {"task_id": "unicode/0", "prompt": "\u4f60\u597d"}
                second = {"task_id": "unicode/1", "prompt": "\U0001f600"}
                module.write_jsonl(path, [first])
                module.write_jsonl(path, [second], append=True)
                self.assertEqual(list(module.stream_jsonl(path)), [first, second])


if __name__ == "__main__":
    unittest.main()
