import importlib.util
import unittest
from pathlib import Path


def _load_parser():
    path = Path(__file__).resolve().parent / "parser.py"
    spec = importlib.util.spec_from_file_location("pal_parser_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExtractAnswerTest(unittest.TestCase):
    def test_a_word_containing_and_keeps_those_letters(self):
        extract = _load_parser().extract_answer
        self.assertEqual(extract("the answer is thousand"), "thousand")
        self.assertEqual(extract("the answer is sand"), "sand")
        self.assertEqual(extract("the answer is standard"), "standard")

    def test_a_plain_number_and_a_fraction_stay(self):
        extract = _load_parser().extract_answer
        self.assertEqual(extract("the answer is 1000"), "1000")
        self.assertEqual(extract("\\boxed{1/2}"), "\\frac{1}{2}")


if __name__ == "__main__":
    unittest.main()
