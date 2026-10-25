import sys
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mock heavy optional dependencies so tests run lightweight anywhere
sys.modules.setdefault("torch", mock.MagicMock())
sys.modules.setdefault("transformers", mock.MagicMock())
sys.modules.setdefault("human_eval", mock.MagicMock())
sys.modules.setdefault("human_eval.evaluation", mock.MagicMock())
sys.modules.setdefault("tqdm", mock.MagicMock())

# Import HumanEval utils
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "HumanEval"))
from utils.utils import extract_generation_code, get_function_name

# Import MBPP eval
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "MBPP"))
from eval_instruct import convert_for_evaluation

# Import LeetCode eval
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "LeetCode"))
from evaluate_leetcode import extract_python_code


def test_get_function_name_indented_and_standard():
    prompt_std = "def add(a, b):\n    pass"
    name, prefix = get_function_name(prompt_std, "python")
    assert name == "add"
    assert prefix == ""

    prompt_indented = "    def multiply(x: int, y: int) -> int:\n        pass"
    name_ind, _ = get_function_name(prompt_indented, "python")
    assert name_ind == "multiply"

    prompt_cpp = "int sum(int a, int b) {\n"
    name_cpp, _ = get_function_name(prompt_cpp, "cpp")
    assert "sum" in name_cpp


def test_extract_generation_code_crlf():
    example = {
        "task_id": "Python/0",
        "prompt": "def add(a, b):\n",
        "output": "```python\r\ndef add(a, b):\r\n    return a + b\r\n```",
    }
    res = extract_generation_code(example, "python")
    assert "return a + b" in res["generation"]


def test_extract_generation_code_trailing_whitespace_and_unclosed():
    example = {
        "task_id": "Python/1",
        "prompt": "def solve(n):\n",
        "output": "Here is the code:\n```python   \ndef solve(n):\n    return n * 2",
    }
    res = extract_generation_code(example, "python")
    assert "return n * 2" in res["generation"]


def test_extract_generation_code_no_negative_index_underflow():
    # When code_block ends with a space, start - indent must not check code_block[-1]
    example = {
        "task_id": "Python/2",
        "prompt": "def zero():\n",
        "output": "```python\ndef zero():\n    return 0 ",
    }
    res = extract_generation_code(example, "python")
    assert "return 0" in res["generation"]


def test_extract_generation_code_generic_fence():
    example = {
        "task_id": "Python/3",
        "prompt": "def compute():\n",
        "output": "```\ndef compute():\n    return 42\n```",
    }
    res = extract_generation_code(example, "python")
    assert "return 42" in res["generation"]


def test_mbpp_convert_for_evaluation():
    # CRLF support
    ex_crlf = {"gpt_completion": "```python\r\ndef foo():\r\n    return True\r\n```"}
    res_crlf = convert_for_evaluation(ex_crlf)
    assert "return True" in res_crlf["generation"]

    # Unclosed block support
    ex_unclosed = {"gpt_completion": "```python\ndef bar():\n    return False"}
    res_unclosed = convert_for_evaluation(ex_unclosed)
    assert "return False" in res_unclosed["generation"]


def test_leetcode_extract_python_code():
    # Empty string
    assert extract_python_code("") == ""

    # CRLF code block
    gen_crlf = "```python\r\ndef twoSum(nums, target):\r\n    return [0, 1]\r\n```"
    assert "return [0, 1]" in extract_python_code(gen_crlf)

    # Unclosed code block
    gen_unclosed = "```python\ndef twoSum(nums, target):\n    return [0, 1]"
    assert "return [0, 1]" in extract_python_code(gen_unclosed)

    # Non-code block fallback
    gen_raw = "def twoSum(nums, target):\n    return []\ndef helper():\n    pass"
    assert "return []" in extract_python_code(gen_raw)
