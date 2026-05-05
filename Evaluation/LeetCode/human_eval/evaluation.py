import os
import json
import gzip
import numpy as np
import itertools

from typing import *
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from human_eval.data import stream_jsonl
from human_eval.execution import check_correctness

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
        "from functools import *"
    ],
    "go"   : [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp"   : [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
    ],
}


LANGUAGE_NAME = {
    "cpp"   : "CPP",
    "go"    : "Go",
    "java"  : "Java",
    "js"    : "JavaScript",
    "python": "Python",
}


def read_dataset(
    data_file: str = None,
    dataset_type: str = "humaneval",
    num_shot=None,
) -> Dict:
    if num_shot is not None:
        print(f"{num_shot}-shot setting...")
    if "humaneval" in dataset_type.lower():
        if data_file is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(current_path, "..", "humaneval-x", "python", "data", "humaneval_python.jsonl.gz")
        dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset

def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def process_humaneval_test(sample, problems, example_test=False, is_mbpp=False, language="python"):
    task_id = sample["task_id"]
    
    if is_mbpp:
        return sample["generation"] + "\n" + "\n".join(problems[task_id]["test"])
    #language = task_id.split("/")[0].lower()

    prompt = sample.get("prompt", "")
    if example_test and "example_test" in problems[task_id] and problems[task_id]["example_test"] != "":
        test = problems[task_id]["example_test"]
    else:
        test = problems[task_id]["test"]
    code = sample["generation"]

    # Pre-process for different languages
    if language == "python":
        '''code_ = []
        for line in code.split("\n"):
            if (len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t'):
                break
            code_.append(line)
        code = "\n".join(code_)'''
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        test_string = test_setup + code + "\n" + test + "\n"
    elif language == "cpp":
        test_set_up = ""
        for s in IMPORT_HELPER["cpp"]:
            if s not in prompt:
                test_set_up += s + "\n"
        test_string = test_set_up + "\n" + code + "\n" + test
    elif language == "java":
        test_string = code + "\n" + test
    elif language in ["js", "javascript", "ts", "cs", "sh"]:
        test_string = code + "\n" + test
    elif language == "go":
        import_string = problems[task_id]["import"]
        prompt = prompt.replace(import_string, "")
        if example_test and "example_test" in problems[task_id]:
            test = problems[task_id]["example_test"]
        else:
            test = problems[task_id]["test"]
        test_setup = problems[task_id]["test_setup"]
        other_pkgs = []
        for pkg in IMPORT_HELPER["go"]:
            if pkg not in test_setup:
                p = pkg.split("/")[-1]
                if p + "." in code:
                    other_pkgs.append(f"\"{pkg}\"")
        if other_pkgs:
            import_other_pkgs = "import (\n" + "    ".join([p + "\n" for p in other_pkgs]) + ")"
            test_string = test_setup + "\n" + import_other_pkgs + "\n" + prompt + code + "\n" + test
        else:
            test_string = test_setup + "\n" + prompt + code + "\n" + test
    elif language == "rust":
        main = "\nfn main(){ \n } \n"
        declaration = problems[task_id]["declaration"]
        test_string = main + declaration + prompt + code + test
    elif language == "php":
        test_string = code + "\n" + test + "?>"
    return test_string


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def evaluate_functional_correctness(
    input_file: str = None,
    tmp_dir: str = "./",
    n_workers: int = 32,
    timeout: float = 10.0,
    problem_file: str = "../data/humaneval_python.jsonl.gz",
    result_path: str = None,
    k: List[int] = [1, 10, 100],
    test_groundtruth: bool = False,
    example_test: bool = False,
    is_mbpp: bool = False,
    language: str = "python",
):
    if example_test:
        print("Example test...")

    problems = read_dataset(problem_file, dataset_type="humaneval")
    sample_jsonl = stream_jsonl_all(input_file)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        if test_groundtruth:
            print("Testing ground truth...")
            for sample in tqdm(problems.values()):
                task_id = sample["task_id"]
                lang = task_id.split("/")[0].lower()
                if lang == "javascript":
                    lang = "js"
                tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
                sample["generation"] = sample["canonical_solution"]
                sample["test_code"] = process_humaneval_test(sample, problems, example_test, language)
                if sample["test_code"] is None:
                    continue
                args = (task_id, sample, lang, timeout, tmp_dir_, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1
        else:
            print("Reading Samples...")
            id2samples = {}
            for sample in tqdm(sample_jsonl):
                task_id = sample["task_id"]

                if not is_mbpp:
                    lang = language
                if not is_mbpp and lang == "javascript":
                    lang = "js"
                if is_mbpp:
                    lang = "python"
                tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
                sample["task_id"] = task_id
                sample["test_code"] = process_humaneval_test(sample, problems, example_test, is_mbpp, language)
                if sample["test_code"] is None:
                    continue
                if "completion_id" in sample:
                    completion_id_ = sample["completion_id"]
                else:
                    completion_id_ = completion_id[task_id]
                args = (task_id, sample, lang, timeout, tmp_dir_, completion_id_)
                id2samples[(task_id, completion_id_)] = sample
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        if len(completion_id) == len(problems):
            evaluate_pass_at_k = True
        else:
            evaluate_pass_at_k = False

        print("Running test suites...")
        sample_with_results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

            sample = id2samples[(result["task_id"], result["completion_id"])]
            sample_with_results.append({
                'task_id': result['task_id'],
                'completion_id': result["completion_id"],
                'passed': result['passed'],
                'generation': sample['generation']
            })

            for key in sample:
                if key not in sample_with_results[-1]:
                    sample_with_results[-1][key] = sample[key]

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)
    if evaluate_pass_at_k:
        ks = k
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
            for k in ks if (total >= k).all()
        }
        print(pass_at_k)
    else:
        print("Total:", np.sum(total))
        print("Correct:", np.sum(correct))
    
    if result_path is not None:
        with open(result_path, 'w', encoding='utf-8') as fw:
            for sample_with_result in sample_with_results:
                fw.write(json.dumps(sample_with_result) + '\n')
            print("Save evaluation results to\n{}".format(result_path))

    return pass_at_k
