import re
import json
from pathlib import Path
from collections import defaultdict
from human_eval.evaluation import evaluate_functional_correctness

version = "20240121-Jul"

DATA_DIR = Path(__file__).parent / "data"

def extract_python_code(generation: str):
    generation = generation.replace("[PYTHON]", '```python').replace("[/PYTHON]", '```')
    if '```python' in generation:
        p_code = re.compile(r'```python\n(.*?)\n```', flags=re.DOTALL)
        code_block = p_code.findall(generation)[0]
        return code_block
    else:
        codelist = re.split("\ndef|\nclass|\nif|\n#|\nprint", generation)
        return codelist[0]
    
def evaluate_main(generation_path: str, result_path: str, temp_dir: str):
    problem_path = (DATA_DIR / f"{version}.jsonl").as_posix()

    print(problem_path)
    problems = [json.loads(line) for line in open(problem_path, 'r')]

    id2problems = { x['task_id']: x for x in problems }

    results = [json.loads(line) for line in open(generation_path, 'r')]
    for result in results:
        if 'task_id' not in result:
            result['task_id'] = problems[result['index']]['task_id']

        if 'generation' not in result:
            try:
                if 'output' not in result:
                    result['output'] = result['response']
                if result['output'].startswith("\n        "):
                    func_code = extract_python_code(result['prompt_sft']).strip()
                    result['generation'] = func_code + '\n' + result['output']
                else:
                    result['generation'] = extract_python_code(result['output'])
            except:
                result['generation'] = result['output']
    
    with open(result_path, 'w') as fr:
        for result in results:
            fr.write(json.dumps(result) + "\n")

    score = evaluate_functional_correctness(
        input_file=result_path,
        tmp_dir=temp_dir,
        problem_file=problem_path,
        result_path=result_path
    )

    hardness_results = defaultdict(int)
    for result in [json.loads(line) for line in open(result_path, 'r')]:
        problem = id2problems[result['task_id']]

        hardness = problem['meta']['difficulty']
        hardness_results[hardness] += 1
        hardness_results[hardness + "_correct"] += result['passed']

    print("="*100)
    print("Evaluate {} over.".format(generation_path))
    print("Pass@1: {:.3f}".format(score["pass@1"]))
    for key in ["Easy", "Medium", "Hard"]:
        if key.endswith("_correct"):
            continue
        acc = hardness_results[key+"_correct"] / hardness_results[key]
        print("{}: {:.3f}({}/{})".format(key, acc, hardness_results[key+"_correct"],  hardness_results[key]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_path", type=str, required=True)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--temp_dir", type=str, default="output/temp")
    args = parser.parse_args()

    if args.result_path is None:
        args.result_path = args.generation_path.replace(".jsonl", "_result.jsonl")
    
    evaluate_main(args.generation_path, args.result_path, temp_dir=args.temp_dir)
    pass
