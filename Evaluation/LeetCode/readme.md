## 1. Introduction
We construct the LeetCode Contest benchmark to to further validate the model's capability in real-world programming problems.
[LeetCode](https://leetcode.com/) presents competition-level problems, offering significant challenges that test the model's problem understanding and code generation skills. We collected the latest problems from LeetCode Contests to prevent the appearance of both the problems or their solutions in our pre-training data. A total of `180` problems were collected from July 2023 to January 2024. For each problem, we collected `100` test cases. The data format is the same as human-eval. For more details, please refer to [leetcode_contest_data](./data/20240121-Jul.jsonl).

## 2. Evaluation
Please follow the following two steps to evaluate the model's performance on our LeetCode Contest benchmark:

1. Run `vllm_inference.py` to get generation results.
```bash
cd Evaluation/LeetCode

# Set the model or path here
MODEL="deepseek-ai/deepseek-coder-7b-instruct"

python vllm_inference.py --model_name_or_path $MODEL --saved_path output/20240121-Jul.deepseek-coder-7b-instruct.jsonl
```

If you want to evaluate the model with COT, please add `--cot` to the command:
```bash
python vllm_inference.py --model_name_or_path $MODEL --saved_path output/20240121-Jul.deepseek-coder-7b-instruct.jsonl --cot
```

2. Run `evaluate_leetcode.py` to get evaluation results.
```bash
python evaluate_leetcode.py --generation_path output/20240121-Jul.deepseek-coder-7b-instruct.jsonl --result_path output/20240121-Jul.deepseek-coder-7b-instruct.result.jsonl
```

## 3. Experimental Results
We report experimental results here:

| Model                       | Size | Easy (45) | Medium (91) | Hard (44) | Overall(180) |
|-----------------------------|------|-----------|-------------|-----------|--------------|
| WizardCoder-V1.0            | 15B  | 17.8%     | 1.1%        | 0.0%      | 5.0%         |
| CodeLlama-Instruct          | 34B  | 24.4%     | 4.4%        | 4.5%      | 9.4%         |
| Phind-CodeLlama-V2          | 34B  | 26.7%     | 8.8%        | 9.1%      | 13.3%        |
| | | | |
| GPT-3.5-Turbo               | -    | 46.7%     | 15.4 %      | 15.9%     | 23.3%        |
| GPT-3.5-Turbo + CoT         | -    | 42.2%     | 15.4%       | 20.5%     | 23.3%        |
| GPT-4-Turbo                 | -    | 73.3%     | 31.9%       | 25.0%     | 40.6%        |
| GPT-4-Turbo + CoT           | -    | 71.1%     | 35.2%       | 25.0%     | 41.8%        |
| | | | |
| DeepSeek-Coder-Instruct     | 1.3B | 22.2%     | 1.1%        | 4.5%      | 7.2%         |
| DeepSeek-Coder-Instruct + CoT | 1.3B | 22.2%   | 2.2%        | 2.3%      | 7.2%         |
| DeepSeek-Coder-Instruct     | 6.7B | 44.4%     | 12.1%       | 9.1%      | 19.4%        |
| DeepSeek-Coder-Instruct + CoT | 6.7B | 44.4%   | 17.6%       | 4.5%      | 21.1%        |
| DeepSeek-Coder-Instruct     | 33B  | 57.8%     | 22.0%       | 9.1%      | 27.8%        |
| DeepSeek-Coder-Instruct + CoT | 33B | 53.3%    | 25.3%       | 11.4%     | 28.9%        |

