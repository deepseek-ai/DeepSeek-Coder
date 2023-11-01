## 1. Introduction

We provide a test script to evaluate the performance of the **deepseek-coder** model on code generation benchmarks. We select the widely-used benchmarks: **[HumanEval-Python](https://huggingface.co/datasets/openai_humaneval), [HumanEval-Multilingual](https://huggingface.co/datasets/nuprl/MultiPL-E)**.



## 2. Setup

```
pip install accelerate
pip install attrdict
pip install transformers
pip install pytorch
```


## 3. Evaluation

We've created a sample script, **eval.sh**, that demonstrates how to test the **deepseek-coder-1b-python** model on the HumanEval dataset leveraging **8** GPUs. If your use case involves a different model or dataset, simply adjust the script to fit your needs.

Additionally, for various programming languages, the execution path may differ. Please ensure you update the appropriate paths in the **humaneval/execution.py** file accordingly.

```bash
MODEL_NAME_OR_PATH="deepseek/deepseek-coder-1b-python"
DATASET_ROOT="data/"
LANGUAGE="python"
python -m accelerate.commands.launch --config_file test_config.yaml eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --language ${LANGUAGE} --dataroot ${DATASET_ROOT} 
```



## 4. Experimental Results

We report experimental results here for 8 main-stream programming languages, **python**, **c++**, **java**, **PHP**, **TypeScript**, **C#**, **Bash**, and **JavaScript**. For all open-source models, we utilize this repository to obtain the performance of the models on the HumanEval dataset. We set the maximum input length to **4096** and the maximum output length to **500**, and employ the **greedy search strategy**.


#### (1) Multilingual Base Models

| Model             | Size | Python | C++   | Java | PHP  | TS   | C#   | Bash | JS   | Avg  |
|-------------------|------|--------|-------|------|------|------|------|------|------|------|
| code-cushman-001  | 12B  | 33.5%  | 31.9% | 30.6%| 28.9%| 31.3%| 22.1%| 11.7%| -    | -    |
| CodeShell         | 7B   | 35.4%  | 32.9% | 34.2%| 31.7%| 30.2%| 38.0%| 7.0% | 33.5%| 30.4%|
| CodeGeeX2         | 6B   | 36.0%  | 29.2% | 25.9%| 23.6%| 20.8%| 29.7%| 6.3% | 24.8%| 24.5%|
| StarCoderBase     | 16B  | 31.7%  | 31.1% | 28.5%| 25.4%| 34.0%| 34.8%| 8.9% | 29.8%| 28.0%|
| CodeLLama (7B)    | 7B   | 31.7%  | 29.8% | 34.2%| 23.6%| 36.5%| 36.7%| 12.0%| 29.2%| 29.2%|
| CodeLLama (13B)   | 13B  | 36.0%  | 37.9% | 38.0%| 34.2%| 45.2%| 43.0%| 16.5%| 32.3%| 35.4%|
| CodeLLama (34B)   | 34B  | 48.2%  | 44.7% | 44.9%| 41.0%| 42.1%| 48.7%| 15.8%| 42.2%| 41.0%|
| | | | |  |  |  |  |  |  | |
| OraCoder-Base (1B)| 1B   | 34.8%  | 31.1% | 32.3%| 24.2%| 28.9%| 36.7%| 10.1%| 28.6%| 28.3%|
| OraCoder-Base (7B)| 7B   | 49.4%  | 50.3% | 43.0%| 38.5%| 49.7%| 50.0%| 28.5%| 48.4%| 44.7%|
| OraCoder-Base (33B)|33B  | -      | -     | -    | -    | -    | -    | -    | -    | -    |

#### (3) Instruction-Tuned Models
| Model               | Size | Python | C++   | Java | PHP  | TS   | C#   | Bash | JS   | Avg  |
|---------------------|------|--------|-------|------|------|------|------|------|------|------|
| GPT35-turbo         | -    | 76.2%  | 63.4% | 69.2%| 60.9%| 69.1%| 70.8%| 42.4%| 67.1%| 64.9%|
| GPT-4               | -    | 84.1%  | 76.4% | 81.6%| 77.2%| 77.4%| 79.1%| 58.2%| 78.0%| 76.5%|
| WizardCoder         | 16B  | 51.8%  | 41.6% | 41.1%| 42.2%| 44.7%| 46.8%| 12.7%| 42.8%| 40.5%|
| Phind-CodeLlama     | 34B  | -      | -     | -    | -    | -    | -    | -    | -    | -    |
| | | | |  |  |  |  |  |  | |
| OraCoder-Chat (1B)  | 1B  | -      | -     | -    | -    | -    | -    | -    | -    | -    |
| OraCoder-Chat (7B)  | 7B  | 78.9%  | 63.4% | 68.4% | 68.9%| 67.2%| 72.8%| 36.7%| 72.7%| 66.1%|
| OraCoder-Chat (33B) | 33B | 79.3%  | 68.9% | 73.4% | 72.7%| 67.9%| 74.1%| 43.0%| 73.9%| 69.2%|

