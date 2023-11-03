## 1. Introduction

We provide a test script to evaluate the performance of the **deepseek-coder** model on code generation benchmarks, [**MBPP**](https://huggingface.co/datasets/mbpp), with 3-shot setting.



## 2. Setup

```
pip install accelerate
pip install attrdict
pip install transformers
pip install pytorch
```



## 3. Evaluation

We've created a sample script, **eval.sh**, that demonstrates how to test the **deepseek-coder-1.3b-base** model on the MBPP dataset leveraging **8** GPUs.

```bash
MODEL_NAME_OR_PATH="deepseek-ai/deepseek-coder-1.3b-base"
DATASET_ROOT="data/"
LANGUAGE="python"
python -m accelerate.commands.launch --config_file test_config.yaml eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --dataroot ${DATASET_ROOT} 
```

## 4. Experimental Results

We report experimental results here for several models. We set the maximum input length to **4096** and the maximum output length to **500**, and employ the **greedy search strategy**.



#### (1) Multilingual Base Models

| Model             | Size | Pass@1 | 
|-------------------|------|--------|
| CodeShell         | 7B   | 38.6%  | 
| CodeGeeX2         | 6B   | 36.2%  |
| StarCoder     | 16B  | 42.8%  | 
| CodeLLama-Base   | 7B   | 38.6%  | 
| CodeLLama-Base    | 13B  | 47.0%  | 
| CodeLLama-Base    | 34B  | 55.0%  | 
| | | | |  |  |  |  |  |  | |
| DeepSeek-Coder-Base| 1.3B   | 46.8%  |
| DeepSeek-Coder-Base| 5.7B   | 57.2%  | 
| DeepSeek-Coder-Base| 6.7B   | 60.6%  | 
| DeepSeek-Coder-Base|33B  | **66.0%**  |

#### (2) Instruction-Tuned Models
| Model               | Size | Pass@1  |
|---------------------|------|--------|
| GPT-3.5-Turbo            | -    | 70.8%  | 
| GPT-4               | -    | **80.0%**  |
| | | | |  |  |  |  |  |  | |
| DeepSeek-Coder-Instruct | 1.3B  | 49.4%      |
| DeepSeek-Coder-Instruct  | 6.7B  | 65.4%     |
| DeepSeek-Coder-Instruct  | 33B | **70.0%**     | 


