## 1. Introduction

We provide a test script to evaluate the capability of the **deepseek-coder** model to solve mathematical problems using external tools (Python interpreter). We evaluate it using the [PAL](https://arxiv.org/pdf/2211.10435.pdf) method on seven datasets: **GSM8k, MATH, GSM-Hard, SVAMP, TabMWP, ASDiv, and MAWPS**.



## 2. Setup

```
pip install sympy==1.12 pebble timeout-decorator transformers
```



## 3. Evaluation

We provide an example of testing the **deepseek-coder-1.3b-base** model on the **gsm8k** dataset using **8** GPUs. If you wish to use a different model or dataset, you can modify it as needed.

```bash
MODEL_NAME_OR_PATH=deepseek-ai/deepseek-coder-1.3b-base
DATA=gsm8k # 'math' 'gsm8k' 'gsm-hard' 'svamp' 'tabmwp' 'asdiv' 'mawps'
MODEL_DIR_NAME=${MODEL_NAME_OR_PATH##*/}
GPU_NUM=8
for rank in {0..7}; do
    CUDA_VISIBLE_DEVICES=$rank nohup  python run.py \
    --data_name ${DATA} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --batch_size 16 \
    --do_inference \
    --rank $rank \
    --world_size $GPU_NUM 2>&1 &
done

# Wait for all processes to finish
wait
echo "All processes completed."
python run.py --do_eval --data_name ${DATA}  --model_name_or_path ${MODEL_NAME_OR_PATH}  --world_size $GPU_NUM | tee outputs/${MODEL_DIR_NAME}/${DATA}/result.out
```



## 4. Experimental Results

We report experimental results here for mathematical reasoning tasks by using python program. For all open-source models, we utilize this repository and test with the same prompt. We set the maximum input length to **2048** and the maximum output length to **512**, and employ the **greedy search strategy**.




| Model          | Size | GSM8k | MATH  | GSM-Hard | SVAMP | TabMWP | ASDiv | MAWPS | Avg   |
| -------------- | ---- | ----- | ----- | -------- | ----- | ------ | ----- | ----- | ----- |
| CodeShell      | 7B   | 17.0% | 9.1%  | 18.2%    | 45.6% | 29.6%  | 46.6% | 56.8% | 31.8% |
| CodeGeex-2     | 7B   | 23.6% | 9.6%  | 22.4%    | 48.0% | 47.2%  | 46.9% | 66.0% | 37.7% |
| StarCoder-Base | 16B  | 27.3% | 11.5% | 24.2%    | 44.0% | 45.6%  | 54.9% | 73.4% | 40.1% |
| CodeLLama-Base | 7B   | 36.4% | 12.3% | 29.7%    | 57.6% | 58.4%  | 59.6% | 82.6% | 48.0% |
| CodeLLama-Base | 13B  | 44.2% | 15.5% | 42.4%    | 65.6% | 61.6%  | 65.3% | 85.3% | 54.3% |
| CodeLLama-Base | 34B  | 58.2% | 22.1% | **55.2%**    | 77.2% | 69.6%  | 70.0% | 92.8% | 63.6% |
|                |      |       |       |          |       |        |       |       |       |
| DeepSeek-Coder-Base  | 1.3B   | 15.8% | 16.3% | 14.5%    | 38.4% | 28.8%  | 51.3% | 66.0% | 33.0% |
| DeepSeek-Coder-MQA-Base  | 5.7B   | 44.8% | 25.4% | 40.6%    | 56.8% | 62.4%  | 66.8% | 84.2% | 54.4% |
| DeepSeek-Coder-Base  | 6.7B   | 46.1% | 25.6% | 40.0%    | 67.2% | 71.2%  | 69.0% | 89.2% | 58.3% |
| DeepSeek-Coder-Base  | 33B  | **58.2%** | **35.3%** | 54.5%    | **78.4%** | **76.8%** | **78.2%** | **94.0%** | **67.9%** |


