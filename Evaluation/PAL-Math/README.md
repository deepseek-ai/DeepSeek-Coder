## 1. Introduction

We provide a test script to evaluate the capability of the **deepseek-coder** model to solve mathematical problems using external tools (Python interpreter). We evaluate it using the [PAL](https://arxiv.org/pdf/2211.10435.pdf) method on seven datasets: **GSM8k, MATH, GSM-Hard, SVAMP, TabMWP, ASDiv, and MAWPS**.



## 2. Setup

```
pip install sympy==1.12 pebble timeout-decorator transformers
```



## 3. Evaluation

We provide an example of testing the **deepseek-coder-1b-python** model on the **gsm8k** dataset using **8** GPUs. If you wish to use a different model or dataset, you can modify it as needed.

```bash
MODEL_NAME_OR_PATH=deepseek/deepseek-coder-1b-python
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



#### (1) Multilingual Base Models

| Model          | Size | GSM8k | MATH  | GSM-Hard | SVAMP | TabMWP | ASDiv | MAWPS | Avg   |
| -------------- | ---- | ----- | ----- | -------- | ----- | ------ | ----- | ----- | ----- |
| CodeShell      | 7B   | 17.0% | 9.1%  | 18.2%    | 45.6% | 29.6%  | 46.6% | 56.8% | 31.8% |
| CodeGeex-2     | 7B   | 23.6% | 9.6%  | 22.4%    | 48.0% | 47.2%  | 46.9% | 66.0% | 37.7% |
| StarCoder-Base | 16B  | 27.3% | 11.5% | 24.2%    | 44.0% | 45.6%  | 54.9% | 73.4% | 40.1% |
| CodeLLama-Base | 7B   | 36.4% | 12.3% | 29.7%    | 57.6% | 58.4%  | 59.6% | 82.6% | 48.0% |
| CodeLLama-Base | 13B  | 44.2% | 15.5% | 42.4%    | 65.6% | 61.6%  | 65.3% | 85.3% | 54.3% |
| CodeLLama-Base | 34B  | 58.2% | 22.1% | 55.2%    | 77.2% | 69.6%  | 70.0% | 92.8% | 63.6% |
|                |      |       |       |          |       |        |       |       |       |
| OraCoder-Base  | 1B   | 17.0% | 13.4% | 13.3%    | 39.2% | 42.4%  | 44.8% | 66.0% | 33.7% |
| OraCoder-Base  | 7B   | 46.0% | 20.6% | 40.0%    | 67.2% | 71.2%  | 67.1% | 89.1% | 57.3% |
| OraCoder-Base  | 33B  | -     | -     | -        | -     | -      | -     | -     | -     |

#### (2) Python Base Models

| Model          | Size | GSM8k | MATH  | GSM-Hard | SVAMP | TabMWP | ASDiv | MAWPS | Avg   |
| -------------- | ---- | ----- | ----- | -------- | ----- | ------ | ----- | ----- | ----- |
| StarCoder          | 16B  | 31.5% | 13.8% | 26.7%    | 48.8% | 47.2%  | 54.9% | 76.1% | 42.7% |
| CodeLLama-Python   | 7B   | 35.2% | 14.7% | 34.5%    | 70.4% | 55.2%  | 62.1% | 84.2% | 50.9% |
| CodeLLama-Python   | 13B  | 44.8% | 17.4% | 45.5%    | 65.6% | 60.8%  | 69.0% | 89.6% | 56.1% |
| CodeLLama-Python   | 34B  | 57.6% | 21.1% | 54.5%    | 76.8% | 66.8%  | 69.5% | 94.2% | 62.9% |
|  |  |  |  |  |  |  |  |  |  |
| OraCoder-Python    | 1B   | 17.6% | 15.0% | 18.2%    | 40.0% | 38.4%  | 49.5% | 64.1% | 34.7% |
| OraCoder-Python    | 7B   | 50.3% | 24.3% | 43.0%    | 71.2% | 73.6%  | 69.7% | 88.0% | 60.0% |
| OraCoder-Python    | 33B  | -     | -     | -        | -     | -      | -     | -     | -     |

#### (3) Instruction-Tuned Models
| Model          | Size | GSM8k | MATH  | GSM-Hard | SVAMP | TabMWP | ASDiv | MAWPS | Avg   |
| -------------- | ---- | ----- | ----- | -------- | ----- | ------ | ----- | ----- | ----- |
| ChatGPT            | -    | 78.6% | 38.7% | 67.6%    | 77.8% | 79.9%  | 81.0% | 89.4% | 73.3% |
| GPT-4              | -    | 94.2% | 51.8% | 77.6%    | 94.8% | 95.9%  | 92.6% | 97.7% | 86.4% |
|               |      |       |       |          |       |        |       |       |       |
| OraCoder-Chat      | 1B   | -     | -     | -        | -     | -      | -     | -     | -     |
| OraCoder-Chat      | 7B   | -     | -     | -        | -     | -      | -     | -     | -     |
| OraCoder-Chat      | 33B  | -     | -     | -        | -     | -      | -     | -     | -     |

