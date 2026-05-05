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
| CodeShell      | 7B   | 15.8% | 8.6%  | 17.3%    | 35.5% | 28.2%  | 44.4% | 59.8% | 29.9% |
| CodeGeex-2     | 7B   | 22.2% | 9.7%  | 23.6%    | 39.0% | 44.6%  | 48.5% | 66.0% | 36.2% |
| StarCoder-Base | 16B  | 23.4% | 10.3% | 23.0%    | 42.4% | 45.0%  | 54.9% | 81.1% | 40.0% |
| CodeLLama-Base | 7B   | 31.2% | 12.1% | 30.2%    | 54.2% | 52.9%  | 59.6% | 82.6% | 46.1% |
| CodeLLama-Base | 13B  | 43.1% | 14.4% | 40.2%    | 59.2% | 60.3%  | 63.6% | 85.3% | 52.3% |
| CodeLLama-Base | 34B  | 58.2% | 21.2% | 51.8%    | 70.3% | 69.8%  | 70.7% | 91.8% | 62.0% |
|                |      |       |       |          |       |        |       |       |       |
| DeepSeek-Coder-Base  | 1.3B   | 14.6% | 16.8% | 14.5%    | 36.7% | 30.0%  | 48.2% | 62.3% | 31.9% |
| DeepSeek-Coder-MQA-Base  | 5.7B   | 38.8% | 20.0% | 36.8%    | 52.5% | 55.9%  | 63.9% | 84.8% | 50.4% |
| DeepSeek-Coder-Base  | 6.7B   | 43.2% | 19.2% | 40.3%    | 58.4% | 67.9%  | 67.2% | 87.0% | 54.7% |
| DeepSeek-Coder-Base  | 33B  | **60.7%** | **29.1%** | **54.1%**    | **71.6%** | **75.3%** | **76.7%** | **93.3%** | **65.8%** |


