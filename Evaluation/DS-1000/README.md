## 1. Introduction

We provide a test script to evaluate the performance of the **deepseek-coder** model on code completion benchmarks. We select the widely-used benchmarks: [**DS-1000**](https://github.com/xlang-ai/DS-1000).

## 2. Evaluation

We directly use the scripts provided by the DS-1000 repository to evaluate the performance of the models. You can refer to [**DS-1000**](https://github.com/xlang-ai/DS-1000) to find more details about the evaluation.


## 3. Experimental Results

We report experimental results here for the completion mode of DS-1000. We set the maximum length to **2048**, and employ the **greedy search strategy**.  To ensure a fair comparison, we apply identical hyper-parameters across all open-source models under evaluation.

| Model                  | Size | Matplotlib | Numpy | Pandas | Pytorch | Scipy | Scikit-Learn | Tensorflow | Avg   |
|------------------------|------|------------|-------|--------|---------|-------|-------------|------------|-------|
| Codex-001              | -    | 41.8%      | 26.6% | 9.4%   | 9.7%    | 15.0% | 18.5%        | 17.2%      | 20.2% |
| Codex-002              | -    | **57.0%**      | 43.1% | **26.5%**  | **41.8%**   | 31.8% | **44.8%**        | 39.3%      | 39.2% |
| CodeShell              | 7B   | 34.1%      | 21.8% | 10.7%  | 11.8%   | 17.0% | 20.0%        | 15.6%      | 18.8% |
| CodeGeeX2              | 6B   | 38.7%      | 26.8% | 14.4%  | 11.8%   | 19.8% | 27.0%        | 17.8%      | 22.9% |
| StarCoder         | 16B  | 47.7%      | 31.4% | 12.7%  | 25%   | 22.6% | 35.7%        | 22.2%      | 27.2% |
| CodeLLama-Base         | 7B   | 41.9%      | 24.6% | 14.8%  | 16.2%   | 18.9% | 17.4%        | 17.8%      | 22.1% |
| CodeLLama-Base         | 13B  | 46.5%      | 28.6% | 18.2%  | 19.1%   | 18.9% | 27.8%        | 33.3%      | 26.8% |
| CodeLLama-Base         | 34B  | 50.3%      | 42.7% | 23.0%  | 25.0%   | 28.3% | 33.9%        | 40.0%      | 34.3% |
| | | | |  |  |  |  |  |  | |
| DeepSeek-Coder-Base    | 1.3B   | 32.3%      | 21.4% | 9.3%   | 8.8%    | 8.5%  | 16.5%        | 8.9%       | 16.2% |
| DeepSeek-Coder-Base    | 5.7B   | 51.1%      | 31.8% | 19.9%  | 14.7%   | 17.0% | 29.6%        | 15.6%      | 27.7% |
| DeepSeek-Coder-Base    | 6.7B   | 48.4%      | 35.5% | 20.6%  | 19.1%   | 22.6% | 38.3%        | 24.4%      | 30.5% |
| DeepSeek-Coder-Base    | 33B  | 56.1%      | **49.6%** | 25.8%  | 36.8%   | **36.8%** | 40.0%        | **46.7%**      | **40.2%** |

