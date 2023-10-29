<p align="center">
<img width="1000px" alt="DeepSeek Coder" src="pictures/logo.jpeg">
</p>
<p align="center"><a href="https://www.deepseek.com/">[<img src="pictures/home.png" width="20px">主页]</a> | <a href="https://coder.deepseek.com/">[🤖 在线体验] | <a href="https://huggingface.co/deepseek-ai">[🤗 模型下载]</a> | <a href="README.md">[📄 English Version] </a> </p>
<hr>


### 1. Deepseek Coder简介

Deepseek Coder 包括一系列代码预训练模型，这些模型在87%的代码和13%的中英文自然语言数据上进行了预训练，共2T的单词。
Deepseek Coder提供各种参数大小的代码模型，范围从1B到33B版本。每个模型都在项目级代码数据上进行预训练，采用16K的窗口大小和额外的Fill-in-the-blank任务，以支持项目级别的代码补全和填充。在代码能力方面，Deepseek Coder在多种编程语言和各种测试基准测试上都达到了目前开源代码模型的最优性能。

<img src="pictures/result.png" alt="result" width="85%">

- **海量训练数据**：在2万亿单词上进行训练，包括87%的代码和13%的英文和中文语言数据。

- **灵活可扩展**：提供1B、7B和33B的模型大小，使用户能够选择最适合其需求的模型。

- **模型性能强大**：在 HumanEval, MultiPL-E, MBPP, DS-1000, 和 APPS 基准测试上，DeepSeek Coder在公开可用的代码模型中性能最优。

- **项目级代码补全**：采用16K的窗口大小和Fill-in-the-blank训练任务，支持项目级代码补全和填充任务。

### 2. 数据处理和模型训练

#### 数据处理

- 步骤1：从GitHub收集代码数据，并采用与[StarcoderData](https://github.com/bigcode-project/bigcode-dataset)相同的过滤规则来筛选数据。
- 步骤2：解析同一仓库中文件的依赖关系，根据它们的依赖关系重新排列文件位置。
- 步骤3：组织依赖文件以形成单一示例，并使用仓库级别的minhash算法进行去重。
- 步骤4：进一步过滤掉低质量的代码，例如语法错误或可读性差的代码。

<img src="pictures/data_clean.png" alt="data_creation" width="100%">

#### 模型训练

- 步骤1：首先使用处理后数据进行预训练，该数据由87%的代码、10%与代码相关的语言数据（Github Markdown和Stack Exchange）以及3%与代码无关的中文语言数据组成。在此步骤中，使用4K的窗口大小在1.8万亿单词上进行模型的预训练。

- 步骤2：扩展的窗口至16K并使用额外的2千亿单词进一步进行预训练，从而得到基础版本模型（**DeepSeek-Coder-Base**）。

- 步骤3：使用20亿单词的指令数据进行微调，得到经过指令调优的模型（**DeepSeek-Coder-Instruct**）。

<img src="pictures/model_pretraining.png" alt="model_pretraining" width="100%">


### 3. 下载和环境依赖

我们提供了基于Hai-LLM的 pytorch 兼容版本，支持transformers(3.34+)，以便在其他GPU平台上使用。

同时模型的权重已上传到至 [huggingface](https://huggingface.co/deepseek-ai)。

#### 环境依赖
Python 3.8+ / CUDA 11+ / PyTorch 2.0+ / transformers 3.34+.

### 4. 模型推理
请参考下面样例来使用我们模型：

#### 1）代码补全
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek/deepseek-coder-7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek/deepseek-coder-7b-base", trust_remote_code=True).cuda()
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

这段代码将输入以下结果:

```
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
    for i in range(1, len(arr)):
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)
```

#### 2）代码填充

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek/deepseek-coder-7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek/deepseek-coder-7b-base", trust_remote_code=True).cuda()
input_text = """<fim_prefix>def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
<fim_middle>
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)<fim_suffix>"""
inputs = tokenizer(input_text, return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):])
```

这段代码将输入以下结果:

```
   for i in range(1, len(arr)):
```

#### 3）仓库级别的代码补全

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("deepseek/deepseek-coder-7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek/deepseek-coder-7b-base", trust_remote_code=True).cuda()

input_text = """#utils.py
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert numpy data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)
    
    return X_train, X_test, y_train, y_test

def evaluate_predictions(y_test, y_pred):
    return accuracy_score(y_test, y_pred)
#model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.fc(x)

    def train_model(self, X_train, y_train, epochs, lr, batch_size):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Create DataLoader for batches
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        with torch.no_grad():
            outputs = self(X_test)
            _, predicted = outputs.max(1)
        return predicted.numpy()
#main.py
from utils import load_data, evaluate_predictions
from model import IrisClassifier as Classifier

def main():
    # Model training and evaluation
"""
inputs = tokenizer(input_text, return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_new_tokens=140)
print(tokenizer.decode(outputs[0]))
```

---
在下面样例中，Deepseek-Coder 7B 模型有效地从 `model.py` 文件中调用了一个名为 `IrisClassifier` 的类及其成员函数，并利用了 `utils.py` 文件中的函数，以正确地完成`main.py` 文件中的模型的训练和评估的功能。

![Completion GIF](pictures/completion_demo.gif)

#### 4）对话功能
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("deepseek/deepseek-coder-7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek/deepseek-coder-7b-base", trust_remote_code=True).cuda()
prompt = "write a quick sort algorithm in python."
prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\nWrite a program to perform the given task.\n\nInput:\n{prompt}\n\n### Response:\n"""
inputs = tokenizer.encode(prompt, return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0]))
```



### 5. 评测结果

以下评测结果的复现代码可查看[Evaluation](https://github.com/deepseek-ai/deepseek-coder/tree/main/Evaluation)目录

#### 1) [HumanEval](https://github.com/deepseek-ai/deepseek-coder/tree/main/Evaluation/HumanEval)

Multilingual Base Models

| Model               | Size | Python | C++   | Java  | PHP   | TS    | C#    | Bash  | JS    | Avg   |
| ------------------- | ---- | ------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| code-cushman-001    | 12B  | 33.5%  | 31.9% | 30.6% | 28.9% | 31.3% | 22.1% | 11.7% | -     | -     |
| CodeShell           | 7B   | 35.4%  | 32.9% | 34.2% | 31.7% | 30.2% | 38.0% | 7.0%  | 33.5% | 30.4% |
| CodeGeeX2           | 6B   | 36.0%  | 29.2% | 25.9% | 23.6% | 20.8% | 29.7% | 6.3%  | 24.8% | 24.5% |
| StarCoderBase       | 16B  | 31.7%  | 31.1% | 28.5% | 25.4% | 34.0% | 34.8% | 8.9%  | 29.8% | 28.0% |
| CodeLLama     | 7B   | 31.7%  | 29.8% | 34.2% | 23.6% | 36.5% | 36.7% | 12.0% | 29.2% | 29.2% |
| CodeLLama     | 13B  | 36.0%  | 37.9% | 38.0% | 34.2% | 45.2% | 43.0% | 16.5% | 32.3% | 35.4% |
| CodeLLama     | 34B  | 48.2%  | 44.7% | 44.9% | 41.0% | 42.1% | 48.7% | 15.8% | 42.2% | 41.0% |
|                     |      |        |       |       |       |       |       |       |       |       |
| DeepSeek-Coder-Base  | 1B   | 34.8%  | 31.1% | 32.3% | 24.2% | 28.9% | 36.7% | 10.1% | 28.6% | 28.3% |
| DeepSeek-Coder-Base | 7B   | 49.4%  | 50.3% | 43.0% | 38.5% | 49.7% | 50.0% | 28.5% | 48.4% | 44.7% |
| DeepSeek-Coder-Base | 33B  | -      | -     | -     | -     | -     | -     | -     | -     | -     |

Instruction-Tuned Models
| Model               | Size | Python | C++   | Java  | PHP   | TS    | C#    | Bash  | JS    | Avg   |
| ------------------- | ---- | ------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| ChatGPT             | -    | 70.7%  | 50.3% | 54.5% | 52.2% | 62.3% | 64.6% | 34.8% | 60.9% | 52.2% |
| GPT-4               | -    | 82.3%  | 70.2% | 74.8% | 70.8% | 73.0% | 77.9% | 51.3% | 83.2% | 72.9% |
| WizardCoder         | 16B  | 51.8%  | 41.6% | 41.1% | 42.2% | 44.7% | 46.8% | 12.7% | 42.8% | 40.5% |
| Phind-CodeLlama     | 34B  | -      | -     | -     | -     | -     | -     | -     | -     | -     |
|                     |      |        |       |       |       |       |       |       |       |       |
| DeepSeek-Coder-Instruct | 1B   | -      | -     | -     | -     | -     | -     | -     | -     | -     |
| DeepSeek-Coder-Instruct | 7B   | -      | -     | -     | -     | -     | -     | -     | -     | -     |
| DeepSeek-Coder-Instruct | 33B  | -      | -     | -     | -     | -     | -     | -     | -     | -     |



#### 2) [Math Reasoning](https://github.com/deepseek-ai/deepseek-coder/tree/main/Evaluation/PAL-Math)

Multilingual Base Models

| Model          | Size | GSM8k | MATH  | GSM-Hard | SVAMP | TabMWP | ASDiv | MAWPS | Avg   |
| -------------- | ---- | ----- | ----- | -------- | ----- | ------ | ----- | ----- | ----- |
| CodeShell      | 7B   | 17.0% | 9.1%  | 18.2%    | 45.6% | 29.6%  | 46.6% | 56.8% | 31.8% |
| CodeGeex-2     | 7B   | 23.6% | 9.6%  | 22.4%    | 48.0% | 47.2%  | 46.9% | 66.0% | 37.7% |
| StarCoder-Base | 16B  | 27.3% | 11.5% | 24.2%    | 44.0% | 45.6%  | 54.9% | 73.4% | 40.1% |
| CodeLLama-Base | 7B   | 36.4% | 12.3% | 29.7%    | 57.6% | 58.4%  | 59.6% | 82.6% | 48.0% |
| CodeLLama-Base | 13B  | 44.2% | 15.5% | 42.4%    | 65.6% | 61.6%  | 65.3% | 85.3% | 54.3% |
| CodeLLama-Base | 34B  | 58.2% | 22.1% | 55.2%    | 77.2% | 69.6%  | 70.0% | 92.8% | 63.6% |
|                |      |       |       |          |       |        |       |       |       |
| DeepSeek-Coder-Base  | 1B   | 17.0% | 13.4% | 13.3%    | 39.2% | 42.4%  | 44.8% | 66.0% | 33.7% |
| DeepSeek-Coder-Base  | 7B   | 46.0% | 20.6% | 40.0%    | 67.2% | 71.2%  | 67.1% | 89.1% | 57.3% |
| DeepSeek-Coder-Base  | 33B  | -     | -     | -        | -     | -      | -     | -     | -     |


Instruction-Tuned Models
| Model         | Size | GSM8k | MATH  | GSM-Hard | SVAMP | TabMWP | ASDiv | MAWPS | Avg   |
| ------------- | ---- | ----- | ----- | -------- | ----- | ------ | ----- | ----- | ----- |
| ChatGPT       | -    | 78.6% | 38.7% | 67.6%    | 77.8% | 79.9%  | 81.0% | 89.4% | 73.3% |
| GPT-4         | -    | 94.2% | 51.8% | 77.6%    | 94.8% | 95.9%  | 92.6% | 97.7% | 86.4% |
|               |      |       |       |          |       |        |       |       |       |
| DeepSeek-Coder-Instruct | 1B   | -     | -     | -        | -     | -      | -     | -     | -     |
| DeepSeek-Coder-Instruct | 7B   | -     | -     | -        | -     | -      | -     | -     | -     |
| DeepSeek-Coder-Instruct | 33B  | -     | -     | -        | -     | -      | -     | -     | -     |

### 6. 协议



### 7. 联系方式

如果有任何问题，请提出raise或通过 [agi_code@deepseek.com](mailto:agi_code@deepseek.com) 与我们联系。



