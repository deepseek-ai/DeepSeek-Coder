import time
import string
import multiprocessing
import os
import numpy as np
import json
import re
import torch
import datetime
import subprocess
import torch.distributed as dist
from attrdict import AttrDict
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoTokenizer

class HumanEvalDataset:

    def __init__(self, root, sample_num=1, language="python", issft=False):
        """
        root: the path to the HumanEval dataset
        sample_num: the number of samples for each prompt
        language: the language of the HumanEval dataset
        issft: whether to use the SFT setting
        """
        self.root = root
        self.data = open(os.path.join(self.root, f"humaneval-{language}.jsonl")).readlines()

        tmp = self.get_qa_only_data(self.data, issft)
        self.clean_data = []
        for i in range(len(tmp)):
            for j in range(sample_num):
                self.clean_data.append(tmp[i])
        self.stopwords = self.clean_data[0]["stopwords"]
        np.random.seed(1234)
        print(f"Read HumanEval from {root}, number of samples {len(self.clean_data)}")

    def get_qa_only_data(self, data_json, sft=False):
        """
        data_json: the jsonl file of HumanEval
        sft: whether to use the SFT setting
        return: a list of dict, each dict contains the prompt, task_id and stopwords
        """
        ans = []
        for line in data_json:
            line = json.loads(line)
            prompt = line["prompt"].strip()
            if "prefix" in line:
                origin_prompt = line["prefix"]
            else:
                origin_prompt = line["prompt"]

            if sft:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\nWrite a program to perform the given task.\n\nInput:\n{prompt}\n\n### Response:\n"""
            if "stop_tokens" in line:
                s = line["stop_tokens"]
            else:
                s = []
            ans.append({"prompt":prompt, "task_id":line["task_id"], "original_prompt": origin_prompt, "stopwords":s})
        return ans

    def __len__(self):
        """
        return the number of samples in the dataset
        """
        return len(self.clean_data)

    def __getitem__(self, index):
        """
        return the sample at index
        """
        sample = self.clean_data[index]
        return sample

def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
    issft: bool = False,
    stop_words = []
):
    """
    Cleans up the generated code.
    """
    if language_type is None or dataset is None:
        return code

    if "humaneval" in dataset.lower():
        if language_type.lower() == "python":
            if issft:
                copycode = code
                completion = code.replace("\r", "")
                if "```python" in completion:
                    def_line = completion.index("```python")
                    completion = completion[def_line:].strip()
                    completion = completion.replace("```python", "")
                    # print(completion)
                    try:
                        next_line = completion.index("```")
                        completion = completion[:next_line].strip()
                    except:
                        print(code)
                        print("error================\n")
                    # print(completion)
                code = completion.strip()          
            if True:
                codelist = re.split("\ndef|\nclass|\nif|\n#|\nprint", code)
                if "def" not in codelist[0] and issft:
                    if len(codelist) == 1:
                        print(copycode)
                        code = codelist[0] + "\ndef"
                    try:
                        code = codelist[0] + "\ndef" + codelist[1]
                    except:
                        print("index error")
                        print(copycode)
                        print("===================================")
                else:
                    code = codelist[0]

        elif language_type.lower() == "ts":
            min_stop_idx = len(code)
            stop_words += ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"]
            for stop_word in stop_words:
                stop_index = code.find(stop_word)
                if stop_index != -1 and stop_index < min_stop_idx:
                    min_stop_idx = stop_index
            code = code[:min_stop_idx]

        else:
            min_stop_idx = len(code)
            for stop_word in stop_words:
                stop_index = code.find(stop_word)
                if stop_index != -1 and stop_index < min_stop_idx:
                    min_stop_idx = stop_index
            code = code[:min_stop_idx]
    return code

class HumanEval:
    """
    HumanEval evaluation class.
    """
    def __init__(self, data_root, max_seq_len=2048, language="python", max_gen_len=200, batch_size=512,
                 log_dir=None, temperature=0, issft=False, top_p=0.95,
                 model_name="", inference_increment=True, tokenizer_cfg=None, n_sample=40, k_sample=1):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.k = k_sample
        self.n_sample = n_sample
        self.language = language
        self.log_dir = log_dir
        self.sft = issft
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = tokenizer_cfg["model_path"].replace("/", "_")
        self.inference_increment = inference_increment
        tokenizer_cls = tokenizer_cfg.pop('cls')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg.pop("model_path"), trust_remote_code=True)       
        except Exception as e:
            print(e)
            assert False

    @torch.no_grad()
    def eval_model(self, gpt, accelerator):
        """
        Evaluate the model on HumanEval.
        """
        assert self.log_dir is not None, "log_dir should not be None when evaluating humaneval"
        dataset = HumanEvalDataset(self.data_root, sample_num=self.n_sample, language=self.language, issft=self.sft)
        nprompt = len(dataset) // self.n_sample
        dp_rank = accelerator.process_index 
        dp_size = accelerator.num_processes 
        if self.k > 1:
            assert self.n_sample >= 100, "HumanEval PASS@100 needs n_sample >= 100"
        gpt.eval()
        # 每个 DP rank 负责一部分的数据
        prompt_indices_split = np.array_split(range(nprompt), dp_size)
        prompt_indices = prompt_indices_split[dp_rank]
        indices = []
        for x in prompt_indices:
            for j in range(self.n_sample):
                indices.append(x * self.n_sample + j)
        all_num = len(indices) 
        processed_num = 0
        if self.log_dir:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            log_file = os.path.join(self.log_dir,
                                    f'{self.model_name}_rank{dp_rank}_bs{self.batch_size}_shot_log_{self.language}.json')
            latest_log_file = os.path.join(self.log_dir, f'latest_log.json')
            print('Logs are saved to', log_file)
        totoalnum = 0
        tmpfile = open(log_file, "w")
        start_time = time.time()
        for idx in range(0, len(indices), self.batch_size):
            prompt_list = []
            prompt_lens = []
            orriginal_prompt_list = []
            tokenized_prompt_lens = []
            taskid = []
            for j in indices[idx:idx + self.batch_size]:
                data = dataset[j]
                fprompt = data["prompt"].strip()
                prompt_list.append(fprompt)
                tmp = self.tokenizer.encode(fprompt)
                orriginal_prompt_list.append(data["original_prompt"])
                prompt_lens.append(len(fprompt))
                tokenized_prompt_lens.append(tmp)
                taskid.append(data["task_id"])
            input_ids = torch.tensor(tokenized_prompt_lens).to(accelerator.device)
            if self.temperature != 0:       
                decoded = gpt.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_gen_len,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                decoded = gpt.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_gen_len,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            for local_idx, text in enumerate(decoded):
                prediction = decoded[local_idx]
                prediction = self.tokenizer.decode(prediction, skip_special_tokens=True)
                suffixprediction = prediction[prompt_lens[local_idx]:]
                suffixprediction = cleanup_code(suffixprediction, self.language, "humaneval", self.sft, dataset.stopwords)
                if not self.sft:
                    suffixprediction = orriginal_prompt_list[local_idx] + "\n" + suffixprediction
                res = {"task_id": taskid[local_idx], "generation": suffixprediction, "prompt": orriginal_prompt_list[local_idx], "wholecode":prediction}
                tmpfile.write(json.dumps(res) + "\n")
                tmpfile.flush()
                if (idx + local_idx) % self.n_sample == self.n_sample - 1:
                    processed_num += 1
                totoalnum += 1

            self.log_score(dp_rank, totoalnum, all_num, start_time, self.batch_size)
        tmpfile.close()        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and processed_num > 0:
            print('ALL REDUCE!')
            logfilepath = os.path.join(self.log_dir, f'final_{time.time()}.jsonl')
            logfile = open(logfilepath, "w")
            for i in range(dp_size):
                tmplogfile = os.path.join(self.log_dir,
                                f'{self.model_name}_rank{i}_bs{self.batch_size}_shot_log_{self.language}.json')
                logfile.write(open(tmplogfile).read().strip() + "\n")
            logfile.close()
            if self.language == 'python':
                timeout = 10
            else:
                timeout = 5
            runlang = self.language
            res = evaluate_functional_correctness(input_file=logfilepath, problem_file=os.path.join(self.data_root, f"humaneval-{self.language}.jsonl"), tmp_dir=self.log_dir, timeout=timeout, language=runlang)
            print("score is", res['pass@%d' % self.k])
            acc = res['pass@%d' % self.k]
            os.system(f"rm -rf {self.log_dir}")
        else:
            acc = 0
        accelerator.wait_for_everyone()
        return acc if processed_num > 0 else None

    def log_score(self, dp_rank, processed_num, all_num, start_time, bs):
        """
        Log the score.
        """
        mem = torch.cuda.max_memory_allocated() / (1 << 30)
        avg_time = (time.time() - start_time) / processed_num * bs
        print(
            f'DP RANK:{dp_rank} process_num/all_num:{int(processed_num)}/{all_num} '
            f'avg_time_per_batch:{avg_time:.2f} s '
            f'still_need:{((all_num - processed_num) // bs + 1) * avg_time / 60:.2f} m',
            f'mem:{mem:.3f} GiB bs:{bs}',
            flush=True
        )
        if processed_num == all_num:
            print(f'EVAL DONE! Process time {(time.time() - start_time) / 60:.2f} m', flush=True)
