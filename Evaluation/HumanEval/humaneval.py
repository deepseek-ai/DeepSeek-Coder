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
from utils.dataset import HumanEvalDataset
from utils.utils import cleanup_code

class HumanEval:
    """
    HumanEval evaluation class.
    """
    def __init__(self, data_root, max_seq_len=2048,
                language="python", max_gen_len=200, batch_size=512,
                log_dir=None, temperature=0, issft=False, top_p=0.95,
                model_name="", inference_increment=True,
                tokenizer_cfg=None, n_sample=40, k_sample=1):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.k = k_sample
        self.n_sample = n_sample
        self.language = language
        self.log_dir = log_dir
        self.sft = issft
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = tokenizer_cfg["model_path"].replace("/", "_")
        self.inference_increment = inference_increment
        os.makedirs(self.log_dir, exist_ok=True)
        tokenizer_cls = tokenizer_cfg.pop('cls')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg.pop("model_path"), trust_remote_code=True)       
        except Exception as e:
            print(e)
            assert False

    @torch.no_grad()
    def eval_model(self, gpt, accelerator, model_path, language, start_time):
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
        # each process will process a subset of the dataset
        prompt_indices_split = np.array_split(range(nprompt), dp_size)
        prompt_indices = prompt_indices_split[dp_rank]
        indices = [x * self.n_sample + j for x in prompt_indices for j in range(self.n_sample)]
        all_num = len(indices)
        processed_num = 0
        log_file = os.path.join(self.log_dir,
                                        f'{self.model_name}_rank{dp_rank}_bs{self.batch_size}_shot_log_{self.language}.json')
        tmpfile = open(log_file, "w")
        
        # --- FIX: Start a new timer here for the loop using time.time() ---
        loop_start_time = time.time()

        for idx in range(0, len(indices), self.batch_size):
            # Prepare data for the current batch
            batch_indices = indices[idx:idx + self.batch_size]
            prompt_list = []
            prompt_lens = []
            original_prompt_list = []
            taskid_list = []

            for j in batch_indices:
                data = dataset[j]
                fprompt = data["prompt"].strip()
                prompt_list.append(fprompt)
                prompt_lens.append(len(fprompt))
                original_prompt_list.append(data["original_prompt"])
                taskid_list.append(data["task_id"])
            
            # Batch process with the tokenizer, which handles padding and truncation automatically
            inputs = self.tokenizer(
                prompt_list,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_seq_len - self.max_gen_len
            )
            
            # Move the batch data to the corresponding device
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

            # Generate the code
            if self.temperature != 0:       
                decoded = gpt.generate(
                    **inputs,
                    max_new_tokens=self.max_gen_len,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                decoded = gpt.generate(
                    **inputs,
                    max_new_tokens=self.max_gen_len,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # save the results to a file
            for local_idx, text in enumerate(decoded):
                prediction = decoded[local_idx]
                prediction = self.tokenizer.decode(prediction, skip_special_tokens=True)
                suffixprediction = prediction[prompt_lens[local_idx]:]
                suffixprediction = cleanup_code(suffixprediction, self.language, "humaneval", self.sft, dataset.stopwords)
                if not self.sft:
                    suffixprediction = original_prompt_list[local_idx] + "\n" + suffixprediction
                res = {"task_id": taskid_list[local_idx], "generation": suffixprediction, "prompt": original_prompt_list[local_idx], "wholecode":prediction}
                tmpfile.write(json.dumps(res) + "\n")
                tmpfile.flush()
                processed_num += 1

            # --- FIX: Pass the correct float timer to log_score ---
            self.log_score(dp_rank, processed_num, all_num, loop_start_time, self.batch_size)
        tmpfile.close()      
        accelerator.wait_for_everyone()
        # The original start_time is passed here for the final report, which is fine.
        self._calculate_final_score(accelerator, model_path, language, start_time) 
        accelerator.wait_for_everyone()
        return

    def log_score(self, dp_rank, processed_num, all_num, start_time, bs):
        """
        Log the score.
        """
        mem = torch.cuda.max_memory_allocated() / (1 << 30)
        
        # --- FIX: This calculation will now work correctly (float - float) ---
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
    
    def _calculate_final_score(self, accelerator, model_path, language, start_time):
        """
        Calculate the final score.
        """
        if accelerator.is_local_main_process:
            logfilepath = os.path.join(self.log_dir, f'final_{self.model_name}.jsonl')
            logfile = open(logfilepath, "w")
            for i in range(accelerator.num_processes):
                tmplogfile = os.path.join(self.log_dir, f'{self.model_name}_rank{i}_bs{self.batch_size}_shot_log_{self.language}.json')
                logfile.write(open(tmplogfile).read().strip() + "\n")
                os.remove(tmplogfile)
            logfile.close()
            timeout = 30
            runlang = self.language
            res = evaluate_functional_correctness(input_file=logfilepath, problem_file=os.path.join(self.data_root, f"humaneval-{self.language}.jsonl"), tmp_dir=self.log_dir, timeout=timeout, language=runlang)

            end_time = datetime.datetime.now()

            print("\n" + "="*45)
            print("Evaluation Done!")
            print(f"Model Path: {model_path}")
            print(f"Language: {language}")
            print("Score is", res['pass@%d' % self.k])
            print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*45)

            # os.remove(logfilepath)
        return
            