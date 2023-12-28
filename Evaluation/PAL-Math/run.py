import os
import re
import json
import argparse
import torch
import numpy as np
from utils.parser import *
from utils.grader import *
from utils.python_executor import PythonExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def extract_python_block_with_solution(text):
    """
    Extract the code block from the text that contains the solution function.
    :param text: The text to search for the code block.
    :return: The extracted code block.
    """
    pattern = r'```python\n(.*?)def solution\(\):\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1) + 'def solution():\n' + match.group(2)
    else:
        return ""
    
def load_data(args):
    """
    Load data from file.
    :param args: Arguments.
    :return: A list of examples.
    """
    if args.data_name != "math":
        prompt = open("prompts/gsm8k.md").read()
    else:
        prompt = open("prompts/math.md").read()

    examples = []
    with open(f"datasets/{args.data_name}/test.json", "r") as f: 
        for line in f:
            js = json.loads(line)
            examples.append(js)

    # parse data
    samples = []
    for example in examples:
        idx = example['idx']
        example['question'] = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        example["input"] = f"{prompt}\n\nQuestion: {example['question']}\n"
        example = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': example["input"]}
        samples.append(example)  

    return samples

def inference(args):
    """
    Inference on the dataset.
    :param args: Arguments.
    :return: None
    """
    # load data
    samples = load_data(args)
    samples = [sample for i,sample in enumerate(samples) if i%args.world_size==args.rank]

    # create directory for saving results
    os.makedirs(f'outputs/{args.model_name}/{args.data_name}', exist_ok=True)

    # init python executor
    executor = PythonExecutor(get_answer_expr='solution()')

    # load model
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True,padding_side="left")
    try:
        tokenizer.pad_token_id = 0
    except:
        # Deal with CodeGeex-2
        pass
    llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)

    #samples = samples[:32]
    print("dataset:", args.data_name, "samples:", len(samples))
    if len(samples) > 0:
        print("=" * 50)
        print("sample:", samples[0]['prompt'])
        print("=" * 50)

    stop_ids = []
    stop_words = ["Question","----------------"]
    for x in stop_words:
        ids = tokenizer.encode(x)
        if tokenizer.decode(ids[-1:]) == x:
            stop_ids.append(ids[-1])               
    print("stop ids:", stop_ids)



    outputs = []
    generation_config = GenerationConfig(num_beams=1,)
    for i in range(0, len(samples), args.batch_size):
        chunk = [x["prompt"] for x in samples[i:i+args.batch_size]]
        if "llama" in args.model_name_or_path.lower() and args.rank==3 and (i==164 or i==328):
            for x in chunk:
                outputs.append(x)
            continue
        inputs = tokenizer(chunk, return_tensors="pt",padding=True)
        input_ids = inputs["input_ids"].cuda()[:,-args.max_context_length:]
        attention_mask = inputs["attention_mask"].cuda()[:,-args.max_context_length:]

        with torch.no_grad():
            generation_output = llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
                max_new_tokens=args.max_output_length,
                eos_token_id=stop_ids,
                pad_token_id=0
            )

        answers = []

        for i, a in enumerate(generation_output.sequences):
            a = a.tolist()
            a = a[input_ids.shape[-1]:]
            a = tokenizer.decode(a)
            for x in stop_words:
                if x in a:
                    a = a[:a.index(x)]
            ans = extract_python_block_with_solution(a)
            answers.append(ans)
            if i == 0:
                print("="*80)
                print("Response:\n")
                print(a)
                print("Program:\n")
                print(ans)               
                print("="*80)
        outputs.extend(answers)
        print("Rank",args.rank,"Processed Number:",len(outputs),flush=True)

    assert len(outputs) == len(samples)

    results = [x[0] for x in executor.batch_apply(outputs)]
    for result,code,sample in zip(results, outputs, samples):
        sample["code"] = code
        sample["pred"] = strip_string(result)

    # save results
    out_file = f"world_size_{args.world_size}_rank_{args.rank}.json"
    with open(f"outputs/{args.model_name}/{args.data_name}/{out_file}", "w") as f:
        json.dump(samples,f,indent=4)

def eval(args):
    """
    Evaluate the results.
    :param args: Arguments.
    :return: None
    """
    # load data
    samples = []
    for rank in range(args.world_size):
        out_file = f"outputs/{args.model_name}/{args.data_name}/world_size_{args.world_size}_rank_{rank}.json"
        if not os.path.exists(out_file):
            raise FileNotFoundError(f"File {out_file} does not exist.")
        samples.extend(json.load(open(out_file,"r")))
    print("Dataset:",args.data_name)
    print("Model:",args.model_name)
    print("Loaded Examples:",len(samples))
    scores = []
    for x in samples:
        scores.append(math_equal(x["gt"],x["pred"]))
    print("Mean Score",np.mean(scores))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="math", type=str)
    parser.add_argument("--model_name_or_path", default="deepseek/deepseek-coder-1b-python", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_context_length", default=2048, type=int)
    parser.add_argument("--max_output_length", default=512, type=int)
    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size",default=1, type=int)
    args = parser.parse_args()
    
    args.model_name = args.model_name_or_path.strip("/").split("/")[-1]
    if args.do_inference:
        print(args)
        inference(args)
    elif args.do_eval:
        eval(args)
