import copy
import inspect
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def build_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional tokenizer source. Example: openai/gpt-oss-20b"},
    )


def project_vocab_weights_(new_weight: torch.Tensor, old_weight: torch.Tensor) -> None:
    """Copy overlapping vocab rows and initialize extra rows."""
    if old_weight.ndim != 2 or new_weight.ndim != 2:
        return
    if old_weight.shape[1] == 0 or new_weight.shape[1] == 0:
        return
    if old_weight.shape[1] != new_weight.shape[1]:
        return

    rows_to_copy = min(new_weight.shape[0], old_weight.shape[0])
    if rows_to_copy == 0:
        return

    new_weight[:rows_to_copy].copy_(old_weight[:rows_to_copy])
    if new_weight.shape[0] > rows_to_copy:
        old_std = old_weight.std().item()
        if old_std <= 0:
            old_std = 0.02
        torch.nn.init.normal_(new_weight[rows_to_copy:], mean=0.0, std=old_std)


def adapt_model_to_tokenizer_vocab_(model, tokenizer, local_rank: int) -> None:
    """Resize token embeddings/lm_head to tokenizer vocab and align special ids."""
    new_vocab_size = len(tokenizer)
    old_vocab_size = model.get_input_embeddings().weight.shape[0]

    if local_rank in (-1, 0):
        print(f"Model vocab size: {old_vocab_size}")
        print(f"Tokenizer vocab size: {new_vocab_size}")

    # Preserve old projection weights before resize.
    with torch.no_grad():
        old_input_weight = model.get_input_embeddings().weight.detach().clone()
        old_lm_head = model.get_output_embeddings()
        old_lm_head_weight = None
        if old_lm_head is not None:
            old_lm_head_weight = old_lm_head.weight.detach().clone()

    model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size

    # Align model/generation special token ids with tokenizer.
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    can_project_input = (
        old_input_weight.ndim == 2
        and old_input_weight.shape[0] > 0
        and old_input_weight.shape[1] > 0
        and model.get_input_embeddings().weight.ndim == 2
        and model.get_input_embeddings().weight.shape[1] == old_input_weight.shape[1]
    )
    can_project_lm_head = (
        old_lm_head_weight is not None
        and old_lm_head_weight.ndim == 2
        and old_lm_head_weight.shape[0] > 0
        and old_lm_head_weight.shape[1] > 0
        and model.get_output_embeddings() is not None
        and model.get_output_embeddings().weight.ndim == 2
        and model.get_output_embeddings().weight.shape[1] == old_lm_head_weight.shape[1]
    )

    with torch.no_grad():
        if can_project_input:
            project_vocab_weights_(model.get_input_embeddings().weight, old_input_weight)
        elif local_rank in (-1, 0):
            print("Skipping input embedding projection: source weights are empty/unmaterialized on this rank.")

        new_lm_head = model.get_output_embeddings()
        if new_lm_head is not None and can_project_lm_head:
            project_vocab_weights_(new_lm_head.weight, old_lm_head_weight)
        elif new_lm_head is not None and local_rank in (-1, 0):
            print("Skipping lm_head projection: source weights are empty/unmaterialized on this rank.")

    if local_rank in (-1, 0):
        print("Applied tokenizer-vocab projection to model embeddings/lm_head.")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, eot_token):
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{eot_token}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.local_rank == 0:
        print('='*100)
        print(training_args)
    
    tokenizer_source = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(tokenizer_source))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        adapt_model_to_tokenizer_vocab_(model, tokenizer, training_args.local_rank)
    else:
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    if training_args.local_rank == 0:
        print("Load model from {} over.".format(model_args.model_name_or_path))


    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )
    if training_args.local_rank > 0: 
        torch.distributed.barrier()
    
    eot_token = tokenizer.eos_token or EOT_TOKEN
    
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer, "eot_token": eot_token}
    )

    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    trainer_kwargs = dict(model=model, args=training_args, **data_module)
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        # transformers>=5 renamed `tokenizer` to `processing_class`
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    resume_checkpoint = training_args.resume_from_checkpoint
    if resume_checkpoint is None and os.path.isdir(training_args.output_dir):
        resume_checkpoint = get_last_checkpoint(training_args.output_dir)
        if resume_checkpoint is not None and training_args.local_rank in (-1, 0):
            print(f"Auto-resuming from latest checkpoint: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
