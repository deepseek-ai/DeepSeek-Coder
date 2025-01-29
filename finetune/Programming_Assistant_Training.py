
import copy
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments
from datasets import load_dataset

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Set up logging
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


def build_instruction_prompt(instruction: str) -> str:
    """
    Build a formatted instruction prompt for the model to respond to.
    """
    return f'''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, 
and you only answer questions related to computer science. For politically sensitive questions, security 
and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{instruction.strip()}
### Response:
'''.lstrip()


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model(trainer: Trainer, output_dir: str):
    """
    Collects the state dict and saves it to disk.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        trainer._save(output_dir, state_dict=cpu_state_dict)
    logging.info(f"Model saved to {output_dir}")


def tokenize_texts(strings: Sequence[str], tokenizer: AutoTokenizer) -> Dict:
    """
    Tokenize a list of strings.
    """
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

    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess_data(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: AutoTokenizer,
) -> Dict:
    """
    Preprocess the data by tokenizing it and preparing the labels.
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [tokenize_texts(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset:
    """
    Collator for supervised fine-tuning.
    """
    tokenizer: AutoTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in input_ids], 
                                                    batch_first=True, 
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in labels], 
                                                 batch_first=True, 
                                                 padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def tokenize_for_training(examples, tokenizer):
    """
    Tokenize the examples for training.
    """
    sources = [build_instruction_prompt(instruction) for instruction in examples['instruction']]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    return preprocess_data(sources, targets, tokenizer)


def train():
    """
    Main training loop.
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)

    # Load dataset
    raw_train_datasets = load_dataset('json', data_files=data_args.data_path, split="train", cache_dir=training_args.cache_dir)
    train_dataset = raw_train_datasets.map(
        tokenize_for_training,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        desc="Tokenizing",
        fn_kwargs={"tokenizer": tokenizer}
    )

    logging.info(f"Loaded dataset with {len(train_dataset)} samples.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset, data_collator=data_collator)

    # Start training
    trainer.train()
    trainer.save_state()
    safe_save_model(trainer, training_args.output_dir)


if __name__ == "__main__":
    train()
