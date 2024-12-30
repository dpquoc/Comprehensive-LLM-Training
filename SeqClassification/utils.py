import os
from enum import Enum

import json
import packaging.version
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from datasets import DatasetDict, load_dataset, load_from_disk
from peft import get_peft_model
from datasets import Dataset as HFDataset

from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification
)
import pandas as pd
from typing import Dict, Optional, List, Union
from peft import LoraConfig


DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def preprocess(
    sources: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 2048,
    spread_max_length: bool = False,
) -> Dict:
    """
    Preprocess the input data for the model.
    
    Args:
        sources: Either a dictionary with lists as values, or a list of dictionaries
        tokenizer: The tokenizer to use
        max_len: Maximum sequence length
        spread_max_length: Whether to equally distribute max_length among components
    """
    # Convert list of dicts to dict of lists if necessary
    if isinstance(sources, list):
        converted_sources = {
            "prompt": [],
            "response_a": [],
            "response_b": [],
            "winner": []
        }
        for item in sources:
            converted_sources["prompt"].append(item["prompt"])
            converted_sources["response_a"].append(item["response_a"])
            converted_sources["response_b"].append(item["response_b"])
            converted_sources["winner"].append(item["winner"])
        sources = converted_sources
    
    # Define special tokens
    im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    nl_tokens = tokenizer('\n').input_ids
    
    # Define role tokens
    user_tokens = tokenizer('user').input_ids + nl_tokens
    assistant_tokens = tokenizer('assistant').input_ids + nl_tokens
    
    # Process prompts and responses
    custom_prompt = (
        "Read the following prompt carefully. Compare the two responses provided "
        "and determine which response better addresses the user's needs.\n\n"
    )
    prompts = [custom_prompt + "<prompt>: " + p for p in sources["prompt"]]
    responses_a = ["\n\n<response_a>: " + r_a for r_a in sources["response_a"]]
    responses_b = ["\n\n<response_b>: " + r_b for r_b in sources["response_b"]]
    
    # Rest of the function remains the same
    if spread_max_length:
        special_tokens_length = len([im_start] + user_tokens + [im_end] + [im_start] + assistant_tokens)
        component_max_len = (max_len - special_tokens_length) // 3
        
        # Tokenize without special tokens
        prompt_tokens = tokenizer(prompts, add_special_tokens=False, max_length=component_max_len, 
                                truncation=True, padding=False).input_ids
        response_a_tokens = tokenizer(responses_a, add_special_tokens=False, max_length=component_max_len, 
                                    truncation=True, padding=False).input_ids
        response_b_tokens = tokenizer(responses_b, add_special_tokens=False, max_length=component_max_len, 
                                    truncation=True, padding=False).input_ids
        
        input_ids = []
        for p_tok, ra_tok, rb_tok in zip(prompt_tokens, response_a_tokens, response_b_tokens):
            sequence = [im_start] + user_tokens + p_tok + ra_tok + rb_tok + [im_end] + [im_start] + assistant_tokens
            sequence = sequence[:max_len]
            sequence += [tokenizer.pad_token_id] * (max_len - len(sequence))
            input_ids.append(sequence)
    else:
        combined_texts = [p + ra + rb for p, ra, rb in zip(prompts, responses_a, responses_b)]
        
        tokenized = tokenizer(
            combined_texts,
            add_special_tokens=False,
            max_length=max_len - len([im_start] + user_tokens + [im_end] + [im_start] + assistant_tokens),
            truncation=True,
            padding=False,
        ).input_ids
        
        input_ids = []
        for seq in tokenized:
            sequence = [im_start] + user_tokens + seq + [im_end] + [im_start] + assistant_tokens
            sequence = sequence[:max_len]
            sequence += [tokenizer.pad_token_id] * (max_len - len(sequence))
            input_ids.append(sequence)
    
    # Convert to tensor and create attention mask
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # Handle the winner labels
    labels = []
    winners = sources.get("winner", [])
    
    for winner in winners:
        if winner == 'model_a':
            label = 0
        elif winner == 'model_b':
            label = 1
        else:
            continue  # Skip invalid labels
            
        labels.append(label)
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

class SupervisedDataset(HFDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data: Union[List, pd.DataFrame], tokenizer: transformers.PreTrainedTokenizer, max_len: int, spread_max_length: bool = False):
        super(SupervisedDataset, self).__init__()
        
        # Process raw data to consistent format
        sources = self._process_raw_data(raw_data)
        
        # Let preprocess handle the full dictionary including conversations and winner
        data_dict = preprocess(sources, tokenizer, max_len, spread_max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def _process_raw_data(self, raw_data: Union[List, pd.DataFrame]) -> List:
        """Convert input data to list format."""
        if isinstance(raw_data, list):
            return raw_data
        
        elif isinstance(raw_data, pd.DataFrame):
            # Convert DataFrame to list of dictionaries
            return raw_data.to_dict('records')
        
        else:
            raise TypeError(
                f"Unsupported input type: {type(raw_data)}. "
                "Please provide a list or pandas DataFrame."
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

class LazySupervisedDataset(HFDataset):
    """Dataset for supervised fine-tuning with lazy loading."""

    def __init__(self, raw_data: Union[List, pd.DataFrame], tokenizer: transformers.PreTrainedTokenizer, max_len: int, spread_max_length: bool = False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.spread_max_length = spread_max_length
        self.raw_data = self._process_raw_data(raw_data)
        self.cached_data_dict = {}

    def _process_raw_data(self, raw_data: Union[List, pd.DataFrame]) -> List:
        """Convert input data to list format."""
        if isinstance(raw_data, list):
            return raw_data
        
        elif isinstance(raw_data, pd.DataFrame):
            return raw_data.to_dict('records')
        
        else:
            raise TypeError(
                f"Unsupported input type: {type(raw_data)}. "
                "Please provide a list or pandas DataFrame."
            )

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer, self.max_len, self.spread_max_length)
        ret = {
            'input_ids': ret['input_ids'][0],
            'attention_mask': ret['attention_mask'][0],
            'labels': ret['labels'][0]
        }
        self.cached_data_dict[i] = ret

        return ret

def load_data(file_path: str) -> Union[List, pd.DataFrame]:
    """Load data from a file, supporting JSON, CSV, Excel, and Parquet."""
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == ".json":
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_extension.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_extension.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    elif file_extension.lower() == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported file extension: {file_extension}. "
            "Supported formats are JSON, CSV, Excel, and Parquet."
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )

    # Load training data
    train_data = load_data(data_args.data_path)
    train_dataset = dataset_cls(
        train_data, tokenizer=tokenizer, max_len=data_args.my_max_len, spread_max_length=data_args.spread_max_length
    )

    # Load evaluation data if provided
    if data_args.eval_data_path:
        eval_data = load_data(data_args.eval_data_path)
        eval_dataset = dataset_cls(
            eval_data, tokenizer=tokenizer, max_len=data_args.my_max_len, spread_max_length=data_args.spread_max_length
        )
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def create_and_prepare_model(args, data_args):
    bnb_config = None
    quant_storage_dtype = None

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    torch_dtype = (
        quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        torch_dtype=torch_dtype,
        # device_map="auto"  # Add this, not work when using DeepSpeed 3
    )

    peft_config = None
    chat_template = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    # Chat template and tokenizer configuration
    special_tokens = None
    chat_template = None
    if data_args.apply_chat_template == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif data_args.apply_chat_template == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template

        # make embedding resizing configurable?
        # Transformers 4.46.0+ defaults uses mean_resizing by default, which fails with QLoRA + FSDP because the
        # embedding could be on meta device, therefore, we set mean_resizing=False in that case (i.e. the status quo
        # ante). See https://github.com/huggingface/accelerate/issues/1620.
        uses_transformers_4_46 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")
        uses_fsdp = os.environ.get("ACCELERATE_USE_FSDP").lower() == "true"
        if (bnb_config is not None) and uses_fsdp and uses_transformers_4_46:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
        else:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, model_max_length=args.model_max_length, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token


    return model, peft_config, tokenizer