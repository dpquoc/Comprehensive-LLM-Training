import os
from enum import Enum

import json
import packaging.version
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import pandas as pd


from typing import Dict, Optional, List
from peft import LoraConfig
from typing import Optional, Union, Callable



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

def simple_preprocess(
    conversations: List[Dict],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 2048,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    """
    Simplified preprocessing for chat conversations.
    
    Args:
        conversations: List of conversation turns [{"role": "user/assistant", "content": "message"}]
        tokenizer: The tokenizer to use
        max_len: Maximum sequence length
        system_message: System message to prepend
    """
    template = "<|im_start|>{role}\n{content}<|im_end|>\n"
    system_prompt = template.format(role="system", content=system_message)
    
    # For Gemma model ( no system prompt )
    system_prompt= ""

    full_texts = []
    targets = []
    
    for conv in conversations:
        # Build conversation text
        text = system_prompt
        target = ["<|im_start|>"] + [IGNORE_TOKEN_ID] * (len(tokenizer(system_prompt).input_ids) - 3) + ["<|im_end|>\n"]
        
        for turn in conv:
            turn_text = template.format(role=turn["role"], content=turn["content"])
            text += turn_text
            
            # For user messages, mask out the entire turn
            if turn["role"] == "user":
                target += ["<|im_start|>"] + [IGNORE_TOKEN_ID] * (len(tokenizer(turn_text).input_ids) - 3) + ["<|im_end|>\n"]
            # For assistant messages, only mask the role tokens
            else:
                role_tokens = len(tokenizer("<|im_start|>assistant\n").input_ids)
                content_tokens = tokenizer(turn["content"] + "<|im_end|>\n").input_ids
                target += ["<|im_start|>"] + [IGNORE_TOKEN_ID] * (role_tokens - 1) + content_tokens
                
        full_texts.append(text)
        targets.append(target)

    # Tokenize and pad
    tokenized = tokenizer(
        full_texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Convert targets to tensor and pad
    padded_targets = [t + [IGNORE_TOKEN_ID] * (max_len - len(t)) for t in targets]
    target_tensor = torch.tensor(padded_targets)[:, :max_len]
    
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "labels": target_tensor
    }


# def preprocess(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     max_len: int = 2048,
#     system_message: str = "You are a helpful assistant."
# ) -> Dict:
#     # start_text = '<|im_start|>'
#     # end_text = '<|im_end|>'
#     # roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

#     start_text = '<start_of_turn>'
#     end_text = '<end_of_turn>'
#     roles = {"user": "<start_of_turn>user", "assistant": "<start_of_turn>model"}

#     im_start = tokenizer.convert_tokens_to_ids(start_text)
#     im_end = tokenizer.convert_tokens_to_ids(end_text)
#     nl_tokens = tokenizer('\n', add_special_tokens=False).input_ids  # Add this flag
#     _system = tokenizer('system', add_special_tokens=False).input_ids + nl_tokens
#     _user = tokenizer('user', add_special_tokens=False).input_ids + nl_tokens
#     _assistant = tokenizer('assistant', add_special_tokens=False).input_ids + nl_tokens

#     # Apply prompt templates
#     input_ids, targets = [], []
#     for i, source in enumerate(sources):
#         if roles[source[0]["role"]] != roles["user"]:
#             source = source[1:]

#         input_id, target = [], []

#         ## System handling
#         # system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
#         # input_id += system
#         # target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens

#         # System handling
#         bos_token = tokenizer.convert_tokens_to_ids('<bos>')
#         system = [bos_token]
#         input_id += system
#         target += [bos_token]
#         assert len(input_id) == len(target)

#         # Conversation handling
#         for j, sentence in enumerate(source):
#             role = roles[sentence["role"]]

#             # Input IDs
#             _input_id = tokenizer(role, add_special_tokens=False).input_ids + nl_tokens + \
#                 tokenizer(sentence["content"], add_special_tokens=False).input_ids + [im_end] + nl_tokens
#             input_id += _input_id

#             # Target IDs
#             # if role == '<|im_start|>user':
#             if role == '<start_of_turn>user':
#                 _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
#             # elif role == '<|im_start|>assistant':
#             elif role == '<start_of_turn>model':
#                 _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role, add_special_tokens=False).input_ids) + \
#                     _input_id[len(tokenizer(role, add_special_tokens=False).input_ids)+1:-2] + [im_end] + nl_tokens
#             else:
#                 raise NotImplementedError
#             target += _target

#         # break
#         assert len(input_id) == len(target)
#         input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
#         target += [IGNORE_TOKEN_ID] * (max_len - len(target))
#         input_ids.append(input_id[:max_len])
#         targets.append(target[:max_len])
#     input_ids = torch.tensor(input_ids, dtype=torch.int)
#     targets = torch.tensor(targets, dtype=torch.int)

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#         attention_mask=input_ids.ne(tokenizer.pad_token_id),
#     )


# For Gemma 2 A/B
def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 2048,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    start_text = '<start_of_turn>'
    end_text = '<end_of_turn>'
    roles = {"user": "<start_of_turn>user", "assistant": "<start_of_turn>model"}

    im_start = tokenizer.convert_tokens_to_ids(start_text)
    im_end = tokenizer.convert_tokens_to_ids(end_text)
    nl_tokens = tokenizer('\n', add_special_tokens=False).input_ids
    _system = tokenizer('system', add_special_tokens=False).input_ids + nl_tokens
    _user = tokenizer('user', add_special_tokens=False).input_ids + nl_tokens
    _assistant = tokenizer('assistant', add_special_tokens=False).input_ids + nl_tokens

    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if len(source) < 2:
            continue  # Skip invalid sources

        # Ensure the first message is from the user
        if roles[source[0]["role"]] != roles["user"]:
            continue  # Skip if the first message is not user

        # System handling (bos_token)
        bos_token = tokenizer.convert_tokens_to_ids('<bos>')
        system = [bos_token]
        input_id = system.copy()
        target = [bos_token]

        # Process each sentence in the source (user and assistant)
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            current_role = sentence["role"]

            if j == 0:  # User turn: handle truncation based on assistant's content
                # Precompute assistant's part
                assistant_sentence = source[1]
                assistant_role = roles[assistant_sentence["role"]]
                assistant_content = assistant_sentence["content"]
                assistant_role_tokens = tokenizer(assistant_role, add_special_tokens=False).input_ids
                assistant_content_tokens = tokenizer(assistant_content, add_special_tokens=False).input_ids
                assistant_part = (
                    assistant_role_tokens +
                    nl_tokens +
                    assistant_content_tokens +
                    [im_end] +
                    nl_tokens
                )
                assistant_part_length = len(assistant_part)

                # Calculate available space for user's content
                system_length = len(system)
                user_role_tokens = tokenizer(role, add_special_tokens=False).input_ids
                user_role_length = len(user_role_tokens)
                user_end_part = nl_tokens + [im_end] + nl_tokens
                user_end_length = len(user_end_part)
                available_space = max_len - (
                    system_length +
                    user_role_length +
                    user_end_length +
                    assistant_part_length
                )

                # Tokenize user's content with truncation
                user_content = sentence["content"]
                user_content_tokens = tokenizer(user_content, add_special_tokens=False).input_ids
                if available_space > 0:
                    user_content_tokens = user_content_tokens[:available_space]
                else:
                    user_content_tokens = []

                # Build user's input_id part
                _input_id = (
                    user_role_tokens +
                    nl_tokens +
                    user_content_tokens +
                    [im_end] +
                    nl_tokens
                )
            else:  # Assistant turn: use full content
                _input_id = (
                    tokenizer(role, add_special_tokens=False).input_ids +
                    nl_tokens +
                    tokenizer(sentence["content"], add_special_tokens=False).input_ids +
                    [im_end] +
                    nl_tokens
                )

            # Append to input_id and target
            input_id += _input_id

            # Target processing
            if current_role == "user":
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif current_role == "assistant":
                role_tokens = tokenizer(role, add_special_tokens=False).input_ids
                _target = (
                    [im_start] +
                    [IGNORE_TOKEN_ID] * len(role_tokens) +
                    _input_id[len(role_tokens) + 1:-2] +
                    [im_end] +
                    nl_tokens
                )
            else:
                raise NotImplementedError(f"Role {current_role} not supported")
            target += _target

        # print("\nFinal lengths before assertion:")
        # print(f"input_id length: {len(input_id)}")
        # print(f"target length: {len(target)}")
        # print(f"Full final decoded input: {tokenizer.decode(input_id)}")
        # break
        assert len(input_id) == len(target)

        # Pad or truncate to max_len
        input_id = input_id[:max_len]
        target = target[:max_len]
        pad_len = max_len - len(input_id)
        input_id += [tokenizer.pad_token_id] * pad_len
        target += [IGNORE_TOKEN_ID] * pad_len

        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
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
        train_data, tokenizer=tokenizer, max_len=data_args.my_max_len
    )

    # Load evaluation data if provided
    if data_args.eval_data_path:
        eval_data = load_data(data_args.eval_data_path)
        eval_dataset = dataset_cls(
            eval_data, tokenizer=tokenizer, max_len=data_args.my_max_len
        )
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)



def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

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

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.my_max_len,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
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
        # tokenizer.pad_token = tokenizer.eos_token # This line doesnt matter , since Qwen already has its own pad token
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.my_max_len,
        )

    return model, peft_config, tokenizer