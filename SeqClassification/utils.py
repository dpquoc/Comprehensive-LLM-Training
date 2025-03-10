import os
from enum import Enum

import json
import packaging.version
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from peft import get_peft_model
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
)
import pandas as pd
from typing import Dict, Optional, List, Union
from peft import LoraConfig


DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def save_score_layer_weights(model, save_path):
    """
    Save only the score layer weights from a sequence classification model.
    
    Args:
        model: The sequence classification model containing the score layer
        save_path (str): Path where to save the weights
    """
    # Extract score layer weights
    score_weights = {
        'weight': model.score.weight.detach().cpu(),
    }
    if hasattr(model.score, 'bias') and model.score.bias is not None:
        score_weights['bias'] = model.score.bias.detach().cpu()
    
    # Save weights
    torch.save(score_weights, save_path)
    print(f"Score layer weights saved to {save_path}")

def load_score_layer_weights(model, weights_path):
    """
    Load and replace score layer weights in a model.
    
    Args:
        model: The sequence classification model to update
        weights_path (str): Path to the saved weights
    """
    # Load the saved weights
    score_weights = torch.load(weights_path)
    
    # Replace the weights in the model
    model.score.weight = torch.nn.Parameter(score_weights['weight'].to(model.score.weight.device))
    if 'bias' in score_weights and hasattr(model.score, 'bias'):
        model.score.bias = torch.nn.Parameter(score_weights['bias'].to(model.score.bias.device))
    
    print(f"Score layer weights loaded from {weights_path}")
    return model

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
    sources: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 2048,
    spread_max_length: bool = False,
) -> Dict:
    """
    A simplified preprocessing function that uses a template and tokenizes the full text.
    
    Args:
        sources: Dictionary containing lists of 'prompt', 'response_a', 'response_b', and optionally 'winner'
        tokenizer: The tokenizer to use
        max_len: Maximum sequence length
        spread_max_length: Whether to equally distribute max_length among components
        
    Returns:
        Dictionary containing input_ids, attention_mask, and (optionally) labels tensors
    """
    # Convert list of dicts to dict of lists if necessary
    if isinstance(sources, list):
        # Check if all items have 'winner' to determine if we should collect labels
        has_winner = all('winner' in item for item in sources)
        converted_sources = {
            "prompt": [],
            "response_a": [],
            "response_b": [],
        }
        if has_winner:
            converted_sources["winner"] = []
        for item in sources:
            converted_sources["prompt"].append(item["prompt"])
            converted_sources["response_a"].append(item["response_a"])
            converted_sources["response_b"].append(item["response_b"])
            if has_winner:
                converted_sources["winner"].append(item["winner"])
        sources = converted_sources
    
    # Cohere2 template
    template = """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful assistant. Your task is to read the following prompt carefully. Compare the two responses provided and determine which response better.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}
<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"""

    if spread_max_length:
        # Calculate the special tokens length in the template
        template_special_tokens = tokenizer(
            template.format(prompt="", response_a="", response_b=""),
            return_tensors="pt"
        ).input_ids.shape[1]
        
        # Distribute remaining length among prompt, response_a, and response_b
        component_max_len = (max_len - template_special_tokens) // 3
        
        # Process each component separately
        processed_texts = []
        for prompt, response_a, response_b in zip(
            sources["prompt"],
            sources["response_a"],
            sources["response_b"]
        ):
            # Tokenize and truncate each component
            prompt_tokens = tokenizer(prompt, max_length=component_max_len, 
                                   truncation=True, add_special_tokens=False).input_ids
            response_a_tokens = tokenizer(response_a, max_length=component_max_len, 
                                       truncation=True, add_special_tokens=False).input_ids
            response_b_tokens = tokenizer(response_b, max_length=component_max_len, 
                                       truncation=True, add_special_tokens=False).input_ids
            
            # Decode back to text
            processed_texts.append(template.format(
                prompt=tokenizer.decode(prompt_tokens),
                response_a=tokenizer.decode(response_a_tokens),
                response_b=tokenizer.decode(response_b_tokens)
            ))
    else:
        # Create full texts by filling in template directly
        processed_texts = [
            template.format(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b
            )
            for prompt, response_a, response_b in zip(
                sources["prompt"],
                sources["response_a"],
                sources["response_b"]
            )
        ]
    
    # Tokenize
    tokenized = tokenizer(
        processed_texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    # Create labels tensor if 'winner' exists and is non-empty
    labels = None
    if "winner" in sources and len(sources["winner"]) > 0:
        labels = torch.tensor([1 if w == 'model_b' else 0 for w in sources["winner"]], dtype=torch.long)
    
    # Prepare return dictionary
    result = {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
    }
    if labels is not None:
        result["labels"] = labels
    
    return result

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
            sequence = [im_start] + user_tokens + p_tok + ra_tok + rb_tok + [im_end] + nl_tokens + [im_start] + assistant_tokens
            sequence = sequence[:max_len]
            sequence += [tokenizer.pad_token_id] * (max_len - len(sequence))
            input_ids.append(sequence)
    else:
        combined_texts = [p + ra + rb for p, ra, rb in zip(prompts, responses_a, responses_b)]
        
        tokenized = tokenizer(
            combined_texts,
            add_special_tokens=False,
            max_length=max_len - len([im_start] + user_tokens + [im_end] + nl_tokens + [im_start] + assistant_tokens),
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

    labels = torch.tensor([1 if w == 'model_b' else 0 for w in sources["winner"]], dtype=torch.long)
    
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data: Union[List, pd.DataFrame], tokenizer: transformers.PreTrainedTokenizer, max_len: int, spread_max_length: bool = False):
        super(SupervisedDataset, self).__init__()
        
        # Process raw data to consistent format
        sources = self._process_raw_data(raw_data)
        
        # Let preprocess handle the full dictionary including conversations and winner
        data_dict = simple_preprocess(sources, tokenizer, max_len, spread_max_length)

        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.labels = data_dict.get("labels")  # Use get to handle missing labels

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
        item = {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
        }
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


class LazySupervisedDataset(Dataset):
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

        ret = simple_preprocess([self.raw_data[i]], self.tokenizer, self.max_len, self.spread_max_length)
        processed_ret = {
            'input_ids': ret['input_ids'][0],
            'attention_mask': ret['attention_mask'][0],
        }
        if 'labels' in ret:
            processed_ret['labels'] = ret['labels'][0]
        self.cached_data_dict[i] = processed_ret
        return processed_ret

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

    # Load training data if in training mode
    train_dataset = None
    if data_args.data_path:
        train_data = load_data(data_args.data_path)
        train_dataset = dataset_cls(
            train_data, tokenizer=tokenizer, max_len=data_args.my_max_len, 
            spread_max_length=data_args.spread_max_length
        )

    # Load evaluation data if provided
    eval_dataset = None
    if data_args.eval_data_path:
        eval_data = load_data(data_args.eval_data_path)
        eval_dataset = dataset_cls(
            eval_data, tokenizer=tokenizer, max_len=data_args.my_max_len,
            spread_max_length=data_args.spread_max_length
        )

    # Load prediction data if in predict mode
    test_dataset = None
    if data_args.my_do_predict and data_args.test_data_path:
        test_data = load_data(data_args.test_data_path)
        test_dataset = dataset_cls(
            test_data, tokenizer=tokenizer, max_len=data_args.my_max_len,
            spread_max_length=data_args.spread_max_length
        )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset
    )

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
        quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
        quantization_config=bnb_config,
        trust_remote_code=False,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        torch_dtype=torch_dtype,
        # device_map="auto"  # Add this, not work when using DeepSpeed 3
    )

    if args.score_layer_path:
        model = load_score_layer_weights(model, args.score_layer_path)

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
        # tokenizer.pad_token = tokenizer.eos_token # This line doesnt matter , since Qwen already has its own pad token
        model.config.pad_token_id = tokenizer.pad_token_id




    return model, peft_config, tokenizer