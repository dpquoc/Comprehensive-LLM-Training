import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from sklearn.metrics import log_loss, accuracy_score
from datasets import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    EvalPrediction,
    set_seed,
    is_wandb_available,
)
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback

from peft import get_peft_model
from trl import SFTConfig, SFTTrainer
from trl.trainer.utils import ConstantLengthDataset

from utils import make_supervised_data_module, create_and_prepare_model

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    score_layer_path: str = field(
        default=None,
        metadata={"help": "Path to scorer weights layer for seq classification head of LLM"}
    )
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    num_labels: int = field(
    default=2,
    metadata={"help": "Number of labels for classification"}
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
    

@dataclass
class DataTrainingArguments:
    # dataset_name: Optional[str] = field(
    #     default="timdettmers/openassistant-guanaco",
    #     metadata={"help": "The preference dataset to use."},
    # )
    apply_chat_template: Optional[str] = field(
        default="default",
        metadata={
            "help": "default|chatml|zephyr|none. Pass `default` to apply chat template existed in tokenizer. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    remove_system_message: Optional[str] = field(
        default=True,
        metadata={"help": "If True delete the system message in chat if exists"},
    )
    my_max_len: Optional[int] = field(
        default=2048,
        metadata={"help": "Max length of sequence data"}  # Changed from set to dictionary
    )
    spread_max_length: Optional[bool] = field(
        default=False,
        metadata={"help": "If True delete spread the max length over all prompt and repsonses"},
    )
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."} 
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, each text batch will preprocess during training or predict"},
    )
    # Predict arguments
    test_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test data for prediction"}
    )
    do_predict: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set"}
    )

@dataclass
class SFTClassificationConfig(SFTConfig):
    # Remove num_labels from config since it's already in ModelArguments
    problem_type: Optional[str] = None

class SFTClassificationTrainer(SFTTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SFTClassificationConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable] = None,
        peft_config = None,
        num_labels: int = 2,  # Add num_labels as a direct parameter
    ):
        if isinstance(model, str):
            model = AutoModelForSequenceClassification.from_pretrained(
                model,
                num_labels=num_labels,
                problem_type=args.problem_type if args is not None else None
            )
            
        if args is None:
            args = SFTClassificationConfig(output_dir="tmp_trainer")
            
        # Initialize the parent SFTTrainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )
        
    

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}

def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Model and tokenizer
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args)
    
    # Convert training args to SFTClassificationConfig
    sft_args = SFTClassificationConfig(**training_args.to_dict())
    
    if data_args.do_predict:
        # Initialize trainer with trained model
        trainer = SFTClassificationTrainer(
            model=model,
            args=SFTClassificationConfig(**training_args.to_dict()),
            data_collator=collate_fn,
            processing_class=tokenizer,
            num_labels=model_args.num_labels,
        )

        # Get predictions
        test_dataset = data_module["test_dataset"]
        predictions = trainer.predict(test_dataset)
        
        # Process and save predictions
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
        results = pd.DataFrame({
            "predicted_label": probs.argmax(axis=1),
            **{f"prob_class_{i}": probs[:, i] for i in range(probs.shape[1])}
        })
        
        output_path = os.path.join(training_args.output_dir, "predictions.parquet")
        results.to_parquet(output_path)
        print(f"Saved predictions to {output_path}")
        return

    # Dataset preparation
    data_module = make_supervised_data_module(tokenizer, data_args, max_len=data_args.my_max_len)
    train_dataset = data_module["train_dataset"]
    eval_dataset = data_module["eval_dataset"]
    
    # Initialize SFTClassificationTrainer
    trainer = SFTClassificationTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,  # Add this line
        processing_class=tokenizer,
        peft_config=peft_config,
        num_labels=model_args.num_labels,
    )

    # Rest of your training code remains the same
    trainer.accelerator.print(f"{trainer.model}")
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTClassificationConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)