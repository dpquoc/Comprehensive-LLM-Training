import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
from sklearn.metrics import log_loss, accuracy_score
from transformers import (
    HfArgumentParser,
    set_seed,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
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
    lazy_preprocess: bool = False



def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Model and tokenizer
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args)
    
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Gradient checkpointing setup
    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}
    
    # Dataset preparation
    data_module = make_supervised_data_module(tokenizer, data_args, max_len=data_args.my_max_len)
    train_dataset = data_module["train_dataset"]
    eval_dataset = data_module["eval_dataset"]
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # peft_config=peft_config,

        # data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    # Print model information
    trainer.accelerator.print(f"{trainer.model}")
    
    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save the final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)