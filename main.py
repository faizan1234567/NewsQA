"""
fine-tune LLM on Alpaca like QA datasets
"""
import warnings
warnings.filterwarnings("ignore")
import random
import time
import wandb
import logging
import argparse
import yaml
from pathlib import Path
from cfg import from_dict

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


# initilize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

def formatting_prompts_func(examples):
    """format dataset in alpaca format"""
    # Prompte template
    QA_prompt = """
    Below is an instruction that describes a task.  Write a response that appropriately completes the request.

    ### Instruction:
    {}


    ### Response:
    {}
    """
    EOS_TOKEN = tokenizer.eos_token
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # EOS_TOKEN added, otherwise model will generate forevoer
        text = QA_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tiny-llama", help= "name of the model")
    parser.add_argument("--epochs", type = int, default=50, help = "fine tunnig epochs")
    parser.add_argument("--cfg", type = str, default="cfg/fine_tuning.yaml", help="configuration settings")
    parser.add_argument("--dataset", type = str, default="dataset/alpaca_format/qa_chunked_data_5000.json", help = "path to the dataset file")
    opt = parser.parse_args()

    # check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.info("WARNING: Running with CPU could be slow")

    # Init config
    cfg = yaml.safe_load(Path(opt.cfg).open('r'))
    cfg = from_dict(cfg)  # convert dict

    # Initialize weight & baises for experiment tracking
    wandb.init(
    project="QA LLM fine tuning",
    
    config={
    "learning_rate": 2e-5,
    "architecture": f"{opt.model}",
    "dataset": "QA in alpaca format",
    "epochs": opt.epochs,
    })

    max_seq_length = cfg.max_seq_length
    dtype = cfg.dtype
    load_in_4bit = cfg.load_in_4bit # Use 4bit quantization to reduce memory usage
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    tokenizer.padding_side = 'right'

    # Create model
    model = FastLanguageModel.get_peft_model(
                    model,
                    r = 16, #  Suggested 8, 16, 32, 64, 128
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj",],
                    lora_alpha = cfg.lora_alpha,
                    lora_dropout = 0,
                    bias = "none",
                    use_gradient_checkpointing = "unsloth",
                    random_state = cfg.random_state,
                    use_rslora = False,  #Rank stablization LORA
                    loftq_config = None,
                )
    
    # Load dataset and map to the desired format
    logger.info("Load the dataset")
    dataset = load_dataset("json", data_files= opt.dataset, split = "train").train_test_split(test_size = 0.2) 
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 8,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            num_train_epochs = cfg.total_epochs,
            per_device_train_batch_size = cfg.batch_size,
            gradient_accumulation_steps = cfg.gradient_acu_steps,
            warmup_steps = 5,
            learning_rate = cfg.lr,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 2,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "wandb",
            output_dir = "outputs",
            save_steps= 5000,  # Default value, actual saving handled by callback
            save_total_limit=5,
        ),
    )

    print('Starting training')
    trainer_stats = trainer.train()



    




