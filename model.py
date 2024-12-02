import argparse
import torch
from helper.pdf_reader import create_conversation, conevrt_to_jsonl
from helper import config
 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import setup_chat_format
from datasets import load_dataset
from peft import LoraConfig 
from trl import SFTTrainer
 
def main(args):
    conevrt_to_jsonl('./movies/manual/american_psycho.txt', 'BATEMAN') #TODO: Change to an automated process
    
    # Base model id
    model_id = config.MODEL_ID
 
    # Finetuned model id
    output_directory=config.OUTPUT_DIRECTORY
    peft_model_id=config.PEFT_MODEL_ID
 
    # Load training data and split
    train_dataset = load_dataset("json", data_files='./movie_scripts/final_scripts/BATEMAN_lines.jsonl', split="train") # [file for file in config.MOVIE_OUTPUT_FINAL]
    #new_dataset = train_dataset.map(create_conversation, batched=False)
    
    # Split dataset into 90-10%
    new_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    torch.utils.checkpoint.use_reentrant=True
 
    # Configure the Bits and Bites quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16  # Change from Niklas / Different from Phil Schmid's blog post
    )
 
    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16, # Change from Niklas / Different from Phil Schmid's blog post
        quantization_config=bnb_config,
        return_dict=False
    )
    model.config.use_cache = False
 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure the tokenizer's chat template is cleared before setup
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = None  # Clear the existing template if present
 
    # Use for the training the chat format
    model, tokenizer = setup_chat_format(model, tokenizer)
 
    peft_config = LoraConfig(
            lora_alpha=128, 
            lora_dropout=0.05,
            r=128, # Change from Niklas / Different from Phil Schmid's blog post
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"
    )
 
    args = TrainingArguments(
        output_dir=output_directory+"checkpoints", # The output directory where the model predictions and checkpoints will be written.
        logging_dir=output_directory+"logs", # Tensorboard log directory. Will default to runs/**CURRENT_DATETIME_HOSTNAME**.
        logging_strategy="epoch",
        eval_strategy="epoch", # Evaluate at the end of each epoch.
        save_strategy="epoch", # Save checkpoints at the end of each epoch.
        num_train_epochs=4, # Total number of training epochs to perform.          
        per_device_train_batch_size=5, # The batch size per GPU/TPU core/CPU for training.
        gradient_accumulation_steps=2, # Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},# Added by Thomas
        optim="adamw_torch_fused",      
        learning_rate=2e-5,
        fp16=True, # Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training. Change from Niklas / Different from Phil Schmid's blog post
        max_grad_norm=0.3, # Maximum gradient norm (for gradient clipping).                   
        warmup_ratio=0.03, # Number of steps used for a linear warmup from 0 to learning_rate.                  
        lr_scheduler_type="constant",          
        push_to_hub=False,  # Change from Niklas / Different from Phil Schmid's blog post               
        auto_find_batch_size=True # Change from Niklas / Different from Phil Schmid's blog post
    )
 
    # Supervised fine-tuning (or SFT for short) 
    max_seq_length = 1024
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=new_dataset['train'],
        eval_dataset=new_dataset['test'],
        peft_config=peft_config,
        max_seq_length=max_seq_length, # maximum packed length 
        tokenizer=tokenizer,
        dataset_kwargs={
            "add_special_tokens": False, 
            "append_concat_token": False,
        }
    )
 
    # Train the model
    trainer.train()
 
    # Save the model and tokenizer
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
 
    # Clean-up
    del model
    del trainer
    torch.cuda.empty_cache()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)