import torch
import json
import os
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
# [Updated Import] Use SFTConfig for newer TRL versions
from trl import SFTConfig, SFTTrainer

# Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_PATH = "fine_tune/labeled_dataset.jsonl"
OUTPUT_DIR = "fine_tune/results/Llama-3.1-Comedy-Adapter"

def load_jsonl_data(file_path):
    """
    Loads JSONL and ensures the assistant's content is a string, not a list.
    """
    processed_data = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    print(f"Loading JSONL data from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = entry["messages"]
            new_messages = []
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                # Convert list content (JSON object) to string text
                if role == "assistant" and isinstance(content, list):
                    formatted_lines = []
                    for item in content:
                        c_type = item.get('type', 'Unknown')
                        c_text = item.get('text', '')
                        formatted_lines.append(f"({c_type}): \"{c_text}\"")
                    content = "\n".join(formatted_lines)
                
                new_messages.append({"role": role, "content": str(content)})
            
            processed_data.append({"messages": new_messages})
        
    return Dataset.from_list(processed_data)

def formatting_prompts_func(example):
    """
    Formats a SINGLE example into the Llama 3 prompt format.
    [Fix] Removed the double loop causing the TypeError.
    """
    output_texts = []
    
    # example['messages'] is a list of dicts: [{'role':..}, {'role':..}]
    messages = example['messages'] 
    
    text = "<|begin_of_text|>"
    for message in messages:
        # Check if message is valid dict
        if not isinstance(message, dict):
            continue
            
        role = message["role"]
        content = message["content"]
        
        # Apply Llama 3 special tokens
        text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    
    # SFTTrainer expects a list of strings
    output_texts.append(text)
        
    return output_texts

def train():
    # 1. Load Dataset
    try:
        dataset = load_jsonl_data(DATASET_PATH)
        print(f"Loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fixed: SFTTrainer needs right padding

    # 4. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Optional: You can attach peft config to model here, or let SFTTrainer handle it.
    # Letting SFTTrainer handle it via the argument is usually cleaner.

    # 6. Training Configuration (Using SFTConfig)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=2048,        # moved here
        packing=False,              # moved here
        dataset_text_field="text",  # required when using formatting_func
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        optim="paged_adamw_32bit",
        report_to="none",
        lr_scheduler_type="cosine",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    # 8. Train & Save
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()