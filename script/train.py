import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def prepare_data(data):
    # Combine prompt and completion for Causal LM
    texts = []
    for d in data:
        full_text = f"Prompt: {d['prompt']}\nQuest: {d['completion']}"
        texts.append({"text": full_text})
    return Dataset.from_list(texts)

def main():
    model_name = "distilgpt2"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configure LoRA
    print("Setting up LoRA PEFT...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    train_file = os.path.join(target_dir, "train.jsonl")
    val_file = os.path.join(target_dir, "val.jsonl")
    
    # Load and process data
    print("Loading data...")
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    
    # Downsample for quick local run (use 500 train / 100 val)
    train_data = train_data[:500]
    val_data = val_data[:100]
    
    train_dataset = prepare_data(train_data)
    val_dataset = prepare_data(val_data)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tuned_questcrafter")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        max_steps=1000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
