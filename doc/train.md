# Fine-Tuning Pipeline (`script/train.py`)

## Purpose
This script is responsible for adapting the generic `distilgpt2` model to our specific `TinyStories` quest generation task using a lightweight LoRA (Low-Rank Adaptation) approach.

## Tasks Performed
1. **Model & Tokenizer Initialization**: Loads the base model from Hugging Face and initializes the corresponding tokenizer.
2. **LoRA Configuration**: Injects PEFT (Parameter-Efficient Fine-Tuning) LoRA adapters dynamically into the model, vastly reducing the trainable parameters to a fraction of the total parameters (only 8 rank).
3. **Dataset Preparation**: Formats the `processed` jsonl files into HuggingFace dataset format containing `"Prompt: ...\nQuest: ..."` blocks.
4. **Trainer**: Executes fine-tuning for 1 epoch using `Trainer` with logging, saving the final updated LoRA weights into the `models/tuned_questcrafter` index.

## How to Run
```bash
python script/train.py
```
