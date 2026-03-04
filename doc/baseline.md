# Baseline Evaluation (`script/baseline.py`)

## Purpose
Tests the raw (un-tuned) conversational capabilities of the core `distilgpt2` model so we can measure the impact of our later fine-tuning stages.

## Tasks Performed
1. **Model Loading**: Initializes the default Hugging Face `pipeline` for text-generation based strictly on the un-trained base parameters.
2. **Prompt Generation**: Constructs an array of 60 standard prompts iterating through variable control arguments for `level`, `setting`, and `tone`.
3. **Inference**: Invokes the pipeline sequentially.
4. **Saving Generation**: Saves the results into a file within the `data/generations/` folder, allowing for post-analysis by the `evaluate.py` module.

## How to Run
```bash
python script/baseline.py
```
