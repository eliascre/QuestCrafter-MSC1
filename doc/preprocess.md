# Preprocessing Script (`script/preprocess.py`)

## Purpose
This script is responsible for downloading the `TinyStories` dataset and forming an instruction-driven dataset ready for fine-tuning our generative model.

## Tasks Performed
1. **Download**: Utilizes `kagglehub` via the `dataset_download()` method to retrieve the latest version of the `tinystories-narrative-classification` dataset directly from Kaggle.
2. **Formatting**: Reads a subset of records from `train.csv` and structures each record into a supervised formatting object containing a `prompt` (the instruction to write a short story) and a `completion` (the target generative path).
3. **Splitting and Saving**: Cuts the dataset down into an 80 / 10 / 10 standard split representing Train, Validation, and Test collections. Finally, it serializes these dictionaries into JSONL formats and saves them to the `data/processed/` directory.

## How to Run
```bash
python script/preprocess.py
```
