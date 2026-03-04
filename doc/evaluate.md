# Evaluation Rubrics (`script/evaluate.py`)

## Purpose
Quantifies the text diversity and outputs a standard framework to human-score the generative output arrays (`data/generations/`).

## Automated Metrics
In generative storytelling, outputs should not collapse into singular repeated outputs. The script analyzes the distinct variation within generations.
- **Distinct-1 (Unigrams)**: Reflects basic vocabulary breadth.
- **Distinct-2 (Bigrams)**: Shows syntactic distinctiveness.
- **Average length (words)**: Validates response limits constraints.

## Human Rubric (1-5 Scale)
1. **Coherence**: Evaluation of the sequence of phrasing logically spanning forward.
2. **Faithfulness**: How closely the resulting text answers the user's specific context requests.
3. **Creativity**: Rewards outputs possessing unique details or concepts non-prevalent across the output batch.

## How to Run
```bash
python script/evaluate.py data/generations/baseline_generations.json
python script/evaluate.py data/generations/tuned_generations.json
```
