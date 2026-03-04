# Tuned Generation (`script/tuned_generate.py`)

## Purpose
Like `baseline.py`, this script generates text outputs from test prompts. However, it applies the tuned LoRA weights trained in `train.py`, measuring the concrete improvements made.

## Tasks Performed
1. **Adapter Loading**: Loads the base `distilgpt2` and strategically patches it with the adapter matrix saved inside `models/tuned_questcrafter`.
2. **Prompt Generation**: Mirrors the prompt loop testing variations of levels, tones, and settings.
3. **Pipelining**: Feeds those targeted prompts through the adjusted model to generate quests.
4. **Saving**: Logs the exact outputs into `data/generations/tuned_generations.json` for analysis against the baseline.

## How to Run
```bash
python script/tuned_generate.py
```
