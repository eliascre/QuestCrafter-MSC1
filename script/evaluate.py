import json
import os
import re

def compute_distinct_n(text_list, n=2):
    """Computes Distinct-N metric for a list of generated texts."""
    ngrams = set()
    total_ngrams = 0
    for text in text_list:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.add(ngram)
            total_ngrams += 1
    
    if total_ngrams == 0:
        return 0.0
    return len(ngrams) / total_ngrams

def evaluate_generations(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    generated_texts = [d['generated'] for d in data]
    
    distinct_1 = compute_distinct_n(generated_texts, n=1)
    distinct_2 = compute_distinct_n(generated_texts, n=2)
    
    print(f"--- Evaluation for {filepath} ---")
    print(f"Total generations: {len(generated_texts)}")
    print(f"Distinct-1 (Unigram diversity): {distinct_1:.4f}")
    print(f"Distinct-2 (Bigram diversity):  {distinct_2:.4f}")
    
    print("\n--- Human Evaluation Rubric (1-5 scale) ---")
    print("For each generation, score the following:")
    print("1. Coherence (logical story flow)")
    print("2. Prompt-faithfulness (respects max length, tone, setting constraints)")
    print("3. Creativity (interesting elements, not generic)\n")
    
    # Analyze common issues (document typical issues)
    # E.g., check average length vs requested (if length requested)
    avg_length = sum(len(t.split()) for t in generated_texts) / len(generated_texts)
    print(f"Average length in words: {avg_length:.1f}")

import sys

if __name__ == "__main__":
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "generations")
    baseline_file = os.path.join(target_dir, "baseline_generations.json")
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        evaluate_generations(filepath)
    else:
        evaluate_generations(baseline_file)

