import json
import os
from transformers import pipeline, set_seed

def generate_prompts():
    settings = ["forest", "desert", "cyberpunk", "medieval city", "dark cave"]
    levels = [1, 5, 10]
    tones = ["epic", "humorous", "dark", "casual"]
    
    prompts = []
    for s in settings:
        for l in levels:
            for t in tones:
                prompt_text = f"Create a level-{l} quest in a {s} space with a {t} tone."
                # Adding the control tokens for W4 compatibility early testing
                # <LEVEL={l}><SETTING={s}><TONE={t}>
                prompts.append(prompt_text)
    return prompts

def main():
    print("Loading distilgpt2 baseline...")
    generator = pipeline('text-generation', model='distilgpt2', device="cpu")
    set_seed(42)
    
    prompts = generate_prompts()
    print(f"Generated {len(prompts)} test prompts.")
    
    results = []
    
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "generations")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    out_file = os.path.join(target_dir, "baseline_generations.json")
    
    print(f"Generating baseline outputs to {out_file}...")
    for i, p in enumerate(prompts):
        out = generator(p, max_new_tokens=50, num_return_sequences=1, truncation=True)
        gen_text = out[0]['generated_text']
        results.append({
            "prompt": p,
            "generated": gen_text
        })
        
        # Print the first few for visual inspection
        if i < 3:
            print(f"---\nPrompt: {p}\nGeneration:\n{gen_text}\n")
            
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("Done! Check data/baseline_generations.json for full results.")

if __name__ == "__main__":
    main()
