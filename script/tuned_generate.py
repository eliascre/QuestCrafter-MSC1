import json
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_tuned_model():
    model_name = "distilgpt2"
    print(f"Loading base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    adapter_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tuned_questcrafter")
    print(f"Loading tuned adapter from {adapter_path}...")
    
    if not os.path.exists(adapter_path):
        print("Adapter not found. Did you run train.py successfully?")
        return None
        
    model = PeftModel.from_pretrained(base_model, adapter_path)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device="cpu")
    return generator

def generate_prompts():
    settings = ["forest", "desert", "cyberpunk", "medieval city", "dark cave"]
    levels = [1, 5, 10]
    tones = ["epic", "humorous", "dark", "casual"]
    
    prompts = []
    for s in settings:
        for l in levels:
            for t in tones:
                prompt_text = f"Create a level-{l} quest in a {s} space with a {t} tone.\nQuest:"
                prompts.append(prompt_text)
    return prompts

def main():
    generator = load_tuned_model()
    if generator is None: return
    
    prompts = generate_prompts()
    print(f"Generated {len(prompts)} test prompts for tuned generation.")
    
    results = []
    
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "generations")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    out_file = os.path.join(target_dir, "tuned_generations.json")
    
    print(f"Generating tuned outputs to {out_file}...")
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
        
    print("Done! Check data/tuned_generations.json for full results. Run 'python evaluate.py data/tuned_generations.json' to evaluate.")

if __name__ == "__main__":
    main()
