import os
import sys
import pandas as pd
import json

def format_and_split(dataset_path):
    print("Formatting to JSONL and creating splits...")
    train_csv = os.path.join(dataset_path, "train.csv")
    
    if not os.path.exists(train_csv):
        print(f"Error: Could not find train.csv in {dataset_path}")
        return
    
    # Load a subset to keep training manageable
    df = pd.read_csv(train_csv, nrows=12000)
    
    formatted_data = []
    for text in df['text']:
        formatted_data.append({
            "prompt": "Write a short story.",
            "completion": str(text)
        })
    
    # 80 / 10 / 10 Split (9600 / 1200 / 1200)
    train_data = formatted_data[:9600]
    val_data = formatted_data[9600:10800]
    test_data = formatted_data[10800:12000]
    
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    def write_jsonl(data, filename):
        with open(os.path.join(target_dir, filename), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
                
    write_jsonl(train_data, "train.jsonl")
    write_jsonl(val_data, "val.jsonl")
    write_jsonl(test_data, "test.jsonl")
    
    print(f"Created splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_dataset>")
        sys.exit(1)
    format_and_split(sys.argv[1])
