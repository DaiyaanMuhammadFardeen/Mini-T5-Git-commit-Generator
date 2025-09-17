import json
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
parser.add_argument("-o", "--output", default="temp_combined.csv", help="Output CSV file")
args = parser.parse_args()

data = []
with open(args.input, "r") as f:
    for line in tqdm(f, desc=f"Processing {args.input}"):
        try:
            json_obj = json.loads(line.strip())
            row = {
                "message": json_obj.get("message", ""),
                "old_contents": json_obj.get("old_contents", ""),
                "new_contents": json_obj.get("new_contents", "")
            }
            data.append(row)
        except json.JSONDecodeError:
            continue

df = pd.DataFrame(data)
df.to_csv(args.output, index=False)
print(f"[+] CSV saved: {args.output}")
