from huggingface_hub import hf_hub_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", required=True, help="File to download, e.g., python-0001.jsonl")
args = parser.parse_args()

repo_id = "bigcode/commitpack"
filename = f"data/python/{args.file}"

print(f"[+] Downloading {filename} ...")
local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset"
)
print(f"[+] File available in cache: {local_path}")

