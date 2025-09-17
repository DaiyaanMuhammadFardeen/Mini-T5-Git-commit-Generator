import os
import subprocess
from queue import Queue
from threading import Thread
from tqdm import tqdm
import glob
import shutil

HF_CACHE_BASE = os.path.expanduser("~/.cache/huggingface/hub/datasets--bigcode--commitpack/snapshots")

TOTAL_FILES = 457
QUEUE = Queue()
idx = 153

# ---------- Find latest snapshot path ----------
def get_snapshot_dir():
    snapshots = glob.glob(os.path.join(HF_CACHE_BASE, "*"))
    if not snapshots:
        raise RuntimeError("No Hugging Face snapshots found yet. Run one download first.")
    # Pick the most recent snapshot (last modified)
    return max(snapshots, key=os.path.getmtime)

# ---------- Helper to clean HF cache ----------
def clean_hf_cache():
    dataset_blobs = glob.glob(os.path.join(HF_CACHE_BASE, "**"), recursive=True)
    for path in dataset_blobs:
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"[!] Failed to delete {path}: {e}")

# ---------- Producer ----------
def producer():
    for i in range(idx, TOTAL_FILES + 1):
        filename = f"python-{i:04d}.jsonl"
        QUEUE.put(filename)

# ---------- Worker ----------
def worker():
    global idx
    pbar = tqdm(total=TOTAL_FILES, desc="Overall Progress", ncols=100)
    while True:
        filename = QUEUE.get()
        if filename is None:
            break

        # --- Download ---
        download_pbar = tqdm(total=1, desc=f"Downloading {filename}", leave=False, ncols=100)
        subprocess.run(["python", "download.py", "-f", filename])
        download_pbar.update(1)
        download_pbar.close()

        # Resolve cache path
        snapshot_dir = get_snapshot_dir()
        json_path = os.path.join(snapshot_dir, "data", "python", filename)

        if not os.path.exists(json_path):
            print(f"[!] File not found in cache: {json_path}")
            QUEUE.task_done()
            continue

        # --- Combine ---
        temp_csv = "temp_combined.csv"
        subprocess.run(["python", "combine.py", "-i", json_path, "-o", temp_csv])

        # --- Partition ---
        parquet_file = f"commitpack_parted_{idx}.parquet"
        subprocess.run([
            "python", "partition.py",
            "-i", temp_csv,
            "-o", parquet_file,
            "--drop-both-empty"
        ])
        idx += 1

        # --- Delete temporary files ---
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        clean_hf_cache()

        pbar.update(1)
        QUEUE.task_done()

    pbar.close()

# ---------- Start threads ----------
Thread(target=producer, daemon=True).start()
Thread(target=worker, daemon=True).start()

# ---------- Wait for queue ----------
QUEUE.join()
print("[+] All files processed!")

