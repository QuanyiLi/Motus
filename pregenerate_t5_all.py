import os
import subprocess
import sys

root_dir = "/work/vita/lanfeng/vlas/vla/wise_dataset_0.3.2"
wan_path = "/work/vita/lanfeng/vlas/Motus/pretrained_models"

script_path = "/work/vita/lanfeng/vlas/Motus/data/lerobot/add_t5_cache_to_lerobot_dataset.py"

for dirpath, dirnames, filenames in os.walk(root_dir):
    meta_dir = os.path.join(dirpath, "meta")
    if "meta" in dirnames and os.path.isdir(meta_dir) and ("info.json" in os.listdir(meta_dir) or "episodes.jsonl" in os.listdir(meta_dir)):
        print(f"Found dataset at: {dirpath}")
        # The true repo_id should be config_X_train, but dirpath ends in lerobot_data
        repo_id = os.path.basename(os.path.dirname(dirpath)) if os.path.basename(dirpath) == 'lerobot_data' else os.path.basename(dirpath)
        
        cmd = [
            sys.executable, script_path,
            "--repo_id", repo_id,
            "--root", dirpath,
            "--wan_path", wan_path,
            "--text_len", "512",
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Prevent recursing into subdirectories of a valid dataset
        dirnames.clear()

print("Completed T5 preprocessing for all nested datasets.")
