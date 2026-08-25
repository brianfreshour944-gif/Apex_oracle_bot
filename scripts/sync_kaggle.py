#!/usr/bin/env python3
"""
Kaggle Sync & Automated MLOps Script

This script decouples the local trading bot from the heavy nightly research tasks.
It uploads the live experiences to a Kaggle Dataset, triggers the cloud GPU notebook,
waits for the NAS/Evolutionary training to finish, and downloads the new Champion models.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.logging_config import get_logger

logger = get_logger("kaggle_sync")

# Kaggle username config
# Change this to your actual Kaggle username
KAGGLE_USERNAME = "brianfreshour" 
DATASET_SLUG = f"{KAGGLE_USERNAME}/apex-oracle-live-data"
KERNEL_SLUG = f"{KAGGLE_USERNAME}/apex-oracle-nightly-research"

def ensure_kaggle_api():
    try:
        import kaggle
        return True
    except OSError:
        logger.error("Kaggle API key not found! Please place kaggle.json in ~/.kaggle/kaggle.json")
        return False
    except ImportError:
        logger.error("Kaggle package not installed. Run `pip install kaggle`.")
        return False

def push_dataset():
    logger.info("Uploading live experiences to Kaggle Dataset...")
    dataset_dir = "kaggle_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Copy live experiences if it exists
    live_data_path = "data/live_experiences.jsonl"
    if os.path.exists(live_data_path):
        import shutil
        shutil.copy(live_data_path, os.path.join(dataset_dir, "live_experiences.jsonl"))
    else:
        logger.warning(f"No live data found at {live_data_path}. Uploading empty dataset.")
        open(os.path.join(dataset_dir, "live_experiences.jsonl"), 'w').close()

    # Copy historical (bootstrap) experiences if it exists
    hist_data_path = "data/historical_experiences.jsonl"
    if os.path.exists(hist_data_path):
        import shutil
        shutil.copy(hist_data_path, os.path.join(dataset_dir, "historical_experiences.jsonl"))
        logger.info("Historical experiences included in dataset.")

    # Create dataset metadata if it doesn't exist
    meta_path = os.path.join(dataset_dir, "dataset-metadata.json")
    if not os.path.exists(meta_path):
        meta = {
            "title": "Apex Oracle Live Data",
            "id": DATASET_SLUG,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
            
    # Push using Kaggle API
    cmd = ["kaggle", "datasets", "version", "-p", dataset_dir, "-m", "Nightly automated update"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if "error" in result.stderr.lower():
        # If it's a new dataset, we need to create it instead of versioning it
        logger.info("Dataset might not exist yet, attempting to create...")
        cmd_create = ["kaggle", "datasets", "create", "-p", dataset_dir]
        subprocess.run(cmd_create, capture_output=True, text=True)
        
    logger.info("✅ Live data uploaded successfully.")

def trigger_kernel_and_wait():
    logger.info("Triggering Kaggle GPU Notebook...")
    
    kernel_dir = "kaggle_kernel"
    os.makedirs(kernel_dir, exist_ok=True)
    
    import shutil
    shutil.copy("kaggle_nightly_research.ipynb", os.path.join(kernel_dir, "kaggle_nightly_research.ipynb"))
    
    meta_path = os.path.join(kernel_dir, "kernel-metadata.json")
    if not os.path.exists(meta_path):
        meta = {
            "id": KERNEL_SLUG,
            "title": "Apex Oracle Nightly Research",
            "code_file": "kaggle_nightly_research.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [DATASET_SLUG],
            "competition_sources": [],
            "kernel_sources": []
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
            
    # Push kernel
    cmd = ["kaggle", "kernels", "push", "-p", kernel_dir]
    subprocess.run(cmd, capture_output=True, text=True)
    logger.info("✅ Kernel pushed. Waiting for execution to finish (this may take 1-3 hours)...")
    
    # Wait loop
    status = "running"
    while status in ["running", "queued"]:
        time.sleep(120)  # Check every 2 minutes
        cmd_status = ["kaggle", "kernels", "status", KERNEL_SLUG]
        result = subprocess.run(cmd_status, capture_output=True, text=True)
        out = result.stdout.lower()
        if "complete" in out:
            status = "complete"
        elif "error" in out or "cancel" in out:
            status = "error"
            
    if status == "error":
        logger.error("❌ Kaggle Kernel failed! Check the Kaggle website for logs.")
        sys.exit(1)
        
    logger.info("✅ Kernel execution finished successfully!")

def download_models():
    logger.info("Downloading new Champion models from Kaggle...")
    cmd = ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", "models/"]
    subprocess.run(cmd, capture_output=True, text=True)
    logger.info("✅ Models successfully downloaded and synced to local environment!")

def main() -> int:
    if not ensure_kaggle_api():
        return 1
        
    logger.info("=========================================")
    logger.info("🚀 Initiating Kaggle MLOps Nightly Sync")
    logger.info("=========================================")
    
    push_dataset()
    trigger_kernel_and_wait()
    download_models()
    
    logger.info("🎉 Nightly Research Pipeline fully synchronized.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
