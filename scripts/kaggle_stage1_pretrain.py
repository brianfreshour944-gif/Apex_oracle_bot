#!/usr/bin/env python3
"""
Kaggle Stage 1: Historical Pre-Training (Offline)

This script acts as the primary orchestrator for the first stage of the 
AI's lifecycle. It:
1. Generates a deep historical replay dataset (2-10 years across assets).
2. Uses that dataset to train the foundation Transformer model from scratch.
3. Exports the resulting weights and scalers to the Kaggle working directory.
"""

import os
import sys
import shutil
import subprocess

# Add project root to path before importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logging_config import get_logger

logger = get_logger("kaggle_stage1")

def run_script(script_path: str):
    logger.info(f"Executing: {script_path}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        subprocess.run([sys.executable, script_path], env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute {script_path}. Error: {e}")
        return False

def main() -> int:
    logger.info("=====================================================")
    logger.info("STAGE 1: HISTORICAL PRE-TRAINING (OFFLINE)")
    logger.info("=====================================================")
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    
    # Ensure working directory is project root
    os.chdir(project_root)
    
    # 1. Generate Historical Dataset
    logger.info("\n--- STEP 1: Generating Historical Trades ---")
    gen_script = os.path.join(scripts_dir, "generate_replay_dataset.py")
    if not run_script(gen_script):
        logger.error("Dataset generation failed! Aborting.")
        sys.exit(1)
        
    # 2. Train Foundation Model
    logger.info("\n--- STEP 2: Training Transformer Foundation Model ---")
    train_script = os.path.join(scripts_dir, "train_foundation_model.py")
    if not run_script(train_script):
        logger.error("Transformer training failed! Aborting.")
        sys.exit(1)
        
    # 3. Export to Kaggle Working Directory
    logger.info("\n--- STEP 3: Exporting Models to Kaggle Output ---")
    kaggle_out = "/kaggle/working/models"
    os.makedirs(kaggle_out, exist_ok=True)
    
    models_dir = os.path.join(project_root, "models")
    transformer_path = os.path.join(models_dir, "grok_gqa_v9_best.pth")
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    
    if os.path.exists(transformer_path):
        shutil.copy(transformer_path, os.path.join(kaggle_out, "grok_gqa_v9_best.pth"))
        logger.info(f"✅ Saved grok_gqa_v9_best.pth -> {kaggle_out}")
    else:
        logger.warning(f"❌ Could not find {transformer_path}")
        
    if os.path.exists(scaler_path):
        shutil.copy(scaler_path, os.path.join(kaggle_out, "feature_scaler.pkl"))
        logger.info(f"✅ Saved feature_scaler.pkl -> {kaggle_out}")
    else:
        logger.warning(f"❌ Could not find {scaler_path}")
        
    logger.info("=====================================================")
    logger.info("STAGE 1 COMPLETE. Ready for Stage 2 (Backtest Learning).")
    logger.info("=====================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
