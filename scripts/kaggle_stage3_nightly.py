#!/usr/bin/env python3
"""
Kaggle Stage 3: Live Learning (Nightly Cron)

This script acts as the nightly research pipeline. Once deployed to live or 
paper trading, the bot generates daily experiences (trade buffers).
Every night at 2 AM, this script:
1. Fine-tunes the Transformer on the latest live experiences.
2. Updates the PPO Meta-Learner (via evolutionary backtest on recent data).
3. Compares the newly trained Candidate against the current Champion.
4. Deploys the new models only if they outperform.
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime
from src.logging_config import get_logger

logger = get_logger("kaggle_stage3")

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

def main():
    logger.info("=====================================================")
    logger.info(f"STAGE 3: NIGHTLY LIVE LEARNING PIPELINE - {datetime.now()}")
    logger.info("=====================================================")
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    os.chdir(project_root)
    
    # 1. Fine-Tune Transformer based on new trade outcomes
    logger.info("\n--- STEP 1: Fine-Tuning Transformer on Live Trades ---")
    transformer_retrain = os.path.join(scripts_dir, "retrain_transformer.py")
    if not run_script(transformer_retrain):
        logger.warning("Transformer retraining failed or skipped (e.g. no new trades).")
        
    # 2. Update PPO Meta-Learner via Evolutionary Backtest
    logger.info("\n--- STEP 2: Updating PPO Meta-Learner (Evolutionary Tournament) ---")
    ppo_script = os.path.join(scripts_dir, "evolutionary_ppo_trainer.py")
    if not run_script(ppo_script):
        logger.error("Evolutionary PPO Trainer failed! Aborting Stage 3.")
        sys.exit(1)
        
    # Note: evolutionary_ppo_trainer.py inherently compares Candidate vs Champion
    # and only saves `champion_model.pth` if the candidate wins.
    
    # 3. Export Updated Models to Kaggle Working Directory for Deployment
    logger.info("\n--- STEP 3: Exporting Updated Models for Deployment ---")
    kaggle_out = "/kaggle/working/models"
    os.makedirs(kaggle_out, exist_ok=True)
    
    models_dir = os.path.join(project_root, "models")
    
    for filename in ["grok_gqa_v9_best.pth", "champion_model.pth", "ppo_meta.zip", "feature_scaler.pkl"]:
        src = os.path.join(models_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(kaggle_out, filename))
            logger.info(f"✅ Synced {filename} -> {kaggle_out}")
            
    logger.info("=====================================================")
    logger.info("STAGE 3 COMPLETE. Nightly adaptation finished successfully.")
    logger.info("=====================================================")

if __name__ == "__main__":
    main()
