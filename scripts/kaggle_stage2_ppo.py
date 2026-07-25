#!/usr/bin/env python3
"""
Kaggle Stage 2: Backtest Learning (PPO Meta-Learner)

This script acts as the orchestrator for the second stage of the AI's lifecycle.
It utilizes the pre-trained Transformer (from Stage 1) to evaluate trade 
signals and trains the PPO Meta-Learner to optimize strategy weightings via
a genetic algorithm and massive backtesting across all assets.
"""

import os
import sys
import shutil
import subprocess
from src.logging_config import get_logger

logger = get_logger("kaggle_stage2")

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
    logger.info("STAGE 2: BACKTEST LEARNING (PPO META-LEARNER)")
    logger.info("=====================================================")
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    os.chdir(project_root)
    
    # Check if Stage 1 models exist
    transformer_path = os.path.join(project_root, "models", "grok_gqa_v9_best.pth")
    if not os.path.exists(transformer_path):
        logger.warning("⚠️ grok_gqa_v9_best.pth not found! The backtest will run using fallback heuristics.")
    else:
        logger.info("✅ Found pre-trained grok_gqa_v9_best.pth. Ready for PPO Backtest Learning.")

    # 1. Run Evolutionary PPO Trainer
    logger.info("\n--- STEP 1: Running Evolutionary PPO Tournament ---")
    ppo_script = os.path.join(scripts_dir, "evolutionary_ppo_trainer.py")
    if not run_script(ppo_script):
        logger.error("Evolutionary PPO Trainer failed! Aborting.")
        sys.exit(1)
        
    # 2. Export to Kaggle Working Directory
    logger.info("\n--- STEP 2: Exporting Models to Kaggle Output ---")
    kaggle_out = "/kaggle/working/models"
    os.makedirs(kaggle_out, exist_ok=True)
    
    models_dir = os.path.join(project_root, "models")
    champion_path = os.path.join(models_dir, "champion_model.pth")
    ppo_meta_path = os.path.join(models_dir, "ppo_meta.zip")
    
    if os.path.exists(champion_path):
        shutil.copy(champion_path, os.path.join(kaggle_out, "champion_model.pth"))
        logger.info(f"✅ Saved champion_model.pth -> {kaggle_out}")
        
    if os.path.exists(ppo_meta_path):
        shutil.copy(ppo_meta_path, os.path.join(kaggle_out, "ppo_meta.zip"))
        logger.info(f"✅ Saved ppo_meta.zip -> {kaggle_out}")
        
    logger.info("=====================================================")
    logger.info("STAGE 2 COMPLETE. Ready for Stage 3 (Live Trading).")
    logger.info("=====================================================")

if __name__ == "__main__":
    main()
