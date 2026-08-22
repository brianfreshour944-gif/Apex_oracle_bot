#!/usr/bin/env python3
"""
Nightly Research Pipeline Orchestrator

This script executes the complete autonomous daily research loop:
1. Retrains the Transformer on the latest live trade outcomes.
2. Runs the Universal Multi-Asset Evolutionary Trainer.
3. Updates the PPO Meta-Learner weights if a new candidate champion emerges.

Schedule this to run daily (e.g., at 2:00 AM) via cron.
"""

import sys
import os
import subprocess
from datetime import datetime


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.logging_config import get_logger

logger = get_logger("nightly_orchestrator")

def run_script(script_path: str):
    logger.info(f"Starting execution: {script_path}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        for line in result.stdout.splitlines():
            logger.info(f"[{os.path.basename(script_path)}] {line}")
        logger.info(f"✅ Successfully completed: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to execute {script_path}")
        for line in e.stdout.splitlines():
            logger.info(f"[{os.path.basename(script_path)}] {line}")
        for line in e.stderr.splitlines():
            logger.error(f"[{os.path.basename(script_path)}] {line}")
        return False

def main() -> int:
    logger.info("==================================================")
    logger.info(f"NIGHTLY RESEARCH PIPELINE STARTED: {datetime.now()}")
    logger.info("==================================================")
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Retrain Transformer based on new trade outcomes in replay buffer
    transformer_trainer = os.path.join(scripts_dir, "retrain_transformer.py")
    if not run_script(transformer_trainer):
        logger.warning("Transformer retraining failed or skipped. Continuing pipeline...")
        
    # 2. Run 500-generation multi-asset evolutionary PPO trainer
    # This automatically fetches data, mutates, backtests, runs walk-forward/monte carlo, and trains PPO.
    ppo_trainer = os.path.join(scripts_dir, "evolutionary_ppo_trainer.py")
    if not run_script(ppo_trainer):
        logger.error("Evolutionary PPO Trainer failed. Aborting pipeline.")
        sys.exit(1)
        
    logger.info("==================================================")
    logger.info(f"NIGHTLY RESEARCH PIPELINE COMPLETED: {datetime.now()}")
    logger.info("==================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
