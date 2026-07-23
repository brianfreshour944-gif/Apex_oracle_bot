"""Genetic Algorithm Researcher.

Evolves the best trading parameters over multiple generations using 
cross-breeding, mutation, and Monte Carlo survival-of-the-fittest.
"""

import sys
import os
import asyncio
import random
import copy
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.backtest import run_backtest, BacktestResult, run_monte_carlo_analysis
from src.logging_config import get_logger

logger = get_logger("genetic_researcher")

# Genetic Algorithm Parameters
POPULATION_SIZE = 20
GENERATIONS = 5
MUTATION_RATE = 0.2
ELITISM_COUNT = 3

# Gene pools
PARAM_BOUNDS = {
    "STOP_LOSS_PCT": (0.01, 0.10),
    "PROFIT_TARGET_PCT": (0.01, 0.15),
    "RSI_OVERSOLD": (15.0, 45.0),
    "RSI_OVERBOUGHT": (55.0, 85.0),
}

def generate_random_genome() -> Dict[str, Any]:
    """Create a random parameter configuration."""
    return {
        "STOP_LOSS_PCT": round(random.uniform(*PARAM_BOUNDS["STOP_LOSS_PCT"]), 3),
        "PROFIT_TARGET_PCT": round(random.uniform(*PARAM_BOUNDS["PROFIT_TARGET_PCT"]), 3),
        "RSI_OVERSOLD": round(random.uniform(*PARAM_BOUNDS["RSI_OVERSOLD"]), 1),
        "RSI_OVERBOUGHT": round(random.uniform(*PARAM_BOUNDS["RSI_OVERBOUGHT"]), 1),
    }

def crossover(parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
    """Combine genes from two parents."""
    child = {}
    for key in PARAM_BOUNDS.keys():
        child[key] = parent_a[key] if random.random() > 0.5 else parent_b[key]
    return child

def mutate(genome: Dict[str, Any]) -> Dict[str, Any]:
    """Randomly mutate genes."""
    child = copy.deepcopy(genome)
    for key, bounds in PARAM_BOUNDS.keys():
        if random.random() < MUTATION_RATE:
            if isinstance(child[key], float):
                # Nudge by a random amount
                nudge = child[key] * random.uniform(-0.2, 0.2)
                child[key] = round(max(bounds[0], min(bounds[1], child[key] + nudge)), 3)
    return child


async def evaluate_fitness(symbol: str, config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """Run a backtest and Monte Carlo to determine fitness score."""
    originals = {k: getattr(settings, k) for k in config.keys()}
    try:
        for k, v in config.items():
            setattr(settings, k, v)
            
        res = await run_backtest(symbol=symbol, n_bars=800, start_equity=10000.0, seed=seed, regime="all")
        mc_res = run_monte_carlo_analysis(res, n_simulations=500)
        
        # Fitness penalizes Risk of Ruin severely
        fitness = res.sharpe * 100 + res.total_return_pct
        if mc_res["risk_of_ruin_pct"] > 5.0:
            fitness -= 1000  # Blow up penalty
            
        return {
            "config": config,
            "fitness": fitness,
            "return": res.total_return_pct,
            "sharpe": res.sharpe,
            "mc_risk_of_ruin": mc_res["risk_of_ruin_pct"]
        }
    finally:
        for k, v in originals.items():
            setattr(settings, k, v)


async def main():
    logger.info(f"Starting Genetic Algorithm: {POPULATION_SIZE} pop, {GENERATIONS} gens")
    symbol = "BTC/USD"
    
    # 1. Evaluate Baseline
    baseline_config = {k: getattr(settings, k) for k in PARAM_BOUNDS.keys()}
    baseline_res = await evaluate_fitness(symbol, baseline_config, seed=99)
    logger.info(f"Baseline Fitness: {baseline_res['fitness']:.2f} (Sharpe: {baseline_res['sharpe']:.2f}, RoR: {baseline_res['mc_risk_of_ruin']:.1f}%)")
    
    # 2. Initialize Population
    population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    # Seed the baseline as one of the population to ensure we never regress
    population[0] = baseline_config
    
    best_ever = baseline_res
    
    for generation in range(GENERATIONS):
        logger.info(f"--- Generation {generation + 1}/{GENERATIONS} ---")
        
        # Evaluate
        evaluations = []
        for genome in population:
            ev = await evaluate_fitness(symbol, genome, seed=99)
            evaluations.append(ev)
            
        # Sort by fitness descending
        evaluations.sort(key=lambda x: x["fitness"], reverse=True)
        
        gen_best = evaluations[0]
        logger.info(f"Gen {generation+1} Best Fitness: {gen_best['fitness']:.2f} | Config: {gen_best['config']}")
        
        if gen_best["fitness"] > best_ever["fitness"]:
            best_ever = gen_best
            
        # Next generation
        next_population = []
        
        # Elitism
        for i in range(ELITISM_COUNT):
            next_population.append(evaluations[i]["config"])
            
        # Breed the rest
        while len(next_population) < POPULATION_SIZE:
            # Tournament selection
            parent_a = random.choice(evaluations[:POPULATION_SIZE//2])["config"]
            parent_b = random.choice(evaluations[:POPULATION_SIZE//2])["config"]
            
            child = crossover(parent_a, parent_b)
            child = mutate(child)
            next_population.append(child)
            
        population = next_population
        
    logger.info("Evolution Complete.")
    
    # 3. Generate Report
    report_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'research_report.md')
    
    with open(report_path, "w") as f:
        f.write("# 🧬 Genetic Quant Research Report\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 Baseline Model\n")
        for k, v in baseline_config.items():
            f.write(f"- **{k}:** {v}\n")
        f.write(f"**Baseline Fitness:** {baseline_res['fitness']:.2f}\n")
        f.write(f"**Baseline Risk of Ruin:** {baseline_res['mc_risk_of_ruin']:.1f}%\n\n")
        
        f.write("## 🏆 Evolutionary Champion\n")
        if best_ever["fitness"] <= baseline_res["fitness"]:
            f.write("The Genetic Algorithm could not beat the baseline. The production model remains mathematically optimal.\n")
        else:
            f.write("The Genetic Algorithm has bred a superior configuration!\n")
            for k, v in best_ever["config"].items():
                f.write(f"- **{k}:** {v}\n")
            f.write(f"\n**Champion Fitness:** {best_ever['fitness']:.2f}\n")
            f.write(f"**Champion Return:** {best_ever['return']:.2f}%\n")
            f.write(f"**Champion Sharpe:** {best_ever['sharpe']:.2f}\n")
            f.write(f"**Champion Risk of Ruin:** {best_ever['mc_risk_of_ruin']:.1f}%\n")
            
    logger.info(f"Genetic report generated at {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
