#!/usr/bin/env python3
"""
Train Decision Transformer on historical decision snapshots.

Run: python -m scripts.train_decision_transformer [--epochs N] [--batch-size N] [--lr N]
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("ALPACA_API_KEY", "test")
os.environ.setdefault("ALPACA_SECRET_KEY", "test")

from src.committee.decision_transformer import train_decision_transformer


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()

    print(f"Training Decision Transformer: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    result = train_decision_transformer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(json.dumps(result, indent=2))

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()