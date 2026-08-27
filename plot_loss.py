#!/usr/bin/env python3
"""Plot training loss curves from grungeGPT loss log JSON.

Usage:
    python3 plot_loss.py checkpoints/grungegpt/loss.json
    python3 plot_loss.py checkpoints/grungegpt/loss.json --output loss.png --smooth 20
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_loss_log(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Exponential moving average smoothing."""
    if window <= 1:
        return values
    alpha = 2.0 / (window + 1)
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def plot_loss(entries: list[dict], output: Path, smooth_window: int) -> None:
    steps = np.array([e["step"] for e in entries])
    losses = np.array([e["loss"] for e in entries])
    elapsed = np.array([e["elapsed"] for e in entries])
    smoothed = smooth(losses, smooth_window)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("grungeGPT Training Loss", fontsize=14, fontweight="bold")

    ax1.plot(steps, losses, alpha=0.3, color="#4a90d9", linewidth=0.8, label="raw")
    ax1.plot(steps, smoothed, color="#e74c3c", linewidth=2, label=f"EMA (w={smooth_window})")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss vs Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    hours = elapsed / 3600
    ax2.plot(hours, losses, alpha=0.3, color="#4a90d9", linewidth=0.8, label="raw")
    ax2.plot(hours, smoothed, color="#e74c3c", linewidth=2, label=f"EMA (w={smooth_window})")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.set_title("Loss vs Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved plot to {output}")

    final_loss = losses[-1]
    min_loss = losses.min()
    total_hours = elapsed[-1] / 3600
    print(f"final loss: {final_loss:.4f}  |  min loss: {min_loss:.4f}  |  total time: {total_hours:.2f}h")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grungeGPT training loss")
    parser.add_argument("loss_log", type=Path, help="Path to loss JSON file")
    parser.add_argument("--output", "-o", type=Path, default=Path("loss.png"), help="Output PNG path")
    parser.add_argument("--smooth", "-s", type=int, default=20, help="EMA smoothing window")
    args = parser.parse_args()

    if not args.loss_log.exists():
        print(f"error: {args.loss_log} not found", file=sys.stderr)
        sys.exit(1)

    entries = load_loss_log(args.loss_log)
    if not entries:
        print("error: loss log is empty", file=sys.stderr)
        sys.exit(1)

    plot_loss(entries, args.output, args.smooth)


if __name__ == "__main__":
    main()
