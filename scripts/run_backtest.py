#!/usr/bin/env python
"""
CLI: run a full options MM backtest and print results.

Usage:
    python scripts/run_backtest.py --steps 252 --vol 0.20 --seed 42
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.scenario import ScenarioConfig
from src.hedging.delta_hedger import HedgeFrequency
from src.quoting.market_maker import MMParams
from src.risk.metrics import RiskMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Options MM Backtest")
    p.add_argument("--steps",    type=int,   default=252)
    p.add_argument("--vol",      type=float, default=0.20)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--S0",       type=float, default=100.0)
    p.add_argument("--hedge",    choices=["band", "periodic", "gamma"], default="band")
    p.add_argument("--plot",     action="store_true")
    p.add_argument("--out",      type=str,   default="backtest_result.png")
    return p.parse_args()


def main():
    args = parse_args()

    freq = {
        "band":     HedgeFrequency.BAND,
        "periodic": HedgeFrequency.PERIODIC,
        "gamma":    HedgeFrequency.GAMMA,
    }[args.hedge]

    config = BacktestConfig(
        scenario=ScenarioConfig(
            S0=args.S0, sigma=args.vol, T=1.0,
            n_steps=args.steps, seed=args.seed
        ),
        strikes_pct=[0.90, 0.95, 1.00, 1.05, 1.10],
        expiry_years=0.25,
        sigma=args.vol,
        hedge_freq=freq,
        mm_params=MMParams(base_spread_vol=0.005),
    )

    print(f"Running backtest: {args.steps} steps, vol={args.vol:.0%}, hedge={args.hedge}")
    engine = BacktestEngine(config)
    result = engine.run()
    print(result.summary())

    pnl_series = result.records["mtm"].diff().dropna().values
    metrics = RiskMetrics.summary(pnl_series)
    print("\nRisk Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")

    if args.plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        df = result.records

        axes[0].plot(df["S"], label="Spot")
        axes[0].set_title("Spot Price Path")
        axes[0].set_ylabel("Price ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df["mtm"], label="MtM P&L", color="green")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_title("Mark-to-Market P&L")
        axes[1].set_ylabel("P&L ($)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(df["net_delta"], label="Net Delta", color="orange")
        axes[2].axhline(0, color="black", linewidth=0.5)
        axes[2].set_title("Net Delta Exposure")
        axes[2].set_ylabel("Delta")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
