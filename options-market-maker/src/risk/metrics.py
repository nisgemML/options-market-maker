"""P&L risk metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


class RiskMetrics:
    """Compute standard risk metrics from a P&L series."""

    @staticmethod
    def sharpe(pnl: np.ndarray, periods_per_year: float = 252.0) -> float:
        pnl = np.asarray(pnl)
        if pnl.std() == 0:
            return 0.0
        return float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(cumulative_pnl: np.ndarray) -> float:
        """Max drawdown as a fraction of peak."""
        c = np.asarray(cumulative_pnl)
        peak = np.maximum.accumulate(c)
        dd = np.where(peak != 0, (c - peak) / np.abs(peak), 0.0)
        return float(dd.min())

    @staticmethod
    def var(pnl: np.ndarray, confidence: float = 0.95) -> float:
        """Historical VaR at given confidence level (positive = loss)."""
        return float(-np.percentile(pnl, (1 - confidence) * 100))

    @staticmethod
    def cvar(pnl: np.ndarray, confidence: float = 0.95) -> float:
        """Conditional VaR (expected shortfall)."""
        threshold = RiskMetrics.var(pnl, confidence)
        tail = pnl[pnl <= -threshold]
        return float(-tail.mean()) if len(tail) > 0 else threshold

    @staticmethod
    def summary(pnl_series: np.ndarray, periods_per_year: float = 252.0) -> dict:
        cumulative = np.cumsum(pnl_series)
        return {
            "total_pnl":    float(cumulative[-1]),
            "mean_daily":   float(pnl_series.mean()),
            "std_daily":    float(pnl_series.std()),
            "sharpe":       RiskMetrics.sharpe(pnl_series, periods_per_year),
            "max_drawdown": RiskMetrics.max_drawdown(cumulative),
            "var_95":       RiskMetrics.var(pnl_series, 0.95),
            "cvar_95":      RiskMetrics.cvar(pnl_series, 0.95),
            "win_rate":     float((pnl_series > 0).mean()),
        }
