"""
Inventory skew module.

Adjusts bid/ask quotes based on current delta/gamma exposure
to manage inventory risk. Core idea: if we are long delta,
skew bids down (less eager to buy more delta) and asks down
(more eager to sell delta), so quotes self-correct inventory.

Reference: Avellaneda & Stoikov (2008), adapted for options.
"""
from __future__ import annotations

import numpy as np


class InventorySkew:
    """
    Quote skew as a function of inventory (net delta) and market conditions.

    skew(delta) = -alpha * delta
      where alpha is sensitivity parameter.

    Applied symmetrically:
      bid_adjusted = fair_value + skew - half_spread
      ask_adjusted = fair_value + skew + half_spread

    So positive skew (short delta → want to buy) raises both quotes.
    Negative skew (long delta  → want to sell) lowers both quotes.
    """

    def __init__(
        self,
        alpha: float = 0.01,         # skew sensitivity per unit delta
        max_skew: float = 0.05,      # cap skew at this fraction of fair value
        gamma_adjust: bool = True,   # also widen spread when gamma is high
        gamma_spread_mult: float = 0.5,  # extra spread = gamma_spread_mult * gamma * S
    ) -> None:
        self.alpha = alpha
        self.max_skew = max_skew
        self.gamma_adjust = gamma_adjust
        self.gamma_spread_mult = gamma_spread_mult

    def mid_skew(self, net_delta: float, fair_value: float) -> float:
        """
        Shift of mid-price due to inventory.
        Positive = quote higher (short delta, want to buy).
        Negative = quote lower  (long delta, want to sell).
        """
        raw_skew = -self.alpha * net_delta
        max_abs = self.max_skew * abs(fair_value) if fair_value != 0 else self.max_skew
        return float(np.clip(raw_skew, -max_abs, max_abs))

    def spread_adjustment(self, gamma: float, S: float, base_spread: float) -> float:
        """
        Widen spread when gamma exposure is high (more uncertain P&L).
        Returns adjusted spread (always >= base_spread).
        """
        if not self.gamma_adjust:
            return base_spread
        gamma_component = self.gamma_spread_mult * abs(gamma) * S
        return base_spread + gamma_component

    def adjusted_quotes(
        self,
        fair_value: float,
        base_spread: float,
        net_delta: float,
        gamma: float,
        S: float,
    ) -> tuple[float, float]:
        """
        Compute skew-adjusted bid and ask.

        Returns:
            (bid, ask) — both > 0 (floored at 1e-6)
        """
        skew = self.mid_skew(net_delta, fair_value)
        spread = self.spread_adjustment(gamma, S, base_spread)
        half = spread / 2.0
        bid = fair_value + skew - half
        ask = fair_value + skew + half
        return max(bid, 1e-6), max(ask, 1e-6)
