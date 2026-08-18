"""Risk limit enforcement."""
from __future__ import annotations

from dataclasses import dataclass

from src.hedging.portfolio import HedgePortfolio


class RiskBreachError(Exception):
    pass


@dataclass
class RiskLimits:
    max_delta:   float = 100.0
    max_gamma:   float = 10.0
    max_vega:    float = 1000.0
    max_theta:   float = 500.0    # absolute value
    max_loss:    float = 10_000.0  # stop-loss in $

    def check(self, portfolio: HedgePortfolio, S: float, pnl: float) -> None:
        """Raise RiskBreachError if any limit is exceeded."""
        greeks = portfolio.net_greeks(S)
        net_delta = portfolio.net_delta(S)

        breaches = []
        if abs(net_delta) > self.max_delta:
            breaches.append(f"delta={net_delta:.1f} > {self.max_delta}")
        if abs(greeks.gamma) > self.max_gamma:
            breaches.append(f"gamma={greeks.gamma:.2f} > {self.max_gamma}")
        if abs(greeks.vega) > self.max_vega:
            breaches.append(f"vega={greeks.vega:.1f} > {self.max_vega}")
        if abs(greeks.theta) > self.max_theta:
            breaches.append(f"theta={greeks.theta:.1f} > {self.max_theta}")
        if pnl < -self.max_loss:
            breaches.append(f"loss={pnl:.2f} > {self.max_loss}")

        if breaches:
            raise RiskBreachError("Risk limit breached: " + "; ".join(breaches))
