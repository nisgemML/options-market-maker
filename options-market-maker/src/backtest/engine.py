"""
Backtest engine — simulates the market maker over a price path.

Loop per step:
  1. Generate new spot price from scenario
  2. Roll time forward (decay all option expiries)
  3. MM generates quotes for each option in the universe
  4. Simulate fills: incoming orders hit MM quotes with fill probability
  5. Delta hedger evaluates trigger; hedges if needed
  6. Record P&L, Greeks, hedge trades

All state is immutable per step — engine returns a complete BacktestResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.pricing import OptionType
from src.hedging.portfolio import HedgePortfolio, OptionPosition
from src.hedging.delta_hedger import DeltaHedger, HedgeFrequency
from src.quoting.market_maker import MarketMaker, MMParams, Quote
from .scenario import GBMScenario, ScenarioConfig


@dataclass
class BacktestConfig:
    """Full backtest configuration."""
    # Scenario
    scenario:       ScenarioConfig = field(default_factory=ScenarioConfig)

    # Options universe
    strikes_pct:    list[float] = field(
        default_factory=lambda: [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    )  # as fraction of S0
    expiry_years:   float = 0.25        # 3-month options
    option_types:   list[OptionType] = field(
        default_factory=lambda: [OptionType.CALL, OptionType.PUT]
    )
    r:              float = 0.05
    q:              float = 0.0
    sigma:          float = 0.20        # flat vol (no surface for simplicity)

    # MM params
    mm_params:      MMParams = field(default_factory=MMParams)

    # Hedging
    hedge_freq:     HedgeFrequency = HedgeFrequency.BAND
    band_threshold: float = 0.05
    tc_bps:         float = 1.0

    # Fill model: probability that a quote is hit each step
    fill_prob_bid:  float = 0.15
    fill_prob_ask:  float = 0.15
    fill_seed:      int   = 99


class StepRecord(NamedTuple):
    step:           int
    S:              float
    net_delta:      float
    net_gamma:      float
    net_vega:       float
    net_theta:      float
    mtm:            float
    option_pnl:     float
    hedge_pnl:      float
    cash:           float
    n_hedge_trades: int
    hedge_cost:     float
    n_fills:        int


@dataclass
class BacktestResult:
    records:        pd.DataFrame
    hedge_trades:   list
    quotes_history: list[list[Quote]]
    final_portfolio: HedgePortfolio
    config:         BacktestConfig

    @property
    def total_pnl(self) -> float:
        return float(self.records["mtm"].iloc[-1] - self.records["mtm"].iloc[0])

    @property
    def sharpe(self) -> float:
        pnl_series = self.records["mtm"].diff().dropna()
        if pnl_series.std() == 0:
            return 0.0
        return float(pnl_series.mean() / pnl_series.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        mtm = self.records["mtm"].values
        peak = np.maximum.accumulate(mtm)
        dd = (mtm - peak) / np.where(peak != 0, peak, 1)
        return float(dd.min())

    @property
    def total_hedge_cost(self) -> float:
        return float(self.records["hedge_cost"].sum())

    def summary(self) -> str:
        return (
            f"BacktestResult:\n"
            f"  Steps:           {len(self.records)}\n"
            f"  Total PnL:       {self.total_pnl:+.2f}\n"
            f"  Sharpe:          {self.sharpe:.3f}\n"
            f"  Max Drawdown:    {self.max_drawdown:.2%}\n"
            f"  Total Hedge Cost:{self.total_hedge_cost:.2f}\n"
            f"  Total Fills:     {int(self.records['n_fills'].sum())}\n"
            f"  Hedge Trades:    {len(self.hedge_trades)}\n"
        )


class BacktestEngine:
    """
    Event-driven backtest loop.

    Design:
      - Pure functional per step: all state threaded explicitly
      - Fill model: Bernoulli trials per quote per step
      - Supports single-path GBM; extend by passing external price paths
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.mm = MarketMaker(config.mm_params)
        self.hedger = DeltaHedger(
            hedge_freq=config.hedge_freq,
            band_threshold=config.band_threshold,
            transaction_cost_bps=config.tc_bps,
        )

    def _build_universe(self, S0: float) -> list[tuple[float, OptionType]]:
        """Return (strike, option_type) pairs for the options universe."""
        cfg = self.config
        universe = []
        for pct in cfg.strikes_pct:
            K = round(S0 * pct, 2)
            for ot in cfg.option_types:
                universe.append((K, ot))
        return universe

    def _simulate_fills(
        self,
        quotes: list[Quote],
        rng: np.random.Generator,
        portfolio: HedgePortfolio,
        S: float,
        T_remaining: float,
        r: float,
        sigma: float,
    ) -> tuple[HedgePortfolio, int]:
        """
        Simulate incoming flow hitting MM quotes.
        Returns (updated_portfolio, n_fills).
        """
        cfg = self.config
        n_fills = 0

        for quote in quotes:
            if quote.pulled:
                continue

            # Bid hit: client sells to MM → MM buys (long)
            if rng.random() < cfg.fill_prob_bid and quote.bid_size > 0:
                pos = OptionPosition(
                    strike=quote.strike,
                    expiry=T_remaining,
                    option_type=quote.option_type,
                    quantity=+1,          # MM buys
                    entry_price=quote.bid,
                    r=r,
                    sigma=sigma,
                )
                portfolio.add_position(pos)
                n_fills += 1

            # Ask hit: client buys from MM → MM sells (short)
            if rng.random() < cfg.fill_prob_ask and quote.ask_size > 0:
                pos = OptionPosition(
                    strike=quote.strike,
                    expiry=T_remaining,
                    option_type=quote.option_type,
                    quantity=-1,          # MM sells
                    entry_price=quote.ask,
                    r=r,
                    sigma=sigma,
                )
                portfolio.add_position(pos)
                n_fills += 1

        return portfolio, n_fills

    def run(self, price_path: np.ndarray | None = None) -> BacktestResult:
        """
        Run the backtest.

        Args:
            price_path: optional (n_steps+1,) array of spot prices.
                        If None, generates a GBM path from config.scenario.
        """
        cfg = self.config

        # Generate path if not provided
        if price_path is None:
            scenario = GBMScenario(cfg.scenario)
            paths = scenario.generate()
            price_path = paths[:, 0]  # first path

        n_steps = len(price_path) - 1
        dt = cfg.scenario.T / cfg.scenario.n_steps

        # Build options universe
        S0 = price_path[0]
        universe = self._build_universe(S0)
        strikes = [k for k, _ in universe]

        # Initial portfolio
        portfolio = HedgePortfolio()
        rng_fill = np.random.default_rng(cfg.fill_seed)

        records: list[StepRecord] = []
        hedge_trades = []
        quotes_history = []
        initial_mtm = 0.0

        for step in range(n_steps + 1):
            S = float(price_path[step])
            T_remaining = max(cfg.expiry_years - step * dt, 1e-5)

            # Roll time on existing positions
            portfolio = portfolio.roll_time(dt if step > 0 else 0.0)

            # Generate quotes
            quotes = []
            for K, ot in universe:
                q = self.mm.quote(
                    S=S, K=K, T=T_remaining, r=cfg.r,
                    option_type=ot, portfolio=portfolio,
                    sigma_override=cfg.sigma, q=cfg.q,
                )
                quotes.append(q)
            quotes_history.append(quotes)

            # Simulate fills
            portfolio, n_fills = self._simulate_fills(
                quotes, rng_fill, portfolio, S, T_remaining,
                cfg.r, cfg.sigma,
            )

            # Delta hedge
            portfolio, hedge_rec = self.hedger.step(portfolio, S, step, dt)
            if hedge_rec:
                hedge_trades.append(hedge_rec)

            # Record state
            greeks = portfolio.net_greeks(S)
            mtm = portfolio.mtm_value(S)
            if step == 0:
                initial_mtm = mtm

            rec = StepRecord(
                step=step,
                S=S,
                net_delta=portfolio.net_delta(S),
                net_gamma=greeks.gamma,
                net_vega=greeks.vega,
                net_theta=greeks.theta,
                mtm=mtm,
                option_pnl=portfolio.option_pnl(S),
                hedge_pnl=portfolio.delta_hedge * S + portfolio.cash,
                cash=portfolio.cash,
                n_hedge_trades=1 if hedge_rec else 0,
                hedge_cost=hedge_rec.cost if hedge_rec else 0.0,
                n_fills=n_fills,
            )
            records.append(rec)

        df = pd.DataFrame(records)

        return BacktestResult(
            records=df,
            hedge_trades=hedge_trades,
            quotes_history=quotes_history,
            final_portfolio=portfolio,
            config=cfg,
        )
