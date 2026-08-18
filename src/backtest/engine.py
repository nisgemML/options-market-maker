"""
Backtest engine — simulates the market maker over a price path.

Loop per step:
  1. Generate new spot price from scenario
  2. Roll time forward (decay all option expiries)
  3. MM generates quotes for each option in the universe
  4. Simulate fills with adverse selection (Kyle 1985 / Glosten-Milgrom model)
  5. Delta hedger evaluates trigger; hedges if needed
  6. Record P&L, Greeks, hedge trades

Fill model — adverse selection:
  Bernoulli base fills are augmented with a toxicity parameter that models
  informed trading (Kyle 1985):
    - Informed traders hit the bid when price is about to fall (directional).
    - Informed traders hit the ask when price is about to rise.
  Toxic fills impose adverse realised P&L on the MM beyond the quoted spread.

  fill_prob = base_prob + toxicity * |ΔS_next| / S   (heuristic Kyle proxy)
  This captures the stylised fact that MM fill rates spike before large moves.

  Reference: Kyle, A.S. (1985). Continuous auctions and insider trading.
  Econometrica, 53(6), 1315-1335.
  Glosten, L.R. & Milgrom, P.R. (1985). Bid, ask and transaction prices in a
  specialist market with heterogeneously informed traders. J. Financial Economics.

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
    scenario:           ScenarioConfig = field(default_factory=ScenarioConfig)

    # Options universe
    strikes_pct:        list[float] = field(
        default_factory=lambda: [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    )
    expiry_years:       float = 0.25
    option_types:       list[OptionType] = field(
        default_factory=lambda: [OptionType.CALL, OptionType.PUT]
    )
    r:                  float = 0.05
    q:                  float = 0.0
    sigma:              float = 0.20

    # MM params
    mm_params:          MMParams = field(default_factory=MMParams)

    # Hedging
    hedge_freq:         HedgeFrequency = HedgeFrequency.BAND
    band_threshold:     float = 0.05
    tc_bps:             float = 1.0

    # Fill model
    fill_prob_bid:      float = 0.15    # base fill prob on bid
    fill_prob_ask:      float = 0.15    # base fill prob on ask
    # Adverse selection (Kyle 1985):
    # fill_prob += toxicity * |realised_move| / S
    # 0.0 = pure Bernoulli (no informed flow)
    # 0.3 = ~30% of flow is directional (stylised HFT parameter)
    toxicity:           float = 0.20
    fill_seed:          int   = 99


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
    toxic_fills:    int     # fills attributed to informed flow


@dataclass
class BacktestResult:
    records:         pd.DataFrame
    hedge_trades:    list
    quotes_history:  list[list[Quote]]
    final_portfolio: HedgePortfolio
    config:          BacktestConfig

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

    @property
    def toxic_fill_rate(self) -> float:
        """Fraction of fills attributed to informed/toxic flow."""
        total = int(self.records["n_fills"].sum())
        toxic = int(self.records["toxic_fills"].sum())
        return toxic / total if total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"BacktestResult:\n"
            f"  Steps:            {len(self.records)}\n"
            f"  Total PnL:        {self.total_pnl:+.2f}\n"
            f"  Sharpe:           {self.sharpe:.3f}\n"
            f"  Max Drawdown:     {self.max_drawdown:.2%}\n"
            f"  Total Hedge Cost: {self.total_hedge_cost:.2f}\n"
            f"  Total Fills:      {int(self.records['n_fills'].sum())}\n"
            f"  Toxic Fill Rate:  {self.toxic_fill_rate:.1%}\n"
            f"  Hedge Trades:     {len(self.hedge_trades)}\n"
        )


class BacktestEngine:
    """
    Event-driven backtest loop.

    Design:
      - Pure functional per step: all state threaded explicitly via frozen dataclasses.
      - Fill model: Bernoulli + adverse selection (Kyle 1985 toxicity proxy).
      - Supports single-path GBM; extend by passing external price paths.
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
        S_next: float,              # next step's price — used for adverse selection
        T_remaining: float,
        r: float,
        sigma: float,
    ) -> tuple[HedgePortfolio, int, int]:
        """
        Simulate incoming flow hitting MM quotes with adverse selection.

        Adverse selection (Kyle 1985 proxy):
          Informed traders know the next price move:
            - They hit the bid  (sell to MM) when price is about to fall.
            - They hit the ask  (buy from MM) when price is about to rise.
          This means MM buys at the bid just before a down-move (adverse),
          and sells at the ask just before an up-move (adverse).

        Returns:
            (updated_portfolio, n_fills, n_toxic_fills)
        """
        cfg = self.config
        n_fills = 0
        n_toxic = 0

        # Realised move as fraction of S — proxy for informed trader signal
        dS_frac = abs(S_next - S) / max(S, 1e-6)
        toxic_boost = cfg.toxicity * dS_frac
        price_fell = S_next < S

        for quote in quotes:
            if quote.pulled:
                continue

            # Adverse fill adjustment: informed sellers hit bid before price falls
            bid_toxic_boost = toxic_boost if price_fell else 0.0
            ask_toxic_boost = toxic_boost if not price_fell else 0.0

            # Bid hit: client sells to MM → MM buys (long) — adverse if price falls
            p_bid = min(cfg.fill_prob_bid + bid_toxic_boost, 1.0)
            if rng.random() < p_bid and quote.bid_size > 0:
                pos = OptionPosition(
                    strike=quote.strike,
                    expiry=T_remaining,
                    option_type=quote.option_type,
                    quantity=+1,
                    entry_price=quote.bid,
                    r=r,
                    sigma=sigma,
                )
                portfolio = portfolio.add_position(pos)
                n_fills += 1
                if bid_toxic_boost > 0:
                    n_toxic += 1

            # Ask hit: client buys from MM → MM sells (short) — adverse if price rises
            p_ask = min(cfg.fill_prob_ask + ask_toxic_boost, 1.0)
            if rng.random() < p_ask and quote.ask_size > 0:
                pos = OptionPosition(
                    strike=quote.strike,
                    expiry=T_remaining,
                    option_type=quote.option_type,
                    quantity=-1,
                    entry_price=quote.ask,
                    r=r,
                    sigma=sigma,
                )
                portfolio = portfolio.add_position(pos)
                n_fills += 1
                if ask_toxic_boost > 0:
                    n_toxic += 1

        return portfolio, n_fills, n_toxic

    def run(self, price_path: np.ndarray | None = None) -> BacktestResult:
        """
        Run the backtest.

        Args:
            price_path: optional (n_steps+1,) array of spot prices.
                        If None, generates a GBM path from config.scenario.
        """
        cfg = self.config

        if price_path is None:
            scenario = GBMScenario(cfg.scenario)
            paths = scenario.generate()
            price_path = paths[:, 0]

        n_steps = len(price_path) - 1
        dt = cfg.scenario.T / cfg.scenario.n_steps

        S0 = price_path[0]
        universe = self._build_universe(S0)

        portfolio: HedgePortfolio = HedgePortfolio()
        rng_fill = np.random.default_rng(cfg.fill_seed)

        records: list[StepRecord] = []
        hedge_trades = []
        quotes_history = []

        for step in range(n_steps + 1):
            S = float(price_path[step])
            S_next = float(price_path[min(step + 1, n_steps)])
            T_remaining = max(cfg.expiry_years - step * dt, 1e-5)

            portfolio = portfolio.roll_time(dt if step > 0 else 0.0)

            # Generate quotes
            quotes = [
                self.mm.quote(
                    S=S, K=K, T=T_remaining, r=cfg.r,
                    option_type=ot, portfolio=portfolio,
                    sigma_override=cfg.sigma, q=cfg.q,
                )
                for K, ot in universe
            ]
            quotes_history.append(quotes)

            # Fills with adverse selection
            portfolio, n_fills, n_toxic = self._simulate_fills(
                quotes, rng_fill, portfolio, S, S_next,
                T_remaining, cfg.r, cfg.sigma,
            )

            # Delta hedge
            portfolio, hedge_rec = self.hedger.step(portfolio, S, step, dt)
            if hedge_rec:
                hedge_trades.append(hedge_rec)

            greeks = portfolio.net_greeks(S)
            mtm = portfolio.mtm_value(S)

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
                toxic_fills=n_toxic,
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
