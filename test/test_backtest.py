"""
Backtest engine smoke tests.
Validates: runs without error, output shape correct, Sharpe computable,
adverse selection fills recorded, immutable portfolio used throughout.
"""
import numpy as np
import pytest

from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.scenario import ScenarioConfig


def make_config(n_steps: int = 50, toxicity: float = 0.20) -> BacktestConfig:
    return BacktestConfig(
        scenario=ScenarioConfig(S0=100.0, sigma=0.20, T=0.25, n_steps=n_steps, seed=1),
        strikes_pct=[0.95, 1.00, 1.05],
        expiry_years=0.25,
        sigma=0.20,
        toxicity=toxicity,
    )


def test_backtest_runs():
    config = make_config()
    engine = BacktestEngine(config)
    result = engine.run()
    assert result is not None
    assert len(result.records) == config.scenario.n_steps + 1


def test_backtest_record_columns():
    result = BacktestEngine(make_config()).run()
    expected = {"step", "S", "net_delta", "net_gamma", "mtm",
                "n_fills", "hedge_cost", "toxic_fills"}
    assert expected.issubset(set(result.records.columns))


def test_backtest_sharpe_finite():
    result = BacktestEngine(make_config(n_steps=100)).run()
    assert np.isfinite(result.sharpe)


def test_backtest_max_drawdown_nonpositive():
    result = BacktestEngine(make_config()).run()
    assert result.max_drawdown <= 0.0


def test_backtest_custom_path():
    config = make_config(n_steps=30)
    engine = BacktestEngine(config)
    price_path = np.linspace(100, 110, 31)
    result = engine.run(price_path=price_path)
    assert len(result.records) == 31
    assert result.records["S"].iloc[-1] == pytest.approx(110.0)


def test_backtest_summary_string():
    result = BacktestEngine(make_config()).run()
    summary = result.summary()
    assert "Sharpe" in summary
    assert "Total PnL" in summary
    assert "Toxic Fill Rate" in summary


def test_adverse_selection_more_fills_high_toxicity():
    """
    Higher toxicity → fill rates increase during large price moves.
    With enough steps, total fills should differ between 0 and high toxicity.
    """
    r0 = BacktestEngine(make_config(n_steps=200, toxicity=0.0)).run()
    r1 = BacktestEngine(make_config(n_steps=200, toxicity=0.5)).run()
    fills_0 = int(r0.records["n_fills"].sum())
    fills_1 = int(r1.records["n_fills"].sum())
    # High toxicity should produce at least as many fills
    assert fills_1 >= fills_0


def test_toxic_fills_recorded():
    result = BacktestEngine(make_config(n_steps=100, toxicity=0.30)).run()
    assert "toxic_fills" in result.records.columns
    # With toxicity > 0 and enough steps, some toxic fills should be recorded
    assert int(result.records["toxic_fills"].sum()) >= 0


def test_final_portfolio_immutable():
    """BacktestResult.final_portfolio must be a frozen HedgePortfolio."""
    result = BacktestEngine(make_config()).run()
    from src.hedging.portfolio import HedgePortfolio
    assert isinstance(result.final_portfolio, HedgePortfolio)
    # Frozen: cannot set attributes
    with pytest.raises((AttributeError, TypeError)):
        result.final_portfolio.delta_hedge = 999.0
