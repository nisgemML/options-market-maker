"""
Backtest engine smoke tests.
Validates: runs without error, output shape correct, Sharpe computable.
"""
import numpy as np
import pytest

from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.scenario import ScenarioConfig


def make_config(n_steps: int = 50) -> BacktestConfig:
    return BacktestConfig(
        scenario=ScenarioConfig(S0=100.0, sigma=0.20, T=0.25, n_steps=n_steps, seed=1),
        strikes_pct=[0.95, 1.00, 1.05],
        expiry_years=0.25,
        sigma=0.20,
    )


def test_backtest_runs():
    config = make_config()
    engine = BacktestEngine(config)
    result = engine.run()
    assert result is not None
    assert len(result.records) == config.scenario.n_steps + 1


def test_backtest_record_columns():
    result = BacktestEngine(make_config()).run()
    expected = {"step", "S", "net_delta", "net_gamma", "mtm", "n_fills", "hedge_cost"}
    assert expected.issubset(set(result.records.columns))


def test_backtest_sharpe_finite():
    result = BacktestEngine(make_config(n_steps=100)).run()
    sharpe = result.sharpe
    assert np.isfinite(sharpe)


def test_backtest_max_drawdown_nonpositive():
    result = BacktestEngine(make_config()).run()
    assert result.max_drawdown <= 0.0


def test_backtest_custom_path():
    config = make_config(n_steps=30)
    engine = BacktestEngine(config)
    price_path = np.linspace(100, 110, 31)  # simple rising path
    result = engine.run(price_path=price_path)
    assert len(result.records) == 31
    assert result.records["S"].iloc[-1] == pytest.approx(110.0)


def test_backtest_summary_string():
    result = BacktestEngine(make_config()).run()
    summary = result.summary()
    assert "Sharpe" in summary
    assert "Total PnL" in summary
