"""Shared pytest fixtures."""
import pytest
import numpy as np

from src.pricing import OptionType
from src.hedging.portfolio import HedgePortfolio, OptionPosition


@pytest.fixture
def atm_call_portfolio():
    portfolio = HedgePortfolio()
    pos = OptionPosition(
        strike=100.0, expiry=0.25, option_type=OptionType.CALL,
        quantity=1.0, entry_price=5.0, r=0.05, sigma=0.20
    )
    portfolio.add_position(pos)
    return portfolio


@pytest.fixture
def flat_price_path():
    return np.full(253, 100.0)
