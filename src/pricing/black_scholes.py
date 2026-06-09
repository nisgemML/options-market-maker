"""
Black-Scholes pricing engine.

All functions are pure: (params) -> float.
Vectorised over numpy arrays for backtest performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class BSParams:
    """Immutable Black-Scholes parameter set."""
    S: float          # spot price
    K: float          # strike
    T: float          # time to expiry (years)
    r: float          # risk-free rate (continuously compounded)
    sigma: float      # implied volatility
    q: float = 0.0    # continuous dividend yield


class BlackScholes:
    """
    Black-Scholes-Merton pricing and Greeks.

    Design: static methods only — no state, fully composable.
    Vectorised signatures accept np.ndarray for all float params.
    """

    @staticmethod
    def _d1_d2(
        S: np.ndarray | float,
        K: np.ndarray | float,
        T: np.ndarray | float,
        r: float,
        sigma: np.ndarray | float,
        q: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute d1 and d2. T=0 returns ±inf safely via np.where."""
        T = np.asarray(T, dtype=float)
        sqrt_T = np.where(T > 0, np.sqrt(np.maximum(T, 1e-10)), 1e-10)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    @staticmethod
    def price(
        S: np.ndarray | float,
        K: np.ndarray | float,
        T: np.ndarray | float,
        r: float,
        sigma: np.ndarray | float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> np.ndarray | float:
        """Black-Scholes option price."""
        S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
        d1, d2 = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        df = np.exp(-r * T)
        dq = np.exp(-q * T)

        if option_type == OptionType.CALL:
            price = dq * S * norm.cdf(d1) - df * K * norm.cdf(d2)
        else:
            price = df * K * norm.cdf(-d2) - dq * S * norm.cdf(-d1)

        # At expiry: intrinsic value
        T_arr = np.asarray(T)
        if T_arr.ndim == 0 and float(T_arr) == 0.0:
            if option_type == OptionType.CALL:
                return float(np.maximum(S - K, 0.0))
            else:
                return float(np.maximum(K - S, 0.0))

        return price

    @staticmethod
    def delta(
        S: np.ndarray | float,
        K: np.ndarray | float,
        T: np.ndarray | float,
        r: float,
        sigma: np.ndarray | float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> np.ndarray | float:
        S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
        d1, _ = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        dq = np.exp(-q * T)
        if option_type == OptionType.CALL:
            return dq * norm.cdf(d1)
        return dq * (norm.cdf(d1) - 1.0)

    @staticmethod
    def gamma(
        S: np.ndarray | float,
        K: np.ndarray | float,
        T: np.ndarray | float,
        r: float,
        sigma: np.ndarray | float,
        q: float = 0.0,
    ) -> np.ndarray | float:
        """Gamma is identical for calls and puts."""
        S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
        d1, _ = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        dq = np.exp(-q * T)
        sqrt_T = np.sqrt(np.maximum(T, 1e-10))
        return dq * norm.pdf(d1) / (S * sigma * sqrt_T)

    @staticmethod
    def vega(
        S: np.ndarray | float,
        K: np.ndarray | float,
        T: np.ndarray | float,
        r: float,
        sigma: np.ndarray | float,
        q: float = 0.0,
    ) -> np.ndarray | float:
        """Vega (per 1-point move in sigma, not 1%)."""
        S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
        d1, _ = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        dq = np.exp(-q * T)
        sqrt_T = np.sqrt(np.maximum(T, 1e-10))
        return dq * S * norm.pdf(d1) * sqrt_T

    @staticmethod
    def theta(
        S: np.ndarray | float,
        K: np.ndarray | float,
        T: np.ndarray | float,
        r: float,
        sigma: np.ndarray | float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> np.ndarray | float:
        """Theta (per calendar day, not per year)."""
        S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
        d1, d2 = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        dq = np.exp(-q * T)
        df = np.exp(-r * T)
        sqrt_T = np.sqrt(np.maximum(T, 1e-10))

        common = -(dq * S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
        if option_type == OptionType.CALL:
            theta_annual = common - r * K * df * norm.cdf(d2) + q * S * dq * norm.cdf(d1)
        else:
            theta_annual = common + r * K * df * norm.cdf(-d2) - q * S * dq * norm.cdf(-d1)
        return theta_annual / 365.0

    @staticmethod
    def rho(
        S: np.ndarray | float,
        K: np.ndarray | float,
        T: np.ndarray | float,
        r: float,
        sigma: np.ndarray | float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> np.ndarray | float:
        """Rho (per 1-point move in r, not 1bp)."""
        S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
        _, d2 = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        df = np.exp(-r * T)
        if option_type == OptionType.CALL:
            return K * T * df * norm.cdf(d2)
        return -K * T * df * norm.cdf(-d2)

    @staticmethod
    def vanna(
        S: np.ndarray | float,
        K: np.ndarray | float,
        T: np.ndarray | float,
        r: float,
        sigma: np.ndarray | float,
        q: float = 0.0,
    ) -> np.ndarray | float:
        """Vanna = d(delta)/d(sigma) = d(vega)/dS."""
        S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
        d1, d2 = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        dq = np.exp(-q * T)
        return -dq * norm.pdf(d1) * d2 / sigma

    @staticmethod
    def volga(
        S: np.ndarray | float,
        K: np.ndarray | float,
        T: np.ndarray | float,
        r: float,
        sigma: np.ndarray | float,
        q: float = 0.0,
    ) -> np.ndarray | float:
        """Volga/Vomma = d(vega)/d(sigma)."""
        S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
        d1, d2 = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        vega = BlackScholes.vega(S, K, T, r, sigma, q)
        return vega * d1 * d2 / sigma

    @staticmethod
    def put_call_parity_check(
        call_price: float,
        put_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        tol: float = 1e-6,
    ) -> bool:
        """Verify C - P = S*exp(-qT) - K*exp(-rT)."""
        lhs = call_price - put_price
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        return bool(abs(lhs - rhs) < tol)
