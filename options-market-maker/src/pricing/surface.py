"""
Volatility surface — SVI (Stochastic Volatility Inspired) parameterisation.

SVI raw parameterisation (Gatheral 2004):
    w(k) = a + b * (rho*(k - m) + sqrt((k-m)^2 + sigma^2))

where k = log(K/F) is log-moneyness and w = sigma_implied^2 * T.

The surface is arbitrage-free if:
  - Calendar spread: w(k, T1) <= w(k, T2) for T1 < T2 (total variance non-decreasing)
  - Butterfly: second derivative of w w.r.t. k >= 0 (convexity)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class SVIParams:
    """SVI raw parameterisation for one expiry slice."""
    a: float      # overall level
    b: float      # angle between left and right asymptotes (>= 0)
    rho: float    # correlation (-1 < rho < 1) — skew
    m: float      # translation of smile
    sigma: float  # ATM curvature (> 0)

    def __post_init__(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"SVI rho must be in (-1,1), got {self.rho}")
        if self.b < 0:
            raise ValueError(f"SVI b must be >= 0, got {self.b}")
        if self.sigma <= 0:
            raise ValueError(f"SVI sigma must be > 0, got {self.sigma}")

    def total_variance(self, log_moneyness: np.ndarray) -> np.ndarray:
        """w(k) = total implied variance = sigma_impl^2 * T."""
        k = np.asarray(log_moneyness)
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def implied_vol(self, log_moneyness: np.ndarray, T: float) -> np.ndarray:
        """Convert total variance to annualised implied vol."""
        w = self.total_variance(log_moneyness)
        return np.sqrt(np.maximum(w, 0.0) / T)

    def is_convex(self, log_moneyness: np.ndarray) -> bool:
        """Butterfly arbitrage check: d²w/dk² >= 0 everywhere."""
        k = np.asarray(log_moneyness)
        disc = (k - self.m)**2 + self.sigma**2
        d2w = self.b * self.sigma**2 / (disc ** 1.5)
        return bool(np.all(d2w >= 0))


class VolSurface:
    """
    Multi-expiry implied vol surface built from SVI slices.

    Usage:
        surface = VolSurface.fit(market_vols, strikes, expiries, S, r)
        vol = surface.interpolate(K=100, T=0.25)
    """

    def __init__(self, slices: dict[float, SVIParams], S: float, r: float) -> None:
        """
        Args:
            slices: {T -> SVIParams} — one calibrated SVI per expiry
            S: spot at calibration time
            r: risk-free rate
        """
        self.slices = dict(sorted(slices.items()))  # sorted by T
        self.S = S
        self.r = r
        self.expiries = np.array(sorted(slices.keys()))

    @staticmethod
    def _log_moneyness(K: np.ndarray | float, F: float) -> np.ndarray:
        return np.log(np.asarray(K) / F)

    def interpolate(self, K: float, T: float) -> float:
        """
        Interpolate the vol surface at (K, T).

        - If T matches a calibrated expiry exactly, return SVI vol.
        - Otherwise: linear interpolation in total variance (calendar-spread safe).
        """
        F = self.S * np.exp(self.r * T)
        k = self._log_moneyness(K, F)

        expiries = self.expiries

        if T <= expiries[0]:
            params = self.slices[expiries[0]]
            return float(params.implied_vol(k, expiries[0]))
        if T >= expiries[-1]:
            params = self.slices[expiries[-1]]
            return float(params.implied_vol(k, expiries[-1]))

        # Find bracketing expiries
        idx = np.searchsorted(expiries, T)
        T1, T2 = expiries[idx - 1], expiries[idx]
        w1 = self.slices[T1].total_variance(k) 
        w2 = self.slices[T2].total_variance(k)

        # Linear interpolation in total variance
        alpha = (T - T1) / (T2 - T1)
        w = (1 - alpha) * w1 + alpha * w2
        return float(np.sqrt(max(w, 0.0) / T))

    @staticmethod
    def fit_slice(
        market_vols: np.ndarray,
        strikes: np.ndarray,
        T: float,
        S: float,
        r: float,
        q: float = 0.0,
    ) -> SVIParams:
        """
        Calibrate SVI params to market implied vols for one expiry.

        Minimises sum of squared vol errors with parameter constraints.
        """
        F = S * np.exp((r - q) * T)
        log_mon = np.log(strikes / F)
        w_market = market_vols**2 * T  # target: total variance

        def objective(params: np.ndarray) -> float:
            a, b, rho, m, sigma = params
            if b < 0 or sigma <= 0 or abs(rho) >= 1:
                return 1e9
            w = a + b * (rho * (log_mon - m) + np.sqrt((log_mon - m)**2 + sigma**2))
            if np.any(w < 0):
                return 1e9
            return float(np.sum((w - w_market)**2))

        atm_var = float(np.mean(market_vols)**2 * T)
        x0 = np.array([atm_var * 0.5, 0.1, -0.3, 0.0, 0.1])

        bounds = [
            (-1.0, 1.0),    # a
            (1e-4, 2.0),    # b
            (-0.999, 0.999),# rho
            (-1.0, 1.0),    # m
            (1e-4, 1.0),    # sigma
        ]

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-12})

        a, b, rho, m, sigma = result.x
        return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

    @classmethod
    def fit(
        cls,
        market_vols: np.ndarray,   # shape (n_expiries, n_strikes)
        strikes: np.ndarray,
        expiries: np.ndarray,
        S: float,
        r: float,
        q: float = 0.0,
    ) -> "VolSurface":
        """Calibrate full surface — one SVI slice per expiry."""
        slices = {}
        for i, T in enumerate(expiries):
            slices[T] = cls.fit_slice(market_vols[i], strikes, T, S, r, q)
        return cls(slices=slices, S=S, r=r)

    def calendar_spread_check(self, log_moneyness: float = 0.0) -> bool:
        """
        Verify total variance is non-decreasing across expiries (no calendar arb).
        Checks at a given log-moneyness level (default: ATM).
        """
        k = np.array([log_moneyness])
        prev_w = -np.inf
        for T, params in self.slices.items():
            w = float(params.total_variance(k))
            if w < prev_w - 1e-8:
                return False
            prev_w = w
        return True
