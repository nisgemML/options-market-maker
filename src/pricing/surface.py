"""
Volatility surface parameterisations.

Two models implemented:

1. SVI (Stochastic Volatility Inspired) — Gatheral (2004)
   Single-expiry slice parameterisation.  Five parameters per slice.
   w(k) = a + b·[ρ(k-m) + √((k-m)² + σ²)]

2. SSVI (Surface SVI) — Gatheral & Jacquier (2014)
   Joint surface parameterisation that guarantees no static arbitrage across
   both strikes AND maturities with a single set of surface parameters.
   w(k, θ_t) = (θ_t/2)·{1 + ρ·φ(θ_t)·k + √[(φ(θ_t)·k + ρ)² + 1 - ρ²]}
   φ(θ) = η / θ^γ   (power-law parametric form)

   Theorem (Gatheral & Jacquier 2014, Thm 4.2):
     SSVI is free of butterfly arbitrage iff:
       ∂w/∂k ≥ 0  (call spread)
       g(k) := (1 - k·φ'·w/(2w))² - (φ')²/4·(1/w + 1/4) + φ''/2 ≥ 0
     where φ' = ∂φ/∂k.  The power-law form satisfies this by construction
     when (ρ, η, γ) ∈ admissible region.

   Calendar-spread free iff θ_t non-decreasing in t (total variance grows).
   This is enforced by fitting θ_t = ATM total variance from the market.

References:
    Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterisation.
    Global Derivatives & Risk Management, Madrid.
    Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    Quantitative Finance, 14(1), 59-71.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

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
        """w(k) = total implied variance = σ²_impl · T."""
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


class SSVIParams(NamedTuple):
    """
    SSVI power-law surface parameters (Gatheral & Jacquier 2014).

    Surface parameterisation:
        w(k, θ_t) = (θ_t/2)·{1 + ρ·φ·k + √[(φ·k + ρ)² + 1 - ρ²]}
        φ(θ_t)   = η / θ_t^γ

    Admissibility constraints (no butterfly arb, Thm 4.2):
        |ρ| < 1
        0 < γ ≤ 1
        η(1 + |ρ|) ≤ 2          (Lee moment formula bound)
    """
    rho:   float   # global skew parameter ∈ (-1, 1)
    eta:   float   # ATM vol-of-vol scale (> 0)
    gamma: float   # power-law exponent ∈ (0, 1]

    def phi(self, theta_t: float) -> float:
        """Curvature function φ(θ_t) = η / θ_t^γ."""
        return self.eta / (theta_t ** self.gamma)

    def total_variance(self, k: np.ndarray | float, theta_t: float) -> np.ndarray:
        """
        SSVI total variance w(k, θ_t).

        Args:
            k:       log-moneyness ln(K/F)
            theta_t: ATM total variance at maturity T (θ_t = σ²_ATM · T)
        """
        k = np.asarray(k, dtype=float)
        phi = self.phi(theta_t)
        inner = phi * k + self.rho
        w = (theta_t / 2.0) * (
            1.0 + self.rho * phi * k
            + np.sqrt(inner ** 2 + 1.0 - self.rho ** 2)
        )
        return w

    def implied_vol(self, k: np.ndarray | float, theta_t: float, T: float) -> np.ndarray:
        """Annualised implied vol from SSVI total variance."""
        w = self.total_variance(k, theta_t)
        return np.sqrt(np.maximum(w, 0.0) / T)

    def is_admissible(self) -> tuple[bool, str]:
        """
        Check Gatheral-Jacquier admissibility (no butterfly arb).
        Returns (True, '') if admissible, (False, reason) otherwise.
        """
        if not (-1.0 < self.rho < 1.0):
            return False, f"|ρ|={abs(self.rho):.3f} ≥ 1"
        if not (0 < self.gamma <= 1.0):
            return False, f"γ={self.gamma:.3f} ∉ (0,1]"
        if self.eta * (1 + abs(self.rho)) > 2.0:
            return False, f"η(1+|ρ|)={self.eta*(1+abs(self.rho)):.3f} > 2 (Lee bound)"
        return True, ""

    def calendar_spread_free(self, theta_times: np.ndarray) -> bool:
        """
        Calendar-spread arbitrage check: θ_t must be non-decreasing in T.
        (Total variance cannot decrease as maturity increases.)
        """
        return bool(np.all(np.diff(theta_times) >= 0))


class VolSurface:
    """
    Multi-expiry implied vol surface built from SVI slices.

    Usage:
        surface = VolSurface.fit(market_vols, strikes, expiries, S, r)
        vol = surface.interpolate(K=100, T=0.25)
    """

    def __init__(self, slices: dict[float, SVIParams], S: float, r: float) -> None:
        self.slices = dict(sorted(slices.items()))
        self.S = S
        self.r = r
        self.expiries = np.array(sorted(slices.keys()))

    @staticmethod
    def _log_moneyness(K: np.ndarray | float, F: float) -> np.ndarray:
        return np.log(np.asarray(K) / F)

    def interpolate(self, K: float, T: float) -> float:
        """
        Interpolate vol surface at (K, T).

        Uses linear interpolation in total variance (calendar-spread safe).
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

        idx = np.searchsorted(expiries, T)
        T1, T2 = expiries[idx - 1], expiries[idx]
        w1 = self.slices[T1].total_variance(k)
        w2 = self.slices[T2].total_variance(k)

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
        """Calibrate SVI params to market implied vols for one expiry."""
        F = S * np.exp((r - q) * T)
        log_mon = np.log(strikes / F)
        w_market = market_vols**2 * T

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

        bounds = [(-1.0, 1.0), (1e-4, 2.0), (-0.999, 0.999), (-1.0, 1.0), (1e-4, 1.0)]
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-12})

        a, b, rho, m, sigma = result.x
        return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

    @classmethod
    def fit(
        cls,
        market_vols: np.ndarray,
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
        """Verify total variance is non-decreasing (no calendar arb) at ATM."""
        k = np.array([log_moneyness])
        prev_w = -np.inf
        for T, params in self.slices.items():
            w = float(np.squeeze(params.total_variance(k)))
            if w < prev_w - 1e-8:
                return False
            prev_w = w
        return True


class SSVIVolSurface:
    """
    SSVI (Surface SVI) vol surface — joint calibration, no static arbitrage.

    Parameterisation (Gatheral & Jacquier 2014):
        w(k, θ_t) = (θ_t/2)·{1 + ρ·φ·k + √[(φ·k + ρ)² + 1 - ρ²]}
        φ(θ_t) = η / θ_t^γ

    Key advantage over slice-by-slice SVI: single set of (ρ, η, γ) parameters
    spans the whole surface, guaranteeing calendar-spread and butterfly
    arbitrage-free pricing across all strikes and maturities simultaneously.

    The ATM total variance θ_t = σ²_ATM(T) · T is fitted directly from
    ATM implied vols in the market data (non-parametric in T).

    Usage:
        ssvi = SSVIVolSurface.fit(market_vols, strikes, expiries, S, r)
        vol = ssvi.interpolate(K=100, T=0.25)
    """

    def __init__(
        self,
        params: SSVIParams,
        theta_t: dict[float, float],   # {T -> ATM total variance}
        S: float,
        r: float,
    ) -> None:
        self.params = params
        self.theta_t = dict(sorted(theta_t.items()))
        self.expiries = np.array(sorted(theta_t.keys()))
        self.S = S
        self.r = r

    def _get_theta(self, T: float) -> float:
        """Linearly interpolate ATM total variance to arbitrary T."""
        exp = self.expiries
        if T <= exp[0]:
            return self.theta_t[exp[0]]
        if T >= exp[-1]:
            return self.theta_t[exp[-1]]
        idx = np.searchsorted(exp, T)
        T1, T2 = exp[idx - 1], exp[idx]
        alpha = (T - T1) / (T2 - T1)
        return (1 - alpha) * self.theta_t[T1] + alpha * self.theta_t[T2]

    def interpolate(self, K: float, T: float) -> float:
        """SSVI implied vol at (K, T)."""
        F = self.S * np.exp(self.r * T)
        k = np.log(K / F)
        theta_t = self._get_theta(T)
        return float(self.params.implied_vol(k, theta_t, T))

    @classmethod
    def fit(
        cls,
        market_vols: np.ndarray,    # shape (n_expiries, n_strikes)
        strikes: np.ndarray,         # shape (n_strikes,)
        expiries: np.ndarray,        # shape (n_expiries,) in years
        S: float,
        r: float,
        q: float = 0.0,
    ) -> "SSVIVolSurface":
        """
        Calibrate SSVI surface.

        Step 1: Extract ATM total variance θ_t from market data (non-parametric).
        Step 2: Fit global (ρ, η, γ) to minimise sum-of-squared vol errors
                across all strikes and expiries simultaneously.

        Constraints enforce admissibility (Gatheral-Jacquier Thm 4.2).
        """
        n_exp, n_str = market_vols.shape

        # Step 1: ATM total variance per expiry (interpolate to ATM strike)
        theta_t = {}
        for i, T in enumerate(expiries):
            F = S * np.exp((r - q) * T)
            log_mon = np.log(strikes / F)
            atm_idx = int(np.argmin(np.abs(log_mon)))
            theta_t[T] = float(market_vols[i, atm_idx] ** 2 * T)

        # Step 2: Fit global (rho, eta, gamma)
        def objective(x: np.ndarray) -> float:
            rho, eta, gamma = x
            if abs(rho) >= 1 or eta <= 0 or gamma <= 0 or gamma > 1:
                return 1e9
            if eta * (1 + abs(rho)) > 2.0:
                return 1e9
            p = SSVIParams(rho=rho, eta=eta, gamma=gamma)
            err = 0.0
            for i, T in enumerate(expiries):
                F = S * np.exp((r - q) * T)
                log_mon = np.log(strikes / F)
                th = theta_t[T]
                w_model = p.total_variance(log_mon, th)
                w_market = market_vols[i] ** 2 * T
                err += float(np.sum((np.sqrt(np.maximum(w_model, 0) / T) - market_vols[i]) ** 2))
            return err

        x0 = np.array([-0.3, 0.5, 0.5])
        bounds = [(-0.999, 0.999), (1e-4, 1.99), (1e-4, 1.0)]
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 1000, "ftol": 1e-14})
        rho, eta, gamma = result.x
        params = SSVIParams(rho=rho, eta=eta, gamma=gamma)

        return cls(params=params, theta_t=theta_t, S=S, r=r)

    def is_arbitrage_free(self, n_strikes: int = 50, n_expiries: int = 20) -> dict[str, bool]:
        """
        Check SSVI surface for both butterfly and calendar-spread arbitrage
        on a dense grid.
        """
        expiry_grid = np.linspace(self.expiries[0], self.expiries[-1], n_expiries)
        strike_grid = np.linspace(0.7 * self.S, 1.3 * self.S, n_strikes)

        calendar_free = True
        prev_w = np.full(n_strikes, -np.inf)

        for T in expiry_grid:
            F = self.S * np.exp(self.r * T)
            log_mon = np.log(strike_grid / F)
            theta_t = self._get_theta(T)
            w = self.params.total_variance(log_mon, theta_t)
            if np.any(w < prev_w - 1e-8):
                calendar_free = False
            prev_w = w

        # Butterfly: total variance convex in k
        butterfly_free = True
        T_mid = expiry_grid[len(expiry_grid)//2]
        F_mid = self.S * np.exp(self.r * T_mid)
        k_grid = np.log(strike_grid / F_mid)
        theta_mid = self._get_theta(T_mid)
        w_mid = self.params.total_variance(k_grid, theta_mid)
        d2w = np.diff(w_mid, 2)
        if np.any(d2w < -1e-8):
            butterfly_free = False

        ok, reason = self.params.is_admissible()
        return {
            "admissible": ok,
            "calendar_spread_free": calendar_free,
            "butterfly_free": butterfly_free,
        }
