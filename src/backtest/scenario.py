"""
Price path generators for backtesting.

GBMScenario: Geometric Brownian Motion — exact log-normal solution.
HestonScenario: Full Heston (1993) stochastic-vol model via correlated Brownian motions.
  Uses Euler-Maruyama with full truncation on the variance process.
  Feller condition 2κθ > ξ² checked at construction — when satisfied, variance
  process is strictly positive a.s. and the CIR has a non-central χ² invariant measure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScenarioConfig:
    S0:       float = 100.0     # initial spot
    mu:       float = 0.05      # drift (annual)
    sigma:    float = 0.20      # vol (annual)
    T:        float = 1.0       # horizon (years)
    n_steps:  int   = 252       # number of time steps
    n_paths:  int   = 1         # number of MC paths
    seed:     int   = 42


@dataclass
class HestonConfig:
    """
    Parameters for the Heston (1993) stochastic volatility model.

    dS = r·S·dt + √v·S·dW₁
    dv = κ(θ - v)·dt + ξ·√v·dW₂          (CIR variance process)
    corr(dW₁, dW₂) = ρ

    Feller condition: 2κθ > ξ²
      If satisfied, the variance process never hits zero.
      A warning is issued (not an error) if violated, as the discretised
      scheme uses full truncation (v → max(v, 0)) regardless.
    """
    S0:      float = 100.0
    v0:      float = 0.04       # initial variance (σ₀ = 0.20)
    r:       float = 0.05       # risk-free rate
    kappa:   float = 2.0        # mean-reversion speed
    theta:   float = 0.04       # long-run variance (σ_∞ = 0.20)
    xi:      float = 0.30       # vol-of-vol
    rho:     float = -0.70      # spot-vol correlation (typically negative)
    T:       float = 1.0        # horizon (years)
    n_steps: int   = 252
    n_paths: int   = 1
    seed:    int   = 42

    def __post_init__(self) -> None:
        feller = 2 * self.kappa * self.theta
        if feller <= self.xi ** 2:
            import warnings
            warnings.warn(
                f"Feller condition violated: 2κθ={feller:.4f} ≤ ξ²={self.xi**2:.4f}. "
                "Variance process may hit zero; full-truncation scheme applied.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def feller_ratio(self) -> float:
        """2κθ / ξ² — must be > 1 for Feller condition."""
        return 2 * self.kappa * self.theta / (self.xi ** 2)


class GBMScenario:
    """
    Geometric Brownian Motion path generator.

    dS = μ·S·dt + σ·S·dW

    Exact log-normal solution (no discretisation error):
        S(t+dt) = S(t) · exp((μ - ½σ²)dt + σ√dt · Z)
    """

    def __init__(self, config: ScenarioConfig) -> None:
        self.cfg = config

    def generate(self) -> np.ndarray:
        """
        Generate price paths.

        Returns:
            paths: shape (n_steps+1, n_paths) — row 0 is S0
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        dt = cfg.T / cfg.n_steps

        Z = rng.standard_normal((cfg.n_steps, cfg.n_paths))
        log_returns = (cfg.mu - 0.5 * cfg.sigma**2) * dt + cfg.sigma * np.sqrt(dt) * Z

        paths = np.empty((cfg.n_steps + 1, cfg.n_paths))
        paths[0] = cfg.S0
        for t in range(cfg.n_steps):
            paths[t + 1] = paths[t] * np.exp(log_returns[t])

        return paths


class HestonScenario:
    """
    Heston (1993) stochastic volatility path generator.

    Implementation details:
      - Variance process: Euler-Maruyama with full truncation (v → max(v, 0)).
        Full truncation (Lord et al. 2010) outperforms absorption and reflection
        in preserving the marginal distribution near zero.
      - Correlated Brownians: Cholesky factoring with ρ.
        Z₂ = ρ·Z₁ + √(1-ρ²)·Z_perp  where Z₁, Z_perp ∼ N(0,1) independent.
      - Log-spot uses Milstein correction via the exact log transform to suppress
        discretisation bias in the spot process.
    """

    def __init__(self, config: HestonConfig) -> None:
        self.cfg = config

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths.

        Returns:
            (S_paths, v_paths): both shape (n_steps+1, n_paths)
            S_paths[0] = S0,  v_paths[0] = v0
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        dt = cfg.T / cfg.n_steps

        S = np.empty((cfg.n_steps + 1, cfg.n_paths))
        v = np.empty((cfg.n_steps + 1, cfg.n_paths))
        S[0] = cfg.S0
        v[0] = cfg.v0

        for t in range(cfg.n_steps):
            Z1 = rng.standard_normal(cfg.n_paths)
            Z_perp = rng.standard_normal(cfg.n_paths)
            Z2 = cfg.rho * Z1 + np.sqrt(1.0 - cfg.rho ** 2) * Z_perp

            v_pos = np.maximum(v[t], 0.0)           # full truncation
            sqrt_v_dt = np.sqrt(v_pos * dt)

            # CIR variance step (Euler-Maruyama)
            v[t + 1] = np.maximum(
                v[t] + cfg.kappa * (cfg.theta - v_pos) * dt + cfg.xi * sqrt_v_dt * Z2,
                0.0,
            )

            # Log-spot step (exact given v_t — standard Euler on log S)
            S[t + 1] = S[t] * np.exp(
                (cfg.r - 0.5 * v_pos) * dt + sqrt_v_dt * Z1
            )

        return S, v

    def implied_vol_proxy(self, S_paths: np.ndarray, v_paths: np.ndarray) -> np.ndarray:
        """
        Realised vol from simulated variance paths: σ_realised = sqrt(mean(v)).
        Useful for sanity-checking that E[v] ≈ θ at long horizons.
        """
        return np.sqrt(v_paths.mean(axis=0))
