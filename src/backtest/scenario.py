"""
Price path generators for backtesting.

GBMScenario: Geometric Brownian Motion with optional vol-of-vol (Heston-like jumps).
All generators are pure functions: (seed, params) -> paths.
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


class GBMScenario:
    """
    Geometric Brownian Motion path generator.

    dS = mu*S*dt + sigma*S*dW

    Exact log-normal solution (no discretisation error):
        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
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

    def generate_with_vol(
        self,
        vol_of_vol: float = 0.3,
        mean_reversion: float = 2.0,
        long_run_vol: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Heston-like stochastic vol using Euler-Maruyama on variance process.

        dv = kappa*(theta - v)*dt + vol_of_vol*sqrt(v)*dW_v
        corr(dW_S, dW_v) = rho (set to -0.7 by default)

        Returns:
            (paths, vol_paths): both shape (n_steps+1, n_paths)
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed + 1)
        dt = cfg.T / cfg.n_steps
        theta = long_run_vol or cfg.sigma
        kappa = mean_reversion
        rho = -0.7

        Z1 = rng.standard_normal((cfg.n_steps, cfg.n_paths))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal((cfg.n_steps, cfg.n_paths))

        paths = np.empty((cfg.n_steps + 1, cfg.n_paths))
        vols  = np.empty((cfg.n_steps + 1, cfg.n_paths))
        paths[0] = cfg.S0
        vols[0]  = cfg.sigma

        for t in range(cfg.n_steps):
            v = np.maximum(vols[t], 0.0)
            sqrt_v = np.sqrt(v)
            vols[t + 1] = np.maximum(
                v + kappa * (theta - v) * dt + vol_of_vol * sqrt_v * np.sqrt(dt) * Z2[t],
                1e-6,
            )
            paths[t + 1] = paths[t] * np.exp(
                (cfg.mu - 0.5 * v) * dt + sqrt_v * np.sqrt(dt) * Z1[t]
            )

        return paths, vols
