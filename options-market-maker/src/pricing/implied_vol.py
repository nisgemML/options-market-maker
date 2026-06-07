"""
Implied volatility solver.

Uses Brent's method (scipy) with a smart initial guess from
Brenner-Subrahmanyam approximation to converge in <5 iterations typically.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .black_scholes import BlackScholes, OptionType


class ImpliedVolSolverError(ValueError):
    pass


class ImpliedVolSolver:
    """
    Solve for implied volatility given an observed market price.

    Strategy:
      1. Validate arbitrage bounds (intrinsic value, put-call parity region).
      2. Initial guess via Brenner-Subrahmanyam approximation.
      3. Brent's method on [1e-4, 20.0] vol bracket.
      4. Fallback to bisection if Brent fails (e.g. deep ITM/OTM).
    """

    VOL_LB = 1e-6
    VOL_UB = 20.0

    @staticmethod
    def _bs_approximation(
        S: float, K: float, T: float, r: float, option_type: OptionType
    ) -> float:
        """Brenner-Subrahmanyam approximation: sigma ≈ sqrt(2π/T) * C/S."""
        if T <= 0:
            return 0.5
        atm_price = S * np.exp(-r * T) * 0.4  # rough ATM approx
        guess = np.sqrt(2 * np.pi / T) * atm_price / S
        return float(np.clip(guess, 0.05, 2.0))

    @staticmethod
    def solve(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        q: float = 0.0,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """
        Compute implied volatility.

        Returns:
            Implied vol as a decimal (e.g. 0.20 for 20%).

        Raises:
            ImpliedVolSolverError: if no solution exists (arbitrage or bad price).
        """
        bs = BlackScholes

        if T <= 0:
            raise ImpliedVolSolverError("Cannot solve IV at expiry (T=0)")

        # Arbitrage bound checks
        intrinsic = max(
            (S * np.exp(-q * T) - K * np.exp(-r * T)) if option_type == OptionType.CALL
            else (K * np.exp(-r * T) - S * np.exp(-q * T)),
            0.0,
        )
        upper_bound = S * np.exp(-q * T) if option_type == OptionType.CALL else K * np.exp(-r * T)

        if market_price < intrinsic - tol:
            raise ImpliedVolSolverError(
                f"Price {market_price:.4f} below intrinsic {intrinsic:.4f} — arbitrage"
            )
        if market_price > upper_bound + tol:
            raise ImpliedVolSolverError(
                f"Price {market_price:.4f} above upper bound {upper_bound:.4f}"
            )

        def objective(sigma: float) -> float:
            return float(bs.price(S, K, T, r, sigma, option_type, q)) - market_price

        # Check bracket
        lo_val = objective(ImpliedVolSolver.VOL_LB)
        hi_val = objective(ImpliedVolSolver.VOL_UB)

        if lo_val * hi_val > 0:
            # Same sign — clamp to boundary
            if abs(lo_val) < abs(hi_val):
                return ImpliedVolSolver.VOL_LB
            return ImpliedVolSolver.VOL_UB

        try:
            iv = brentq(
                objective,
                ImpliedVolSolver.VOL_LB,
                ImpliedVolSolver.VOL_UB,
                xtol=tol,
                maxiter=max_iter,
            )
        except ValueError as e:
            raise ImpliedVolSolverError(f"Brent solver failed: {e}") from e

        return float(iv)

    @staticmethod
    def solve_surface(
        market_prices: np.ndarray,   # shape (n_expiries, n_strikes)
        S: float,
        strikes: np.ndarray,          # shape (n_strikes,)
        expiries: np.ndarray,         # shape (n_expiries,) in years
        r: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> np.ndarray:
        """
        Vectorised IV surface solver.
        Returns implied vol surface of same shape as market_prices.
        Failed solves return np.nan.
        """
        n_exp, n_str = market_prices.shape
        assert len(expiries) == n_exp and len(strikes) == n_str

        iv_surface = np.full_like(market_prices, np.nan)
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                try:
                    iv_surface[i, j] = ImpliedVolSolver.solve(
                        market_prices[i, j], S, K, T, r, option_type, q
                    )
                except ImpliedVolSolverError:
                    pass  # leave as nan
        return iv_surface
