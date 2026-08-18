# Benchmark Results — Options Market Maker

All results produced on this machine and committed. Reproducible:

```bash
pip install numpy pandas scipy matplotlib
python scripts/run_backtest.py    # backtest
pytest test/ -q                   # 89 Python tests
cd ocaml && ocamlfind ocamlopt -package qcheck -linkpkg \
    black_scholes.ml black_scholes_test.ml -o test_bs && ./test_bs  # 12 QCheck
```

---

## Test results

```
89 Python tests passed  (27.38s)
12 OCaml QCheck properties passed

Python tests cover:
  - Black-Scholes: put-call parity, delta bounds [0,1], gamma ≥ 0, vega ≥ 0
  - IV solver: round-trip price(IV(price)) = price
  - Heston: Gil-Pelaez Fourier inversion, BS limit (ξ→0)
  - SSVI: butterfly arbitrage-free, calendar arbitrage-free (∂w/∂θ ≥ 0)
  - Vanna-volga: weight decomposition, ATM/RR/BF pillar consistency
  - Hedging: delta-band P&L attribution, hedge cost accounting

OCaml QCheck properties (12):
  put-call parity, call delta ∈ [0,1], put delta ∈ [-1,0], gamma ≥ 0,
  vega ≥ 0, IV round-trip, call ≥ intrinsic, prices ≥ 0,
  call increasing in S, prices increasing in vol, OTM call decays with time,
  homogeneity C(λS,λK) = λC(S,K)
```

---

## Backtest results

**Setup:** 252 trading days, σ=20%, delta-band hedging (rehedge when |Δ| > band).
One at-the-money call option position, daily rebalancing of delta hedge.

```
Steps            : 253
Total PnL        : +489.72
Sharpe           : 2.26  (annualised)
Max Drawdown     : -25.93%
Total Hedge Cost :   2.80  (transaction costs)
Total Fills      : 780   (options fills)
Toxic Fill Rate  :  51.9% (fills from informed flow)
Hedge Trades     : 200   (delta rehedges triggered)

Risk metrics:
  Mean daily PnL   :   +1.94
  Std daily PnL    :   13.64
  VaR 95%          :   17.21
  CVaR 95%         :   28.29
  Win rate         :   63.5%
```

**Key observations:**

**Sharpe 2.26** — strong for a market making strategy. The delta-band
hedging (not daily fixed) reduces unnecessary hedge trades (200 vs 252)
while maintaining gamma exposure during favourable vol moves.

**Toxic fill rate 51.9%** — just over half of fills come from informed
flow. This is high; in production, the adverse selection model (Glosten-Milgrom
α parameterisation) would widen spreads when the informed fraction exceeds
the break-even threshold (~0.30 for this parameter regime).

**Hedge cost 2.80** — very low relative to total PnL of 489.72 (0.57%).
Delta-band hedging avoids the transaction cost drag of daily rebalancing
while keeping delta exposure bounded.

---

## Pricing model accuracy

### Black-Scholes IV round-trip error

```
|price(IV(market_price)) - market_price| < 1e-4 for all 2000 test cases
IV solver: Brent's method, tolerance 1e-6, max 100 iterations
Typical convergence: 8-12 iterations
```

### Heston vs Black-Scholes at zero vol-of-vol

```
When ξ→0 (vol-of-vol → 0), Heston reduces to Black-Scholes.
Test uses ξ=0.01 (not ξ=0 — degenerate in the square-root discriminant).
Max error at ξ=0.01: |Heston - BS| < 0.01 across all strikes/maturities.
```

### SSVI surface properties

```
Calendar arbitrage-free: ∂w/∂θ ≥ 0 verified for all (k,θ) combinations.
Butterfly arbitrage-free: g(k,θ) ≥ 0 verified (Gatheral-Jacquier condition).
Joint calibration: fitted across all expiries simultaneously — per-slice
calibration can violate calendar arbitrage between adjacent expiries.
```
