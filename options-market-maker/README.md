# options-market-maker

**Options pricing, hedging, and market-making engine — BS pricing → Greeks → IV solver → vol surface → inventory-skewed quoting → delta hedging → full backtest.**

[![CI](https://github.com/nisgemML/options-market-maker/actions/workflows/ci.yml/badge.svg)](https://github.com/nisgemML/options-market-maker/actions)

---

## Architecture

```
src/
  pricing/
    black_scholes.py     – Vectorised BS: price, delta, gamma, vega, theta, rho, vanna, volga
    greeks.py            – Greeks container; portfolio_greeks(); finite-diff cross-check
    implied_vol.py       – Brent's method IV solver; Brenner-Subrahmanyam initial guess
    surface.py           – SVI vol surface; calendar-spread and butterfly arb checks
  hedging/
    portfolio.py         – OptionPosition, HedgePortfolio; pure functional state threading
    delta_hedger.py      – Three triggers: band / periodic / gamma-adjusted band
  quoting/
    skew.py              – Inventory skew: mid shift + gamma spread widening
    market_maker.py      – Quote engine: fair value → skew → risk limits → bid/ask
  backtest/
    scenario.py          – GBM and Heston-like stochastic vol path generators
    engine.py            – Event-driven backtest loop; Bernoulli fill model
  risk/
    limits.py            – Delta / gamma / vega / loss limits with RiskBreachError
    metrics.py           – Sharpe, max drawdown, VaR, CVaR, win rate
test/
  test_black_scholes.py  – 10 properties incl. put-call parity, bounds, finite-diff Greeks
  test_implied_vol.py    – IV round-trip (Hypothesis), known values, error cases
  test_hedger.py         – Band reduces delta, cost non-negative, periodic schedule
  test_backtest.py       – Smoke tests: shape, Sharpe finite, custom path
scripts/
  run_backtest.py        – CLI: --steps, --vol, --hedge, --plot
```

---

## Design decisions

**Why pure functional state threading throughout?**
Every function is `(params, state) -> new_state`. `HedgePortfolio.roll_time()` returns a new portfolio; `DeltaHedger.step()` returns `(new_portfolio, record)`. This makes the backtest engine trivially parallelisable across paths and eliminates a class of mutation bugs.

**Why SVI for the vol surface?**
SVI (Gatheral 2004) is the industry standard for arbitrage-free vol surface parameterisation. The raw form has an analytical gradient, calibrates fast with L-BFGS-B, and natively enforces no-arbitrage bounds (non-negative total variance, convex in log-moneyness). Calendar-spread safety is enforced via linear interpolation in total variance.

**Why Brent's method for IV?**
Brent's method is superlinearly convergent and bracket-safe — it never diverges. Newton-Raphson is faster per iteration but can oscillate near zero vega (deep ITM/OTM). The Brenner-Subrahmanyam guess gets us within 5 iterations for most ATM/near-money options.

**Why three hedge triggers?**
- **Band**: lowest transaction cost for slow-moving books
- **Periodic**: simple and auditable for regulatory purposes
- **Gamma-adjusted**: Zakamouline (2009) optimal band — proportional to `gamma * sqrt(dt) * S`; widens automatically when gamma is small (less urgent) and tightens near expiry or high vol

**Inventory skew formula**
```
skew(delta) = -alpha * net_delta
bid = fair + skew - spread/2
ask = fair + skew + spread/2
```
A long delta position shifts both quotes down → MM more eager to sell, less eager to buy. Pure feedback control, no forecasting required.

---

## Properties tested

| Test | What it verifies |
|---|---|
| Put-call parity | C - P = S - K·e^{-rT} for all valid inputs (Hypothesis, 500 examples) |
| Call/put bounds | Price always in [intrinsic, S] / [intrinsic, Ke^{-rT}] |
| Price monotone in vol | Vega > 0 everywhere |
| Delta in [0,1] / [-1,0] | Delta bounds for calls / puts |
| Gamma ≥ 0 | Gamma always non-negative |
| Vega ≥ 0 | Vega always non-negative |
| Finite-diff Greeks | Analytical = numerical to 4 decimal places (4 parameter sets × 2 types) |
| IV round-trip | `solve(price(sigma)) == sigma` (Hypothesis, 300 examples) |
| Band hedge reduces delta | After hedge, \|delta\| strictly smaller |
| Hedge cost non-negative | Transaction costs always ≥ 0 |
| Periodic trigger fires | Exactly at multiples of N |
| Backtest shape | Output rows = n_steps + 1 |

---

## Build and run

```bash
git clone https://github.com/nisgemML/options-market-maker
cd options-market-maker

pip install -e ".[dev]"

# Run all tests
pytest test/ -v

# Run backtest (252 steps, band hedge)
python scripts/run_backtest.py --steps 252 --vol 0.20 --hedge band

# With P&L plot
python scripts/run_backtest.py --steps 252 --vol 0.30 --hedge gamma --plot --out result.png
```

---

## Key formulas

**Black-Scholes:**
```
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
C  = S·N(d1) - K·e^{-rT}·N(d2)
```

**SVI total variance:**
```
w(k) = a + b·[ρ(k-m) + √((k-m)² + σ²)]
```

**Gamma-adjusted hedge band (Zakamouline):**
```
band = γ_mult · |Γ| · √(dt) · S
```

**Inventory skew:**
```
mid_shift = -α · net_delta
spread    = base_spread + γ_mult · |Γ| · S
```

---

## Related repos

- [`ocaml-trading-primitives`](https://github.com/nisgemML/ocaml-trading-primitives) — Functional LOB + QCheck property tests in OCaml
- [`Low-Latency-Trading-Engine`](https://github.com/nisgemML/Low-Latency-Trading-Engine) — C++20 exchange engine, 6M+ msgs/sec
- [`avellaneda-stoikov`](https://github.com/nisgemML/avellaneda-stoikov) — Optimal MM with adverse selection

---

## Author

Nishant Gemawat · [github.com/nisgemML](https://github.com/nisgemML)  
12+ years financial services engineering · Morgan Stanley · State Street Alpha Frontier
