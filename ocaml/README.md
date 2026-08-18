# ocaml/black_scholes.ml

Black-Scholes pricer in OCaml — mirrors `src/pricing/black_scholes.py`.

Same mathematical model, different type-system guarantees:

- **Exhaustive pattern matching** on `option_type` — impossible to forget a branch.
- **`result` return type** on `implied_vol` — the caller must handle solver failure at compile time, no runtime exceptions.
- **QCheck property tests** in `black_scholes_test.ml` — same properties as the Python Hypothesis suite (put-call parity, delta bounds, gamma non-negative).

This module also lives in [`ocaml-trading-primitives`](https://github.com/nisgemML/ocaml-trading-primitives), where it sits alongside the limit order book and matching engine implementations.

## Build

```bash
# Requires ocamlfind and qcheck
ocamlfind ocamlopt -package qcheck -linkpkg \
  black_scholes.ml black_scholes_test.ml -o test_bs
./test_bs
```

## Key difference from Python

```ocaml
(* OCaml: compiler enforces both cases are handled *)
match opt with
| Call -> dq *. s *. norm_cdf d1 -. df *. k *. norm_cdf d2
| Put  -> df *. k *. norm_cdf (-. d2) -. dq *. s *. norm_cdf (-. d1)

(* Python: same logic, enforced at runtime via enum + if/else *)
if option_type == OptionType.CALL:
    return dq * S * _Phi(d1) - df * K * _Phi(d2)
else:
    return df * K * _Phi(-d2) - dq * S * _Phi(-d1)
```

The OCaml version makes it a **compile-time error** to add a new `option_type` variant without updating the pricer. This is Jane Street's core argument for ML-family languages in trading systems.
