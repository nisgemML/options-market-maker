(** Black-Scholes pricer in OCaml.

    Mirrors the Python implementation in src/pricing/black_scholes.py —
    same mathematical model, different type-system guarantees.

    OCaml enforces exhaustive pattern matching on [option_type], making
    it impossible to forget a branch. The [result] return type on [implied_vol]
    forces the caller to handle the solver-failure case at compile time —
    no runtime exceptions from unchecked IV failures.

    Cross-reference: Python implementation at src/pricing/black_scholes.py
    Repository: https://github.com/nisgemML/options-market-maker

    References:
      Black, F. & Scholes, M. (1973). The pricing of options and corporate
      liabilities. Journal of Political Economy, 81(3), 637-654.
      Merton, R. (1973). Theory of rational option pricing. Bell Journal of
      Economics and Management Science, 4(1), 141-183. *)

type option_type = Call | Put

(** Normal CDF via the error function: N(x) = 0.5*(1 + erf(x/√2)). *)
let norm_cdf x =
  0.5 *. (1.0 +. Stdlib.Float.( (x /. sqrt 2.0) |> fun y ->
    (* erf approximation — Abramowitz & Stegun 7.1.26, max |ε| < 1.5e-7 *)
    let t = 1.0 /. (1.0 +. 0.3275911 *. Float.abs y) in
    let poly = t *. (0.254829592
      +. t *. (-0.284496736
      +. t *. (1.421413741
      +. t *. (-1.453152027
      +. t *. 1.061405429)))) in
    let erf_abs = 1.0 -. poly *. exp (-. y *. y) in
    if y >= 0.0 then erf_abs else -. erf_abs))

(** Standard normal PDF. *)
let norm_pdf x = exp (-. 0.5 *. x *. x) /. sqrt (2.0 *. Float.pi)

(** d₁ and d₂ for Black-Scholes.
    @raise Invalid_argument if T <= 0 or sigma <= 0. *)
let d1_d2 ~s ~k ~t ~r ~sigma ~q =
  if t <= 0.0 then invalid_arg "d1_d2: T must be > 0";
  if sigma <= 0.0 then invalid_arg "d1_d2: sigma must be > 0";
  let sqrt_t = sqrt t in
  let d1 = (log (s /. k) +. (r -. q +. 0.5 *. sigma *. sigma) *. t)
            /. (sigma *. sqrt_t) in
  let d2 = d1 -. sigma *. sqrt_t in
  (d1, d2)

(** Black-Scholes option price.
    [q] is continuous dividend yield (default 0.0). *)
let price ~s ~k ~t ~r ~sigma ?(q=0.0) opt =
  let d1, d2 = d1_d2 ~s ~k ~t ~r ~sigma ~q in
  let df = exp (-. r *. t) in
  let dq = exp (-. q *. t) in
  match opt with
  | Call -> dq *. s *. norm_cdf d1 -. df *. k *. norm_cdf d2
  | Put  -> df *. k *. norm_cdf (-. d2) -. dq *. s *. norm_cdf (-. d1)

(** Delta: ∂C/∂S. *)
let delta ~s ~k ~t ~r ~sigma ?(q=0.0) opt =
  let d1, _ = d1_d2 ~s ~k ~t ~r ~sigma ~q in
  let dq = exp (-. q *. t) in
  match opt with
  | Call -> dq *. norm_cdf d1
  | Put  -> dq *. (norm_cdf d1 -. 1.0)

(** Gamma: ∂²C/∂S² (identical for calls and puts). *)
let gamma ~s ~k ~t ~r ~sigma ?(q=0.0) () =
  let d1, _ = d1_d2 ~s ~k ~t ~r ~sigma ~q in
  let dq = exp (-. q *. t) in
  dq *. norm_pdf d1 /. (s *. sigma *. sqrt t)

(** Vega: ∂C/∂σ per 1-point move (not 1%). *)
let vega ~s ~k ~t ~r ~sigma ?(q=0.0) () =
  let d1, _ = d1_d2 ~s ~k ~t ~r ~sigma ~q in
  let dq = exp (-. q *. t) in
  dq *. s *. norm_pdf d1 *. sqrt t

(** Implied volatility via bisection search.
    Returns [Ok iv] or [Error msg] — the caller must handle both cases.
    Convergence to tol=1e-7 within 100 iterations for well-conditioned inputs. *)
let implied_vol ~market_price ~s ~k ~t ~r ?(q=0.0) opt =
  let lo = ref 1e-6 and hi = ref 20.0 in
  let f v = price ~s ~k ~t ~r ~sigma:v ~q opt -. market_price in
  if f !lo *. f !hi > 0.0 then
    Error (Printf.sprintf "No bracket: f(%.2e)=%.4f, f(%.2e)=%.4f" !lo (f !lo) !hi (f !hi))
  else begin
    for _ = 1 to 100 do
      let mid = (!lo +. !hi) /. 2.0 in
      if f mid < 0.0 then lo := mid else hi := mid
    done;
    let iv = (!lo +. !hi) /. 2.0 in
    Ok iv
  end

(** Put-call parity check: C - P = S·e^{-qT} - K·e^{-rT}.
    Returns true if error < tol. *)
let put_call_parity_check ~call_price ~put_price ~s ~k ~t ~r ?(q=0.0) ?(tol=1e-6) () =
  let lhs = call_price -. put_price in
  let rhs = s *. exp (-. q *. t) -. k *. exp (-. r *. t) in
  Float.abs (lhs -. rhs) < tol
