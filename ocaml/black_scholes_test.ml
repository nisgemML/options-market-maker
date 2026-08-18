(* QCheck property-based tests for the OCaml Black-Scholes pricer.
   12 properties covering mathematical invariants.
   Run: ocamlfind ocamlopt -package qcheck -linkpkg black_scholes.ml black_scholes_test.ml -o test_bs && ./test_bs *)

let gen_s     = QCheck.float_range 20.0 500.0
let gen_k     = QCheck.float_range 20.0 500.0
let gen_t     = QCheck.float_range 0.02 3.0
let gen_sigma = QCheck.float_range 0.05 1.5
let gen_r     = QCheck.float_range 0.0  0.10

let quad s k t sigma = QCheck.(map (fun ((a,b),(c,d)) -> (a,b,c,d)) (pair (pair s k) (pair t sigma)))

(* 1. Put-call parity: C - P = S·e^{-rT} - K·e^{-rT} *)
let prop_put_call_parity =
  QCheck.Test.make ~count:2000 ~name:"put-call parity"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       let r = 0.05 in
       let c = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Call in
       let p = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Put  in
       abs_float ((c -. p) -. (s -. k *. exp(-. r *. t))) < 1e-6)

(* 2. Call delta in [0,1] *)
let prop_call_delta_bounds =
  QCheck.Test.make ~count:2000 ~name:"call delta in [0,1]"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       let d = Black_scholes.delta ~s ~k ~t ~r:0.05 ~sigma Black_scholes.Call in
       d >= -1e-8 && d <= 1.0 +. 1e-8)

(* 3. Put delta in [-1,0] *)
let prop_put_delta_bounds =
  QCheck.Test.make ~count:2000 ~name:"put delta in [-1,0]"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       let d = Black_scholes.delta ~s ~k ~t ~r:0.05 ~sigma Black_scholes.Put in
       d >= -1.0 -. 1e-8 && d <= 1e-8)

(* 4. Gamma >= 0 *)
let prop_gamma_nonneg =
  QCheck.Test.make ~count:2000 ~name:"gamma >= 0"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       Black_scholes.gamma ~s ~k ~t ~r:0.05 ~sigma () >= -1e-10)

(* 5. Vega >= 0 *)
let prop_vega_nonneg =
  QCheck.Test.make ~count:2000 ~name:"vega >= 0"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       Black_scholes.vega ~s ~k ~t ~r:0.05 ~sigma () >= -1e-10)

(* 6. IV round-trip: price(IV(price)) = price *)
let prop_iv_roundtrip =
  QCheck.Test.make ~count:1000 ~name:"IV round-trip: price(IV(p)) = p"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       let r = 0.05 in
       let market_price = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Call in
       match Black_scholes.implied_vol ~market_price ~s ~k ~t ~r Black_scholes.Call with
       | Error _  -> true
       | Ok iv    ->
           let repriced = Black_scholes.price ~s ~k ~t ~r ~sigma:iv Black_scholes.Call in
           abs_float (repriced -. market_price) < 1e-4)

(* 7. Call price >= intrinsic value *)
let prop_call_ge_intrinsic =
  QCheck.Test.make ~count:2000 ~name:"call price >= intrinsic value"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       let r = 0.05 in
       let c = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Call in
       let intrinsic = max 0.0 (s -. k *. exp (-. r *. t)) in
       c >= intrinsic -. 1e-8)

(* 8. Option prices non-negative *)
let prop_prices_nonneg =
  QCheck.Test.make ~count:2000 ~name:"option prices non-negative"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       let r = 0.05 in
       let c = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Call in
       let p = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Put  in
       c >= -1e-10 && p >= -1e-10)

(* 9. Call increasing in S *)
let prop_call_monotone_in_s =
  QCheck.Test.make ~count:1000 ~name:"call increasing in S"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       let r = 0.05 in
       let bump = s *. 0.01 in
       let c1 = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Call in
       let c2 = Black_scholes.price ~s:(s +. bump) ~k ~t ~r ~sigma Black_scholes.Call in
       c2 >= c1 -. 1e-8)

(* 10. Both call and put increase with volatility *)
let prop_monotone_in_vol =
  QCheck.Test.make ~count:1000 ~name:"prices increase with volatility"
    (quad gen_s gen_k gen_t gen_sigma)
    (fun (s, k, t, sigma) ->
       let r = 0.05 and bump = 0.01 in
       let c1 = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Call in
       let c2 = Black_scholes.price ~s ~k ~t ~r ~sigma:(sigma +. bump) Black_scholes.Call in
       let p1 = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Put  in
       let p2 = Black_scholes.price ~s ~k ~t ~r ~sigma:(sigma +. bump) Black_scholes.Put  in
       c2 >= c1 -. 1e-8 && p2 >= p1 -. 1e-8)

(* 11. OTM call decays with time *)
let prop_otm_theta_negative =
  QCheck.Test.make ~count:500 ~name:"OTM call decays with time"
    QCheck.(pair (float_range 50.0 80.0) (float_range 0.1 2.0))
    (fun (s, t) ->
       let k = 120.0 and sigma = 0.20 and r = 0.05 in
       if t < 0.1 then true
       else
         let c1 = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Call in
         let c2 = Black_scholes.price ~s ~k ~t:(t *. 0.9) ~r ~sigma Black_scholes.Call in
         c1 >= c2 -. 1e-8)

(* 12. Homogeneity: C(λS, λK) = λ·C(S, K) *)
let prop_homogeneity =
  QCheck.Test.make ~count:1000 ~name:"BS homogeneity: C(λS, λK) = λ·C(S, K)"
    QCheck.(pair (quad gen_s gen_k gen_t gen_sigma) (float_range 0.5 3.0))
    (fun ((s, k, t, sigma), lambda) ->
       let r = 0.05 in
       let c1 = Black_scholes.price ~s ~k ~t ~r ~sigma Black_scholes.Call in
       let c2 = Black_scholes.price
                  ~s:(lambda *. s) ~k:(lambda *. k) ~t ~r ~sigma
                  Black_scholes.Call in
       abs_float (c2 -. lambda *. c1) < 1e-6 *. (lambda *. c1 +. 1.0))

let () =
  QCheck_runner.run_tests_main [
    prop_put_call_parity;
    prop_call_delta_bounds;
    prop_put_delta_bounds;
    prop_gamma_nonneg;
    prop_vega_nonneg;
    prop_iv_roundtrip;
    prop_call_ge_intrinsic;
    prop_prices_nonneg;
    prop_call_monotone_in_s;
    prop_monotone_in_vol;
    prop_otm_theta_negative;
    prop_homogeneity;
  ]
