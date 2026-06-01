"""
Analyzer for Experiment 3 (install-then-remove with AR completion).

Inputs:
  --in_json   full_sweep.json from probe_install_then_remove.py

Outputs:
  * stdout: per-cell rates, Wilson 95% CIs, McNemar canon-vs-shuf,
    donor-specific Δ (canon − random) bootstrap CIs, pre-registered
    PASS bar evaluation.
  * --out_json: machine-readable per-cell stats.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial test on (b,c) discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k or X >= n-k) under Binomial(n, 0.5)
    cum = 0.0
    for i in range(0, k + 1):
        cum += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2.0 * cum)
    return p


def bootstrap_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                      seed: int = 0) -> Tuple[float, float, float]:
    """Returns (mean_diff, lo, hi). x and y are 0/1 arrays of equal length."""
    rng = np.random.default_rng(seed)
    n = len(x)
    obs = float(x.mean() - y.mean())
    if n == 0:
        return obs, 0.0, 0.0
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(x[idx].mean() - y[idx].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs, float(lo), float(hi)


METRICS = ("rate_substring", "rate_first2piece", "rate_firstpiece", "rate_cont_token_1")
PRIMARY_METRIC = "rate_first2piece"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--alpha_ci", type=float, default=0.05)
    args = ap.parse_args()

    with open(args.in_json) as f:
        payload = json.load(f)
    meta = payload["meta"]
    cells = payload["cells"]

    # Index by (tau, alpha, donor, seed) → records
    by_cell: Dict[Tuple[int, float, str, str], List[Dict[str, Any]]] = {}
    for cell in cells:
        key = (int(cell["tau"]), float(cell["alpha"]), str(cell["donor"]), str(cell["seed"]))
        by_cell[key] = cell["records"]

    taus = sorted({k[0] for k in by_cell})
    alphas = sorted({k[1] for k in by_cell})
    donors = sorted({k[2] for k in by_cell})
    seeds = sorted({k[3] for k in by_cell})

    print(f"Cells: {len(by_cell)} | τ={taus} α={alphas} donors={donors} seeds={seeds}")
    print(f"Primary metric: {PRIMARY_METRIC}\n")

    summary: List[Dict[str, Any]] = []
    pass_cells: List[Dict[str, Any]] = []

    for tau in taus:
        for alpha in alphas:
            # Find canonical row (seed='any')
            canon_key = (tau, alpha, "canon", "any")
            if canon_key not in by_cell:
                continue
            canon_rows = by_cell[canon_key]
            cid_to_canon = {r["case_id"]: r for r in canon_rows}

            # For shuf/random, pool seeds (per Codex correction: seeds vary
            # ONLY the control draws; per-seed CI then pooled across seeds
            # for the headline rate).
            shuf_rows: List[Dict[str, Any]] = []
            rand_rows: List[Dict[str, Any]] = []
            for s in seeds:
                if s == "any":
                    continue
                shuf_rows.extend(by_cell.get((tau, alpha, "shuf", s), []))
                rand_rows.extend(by_cell.get((tau, alpha, "random", s), []))

            for metric in METRICS:
                key_field = metric.replace("rate_", "gold_")
                key_field_map = {
                    "rate_substring": "gold_substring_norm",
                    "rate_first2piece": "gold_first10_first2piece",
                    "rate_firstpiece": "gold_first10_firstpiece",
                    "rate_cont_token_1": "gold_at_cont_token_1",
                }
                kf = key_field_map[metric]
                canon = np.array([1 if r[kf] else 0 for r in canon_rows], dtype=np.int64)
                shuf = np.array([1 if r[kf] else 0 for r in shuf_rows], dtype=np.int64)
                rand = np.array([1 if r[kf] else 0 for r in rand_rows], dtype=np.int64)

                c_rate = float(canon.mean()) if len(canon) else 0.0
                s_rate = float(shuf.mean()) if len(shuf) else 0.0
                r_rate = float(rand.mean()) if len(rand) else 0.0
                c_lo, c_hi = wilson_ci(int(canon.sum()), len(canon), args.alpha_ci)
                s_lo, s_hi = wilson_ci(int(shuf.sum()), len(shuf), args.alpha_ci)
                r_lo, r_hi = wilson_ci(int(rand.sum()), len(rand), args.alpha_ci)

                # McNemar canon vs shuf: pair by case_id (within seed); pool over seeds.
                b_count = 0
                c_count_disc = 0
                for s in seeds:
                    if s == "any":
                        continue
                    s_rows = by_cell.get((tau, alpha, "shuf", s), [])
                    for r in s_rows:
                        cid = r["case_id"]
                        cn = cid_to_canon.get(cid)
                        if cn is None:
                            continue
                        cn_v = 1 if cn[kf] else 0
                        sh_v = 1 if r[kf] else 0
                        if cn_v == 1 and sh_v == 0:
                            b_count += 1
                        elif cn_v == 0 and sh_v == 1:
                            c_count_disc += 1
                p_mc = mcnemar_exact_p(b_count, c_count_disc)

                # Donor-specific Δ = canon_rate − random_rate, paired by case_id
                if len(rand) and len(canon):
                    # Align random by case_id; each canon case has 2 random rows (per seed); use mean per case.
                    rand_by_case = defaultdict(list)
                    for r in rand_rows:
                        rand_by_case[r["case_id"]].append(1 if r[kf] else 0)
                    paired_canon: List[int] = []
                    paired_rand: List[float] = []
                    for cid, cn in cid_to_canon.items():
                        if cid not in rand_by_case:
                            continue
                        paired_canon.append(1 if cn[kf] else 0)
                        paired_rand.append(float(np.mean(rand_by_case[cid])))
                    pc = np.array(paired_canon, dtype=np.float64)
                    pr = np.array(paired_rand, dtype=np.float64)
                    delta, dlo, dhi = bootstrap_diff_ci(pc, pr,
                                                       n_boot=2000,
                                                       seed=int(1000 * float(alpha)) + tau)
                else:
                    delta, dlo, dhi = (0.0, 0.0, 0.0)

                # Also canon - shuf paired
                if len(shuf) and len(canon):
                    shuf_by_case = defaultdict(list)
                    for r in shuf_rows:
                        shuf_by_case[r["case_id"]].append(1 if r[kf] else 0)
                    pc2: List[int] = []
                    pshuf: List[float] = []
                    for cid, cn in cid_to_canon.items():
                        if cid not in shuf_by_case:
                            continue
                        pc2.append(1 if cn[kf] else 0)
                        pshuf.append(float(np.mean(shuf_by_case[cid])))
                    pc2a = np.array(pc2, dtype=np.float64)
                    psa = np.array(pshuf, dtype=np.float64)
                    delta_s, dlo_s, dhi_s = bootstrap_diff_ci(pc2a, psa,
                                                              n_boot=2000,
                                                              seed=int(2000 * float(alpha)) + tau)
                else:
                    delta_s, dlo_s, dhi_s = (0.0, 0.0, 0.0)

                # PASS bar (pre-registered, for primary metric only):
                pass_bar = (
                    metric == PRIMARY_METRIC and
                    c_rate >= 0.40 and
                    (c_rate - s_rate) >= 0.20 and
                    delta >= 0.10
                )
                rec = {
                    "tau": tau, "alpha": alpha, "metric": metric,
                    "canon_rate": c_rate, "canon_n": int(len(canon)),
                    "canon_ci": [c_lo, c_hi],
                    "shuf_rate": s_rate, "shuf_n": int(len(shuf)),
                    "shuf_ci": [s_lo, s_hi],
                    "rand_rate": r_rate, "rand_n": int(len(rand)),
                    "rand_ci": [r_lo, r_hi],
                    "canon_minus_shuf": delta_s,
                    "canon_minus_shuf_ci": [dlo_s, dhi_s],
                    "canon_minus_random": delta,
                    "canon_minus_random_ci": [dlo, dhi],
                    "mcnemar_b": b_count, "mcnemar_c": c_count_disc,
                    "mcnemar_p": p_mc,
                    "pass_bar": bool(pass_bar),
                }
                summary.append(rec)
                if pass_bar:
                    pass_cells.append(rec)

    # ---- print primary metric cells ----------------------------------------
    print(f"=== Primary metric ({PRIMARY_METRIC}) per (τ, α) cell ===")
    print(f"{'τ':>3} {'α':>5}  {'canon':>7} {'shuf':>7} {'rand':>7}  "
          f"{'c−s':>7} {'c−r':>7}  {'McNB':>4}/{'McNC':>4}  {'McNp':>9}  {'PASS':>5}")
    for tau in taus:
        for alpha in alphas:
            recs = [r for r in summary
                    if r["tau"] == tau and r["alpha"] == alpha
                    and r["metric"] == PRIMARY_METRIC]
            if not recs:
                continue
            r = recs[0]
            print(f"{tau:>3} {alpha:>5g}  "
                  f"{r['canon_rate']:>7.3f} {r['shuf_rate']:>7.3f} {r['rand_rate']:>7.3f}  "
                  f"{r['canon_minus_shuf']:>+7.3f} {r['canon_minus_random']:>+7.3f}  "
                  f"{r['mcnemar_b']:>4d}/{r['mcnemar_c']:>4d}  "
                  f"{r['mcnemar_p']:>9.3e}  {'YES' if r['pass_bar'] else 'no':>5}")

    print()
    print(f"=== All other metrics (best cell per metric, by canon rate) ===")
    for m in METRICS:
        if m == PRIMARY_METRIC:
            continue
        recs = [r for r in summary if r["metric"] == m]
        recs.sort(key=lambda x: x["canon_rate"], reverse=True)
        if not recs:
            continue
        top = recs[0]
        print(f"  {m:>22s}: best τ={top['tau']} α={top['alpha']:g} "
              f"canon={top['canon_rate']:.3f} shuf={top['shuf_rate']:.3f} "
              f"rand={top['rand_rate']:.3f} c−s={top['canon_minus_shuf']:+.3f} "
              f"c−r={top['canon_minus_random']:+.3f}")

    print()
    if pass_cells:
        print(f"=== PASS cells (all 3 pre-registered bars on {PRIMARY_METRIC}) ===")
        for r in pass_cells:
            print(f"  τ={r['tau']} α={r['alpha']:g}  canon={r['canon_rate']:.3f} "
                  f"shuf={r['shuf_rate']:.3f} rand={r['rand_rate']:.3f} "
                  f"c−s={r['canon_minus_shuf']:+.3f} c−r={r['canon_minus_random']:+.3f}")
    else:
        # Verdict per handoff decision tree
        canon_max = max((r["canon_rate"] for r in summary if r["metric"] == PRIMARY_METRIC), default=0.0)
        if canon_max < 0.40:
            verdict = "NO_CONTINUATION_EFFECT"
            line = "Discrete-token causal mediation does not carry install signal."
        else:
            # canon ≥ 0.40 somewhere; check whether donor-specific Δ holds
            cands = [r for r in summary if r["metric"] == PRIMARY_METRIC
                     and r["canon_rate"] >= 0.40
                     and (r["canon_rate"] - r["shuf_rate"]) >= 0.20]
            if cands and max(r["canon_minus_random"] for r in cands) < 0.10:
                verdict = "DONOR_NON_SPECIFIC"
                line = "Canon ≥0.40 and canon−shuf ≥0.20 in some cell, but canon−random < 0.10. KV-lease pattern."
            else:
                verdict = "PARTIAL"
                line = "Some cell at/above bar but not all three together. Inspect."
        print(f"=== VERDICT: {verdict} ===")
        print(f"  {line}")
        print(f"  primary canon max = {canon_max:.3f}")

    out: Dict[str, Any] = {
        "meta": meta,
        "primary_metric": PRIMARY_METRIC,
        "summary": summary,
        "pass_cells": pass_cells,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
