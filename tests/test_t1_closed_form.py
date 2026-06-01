#!/usr/bin/env python3
"""Model-free verification of the T1 readout transform (paper Lemmas 1 and 3).

T1 is an *algebraic* identity, not an empirical fit: for a final-residual install
delta = alpha * U[t] added to the pre-norm residual h, the induced logit bias is,
exactly,

    no-norm (Lemma 1):   b = U @ delta = alpha * (U @ U[t]) = alpha * g_t
    RMSNorm  (Lemma 3):  b = (alpha / r') * u_t  +  ((r - r') / (r * r')) * U(gamma . h)
                         with u_t = U(gamma . U[t]),  r = sqrt(mean(h^2)+eps),  r' = r(h+delta)

This test reconstructs the bias two independent ways on a *random* synthetic
unembedding (no model, no GPU): (a) "direct", by literally forming the logits
before/after the install and subtracting, and (b) "closed form", via the formula
above. They must agree to floating-point precision. This is the same check the
paper runs on real 7B checkpoints (experiments/01_closed_form), where the residual
max|direct - closed| is ~5e-6 in fp32; in float64 here it is ~1e-12.

Run:  python tests/test_t1_closed_form.py      (prints a report; exit 0 on pass)
      pytest tests/test_t1_closed_form.py       (also works)
"""
from __future__ import annotations

import numpy as np

# numpy 2.x with some BLAS backends (notably macOS Accelerate) emits spurious
# divide/overflow/invalid FP-flag warnings from matmul even when the result is
# exact (the residuals below are ~1e-15). Silence those flags; correctness is
# still enforced by the explicit assertions on the computed values.
np.seterr(all="ignore")

RNG = np.random.default_rng(0)


def rmsnorm_denom(h: np.ndarray, eps: float) -> float:
    return float(np.sqrt(np.mean(h * h) + eps))


def make_problem(vocab: int = 2000, dim: int = 256, eps: float = 1e-6):
    U = RNG.standard_normal((vocab, dim)) / np.sqrt(dim)   # unembedding rows
    h = RNG.standard_normal(dim)                            # pre-norm residual
    gamma = 1.0 + 0.1 * RNG.standard_normal(dim)            # RMSNorm gain
    return U, h, gamma, eps


def test_lemma1_no_norm():
    """N = Identity: the induced bias is exactly alpha * (U @ U[t])."""
    U, h, _, _ = make_problem()
    t, alpha = 42, 3.0
    delta = alpha * U[t]
    b_direct = U @ (h + delta) - U @ h
    b_closed = alpha * (U @ U[t])            # = alpha * g_t  (Gram column)
    err = float(np.max(np.abs(b_direct - b_closed)))
    assert err < 1e-10, f"Lemma 1 residual {err:.2e}"
    return err


def test_lemma3_rmsnorm():
    """RMSNorm: temperature s = r/r' plus the fixed-direction additive term."""
    U, h, gamma, eps = make_problem()
    t, alpha = 137, 5.0
    delta = alpha * U[t]
    r = rmsnorm_denom(h, eps)
    rp = rmsnorm_denom(h + delta, eps)

    def norm_logits(x):
        return U @ (gamma * x / rmsnorm_denom(x, eps))

    z = norm_logits(h)
    z_tilde = norm_logits(h + delta)
    b_direct = z_tilde - z

    u_t = U @ (gamma * U[t])                         # fixed bias direction
    b_closed = (alpha / rp) * u_t + ((r - rp) / (r * rp)) * (U @ (gamma * h))
    err = float(np.max(np.abs(b_direct - b_closed)))
    assert err < 1e-9, f"Lemma 3 residual {err:.2e}"

    # the rescale term is a pure positive temperature s = r/r' on z (ranking-preserving)
    s = r / rp
    assert s > 0.0
    rescale_only = s * z
    # z_tilde = s*z + additive(u_t); check the additive remainder lies along u_t's span
    remainder = z_tilde - rescale_only
    additive = (alpha / rp) * u_t
    err_add = float(np.max(np.abs(remainder - additive)))
    assert err_add < 1e-9, f"additive-term residual {err_add:.2e}"
    return err, s


def test_cone_confinement_rank():
    """T2/T2': m installed slots -> re-ranking residual has affine rank <= m,
    for arbitrary non-negative input-addressed weights."""
    U, h, gamma, eps = make_problem()
    m = 4
    targets = [11, 200, 555, 1234]
    alphas = [2.0, -1.5, 3.0, 0.5]
    g = np.stack([U @ (gamma * U[t]) for t in targets])      # m fixed generators

    def install_bias(weights):
        delta = sum(w * a * U[t] for w, a, t in zip(weights, alphas, targets))
        rp = rmsnorm_denom(h + delta, eps)
        z = U @ (gamma * h / rmsnorm_denom(h, eps))
        z_tilde = U @ (gamma * (h + delta) / rp)
        s = rmsnorm_denom(h, eps) / rp
        return z_tilde - s * z            # d(x) at the structural temperature

    # 50 random non-negative weight vectors (arbitrary "addressor" outputs)
    D = np.stack([install_bias(np.abs(RNG.standard_normal(m))) for _ in range(50)])
    rank = np.linalg.matrix_rank(D, tol=1e-6)
    assert rank <= m, f"re-ranking residual rank {rank} exceeds slot count {m}"
    return rank, m


def main():
    e1 = test_lemma1_no_norm()
    e3, s = test_lemma3_rmsnorm()
    rank, m = test_cone_confinement_rank()
    print("T1 / T2 model-free verification (float64, synthetic unembedding)")
    print("-" * 64)
    print(f"  Lemma 1 (no-norm)   max|direct - closed| = {e1:.2e}   (< 1e-10)")
    print(f"  Lemma 3 (RMSNorm)   max|direct - closed| = {e3:.2e}   (< 1e-9)")
    print(f"  temperature s = r/r' = {s:.6f}  (> 0, ranking-preserving)")
    print(f"  T2' cone rank: re-ranking residual rank = {rank} <= m = {m} slots")
    print("-" * 64)
    print("PASS: the readout transform is an exact identity; re-ranking is")
    print("confined to the <= m fixed installed directions.")


if __name__ == "__main__":
    main()
