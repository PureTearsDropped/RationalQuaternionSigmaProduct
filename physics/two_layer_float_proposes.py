#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""(iv) two layers, float proposes / lattice decides: every inner parameter (inner coefficients W_a, W_b AND the
exponents w₁, w₂, v) is continuous in a Levenberg–Marquardt fit with A₂ by LS (variable projection), from the seed
Z₁ = X₁ + X₂, y = A₂ Z₁ X₃; then everything is snapped to the nearest ledger of bound m and verified exactly.
Compared with the single-/pair-move lattice searches that stop at (w₂ = 1, v = 2/3, W_a − W_b = 1/3).
   python physics/two_layer_float_proposes.py"""
from __future__ import annotations
import math
import sys
import time
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import flexible_rational_quaternion_sigma_product as ref
import bounded_exponent as be
from lattice_coefficients import nearest_ledger
from two_layer_lattice import data, seed, forward_fit, exponents_of, coef_ratio_lattice, layer1, layer2, M, ONES


def torch_forward_params(xg, p, A2):
    """p = (a, b, w1, w2, v) real; Z₁ = e^a X₁^{w1} + e^b X₂^{w2}; y = A₂ Z₁^{v} X₃ (torch, differentiable)."""
    X1, X2, X3 = xg[:, 0], xg[:, 1], xg[:, 2]
    w1 = torch.stack([p[2], p[2] * 0, p[2] * 0, p[2] * 0])
    w2 = torch.stack([p[3], p[3] * 0, p[3] * 0, p[3] * 0])
    v = torch.stack([p[4], p[4] * 0, p[4] * 0, p[4] * 0])
    Z1 = torch.exp(p[0]) * ref.tqpow(X1, w1.expand_as(X1), 'L') + torch.exp(p[1]) * ref.tqpow(X2, w2.expand_as(X2), 'L')
    P2 = ref.tqmul(ref.tqpow(Z1, v.expand_as(Z1), 'L'), X3)
    return ref.tqmul(A2.expand_as(P2), P2)


def ls_A2(x, y, p):
    n1 = layer1(None, None, (1, 1), (1, 1), lattice=False)     # placeholders; exponents come from p via numpy below
    # numpy forward for LS of A₂ with float exponents:
    X1, X2, X3 = x[:, 0], x[:, 1], x[:, 2]
    Z1 = math.exp(p[0]) * ref.qpow(X1, np.array([p[2], 0, 0, 0]), 'L') + math.exp(p[1]) * ref.qpow(X2, np.array([p[3], 0, 0, 0]), 'L')
    P2 = ref.qmul(ref.qpow(Z1, np.array([p[4], 0, 0, 0]), 'L'), X3)
    D = ref.left_multiply_design(P2[:, None, None, :])[:, 0, 0]            # [N,4,4]
    Dm = D.reshape(-1, 4)
    A2 = np.linalg.solve(Dm.T @ Dm + 1e-10 * np.eye(4), Dm.T @ y.reshape(-1))
    pred = (D @ A2)
    return float(np.mean((pred - y) ** 2)), A2


def lm_fit(x, y, p0, iters=200):
    xg = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    p = np.array(p0, dtype=np.float64)
    lam = 1e-2
    mse, A2 = ls_A2(x, y, p)
    for it in range(iters):
        pt = torch.tensor(p, requires_grad=True)
        A2t = torch.tensor(A2)
        resid = lambda q: (torch_forward_params(xg, q, A2t) - yt).reshape(-1)
        J = torch.autograd.functional.jacobian(resid, pt)
        r = resid(pt).detach()
        JtJ = J.T @ J
        step = torch.linalg.solve(JtJ + lam * torch.diag(torch.diag(JtJ) + 1e-12), -J.T @ r).numpy()
        p_new = p + step
        mse_new, A2_new = ls_A2(x, y, p_new)
        if mse_new < mse:
            p, mse, A2, lam = p_new, mse_new, A2_new, max(lam / 3, 1e-12)
        else:
            lam *= 10
        if mse < 1e-28 or lam > 1e12:
            break
    return p, mse, A2, it + 1


def snap_all(p):
    """(a, b, w1, w2, v) → ledgers; returns the lattice networks and the fractions."""
    fr = []
    for val in p:
        sign, n, o = nearest_ledger(float(val), M)
        fr.append((sign, n, o, (-1) ** sign * be.total_fraction(n, o)))
    wa = be.bounded_state(M, fr[0][1], fr[0][2], sign=fr[0][0])
    wb = be.bounded_state(M, fr[1][1], fr[1][2], sign=fr[1][0])
    # exponents w1, w2, v are ledgers with sign channel
    def st(f):
        return be.bounded_state(M, f[1], f[2], sign=f[0])
    n1 = layer1(wa, wb, (0, 0), (0, 0))
    for u, f in zip(n1.units, (fr[2], fr[3])):
        u.factors[1].exponent_layers[0].state = st(f)
    n2 = layer2((0, 0))
    n2.units[0].factors[0].exponent_layers[0].state = st(fr[4])
    return n1, n2, [f[3] for f in fr]


def main():
    out = []
    P = lambda *a: (print(*a, flush=True), out.append(" ".join(str(s) for s in a)))
    P(__doc__)
    TRUE_X = (Fraction(1), Fraction(2), Fraction(1, 2))
    wins = 0
    for s in range(8):
        x, y = data(s)
        t0 = time.time()
        p, mse_f, A2, it = lm_fit(x, y, [0, 0, 1, 1, 1])
        n1, n2, fr = snap_all(p)
        mse_l, A2l = forward_fit(x, y, n1, n2)
        ok = exponents_of(n1, n2) == TRUE_X and coef_ratio_lattice(n1) == Fraction(3, 2) and mse_l / np.var(y) < 1e-20
        wins += ok
        P(f"  seed {s}: LM {it:3d} iters rel mse {mse_f/np.var(y):.2e}  p = (a {p[0]:+.4f}, b {p[1]:+.4f}, w₁ {p[2]:.4f}, w₂ {p[3]:.4f}, v {p[4]:.4f})"
          f"  → snap {tuple(str(f) for f in fr)}  lattice rel mse {mse_l/np.var(y):.2e}  {time.time()-t0:4.1f}s  {'exact' if ok else 'wrong'}")
    P(f"  {wins}/8   (gauge: a − b = 3/2 is what is checked; the LM may land on any gauge copy)")
    Path(__file__).resolve().parent.joinpath("results", "two_layer_float_proposes.txt").write_text("\n".join(out))


if __name__ == "__main__" and "--off" not in sys.argv:
    main()


def off_lattice(seeds=(0, 1, 2), wa_true=0.53):
    """(iii′) teacher inner coefficient off the lattice: LM proposes → snap (the lattice floor) → LM refit from the
    lattice point recovers the float value."""
    out = []
    P = lambda *a: (print(*a, flush=True), out.append(" ".join(str(s) for s in a)))
    P(f"== off-lattice inner coefficient W_a = {wa_true} (W_b = −1, ratio a − b = {wa_true + 1})")
    for s in seeds:
        x, y = data(s, wa_true=wa_true)
        p, mse_f, A2, it = lm_fit(x, y, [0, 0, 1, 1, 1])
        n1, n2, fr = snap_all(p)
        mse_l, _ = forward_fit(x, y, n1, n2)
        p_l = [float(f) for f in fr]
        p2, mse_r, _, it2 = lm_fit(x, y, p_l, iters=100)
        P(f"  seed {s}: LM rel mse {mse_f/np.var(y):.2e} (a−b {p[0]-p[1]:.4f}) → snap {tuple(str(f) for f in fr)} lattice rel mse {mse_l/np.var(y):.2e}"
          f" → refit from the lattice point: rel mse {mse_r/np.var(y):.2e} (a−b {p2[0]-p2[1]:.6f}, w₂ {p2[3]:.6f}, v {p2[4]:.6f}) in {it2} iters")
    with open(Path(__file__).resolve().parent / "results" / "two_layer_float_proposes.txt", "a") as fh:
        fh.write("\n\n" + "\n".join(out))


if __name__ == "__main__" and "--off" in sys.argv:
    off_lattice()
