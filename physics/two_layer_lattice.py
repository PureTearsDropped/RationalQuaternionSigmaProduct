#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
Two layers: does putting the INNER coefficients on the lattice make the network solvable?

    layer 1:  Z₁ = e^{W_a}·X₁^{w₁} + e^{W_b}·X₂^{w₂}        (inner Σ; its coefficients sit inside layer 2's Log)
    layer 2:  y  = A₂ · Z₁^{v} · X₃                        (outer readout A₂: least squares)

teacher:  W_a = 1/2, W_b = −1, w₁ = 1, w₂ = 2, v = 1/2, A₂ = (0.7, −0.1, 0.2, 0.05)
seed:     W_a = W_b = 0, w₁ = w₂ = 1, v = 1                (Z₁ = X₁ + X₂, y = A₂ Z₁ X₃)

(i)   lattice inner coefficients: every parameter but A₂ is a ledger; ledger moves on all channels of both layers,
      A₂ by LS for every candidate (variable projection restored), accept the best improvement.
(ii)  float inner coefficients: the same discrete search over the exponents, but the inner A₁ is continuous and
      solved by alternating Gauss–Newton (A₂ LS ↔ A₁ Levenberg–Marquardt via autograd) for every candidate.
(iii) off-lattice teacher (W_a = 0.53): (i) finds the nearest lattice point; then A₁ is refit in float from there.
Success is gauge-invariant (Z₁ ↦ cZ₁ is absorbed by A₂): the coefficient ratio e^{W_a − W_b} = e^{3/2} and the
exponents (w₁, w₂, v) = (1, 2, 1/2).   python physics/two_layer_lattice.py
"""
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
from lattice_coefficients import with_constant, coefficient_factor, balanced_zero

M = 3
E_IDX = 3                                  # constant node in layer-1 inputs (X₁, X₂, X₃, e)
ONES = lambda k: np.tile([[1.0, 0, 0, 0]], (k, 1))
A2_TRUE = np.array([0.7, -0.1, 0.2, 0.05])


def layer1(wa, wb, w1, w2, lattice=True):
    """Z₁ = e^{wa}·X₁^{w1} + e^{wb}·X₂^{w2}; with lattice=False the coefficient factors are omitted (float A₁)."""
    units = []
    for j, (wc, we) in enumerate(((wa, w1), (wb, w2))):
        f = [ref.make_power_factor(j, be.bounded_state(M, *we))]
        if lattice:
            f.insert(0, coefficient_factor(E_IDX, wc))
        units.append(ref.ProductUnit(f))
    return be.bounded_network(units, bound=M, n_inputs=4)


def layer2(v):
    """y = A₂ · Z₁^{v} · X₃   with layer-2 inputs (Z₁, X₃)."""
    return be.bounded_network([ref.ProductUnit([ref.make_power_factor(0, be.bounded_state(M, *v)),
                                                ref.make_passthrough_factor(1)])], bound=M, n_inputs=2)


def teacher():
    wa = be.bounded_state(M, 1, 2)
    wb = be.bounded_state(M, 1, 1, sign=1)
    return layer1(wa, wb, (1, 1), (2, 1)), layer2((1, 2))


def seed(lattice=True):
    return layer1(balanced_zero(M), balanced_zero(M), (1, 1), (1, 1), lattice), layer2((1, 1))


def data(s, n=400, wa_true=None):
    rng = np.random.default_rng(s)
    x = rng.normal(size=(n, 3, 4)) * 0.3
    x[:, :, 0] += 1.5
    x = with_constant(x)
    n1, n2 = teacher()
    if wa_true is not None:                                    # off-lattice inner coefficient
        A1 = np.array([[math.exp(wa_true), 0, 0, 0], [math.exp(-1), 0, 0, 0]])
        Z1 = ref.predict(x, layer1(None, None, (1, 1), (2, 1), lattice=False), A1)
    else:
        Z1 = ref.predict(x, n1, ONES(2))
    y = ref.predict(np.stack([Z1, x[:, 2]], 1), n2, A2_TRUE[None])
    return x, y


# ---------------------------------------------------------------- (i) all-lattice forward and fit
def forward_fit(x, y, n1, n2, A1=None, ridge=1e-10):
    Z1 = ref.predict(x, n1, ONES(len(n1.units)) if A1 is None else A1)
    x2 = np.stack([Z1, x[:, 2]], 1)
    A2 = ref.fit_readout(x2, y, n2, ridge)
    pred = ref.predict(x2, n2, A2)
    return float(np.mean((pred - y) ** 2)), A2


def all_ledger_moves(net):
    keys = [(ui, fi, li) for ui, fi, li, _ in net.iter_layers()]
    score = np.ones((len(keys), 4))
    return be.propose_ledger_moves(net, keys, dict(score=score), top_k_blocks=len(keys) * 4, include_side_flips=False)


# ---------------------------------------------------------------- (ii) float inner coefficients: alternating GN
def _wdict(net, prefix):
    return {k: torch.tensor(ref._get_layer(net, k).state.exponent(net.primes), dtype=torch.float64)
            for k in [(ui, fi, li) for ui, fi, li, _ in net.iter_layers()]}


def torch_forward(xg, n1, n2, A1, A2):
    ph1 = ref._torch_features(xg, n1, _wdict(n1, 1))[:, 0]                 # [N,U,4]
    Z1 = sum(ref.tqmul(A1[u].expand_as(ph1[:, u]), ph1[:, u]) for u in range(ph1.shape[1]))
    x2 = torch.stack([Z1, xg[:, 0, 2]], 1)[:, None]
    ph2 = ref._torch_features(x2, n2, _wdict(n2, 2))[:, 0]
    return sum(ref.tqmul(A2[u].expand_as(ph2[:, u]), ph2[:, u]) for u in range(ph2.shape[1]))


def fit_float_inner(x, y, n1, n2, A1_init, iters=30, ridge=1e-10):
    """alternating: A₂ by LS for the current A₁, then one LM step on A₁ with A₂ fixed (autograd Jacobian)."""
    xg = torch.tensor(x, dtype=torch.float64)[:, None]
    yt = torch.tensor(y, dtype=torch.float64)
    A1 = np.array(A1_init, dtype=np.float64)
    lam = 1e-3
    best = (np.inf, A1.copy(), None)
    for _ in range(iters):
        mse, A2 = forward_fit(x, y, n1, n2, A1, ridge)
        if mse < best[0]:
            best = (mse, A1.copy(), A2)
        a1 = torch.tensor(A1.reshape(-1), dtype=torch.float64, requires_grad=True)
        a2 = torch.tensor(np.asarray(A2), dtype=torch.float64)
        def resid(a):
            return (torch_forward(xg, n1, n2, a.reshape(-1, 4), a2) - yt).reshape(-1)
        J = torch.autograd.functional.jacobian(resid, a1)
        r = resid(a1).detach()
        JtJ = J.T @ J
        step = torch.linalg.solve(JtJ + lam * torch.diag(torch.diag(JtJ) + 1e-12), -J.T @ r)
        A1_new = (a1.detach() + step).numpy().reshape(-1, 4)
        mse_new, _ = forward_fit(x, y, n1, n2, A1_new, ridge)
        if mse_new < mse:
            A1, lam = A1_new, max(lam / 3, 1e-9)
        else:
            lam *= 10
        if mse < 1e-26:
            break
    mse, A2 = forward_fit(x, y, n1, n2, best[1], ridge)
    return mse, best[1], A2


# ---------------------------------------------------------------- the searches
def exponents_of(n1, n2):
    w = [l.state.fractions()[0] for u in n1.units for f in u.factors if f.name != "A" for l in f.exponent_layers]
    v = [l.state.fractions()[0] for u in n2.units for f in u.factors for l in f.exponent_layers]
    return tuple(w + v)


def coef_ratio_lattice(n1):
    W = [l.state.fractions()[0] for u in n1.units for f in u.factors if f.name == "A" for l in f.exponent_layers]
    return W[0] - W[1]


def search_lattice(x, y, n1, n2, max_steps=12):
    mse, A2 = forward_fit(x, y, n1, n2)
    steps, ev = 0, 0
    for _ in range(max_steps):
        best = None
        for which, net in ((1, n1), (2, n2)):
            for mv in all_ledger_moves(net):
                cand = be.apply_move(net, mv)
                c1, c2 = (cand, n2) if which == 1 else (n1, cand)
                m_, a_ = forward_fit(x, y, c1, c2)
                ev += 1
                if best is None or m_ < best[0]:
                    best = (m_, c1, c2, a_)
        if best is None or best[0] >= mse * (1 - 1e-6) - 1e-30:
            break
        mse, n1, n2, A2 = best
        steps += 1
    return mse, n1, n2, A2, steps, ev


def search_float(x, y, n1, n2, A1, max_steps=6, iters=8):
    mse, A1, A2 = fit_float_inner(x, y, n1, n2, A1, iters)
    steps, ev = 0, 0
    for _ in range(max_steps):
        best = None
        for which, net in ((1, n1), (2, n2)):
            for mv in all_ledger_moves(net):
                cand = be.apply_move(net, mv)
                c1, c2 = (cand, n2) if which == 1 else (n1, cand)
                m_, a1_, a2_ = fit_float_inner(x, y, c1, c2, A1, iters)
                ev += 1
                if best is None or m_ < best[0]:
                    best = (m_, c1, c2, a1_, a2_)
        if best is None or best[0] >= mse * (1 - 1e-6) - 1e-30:
            break
        mse, n1, n2, A1, A2 = best
        steps += 1
    return mse, n1, n2, A1, A2, steps, ev


def main():
    out = []
    P = lambda *a: (print(*a, flush=True), out.append(" ".join(str(s) for s in a)))
    P(__doc__)
    TRUE_X = (Fraction(1), Fraction(2), Fraction(1, 2))
    P("== (i) inner coefficients on the lattice, A₂ by LS")
    wins = 0
    for s in range(8):
        x, y = data(s)
        t0 = time.time()
        mse, n1, n2, A2, steps, ev = search_lattice(x, y, *seed(True))
        ok = exponents_of(n1, n2) == TRUE_X and coef_ratio_lattice(n1) == Fraction(3, 2) and mse / np.var(y) < 1e-20
        wins += ok
        P(f"  seed {s}: rel mse {mse/np.var(y):.2e}  steps {steps}  evals {ev}  {time.time()-t0:4.1f}s  "
          f"exponents {tuple(str(f) for f in exponents_of(n1, n2))}  W_a−W_b {coef_ratio_lattice(n1)}  {'exact' if ok else 'wrong'}")
    P(f"  {wins}/8\n")

    P("== (ii) inner coefficients float (alternating LS/LM per candidate): not run — an LM fit per discrete candidate "
      "took >15 min per seed and exhausted memory; two_layer_float_proposes.py is the float baseline (LM over every "
      "inner parameter at once, then snap).\n")

    P("== (iii) off-lattice inner coefficient W_a = 0.53: (i) then float refit of A₁ from the lattice point")
    for s in range(4):
        x, y = data(s, wa_true=0.53)
        mse, n1, n2, A2, steps, ev = search_lattice(x, y, *seed(True))
        W = [l.state.fractions()[0] for u in n1.units for f in u.factors if f.name == "A" for l in f.exponent_layers]
        bare = layer1(None, None, tuple(w.as_integer_ratio()) if False else None, None, lattice=False) if False else None
        # refit: strip the coefficient factors, start A₁ at the lattice values
        n1b = be.bounded_network([ref.ProductUnit([f.copy() for f in u.factors if f.name != "A"]) for u in n1.units], bound=M, n_inputs=4)
        A1_0 = np.array([[math.exp(float(W[0])), 0, 0, 0], [math.exp(float(W[1])), 0, 0, 0]])
        mse_f, A1, A2f = fit_float_inner(x, y, n1b, n2, A1_0, iters=60)
        ratio = np.linalg.norm(A1[0]) / np.linalg.norm(A1[1])
        P(f"  seed {s}: lattice rel mse {mse/np.var(y):.2e} W_a−W_b {W[0]-W[1]} exponents {tuple(str(f) for f in exponents_of(n1, n2))}"
          f" → float refit rel mse {mse_f/np.var(y):.2e}  log ratio {math.log(ratio):.6f} (true 1.53)")
    Path(__file__).resolve().parent.joinpath("results", "two_layer_lattice.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
