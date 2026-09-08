#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
Approximating a function that is NOT on the lattice, with every coefficient on the lattice (A ≡ 1).

Seed: log-Fourier modes X^{±ik}, k = 1..K (and a constant unit X⁰), each unit led by the coefficient factor e^{W}.
Targets (inputs real, x = (x, 0, 0, 0); the constant node e is the last input):
    T1  f(x)     = 1/(1 + 4(log x − 1)²)                on x ∈ [1, e²]           (not a power law, not periodic)
    T2  f(x)     = sin(3x)/x                            on x ∈ [1, e²]
    T3  f(x₁,x₂) = log(x₁ + x₂)                         on [1, 3]²               (the sum sits outside one layer)
Modes:
    A   LS readout (the reference), exponents refined by ±1 count moves          — the continuous ceiling
    D0  A's exponents, Log A snapped to the nearest ledger of bound m, A ≡ 1     — the snap alone
    D   D0 then ledger moves on every channel (exponents and coefficients)        — LS proposes, the lattice decides
Reported: relative test MSE  = MSE / Var(f), on 2000 held-out samples.        python physics/lattice_approximation.py
"""
from __future__ import annotations
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import flexible_rational_quaternion_sigma_product as ref
import bounded_exponent as be
from lattice_coefficients import (with_constant, coefficient_factor, fit_quaternion_ls, fit_fixed_one, search,
                                  snap_coefficients, balanced_zero)

TARGETS = {
    "T1 1/(1+4(log x−1)²)": (1, lambda x: 1 / (1 + 4 * (np.log(x[:, 0]) - 1) ** 2), (1.0, math.e ** 2)),
    "T2 sin(3x)/x":           (1, lambda x: np.sin(3 * x[:, 0]) / x[:, 0], (1.0, math.e ** 2)),
    "T3 log(x₁+x₂)":          (2, lambda x: np.log(x[:, 0] + x[:, 1]), (1.0, 3.0)),
}


def make_data(seed, n, n_inputs, f, lo, hi):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(lo, hi, size=(n, n_inputs))
    x = np.zeros((n, n_inputs, 4))
    x[:, :, 0] = xs
    y = np.zeros((n, 4))
    y[:, 0] = f(xs)
    return with_constant(x), y


def fourier_seed(n_inputs, K, m):
    """units X^{ik} for every mode vector k ∈ {−K..K}^I (one unit per vector, k = 0 the constant), coefficient e^{0}."""
    C = n_inputs
    units = []
    grid = range(-K, K + 1)
    import itertools
    for ks in itertools.product(grid, repeat=n_inputs):
        factors = [coefficient_factor(C, balanced_zero(m))]
        for j, k in enumerate(ks):
            st = be.BoundedRationalQuaternionState.zero(m)
            st.set(0, 0, 1, 1).set(0, 1, 1, 1)                       # scalar exponent balanced 0
            if k:
                st.set(1, 0 if k > 0 else 1, abs(k), 1)              # ± i k
            factors.append(ref.Factor(j, exponent_layers=[ref.ExponentLayer(st, 'L')], name=f"X{j}"))
        units.append(ref.ProductUnit(factors))
    return be.bounded_network(units, bound=m)


def rel_mse(pred, y):
    return float(np.mean((pred - y) ** 2) / np.var(y[:, 0]))


def run(target, K, m, seed=0, n=2000, steps=8, P=print):
    n_inputs, f, (lo, hi) = TARGETS[target]
    xtr, ytr = make_data(seed, n, n_inputs, f, lo, hi)
    xte, yte = make_data(seed + 100, n, n_inputs, f, lo, hi)
    C = n_inputs
    net0 = fourier_seed(n_inputs, K, m)
    U = len(net0.units)
    # mode A: the LS readout (no coefficient factor in the readout: strip it, since A is free)
    netA0 = be.bounded_network([ref.ProductUnit([fc.copy() for fc in u.factors if fc.name != "A"]) for u in net0.units], bound=m)
    t0 = time.time()
    netA, coeffA, mseA, trA, evA = search(xtr, ytr, netA0, fit_quaternion_ls, max_steps=steps, moves='count')
    A_test = rel_mse(ref.predict(xte, netA, coeffA), yte)
    tA = time.time() - t0
    # D0: snap
    t0 = time.time()
    snapped = snap_coefficients(netA, coeffA, m, C)
    D0_train, _ = fit_fixed_one(xtr, ytr, snapped)
    D0_test = rel_mse(ref.predict(xte, snapped, np.tile([[1.0, 0, 0, 0]], (U, 1))), yte)
    # D: ledger moves with A ≡ 1
    netD, coeffD, mseD, trD, evD = search(xtr, ytr, snapped, fit_fixed_one, max_steps=steps, moves='ledger')
    D_test = rel_mse(ref.predict(xte, netD, coeffD), yte)
    tD = time.time() - t0
    P(f"  {target:24s} K={K} U={U:2d} m={m:2d} | A  train {mseA/np.var(ytr[:,0]):.2e} test {A_test:.2e} ({len(trA)-1} steps, {tA:4.0f}s)"
      f" | D0 test {D0_test:.2e} | D train {mseD/np.var(ytr[:,0]):.2e} test {D_test:.2e} ({len(trD)-1} steps, {evD} evals, {tD:4.0f}s)")
    return dict(A=A_test, D0=D0_test, D=D_test, U=U, netD=netD, coeffD=coeffD)


def main():
    out = []
    P = lambda *a: (print(*a, flush=True), out.append(" ".join(str(s) for s in a)))
    P(__doc__)
    from rational_quaternion_sigma_product import formula
    P("== bound m sweep (the coefficient set and the frequency set both grow with m)")
    for target, K in (("T1 1/(1+4(log x−1)²)", 2), ("T2 sin(3x)/x", 3), ("T3 log(x₁+x₂)", 1)):
        for m in (K, 4, 6, 8, 12):
            if m < K:
                continue
            r = run(target, K, m, P=P)
            if m == 8:
                names = tuple(f"X{j}" for j in range(TARGETS[target][0])) + ("e",)
                P("      " + formula(r["netD"], r["coeffD"], names=names)[:400])
    P("\n== unit sweep at m = 8 (more modes: the lattice keeps up with the continuous readout or not)")
    for target, Ks in (("T1 1/(1+4(log x−1)²)", (1, 2, 3, 4)), ("T3 log(x₁+x₂)", (1, 2))):
        for K in Ks:
            run(target, K, 8, P=P)
    Path(__file__).resolve().parent.joinpath("results", "lattice_approximation.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
