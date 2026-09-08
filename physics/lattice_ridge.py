#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
Why the snap fails: the least-squares proposal cancels large coefficients, and a 1 % lattice rounding of each breaks
the cancellation.  Ridge on the proposal keeps |A| = O(1); then the lattice's own floor (the Farey spacing of n/o at
bound m, ~1/m² in log|A|) should show.   python physics/lattice_ridge.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import flexible_rational_quaternion_sigma_product as ref
import bounded_exponent as be
from lattice_coefficients import fit_fixed_one, search, snap_coefficients
from lattice_approximation import TARGETS, make_data, fourier_seed, rel_mse


def farey_floor(m):
    """largest gap between consecutive fractions n/o (n, o ≤ m) in [0, 1] — the worst rounding of log|A| or an angle."""
    fr = sorted({be.total_fraction(n, o) for n in range(m + 1) for o in range(1, m + 1) if n <= o})
    return max(float(b - a) for a, b in zip(fr, fr[1:])) / 2


def run(target, K, m, ridge, seed=0, n=2000, steps=8, P=print):
    n_inputs, f, (lo, hi) = TARGETS[target]
    xtr, ytr = make_data(seed, n, n_inputs, f, lo, hi)
    xte, yte = make_data(seed + 100, n, n_inputs, f, lo, hi)
    net0 = fourier_seed(n_inputs, K, m)
    netA0 = be.bounded_network([ref.ProductUnit([fc.copy() for fc in u.factors if fc.name != "A"]) for u in net0.units], bound=m)
    fit = lambda x, y, net: ref.exact_fit_mse(x, y, net, ridge)
    netA, coeffA, mseA, trA, _ = search(xtr, ytr, netA0, fit, max_steps=steps, moves='count')
    A_test = rel_mse(ref.predict(xte, netA, coeffA), yte)
    amax = float(np.max(np.linalg.norm(coeffA, axis=1)))
    snapped = snap_coefficients(netA, coeffA, m, n_inputs)
    U = len(net0.units)
    D0 = rel_mse(ref.predict(xte, snapped, np.tile([[1.0, 0, 0, 0]], (U, 1))), yte)
    netD, coeffD, mseD, trD, evD = search(xtr, ytr, snapped, fit_fixed_one, max_steps=steps, moves='ledger')
    D = rel_mse(ref.predict(xte, netD, coeffD), yte)
    P(f"  {target:22s} K={K} m={m:2d} ridge={ridge:6.0e} | A test {A_test:.2e} max|A| {amax:8.2f} | D0 {D0:.2e} | D {D:.2e} ({len(trD)-1} steps)")
    return A_test, D0, D


def main():
    out = []
    P = lambda *a: (print(*a, flush=True), out.append(" ".join(str(s) for s in a)))
    P(__doc__)
    P("Farey half-gap of the ledger at bound m (worst rounding of log|A|, in nats):  " +
      "  ".join(f"m={m}: {farey_floor(m):.3f}" for m in (4, 8, 12, 16)))
    P("(the normal matrix has entries ~N = 2000, so ridge 1 is a relative 5e-4)")
    for target, K, m in (("T1 1/(1+4(log x−1)²)", 2, 8), ("T1 1/(1+4(log x−1)²)", 4, 8), ("T1 1/(1+4(log x−1)²)", 4, 12),
                         ("T2 sin(3x)/x", 3, 8), ("T3 log(x₁+x₂)", 1, 8), ("T3 log(x₁+x₂)", 2, 8), ("T3 log(x₁+x₂)", 2, 12)):
        for ridge in (1e-10, 1.0, 10.0, 100.0):
            run(target, K, m, ridge, P=P)
        P("")
    Path(__file__).resolve().parent.joinpath("results", "lattice_ridge.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
