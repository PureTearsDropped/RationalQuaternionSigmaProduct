#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
Discrete first, float after: take the structure D found with every coefficient on the lattice, drop the lattice
coefficients, and solve A in float by least squares on that structure.  Compared with A (the reference: structure
searched with the float readout).   python physics/lattice_refit.py
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


def strip_coefficients(net):
    return be.bounded_network([ref.ProductUnit([f.copy() for f in u.factors if f.name != "A"], u.right_basis)
                               for u in net.units], bound=be.max_count_of(net))


def refit(net, xtr, ytr, xte, yte, ridge):
    bare = strip_coefficients(net)
    coeff = ref.fit_readout(xtr, ytr, bare, ridge)
    return rel_mse(ref.predict(xte, bare, coeff), yte), float(np.max(np.linalg.norm(coeff, axis=1)))


def run(target, K, m, ridge_prop, seed=0, n=2000, steps=8, P=print):
    n_inputs, f, (lo, hi) = TARGETS[target]
    xtr, ytr = make_data(seed, n, n_inputs, f, lo, hi)
    xte, yte = make_data(seed + 100, n, n_inputs, f, lo, hi)
    net0 = strip_coefficients(fourier_seed(n_inputs, K, m))
    fit = lambda x, y, net: ref.exact_fit_mse(x, y, net, ridge_prop)
    netA, coeffA, *_ = search(xtr, ytr, net0, fit, max_steps=steps, moves='count')
    A_r0, _ = refit(netA, xtr, ytr, xte, yte, 1e-10)
    A_r1, _ = refit(netA, xtr, ytr, xte, yte, 1.0)
    snapped = snap_coefficients(netA, coeffA, m, n_inputs)
    netD, coeffD, *_ = search(xtr, ytr, snapped, fit_fixed_one, max_steps=steps, moves='ledger')
    D = rel_mse(ref.predict(xte, netD, coeffD), yte)
    D_r0, a0 = refit(netD, xtr, ytr, xte, yte, 1e-10)
    D_r1, a1 = refit(netD, xtr, ytr, xte, yte, 1.0)
    same = ref._network_key(strip_coefficients(netD)) == ref._network_key(netA)
    P(f"  {target:22s} K={K} m={m:2d} proposal ridge {ridge_prop:5.0e} | A structure: LS {A_r0:.2e} ridge1 {A_r1:.2e}"
      f" | D lattice {D:.2e} → D structure refit: LS {D_r0:.2e} (max|A| {a0:6.2f}) ridge1 {D_r1:.2e}"
      f" | structures {'identical' if same else 'differ'}")


def main():
    out = []
    P = lambda *a: (print(*a, flush=True), out.append(" ".join(str(s) for s in a)))
    P(__doc__)
    for target, K, m in (("T1 1/(1+4(log x−1)²)", 2, 8), ("T1 1/(1+4(log x−1)²)", 4, 8), ("T1 1/(1+4(log x−1)²)", 4, 12),
                         ("T3 log(x₁+x₂)", 1, 8), ("T3 log(x₁+x₂)", 2, 12)):
        for ridge_prop in (1e-10, 1.0):
            run(target, K, m, ridge_prop, P=P)
    Path(__file__).resolve().parent.joinpath("results", "lattice_refit.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
