#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""(v) shortlist per site, brute force across sites.  Sites = the five scalar ledgers (a, b, w₁, w₂, v).  For each
site alone, every value ±n/o (n, o ≤ m) is tried with the others fixed and the best k are kept; then all k⁵
combinations are evaluated exactly (A₂ by LS) and the best is accepted; repeat until nothing improves.
The single-site greedy and the pair moves stop at (w₂ = 1, v = 2/3, a − b = 1/3); this asks whether the true
values are on the per-site shortlists and whether their combination wins.   python physics/two_layer_shortlist.py"""
from __future__ import annotations
import itertools
import sys
import time
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import bounded_exponent as be
from two_layer_lattice import data, forward_fit, layer1, layer2, exponents_of, coef_ratio_lattice, M

VALUES = sorted({s * be.total_fraction(n, o) for n in range(M + 1) for o in range(M + 1) for s in (1, -1)})


def build(p):
    """p = (a, b, w1, w2, v) as Fractions → the two lattice networks."""
    def st(f):
        return be.bounded_state(M, abs(f.numerator), f.denominator, sign=int(f < 0))
    n1 = layer1(st(p[0]), st(p[1]), (0, 0), (0, 0))
    n1.units[0].factors[1].exponent_layers[0].state = st(p[2])
    n1.units[1].factors[1].exponent_layers[0].state = st(p[3])
    n2 = layer2((0, 0))
    n2.units[0].factors[0].exponent_layers[0].state = st(p[4])
    return n1, n2


def mse_of(x, y, p):
    return forward_fit(x, y, *build(p))[0]


def search(x, y, p0, k=3, max_rounds=6):
    p = list(p0)
    mse = mse_of(x, y, p)
    ev, rounds, log = 1, 0, []
    for _ in range(max_rounds):
        shortlists = []
        for site in range(5):
            scored = []
            for val in VALUES:
                q = list(p); q[site] = val
                scored.append((mse_of(x, y, q), val)); ev += 1
            scored.sort(key=lambda t: t[0])
            shortlists.append([v for _, v in scored[:k]])
        best = None
        for combo in itertools.product(*shortlists):
            m_ = mse_of(x, y, list(combo)); ev += 1
            if best is None or m_ < best[0]:
                best = (m_, list(combo))
        log.append((shortlists, best))
        if best[0] >= mse * (1 - 1e-6) - 1e-30:
            break
        mse, p = best
        rounds += 1
    return p, mse, rounds, ev, log


def main():
    out = []
    P = lambda *a: (print(*a, flush=True), out.append(" ".join(str(s) for s in a)))
    P(__doc__)
    TRUE = [Fraction(1, 2), Fraction(-1), Fraction(1), Fraction(2), Fraction(1, 2)]
    seed0 = [Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(1)]
    for k in (2, 3, 4):
        P(f"== shortlist k = {k} per site, {k}^5 = {k**5} combinations per round")
        wins = 0
        for s in range(8):
            x, y = data(s)
            t0 = time.time()
            p, mse, rounds, ev, log = search(x, y, seed0, k=k)
            ok = (p[2], p[3], p[4]) == (TRUE[2], TRUE[3], TRUE[4]) and p[0] - p[1] == Fraction(3, 2) and mse / np.var(y) < 1e-20
            wins += ok
            on_list = [TRUE[i] in log[0][0][i] for i in range(5)]
            P(f"  seed {s}: rel mse {mse/np.var(y):.2e}  rounds {rounds}  evals {ev:5d}  {time.time()-t0:5.1f}s  "
              f"p = ({', '.join(str(v) for v in p)})  true on round-1 shortlists {on_list}  {'exact' if ok else 'wrong'}")
            if s == 0:
                P("    round-1 shortlists: " + " | ".join(",".join(str(v) for v in sl) for sl in log[0][0]))
        P(f"  {wins}/8\n")
    Path(__file__).resolve().parent.joinpath("results", "two_layer_shortlist.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
