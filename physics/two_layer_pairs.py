#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""(i′) the all-lattice two-layer search with PAIR moves: two scalar channels (any two of the inner coefficients,
the inner exponents, the outer exponent) changed together, A₂ by LS per candidate.  The single-channel greedy of
two_layer_lattice.py stops at a coupled local minimum (v = 2/3, W_a − W_b = 1/3).   python physics/two_layer_pairs.py"""
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
from two_layer_lattice import data, seed, forward_fit, exponents_of, coef_ratio_lattice, M


def scalar_channel_moves(net, which):
    keys = [(ui, fi, li) for ui, fi, li, _ in net.iter_layers()]
    out = []
    for key in keys:
        st = be.ref._get_layer(net, key).state
        for sign in range(2):
            cur = st.fraction(0, sign)
            seen = {cur}
            for n in range(M + 1):
                for o in range(M + 1):
                    fr = be.total_fraction(n, o)
                    if fr in seen:
                        continue
                    seen.add(fr)
                    out.append((which, key, sign, be.LedgerMove(key, 0, sign, n, o)))
    return out


def apply(n1, n2, mv):
    which, key, sign, m = mv
    return (be.apply_move(n1, m), n2) if which == 1 else (n1, be.apply_move(n2, m))


def search_pairs(x, y, n1, n2, max_steps=8):
    mse, A2 = forward_fit(x, y, n1, n2)
    steps = ev = 0
    for _ in range(max_steps):
        singles = scalar_channel_moves(n1, 1) + scalar_channel_moves(n2, 2)
        best = None
        cands = [(a,) for a in singles] + [(a, b) for a, b in itertools.combinations(singles, 2) if (a[0], a[1], a[2]) != (b[0], b[1], b[2])]
        for combo in cands:
            c1, c2 = n1, n2
            for mv in combo:
                c1, c2 = apply(c1, c2, mv)
            m_, a_ = forward_fit(x, y, c1, c2)
            ev += 1
            if best is None or m_ < best[0]:
                best = (m_, c1, c2, a_, len(combo))
        if best is None or best[0] >= mse * (1 - 1e-6) - 1e-30:
            break
        mse, n1, n2, A2, k = best
        steps += 1
    return mse, n1, n2, A2, steps, ev


def main():
    out = []
    P = lambda *a: (print(*a, flush=True), out.append(" ".join(str(s) for s in a)))
    P(__doc__)
    TRUE_X = (Fraction(1), Fraction(2), Fraction(1, 2))
    wins = 0
    for s in range(8):
        x, y = data(s)
        t0 = time.time()
        mse, n1, n2, A2, steps, ev = search_pairs(x, y, *seed(True))
        ok = exponents_of(n1, n2) == TRUE_X and coef_ratio_lattice(n1) == Fraction(3, 2) and mse / np.var(y) < 1e-20
        wins += ok
        P(f"  seed {s}: rel mse {mse/np.var(y):.2e}  steps {steps}  evals {ev}  {time.time()-t0:5.1f}s  "
          f"exponents {tuple(str(f) for f in exponents_of(n1, n2))}  W_a−W_b {coef_ratio_lattice(n1)}  {'exact' if ok else 'wrong'}")
    P(f"  {wins}/8")
    Path(__file__).resolve().parent.joinpath("results", "two_layer_pairs.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
