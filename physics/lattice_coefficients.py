#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
Coefficients on the lattice: a constant input node e and a leading factor e^{W} per unit instead of the readout A.

    Z = Σ_n A_n · P_n              (A_n: quaternion least squares, the reference)
    Z = Σ_n [e^{W_n}] · P_n        (A_n ≡ 1; W_n a bounded rational-quaternion ledger, searched like any exponent)
    Z = Σ_n a_n [e^{W_n}] · P_n    (middle: the direction on the lattice, the magnitude a_n ∈ ℝ by least squares)

Teacher (2 data inputs + the constant e, bound m = 3 = number of input nodes):

    y = e^{1/2 + (2/3) i} · X₀^{1/2}  +  e^{−1 + (1/3) j} · X₁²

Every search starts from X₀ · X₁ with A = 1 (the coefficient ledger balanced: 1/1 − 1/1 = 0 on every component, so
that every ±1 move is live — the empty ledger 0/0 is a plateau, shown as mode B0).   python physics/lattice_coefficients.py
"""
from __future__ import annotations
import math
import sys
import time
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import flexible_rational_quaternion_sigma_product as ref
import bounded_exponent as be
from rational_quaternion_sigma_product import formula

E = np.array([math.e, 0, 0, 0])


def with_constant(x: np.ndarray, c: np.ndarray = E) -> np.ndarray:
    """append the constant input node as the last column of x[N, I, 4]."""
    return np.concatenate([x, np.tile(c[None, None, :], (x.shape[0], 1, 1))], axis=1)


def balanced_zero(bound: int) -> be.BoundedRationalQuaternionState:
    """W = 0 with every ±1 move live: 1/1 − 1/1 on all four components."""
    st = be.BoundedRationalQuaternionState.zero(bound)
    for mu in range(4):
        st.set(mu, 0, 1, 1).set(mu, 1, 1, 1)
    return st


def coefficient_factor(const_index: int, state, name="A"):
    return ref.Factor(const_index, exponent_layers=[ref.ExponentLayer(state, 'L', name)], name=name)


# ---------------------------------------------------------------- the three readouts
def fit_quaternion_ls(x, y, net, ridge=1e-10):
    return ref.exact_fit_mse(x, y, net, ridge)


def fit_fixed_one(x, y, net, ridge=None):
    coeff = np.tile([[1.0, 0, 0, 0]], (len(net.units), 1))
    pred = ref.predict(x, net, coeff)
    return float(np.mean((pred - y) ** 2)), coeff


def fit_real_scalar(x, y, net, ridge=1e-10):
    phi, _ = ref.features_numpy(x, net)          # [N,1,M,4]
    D = phi[:, 0].transpose(0, 2, 1).reshape(-1, phi.shape[2])   # rows: (sample, component); cols: unit
    a = np.linalg.solve(D.T @ D + ridge * np.eye(D.shape[1]), D.T @ y.reshape(-1))
    coeff = np.stack([a, 0 * a, 0 * a, 0 * a], 1)
    pred = ref.predict(x, net, coeff)
    return float(np.mean((pred - y) ** 2)), coeff


# ---------------------------------------------------------------- the reference's search with a pluggable readout
def search(x, y, net0, fit, *, max_steps=40, top_k_blocks=8, move_filter=None, min_improvement=1e-14, moves='count'):
    """moves = 'count' (the reference's ±1 count steps) or 'ledger' (any ledger of a selected channel)."""
    net = net0.copy()
    mse, coeff = fit(x, y, net)
    evaluated = 0
    trace = [(0, mse, None)]
    for it in range(max_steps):
        keys, grads = ref.effective_w_chunk_gradients(x, y, net, coeff, 8)
        sensor = ref.gradient_consensus_sensor(grads)
        if moves == 'ledger':
            proposals = be.propose_ledger_moves(net, keys, sensor, top_k_blocks=top_k_blocks)
        else:
            proposals = ref.propose_moves(net, keys, sensor, top_k_blocks=top_k_blocks, max_count=None)
        if move_filter is not None:
            proposals = [m for m in proposals if move_filter(net, m)]
        seen = {ref._network_key(net)}
        best = None
        for mv in proposals:
            cand = be.apply_move(net, mv)
            k = ref._network_key(cand)
            if k in seen:
                continue
            seen.add(k)
            cm, cc = fit(x, y, cand)
            evaluated += 1
            if best is None or cm < best[0]:
                best = (cm, mv, cand, cc)
        if best is None or best[0] >= mse - min_improvement:
            break
        mse, _, net, coeff = best
        trace.append((it + 1, mse, best[1]))
    return net, coeff, mse, trace, evaluated


def is_coefficient_layer(net, key):
    ui, fi, li = key
    return net.units[ui].factors[fi].name == "A"


def freeze_scalar_of_coefficient(net, mv):
    nat = mv.natural if isinstance(mv, ref.NaturalSideFlipMove) else mv
    if isinstance(nat, (ref.NaturalMove, be.LedgerMove)) and is_coefficient_layer(net, nat.key) and nat.component == 0:
        return False
    return True


# ---------------------------------------------------------------- LS proposes, the lattice decides
def nearest_ledger(v: float, m: int):
    """the ledger (sign, n, o) with n/o closest to v (n, o ≤ m; n/0 = 0)."""
    best = None
    for n in range(m + 1):
        for o in range(m + 1):
            fr = float(be.total_fraction(n, o))
            for sign, val in ((0, fr), (1, -fr)):
                d = abs(val - v)
                if best is None or d < best[0]:
                    best = (d, sign, n, o)
    return best[1:]


def snap_coefficients(net_ls, coeff, m, const_index):
    """Log A_n → the nearest bounded ledger on each component → a leading factor e^{W_n} per unit, A ≡ 1."""
    units = []
    for unit, A in zip(net_ls.units, coeff):
        L = ref.qlog(A[None, :])[0]                                   # Log A (principal branch)
        st = be.BoundedRationalQuaternionState.zero(m)
        for mu in range(4):
            sign, n, o = nearest_ledger(float(L[mu]), m)
            st.set(mu, sign, n, o)
        units.append(ref.ProductUnit([coefficient_factor(const_index, st)] + [f.copy() for f in unit.factors],
                                     unit.right_basis, unit.name))
    return be.bounded_network(units, bound=m)


def search_ls_then_snap(x, y, net0, **kw):
    """mode D: the reference search with the LS readout, then the coefficients snapped to the lattice, then ledger
    moves on the whole network with A ≡ 1 (the snap is a proposal; the exact evaluation decides)."""
    net, coeff, mse, trace, ev = search(x, y, net0, fit_quaternion_ls, **kw)
    snapped = snap_coefficients(net, coeff, M, C)
    mse_snap, _ = fit_fixed_one(x, y, snapped)
    net2, coeff2, mse2, trace2, ev2 = search(x, y, snapped, fit_fixed_one, moves='ledger', **kw)
    trace = trace + [(len(trace), mse_snap, 'snap')] + [(len(trace) + 1 + i, m_, mv) for i, m_, mv in trace2[1:]]
    return net2, coeff2, mse2, trace, ev + ev2


# ---------------------------------------------------------------- teacher and seeds
M = 3
C = 2                                   # index of the constant input node
TEACHER_A = [((1, 2), (1, (2, 3))), (((-1), 1), (2, (1, 3)))]    # for the docstring only


def teacher_net():
    a1 = be.bounded_state(M, 1, 2).set(1, 0, 2, 3)                # 1/2 + (2/3) i
    a2 = be.bounded_state(M, 1, 1, sign=1).set(2, 0, 1, 3)        # −1 + (1/3) j
    return be.bounded_network([
        ref.ProductUnit([coefficient_factor(C, a1), be.make_bounded_power_factor(0, M, 1, 2)]),
        ref.ProductUnit([coefficient_factor(C, a2), be.make_bounded_power_factor(1, M, 2, 1)]),
    ], bound=M)


def seed_net(coefficients: str):
    """coefficients: 'none' (readout only), 'balanced' (e^{0} with live moves), 'empty' (0/0 plateau)."""
    units = []
    for j in range(2):
        f = [be.make_bounded_power_factor(j, M, 1, 1)]
        if coefficients == 'balanced':
            f.insert(0, coefficient_factor(C, balanced_zero(M)))
        elif coefficients == 'empty':
            f.insert(0, coefficient_factor(C, be.BoundedRationalQuaternionState.zero(M)))
        units.append(ref.ProductUnit(f))
    return be.bounded_network(units, bound=M)


def data(seed, n=400):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2, 4)) * 0.4
    x[:, :, 0] += 1.5
    x = with_constant(x)
    y = ref.predict(x, teacher_net(), np.tile([[1.0, 0, 0, 0]], (2, 1)))
    return x, y


def exponents(net):
    return [(f.name or f"X{f.input_index}", tuple(l.state.fractions()) if f.exponent_layers else None)
            for u in net.units for f in u.factors for l in (f.exponent_layers or [None])]


def main():
    out = []
    P = lambda *a: (print(*a), out.append(" ".join(str(s) for s in a)))
    P(__doc__)
    P(f"teacher: {formula(teacher_net(), np.tile([[1.0, 0, 0, 0]], (2, 1)), names=('X₀', 'X₁', 'e'))}")
    A1 = math.exp(0.5) * np.array([math.cos(2 / 3), math.sin(2 / 3), 0, 0])
    A2 = math.exp(-1) * np.array([math.cos(1 / 3), 0, math.sin(1 / 3), 0])
    P(f"         A₁ = {np.round(A1, 6)}   A₂ = {np.round(A2, 6)}\n")
    modes = [
        ("A  readout LS (reference)",        'none',     fit_quaternion_ls, None, 'count'),
        ("B  A ≡ 1, coefficient on lattice", 'balanced', fit_fixed_one,     None, 'count'),
        ("B0 same, empty ledger seed",       'empty',    fit_fixed_one,     None, 'count'),
        ("C  direction lattice + real LS",   'balanced', fit_real_scalar,   freeze_scalar_of_coefficient, 'count'),
        ("A' readout LS, ledger moves",      'none',     fit_quaternion_ls, None, 'ledger'),
        ("B' A ≡ 1, ledger moves",           'balanced', fit_fixed_one,     None, 'ledger'),
        ("B0' A ≡ 1, ledger moves, empty seed", 'empty', fit_fixed_one,     None, 'ledger'),
        ("C' direction lattice + real LS, ledger moves", 'balanced', fit_real_scalar, freeze_scalar_of_coefficient, 'ledger'),
    ]
    modes.append(("D  LS proposes → snap Log A to the lattice → ledger moves with A ≡ 1", 'none', None, None, 'snap'))
    seeds = range(8)
    for label, coef, fit, filt, mv in modes:
        P(f"== {label}")
        wins = 0
        rows = []
        for s in seeds:
            x, y = data(s)
            t0 = time.time()
            if mv == 'snap':
                net, coeff, mse, trace, ev = search_ls_then_snap(x, y, seed_net(coef))
            else:
                net, coeff, mse, trace, ev = search(x, y, seed_net(coef), fit, move_filter=filt, moves=mv)
            dt = time.time() - t0
            xw = [fr for name, fr in exponents(net) if name != "A"]
            ok_x = xw == [(Fraction(1, 2), 0, 0, 0), (Fraction(2), 0, 0, 0)]
            ok = ok_x and mse < 1e-20
            if mv == 'snap':
                cw = [fr for name, fr in exponents(net) if name == "A"]
                ok = ok and cw == [(Fraction(1, 2), Fraction(2, 3), 0, 0), (Fraction(-1), 0, Fraction(1, 3), 0)]
            wins += ok
            rows.append((s, mse, len(trace) - 1, ev, dt, ok))
            P(f"  seed {s}: mse {mse:.2e}  steps {len(trace)-1:2d}  evaluated {ev:4d}  {dt:5.1f}s  "
              f"{'exact' if ok else ('exponents ok' if ok_x else 'exponents wrong')}")
            if s == 0:
                P("    " + formula(net, coeff, names=('X₀', 'X₁', 'e'), snap=1e-6))
                P("    path: " + " → ".join(f"{m:.1e}" for _, m, _ in trace))
        P(f"  {wins}/{len(seeds)} exact;  median steps {np.median([r[2] for r in rows]):.0f}, "
          f"median evaluations {np.median([r[3] for r in rows]):.0f}, median {np.median([r[4] for r in rows]):.1f}s\n")
    Path(__file__).resolve().parent.joinpath("results", "lattice_coefficients.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
