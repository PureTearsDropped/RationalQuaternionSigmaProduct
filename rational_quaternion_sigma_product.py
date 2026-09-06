#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""rational_quaternion_sigma_product — RationalQuaternionSigmaProduct: the flexible rational-quaternion Σ-Product network on
total arithmetic (the `Tot` of total-arith-cuda ≥ v1.2.0, which must be importable: a checkout next to this repository, the
environment variable TOTAL_ARITH_CUDA, or sys.path).

The structure (inputs → per-factor reciprocal → rational-quaternion exponent layers Pow_L / Pow_R → ordered Hamilton
product → fixed right basis → quaternion-linear Σ readout, with natural-number prime-valuation exponent states and the
gradient-sensor + exact discrete search) is the one of `flexible_rational_quaternion_sigma_product.py` (the reference,
NumPy/IEEE, imported here as `ref` — its dataclasses, its move generator and its consensus sensor are reused verbatim).
This module replaces its arithmetic by the `Tot` (float32 value + uint8 flag) of total-arith-cuda:

    Hamilton product        group_mul with the Cayley–Dickson wiring table cd4 (i·j = k, j·k = i, k·i = j), float64
                            accumulation, one saturation, the pattern flag rules of total-arith-cuda
    X⁻¹                     conj(X)/|X|² through tot_div: a/0 = 0 ⟹ 0⁻¹ = 0 (Moore–Penrose, the reference clamps |X|² ≥ 1e-12)
    Log₀(X)                 L₀(|X|) + (v/|v|)·Arg₀(a, |v|):  L₀(0) = 0, Arg₀(0, 0) = 0 (the reserved words), a true-zero
                            vector part has direction 0 — except on the negative real axis, where Log₀(−r) = log r + iπ by
                            definition (ℂ embedded along i; the reference sets the vector part to 0 there, so Exp∘Log(−1) = 1)
    Exp(U)                  e^{U₀} saturates (MAX⟦≥⟧ / MIN⟦≤⟧, never 0), the rotation (cos|v|, sinc|v|·v) is exact when v is
                            exact and carries ⟦no bound, direction unknown⟧ when any component of v is flagged (periodic)
    Pow_L / Pow_R           Exp(W·Log₀ X) / Exp((Log₀ X)·W) — so Pow(0, W) = 1: a zero input drops out of the product
    Σ readout               A_n solved by ridge least squares on the USABLE samples only (every flag 0 or ⟦≤⟧ alone);
                            the prediction is built with the same Tot operations, so its flags say which outputs are numbers

An input enters through the customs of `Tot(x)`: NaN → (0, unknown), |x| > f32 MAX → MAX⟦≥⟧, subnormal → MIN⟦≤⟧ = ε.
Nothing downstream ever produces NaN or Inf; a sample the arithmetic cannot vouch for is excluded from the fit and from the
search's loss, and the search never sees a silently clipped number (the reference clips exp's argument to ±80 without a mark).

Shapes follow the reference: inputs [N, I, 4] or [N, G, I, 4] (grouped: shared structure, per-group readout), targets [N, 4]
or [N, G, 4].  Values are float32 (as in `Tot`), so the fit floor is ~1e-13 in MSE, not the 1e-30 of float64.

Known limit (inherited from total-arith-cuda: ordinary float rounding is not flagged, only totalization events are): a
component that is mathematically 0 after a rotation (cos π/2) is a rounding residue of ~1e-7, and when the amplitude
saturated at MAX⟦≥⟧ that residue is shown as ~1e31⟦≥⟧ — a claim on a component whose truth is 0.  The sample is flagged
and excluded either way; the flag is right about the sample, not about that component's magnitude.  The complex unit
shares this.  Exact rationals (1/3, 2/5, …) are rounded once to float32 in W, as they are rounded to float64 in the reference.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from cuda_total import (Tot, GE, LE, SUNK, MAX, MIN, F32, _sat, tot_mul, tot_add, tot_div, _true_zero, _danger_zero,
                            wiring_tensor, group_mul)
except ImportError:                                                  # total-arith-cuda: TOTAL_ARITH_CUDA or ../total-arith-cuda
    sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parent.parent / 'total-arith-cuda')))
    from cuda_total import (Tot, GE, LE, SUNK, MAX, MIN, F32, _sat, tot_mul, tot_add, tot_div, _true_zero, _danger_zero,
                            wiring_tensor, group_mul)
import flexible_rational_quaternion_sigma_product as ref                                                              # noqa: E402

NOB = GE | LE
UNKNOWN = GE | LE | SUNK
LOG_MIN = math.log(MIN)


def _where_flag(cond, value, flag):
    return torch.where(cond, torch.full_like(flag, value), flag)


# ---------------------------------------------------------------- the scalar reserved-word functions
# (as in ComplexSigmaProductUnit v1.0.0 `complex_sigma_product_unit.py`, unchanged: L₀ of a magnitude, Arg₀ of a pair,
#  the saturating exponential — the same three that build the complex unit build the quaternion one on |q|, (a, |v|), U₀)
def log0(r: Tot) -> Tot:
    """L₀ of a magnitude (r ≥ 0).  L₀(0) = 0 exact and unflagged (the reserved word); ε = (MIN, ⟦≤⟧) → log MIN ⟦≥⟧;
    otherwise log r with ScalarTot's rule — a bound survives only where it is provable (⟦≥⟧ above 1 or ⟦≤⟧ below 1 →
    ⟦≥⟧ on |log|); sign unknown → unknown."""
    v = r.val.double()
    z = v == 0
    safe = torch.where(z, torch.ones_like(v), v)
    u = torch.where(z, torch.zeros_like(v), torch.log(safe))
    f = r.flag
    ge = (f & GE) > 0; le = (f & LE) > 0; sunk = (f & SUNK) > 0
    out = torch.zeros_like(f)
    out = _where_flag(sunk | (ge & le), UNKNOWN, out)
    prov = ~sunk & ~(ge & le) & ((ge & (v > 1)) | (le & (v < 1)))
    out = _where_flag(prov, GE, out)
    out = _where_flag(~sunk & ~(ge & le) & ((ge & (v <= 1)) | (le & (v >= 1))) & ~z, NOB | SUNK, out)
    out = _where_flag(z & ge, UNKNOWN, out)
    out = _where_flag(z & ~ge, 0, out)
    val, sflag = _sat(u, v.device)
    return Tot(val, sflag | out)


def arg0(a: Tot, b: Tot) -> Tot:
    """Arg₀(a, b): atan2(b, a) ∈ (−π, π], and 0 at (0, 0).  The direction is claimed exact only when both components
    are exact, or the flagged one is the only nonzero one."""
    av, bv = a.val.double(), b.val.double()
    z = (av == 0) & (bv == 0)
    a_s = torch.where(z, torch.ones_like(av), av)
    v = torch.atan2(bv, a_s)
    fa, fb = a.flag, b.flag
    sunk = ((fa | fb) & SUNK) > 0
    bnd_a = (fa & NOB) > 0; bnd_b = (fb & NOB) > 0
    tz_a = _true_zero(a.val, fa); tz_b = _true_zero(b.val, fb)
    moves = (bnd_a & ~tz_b) | (bnd_b & ~tz_a)
    danger = _danger_zero(a.val, fa) | _danger_zero(b.val, fb)
    out = torch.zeros_like(fa)
    out = _where_flag(sunk | moves | danger, NOB | SUNK, out)
    out = _where_flag(z & ~danger, 0, out)
    val, sflag = _sat(v, av.device)
    return Tot(val, sflag | out)


def tot_exp(U: Tot) -> Tot:
    """e^U with the amplitude saturated: overflow → MAX⟦≥⟧, underflow → MIN⟦≤⟧, never 0 (e^U > 0).
    Flags: ScalarTot's exp rule — the sign of e^U is certain, a bound on U flips with the sign of U."""
    v = U.val.double()
    raw = torch.exp(v)
    val, sflag = _sat(raw, v.device)
    under = raw == 0
    val = torch.where(under, torch.full_like(val, MIN), val)
    sflag = sflag | under.to(torch.uint8) * LE
    f = U.flag
    ge = (f & GE) > 0; le = (f & LE) > 0; sunk = (f & SUNK) > 0
    out = torch.zeros_like(f)
    out = _where_flag(sunk | (ge & le), NOB, out)
    pos = v > 0; zero = v == 0
    out = _where_flag(~sunk & ge & ~le & pos, GE, out); out = _where_flag(~sunk & ge & ~le & ~pos & ~zero, LE, out)
    out = _where_flag(~sunk & le & ~ge & pos, LE, out); out = _where_flag(~sunk & le & ~ge & ~pos & ~zero, GE, out)
    out = _where_flag(~sunk & ge & ~le & zero, NOB, out)
    return Tot(val, sflag | out)

# the structure is the reference's — re-exported so a user of this module needs one import
NaturalQuaternionState = ref.NaturalQuaternionState
ExponentLayer = ref.ExponentLayer
Factor = ref.Factor
ProductUnit = ref.ProductUnit
SigmaProductNetwork = ref.SigmaProductNetwork
make_scalar_plus_one_state = ref.make_scalar_plus_one_state
make_scalar_minus_one_state = ref.make_scalar_minus_one_state
make_passthrough_factor = ref.make_passthrough_factor
make_power_factor = ref.make_power_factor
make_alternating_linear_seed = ref.make_alternating_linear_seed

F64 = torch.float64
NEG_REAL_AXIS = 0            # Log₀(−r) = log r + π·e_{1+NEG_REAL_AXIS}: the i axis.  None → vector part 0 (the reference)


# ---------------------------------------------------------------- quaternion Tot algebra (a Tot whose last axis is 4)
_TABLES: dict = {}


def hamilton_table(device) -> torch.Tensor:
    """the wiring table T[k, i, j] of ℍ (Cayley–Dickson, M = 4), one per device."""
    key = str(device)
    if key not in _TABLES:
        _TABLES[key] = wiring_tensor('cd', 4, device=device)
    return _TABLES[key]


def comp(q: Tot, i: int) -> Tot:
    return Tot(q.val[..., i], q.flag[..., i])


def stack(parts) -> Tot:
    return Tot(torch.stack([p.val for p in parts], -1), torch.stack([p.flag for p in parts], -1))


def bcast(t: Tot, shape) -> Tot:
    return Tot(t.val.expand(shape), t.flag.expand(shape))


def const_quat(w, like: Tot) -> Tot:
    """a constant quaternion (exact, unflagged — rounded once to float32) broadcast to the shape of `like`."""
    if torch.is_tensor(w):
        t = w.to(device=like.device, dtype=F64)
    else:
        t = torch.as_tensor(np.asarray(w, dtype=np.float64), dtype=F64, device=like.device)
    c = Tot(t)
    shape = like.val.shape
    return Tot(c.val.expand(shape), c.flag.expand(shape))


def qmul_t(a: Tot, b: Tot) -> Tot:
    """the Hamilton product a·b as a wiring product (the flags follow total-arith-cuda's pattern rules)."""
    return group_mul(hamilton_table(a.device), a, b)


def qconj_t(q: Tot) -> Tot:
    """conj: negation keeps every flag (a bound is on |·|, SUNK is sign-free)."""
    sgn = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=F32, device=q.device)
    return Tot(q.val * sgn, q.flag)


def hypot_t(parts) -> Tot:
    """√Σ c² over a list of Tots (same shape): monotone in every |c| ⟹ ⟦≥⟧ / ⟦≤⟧ survive as an OR, both → no bound;
    SUNK does not touch a magnitude.  A shown 0 is a true 0 unless a component is a dangerous zero (then ⟦≥⟧)."""
    vals = [p.val.double() for p in parts]
    sq = vals[0] * vals[0]
    for v in vals[1:]:
        sq = sq + v * v
    z = sq == 0
    r = torch.where(z, torch.zeros_like(sq), torch.sqrt(torch.where(z, torch.ones_like(sq), sq)))
    fl = parts[0].flag & NOB
    for p in parts[1:]:
        fl = fl | (p.flag & NOB)
    ge = (fl & GE) > 0
    le = (fl & LE) > 0
    out = torch.zeros_like(fl)
    out = torch.where(ge & ~le, torch.full_like(out, GE), out)
    out = torch.where(le & ~ge, torch.full_like(out, LE), out)
    out = torch.where(ge & le, torch.full_like(out, NOB), out)
    val, sflag = _sat(r, r.device)
    return Tot(val, sflag | out)


def qabs_t(q: Tot) -> Tot:
    return hypot_t([comp(q, i) for i in range(4)])


def qinv_t(q: Tot) -> Tot:
    """X⁻¹ = conj(X)/|X|² through tot_div: 0⁻¹ = 0 (a/0 = 0).  Bounds on the input make the quotient boundless (tot_div)."""
    n = qabs_t(q)
    n2 = tot_mul(n, n)
    c = qconj_t(q)
    return stack([tot_div(comp(c, i), n2) for i in range(4)])


def _direction(vparts, vr: Tot, a: Tot, neg_axis):
    """u = v/|v| for the vector part v = (v₁, v₂, v₃) with |v| = vr.  Exact when the direction is provable: a component
    that is a true zero is exactly 0; if exactly one component is nonzero its direction is ±axis (SUNK if its sign is);
    otherwise a bound or SUNK on any nonzero component makes the direction of every nonzero component unknown; a dangerous
    zero anywhere makes the whole direction unknown.  |v| = 0 (true zero): u = 0 — the reserved word — except on the
    negative real axis (a < 0) where u = e_{neg_axis} by definition."""
    vv = [p.val.double() for p in vparts]
    vrv = vr.val.double()
    z = vrv == 0
    safe = torch.where(z, torch.ones_like(vrv), vrv)
    tz = [_true_zero(p.val, p.flag) for p in vparts]
    dz = [_danger_zero(p.val, p.flag) for p in vparts]
    danger = dz[0] | dz[1] | dz[2]
    nonzero = [~t & ~d for t, d in zip(tz, dz)]
    n_nz = sum(nz.to(torch.int8) for nz in nonzero)
    flagged_nz = None
    for p, nz in zip(vparts, nonzero):
        f = (p.flag & (NOB | SUNK)) > 0
        flagged_nz = (f & nz) if flagged_nz is None else (flagged_nz | (f & nz))
    outs = []
    on_neg_axis = z & ~danger & (a.val.double() < 0) & (neg_axis is not None)
    for i, (p, x) in enumerate(zip(vparts, vv)):
        u = torch.where(z, torch.zeros_like(x), x / safe)
        if neg_axis is not None and i == neg_axis:
            u = torch.where(on_neg_axis, torch.ones_like(u), u)
        f = torch.zeros_like(p.flag)
        single = (n_nz == 1) & nonzero[i]
        f = torch.where(single & ((p.flag & SUNK) > 0), torch.full_like(f, SUNK), f)
        many = (n_nz >= 2) & nonzero[i] & flagged_nz
        f = torch.where(many, torch.full_like(f, NOB | SUNK), f)
        f = torch.where(danger, torch.full_like(f, NOB | SUNK), f)
        val, sflag = _sat(u, u.device)
        outs.append(Tot(val, sflag | f))
    return outs


def qlog0_t(q: Tot, neg_axis=NEG_REAL_AXIS) -> Tot:
    """Log₀(a + v) = L₀(|q|) + (v/|v|)·Arg₀(a, |v|) with the reserved words L₀(0) = 0, Arg₀(0, 0) = 0, direction of 0 = 0;
    ε = MIN⟦≤⟧ → L₀ = log MIN ⟦≥⟧ and the direction ε carries; Log₀(−r) = log r + π·e_{neg_axis} (definition)."""
    a = comp(q, 0)
    vparts = [comp(q, i) for i in (1, 2, 3)]
    n = qabs_t(q)
    vr = hypot_t(vparts)
    u0 = log0(n)
    theta = arg0(a, vr)                                    # atan2(|v|, a) ∈ [0, π] with the pair's flag rule
    u = _direction(vparts, vr, a, neg_axis)
    return stack([u0] + [tot_mul(theta, ui) for ui in u])


def qexp_t(q: Tot) -> Tot:
    """Exp(a + v) = e^a (cos|v| + (v/|v|) sin|v|): e^a by tot_exp (saturates, never 0); the rotation is exact for an exact
    v, and ⟦no bound, direction unknown⟧ on all four components when any component of v is flagged (a rotation is periodic:
    a flagged angle says nothing)."""
    a = comp(q, 0)
    vparts = [comp(q, i) for i in (1, 2, 3)]
    E = tot_exp(a)
    vv = [p.val.double() for p in vparts]
    vr2 = vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2]
    z = vr2 == 0
    vr = torch.where(z, torch.zeros_like(vr2), torch.sqrt(torch.where(z, torch.ones_like(vr2), vr2)))
    small = vr < 1e-4
    safe = torch.where(small, torch.ones_like(vr), vr)
    sinc = torch.where(small, 1.0 - vr2 / 6.0 + vr2 * vr2 / 120.0, torch.sin(safe) / safe)
    c = torch.cos(vr)
    fl = vparts[0].flag | vparts[1].flag | vparts[2].flag
    rf = torch.where(fl > 0, torch.full_like(fl, NOB | SUNK), torch.zeros_like(fl))
    cv, cs = _sat(c, c.device)
    parts = [tot_mul(E, Tot(cv, cs | rf))]
    for x in vv:
        sv, ss = _sat(sinc * x, x.device)
        parts.append(tot_mul(E, Tot(sv, ss | rf)))
    return stack(parts)


def qpow_t(x: Tot, w, side: str) -> Tot:
    """Pow_L(x, W) = Exp(W·Log₀ x), Pow_R(x, W) = Exp((Log₀ x)·W); W a constant quaternion (array, tensor or Tot)."""
    lx = qlog0_t(x)
    W = w if isinstance(w, Tot) else const_quat(w, lx)
    if side == 'L':
        return qexp_t(qmul_t(W, lx))
    if side == 'R':
        return qexp_t(qmul_t(lx, W))
    raise ValueError(f'unknown side: {side!r}')


def qright_basis_t(q: Tot, r: int) -> Tot:
    """q·e_r for e_r ∈ {1, i, j, k}: a wiring product with an exact one-hot constant (pattern P1: every flag survives)."""
    e = torch.zeros(4, dtype=F64, device=q.device)
    e[r] = 1.0
    return qmul_t(q, const_quat(e, q))


# ---------------------------------------------------------------- inputs / targets
def as_quat_tot(x, device=None):
    """[N, I, 4] / [N, G, I, 4] (numpy, tensor, or Tot) → (Tot [N, G, I, 4], grouped).  A raw array passes the customs of
    `Tot(x)` (NaN → unknown, overflow → MAX⟦≥⟧, subnormal → ε); a Tot is taken as it is (flags in)."""
    if isinstance(x, Tot):
        if x.val.dim() == 3:
            return Tot(x.val[:, None], x.flag[:, None]), False
        if x.val.dim() == 4:
            return x, True
        raise ValueError('a Tot input must have shape [N,I,4] or [N,G,I,4]')
    t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, dtype=np.float64))
    t = t.to(dtype=F64, device=device) if device is not None else t.to(F64)
    if t.dim() == 3 and t.shape[-1] == 4:
        return Tot(t[:, None]), False
    if t.dim() == 4 and t.shape[-1] == 4:
        return Tot(t), True
    raise ValueError('x must have shape [N,I,4] or [N,G,I,4]')


def as_target_tot(y, device=None):
    """[N, 4] / [N, G, 4] → (Tot [N, G, 4], grouped)."""
    if isinstance(y, Tot):
        if y.val.dim() == 2:
            return Tot(y.val[:, None], y.flag[:, None]), False
        return y, True
    t = y if torch.is_tensor(y) else torch.as_tensor(np.asarray(y, dtype=np.float64))
    t = t.to(dtype=F64, device=device) if device is not None else t.to(F64)
    if t.dim() == 2 and t.shape[-1] == 4:
        return Tot(t[:, None]), False
    if t.dim() == 3 and t.shape[-1] == 4:
        return Tot(t), True
    raise ValueError('y must have shape [N,4] or [N,G,4]')


# ---------------------------------------------------------------- forward evaluation
def _layer_w(layer, primes, key, wvars):
    if wvars is not None and key in wvars:
        return wvars[key]
    return torch.as_tensor(layer.state.exponent(primes), dtype=F64)


def eval_factor_total(xg: Tot, factor, primes, key=(0, 0), wvars=None) -> Tot:
    if factor.input_index < 0 or factor.input_index >= xg.val.shape[2]:
        raise IndexError(f'input_index {factor.input_index} out of range')
    h = Tot(xg.val[:, :, factor.input_index, :], xg.flag[:, :, factor.input_index, :])
    if factor.reciprocal:
        h = qinv_t(h)
    for li, layer in enumerate(factor.exponent_layers):
        h = qpow_t(h, _layer_w(layer, primes, key + (li,), wvars), layer.side)
    return h


def eval_unit_total(xg: Tot, unit, primes, ui=0, wvars=None) -> Tot:
    out = None
    for fi, factor in enumerate(unit.factors):
        f = eval_factor_total(xg, factor, primes, (ui, fi), wvars)
        out = f if out is None else qmul_t(out, f)
    if unit.right_basis is not None:
        out = qright_basis_t(out, unit.right_basis)
    return out


def features_total(x, network, wvars=None):
    """Φ as a Tot [N, G, M, 4] (M product units) and the grouped flag."""
    xg, grouped = as_quat_tot(x)
    units = [eval_unit_total(xg, unit, network.primes, ui, wvars) for ui, unit in enumerate(network.units)]
    phi = Tot(torch.stack([u.val for u in units], 2), torch.stack([u.flag for u in units], 2))
    return phi, grouped


# ---------------------------------------------------------------- the Σ readout: least squares on the usable samples
def usable_rows(*tots):
    """[N, G] mask: a sample is usable iff every flag on it (over the trailing axes) is 0 or ⟦≤⟧ alone."""
    m = None
    for t in tots:
        ok = ((t.flag & ~LE) == 0)
        while ok.dim() > 2:
            ok = ok.all(-1)
        m = ok if m is None else (m & ok)
    return m


def left_multiply_design_t(phi: torch.Tensor) -> torch.Tensor:
    """M(φ) with A·φ = M(φ) @ [a, b, c, d] (the reference's left_multiply_design, in torch)."""
    w, x, y, z = [phi[..., i] for i in range(4)]
    return torch.stack([torch.stack([w, -x, -y, -z], -1),
                        torch.stack([x, w, z, -y], -1),
                        torch.stack([y, -z, w, x], -1),
                        torch.stack([z, y, -x, w], -1)], -2)


def fit_readout_total(x, y, network, ridge=1e-8, return_counts=False):
    """A [G, M, 4] (or [M, 4] for ungrouped data) by ridge least squares over the usable samples of each group.
    Returns also (n_used, n_excluded) when asked.  A group without a usable sample gets A = 0."""
    phi, gx = features_total(x, network)
    yg, gy = as_target_tot(y, device=phi.device)
    if phi.val.shape[:2] != yg.val.shape[:2]:
        raise ValueError('x/y sample or group mismatch')
    m = usable_rows(phi, yg)
    n, g, mm = phi.val.shape[:3]
    coeff = torch.zeros(g, mm, 4, dtype=F64, device=phi.device)
    used = 0
    for gi in range(g):
        idx = m[:, gi].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        used += int(idx.numel())
        D = left_multiply_design_t(phi.val[idx, gi].double()).permute(0, 2, 1, 3).reshape(idx.numel() * 4, mm * 4)
        yy = yg.val[idx, gi].double().reshape(-1)
        G = D.T @ D + ridge * torch.eye(mm * 4, dtype=F64, device=D.device)
        coeff[gi] = torch.linalg.solve(G, D.T @ yy).reshape(mm, 4)
    if not gx and not gy:
        coeff = coeff[0]
    if return_counts:
        return coeff, used, int(n * g - used)
    return coeff


def predict_total(x, network, coeff) -> Tot:
    """Z = Σ_n A_n Φ_n as a Tot [N, 4] or [N, G, 4] — the readout with the same arithmetic, flags included."""
    phi, grouped = features_total(x, network)
    n, g, mm = phi.val.shape[:3]
    c = coeff if torch.is_tensor(coeff) else torch.as_tensor(np.asarray(coeff, dtype=np.float64))
    c = c.to(dtype=F64, device=phi.device)
    if c.dim() == 2:
        c = c[None]
    if tuple(c.shape) != (g, mm, 4):
        raise ValueError(f'coeff must have shape {(g, mm, 4)} or {(mm, 4)}')
    out = None
    for u in range(mm):
        A = Tot(c[:, u, :][None].expand(n, g, 4))
        term = qmul_t(A, Tot(phi.val[:, :, u], phi.flag[:, :, u]))
        out = term if out is None else stack([tot_add(comp(out, i), comp(term, i)) for i in range(4)])
    return out if grouped else Tot(out.val[:, 0], out.flag[:, 0])


def masked_mse(pred: Tot, y) -> tuple:
    """mean over the usable samples of |Z − y|² (per component); returns (mse, n_used, n_excluded); mse None if none."""
    yg, _ = as_target_tot(y, device=pred.device)
    p = pred if pred.val.dim() == 3 else Tot(pred.val[:, None], pred.flag[:, None])
    m = usable_rows(p, yg)
    idx = m.reshape(-1).nonzero(as_tuple=True)[0]
    n_all = m.numel()
    if idx.numel() == 0:
        return None, 0, n_all
    d = (p.val.reshape(-1, 4).double()[idx] - yg.val.reshape(-1, 4).double()[idx])
    return (d * d).mean(), int(idx.numel()), int(n_all - idx.numel())


def exact_fit_mse_total(x, y, network, ridge=1e-8):
    """(mse over the usable samples, coeff, n_used, n_excluded) for a fixed structure — the decision of the search."""
    coeff = fit_readout_total(x, y, network, ridge)
    pred = predict_total(x, network, coeff)
    mse, used, excl = masked_mse(pred, y)
    return (float('inf') if mse is None else float(mse)), coeff, used, excl


# ---------------------------------------------------------------- the gradient sensor (autograd through the Tot values)
def chunk_gradients_total(x, y, network, coeff, n_chunks=8):
    """dL_chunk/dW for every exponent layer, L = the masked MSE with A fixed.  Returns (keys, grads [B, L, 4]).  The
    gradients are totalized (`_sat`): an over/underflowed or NaN-derived gradient is 0 in the sensor, never a NaN."""
    keys = [(ui, fi, li) for ui, fi, li, _ in network.iter_layers()]
    if not keys:
        return keys, np.zeros((0, 0, 4), dtype=np.float64)
    xg, _ = as_quat_tot(x)
    yg, _ = as_target_tot(y, device=xg.device)
    c = coeff if torch.is_tensor(coeff) else torch.as_tensor(np.asarray(coeff, dtype=np.float64))
    c = c.to(dtype=F64, device=xg.device)
    if c.dim() == 2:
        c = c[None]
    n = xg.val.shape[0]
    chunks = [idx for idx in np.array_split(np.arange(n), min(n_chunks, n)) if len(idx)]
    out = []
    for idx in chunks:
        it = torch.as_tensor(idx, device=xg.device)
        wvars = {k: torch.tensor(ref._get_layer(network, k).state.exponent(network.primes), dtype=F64, device=xg.device,
                                 requires_grad=True) for k in keys}
        xc = Tot(xg.val[it], xg.flag[it])
        yc = Tot(yg.val[it], yg.flag[it])
        phi, _ = features_total(xc, network, wvars)
        nn_, g, mm = phi.val.shape[:3]
        pred = None
        for u in range(mm):
            A = Tot(c[:, u, :][None].expand(nn_, g, 4))
            term = qmul_t(A, Tot(phi.val[:, :, u], phi.flag[:, :, u]))
            pred = term if pred is None else stack([tot_add(comp(pred, i), comp(term, i)) for i in range(4)])
        loss, used, _ = masked_mse(pred, yc)
        gr = np.zeros((len(keys), 4), dtype=np.float64)
        if loss is not None and used > 0:
            loss.backward()
            for li, k in enumerate(keys):
                gk = wvars[k].grad
                if gk is not None:
                    v, f = _sat(gk.double(), gk.device)
                    v = torch.where((f & (GE | SUNK)) > 0, torch.zeros_like(v), v)
                    gr[li] = v.double().cpu().numpy()
        out.append(gr)
    return keys, np.stack(out, 0)


# ---------------------------------------------------------------- exact discrete structure search (the reference's loop)
def search_structure_total(x, y, initial_network, *, ridge=1e-8, max_steps=20, n_chunks=8, top_k_blocks=4, max_count=6,
                           min_improvement=1e-12, verbose=False):
    """gradient = sensor, exact discrete evaluation = decision — with every evaluation on total arithmetic and the loss
    over the usable samples.  Returns (network, coeff, history) like the reference."""
    net = initial_network.copy()
    mse, coeff, used, excl = exact_fit_mse_total(x, y, net, ridge)
    history = []
    for it in range(max_steps + 1):
        history.append(ref.SearchRecord(
            iteration=it, train_mse=mse, accepted_move=None,
            layer_exponents={(ui, fi, li): layer.state.fractions(net.primes) for ui, fi, li, layer in net.iter_layers()},
            layer_sides={(ui, fi, li): layer.side for ui, fi, li, layer in net.iter_layers()}))
        if it == max_steps:
            break
        keys, grads = chunk_gradients_total(x, y, net, coeff, n_chunks)
        if not keys:
            break
        sensor = ref.gradient_consensus_sensor(grads)
        proposals = ref.propose_moves(net, keys, sensor, top_k_blocks=top_k_blocks, max_count=max_count)
        seen = {ref._network_key(net)}
        exact = []
        for move in proposals:
            cand = ref.apply_move(net, move)
            key = ref._network_key(cand)
            if key in seen:
                continue
            seen.add(key)
            cmse, ccoeff, _, _ = exact_fit_mse_total(x, y, cand, ridge)
            exact.append((cmse, move, cand, ccoeff))
        if not exact:
            break
        exact.sort(key=lambda t: t[0])
        best_mse, best_move, best_net, best_coeff = exact[0]
        if best_mse >= mse - min_improvement:
            break
        history[-1].accepted_move = best_move
        net, mse, coeff = best_net, best_mse, best_coeff
        if verbose:
            print(f'iter={it:02d} mse={mse:.6e} move={best_move}')
    return net, coeff, history


# ---------------------------------------------------------------- self test
def _flags(t: Tot):
    return t.flag.reshape(-1).tolist()


def self_test():
    """the reserved words and the boundaries of ℍ, as numbers."""
    T = lambda *c: Tot(torch.tensor([list(c)], dtype=F64))
    eps = Tot(torch.tensor([[MIN, 0.0, 0.0, 0.0]], dtype=F32), torch.tensor([[LE, 0, 0, 0]], dtype=torch.uint8))
    print('i·j, j·k, k·i =', qmul_t(T(0, 1, 0, 0), T(0, 0, 1, 0)).val.tolist(), qmul_t(T(0, 0, 1, 0), T(0, 0, 0, 1)).val.tolist(),
          qmul_t(T(0, 0, 0, 1), T(0, 1, 0, 0)).val.tolist())
    L = qlog0_t(T(0, 0, 0, 0)); print('Log₀(0)   =', L.val.tolist(), 'flags', _flags(L))
    L = qlog0_t(eps); print('Log₀(ε)   =', L.val.tolist(), 'flags', _flags(L), ' (log MIN =', round(LOG_MIN, 3), '⟦≥⟧)')
    L = qlog0_t(T(-1, 0, 0, 0)); print('Log₀(−1)  =', L.val.tolist(), 'flags', _flags(L), ' (= iπ by definition)')
    L = qlog0_t(T(0, 0, 2, 0)); print('Log₀(2j)  =', L.val.tolist(), 'flags', _flags(L))
    print('Exp(Log₀(−1)) =', qexp_t(qlog0_t(T(-1, 0, 0, 0))).val.tolist())
    print('0⁻¹       =', qinv_t(T(0, 0, 0, 0)).val.tolist(), '(a/0 = 0)')
    print('Pow_L(0, 1) =', qpow_t(T(0, 0, 0, 0), [1, 0, 0, 0], 'L').val.tolist(), '(Exp∘Log₀: a zero input drops out)')
    q = T(0.3, -1.2, 0.5, 2.0)
    print('Pow_L(q, 1) − q =', (qpow_t(q, [1, 0, 0, 0], 'L').val - q.val).abs().max().item())
    print('Pow_L(q,−1) − q⁻¹ =', (qpow_t(q, [-1, 0, 0, 0], 'L').val - qinv_t(q).val).abs().max().item())
    big = Tot(torch.tensor([[1e30, 1e30, 0, 0]], dtype=F64))
    P = qpow_t(big, [2, 0, 0, 0], 'L'); print('Pow_L(1e30(1+i), 2) =', P.val.tolist(), 'flags', _flags(P), '(saturated, marked)')
    nanq = Tot(torch.tensor([[float('nan'), 1, 0, 0]], dtype=F64))
    P = qpow_t(nanq, [1, 0, 0, 0], 'L'); print('Pow_L(NaN+i, 1) =', P.val.tolist(), 'flags', _flags(P), '(unknown, never NaN)')
    print('OK')


if __name__ == '__main__':
    self_test()
