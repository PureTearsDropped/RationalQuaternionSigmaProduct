#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""is the network the reference's?  60 random structures (1–3 units, 1–3 factors, X or X⁻¹, 0–2 exponent layers L/R with
random natural-number states, random right basis, 1–2 groups), features compared entry by entry on the UNFLAGGED entries.

The reference carries two unmarked guards: exp's argument is clipped to ±80, and log|q| / the vector direction are floored
at |·| < 1e-12.  Off the boundary they are silent; a chained exponent layer whose first output is ~1e-13 or whose amplitude
exponent is > 80 hits them.  The check runs the reference with the guards on and off, and the total arithmetic with its
values in float32 (as shipped) and float64 (to separate precision from structure).   python physics/parity_check.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import rational_quaternion_sigma_product as qsp
import flexible_rational_quaternion_sigma_product as ref
import cuda_total

REF_QLOG, REF_QEXP = ref.qlog, ref.qexp


def qexp_noclip(q):
    q = np.asarray(q, float); a = q[..., 0]; v = q[..., 1:]; vr = np.linalg.norm(v, axis=-1)
    ea = np.exp(np.clip(a, -700.0, 700.0))
    sinc = np.ones_like(vr); m = vr > 1e-10; sinc[m] = np.sin(vr[m]) / vr[m]; r = vr[~m]; sinc[~m] = 1 - r * r / 6 + r ** 4 / 120
    return np.concatenate([(ea * np.cos(vr))[..., None], ea[..., None] * v * sinc[..., None]], -1)


def random_network(rng, primes, I, maxcount):
    units = []
    for _ in range(rng.integers(1, 4)):
        factors = []
        for _ in range(rng.integers(1, 4)):
            layers = []
            for _ in range(rng.integers(0, 3)):
                st = ref.NaturalQuaternionState(np.minimum(rng.integers(0, 3, size=(4, 2, 2, len(primes))), maxcount))
                layers.append(ref.ExponentLayer(st, side=rng.choice(['L', 'R'])))
            factors.append(ref.Factor(int(rng.integers(0, I)), bool(rng.integers(0, 2)), layers))
        rb = rng.choice([None, 0, 1, 2, 3])
        units.append(ref.ProductUnit(factors, None if rb is None else int(rb)))
    return ref.SigmaProductNetwork(primes, units)


def run(maxcount, f64, guards, trials=60, seed=0):
    cuda_total.F32 = torch.float64 if f64 else torch.float32           # the Tot value type; the saturation domain stays f32 MAX/MIN
    ref.qexp = REF_QEXP if guards else qexp_noclip
    ref.qlog = REF_QLOG if guards else (lambda q, eps=1e-300: REF_QLOG(q, eps))
    rng = np.random.default_rng(seed); primes = (2, 3, 5, 7)
    n = bad = 0; worst = 0.0; maxw = 0.0
    with np.errstate(all='ignore'):
        for _ in range(trials):
            I = int(rng.integers(1, 4)); G = int(rng.integers(1, 3)); N = 60
            x = rng.normal(size=(N, G, I, 4)); x[..., 0] += 2.0                    # away from the negative real axis (a definition differs there)
            net = random_network(rng, primes, I, maxcount)
            for _, _, _, layer in net.iter_layers():
                maxw = max(maxw, float(np.abs(layer.state.exponent(primes)).max()))
            pr, _ = ref.features_numpy(x, net); pt, _ = qsp.features_total(x, net)
            ptv = pt.val.double().numpy(); unfl = (pt.flag.numpy() == 0) & np.isfinite(pr)
            rel = np.abs(ptv - pr) / (1 + np.abs(pr))
            n += int(unfl.sum()); bad += int(((rel > 1e-3) & unfl).sum())
            if unfl.any():
                worst = max(worst, float(rel[unfl].max()))
    cuda_total.F32 = torch.float32; ref.qexp = REF_QEXP; ref.qlog = REF_QLOG
    return n, bad, worst, maxw


if __name__ == '__main__':
    print('60 random structures; "disagree" = relative difference > 1e-3 on an entry the total arithmetic left unflagged')
    print('reference guards = exp clip ±80 and the 1e-12 floor on log|q| and on the vector direction, both unmarked')
    for mc in (1, 2):
        for guards in (True, False):
            for f64 in (True, False):
                n, b, w, mw = run(mc, f64, guards)
                print(f'counts ≤{mc} (max |W component| {mw:6.0f})  guards {"on " if guards else "off"}  Tot values {"f64" if f64 else "f32"}: '
                      f'unflagged {n:6d}  disagree {b:4d}  worst rel {w:.1e}')
