#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""the quaternion Σ-Product network on total arithmetic: reserved words, agreement with the IEEE reference off the
boundary, and the boundary itself.   python test_rational_quaternion_sigma_product.py"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import rational_quaternion_sigma_product as qsp
import flexible_rational_quaternion_sigma_product as ref
from rational_quaternion_sigma_product import Tot, GE, LE, SUNK, MIN, LOG_MIN, F64


def T(*rows):
    return Tot(torch.tensor([list(r) for r in rows], dtype=F64))


def test_hamilton_table_is_hamilton():
    """i·j = k, j·k = i, k·i = j, i² = j² = k² = −1 — and the wiring product equals the reference qmul on random data."""
    e = np.eye(4)
    for a, b, c, s in ((1, 2, 3, 1), (2, 3, 1, 1), (3, 1, 2, 1), (2, 1, 3, -1), (3, 2, 1, -1), (1, 3, 2, -1)):
        out = qsp.qmul_t(T(*[e[a]]), T(*[e[b]])).val[0].numpy()
        assert np.allclose(out, s * e[c]), (a, b, out)
    for a in (1, 2, 3):
        assert np.allclose(qsp.qmul_t(T(*[e[a]]), T(*[e[a]])).val[0].numpy(), -e[0])
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))
    got = qsp.qmul_t(Tot(torch.tensor(x)), Tot(torch.tensor(y))).val.double().numpy()
    assert np.allclose(got, ref.qmul(x, y), rtol=1e-5, atol=1e-5)


def test_reserved_words():
    """Log₀(0) = 0 unflagged; Log₀(ε) = log MIN ⟦≥⟧ on the scalar part; Arg₀ of a pure imaginary is π/2 with its axis;
    Log₀(−1) = iπ; 0⁻¹ = 0; Pow(0, W) = 1 (the zero input drops out); Pow(q, 1) = q; Pow(q, −1) = q⁻¹."""
    L = qsp.qlog0_t(T((0, 0, 0, 0)))
    assert L.val.abs().max().item() == 0 and int(L.flag.max()) == 0
    eps = Tot(torch.tensor([[MIN, 0, 0, 0]], dtype=torch.float32), torch.tensor([[LE, 0, 0, 0]], dtype=torch.uint8))
    L = qsp.qlog0_t(eps)
    assert abs(L.val[0, 0].item() - LOG_MIN) < 1e-4 and L.flag[0, 0].item() == GE and int(L.flag[0, 1:].max()) == 0
    L = qsp.qlog0_t(T((0, 0, 2, 0)))
    assert abs(L.val[0, 0].item() - math.log(2)) < 1e-6 and abs(L.val[0, 2].item() - math.pi / 2) < 1e-6
    assert L.val[0, 1].item() == 0 and L.val[0, 3].item() == 0 and int(L.flag.max()) == 0
    L = qsp.qlog0_t(T((-1, 0, 0, 0)))
    assert abs(L.val[0, 1].item() - math.pi) < 1e-6 and L.val[0, 0].item() == 0 and int(L.flag.max()) == 0
    E = qsp.qexp_t(L)
    assert abs(E.val[0, 0].item() + 1) < 1e-6 and E.val[0, 1:].abs().max().item() < 1e-6
    assert qsp.qinv_t(T((0, 0, 0, 0))).val.abs().max().item() == 0
    P = qsp.qpow_t(T((0, 0, 0, 0)), [1, 0, 0, 0], 'L')
    assert P.val[0].tolist() == [1.0, 0.0, 0.0, 0.0] and int(P.flag.max()) == 0
    q = T((0.3, -1.2, 0.5, 2.0))
    assert (qsp.qpow_t(q, [1, 0, 0, 0], 'L').val - q.val).abs().max().item() < 1e-6
    assert (qsp.qpow_t(q, [-1, 0, 0, 0], 'R').val - qsp.qinv_t(q).val).abs().max().item() < 1e-6


def test_agrees_with_reference_off_the_boundary():
    """on ordinary data the total evaluation, the readout fit and the prediction equal the IEEE reference to f32."""
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=(n, 2, 4))
    x[:, 0, 0] += 1.5
    x[:, 1, 0] += 1.2
    primes = (2, 3, 5, 7)
    w1 = ref.make_scalar_plus_one_state(primes)
    w2 = ref.make_scalar_plus_one_state(primes)
    w2.counts[1, 1, 1, primes.index(2)] += 1                                   # 1 + ½ i
    net = ref.SigmaProductNetwork(primes=primes, units=[ref.ProductUnit([
        ref.Factor(input_index=0, exponent_layers=[ref.ExponentLayer(w1, 'L'), ref.ExponentLayer(w2, 'R')]),
        ref.make_passthrough_factor(1, reciprocal=True)], right_basis=2)])
    phi_ref, _ = ref.features_numpy(x, net)
    phi_tot, _ = qsp.features_total(x, net)
    assert int(phi_tot.flag.max()) == 0
    assert np.allclose(phi_tot.val.double().numpy(), phi_ref, rtol=2e-4, atol=2e-4), np.abs(phi_tot.val.double().numpy() - phi_ref).max()
    y = ref.qmul(np.broadcast_to(np.array([0.8, -0.2, 0.15, 0.1]), phi_ref[:, 0, 0].shape), phi_ref[:, 0, 0])
    c_ref = ref.fit_readout(x, y, net, ridge=1e-12)
    c_tot = qsp.fit_readout_total(x, y, net, ridge=1e-12)
    assert np.allclose(c_tot.numpy(), c_ref, atol=2e-4), (c_tot, c_ref)
    Z = qsp.predict_total(x, net, c_tot)
    assert int(Z.flag.max()) == 0
    mse, used, excl = qsp.masked_mse(Z, y)
    assert used == n and excl == 0 and float(mse) < 1e-8, float(mse)


def test_boundary_is_marked_not_nan():
    """a NaN input, an input beyond f32 MAX and an ε input give flagged outputs (never NaN/Inf), are excluded from the
    fit, and the readout fitted on the clean rows is unchanged by them."""
    rng = np.random.default_rng(5)
    n = 200
    x = rng.normal(size=(n, 1, 4))
    x[:, 0, 0] += 1.7
    net = ref.make_alternating_linear_seed(n_units=4)
    y = ref.predict(x, net, np.array([[0.7, -0.1, 0.2, 0.05]] * 4))
    c_clean = qsp.fit_readout_total(x, y, net, ridge=1e-10)
    xb = np.concatenate([x, np.array([[[np.nan, 1, 0, 0]], [[1e40, 0, 0, 0]], [[5e-39, 0, 0, 0]]])], 0)
    yb = np.concatenate([y, np.zeros((3, 4))], 0)
    c_dirty, used, excl = qsp.fit_readout_total(xb, yb, net, ridge=1e-10, return_counts=True)
    assert used == n and excl == 3
    assert torch.allclose(c_dirty, c_clean, atol=1e-6)
    Z = qsp.predict_total(xb, net, c_dirty)
    assert torch.isfinite(Z.val).all()
    assert int(Z.flag[:n].max()) == 0 and all(int(Z.flag[n + k].max()) > 0 for k in range(3))
    xt, _ = qsp.as_quat_tot(xb)
    assert xt.flag[n, 0, 0, 0].item() == GE | LE | SUNK and xt.flag[n + 1, 0, 0, 0].item() == GE and xt.flag[n + 2, 0, 0, 0].item() == LE


def test_exact_product_x1x2():
    rng = np.random.default_rng(10)
    x1, x2 = rng.normal(size=(200, 4)), rng.normal(size=(200, 4))
    x = np.stack([x1, x2], 1)
    y = ref.qmul(x1, x2)
    net = ref.SigmaProductNetwork(primes=(2, 3, 5, 7), units=[ref.ProductUnit([ref.make_passthrough_factor(0), ref.make_passthrough_factor(1)])])
    mse, coeff, used, excl = qsp.exact_fit_mse_total(x, y, net, ridge=1e-12)
    assert mse < 1e-10 and excl == 0
    assert np.allclose(coeff.numpy(), [[1, 0, 0, 0]], atol=1e-5)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
    print('OK: the quaternion reserved words hold')
