#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""the three demos of the reference (`flexible_rational_quaternion_sigma_product.py`) on total arithmetic, and a fourth:
the same gradient-sensor + exact discrete search on data with a few boundary rows (a missing measurement = NaN, a
value beyond f32 MAX, an ε), IEEE reference vs total.   python physics/demos.py"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import rational_quaternion_sigma_product as qsp
import flexible_rational_quaternion_sigma_product as ref

torch.set_num_threads(max(1, torch.get_num_threads()))


def fmt_layers(net):
    return '  '.join(f'{k}: W={tuple(str(v) for v in layer.state.fractions(net.primes))} {layer.side}'
                     for k, layer in ((( ui, fi, li), layer) for ui, fi, li, layer in net.iter_layers()))


def demo_exact_x1_times_x2(seed=10):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(500, 4)), rng.normal(size=(500, 4))
    x = np.stack([x1, x2], 1)
    y = ref.qmul(x1, x2)
    net = ref.SigmaProductNetwork(primes=(2, 3, 5, 7), units=[ref.ProductUnit([ref.make_passthrough_factor(0), ref.make_passthrough_factor(1)])])
    mse_r, c_r = ref.exact_fit_mse(x, y, net, ridge=1e-12)
    mse_t, c_t, used, excl = qsp.exact_fit_mse_total(x, y, net, ridge=1e-12)
    print('demo 1  y = x1·x2 (no exponent layer)')
    print(f'  reference MSE {mse_r:.3e}  A = {np.round(c_r[0], 6)}')
    print(f'  total     MSE {mse_t:.3e}  A = {np.round(c_t[0].numpy(), 6)}   used {used} excluded {excl}')


def demo_free_depth(seed=11):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(700, 2, 4)); x[:, 0, 0] += 1.5; x[:, 1, 0] += 1.2
    primes = (2, 3, 5, 7)
    w1 = ref.make_scalar_plus_one_state(primes); w2 = ref.make_scalar_plus_one_state(primes)
    w2.counts[1, 1, 1, primes.index(2)] += 1
    net = ref.SigmaProductNetwork(primes=primes, units=[ref.ProductUnit([
        ref.Factor(input_index=0, exponent_layers=[ref.ExponentLayer(w1, 'L'), ref.ExponentLayer(w2, 'R')]), ref.make_passthrough_factor(1)])])
    phi, _ = ref.features_numpy(x, net)
    y = ref.qmul(np.broadcast_to(np.array([0.8, -0.2, 0.15, 0.1]), phi[:, 0, 0].shape), phi[:, 0, 0])
    mse_r, c_r = ref.exact_fit_mse(x, y, net, ridge=1e-12)
    mse_t, c_t, used, excl = qsp.exact_fit_mse_total(x, y, net, ridge=1e-12)
    print('demo 2  P = Pow_R(Pow_L(X1, 1), 1 + ½i)·X2  (two exponent layers)')
    print(f'  reference MSE {mse_r:.3e}  A = {np.round(c_r[0], 5)}')
    print(f'  total     MSE {mse_t:.3e}  A = {np.round(c_t[0].numpy(), 5)}   used {used} excluded {excl}')


def _search_setup(seed, n=1200):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, 4)); x0[:, 0] += 1.7
    x = x0[:, None, :]
    primes = (2, 3, 5, 7)
    init = ref.make_scalar_plus_one_state(primes)
    target = init.copy(); target.counts[1, 1, 1, primes.index(2)] += 1        # 1 + ½ i
    teacher = ref.SigmaProductNetwork(primes=primes, units=[ref.ProductUnit([ref.make_power_factor(0, target, side='R')])])
    y = ref.predict(x, teacher, np.array([[0.7, -0.1, 0.2, 0.05]]))
    start = ref.SigmaProductNetwork(primes=primes, units=[ref.ProductUnit([ref.make_power_factor(0, init, side='L')])])
    return x, y, start, teacher


def demo_gradient_search(seed=12):
    x, y, start, teacher = _search_setup(seed)
    print('demo 3  search: start Pow_L(X, 1) → teacher Pow_R(X, 1 + ½i)')
    t0 = time.time()
    fr, cr, hr = ref.search_structure(x, y, start, ridge=1e-10, max_steps=5, n_chunks=6, top_k_blocks=4)
    print(f'  reference: {len([h for h in hr if h.accepted_move])} accepted moves, final MSE {hr[-1].train_mse:.3e}, {time.time() - t0:.1f}s')
    print('    ', fmt_layers(fr))
    t0 = time.time()
    ft, ct, ht = qsp.search_structure_total(x, y, start, ridge=1e-10, max_steps=5, n_chunks=6, top_k_blocks=4, min_improvement=1e-10)
    print(f'  total:     {len([h for h in ht if h.accepted_move])} accepted moves, final MSE {ht[-1].train_mse:.3e}, {time.time() - t0:.1f}s')
    print('    ', fmt_layers(ft))
    print('  teacher:  ', fmt_layers(teacher))


BAD_ROWS = {'NaN': [np.nan, 1.0, 0.0, 0.0], '1e40': [1e40, 1e40, 0.0, 0.0], 'ε': [5e-39, 0.0, 0.0, 0.0]}


def demo_contaminated_search(seed=12, rows=('NaN', '1e40', 'ε'), label='4'):
    """the same search, but the data carry boundary rows: NaN (a missing measurement), 1e40 (beyond f32 MAX), ε.
    The reference has no mark for them (NaN spreads; 1e40 is clipped inside exp to e^80 without a flag, and that one
    silently clipped row dominates the least squares and steers the search); total arithmetic flags them at the customs
    and the fit/search never sees them."""
    x, y, start, teacher = _search_setup(seed)
    n_bad = len(rows)
    bad = np.array([[BAD_ROWS[r]] for r in rows])
    xb = np.concatenate([x, bad], 0)
    yb = np.concatenate([y, np.zeros((n_bad, 4))], 0)
    print(f'demo {label}  the same search with {n_bad} boundary rows appended ({", ".join(rows)})')
    t0 = time.time()
    with np.errstate(all='ignore'):
        try:
            fr, cr, hr = ref.search_structure(xb, yb, start, ridge=1e-10, max_steps=5, n_chunks=6, top_k_blocks=4)
            print(f'  reference: {len([h for h in hr if h.accepted_move])} accepted moves, final MSE {hr[-1].train_mse:.3e}, {time.time() - t0:.1f}s')
            print('    ', fmt_layers(fr), '  A =', np.round(cr[0], 4) if np.all(np.isfinite(cr)) else cr[0])
            if np.all(np.isfinite(cr)):
                pr = ref.predict(x, fr, cr)
                print(f'     its MSE on the {len(x)} clean rows: {np.mean((pr - y) ** 2):.3e}   (the clipped feature of the 1e40 row, unmarked: '
                      f'{np.round(ref.features_numpy(xb, fr)[0][len(x) + list(rows).index("1e40"), 0, 0], 3) if "1e40" in rows else "-"})')
        except Exception as e:                                             # a LinAlgError from a NaN Gram matrix is the honest outcome too
            print(f'  reference: crashed — {type(e).__name__}: {e}')
    t0 = time.time()
    ft, ct, ht = qsp.search_structure_total(xb, yb, start, ridge=1e-10, max_steps=5, n_chunks=6, top_k_blocks=4, min_improvement=1e-10)
    mse, c, used, excl = qsp.exact_fit_mse_total(xb, yb, ft, ridge=1e-10)
    print(f'  total:     {len([h for h in ht if h.accepted_move])} accepted moves, final MSE {mse:.3e} on {used} usable rows ({excl} excluded), {time.time() - t0:.1f}s')
    print('    ', fmt_layers(ft), '  A =', np.round(c[0].numpy(), 4))
    Z = qsp.predict_total(xb, ft, c)
    print('  the excluded rows come back flagged, never NaN:')
    for k in range(n_bad):
        print(f'    row {len(x) + k}: Z = [' + ' '.join(f'{v:+.3e}' for v in Z.val[len(x) + k].tolist()) + f']  flags {Z.flag[len(x) + k].tolist()}  finite={bool(torch.isfinite(Z.val[len(x) + k]).all())}')
    print('  teacher:  ', fmt_layers(teacher))


if __name__ == '__main__':
    demo_exact_x1_times_x2(); print()
    demo_free_depth(); print()
    demo_gradient_search(); print()
    demo_contaminated_search(); print()
    demo_contaminated_search(rows=('1e40', 'ε'), label='4b')
