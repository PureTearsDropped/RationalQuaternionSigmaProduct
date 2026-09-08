#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""bounded rational exponents x^(n/o), 0 ≤ n, o ≤ m: the ledger, its exact place inside the reference's prime lattice,
the ledger spelled out as layers, and the structure search (IEEE reference and total arithmetic) confined to the bound.
   python test_bounded_exponent.py"""
import sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import flexible_rational_quaternion_sigma_product as ref
import bounded_exponent as be


def test_ledger_semantics():
    """n/0 = 0 (the reserved word), the empty ledger is W = 0, counts stay inside [0, m], the lattice is finite."""
    m = 3
    assert be.bounded_state(m, 2, 0).fractions() == (0, 0, 0, 0)
    assert be.bounded_state(m, 0, 2).fractions() == (0, 0, 0, 0)
    assert be.BoundedRationalQuaternionState.zero(m).exponent().tolist() == [0, 0, 0, 0]
    st = be.bounded_state(m, 2, 3).set(0, 1, 1, 2)                   # 2/3 − 1/2 = 1/6
    assert st.fractions()[0] == Fraction(1, 6) and st.layers_used() == 3
    for bad in ((4, 1), (1, 4)):
        try:
            be.bounded_state(m, *bad)
            assert False
        except ValueError:
            pass
    values = {be.total_fraction(n, o) for n in range(m + 1) for o in range(m + 1)}
    assert values == {Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)}
    one = {be.total_fraction(n, o) for n in range(2) for o in range(2)}
    assert one == {Fraction(0), Fraction(1)}                          # m = 1: only 0 and ±1 — the bound must be free


def test_bound_is_a_free_parameter():
    """one input node with bound 3: the network accepts it, and the number of inputs is not the bound."""
    net = be.make_bounded_alternating_seed(n_units=4, bound=3)
    assert be.max_count_of(net) == 3 and net.primes == be.UNARY
    try:
        be.bounded_network(net.units, bound=2)
        assert False
    except ValueError:
        pass


def test_moves_respect_bound():
    """the reference's move generator never proposes a count above the bound, apply_move refuses one."""
    m = 2
    st = be.bounded_state(m, 2, 1)                                     # n at the bound
    net = be.bounded_network([ref.ProductUnit([ref.make_power_factor(0, st)])], bound=m)
    key = (0, 0, 0)
    score = np.ones((1, 4))
    props = ref.propose_moves(net, [key], dict(score=score), top_k_blocks=4, max_count=None,
                              include_side_flips=False, include_compound=False)
    for mv in props:
        cand = ref.apply_move(net, mv)
        assert cand.units[0].factors[0].exponent_layers[0].state.counts.max() <= m
    ups = [mv for mv in props if mv.component == 0 and mv.sign == 0 and mv.leg == be.NUM and mv.delta == +1]
    assert not ups                                                     # 2 → 3 is never proposed
    try:
        ref.apply_move(net, ref.NaturalMove(key, 0, 0, be.NUM, 0, +1))
        assert False
    except ValueError:
        pass


def _data(rng, n, inputs):
    x = rng.normal(size=(n, inputs, 4)) * 0.4
    x[:, :, 0] += 1.5                                                  # away from the branch cut
    return x


def test_bounded_equals_prime_state():
    """a bounded network and its to_natural conversion give identical features (the bounded lattice sits inside the
    reference's lattice), including quaternion exponents and L/R sides."""
    rng = np.random.default_rng(1)
    m = 3
    x = _data(rng, 100, m)
    w1 = be.bounded_state(m, 2, 3).set(1, 0, 1, 2).set(2, 1, 3, 1)     # 2/3 + ½ i − 3 j
    w2 = be.bounded_state(m, 1, 1).set(3, 0, 1, 3)                     # 1 + ⅓ k
    net = be.bounded_network([ref.ProductUnit([
        ref.Factor(0, exponent_layers=[ref.ExponentLayer(w1, 'L'), ref.ExponentLayer(w2, 'R')]),
        be.make_bounded_power_factor(2, m, 3, 2, side='R'),
        ref.make_passthrough_factor(1, reciprocal=True)], right_basis=1)], bound=m)
    primes = be.primes_up_to(m)
    nat = ref.SigmaProductNetwork(primes=primes, units=[ref.ProductUnit([
        ref.Factor(0, exponent_layers=[ref.ExponentLayer(w1.to_natural(primes), 'L'), ref.ExponentLayer(w2.to_natural(primes), 'R')]),
        ref.make_power_factor(2, be.bounded_state(m, 3, 2).to_natural(primes), side='R'),
        ref.make_passthrough_factor(1, reciprocal=True)], right_basis=1)])
    a, _ = ref.features_numpy(x, net)
    b, _ = ref.features_numpy(x, nat)
    assert np.array_equal(a, b)                                        # bit-identical: the same float exponent


def test_ledger_unrolls_into_layers():
    """X^(n/o) = (X^(1/o))·…·(X^(1/o)) with n product slots: the ledger is literally a layer count, and a difference
    n⁺/o⁺ − n⁻/o⁻ unrolls into n⁺ + n⁻ slots.  Both sides agree to machine precision."""
    rng = np.random.default_rng(2)
    m = 3
    x = _data(rng, 100, 2)
    net = be.bounded_network([ref.ProductUnit([
        be.make_bounded_power_factor(0, m, 3, 2, name='f0'),                                  # X₀^{3/2}
        ref.make_power_factor(1, be.bounded_state(m, 2, 3).set(0, 1, 1, 2), name='f1')],      # X₁^{2/3 − 1/2}
        right_basis=3)], bound=m)
    un = be.unroll(net)
    assert len(un.units[0].factors) == 3 + (2 + 1)
    assert all(len(f.exponent_layers) == 1 for f in un.units[0].factors)
    roots = [f.exponent_layers[0].state.fractions(un.primes)[0] for f in un.units[0].factors]
    assert roots == [Fraction(1, 2)] * 3 + [Fraction(1, 3)] * 2 + [Fraction(1, 2)]
    recips = [f.reciprocal for f in un.units[0].factors]
    assert recips == [False] * 5 + [True]
    a, _ = ref.features_numpy(x, net)
    b, _ = ref.features_numpy(x, un)
    assert np.abs(a - b).max() < 1e-12, np.abs(a - b).max()


def _teacher_net(m):
    return be.bounded_network([ref.ProductUnit([be.make_bounded_power_factor(0, m, 1, 2),
                                                be.make_bounded_power_factor(1, m, 2, 1)])], bound=m)


def _seed_net(m):
    return be.bounded_network([ref.ProductUnit([be.make_bounded_power_factor(0, m, 1, 1),
                                                be.make_bounded_power_factor(1, m, 1, 1)])], bound=m)


def test_search_on_the_bounded_lattice_reference():
    """teacher X₀^{1/2}·X₁²  (m = 2 inputs, bound 2): the reference search starting from X₀·X₁ walks two ±1 ledger moves,
    never leaves [0, 2], and lands on the teacher exactly."""
    rng = np.random.default_rng(4)
    m = 2
    x = _data(rng, 400, m)
    y = ref.predict(x, _teacher_net(m), np.array([[0.7, -0.1, 0.2, 0.05]]))
    net, coeff, hist = ref.search_structure(x, y, _seed_net(m), max_steps=6, max_count=None)
    fr = [l.state.fractions()[0] for *_, l in net.iter_layers()]
    assert fr == [Fraction(1, 2), Fraction(2)], fr
    assert hist[-1].train_mse < 1e-20, hist[-1].train_mse
    assert max(l.state.counts.max() for *_, l in net.iter_layers()) <= m
    assert np.allclose(coeff, [[0.7, -0.1, 0.2, 0.05]], atol=1e-8)


def test_search_on_the_bounded_lattice_total():
    """the same search on total arithmetic (Tot), with a NaN row and an ε row in the data: excluded, not fatal."""
    import torch
    import rational_quaternion_sigma_product as qsp
    rng = np.random.default_rng(4)
    m = 2
    x = _data(rng, 400, m)
    y = ref.predict(x, _teacher_net(m), np.array([[0.7, -0.1, 0.2, 0.05]]))
    xb = np.concatenate([x, np.array([[[np.nan, 1, 0, 0], [1, 0, 0, 0]], [[5e-39, 0, 0, 0], [1, 0, 0, 0]]])], 0)
    yb = np.concatenate([y, np.zeros((2, 4))], 0)
    net, coeff, hist = qsp.search_structure_total(xb, yb, _seed_net(m), max_steps=6, max_count=None)
    fr = [l.state.fractions()[0] for *_, l in net.iter_layers()]
    assert fr == [Fraction(1, 2), Fraction(2)], fr
    assert hist[-1].train_mse < 1e-9, hist[-1].train_mse
    assert np.allclose(coeff.numpy(), [[0.7, -0.1, 0.2, 0.05]], atol=1e-4)
    print('   ', qsp.formula(net, coeff, names=('X₀', 'X₁'), snap=1e-3))


def test_ledger_moves():
    """a LedgerMove replaces one channel's ledger; the proposals of a block are every distinct fraction of both
    channels (deduplicated: 1/1 and 2/2 are one), all inside the bound."""
    m = 3
    st = be.bounded_state(m, 1, 1)
    net = be.bounded_network([ref.ProductUnit([ref.make_power_factor(0, st)])], bound=m)
    key = (0, 0, 0)
    cand = be.apply_move(net, be.LedgerMove(key, 0, 0, 2, 3))
    assert cand.units[0].factors[0].exponent_layers[0].state.fractions()[0] == Fraction(2, 3)
    assert net.units[0].factors[0].exponent_layers[0].state.fractions()[0] == 1          # the original untouched
    props = be.propose_ledger_moves(net, [key], dict(score=np.array([[1.0, 0, 0, 0]])), include_side_flips=False)
    fr = {(p.sign, be.total_fraction(p.n, p.o)) for p in props}
    values = {Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)}
    assert fr == {(0, v) for v in values - {Fraction(1)}} | {(1, v) for v in values - {Fraction(0)}}
    assert len(props) == len(fr)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
    print('OK: the bounded ledger x^(n/o), 0 ≤ n, o ≤ m')
