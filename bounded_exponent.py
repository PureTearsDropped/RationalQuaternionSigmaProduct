# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
Bounded rational exponents: x^(n/o) with 0 ≤ n, o ≤ m,  m = number of input nodes
=================================================================================

The reference encodes an exponent component as w = R₊ − R₋ with R± = ∏_p p^(num−den) over a
prime list — an unbounded lattice (the counts are prime valuations).  This module offers the
other natural-number ledger: **the count itself is the exponent's numerator or denominator**,

    w_μ = n⁺_μ / o⁺_μ  −  n⁻_μ / o⁻_μ,        n, o ∈ {0, 1, …, m},

with the total-arithmetic reading of the division, n/0 := 0 (a/0 = 0 is the Moore–Penrose
inverse of 0; the empty ledger 0/0 is the exponent 0, i.e. the factor 1).  m is the number of
input nodes of the network: a numerator n is "n layers of the input multiplied together", a
denominator o is "the o-th root", so every exponent the network can hold is realised by at most
m layers in each leg — see `unroll` below, which spells a factor X^(n/o) out as o-th root
followed by n product slots and lets you check the two agree to machine precision.

The state has the same duck-type as `NaturalQuaternionState` (counts[μ, sign, leg, digit],
`.copy()`, `.zero()`, `.fractions(primes)`, `.exponent(primes)`, `.n_primes`), so the
reference's evaluation, the gradient sensor, the ±1 move generator and both search loops (IEEE
reference and total arithmetic) run on it unchanged.  The one difference is the "prime" tuple of
the network: a bounded network has `primes = UNARY = (1,)` — one digit, the count itself — and
`apply_move`/`propose_moves` refuse any count above the state's bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence
import numpy as np

import flexible_rational_quaternion_sigma_product as ref

UNARY = (1,)          # the "prime" tuple of a bounded network: one digit, the count itself
NUM, DEN = 0, 1       # the two legs of a count ledger


def total_fraction(n: int, o: int) -> Fraction:
    """n/o with the reserved word n/0 = 0."""
    return Fraction(int(n), int(o)) if o else Fraction(0)


@dataclass
class BoundedRationalQuaternionState:
    """
    counts[μ, sign, leg, 0]  ∈ {0, …, bound}

      μ    = 0,1,2,3  -> scalar, i, j, k
      sign = 0,1      -> the + channel, the − channel
      leg  = 0,1      -> numerator n, denominator o

    w_μ = n⁺/o⁺ − n⁻/o⁻  with n/0 = 0.
    """
    counts: np.ndarray
    bound: int

    def __post_init__(self):
        a = np.asarray(self.counts)
        if a.ndim != 4 or a.shape != (4, 2, 2, 1):
            raise ValueError("counts must have shape [4,2,2,1] (μ, sign, leg, one digit)")
        if int(self.bound) < 0:
            raise ValueError("bound must be a natural number")
        if np.any(a < 0) or np.any(a > int(self.bound)):
            raise ValueError(f"every count must lie in [0, {self.bound}]")
        self.counts = a.astype(np.int64, copy=True)
        self.bound = int(self.bound)

    # ---- the NaturalQuaternionState duck-type ------------------------------------------------
    @property
    def n_primes(self) -> int:
        return 1

    @classmethod
    def zero(cls, bound: int) -> "BoundedRationalQuaternionState":
        return cls(np.zeros((4, 2, 2, 1), dtype=np.int64), bound)

    def copy(self) -> "BoundedRationalQuaternionState":
        return BoundedRationalQuaternionState(self.counts.copy(), self.bound)

    def _check(self, primes: Sequence[int]):
        if len(primes) != 1:
            raise ValueError("a bounded state lives in a network with primes = UNARY = (1,)")

    def fraction(self, mu: int, sign: int) -> Fraction:
        return total_fraction(self.counts[mu, sign, NUM, 0], self.counts[mu, sign, DEN, 0])

    def channels(self, primes: Sequence[int] = UNARY):
        self._check(primes)
        return tuple((self.fraction(mu, 0), self.fraction(mu, 1)) for mu in range(4))

    def fractions(self, primes: Sequence[int] = UNARY):
        return tuple(rp - rm for rp, rm in self.channels(primes))

    def exponent(self, primes: Sequence[int] = UNARY) -> np.ndarray:
        return np.array([float(v) for v in self.fractions(primes)], dtype=np.float64)

    # ---- the ledger as numbers ----------------------------------------------------------------
    def set(self, mu: int, sign: int, n: int, o: int) -> "BoundedRationalQuaternionState":
        """write the ledger n/o into (μ, sign); returns self."""
        for v in (n, o):
            if not 0 <= int(v) <= self.bound:
                raise ValueError(f"count {v} outside [0, {self.bound}]")
        self.counts[mu, sign, NUM, 0] = int(n)
        self.counts[mu, sign, DEN, 0] = int(o)
        return self

    def ledger(self):
        """((n⁺, o⁺), (n⁻, o⁻)) for μ = 0..3."""
        return tuple(((int(self.counts[mu, 0, NUM, 0]), int(self.counts[mu, 0, DEN, 0])),
                      (int(self.counts[mu, 1, NUM, 0]), int(self.counts[mu, 1, DEN, 0]))) for mu in range(4))

    def layers_used(self) -> int:
        """the largest count in the ledger = the number of layers this exponent needs."""
        return int(self.counts.max())

    # ---- exact conversion to the prime-valuation state ----------------------------------------
    def to_natural(self, primes: Sequence[int], search: int = 10000) -> ref.NaturalQuaternionState:
        """the same exponent as a prime-valuation state (exact).  The reference's channels are the
        positive rationals R₊, R₋ (R = 1 for an empty ledger) and w = R₊ − R₋; a bounded w = a/b is
        written as R₊ = (a + c)/b, R₋ = c/b with the smallest c ≥ 1 making both p-smooth over `primes`
        (c is searched up to `search`).  Proves the bounded lattice is a finite subset of the reference's
        lattice: the two states evaluate identically."""
        primes = tuple(primes)

        def smooth(v: int) -> dict | None:
            out = {}
            for pi, p in enumerate(primes):
                while v % p == 0:
                    out[pi] = out.get(pi, 0) + 1
                    v //= p
            return out if v == 1 else None

        st = ref.NaturalQuaternionState.zero(len(primes))
        for mu in range(4):
            w = self.fractions(UNARY)[mu]
            if w == 0:
                continue                                 # R₊ = R₋ = 1
            plus, minus = (0, 1) if w > 0 else (1, 0)
            a, b = abs(w.numerator), w.denominator
            fb = smooth(b)
            if fb is None:
                raise ValueError(f"denominator {b} of {w} is not smooth over {primes}")
            for c in range(1, search + 1):
                fa, fc = smooth(a + c), smooth(c)
                if fa is not None and fc is not None:
                    break
            else:
                raise ValueError(f"{w} has no R₊ − R₋ encoding over {primes} within c ≤ {search}")
            for sign, num in ((plus, fa), (minus, fc)):
                for pi, k in num.items():
                    st.counts[mu, sign, NUM, pi] += k
                for pi, k in fb.items():
                    st.counts[mu, sign, DEN, pi] += k
        return st


def primes_up_to(m: int) -> tuple[int, ...]:
    out = []
    for k in range(2, max(m, 1) + 1):
        if all(k % p for p in out):
            out.append(k)
    return tuple(out) or (2,)


# ---- builders -----------------------------------------------------------------------------------
def bounded_state(bound: int, n: int = 0, o: int = 0, *, mu: int = 0, sign: int = 0) -> BoundedRationalQuaternionState:
    """the exponent (±) n/o on component μ; the empty ledger is W = 0."""
    return BoundedRationalQuaternionState.zero(bound).set(mu, sign, n, o)


def make_bounded_plus_one_state(bound: int) -> BoundedRationalQuaternionState:
    return bounded_state(bound, 1, 1)


def make_bounded_minus_one_state(bound: int) -> BoundedRationalQuaternionState:
    return bounded_state(bound, 1, 1, sign=1)


def bounded_network(units, bound: int | None = None, *, n_inputs: int | None = None) -> ref.SigmaProductNetwork:
    """a network whose exponents are bounded ledgers.  `bound` is m; it defaults to the number of
    input nodes (n_inputs, or the largest input index + 1).  It is a free parameter on purpose: with
    one input node m = 1 holds only the exponents 0 and ±1 (per component), so a one-input network
    wants a bound of its own — the number of layers you are willing to stack — rather than the
    input count.  Every exponent layer must carry a BoundedRationalQuaternionState with this bound."""
    if n_inputs is None:
        n_inputs = 1 + max(f.input_index for u in units for f in u.factors)
    if bound is None:
        bound = n_inputs
    for unit in units:
        for factor in unit.factors:
            if not 0 <= factor.input_index < n_inputs:
                raise ValueError(f"input_index {factor.input_index} outside the {n_inputs} inputs")
            for layer in factor.exponent_layers:
                st = layer.state
                if not isinstance(st, BoundedRationalQuaternionState):
                    raise TypeError("bounded_network needs BoundedRationalQuaternionState exponents")
                if st.bound != bound:
                    raise ValueError(f"state bound {st.bound} ≠ network bound {bound}")
    return ref.SigmaProductNetwork(primes=UNARY, units=list(units))


def make_bounded_power_factor(input_index, bound, n=1, o=1, *, mu=0, sign=0, side="L", reciprocal=False, name=""):
    return ref.make_power_factor(input_index, bounded_state(bound, n, o, mu=mu, sign=sign),
                                 side=side, reciprocal=reciprocal, name=name)


def make_bounded_alternating_seed(n_units: int = 4, bound: int = 1) -> ref.SigmaProductNetwork:
    """the reference's one-input alternating X / X⁻¹ seed with bounded ledgers (W = ±1 = 1/1).
    One input node: the bound is the seed's own parameter (bound = 1 would freeze W at 0, ±1)."""
    units = []
    for k in range(n_units):
        reciprocal = bool(k % 2)
        st = make_bounded_minus_one_state(bound) if reciprocal else make_bounded_plus_one_state(bound)
        units.append(ref.ProductUnit(
            factors=[ref.make_power_factor(0, st, side="L", reciprocal=reciprocal, name=f"branch{k}")],
            right_basis=(k % 4), name=f"unit{k}"))
    return bounded_network(units, bound, n_inputs=1)


def max_count_of(network: ref.SigmaProductNetwork) -> int | None:
    """the bound shared by the network's ledgers (None for a prime-valuation network)."""
    bounds = {layer.state.bound for *_, layer in network.iter_layers()
              if isinstance(layer.state, BoundedRationalQuaternionState)}
    if not bounds:
        return None
    if len(bounds) > 1:
        raise ValueError(f"mixed bounds {sorted(bounds)}")
    return bounds.pop()


# ---- the ledger as layers ------------------------------------------------------------------------
def unroll(network: ref.SigmaProductNetwork, primes: Sequence[int] | None = None) -> ref.SigmaProductNetwork:
    """
    Spell every bounded factor out as layers: for a real exponent n⁺/o⁺ − n⁻/o⁻ on one exponent layer,

        X^(n⁺/o⁺) · X^(−n⁻/o⁻)  =  (X^(1/o⁺))·…·(X^(1/o⁺)) · (X^(−1/o⁻))·…·(X^(−1/o⁻))
                                      └──── n⁺ slots ────┘   └──── n⁻ slots ─────┘

    (powers of one quaternion commute, so the ordered product of copies is the power).  The o-th root
    is one exponent layer 1/o in the reference's prime encoding, the numerator n is n product slots —
    the "layers" of the ledger, at most m of each.  Non-real exponents and stacked exponent layers are
    left as a single converted layer (to_natural).  Returns a prime-valuation network.
    """
    m = max_count_of(network)
    if m is None:
        return network.copy()
    primes = tuple(primes) if primes is not None else primes_up_to(m)
    units = []
    for unit in network.units:
        factors = []
        for factor in unit.factors:
            st = factor.exponent_layers[0].state if factor.exponent_layers else None
            simple = (len(factor.exponent_layers) == 1 and isinstance(st, BoundedRationalQuaternionState)
                      and not np.any(st.counts[1:]))
            if not simple:
                layers = [ref.ExponentLayer(l.state.to_natural(primes) if isinstance(l.state, BoundedRationalQuaternionState)
                                            else l.state.copy(), l.side, l.name) for l in factor.exponent_layers]
                factors.append(ref.Factor(factor.input_index, factor.reciprocal, layers, factor.name))
                continue
            side = factor.exponent_layers[0].side
            for sign in range(2):
                n, o = int(st.counts[0, sign, NUM, 0]), int(st.counts[0, sign, DEN, 0])
                if n == 0 or o == 0:
                    continue                                       # n/0 = 0 and 0/o = 0: no slots
                root = bounded_state(m, 1, o).to_natural(primes)  # the o-th root, W = 1/o
                for k in range(n):
                    factors.append(ref.Factor(factor.input_index, reciprocal=(factor.reciprocal != bool(sign)),
                                              exponent_layers=[ref.ExponentLayer(root.copy(), side, f"{factor.name}:root{o}")],
                                              name=f"{factor.name}:slot{sign}{k}"))
            if all(int(st.counts[0, s, NUM, 0]) == 0 or int(st.counts[0, s, DEN, 0]) == 0 for s in range(2)):
                # W = 0: the factor is 1; keep a placeholder so the unit is not empty
                factors.append(ref.Factor(factor.input_index, False,
                                          [ref.ExponentLayer(ref.NaturalQuaternionState.zero(len(primes)), side, f"{factor.name}:one")],
                                          factor.name))
        units.append(ref.ProductUnit(factors, unit.right_basis, unit.name))
    return ref.SigmaProductNetwork(primes=primes, units=units)


# ---- ledger moves: the finite lattice makes "any ledger of one channel" a legal neighbourhood -------------------
@dataclass(frozen=True)
class LedgerMove:
    """replace the ledger (n, o) of one (layer, component, sign) channel by another — the bounded lattice is finite
    ((m+1)² ledgers per channel), so the whole channel is a legal neighbourhood, where the reference's ±1 count
    steps are not: from n/o a single count step jumps to (n±1)/o or n/(o±1), and 1 → 2/3 needs two steps through
    a worse point (2/2 = 1 → 2/3) or through 0, which a greedy search never takes."""
    key: tuple[int, int, int]
    component: int
    sign: int
    n: int
    o: int


def apply_move(network: ref.SigmaProductNetwork, move):
    """ref.apply_move plus LedgerMove."""
    if isinstance(move, LedgerMove):
        cand = network.copy()
        layer = ref._get_layer(cand, move.key)
        if not isinstance(layer.state, BoundedRationalQuaternionState):
            raise TypeError("LedgerMove needs a bounded state")
        layer.state.set(move.component, move.sign, move.n, move.o)
        return cand
    return ref.apply_move(network, move)


def propose_ledger_moves(network, layer_keys, sensor, top_k_blocks: int = 4, include_side_flips: bool = True):
    """the gradient sensor selects (layer, component) blocks as in the reference; within a block every distinct
    ledger of both sign channels is proposed (fractions deduplicated: 1/1 and 2/2 are one move)."""
    score = sensor["score"]
    if score.size == 0:
        return []
    order = np.argsort(score.reshape(-1))[::-1]
    blocks = []
    for flat in order:
        li, mu = np.unravel_index(flat, score.shape)
        if score[li, mu] <= 0:
            break
        blocks.append((int(li), int(mu)))
        if len(blocks) >= top_k_blocks:
            break
    proposals = []
    for li, mu in blocks:
        key = layer_keys[li]
        st = ref._get_layer(network, key).state
        if not isinstance(st, BoundedRationalQuaternionState):
            continue
        m = st.bound
        for sign in range(2):
            cur = st.fraction(mu, sign)
            seen = {cur}
            for n in range(m + 1):
                for o in range(m + 1):
                    fr = total_fraction(n, o)
                    if fr in seen:
                        continue
                    seen.add(fr)
                    proposals.append(LedgerMove(key, mu, sign, n, o))
    if include_side_flips:
        layer_score = np.sqrt(np.sum(score * score, axis=-1))
        for li in np.argsort(layer_score)[::-1][:top_k_blocks]:
            key = layer_keys[int(li)]
            if ref._nonreal(ref._get_layer(network, key), network.primes):
                proposals.append(ref.SideFlipMove(key))
    return proposals
