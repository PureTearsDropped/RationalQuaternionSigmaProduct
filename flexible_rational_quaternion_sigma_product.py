"""
Flexible Rational-Quaternion Sigma-Product Network
===================================================

This module generalizes the previous fixed 4-channel QCSPU into a configurable
ordered Sigma-Product architecture:

    inputs X_1, ..., X_J
        -> per-factor reciprocal selection
        -> zero or more rational-quaternion exponent layers
        -> ordered Hamilton product inside each Product Unit
        -> optional fixed right basis
        -> final quaternion-linear Sigma readout

A Product Unit n is

    P_n = F_{n,1} F_{n,2} ... F_{n,K_n}

where each factor is

    H_0 = X_{input_index}^{sigma},   sigma in {+1,-1}

    H_{ell+1} =
        Pow_L(H_ell, W_ell) = Exp(W_ell Log H_ell)
    or
        Pow_R(H_ell, W_ell) = Exp((Log H_ell) W_ell)

and every W_ell is a rational quaternion generated only from natural-number
prime-valuation states.

The output is

    Z = sum_n A_n P_n e_{r_n}

with quaternion A_n.  For fixed structure, A_n is solved by ridge least squares.

The structure search follows:

    exact LS fit of A
    -> backprop gradient only as a structure sensor
    -> select distorted (layer, quaternion-component) blocks
    -> enumerate legal natural +/-1 moves
    -> exact LS-refit every proposed structure
    -> accept only true train-loss improvement

L/R flips and natural-move + L/R-flip compound candidates are supported.

Input shapes:
    [N, I, 4]       : N samples, I quaternion inputs
    [N, G, I, 4]    : grouped/bin-wise data; readout A is independent per group

NumPy + optional PyTorch (required only for gradient-sensor search).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Literal, Sequence
import copy
import itertools
import math
import numpy as np

try:
    import torch
except Exception:
    torch = None

Side = Literal["L", "R"]

RIGHT_BASIS = np.eye(4, dtype=np.float64)


# ============================================================================
# Quaternion algebra: NumPy
# ============================================================================

def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = [a[..., i] for i in range(4)]
    bw, bx, by, bz = [b[..., i] for i in range(4)]
    return np.stack(
        [
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ],
        axis=-1,
    )


def qconj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1.0
    return q


def qnorm2(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.sum(q*q, axis=-1, keepdims=True)


def qinv(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return qconj(q) / np.maximum(qnorm2(q), eps)


def qlog(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Principal quaternion logarithm.

        Log(a+v) = log|q| + v/|v| * atan2(|v|, a)

    Near v=0, vector part is set to zero.
    """
    q = np.asarray(q, dtype=np.float64)
    a = q[..., 0]
    v = q[..., 1:]
    vr = np.linalg.norm(v, axis=-1)
    n = np.sqrt(a*a + vr*vr)
    theta = np.arctan2(vr, a)

    scale = np.zeros_like(vr)
    mask = vr > eps
    scale[mask] = theta[mask] / vr[mask]

    return np.concatenate(
        [np.log(np.maximum(n, eps))[..., None], v*scale[..., None]],
        axis=-1,
    )


def qexp(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    a = q[..., 0]
    v = q[..., 1:]
    vr = np.linalg.norm(v, axis=-1)
    ea = np.exp(np.clip(a, -80.0, 80.0))

    sinc = np.ones_like(vr)
    mask = vr > 1e-10
    sinc[mask] = np.sin(vr[mask]) / vr[mask]
    r = vr[~mask]
    sinc[~mask] = 1.0 - r*r/6.0 + r**4/120.0

    return np.concatenate(
        [(ea*np.cos(vr))[..., None],
         ea[..., None]*v*sinc[..., None]],
        axis=-1,
    )


def qpow(x: np.ndarray, w: np.ndarray, side: Side) -> np.ndarray:
    lx = qlog(x)
    wb = np.broadcast_to(np.asarray(w, dtype=np.float64), lx.shape)
    if side == "L":
        return qexp(qmul(wb, lx))
    if side == "R":
        return qexp(qmul(lx, wb))
    raise ValueError(f"unknown side: {side!r}")


def left_multiply_design(phi: np.ndarray) -> np.ndarray:
    """
    M(phi) such that A*phi = M(phi) @ [a,b,c,d].
    """
    w, x, y, z = [phi[..., i] for i in range(4)]
    return np.stack(
        [
            np.stack([w, -x, -y, -z], axis=-1),
            np.stack([x,  w,  z, -y], axis=-1),
            np.stack([y, -z,  w,  x], axis=-1),
            np.stack([z,  y, -x,  w], axis=-1),
        ],
        axis=-2,
    )


# ============================================================================
# Quaternion algebra: PyTorch (gradient sensor only)
# ============================================================================

def _need_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for gradient-sensor search")


def tqmul(a, b):
    aw, ax, ay, az = [a[..., i] for i in range(4)]
    bw, bx, by, bz = [b[..., i] for i in range(4)]
    return torch.stack(
        [
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ],
        dim=-1,
    )


def tqconj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def tqinv(q, eps: float = 1e-12):
    return tqconj(q) / torch.clamp(torch.sum(q*q, dim=-1, keepdim=True), min=eps)


def tqlog(q, eps: float = 1e-12):
    a = q[..., 0]
    v = q[..., 1:]
    vr = torch.linalg.vector_norm(v, dim=-1)
    n = torch.sqrt(a*a + vr*vr)
    theta = torch.atan2(vr, a)
    scale = torch.where(vr > eps, theta/torch.clamp(vr, min=eps), torch.zeros_like(vr))
    return torch.cat(
        [torch.log(torch.clamp(n, min=eps))[..., None],
         v*scale[..., None]],
        dim=-1,
    )


def tqexp(q):
    a = q[..., 0]
    v = q[..., 1:]
    vr = torch.linalg.vector_norm(v, dim=-1)
    ea = torch.exp(torch.clamp(a, -80.0, 80.0))
    safe = torch.clamp(vr, min=1e-10)
    sinc = torch.where(
        vr > 1e-10,
        torch.sin(vr)/safe,
        1.0 - vr*vr/6.0 + vr**4/120.0,
    )
    return torch.cat(
        [(ea*torch.cos(vr))[..., None],
         ea[..., None]*v*sinc[..., None]],
        dim=-1,
    )


def tqpow(x, w, side: Side):
    lx = tqlog(x)
    wb = w
    while wb.ndim < lx.ndim:
        wb = wb.unsqueeze(0)
    wb = wb.expand_as(lx)
    if side == "L":
        return tqexp(tqmul(wb, lx))
    if side == "R":
        return tqexp(tqmul(lx, wb))
    raise ValueError(side)


# ============================================================================
# Natural-number state -> rational quaternion exponent
# ============================================================================

@dataclass
class NaturalQuaternionState:
    """
    counts[mu, sign, leg, p]

      mu   = 0,1,2,3  -> scalar, i, j, k
      sign = 0,1      -> R_plus, R_minus
      leg  = 0,1      -> prime numerator count, denominator count
      p                  prime index

    R(sign) = product_p p^(num-den)
    w_mu    = R_plus - R_minus
    W       = w0 + w1*i + w2*j + w3*k

    Every stored count is a natural integer.
    """
    counts: np.ndarray

    def __post_init__(self):
        a = np.asarray(self.counts)
        if a.ndim != 4 or a.shape[:3] != (4, 2, 2):
            raise ValueError("counts must have shape [4,2,2,n_primes]")
        if np.any(a < 0):
            raise ValueError("all internal states must be natural numbers")
        self.counts = a.astype(np.int64, copy=True)

    @property
    def n_primes(self) -> int:
        return int(self.counts.shape[-1])

    @classmethod
    def zero(cls, n_primes: int) -> "NaturalQuaternionState":
        # R_plus = R_minus = 1 for every component, hence W=0.
        return cls(np.zeros((4, 2, 2, n_primes), dtype=np.int64))

    def copy(self) -> "NaturalQuaternionState":
        return NaturalQuaternionState(self.counts.copy())

    def _positive_rational(self, mu: int, sign: int, primes: Sequence[int]) -> Fraction:
        if len(primes) != self.n_primes:
            raise ValueError("prime count mismatch")
        out = Fraction(1, 1)
        for pi, p in enumerate(primes):
            d = int(self.counts[mu, sign, 0, pi]) - int(self.counts[mu, sign, 1, pi])
            if d >= 0:
                out *= Fraction(int(p)**d, 1)
            else:
                out *= Fraction(1, int(p)**(-d))
        return out

    def channels(self, primes: Sequence[int]):
        return tuple(
            (self._positive_rational(mu, 0, primes),
             self._positive_rational(mu, 1, primes))
            for mu in range(4)
        )

    def fractions(self, primes: Sequence[int]):
        return tuple(rp-rm for rp, rm in self.channels(primes))

    def exponent(self, primes: Sequence[int]) -> np.ndarray:
        return np.array([float(v) for v in self.fractions(primes)], dtype=np.float64)


def make_scalar_plus_one_state(primes=(2,3,5,7)) -> NaturalQuaternionState:
    """
    Natural-number encoding of W=+1:
      scalar: R+ = 2, R- = 1 => 1
      imaginary components: 1-1 => 0
    """
    primes = tuple(primes)
    if 2 not in primes:
        raise ValueError("this convenience seed needs prime 2")
    st = NaturalQuaternionState.zero(len(primes))
    p2 = primes.index(2)
    st.counts[0, 0, 0, p2] += 1
    return st


def make_scalar_minus_one_state(primes=(2,3,5,7)) -> NaturalQuaternionState:
    """
    Natural-number encoding of W=-1:
      scalar: R+ = 1, R- = 2 => -1
    """
    primes = tuple(primes)
    if 2 not in primes:
        raise ValueError("this convenience seed needs prime 2")
    st = NaturalQuaternionState.zero(len(primes))
    p2 = primes.index(2)
    st.counts[0, 1, 0, p2] += 1
    return st


# ============================================================================
# Configurable architecture
# ============================================================================

@dataclass
class ExponentLayer:
    state: NaturalQuaternionState
    side: Side = "L"
    name: str = ""

    def copy(self):
        return ExponentLayer(self.state.copy(), self.side, self.name)


@dataclass
class Factor:
    """
    One ordered factor inside a Product Unit.

    Starts from one selected input X[input_index].
    If reciprocal=True, starts from X^{-1}.
    Then applies exponent_layers sequentially.
    """
    input_index: int
    reciprocal: bool = False
    exponent_layers: list[ExponentLayer] = field(default_factory=list)
    name: str = ""

    def copy(self):
        return Factor(
            self.input_index,
            self.reciprocal,
            [x.copy() for x in self.exponent_layers],
            self.name,
        )


@dataclass
class ProductUnit:
    """
    Ordered Hamilton product of factors:
        P = F_1 F_2 ... F_K

    right_basis:
        None or 0/1/2/3 for multiplication by 1/i/j/k after the product.
    """
    factors: list[Factor]
    right_basis: int | None = None
    name: str = ""

    def __post_init__(self):
        if not self.factors:
            raise ValueError("a ProductUnit needs at least one factor")
        if self.right_basis not in (None, 0, 1, 2, 3):
            raise ValueError("right_basis must be None or 0/1/2/3")

    def copy(self):
        return ProductUnit(
            [f.copy() for f in self.factors],
            self.right_basis,
            self.name,
        )


@dataclass
class SigmaProductNetwork:
    primes: tuple[int, ...]
    units: list[ProductUnit]

    def __post_init__(self):
        if not self.units:
            raise ValueError("network needs at least one ProductUnit")
        for unit in self.units:
            for factor in unit.factors:
                for layer in factor.exponent_layers:
                    if layer.state.n_primes != len(self.primes):
                        raise ValueError("state/prime length mismatch")

    def copy(self):
        return SigmaProductNetwork(
            tuple(self.primes),
            [u.copy() for u in self.units],
        )

    def iter_layers(self):
        """
        Yields:
            (unit_index, factor_index, layer_index, layer)
        """
        for ui, unit in enumerate(self.units):
            for fi, factor in enumerate(unit.factors):
                for li, layer in enumerate(factor.exponent_layers):
                    yield ui, fi, li, layer


# ============================================================================
# Forward evaluation
# ============================================================================

def _as_grouped_inputs(x: np.ndarray):
    """
    [N,I,4]    -> [N,1,I,4], grouped=False
    [N,G,I,4]  -> unchanged, grouped=True
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3 and x.shape[-1] == 4:
        return x[:, None, :, :], False
    if x.ndim == 4 and x.shape[-1] == 4:
        return x, True
    raise ValueError("x must have shape [N,I,4] or [N,G,I,4]")


def _as_grouped_targets(y: np.ndarray):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[-1] == 4:
        return y[:, None, :], False
    if y.ndim == 3 and y.shape[-1] == 4:
        return y, True
    raise ValueError("y must have shape [N,4] or [N,G,4]")


def eval_factor_numpy(xg: np.ndarray, factor: Factor, primes: Sequence[int]) -> np.ndarray:
    if factor.input_index < 0 or factor.input_index >= xg.shape[2]:
        raise IndexError(f"input_index {factor.input_index} out of range")
    h = xg[:, :, factor.input_index, :]
    if factor.reciprocal:
        h = qinv(h)
    for layer in factor.exponent_layers:
        h = qpow(h, layer.state.exponent(primes), layer.side)
    return h


def eval_unit_numpy(xg: np.ndarray, unit: ProductUnit, primes: Sequence[int]) -> np.ndarray:
    factors = [eval_factor_numpy(xg, f, primes) for f in unit.factors]
    out = factors[0]
    for nxt in factors[1:]:
        out = qmul(out, nxt)

    if unit.right_basis is not None:
        e = np.broadcast_to(RIGHT_BASIS[unit.right_basis], out.shape)
        out = qmul(out, e)
    return out


def features_numpy(x: np.ndarray, network: SigmaProductNetwork):
    xg, grouped = _as_grouped_inputs(x)
    phi = np.stack(
        [eval_unit_numpy(xg, unit, network.primes) for unit in network.units],
        axis=2,
    )  # [N,G,M,4]
    return phi, grouped


# ============================================================================
# Exact quaternion-linear Sigma readout
# ============================================================================

def fit_readout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    network: SigmaProductNetwork,
    ridge: float = 1e-8,
):
    xg, grouped_x = _as_grouped_inputs(x_train)
    yg, grouped_y = _as_grouped_targets(y_train)

    if xg.shape[:2] != yg.shape[:2]:
        raise ValueError("x/y sample or group mismatch")

    phi, _ = features_numpy(x_train, network)        # [N,G,M,4]
    mats = left_multiply_design(phi)                 # [N,G,M,4out,4coef]
    n, g, m = phi.shape[:3]

    D = np.transpose(mats, (1,0,3,2,4)).reshape(g, n*4, m*4)
    yy = np.transpose(yg, (1,0,2)).reshape(g, n*4)

    Dt = np.transpose(D, (0,2,1))
    G = Dt @ D + ridge*np.eye(m*4)[None, :, :]
    h = Dt @ yy[..., None]
    coeff = np.linalg.solve(G, h)[..., 0].reshape(g, m, 4)

    if not grouped_x and not grouped_y:
        return coeff[0]
    return coeff


def predict(
    x: np.ndarray,
    network: SigmaProductNetwork,
    coeff: np.ndarray,
):
    xg, grouped = _as_grouped_inputs(x)
    phi, _ = features_numpy(x, network)
    mats = left_multiply_design(phi)
    n, g, m = phi.shape[:3]
    D = np.transpose(mats, (1,0,3,2,4)).reshape(g, n*4, m*4)

    c = np.asarray(coeff, dtype=np.float64)
    if c.ndim == 2:
        c = c[None, :, :]
    if c.shape != (g, m, 4):
        raise ValueError(f"coeff must have shape {(g,m,4)} or {(m,4)}")

    yp = (D @ c.reshape(g, m*4, 1))[..., 0]
    yp = np.transpose(yp.reshape(g, n, 4), (1,0,2))
    return yp[:, 0, :] if not grouped else yp


def exact_fit_mse(
    x_train: np.ndarray,
    y_train: np.ndarray,
    network: SigmaProductNetwork,
    ridge: float = 1e-8,
):
    coeff = fit_readout(x_train, y_train, network, ridge)
    pred = predict(x_train, network, coeff)
    mse = float(np.mean((pred - np.asarray(y_train, dtype=np.float64))**2))
    return mse, coeff


# ============================================================================
# Gradient sensor for arbitrary exponent-layer layouts
# ============================================================================

def _torch_group_inputs(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x[:, None, :, :]
    return torch.tensor(x, dtype=torch.float64)


def _torch_group_targets(y):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2:
        y = y[:, None, :]
    return torch.tensor(y, dtype=torch.float64)


def _torch_eval_factor(xg, factor, wvars, layer_keys):
    h = xg[:, :, factor.input_index, :]
    if factor.reciprocal:
        h = tqinv(h)

    for li, layer in enumerate(factor.exponent_layers):
        key = layer_keys[li]
        h = tqpow(h, wvars[key], layer.side)
    return h


def _torch_features(xg, network: SigmaProductNetwork, wvars):
    outs = []
    for ui, unit in enumerate(network.units):
        factors = []
        for fi, factor in enumerate(unit.factors):
            keys = [(ui,fi,li) for li in range(len(factor.exponent_layers))]
            factors.append(_torch_eval_factor(xg, factor, wvars, keys))

        out = factors[0]
        for nxt in factors[1:]:
            out = tqmul(out, nxt)

        if unit.right_basis is not None:
            e = torch.tensor(
                RIGHT_BASIS[unit.right_basis],
                dtype=xg.dtype,
                device=xg.device,
            )
            while e.ndim < out.ndim:
                e = e.unsqueeze(0)
            out = tqmul(out, e.expand_as(out))
        outs.append(out)
    return torch.stack(outs, dim=2)


def effective_w_chunk_gradients(
    x_train,
    y_train,
    network: SigmaProductNetwork,
    coeff,
    n_chunks: int = 8,
):
    """
    Gradients dL_chunk/dW for every exponent layer.

    Returned:
        keys: list[(unit,factor,layer)]
        grads: [B, L, 4]
    """
    _need_torch()

    keys = [(ui,fi,li) for ui,fi,li,_ in network.iter_layers()]
    if not keys:
        return keys, np.zeros((0,0,4), dtype=np.float64)

    base_w = {}
    for ui,fi,li,layer in network.iter_layers():
        base_w[(ui,fi,li)] = layer.state.exponent(network.primes)

    xg = _torch_group_inputs(x_train)
    yg = _torch_group_targets(y_train)

    c = np.asarray(coeff, dtype=np.float64)
    if c.ndim == 2:
        c = c[None,:,:]
    ct = torch.tensor(c, dtype=torch.float64)

    n = xg.shape[0]
    chunks = [idx for idx in np.array_split(np.arange(n), min(n_chunks,n)) if len(idx)]
    out = []

    for idx in chunks:
        wvars = {
            k: torch.tensor(base_w[k], dtype=torch.float64, requires_grad=True)
            for k in keys
        }

        ph = _torch_features(xg[idx], network, wvars)  # [Nc,G,M,4]

        # Build A_m * phi_m explicitly.
        pred = torch.zeros_like(yg[idx])
        for m in range(len(network.units)):
            A = ct[:,m,:]  # [G,4]
            while A.ndim < ph[:,:,m,:].ndim:
                A = A.unsqueeze(0)
            pred = pred + tqmul(A.expand_as(ph[:,:,m,:]), ph[:,:,m,:])

        loss = torch.mean((pred - yg[idx])**2)
        loss.backward()

        out.append(
            np.stack([wvars[k].grad.detach().cpu().numpy() for k in keys], axis=0)
        )

    return keys, np.stack(out, axis=0)


def gradient_consensus_sensor(grads: np.ndarray, eps: float = 1e-15):
    """
    grads[B,L,4].

    C = |sum_b g_b| / (sum_b |g_b| + eps)
    score = C * RMS(g_b)
    """
    if grads.size == 0:
        return dict(consensus=np.zeros((0,4)), rms=np.zeros((0,4)), score=np.zeros((0,4)))

    abs_sum = np.sum(np.abs(grads), axis=0)
    consensus = np.abs(np.sum(grads, axis=0)) / (abs_sum + eps)
    rms = np.sqrt(np.mean(grads*grads, axis=0))
    score = consensus * rms
    return dict(consensus=consensus, rms=rms, score=score)


# ============================================================================
# Generic exact discrete structure search
# ============================================================================

@dataclass(frozen=True)
class NaturalMove:
    key: tuple[int,int,int]       # unit, factor, layer
    component: int               # scalar/i/j/k
    sign: int                    # plus/minus rational channel
    leg: int                     # numerator/denominator
    prime_index: int
    delta: int


@dataclass(frozen=True)
class SideFlipMove:
    key: tuple[int,int,int]


@dataclass(frozen=True)
class NaturalSideFlipMove:
    natural: NaturalMove


@dataclass
class SearchRecord:
    iteration: int
    train_mse: float
    accepted_move: object | None
    layer_exponents: dict
    layer_sides: dict


def _get_layer(network: SigmaProductNetwork, key):
    ui,fi,li = key
    return network.units[ui].factors[fi].exponent_layers[li]


def _network_key(network: SigmaProductNetwork):
    out = []
    for ui,fi,li,layer in network.iter_layers():
        out.append((
            ui,fi,li,
            tuple(layer.state.fractions(network.primes)),
            layer.side,
        ))
    return tuple(out)


def _nonreal(layer: ExponentLayer, primes):
    w = layer.state.exponent(primes)
    return bool(np.any(np.abs(w[1:]) > 1e-15))


def apply_move(network: SigmaProductNetwork, move):
    cand = network.copy()

    if isinstance(move, NaturalMove):
        layer = _get_layer(cand, move.key)
        cur = int(layer.state.counts[
            move.component, move.sign, move.leg, move.prime_index
        ])
        nxt = cur + move.delta
        if nxt < 0:
            raise ValueError("illegal negative natural state")
        layer.state.counts[
            move.component, move.sign, move.leg, move.prime_index
        ] = nxt
        return cand

    if isinstance(move, SideFlipMove):
        layer = _get_layer(cand, move.key)
        layer.side = "R" if layer.side == "L" else "L"
        return cand

    if isinstance(move, NaturalSideFlipMove):
        cand = apply_move(cand, move.natural)
        layer = _get_layer(cand, move.natural.key)
        layer.side = "R" if layer.side == "L" else "L"
        return cand

    raise TypeError(type(move))


def propose_moves(
    network: SigmaProductNetwork,
    layer_keys,
    sensor,
    top_k_blocks: int = 4,
    max_count: int | None = 6,
    include_side_flips: bool = True,
    include_compound: bool = True,
):
    """
    Gradient selects only (layer, quaternion-component) blocks.

    Within selected blocks, ALL legal natural +/-1 moves are exact-tested.
    This avoids trusting raw prime-coordinate gradient magnitudes.
    """
    score = sensor["score"]
    if score.size == 0:
        return []

    order = np.argsort(score.reshape(-1))[::-1]
    blocks = []
    for flat in order:
        li, mu = np.unravel_index(flat, score.shape)
        if score[li,mu] <= 0:
            break
        blocks.append((li,mu))
        if len(blocks) >= top_k_blocks:
            break

    proposals = []

    for li,mu in blocks:
        key = layer_keys[li]
        layer = _get_layer(network, key)

        for sign in range(2):
            for leg in range(2):
                for pi in range(len(network.primes)):
                    cur = int(layer.state.counts[mu,sign,leg,pi])
                    for delta in (-1,+1):
                        nxt = cur + delta
                        if nxt < 0:
                            continue
                        if max_count is not None and nxt > max_count:
                            continue
                        nm = NaturalMove(key,mu,sign,leg,pi,delta)
                        proposals.append(nm)

                        if include_compound and mu > 0:
                            cand = apply_move(network, nm)
                            if _nonreal(_get_layer(cand,key), cand.primes):
                                proposals.append(NaturalSideFlipMove(nm))

    if include_side_flips:
        # Test direct side flips for high-score non-real layers.
        layer_score = np.sqrt(np.sum(score*score, axis=-1))
        for li in np.argsort(layer_score)[::-1][:top_k_blocks]:
            key = layer_keys[int(li)]
            layer = _get_layer(network, key)
            if _nonreal(layer, network.primes):
                proposals.append(SideFlipMove(key))

    return proposals


def search_structure(
    x_train,
    y_train,
    initial_network: SigmaProductNetwork,
    *,
    ridge: float = 1e-8,
    max_steps: int = 20,
    n_chunks: int = 8,
    top_k_blocks: int = 4,
    max_count: int | None = 6,
    min_improvement: float = 1e-12,
    verbose: bool = False,
):
    """
    Generic gradient-sensor + exact discrete search over ANY configured
    exponent-layer layout.

    Input selection, reciprocal flags, Product Unit factor ordering, number of
    Product Units, and number of exponent layers are specified by the network
    config and can be freely changed before calling this function.
    """
    net = initial_network.copy()
    mse, coeff = exact_fit_mse(x_train, y_train, net, ridge)
    history = []

    for it in range(max_steps+1):
        history.append(
            SearchRecord(
                iteration=it,
                train_mse=mse,
                accepted_move=None,
                layer_exponents={
                    (ui,fi,li): layer.state.fractions(net.primes)
                    for ui,fi,li,layer in net.iter_layers()
                },
                layer_sides={
                    (ui,fi,li): layer.side
                    for ui,fi,li,layer in net.iter_layers()
                },
            )
        )

        if it == max_steps:
            break

        keys, grads = effective_w_chunk_gradients(
            x_train, y_train, net, coeff, n_chunks
        )
        if not keys:
            break

        sensor = gradient_consensus_sensor(grads)
        proposals = propose_moves(
            net,
            keys,
            sensor,
            top_k_blocks=top_k_blocks,
            max_count=max_count,
        )

        seen = {_network_key(net)}
        exact = []

        for move in proposals:
            cand = apply_move(net, move)
            key = _network_key(cand)
            if key in seen:
                continue
            seen.add(key)

            cmse, ccoeff = exact_fit_mse(x_train, y_train, cand, ridge)
            exact.append((cmse, move, cand, ccoeff))

        if not exact:
            break

        exact.sort(key=lambda x: x[0])
        best_mse, best_move, best_net, best_coeff = exact[0]

        if best_mse >= mse - min_improvement:
            break

        history[-1].accepted_move = best_move
        net = best_net
        mse = best_mse
        coeff = best_coeff

        if verbose:
            print(
                f"iter={it:02d} mse={mse:.6e} move={best_move}"
            )

    return net, coeff, history


# ============================================================================
# Convenience architecture builders
# ============================================================================

def make_passthrough_factor(input_index: int, reciprocal: bool = False, name: str = ""):
    """
    No exponent layer:
        F = X_j
    or
        F = X_j^{-1}

    This is useful for exact products such as X1*X2.
    """
    return Factor(
        input_index=input_index,
        reciprocal=reciprocal,
        exponent_layers=[],
        name=name,
    )


def make_power_factor(
    input_index: int,
    exponent_state: NaturalQuaternionState,
    *,
    side: Side = "L",
    reciprocal: bool = False,
    name: str = "",
):
    return Factor(
        input_index=input_index,
        reciprocal=reciprocal,
        exponent_layers=[
            ExponentLayer(exponent_state.copy(), side=side, name=f"{name}:exp0")
        ],
        name=name,
    )


def make_alternating_linear_seed(
    n_units: int = 4,
    primes=(2,3,5,7),
):
    """
    One-input alternating X / X^-1 architecture.

    Unit m starts from:
        X      with W=+1 for even m
        X^-1   with W=-1 for odd m

    Hence every branch initially equals X before the fixed right basis:
        X, Xi, Xj, Xk

    for n_units=4.
    """
    primes = tuple(primes)
    units = []

    for m in range(n_units):
        reciprocal = bool(m % 2)
        state = (
            make_scalar_minus_one_state(primes)
            if reciprocal
            else make_scalar_plus_one_state(primes)
        )

        units.append(
            ProductUnit(
                factors=[
                    make_power_factor(
                        0,
                        state,
                        side="L",
                        reciprocal=reciprocal,
                        name=f"branch{m}",
                    )
                ],
                right_basis=(m % 4),
                name=f"unit{m}",
            )
        )

    return SigmaProductNetwork(primes=primes, units=units)


# ============================================================================
# Examples / smoke tests
# ============================================================================

def demo_exact_x1_times_x2(seed: int = 10):
    """
    Demonstrates exact representation of:
        y = x1 * x2

    There are no exponent layers at all.
    """
    rng = np.random.default_rng(seed)
    n = 500
    x1 = rng.normal(size=(n,4))
    x2 = rng.normal(size=(n,4))
    x = np.stack([x1,x2], axis=1)

    y = qmul(x1,x2)

    net = SigmaProductNetwork(
        primes=(2,3,5,7),
        units=[
            ProductUnit(
                factors=[
                    make_passthrough_factor(0, name="x1"),
                    make_passthrough_factor(1, name="x2"),
                ],
                name="x1_times_x2",
            )
        ],
    )

    coeff = fit_readout(x,y,net,ridge=1e-12)
    pred = predict(x,net,coeff)
    mse = np.mean((pred-y)**2)

    print("demo_exact_x1_times_x2")
    print("MSE:", mse)
    print("readout A:", coeff)


def demo_free_depth(seed: int = 11):
    """
    One factor with TWO exponent layers, multiplied by a second raw input:

        F1 = Pow_R(Pow_L(X1, +1), 1 + 1/2 i)
        F2 = X2
        P  = F1 * F2
        Z  = A * P

    Shows that exponent depth is arbitrary.
    """
    rng = np.random.default_rng(seed)
    n = 700
    x = rng.normal(size=(n,2,4))
    x[:,0,0] += 1.5
    x[:,1,0] += 1.2

    primes = (2,3,5,7)

    w1 = make_scalar_plus_one_state(primes)
    w2 = make_scalar_plus_one_state(primes)

    # From zero i-component, one natural move:
    # R_minus(i) = 1/2 -> i-component = 1 - 1/2 = +1/2.
    p2 = primes.index(2)
    w2.counts[1,1,1,p2] += 1

    factor1 = Factor(
        input_index=0,
        exponent_layers=[
            ExponentLayer(w1, "L", "layer0"),
            ExponentLayer(w2, "R", "layer1"),
        ],
        name="deep_x1",
    )
    factor2 = make_passthrough_factor(1, name="x2")

    net = SigmaProductNetwork(
        primes=primes,
        units=[
            ProductUnit([factor1,factor2], name="deep_product")
        ],
    )

    phi,_ = features_numpy(x,net)
    teacher_A = np.array([0.8,-0.2,0.15,0.1])
    y = qmul(
        np.broadcast_to(teacher_A, phi[:,0,0,:].shape),
        phi[:,0,0,:],
    )

    coeff = fit_readout(x,y,net,ridge=1e-12)
    pred = predict(x,net,coeff)
    mse = np.mean((pred-y)**2)

    print("demo_free_depth")
    print("MSE:", mse)
    print("layer exponents:")
    for key, layer in [
        ((0,0,li), layer)
        for li, layer in enumerate(net.units[0].factors[0].exponent_layers)
    ]:
        print(" ", key, layer.state.fractions(primes), layer.side)


def demo_gradient_search(seed: int = 12):
    """
    Search demonstration on a configurable one-input, one-unit architecture.

    Start:
        Pow_L(X, 1)

    Teacher:
        Pow_R(X, 1 + 1/2 i)

    The target exponent is one legal natural-number move away in the i-component,
    but the correct structure also requires L->R.  Compound exact candidates allow
    the search to cross that barrier in one accepted step.
    """
    if torch is None:
        print("PyTorch unavailable; skipping gradient-search demo")
        return

    rng = np.random.default_rng(seed)
    n = 1200

    x0 = rng.normal(size=(n,4))
    x0[:,0] += 1.7
    x = x0[:,None,:]

    primes = (2,3,5,7)
    init = make_scalar_plus_one_state(primes)

    target = init.copy()
    p2 = primes.index(2)
    target.counts[1,1,1,p2] += 1   # +1/2 i

    teacher_net = SigmaProductNetwork(
        primes=primes,
        units=[
            ProductUnit(
                [
                    make_power_factor(
                        0,target,side="R",name="teacher"
                    )
                ]
            )
        ],
    )

    teacher_A = np.array([[0.7,-0.1,0.2,0.05]])
    y = predict(x,teacher_net,teacher_A)

    init_net = SigmaProductNetwork(
        primes=primes,
        units=[
            ProductUnit(
                [
                    make_power_factor(
                        0,init,side="L",name="search"
                    )
                ]
            )
        ],
    )

    found, coeff, hist = search_structure(
        x,y,init_net,
        ridge=1e-10,
        max_steps=5,
        n_chunks=6,
        top_k_blocks=4,
        verbose=True,
    )

    pred = predict(x,found,coeff)
    mse = np.mean((pred-y)**2)

    print("demo_gradient_search")
    print("final MSE:", mse)
    for key, layer in [
        ((ui,fi,li), layer)
        for ui,fi,li,layer in found.iter_layers()
    ]:
        print(" ", key, layer.state.fractions(primes), layer.side)


if __name__ == "__main__":
    demo_exact_x1_times_x2()
    print()
    demo_free_depth()
    print()
    demo_gradient_search()
