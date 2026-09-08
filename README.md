# RationalQuaternionSigmaProduct

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22443844.svg)](https://doi.org/10.5281/zenodo.22443844)

A **rational-quaternion Σ-Product network** on total arithmetic: quaternion inputs, exponent layers stacked
in depth with exponents W ∈ ℍ_ℚ generated from natural-number prime valuations, ordered Hamilton products,
a fixed right basis, and a quaternion-linear Σ readout solved by least squares — with the structure found by
a gradient *sensor* and an exact discrete search — implemented on the value + flag arithmetic of
[total-arith-cuda](https://github.com/PureTearsDropped/total-arith-cuda) instead of IEEE.

    P_n = F_{n,1} F_{n,2} ⋯ F_{n,K_n}                       an ordered Hamilton product of factors
    F   = Pow_{s_L}( ⋯ Pow_{s_1}( X_j^{±1}, W_1 ) ⋯, W_L )   Pow_L(H, W) = Exp(W·Log H),  Pow_R(H, W) = Exp((Log H)·W)
    W   = w_0 + w_1 i + w_2 j + w_3 k,   w_μ = R_{μ,+} − R_{μ,−},   R = ∏_p p^{n_num − n_den},  n ∈ ℕ
    Z   = Σ_n A_n P_n e_{r_n}                                A_n ∈ ℍ by ridge least squares, e_r ∈ {1, i, j, k}

The addition and the multiplication live in different layers (a Σ-Π network); depth is in the exponent.  In ℂ the
exponent depth collapses (W₁W₂); in ℍ a second layer gives the sandwich Exp(W₁·Log X·W₂), which the non-commutativity
keeps — the network is the quaternion successor of
[ComplexSigmaProductUnit](https://github.com/PureTearsDropped/ComplexSigmaProductUnit).

## The two files

- `flexible_rational_quaternion_sigma_product.py` — **the reference**: the structure (dataclasses for the natural-number
  exponent state, the exponent layer, the factor, the product unit, the network), the NumPy/IEEE evaluation, the
  least-squares readout, the gradient-consensus sensor, the legal natural ±1 moves, the L/R flips and the compound
  move + flip candidates, and the search loop.  Design note (Japanese): `docs/flexible_rational_quaternion_sigma_product.txt`.
- `rational_quaternion_sigma_product.py` — **the same network on total arithmetic**.  It imports the reference's
  dataclasses, move generator, sensor and search bookkeeping unchanged and replaces only the arithmetic by the `Tot`
  (float32 value + uint8 flag: ⟦≥⟧ true at least the shown magnitude, ⟦≤⟧ at most, SUNK sign unknown) of total-arith-cuda:

  | | reference (IEEE, float64) | total (`Tot`) |
  |---|---|---|
  | Hamilton product | component formula | `group_mul` with the Cayley–Dickson wiring table cd4 (i·j = k), float64 accumulation, one saturation, the pattern flag rules |
  | X⁻¹ | conj/max(\|X\|², 1e-12) | conj/\|X\|² with a/0 = 0 ⟹ 0⁻¹ = 0 |
  | Log X | log max(\|X\|, 1e-12); vector part 0 when \|v\| < 1e-12 | L₀(\|X\|) + (v/\|v\|)·Arg₀(a, \|v\|): **L₀(0) = 0, Arg₀(0,0) = 0, direction(0) = 0** (the reserved words); ε = MIN⟦≤⟧ → log MIN⟦≥⟧ and its direction is kept; Log₀(−r) = log r + iπ by definition |
  | Exp U | exp clipped to ±80 | e^{U₀} saturates (MAX⟦≥⟧ / MIN⟦≤⟧, never 0); the rotation is exact for an exact angle, ⟦unknown⟧ for a flagged one |
  | Pow(0, W) | (1e-12)^W | Exp(W·Log₀ 0) = **1**: a zero input drops out of the product |
  | readout | least squares over all samples | least squares over the **usable** samples (every flag 0 or ⟦≤⟧ alone); the prediction is a `Tot` |
  | input | as given | the customs of `Tot(x)`: NaN → (0, unknown), \|x\| > f32 MAX → MAX⟦≥⟧, subnormal → ε |

  Nothing downstream produces NaN or Inf.  A sample the arithmetic cannot vouch for is excluded from the fit and from
  the search's loss; a silently clipped number never enters.  Pow(x, W): x is the base (the input), W the exponent.

`formula(network, coeff, names=…)` prints the network as an expression — the structure is symbolic by construction
(rational exponents, order, right basis); only the readout A is a float, and `snap=1e-3` shows a coefficient as a
fraction with ≈ when one with denominator ≤ 12 is that close (a marked decision, never silent):

    Z = (0.7 − 0.1i + 0.2j + 0.05k)·[Exp(Log X·(1 + 1/2i))]                        # demo 3, as found
    y = (0.7 − 0.1i + 0.2j + 0.05k)·[Exp((1/2)·Log ([X₁] + [X₂]))·X₃]              # two layers: layer 1's formula names layer 2's input

## Is it the reference's network?

`physics/parity_check.py` compares the features of 60 random structures (1–3 units, 1–3 factors, X or X⁻¹, 0–2
exponent layers L/R with random natural-number states, random right basis, 1–2 groups) entry by entry on the entries
the total arithmetic left unflagged (`physics/results/parity_check.txt`):

| exponent counts | reference guards | `Tot` values | unflagged entries | disagree (rel > 1e-3) | worst rel |
|---|---|---|---|---|---|
| ≤ 1 (max \|W\| 210) | on | float64 | 40 226 | 591 | 2.6e+31 |
| ≤ 1 | **off** | **float64** | 40 226 | **0** | 1.9e-09 |
| ≤ 1 | off | float32 (as shipped) | 40 226 | 105 | 2.4e-01 |
| ≤ 2 (max \|W\| 22 041) | off | float64 | 20 934 | **0** | 6.1e-10 |

With the reference's two unmarked guards switched off (exp clipped to ±80; log|q| and the vector direction floored at
1e-12) and the `Tot` values in float64, the two agree on every entry: the structure is the reference's.  With the
guards on, every disagreeing entry is a chained exponent layer whose first output fell below 1e-12 or whose amplitude
exponent passed 80 — where the reference clamps silently.  In float32 a further 0.3–1.4 % differ where |W| is in the
hundreds: a phase W·Log X of hundreds of radians loses its 1e-7 to float32 rounding.  That is precision, not structure.

## Bounded exponents: x^(n/o) with 0 ≤ n, o ≤ m

`bounded_exponent.py` is the other natural-number ledger for an exponent.  The reference's counts are prime
valuations (w = R₊ − R₋, R = ∏ p^(num−den): an unbounded lattice); here **the count is the numerator or the
denominator itself**,

    w_μ = n⁺_μ/o⁺_μ − n⁻_μ/o⁻_μ,        n, o ∈ {0, 1, …, m},        n/0 := 0,

with the total-arithmetic reading of the division (a/0 = 0 is the Moore–Penrose inverse of 0, so the empty ledger
0/0 is the exponent 0 and the factor 1).  m is the number of layers the exponent may use: a numerator n is n product
slots of the input, a denominator o is one o-th root, and `unroll(network)` spells a factor out that way —
X^(3/2) becomes three slots of X^(1/2), X^(2/3 − 1/2) becomes two slots of X^(1/3) and one of X^(−1/2) — with the
two evaluations agreeing to 1e-12.  The default bound is the number of input nodes; it is a free parameter because
**one input node with m = 1 holds only the exponents 0 and ±1**, so a one-input network chooses its own m.

The state has the reference's duck-type, so the evaluation, the gradient sensor, the ±1 move generator and both search
loops run on it unchanged; the network's prime tuple is `UNARY = (1,)` (one digit, the count itself) and
`apply_move`/`propose_moves` refuse a count above the bound.  `to_natural(primes)` writes any ledger as a
prime-valuation state (exact, bit-identical features: the bounded lattice is a finite subset of the reference's
lattice; every n/o with n, o ≤ m factors over the primes ≤ m).  `test_bounded_exponent.py`: the ledger semantics,
the bound as a free parameter, the moves confined to [0, m], bounded = prime state, the unroll, and the search
from X₀·X₁ to the teacher X₀^{1/2}·X₁² in two ledger moves on the reference and on `Tot` (with a NaN row and an
ε row in the data, excluded rather than fatal).

The exponent set at bound m is finite: m = 2 gives {0, ½, 1, 2} per channel, m = 3 gives {0, ⅓, ½, ⅔, 1, 3/2, 2, 3},
and the channel difference doubles it around 0.  The count in [0, m] is what the search moves ±1; the fraction is what
the layer evaluates.

### Coefficients on the lattice (a constant input node)

With A ≡ 1 the readout is gone, and a coefficient can only come from a constant input node e = (e, 0, 0, 0) and a
leading factor e^{W_n} = Exp(W_n · Log e) = Exp(W_n) in each product unit — W_n = Log A_n as a ledger.  With A ≡ 1
and no constant node the network is not universal: at X = 1 every product is its right basis, so Z(1) ∈ ℤ⁴.
`physics/lattice_coefficients.py` (teacher y = e^{1/2 + (2/3)i}·X₀^{1/2} + e^{−1 + (1/3)j}·X₁², m = 3 = the three
input nodes, 8 data seeds; `physics/results/lattice_coefficients.txt`):

| search | readout | moves | exact |
|---|---|---|---|
| A  reference | quaternion least squares | ±1 counts | **8/8** |
| B  coefficient on the lattice, A ≡ 1 | none | ±1 counts | 0/8 (stuck at MSE 8e-2 after 2 steps) |
| B′ same | none | any ledger of a channel (`LedgerMove`) | 0/8 (stuck at 4e-2 after 3 steps) |
| C, C′ direction on the lattice, magnitude real LS | real scalar LS | either | 0/8 |
| **D  LS proposes, the lattice decides** | LS → snap Log A to the nearest ledger → A ≡ 1 | ledger moves after the snap | **8/8, MSE 0** |

The lattice coefficient loses the variable projection: with A solved exactly for every exponent proposal (A), the
loss over exponents is clean, but with A on the lattice a wrong coefficient makes the best exponent wrong and a
greedy coordinate search stops in a coupled local minimum, whatever the neighbourhood.  What works is the pattern of
the complex unit's learning rule: the continuous fit is the proposal, the lattice is the decision, and the exact
evaluation after the snap verifies it (here MSE exactly 0: the found network *is* the teacher, with no float in it).
The empty ledger 0/0 is a plateau for ±1 count moves (every single step leaves the exponent 0), so a neutral seed is
the balanced ledger 1/1 − 1/1; ledger moves have no plateau.

**Off the lattice** (`physics/lattice_approximation.py`, `physics/lattice_ridge.py`; results in `physics/results/`):
targets that are not on any lattice — 1/(1+4(log x−1)²) on [1, e²], sin(3x)/x, log(x₁+x₂) on [1, 3]² — from a
log-Fourier seed X^{±ik}, k ≤ K, 2000 train / 2000 test samples, relative test MSE = MSE/Var(f):

| target | K (units) | m | A: LS readout | A with ridge 1 | D0: snap of the ridge-1 proposal | D: + ledger moves, A ≡ 1 |
|---|---|---|---|---|---|---|
| 1/(1+4(log x−1)²) | 4 (9) | 8 | 2.9e-5 (max\|A\| 8.7) | 7.0e-5 (max\|A\| 0.19) | 6.1e-3 | **8.6e-4** |
| same | 4 (9) | 12 | 2.9e-5 | 7.0e-5 | 5.6e-3 | **6.7e-4** (ridge 10: 3.4e-4) |
| log(x₁+x₂) | 2 (25) | 8 | 1.3e-10 (max\|A\| 2.2) | 5.2e-6 (max\|A\| 0.22) | 6.7e-3 | **2.0e-4** |
| same | 2 (25) | 12 | 1.3e-10 | 5.2e-6 | 2.4e-3 | **1.4e-4** |
| sin(3x)/x | 3 (7) | 8 | 6.4e-2 | 8.4e-2 | 8.9e-2 | 8.6e-2 (the seed's modes are the limit, not the lattice) |

Two things decide it.  **The proposal must not cancel**: the plain least-squares readout puts large coefficients
against each other (max|A| 8.7 for a target of size 1; 11.8 for sin(3x)/x), and rounding each to the lattice breaks
the cancellation — the snapped network is worse than the constant (relative MSE 19, 21).  A ridge that keeps |A| = O(1)
costs the continuous readout ×2–×40 and gives the lattice a proposal it can hold.  **The lattice has a floor**: the
Farey half-gap of n/o at bound m (0.062 nats at m = 8, 0.042 at m = 12) is the worst rounding of log|A| and of an
angle; the snap alone lands at a few 1e-3, the ledger moves after it reach 1e-4–1e-3, one to one-and-a-half orders
behind the ridge readout and three to five behind the unregularised one.  The lattice cannot use cancellation, which
is the same fact as "only a sparse rational solution is evidence": a fit that needs cancelling coefficients is not
on any lattice.

## The demos

`physics/demos.py` → `physics/results/demos.txt` (CPU, ~40 s):

1. y = x₁x₂ with no exponent layer: reference MSE 1.7e-30, total 9.5e-15 (the float32 floor), A = 1 both.
2. P = Pow_R(Pow_L(X₁, 1), 1 + ½i)·X₂, two exponent layers: A recovered to 5 digits, both.
3. The search from Pow_L(X, 1) to the teacher Pow_R(X, 1 + ½i): both accept the one compound move (natural move + L→R)
   and find the teacher; 0.1 s against 6 s (the flag logic is Python on the CPU).
4. The same search with three rows appended to the 1200 clean ones — a NaN (a missing measurement), a 1e40 (beyond
   float32 MAX) and an ε: the reference accepts five NaN "moves" and ends with A = NaN.  **4b**, without the NaN row:
   the reference accepts three moves toward a wrong structure whose MSE on the clean rows is 0.55 — the 1e40 row's
   feature was clipped inside exp to e^80 without a mark and dominated the least squares.  Total arithmetic excludes
   the 2–3 flagged rows, finds the teacher's structure and coefficients (MSE 9e-15 on the 1200 usable rows) and returns
   the excluded rows' predictions finite and flagged ⟦unknown⟧.

## Run

```bash
git clone https://github.com/PureTearsDropped/total-arith-cuda          # ≥ v1.2.0 (the arithmetic), next to this repository
git clone https://github.com/PureTearsDropped/RationalQuaternionSigmaProduct
cd RationalQuaternionSigmaProduct && pip install -r requirements.txt
python test_rational_quaternion_sigma_product.py    # reserved words, Hamilton table, agreement with the reference, the boundary
python test_bounded_exponent.py                      # x^(n/o), 0 ≤ n, o ≤ m: ledger, unroll, bounded search (reference and Tot)
python physics/lattice_coefficients.py               # coefficients on the lattice via a constant input node, 5 search modes
python physics/lattice_approximation.py              # off-lattice targets: bound m and unit sweeps (~4 min)
python physics/lattice_ridge.py                      # ridge on the proposal before the snap, the Farey floor (~15 min)
python physics/demos.py                              # the four demos, reference vs total
python physics/parity_check.py                       # 60 random structures, guards on/off, float32/float64 (~5 min)
python flexible_rational_quaternion_sigma_product.py # the reference's own demos (NumPy + PyTorch)
```

The environment variable `TOTAL_ARITH_CUDA` points to the arithmetic if it is not next to this repository.

## Related work

Product units (Durbin & Rumelhart 1989) and Σ-Π / Π-Σ networks separate addition from multiplication; Neural Power
Units (Heim, Pevný & Šmídl 2020) use the complex logarithm for negative inputs and complex exponents; **Quaternion
Product Units** (Zhang, Shi, Ni & Gu, CVPR 2020) power unit quaternions by real exponents and chain them by Hamilton
products for SO(3).  This network is that chain on general quaternions with quaternion (rational) exponents on either
side, exponent depth, a least-squares Σ readout, a discrete structure search — and an arithmetic in which the
boundaries are values with flags rather than clamps.  The name avoids "QPU", which is taken.

## Known limits

- Float rounding is not flagged (only totalization events are): after a saturated amplitude, a rotated component whose
  truth is 0 shows a rounding residue × MAX with ⟦≥⟧.  The sample's exclusion is right; that component's magnitude
  claim is not.  The complex unit shares this.
- W is rounded once to float32 (the reference rounds it to float64).  The fit floor is ~1e-14 in MSE.
- The flag logic runs in Python over the four output components; the search is ~50× slower than the reference on the CPU.
- `numpy.round(x, 3)` on float32 values near 1e38 overflows to inf — a display trap, not an arithmetic one.

License: 0BSD.  Cite with `CITATION.cff` (please record the commit ID and the total-arith-cuda version).  Archived on Zenodo:
concept DOI [10.5281/zenodo.22443844](https://doi.org/10.5281/zenodo.22443844) (all versions), v1.0.0 [10.5281/zenodo.22443845](https://doi.org/10.5281/zenodo.22443845).

⚠️ AI-assisted; verify. / 生成AI使用・要検証
