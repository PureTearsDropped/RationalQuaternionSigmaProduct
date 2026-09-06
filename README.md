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
