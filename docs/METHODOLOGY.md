# Methodology

Notation matches `paper/main.tex` exactly. Every variable in code maps to one
symbol here.

## Trajectory

A question `q` induces a reasoning trajectory

```
𝒯 = (s_1, s_2, …, s_n, a)
```

where `s_i` is the i-th reasoning step and `a` is the final answer. In code: a
`Trace` (`src/data/loaders.py`) with `steps = [s_1 … s_n]`, `n_steps = n`.

## Outcome and onset

- `y ∈ {0, 1}` — answer incorrectness (`has_error` in dataframes). `y = 1`
  when the final answer disagrees with gold.
- `e ∈ {1, …, n} ∪ {∅}` — first erroneous step (`first_error_idx`; `None`
  for `∅`, i.e. error-free). 1-indexed, matching ProcessBench annotation.
- **Relative onset** `ρ = e / n ∈ (0, 1]`: `ρ = 0.3` means the error begins
  30% into the chain regardless of `n`. Computed by P4
  (`src/data/preprocessing.py::apply_p4_relative_onset`).

## Detector

`𝒯_{1:k}` is the prefix of the first `k` steps. A detector

```
f_θ : 𝒯_{1:k} ↦ r_k ∈ [0,1],   r_k ≈ P(y = 1 | 𝒯_{1:k})
```

is evaluated by AUROC at each prefix fraction `k/n` — a predictability curve,
not a scalar (`src/analysis/predictability.py`). Standardisation uses
train-split statistics only (paper Algorithm 1, line 9).

## Lead time

For threshold `τ`, `k* = min{k : r_k ≥ τ}` is the first alarm. Lead time

```
Δ = e − k*
```

is positive when the detector fires before the error (`src/analysis/lead_time.py`).

## Intervention value

Not accuracy alone but the pair

```
Recovery   = P(corrected | y = 1)
Corruption = P(broken    | y = 0)
```

subject to a token budget (`src/intervention/simulate.py`). Net benefit =
recovery − corruption at matched token budget.
