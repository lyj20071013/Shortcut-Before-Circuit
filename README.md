<div align="center">

# Same Task Performance, Different Intervention Readouts

### Cross-Run Variation under In-Context Rule Aliasing

<a href="https://arxiv.org/abs/2608.24460">
  <img src="https://img.shields.io/badge/arXiv-2608.24460-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv:2608.24460">
</a>
<a href="https://github.com/lyj20071013/Shortcut-Before-Circuit">
  <img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github" alt="GitHub repository">
</a>
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
<img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">

**Yijun Liao · Fanwei Liang**

Controlled experiments on the reproducibility of intervention-based
mechanistic measurements across independently trained transformers.

</div>

---

## Overview

Can two models solve the same task almost perfectly while responding very
differently to the same mechanistic intervention?

We study this question in a controlled in-context retrieval setting. Every
training document is constructed so that two candidate answer rules are
observationally equivalent:

```text
RECENCY: choose the most recently assigned value
RARITY:  choose the least frequent value in the queried slot
```

Because both rules select the same answer on every training example, task
accuracy cannot distinguish them. We then apply an answer-preserving
multiplicity intervention that makes their predictions diverge while keeping
the correct answer, answer position, token count, statement count, and all
non-target tokens fixed.

Across **75 independently trained 26.1M-parameter transformers** in
**25 configurations**:

- every analyzed run reaches in-distribution accuracy of at least **0.999**;
- **13/25 configurations** have a cross-seed intervention-readout range above
  `0.3`;
- the largest range is **0.879**;
- replacing the probe batch changes the readout by at most **0.034** in four
  tested extreme runs;
- on separately generated documents where RECENCY and RARITY disagree,
  **70/75 baseline runs never select the rarity answer**, and the remaining
  five do so at a rate of at most `0.003`.

The central result is therefore not that different seeds implement different
observable answer rules.

> **Task behavior can reproduce while an intervention-based sensitivity
> measurement does not.**

A directional result can be statistically decisive within one trained model
and still fail to reproduce after retraining the same architecture on the same
task.

---

## What the readout measures

Let `v*` be the contrast value and `v_truth` the preserved correct answer. For
the base and edited versions of the same document, we measure

```text
Δ = [log p(v*) - log p(v_truth)]edit
  - [log p(v*) - log p(v_truth)]base
```

The primary summary is the fraction of valid document pairs with `Δ > 0`.
A positive value is a rarity-directed displacement; a negative value is an
occurrence-count-directed displacement.

This quantity is an **intervention response**. It is not, by itself:

- a classifier of the model's answer policy;
- evidence that a unique internal mechanism has been identified;
- a configuration-level label that can safely be inferred from one seed; or
- proof that a particular circuit exists.

The retrieval-state criteria determine which terminal runs are eligible for
comparison. Passing those criteria does not identify an internal circuit and
does not guarantee temporal stability of the readout.

---

## Main findings

### 1. Cross-run variation

All 75 main-grid runs solve the task, but independently trained models in the
same configuration can give very different intervention readouts.

| Cell | seed 0 | seed 1 | seed 2 | range |
| --- | ---: | ---: | ---: | ---: |
| `R_old=3, ΔD=8` | 0.098 | 0.477 | **0.977** | **0.879** |
| `R_old=3, ΔD=5` | 0.126 | **0.972** | 0.270 | 0.845 |
| `R_old=5, ΔD=2` | 0.781 | 0.175 | **0.967** | 0.792 |
| `R_old=3, ΔD=3` | 0.365 | **0.934** | 0.930 | 0.569 |
| `R_old=16, ΔD=16` | **0.977** | 0.469 | 0.969 | 0.508 |

The largest differences are not explained by resampling probe documents. In
the two widest cells, exchanging the trained model while holding the probe
batch fixed reproduces the cross-run gap within sampling error.

### 2. Intervention sensitivity is not answer policy

We evaluate the same checkpoints on newly generated documents where RECENCY
and RARITY select different answers. The rarity-answer rate is:

- exactly `0.000` in 70 of the 75 main-grid runs;
- at most `0.003` in the other five.

Across all 28,563 valid edited pairs, 98.68% of value-vocabulary argmaxes remain
on the preserved correct answer. The cross-seed variation therefore concerns
log-odds movement under intervention, not different observable answer
policies.

### 3. Supervision changes the measurement

We introduce disagreement documents during training and label them according
to either RECENCY or RARITY.

- The **RARITY-labeled** arm produces a behavioral rarity preference and large
  positive displacements, providing a trained positive control.
- In the selected `R_old=3, ΔD=8` cell, the **RECENCY-labeled** arm has lower
  observed cross-seed dispersion than the aliased baseline.

The dispersion comparisons are exploratory: the cell was selected using the
original grid, sample sizes remain limited, and the comparisons are not
selection-adjusted. They should not be interpreted as a general causal proof
that rule aliasing is sufficient for cross-run dispersion.

### 4. The readout depends on the training process

The saved experiments separate several sources of variation:

- replacing the probe batch has a small effect in the tested extreme runs;
- changing the training-document stream at fixed initialization produces
  substantial dispersion;
- the readout can drift across post-gate checkpoints at unchanged accuracy;
- changing the cosine period or removing annealing can move individual runs
  substantially;
- width, depth, supervision, and surface form can all change the measurement.

Document-sampling uncertainty, within-run temporal variation, and variation
between independently trained models are different quantities and are reported
separately.

### 5. Pre-retrieval readouts can point in the opposite direction

Predominantly negative readouts occur before the retrieval-state criteria are
met in every redundancy row. These values are retained as training-dynamics
diagnostics, not as answer-rule or circuit attributions.

Passing the retrieval gate is necessary for the terminal comparison, but it is
not sufficient to make a single-run mechanistic conclusion temporally stable.

---

## Repository contents

The repository includes source code and the available machine-readable records
used by the analyses. It does **not** include model weights or checkpoints.

| Path | Contents |
| --- | --- |
| `runs_g2/` | Canonical 75-run grid, terminal and per-document probes, transition summaries, and selected intervention records |
| `runs_g2_seeds/` | Expanded seed sample for `R_old=3, ΔD=8` |
| `runs_nb/` | RECENCY/RARITY disagreement-supervision arms, dose extensions, failures, and rescue runs |
| `runs_constlr/`, `runs_cos32/` | Constant-learning-rate and changed-cosine-period experiments |
| `runs_dseeds/` | Fixed initialization with different training-document streams |
| `runs_depth/`, `runs_w1024/` | Depth and 102.2M-parameter width experiments |
| `runs_gamma1/` | QK-normalization gain control |
| `runs_slot/`, `runs_pupd/`, `runs_fixband/` | Slot-count, update-density, and fixed-band controls |
| `runs_nl/`, `runs_ft/` | Rendered-English and Qwen2.5-0.5B fine-tuning trajectories |
| `runs_flat/` | Available exploratory readout/loss-geometry records |
| `nl_data/`, `ft_data/` | Archived probe arrays and tokenizer/vocabulary metadata |
| `CODE_INDEX.md` | Script-by-script source-code index |
| `COVERAGE.md` | Paper-to-artifact coverage map and missing-output disclosures |
| `manifest.json` | File sizes, checksums, record counts, and packaging provenance |

The pair `(experiment directory, run tag)` identifies a run. Some directories
reuse tags, so runs must not be merged by tag alone.

---

## Quick start: analyze the saved records

Clone the repository and create an analysis environment:

```bash
git clone https://github.com/lyj20071013/Shortcut-Before-Circuit.git
cd Shortcut-Before-Circuit

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
```

Verify the distributed artifact and regenerate the two paper figures:

```bash
python validate_bundle.py
python figs2.py --prefix outputs/fig
```

These commands use the saved JSON/JSONL records and do not require a GPU or a
checkpoint.

Additional audit commands:

```bash
# Main-grid quantities and run inventory
python paper_numbers.py \
  --s0-dir runs_g2 --s1-dir runs_g2 --s2-dir runs_g2 \
  --gonogo runs_g2/go_nogo.txt.jsonl
python run_ledger.py --root . --out outputs/ledger.tsv

# Argmax transitions and magnitude thresholds
python argmax_transitions.py \
  runs_g2/go_nogo_argmax_v1.txt.jsonl \
  --perdoc runs_g2/go_nogo_argmax_v1.txt.perdoc.jsonl \
  --json outputs/argmax.json \
  --txt outputs/argmax.txt \
  --tex outputs/argmax.tex

python magnitude_thresholds.py \
  --perdoc runs_g2/go_nogo_argmax_v1.txt.perdoc.jsonl \
  --summary runs_g2/go_nogo_argmax_v1.txt.jsonl \
  --out outputs/magnitude

# Supervision, schedules, data streams, and architecture
python verify_dose01.py --root .
python constlr_read.py --seeds 0 1 2 3 4 5 6 7 8 9 --at 16000 \
  --json outputs/constlr.json
python dseed_read.py --dir runs_dseeds --out outputs/dseeds.json
python depth_peaks.py --dir runs_depth
python fixband_analyze.py runs_fixband
python ft_read.py --dir runs_ft --out outputs/ft_summary.json
```

The permutation calculations in the supervision analyses enumerate the stated
reference assignments. Enumeration does not remove selection bias, establish
exchangeability, or turn the selected-cell comparisons into confirmatory
evidence.

---

## Recreate a synthetic training run

Install the training dependencies with a PyTorch build appropriate for your
CUDA environment:

```bash
python -m pip install -r requirements-training.txt
```

The safest way to reproduce a configuration is to start from its recorded log.
The following command prints a retraining recipe without launching training:

```bash
python retrain_from_log.py \
  runs_g2/R3_D8_s0_grid.jsonl \
  --out outputs/retrained_main
```

Add `--execute` to start the run:

```bash
python retrain_from_log.py \
  runs_g2/R3_D8_s0_grid.jsonl \
  --out outputs/retrained_main \
  --execute
```

To list or launch the complete 5 × 5 × 3 grid:

```bash
python sweep.py --dry-run --out outputs/retrained_grid
python sweep.py --out outputs/retrained_grid
```

After training, evaluate a checkpoint with:

```bash
python go_nogo.py R3_D8_s0_grid \
  --out outputs/retrained_main \
  --docs 400
```

Training uses streamed synthetic documents and GPU mixed precision. Different
worker counts, CUDA/PyTorch versions, or numerical environments may change the
sampled stream or optimization trajectory. Cross-platform bitwise identity is
not guaranteed.

---

## Rendered-language and pretrained-model arms

Restore the archived probe arrays, including dtype, shape, and checksum
validation:

```bash
python restore_probe_arrays.py --out generated_inputs
```

Generate current rendered-language inputs and train from scratch:

```bash
python nl_corpus.py --rows 3 --cols 8 --out generated_inputs/nl_current
python nl_train.py \
  --r 3 --d 8 --seed 0 \
  --data generated_inputs/nl_current \
  --out outputs/retrained_nl
```

Rebuild tokenizer-level inputs and fine-tune Qwen2.5-0.5B:

```bash
python ft_data.py \
  --model Qwen/Qwen2.5-0.5B \
  --src hf --no-mirror \
  --out generated_inputs/ft_current

python ft_train.py \
  --model Qwen/Qwen2.5-0.5B \
  --src hf --no-mirror \
  --seed 0 \
  --data generated_inputs/ft_current \
  --out outputs/retrained_ft \
  --tag ft
```

The upstream model and tokenizer are not distributed here, and the archived
logs do not record an immutable upstream revision.

---

## Data schema notes

- Training trajectories normally begin with a `kind="meta"` object, followed
  by training, evaluation, probe, and completion records.
- Terminal summaries and per-document records are separate files and should not
  be pooled silently.
- `frac_positive` is the fraction of valid edit pairs with positive
  displacement; it is not a behavioral rarity-answer rate.
- `NaN` represents an undefined measurement, not zero. Some historical JSONL
  files use Python's non-strict `NaN` literal.
- Mass-restricted and unrestricted summaries describe different document
  populations. Use the population specified by the corresponding analysis.
- Failed and historical runs are retained when they are part of a reported
  control. Their presence does not make them eligible for every pooled result.

---

## Coverage and limitations

The repository supports the saved-record analyses for the main grid and the
available extensions, but it does not claim direct reconstruction of every
appendix result.

The following original artifacts are not distributed:

1. raw logs for the excluded `R_old=2` group;
2. the complete final geometry-control output, including two
   objective-constrained readouts;
3. original non-aliased head- and residual-patching output files;
4. terminal and intermediate model weights or checkpoints; and
5. a complete original training lockfile and immutable revision of external
   pretrained assets.

`runs_flat/` therefore provides only the available exploratory geometry
records. The supplied `flatctrl.py` should not be treated as a reconstruction
of the complete final geometry table.

See [`COVERAGE.md`](COVERAGE.md) for the paper-to-artifact map and
[`CODE_INDEX.md`](CODE_INDEX.md) for the source-code index.

---

## Artifact integrity

`manifest.json` records file sizes, SHA-256 checksums, JSON/JSONL record counts,
and packaging provenance. Run:

```bash
python validate_bundle.py
```

This checks distribution integrity and parseability. It does not independently
recompute or certify the scientific conclusions.

---

## Citation

```bibtex
@misc{liao2026interventionreadouts,
  title        = {Same Task Performance, Different Intervention Readouts:
                  Cross-Run Variation under In-Context Rule Aliasing},
  author       = {Liao, Yijun and Liang, Fanwei},
  year         = {2026},
  eprint       = {2608.24460},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2608.24460}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE).

The paper is distributed separately under the license shown on its arXiv
record. Model and tokenizer assets obtained from external providers remain
subject to their respective licenses.
