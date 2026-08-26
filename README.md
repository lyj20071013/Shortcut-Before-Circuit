<div align="center">

# Shortcut Before Circuit: Document Statistics Time In-Context Conflict Resolution

<a href="https://arxiv.org/abs/2608.24460"><img src="https://img.shields.io/badge/arXiv-2608.24460-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv:2608.24460"></a>
<a href="https://github.com/lyj20071013/Shortcut-Before-Circuit"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github" alt="GitHub repository"></a>
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
<img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.4+">

**Yijun Liao · Fanwei Liang**

Controlled synthetic-language experiments for separating behaviorally aliased
in-context decision rules.

</div>

---

> [!IMPORTANT]
> The paper is on arXiv as [arXiv:2608.24460](https://arxiv.org/abs/2608.24460) (cs.CL).
> This repository contains the source code and `paper.pdf`. It does **not** include
> pretrained checkpoints or raw experiment outputs. The dataset is generated online and
> therefore does not require a download. See [License](#-license) for the current
> licensing state.

## 📢 News

- **2026-08-25:** Preprint posted as [arXiv:2608.24460](https://arxiv.org/abs/2608.24460)
  (cs.CL). README audited against the current command-line interfaces; the `p_update` arm
  and the geometry positive-control suite are now documented.
- **2026-08-24:** Public source snapshot released with the 75-run grid, causal readout,
  generator checks, architecture/data controls, and paper-table utilities.

## ⚡️ Overview

When several cues always agree in observational data, identical task performance does not
identify which cue a model actually uses. This project constructs that ambiguity exactly.

Each synthetic document contains repeated assignments to an entity–attribute slot. For the
queried slot, a superseded value appears `R_old` times and the current value appears once.
Consequently, two human-interpretable policies agree on every training document:

```text
RECENCY: choose the value in the latest assignment
RARITY:  choose the value occurring least often within the queried slot
```

The training objective cannot distinguish the two policies. A minimal held-out edit reverses
their predictions by inverting the multiplicities of the old and current values while keeping
the ground truth, token count, answer position, statement count, and all non-target tokens
fixed.

The main empirical result is a split:

- **Which rule a run expresses does not replicate reliably across seeds.**
- **When the model escapes a positional shortcut and forms a retrieval circuit does.**

## Repository status

| Item | Status |
| --- | --- |
| Main grid | 25 cells × 3 seeds = 75 runs |
| Dataset | Generated online; no static corpus required |
| Main readout | `go_nogo.py`, 400 held-out documents per run |
| Checkpoints/results | Not included in the repository |
| Paper | [arXiv:2608.24460](https://arxiv.org/abs/2608.24460); `paper.pdf` in this repository |
| Dependency lock file | Not included; dependencies are listed below |
| License | See [License](#-license) — a `LICENSE` file still needs to be committed |

## Experimental design

### The exact alias

For the queried slot `s` in every generated training document `d`, the current value is both
the last assigned value and a least-frequent value:

```text
argmax_i pos(s_i)  =  argmin_v |{i : val(s_i) = v}|      for every d
       RECENCY                       RARITY
```

This is a generator-level construction, not an empirical correlation estimated from a finite
sample. `collide.py` measures the resulting rule-collision rates, while `selfcheck.py` and
`sweep.py --check` verify the structural invariants before training.

### Main grid

| Quantity | Canonical value |
| --- | --- |
| Redundancy axis | `R_old ∈ {3, 5, 8, 12, 16}` |
| Nominal distance axis | `d ∈ {2, 3, 5, 8, 16}` |
| Random seeds | `{0, 1, 2}` |
| Runs | `5 × 5 × 3 = 75` |
| Statements per document | Uniform on `[45, 55]` |
| Filler-slot update probability | `p_update = 0.5` |
| Entities / attributes / values | `200 / 8 / 512` |
| Effective context length | `600` tokens |
| Model | 8 layers, `d_model=512`, 8 heads, SwiGLU MLP |
| Position encoding | RoPE |
| Attention | Per-head QK RMS normalization, gain initialized at `2.0` |
| Parameters | 26.1M |
| Training | 16,000 steps, batch 256, AdamW, cosine LR |
| Learning rate / weight decay | `1e-3 / 0.1` |
| Evaluation | Every 1,000 steps on 4,000 fixed held-out documents |

`d` is a label for a non-degenerate support interval produced by `dd_band(d)`:

| Nominal `d` | Realized `ΔD` support | Positional ceiling |
| ---: | ---: | ---: |
| 2 | `[1, 3]` | `1/3 = 0.333` |
| 3 | `[2, 4]` | `1/3 = 0.333` |
| 5 | `[2, 8]` | `1/7 = 0.143` |
| 8 | `[4, 12]` | `1/9 = 0.111` |
| 16 | `[8, 24]` | `1/17 = 0.059` |

A constant `ΔD` would make “copy the value at a fixed offset from the end” perfectly
accurate. `validate_cfg` therefore rejects a one-point support.

### Causal readout

The `break_rarity` edit changes the correct value from one occurrence to `R` occurrences and
the superseded value from `R` occurrences to one. For a fixed contrast value `v*` and ground
truth `v_truth`, the paired readout is

```text
Δ = [log p(v*) - log p(v_truth)]edit
  - [log p(v*) - log p(v_truth)]base
```

- `Δ > 0`: rarity-type response.
- `Δ < 0`: frequency/accumulation-type response.
- `Δ ≈ 0`: the edit does not move the contrast.

The primary dependent variable is `frac+`, the fraction of edited documents with `Δ > 0`,
reported with an exact two-sided binomial test. The median, IQR, trimmed mean, probability
mass on the contrast pair, and a same-size filler-slot control are reported alongside it.

Two gates matter:

1. `copy_acc ≥ 0.95` and task accuracy `≥ 0.99` are required before a run is classified as
   retrieval-capable.
2. Per-document contrast-pair mass below `0.5` invalidates magnitude-based readouts on that
   document. `frac+` is intentionally computed over the edit domain rather than the gated
   subset.

## 🌟 Key results

### 1. Per-cell mechanism readout does not replicate

All 75 analyzed runs reach in-distribution accuracy at least 0.999, including the stratum on
which the trivial global-last-update heuristic fails. Under the causal edit, however, the
same cell can point in different directions across seeds.

| Cell | seed 0 | seed 1 | seed 2 | range |
| --- | ---: | ---: | ---: | ---: |
| `R_old=3, ΔD=8` | 0.098 | 0.477 | **0.977** | **0.879** |
| `R_old=3, ΔD=5` | 0.126 | 0.972 | 0.270 | 0.845 |
| `R_old=5, ΔD=2` | 0.781 | 0.175 | 0.967 | 0.792 |
| `R_old=16, ΔD=16` | 0.977 | 0.469 | 0.969 | 0.508 |

Thirteen of the 25 cells span more than 0.3 across seeds, and eight straddle 0.5. At
`n=400`, the maximum binomial standard error is 0.025, so the widest range is approximately
35 standard errors. The unit of replication is therefore the training run, not the held-out
document.

### 2. Timing is substantially more stable

| Quantity | Result | Replicates? |
| --- | --- | :---: |
| Escape step from loss-derivative peak | Monotone in `R_old` across four separable levels | ✅ |
| Positional shortcut ceiling | Saturated to within 2% at narrow support | ✅ |
| Within-run redundancy dose response | Same direction as each run’s effect in 65/75 runs | ✅ |
| Per-cell rule identity | Range up to 0.879 across seeds | ❌ |

Loss-derivative peak row means, ordered by `R_old = 3, 5, 8, 12, 16`, are:

- seed 0: `5500 / 1740 / 740 / 400 / 400`
- seed 1: `6100 / 2660 / 1180 / 420 / 400`
- seed 2: `4220 / 1960 / 820 / 400 / 375`

The last two rows are not cleanly separated because the 100-step recording grid begins to
bind.

### 3. Probing before circuit formation reverses the conclusion

In 32 of 75 runs the last pre-escape sign fraction is below 0.20, and in 22 it is below 0.05.
Taken as a rule attribution, those values would indicate a clean frequency-type region even
though the copy diagnostic says the model has not formed a retrieval circuit. Circuit
formation is therefore a necessary gate, but it is not sufficient to make a single-run
mechanistic attribution replicable.

## 💻 Installation

Clone the repository and run all commands from its root:

```bash
git clone https://github.com/lyj20071013/Shortcut-Before-Circuit.git
cd Shortcut-Before-Circuit

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib
```

For a CUDA-specific PyTorch wheel, use the command generated by the
[official PyTorch installer](https://pytorch.org/get-started/locally/). The code falls back
to fp16 on CUDA devices without bf16 and to fp32 on CPU, but reproducing the reported
training setup requires a CUDA GPU with bf16 support.

There is currently no pinned `requirements.txt`. The code uses only the Python standard
library plus PyTorch, NumPy, and Matplotlib.

## Canonical settings live in `sweep.py`

Running `train.py` with only `--r` and `--d` does **not** reproduce a paper cell. The
dataclass defaults preserve pilot-era values; `sweep.py` is the authoritative main-grid
entry point.

| Field | Direct-code default | Canonical sweep |
| --- | ---: | ---: |
| `LangSpec.n_entities` | 2000 | **200** |
| `LangSpec.n_values` | 512 | **512** |
| Effective `ctx_len` | 600 | **600** |
| `CorpusCfg.n_stmts_lo/hi` | 60 / 100 | **45 / 55** |
| `TrainCfg.total_steps` | 15000 | **16000** |
| `TrainCfg.batch_docs` | 192 | **256** |
| `TrainCfg.eval_every` | 1500 | **1000** |
| `TrainCfg.eval_docs` | 8000 | **4000** |
| `TrainCfg.num_workers` | 8 | **4** |
| Redundancy grid | `config.R_OLD_GRID=[1,2,3,5,8,12]` | **`[3,5,8,12,16]`** |

`ModelCfg.ctx_len` has a standalone default of 1024, but `train.py` explicitly constructs
the model with `LangSpec.ctx_len`; canonical CLI runs therefore use 600. The older
`phase_configs()`, `EXTREME_A/B`, and `hist_configs()` objects in `config.py` are pilot
scaffolding and do not define the reported 75-run grid.

## 🚀 Main reproduction pipeline

### Stage 0 — pre-flight checks

Do not start the grid until the generator check passes:

```bash
python sweep.py --check
```

Optional deeper checks:

```bash
# Generator and edit invariants
python selfcheck.py

# Can the architecture learn one-key, two-key, and long-range induction?
python bench.py --qk-gain 2.0 --steps 4000 --seeds 3
```

The checks run on CPU, but the current import graph still requires PyTorch to be installed.

### Stage 1 — train the 75-run grid

```bash
# Print every command without training
python sweep.py --dry-run --out runs_g2

# Run all 25 cells for seeds 0, 1, and 2
python sweep.py --out runs_g2

# Summarize whatever has completed
python sweep.py --collect --out runs_g2
```

`sweep.py` runs seed-major and executes the six `HEAD` cells first. It skips a run when its
final `.pt` file already exists. This provides **run-level restartability**: an interrupted
cell is restarted from step 1; it is not resumed from optimizer state.

To distribute seeds across machines or GPUs:

```bash
python sweep.py --seed 0 --out runs_s0
python sweep.py --seed 1 --out runs_s1
python sweep.py --seed 2 --out runs_s2
```

One canonical cell by hand:

```bash
python -u train.py \
  --r 3 --d 5 --seed 0 --steps 16000 \
  --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
  --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
  --eval-every 1000 --eval-docs 4000 --workers 4 \
  --out runs_g2 --tag grid
```

### Stage 2 — terminal readout and paper numbers

For all seeds stored in one directory:

```bash
# Authoritative final readout: 400 documents per run
python go_nogo.py --pattern 'R*_grid' --out runs_g2

# Escape timing from the 100-step training-loss trace
python traj.py runs_g2 --seeds 0 1 2 --suffix _grid

# Generator invariants and candidate-rule collision rates
python covar.py --txt runs_g2/covar.txt --tex runs_g2/covar.tex
python collide.py --out runs_g2/collide

# Paper-number and provenance utilities
python paper_numbers.py \
  --s0-dir runs_g2 --s1-dir runs_g2 \
  --gonogo runs_g2/go_nogo.txt
python ledger.py --dirs runs_g2 --txt runs_g2/ledger.txt --csv runs_g2/ledger.csv
python verify_seed2.py runs_g2/go_nogo.txt
```

If the three seeds are in separate directories, pass the directories in seed order:

```bash
python traj.py runs_s0 runs_s1 runs_s2 --seeds 0 1 2 --suffix _grid
python paper_numbers.py \
  --s0-dir runs_s0 --s1-dir runs_s1 \
  --gonogo runs_s0/go_nogo.txt runs_s1/go_nogo.txt runs_s2/go_nogo.txt
```

`sweep.py --collect` uses the 200-document online probe. `go_nogo.py` is the final
400-document path and is the source for the reported terminal sign fractions, medians,
controls, and binomial tests.

### Stage 3 — figures

```bash
# Replacement Figure 1 and Figure 2
python figs2.py \
  --s0-dir runs_g2 --s1-dir runs_g2 \
  --gonogo runs_g2/go_nogo.txt \
  --flip-run 3 3 1 --prefix fig

# Phase/escape plots plus the per-document distribution figure
python figs.py runs_g2/go_nogo.txt \
  --perdoc runs_g2/go_nogo.txt.perdoc.jsonl \
  --prefix fig

# Continuous-metric check against a threshold artifact
python plot_continuous.py \
  --runs runs_g2 --cells R3_D2,R16_D2 --seeds 1
```

`fig_traj.py` is a hardcoded-path alternative for an earlier version of the escape figure.
`plot_perdoc.py` is legacy and is not compatible with the current default per-document
cache without manual conversion; use `figs.py --perdoc` instead.

## Geometry diagnostic — a reported negative result

The geometry suite measures whether the direction changing the causal readout is flat
relative to the training loss. It computes gradient alignment, a split-batch noise floor,
finite differences, and curvature in strict fp32 with TF32 disabled.

The current paper does **not** use this suite as evidence for the main claim. The aliased
direction appears flat, but `flatctrl.py` shows that several separable and directly
objective-constrained readouts constructed by the same procedure also appear flat. The
measurement therefore fails to distinguish aliasing from generic answer-position geometry
in a 26M-parameter model. The main argument instead rests on the document-wise identity and
the observed ordering of variance across documents, checkpoints, schedules, and seeds.

### Produce fp32 intermediate checkpoints

Final checkpoints are stored in bf16. Geometry runs need fp32 checkpoints generated with
`--ckpt-every` or `--ckpt-steps`:

```bash
python -u train.py \
  --r 3 --d 5 --seed 0 --steps 16000 \
  --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
  --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
  --eval-every 1000 --eval-docs 4000 --workers 4 \
  --out runs_flat --tag flat --ckpt-every 2000
```

Repeat for the other cell/seed combinations required by the comparison.

### Run the primary measurement and controls

```bash
# flatdir.py takes checkpoint paths as positional arguments
python flatdir.py runs_flat/ckpt/R3_D5_s0_flat_s*.pt \
  --out runs_flat/flat_R3D5_s0.jsonl \
  --loss-docs 512 --probe-docs 400

python summarize_flat.py runs_flat/flat_R3D5_s0.jsonl

# Apply the same measurement to all available causal edits
python flatctrl.py runs_flat/ckpt/R3_D5_s0_flat_s*.pt \
  --out runs_flat/flatctrl_R3D5_s0.jsonl \
  --loss-docs 512 --probe-docs 400
```

Table and cross-seed utilities:

```bash
python flat_find.py \
  runs_flat/flat_R3D5_s0.jsonl \
  runs_flat/flat_R16D2_s0.jsonl > flat_rows.tex

python flat_compare.py \
  runs_flat/flat_R3D5_s0.jsonl \
  runs_flat/flat_R3D5_s1.jsonl

python flat_ratios.py \
  runs_flat/flat_R3D5_s0.jsonl \
  runs_flat/flat_R3D5_s1.jsonl
```

Earlier final-checkpoint diagnostics remain available:

```bash
python an_grad.py R3_D5_s0_grid R16_D2_s0_grid \
  --out runs_g2 --docs 200 --json runs_g2/an_grad.json

python an_fd.py R3_D5_s0_grid R16_D2_s0_grid \
  --out runs_g2 --docs 100
```

Intermediate geometry checkpoints do not contain optimizer state and cannot resume
training. Checkpointing itself does not consume RNG or advance the dataloader.

## 🧪 Side arms and controls

### QK-normalization gain

The reported gain arm tests `γ=1` against the main-grid value `γ=2` at
`(R_old, d) ∈ {(3,5), (16,2)}` for seeds 0 and 1.

```bash
for SEED_ID in 0 1; do
  python -u train.py --r 3 --d 5 --seed "$SEED_ID" --steps 16000 \
    --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
    --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
    --eval-every 1000 --eval-docs 4000 --workers 4 \
    --qk-gain 1.0 --out runs_gamma --tag gamma1

  python -u train.py --r 16 --d 2 --seed "$SEED_ID" --steps 16000 \
    --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
    --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
    --eval-every 1000 --eval-docs 4000 --workers 4 \
    --qk-gain 1.0 --out runs_gamma --tag gamma1
done

python go_nogo.py --pattern 'R*_gamma1' --out runs_gamma
python gamma_table.py runs_g2 runs_gamma
```

### Fixed-width `ΔD` band

This arm holds the positional ceiling at `1/9` while moving the band location.

```bash
# CPU gate; exits non-zero on a structural failure
python fixband_check.py --r 3 --lo 1 4 8 16 --width 9 --docs 1500

for SEED_ID in 0 1 2; do
  for BAND_LO in 1 4 8 16; do
    BAND_HI=$((BAND_LO + 8))
    python -u train.py \
      --r 3 --d "$BAND_LO" --seed "$SEED_ID" --steps 16000 \
      --dd-lo "$BAND_LO" --dd-hi "$BAND_HI" \
      --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
      --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
      --eval-every 1000 --eval-docs 4000 --workers 4 \
      --out runs_fixband --tag "fb${BAND_LO}-${BAND_HI}"
  done
done

python go_nogo.py --pattern 'R*_fb*' --out runs_fixband
python fixband_analyze.py runs_fixband --pattern '_fb(\d+)-(\d+)$'
python traj.py runs_fixband --seeds 0 1 2 --suffix '_fb*' \
  --rows 3 --cols 1 4 8 16
python covar.py --fixband 9 --rows 3 --cols 1 4 8 16 \
  --txt runs_fixband/covar.txt --tex runs_fixband/covar.tex
python collide.py --fixband 9 --rows 3 --cols 1 4 8 16 \
  --out runs_fixband/collide
```

Passing only one of `--dd-lo` and `--dd-hi` is an error. Here `--d` is a label; the explicit
band controls the actual support.

### Slot-count arm by shortening documents

`calib.py` searches for a shorter `R_old=8, d=2` configuration whose slot count matches the
main-grid `R_old=16, d=2` cell.

```bash
mkdir -p runs_slot
python calib.py --dd 2 --docs 2000 --out runs_slot/calib_R8

# The reported calibration selects 23–33 statements, not 20–30.
for SEED_ID in 0 1 2; do
  python -u train.py \
    --r 8 --d 2 --seed "$SEED_ID" --steps 16000 \
    --stmts-lo 23 --stmts-hi 33 \
    --n-values 512 --n-entities 200 \
    --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
    --eval-every 1000 --eval-docs 4000 --workers 4 \
    --out runs_slot --tag slotmatch
done

python go_nogo.py --pattern 'R*_slotmatch' --out runs_slot
```

`pair.py` performs the matched-`q_kept` analysis, but it expects a **flattened per-document
JSONL** containing scalar `delta`, `q_kept`, and optional `mass` fields. The current
`go_nogo.py` per-document cache instead stores one `{tag, d_all}` object per run and therefore
cannot be passed directly to `pair.py`. A converter/export path is not currently included.

With compatible flattened files, the interface is:

```bash
python pair.py <R8-main-perdoc.jsonl> <R8-short-perdoc.jsonl> \
  --ref <R16-main-perdoc.jsonl> --band 3,5 --out pair
```

### Slot-count arm at fixed length with `p_update`

The newer `p_update` arm changes filler-slot density while keeping document length fixed.
The reported calibration uses `p_update=0.90` at `R_old=8` and `p_update=0.28` at
`R_old=16`, producing approximately 11.51 and 11.55 slots at approximately 205 tokens.

```bash
python calib_pupd.py \
  --r 8 16 --d 2 --p 0.28 0.50 0.90 --docs 2000

for SEED_ID in 0 1 2; do
  python -u train.py \
    --r 8 --d 2 --seed "$SEED_ID" --steps 16000 \
    --p-update 0.90 --tag pupd090 \
    --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
    --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
    --eval-every 1000 --eval-docs 4000 --workers 4 \
    --out runs_pupd

  python -u train.py \
    --r 16 --d 2 --seed "$SEED_ID" --steps 16000 \
    --p-update 0.28 --tag pupd028 \
    --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
    --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
    --eval-every 1000 --eval-docs 4000 --workers 4 \
    --out runs_pupd
done

python go_nogo.py --pattern 'R*_pupd*' --out runs_pupd
```

`train.py` requires a non-empty `--tag` whenever `p_update != 0.5`, preventing an arm from
silently overwriting a main-grid run. This arm holds token length nearly fixed but does not
fully match the queried-slot copy dispersion; that remaining difference must be reported.

### Depth arm

```bash
for LAYER_COUNT in 4 12; do
  for R_OLD in 3 8 16; do
    for SEED_ID in 0 1; do
      python -u train.py \
        --r "$R_OLD" --d 5 --seed "$SEED_ID" --steps 16000 \
        --n-layer "$LAYER_COUNT" \
        --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
        --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
        --eval-every 1000 --eval-docs 4000 --workers 4 \
        --out runs_depth --tag "L${LAYER_COUNT}"
    done
  done
done

python go_nogo.py --pattern 'R*_L4' --out runs_depth \
  --txt runs_depth/go_nogo_L4.txt
python go_nogo.py --pattern 'R*_L12' --out runs_depth \
  --txt runs_depth/go_nogo_L12.txt

python traj.py runs_depth --seeds 0 1 --suffix _L4 \
  --rows 3 8 16 --cols 5
python traj.py runs_depth --seeds 0 1 --suffix _L12 \
  --rows 3 8 16 --cols 5

# Loss-derivative peaks for the depth arm. paper_numbers.py matches
# R{r}_D{d}_s{s}_grid.jsonl and cannot see the _L4/_L12 tags, so this is a
# separate script; its deriv/escape_peak/read_run are line-for-line identical to
# paper_numbers.py so that peak positions stay comparable across the two.
python depth_peaks.py --dir runs_depth
python depth_peaks.py --dir runs_depth --tail-frac 0.8
```

At fixed `d_model=512`, changing depth also changes capacity: approximately 13.4M, 26.1M,
and 38.7M parameters for 4, 8, and 12 layers. The arm therefore does not isolate depth from
parameter count.

### Plateau-phase positional account

```bash
python overlap.py --docs 1500 --out runs_g2/overlap
```

This CPU-only analysis tests whether the offset `4ΔD+6` lands on one of the copies rewritten
by the causal edit, explaining the negative readout observed before retrieval forms.

## 🩺 Diagnostics

```bash
# Inspect every current per-document distribution before summarizing it
python hist.py runs_g2/go_nogo.txt.perdoc.jsonl --bins 25

# Learnability ladder
python bench.py --qk-gain 2.0 --steps 4000 --seeds 3

# Diagnostic pilot suite; currently blocked by the import typo noted below
python pilots.py --out runs_pilots

# Print detected JSON schema without plotting
python plot_continuous.py --runs runs_g2 --schema-only
```

For `bench.py`, read the metrics in this order:

1. `acc` reaches the target: the architecture can solve the diagnostic task.
2. Accuracy is on the positional plateau and `prec` is flat: sharpening did not start.
3. `prec` rises while accuracy remains short: increase the training budget.
4. `prec` stays at chance: inspect the model or metric implementation.

## Output conventions

| Path | Contents |
| --- | --- |
| `<out>/manifest.json` | Canonical sweep configuration and start time |
| `<out>/<tag>.jsonl` | Training metadata, 100-step loss records, evals, online probes |
| `<out>/<tag>.pt` | Final bf16 model checkpoint |
| `<out>/ckpt/<tag>_s<step>.pt` | Optional fp32 geometry checkpoint |
| `<out>/grid.txt` | `sweep.py --collect` summary |
| `<out>/go_nogo.txt` | Human-readable terminal readout |
| `<out>/go_nogo.txt.jsonl` | Incremental machine-readable readout cache |
| `<out>/go_nogo.txt.perdoc.jsonl` | `{tag, d_all}` records for distributions |
| `<out>/log/<tag>.txt` | Per-run stdout/stderr captured by `sweep.py` |

Keep each final `.pt` next to its matching training `.jsonl`: final checkpoints do not store
the complete `LangSpec`, and downstream readers recover it from the log’s first `meta` record.

## Project structure

```text
.
├── config.py               # Dataclasses, validation, dd_band, legacy pilot configs
├── vocab.py                # Integer vocabulary; one value is one token
├── generator.py            # Online document generator and structural statistics
├── model.py                # Decoder-only Transformer, RoPE, QK RMS normalization
├── probe.py                # Observational rules and causal edit library
├── probes.py               # Frozen probe-suite v2 utilities
├── train.py                # Single-cell training and online probes
│
├── sweep.py                # ★ Canonical 75-run main-grid driver
├── selfcheck.py            # Generator/edit correctness assertions
├── bench.py                # Induction learnability ladder
├── pilots.py               # Diagnostic pilot runner
│
├── go_nogo.py              # ★ Final causal readout and exact sign tests
├── traj.py                 # ★ Multi-seed escape-timing analysis
├── depth_peaks.py          # ★ Depth-arm loss-derivative peaks (_L4/_L12 tags)
├── ledger.py               # Result provenance across incompatible calibres
├── paper_numbers.py        # ★ Paper-number report by section
├── verify_seed2.py         # Seed-2 consistency checks against paper tables
├── seed2_traj.py           # Legacy seed-2 helper; superseded by traj.py
│
├── covar.py                # ★ Generator invariants and per-cell covariates
├── collide.py              # ★ Seven-rule collision rates
├── overlap.py              # Plateau-phase positional-overlap calculation
├── hist.py                 # ASCII per-document Δ histograms
│
├── flatdir.py              # Gradient/finite-difference/curvature measurement
├── flatctrl.py             # break_rarity plus six additional causal-edit controls
├── summarize_flat.py       # Geometry summary and consistency checks
├── flat_find.py            # Geometry LaTeX rows
├── flat_compare.py         # Same-cell cross-seed geometry comparison
├── flat_ratios.py          # Post-escape curvature and nats ratios
├── an_grad.py              # Earlier gradient-alignment diagnostic
├── an_fd.py                # Earlier final-checkpoint finite-difference diagnostic
│
├── calib.py                # Short-document slot-count calibration
├── pair.py                 # Matched-q_kept analysis; requires flattened per-doc input
├── calib_pupd.py           # Fixed-length p_update calibration
├── fixband_check.py        # Fixed-width-band structural gate
├── fixband_analyze.py      # Fixed-width-band results
├── gamma_table.py          # QK-gain LaTeX table
│
├── figs2.py                # ★ Replacement Figure 1 and Figure 2
├── figs.py                 # Phase, escape, and per-document distribution figures
├── plot_continuous.py      # Continuous escape metrics
├── fig_traj.py             # Earlier hardcoded escape figure
└── plot_perdoc.py          # Legacy hardcoded per-document plotter
```

`★` marks a canonical paper-output or paper-number path.

## 📐 Determinism and numerical precision

- Model initialization and Python sampling are seeded by `--seed`.
- Training workers use streams beginning at `seed_offset=1000+w`; evaluation and probe
  documents use `seed_offset=1`.
- The training stream is infinite, and worker offsets advance after every 4,096-document
  chunk rather than replaying the same corpus.
- Saving an intermediate checkpoint does not consume RNG or advance the dataloader.
- The original experiments report bit-identical reruns in the same software/hardware
  environment. Bit identity across different CUDA, PyTorch, or device versions is not
  guaranteed.
- Final model files are bf16. Geometry checkpoints are fp32 and geometry scripts explicitly
  disable TF32 because the relevant loss changes can be around `1e-5`.

## 🐛 Known issues and integration gaps

1. **Two copy thresholds.** `train.py` stores `converged = copy_acc > 0.9`, while the
   authoritative readout uses `COPY_FLOOR = 0.95`. Recompute state with `go_nogo.py` or
   `sweep.py --collect`; do not trust the stored boolean near the boundary.
2. **Direct defaults are not paper defaults.** Use `sweep.py` or spell out every canonical
   flag.
3. **Two context-length class defaults exist.** `LangSpec.ctx_len=600` is the effective CLI
   value; the standalone `ModelCfg.ctx_len=1024` fallback is overridden by `train.py`.
4. **Several outputs are append-mode.** Re-running `go_nogo.py`, `flatdir.py`, or
   `flatctrl.py` with the same destination can preserve or duplicate old records. Use a new
   output name when changing the document count or configuration. In particular,
   `go_nogo.py --force` does not truncate the per-document cache.
5. **`pair.py` is not connected to the current per-document cache schema.** It needs
   flattened scalar records, while `go_nogo.py` writes `{tag, d_all}` objects.
6. **`plot_perdoc.py` is legacy.** It expects a hardcoded `runs_g2/ctrl_table2.txt` carrying
   `d_all`, which the current human-readable `go_nogo` report does not contain. Use
   `figs.py --perdoc`.
7. **Some plotting helpers use hardcoded paths.** `fig_traj.py` and `plot_perdoc.py` must be
   edited if the run layout differs.
8. **Pilot-era docstrings remain.** Some comments still mention 90 runs, 110 statements,
   445-token documents, four probe points, or a nonexistent `--no-qk-norm` flag. For the
   probe count specifically, `TrainCfg.probe_points = 7` is the real value against the
   `train.py` docstring's "4 log-spaced probe points." The executable interfaces and
   `sweep.py` constants are authoritative.
9. **`pilots.py` currently has an import typo.** It imports `Option` from `typing`; this must
   be changed to `Optional` before the pilot runner can start.
10. **No optimizer state is saved.** Neither final nor geometry checkpoints support true
      step-level resume.
11. **No dependency lock, checkpoints, or `LICENSE` file is included.** `paper.pdf` is in the
    repository; the preprint is at [arXiv:2608.24460](https://arxiv.org/abs/2608.24460). Do
    not infer redistribution rights from the repository being public.

## Present in code but not completed as reported controls

- **Remove QK normalization entirely.** `ModelCfg.qk_norm` exists, but `train.py` exposes
  only `--qk-gain`; there is no `--no-qk-norm` route in the main trainer.
- **Learned absolute or no positional encoding.** `ModelCfg.pos` supports `learned` and
  `nope`, but `train.py` does not expose a `--pos` flag.
- **Width arm.** `--d-model` works but is not a pure width knob: `train.py` also sets
  `d_mlp = d_model * 8 // 3`, so width and MLP capacity move together. No reported
  fixed-depth width sweep is included.
- **Non-aliased training control.** Every main-grid cell preserves the recency–rarity alias;
  no training condition breaks it while retaining the rest of the pipeline.
- **Marker/history-query causal readouts.** The generator fields exist, but the current
  minimal edit does not support marked updates or historical queries without changing the
  causal estimand.

Note that `model.py`'s own docstring asks for two of these as appendix robustness checks —
re-running a few cells with learned-absolute position encoding, and with QK normalization
removed. Neither flag is exposed in `train.py`, so the docstrings describe intent rather
than an available route. Where a code comment and this section disagree, this section is
current.

## 📜 Citation

```bibtex
@misc{liao2026shortcut,
      title={Shortcut Before Circuit: Document Statistics Time In-Context Conflict Resolution}, 
      author={Liao, Yijun and Liang, Fanwei},
      year={2026},
      eprint={2608.24460},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.24460}
}
```

## 📄 License

Released under the Apache License 2.0. See [`LICENSE`](LICENSE).
Copyright 2026 Yijun Liao and Fanwei Liang.

The paper itself is separately licensed: the arXiv record is posted under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which covers `paper.pdf` and not
the code in this repository.
