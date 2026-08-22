<div align="center">

# Shortcut Before Circuit: Document Statistics Time In-Context Conflict Resolution

<a href="paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF_Available-b31b1b?style=for-the-badge&logo=adobeacrobatreader" alt="Paper PDF"></a>
<a href="#"><img src="https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
<a href="#"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge" alt="License"></a>

**Yijun Liao · Fanwei Liang**

[**Read the Paper (PDF)**](paper.pdf)

</div>

---

## 📢 News

- **[2025/xx/xx]** 🧪 **Side arms released.** Fixed-band, slot-matched, QK-gain and depth arms, plus the flat-direction geometry suite.
- **[2025/xx/xx]** 🔥 **Third seed complete.** The 75-run grid (25 cells × 3 seeds) and the full verification pipeline are in.
- **[2025/xx/xx]** 💻 **Code released.** Generator, training, causal probe and every table-producing script.

---

## ⚡️ Abstract

When a context asserts two values for one fact, a model commits to a cue — recency,
repetition, position — but natural data never makes these cues disagree, so behavior cannot
reveal which one it uses.

We train 26.1M-parameter transformers on a synthetic assignment language in which
**recency** ("take the most recently written value") and **rarity** ("take the value written
fewest times") are *exactly coextensive*: because each superseded value is repeated
`R_old` times while the current value is written once, the two rules select the same value on
**every** training document. The objective is indifferent between them and no observational
analysis can separate them. We separate them with a minimal causal edit that inverts the
multiplicities of the two competing values while holding the ground truth, the token count,
the answer position, and every other token fixed.

The result is a split. **Which** rule a model commits to does not replicate. **When** a
mechanism appears does.

---

## 🌟 Key Results

### 1. The per-cell readout does not replicate

All 75 runs reach in-distribution accuracy ≥ 0.999 — including on the stratum where the
trivial "the answer is the last update" heuristic fails — so no held-in evaluation separates
any pair of them. Under intervention they separate, and the separation is a property of the
run rather than of the cell.

| Cell | seed 0 | seed 1 | seed 2 | range |
| :--- | :---: | :---: | :---: | :---: |
| `R_old=3, ΔD=8` | 0.098 | 0.477 | **0.977** | **0.879** |
| `R_old=3, ΔD=5` | 0.126 | 0.972 | 0.270 | 0.845 |
| `R_old=5, ΔD=2` | 0.781 | 0.175 | 0.967 | 0.792 |
| `R_old=16, ΔD=16` | 0.977 | 0.469 | 0.969 | 0.508 |

> 13 of 25 cells span more than 0.3; in 8 the three seeds disagree about which side of
> indifference the cell lies on. The binomial standard error at n=400 is at most 0.025, so
> the largest range is **35 standard errors**. The endpoints of the widest cell are
> individually decisive at p = 1e-54 and p = 5e-88 — **in opposite directions**. Either run
> alone would license a confident mechanistic attribution.

### 2. Timing survives every comparison

| Quantity | Result | Replicates? |
| :--- | :--- | :---: |
| Escape step (loss-derivative peak) | monotone in `R_old` across 4 separable levels | ✅ all 3 seeds |
| Positional shortcut ceiling `1/\|supp(ΔD)\|` | saturated to within 2% at narrow support | ✅ closed form |
| Within-cell dose-response | holds with the run held fixed | ✅ 65 of 75 runs |
| Per-cell rule identity | range up to 0.879 across seeds | ❌ |

Escape row means, in steps: `5500 / 1740 / 740 / 400 / 400` (seed 0),
`6100 / 2660 / 1180 / 420 / 400` (seed 1), `4220 / 1960 / 820 / 400 / 375` (seed 2) at
`R_old = 3, 5, 8, 12, 16`.

### 3. Probing before the circuit exists reverses the conclusion

In **32 of 75 runs** the pre-escape sign fraction is below 0.20 (below 0.05 in 22), which
read as a rule attribution would report a clean "frequency-type" region — produced by models
that have **no retrieval circuit at all**. Accuracy gives no warning: it is 1.000 on both
sides of a readout that moves 0.71 in sign fraction.

---

## 🛠️ Methodology

**The alias.** For the queried slot `s` of any document `d`:

```
argmax_i pos(s_i)   =   argmin_v |{ i : val(s_i) = v }|      for all d
     RECENCY                      RARITY
```

This is an identity of the generator, not a statistical tendency.

**The readout.** A minimal edit inverts multiplicities: the correct answer goes from 1
occurrence to `R`, the superseded value from `R` to 1. Any mechanism that accumulates
evidence with occurrence count must move negative. The dependent variable is the **fraction
of held-out documents on which Δ has the expected sign**, with an exact binomial test — not
the mean, which is heavy-tailed, and not the median, which is bounded above by contrast-pair
mass in exactly the high-effect cells.

**The gate.** Every readout is gated on the copy diagnostic (accuracy on in-context value
tokens repeating an earlier value of the same slot) at 0.95, which requires the same
slot-matching computation the answer needs but is scored on 50× more tokens. The paper's
sharpest negative result is that this gate is **necessary but not sufficient**.

---

## ⚠️ Defaults do not reproduce the paper

The dataclass defaults in `config.py`, `train.py` and `model.py` are from pilot rounds.
`python train.py --r 3 --d 5` will **not** give you a paper cell. Use `sweep.py`, which
hardcodes the correct values in its `FIXED` dict, or pass them explicitly as shown below.

| Field | Code default | Paper |
| :--- | :---: | :---: |
| `LangSpec.n_entities` | 2000 | **200** |
| `LangSpec.ctx_len` | 600 | 1024 (`ModelCfg.ctx_len`) |
| `CorpusCfg.n_stmts_lo/hi` | 60 / 100 | **45 / 55** |
| `TrainCfg.total_steps` | 15000 | **16000** |
| `TrainCfg.batch_docs` | 192 | **256** |
| `TrainCfg.eval_every` | 1500 | **1000** |
| `TrainCfg.eval_docs` | 8000 | **4000** |
| `config.R_OLD_GRID` | `[1,2,3,5,8,12]` | **`[3,5,8,12,16]`** |

`R_OLD_GRID` matters most. The analyzed grid drops `R_old ∈ {1,2}` (at 1 the edit has no
domain; at 2 the retrieval circuit fails to form within budget in four of five columns) and
adds 16. `phase_configs()`, `EXTREME_A/B` and `hist_configs()` are pilot scaffolding and did
**not** produce any published result.

---

## 📂 Project Structure

```text
.
├── config.py               # LangSpec, CorpusCfg (5 data knobs), validate_cfg, dd_band
├── vocab.py                # Integer tokens, no BPE — one value = one token
├── generator.py            # Document sampling + 5 structural invariants
├── model.py                # Decoder-only LM: RoPE, untied embed, per-head QK-RMSNorm
├── probe.py                # The causal edit + observational attribution (frozen)
├── probes.py               # Probe suite v2 (frozen before first training run)
├── train.py                # One cell per run, online probing, final ckpt only
│
├── sweep.py                # ★ Grid driver — hardcodes the paper's hyperparameters
├── selfcheck.py            # Pre-flight: design-correctness assertions
├── bench.py                # Learnability ladder (1key / 2key / 2key_far)
├── pilots.py               # Diagnostic pilot runner (subprocess per pilot)
│
├── go_nogo.py              # ★ Terminal readout: median, sign fraction, binomial p, state
├── traj.py                 # ★ Escape timing from the loss derivative, multi-seed
├── ledger.py               # Experiment ledger — which number came from which pipeline
├── paper_numbers.py        # ★ Single source of truth for every number in the paper
├── verify_seed2.py         # Cross-checks seed 2 against the paper's hardcoded s0/s1
├── seed2_traj.py           # Seed-2 trajectory (superseded by traj.py)
│
├── covar.py                # ★ tab:covar + app:gen — generator invariants, CPU only
├── collide.py              # ★ tab:collide — 7-rule collision rates vs truth
├── overlap.py              # app:posneg — positional account of plateau-phase Δ
├── hist.py                 # Per-document Δ distribution shape (ASCII, check bimodality)
│
├── flatdir.py              # ★ Flat-direction geometry: 4 layers of evidence
├── summarize_flat.py       # flatdir → narrow summary + 3 consistency checks
├── flat_find.py            # flatdir → app:flat LaTeX rows
├── flat_compare.py         # Two-seed geometry comparison (kills the one-seed limit)
├── flat_ratios.py          # Curvature and nats ratios, post-escape only
├── an_grad.py              # Gradient orthogonality + noise floor + positive control
├── an_fd.py                # Finite-difference version (measures L, not grad L)
│
├── calib.py                # Slot-sparsity arm: length calibration
├── pair.py                 # Slot-sparsity arm: paired readout on the overlap band
├── fixband_check.py        # Fixed-band arm: CPU pre-flight, exits nonzero on FAIL
├── fixband_analyze.py      # Fixed-band arm: escape, plateau height, terminal readout
├── gamma_table.py          # app:qk — the 8-row QK-gain table
│
├── figs.py                 # fig_dist (per-doc distributions) + legacy fig_phase
├── figs2.py                # ★ Fig 1 + Fig 2 (replacement versions) + shared defs
├── fig_traj.py             # Continuous escape trajectories
├── plot_continuous.py      # Metric-artifact rebuttal (Schaeffer et al. 2023)
└── plot_perdoc.py          # Six-cell per-doc Δ panel
```

★ = produces something that appears in the paper.

---

## 💻 Installation

```bash
git clone https://github.com/YourUsername/shortcut-before-circuit.git
cd shortcut-before-circuit
pip install torch matplotlib numpy
```

Requires PyTorch ≥ 2.4 (for `nn.RMSNorm`) and a CUDA GPU with bf16 support. Each run is
26.1M parameters × 16000 steps. No `requirements.txt` is pinned yet.

---

## 🚀 Reproduction Pipeline

### Stage 0 — Pre-flight (CPU, ~30s)

Never train before this passes. Any failed assertion means the generator has a shortcut and
training is meaningless.

```bash
python sweep.py --check
```

Optional deeper checks:

```bash
# design-correctness assertions on one config (4000 docs)
python selfcheck.py

# can the architecture do induction at all, and how fast?
python bench.py --qk-gain 2.0 --steps 4000 --seeds 3
```

### Stage 1 — Train the grid (GPU)

`sweep.py` is the intended entry point. It refuses to start if the pre-check fails, skips
completed runs (`.pt` exists), and runs seed-major with the four corners and centre first so
the phase diagram's shape is visible after ~7 hours rather than ~91.

```bash
python sweep.py --dry-run                    # print the commands, run nothing
python sweep.py --out runs_g2                # the 75-run grid
python sweep.py --collect                    # summarize completed runs, anytime
```

One cell by hand (the `FIXED` dict of `sweep.py`, spelled out):

```bash
python -u train.py \
  --r 3 --d 5 --seed 0 --steps 16000 \
  --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
  --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
  --eval-every 1000 --eval-docs 4000 --workers 4 \
  --out runs_g2 --tag grid
```

`--r` is `R_old`; `--d` is the *nominal* ΔD — the actual support is `dd_band(d)`, always an
interval. A constant ΔD makes "copy the value at a fixed token offset" a 100%-correct rule,
and `validate_cfg` rejects it.

### Stage 2 — Readout

```bash
# terminal causal readout, 400 docs/cell, incremental cache (~1.5h for 75 cells)
python go_nogo.py --pattern 'R*_grid' --out runs_g2

# escape timing at 100-step resolution. Calibrate against the paper first:
python traj.py runs_g2 --seeds 0 --suffix _grid
python traj.py runs_g2 /path/s1 /path/s2 --seeds 0 1 2 --suffix _grid

# generator invariants + covariates (CPU, ~20min, parallel with training)
python covar.py --txt runs_g2/covar.txt --tex runs_g2/covar.tex

# 7-rule collision rates
python collide.py --out runs_g2/collide

# every number in the paper, sectioned by paper location
python paper_numbers.py --s0-dir runs_g2 --s1-dir /path/s1 \
  --gonogo runs_g2/go_nogo.txt /path/s1/go_nogo.txt

# which number came from which pipeline (A/B/C/X calibres are not comparable)
python ledger.py --dirs runs runs_g2 --csv ledger.csv

# cross-check seed 2 against the paper's hardcoded seed 0/1 columns
python verify_seed2.py /path/s2/go_nogo.txt
```

### Stage 3 — Figures

```bash
# Fig 1 (paired-point replication + flip run) and Fig 2 (escape)
python figs2.py --s0-dir runs_g2 --s1-dir /path/s1 \
  --gonogo runs_g2/go_nogo.txt /path/s1/go_nogo.txt \
  --flip-run 3 3 1 --prefix fig

# Fig 3: per-document Δ distributions along the ΔD=5 column
python figs.py runs_g2/go_nogo.txt \
  --perdoc runs_g2/go_nogo.txt.perdoc.jsonl --prefix fig

# metric-artifact rebuttal
python plot_continuous.py --runs runs_g2 --cells R3_D2,R16_D2 --seeds 1
python plot_continuous.py --runs runs_g2 --schema-only   # dump keys, exit
```

`plot_perdoc.py` and `fig_traj.py` read hardcoded paths (`runs_g2/...`) and take no
arguments — edit the constants at the top if your layout differs.

```bash
python plot_perdoc.py
python fig_traj.py
```

---

## 🔬 The Flat-Direction Suite

This is the machinery behind the paper's central claim: the direction in parameter space that
trades recency for rarity is one the objective supplies almost no pressure along. Four layers
of evidence, because no single one is sufficient — a first-order angle can be overturned by
second-order effects, and an angle near zero is meaningless without a noise floor.

Training with intermediate checkpoints (geometry needs weights, not just readouts):

```bash
python -u train.py --r 3 --d 5 --seed 0 --steps 16000 \
  --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
  --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
  --eval-every 1000 --eval-docs 4000 --workers 4 \
  --out runs_g2 --tag grid \
  --ckpt-every 2000
```

Saving a checkpoint consumes no RNG and does not touch the dataloader, so the trajectory
stays bit-identical.

```bash
# the measurement itself: cos(g_L, g_Δ), noise floor, finite differences, curvature
python flatdir.py --tag R3_D5_s0_grid --out runs_g2 --docs 400 \
  --json runs_g2/flat_R3D5_s0.jsonl

# narrow summary + 3 internal consistency checks (directional derivative,
# precision floor, sigmoid regime)
python summarize_flat.py runs_g2/flat_R3D5_s0.jsonl

# → app:flat LaTeX rows. Takes exactly two files, in this order.
python flat_find.py runs_g2/flat_R3D5_s0.jsonl runs_g2/flat_R16D2_s0.jsonl > flat_rows.tex

# two seeds of the same cell: is flatness a property of the construction
# or of the run? (This is what removes the one-seed limitation.)
python flat_compare.py runs_g2/flat_R3D5_s0.jsonl runs_g2/flat_R3D5_s1.jsonl
python flat_ratios.py  runs_g2/flat_R3D5_s0.jsonl runs_g2/flat_R3D5_s1.jsonl

# standalone gradient orthogonality with noise floor and positive control
python an_grad.py --tags R3_D5_s0_grid R16_D2_s0_grid --out runs_g2 --docs 400

# finite-difference version: measures L rather than grad L
python an_fd.py --tag R3_D5_s0_grid --out runs_g2 --docs 400
```

Everything here runs in **fp32 with TF32 explicitly disabled** — the ΔL of interest is around
1e-5, and TF32's 10-bit mantissa would turn it into rounding noise. `summarize_flat.py`'s
precision-floor check exists to catch exactly that.

> **What this suite cannot establish.** Checkpoints do not include optimizer state, so this
> measures *geometry*: "the objective supplies no usable first- or second-order signal along
> this direction." It cannot say "the optimizer did not travel along it."

---

## 🧪 Side Arms

Each arm breaks one covariate that is collinear with an axis in the main grid.

### QK-gain (`γ`) — is the analyzed grid partly defined by the architecture?

```bash
python -u train.py --r 3 --d 5 --seed 0 --steps 16000 \
  --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
  --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
  --eval-every 1000 --eval-docs 4000 --workers 4 \
  --qk-gain 1.0 --out runs_gamma --tag gamma1

python go_nogo.py --pattern 'R*_gamma1' --out runs_gamma
python gamma_table.py runs_g2 runs_gamma          # → the 8-row LaTeX table
```

`gamma_table.py` validates three things and warns on stderr: n near 400 (mixing in
`--docs 200` results shifts frac+ in the third decimal), mass ≥ 0.5, and both gains present
for every cell.

### Fixed band — breaks `posCeil` × ΔD collinearity

The main grid's ΔD support widens with `d`, so the positional ceiling falls from 0.333 to
0.059 along the axis. A fixed-width band holds it at 0.111 everywhere.

```bash
# CPU pre-flight. Exits nonzero on FAIL — usable as a runner gate.
python fixband_check.py --r 3 --lo 1 4 8 16 --width 9 --docs 1500

# train: --dd-lo/--dd-hi override dd_band; --d becomes a label only
python -u train.py --r 3 --d 1 --seed 0 --steps 16000 \
  --dd-lo 1 --dd-hi 9 \
  --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
  --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
  --eval-every 1000 --eval-docs 4000 --workers 4 \
  --out runs_fixband --tag fb1-9

python fixband_analyze.py runs_fixband --pattern '_fb(\d+)-(\d+)$'
python covar.py   --fixband 9 --rows 3 --cols 1 4 8 16 --txt fb_covar.txt --tex fb_covar.tex
python collide.py --fixband 9 --rows 3 --cols 1 4 8 16 --out fb_collide
```

Passing only one of `--dd-lo`/`--dd-hi` is an error rather than a silent fallback to
`dd_band`.

### Slot sparsity — breaks slot-count × `R_old` collinearity

Slot count falls from 27.6 at `R_old=3` to 8.7 at 16 and cannot be held fixed without making
document length depend on `R_old`.

```bash
# find the document length at which R8's slot count matches R16's
python calib.py --dd 2 --docs 1500 --out calib_R8

# train R8 at that length, then read out
python -u train.py --r 8 --d 2 --seed 0 --steps 16000 \
  --stmts-lo 20 --stmts-hi 30 \
  --n-values 512 --n-entities 200 \
  --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
  --eval-every 1000 --eval-docs 4000 --workers 4 \
  --out runs_slot --tag slotmatch

# paired readout on the q_kept overlap band, where R_old and realized
# redundancy agree and only slot count differs
python pair.py runs_slot/go_nogo.txt.perdoc.jsonl runs_g2/go_nogo.txt.perdoc.jsonl
```

Matching slot count also moves realized redundancy (14.59 → 8.95 slots costs 5.75 → 4.02
redundancy), and §4.4 shows the effect scales with realized redundancy — hence the paired
subset rather than a whole-cell comparison.

### Depth — breaks nothing on its own, and says so

```bash
for L in 4 12; do for R in 3 8 16; do for S in 0 1; do
  python -u train.py --r $R --d 5 --seed $S --steps 16000 --n-layer $L \
    --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
    --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
    --eval-every 1000 --eval-docs 4000 --workers 4 \
    --out runs_depth --tag L${L}
done; done; done

python go_nogo.py --pattern 'R*_L4'  --out runs_depth
python traj.py runs_depth --seeds 0 1 --suffix _L4  --rows 3 8 16 --cols 5
python traj.py runs_depth --seeds 0 1 --suffix _L12 --rows 3 8 16 --cols 5
```

`d_model` stays at 512, so parameter count moves with depth (13.4M / 26.1M / 38.7M). Depth
and capacity are deliberately confounded here; a width arm at fixed depth is the complement
and is **not** in this repo.

Run the two suffixes separately — `_L*` matches both `_L4` and `_L12`, and `traj.py` prints
`AMBIGUOUS` then takes the first hit.

### Plateau-phase positional account

```bash
python overlap.py --docs 1500 --out overlap
```

Tests whether the offset `4ΔD+6` landing on a rewritten copy predicts the magnitude of the
plateau-phase negative Δ. Needs no new training.

---

## 🩺 Diagnostics

```bash
# per-document Δ shape, ASCII histogram. Run this before writing any summary
# sentence: median + sign fraction can hide a bimodal cell.
python hist.py runs_g2/go_nogo.txt.perdoc.jsonl --bins 25

# learnability ladder — is a failure the model or the data?
python bench.py --qk-gain 2.0 --steps 4000 --seeds 3

# diagnostic pilots, one subprocess each, logs to runs/<tag>.log
python pilots.py
```

`bench.py` reads: acc met → fine; stuck on the shortcut plateau with `prec` flat →
sharpening never started, raise `--qk-gain`; `prec` rising but acc short → raise `--steps`;
`prec` at chance → bug in `model.py`.

---

## 📐 Determinism

Training is bit-identical on rerun at a fixed seed. This is what licenses attributing the
cross-seed spread to initialization and data order rather than to nondeterminism. The
fixed-band `d=4` cell coincides with the main grid's `R_old=3, ΔD=8` cell (band `[4,12]`
already has width 9) and reproduces it to all recorded digits under a different configuration
name.

The training stream is infinite and the eval/probe sets are pinned to a disjoint RNG offset
(`seed_offset=1` vs the training stream's `1000+w`), so no evaluated document is ever
trained on.

---

## 🐛 Known Issues

**Two copy-diagnostic thresholds.** `train.py` writes `converged = copy_acc > 0.9` into each
checkpoint; `go_nogo.py` gates at `COPY_FLOOR = 0.95`, which is the paper's gate. Every
analyzed run sits at 1.000 so no published result depends on the gap, but a run with
`copy_acc ∈ (0.9, 0.95]` would be labelled converged in its checkpoint and rejected by the
readout. Prefer recomputing over reading the stored field.

**`ctx_len` disagreement.** `LangSpec.ctx_len = 600` but `ModelCfg.ctx_len = 1024`.
`validate_cfg` checks the token budget against the former while the model allocates the
latter. At `n_stmts ≤ 55` the longest document is ~225 tokens so neither bound binds, but
they should be unified.

**`.perdoc.jsonl` is append-mode.** Re-running `go_nogo.py` to the same path accumulates
rows across seeds, and `figs.py:fig_dist()` matches on `tag.startswith(...)` — it may pick a
row you did not intend. Check the line count before plotting.

**Fig 3 is seed 0.** The `R_old=3, ΔD=5` panel shows seed 0's shape (median −0.09,
frac+ 0.13). That cell reads 0.972 in seed 1, so the distribution shape shown is a property
of the run, not the cell.

**Stale pilot-era docstrings.** Several modules still describe a 90-run / 30-configuration
design, `n_values=128`, `ctx_len=1024` padding waste, or `probe_points=4`. The code is
current; the prose above it is not.

## 🚧 Not in this repo

Controls the paper names as incomplete. Each is a field that exists with no command-line
route:

- **No-QK-norm arm.** `ModelCfg.qk_norm` is a bool but `train.py` exposes only `--qk-gain`.
  Removing the normalization entirely is a different parameterization, not `gain=0`.
- **Learned-absolute positions.** `ModelCfg.pos` accepts `"learned"` and `"nope"`; both are
  diagnostic. `model.py`'s own docstring calls for a learned-absolute rerun on a few cells.
- **`p_update` arm.** `CorpusCfg.p_update` is fixed at 0.5. Varying it at fixed document
  length is what would separate slot count from copy dispersion and length.
- **Width arm.** The complement to the depth arm — vary `d_model` at fixed `n_layer`.
  `--d-model` exists and works; the runs were not done.
- **Non-aliased control.** Every cell is aliased by construction, so there is no condition in
  which to measure the same geometry and find it absent.

---

## 📜 Citation

```bibtex
@misc{liao2025shortcut,
      title={Shortcut Before Circuit: Document Statistics Time In-Context Conflict Resolution},
      author={Yijun Liao and Fanwei Liang},
      year={2025},
      eprint={XXXX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/XXXX.XXXXX},
}
```

## 📄 License

TODO — pick one before making the repo public. Apache 2.0 is assumed by the badge above.

<div align="center">
</div>
