# Shortcut Before Circuit

Code for *Shortcut Before Circuit: Document Statistics Time In-Context Conflict Resolution*.

We train small transformers on a synthetic assignment language in which two candidate
resolution rules — **recency** ("take the most recently written value") and **rarity**
("take the value written fewest times") — are *exactly coextensive*: they select the same
value on every training document. The training objective is therefore indifferent between
them, and no observational analysis can separate them. We separate them with a minimal
causal edit that inverts the multiplicities of the two competing values while holding the
ground truth, the token count, and the answer position fixed.

The main finding is a split. **Which** rule a model commits to does not replicate across
seeds: 13 of 25 cells differ by more than 0.3 in sign fraction across three seeds, the
largest by 0.879 against a binomial standard error of 0.025, with individual endpoints
decisive at p < 1e-50 in *opposite* directions. **When** a mechanism appears does
replicate: escape from a closed-form positional shortcut is monotone in redundancy across
four separable levels in all three seeds.

---

## ⚠️ Defaults do not reproduce the paper

The dataclass defaults in `config.py`, `train.py` and `model.py` are from earlier pilot
rounds. Running `python train.py` with no arguments will **not** give you a cell from the
paper's grid. Every difference we know of:

| Field | Code default | Paper |
|---|---|---|
| `LangSpec.n_entities` | 2000 | **200** |
| `LangSpec.ctx_len` | 600 | 1024 (see `ModelCfg.ctx_len`) |
| `CorpusCfg.n_stmts_lo/hi` | 60 / 100 | **45 / 55** |
| `TrainCfg.total_steps` | 15000 | **16000** |
| `TrainCfg.batch_docs` | 192 | **256** |
| `TrainCfg.eval_every` | 1500 | **1000** |
| `TrainCfg.eval_docs` | 8000 | **4000** |
| `config.R_OLD_GRID` | `[1, 2, 3, 5, 8, 12]` | **`[3, 5, 8, 12, 16]`** |

`R_OLD_GRID` is the important one. The analyzed grid drops `R_old ∈ {1, 2}` and adds
`16`. `R_old = 1` leaves the edit with no domain (there are no copies to rewrite) and
`R_old = 2` fails to form the retrieval circuit within budget in four of five columns; both
exclusions are argued in the paper. `phase_configs()` and the `EXTREME_A`/`EXTREME_B` /
`hist_configs()` helpers are pilot-era scaffolding and are **not** what produced the
results — pass parameters explicitly, as below.

`LangSpec.ctx_len = 600` vs `ModelCfg.ctx_len = 1024` is a real inconsistency we have not
resolved. `validate_cfg` checks token budget against `LangSpec.ctx_len`, so the corpus is
validated against 600 while the model allocates 1024 positions. At `n_stmts ≤ 55` the
longest document is ~225 tokens, so neither bound binds and results are unaffected, but
the two should be unified.

---

## Setup

```bash
pip install torch          # ≥ 2.4 (nn.RMSNorm); CUDA GPU with bf16 support
```

`requirements.txt` is not pinned yet. The runs in the paper used bfloat16 autocast on a
single GPU; each cell is ~26.1M parameters and 16000 steps.

## Reproducing one cell

```bash
python -u train.py \
  --r 3 --d 5 --seed 0 --steps 16000 \
  --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
  --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
  --eval-every 1000 --eval-docs 4000 --workers 4 \
  --out runs_main --tag grid
```

`--r` is `R_old`, `--d` is the nominal `ΔD` (the actual band is
`config.dd_band(d)` — an interval, never a point; a constant `ΔD` makes "copy the value at
a fixed token offset" a 100%-correct rule and `validate_cfg` rejects it).

## Reproducing the grid

`R_old ∈ {3, 5, 8, 12, 16} × ΔD ∈ {2, 3, 5, 8, 16} × seed ∈ {0, 1, 2}` = 75 runs.

```bash
for S in 0 1 2; do for R in 3 5 8 12 16; do for D in 2 3 5 8 16; do
  python -u train.py --r $R --d $D --seed $S --steps 16000 \
    --n-values 512 --n-entities 200 --stmts-lo 45 --stmts-hi 55 \
    --batch 256 --lr 1e-3 --sched cos --wd 0.1 \
    --eval-every 1000 --eval-docs 4000 --workers 4 \
    --out runs_main --tag grid > runs_main/R${R}_D${D}_s${S}_grid.log 2>&1
done; done; done
```

## Readout and tables

```bash
# terminal causal readout: median Δ, sign fraction, exact binomial p, state label
python go_nogo.py runs_main --out go_nogo.txt

# escape timing from the loss derivative (100-step resolution) + gate crossings
python traj.py runs_main --seeds 0 1 2 --suffix _grid --rows 3 5 8 12 16 --cols 2 3 5 8 16

# generator invariants and per-cell covariates (pre-training, no model involved)
python covar.py   --txt covar.txt --tex covar.tex
python collide.py --out collide          # writes collide.tex + collide.jsonl

# figures
python figs.py  go_nogo.txt --perdoc go_nogo.txt.perdoc.jsonl --prefix fig
python figs2.py ...
```

`go_nogo.py` gates every readout on four conditions, all four required: a cross-cell
gradient in `R_old`; the sign of Δ; probability mass remaining on `{v_old, v_new}`
(`MASS_FLOOR = 0.50`, since the edit is off-distribution and a model that has left the task
gives a meaningless readout); and a control edit on a non-queried slot whose `|Δ|` must be
far below the main readout. The primary dependent variable is the **sign fraction** with an
exact binomial test, not the mean — per-document Δ is heavy-tailed, and on one cell the
mean moves 27% across post-escape checkpoints while the sign fraction moves 3.3%.

`.perdoc.jsonl` files are opened in append mode. If you re-run `go_nogo.py` into the same
path you will get multiple seeds' rows in one file, and `figs.py:fig_dist()` matches on
`tag.startswith(...)`, so it may pick a row you did not intend. Check the line count before
plotting.

---

## Repository map

Verified against the source:

| File | Role |
|---|---|
| `config.py` | `LangSpec` (language scale), `CorpusCfg` (the five data-side knobs), `validate_cfg` (fail-before-generate checks), `dd_band` (nominal `d` → uniform `ΔD` interval) |
| `generator.py` | Document sampling. Five structural invariants documented in the module docstring: per-document rebinding, exact non-degenerate `ΔD`, no repeated value ids within a document, uniform update placement, and matched local structure between the queried slot and filler slots |
| `model.py` | Decoder-only transformer. RoPE (not learned-absolute, which would inflate the positional rule), untied embeddings (needed for logit attribution), no dropout (infinite stream, nothing to overfit), per-head QK RMSNorm with learned gain at 2.0 |
| `train.py` | One cell per run. Infinite training stream; eval and probe sets pinned to a disjoint RNG offset. Online probing at log-spaced points, storing only the final checkpoint |
| `probe.py` | The causal edit and the observational attribution. Hard constraints: token count, `answer_pos`, ground truth, statement count and `q_gap` all unchanged by the edit. `MAIN_GRID_EXCLUDE` drops `frequency` and `rarity` from *observational* attribution because both are identities under `max_updates = 1` |
| `go_nogo.py` | Terminal readout from checkpoints, with the four-condition gate and the state classifier |

Present but not yet documented here — grouped by apparent role, **not verified**, do not
trust these descriptions without checking:

- **Analysis / tables**: `paper_numbers.py`, `ledger.py`, `summarize_flat.py`, `gamma_table.py`, `overlap.py`, `hist.py`, `calib.py`
- **Flatness geometry (§4.6)**: `flatdir.py`, `flat_find.py`, `flat_compare.py`, `flat_ratios.py`, `an_grad.py`, `an_fd.py`
- **Arms**: `fixband_analyze.py`, `fixband_check.py`, `seed2_traj.py`, `verify_seed2.py`
- **Plotting**: `figs.py`, `figs2.py`, `fig_traj.py`, `plot_continuous.py`, `plot_perdoc.py`
- **Infrastructure**: `vocab.py`, `sweep.py`, `bench.py`, `pilots.py`, `selfcheck.py`, `probes.py`, `pair.py`

## Determinism

Training is bit-identical on rerun at a fixed seed, which is what licenses attributing the
cross-seed spread to initialization and data order rather than to nondeterminism. One
fixed-band cell coincides with a main-grid cell and reproduces it to all recorded digits
under a different configuration name.

## Not in this repo

The paper names several controls as incomplete. In the code they appear as `ModelCfg` /
`CorpusCfg` fields with no command-line route:

- `ModelCfg.qk_norm` (bool, default `True`). `--qk-gain` exposes the *gain*, so the
  halved-gain arm is reachable, but removing the normalization entirely is not. Note that
  `qk_norm=False` is a different parameterization, not `gain=0`.
- `ModelCfg.pos` (default `"rope"`). The `"learned"` and `"nope"` branches are diagnostic;
  `model.py`'s docstring calls for a learned-absolute rerun on a few cells as a robustness
  check, which we did not do.
- `CorpusCfg.p_update` (default `0.5`). Varying it at fixed document length is what would
  separate slot count from copy dispersion and length; these three move together along the
  `R_old` axis in every run reported.

## Known inconsistency: two copy-diagnostic thresholds

`train.py` writes `converged = copy_acc > 0.9` into each checkpoint. `go_nogo.py` gates on
`COPY_FLOOR = 0.95`. The paper's gate is 0.95. Every analyzed run sits at 1.000, so nothing
in the results depends on the gap, but a run with `copy_acc ∈ (0.9, 0.95]` would be labelled
`converged=True` in its checkpoint and rejected by the readout. Prefer recomputing the
diagnostic over reading the stored field.

## Citation

```bibtex
@misc{shortcut-before-circuit,
  title  = {Shortcut Before Circuit: Document Statistics Time In-Context Conflict Resolution},
  author = {Liao, Yijun and Liang, Fanwei},
  year   = {2025},
  eprint = {arXiv:XXXX.XXXXX}
}
```

## License

TODO — pick one before making the repo public.
