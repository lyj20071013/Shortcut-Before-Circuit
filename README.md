# Anonymous code and data supplement

**Paper:** Same Task Performance, Different Intervention Readouts: Cross-Run Variation under In-Context Rule Aliasing

This archive contains the available source code, saved training trajectories, terminal probe summaries, per-document intervention records, and selected probe inputs. The main study uses 25 configurations and three seeds. Additional directories retain seed extensions, schedule changes, supervision controls, architecture changes, and rendered-language/fine-tuning arms.

**No model weights or checkpoints are distributed in this archive.** Saved-record analysis works without them. New model evaluations, gradient measurements, and ablations require training the relevant models first. See the explicit coverage limits below; this is not a claim that every appendix table can be reconstructed directly from the bundled logs.

## 1. Start here

Run commands from the directory containing this README. Use UTF-8 output when redirecting script output on Windows.

~~~bash
python -m pip install -r requirements-analysis.txt
python validate_bundle.py
python figs2.py --prefix outputs/fig
~~~

The last command regenerates the two final figures from the supplied records. It performs no training and reads no checkpoint. The scripts create the output directory.

Python 3.13.1, NumPy 2.2.6, Matplotlib 3.10.6 and SciPy 1.16.3 were available for packaging checks. These are the packaging environment, not a recovered lockfile for the original training jobs. The analysis requirements pin that environment. The original logs do not capture a complete package lockfile or all GPU/driver details.

For training, install a PyTorch build appropriate for the available CUDA environment and the optional pretrained-model dependencies:

~~~bash
python -m pip install -r requirements-training.txt
~~~

The training code uses PyTorch, streamed synthetic data, and GPU mixed precision. The provided multi-worker training configuration is intended for Linux/CUDA. Using a different worker count or numerical environment may change the sampled stream or optimization trajectory; no cross-platform bitwise-reproduction guarantee is made. Full training was not run as part of packaging.

## 2. What is in the archive

| Path | Purpose |
|---|---|
| Top-level Python files | Training, generators, model, probe, and analysis code. See CODE_INDEX.md. |
| runs_g2/ | Canonical 75-run grid, terminal and per-document probes, argmax/threshold summaries, shared-batch and model-intervention records, and a few auxiliary records. |
| runs_g2_seeds/ | Expanded seed sample for the selected R_old=3, D=8 cell. |
| runs_nb/ | Disagreement-supervision doses, both labeling rules, second-cell extensions, failed/rescue conditions, and available non-aliased geometry records. |
| runs_constlr/, runs_cos32/ | Constant-learning-rate and changed-cosine-period arms. |
| runs_dseeds/ | Initialization held fixed while the document-stream seed varies. |
| runs_gamma1/, runs_depth/, runs_w1024/ | QK-gain, depth, and width controls. |
| runs_slot/, runs_pupd/, runs_fixband/ | Slot-count, update-probability, and fixed-distance-band controls. |
| runs_nl/, runs_ft/ | Rendered-language training and pretrained-model fine-tuning trajectories. |
| runs_flat/ | Available exploratory geometry trajectories, terminal probes and flatdir outputs. These do not include the complete final flatctrl table outputs. |
| nl_data/, ft_data/ | Vocabulary/token-pool metadata and JSON-encoded archived probe arrays. |
| nl_collide.jsonl | Saved rendered-language candidate-rule collision records. |
| archived_reports/ | Historical calibration reports preserved as JSON text strings; these are not newly reconstructed statistics. |
| archived_summaries/ | An additional fine-tuning summary export with a different schema. Use the main runs_ft logs and current ft_read.py for analysis. |
| manifest.json | File sizes, SHA-256 checksums, record counts, and packaging provenance. |
| COVERAGE.md | Mapping from paper topics to available inputs and missing original outputs. |

Training logs, terminal summaries, and per-document records are intentionally distinct. A directory also contains unsuccessful or historical runs when they belong to the reported controls. Their presence does not make them eligible for a pooled result.

## 3. Data schema and interpretation

- A training trajectory usually begins with a kind=meta object containing corpus, train, model and spec fields, followed by train, eval, probe and done records. NL and FT use related but different metadata schemas.
- Terminal go_nogo summaries contain run tags and aggregate measurements. Per-document records are stored separately in files ending in perdoc.jsonl.
- The pair **(experiment directory, run tag)** identifies a run. Some extension directories reuse tags; do not merge runs by tag alone.
- Historical summaries may contain legacy seed fields. Canonical grid tags encode the seed used by figs2.py and the current main-grid analysis.
- JSONL means one JSON object per line. Existing Python-generated logs retain NaN for undefined measurements; Python's json reader accepts this extension. NaN is not zero, and strict JSON tools may need to handle it explicitly.
- Numeric observations have not been edited during packaging. A server-directory prefix was removed from the additional geometry records; probe arrays were represented by JSON values, exact dtype, shape and array checksums.
- Training-time probes and terminal probes can have different document counts. Do not silently pool them.
- In the main grid, frac_positive/frac_expected denotes positive intervention displacement among valid edit pairs; it is not a behavioral rarity-answer rate. The exact field meaning is arm-specific, particularly for FT gated versus ungated summaries.
- A mass-restricted median and an unrestricted median summarize different document populations. Use the paper's stated population when selecting fields.
- Implementation comments are retained. Author-facing revision notes and overstrong diagnostic interpretations were clarified for distribution. Archived reference constants and diagnostic checks remain; saved measurements are unchanged. Historical checks can still refer to earlier drafts, so use the final paper for the stated analysis populations and claims.

## 4. Analyses from saved records

These commands use existing JSON/JSONL inputs. Creating the figures and validating the archive were exercised during packaging; the numerical audit commands below were not rerun as part of packaging.

First create a fresh output directory if needed:

~~~bash
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
~~~

### Main grid and figures

~~~bash
python figs2.py --prefix outputs/fig
python paper_numbers.py --s0-dir runs_g2 --s1-dir runs_g2 --s2-dir runs_g2 --gonogo runs_g2/go_nogo.txt.jsonl
python run_ledger.py --root . --out outputs/ledger.tsv
~~~

Use the final figs2.py for the candidate-peak definitions and final figure labels. Historical figure scripts are excluded.

### Argmax transitions and magnitude thresholds

~~~bash
python argmax_transitions.py runs_g2/go_nogo_argmax_v1.txt.jsonl --perdoc runs_g2/go_nogo_argmax_v1.txt.perdoc.jsonl --json outputs/argmax.json --txt outputs/argmax.txt --tex outputs/argmax.tex
python magnitude_thresholds.py --perdoc runs_g2/go_nogo_argmax_v1.txt.perdoc.jsonl --summary runs_g2/go_nogo_argmax_v1.txt.jsonl --out outputs/magnitude
~~~

### Supervision, schedules and architecture

~~~bash
python verify_dose01.py --root .
python constlr_read.py --seeds 0 1 2 3 4 5 6 7 8 9 --at 16000 --json outputs/constlr.json
python dseed_read.py --dir runs_dseeds --out outputs/dseeds.json
python depth_peaks.py --dir runs_depth
python fixband_analyze.py runs_fixband
python ft_read.py --dir runs_ft --out outputs/ft_summary.json
python traj.py runs_g2 --seeds 0 1 2 --suffix _grid
~~~

The label-permutation calculations in run_ledger.py and verify_dose01.py enumerate the specified reference assignments. Their output does not resolve selection bias, establish exchangeability, or turn the selected-cell comparisons into confirmatory evidence.

nb_dose.py can display selected cache sets. The expanded-seed comparisons should use the explicit run identities in run_ledger.py / verify_dose01.py, rather than indiscriminately merging every cache.

### Available geometry records

~~~bash
python summarize_flat.py runs_flat/flatdirR3_D5.jsonl runs_flat/flatdirR16_D2.jsonl runs_flat/flatdirR3_D5_s1.jsonl
~~~

These files describe the available exploratory readout/loss geometry measurements. They are not a substitute for the missing full control-readout output listed in COVERAGE.md.

## 5. Recreate synthetic training runs

Prefer the configuration recorded in a particular log to a script's general-purpose defaults. In particular, the model MLP width must be copied exactly, not inferred from a rounded width ratio.

The supplied helper first prints a recipe without loading a model:

~~~bash
python retrain_from_log.py runs_g2/R3_D8_s0_grid.jsonl --out outputs/retrained_main
~~~

To run that recipe:

~~~bash
python retrain_from_log.py runs_g2/R3_D8_s0_grid.jsonl --out outputs/retrained_main --execute
~~~

It calls the provided training function with the saved corpus, training, model and vocabulary settings. The output directory is changed so archived logs are not overwritten. It is a retraining recipe, not a checkpoint-resume command. Use a fresh destination.

For the entire main grid, sweep.py contains the explicit 5 x 5 x 3 configuration:

~~~bash
python sweep.py --dry-run --out outputs/retrained_grid
python sweep.py --out outputs/retrained_grid
~~~

The second command starts the grid. Completed training creates weights that can then be evaluated:

~~~bash
python go_nogo.py R3_D8_s0_grid --out outputs/retrained_main --docs 400
~~~

Other synthetic arms with complete corpus/train/model/spec metadata can use retrain_from_log.py in the same way. The helper preserves the recorded MLP width, learning-rate schedule, initial gain, seed, worker count and training budget. Historical logging/probe density can differ between recorded jobs and the current source, so this is not a guarantee of identical intermediate logging.

For FP32 intermediate checkpoints needed by geometry analyses, specify steps explicitly:

~~~bash
python retrain_from_log.py runs_flat/R3_D5_s0_flat.jsonl --out outputs/retrained_flat --ckpt-steps 400,1000,2000,3000,4000,5000,6000,8000,10000,12000,14000,16000 --execute
~~~

The resulting checkpoints can be passed to flatdir.py and flatctrl.py. The existing flatctrl.py covers edit-based controls; it does not implement the two objective-constrained readouts needed to reconstruct the complete final geometry comparison. Do not treat a run of that script as reproduction of the entire final table.

qk_bound.py implements the RoPE-safe upper bound used in the corrected appendix. It requires a supplied or retrained checkpoint:

~~~bash
python qk_bound.py outputs/retrained_gamma/R3_D5_s0_gamma1.pt --json outputs/qk_bounds.json
~~~

This computes an upper bound from gain vectors, not the attained attention gaps. gamma_table.py summarizes terminal probe measurements and does not compute this bound.

## 6. Rendered-language and fine-tuning inputs

The original small probe arrays are encoded as readable JSON values. Restore them, with dtype/shape/checksum validation, by running:

~~~bash
python restore_probe_arrays.py --out generated_inputs
~~~

This restores ft_data/probe_R3_D8.npz and the archived nl_data/probe_R3_D8_s0.npz under generated_inputs, together with available metadata. It does not create model weights.

The saved NL probe uses an older seed-suffixed filename. The current nl_train.py expects a shared probe_R3_D8.npz generated by the current nl_corpus.py. Keep the archived probe for inspection; generate current NL inputs explicitly instead of silently renaming it:

~~~bash
python nl_corpus.py --rows 3 --cols 8 --out generated_inputs/nl_current
python nl_train.py --r 3 --d 8 --seed 0 --data generated_inputs/nl_current --out outputs/retrained_nl
~~~

Repeat with the recorded seeds for the other runs. NL training data stream online; the unrelated legacy finite training-array file is excluded.

FT uses the external pretrained model Qwen/Qwen2.5-0.5B. Its weights and tokenizer are not bundled. The archived logs record the model name but not an immutable upstream revision. Pretrained-model reproducibility consequently also depends on obtaining the intended upstream assets.

The archived FT probe can be restored as above. Alternatively, rebuild tokenizer-level inputs:

~~~bash
python ft_data.py --model Qwen/Qwen2.5-0.5B --src hf --no-mirror --out generated_inputs/ft_current
python ft_train.py --model Qwen/Qwen2.5-0.5B --src hf --no-mirror --seed 0 --data generated_inputs/ft_current --out outputs/retrained_ft --tag ft
~~~

The explicit --no-mirror option uses the configured official Hugging Face source rather than the historical mirror default. ModelScope support is optional and is not needed for these commands.

## 7. Coverage limits

The following original artifacts were not available for this archive:

1. Raw logs for the excluded R_old=2 group discussed in the positional-shortcut appendix. That group is outside the 75-run main grid. The general generator, train.py, overlap.py and evaluation code are provided, but a complete original-run configuration/log export for that group is not bundled.
2. The full final flatctrl output containing the edit-based controls and the two objective-constrained readouts. Available flatdir records, geometry trajectories and non-aliased geometry records are included. The supplied flatctrl.py does not contain those two additional objective-constrained readouts.
3. Original output files for the non-aliased head/residual-patching experiments. heads.py is provided. Available aliased-grid ablation, joint-ablation and interpolation/shared-batch records are included in runs_g2.
4. All terminal and intermediate model weights, including the QK gains needed to recompute the terminal bound column directly.
5. A complete locked training environment and immutable revision of external pretrained assets.

These limits concern archive coverage. The paper contains the reported summaries, and the provided generators/training code support new runs, but unavailable original records are not fabricated or reconstructed from rounded paper values.

The reproducibility statement should describe the supplied code and available logs, rather than promise direct reconstruction of every table from this archive.

## 8. File integrity and provenance

Run python validate_bundle.py to verify hashes and parse the distributed JSON/JSONL. It checks packaging integrity, not the scientific correctness of measurements. manifest.json records per-file checksums, record counts and transformations.

Packaging changes include short English module introductions, clarified internal documentation and diagnostic messages, portable defaults for several readers, a server-path-prefix removal in additional geometry records, and reversible representation of small probe arrays as JSON. The three reproduction helpers and the validator are supplied with this archive. The source experiment functions were not replaced by alternative implementations.

No manuscript source, author names, email addresses, repository history, cache folders, checkpoints, API credentials or unrelated external-source extraction corpora are intentionally included. No new software/data license is assigned by this packaging step.

