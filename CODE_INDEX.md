# Source-code index

Run the documented commands from the archive root. GPU-dependent scripts require regenerated or separately supplied checkpoints.

| Script | Role |
|---|---|
| ablate.py | Single-head ablations in the aliased main grid. |
| ablate_joint.py | Joint-head ablations and accuracy/copy diagnostics. |
| an_fd.py | Exploratory finite-difference measurements requiring checkpoints. |
| an_grad.py | Exploratory gradient measurements requiring checkpoints. |
| argmax_transitions.py | Aggregate base/edit argmax transitions from saved probe records. |
| calib.py | Generator calibration for slot-count matching. |
| calib_pupd.py | Generator calibration for update-probability matching. |
| coext_audit.py | Sequence-rule collision utilities used by the rendered-language checks. |
| collide.py | Evaluate candidate-rule collisions on generated documents. |
| config.py | Dataclass definitions for the synthetic language and corpus. |
| constlr_read.py | Summarize constant-learning-rate trajectories from saved records. |
| covar.py | Measure corpus covariates and generator invariants. |
| depth_peaks.py | Summarize candidate peaks and trajectories in the depth arm. |
| dseed_read.py | Summarize runs with initialization and data-stream seeds separated. |
| figs2.py | Render the final paper figures from cached main-grid records. |
| fixband_analyze.py | Summarize the fixed-distance-band control arm. |
| fixband_check.py | Check the fixed-distance-band generator configuration. |
| flat_compare.py | Compare saved exploratory geometry measurements across seeds. |
| flat_find.py | Extract rows from saved exploratory geometry measurements. |
| flat_ratios.py | Summarize finite-difference measurements from saved records. |
| flatctrl.py | Measure answer-preserving edit controls using FP32 checkpoints. |
| flatdir.py | Measure readout/loss gradients and symmetric finite differences. |
| ft_data.py | Construct tokenizer-level probe data for the fine-tuning arm. |
| ft_pool.py | Build single-token value pools for the selected tokenizer. |
| ft_posceil.py | Estimate the tokenizer-level fixed-position diagnostic. |
| ft_read.py | Summarize fine-tuning trajectories from saved records. |
| ft_ruleprior.py | Measure pretrained-model rule preferences before fine-tuning. |
| ft_tokcheck.py | Check token-pool compatibility for pretrained tokenizers. |
| ft_train.py | Fine-tune a pretrained model on the rendered synthetic task. |
| ft_zeroshot.py | Measure pretrained-model task and copy diagnostics. |
| gamma_table.py | Summarize gain-arm terminal readouts; does not compute the RoPE-aware bound. |
| generator.py | Generate controlled assignment-language documents. |
| go_nogo.py | Evaluate terminal checkpoints and save aggregate and per-document probes. |
| heads.py | Patch residual streams and ablate heads in disagreement-supervision arms. |
| interp.py | Evaluate interpolation and shared-batch comparisons between models. |
| magnitude_thresholds.py | Aggregate displacement thresholds from saved per-document probes. |
| model.py | Decoder-only Transformer with RoPE, RMS normalization, and SwiGLU. |
| nb_dose.py | Summarize disagreement-supervision dose arms from terminal caches. |
| nb_flat.py | Compare saved geometry measurements between supervision arms. |
| nb_pair.py | Check token/label matching between the two supervision conditions. |
| nb_stream.py | Inspect disagreement-supervision training-stream construction. |
| nl_collide.py | Evaluate candidate-rule collisions in the rendered-language arm. |
| nl_corpus.py | Build rendered-language vocabulary and probe arrays; training data stream online. |
| nl_gates.py | Check rendered-language construction and edit invariants. |
| nl_generator.py | Vocabulary pools and supporting rendered-language definitions. |
| nl_preflight.py | Check the rendered-language training and probe interfaces. |
| nl_regate.py | Re-evaluate rendered-language checkpoints with mass-restricted summaries. |
| nl_render.py | Render synthetic assignment documents as English statements. |
| nl_train.py | Train a Transformer from scratch on rendered-language documents. |
| overlap.py | Measure fixed-position overlap with rewritten statements. |
| pair.py | Summarize matched subsets from per-document records. |
| paper_numbers.py | Summarize manuscript-related quantities from main-grid trajectories and caches. |
| probe.py | Candidate rules, answer-preserving edits, and intervention measurements. |
| propB.py | Measure behavioral answers on generated rule-disagreement documents. |
| run_ledger.py | Build a run ledger and summarize the saved terminal/trajectory records. |
| seed2_traj.py | Summarize seed-2 trajectories using the final figure definitions. |
| seed_ext.py | Summarize seed-extension terminal records. |
| selfcheck.py | Generator and edit invariant checks. |
| summarize_flat.py | Summarize saved geometry records. |
| sweep.py | Launch or list the 75-run main grid. |
| train.py | Train one synthetic-task configuration and record diagnostics. |
| traj.py | Summarize candidate peaks and temporal readout variation. |
| verify_dose01.py | Reconstruct the exploratory 1% dose comparisons from saved records. |
| vocab.py | Integer-token vocabulary for the assignment language. |

## Packaging helpers

| Script | Role |
|---|---|
| validate_bundle.py | Verify distributed file hashes and parse formats without recalculating results. |
| restore_probe_arrays.py | Restore the archived arrays from JSON and verify array checksums. |
| retrain_from_log.py | Display a recorded synthetic-run recipe; train only with --execute. |
| qk_bound.py | Compute the RoPE-safe gain-vector upper bound from supplied weights. |
