# Paper-to-artifact coverage

Table labels below refer to the LaTeX identifiers in the submitted manuscript, so the map remains useful if printed table numbers change.

| Paper topic / label | Available artifact or entry point | Coverage |
|---|---|---|
| Main sign-fraction spread, tab:spread | runs_g2/go_nogo.txt.jsonl; figs2.py; run_ledger.py | Canonical 75-run records included. |
| Main terminal statistics and within-run strata, tab:stats, tab:within | runs_g2/go_nogo*.jsonl and perdoc records | Aggregate and per-document records included. Select the stated population. |
| Argmax transitions, tab:argmax-transition | runs_g2/go_nogo_argmax_v1.txt.jsonl and matching perdoc.jsonl; argmax_transitions.py | Inputs and cached summaries included. |
| Magnitude thresholds, tab:magnitude-threshold | Same argmax-enriched records; magnitude_thresholds.py | Inputs and cached summaries included. |
| Generator invariants, covariates and collisions | config.py, generator.py, covar.py, collide.py; archived_reports/; runs_fixband/collide.jsonl | Generators and available report snapshots included; not every final main-grid calibration export is present. |
| Disagreement-supervision comparisons, tab:nbdose, tab:nbruns | runs_g2_seeds/ and runs_nb/; run_ledger.py, verify_dose01.py, nb_pair.py | Recorded trajectories/caches and analysis code included. Failed and rescue conditions retained. |
| Candidate peaks, tab:escape, Figure 2 | runs_g2/ trajectories; figs2.py, paper_numbers.py, traj.py | Main-grid trajectories included; seven flagged candidates remain identified. |
| Pre-escape and temporal variation | runs_g2/, runs_constlr/, runs_cos32/; trajectory readers | Included records cover these main-grid and schedule arms; do not confuse all these files with one pooled population. |
| Excluded R_old=2 group, tab:posneg | General train.py and overlap.py | Original excluded-group logs not provided. |
| Shared-batch / interpolation checks | runs_g2/interp_R3D8.json, xc_R3D5.json | Available saved reports included. |
| Data-stream seeds, tab:dseed | runs_dseeds/; dseed_read.py | Saved records included. |
| Training budget comparisons, tab:budget | Available s4k/s32k records in runs_g2 and schedule directories | Only available records included; historical baseline/budget variants outside these directories are not recreated. |
| QK gain, tab:gamma / tab:gain | runs_gamma1/; gamma_table.py; qk_bound.py | Training/probe records included. Bound recomputation requires omitted weights. |
| Depth and width | runs_depth/, runs_w1024/; depth_peaks.py | Saved records included. |
| Rendered-language and fine-tuning scope, tab:surface | runs_nl/, runs_ft/, nl_data/, ft_data/, nl_collide.jsonl | Logs and metadata included; archived probes encoded as JSON. External pretrained weights are required for new FT runs. |
| Slot, update-probability and fixed-band controls | runs_slot/, runs_pupd/, runs_fixband/; calib.py, calib_pupd.py, pair.py, fixband_check.py | Available logs/caches included. Generator checks can be rerun. |
| High-redundancy non-aliased construction, tab:nbr16 | archived_reports/runs_nb__covar_r16_k*.txt.json; covar.py, nb_pair.py | Available construction reports retained; no invented successful run added. |
| Non-aliased heads and residual patching, app:heads | heads.py | Code included; original output logs absent. |
| Aliased-grid head ablations, app:headsalias | runs_g2/ablate_R3D8.json and joint_s2.json; ablate.py, ablate_joint.py | Available reports and code included. |
| Geometry, tab:flatctrl | runs_flat/, runs_nb/flat_nbrec30.jsonl, flat_nbrar30.jsonl; flatdir.py, flatctrl.py | Partial original geometry outputs included. Final objective-constrained control outputs and their complete implementation are not present in the supplied source collection. |

This map does not certify numerical reproduction of each result. It documents which original artifacts were available when the archive was assembled.

