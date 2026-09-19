"""Aggregate base/edit argmax transitions from saved probe records."""
import argparse
import json
import math
import os
import re

LABELS = ("old", "new", "other")
SCHEMA = "old-new-other-v1"
COUNT_FIELD = "argmax_transition_counts"
MASS_COUNT_FIELD = "argmax_transition_counts_mass"
MASS_FLOOR_FIELD = "argmax_transition_mass_floor"
ID_NAMESPACE_FIELD = "argmax_id_namespace"
ID_NAMESPACE = "raw_value_id"
DEFAULT_MASS_FLOOR = 0.5
R_VALUES = (3, 5, 8, 12, 16)
D_VALUES = (2, 3, 5, 8, 16)
SEEDS = (0, 1, 2)
GRID_RE = re.compile(r"^R(3|5|8|12|16)_D(2|3|5|8|16)_s([012])_grid$")


def canonical_tags():
    return tuple(f"R{r}_D{d}_s{s}_grid" for r in R_VALUES
                 for d in D_VALUES for s in SEEDS)


def transition_keys():
    return tuple(f"{a}->{b}" for a in LABELS for b in LABELS)


def empty_counts():
    return {key: 0 for key in transition_keys()}


def classify_argmax(prediction, new_value, old_value):
    """Map a raw value id to ``new``, ``old``, or ``other``.

    ``new_value`` is the preserved ground truth and ``old_value`` is the
    intervention contrast.  They must be distinct for the edit to be in scope.
    """
    if new_value == old_value:
        raise ValueError("new and old values must be distinct")
    if prediction == new_value:
        return "new"
    if prediction == old_value:
        return "old"
    return "other"


def transition_key(base_prediction, edit_prediction, new_value, old_value):
    base = classify_argmax(base_prediction, new_value, old_value)
    edit = classify_argmax(edit_prediction, new_value, old_value)
    return f"{base}->{edit}"


def normalize_counts(counts):
    """Return all nine cells and reject unknown, negative, or non-integer data."""
    if counts is None:
        counts = {}
    if not isinstance(counts, dict):
        raise ValueError(f"transition counts must be an object, got {type(counts).__name__}")
    unknown = set(counts) - set(transition_keys())
    if unknown:
        raise ValueError(f"unknown transition cells: {sorted(unknown)}")
    out = empty_counts()
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer, got {value!r}")
        out[key] = value
    return out


def validate_counts(counts, expected_n, *, label="transition counts"):
    out = normalize_counts(counts)
    if sum(out.values()) != expected_n:
        raise ValueError(
            f"{label} sum to {sum(out.values())}, expected {expected_n}")
    return out


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _mass_floor(value, label="mass floor"):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{label} must be a finite number in [0, 1], got {value!r}")
    return float(value)


def validate_row(row, mass_floor=DEFAULT_MASS_FLOOR):
    """Validate one run and return its unrestricted and restricted counts."""
    if not isinstance(row, dict):
        raise ValueError("run summary must be an object")
    tag = row.get("tag", "<untagged>")
    if row.get("argmax_transition_schema") != SCHEMA:
        raise ValueError(f"{tag}: missing schema {SCHEMA}")
    if row.get(ID_NAMESPACE_FIELD) != ID_NAMESPACE:
        raise ValueError(f"{tag}: id namespace is not {ID_NAMESPACE}")
    expected_floor = _mass_floor(mass_floor)
    stored_floor = _mass_floor(
        row.get(MASS_FLOOR_FIELD), f"{tag} stored mass floor")
    if stored_floor != expected_floor:
        raise ValueError(
            f"{tag}: mass floor is {stored_floor}, expected {expected_floor}")
    requested = _nonnegative_int(
        row.get("n_docs_requested"), f"{tag} n_docs_requested")
    if requested == 0:
        raise ValueError(f"{tag}: n_docs_requested must be positive")
    n = _nonnegative_int(row.get("n"), f"{tag} n")
    n_valid = _nonnegative_int(row.get("n_valid"), f"{tag} n_valid")
    if n > requested:
        raise ValueError(f"{tag}: n={n} exceeds n_docs_requested={requested}")
    if n_valid > n:
        raise ValueError(f"{tag}: n_valid={n_valid} exceeds n={n}")
    all_counts = validate_counts(
        row.get(COUNT_FIELD), n, label=f"{tag} unrestricted counts")
    mass_counts = validate_counts(
        row.get(MASS_COUNT_FIELD), n_valid,
        label=f"{tag} mass-restricted counts")
    for key in transition_keys():
        if mass_counts[key] > all_counts[key]:
            raise ValueError(f"{tag}: mass subset exceeds all for {key}")
    return all_counts, mass_counts


def counts_from_records(records, mass_floor=DEFAULT_MASS_FLOOR):
    """Rebuild both 3x3 tables from per-document raw argmax records."""
    threshold = _mass_floor(mass_floor)
    if not isinstance(records, list):
        raise ValueError("per-document records must be a list")
    all_counts, mass_counts = empty_counts(), empty_counts()
    seen = set()
    for pos, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {pos} is not an object")
        required = ("doc_index", "base_argmax", "edit_argmax", "old_value",
                    "new_value", "base_class", "edit_class", "transition",
                    "delta", "mass", "q_kept")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"record {pos} missing {missing}")
        doc_index = _nonnegative_int(record["doc_index"],
                                     f"record {pos} doc_index")
        if doc_index in seen:
            raise ValueError(f"duplicate doc_index {doc_index}")
        seen.add(doc_index)
        for field in ("base_argmax", "edit_argmax", "old_value", "new_value"):
            _nonnegative_int(record[field], f"record {pos} {field}")
        _nonnegative_int(record["q_kept"], f"record {pos} q_kept")
        delta = record["delta"]
        if (isinstance(delta, bool) or not isinstance(delta, (int, float))
                or not math.isfinite(float(delta))):
            raise ValueError(f"record {pos} has invalid delta {delta!r}")
        mass = record["mass"]
        if (isinstance(mass, bool) or not isinstance(mass, (int, float))
                or not math.isfinite(float(mass)) or not 0.0 <= mass <= 1.000001):
            raise ValueError(f"record {pos} has invalid probability mass {mass!r}")
        base_class = classify_argmax(record["base_argmax"], record["new_value"],
                                     record["old_value"])
        edit_class = classify_argmax(record["edit_argmax"], record["new_value"],
                                     record["old_value"])
        if record["base_class"] != base_class:
            raise ValueError(f"record {pos} base_class mismatch")
        if record["edit_class"] != edit_class:
            raise ValueError(f"record {pos} edit_class mismatch")
        key = f"{base_class}->{edit_class}"
        if record["transition"] != key:
            raise ValueError(
                f"record {pos} transition={record['transition']!r}, expected {key!r}")
        all_counts[key] += 1
        if mass >= threshold:
            mass_counts[key] += 1
    return all_counts, mass_counts


def validate_perdoc(summary_row, perdoc_row,
                    mass_floor=DEFAULT_MASS_FLOOR):
    """Cross-check a run summary against independently serialised records."""
    if not isinstance(perdoc_row, dict):
        raise ValueError("per-document row must be an object")
    tag = summary_row.get("tag", "<untagged>")
    if perdoc_row.get("tag") != tag:
        raise ValueError(f"per-document tag mismatch for {tag}")
    if perdoc_row.get("argmax_transition_schema") != SCHEMA:
        raise ValueError(f"{tag}: per-document schema is not {SCHEMA}")
    if perdoc_row.get(ID_NAMESPACE_FIELD) != ID_NAMESPACE:
        raise ValueError(f"{tag}: per-document id namespace is not {ID_NAMESPACE}")
    threshold = _mass_floor(mass_floor)
    perdoc_floor = _mass_floor(
        perdoc_row.get(MASS_FLOOR_FIELD), f"{tag} per-document mass floor")
    if perdoc_floor != threshold:
        raise ValueError(
            f"{tag}: per-document mass floor is {perdoc_floor}, expected {threshold}")
    expected_all, expected_mass = validate_row(summary_row, threshold)
    requested = _nonnegative_int(
        perdoc_row.get("n_docs_requested"),
        f"{tag} per-document n_docs_requested")
    if requested != summary_row["n_docs_requested"]:
        raise ValueError(f"{tag}: per-document n_docs_requested mismatch")
    records = perdoc_row.get("records")
    all_counts, mass_counts = counts_from_records(records, threshold)
    if any(record["doc_index"] >= requested for record in records):
        raise ValueError(f"{tag}: doc_index exceeds requested document population")
    if all_counts != expected_all:
        raise ValueError(f"{tag}: per-document unrestricted table mismatch")
    if mass_counts != expected_mass:
        raise ValueError(f"{tag}: per-document mass table mismatch")
    n = _nonnegative_int(perdoc_row.get("n"), f"{tag} per-document n")
    n_valid = _nonnegative_int(
        perdoc_row.get("n_valid"), f"{tag} per-document n_valid")
    if n != len(records) or n != summary_row["n"]:
        raise ValueError(f"{tag}: per-document n mismatch")
    if n_valid != sum(mass_counts.values()) or n_valid != summary_row["n_valid"]:
        raise ValueError(f"{tag}: per-document n_valid mismatch")
    d_all, mass_all = perdoc_row.get("d_all"), perdoc_row.get("mass_all")
    if not isinstance(d_all, list) or len(d_all) != n:
        raise ValueError(f"{tag}: d_all length mismatch")
    if not isinstance(mass_all, list) or len(mass_all) != n:
        raise ValueError(f"{tag}: mass_all length mismatch")
    for pos, (record, delta, mass) in enumerate(zip(records, d_all, mass_all)):
        if (isinstance(delta, bool) or not isinstance(delta, (int, float))
                or not math.isfinite(float(delta))):
            raise ValueError(f"{tag}: d_all[{pos}] is not finite")
        if (isinstance(mass, bool) or not isinstance(mass, (int, float))
                or not math.isfinite(float(mass))):
            raise ValueError(f"{tag}: mass_all[{pos}] is not finite")
        if not math.isclose(float(delta), float(record.get("delta")), abs_tol=5.1e-5):
            raise ValueError(f"{tag}: d_all[{pos}] disagrees with record")
        if not math.isclose(float(mass), float(record["mass"]), abs_tol=5.1e-7):
            raise ValueError(f"{tag}: mass_all[{pos}] disagrees with record")
    return all_counts, mass_counts


def _objects(line):
    line = line.strip()
    if not line:
        return []
    try:
        if line.startswith("raw:"):
            value = json.loads(line[4:])
            return value if isinstance(value, list) else []
        value = json.loads(line)
        return [value] if isinstance(value, dict) else []
    except json.JSONDecodeError:
        return []


def load_rows(paths, *, canonical_grid=True):
    """Load latest row per tag from snapshots or append-only JSONL caches."""
    latest = {}
    for path in paths:
        if "perdoc" in os.path.basename(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for row in _objects(line):
                    tag = row.get("tag")
                    if tag:
                        latest[tag] = row
    rows = list(latest.values())
    if canonical_grid:
        rows = [row for row in rows if GRID_RE.fullmatch(row.get("tag", ""))
                and row.get("step") == 16000
                and row.get("total_steps") == 16000]
        tags = {row["tag"] for row in rows}
        expected = set(canonical_tags())
        missing, extra = sorted(expected - tags), sorted(tags - expected)
        if missing or extra:
            raise ValueError(f"canonical grid mismatch: missing={missing}, extra={extra}")
    if not rows:
        raise ValueError("no run summaries loaded")
    return sorted(rows, key=lambda row: row["tag"])


def load_perdoc_rows(paths):
    """Load the latest per-document object per tag."""
    latest = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for row in _objects(line):
                    tag = row.get("tag")
                    if tag:
                        latest[tag] = row
    return latest


def validate_row_pairs(rows, perdoc_paths):
    """Rebuild every supplied run table from its per-document raw ids."""
    perdocs = load_perdoc_rows(perdoc_paths)
    missing = sorted(row["tag"] for row in rows if row["tag"] not in perdocs)
    if missing:
        raise ValueError(f"missing per-document rows: {missing}")
    for row in rows:
        validate_perdoc(row, perdocs[row["tag"]], DEFAULT_MASS_FLOOR)
    return perdocs


def add_counts(target, source):
    source = normalize_counts(source)
    for key in transition_keys():
        target[key] += source[key]


def _rate(counts, predicate):
    n = sum(counts.values())
    return (sum(value for key, value in counts.items() if predicate(key)) / n
            if n else float("nan"))


def count_metrics(counts):
    """Rates used in pooled and run-level reporting."""
    n = sum(counts.values())
    cell_rates = {
        key.replace("->", "_to_"): (value / n if n else float("nan"))
        for key, value in counts.items()
    }
    return {
        **cell_rates,
        "base_other": _rate(counts, lambda key: key.startswith("other->")),
        "edit_other": _rate(counts, lambda key: key.endswith("->other")),
        "any_other": _rate(counts, lambda key: "other" in key),
        "changed": _rate(counts, lambda key: key.split("->")[0]
                          != key.split("->")[1]),
    }


def aggregate(rows, *, mass_restricted=False):
    """Aggregate counts while retaining every training run as the unit."""
    pooled, per_run = empty_counts(), []
    for row in rows:
        all_counts, mass_counts = validate_row(row)
        counts = mass_counts if mass_restricted else all_counts
        n = sum(counts.values())
        add_counts(pooled, counts)
        per_run.append({"tag": row["tag"], "n": n,
                        "counts": counts, **count_metrics(counts)})
    n_total = sum(pooled.values())
    return {"schema": SCHEMA, "mass_floor": DEFAULT_MASS_FLOOR,
            "mass_restricted": mass_restricted,
            "inference_unit": "training_run",
            "pooled_role": "descriptive",
            "n_runs": len(rows), "n_runs_nonempty": sum(r["n"] > 0 for r in per_run),
            "n_documents": n_total, "counts": pooled,
            "rates": count_metrics(pooled), "per_run": per_run}


def metrics(summary):
    return summary.get("rates") or count_metrics(summary["counts"])


def _fmt_rate(value):
    return "n/a" if value != value else f"{value:.4f}"


def _max_finite_run(per_run, name):
    finite = [row for row in per_run
              if isinstance(row.get(name), (int, float))
              and math.isfinite(float(row[name]))]
    return max(finite, key=lambda row: row[name]) if finite else None


def format_text(summary, mass_summary):
    lines = [
        "Base/edit argmax transitions (old=contrast, new=preserved truth)",
        "Pooled document counts are descriptive; the training run is the replication unit.",
    ]
    panels = (("all edit-domain documents", summary),
              (f"documents with contrast-pair mass >= {DEFAULT_MASS_FLOOR}",
               mass_summary))
    key_rates = ("new_to_new", "new_to_old", "old_to_new",
                 "base_other", "edit_other", "any_other", "changed")
    for title, current in panels:
        counts = current["counts"]
        lines += ["", title,
                  f"  runs={current['n_runs']} "
                  f"(nonempty={current['n_runs_nonempty']})  "
                  f"documents={current['n_documents']}",
                  f"  {'base/edit':<12}{'old':>9}{'new':>9}"
                  f"{'other':>9}{'row total':>11}"]
        for base in LABELS:
            values = [counts[f"{base}->{edit}"] for edit in LABELS]
            lines.append(f"  {base:<12}" + "".join(f"{v:>9}" for v in values)
                         + f"{sum(values):>11}")
        met = metrics(current)
        lines.append("  pooled rates (descriptive): " + ", ".join(
            f"{name}={_fmt_rate(met[name])}" for name in key_rates))
        for name in key_rates:
            worst = _max_finite_run(current["per_run"], name)
            if worst is not None:
                lines.append(f"  max run {name}: {worst[name]:.4f} "
                             f"({worst['tag']}, n={worst['n']})")
    return "\n".join(lines)


def json_ready(value):
    """Replace non-finite floats with null for standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def _latex_cell(value, total):
    pct = 100.0 * value / total if total else float("nan")
    return f"{value} ({pct:.2f}\\%)" if pct == pct else str(value)


def format_latex(summary, mass_summary):
    row_end = r"\\"
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Base-to-edited argmax transitions. \emph{old} is the superseded",
        r"contrast value, \emph{new} is the preserved ground-truth/current value, and",
        r"\emph{other} is any third value. Cells give pooled document counts (percent",
        r"within panel); pooling is descriptive because the training run is the",
        r"replication unit.}", r"\label{tab:argmax-transition}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"subset & base $\backslash$ edited & old & new & other " + row_end,
        r"\midrule",
    ]
    for panel, current in (("all", summary), ("mass", mass_summary)):
        total = current["n_documents"]
        for i, base in enumerate(LABELS):
            prefix = panel if i == 0 else ""
            cells = [_latex_cell(current["counts"][f"{base}->{edit}"], total)
                     for edit in LABELS]
            lines.append(f"{prefix} & {base} & " + " & ".join(cells)
                         + " " + row_end)
        if panel == "all":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Validate and summarize old/new/other argmax transitions")
    parser.add_argument("inputs", nargs="+", help="go_nogo summary .txt/.jsonl files")
    parser.add_argument("--perdoc", nargs="+", default=None,
                        help="per-document JSONL; rebuild and cross-check every run")
    parser.add_argument("--all-runs", action="store_true",
                        help="do not require the canonical 75-run grid")
    parser.add_argument("--txt", default=None, help="write the text report")
    parser.add_argument("--json", default=None, help="write machine-readable summaries")
    parser.add_argument("--tex", default=None, help="write a LaTeX table")
    args = parser.parse_args()

    rows = load_rows(args.inputs, canonical_grid=not args.all_runs)
    if args.perdoc:
        validate_row_pairs(rows, args.perdoc)
    unrestricted = aggregate(rows)
    restricted = aggregate(rows, mass_restricted=True)
    text = format_text(unrestricted, restricted)
    print(text)

    payload = json_ready({"unrestricted": unrestricted,
                          "mass_restricted": restricted})
    outputs = ((args.txt, text),
               (args.tex, format_latex(unrestricted, restricted)),
               (args.json, json.dumps(payload, indent=2, sort_keys=True,
                                      allow_nan=False)))
    for path, content in outputs:
        if not path:
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

