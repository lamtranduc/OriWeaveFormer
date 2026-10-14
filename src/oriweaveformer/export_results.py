"""Generate paper result tables from the frozen final-experiment JSON records."""
import argparse
import csv
import json
from pathlib import Path


FIELDS = ("precision", "recall", "f1", "iou")


def _load_rows(experiment):
    rows = []
    for dataset_dir in sorted(Path(experiment).iterdir()):
        metric_path = dataset_dir / "metrics.json"
        if (not dataset_dir.is_dir() or dataset_dir.name == "summary"
                or not metric_path.is_file()):
            continue
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        metric = payload.get("reported_test_metrics", payload.get("test_metrics", {}).get("val_tuned", {}))
        if not all(key in metric for key in FIELDS):
            raise ValueError(f"Missing final metrics in {metric_path}")
        rows.append({"dataset": payload.get("dataset", dataset_dir.name),
                     "checkpoint_epoch": payload.get("checkpoint_epoch"),
                     "threshold": payload.get("selected_threshold", metric.get("threshold")),
                     **{key: float(metric[key]) for key in FIELDS}})
    if not rows:
        raise ValueError("No dataset metrics.json files found")
    return rows


def export(experiment, paper_dir=None):
    experiment = Path(experiment)
    rows = _load_rows(experiment)
    summary = experiment / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    csv_path = summary / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "checkpoint_epoch", "threshold", *FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    tex_path = None
    if paper_dir is not None:
        paper_dir = Path(paper_dir)
        table_dir = paper_dir / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        tex_path = table_dir / "results.tex"
        lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
                 r"Dataset & Precision & Recall & F1 & IoU \\", r"\midrule"]
        for row in rows:
            label = str(row["dataset"]).replace("_", r"\_")
            values = " & ".join(f"{row[key]:.4f}" for key in FIELDS)
            lines.append(f"{label} & {values} " + r"\\")
        lines.extend((r"\bottomrule", r"\end{tabular}"))
        tex_path.open("w", encoding="utf-8").write("\n".join(lines) + "\n")
    return csv_path, tex_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiments/oriweaveformer_final")
    parser.add_argument("--paper-dir", default=None, help="Optional path to paper directory for LaTeX table export")
    args = parser.parse_args()
    csv_path, tex_path = export(args.experiment, args.paper_dir)
    if tex_path:
        print(f"wrote {csv_path} and {tex_path}")
    else:
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
