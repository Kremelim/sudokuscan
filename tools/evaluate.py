"""Section 7 metrics for a results directory against ground-truth annotations.

Usage:
    python tools/evaluate.py --results results/ --annotations data/devset/annotations/
"""
import argparse
import json
from pathlib import Path

import numpy as np


def corner_error(predicted, truth):
    """Mean corner distance as a percentage of the true puzzle side length."""
    predicted, truth = np.array(predicted), np.array(truth)
    side = np.mean(np.linalg.norm(truth - np.roll(truth, -1, axis=0), axis=1))
    return 100 * np.mean(np.linalg.norm(predicted - truth, axis=1)) / side


def evaluate(results_dir, annotations_dir):
    """One row of metrics per annotated image."""
    rows = []
    for file in sorted(Path(annotations_dir).glob("*.json")):
        truth = json.loads(file.read_text())
        result_file = Path(results_dir) / (file.stem + "_result.json")
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        true_grid = np.array(truth["grid"])
        grid = np.array(result.get("grid", np.zeros((9, 9), int)))
        rows.append({
            "name": file.stem,
            "conditions": truth.get("conditions", []),
            "corner_err": corner_error(result["corners"], truth["corners"]) if "corners" in result else np.nan,
            "cell_acc": np.mean(grid == true_grid),
            "exact": bool(np.all(grid == true_grid)),
            "tp": np.sum((grid == 0) & (true_grid == 0)),
            "fp": np.sum((grid == 0) & (true_grid != 0)),
            "fn": np.sum((grid != 0) & (true_grid == 0)),
            "robust": bool(result) and result.get("runtime_s", 99) <= 20,
        })
    return rows


def summary(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return {
        "images": len(rows),
        "corner_err_%": np.nanmean([r["corner_err"] for r in rows]),
        "cell_acc": np.mean([r["cell_acc"] for r in rows]),
        "empty_f1": 2 * tp / (2 * tp + fp + fn),
        "exact_match": np.mean([r["exact"] for r in rows]),
        "robustness": np.mean([r["robust"] for r in rows]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--annotations", required=True)
    args = parser.parse_args()

    rows = evaluate(args.results, args.annotations)
    print(f"{'image':<16}{'corner %':>9}{'cells':>8}  conditions")
    for r in rows:
        print(f"{r['name']:<16}{r['corner_err']:>9.2f}{r['cell_acc']:>8.3f}  {' '.join(r['conditions'])}")

    groups = {"all": rows}
    for condition in ("C1", "C2", "C3", "C4", "C5"):
        groups[condition] = [r for r in rows if condition in r["conditions"]]
    print(f"\n{'group':<6}{'n':>4}{'corner %':>10}{'cell acc':>10}{'empty F1':>10}{'exact':>8}{'robust':>8}")
    for name, group in groups.items():
        if group:
            s = summary(group)
            print(f"{name:<6}{s['images']:>4}{s['corner_err_%']:>10.2f}{s['cell_acc']:>10.3f}"
                  f"{s['empty_f1']:>10.3f}{s['exact_match']:>8.2f}{s['robustness']:>8.2f}")


if __name__ == "__main__":
    main()
