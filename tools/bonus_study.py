"""Bonus study: solver (B1), many-point homography (B3) and orientation (B4)
on annotated images.

Every photograph goes through the whole pipeline four times, turned by 0, 90,
180 and 270 degrees. The upright run gives the solver result and the H4 / HN
comparison (grid line deviation and OCR cell accuracy). The four runs together
show whether the grid is read correctly in any orientation.

Usage:
    python tools/bonus_study.py --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudoku_scan import process, read_grid, solve
from tools.compare_models import line_deviation

ANGLES = (0, 90, 180, 270)      # counter-clockwise turn given to the photograph


def study(image, truth, workdir):
    """One row of bonus measurements for one annotated photograph."""
    true_grid = np.array(truth["grid"])
    true_solutions = solve(true_grid)[0]
    row, turns = {"image": truth["image"]}, {}
    for angle in ANGLES:
        turned = Path(workdir) / f"turned_{angle}.png"
        cv2.imwrite(str(turned), np.rot90(image, angle // 90))
        result = process(turned, workdir)
        turns[f"cell_acc_{angle}"] = float(np.mean(np.array(result["grid"]) == true_grid))
        turns[f"rotation_{angle}"] = result.get("rotation_deg", "")
        if angle == 0 and "HN" in result:
            row.update(solution_status=result["solution_status"], solver_ms=result["solver_ms"],
                       solution_correct=result["solution"] in true_solutions,
                       hn_points=result["hn_points"], hn_inliers=result["hn_inliers"])
            for model in ("H4", "HN"):
                rect = cv2.imread(str(Path(workdir) / f"turned_0_rect_{model}.png"))
                row["line_dev_" + model] = line_deviation(rect)
                row["grid_acc_" + model] = float(np.mean(read_grid(rect) == true_grid))
    return {**row, **turns}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for file in sorted(Path(args.annotations).glob("*.json")):
        truth = json.loads(file.read_text())
        image = cv2.imread(str(Path(args.images) / truth["image"]))
        with tempfile.TemporaryDirectory() as workdir:
            rows.append(study(image, truth, workdir))
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()})
    if not rows:
        sys.exit(f"no annotated images found in {args.annotations}")

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(outdir / "bonus_study.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    found = [r for r in rows if "solution_status" in r]       # puzzle found when upright
    if not found:
        sys.exit("no puzzle was found")
    unique = [r for r in found if r["solution_status"] == "unique"]
    times = [r["solver_ms"] for r in found]
    print(f"\nB1 solver: unique solution for {len(unique)} of {len(found)} read grids, "
          f"{sum(r['solution_correct'] for r in unique)} of them correct, "
          f"mean {np.mean(times):.1f} ms, max {np.max(times):.1f} ms")
    print(f"\nB3 {'model':<7}{'line deviation (px)':>21}{'cell accuracy':>15}")
    for model in ("H4", "HN"):
        print(f"   {model:<7}{np.mean([r['line_dev_' + model] for r in found]):>21.2f}"
              f"{np.mean([r['grid_acc_' + model] for r in found]):>15.3f}")
    print(f"   HN uses {np.mean([r['hn_points'] for r in found]):.0f} point pairs on average, "
          f"{np.mean([r['hn_inliers'] for r in found]):.0f} of them RANSAC inliers")
    print(f"\nB4 {'photo turned by':<17}{'orientation found':>19}{'cell accuracy':>15}")
    for angle in ANGLES:
        right = sum(r[f"rotation_{angle}"] == (r["rotation_0"] - angle) % 360 for r in found)
        print(f"   {angle:<17}{right:>14} / {len(found):<2}"
              f"{np.mean([r[f'cell_acc_{angle}'] for r in rows]):>15.3f}")

    found.sort(key=lambda r: r["line_dev_H4"])
    x = range(len(found))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x, [r["line_dev_H4"] for r in found], "o-", label="H4 (4 corners)")
    axis.plot(x, [r["line_dev_HN"] for r in found], "s-", label="HN (grid intersections, RANSAC)")
    axis.set_xlabel("image (sorted by H4 deviation)")
    axis.set_ylabel("mean grid line deviation (px)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(outdir / "bonus_homography.png", dpi=150)


if __name__ == "__main__":
    main()
