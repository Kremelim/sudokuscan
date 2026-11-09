"""Stage 4: affine (A3, A4) versus homography (H4) on annotated images.

For every image it measures the obliqueness, the corner residuals, the grid
line deviation and the OCR cell accuracy of the three rectifications.

Usage:
    python tools/compare_models.py --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudoku_scan import CELL, SIZE, binarise, detect_corners, odd, read_grid, rectify

MODELS = ("A3", "A4", "H4")


def obliqueness(corners):
    """Largest length ratio between opposite sides of the quadrilateral.
    It is 1 for any parallelogram, so it measures only the perspective part."""
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    top, right, bottom, left = sides
    return float(max(top / bottom, bottom / top, left / right, right / left))


def line_deviation(rect):
    """Mean distance (px) of the 20 grid lines from their ideal positions,
    located as the strongest peak of the projection profile near each one."""
    binary = binarise(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), odd(CELL / 2))
    deviations = []
    for axis in (0, 1):
        profile = binary.sum(axis=axis).astype(float)
        for ideal in range(0, SIZE + 1, CELL):
            low, high = max(0, ideal - CELL // 2), min(SIZE, ideal + CELL // 2)
            deviations.append(abs(low + int(np.argmax(profile[low:high])) - ideal))
    return float(np.mean(deviations))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, most_oblique = [], None
    for file in sorted(Path(args.annotations).glob("*.json")):
        truth = json.loads(file.read_text())
        image = cv2.imread(str(Path(args.images) / truth["image"]))
        corners = detect_corners(image)[0]
        if corners is None:
            continue
        warped = rectify(image, corners)
        mapped = cv2.perspectiveTransform(np.float32([corners]), warped["H4"])[0]
        target = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
        row = {"image": truth["image"], "obliqueness": obliqueness(corners),
               "a3_corner_err_px": warped["a3_corner_err_px"], "a4_rms_px": warped["a4_rms_px"],
               "h4_max_err_px": float(np.abs(mapped - target).max())}
        for model in MODELS:
            rect = warped["rect_" + model]
            row["line_dev_" + model] = line_deviation(rect)
            row["cell_acc_" + model] = float(np.mean(read_grid(rect) == np.array(truth["grid"])))
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
        if most_oblique is None or row["obliqueness"] > most_oblique[0]:
            most_oblique = (row["obliqueness"], truth["image"], warped)

    rows.sort(key=lambda r: r["obliqueness"])
    with open(outdir / "model_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = [r["obliqueness"] for r in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x, [r["a3_corner_err_px"] for r in rows], "o-", label="A3 fourth-corner error")
    axes[0].plot(x, [r["a4_rms_px"] for r in rows], "s-", label="A4 RMS residual")
    axes[0].set_ylabel("residual (px)")
    for model, marker in zip(MODELS, "os^"):
        axes[1].plot(x, [r["line_dev_" + model] for r in rows], marker + "-", label=model)
        axes[2].plot(x, [r["cell_acc_" + model] for r in rows], marker + "-", label=model)
    axes[1].set_ylabel("mean grid line deviation (px)")
    axes[2].set_ylabel("cell accuracy")
    for axis in axes:
        axis.set_xlabel("obliqueness (opposite side ratio)")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(outdir / "model_comparison.png", dpi=150)

    _, name, warped = most_oblique
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for axis, model in zip(axes, MODELS):
        axis.imshow(cv2.cvtColor(warped["rect_" + model], cv2.COLOR_BGR2RGB))
        axis.set_title(model)
        axis.axis("off")
    figure.suptitle(f"Most oblique image: {name}")
    figure.tight_layout()
    figure.savefig(outdir / "most_oblique.png", dpi=150)


if __name__ == "__main__":
    main()
