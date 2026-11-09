"""Stage 2 parameter study: effect of the Harris parameters on the number of
candidates and on the final corner error.

Usage:
    python tools/param_study.py --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudoku_scan import BLOCK_SIZE, K, KSIZE, THRESHOLD, detect_corners
from tools.evaluate import corner_error

DEFAULTS = {"block_size": BLOCK_SIZE, "ksize": KSIZE, "k": K, "threshold": THRESHOLD}
VALUES = {
    "block_size": [2, 3, 5, 7, 9, 13],
    "ksize": [3, 5, 7],
    "k": [0.02, 0.04, 0.06, 0.10, 0.15, 0.20],
    "threshold": [0.01, 0.05, 0.1, 0.2, 0.4, 0.6],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    data = []
    for file in sorted(Path(args.annotations).glob("*.json")):
        truth = json.loads(file.read_text())
        data.append((cv2.imread(str(Path(args.images) / truth["image"])), truth["corners"]))
    if not data:
        sys.exit(f"no annotations found in {args.annotations}")
    print(f"{len(data)} images")

    figure, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for axis, (name, values) in zip(axes, VALUES.items()):
        counts, errors = [], []
        print(f"\n{name:<12}{'candidates':>12}{'corner %':>10}{'failures':>10}")
        for value in values:
            n, e = [], []
            for image, truth in data:
                corners, _, candidates, _ = detect_corners(image, **{**DEFAULTS, name: value})
                n.append(len(candidates))
                e.append(corner_error(corners, truth) if corners is not None else np.nan)
            failures = int(np.sum(np.isnan(e) | (np.array(e) > 5)))
            counts.append(np.mean(n))
            errors.append(np.nanmean(e))
            print(f"{value:<12}{counts[-1]:>12.0f}{errors[-1]:>10.2f}{failures:>10}")
        axis.plot(values, counts, "o-", color="tab:blue")
        axis.set_xlabel(name)
        axis.set_ylabel("mean candidates", color="tab:blue")
        twin = axis.twinx()
        twin.plot(values, errors, "s--", color="tab:red")
        twin.set_ylabel("mean corner error (%)", color="tab:red")
        twin.set_yscale("log")
        twin.set_ylim(0.1, 100)
        axis.axvline(DEFAULTS[name], color="grey", linewidth=0.8)
    figure.tight_layout()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    figure.savefig(Path(args.outdir) / "param_study.png", dpi=150)


if __name__ == "__main__":
    main()
