"""Stage 5 study: effect of cell preparation and of the Tesseract page
segmentation mode on cell accuracy, plus the most frequent confusions.

Usage:
    python tools/ocr_study.py --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudoku_scan import CELL, WHITELIST, detect_corners, read_grid, rectify


def single(options, pad=2.5):
    """One OCR attempt with the given Tesseract options."""
    return [(options, pad)]


VARIANTS = {
    "prepared, default attempts": dict(),
    "prepared, psm 10": dict(attempts=single("--psm 10" + WHITELIST)),
    "raw cells, psm 10": dict(attempts=single("--psm 10" + WHITELIST), prepare=False),
    "prepared, psm 6": dict(attempts=single("--psm 6" + WHITELIST)),
    "prepared, psm 7": dict(attempts=single("--psm 7" + WHITELIST)),
    "prepared, psm 8": dict(attempts=single("--psm 8" + WHITELIST)),
    "prepared, psm 13": dict(attempts=single("--psm 13" + WHITELIST)),
    "prepared, psm 10, no whitelist": dict(attempts=single("--psm 10")),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    crops = Path(args.outdir) / "confusions"
    crops.mkdir(parents=True, exist_ok=True)

    accuracy = {name: [] for name in VARIANTS}
    confusions = Counter()
    for file in sorted(Path(args.annotations).glob("*.json")):
        truth = json.loads(file.read_text())
        image = cv2.imread(str(Path(args.images) / truth["image"]))
        corners = detect_corners(image)[0]
        if corners is None:
            continue
        rect = rectify(image, corners)["rect_H4"]
        true_grid = np.array(truth["grid"])
        for name, options in VARIANTS.items():
            grid = read_grid(rect, **options)
            accuracy[name].append(np.mean(grid == true_grid))
            if name == "prepared, default attempts":
                for r, c in zip(*np.nonzero(grid != true_grid)):
                    confusions[(true_grid[r, c], grid[r, c])] += 1
                    crop = rect[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
                    cv2.imwrite(str(crops / f"true{true_grid[r, c]}_read{grid[r, c]}_{file.stem}_r{r}c{c}.png"), crop)
        print(file.stem, "done")

    if not confusions and not any(accuracy.values()):
        sys.exit(f"no annotated images found in {args.annotations}")
    print(f"\n{'variant':<34}{'cell accuracy':>14}")
    for name, values in accuracy.items():
        print(f"{name:<34}{np.mean(values):>14.3f}")
    print("\nmost frequent confusions (true -> read, 0 = empty):")
    for (true, read), count in confusions.most_common(10):
        print(f"  {true} -> {read}: {count}")


if __name__ == "__main__":
    main()
