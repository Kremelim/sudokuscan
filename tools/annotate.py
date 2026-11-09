"""Create the ground-truth annotation (Appendix A format) for one photograph.

The grid is read from a text file with 9 lines of 9 digits (0 = empty), typed
once per printed puzzle. The corners are clicked by hand: first roughly on the
whole photo, then precisely on a zoomed view, in the order TL, TR, BR, BL.

Usage:
    python tools/annotate.py --image data/personal/images/s123_01.jpg --grid data/personal/puzzles/puzzle1.txt --conditions C2 C4
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

ZOOM = 60     # half-size of the zoomed view in pixels


def click_corner(image, label):
    plt.clf()
    plt.imshow(image)
    plt.title(f"Click near the {label} corner")
    x, y = plt.ginput(1, timeout=0)[0]
    plt.xlim(x - ZOOM, x + ZOOM)
    plt.ylim(y + ZOOM, y - ZOOM)
    plt.title(f"Click exactly on the outer {label} corner")
    x, y = plt.ginput(1, timeout=0)[0]
    return [round(x, 1), round(y, 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--grid", required=True, help="text file with 9 lines of 9 digits")
    parser.add_argument("--conditions", nargs="+", required=True, help="e.g. C2 C4")
    args = parser.parse_args()

    path = Path(args.image)
    grid = [[int(d) for d in line.strip()] for line in Path(args.grid).read_text().split()]
    assert len(grid) == 9 and all(len(row) == 9 for row in grid), "grid must be 9x9"

    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    corners = [click_corner(image, label) for label in ("top-left", "top-right", "bottom-right", "bottom-left")]
    plt.close()

    outdir = path.parent.parent / "annotations"
    outdir.mkdir(exist_ok=True)
    annotation = {"image": path.name, "conditions": args.conditions, "corners": corners, "grid": grid}
    (outdir / (path.stem + ".json")).write_text(json.dumps(annotation, indent=2))
    print("saved", outdir / (path.stem + ".json"))


if __name__ == "__main__":
    main()
