"""SudokuScan: photograph of a printed sudoku -> 9x9 digital grid.

Usage:
    python sudoku_scan.py --input photo.jpg --outdir out/
    python sudoku_scan.py --batch images_dir/ --outdir results/
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract

SIZE = 450            # side of the rectified image in pixels
CELL = SIZE // 9      # side of one cell
WORK = 1000           # longest image side used for corner detection
INK = 0.85            # ink is darker than this fraction of the local paper brightness

# Harris parameters (chosen in the parameter study, see the report)
BLOCK_SIZE = 5
KSIZE = 3
K = 0.04
THRESHOLD = 0.05      # fraction of the maximum response

MARGIN = round(0.14 * CELL)   # inner margin of a cell, where the grid lines are
MIN_HEIGHT = 0.3              # a digit is at least this fraction of the cell height
OCR_SIZE = 40                 # side of the picture given to Tesseract
WHITELIST = " -c tessedit_char_whitelist=123456789"
# (Tesseract options, padding) tried in this order until a digit is returned
OCR_ATTEMPTS = [("--psm 10" + WHITELIST, pad) for pad in (2.5, 2.0, 3.0, 1.6)]
OCR_ATTEMPTS.append(("--psm 13" + WHITELIST, 2.5))


def odd(x):
    """Nearest odd integer >= 3, used for kernel sizes."""
    return max(3, int(x) | 1)


# ---------------------------------------------------------------- Stage 1
def binarise(grey, size):
    """Ink mask (ink = 255) that does not depend on the lighting.

    After a light blur against noise, the local paper brightness is estimated
    with a morphological closing, which removes dark structures thinner than
    size. A pixel is ink when it is darker than INK times the paper around it."""
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    paper = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    return np.uint8(grey < INK * paper) * 255


def preprocess(image):
    """Return (grey, binary, scale) at the working resolution."""
    scale = min(1.0, WORK / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return grey, binarise(grey, odd(max(grey.shape) * 0.02)), scale


# ---------------------------------------------------------------- Stage 2
def find_region(binary):
    """Mask of the puzzle: the connected component whose number of
    cell-sized holes is closest to 81 and that is not cut by the image border."""
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    areas = [cv2.contourArea(c) for c in contours]
    holes = np.zeros(len(contours), int)
    for i, (_, _, _, parent) in enumerate(hierarchy[0]):
        if parent >= 0 and areas[parent] / 400 < areas[i] < areas[parent] / 5:
            holes[parent] += 1
    height, width = binary.shape
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x + w == width or y + h == height:
            holes[i] = 0
    best = int(np.argmin(np.abs(holes - 81)))
    if holes[best] < 4:
        return None
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, best, 255, -1)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def harris_candidates(binary, block_size=BLOCK_SIZE, ksize=KSIZE, k=K, threshold=THRESHOLD):
    """Harris response map and its local maxima above the threshold."""
    response = cv2.cornerHarris(np.float32(binary), block_size, ksize, k)
    window = odd(max(binary.shape) * 0.005)
    peaks = response == cv2.dilate(response, np.ones((window, window), np.uint8))
    ys, xs = np.nonzero(peaks & (response > threshold * response.max()))
    return response, np.float32(np.column_stack([xs, ys]))


def select_corners(candidates, mask):
    """Largest-area quadrilateral whose vertices are candidates inside the
    puzzle region, ordered TL, TR, BR, BL."""
    inside = mask[candidates[:, 1].astype(int), candidates[:, 0].astype(int)] > 0
    if inside.sum() < 4:
        return None
    hull = cv2.convexHull(candidates[inside]).reshape(-1, 2)
    if len(hull) < 4:
        return None
    quad = max(itertools.combinations(hull, 4), key=lambda q: cv2.contourArea(np.array(q)))
    quad = np.array(quad)
    centre = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    return quad[np.argsort(angles)]


def detect_corners(image, **harris):
    """Stages 1-2. Returns corners in original-image pixels (or None),
    the response map, all candidates and the working scale."""
    grey, binary, scale = preprocess(image)
    response, candidates = harris_candidates(binary, **harris)
    mask = find_region(binary)
    corners = None
    if mask is not None and len(candidates) >= 4:
        corners = select_corners(candidates, mask)
    if corners is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(grey, corners.copy(), (5, 5), (-1, -1), criteria) / scale
    return corners, response, candidates, scale


# ---------------------------------------------------------------- Stage 3
def rectify(image, corners):
    """Warp the puzzle to SIZE x SIZE with the A3, A4 and H4 models."""
    src = np.float32(corners)
    dst = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
    A3 = cv2.getAffineTransform(src[[0, 1, 3]], dst[[0, 1, 3]])
    A4 = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=SIZE)[0]
    H4 = cv2.getPerspectiveTransform(src, dst)

    homogeneous = np.column_stack([src, np.ones(4)])
    a3_error = float(np.linalg.norm(A3 @ homogeneous[2] - dst[2]))
    a4_rms = float(np.sqrt(np.mean(np.sum((homogeneous @ A4.T - dst) ** 2, axis=1))))

    return {
        "A3": A3, "A4": A4, "H4": H4,
        "a3_corner_err_px": a3_error, "a4_rms_px": a4_rms,
        "rect_A3": cv2.warpAffine(image, A3, (SIZE, SIZE)),
        "rect_A4": cv2.warpAffine(image, A4, (SIZE, SIZE)),
        "rect_H4": cv2.warpPerspective(image, H4, (SIZE, SIZE)),
    }


# ---------------------------------------------------------------- Stage 5
def find_digits(binary):
    """Empty-cell criterion. Returns {(row, column): digit mask} for filled cells.

    A cell is filled when a connected component that is smaller than a cell
    and at least MIN_HEIGHT of its height has its centre in the inner region
    of the cell (the cell without MARGIN on each side). Grid lines are larger
    than a cell or centred in the margin, so they never reach the OCR."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    digits = {}
    for i in np.argsort(stats[:, cv2.CC_STAT_AREA]):     # largest last, so it wins
        x, y, w, h, _ = stats[i]
        cx, cy = x + w / 2, y + h / 2
        in_inner = (MARGIN <= cx % CELL <= CELL - MARGIN and
                    MARGIN <= cy % CELL <= CELL - MARGIN)
        if in_inner and MIN_HEIGHT * CELL <= h < CELL and w < CELL:
            mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
            digits[int(cy // CELL), int(cx // CELL)] = mask
    return digits


def prepare_digit(digit, pad):
    """Black digit centred on a white square of side pad x digit height,
    with strokes thinned by one pixel, resized to OCR_SIZE."""
    h, w = digit.shape
    side = max(int(h * pad), w + 2)
    canvas = np.full((side, side), 255, np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y + h, x:x + w] = 255 - digit
    canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8))
    return cv2.resize(canvas, (OCR_SIZE, OCR_SIZE), interpolation=cv2.INTER_AREA)


def ocr(picture, config):
    """First digit 1-9 that Tesseract finds in the picture, or 0."""
    text = pytesseract.image_to_string(picture, config=config)
    digits = [int(ch) for ch in text if ch in "123456789"]
    return digits[0] if digits else 0


def read_grid(rect, prepare=True, attempts=OCR_ATTEMPTS):
    """Stage 5: 9x9 grid of givens from a rectified image, 0 for empty.

    Empty cells are decided by find_digits and never sent to the OCR.
    Tesseract sometimes returns nothing for a clear digit, so a filled cell
    is tried with each entry of attempts until a digit is read."""
    grey = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    binary = binarise(grey, odd(CELL / 2))
    grid = np.zeros((9, 9), int)
    for (r, c), digit in find_digits(binary).items():
        raw = grey[r * CELL + MARGIN:(r + 1) * CELL - MARGIN,
                   c * CELL + MARGIN:(c + 1) * CELL - MARGIN]
        for config, pad in attempts:
            grid[r, c] = ocr(prepare_digit(digit, pad) if prepare else raw, config)
            if grid[r, c]:
                break
    return grid


# ---------------------------------------------------------------- Stage 6
def find_conflicts(grid):
    """Cells that repeat a digit in their row, column or 3x3 box."""
    conflicts = set()
    cells = [(r, c) for r in range(9) for c in range(9) if grid[r, c]]
    for (r1, c1), (r2, c2) in itertools.combinations(cells, 2):
        same_unit = r1 == r2 or c1 == c2 or (r1 // 3, c1 // 3) == (r2 // 3, c2 // 3)
        if same_unit and grid[r1, c1] == grid[r2, c2]:
            conflicts.update([(r1, c1), (r2, c2)])
    return sorted(conflicts)


# ---------------------------------------------------------------- Output
def draw_harris(response, candidates, corners, scale):
    picture = cv2.normalize(np.maximum(response, 0) ** 0.25, None, 0, 255, cv2.NORM_MINMAX)
    picture = cv2.cvtColor(picture.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y in candidates:
        cv2.circle(picture, (int(x), int(y)), 3, (0, 255, 255), 1)
    if corners is not None:
        for x, y in corners * scale:
            cv2.circle(picture, (int(x), int(y)), 9, (0, 0, 255), 2)
    return picture


def draw_debug(rect, grid, conflicts):
    picture = rect.copy()
    for i in range(10):
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(picture, (i * CELL, 0), (i * CELL, SIZE), (0, 160, 0), thickness)
        cv2.line(picture, (0, i * CELL), (SIZE, i * CELL), (0, 160, 0), thickness)
    for r, c in conflicts:
        cv2.rectangle(picture, (c * CELL + 2, r * CELL + 2),
                      ((c + 1) * CELL - 2, (r + 1) * CELL - 2), (0, 0, 255), 2)
    for r, c in zip(*np.nonzero(grid)):
        cv2.putText(picture, str(grid[r, c]), (c * CELL + 3, r * CELL + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    return picture


def process(path, outdir):
    """Run the whole pipeline on one image and write all output files."""
    start = time.time()
    path, outdir = Path(path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(outdir / path.stem)
    result = {"image": path.name, "verdict": "not_found", "grid": [[0] * 9] * 9}
    try:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("cannot read image")
        corners, response, candidates, scale = detect_corners(image)
        cv2.imwrite(prefix + "_harris.png", draw_harris(response, candidates, corners, scale))
        if corners is not None:
            warped = rectify(image, corners)
            grid = read_grid(warped["rect_H4"])
            conflicts = find_conflicts(grid)
            for model in ("A3", "A4", "H4"):
                cv2.imwrite(f"{prefix}_rect_{model}.png", warped["rect_" + model])
                result[model] = warped[model].tolist()
            cv2.imwrite(prefix + "_debug.png", draw_debug(warped["rect_H4"], grid, conflicts))
            result.update(corners=corners.tolist(), grid=grid.tolist(),
                          a3_corner_err_px=warped["a3_corner_err_px"],
                          a4_rms_px=warped["a4_rms_px"],
                          verdict="invalid" if conflicts else "valid")
    except Exception as error:
        print(f"{path.name}: {error}")
    result["runtime_s"] = round(time.time() - start, 3)
    with open(prefix + "_grid.txt", "w") as f:
        f.write("\n".join("".join(map(str, row)) for row in result["grid"]) + "\n")
    with open(prefix + "_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Read a sudoku puzzle from a photograph.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="one image")
    source.add_argument("--batch", help="directory of images")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    if args.input:
        paths = [Path(args.input)]
    else:
        paths = sorted(p for p in Path(args.batch).iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for path in paths:
        result = process(path, args.outdir)
        print(f"{path.name}: {result['verdict']} ({result['runtime_s']} s)")


if __name__ == "__main__":
    main()
