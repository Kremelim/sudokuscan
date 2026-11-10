# SudokuScan

COMP 482 Computer Vision, Project 1. Takes a phone photograph of a printed
sudoku puzzle and returns the puzzle as a 9x9 digital grid:

1. find the four outer corners of the grid with the Harris corner detector,
2. rectify the puzzle with two affine models (A3, A4) and a homography (H4),
3. split the rectified image into cells, decide which are empty and read the
   printed digits with Tesseract,
4. check the result against the sudoku rules.

The four bonus parts are implemented as well: the puzzle is solved (B1), the
solution is drawn on the photograph (B2), a second homography is estimated
from the inner grid intersections with RANSAC (B3), and the puzzle is read in
any of the four orientations (B4). See [Bonus parts](#bonus-parts).

## Installation

Python 3.10+ and the Tesseract OCR engine are required.

```
sudo apt-get install -y tesseract-ocr        # macOS: brew install tesseract
pip install -r requirements.txt
```

On Windows install Tesseract from <https://github.com/UB-Mannheim/tesseract/wiki>
and add its folder to `PATH`. `tesseract --version` must work in the terminal.

## Usage

```
python sudoku_scan.py --input photo.jpg --outdir out/
python sudoku_scan.py --batch images_dir/ --outdir results/
```

For every input image `<name>` the output directory receives:

| File | Content |
|---|---|
| `<name>_grid.txt` | 9 lines of 9 digits, 0 for an empty cell (from H4) |
| `<name>_result.json` | corners (TL, TR, BR, BL), A3, A4, H4, A3 fourth-corner error, A4 RMS residual, grid, verdict, runtime; bonus fields: `solution_status`, `solution`, `solver_ms` (B1), `HN`, `hn_points`, `hn_inliers` (B3), `rotation_deg`, `orientation_confidence` (B4) |
| `<name>_harris.png` | Harris response map with all candidates (yellow) and the four selected corners (red) |
| `<name>_rect_A3.png`, `_rect_A4.png`, `_rect_H4.png` | the three 450x450 rectifications |
| `<name>_debug.png` | H4 rectification with cell boundaries, recognised digits and conflicting cells (red) |
| `<name>_solution.txt` | B1: the solved grid, 9 lines of 9 digits (only when the solution is unique) |
| `<name>_ar.jpg` | B2: the photograph with the missing digits drawn into the empty cells (only when the solution is unique) |
| `<name>_rect_HN.png` | B3: 450x450 rectification with the many-point homography HN |

The verdict is `valid`, `invalid` or `not_found`. The program runs on CPU and
takes 3 to 4 seconds per image.

## Repository structure

```
sudoku_scan.py            the whole pipeline, one function per stage
requirements.txt
tools/
    evaluate.py           Section 7 metrics, overall and per condition C1-C5
    param_study.py        Harris parameter study (Stage 2)
    compare_models.py     affine versus homography comparison (Stage 4)
    ocr_study.py          cell preparation, segmentation modes, confusions (Stage 5)
    bonus_study.py        solver, HN versus H4 and the four orientations (bonus B1, B3, B4)
    annotate.py           ground-truth annotation of a photograph
data/
    devset/               provided development set (images, annotations)
    personal/             own photographs, annotations and puzzle grids
notebooks/
    SudokuScan_Colab.ipynb    runs everything on Google Colab
report/
    report.tex            report source (IEEE two-column)
    figures/              figures produced by the tools
```

## Pipeline

```
photo
  -> preprocess         downscale to 1000 px, grey, blur, binarise      (Stage 1)
  -> harris_candidates  Harris response, threshold, local maxima        (Stage 2)
  -> find_region        puzzle = component with about 81 holes          (Stage 2)
  -> select_corners     largest quadrilateral, TL TR BR BL, sub-pixel   (Stage 2)
  -> rectify            A3, A4, H4 to 450x450, corner residuals         (Stage 3)
  -> find_orientation   best of 0, 90, 180, 270 degrees, corners rolled (Bonus B4)
  -> read_grid (H4)     find_digits -> prepare_digit -> ocr             (Stage 5)
  -> find_conflicts     valid / invalid                                 (Stage 6)
  -> multipoint_homography   HN from grid intersections, RANSAC         (Bonus B3)
  -> solve              backtracking, unique / not_unique / none        (Bonus B1)
  -> draw_solution      solution warped onto the photograph             (Bonus B2)
  -> output files
```

Stage 4 (affine versus homography) is the analysis in
`tools/compare_models.py`; it uses the same functions.

## Method

| Stage | Function | What it does |
|---|---|---|
| 1 | `preprocess`, `binarise` | Downscale to 1000 px, grey, Gaussian blur. A pixel is ink when it is darker than 85 % of the local paper brightness, which is estimated with a morphological closing. All kernel sizes are fractions of the image width. |
| 2 | `harris_candidates` | `cv2.cornerHarris` on the binary image, threshold relative to the maximum, non-maximum suppression. |
| 2 | `find_region` | The puzzle is the connected component whose number of cell-sized holes is closest to 81 and that is not cut by the image border. |
| 2 | `select_corners` | Among the candidates lying on that component, the four that span the largest quadrilateral, ordered TL, TR, BR, BL and refined with `cv2.cornerSubPix`. |
| 3 | `rectify` | A3 (`getAffineTransform`, TL/TR/BL), A4 (`estimateAffine2D`, least squares over four corners) and H4 (`getPerspectiveTransform`), with the A3 fourth-corner error and the A4 RMS residual. |
| 5 | `find_digits` | Empty-cell criterion: a cell is filled when a connected component smaller than a cell and at least 30 % of its height is centred inside the cell's inner region (7 px margin). |
| 5 | `prepare_digit`, `ocr` | The digit mask is centred on a white square, its strokes are thinned by one pixel, and it is given to Tesseract (`--psm 10`, whitelist 1-9). An empty answer is retried with other paddings. |
| 6 | `find_conflicts` | Repeated digits in a row, column or 3x3 box. |

## Bonus parts

| Bonus | Function | What it does |
|---|---|---|
| B1 solver | `solve` | Backtracking. The empty cell with the fewest candidates is filled first, its candidates are tried in ascending order, and the search goes back on a contradiction. It does not stop at the first solution: a second solution gives `not_unique`. The other answers are `unique`, `none` and `timeout` (5 s). The time is measured with `time.perf_counter` and written as `solver_ms`. |
| B2 solution on the photograph | `draw_solution` | The missing digits are written at the cell centres of a blank 450x450 canvas. The canvas is warped to the size of the photograph with the inverse of H4 (`np.linalg.inv`, `cv2.warpPerspective`) and used as a mask to paint the digits on the photograph. Written only when the solution is unique. |
| B3 many-point homography | `multipoint_homography` | The Harris candidates are moved to the square with H4. A candidate within 8 px of a multiple of 50 in x and y belongs to that grid intersection; the candidates of one intersection are averaged and refined with `cv2.cornerSubPix`. The 64 inner intersections and the four detected corners (at most 68 point pairs) are given to `cv2.findHomography` with RANSAC (3 px). Candidates on the outer border are not used: they lie on the inner edge of the thick line, and with them the result was worse than H4. The grid is still read from H4; HN is saved for the comparison. |
| B4 any orientation | `find_orientation`, `reading_confidence` | The H4 rectification is turned by 0, 90, 180 and 270 degrees. For each turn the digits are put on one small page and read with one `pytesseract.image_to_data` call, and the turn with the highest mean confidence wins. The corner list is then rolled so that it starts at the top-left corner of the puzzle, and all later stages use this order. `rotation_deg` is the counter-clockwise turn that made the puzzle upright. This adds about 0.4 s per image. |

## Code map

| Topic | Where | Notes |
|---|---|---|
| Harris parameters | `sudoku_scan.py`: `BLOCK_SIZE`, `KSIZE`, `K`, `THRESHOLD` | 5, 3, 0.04, 0.05; used in `harris_candidates` |
| Harris parameter study | `tools/param_study.py`: `VALUES` | blockSize 2-13, ksize 3-7, k 0.02-0.20, threshold 0.01-0.6 |
| Corner selection rule | `sudoku_scan.py`: `find_region`, `select_corners`, `detect_corners` | region by hole count, then largest quadrilateral of Harris candidates |
| A3, A4, H4 and residuals | `sudoku_scan.py`: `rectify` | `getAffineTransform`, `estimateAffine2D`, `getPerspectiveTransform` |
| Affine versus homography | `tools/compare_models.py`: `obliqueness`, `line_deviation` | plus cell accuracy of each rectification |
| OCR preprocessing | `sudoku_scan.py`: `binarise`, `find_digits`, `prepare_digit` | empty-cell criterion and digit preparation |
| OCR engine | `sudoku_scan.py`: `ocr`, `read_grid`, `OCR_ATTEMPTS` | Tesseract 5 through `pytesseract` |
| OCR variants and confusions | `tools/ocr_study.py`: `VARIANTS` | preparation on/off, page segmentation modes |
| Metrics | `tools/evaluate.py`: `corner_error`, `evaluate`, `summary` | corner error, cell accuracy, empty-cell F1, exact match, robustness |
| Validity check | `sudoku_scan.py`: `find_conflicts` | row, column and box conflicts |
| B1 solver | `sudoku_scan.py`: `solve`, `SOLVE_SECONDS` | backtracking, fewest candidates first, stops at the second solution |
| B2 solution on the photograph | `sudoku_scan.py`: `draw_solution`, `AR_COLOUR` | canvas warped with the inverse of H4 |
| B3 many-point homography | `sudoku_scan.py`: `multipoint_homography`, `TOLERANCE`, `RANSAC_PX` | `cv2.findHomography` with RANSAC |
| B4 orientation | `sudoku_scan.py`: `find_orientation`, `reading_confidence` | mean Tesseract confidence of four orientations |
| Bonus measurements | `tools/bonus_study.py`: `study` | solver time and correctness, HN versus H4, photographs turned by 90, 180, 270 degrees |

## Evaluation and analysis

```
python sudoku_scan.py --batch data/devset/images --outdir results/devset
python tools/evaluate.py --results results/devset --annotations data/devset/annotations

python tools/param_study.py    --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
python tools/compare_models.py --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
python tools/ocr_study.py      --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
python tools/bonus_study.py    --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
```

Replace `devset` with `personal` for the personal dataset.

## Results on the development set

| Group | Images | Corner error (% of side) | Cell accuracy | Empty-cell F1 | Exact match | Robustness |
|---|---|---|---|---|---|---|
| all | 20 | 0.28 | 1.000 | 1.000 | 1.00 | 1.00 |
| C1 near-frontal | 5 | 0.27 | 1.000 | 1.000 | 1.00 | 1.00 |
| C2 oblique | 9 | 0.27 | 1.000 | 1.000 | 1.00 | 1.00 |
| C3 rotated | 4 | 0.31 | 1.000 | 1.000 | 1.00 | 1.00 |
| C4 uneven light | 6 | 0.27 | 1.000 | 1.000 | 1.00 | 1.00 |
| C5 clutter | 6 | 0.31 | 1.000 | 1.000 | 1.00 | 1.00 |

## Results on the personal dataset

Ten phone photographs of three printed puzzles (`data/personal`).

| Group | Images | Corner error (% of side) | Cell accuracy | Empty-cell F1 | Exact match | Robustness |
|---|---|---|---|---|---|---|
| all | 10 | 0.27 | 0.993 | 0.995 | 0.80 | 1.00 |
| C1 near-frontal | 2 | 0.16 | 1.000 | 1.000 | 1.00 | 1.00 |
| C2 oblique | 5 | 0.29 | 0.988 | 0.990 | 0.80 | 1.00 |
| C3 rotated | 2 | 0.19 | 0.994 | 1.000 | 0.50 | 1.00 |
| C4 uneven light | 4 | 0.25 | 0.997 | 1.000 | 0.75 | 1.00 |
| C5 clutter | 3 | 0.32 | 1.000 | 1.000 | 1.00 | 1.00 |

## Bonus results

Measured with `tools/bonus_study.py`, which also turns every photograph by 90,
180 and 270 degrees and runs the whole pipeline on it again.

| Measurement | Development set (20) | Personal dataset (10) |
|---|---|---|
| B1 unique solution found | 20 / 20 | 8 / 10 |
| B1 solution equal to the true solution | 20 / 20 | 8 / 8 |
| B1 solver time, mean / max (ms) | 10 / 39 | 6 / 24 |
| B3 grid line deviation, H4 / HN (px) | 1.01 / 0.80 | 2.49 / 1.14 |
| B3 cell accuracy, H4 / HN | 1.000 / 1.000 | 0.993 / 0.991 |
| B3 point pairs / RANSAC inliers (mean) | 66 / 65 | 62 / 56 |
| B4 orientation found, four turns per photograph | 80 / 80 | 40 / 40 |
| B4 cell accuracy at 0 / 90 / 180 / 270 degrees | 1.000 / 0.999 / 1.000 / 1.000 | 0.993 / 0.990 / 0.993 / 0.986 |

The two personal photographs without a unique solution are the two with OCR
errors: in `IMG_20261109_235720` five digits are missed, so the puzzle has
several solutions, and in `IMG_20261110_013714` a misread digit breaks the
sudoku rules, so there is no solution. HN lowers the line deviation most
where H4 is weakest (`dev_20`: 6.2 to 2.2 px, `IMG_20261109_235720`: 10.3 to
3.5 px), but the cell accuracy stays the same. A turned photograph can differ
from the upright one in one to three cells, because the detected corners move
by up to one pixel in the rectified image.

The analysis of the main results is in `report/report.tex`.
