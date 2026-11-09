# SudokuScan

COMP 482 Computer Vision, Project 1. Takes a phone photograph of a printed
sudoku puzzle and returns the puzzle as a 9x9 digital grid:

1. find the four outer corners of the grid with the Harris corner detector,
2. rectify the puzzle with two affine models (A3, A4) and a homography (H4),
3. split the rectified image into cells, decide which are empty and read the
   printed digits with Tesseract,
4. check the result against the sudoku rules.

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
| `<name>_result.json` | corners (TL, TR, BR, BL), A3, A4, H4, A3 fourth-corner error, A4 RMS residual, grid, verdict, runtime |
| `<name>_harris.png` | Harris response map with all candidates (yellow) and the four selected corners (red) |
| `<name>_rect_A3.png`, `_rect_A4.png`, `_rect_H4.png` | the three 450x450 rectifications |
| `<name>_debug.png` | H4 rectification with cell boundaries, recognised digits and conflicting cells (red) |

The verdict is `valid`, `invalid` or `not_found`. The program runs on CPU and
takes about 3 seconds per image.

## Repository structure

```
sudoku_scan.py            the whole pipeline, one function per stage
requirements.txt
tools/
    evaluate.py           Section 7 metrics, overall and per condition C1-C5
    param_study.py        Harris parameter study (Stage 2)
    compare_models.py     affine versus homography comparison (Stage 4)
    ocr_study.py          cell preparation, segmentation modes, confusions (Stage 5)
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

## Method

| Stage | Function | What it does |
|---|---|---|
| 1 | `preprocess`, `binarise` | Downscale to 1000 px, grey, Gaussian blur. A pixel is ink when it is darker than 85 % of the local paper brightness, which is estimated with a morphological closing. All kernel sizes are fractions of the image width. |
| 2 | `harris_candidates` | `cv2.cornerHarris` on the binary image, threshold relative to the maximum, non-maximum suppression. |
| 2 | `find_region` | The puzzle is the connected component whose number of cell-sized holes is closest to 81 and that is not cut by the image border. |
| 2 | `select_corners` | Among the candidates lying on that component, the four that span the largest quadrilateral, ordered TL, TR, BR, BL and refined with `cv2.cornerSubPix`. |
| 3 | `rectify` | A3 (`getAffineTransform`, TL/TR/BL), A4 (`estimateAffine2D`, least squares over four corners) and H4 (`getPerspectiveTransform`), with the A3 fourth-corner error and the A4 RMS residual. |
| 5 | `find_digits` | Empty-cell criterion: a cell is filled when a connected component smaller than a cell and at least 30 % of its height is centred inside the cell's inner region (7 px margin). |
| 5 | `prepare_digit`, `ocr` | The digit mask is centred on a white square and given to Tesseract (`--psm 10`, whitelist 1-9). |
| 6 | `find_conflicts` | Repeated digits in a row, column or 3x3 box. |

## Evaluation and analysis

```
python sudoku_scan.py --batch data/devset/images --outdir results/devset
python tools/evaluate.py --results results/devset --annotations data/devset/annotations

python tools/param_study.py    --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
python tools/compare_models.py --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
python tools/ocr_study.py      --images data/devset/images --annotations data/devset/annotations --outdir report/figures/devset
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

The development images are synthetic; results on real photographs are in the
report.
