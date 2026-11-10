# Personal dataset

```
images/        original, unedited phone photos (EXIF kept)
annotations/   one JSON per photo with the same base name (Appendix A format)
puzzles/       one text file per printed puzzle: 9 lines of 9 digits, 0 = empty
```

Every photo contains the handwritten card with the student number and the
capture date, placed outside the grid.

Annotating a photo (corners are clicked by hand, the grid comes from the
puzzle file):

```
python tools/annotate.py --image data/personal/images/IMG_20261109_234517.jpg --grid data/personal/puzzles/puzzle2_websudoku.txt --conditions C4 C2
```

| Condition | Meaning | Minimum |
|---|---|---|
| C1 | near-frontal | 1 |
| C2 | oblique view, moderate to strong | 3, at different angles |
| C3 | in-plane rotation of 15-30 degrees | 1 |
| C4 | cast shadow or low light | 1 |
| C5 | cluttered background | 1 |

| Puzzle file | Source |
|---|---|
| `puzzle1_krazydad.txt` | KrazyDad, Easy Sudoku, Volume 21, Book 9, Sudoku #6 |
| `puzzle2_websudoku.txt` | Web Sudoku, Easy Puzzle 8,468,720,789 |
| `puzzle3_gazete.txt` | Istanbul newspaper, 08.11.2026 |

## Photographs

Ten 3072x4096 JPEG files straight from the phone, with EXIF data. The puzzles
have 39, 32 and 25 givens.

| File | Puzzle | Conditions |
|---|---|---|
| `IMG_20261109_233756.jpg` | 1, KrazyDad | C1 |
| `IMG_20261109_233825.jpg` | 1, KrazyDad | C5 C2 |
| `IMG_20261109_234517.jpg` | 2, Web Sudoku | C4 C2 |
| `IMG_20261109_234609.jpg` | 2, Web Sudoku | C2 |
| `IMG_20261109_235321.jpg` | 2, Web Sudoku | C1 C4 |
| `IMG_20261109_235453.jpg` | 3, newspaper | C5 C4 |
| `IMG_20261109_235538.jpg` | 3, newspaper | C2 |
| `IMG_20261109_235720.jpg` | 3, newspaper | C2 |
| `IMG_20261109_235818.jpg` | 3, newspaper | C3 C5 |
| `IMG_20261110_013714.jpg` | 1, KrazyDad | C3 C4 |

Coverage: C1 in 2 photographs, C2 in 5, C3 in 2, C4 in 4, C5 in 3.
