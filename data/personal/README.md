# Personal dataset

```
images/        original, unedited phone photos (EXIF kept), e.g. s<student-number>_01.jpg
annotations/   one JSON per photo with the same base name (Appendix A format)
puzzles/       one text file per printed puzzle: 9 lines of 9 digits, 0 = empty
```

Every photo contains the handwritten card with the student number and the
capture date, placed outside the grid.

Annotating a photo (corners are clicked by hand, the grid comes from the
puzzle file):

```
python tools/annotate.py --image data/personal/images/s123_01.jpg --grid data/personal/puzzles/puzzle1.txt --conditions C2 C4
```

| Condition | Meaning | Minimum |
|---|---|---|
| C1 | near-frontal | 1 |
| C2 | oblique view, moderate to strong | 3, at different angles |
| C3 | in-plane rotation of 15-30 degrees | 1 |
| C4 | cast shadow or low light | 1 |
| C5 | cluttered background | 1 |
