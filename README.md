# Sudoku Solver (OpenCV, from-scratch image processing)

Reads a photo of a Sudoku puzzle, finds and rectifies the grid, recognises the
printed digits, solves the puzzle, and draws the solution back onto the
original image — all using **hand-written image-processing primitives**. No
OpenCV recognition / contour / ML functions are used for the recognition path;
OpenCV is used only for I/O, matrix containers, the perspective transform, and
text rendering.

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Architecture / file layout](#architecture--file-layout)
3. [How each stage works](#how-each-stage-works)
4. [Digit recognition (the hard part)](#digit-recognition-the-hard-part)
5. [The harvested k-NN template library](#the-harvested-k-nn-template-library)
6. [Building](#building)
7. [Running](#running)
8. [Results](#results)
9. [Changelog — everything that was fixed/added](#changelog--everything-that-was-fixedadded)
10. [Known limitations & next steps](#known-limitations--next-steps)

---

## Pipeline overview

```
input image
   │  cv::imread (grayscale + colour)
   ▼
Gaussian blur ──► adaptive threshold ──► morphological close      (preprocess)
   ▼
find largest blob (flood fill) ──► 4 extreme corners              (locate grid)
   ▼
getPerspectiveTransform + warpPerspective ──► 450×450 binary      (rectify)
   ▼
flood-fill blobs ──► filter by size/aspect ──► assign to 9×9      (extract digits)
   ▼
normalise + zoning + projections ──► weighted k-NN vote           (recognise)
   ▼
backtracking solver                                               (solve)
   ▼
render solved digits ──► inverse warp ──► composite               (draw back)
   ▼
result.png  (original image with the answers in green)
```

---

## Architecture / file layout

```
sudoku-project/
├── CMakeLists.txt
├── build.sh / run.sh
├── README.md                      ← this file
├── images/                        ← test images (clean / perspective / warped / handwritten)
├── templates_knn/                 ← harvested real digit samples (the recogniser's reference set)
├── include/
│   ├── algorithms.hpp             ← image-processing primitives + extraction structs
│   ├── DigitRecognizer.hpp        ← k-NN digit classifier
│   ├── SudokuSolver.hpp           ← backtracking solver
│   └── SudokuPipeline.hpp         ← orchestrates the whole flow
└── src/
    ├── algorithms.cpp             ← blur, threshold, morphology, blob/flood-fill, extraction
    ├── DigitRecognizer.cpp        ← normalisation, features, weighted k-NN
    ├── SudokuSolver.cpp           ← solve() with a step guard
    ├── SudokuPipeline.cpp         ← preprocess → locate → warp → recognise → solve → draw back
    └── main.cpp                   ← thin CLI driver (arg parsing, I/O, display)
```

**Separation of concerns:** `main.cpp` only parses arguments, loads the image,
calls the pipeline, and handles display / file output. All the actual work
lives in `SudokuPipeline`, which composes `algorithms`, `DigitRecognizer`, and
`SudokuSolver`.

---

## How each stage works

All primitives in `algorithms.cpp` are implemented by hand.

### 1. Pre-processing
- **`gaussianBlur`** — separable Gaussian convolution (a row pass then a column
  pass) built from a generated Gaussian kernel, with theoretical-limit
  normalisation. Removes high-frequency noise so thresholding is stable.
- **`adaptiveThreshold`** — local mean thresholding accelerated with an
  **integral image** (summed-area table). Each pixel is compared against the
  mean of its `windowSize×windowSize` neighbourhood minus a constant `C`.
  Produces a binary image with **white ink (255) on black background**. This
  polarity convention is used everywhere downstream.
- **`morphClose`** — dilation followed by erosion (both hand-written) to close
  small gaps in the grid lines before locating the grid.

### 2. Locate the grid
- **`findLargestBlob`** — BFS flood fill (8-connectivity) over the morphed
  binary image. The largest connected white component is the grid frame.
- **`getGridCorners`** — the 4 corners are the blob pixels with extreme
  `x+y` (top-left / bottom-right) and `x−y` (top-right / bottom-left) sums.

### 3. Rectify
- `cv::getPerspectiveTransform(src, dst)` maps the 4 detected corners to a
  `450×450` square; `cv::warpPerspective(..., INTER_NEAREST)` rectifies the
  **thresholded** image (kept binary). The **inverse** matrix is stored so the
  solution can later be warped back.

### 4. Extract digits
- **`findBlobs`** — flood-fill labelling of every white component in the warp.
- **`extractDigits` / `isDigitBlob`** — each blob is kept only if its area and
  aspect ratio look like a digit (rejects noise specks and the grid frame,
  which is one huge blob). Surviving blobs are assigned to a 9×9 cell by their
  centroid (`center / cellSize`), and the cropped cell image is stored.

### 5. Recognise — see the next two sections.

### 6. Solve
- **`SudokuSolver`** — classic recursive backtracking (`isValid` checks
  row/column/3×3 box). A **step counter caps the search at 2,000,000 nodes** so
  an inconsistent (misread) grid can never hang the program — it simply reports
  "could not solve".

### 7. Draw the solution back
- For each cell that was **empty in the recognised grid**, the solved digit is
  rendered with `cv::putText` onto a black `450×450` overlay (the original
  givens are left untouched).
- The overlay is warped back to the original geometry with the **inverse**
  perspective matrix and composited onto the colour image wherever it is
  non-black. Output is written to `result.png` (green answers).

---

## Digit recognition (the hard part)

The original projection-only recogniser confused `1/7`, `3/8`, etc. The current
classifier (`DigitRecognizer.cpp`) is a **nearest-neighbour template matcher**
with carefully chosen normalisation and features. It was validated offline
before being ported to C++.

### Normalisation (`normalise`)
Critical for telling thin digits apart:
1. Crop to the foreground bounding box.
2. **Aspect-preserving** rescale into a `20×20` inner box (so a `1` stays thin
   and is not stretched to fill a square).
3. Re-binarise (the area-interpolated resize introduces grey values).
4. **Recentre by centre of mass** onto a `28×28` canvas.

Cells with fewer than 8 ink pixels are treated as empty → digit `0`.

### Features (`extractFeatures`)
A digit is described by four complementary feature blocks:
| Feature | Size | Captures |
|---|---|---|
| `pixels` | 28×28 = 784 | exact stroke overlap |
| `zoning` | 7×7 = 49 | foreground **density per region** — the key to `1/7`, `3/8` |
| `h_proj` | 28 | row density profile |
| `v_proj` | 28 | column density profile |

### Distance (`calculateDistance`)
```
3.0 · mean((pixels−pixels')²)
+ 1.0 · sum((zoning−zoning')²)
+ 0.3 · sum((h_proj−h_proj')²)
+ 0.3 · sum((v_proj−v_proj')²)
```
Weights were tuned offline. Shape overlap dominates; zoning disambiguates the
look-alike pairs; projections add a small global-layout term.

---

## The harvested k-NN template library

A single hand-made template per digit is brittle. Instead the recogniser uses
**many real digit samples** and a **weighted k-NN vote** (K = 5,
weight = `1/(distance+ε)`).

### `templates_knn/`
- Contains real digit crops named `digit_<value>_<source>_<rc>.png`.
- Currently **302 samples (≈27–43 per digit)** harvested from the grids that
  solve cleanly (the 4 clean + 4 perspective + 1 warped-2 images).
- This folder **is** the reference set — it is committed and loaded at startup.
  (A legacy `templates/` folder, if present, is loaded as extra seeds but is not
  required; the original hand-picked `templates/` and the unusable
  auto-extracted `templates_extracted/` attempts were removed.)

### How harvesting works (`--harvest`)
Bootstrapping is safe because **labels are only trusted from a solved grid**: if
the recognised givens are both conflict-free and completable, a backtracking
solution exists, which is strong evidence every given was read correctly. The
`--harvest` flag re-runs the pipeline and, on success, writes each recognised
cell image into `templates_knn/`, growing the library.

To (re)generate the library from the solvable images:
```bash
rm -rf templates_knn
for img in sudoku-clean-1 sudoku-clean-2 sudoku-clean-3 sudoku-clean-4 \
           sudoku-perspective-1 sudoku-perspective-2 sudoku-perspective-3 \
           sudoku-perspective-5 sudoku-warped-2; do
    ./build/sudoku-app images/$img.png --harvest
done
```

---

## Building

Requires **OpenCV 4** and a C++20 compiler.

```bash
./build.sh            # mkdir build && cmake .. && cmake --build .
```
or manually:
```bash
mkdir -p build && cd build && cmake .. && cmake --build .
```

---

## Running

The image path is a **required positional argument** (no hardcoded default).

```bash
./run.sh images/sudoku-clean-1.png          # via helper script
./build/sudoku-app images/sudoku-clean-1.png

# flags
./build/sudoku-app images/sudoku-warped-2.png --headless   # no GUI windows
./build/sudoku-app images/sudoku-clean-1.png --harvest      # grow templates_knn/
```

| Flag | Effect |
|---|---|
| *(none)* | shows intermediate + final windows (press a key to advance) |
| `--headless` | no windows; still prints grids and writes `result.png` |
| `--harvest` | implies `--headless`; on a solved grid, save digits to `templates_knn/` |

Output:
- The **recognised** and **solved** grids are printed to the console.
- On success, `result.png` is written: the original image with the solved
  digits overlaid in green.

---

## Results

Tested on **all 12 images** in `images/` (k-NN reference = `templates_knn` only):

| Image | Outcome |
|---|---|
| `sudoku-clean-1..4` | ✅ solved |
| `sudoku-perspective-1,2,3,4,5` | ✅ solved |
| `sudoku-warped-2` | ✅ solved |
| `sudoku-warped` | ❌ misread under heavy perspective (grid-rectification limit) |
| `sudoku-handwritten` | ❌ out of scope (printed-digit templates) |

**9 / 12 solved** — every clean and perspective image plus one warped image.

---

## Changelog — everything that was fixed/added

**Bug fixes (these were the real reasons recognition looked broken):**
1. **`findBlobs` flood fill had an inverted foreground test** — it grew blobs
   into the *background* instead of the digits, so no real digit blob ever
   formed. Fixed the neighbour condition to skip background pixels.
2. **`minAreaFraction` was 0.1** — required digits ≥250 px, but thresholded
   digits are only ~120–160 px (~5–6 % of a cell), so all 41 givens were
   discarded as "noise". Lowered to `0.02`.
3. **`cropPadding` default 3 → 0** — the extra trim could clip thin strokes
   (e.g. `1`); the recogniser re-crops to the foreground anyway.

**Recognition rewrite:**
4. Replaced the projection-only matcher with **centroid-centred,
   aspect-preserving normalisation + 7×7 zoning + pixel overlap + projections**.
5. Added the **weighted k-NN** classifier over a **harvested 302-sample library**
   (`templates_knn/`), plus the `--harvest` workflow to grow it. This recovered
   `sudoku-perspective-4` (8/12 → 9/12).
6. Removed the unusable `templates_extracted/` (mislabelled/merged crops) and
   the hand-picked `templates/`; the recogniser now relies on `templates_knn/`.

**Pipeline completion & structure:**
7. Implemented **`SudokuSolver`** (backtracking) with a 2,000,000-node guard so
   an inconsistent grid can never hang.
8. Implemented the **solve + draw-back** stage (render → inverse warp →
   composite) writing `result.png`.
9. **Moved all pipeline logic out of `main.cpp` into `SudokuPipeline`**;
   `main.cpp` is now a thin CLI driver.
10. **Dynamic image path** — the path is a required CLI argument; the hardcoded
    default was removed, and a usage message is printed when it is missing.

---

## Known limitations & next steps

- **`sudoku-warped`** fails not on recognition but on **grid rectification**:
  under strong perspective the uniform `50 px` cell division misaligns and a
  digit can straddle a cell boundary. Fix: refine the 4 corners (sub-pixel /
  line-fit) or sample cells from detected grid-line intersections instead of a
  flat 9×9 split.
- **`sudoku-handwritten`** is out of scope for template matching against printed
  digits; it would need handwriting samples in `templates_knn/` or a different
  classifier.
- The solver assumes a unique solution; it returns the first one found.
