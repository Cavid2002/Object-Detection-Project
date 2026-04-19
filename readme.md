# Classical Computer Vision Algorithms
### License Plate Detection Pipeline — CCPD Dataset

> This document explains every classical CV algorithm used in the license plate detection notebook, grouped by pipeline stage. The pipeline is compared against YOLOv11 at the end.

---

## Table of Contents

- [Overview](#overview)
- [Stage 1 — Pre-processing](#stage-1--pre-processing)
  - [Grayscale Conversion](#11-grayscale-conversion)
  - [Histogram Equalization](#12-histogram-equalization)
  - [Gaussian Blur](#13-gaussian-blur)
- [Stage 2 — Feature Extraction](#stage-2--feature-extraction)
  - [Canny Edge Detection](#21-canny-edge-detection)
- [Stage 3 — Region Formation](#stage-3--region-formation)
  - [Morphological Closing](#31-morphological-closing)
- [Stage 4 — Candidate Detection](#stage-4--candidate-detection)
  - [Contour Detection](#41-contour-detection)
  - [Geometric Filtering and Scoring](#42-geometric-filtering-and-scoring)
- [Stage 5 — Evaluation and Tuning](#stage-5--evaluation-and-tuning)
  - [Intersection over Union (IoU)](#51-intersection-over-union-iou)
  - [Grid Search Hyperparameter Tuning](#52-grid-search-hyperparameter-tuning)
- [Results: Classical CV vs YOLO](#results-classical-cv-vs-yolo)

---

## Overview

The classical pipeline is implemented in a single function `detect_plate_classical_from_image()`. It takes a raw BGR image and returns a predicted bounding box for the license plate region. Every stage is a deterministic, hand-engineered operation — no learned weights are involved.

```
Raw Image
   │
   ▼
[Pre-processing]  ──  Normalize lighting and reduce noise
   │
   ▼
[Feature Extraction]  ──  Detect edges
   │
   ▼
[Region Formation]  ──  Bridge gaps, form solid plate-shaped blobs
   │
   ▼
[Candidate Detection]  ──  Find contours, filter by plate geometry
   │
   ▼
Predicted Bounding Box
```

---

## Stage 1 — Pre-processing

Pre-processing standardizes the image before any feature extraction takes place. Its goal is to reduce lighting variation and suppress noise so that subsequent algorithms receive clean, consistent input.

### 1.1 Grayscale Conversion

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Purpose:** Collapse the 3-channel BGR image into a single luminance channel.

**Why it is needed:** All downstream operations (histogram equalization, edge detection, morphology) work on single-channel intensity maps. Discarding color removes irrelevant variation and halves memory usage.

**Formula:**

```
Y = 0.114·B + 0.587·G + 0.299·R
```

OpenCV's weighted luminance formula assigns higher importance to green (most sensitive to the human eye) and lower importance to blue.

---

### 1.2 Histogram Equalization

```python
gray = cv2.equalizeHist(gray)
```

**Purpose:** Redistribute pixel intensities so the image histogram is roughly uniform across 0–255, enhancing contrast.

**Why it is needed:** The CCPD dataset includes `ccpd_weather` (rain, fog, snow) and images with varying brightness (encoded in the filename). A plate captured in fog produces weak edges. Equalization normalizes lighting conditions, making edges consistently detectable regardless of capture environment.

**How it works:**

1. Compute the histogram `H[i]` — the count of pixels at each intensity value `i`.
2. Compute the cumulative distribution function: `CDF[i] = Σ H[0..i]`
3. Map each pixel value to a new value:

```
new_val = round( (CDF[val] − CDF_min) / (N − CDF_min) × 255 )
```

Where `N` is the total number of pixels and `CDF_min` is the first non-zero CDF value.

---

### 1.3 Gaussian Blur

```python
blur = cv2.GaussianBlur(gray, (5, 5), 0)
```

**Purpose:** Smooth the image by convolving it with a 5×5 Gaussian kernel, suppressing high-frequency noise.

**Why it is needed:** Canny edge detection is gradient-based and extremely sensitive to individual noisy pixels. A single JPEG compression artifact or dust speck produces a spurious gradient spike, generating fake edges. Blurring eliminates these before gradient computation.

**The 5×5 kernel** (σ auto-derived from kernel size, normalized by 1/273):

```
 1   4   7   4  1
 4  16  26  16  4
 7  26  41  26  7
 4  16  26  16  4
 1   4   7   4  1
```

Each output pixel is the weighted average of its 5×5 neighborhood, with the highest weight at the center.

---

## Stage 2 — Feature Extraction

### 2.1 Canny Edge Detection

```python
edges = cv2.Canny(blur, canny_low, canny_high)
```

**Tuned parameters:** `canny_low ∈ {50, 100}`, `canny_high ∈ {150, 200}`

**Purpose:** Detect the boundaries of objects in the image by finding pixels where intensity changes sharply. Produces a binary image where white pixels represent edges.

**Why it is used here:** License plates have strong rectangular edges — the plate border, character strokes, and the contrast between characters and the plate background all generate clean gradient responses.

**The algorithm has four internal steps:**

#### Step 1 — Gradient Computation
Applies Sobel filters in X and Y directions to compute the gradient magnitude and direction at each pixel:

```
G = √(Gx² + Gy²)
θ = arctan(Gy / Gx)
```

#### Step 2 — Non-Maximum Suppression
Thins edges to exactly one pixel wide by keeping only pixels that are local maxima along their gradient direction. Thick, blurry edge regions are reduced to clean single-pixel lines.

#### Step 3 — Double Thresholding
Classifies every pixel into one of three categories:

| Condition | Classification |
|---|---|
| `G > canny_high` | **Strong edge** — definitely an edge |
| `canny_low < G < canny_high` | **Weak edge** — possibly an edge |
| `G < canny_low` | **Not an edge** — discarded |

#### Step 4 — Edge Tracking by Hysteresis
Weak edge pixels are kept only if they are directly connected to a strong edge pixel. This links real edges across small gaps while discarding isolated noise blobs.

---

## Stage 3 — Region Formation

### 3.1 Morphological Closing

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
```

**Tuned parameters:** `kernel_w ∈ {13, 17}`, `kernel_h ∈ {3, 5}`

**Purpose:** Bridge gaps between nearby edge pixels to form solid, connected blobs that correspond to the license plate character region.

**Why it is needed:** After Canny, each character on the plate generates its own cluster of disconnected edge fragments. Without closing, the plate appears as 7 separate clusters rather than one unified region, making it impossible to detect as a single rectangle.

**Closing = Dilation followed by Erosion:**

| Operation | Formula | Effect |
|---|---|---|
| **Dilation** | `dst(x,y) = max over kernel of src(x+x', y+y')` | Expands white regions — fills gaps between nearby edges |
| **Erosion** | `dst(x,y) = min over kernel of src(x+x', y+y')` | Shrinks white regions back to their original size |

The net effect is that gaps *between* edges are filled (dilation) while the overall size of the region is preserved (erosion).

**Why a wide, short kernel `(17 × 5)`?**

License plates have an aspect ratio of approximately 4:1. A wide kernel bridges the horizontal gaps between individual characters, merging "皖A 12345" into one connected horizontal region. The small height prevents vertically distant, unrelated regions from merging.

---

## Stage 4 — Candidate Detection

### 4.1 Contour Detection

```python
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**Purpose:** Trace the boundaries of all white connected regions in the morphological output, returning a list of contour point arrays.

**Flags explained:**

| Flag | Meaning |
|---|---|
| `RETR_EXTERNAL` | Retrieve only outermost contours — ignores holes or nested contours inside a region |
| `CHAIN_APPROX_SIMPLE` | Compress runs of collinear points to only their endpoints, saving memory |

`RETR_EXTERNAL` is important here: it avoids retrieving the inner loops of characters (e.g., the enclosed center of "0" or "8"), which would create noisy candidate regions inside the plate.

**Algorithm:** Based on Suzuki–Abe topological structural analysis. Scans the binary image raster-line by raster-line and traces the outer border of each connected foreground region.

---

### 4.2 Geometric Filtering and Scoring

After extracting all contours, each one is passed through a multi-criterion pipeline that first filters out clearly non-plate candidates with hard rules, then scores the survivors to find the best match.

```python
x, y, w, h = cv2.boundingRect(cnt)
ratio       = w / float(h)
area        = w * h
fill_ratio  = cv2.contourArea(cnt) / float(area + 1e-6)
```

#### Hard Filter Rules

All of the following conditions must be satisfied for a contour to proceed:

| Criterion | Default Range | Rationale |
|---|---|---|
| Aspect ratio `w/h` | `[2.0, 6.5]` | Plates are wider than tall |
| Minimum width | `≥ 40 px` | Rejects tiny noise fragments |
| Minimum height | `≥ 15 px` | Rejects thin lines |
| Bounding box area | `≥ 1000 px²` | Rejects microscopic detections |
| Fill ratio | `≥ 0.35` | Rejects L-shapes, arcs, and sparse contours |

#### Soft Scoring Heuristic

Surviving candidates are scored on four criteria. The candidate with the highest total score is returned as the plate prediction:

```python
score += 2.0  if 2.5  <= ratio  <= 5.5   # ideal plate aspect ratio
score += 2.0  if 0.08 <= norm_w <= 0.60  # plate covers 8–60% of image width
score += 2.0  if 0.03 <= norm_h <= 0.25  # plate covers 3–25% of image height
score += fill_ratio                        # bonus for a compact, well-filled shape
```

Maximum possible score: **7.0** (6 from binary checks + up to 1.0 from fill ratio).

---

## Stage 5 — Evaluation and Tuning

### 5.1 Intersection over Union (IoU)

```python
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    union = areaA + areaB - inter_area + 1e-6
    return inter_area / union
```

**Purpose:** Measure the quality of a predicted bounding box by comparing its overlap with the ground-truth box.

**Formula:**

```
IoU = Area of Intersection / Area of Union
```

**Interpretation:**

| IoU Value | Meaning |
|---|---|
| `1.0` | Perfect match — prediction equals ground truth |
| `≥ 0.5` | Standard threshold for a "correct" detection (IoU@0.5) |
| `0.0` | No overlap at all |

The `1e-6` term in the denominator prevents division by zero when both boxes have zero area.

Three aggregate metrics are reported:

- **Detection Rate** — fraction of images where any prediction was returned
- **Mean IoU** — average IoU across all test images (0.0 for missed detections)
- **IoU@0.5** — fraction of images where the prediction achieved IoU ≥ 0.5

---

### 5.2 Grid Search Hyperparameter Tuning

```python
from itertools import product

for canny_low, canny_high, kernel_w, kernel_h, min_ratio, max_ratio in product(
    [50, 100], [150, 200], [13, 17], [3, 5], [2.0, 2.5], [5.5, 6.5]
):
    result = evaluate_classical(val_data, **detector_params)
    score  = result["iou@0.5"]
```

**Purpose:** Exhaustively search the parameter space to find the combination that maximizes IoU@0.5 on the validation set.

**Search space:**

| Parameter | Values Tried | Controls |
|---|---|---|
| `canny_low` | `{50, 100}` | Lower threshold for weak edges |
| `canny_high` | `{150, 200}` | Upper threshold for strong edges |
| `kernel_w` | `{13, 17}` | Horizontal closing reach |
| `kernel_h` | `{3, 5}` | Vertical closing reach |
| `min_ratio` | `{2.0, 2.5}` | Minimum plate aspect ratio |
| `max_ratio` | `{5.5, 6.5}` | Maximum plate aspect ratio |

**Total combinations:** 2 × 2 × 2 × 2 × 2 × 2 = **64** (pairs with `canny_low ≥ canny_high` are skipped)

**Strategy:** Brute-force grid search, feasible here because the parameter space is small and each inference call is fast (no GPU required). A 200-image validation sample is used for speed.

---

## Results: Classical CV vs YOLO

Both methods were evaluated on the same held-out test set using identical metrics.

| Metric | Classical CV | YOLOv11 |
|---|---|---|
| Detection Rate | 0.40 | 0.99 |
| Mean IoU | 0.07 | 0.89 |
| IoU@0.5 | 0.08 | 0.99 |

### Key Observations

**Classical CV limitations — sensitive to:**
- Blur (from `ccpd_blur` subset)
- Rotation (from `ccpd_rotate` subset)
- Weather and lighting variation (from `ccpd_weather` subset)

Each stage uses fixed mathematical rules chosen to work for typical, well-lit, straight-on plates. When these assumptions break, detection fails.

**YOLO advantages — robust across:**
- Blurred images
- Rotated plates
- Rain, fog, and overexposure

YOLO learns which features matter directly from thousands of labeled examples, making it far more robust to the variations present across CCPD subsets.

### Conclusion

The classical pipeline is interpretable and requires no training data, but its hand-engineered assumptions make it brittle in real-world conditions. YOLO trades interpretability for dramatically higher accuracy and generalization.

---

*Generated from `computer-vision-project-1.ipynb` — CCPD License Plate Detection Project*