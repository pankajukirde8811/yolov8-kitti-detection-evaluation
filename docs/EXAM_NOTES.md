# Task 2 – Exam notes: evaluating an object detector

## 1. Goal
Run YOLOv8 on the 20 KITTI scenes, keep only cars, and evaluate the 2D boxes against the
ground truth (GT): IoU for every box, plus precision and recall per scene. These 2D boxes
are the input for Task 3 (distance estimation), so we need to know how reliable they are.

## 2. Data (KITTI_Selection)
* 20 images of 1242 × 375 px (left colour camera), 98 GT cars in total
* `labels/<id>.txt`: `Car xmin ymin xmax ymax gt_distance` (distance in m, for Task 3)
* `calib/<id>.txt`: intrinsic matrix K (3×3, f ≈ 721 px), not needed in Task 2
* Scenes 006121 and 006130 contain **no GT car** (pedestrian zones).
* The professor *reduced* the objects: only "Car" is kept. The full KITTI labels (LiDAR
  folder) also contain Van, Truck, Pedestrian, **DontCare**, truncation and occlusion. I
  use them **only to explain errors**, never for the evaluation itself.

## 3. Detector: YOLOv8
* One-stage detector: a single CNN pass predicts boxes + class scores on a grid, then
  non-maximum suppression (NMS) removes duplicate boxes.
* Pre-trained on COCO (80 classes) → keep class 2 = `car`; drop person, truck, bicycle…
* Confidence threshold 0.25 (ultralytics default). Model: YOLOv8m (n/s compared in
  `results/ablation.md`).

## 4. IoU
```
IoU(A, B) = area(A ∩ B) / area(A ∪ B)        0 = no overlap … 1 = identical
```
Intersection: `max(0, min(x2a,x2b) − max(x1a,x1b)) · max(0, min(y2a,y2b) − max(y1a,y1b))`.
A detection counts as correct if IoU ≥ 0.5 (PASCAL VOC). KITTI's own car benchmark
uses 0.7.

## 5. Matching (the professor stresses this!)
In crowded scenes a detection overlaps several GT cars (parked cars in a row). Rule: **one
detection ↔ at most one GT car**. A second detection on the same car is a false positive
(duplicate).

* **Greedy** (VOC/COCO): sort detections by confidence; each takes the free GT box with the
  highest IoU ≥ threshold.
* **Hungarian** (optimal assignment): maximise the *sum* of IoUs over all one-to-one pairs
  (`scipy.optimize.linear_sum_assignment`).
* **Pitfall found in 006211:** without *gating*, Hungarian traded one good pair (IoU 0.60)
  for two weak ones (0.42 + 0.40 = 0.82 > 0.60). Both were then discarded (< 0.5), so a
  correct detection was lost. Fix: set IoU < threshold to 0 **before** solving.
  Ungated: 76 TP → gated: 78 TP (= greedy).

## 6. Metrics
* TP = matched detections, FP = unmatched detections, FN = unmatched GT cars
* **Precision** = TP / (TP + FP): how many of my detections are real cars?
* **Recall** = TP / (TP + FN): how many of the real cars did I find?
* **Undefined cases:** 006121 has no GT and no detection, so P and R are both 0/0 → n/a.
  006130 has no GT but one detection, so P = 0 and R is undefined. Writing 0 or 1 there
  would distort the average.
* **Micro average** (sum TP/FP/FN over all scenes) vs **macro** (average of the scene
  values): macro gives a scene with 1 car the same weight as one with 13.

## 7. Results (YOLOv8m, conf 0.25, IoU 0.5, gated Hungarian)

| scene | GT | det | TP | FP | FN | P | R | mean IoU |
|---|---|---|---|---|---|---|---|---|
| 006037 | 5 | 6 | 4 | 2 | 1 | 0.67 | 0.80 | 0.68 |
| 006042 | 2 | 3 | 2 | 1 | 0 | 0.67 | 1.00 | 0.84 |
| 006048 | 6 | 6 | 5 | 1 | 1 | 0.83 | 0.83 | 0.86 |
| 006054 | 8 | 10 | 6 | 4 | 2 | 0.60 | 0.75 | 0.77 |
| 006059 | 6 | 7 | 6 | 1 | 0 | 0.86 | 1.00 | 0.75 |
| 006067 | 5 | 5 | 4 | 1 | 1 | 0.80 | 0.80 | 0.86 |
| 006097 | 6 | 10 | 6 | 4 | 0 | 0.60 | 1.00 | 0.80 |
| 006098 | 4 | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 0.82 |
| 006121 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | – |
| 006130 | 0 | 1 | 0 | 1 | 0 | 0.00 | n/a | – |
| 006206 | 1 | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 0.77 |
| 006211 | 13 | 12 | 8 | 4 | 5 | 0.67 | 0.62 | 0.84 |
| 006227 | 2 | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 0.83 |
| 006253 | 6 | 7 | 4 | 3 | 2 | 0.57 | 0.67 | 0.78 |
| 006291 | 12 | 12 | 7 | 5 | 5 | 0.58 | 0.58 | 0.78 |
| 006310 | 7 | 7 | 6 | 1 | 1 | 0.86 | 0.86 | 0.76 |
| 006312 | 8 | 9 | 6 | 3 | 2 | 0.67 | 0.75 | 0.84 |
| 006315 | 2 | 9 | 2 | 7 | 0 | 0.22 | 1.00 | 0.80 |
| 006329 | 3 | 5 | 3 | 2 | 0 | 0.60 | 1.00 | 0.78 |
| 006374 | 2 | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 0.90 |
| **all (micro)** | 98 | 118 | **78** | **40** | **20** | **0.66** | **0.80** | 0.80 |

Macro average: P = 0.69, R = 0.87. IoU of correct detections: median 0.81, minimum 0.55.
IoU of every single box: `results/yolov8m/per_detection.csv` and `report.md`.

## 8. Discussion – where and why it goes wrong

### 8.1 Missed cars (FN = 20): mostly a box-definition problem, not a detection problem
| reason | count |
|---|---|
| found, but IoU < 0.5, **largely occluded** | 15 |
| found, but IoU < 0.5, partly / not occluded | 3 |
| detected as "truck" (pickup in 006054, IoU 0.86) | 1 |
| really not detected (006067: 24 px high, 44 m, partly occluded) | 1 |

Recall by KITTI occlusion level: **0.98** (visible), **0.89** (partly), **0.44** (largely
occluded). At IoU ≥ 0.3 the largely occluded cars reach **0.93**, so YOLO does see them.

**Why?** KITTI's 2D GT box of an occluded car covers the *whole* car, including the hidden
part ("amodal" box). I checked this: projecting the labelled 3D box with P2 into the image
gives the 2D GT box with median IoU 0.97, also for largely occluded cars. YOLO can only box
what it *sees* (the visible part). Example: a car parked behind another one: GT = whole
car, YOLO = only the visible roof/back → IoU 0.2–0.5. This hurts most in crowded scenes
with parked cars in a row (006211, 006291).
Similar: cars truncated at the image border (006291 gt2, 006312 gt6). The GT box is clipped
there, but YOLO's box is shaped differently.

### 8.2 Wrong detections (FP = 40): many are real vehicles the GT doesn't contain
| reason | count |
|---|---|
| the other half of 8.1: detection of an occluded car with IoU < 0.5 | 16 |
| inside a KITTI *DontCare* region (far/hidden cars deliberately unlabelled) | 8 |
| no GT object at all (8) or small < 25 px (4): **checked by eye** → 9 real cars missing from the GT, 3 real false alarms | 12 |
| van in the full labels (YOLO says car, KITTI says Van) | 3 |
| duplicate box on an already matched car | 1 |

Checked by eye (crops of all 12): real but unlabelled cars in 006054, 006097, 006211,
006291, 006312 (2×) and 006315 (3×, one cut off at the left border with conf 0.87). The only
**real false alarms** are a pedestrian area (006130, conf 0.25), a shop window (006211,
conf 0.32) and a dark hedge/shadow (006253, conf 0.43). All three have low confidence.

**Consequence:** a "false positive" here often means "the label is missing", not "YOLO
is wrong". Precision is **underestimated**: 006315 has P = 0.22 although almost every
detection is a vehicle.

### 8.3 Class confusion car / van / truck
COCO and KITTI define classes differently. A pickup (006054) is a KITTI "Car" but a
YOLO "truck", while vans are a YOLO "car" but a KITTI "Van". Counting "truck" as car
gains 1 TP but adds 15 FP, so car only is the better choice.

## 9. How the choices change the result (`results/ablation.md`)
| choice | precision | recall |
|---|---|---|
| default (m, 0.25, IoU 0.5, gated Hungarian) | 0.66 | 0.80 |
| IoU 0.3 / 0.7 | 0.74 / 0.51 | 0.89 / 0.61 |
| Hungarian without gating | 0.64 | 0.78 |
| YOLOv8n / v8s | 0.72 / 0.71 | 0.74 / 0.81 |
| conf 0.5 / 0.7 | 0.84 / 0.90 | 0.70 / 0.58 |

* **Confidence threshold** trades precision against recall (PR curve: `results/pr_curve.png`).
* **The larger model has *lower* precision**, because it finds more of the unlabelled far
  cars. That's another sign that the GT, not the detector, limits precision.
* **The IoU threshold has the biggest effect.** It's not a property of the detector but of
  the evaluation rule.

## 10. Outlook to Task 3
For distance estimation (e.g. `Z = f · H / h`, with real car height H and box height h), a
box that covers only the visible part of an occluded car is too small, so the distance
comes out too large. So for Task 3, prefer fully visible, untruncated cars.

## 11. Typical exam questions
* Define IoU, precision, recall. Why not use accuracy? → There are no "true negatives" in
  detection: you can't count all the empty boxes that weren't predicted.
* Why must matching be one-to-one? → Otherwise 2 boxes on one car would both count as TP.
* Greedy vs. Hungarian? Why gating? → See §5, example 006211.
* Precision/recall of a scene without GT cars? → Recall undefined (0/0), precision 0 if
  there are detections.
* Why is recall low for occluded cars although YOLO sees them? → Amodal GT vs. visible
  detection box (§8.1).
* Why can precision be too pessimistic here? → Missing labels, DontCare, Van vs Car (§8.2).
* Effect of the IoU and confidence thresholds? → §9.
