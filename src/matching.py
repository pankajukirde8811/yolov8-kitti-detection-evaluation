"""
IoU, matching detections <-> ground truth, precision / recall.

IoU (Intersection over Union) of two boxes A, B:
            area(A ∩ B)
    IoU = ---------------         0 = no overlap, 1 = identical boxes
            area(A ∪ B)

Matching problem: a detection may overlap several GT cars (crowded scenes, parked cars in a
row) and a GT car may overlap several detections. Each GT car may be matched to AT MOST ONE
detection and vice versa (one-to-one). Two strategies:

  * greedy  (PASCAL VOC / COCO style): go through the detections by descending confidence,
            each takes the still-free GT box with the highest IoU (if IoU >= threshold).
  * hungarian (optimal assignment): choose the one-to-one assignment that maximises the sum
            of IoUs, solved with the Hungarian algorithm (scipy linear_sum_assignment).
            Pairs below the IoU threshold are excluded BEFORE solving ("gating").
  * hungarian_raw: the same without gating (only to show why gating is necessary).

After matching:
    TP = matched detections, FP = unmatched detections, FN = unmatched GT cars
    precision = TP / (TP + FP)   "how many of my detections are real cars?"
    recall    = TP / (TP + FN)   "how many of the real cars did I find?"
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(dets: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """rows = detections, cols = GT boxes"""
    M = np.zeros((len(dets), len(gts)))
    for i, d in enumerate(dets):
        for j, g in enumerate(gts):
            M[i, j] = iou(d, g)
    return M


def match_greedy(M: np.ndarray, conf: np.ndarray, thr: float) -> list[tuple[int, int]]:
    pairs, used = [], set()
    for i in np.argsort(-conf):
        if M.shape[1] == 0:
            break
        order = np.argsort(-M[i])
        for j in order:
            if M[i, j] < thr:
                break
            if j not in used:
                pairs.append((int(i), int(j)))
                used.add(j)
                break
    return pairs


def match_hungarian(M: np.ndarray, thr: float, gate: bool = True) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment (maximum total IoU).

    gate=True (default): pairs with IoU < thr are set to 0 BEFORE the assignment, so they
    cannot take part. Without gating the algorithm may give up one good pair (e.g. 0.60)
    for two weak ones (0.42 + 0.40 = 0.82 > 0.60) which are then both thrown away because
    they are below the threshold -> a correct detection is lost (seen in scene 006211).
    """
    if M.size == 0:
        return []
    W = np.where(M >= thr, M, 0.0) if gate else M
    rows, cols = linear_sum_assignment(-W)        # maximise total IoU
    return [(int(i), int(j)) for i, j in zip(rows, cols) if M[i, j] >= thr]


def precision_recall(tp: int, fp: int, fn: int):
    """None where the value is undefined (0/0)."""
    p = tp / (tp + fp) if tp + fp > 0 else None
    r = tp / (tp + fn) if tp + fn > 0 else None
    return p, r
