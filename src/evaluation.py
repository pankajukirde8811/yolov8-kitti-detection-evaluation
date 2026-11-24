"""
Per-scene evaluation: IoU for every box, TP / FP / FN, precision / recall, and a reason for
every error (using the full KITTI labels where available).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from detector import COCO_CAR, Detection
from kitti import Scene
from matching import iou, iou_matrix, match_greedy, match_hungarian, precision_recall

SMALL_PX = 25   # KITTI: boxes lower than 25 px are not part of the official "easy/moderate" evaluation


@dataclass
class SceneResult:
    scene: Scene
    dets: list[Detection]            # car detections used for evaluation
    all_dets: list[Detection]        # every YOLO detection (all classes)
    M: np.ndarray                    # IoU matrix, rows = dets, cols = GT cars
    pairs: list[tuple[int, int]]     # (det index, gt index)
    det_rows: list[dict] = field(default_factory=list)
    gt_rows: list[dict] = field(default_factory=list)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float | None = None
    recall: float | None = None
    mean_iou_tp: float | None = None


def _inside_fraction(box, region):
    """Fraction of `box` that lies inside `region`."""
    ix = max(0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0, min(box[3], region[3]) - max(box[1], region[1]))
    a = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / a if a > 0 else 0.0


def explain_fp(det: Detection, scene: Scene, M_row: np.ndarray, matched_gt: set[int], thr: float) -> str:
    if len(M_row):
        j = int(np.argmax(M_row))
        if M_row[j] >= thr and j in matched_gt:
            return f"duplicate (GT car {j} already matched)"
        if M_row[j] >= 0.1:
            return f"poor localisation (IoU {M_row[j]:.2f} with GT car {j})"
    for o in scene.others:
        if o.cls in ("Van", "Truck") and iou(det.box, o.box) >= 0.3:
            return f"{o.cls} in full KITTI labels (not a 'Car')"
    for o in scene.others:
        if o.cls == "DontCare" and _inside_fraction(det.box, o.box) >= 0.5:
            return "inside KITTI 'DontCare' region (unlabelled far/hidden vehicle)"
    h = det.box[3] - det.box[1]
    if h < SMALL_PX:
        return f"small, unlabelled object ({h:.0f} px high) - possibly a real car missing in GT"
    return "no GT object here - unlabelled car or false alarm (check image)"


def explain_fn(j: int, scene: Scene, M_col: np.ndarray, all_dets: list[Detection], thr: float) -> str:
    g = scene.cars[j]
    reasons = []
    if len(M_col) and M_col.max() >= 0.1:
        reasons.append(f"car detection overlaps only with IoU {M_col.max():.2f} < {thr}")
    else:
        other = [(iou(d.box, g.box), d) for d in all_dets if d.cls_id != COCO_CAR]
        other = [x for x in other if x[0] >= thr]
        if other:
            best = max(other, key=lambda x: x[0])
            reasons.append(f"detected as '{best[1].cls_name}' (IoU {best[0]:.2f})")
        else:
            reasons.append("not detected")
    if g.occlusion in (1, 2):
        reasons.append("partly occluded" if g.occlusion == 1 else "largely occluded")
    if g.truncation is not None and g.truncation > 0.15:
        reasons.append(f"truncated at image border ({g.truncation:.0%})")
    h = g.box[3] - g.box[1]
    if h < SMALL_PX:
        reasons.append(f"small ({h:.0f} px high)")
    if g.distance is not None and g.distance > 40:
        reasons.append(f"far away ({g.distance:.0f} m)")
    return ", ".join(reasons)


def evaluate_scene(scene: Scene, all_dets: list[Detection], iou_thr: float = 0.5,
                   method: str = "hungarian", classes: tuple[int, ...] = (COCO_CAR,)) -> SceneResult:
    dets = [d for d in all_dets if d.cls_id in classes]
    D = np.array([d.box for d in dets]).reshape(-1, 4)
    G = np.array([g.box for g in scene.cars]).reshape(-1, 4)
    M = iou_matrix(D, G)
    conf = np.array([d.conf for d in dets])
    if method == "hungarian":
        pairs = match_hungarian(M, iou_thr)
    elif method == "hungarian_raw":
        pairs = match_hungarian(M, iou_thr, gate=False)
    else:
        pairs = match_greedy(M, conf, iou_thr)

    r = SceneResult(scene, dets, all_dets, M, pairs)
    det2gt = {i: j for i, j in pairs}
    gt2det = {j: i for i, j in pairs}
    matched_gt = set(gt2det)

    for i, d in enumerate(dets):
        best_j = int(np.argmax(M[i])) if M.shape[1] else None
        best = float(M[i, best_j]) if best_j is not None else 0.0
        if i in det2gt:
            j = det2gt[i]
            r.det_rows.append(dict(scene=scene.id, det=i, cls=d.cls_name, conf=round(d.conf, 3),
                                   status="TP", gt=j, iou=round(float(M[i, j]), 3),
                                   gt_distance_m=round(scene.cars[j].distance, 1) if scene.cars[j].distance else None,
                                   reason=""))
        else:
            r.det_rows.append(dict(scene=scene.id, det=i, cls=d.cls_name, conf=round(d.conf, 3),
                                   status="FP", gt=best_j if best > 0 else None, iou=round(best, 3),
                                   gt_distance_m=None,
                                   reason=explain_fp(d, scene, M[i], matched_gt, iou_thr)))
    for j, g in enumerate(scene.cars):
        h = g.box[3] - g.box[1]
        row = dict(scene=scene.id, gt=j, distance_m=round(g.distance, 1) if g.distance else None,
                   height_px=round(float(h), 1), occlusion=g.occlusion, truncation=g.truncation)
        if j in gt2det:
            row.update(status="found", det=gt2det[j], iou=round(float(M[gt2det[j], j]), 3), reason="")
        else:
            row.update(status="FN", det=None, iou=round(float(M[:, j].max()), 3) if M.shape[0] else 0.0,
                       reason=explain_fn(j, scene, M[:, j] if M.shape[0] else np.zeros(0), all_dets, iou_thr))
        r.gt_rows.append(row)

    r.tp = len(pairs)
    r.fp = len(dets) - r.tp
    r.fn = len(scene.cars) - r.tp
    r.precision, r.recall = precision_recall(r.tp, r.fp, r.fn)
    if pairs:
        r.mean_iou_tp = float(np.mean([M[i, j] for i, j in pairs]))
    return r
