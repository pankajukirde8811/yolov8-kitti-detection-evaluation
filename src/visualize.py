"""Drawing GT and detections with their match status."""
from __future__ import annotations

import cv2
import numpy as np

from evaluation import SceneResult

GREEN = (0, 200, 0)      # GT car, found
ORANGE = (0, 140, 255)   # GT car, missed (FN)
BLUE = (255, 120, 0)     # detection, correct (TP)
RED = (0, 0, 255)        # detection, wrong (FP)
GREY = (160, 160, 160)   # DontCare region (full labels)


def _dashed_rect(img, p1, p2, color, thick=1, dash=6):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, 2 * dash):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thick)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thick)
    for y in range(y1, y2, 2 * dash):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thick)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thick)


def _label(img, text, org, color):
    x, y = org
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    y = max(y, h + 3)
    cv2.rectangle(img, (x, y - h - 3), (x + w + 2, y + 1), color, -1)
    cv2.putText(img, text, (x + 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def draw_scene(img: np.ndarray, r: SceneResult, show_dontcare: bool = True) -> np.ndarray:
    vis = img.copy()
    if show_dontcare:
        for o in r.scene.others:
            if o.cls == "DontCare":
                b = o.box.astype(int)
                _dashed_rect(vis, tuple(b[:2]), tuple(b[2:]), GREY)
    matched_gt = {j for _, j in r.pairs}
    for j, g in enumerate(r.scene.cars):
        b = g.box.astype(int)
        col = GREEN if j in matched_gt else ORANGE
        cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), col, 2 if j in matched_gt else 3)
        if j not in matched_gt:
            _label(vis, f"FN gt{j}", (b[0], b[3] + 14), ORANGE)
    det2gt = dict(r.pairs)
    for i, d in enumerate(r.dets):
        b = d.box.astype(int)
        if i in det2gt:
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), BLUE, 1)
            _label(vis, f"IoU {r.M[i, det2gt[i]]:.2f}", (b[0], b[1] - 2), BLUE)
        else:
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), RED, 2)
            best = r.M[i].max() if r.M.shape[1] else 0
            _label(vis, f"FP {best:.2f}", (b[0], b[1] - 2), RED)

    p = "n/a" if r.precision is None else f"{r.precision:.2f}"
    rc = "n/a" if r.recall is None else f"{r.recall:.2f}"
    head = f"{r.scene.id}  GT={len(r.scene.cars)}  det={len(r.dets)}  TP={r.tp} FP={r.fp} FN={r.fn}  P={p} R={rc}"
    bar = np.full((28, vis.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, head, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    legend = np.full((24, vis.shape[1], 3), 245, np.uint8)
    x = 8
    for col, txt in ((GREEN, "GT found"), (ORANGE, "GT missed (FN)"), (BLUE, "detection TP"),
                     (RED, "detection FP"), (GREY, "DontCare region")):
        cv2.rectangle(legend, (x, 6), (x + 16, 18), col, -1)
        cv2.putText(legend, txt, (x + 22, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        x += 40 + 9 * len(txt)
    return np.vstack([bar, vis, legend])
