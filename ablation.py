"""
How much do the evaluation choices change the result?

    python ablation.py            -> results/ablation.md

Compares: YOLOv8 n / s / m, matching (gated Hungarian / greedy / Hungarian without gating),
IoU threshold 0.5 / 0.7, "car" only vs. "car + truck", recall per occlusion level, and a
precision-recall curve over the confidence threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from detector import COCO_CAR, Detection, YoloDetector, load_detections, save_detections  # noqa: E402
from evaluation import evaluate_scene  # noqa: E402
from kitti import load_dataset  # noqa: E402
from matching import precision_recall  # noqa: E402

TRUCK = 7
OUT = Path("results")


def run_all(scenes, dets, thr=0.5, method="hungarian", classes=(COCO_CAR,), conf=0.25):
    rs = [evaluate_scene(s, [d for d in dets[s.id] if d.conf >= conf], thr, method, classes) for s in scenes]
    TP, FP, FN = (sum(getattr(r, k) for r in rs) for k in ("tp", "fp", "fn"))
    P, R = precision_recall(TP, FP, FN)
    return rs, TP, FP, FN, P, R


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


def main():
    scenes = load_dataset("data/KITTI_Selection", "data/KITTI_full_labels")
    dets = {}
    for m in ("yolov8n", "yolov8s", "yolov8m"):
        cache = OUT / f"detections_{m}_conf0.01.json"
        if not cache.exists():
            y = YoloDetector(f"models/{m}.pt", conf=0.01)
            save_detections(cache, {s.id: y(cv2.imread(str(s.image_path))) for s in scenes})
        dets[m] = load_detections(cache)

    L = ["# Task 2 - how the evaluation choices change the result", "",
         "All numbers summed over the 20 scenes (micro average). Default setting in **bold**: "
         "YOLOv8m, confidence >= 0.25, IoU >= 0.5, gated Hungarian matching, class car only.", ""]

    def table(title, rows):
        nonlocal L
        L += [f"## {title}", "", "| setting | TP | FP | FN | precision | recall |", "|---|---|---|---|---|---|"]
        for name, (rs, TP, FP, FN, P, R) in rows:
            L.append(f"| {name} | {TP} | {FP} | {FN} | {f2(P)} | {f2(R)} |")
        L.append("")

    m = dets["yolov8m"]
    table("Model size", [(("**yolov8m**" if k == "yolov8m" else k), run_all(scenes, dets[k])) for k in dets])
    table("Matching strategy", [("**Hungarian (gated)**", run_all(scenes, m)),
                                 ("greedy by confidence", run_all(scenes, m, method="greedy")),
                                 ("Hungarian without gating", run_all(scenes, m, method="hungarian_raw"))])
    table("IoU threshold", [(f"{'**0.5**' if t == 0.5 else t}", run_all(scenes, m, thr=t))
                            for t in (0.3, 0.5, 0.7)])
    table("Classes counted as car", [("**car**", run_all(scenes, m)),
                                      ("car + truck", run_all(scenes, m, classes=(COCO_CAR, TRUCK)))])

    # recall per occlusion level
    rs = run_all(scenes, m)[0]
    gt = [row for r in rs for row in r.gt_rows]
    L += ["## Recall per occlusion level (KITTI: 0 visible, 1 partly, 2 largely occluded, 3 unknown)", "",
          "| occlusion | GT cars | found | recall | found with IoU >= 0.3 |", "|---|---|---|---|---|"]
    for o in (0, 1, 2, 3):
        g = [x for x in gt if x["occlusion"] == o]
        if g:
            f = sum(x["status"] == "found" for x in g)
            f3 = sum(x["iou"] >= 0.3 for x in g)
            L.append(f"| {o} | {len(g)} | {f} | {f / len(g):.2f} | {f3 / len(g):.2f} |")
    L.append("")

    # precision-recall over the confidence threshold
    L += ["## Precision / recall vs. confidence threshold (YOLOv8m)", "",
          "| conf | detections | precision | recall |", "|---|---|---|---|"]
    curve = []
    for c in (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        _, TP, FP, FN, P, R = run_all(scenes, m, conf=c)
        curve.append((c, P, R))
        L.append(f"| {'**0.25**' if c == 0.25 else c} | {TP + FP} | {f2(P)} | {f2(R)} |")
    L.append("")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        for k, col in zip(dets, ("#9aa5b1", "#5b7fa6", "#1f3b63")):
            pts = [run_all(scenes, dets[k], conf=c)[4:] for c in np.linspace(0.02, 0.95, 40)]
            pts = [(r, p) for p, r in pts if p is not None and r is not None]
            ax.plot([x[0] for x in pts], [x[1] for x in pts], label=k, color=col, lw=2)
        _, _, _, _, P, R = run_all(scenes, m)
        ax.scatter([R], [P], color="#c0392b", zorder=5, label="yolov8m @ conf 0.25")
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3); ax.legend(loc="lower left", frameon=False); ax.set_title("Precision-recall (IoU >= 0.5)")
        fig.tight_layout(); fig.savefig(OUT / "pr_curve.png", dpi=150)
        L += ["![precision-recall curve](pr_curve.png)", ""]
    except ImportError:
        pass

    (OUT / "ablation.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
