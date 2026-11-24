"""
Task 2 - Evaluating an object detector (YOLOv8) on KITTI_Selection.

    python main.py                                  # yolov8m, conf 0.25, IoU 0.5, Hungarian
    python main.py --model models/yolov8n.pt --match greedy --iou-thr 0.7 --out results/test

Outputs in --out:
    scenes/<id>.jpg       GT (green / orange = missed) + detections (blue = TP, red = FP) with IoU
    per_detection.csv     every car detection: confidence, IoU, TP/FP, reason for FP
    per_gt.csv            every GT car: found / FN, IoU, distance, occlusion, reason for FN
    per_scene.csv         TP FP FN precision recall mean IoU per scene
    report.md             everything on one page (VS Code: Ctrl+K V)
    detections.json       raw YOLO output (all classes)
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from detector import COCO_CAR, YoloDetector, save_detections  # noqa: E402
from evaluation import evaluate_scene  # noqa: E402
from kitti import load_dataset  # noqa: E402
from matching import precision_recall  # noqa: E402
from visualize import draw_scene  # noqa: E402


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def md_table(rows, keys):
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + " |")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/KITTI_Selection")
    ap.add_argument("--full-labels", default="data/KITTI_full_labels",
                    help="full KITTI labels (only used to explain errors); '' to disable")
    ap.add_argument("--model", default="models/yolov8m.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="IoU needed for a correct detection")
    ap.add_argument("--match", choices=["hungarian", "greedy", "hungarian_raw"], default="hungarian")
    ap.add_argument("--with-trucks", action="store_true", help="also count YOLO 'truck' as car")
    ap.add_argument("--out", default="results/yolov8m")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    scenes = load_dataset(args.data, args.full_labels or None)
    det = YoloDetector(args.model, conf=args.conf)
    classes = (COCO_CAR, 7) if args.with_trucks else (COCO_CAR,)

    results, all_dets = [], {}
    for s in scenes:
        img = cv2.imread(str(s.image_path))
        dets = det(img)
        all_dets[s.id] = dets
        r = evaluate_scene(s, dets, args.iou_thr, args.match, classes)
        results.append(r)
        cv2.imwrite(str(out / "scenes" / f"{s.id}.jpg"), draw_scene(img, r), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{s.id}: GT={len(s.cars):2d} det={len(r.dets):2d}  TP={r.tp:2d} FP={r.fp:2d} FN={r.fn:2d}  "
              f"P={f2(r.precision)} R={f2(r.recall)}  meanIoU={f2(r.mean_iou_tp)}")
    save_detections(out / "detections.json", all_dets)

    scene_rows = [dict(scene=r.scene.id, gt_cars=len(r.scene.cars), detections=len(r.dets), TP=r.tp, FP=r.fp,
                       FN=r.fn, precision=f2(r.precision), recall=f2(r.recall), mean_IoU_TP=f2(r.mean_iou_tp))
                  for r in results]
    det_rows = [row for r in results for row in r.det_rows]
    gt_rows = [row for r in results for row in r.gt_rows]
    TP, FP, FN = (sum(getattr(r, k) for r in results) for k in ("tp", "fp", "fn"))
    P, R = precision_recall(TP, FP, FN)
    Ps = [r.precision for r in results if r.precision is not None]
    Rs = [r.recall for r in results if r.recall is not None]
    tp_ious = [row["iou"] for row in det_rows if row["status"] == "TP"]
    overall = dict(scene="ALL (micro)", gt_cars=sum(len(r.scene.cars) for r in results),
                   detections=sum(len(r.dets) for r in results), TP=TP, FP=FP, FN=FN,
                   precision=f2(P), recall=f2(R), mean_IoU_TP=f2(sum(tp_ious) / len(tp_ious) if tp_ious else None))
    macro = dict(scene="mean over scenes (macro)", precision=f2(sum(Ps) / len(Ps)), recall=f2(sum(Rs) / len(Rs)))

    write_csv(out / "per_scene.csv", scene_rows + [overall])
    write_csv(out / "per_detection.csv", det_rows)
    write_csv(out / "per_gt.csv", gt_rows)

    fp_reasons = Counter(row["reason"].split(" (")[0] for row in det_rows if row["status"] == "FP")
    def fn_cat(row):
        first = row["reason"].split(",")[0]
        if first.startswith("car detection overlaps"):
            first = f"found, but box overlaps only with IoU < {args.iou_thr}"
        occl = {1: "partly occluded", 2: "largely occluded"}.get(row["occlusion"], "not occluded/unknown")
        return f"{first.split(' (')[0]}  [{occl}]"
    fn_reasons = Counter(fn_cat(row) for row in gt_rows if row["status"] == "FN")

    L = [f"# Task 2 - YOLOv8 evaluation on KITTI_Selection", "",
         f"Model `{Path(args.model).name}`, confidence >= {args.conf}, IoU threshold {args.iou_thr}, "
         f"matching **{args.match}**, classes: {'car + truck' if args.with_trucks else 'car only'}", "",
         "Colours: green = GT car found, orange = GT car missed (FN), blue = correct detection (TP), "
         "red = wrong detection (FP), grey dashed = KITTI DontCare region.", "",
         "## Per scene", ""]
    L += md_table(scene_rows + [overall, macro],
                  ["scene", "gt_cars", "detections", "TP", "FP", "FN", "precision", "recall", "mean_IoU_TP"])
    L += ["", "*n/a = undefined (0/0): a scene without GT cars has no recall; a scene without detections has "
          "no precision.*", "", "## Why detections were wrong (FP)", ""]
    L += [f"* {n} x {k}" for k, n in fp_reasons.most_common()]
    L += ["", "## Why GT cars were missed (FN)", ""]
    L += [f"* {n} x {k}" for k, n in fn_reasons.most_common()]
    L += ["", "## Scenes", ""]
    for r in results:
        L += [f"### {r.scene.id}  -  P={f2(r.precision)}  R={f2(r.recall)}", "",
              f"![{r.scene.id}](scenes/{r.scene.id}.jpg)", ""]
        rows = [x for x in r.det_rows] + [x for x in r.gt_rows if x["status"] == "FN"]
        if rows:
            L += md_table([dict(what=f"det {x['det']}" if "conf" in x else f"GT {x['gt']}",
                                status=x["status"], conf=x.get("conf"), iou=x["iou"],
                                gt=x.get("gt") if "conf" in x else x["gt"], reason=x["reason"]) for x in rows],
                          ["what", "status", "conf", "iou", "gt", "reason"])
            L.append("")
    (out / "report.md").write_text("\n".join(L))

    print(f"\nALL: TP={TP} FP={FP} FN={FN}  precision={f2(P)}  recall={f2(R)}  "
          f"(macro P={macro['precision']} R={macro['recall']})")
    print("FP reasons:", dict(fp_reasons))
    print("FN reasons:", dict(fn_reasons))
    print(f"Results written to {out}/  ->  open {out}/report.md with Ctrl+K V")


if __name__ == "__main__":
    main()
