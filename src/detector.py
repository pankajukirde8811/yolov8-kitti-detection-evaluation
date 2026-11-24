"""
YOLOv8 (ultralytics) wrapper.

YOLO is trained on COCO (80 classes). We keep only class 2 = "car" for the evaluation.
All other detections are kept as well, so that the analysis can show e.g. cars that YOLO
called "truck".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COCO_CAR = 2


@dataclass
class Detection:
    box: np.ndarray      # xmin ymin xmax ymax
    conf: float
    cls_id: int
    cls_name: str


class YoloDetector:
    def __init__(self, weights: str = "models/yolov8m.pt", conf: float = 0.25, device: str | None = None):
        from ultralytics import YOLO          # imported here so the rest works without ultralytics
        w = Path(weights)
        if not w.exists():
            # first run: ultralytics downloads e.g. "yolov8m.pt" into the current folder -> move it
            YOLO(w.name)
            w.parent.mkdir(parents=True, exist_ok=True)
            if Path(w.name).exists() and Path(w.name).resolve() != w.resolve():
                Path(w.name).replace(w)
        self.model = YOLO(str(w))
        self.weights = Path(weights).name
        self.conf = conf
        self.device = device

    def __call__(self, image_bgr: np.ndarray) -> list[Detection]:
        r = self.model.predict(image_bgr, conf=self.conf, verbose=False, device=self.device)[0]
        out = []
        for box, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            out.append(Detection(box.astype(float), float(c), int(k), r.names[int(k)]))
        return out


def save_detections(path: Path, dets: dict[str, list[Detection]]):
    data = {sid: [dict(box=d.box.round(2).tolist(), conf=round(d.conf, 4), cls_id=d.cls_id, cls_name=d.cls_name)
                  for d in ds] for sid, ds in dets.items()}
    Path(path).write_text(json.dumps(data, indent=1))


def load_detections(path: Path) -> dict[str, list[Detection]]:
    data = json.loads(Path(path).read_text())
    return {sid: [Detection(np.array(d["box"]), d["conf"], d["cls_id"], d["cls_name"]) for d in ds]
            for sid, ds in data.items()}
