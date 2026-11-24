"""
Loading the KITTI_Selection dataset.

KITTI_Selection/
    images/<id>.png    camera image (left colour camera, ~1242 x 375 px)
    labels/<id>.txt    ground truth, one car per line:  type xmin ymin xmax ymax gt_distance
    calib/<id>.txt     3x3 intrinsic camera matrix K (needed in Task 3)

Optional: the full KITTI labels (from "KITTI-Selection_incl_LiDAR") contain the same cars
plus Van / Truck / Pedestrian / DontCare boxes and truncation + occlusion values. They are
NOT used for the evaluation itself - only to explain why something went wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GTObject:
    cls: str
    box: np.ndarray                 # xmin ymin xmax ymax (pixels)
    distance: float | None = None   # metres (KITTI_Selection gt_distance)
    truncation: float | None = None  # 0 = fully visible ... 1 = fully outside (full labels)
    occlusion: int | None = None     # 0 visible, 1 partly, 2 largely occluded, 3 unknown


@dataclass
class Scene:
    id: str
    image_path: Path
    K: np.ndarray
    cars: list[GTObject]                               # evaluation ground truth
    others: list[GTObject] = field(default_factory=list)  # Van/Truck/DontCare/... (analysis only)


def _read_K(path: Path) -> np.ndarray:
    return np.loadtxt(path).reshape(3, 3)


def _read_selection_labels(path: Path) -> list[GTObject]:
    objs = []
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        objs.append(GTObject(p[0], np.array(list(map(float, p[1:5]))), float(p[5]) if len(p) > 5 else None))
    return objs


def _read_full_labels(path: Path) -> list[GTObject]:
    """Official KITTI label format (15 values per line)."""
    objs = []
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        objs.append(GTObject(p[0], np.array(list(map(float, p[4:8]))), None,
                             float(p[1]), int(p[2])))
    return objs


def load_dataset(root: str | Path, full_labels: str | Path | None = None) -> list[Scene]:
    root = Path(root)
    scenes = []
    for img in sorted((root / "images").glob("*.png")):
        sid = img.stem
        cars = _read_selection_labels(root / "labels" / f"{sid}.txt")
        others = []
        if full_labels is not None and (Path(full_labels) / f"{sid}.txt").exists():
            full = _read_full_labels(Path(full_labels) / f"{sid}.txt")
            # attach truncation / occlusion to the evaluation cars (same boxes in both files)
            for c in cars:
                for f in full:
                    if f.cls == "Car" and np.allclose(f.box, c.box, atol=0.05):
                        c.truncation, c.occlusion = f.truncation, f.occlusion
                        break
            others = [f for f in full if f.cls != "Car"]
        scenes.append(Scene(sid, img, _read_K(root / "calib" / f"{sid}.txt"), cars, others))
    return scenes
