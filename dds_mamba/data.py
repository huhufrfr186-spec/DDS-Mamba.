"""Strict image-directory adapters and materialized sequence manifests.

Inference loads only initialization labels. Dense annotations are read only by
training or offline evaluation. Sparse VTUAV annotations are never interpolated.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import numpy as np


def natural_key(path):
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", str(path))]


def frames(directory):
    return sorted([p for p in Path(directory).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")], key=natural_key)


def numbers(path, columns=None):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"empty annotation: {path}")
    values = np.asarray([float(x) for x in re.split(r"[,\s]+", text)], dtype=np.float64)
    return values.reshape(-1, columns) if columns else values


def names(path):
    path = Path(path)
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value) or len(set(value)) != len(value):
        raise ValueError("split must contain unique sequence identifiers")
    return value


def lasot_split(training_names, test_names=(), strategy="sha256", validation_names=None):
    ordered = sorted(training_names)
    if len(ordered) != 1120 or len(set(ordered)) != 1120:
        raise ValueError("LaSOT Protocol II official training list must contain 1120 unique IDs")
    if set(ordered) & set(test_names):
        raise ValueError("train/test sequence leakage")
    if validation_names is not None:
        holdout = set(validation_names)
        if len(holdout) != 224 or not holdout <= set(ordered):
            raise ValueError("validation list must contain 224 official training IDs")
    elif strategy == "sha256":
        salt = "dds-mamba-v1-lasot-trainval-20260712"
        ranked = sorted(ordered, key=lambda n: hashlib.sha256(f"{salt}:{n}".encode()).hexdigest())
        holdout = set(ranked[:224])
    elif strategy == "ordered":
        holdout = set(ordered[-224:])
    else:
        raise ValueError(strategy)
    return [n for n in ordered if n not in holdout], [n for n in ordered if n in holdout]


@dataclass
class Sequence:
    name: str
    frames: list
    initial_xywh: np.ndarray
    record: dict
    root: Path

    def labels(self):
        source = self.root / self.record["annotation"]
        if source.suffix == ".json":
            info = json.loads(source.read_text(encoding="utf-8"))
            raw = info.get("gt_rect", info.get("bbox", info.get("rect")))
            existence = info.get("exist", info.get("visible", info.get("presence")))
            if raw is None or existence is None:
                raise ValueError(f"{source}: missing box/existence arrays")
            boxes = np.asarray(raw, dtype=np.float64).reshape(-1, 4)
            valid = np.asarray(existence, dtype=bool).reshape(-1)
        else:
            boxes = numbers(source, 4)
            valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
        for flag in self.record.get("invisible_flags", []):
            invisible = numbers(self.root / flag).astype(bool)
            if len(invisible) != len(boxes):
                raise ValueError(f"{self.name}: visibility length mismatch")
            valid &= ~invisible
        if len(boxes) != len(self.frames) or len(valid) != len(boxes):
            raise ValueError(f"{self.name}: {len(boxes)} annotations for {len(self.frames)} images. Sparse GT must be evaluated with its official frame indices; do not use dense diagnostics.")
        return boxes, valid


def load_manifest(path, root_override=None):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != 1 or not value.get("sequences"):
        raise ValueError("manifest must be schema 1 and contain sequences")
    root = Path(root_override or value["root"]).resolve()
    ids, sequences = set(), []
    for record in value["sequences"]:
        name = record["name"]
        if name in ids or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"duplicate/unsafe output sequence ID: {name}")
        ids.add(name)
        paths = frames(root / record["frame_dir"])
        if not paths:
            raise FileNotFoundError(f"{name}: no RGB images")
        init = np.asarray(record["initial_xywh"], dtype=np.float64)
        if init.shape != (4,) or not np.isfinite(init).all() or not (init[2:] > 0).all():
            raise ValueError(f"{name}: invalid first-frame initialization")
        if record.get("frame_count") != len(paths):
            raise ValueError(f"{name}: manifest/image count mismatch")
        sequences.append(Sequence(name, paths, init, record, root))
    return sequences, value


def _record(root, directory, frame_dir, annotation, invisible=()):
    image_paths = frames(frame_dir)
    if not image_paths:
        raise FileNotFoundError(f"no image frames in {frame_dir}")
    if annotation.suffix == ".json":
        info = json.loads(annotation.read_text(encoding="utf-8"))
        raw = info.get("gt_rect", info.get("bbox", info.get("rect")))
        if raw is None:
            raise ValueError(f"missing boxes in {annotation}")
        initial = raw[0]
    else:
        initial = numbers(annotation, 4)[0].tolist()
    return {"name": directory.name, "frame_dir": frame_dir.relative_to(root).as_posix(),
            "annotation": annotation.relative_to(root).as_posix(), "initial_xywh": initial,
            "invisible_flags": [p.relative_to(root).as_posix() for p in invisible], "frame_count": len(image_paths)}


def prepare(root, benchmark, split, selected_names=None):
    root = Path(root).resolve()
    records = []
    if benchmark == "lasot":
        if selected_names is None:
            list_path = root / ("training_set.txt" if split == "train" else "testing_set.txt")
            selected_names = names(list_path)
        for name in sorted(selected_names):
            category = name.rsplit("-", 1)[0]
            choices = [root / category / name, root / name]
            directory = next((p for p in choices if (p / "groundtruth.txt").is_file()), None)
            if directory is None:
                raise FileNotFoundError(f"LaSOT sequence {name} missing under {root}")
            invisible = [directory / f for f in ("full_occlusion.txt", "out_of_view.txt") if (directory / f).exists()]
            records.append(_record(root, directory, directory / "img", directory / "groundtruth.txt", invisible))
    else:
        base = root / split
        if not base.is_dir():
            raise FileNotFoundError(base)
        directories = [p for p in sorted(base.iterdir()) if p.is_dir()]
        if selected_names is not None:
            selected = set(selected_names)
            directories = [p for p in directories if p.name in selected]
            missing = selected - {p.name for p in directories}
            if missing:
                raise FileNotFoundError(f"unresolved split IDs: {sorted(missing)[:5]}")
        for directory in directories:
            if benchmark == "anti-uav300":
                annotation = next((directory / n for n in ("RGB_label.json", "label.json") if (directory / n).exists()), None)
                image_dir = next((directory / n for n in ("RGB", "visible", "rgb", "") if (directory / n).is_dir() and frames(directory / n)), None)
                invisible = []
            elif benchmark == "webuav":
                annotation, image_dir = directory / "groundtruth_rect.txt", directory / "img"
                invisible = [directory / "absent.txt"]
            elif benchmark == "vtuav":
                annotation = next((directory / n for n in ("rgb.txt", "groundtruth.txt", "groundtruth_rect.txt", "init.txt") if (directory / n).exists()), None)
                image_dir = next((directory / n for n in ("rgb", "visible", "img") if (directory / n).is_dir() and frames(directory / n)), None)
                invisible = []
            else:
                raise ValueError(benchmark)
            if annotation is None or image_dir is None or not annotation.exists():
                raise FileNotFoundError(f"{directory.name}: expected {benchmark} RGB images and annotations; decode videos to image folders first")
            if any(not p.exists() for p in invisible):
                raise FileNotFoundError(f"{directory.name}: missing absence flags")
            records.append(_record(root, directory, image_dir, annotation, invisible))
    if not records:
        raise ValueError("empty dataset selection")
    if benchmark == "lasot" and split == "test" and len(records) != 280:
        raise ValueError("official LaSOT Protocol II test requires exactly 280 sequences")
    if benchmark == "webuav" and split.lower() == "test" and len(records) != 780:
        raise ValueError("official WebUAV RGB Test requires exactly 780 sequences")
    return {"schema": 1, "benchmark": benchmark, "split": split, "root": str(root), "sequences": records}
