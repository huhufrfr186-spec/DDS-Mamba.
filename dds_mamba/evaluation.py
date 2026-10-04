"""Raw diagnostics are deliberately separate from official benchmark evaluators."""
from pathlib import Path
import csv
import json
import numpy as np
from .geometry import xywh_to_center, iou


def diagnostics(sequences, prediction_dir):
    records = []
    for sequence in sequences:
        predictions = np.loadtxt(Path(prediction_dir) / (sequence.name + ".txt"), delimiter=",").reshape(-1, 4)
        truth, visible = sequence.labels()
        if len(predictions) != len(truth):
            raise ValueError(f"{sequence.name}: prediction/annotation count mismatch")
        overlaps, errors, normalized, absent_correct = [], [], [], []
        # Initialization is excluded from diagnostics. Official evaluators may include it.
        for pred, gt, shown in zip(predictions[1:], truth[1:], visible[1:]):
            present = bool(np.isfinite(pred).all() and (pred[2:] > 0).all())
            overlaps.append(iou(xywh_to_center(pred), xywh_to_center(gt)) if present else 0.0)
            if present and (gt[2:] > 0).all():
                delta = xywh_to_center(pred)[:2] - xywh_to_center(gt)[:2]
                errors.append(float(np.linalg.norm(delta)))
                normalized.append(float(np.linalg.norm(delta / gt[2:])))
            else:
                errors.append(float("inf"))
                normalized.append(float("inf"))
            if not shown:
                absent_correct.append(float(not present))
        if not overlaps:
            raise ValueError("sequence must contain at least one update")
        npre = np.mean([np.mean(np.asarray(normalized) <= threshold) for threshold in np.linspace(0, 0.5, 51)])
        records.append({"sequence": sequence.name, "raw_integral_auc": float(np.mean(overlaps)),
                        "raw_precision20": float(np.mean(np.asarray(errors) <= 20)), "raw_normalized_precision": float(npre),
                        "absence_accuracy": float(np.mean(absent_correct)) if absent_correct else None})
    return {"official": False, "protocol": "raw_outputs_no_invalid_box_replacement_initialization_excluded",
            "sequence_count": len(records), "raw_integral_auc": float(np.mean([r["raw_integral_auc"] for r in records])), "sequences": records}


def bootstrap(paths, metric, paired_paths=None, resamples=10000, seed=2024):
    def read(path):
        with open(path, newline="", encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
        result = {}
        for row in rows:
            key = row["sequence"]
            if key in result:
                raise ValueError("duplicate sequence in evaluator CSV")
            result[key] = float(row[metric])
        if not result or not np.isfinite(list(result.values())).all():
            raise ValueError("empty/nonfinite official sequence scores")
        return result
    values = [read(path) for path in paths]
    ids = sorted(values[0])
    if any(set(value) != set(ids) for value in values):
        raise ValueError("all seeds need the same sequence identifiers")
    a = np.asarray([[value[key] for key in ids] for value in values]).mean(0)
    if paired_paths:
        if len(paired_paths) != len(paths):
            raise ValueError("paired comparison needs the same seed count")
        other = [read(path) for path in paired_paths]
        if any(set(value) != set(ids) for value in other):
            raise ValueError("paired comparison sequence identifiers differ")
        a -= np.asarray([[value[key] for key in ids] for value in other]).mean(0)
    rng = np.random.default_rng(seed)
    estimates = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(resamples)]
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {"metric": metric, "mean": float(a.mean()), "ci95": [float(lower), float(upper)],
            "resamples": resamples, "seed": seed, "sequences": len(ids), "training_seeds": len(paths),
            "paired": bool(paired_paths), "scope": "sequence bootstrap conditional on supplied training runs"}


def export_predictions(sequences, prediction_dir, output_dir, format_name):
    """Format conversion only; it does not run or emulate official metrics."""
    output = Path(output_dir) / "DDS-Mamba_tracking_result" if format_name == "lasot-txt" else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        boxes = np.loadtxt(Path(prediction_dir) / (sequence.name + ".txt"), delimiter=",").reshape(-1, 4)
        if len(boxes) != len(sequence.frames):
            raise ValueError("prediction length mismatch")
        if format_name == "anti-uav-json":
            value = {"res": boxes.tolist()}
            (output / (sequence.name + ".json")).write_text(json.dumps(value), encoding="utf-8")
        elif format_name == "mat-struct":
            try:
                from scipy.io import savemat
            except ImportError as exc:
                raise RuntimeError("MAT export needs scipy: pip install scipy==1.12.0") from exc
            # Optional legacy interchange, not the current LaSOT official text format.
            result = {"res": boxes, "type": "rect", "len": len(boxes), "startFrame": 1, "endFrame": len(boxes)}
            savemat(output / (sequence.name + "_DDS-Mamba.mat"), {"results": np.array([[result]], dtype=object)})
        else:
            np.savetxt(output / (sequence.name + ".txt"), boxes, delimiter="," if format_name == "csv" else "\t", fmt="%.6f")
    return {"format": format_name, "sequences": len(sequences), "official_evaluation_executed": False}
