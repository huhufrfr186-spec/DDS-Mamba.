"""Offline end-to-end verification, clearly marked as synthetic."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from .config import Config
from .encoders import TinyEncoders
from .model import Network
from .data import prepare, load_manifest
from .training import fit, seed_everything


def run(output, device):
    from .cli import save_json, prediction
    torch.set_num_threads(2)
    seed_everything(2024)
    destination = Path(output).resolve()
    root = destination / "synthetic_data"
    for name in ("toy-1", "toy-2"):
        directory = root / "toy" / name
        (directory / "img").mkdir(parents=True, exist_ok=True)
        boxes = []
        for t in range(6):
            im = Image.new("RGB", (192, 128), (25, 35, 45))
            x, y, w, h = 65 + t, 50, 16, 12
            ImageDraw.Draw(im).rectangle([x, y, x + w, y + h], fill=(200, 60, 40))
            im.save(directory / "img" / f"{t + 1:08d}.png")
            boxes.append([x, y, w, h])
        np.savetxt(directory / "groundtruth.txt", boxes, delimiter=",")
    for name, ids in (("train", ["toy-1"]), ("dev", ["toy-2"])):
        save_json(destination / (name + ".json"), prepare(root, "lasot", name, ids))
    train, _ = load_manifest(destination / "train.json")
    dev, _ = load_manifest(destination / "dev.json")
    cfg = Config(d_model=32, d_state=4, dt_rank=2, epochs=1, warmup_epochs=0, clip_length=3,
                 clips_per_epoch=1, checkpoint_branches=False, jitter_translation=0)
    model = Network(cfg, TinyEncoders()).to(device)
    before = {k: v.clone() for k, v in model.tracker_state_dict().items()}
    history = fit(model, train, dev, device, destination / "training", destination / "train.json", destination / "dev.json", max_steps=1)
    changed = sum(not torch.equal(before[k], v) for k, v in model.tracker_state_dict().items())
    if changed == 0:
        raise AssertionError("smoke optimizer did not change any tracker parameter")
    reloaded = Network(cfg, TinyEncoders()).to(device)
    state = torch.load(destination / "training" / "last.pt", map_location=device, weights_only=False)
    reloaded.load_tracker_state_dict(state["model"])
    if any(not torch.equal(reloaded.tracker_state_dict()[k], v) for k, v in model.tracker_state_dict().items()):
        raise AssertionError("checkpoint round trip changed tracker parameters")
    prediction(model.eval(), dev, device, destination / "predictions", warmup=0)
    boxes = np.loadtxt(destination / "predictions" / "toy-2.txt", delimiter=",")
    if boxes.shape != (6, 4) or not np.isfinite(boxes).all():
        raise AssertionError("bad smoke predictions")
    report = {"synthetic": True, "real_benchmark_accuracy_verified": False, "training_backward_optimizer": True,
              "checkpoint_round_trip": True, "prediction_writer": True, "changed_parameter_tensors": changed,
              "final_loss": history[-1]["loss"], "device": str(device), "torch": torch.__version__}
    save_json(destination / "smoke_report.json", report)
    return report
