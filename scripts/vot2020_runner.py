"""Run inside the VOT-LT2020 toolkit with its original Python integration helper.

Install/provide the 2020 `vot.py`/TraX helper separately and put it on PYTHONPATH.
Confidence uses incoming-mode ranking scores for emitted boxes and zero
when no box is emitted.
"""
import argparse
import inspect
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dds_mamba.cli import choose_device, load_network
from dds_mamba.controller import Tracker
from dds_mamba.geometry import load_image, xywh_to_center, center_to_xywh
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=["reference", "mamba_ssm"])
    args = parser.parse_args()
    try:
        import vot
    except ImportError as exc:
        raise RuntimeError("VOT-LT2020 integration helper and TraX are required") from exc
    if "confidence" not in inspect.signature(vot.VOT.report).parameters:
        raise RuntimeError("Use the VOT2020 Python helper whose report(region, confidence) API supports LT scores")
    device = choose_device(args.device)
    model = load_network(args.checkpoint, args.assets, device, args.backend)
    handle = vot.VOT("rectangle")
    initial, path = handle.region(), handle.frame()
    if not path:
        return
    tracker = Tracker(model)
    with torch.no_grad():
        tracker.initialize(load_image(path, device), xywh_to_center([initial.x, initial.y, initial.width, initial.height]))
    while path := handle.frame():
        box, info = tracker.update(load_image(path, device))
        if box is None:
            handle.report(vot.Rectangle(0, 0, 0, 0), 0.0)
        else:
            handle.report(vot.Rectangle(*center_to_xywh(box)), info.get("tracking_confidence", 0.0))


if __name__ == "__main__":
    main()
