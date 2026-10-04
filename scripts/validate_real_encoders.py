"""Validate official frozen asset loading and full-sized neural forward/backward.

Uses generated RGB frames to check asset loading, gradients and parameter updates.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dds_mamba.config import Config
from dds_mamba.encoders import Encoders
from dds_mamba.model import Network
from dds_mamba.controller import Tracker
from dds_mamba.losses import objective
from dds_mamba.cli import choose_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="assets")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/encoder_validation.json")
    args = parser.parse_args()
    torch.manual_seed(2024)
    device = choose_device(args.device)
    net = Network(Config(), Encoders(args.assets)).to(device).train()
    image = torch.rand(1, 3, 320, 480, device=device)
    tracker = Tracker(net)
    tracker.initialize(image, [240, 160, 60, 40])
    candidates, _ = tracker.evaluate(image)
    candidate = candidates[0]
    gt = image.new_tensor(candidate.crop.to_crop([240, 160, 60, 40]))[None]
    identity = net.encoders.identity(image, [240, 160, 60, 40])
    loss, _ = objective(candidate.output, gt, identity, True, tracker.state.appearance, False, net)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None for p in net.encoders.parameters()):
        raise AssertionError("nonfinite loss or unfrozen encoders")
    result = {"real_pretrained_encoders": True, "synthetic_frame": True, "real_benchmark_verified": False,
              "loss": float(loss.detach()), "assets": net.encoders.provenance(),
              "total_parameters": sum(p.numel() for p in net.parameters()),
              "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
              "branch_parameters": sum(p.numel() for p in net.position.parameters()) + sum(p.numel() for p in net.appearance.parameters()),
              "device": str(device), "torch": torch.__version__}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
