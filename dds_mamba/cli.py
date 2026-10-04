import argparse
from dataclasses import asdict
from pathlib import Path
import json
import platform
import time
import numpy as np
import torch
from .config import Config
from .encoders import Encoders, TinyEncoders, download_assets, sha256
from .model import Network
from .controller import Tracker
from .data import names, prepare, lasot_split, load_manifest
from .geometry import load_image, xywh_to_center, center_to_xywh
from .training import fit, seed_everything
from .evaluation import diagnostics, bootstrap, export_predictions


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def choose_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def load_network(checkpoint, assets, device, backend=None):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "dds-mamba-v1":
        raise ValueError("unsupported checkpoint format; use a DDS-Mamba training checkpoint")
    if state.get("synthetic"):
        raise ValueError("synthetic smoke checkpoints cannot be used for real prediction")
    cfg = Config(**state["config"])
    if backend is not None:
        cfg.backend = backend
    encoders = Encoders(assets)
    if encoders.provenance() != state["assets"]:
        raise ValueError("pretrained encoder hashes do not match the training run")
    model = Network(cfg, encoders).to(device)
    model.load_tracker_state_dict(state["model"])
    return model.eval()


def prediction(model, sequences, device, output, warmup=100, checkpoint=None, manifest=None):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    timing = {mode: {"count": 0, "seconds": 0.0} for mode in ("active", "lost")}
    update_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for sequence in sequences:
        image = load_image(sequence.frames[0], device)
        tracker = Tracker(model)
        with torch.no_grad():
            tracker.initialize(image, xywh_to_center(sequence.initial_xywh))
        height, width = image.shape[-2:]
        rows = [sequence.initial_xywh.tolist()]
        with (destination / (sequence.name + ".trace.jsonl")).open("w", encoding="utf-8") as trace:
            for path in sequence.frames[1:]:
                image = load_image(path, device)  # file read, decode and H2D precede timing.
                if image.shape[-2:] != (height, width):
                    raise ValueError(f"{sequence.name}: resolution changed within sequence")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                began = time.perf_counter()
                box, info = tracker.update(image)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - began
                if update_count >= warmup:
                    timing[info["incoming_mode"]]["count"] += 1
                    timing[info["incoming_mode"]]["seconds"] += elapsed
                update_count += 1
                rows.append([0.0] * 4 if box is None else center_to_xywh(box).tolist())
                trace.write(json.dumps(info) + "\n")
        np.savetxt(destination / (sequence.name + ".txt"), rows, delimiter=",", fmt="%.6f")
        print(f"{sequence.name}: wrote {len(rows)} predictions", flush=True)
    for value in timing.values():
        value["fps"] = value["count"] / value["seconds"] if value["seconds"] > 0 else None
    count = sum(v["count"] for v in timing.values())
    elapsed = sum(v["seconds"] for v in timing.values())
    provenance = {"checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                  "manifest_sha256": sha256(manifest) if manifest else None,
                  "config": asdict(model.cfg), "assets": model.encoders.provenance(), "synthetic": model.encoders.synthetic,
                  "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "timing": timing, "timed_updates": count, "overall_fps": count / elapsed if elapsed > 0 else None,
                  "warmup_updates": warmup, "file_decode_h2d_excluded": True,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "official_benchmark_verified": False, "label_access": "first-frame initialization only"}
    save_json(destination / "run.json", provenance)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description="DDS-Mamba training, prediction and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="download official MAE/DINOv2 pretrained encoders")
    p.add_argument("--assets", default="assets")
    p = sub.add_parser("doctor", help="inspect environment and project assets")
    p.add_argument("--assets", default="assets")
    p = sub.add_parser("split", help="materialize official LaSOT train/dev/test manifests")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="data_manifests")
    p.add_argument("--strategy", choices=["sha256", "ordered"], default="sha256")
    p.add_argument("--validation-list")
    p = sub.add_parser("prepare", help="prepare a benchmark image-folder manifest")
    p.add_argument("--root", required=True)
    p.add_argument("--benchmark", choices=["lasot", "anti-uav300", "webuav", "vtuav"], required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--list", dest="sequence_list")
    p.add_argument("--out", required=True)
    p = sub.add_parser("train", help="train full controller-driven 16-frame clips")
    p.add_argument("--config", default="configs/train.json")
    p.add_argument("--assets", default="assets")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--dev-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--backend", choices=["reference", "mamba_ssm"])
    p.add_argument("--device", default="auto")
    p.add_argument("--resume")
    p.add_argument("--max-steps", type=int, help="limit training clips per epoch")
    p.add_argument("--validate-every", type=int, default=1)
    p = sub.add_parser("predict", help="RGB sequence inference and incoming-mode runtime measurement")
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", help="optional relocation of the manifest dataset root")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--assets", default="assets")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=["reference", "mamba_ssm"])
    p.add_argument("--warmup", type=int, default=100)
    p = sub.add_parser("diagnostic-evaluate", help="compute raw prediction diagnostics")
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("export", help="convert raw predictions for official evaluator import")
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["lasot-txt", "mat-struct", "anti-uav-json", "csv", "vtuav-txt"], required=True)
    p = sub.add_parser("bootstrap", help="paired sequence bootstrap from evaluator CSV exports")
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--other", action="append")
    p.add_argument("--metric", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--resamples", type=int, default=10000)
    p = sub.add_parser("smoke", help="synthetic end-to-end test with explicitly fake encoders")
    p.add_argument("--out", default="runs/smoke")
    p.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.command == "download":
        print(json.dumps(download_assets(args.assets), indent=2))
    elif args.command == "doctor":
        print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                          "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          "assets": {p.name: p.stat().st_size for p in Path(args.assets).glob("*.pth")}}, indent=2))
    elif args.command == "split":
        root = Path(args.root)
        training, testing = names(root / "training_set.txt"), names(root / "testing_set.txt")
        if len(testing) != 280:
            raise ValueError("LaSOT Protocol II testing_set.txt must contain 280 IDs")
        train, dev = lasot_split(training, testing, args.strategy, names(args.validation_list) if args.validation_list else None)
        for split_name, selected in (("train", train), ("dev", dev), ("test", testing)):
            manifest = prepare(root, "lasot", split_name, selected)
            manifest["split_origin"] = "provided-validation-list" if args.validation_list else args.strategy
            save_json(Path(args.out) / (split_name + ".json"), manifest)
    elif args.command == "prepare":
        value = prepare(args.root, args.benchmark, args.split, names(args.sequence_list) if args.sequence_list else None)
        save_json(args.out, value)
    elif args.command == "train":
        if args.validate_every < 1:
            raise ValueError("validate-every must be positive")
        cfg = Config.load(args.config)
        if args.seed is not None:
            cfg.seed = args.seed
        if args.backend:
            cfg.backend = args.backend
        seed_everything(cfg.seed)
        device = choose_device(args.device)
        train, train_info = load_manifest(args.train_manifest)
        dev, dev_info = load_manifest(args.dev_manifest)
        if train_info.get("benchmark") != "lasot" or dev_info.get("benchmark") != "lasot":
            raise ValueError("training requires LaSOT manifests")
        if train_info.get("split") != "train" or dev_info.get("split") != "dev" or len(train) != 896 or len(dev) != 224:
            raise ValueError("training requires distinct 896 train / 224 development sequences")
        model = Network(cfg, Encoders(args.assets)).to(device)
        fit(model, train, dev, device, args.out, args.train_manifest, args.dev_manifest, args.resume, args.max_steps, args.validate_every)
    elif args.command == "predict":
        if args.warmup < 0:
            raise ValueError("warmup must be nonnegative")
        device = choose_device(args.device)
        model = load_network(args.checkpoint, args.assets, device, args.backend)
        sequences, _ = load_manifest(args.manifest, args.root)
        prediction(model, sequences, device, args.out, args.warmup, args.checkpoint, args.manifest)
    elif args.command in ("diagnostic-evaluate", "export"):
        sequences, _ = load_manifest(args.manifest)
        report = diagnostics(sequences, args.predictions) if args.command == "diagnostic-evaluate" else export_predictions(sequences, args.predictions, args.out, args.format)
        if args.command == "diagnostic-evaluate":
            save_json(args.out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "bootstrap":
        if args.resamples < 1:
            raise ValueError("resamples must be positive")
        save_json(args.out, bootstrap(args.input, args.metric, args.other, args.resamples, args.seed))
    elif args.command == "smoke":
        from .smoke import run
        print(json.dumps(run(args.out, choose_device(args.device)), indent=2))


if __name__ == "__main__":
    main()
