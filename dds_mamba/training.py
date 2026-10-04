from pathlib import Path
import json
import math
import random
import time
from dataclasses import asdict
import numpy as np
import torch
from .controller import Tracker
from .geometry import Crop, load_image, xywh_to_center, iou, search_crop
from .losses import objective, focal_loss, teacher_probability, learning_rate
from .encoders import sha256


def negative_crop(target, side, width, height):
    # Seek an in-image center with NO geometric target overlap. Border padding
    # is allowed, but the nearest replicated border must also be target-free.
    candidates = [(side / 2, side / 2), (width - side / 2, side / 2),
                  (side / 2, height - side / 2), (width - side / 2, height - side / 2),
                  (0.5, 0.5), (width - 0.5, 0.5), (0.5, height - 0.5), (width - 0.5, height - 0.5)]
    for x, y in candidates:
        x, y = min(max(x, 0.5), width - 0.5), min(max(y, 0.5), height - 0.5)
        left, top = max(0.0, x - side / 2), max(0.0, y - side / 2)
        right, bottom = min(width, x + side / 2), min(height, y + side / 2)
        visible_box = [0.5 * (left + right), 0.5 * (top + bottom), right - left, bottom - top]
        if iou(visible_box, target) == 0:
            return Crop(x, y, side)
    return None  # Large full-frame targets cannot provide honest target-free crops.


def augmentation(image, xywh, flip, factors):
    box = xywh_to_center(xywh)
    if flip:
        image = image.flip(-1)
        box[0] = image.shape[-1] - box[0]
    brightness, contrast, saturation = factors
    image = image * brightness
    image = (image - image.mean((-1, -2), keepdim=True)) * contrast + image.mean((-1, -2), keepdim=True)
    gray = (image * image.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
    image = (image - gray) * saturation + gray
    return image.clamp(0, 1), box


def clip_loss(model, sequence, boxes, visible, start, epoch, rng, device):
    cfg = model.cfg
    flip = rng.random() < cfg.flip_probability
    factors = [rng.uniform(1 - cfg.photometric, 1 + cfg.photometric) for _ in range(3)]
    image, initial = augmentation(load_image(sequence.frames[start], device), boxes[start], flip, factors)
    tracker = Tracker(model)
    tracker.initialize(image, initial)
    previous_valid = False
    losses, diagnostics = [], {"negative_crops": 0, "teacher_crops": 0, "accepted_commits": 0}
    height, width = image.shape[-2:]

    def jitter(crop):
        scale = rng.uniform(*cfg.jitter_scale)
        dx, dy = [rng.uniform(-cfg.jitter_translation, cfg.jitter_translation) * crop.side for _ in range(2)]
        return Crop(crop.cx + dx, crop.cy + dy, crop.side * scale)

    for step in range(1, cfg.clip_length + 1):
        index = start + step
        image, truth_image = augmentation(load_image(sequence.frames[index], device), boxes[index], flip, factors)
        if image.shape[-2:] != (height, width):
            raise ValueError("sequence resolution must remain constant")
        previous_app = tracker.state.appearance
        candidates, context = tracker.evaluate(image, jitter=jitter)
        selected = tracker.state.selected(candidates, raw_fallback=True)
        valid = bool(visible[index] and selected.crop.contains(truth_image))
        truth = selected.output["box"].new_tensor(selected.crop.to_crop(truth_image))[None]
        identity = model.encoders.identity(image, truth_image) if valid else torch.zeros_like(tracker.state.initial_identity)
        loss, _ = objective(selected.output, truth, identity, valid, previous_app, previous_valid, model)

        # Every step attempts absence supervision; do not secretly label a crop
        # containing the target as absent when its geometry cannot exclude it.
        neg_crop = negative_crop(truth_image, selected.crop.side, width, height)
        if neg_crop is not None:
            negative = tracker.candidate(image, neg_crop, tracker.state.last_box, 99, context, detached_states=True)
            loss = loss + focal_loss(negative.output["logits"], torch.zeros_like(negative.output["logits"]))
            diagnostics["negative_crops"] += 1

        # Teacher uses the same pre-transition state and available RFMB context.
        # It can supervise an online miss, and never writes GT to the controller.
        if visible[index] and rng.random() < teacher_probability(epoch, cfg):
            teacher_crop = Crop(truth_image[0], truth_image[1], selected.crop.side)
            if teacher_crop.contains(truth_image):
                teacher = tracker.candidate(image, teacher_crop, truth_image, 100, context, detached_states=True)
                teacher_truth = truth.new_tensor(teacher_crop.to_crop(truth_image))[None]
                teacher_identity = model.encoders.identity(image, truth_image)
                teacher_loss, _ = objective(teacher.output, teacher_truth, teacher_identity, True, previous_app.detach(), previous_valid, model)
                loss = loss + teacher_loss
                diagnostics["teacher_crops"] += 1
        tracker.state.finish(candidates)
        diagnostics["accepted_commits"] += int(tracker.state.accepted)
        previous_valid = valid
        losses.append(loss)
    # One backward call per clip retains selected QACU paths across all T frames.
    return torch.stack(losses).mean(), diagnostics


@torch.no_grad()
def validation(model, sequences, device):
    overlaps_by_sequence = []
    model.eval()
    for sequence in sequences:
        boxes, _ = sequence.labels()
        tracker = Tracker(model)
        tracker.initialize(load_image(sequence.frames[0], device), xywh_to_center(sequence.initial_xywh))
        overlaps = []
        for index, path in enumerate(sequence.frames[1:], 1):
            predicted, _ = tracker.update(load_image(path, device))
            overlaps.append(0 if predicted is None else iou(predicted, xywh_to_center(boxes[index])))
        if not overlaps:
            raise ValueError("development sequences need at least two frames")
        overlaps_by_sequence.append(float(np.mean(overlaps)))
    # Raw-output integral AUC diagnostic; official evaluator has its own invalid-box handling.
    return float(np.mean(overlaps_by_sequence))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit(model, train_sequences, validation_sequences, device, output, train_manifest, validation_manifest, resume=None, max_steps=None, validate_every=1):
    cfg = model.cfg
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    train_ids = {s.name for s in train_sequences}
    if train_ids & {s.name for s in validation_sequences}:
        raise ValueError("training/development leakage")
    labels = [s.labels() for s in train_sequences]
    starts = [np.flatnonzero(v[:max(0, len(v) - cfg.clip_length)]) for _, v in labels]
    if any(len(s) == 0 for s in starts):
        raise ValueError("every optimization sequence must allow a visible initialization followed by 16 frames")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    start_epoch, best, history = 0, -float("inf"), []
    checkpoint_meta = {"format": "dds-mamba-v1", "config": asdict(cfg), "synthetic": model.encoders.synthetic,
                       "assets": model.encoders.provenance(), "train_manifest_sha256": sha256(train_manifest),
                       "validation_manifest_sha256": sha256(validation_manifest)}
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        for key in ("config", "synthetic", "assets", "train_manifest_sha256", "validation_manifest_sha256"):
            # Tuples are serialized by torch unchanged. Assets are identified by bytes/hash, not root.
            if state[key] != checkpoint_meta[key]:
                raise ValueError(f"resume provenance mismatch: {key}")
        model.load_tracker_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best, history = state["epoch"] + 1, state["best_validation"], state["history"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    summary = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "train_sequences": len(train_sequences), "development_sequences": len(validation_sequences),
               "config": asdict(cfg), "synthetic": model.encoders.synthetic,
               "limited_steps": max_steps, "official_benchmark_verified": False}
    (destination / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, cfg.epochs):
        rng = random.Random(cfg.seed + 1000003 * epoch)
        model.train()
        total, counters = 0.0, {"accepted_commits": 0, "negative_crops": 0, "teacher_crops": 0}
        steps = cfg.clips_per_epoch if max_steps is None else min(cfg.clips_per_epoch, max_steps)
        if steps < 1:
            raise ValueError("steps per epoch must be positive")
        began = time.perf_counter()
        for step in range(steps):
            sequence_index = rng.randrange(len(train_sequences))
            possible = starts[sequence_index]
            start = int(possible[rng.randrange(len(possible))])
            optimizer.zero_grad(set_to_none=True)
            rate = learning_rate(epoch + (step + 0.5) / cfg.clips_per_epoch, cfg)
            for group in optimizer.param_groups:
                group["lr"] = rate
            boxes, visible = labels[sequence_index]
            loss, detail = clip_loss(model, train_sequences[sequence_index], boxes, visible, start, epoch, rng, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch={epoch}, step={step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
            for key in counters:
                counters[key] += detail[key]
            if step == 0 or (step + 1) % 50 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step + 1, "loss": float(loss.detach()), "lr": rate, "gradient_norm": float(norm), **detail}), flush=True)
        metric = validation(model, validation_sequences, device) if (epoch + 1) % validate_every == 0 else None
        improved = metric is not None and metric > best
        if improved:
            best = metric
        record = {"epoch": epoch + 1, "loss": total / steps, "diagnostic_dev_auc": metric,
                  "seconds": time.perf_counter() - began, "teacher_probability": teacher_probability(epoch, cfg), **counters}
        history.append(record)
        state = {**checkpoint_meta, "model": model.tracker_state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_validation": best, "history": history, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        torch.save(state, destination / "last.pt")
        if improved:
            torch.save(state, destination / "best.pt")
        (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
    return history
