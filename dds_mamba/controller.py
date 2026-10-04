from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from .geometry import Crop, valid_box, iou, overlap_tensor, search_crop
from .kalman import Kalman
from .memory import Memory


def scalar(x):
    return float(x.detach().reshape(-1)[0].cpu()) if isinstance(x, torch.Tensor) else float(x)


def map_evidence(logits):
    confidence = logits.sigmoid()
    probability = (confidence + 1e-6) / (confidence + 1e-6).sum(-1, keepdim=True)
    entropy = -(probability * probability.log()).sum(-1)
    concentration = (1 - entropy / np.log(logits.shape[-1])).clamp(0, 1)
    return concentration, confidence.amax(-1), probability


@dataclass
class Candidate:
    output: dict
    crop: Crop
    index: int
    box: np.ndarray
    embedding: torch.Tensor
    map_quality: torch.Tensor
    peak: torch.Tensor
    identity: torch.Tensor
    agreement: torch.Tensor
    rate: torch.Tensor

    def score(self, mode):
        value = self.map_quality * self.peak * self.identity
        if mode == "active":
            value = value * self.agreement
        result = scalar(value)
        return result if np.isfinite(result) else -float("inf")

    def eligible(self, mode, cfg):
        if not valid_box(self.box) or not all(np.isfinite(scalar(v)) for v in (self.map_quality, self.peak, self.identity, self.agreement, self.rate)):
            return False
        if not all(torch.isfinite(self.output[k]).all().item() for k in ("position", "appearance", "box", "logits")):
            return False
        base = scalar(self.map_quality) >= cfg.map_threshold and scalar(self.peak) >= cfg.peak_threshold and scalar(self.identity) >= cfg.identity_threshold
        return base and (mode == "lost" or (scalar(self.agreement) >= cfg.agreement_threshold and scalar(self.rate) >= cfg.qacu_threshold))


@dataclass
class CacheRecord:
    frame: int
    box: np.ndarray
    embedding: torch.Tensor


class Controller:
    """Persistent sequence state. Only selected neural states carry gradients."""
    def __init__(self, cfg, box, position, appearance, identity, width, height):
        self.cfg = cfg
        self.last_box = np.asarray(box, dtype=np.float64).copy()
        self.position, self.appearance = position, appearance
        self.initial_identity = identity.detach()
        self.kalman = Kalman(box, width, height, cfg)
        self.memory = Memory(cfg)
        self.mode, self.incoming_mode = "active", "active"
        self.weak_count = self.lost_count = self.frame = self.commit_count = 0
        self.cache = deque(maxlen=cfg.cache_capacity)
        self.reliability = cfg.initial_reliability
        self.last_rate = 0.0
        self.prediction = None
        self.accepted = self.recovered = False

    def begin(self):
        self.frame += 1
        self.incoming_mode = self.mode
        self.prediction = self.kalman.predict()
        self.accepted = self.recovered = False
        motion = self.prediction if self.prediction is not None else self.last_box
        if self.incoming_mode == "active":
            return [(Crop.around(motion, self.cfg.active_scale), motion)]
        return [(Crop.around(motion, self.cfg.lost_motion_scale), motion),
                (Crop.around(self.last_box, self.cfg.lost_last_scale), self.last_box)]

    def selected(self, candidates, raw_fallback=False):
        eligible = [c for c in candidates if c.eligible(self.incoming_mode, self.cfg)]
        pool = eligible or (candidates if raw_fallback else [])
        # Fixed crop order resolves ties; losses can use a raw candidate without committing it.
        return max(pool, key=lambda c: (c.score(self.incoming_mode), -c.index)) if pool else None

    def finish(self, candidates):
        selected = self.selected(candidates)
        # All entries age once, after the pre-frame read and before a possible new write.
        self.memory.age()
        if self.incoming_mode == "active":
            posterior = self.kalman.update(selected.box) if selected is not None else None
            if posterior is not None:
                self.position = selected.output["position"]
                rate = selected.rate.reshape(-1, 1)
                self.appearance = F.normalize((1 - rate) * self.appearance + rate * selected.output["appearance"], dim=-1, eps=1e-6)
                self.last_box = posterior.copy()
                self.last_rate = scalar(selected.rate)
                self.reliability = self.cfg.trajectory_ema * self.reliability + (1 - self.cfg.trajectory_ema) * self.last_rate
                self.commit_count += 1
                self.weak_count = self.lost_count = 0
                self.cache.clear()
                self.accepted = True
                self.memory.write(selected.embedding, self.reliability)
                return posterior
            self.weak_count += 1
            if self.weak_count >= self.cfg.weak_limit:
                self.mode = "lost"  # Current incoming-active output remains the prediction.
                self.cache.clear()
            return None if self.prediction is None else self.prediction.copy()

        self.lost_count += 1
        if selected is None:
            self.cache.clear()
            return None
        record = CacheRecord(self.frame, selected.box.copy(), selected.embedding.detach().clone())
        if self.cache:
            previous = self.cache[-1]
            similarity = scalar((previous.embedding * record.embedding).sum(-1))
            consistent = previous.frame + 1 == self.frame and iou(previous.box, record.box) >= self.cfg.recovery_iou and similarity >= self.cfg.recovery_identity
            if not consistent:
                self.cache.clear()
        self.cache.append(record)
        if len(self.cache) < self.cfg.confirmation_count:
            return None
        posterior = self.kalman.update(selected.box) if scalar(selected.agreement) >= self.cfg.agreement_threshold else None
        if posterior is None:
            self.kalman.reset(selected.box)
            posterior = selected.box.copy()
        self.position = selected.output["position"]
        self.last_box = posterior.copy()
        self.mode = "active"
        self.weak_count = self.lost_count = 0
        self.cache.clear()
        self.recovered = True
        # Preserve appearance, reliability EMA, commit count, last QACU rate, RFMB order.
        return posterior

    def diagnostics(self, output):
        return {"frame": self.frame, "incoming_mode": self.incoming_mode, "next_mode": self.mode,
                "emitted": output is not None, "accepted": self.accepted, "recovered": self.recovered,
                "weak_count": self.weak_count, "lost_count": self.lost_count, "commit_count": self.commit_count,
                "last_rate": self.last_rate, "trajectory_reliability": self.reliability,
                "memory_size": len(self.memory.entries), "cache_size": len(self.cache)}


class Tracker:
    def __init__(self, network):
        self.network = network
        self.cfg = network.cfg
        self.state = None

    def initialize(self, image, center_box):
        height, width = image.shape[-2:]
        if not valid_box(center_box):
            raise ValueError("first-frame initialization must be a valid target box")
        template = search_crop(image, Crop.around(center_box, self.cfg.template_scale), 128)
        self.template_tokens = self.network.encoders.patches(template).detach()
        identity = self.network.encoders.identity(image, center_box).detach()
        position, appearance = self.network.initialize_states(self.template_tokens, identity)
        self.state = Controller(self.cfg, center_box, position, appearance, identity, width, height)

    def candidate(self, image, crop, reference, index, context, detached_states=False):
        state = self.state
        position, appearance = state.position, state.appearance
        if detached_states:
            position, appearance, context = position.detach(), appearance.detach(), context.detach()
        patches = self.network.encoders.patches(search_crop(image, crop))
        reference_tensor = patches.new_tensor(crop.to_crop(reference))[None]
        out = self.network(self.template_tokens, patches, reference_tensor, position, appearance, context)
        image_box = crop.to_image(out["box"])
        box = image_box[0].detach().double().cpu().numpy()
        if valid_box(box):
            embedding = self.network.encoders.identity(image, box)
        else:
            embedding = torch.zeros_like(state.initial_identity)
        quality, peak, _ = map_evidence(out["logits"])
        identity = state.memory.identity_score(embedding, state.initial_identity)
        agreement = overlap_tensor(image_box, image_box.new_tensor(state.prediction)[None]) if state.prediction is not None else quality.new_zeros(1)
        rate = torch.minimum(quality.new_tensor(self.cfg.rho_max), quality * agreement)
        return Candidate(out, crop, index, box, embedding, quality, peak, identity, agreement, rate)

    def evaluate(self, image, jitter=None):
        if self.state is None:
            raise RuntimeError("initialize the tracker before updating")
        specs = self.state.begin()
        context = self.network.read_memory(self.state.appearance, self.state.initial_identity, self.state.memory)
        candidates = []
        for index, (crop, reference) in enumerate(specs):
            if jitter is not None:
                crop = jitter(crop)
            candidates.append(self.candidate(image, crop, reference, index, context))
        return candidates, context

    @torch.no_grad()
    def update(self, image):
        candidates, _ = self.evaluate(image)
        raw = self.state.selected(candidates, raw_fallback=True)
        box = self.state.finish(candidates)
        info = self.state.diagnostics(box)
        if raw is not None:
            info.update({"map_quality": scalar(raw.map_quality), "peak": scalar(raw.peak),
                         "identity": scalar(raw.identity), "agreement": scalar(raw.agreement),
                         "candidate_rate": scalar(raw.rate), "selected_crop": raw.index,
                         "tracking_confidence": max(0.0, raw.score(self.state.incoming_mode)) if box is not None else 0.0})
        return box, info
