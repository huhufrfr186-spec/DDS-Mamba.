import math
import torch
import torch.nn.functional as F
from .geometry import overlap_tensor
from .controller import map_evidence


def grid_centers(reference):
    q = (torch.arange(16, dtype=reference.dtype, device=reference.device) + 0.5) / 16
    y, x = torch.meshgrid(q, q, indexing="ij")
    return torch.stack([x, y], -1).reshape(256, 2)


def focal_loss(logits, target):
    # Equation in Sec 3.5 uses (1-Y), not CenterNet's (1-Y)^4.
    p = logits.sigmoid()
    return -(target * (1 - p).square() * F.logsigmoid(logits)
             + (1 - target) * p.square() * F.logsigmoid(-logits)).mean()


def objective(output, truth, identity, valid, previous_committed_app, previous_valid, model):
    cfg = model.cfg
    logits, box = output["logits"], output["box"]
    zero = logits.sum() * 0
    grid = grid_centers(logits)
    target = torch.exp(-((grid[None] - truth[:, None, :2]) * 16).square().sum(-1) / (2 * cfg.gaussian_sigma**2)) if valid else torch.zeros_like(logits)
    terms = {"ctr": focal_loss(logits, target), "box": zero, "decorr": zero, "temp": zero, "consist": zero, "norm": zero, "align": zero}
    if valid:
        y = output["identity_projection"]
        terms["box"] = (5 * (box - truth).abs().sum(-1) + 2 * (1 - overlap_tensor(box, truth, generalized=True))).mean()
        terms["decorr"] = F.cosine_similarity(output["position"], output["appearance"], dim=-1, eps=1e-6).square().mean()
        if previous_valid:
            prior_y = model.identity_projector(previous_committed_app)
            terms["temp"] = F.relu(0.15 - F.cosine_similarity(y, prior_y, dim=-1, eps=1e-6)).mean()
        terms["consist"] = (1 - F.cosine_similarity(y, identity, dim=-1, eps=1e-6)).mean()
        terms["norm"] = F.relu(0.5 - y.norm(dim=-1)).square().mean()
        _, _, probability = map_evidence(logits)
        center = probability @ grid
        terms["align"] = (box[:, :2] - center).abs().sum(-1).mean()
    total = terms["box"] + terms["ctr"] + cfg.auxiliary_coefficient * sum(terms[k] for k in ("decorr", "temp", "consist", "align")) + cfg.norm_coefficient * terms["norm"]
    # Sum frame terms and divide by clip length in the trainer, including invalid frames.
    return total, terms


def teacher_probability(epoch, cfg):
    cutoff = cfg.epochs * cfg.teacher_cutoff_fraction
    return 0.5 * (1 + math.cos(math.pi * epoch / cutoff)) if epoch < cutoff else 0.0


def learning_rate(epoch_fraction, cfg):
    if epoch_fraction < cfg.warmup_epochs:
        return cfg.learning_rate * (epoch_fraction + 1e-6) / max(cfg.warmup_epochs, 1e-6)
    progress = (epoch_fraction - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
    return cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
