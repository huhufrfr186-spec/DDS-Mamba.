from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Config:
    d_model: int = 256
    d_state: int = 16
    expand: int = 2
    dt_rank: int = 16
    d_conv: int = 4
    position_layers: int = 2
    appearance_layers: int = 4
    backend: str = "reference"
    gate_min: float = 0.5
    gate_max: float = 1.5
    gate_eta: float = 0.5
    gate_temperature: float = 1.0
    map_threshold: float = 0.45
    peak_threshold: float = 0.50
    identity_threshold: float = 0.60
    agreement_threshold: float = 0.25
    qacu_threshold: float = 0.35
    rho_max: float = 0.95
    trajectory_ema: float = 0.90
    memory_capacity: int = 5
    memory_topk: int = 5
    memory_decay: float = 0.01
    memory_write_threshold: float = 0.70
    weak_limit: int = 3
    confirmation_count: int = 2
    cache_capacity: int = 2
    recovery_iou: float = 0.50
    recovery_identity: float = 0.65
    active_scale: float = 4.0
    lost_motion_scale: float = 8.0
    lost_last_scale: float = 8.0
    innovation_gate: float = 9.49
    process_diagonal: tuple = (1e-4, 1e-4, 1e-5, 1e-5, 1e-3, 1e-3, 1e-4, 1e-4)
    measurement_diagonal: tuple = (1e-3, 1e-3, 5e-4, 5e-4)
    # Initialization and geometry settings used by this implementation.
    initial_covariance: tuple = (0.04, 0.04, 0.02, 0.02, 0.10, 0.10, 0.05, 0.05)
    initial_reliability: float = 1.0
    template_scale: float = 2.0
    gaussian_sigma: float = 1.5
    min_box_fraction: float = 1e-3
    epochs: int = 40
    warmup_epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    clip_length: int = 16
    clips_per_epoch: int = 8960
    seed: int = 2024
    jitter_scale: tuple = (0.8, 1.2)
    jitter_translation: float = 0.2
    photometric: float = 0.2
    flip_probability: float = 0.5
    teacher_cutoff_fraction: float = 0.5
    auxiliary_coefficient: float = 0.1
    norm_coefficient: float = 0.01
    checkpoint_branches: bool = True

    def __post_init__(self):
        if self.d_model % 4 or self.d_model <= 0:
            raise ValueError("d_model must be positive and divisible by four")
        if not 0 < self.gate_min <= 1 <= self.gate_max:
            raise ValueError("gate bounds must include 1")
        if self.backend not in ("reference", "mamba_ssm"):
            raise ValueError("backend must be reference or mamba_ssm")
        if not 1 <= self.confirmation_count <= self.cache_capacity:
            raise ValueError("cache capacity must cover confirmation length")
        if len(self.process_diagonal) != 8 or len(self.measurement_diagonal) != 4:
            raise ValueError("Kalman covariance dimensions are 8 and 4")
        if min(self.memory_capacity, self.memory_topk, self.clip_length, self.clips_per_epoch, self.epochs) < 1:
            raise ValueError("memory capacities and training lengths must be positive")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
