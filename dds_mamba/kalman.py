"""Resolution-normalized constant-velocity filter; all measurements are detached."""
import numpy as np
from .geometry import valid_box


class Kalman:
    def __init__(self, box, width, height, cfg):
        self.width, self.height, self.cfg = width, height, cfg
        self.transition = np.eye(8)
        self.transition[:4, 4:] = np.eye(4)
        self.measurement = np.eye(4, 8)
        self.Q = np.diag(cfg.process_diagonal)
        self.R = np.diag(cfg.measurement_diagonal)
        self.reset(box)

    def encode(self, box):
        if not valid_box(box):
            raise ValueError("invalid measurement box")
        size = np.array([self.width, self.height])
        return np.r_[np.asarray(box)[:2] / size, np.log(np.asarray(box)[2:] / size)]

    def decode(self, state=None):
        s = self.x if state is None else state
        if not np.isfinite(s).all():
            return None
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            size = np.array([self.width, self.height])
            box = np.r_[s[:2] * size, np.exp(s[2:4]) * size]
        return box if valid_box(box) else None

    def reset(self, box):
        self.x = np.r_[self.encode(box), np.zeros(4)]
        self.P = np.diag(self.cfg.initial_covariance).astype(np.float64)

    def predict(self):
        self.x = self.transition @ self.x
        self.P = self.transition @ self.P @ self.transition.T + self.Q
        self.P = (self.P + self.P.T) / 2
        return self.decode()

    def update(self, box):
        if not valid_box(box) or not np.isfinite(self.P).all() or not np.isfinite(self.x).all():
            return None
        innovation = self.encode(box) - self.x[:4]
        covariance = self.P[:4, :4] + self.R
        try:
            solved = np.linalg.solve(covariance, innovation)
            distance = float(innovation @ solved)
            if not np.isfinite(distance) or distance > self.cfg.innovation_gate:
                return None
            gain = np.linalg.solve(covariance, self.P[:, :4].T).T
        except np.linalg.LinAlgError:
            return None
        posterior = self.x + gain @ innovation
        box_out = self.decode(posterior)
        if box_out is None:
            return None
        residual = np.eye(8) - gain @ self.measurement
        posterior_cov = residual @ self.P @ residual.T + gain @ self.R @ gain.T  # Joseph form.
        if not np.isfinite(posterior_cov).all():
            return None
        self.x, self.P = posterior, (posterior_cov + posterior_cov.T) / 2
        return box_out
