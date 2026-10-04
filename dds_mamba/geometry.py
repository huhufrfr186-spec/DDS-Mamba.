"""All internal boxes are cx,cy,w,h. File I/O uses x,y,w,h."""
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def xywh_to_center(b):
    x, y, w, h = map(float, b)
    return np.array([x + w / 2, y + h / 2, w, h], dtype=np.float64)


def center_to_xywh(b):
    b = np.asarray(b, dtype=np.float64)
    return np.r_[b[:2] - b[2:] / 2, b[2:]]


def valid_box(b):
    b = np.asarray(b)
    return b.shape == (4,) and np.isfinite(b).all() and (b[2:] > 0).all()


def corners(b):
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1)


def overlap_tensor(a, b, generalized=False):
    a, b = corners(a), corners(b)
    inter = (torch.minimum(a[..., 2:], b[..., 2:]) - torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    aa = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    bb = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    union = aa + bb - inter
    result = inter / union.clamp_min(1e-12)
    if generalized:
        cover = (torch.maximum(a[..., 2:], b[..., 2:]) - torch.minimum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
        result = result - (cover - union) / cover.clamp_min(1e-12)
    return result


def iou(a, b):
    if not valid_box(a) or not valid_box(b):
        return 0.0
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[:2] + a[2:] / 2, b[:2] + b[2:] / 2) - np.maximum(a[:2] - a[2:] / 2, b[:2] - b[2:] / 2)).prod()
    return float(intersection / max(a[2:].prod() + b[2:].prod() - intersection, 1e-12))


@dataclass(frozen=True)
class Crop:
    cx: float
    cy: float
    side: float

    @classmethod
    def around(cls, box, factor):
        return cls(float(box[0]), float(box[1]), max(2.0, factor * math.sqrt(float(box[2] * box[3]))))

    def to_crop(self, box):
        b = np.asarray(box, dtype=np.float64)
        return np.r_[(b[:2] - [self.cx, self.cy]) / self.side + 0.5, b[2:] / self.side]

    def to_image(self, box):
        # Tensor path keeps gradients for QACU agreement.
        if isinstance(box, torch.Tensor):
            c = box.new_tensor([self.cx, self.cy])
            return torch.cat([(box[..., :2] - 0.5) * self.side + c, box[..., 2:] * self.side], -1)
        b = np.asarray(box)
        return np.r_[(b[:2] - 0.5) * self.side + [self.cx, self.cy], b[2:] * self.side]

    def contains(self, box):
        b = self.to_crop(box)
        return bool(valid_box(b) and (b[:2] - b[2:] / 2 >= 0).all() and (b[:2] + b[2:] / 2 <= 1).all())


def load_image(path, device="cpu"):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def region(image, box, size, antialias=True):
    """Half-open ROI, border replication, 2x sampling and antialiased resize."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("native-resolution controller uses a single RGB frame")
    b = torch.as_tensor(box, dtype=image.dtype, device=image.device).reshape(4)
    height, width = image.shape[-2:]
    sample_size = 2 * size if antialias else size
    q = (torch.arange(sample_size, device=image.device, dtype=image.dtype) + 0.5) / sample_size - 0.5
    y, x = torch.meshgrid(q, q, indexing="ij")
    # Image edge coordinates [0,W] map to grid coordinates [-1,1].
    grid = torch.stack([2 * (b[0] + x * b[2]) / width - 1, 2 * (b[1] + y * b[3]) / height - 1], -1)
    sampled = F.grid_sample(image, grid[None], mode="bilinear", padding_mode="border", align_corners=False)
    return F.interpolate(sampled, size=(size, size), mode="bilinear", align_corners=False, antialias=True) if antialias else sampled


def search_crop(image, crop, size=256):
    return region(image, [crop.cx, crop.cy, crop.side, crop.side], size)


def normalize(image):
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (image - mean) / std
