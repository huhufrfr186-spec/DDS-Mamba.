"""Frozen ViT encoders compatible with official MAE and DINOv2 state dictionaries.

No random-pretraining fallback is allowed in a real run. TinyEncoders is for
tests only and checkpoints produced with it are explicitly marked synthetic.
"""
from pathlib import Path
from functools import partial
import hashlib
import json
import urllib.request
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import normalize, region

ASSETS = {
    "mae": ("mae_pretrain_vit_base.pth", "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"),
    "dino": ("dinov2_vits14_pretrain.pth", "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"),
}
ASSET_LOCKS = {
    "mae": (343249461, "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"),
    "dino": (88283115, "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_assets(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, (name, url) in ASSETS.items():
        destination = root / name
        if not destination.exists():
            temporary = destination.with_suffix(".download")
            print(f"Downloading {name} from official Meta endpoint", flush=True)
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
                while data := source.read(1024 * 1024):
                    target.write(data)
            temporary.replace(destination)
        digest = sha256(destination)
        expected_bytes, expected_digest = ASSET_LOCKS[key]
        if destination.stat().st_size != expected_bytes or digest != expected_digest:
            raise RuntimeError(f"official asset integrity mismatch: {destination}")
        manifest[key] = {"name": name, "url": url, "bytes": destination.stat().st_size, "sha256": digest}
    (root / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, layer_scale):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn, self.mlp = Attention(dim, heads), MLP(dim)
        self.ls1 = LayerScale(dim) if layer_scale else nn.Identity()
        self.ls2 = LayerScale(dim) if layer_scale else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class FrozenViT(nn.Module):
    def __init__(self, kind, checkpoint):
        super().__init__()
        if kind not in ("mae", "dino"):
            raise ValueError(kind)
        dim, patch, heads = (768, 16, 12) if kind == "mae" else (384, 14, 6)
        grid = 14 if kind == "mae" else 37
        self.kind, self.patch, self.grid = kind, patch, grid
        self.patch_embed = PatchEmbed(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, kind == "dino") for _ in range(12)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing frozen encoder: {checkpoint}. Run python -m dds_mamba download --assets assets")
        fingerprint = sha256(checkpoint)
        expected_bytes, expected_digest = ASSET_LOCKS[kind]
        if checkpoint.stat().st_size != expected_bytes or fingerprint != expected_digest:
            raise RuntimeError(f"encoder does not match the verified official asset: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("model", payload)
        # Ignore decoder-only MAE tensors and the unused DINO mask token.
        state = {k.removeprefix("module."): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith(("decoder", "mask_token"))}
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.asset_info = {"path": checkpoint.name, "sha256": fingerprint, "bytes": checkpoint.stat().st_size}
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def positional(self, height, width, dtype):
        if height == width == self.grid:
            return self.pos_embed.to(dtype)
        patch_pos = self.pos_embed[:, 1:].float().reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        if self.kind == "dino":
            # Match DINOv2's interpolate_offset=0.1, antialias=False convention.
            pos = F.interpolate(patch_pos, scale_factor=((height + 0.1) / self.grid, (width + 0.1) / self.grid), mode="bicubic", align_corners=False)
        else:
            pos = F.interpolate(patch_pos, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
        if pos.shape[-2:] != (height, width):
            raise RuntimeError("positional interpolation produced the wrong grid")
        pos = pos.flatten(2).transpose(1, 2)
        return torch.cat([self.pos_embed[:, :1], pos], 1).to(dtype)

    @torch.no_grad()
    def forward(self, image):
        h, w = image.shape[-2] // self.patch, image.shape[-1] // self.patch
        tokens = self.patch_embed(image)
        tokens = torch.cat([self.cls_token.expand(image.shape[0], -1, -1), tokens], 1)
        tokens = tokens + self.positional(h, w, tokens.dtype)
        # Deliberately no MAE random_masking: mask_ratio=0 shuffling would break row order.
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class Encoders(nn.Module):
    synthetic = False

    def __init__(self, assets):
        super().__init__()
        root = Path(assets)
        self.mae = FrozenViT("mae", root / ASSETS["mae"][0])
        self.dino = FrozenViT("dino", root / ASSETS["dino"][0])

    @torch.no_grad()
    def patches(self, image):
        return self.mae(normalize(image))[:, 1:]

    @torch.no_grad()
    def identity(self, image, box):
        return F.normalize(self.dino(normalize(region(image, box, 224)))[:, 0], dim=-1, eps=1e-6)

    def provenance(self):
        return {"mae": self.mae.asset_info, "dino": self.dino.asset_info}


class TinyEncoders(nn.Module):
    """Frozen deterministic-size stand-in for offline tests; NEVER benchmark weights."""
    synthetic = True

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 768, 16, stride=16)
        self.embed = nn.Linear(3, 384)
        self.requires_grad_(False)

    @torch.no_grad()
    def patches(self, image):
        return self.patch(image).flatten(2).transpose(1, 2)

    @torch.no_grad()
    def identity(self, image, box):
        rgb = region(image, box, 16).mean((-1, -2))
        return F.normalize(self.embed(rgb), dim=-1, eps=1e-6)

    def provenance(self):
        return {"synthetic": True}
