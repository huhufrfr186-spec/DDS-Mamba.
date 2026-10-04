from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .mamba import Stack, bounded_gate, sincos2d


class Network(nn.Module):
    def __init__(self, cfg, encoders):
        super().__init__()
        self.cfg, self.encoders = cfg, encoders
        d = cfg.d_model
        self.search_projection = nn.Linear(768, d)
        self.template_affine = nn.Linear(768, 2 * d)
        self.initial_position = nn.Linear(768, d)
        self.initial_appearance = nn.Linear(384, d)
        self.box_encoding = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.position_condition = nn.Linear(2 * d, d)
        self.position = Stack(cfg, cfg.position_layers)
        self.appearance = Stack(cfg, cfg.appearance_layers)
        self.box_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.gate_head = nn.Linear(d, 256)
        self.previous_appearance = nn.Linear(d, d)
        self.memory_query = nn.Linear(d + 384, 384)
        self.context_projection = nn.Linear(384, d)
        self.null_context = nn.Parameter(torch.zeros(1, 384))
        self.confidence_head = nn.Linear(2 * d, 1)
        self.identity_projector = nn.Linear(d, 384)
        self.register_buffer("spatial_encoding", sincos2d(16, 16, d), persistent=False)

    def initialize_states(self, template_tokens, identity):
        return self.initial_position(template_tokens.mean(1)), F.normalize(self.initial_appearance(identity), dim=-1, eps=1e-6)

    def read_memory(self, app, initial_identity, memory):
        if not memory.entries:
            return self.null_context.expand(app.shape[0], -1)
        keys, utilities = memory.tensors(app)
        query = F.normalize(self.memory_query(torch.cat([app, initial_identity], -1)), dim=-1, eps=1e-6)
        scores = query @ keys.T * utilities[None]
        # Choice: every finite positive-utility entry is valid; no undocumented cosine gate.
        indices = torch.argsort(scores.detach(), descending=True, stable=True, dim=-1)[:, :self.cfg.memory_topk]
        selected_scores = scores.gather(1, indices)
        weights = selected_scores.softmax(-1)  # literal beta_j=exp(s_j)/sum exp(s_r)
        return (weights[..., None] * keys[indices]).sum(1)

    def forward(self, template_tokens, search_tokens, crop_reference, position_state, appearance_state, context):
        if search_tokens.shape[1:] != (256, 768):
            raise ValueError("search encoder must supply 256 row-major 768-dimensional patch tokens")
        x = self.search_projection(search_tokens)
        scale, bias = self.template_affine(template_tokens.mean(1)).chunk(2, -1)
        # Template-conditioned affine modulation with bounded scale.
        x = x * (1 + torch.tanh(scale[:, None])) + bias[:, None]
        condition = self.position_condition(torch.cat([x.mean(1), self.box_encoding(crop_reference)], -1))
        pos_tokens = torch.stack([position_state, condition], 1)
        position = self.position(pos_tokens)[:, 1]
        raw_box = self.box_head(position)
        minimum = self.cfg.min_box_fraction
        box = torch.cat([raw_box[:, :2].sigmoid(), minimum + (1 - minimum) * raw_box[:, 2:].sigmoid()], -1)
        gate = bounded_gate(self.gate_head(position), self.cfg)
        spatial = x * gate[..., None] + self.spatial_encoding.to(x)
        sequence = torch.cat([self.previous_appearance(appearance_state)[:, None], spatial, self.context_projection(context)[:, None]], 1)
        outputs = self.appearance(sequence)
        appearance = outputs[:, 257]
        logits = self.confidence_head(torch.cat([outputs[:, 1:257], appearance[:, None].expand(-1, 256, -1)], -1)).squeeze(-1)
        y = self.identity_projector(appearance)
        return {"box": box, "position": position, "appearance": appearance, "logits": logits,
                "identity_projection": y, "gate": gate, "spatial_outputs": outputs[:, 1:257]}

    def tracker_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("encoders.")}

    def load_tracker_state_dict(self, state):
        result = self.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith("encoders.")]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"incompatible tracker checkpoint: missing={missing}, unexpected={result.unexpected_keys}")
