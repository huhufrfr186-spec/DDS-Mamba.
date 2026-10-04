from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass
class Entry:
    embedding: torch.Tensor
    weight: float
    age: int = 0


class Memory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.entries = []

    def utility(self, entry):
        return entry.weight * math.exp(-self.cfg.memory_decay * entry.age)

    def age(self):
        for entry in self.entries:
            entry.age += 1

    def tensors(self, reference):
        keys = torch.cat([entry.embedding.to(reference) for entry in self.entries], 0)
        utilities = reference.new_tensor([self.utility(entry) for entry in self.entries])
        return keys, utilities

    def identity_score(self, candidate, initial):
        score = (candidate * initial).sum(-1).clamp_min(0)
        if self.entries:
            keys, utilities = self.tensors(candidate)
            score = torch.maximum(score, (candidate @ keys.T * utilities).clamp_min(0).amax(-1))
        return score.clamp(0, 1)

    def write(self, embedding, reliability):
        if reliability < self.cfg.memory_write_threshold:
            return False
        entry = Entry(F.normalize(embedding.detach().clone(), dim=-1, eps=1e-6), float(reliability))
        if len(self.entries) < self.cfg.memory_capacity:
            self.entries.append(entry)
        else:
            index = min(range(len(self.entries)), key=lambda i: (self.utility(self.entries[i]), i))
            self.entries[index] = entry
        return True
