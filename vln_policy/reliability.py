"""Reliability features, candidate fusion, and offline gate training."""

import torch
from torch import nn
from torch.nn.functional import cosine_similarity


class ReliabilityGate(nn.Module):
    """A scalar fusion weight from 12 per-decision statistics."""

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(12))
        self.register_buffer('std', torch.ones(12))
        self.net = nn.Sequential(nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 1))
        self.register_buffer('temperature', torch.ones(()))

    def forward(self, features):
        normalized = ((features - self.mean) / self.std.clamp_min(0.1)).clamp(-8, 8)
        return (self.net(normalized).squeeze(-1) / self.temperature).sigmoid()


def reliability_features(imagined, lookahead_scores, global_scores, selected):
    """Build one feature vector from [1, 2, views, channels] and two scores."""
    views = imagined.float()[0]
    local = lookahead_scores.float().reshape(-1)
    global_scores = global_scores.float().reshape(-1)
    first, second = map(int, selected)
    norms = views.norm(dim=-1)
    pooled = views.mean(dim=1)
    pooled_similarity = cosine_similarity(pooled[0:1], pooled[1:2]).squeeze()
    view_similarity = cosine_similarity(views[0], views[1], dim=-1)
    local_margin = local[0] - local[1]
    global_margin = global_scores[first] - global_scores[second]
    return torch.stack([
        local_margin,
        global_margin,
        local_margin.abs(),
        global_margin.abs(),
        local.max() - global_scores[0],
        global_scores[[first, second]].max() - global_scores[0],
        pooled_similarity,
        view_similarity.mean(),
        view_similarity.std(unbiased=False),
        norms.mean() / 30,
        (norms[0].mean() - norms[1].mean()) / 30,
        (local[0] - global_scores[first]) - (local[1] - global_scores[second]),
    ]).float()


def fuse_logits(global_logits, valid, selected, lookahead_scores, alpha):
    """Fuse two selected scores per decision, preserving host AMP precision."""
    base_scores = global_logits.gather(1, selected)
    local_scores = base_scores + alpha[:, None] * (lookahead_scores - base_scores)
    fused = global_logits.scatter(1, selected, local_scores.half().to(global_logits.dtype))
    return fused.masked_fill(~valid, -torch.inf)


def ranking_label(predicted_margin, observed_margin, epsilon=1e-4):
    """Label agreement between the imagined and observed candidate order."""
    if abs(predicted_margin) <= epsilon or abs(observed_margin) <= epsilon:
        return None
    return float(predicted_margin * observed_margin > 0)


def offline_loss_terms(logits, reference_logits, valid, action, behavior_logp, weight):
    """Compute the per-decision signed-return surrogate and reference KL."""
    logp = logits.log_softmax(-1)
    reference = reference_logits.detach().log_softmax(-1)
    action = action[:, None]
    behavior_logp = behavior_logp.detach()
    ratio = (logp.gather(1, action).squeeze(1) - behavior_logp).exp()
    reference_ratio = (reference.gather(1, action).squeeze(1) - behavior_logp).exp()
    clipped = torch.maximum(torch.minimum(ratio, reference_ratio + 0.2), reference_ratio - 0.2)
    advantage = weight.detach() - 1
    actor = -torch.minimum(ratio * advantage, clipped * advantage)
    log_ratio = reference.masked_fill(~valid, 0) - logp.masked_fill(~valid, 0)
    anchor = (reference.exp() * log_ratio).sum(dim=1)
    return actor, anchor
