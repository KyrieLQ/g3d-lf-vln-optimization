"""Objectives used by the panoramic g3D-LF GRPO adaptation."""

import torch


def trajectory_reward(final_distance, initial_distance, path_length, success_weight=1.0):
    """Compute a completed rollout's return from distances in meters."""
    success = (final_distance <= 1.5).to(final_distance.dtype)
    denominator = torch.maximum(initial_distance, path_length)
    efficiency = initial_distance / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
    return success * efficiency + success_weight * success + 1 - final_distance / 6


def group_relative_advantages(rewards):
    """Normalize completed rollout rewards along the group axis."""
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    return centered / (rewards.std(dim=-1, unbiased=False, keepdim=True) + 1e-8)


def grpo_loss(logp, old_logp, reference_logp, advantages, epsilon=0.2, beta=0.04):
    """Compute the objective on valid sampled actions using a sampled KL term."""
    ratio = (logp - old_logp.detach()).exp()
    advantages = advantages.detach()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
    policy = -torch.minimum(unclipped, clipped).mean()

    reference_ratio = reference_logp.detach() - logp
    kl = (reference_ratio.exp() - reference_ratio - 1).mean()
    return policy + beta * kl
