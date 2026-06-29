"""Losses for teacher-student behavior cloning."""

from __future__ import annotations

from typing import Dict

import torch as th
import torch.nn.functional as F


def weighted_action_cross_entropy(
    logits: th.Tensor,
    target: th.Tensor,
    *,
    action0_weight: float = 1.0,
    nonidle_weight: float = 1.0,
) -> th.Tensor:
    per_example = F.cross_entropy(logits, target.long(), reduction="none")
    weights = th.where(
        target.long() == 0,
        th.full_like(per_example, float(action0_weight)),
        th.full_like(per_example, float(nonidle_weight)),
    )
    return (per_example * weights).mean()


def intervention_bce_loss(
    logits: th.Tensor,
    target: th.Tensor,
    *,
    pos_weight: float = 1.0,
) -> th.Tensor:
    if logits.shape[-1] <= 1:
        return logits.sum() * 0.0
    intervention_target = (target.long() != 0).float()
    intervention_logit = th.logsumexp(logits[..., 1:], dim=-1) - logits[..., 0]
    return F.binary_cross_entropy_with_logits(
        intervention_logit,
        intervention_target,
        pos_weight=th.as_tensor(float(pos_weight), device=logits.device),
    )


def classification_metrics(logits: th.Tensor, target: th.Tensor) -> Dict[str, float]:
    pred = th.argmax(logits, dim=-1)
    target = target.long()
    total = max(int(target.numel()), 1)
    teacher_nonidle = target != 0
    teacher_idle = target == 0
    false_noop = (pred == 0) & teacher_nonidle
    false_intervention = (pred != 0) & teacher_idle
    nonidle_den = max(int(teacher_nonidle.sum().item()), 1)
    idle_den = max(int(teacher_idle.sum().item()), 1)
    return {
        "n": float(total),
        "accuracy": float((pred == target).float().mean().item()),
        "teacher_nonidle_frac": float(teacher_nonidle.float().mean().item()),
        "pred_nonidle_frac": float((pred != 0).float().mean().item()),
        "false_noop_rate": float(false_noop.float().sum().item() / nonidle_den),
        "false_intervention_rate": float(
            false_intervention.float().sum().item() / idle_den
        ),
    }
