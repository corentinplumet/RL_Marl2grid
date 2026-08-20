"""Losses for teacher-student behavior cloning."""

from __future__ import annotations

from typing import Dict, MutableMapping

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


def masked_centered_utility_huber_loss(
    logits: th.Tensor,
    utility: th.Tensor,
    trainable_mask: th.Tensor,
    *,
    beta: float = 0.5,
    min_candidates: int = 2,
    eps: float = 1e-6,
) -> th.Tensor:
    """Regress candidate scores onto per-state standardized utilities.

    Actor logits are identifiable only up to a state-dependent additive
    constant. Centering both predictions and targets under the safe-candidate
    mask removes that nuisance degree of freedom. Standardizing the target
    keeps one high-rho state from dominating solely because its utility range
    is numerically larger.
    """
    if logits.shape != utility.shape or logits.shape != trainable_mask.shape:
        raise ValueError("logits, utility, and trainable_mask must have equal shape.")
    mask = trainable_mask.bool() & th.isfinite(utility)
    counts = mask.sum(dim=-1, keepdim=True)
    usable_rows = counts.squeeze(-1) >= int(min_candidates)
    if not bool(usable_rows.any()):
        return logits.sum() * 0.0

    safe_counts = counts.clamp_min(1).to(dtype=logits.dtype)
    mask_f = mask.to(dtype=logits.dtype)
    pred_mean = (logits * mask_f).sum(dim=-1, keepdim=True) / safe_counts
    finite_utility = th.where(mask, utility, th.zeros_like(utility))
    target_mean = finite_utility.sum(dim=-1, keepdim=True) / safe_counts
    centered_target = th.where(mask, utility - target_mean, th.zeros_like(utility))
    target_variance = (centered_target.square() * mask_f).sum(
        dim=-1, keepdim=True
    ) / safe_counts
    target_scale = target_variance.sqrt().clamp_min(float(eps))

    centered_prediction = logits - pred_mean
    standardized_target = centered_target / target_scale
    selected = mask & usable_rows.unsqueeze(-1)
    return F.smooth_l1_loss(
        centered_prediction[selected],
        standardized_target[selected],
        beta=max(float(beta), float(eps)),
        reduction="mean",
    )


def observed_unsafe_action_loss(
    logits: th.Tensor,
    utility: th.Tensor,
    trainable_mask: th.Tensor,
    observed_mask: th.Tensor,
    *,
    margin: float = 1.0,
) -> th.Tensor:
    """Push every observed unsafe action below the best safe candidate.

    Invalid and terminal simulations are deliberately absent from the utility
    regression target. Without this separate term they could nevertheless
    receive the largest deployment logit.
    """
    if not (
        logits.shape == utility.shape == trainable_mask.shape == observed_mask.shape
    ):
        raise ValueError(
            "logits, utility, trainable_mask, and observed_mask must have equal shape."
        )
    safe = trainable_mask.bool() & th.isfinite(utility)
    unsafe = observed_mask.bool() & ~trainable_mask.bool()
    usable_rows = safe.any(dim=-1) & unsafe.any(dim=-1)
    if not bool(usable_rows.any()):
        return logits.sum() * 0.0

    target_scores = utility.masked_fill(~safe, float("-inf"))
    best_safe_index = target_scores.argmax(dim=-1)
    best_safe_logit = logits.gather(-1, best_safe_index.unsqueeze(-1))
    comparisons = logits - best_safe_logit + float(margin)
    selected = unsafe & usable_rows.unsqueeze(-1)
    return F.softplus(comparisons[selected]).mean()


UTILITY_RANKING_COUNT_KEYS = (
    "ranking_rows",
    "masked_top1_correct_n",
    "masked_top3_correct_n",
    "masked_regret_sum",
    "deployment_safe_n",
    "deployment_best_correct_n",
    "unsafe_comparison_rows",
    "unsafe_outranks_best_n",
)


def utility_ranking_counts(
    logits: th.Tensor,
    utility: th.Tensor,
    trainable_mask: th.Tensor,
    observed_mask: th.Tensor,
    *,
    min_candidates: int = 2,
) -> Dict[str, float]:
    """Return additive ranking and unsafe-action diagnostics."""
    if not (
        logits.shape == utility.shape == trainable_mask.shape == observed_mask.shape
    ):
        raise ValueError("Utility-ranking tensors must have equal shape.")
    safe = trainable_mask.bool() & th.isfinite(utility)
    usable = safe.sum(dim=-1) >= int(min_candidates)
    if not bool(usable.any()):
        return {key: 0.0 for key in UTILITY_RANKING_COUNT_KEYS}

    safe_target = utility.masked_fill(~safe, float("-inf"))
    safe_prediction = logits.masked_fill(~safe, float("-inf"))
    target_best = safe_target.argmax(dim=-1)
    predicted_safe_best = safe_prediction.argmax(dim=-1)
    max_k = min(3, int(logits.shape[-1]))
    safe_topk = safe_prediction.topk(k=max_k, dim=-1).indices
    masked_top3 = (safe_topk == target_best.unsqueeze(-1)).any(dim=-1)
    best_utility = safe_target.gather(-1, target_best.unsqueeze(-1)).squeeze(-1)
    predicted_utility = safe_target.gather(
        -1, predicted_safe_best.unsqueeze(-1)
    ).squeeze(-1)

    deployment_action = logits.argmax(dim=-1)
    deployment_safe = safe.gather(-1, deployment_action.unsqueeze(-1)).squeeze(-1)
    deployment_best = deployment_action == target_best

    unsafe = observed_mask.bool() & ~trainable_mask.bool()
    unsafe_rows = usable & unsafe.any(dim=-1)
    max_unsafe_logit = logits.masked_fill(~unsafe, float("-inf")).max(dim=-1).values
    best_safe_logit = logits.gather(-1, target_best.unsqueeze(-1)).squeeze(-1)
    unsafe_outranks = unsafe_rows & (max_unsafe_logit >= best_safe_logit)

    return {
        "ranking_rows": float(usable.sum().item()),
        "masked_top1_correct_n": float(
            ((predicted_safe_best == target_best) & usable).sum().item()
        ),
        "masked_top3_correct_n": float((masked_top3 & usable).sum().item()),
        "masked_regret_sum": float(
            (best_utility[usable] - predicted_utility[usable]).sum().item()
        ),
        "deployment_safe_n": float((deployment_safe & usable).sum().item()),
        "deployment_best_correct_n": float((deployment_best & usable).sum().item()),
        "unsafe_comparison_rows": float(unsafe_rows.sum().item()),
        "unsafe_outranks_best_n": float(unsafe_outranks.sum().item()),
    }


def empty_utility_ranking_counts() -> Dict[str, float]:
    return {key: 0.0 for key in UTILITY_RANKING_COUNT_KEYS}


def merge_utility_ranking_counts(
    destination: MutableMapping[str, float], counts: Dict[str, float]
) -> None:
    for key in UTILITY_RANKING_COUNT_KEYS:
        destination[key] = float(destination.get(key, 0.0)) + float(
            counts.get(key, 0.0)
        )


def finalize_utility_ranking_counts(counts: Dict[str, float]) -> Dict[str, float]:
    rows = float(counts.get("ranking_rows", 0.0))
    unsafe_rows = float(counts.get("unsafe_comparison_rows", 0.0))
    return {
        "ranking_rows": rows,
        "masked_top1_accuracy": _safe_rate(
            counts.get("masked_top1_correct_n", 0.0), rows
        ),
        "masked_top3_accuracy": _safe_rate(
            counts.get("masked_top3_correct_n", 0.0), rows
        ),
        "masked_mean_regret": _safe_rate(counts.get("masked_regret_sum", 0.0), rows),
        "deployment_safe_rate": _safe_rate(counts.get("deployment_safe_n", 0.0), rows),
        "deployment_best_accuracy": _safe_rate(
            counts.get("deployment_best_correct_n", 0.0), rows
        ),
        "unsafe_comparison_rows": unsafe_rows,
        "unsafe_outrank_rate": _safe_rate(
            counts.get("unsafe_outranks_best_n", 0.0), unsafe_rows
        ),
    }


def soft_label_kl_loss(
    student_logits: th.Tensor,
    teacher_policy_logits: th.Tensor,
    target: th.Tensor,
    was_overwritten: th.Tensor,
    *,
    temperature: float = 1.0,
) -> th.Tensor:
    """KL(q_teacher || pi_student) for optional soft-label distillation.

    For states where the heuristic did not overwrite the base policy, the target
    distribution is the teacher base-policy softmax. When the heuristic
    overwrote the action, the target is the final hard teacher action, usually
    action 0. This follows the roadmap's "heuristic-adjusted" soft target.
    """
    temperature = max(float(temperature), 1e-6)
    target = target.long()
    was_overwritten = was_overwritten.bool()

    teacher_probs = F.softmax(teacher_policy_logits / temperature, dim=-1)
    hard_probs = F.one_hot(target, num_classes=student_logits.shape[-1]).to(
        dtype=student_logits.dtype,
        device=student_logits.device,
    )
    soft_target = th.where(was_overwritten.unsqueeze(-1), hard_probs, teacher_probs)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, soft_target, reduction="batchmean") * (
        temperature * temperature
    )


CLASSIFICATION_COUNT_KEYS = (
    "n",
    "correct_n",
    "idle_n",
    "nonidle_n",
    "idle_correct_n",
    "nonidle_correct_n",
    "nonidle_top3_correct_n",
    "nonidle_top5_correct_n",
    "nonidle_rank_top1_correct_n",
    "nonidle_rank_top3_correct_n",
    "nonidle_rank_top5_correct_n",
    "pred_nonidle_n",
    "false_noop_n",
    "false_intervention_n",
)


def classification_counts(
    logits: th.Tensor,
    target: th.Tensor,
) -> Dict[str, float]:
    """Return additive classification counts for exact dataset aggregation.

    Conditional rates such as non-idle accuracy cannot be averaged using total
    batch size: batches with no non-idle labels would bias the result toward
    zero. Keeping integer-like counts makes aggregation exact across arbitrary
    shard and batch boundaries.
    """
    if logits.dim() < 2:
        raise ValueError("logits must have an action dimension.")
    target = target.long()
    if tuple(logits.shape[:-1]) != tuple(target.shape):
        raise ValueError(
            f"logits shape {tuple(logits.shape)} and target shape "
            f"{tuple(target.shape)} are incompatible."
        )

    n_actions = int(logits.shape[-1])
    if n_actions < 1:
        raise ValueError("logits must contain at least one action.")

    pred = th.argmax(logits, dim=-1)
    teacher_nonidle = target != 0
    teacher_idle = target == 0
    false_noop = (pred == 0) & teacher_nonidle
    false_intervention = (pred != 0) & teacher_idle

    max_k = min(5, n_actions)
    top_indices = th.topk(logits, k=max_k, dim=-1).indices

    def topk_correct(k: int) -> th.Tensor:
        width = min(int(k), n_actions)
        return (top_indices[..., :width] == target.unsqueeze(-1)).any(dim=-1)

    if n_actions > 1:
        nonidle_width = n_actions - 1
        nonidle_max_k = min(5, nonidle_width)
        nonidle_top_indices = (
            th.topk(logits[..., 1:], k=nonidle_max_k, dim=-1).indices + 1
        )

        def nonidle_rank_correct(k: int) -> th.Tensor:
            width = min(int(k), nonidle_width)
            return (nonidle_top_indices[..., :width] == target.unsqueeze(-1)).any(
                dim=-1
            )

    else:

        def nonidle_rank_correct(k: int) -> th.Tensor:
            del k
            return th.zeros_like(teacher_nonidle)

    return {
        "n": float(target.numel()),
        "correct_n": float((pred == target).sum().item()),
        "idle_n": float(teacher_idle.sum().item()),
        "nonidle_n": float(teacher_nonidle.sum().item()),
        "idle_correct_n": float(((pred == target) & teacher_idle).sum().item()),
        "nonidle_correct_n": float(((pred == target) & teacher_nonidle).sum().item()),
        "nonidle_top3_correct_n": float(
            (topk_correct(3) & teacher_nonidle).sum().item()
        ),
        "nonidle_top5_correct_n": float(
            (topk_correct(5) & teacher_nonidle).sum().item()
        ),
        "nonidle_rank_top1_correct_n": float(
            (nonidle_rank_correct(1) & teacher_nonidle).sum().item()
        ),
        "nonidle_rank_top3_correct_n": float(
            (nonidle_rank_correct(3) & teacher_nonidle).sum().item()
        ),
        "nonidle_rank_top5_correct_n": float(
            (nonidle_rank_correct(5) & teacher_nonidle).sum().item()
        ),
        "pred_nonidle_n": float((pred != 0).sum().item()),
        "false_noop_n": float(false_noop.sum().item()),
        "false_intervention_n": float(false_intervention.sum().item()),
    }


def merge_classification_counts(
    destination: MutableMapping[str, float],
    counts: Dict[str, float],
) -> None:
    """Add one batch of :func:`classification_counts` in place."""
    for key in CLASSIFICATION_COUNT_KEYS:
        destination[key] = float(destination.get(key, 0.0)) + float(
            counts.get(key, 0.0)
        )


def empty_classification_counts() -> Dict[str, float]:
    return {key: 0.0 for key in CLASSIFICATION_COUNT_KEYS}


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")


def finalize_classification_counts(counts: Dict[str, float]) -> Dict[str, float]:
    """Convert additive counts into user-facing exact rates."""
    n = float(counts.get("n", 0.0))
    idle_n = float(counts.get("idle_n", 0.0))
    nonidle_n = float(counts.get("nonidle_n", 0.0))
    intervention_correct = (
        float(counts.get("idle_correct_n", 0.0))
        + nonidle_n
        - float(counts.get("false_noop_n", 0.0))
    )
    return {
        "n": n,
        "idle_n": idle_n,
        "nonidle_n": nonidle_n,
        "accuracy": _safe_rate(counts.get("correct_n", 0.0), n),
        "action0_accuracy": _safe_rate(counts.get("idle_correct_n", 0.0), idle_n),
        "nonidle_accuracy": _safe_rate(counts.get("nonidle_correct_n", 0.0), nonidle_n),
        "nonidle_top3_accuracy": _safe_rate(
            counts.get("nonidle_top3_correct_n", 0.0), nonidle_n
        ),
        "nonidle_top5_accuracy": _safe_rate(
            counts.get("nonidle_top5_correct_n", 0.0), nonidle_n
        ),
        "nonidle_rank_top1_accuracy": _safe_rate(
            counts.get("nonidle_rank_top1_correct_n", 0.0), nonidle_n
        ),
        "nonidle_rank_top3_accuracy": _safe_rate(
            counts.get("nonidle_rank_top3_correct_n", 0.0), nonidle_n
        ),
        "nonidle_rank_top5_accuracy": _safe_rate(
            counts.get("nonidle_rank_top5_correct_n", 0.0), nonidle_n
        ),
        "intervention_accuracy": _safe_rate(intervention_correct, n),
        "teacher_nonidle_frac": _safe_rate(nonidle_n, n),
        "pred_nonidle_frac": _safe_rate(counts.get("pred_nonidle_n", 0.0), n),
        "false_noop_rate": _safe_rate(counts.get("false_noop_n", 0.0), nonidle_n),
        "false_intervention_rate": _safe_rate(
            counts.get("false_intervention_n", 0.0), idle_n
        ),
    }


def classification_metrics(logits: th.Tensor, target: th.Tensor) -> Dict[str, float]:
    """Backward-compatible single-batch classification metrics."""
    return finalize_classification_counts(classification_counts(logits, target))
