import numpy as np
import pytest
import torch as th

from teacher_student.dataset import make_joint_agent_minibatches
from teacher_student.losses import (
    classification_counts,
    classification_metrics,
    empty_classification_counts,
    finalize_utility_ranking_counts,
    finalize_classification_counts,
    masked_centered_utility_huber_loss,
    merge_classification_counts,
    observed_unsafe_action_loss,
    utility_ranking_counts,
)


def test_classification_counts_aggregate_conditional_rates_exactly():
    aggregate = empty_classification_counts()
    merge_classification_counts(
        aggregate,
        classification_counts(
            th.tensor([[5.0, 0.0, 0.0]]),
            th.tensor([0]),
        ),
    )
    merge_classification_counts(
        aggregate,
        classification_counts(
            th.tensor([[0.0, 5.0, 1.0]]),
            th.tensor([1]),
        ),
    )

    metrics = finalize_classification_counts(aggregate)
    assert metrics["n"] == 2
    assert metrics["idle_n"] == 1
    assert metrics["nonidle_n"] == 1
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["action0_accuracy"] == pytest.approx(1.0)
    assert metrics["nonidle_accuracy"] == pytest.approx(1.0)
    assert metrics["false_noop_rate"] == pytest.approx(0.0)
    assert metrics["false_intervention_rate"] == pytest.approx(0.0)


def test_nonidle_topk_metrics_separate_gate_and_action_ranking():
    logits = th.tensor(
        [
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        ]
    )
    target = th.tensor([1, 3, 5])

    metrics = classification_metrics(logits, target)
    assert metrics["nonidle_accuracy"] == pytest.approx(0.0)
    assert metrics["nonidle_top3_accuracy"] == pytest.approx(1.0 / 3.0)
    assert metrics["nonidle_top5_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["nonidle_rank_top1_accuracy"] == pytest.approx(1.0 / 3.0)
    assert metrics["nonidle_rank_top3_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["nonidle_rank_top5_accuracy"] == pytest.approx(1.0)
    assert metrics["false_noop_rate"] == pytest.approx(1.0)


def test_joint_agent_minibatches_cover_each_unbalanced_epoch_once():
    targets = {
        "agent_0": np.asarray([0, 1, 0, 2, 0, 3, 0, 4, 0, 5]),
        "agent_1": np.asarray([1, 0, 2, 0, 3, 0, 4, 0, 5, 0]),
    }
    batches = list(
        make_joint_agent_minibatches(
            targets,
            batch_size=4,
            rng=np.random.default_rng(7),
            balanced_nonidle_frac=0.0,
        )
    )

    assert len(batches) == 3
    assert all(set(batch) == set(targets) for batch in batches)
    for agent in targets:
        observed = np.concatenate([batch[agent] for batch in batches])
        assert sorted(observed.tolist()) == list(range(len(targets[agent])))


def test_masked_utility_regression_is_shift_invariant_and_ignores_unsafe():
    utility = th.tensor([[0.0, 0.4, -0.2, 99.0]])
    mask = th.tensor([[True, True, True, False]])
    perfect = th.tensor([[2.0, 4.0, 1.0, -100.0]])
    shifted = perfect + 123.0
    wrong = th.tensor([[2.0, 0.0, 4.0, -100.0]])

    perfect_loss = masked_centered_utility_huber_loss(perfect, utility, mask)
    shifted_loss = masked_centered_utility_huber_loss(shifted, utility, mask)
    wrong_loss = masked_centered_utility_huber_loss(wrong, utility, mask)

    assert shifted_loss.item() == pytest.approx(perfect_loss.item(), abs=1e-5)
    assert wrong_loss.item() > perfect_loss.item()


def test_unsafe_loss_and_ranking_metrics_detect_terminal_argmax():
    utility = th.tensor([[0.0, 0.4, -0.2, 0.7], [float("nan")] * 4])
    trainable = th.tensor([[True, True, True, False], [False, False, False, False]])
    observed = th.tensor([[True, True, True, True], [True, False, False, False]])
    safe_logits = th.tensor([[0.0, 3.0, -1.0, -4.0], [5.0, 0.0, 0.0, 0.0]])
    unsafe_logits = th.tensor([[0.0, 3.0, -1.0, 7.0], [5.0, 0.0, 0.0, 0.0]])

    assert observed_unsafe_action_loss(
        unsafe_logits, utility, trainable, observed
    ) > observed_unsafe_action_loss(safe_logits, utility, trainable, observed)

    metrics = finalize_utility_ranking_counts(
        utility_ranking_counts(unsafe_logits, utility, trainable, observed)
    )
    assert metrics["masked_top1_accuracy"] == pytest.approx(1.0)
    assert metrics["deployment_safe_rate"] == pytest.approx(0.0)
    assert metrics["deployment_best_accuracy"] == pytest.approx(0.0)
    assert metrics["unsafe_outrank_rate"] == pytest.approx(1.0)
