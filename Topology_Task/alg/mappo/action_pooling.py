from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch as th
from torch import nn


TypedActionMetadata = Sequence[Tuple[str, th.Tensor, th.Tensor]]


@dataclass
class CandidateAttentionOutput:
    """Action-conditioned contexts and optional attention diagnostics."""

    context: th.Tensor
    weights: Optional[Dict[str, th.Tensor]] = None
    node_indices: Optional[Dict[str, th.Tensor]] = None
    eligible_masks: Optional[Dict[str, th.Tensor]] = None


class CandidateActionMeanPool(nn.Module):
    """The original uniform candidate-node pooling modes."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"mean", "typed_mean"}:
            raise ValueError(
                "CandidateActionMeanPool mode must be 'mean' or "
                f"'typed_mean', got {mode!r}."
            )

    @staticmethod
    def _masked_mean(
        node_embeddings: th.Tensor,
        indices: th.Tensor,
        mask: th.Tensor,
    ) -> th.Tensor:
        safe_indices = indices.clamp_min(0)
        gathered = node_embeddings[:, safe_indices, :]
        weights = mask.to(dtype=node_embeddings.dtype).unsqueeze(0).unsqueeze(-1)
        summed = (gathered * weights).sum(dim=2)
        denominator = weights.sum(dim=2).clamp_min(1.0)
        return summed / denominator

    def forward(
        self,
        node_embeddings: th.Tensor,
        typed_metadata: TypedActionMetadata,
    ) -> CandidateAttentionOutput:
        if self.mode == "typed_mean":
            context = th.cat(
                [
                    self._masked_mean(node_embeddings, indices, mask)
                    for _, indices, mask in typed_metadata
                ],
                dim=-1,
            )
        else:
            all_indices = th.cat(
                [indices for _, indices, _ in typed_metadata], dim=1
            )
            all_masks = th.cat([mask for _, _, mask in typed_metadata], dim=1)
            context = self._masked_mean(
                node_embeddings,
                all_indices,
                all_masks,
            )
        return CandidateAttentionOutput(context=context)


class CandidateActionAttentionPool(nn.Module):
    """Learn action-specific typed pooling over hardcoded affected nodes."""

    def __init__(
        self,
        *,
        graph_dim: int,
        node_dim: int,
        n_actions: int,
        action_feature_dim: int,
        scope: str = "affected",
        heads: int = 1,
        attention_dim: int = 0,
        temperature: float = 1.0,
        query_mode: str = "global_action_features",
        normalizer: str = "softmax",
    ) -> None:
        super().__init__()
        self.graph_dim = int(graph_dim)
        self.node_dim = int(node_dim)
        self.n_actions = int(n_actions)
        self.action_feature_dim = int(action_feature_dim)
        self.scope = str(scope).lower()
        self.heads = int(heads)
        self.attention_dim = int(attention_dim) or self.node_dim
        self.temperature = float(temperature)
        self.query_mode = str(query_mode).lower()
        self.normalizer = str(normalizer).lower()

        if self.scope != "affected":
            raise ValueError(
                "candidate_action_attention_scope currently supports only "
                f"'affected', got {scope!r}."
            )
        if self.heads < 1:
            raise ValueError("candidate_action_attention_heads must be at least 1.")
        if self.attention_dim < 1:
            raise ValueError("candidate_action_attention_dim must be non-negative.")
        if self.temperature <= 0.0:
            raise ValueError(
                "candidate_action_attention_temperature must be greater than 0."
            )
        if self.query_mode not in {
            "global_action_features",
            "learned_action",
            "global_only",
        }:
            raise ValueError(
                "candidate_action_attention_query must be "
                "'global_action_features', 'learned_action', or 'global_only', "
                f"got {query_mode!r}."
            )
        if self.normalizer != "softmax":
            raise ValueError(
                "candidate_action_attention_normalizer currently supports only "
                f"'softmax', got {normalizer!r}."
            )

        projected_query_dim = self.heads * self.attention_dim
        if self.query_mode == "global_action_features":
            self.query_projection = nn.Linear(
                self.graph_dim + self.action_feature_dim,
                projected_query_dim,
            )
            self.action_embedding = None
        elif self.query_mode == "learned_action":
            self.query_projection = nn.Linear(
                self.graph_dim,
                projected_query_dim,
            )
            self.action_embedding = nn.Embedding(
                self.n_actions,
                projected_query_dim,
            )
        else:
            self.query_projection = nn.Linear(
                self.graph_dim,
                projected_query_dim,
            )
            self.action_embedding = None

        self.key_projection = nn.Linear(
            self.node_dim,
            projected_query_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            self.node_dim,
            self.node_dim,
            bias=False,
        )
        self.score_vector = nn.Parameter(
            th.empty(self.heads, self.attention_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.score_vector, mean=0.0, std=1e-3)
        with th.no_grad():
            self.value_projection.weight.copy_(
                th.eye(
                    self.node_dim,
                    dtype=self.value_projection.weight.dtype,
                    device=self.value_projection.weight.device,
                )
            )

    def _action_queries(
        self,
        graph_embedding: th.Tensor,
        action_features: th.Tensor,
    ) -> th.Tensor:
        batch_size = int(graph_embedding.shape[0])
        if self.query_mode == "global_action_features":
            global_context = graph_embedding.unsqueeze(1).expand(
                batch_size,
                self.n_actions,
                -1,
            )
            features = action_features.unsqueeze(0).expand(batch_size, -1, -1)
            query_input = th.cat([global_context, features], dim=-1)
            projected = self.query_projection(query_input)
        else:
            projected = self.query_projection(graph_embedding).unsqueeze(1)
            if self.query_mode == "learned_action":
                action_ids = th.arange(
                    self.n_actions,
                    device=graph_embedding.device,
                )
                projected = projected + self.action_embedding(
                    action_ids
                ).unsqueeze(0)
            else:
                projected = projected.expand(batch_size, self.n_actions, -1)
        return projected.reshape(
            batch_size,
            self.n_actions,
            self.heads,
            self.attention_dim,
        )

    @staticmethod
    def _masked_softmax(scores: th.Tensor, mask: th.Tensor) -> th.Tensor:
        expanded_mask = mask.unsqueeze(0).unsqueeze(2)
        masked_scores = scores.masked_fill(~expanded_mask, -th.inf)
        has_items = expanded_mask.any(dim=-1, keepdim=True)
        safe_scores = th.where(has_items, masked_scores, th.zeros_like(scores))
        weights = th.softmax(safe_scores, dim=-1)
        weights = weights * expanded_mask.to(dtype=weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def _attend_to_gathered_nodes(
        self,
        queries: th.Tensor,
        node_embeddings: th.Tensor,
        indices: th.Tensor,
        mask: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        batch_size = int(node_embeddings.shape[0])
        safe_indices = indices.clamp_min(0)
        gathered = node_embeddings[:, safe_indices, :]
        width = int(gathered.shape[2])

        keys = self.key_projection(gathered).reshape(
            batch_size,
            self.n_actions,
            width,
            self.heads,
            self.attention_dim,
        )
        keys = keys.permute(0, 1, 3, 2, 4)
        additive_features = th.tanh(queries.unsqueeze(3) + keys)
        scores = th.einsum(
            "bahwd,hd->bahw",
            additive_features,
            self.score_vector,
        )
        scores = scores / self.temperature
        weights = self._masked_softmax(scores, mask.bool())

        values = self.value_projection(gathered)
        head_context = th.einsum("bahw,bawd->bahd", weights, values)
        context = head_context.mean(dim=2)
        return context, weights

    def forward(
        self,
        graph_embedding: th.Tensor,
        node_embeddings: th.Tensor,
        action_features: th.Tensor,
        typed_metadata: TypedActionMetadata,
        *,
        return_weights: bool = False,
    ) -> CandidateAttentionOutput:
        queries = self._action_queries(graph_embedding, action_features)
        contexts = []
        weights_by_type = {} if return_weights else None
        indices_by_type = {} if return_weights else None
        masks_by_type = {} if return_weights else None
        for node_type, indices, mask in typed_metadata:
            context, weights = self._attend_to_gathered_nodes(
                queries,
                node_embeddings,
                indices,
                mask,
            )
            contexts.append(context)
            if return_weights:
                weights_by_type[node_type] = weights
                indices_by_type[node_type] = indices
                masks_by_type[node_type] = mask.bool()

        return CandidateAttentionOutput(
            context=th.cat(contexts, dim=-1),
            weights=weights_by_type,
            node_indices=indices_by_type,
            eligible_masks=masks_by_type,
        )
