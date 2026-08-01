import math
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
    affected_masks: Optional[Dict[str, th.Tensor]] = None


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
    """Learn action-specific typed contexts from heterogeneous graph nodes."""

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
        prior_bias: float = 2.0,
        action_chunk_size: int = 64,
        node_types: Optional[th.Tensor] = None,
        node_type_ids: Optional[Dict[str, int]] = None,
        typed_metadata: Optional[TypedActionMetadata] = None,
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
        self.prior_bias = float(prior_bias)
        self.action_chunk_size = int(action_chunk_size)

        if self.scope not in {"affected", "soft_prior", "all"}:
            raise ValueError(
                "candidate_action_attention_scope currently supports only "
                f"'affected', 'soft_prior', or 'all', got {scope!r}."
            )
        if self.heads < 1:
            raise ValueError(
                "candidate_action_attention_heads must be at least 1."
            )
        if self.attention_dim < 1:
            raise ValueError(
                "candidate_action_attention_dim must be non-negative."
            )
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
        if self.scope == "all" and self.query_mode == "global_only":
            raise ValueError(
                "candidate_action_attention_scope=all requires an "
                "action-specific query. Use global_action_features or "
                "learned_action instead of global_only."
            )
        if self.normalizer != "softmax":
            raise ValueError(
                "candidate_action_attention_normalizer currently supports only "
                f"'softmax', got {normalizer!r}."
            )
        if not math.isfinite(self.prior_bias):
            raise ValueError("candidate_action_attention_prior_bias must be finite.")
        if self.action_chunk_size < 1:
            raise ValueError(
                "candidate_action_attention_chunk_size must be at least 1."
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
        if self.scope in {"soft_prior", "all"}:
            if (
                node_types is None
                or node_type_ids is None
                or typed_metadata is None
            ):
                raise ValueError(
                    "Dense candidate attention requires node types, node type "
                    "IDs, and typed action metadata."
                )
            self._register_dense_metadata(
                node_types,
                node_type_ids,
                typed_metadata,
            )
        self._reset_parameters()

    def _register_dense_metadata(
        self,
        node_types: th.Tensor,
        node_type_ids: Dict[str, int],
        typed_metadata: TypedActionMetadata,
    ) -> None:
        node_types = th.as_tensor(node_types, dtype=th.long).clone()
        if node_types.dim() != 1:
            raise ValueError("candidate attention node_types must be 1D.")
        self.register_buffer("node_types", node_types)
        n_nodes = int(node_types.numel())

        for node_type, indices, mask in typed_metadata:
            if node_type not in node_type_ids:
                raise ValueError(
                    f"Missing graph node type ID for candidate type {node_type!r}."
                )
            type_indices = th.nonzero(
                node_types == int(node_type_ids[node_type]),
                as_tuple=False,
            ).flatten()
            row_to_type_position = th.full((n_nodes,), -1, dtype=th.long)
            if type_indices.numel():
                row_to_type_position[type_indices] = th.arange(
                    type_indices.numel(),
                    dtype=th.long,
                )

            safe_indices = indices.clamp_min(0).to(dtype=th.long)
            local_positions = row_to_type_position[safe_indices]
            active_mask = mask.bool()
            if bool(th.any(local_positions[active_mask] < 0)):
                raise ValueError(
                    f"Affected {node_type} rows do not match graph node types."
                )

            affected_mask = th.zeros(
                self.n_actions,
                int(type_indices.numel()),
                dtype=th.bool,
            )
            if bool(active_mask.any()):
                action_rows = th.arange(self.n_actions).unsqueeze(1).expand_as(
                    safe_indices
                )
                affected_mask[
                    action_rows[active_mask],
                    local_positions[active_mask],
                ] = True

            eligible_mask = th.ones_like(affected_mask)
            eligible_mask[0] = False
            self.register_buffer(f"{node_type}_all_indices", type_indices)
            self.register_buffer(f"{node_type}_affected_mask", affected_mask)
            self.register_buffer(f"{node_type}_eligible_mask", eligible_mask)

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

    def _attend_to_dense_type(
        self,
        node_type: str,
        queries: th.Tensor,
        node_embeddings: th.Tensor,
        *,
        return_weights: bool,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        type_indices = getattr(self, f"{node_type}_all_indices")
        affected_mask = getattr(self, f"{node_type}_affected_mask")
        eligible_mask = getattr(self, f"{node_type}_eligible_mask")
        batch_size = int(node_embeddings.shape[0])
        width = int(type_indices.numel())
        if width == 0:
            empty_weights = (
                node_embeddings.new_zeros(
                    batch_size,
                    self.n_actions,
                    self.heads,
                    0,
                )
                if return_weights
                else None
            )
            return (
                node_embeddings.new_zeros(
                    batch_size,
                    self.n_actions,
                    self.node_dim,
                ),
                empty_weights,
            )

        typed_nodes = node_embeddings.index_select(1, type_indices)
        keys = self.key_projection(typed_nodes).reshape(
            batch_size,
            width,
            self.heads,
            self.attention_dim,
        )
        keys = keys.permute(0, 2, 1, 3)
        values = self.value_projection(typed_nodes)
        context_chunks = []
        weight_chunks = [] if return_weights else None
        for start in range(0, self.n_actions, self.action_chunk_size):
            end = min(start + self.action_chunk_size, self.n_actions)
            query_chunk = queries[:, start:end]
            additive_features = th.tanh(
                query_chunk.unsqueeze(3) + keys.unsqueeze(1)
            )
            scores = th.einsum(
                "bahwd,hd->bahw",
                additive_features,
                self.score_vector,
            )
            scores = scores / self.temperature
            if self.scope == "soft_prior":
                scores = scores + (
                    self.prior_bias
                    * affected_mask[start:end]
                    .to(dtype=scores.dtype)
                    .unsqueeze(0)
                    .unsqueeze(2)
                )
            weights = self._masked_softmax(
                scores,
                eligible_mask[start:end],
            )
            head_context = th.einsum("bahw,bwd->bahd", weights, values)
            context_chunks.append(head_context.mean(dim=2))
            if return_weights:
                weight_chunks.append(weights)

        all_contexts = th.cat(context_chunks, dim=1)
        all_weights = (
            th.cat(weight_chunks, dim=1) if return_weights else None
        )
        return all_contexts, all_weights

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
        affected_by_type = {} if return_weights else None
        for node_type, indices, mask in typed_metadata:
            if self.scope == "affected":
                context, weights = self._attend_to_gathered_nodes(
                    queries,
                    node_embeddings,
                    indices,
                    mask,
                )
                diagnostic_indices = indices
                eligible_mask = mask.bool()
                affected_mask = mask.bool()
            else:
                context, weights = self._attend_to_dense_type(
                    node_type,
                    queries,
                    node_embeddings,
                    return_weights=return_weights,
                )
                type_indices = getattr(self, f"{node_type}_all_indices")
                diagnostic_indices = type_indices.unsqueeze(0).expand(
                    self.n_actions,
                    -1,
                )
                eligible_mask = getattr(
                    self,
                    f"{node_type}_eligible_mask",
                )
                affected_mask = getattr(
                    self,
                    f"{node_type}_affected_mask",
                )
            contexts.append(context)
            if return_weights:
                weights_by_type[node_type] = weights
                indices_by_type[node_type] = diagnostic_indices
                masks_by_type[node_type] = eligible_mask
                affected_by_type[node_type] = affected_mask

        return CandidateAttentionOutput(
            context=th.cat(contexts, dim=-1),
            weights=weights_by_type,
            node_indices=indices_by_type,
            eligible_masks=masks_by_type,
            affected_masks=affected_by_type,
        )
