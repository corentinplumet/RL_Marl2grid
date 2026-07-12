from common.imports import *
from common.utils import get_flat_obs


class TokenTransformerEncoder(nn.Module):
    """Transformer encoder for fixed-shape Grid2Op token observations."""

    def __init__(self, token_spec: Dict[str, Any], args: Dict[str, Any]) -> None:
        super().__init__()
        self.token_feature_dim = int(token_spec["token_feature_dim"])
        self.d_model = int(getattr(args, "transformer_d_model", 128))
        self.pool = str(getattr(args, "transformer_pool", "cls")).lower()
        if self.pool not in {"cls", "mean"}:
            raise ValueError(
                f"Unsupported transformer_pool={self.pool!r}. Use 'cls' or 'mean'."
            )

        n_heads = int(getattr(args, "transformer_n_heads", 4))
        n_layers = int(getattr(args, "transformer_layers", 3))
        ff_dim = int(getattr(args, "transformer_ff_dim", 4 * self.d_model))
        dropout = float(getattr(args, "transformer_dropout", 0.05))
        activation = str(getattr(args, "transformer_activation", "gelu")).lower()
        layer_norm_eps = float(getattr(args, "transformer_layer_norm_eps", 1e-5))

        if self.d_model % n_heads != 0:
            raise ValueError(
                f"transformer_d_model={self.d_model} must be divisible by "
                f"transformer_n_heads={n_heads}."
            )
        if n_layers < 1:
            raise ValueError("transformer_layers must be at least 1.")

        self.feature_encoder = nn.Linear(self.token_feature_dim, self.d_model)
        self.token_type_embedding = nn.Embedding(
            int(token_spec.get("n_token_types", 32)), self.d_model, padding_idx=0
        )
        self.entity_type_embedding = nn.Embedding(
            int(token_spec.get("n_entity_types", 16)), self.d_model, padding_idx=0
        )

        self.use_entity_id_embeddings = bool(
            getattr(args, "transformer_use_entity_id_embeddings", True)
        )
        self.use_substation_embeddings = bool(
            getattr(args, "transformer_use_substation_embeddings", True)
        )
        self.use_bus_embeddings = bool(
            getattr(args, "transformer_use_bus_embeddings", True)
        )
        self.use_agent_embeddings = bool(
            getattr(args, "transformer_use_agent_embeddings", True)
        )

        if self.use_entity_id_embeddings:
            self.entity_id_embedding = nn.Embedding(
                int(token_spec.get("max_entity_id", 1)) + 2,
                self.d_model,
                padding_idx=0,
            )
        else:
            self.entity_id_embedding = None

        if self.use_substation_embeddings:
            self.substation_embedding = nn.Embedding(
                int(token_spec.get("n_sub", 1)) + 1,
                self.d_model,
                padding_idx=0,
            )
        else:
            self.substation_embedding = None

        if self.use_bus_embeddings:
            self.bus_embedding = nn.Embedding(
                int(token_spec.get("n_busbar", 2)) + 1,
                self.d_model,
                padding_idx=0,
            )
        else:
            self.bus_embedding = None

        if self.use_agent_embeddings:
            self.agent_embedding = nn.Embedding(
                int(token_spec.get("n_agents", 1)) + 1,
                self.d_model,
                padding_idx=0,
            )
        else:
            self.agent_embedding = None

        self.input_norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        encoder_layer = self._make_encoder_layer(
            d_model=self.d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_dim = self.d_model

    def forward(self, token_obs: Dict[str, th.Tensor]) -> th.Tensor:
        features = token_obs["features"]
        unbatched = features.dim() == 2
        if unbatched:
            features = features.unsqueeze(0)

        feature_mask = token_obs.get("feature_mask")
        if feature_mask is not None:
            if feature_mask.dim() == 2:
                feature_mask = feature_mask.unsqueeze(0)
            features = features * feature_mask.to(features.dtype)

        token_mask = token_obs.get("token_mask")
        if token_mask is None:
            token_mask = th.ones(features.shape[:2], dtype=features.dtype, device=features.device)
        elif token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0)
        token_mask = token_mask.to(features.device)

        x = self.feature_encoder(features)
        x = x + self.token_type_embedding(
            self._bounded_ids(token_obs, "type_ids", self.token_type_embedding.num_embeddings)
        )
        x = x + self.entity_type_embedding(
            self._bounded_ids(
                token_obs,
                "entity_type_ids",
                self.entity_type_embedding.num_embeddings,
            )
        )

        if self.entity_id_embedding is not None:
            x = x + self.entity_id_embedding(
                self._offset_ids(
                    token_obs,
                    "entity_ids",
                    self.entity_id_embedding.num_embeddings,
                )
            )

        if self.substation_embedding is not None:
            for field in ("substation_ids", "line_or_sub_ids", "line_ex_sub_ids"):
                x = x + self.substation_embedding(
                    self._offset_ids(
                        token_obs,
                        field,
                        self.substation_embedding.num_embeddings,
                    )
                )

        if self.bus_embedding is not None:
            for field in ("bus_ids", "line_or_bus_ids", "line_ex_bus_ids"):
                x = x + self.bus_embedding(
                    self._offset_ids(
                        token_obs,
                        field,
                        self.bus_embedding.num_embeddings,
                    )
                )

        if self.agent_embedding is not None:
            x = x + self.agent_embedding(
                self._offset_ids(
                    token_obs,
                    "agent_ids",
                    self.agent_embedding.num_embeddings,
                )
            )

        x = self.input_norm(x)
        padding_mask = token_mask <= 0
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        if self.pool == "cls":
            pooled = x[:, 0]
        else:
            keep = token_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return pooled.squeeze(0) if unbatched else pooled

    def _make_encoder_layer(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
    ) -> nn.TransformerEncoderLayer:
        kwargs = dict(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        try:
            return nn.TransformerEncoderLayer(**kwargs, norm_first=True)
        except TypeError:
            return nn.TransformerEncoderLayer(**kwargs)

    def _raw_ids(self, token_obs: Dict[str, th.Tensor], field: str) -> th.Tensor:
        if field in token_obs:
            ids = token_obs[field]
        else:
            ids = th.zeros(
                token_obs["features"].shape[:-1],
                dtype=th.long,
                device=token_obs["features"].device,
            )
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.to(device=token_obs["features"].device).long()

    def _bounded_ids(
        self,
        token_obs: Dict[str, th.Tensor],
        field: str,
        n_embeddings: int,
    ) -> th.Tensor:
        ids = self._raw_ids(token_obs, field)
        return ids.clamp(min=0, max=n_embeddings - 1)

    def _offset_ids(
        self,
        token_obs: Dict[str, th.Tensor],
        field: str,
        n_embeddings: int,
    ) -> th.Tensor:
        ids = self._raw_ids(token_obs, field)
        ids = ids + 1
        return ids.clamp(min=0, max=n_embeddings - 1)


class TokenAndFlatEncoder(nn.Module):
    """Token transformer with optional flat-observation concatenation."""

    def __init__(
        self,
        token_spec: Dict[str, Any],
        flat_dim: int,
        args: Dict[str, Any],
        use_flat: bool = False,
        token_encoder: Optional[TokenTransformerEncoder] = None,
    ) -> None:
        super().__init__()
        self.use_flat = bool(use_flat)
        self.token_encoder = token_encoder or TokenTransformerEncoder(token_spec, args)
        self.out_dim = self.token_encoder.out_dim + (int(flat_dim) if self.use_flat else 0)

    def forward(self, obs: Dict[str, th.Tensor], token_key: str = "tokens") -> th.Tensor:
        token_embedding = self.token_encoder(obs[token_key])
        if not self.use_flat:
            return token_embedding

        flat = get_flat_obs(obs)
        if flat.dim() == 1 and token_embedding.dim() == 1:
            return th.cat([token_embedding, flat], dim=-1)
        if flat.dim() == 1:
            flat = flat.unsqueeze(0)
        if token_embedding.dim() == 1:
            token_embedding = token_embedding.unsqueeze(0)
        encoded = th.cat([token_embedding, flat], dim=-1)
        return encoded.squeeze(0) if obs[token_key]["features"].dim() == 2 else encoded

