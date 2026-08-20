from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th


ACTION_FEATURE_NAMES = (
    "is_do_nothing",
    "has_busbar_context",
    "has_line",
    "has_load",
    "has_generator",
    "n_substations_scaled",
    "n_topology_changes_scaled",
    "n_line_status_changes_scaled",
    "n_line_endpoint_changes_scaled",
    "n_generator_changes_scaled",
    "n_load_changes_scaled",
    "n_other_changes_scaled",
)

ACTION_DELTA_FEATURE_NAMES = (
    "operation_change_bus",
    "operation_set_bus",
    "operation_change_line_status",
    "operation_connect_line",
    "operation_disconnect_line",
    "object_line_origin",
    "object_line_extremity",
    "object_generator",
    "object_load",
    "object_line_status",
    "has_target_busbar",
    "target_disconnected",
    "target_bus_scaled",
)


@dataclass
class ActionGraphMetadata:
    """Static mapping from discrete action IDs to physical graph rows."""

    action_features: th.Tensor
    busbar_indices: th.Tensor
    busbar_mask: th.Tensor
    line_indices: th.Tensor
    line_mask: th.Tensor
    load_indices: th.Tensor
    load_mask: th.Tensor
    generator_indices: th.Tensor
    generator_mask: th.Tensor
    substation_ids: th.Tensor
    substation_mask: th.Tensor
    delta_object_indices: th.Tensor
    delta_modification_mask: th.Tensor
    delta_target_busbar_indices: th.Tensor
    delta_target_busbar_mask: th.Tensor
    delta_context_busbar_indices: th.Tensor
    delta_context_busbar_mask: th.Tensor
    delta_operation_features: th.Tensor
    is_do_nothing: th.Tensor
    original_action_ids: th.Tensor
    action_feature_names: Tuple[str, ...] = ACTION_FEATURE_NAMES
    action_delta_feature_names: Tuple[str, ...] = ACTION_DELTA_FEATURE_NAMES

    @property
    def n_actions(self) -> int:
        return int(self.action_features.shape[0])

    @property
    def action_feature_dim(self) -> int:
        return int(self.action_features.shape[1])

    def validate(self, n_nodes: Optional[int] = None) -> None:
        if self.action_features.dim() != 2:
            raise ValueError("action_features must have shape [n_actions, dim].")
        if not bool(th.isfinite(self.action_features).all()):
            raise ValueError("action_features contains non-finite values.")

        n_actions = self.n_actions
        paired_fields = (
            ("busbar", self.busbar_indices, self.busbar_mask),
            ("line", self.line_indices, self.line_mask),
            ("load", self.load_indices, self.load_mask),
            ("generator", self.generator_indices, self.generator_mask),
            ("substation", self.substation_ids, self.substation_mask),
        )
        for name, indices, mask in paired_fields:
            if indices.dim() != 2 or tuple(indices.shape) != tuple(mask.shape):
                raise ValueError(
                    f"{name} indices and mask must have the same 2D shape."
                )
            if int(indices.shape[0]) != n_actions:
                raise ValueError(
                    f"{name} metadata has {indices.shape[0]} actions, "
                    f"expected {n_actions}."
                )
            active_indices = indices[mask.bool()]
            if active_indices.numel() and bool(th.any(active_indices < 0)):
                raise ValueError(f"{name} metadata contains a negative active index.")
            if (
                n_nodes is not None
                and name != "substation"
                and active_indices.numel()
                and bool(th.any(active_indices >= int(n_nodes)))
            ):
                raise ValueError(
                    f"{name} metadata contains an index outside [0, {n_nodes})."
                )

        if tuple(self.is_do_nothing.shape) != (n_actions,):
            raise ValueError("is_do_nothing must have shape [n_actions].")
        if tuple(self.original_action_ids.shape) != (n_actions,):
            raise ValueError("original_action_ids must have shape [n_actions].")
        if n_actions < 1 or not bool(self.is_do_nothing[0]):
            raise ValueError("Exposed action 0 must be the do-nothing action.")
        if int(self.is_do_nothing.sum().item()) != 1:
            raise ValueError("Exactly one exposed action must be do-nothing.")

        delta_shape = tuple(self.delta_object_indices.shape)
        if len(delta_shape) != 2 or delta_shape[0] != n_actions:
            raise ValueError(
                "delta_object_indices must have shape [n_actions, n_modifications]."
            )
        if tuple(self.delta_modification_mask.shape) != delta_shape:
            raise ValueError("delta_modification_mask must match delta_object_indices.")
        if tuple(self.delta_target_busbar_indices.shape) != delta_shape:
            raise ValueError(
                "delta_target_busbar_indices must match delta_object_indices."
            )
        if tuple(self.delta_target_busbar_mask.shape) != delta_shape:
            raise ValueError(
                "delta_target_busbar_mask must match delta_object_indices."
            )
        if (
            self.delta_context_busbar_indices.dim() != 3
            or tuple(self.delta_context_busbar_indices.shape[:2]) != delta_shape
        ):
            raise ValueError(
                "delta_context_busbar_indices must have shape "
                "[n_actions, n_modifications, n_context_busbars]."
            )
        if tuple(self.delta_context_busbar_mask.shape) != tuple(
            self.delta_context_busbar_indices.shape
        ):
            raise ValueError(
                "delta_context_busbar_mask must match " "delta_context_busbar_indices."
            )
        if tuple(self.delta_operation_features.shape[:2]) != delta_shape:
            raise ValueError(
                "delta_operation_features must start with "
                "[n_actions, n_modifications]."
            )
        if int(self.delta_operation_features.shape[-1]) != len(
            self.action_delta_feature_names
        ):
            raise ValueError("Unexpected action-delta feature dimension.")

        for name, indices, mask in (
            (
                "delta object",
                self.delta_object_indices,
                self.delta_modification_mask,
            ),
            (
                "delta target busbar",
                self.delta_target_busbar_indices,
                self.delta_target_busbar_mask,
            ),
            (
                "delta context busbar",
                self.delta_context_busbar_indices,
                self.delta_context_busbar_mask,
            ),
        ):
            active = indices[mask.bool()]
            if active.numel() and bool(th.any(active < 0)):
                raise ValueError(f"{name} metadata contains a negative active index.")
            if (
                n_nodes is not None
                and active.numel()
                and bool(th.any(active >= int(n_nodes)))
            ):
                raise ValueError(f"{name} metadata contains an out-of-range index.")


@dataclass
class _DecodedAction:
    substations: set
    lines: set
    line_status_lines: set
    loads: set
    generators: set
    n_topology_changes: int = 0
    n_line_status_changes: int = 0
    n_line_endpoint_changes: int = 0
    n_generator_changes: int = 0
    n_load_changes: int = 0
    n_other_changes: int = 0
    modifications: List["_ActionModification"] = field(default_factory=list)


@dataclass(frozen=True)
class _ActionModification:
    operation: str
    object_kind: str
    object_id: int
    substation_ids: Tuple[int, ...]
    target_bus: int = 0


def _as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    array = np.asarray(value)
    if array.size == 0:
        return []
    return [int(item) for item in array.reshape(-1).tolist()]


def _object_kind(value: Any) -> str:
    object_type = str(value or "").strip().lower()
    if object_type.startswith("line"):
        if "origin" in object_type or "or" in object_type:
            return "line_origin"
        if "extrem" in object_type or "ex" in object_type:
            return "line_extremity"
        return "line_status"
    if object_type in {"generator", "gen", "production"}:
        return "generator"
    if object_type == "load":
        return "load"
    return "other"


def _decode_action(action: Any) -> _DecodedAction:
    try:
        payload = action.as_dict()
    except Exception as exc:
        raise ValueError(
            "Candidate-action metadata requires Grid2Op actions with as_dict()."
        ) from exc

    decoded = _DecodedAction(set(), set(), set(), set(), set())
    for section_name in ("change_bus_vect", "set_bus_vect"):
        section = payload.get(section_name, {}) or {}
        if not isinstance(section, dict):
            continue
        operation = "change_bus" if section_name == "change_bus_vect" else "set_bus"
        decoded.substations.update(_as_int_list(section.get("modif_subs_id", [])))
        for substation_key, objects in section.items():
            try:
                substation_id = int(substation_key)
            except (TypeError, ValueError):
                continue
            if not isinstance(objects, dict):
                continue
            decoded.substations.add(substation_id)
            for details in objects.values():
                if not isinstance(details, dict) or "id" not in details:
                    continue
                object_id = int(details["id"])
                object_type = str(details.get("type", "")).strip().lower()
                object_kind = _object_kind(object_type)
                target_bus = (
                    int(details.get("new_bus", 0)) if operation == "set_bus" else 0
                )
                decoded.modifications.append(
                    _ActionModification(
                        operation=operation,
                        object_kind=object_kind,
                        object_id=object_id,
                        substation_ids=(substation_id,),
                        target_bus=target_bus,
                    )
                )
                decoded.n_topology_changes += 1
                if object_kind.startswith("line"):
                    decoded.lines.add(object_id)
                    decoded.n_line_endpoint_changes += 1
                elif object_kind == "generator":
                    decoded.generators.add(object_id)
                    decoded.n_generator_changes += 1
                elif object_kind == "load":
                    decoded.loads.add(object_id)
                    decoded.n_load_changes += 1
                else:
                    decoded.n_other_changes += 1

    changed_lines = set()
    change_line = payload.get("change_line_status", {}) or {}
    if isinstance(change_line, dict):
        change_ids = _as_int_list(change_line.get("changed_id", []))
        changed_lines.update(change_ids)
        for line_id in change_ids:
            decoded.modifications.append(
                _ActionModification(
                    operation="change_line_status",
                    object_kind="line_status",
                    object_id=line_id,
                    substation_ids=(),
                )
            )
    set_line = payload.get("set_line_status", {}) or {}
    if isinstance(set_line, dict):
        connected_ids = _as_int_list(set_line.get("connected_id", []))
        disconnected_ids = _as_int_list(set_line.get("disconnected_id", []))
        changed_lines.update(connected_ids)
        changed_lines.update(disconnected_ids)
        for operation, line_ids in (
            ("connect_line", connected_ids),
            ("disconnect_line", disconnected_ids),
        ):
            for line_id in line_ids:
                decoded.modifications.append(
                    _ActionModification(
                        operation=operation,
                        object_kind="line_status",
                        object_id=line_id,
                        substation_ids=(),
                    )
                )
    decoded.lines.update(changed_lines)
    decoded.line_status_lines.update(changed_lines)
    decoded.n_line_status_changes = len(changed_lines)
    return decoded


def _action_delta_feature_vector(
    modification: _ActionModification,
    n_busbar: int,
) -> np.ndarray:
    values = np.zeros((len(ACTION_DELTA_FEATURE_NAMES),), dtype=np.float32)
    feature_index = {
        name: index for index, name in enumerate(ACTION_DELTA_FEATURE_NAMES)
    }
    operation_name = f"operation_{modification.operation}"
    object_name = f"object_{modification.object_kind}"
    if operation_name in feature_index:
        values[feature_index[operation_name]] = 1.0
    if object_name in feature_index:
        values[feature_index[object_name]] = 1.0
    if modification.target_bus > 0:
        values[feature_index["has_target_busbar"]] = 1.0
        values[feature_index["target_bus_scaled"]] = float(
            modification.target_bus
        ) / float(max(int(n_busbar), 1))
    elif modification.target_bus < 0:
        values[feature_index["target_disconnected"]] = 1.0
        values[feature_index["target_bus_scaled"]] = -1.0
    return values


def _rows_for_ids(
    ids: Sequence[int],
    row_map: np.ndarray,
    *,
    action_id: int,
    object_name: str,
    strict: bool,
) -> List[int]:
    rows = []
    missing = []
    for object_id in sorted(set(int(value) for value in ids)):
        row = int(row_map[object_id]) if 0 <= object_id < len(row_map) else -1
        if row < 0:
            missing.append(object_id)
        else:
            rows.append(row)
    if missing and strict:
        raise ValueError(
            f"Action {action_id} touches {object_name} IDs {missing}, but they "
            "are absent from the agent graph. The graph construction must "
            "include every physical object in the agent's action space."
        )
    return rows


def _pad_rows(rows_by_action: Sequence[Sequence[int]]) -> Tuple[th.Tensor, th.Tensor]:
    width = max(1, max((len(rows) for rows in rows_by_action), default=0))
    indices = np.full((len(rows_by_action), width), -1, dtype=np.int64)
    mask = np.zeros((len(rows_by_action), width), dtype=np.bool_)
    for action_id, rows in enumerate(rows_by_action):
        unique_rows = sorted(set(int(row) for row in rows))
        if unique_rows:
            indices[action_id, : len(unique_rows)] = unique_rows
            mask[action_id, : len(unique_rows)] = True
    return th.from_numpy(indices), th.from_numpy(mask)


#: Fixed divisor for the action count descriptors. Identical on every grid and
#: for every agent, so the same physical action content maps to the same number
#: wherever it is scored. Chosen to put a typical action near unity.
ACTION_COUNT_SCALE = 4.0

ACTION_FEATURE_SCALINGS = ("fixed", "per_agent")


def _scaled_action_features(
    decoded_actions: Sequence[_DecodedAction],
    scaling: str = "fixed",
) -> th.Tensor:
    """Static per-action descriptor.

    The count columns need a divisor. ``per_agent`` uses the largest value in
    that agent's own action set, which makes the descriptor incomparable
    between agents sharing one scorer and between grids: the same action
    content maps to a different number depending on which action set it was
    normalised against. ``fixed`` divides by a constant instead, so the
    descriptor means the same thing everywhere and a trained scorer transfers.
    """
    if scaling not in ACTION_FEATURE_SCALINGS:
        raise ValueError(
            f"Unsupported action feature scaling '{scaling}'. Use one of: "
            + ", ".join(ACTION_FEATURE_SCALINGS)
        )
    raw = np.zeros((len(decoded_actions), len(ACTION_FEATURE_NAMES)), dtype=np.float32)
    for action_id, decoded in enumerate(decoded_actions):
        raw[action_id] = (
            float(action_id == 0),
            float(bool(decoded.substations)),
            float(bool(decoded.lines)),
            float(bool(decoded.loads)),
            float(bool(decoded.generators)),
            float(len(decoded.substations)),
            float(decoded.n_topology_changes),
            float(decoded.n_line_status_changes),
            float(decoded.n_line_endpoint_changes),
            float(decoded.n_generator_changes),
            float(decoded.n_load_changes),
            float(decoded.n_other_changes),
        )
    for column in range(5, raw.shape[1]):
        scale = (
            ACTION_COUNT_SCALE
            if scaling == "fixed"
            else max(1.0, float(raw[:, column].max()))
        )
        raw[:, column] /= scale
    return th.from_numpy(raw)


def _build_action_delta_tensors(
    decoded_actions: Sequence[_DecodedAction],
    graph_spec: Dict[str, Any],
    *,
    line_or_to_subid: np.ndarray,
    line_ex_to_subid: np.ndarray,
    strict: bool,
) -> Tuple[th.Tensor, ...]:
    """Encode exact action modifications with transferable physical rows."""
    busbar_rows = np.asarray(graph_spec["substation_busbar_node_rows"], dtype=np.int64)
    busbar_id_to_row = np.asarray(graph_spec["busbar_id_to_node_row"], dtype=np.int64)
    line_rows = np.asarray(graph_spec["line_id_to_node_row"], dtype=np.int64)
    load_rows = np.asarray(graph_spec["load_id_to_node_row"], dtype=np.int64)
    generator_rows = np.asarray(graph_spec["gen_id_to_node_row"], dtype=np.int64)
    n_busbar = int(graph_spec.get("n_busbar", busbar_rows.shape[1]))

    max_modifications = max(
        1,
        max((len(decoded.modifications) for decoded in decoded_actions), default=0),
    )
    max_context_busbars = max(
        1,
        2 * n_busbar,
        max((int(np.sum(rows >= 0)) for rows in busbar_rows), default=0),
    )
    n_actions = len(decoded_actions)
    object_indices = np.full((n_actions, max_modifications), -1, dtype=np.int64)
    modification_mask = np.zeros((n_actions, max_modifications), dtype=np.bool_)
    target_indices = np.full_like(object_indices, -1)
    target_mask = np.zeros_like(modification_mask)
    context_indices = np.full(
        (n_actions, max_modifications, max_context_busbars),
        -1,
        dtype=np.int64,
    )
    context_mask = np.zeros_like(context_indices, dtype=np.bool_)
    operation_features = np.zeros(
        (n_actions, max_modifications, len(ACTION_DELTA_FEATURE_NAMES)),
        dtype=np.float32,
    )

    def object_row(modification: _ActionModification) -> int:
        object_id = int(modification.object_id)
        if modification.object_kind.startswith("line"):
            rows = line_rows
        elif modification.object_kind == "generator":
            rows = generator_rows
        elif modification.object_kind == "load":
            rows = load_rows
        else:
            return -1
        return int(rows[object_id]) if 0 <= object_id < len(rows) else -1

    def context_substations(modification: _ActionModification) -> Tuple[int, ...]:
        if modification.substation_ids:
            return modification.substation_ids
        if modification.object_kind == "line_status":
            line_id = int(modification.object_id)
            if 0 <= line_id < len(line_or_to_subid):
                return (
                    int(line_or_to_subid[line_id]),
                    int(line_ex_to_subid[line_id]),
                )
        return ()

    for action_id, decoded in enumerate(decoded_actions):
        for modification_id, modification in enumerate(decoded.modifications):
            row = object_row(modification)
            if row < 0:
                if strict:
                    raise ValueError(
                        f"Action {action_id} modification {modification_id} "
                        f"({modification.object_kind} {modification.object_id}) "
                        "has no graph-represented object row."
                    )
                continue
            object_indices[action_id, modification_id] = row
            modification_mask[action_id, modification_id] = True
            operation_features[action_id, modification_id] = (
                _action_delta_feature_vector(modification, n_busbar)
            )

            context_rows: List[int] = []
            modification_substations = context_substations(modification)
            for substation_id in modification_substations:
                if 0 <= substation_id < len(busbar_rows):
                    context_rows.extend(
                        int(value)
                        for value in busbar_rows[substation_id]
                        if int(value) >= 0
                    )
            context_rows = sorted(set(context_rows))
            if context_rows:
                context_indices[action_id, modification_id, : len(context_rows)] = (
                    context_rows
                )
                context_mask[action_id, modification_id, : len(context_rows)] = True

            target_bus = int(modification.target_bus)
            if target_bus > 0 and modification_substations:
                substation_id = int(modification_substations[0])
                global_busbar_id = substation_id * n_busbar + (target_bus - 1)
                target_row = (
                    int(busbar_id_to_row[global_busbar_id])
                    if 0 <= global_busbar_id < len(busbar_id_to_row)
                    else -1
                )
                if target_row < 0 and strict:
                    raise ValueError(
                        f"Action {action_id} targets busbar {target_bus} of "
                        f"substation {substation_id}, which is absent from the graph."
                    )
                if target_row >= 0:
                    target_indices[action_id, modification_id] = target_row
                    target_mask[action_id, modification_id] = True

    return (
        th.from_numpy(object_indices),
        th.from_numpy(modification_mask),
        th.from_numpy(target_indices),
        th.from_numpy(target_mask),
        th.from_numpy(context_indices),
        th.from_numpy(context_mask),
        th.from_numpy(operation_features),
    )


def build_action_graph_metadata(
    graph_spec: Dict[str, Any],
    actions: Sequence[Any],
    *,
    line_or_to_subid: Sequence[int],
    line_ex_to_subid: Sequence[int],
    original_action_ids: Optional[Sequence[int]] = None,
    strict: bool = True,
    action_feature_scaling: str = "fixed",
) -> ActionGraphMetadata:
    """Build one deterministic candidate-to-node mapping for an agent."""

    if str(graph_spec.get("graph_type", "")) != "heterogeneous_line":
        raise ValueError(
            "Candidate-action pooling requires gnn_graph_type=heterogeneous_line."
        )
    actions = list(actions)
    if not actions:
        raise ValueError("Cannot build candidate metadata for an empty action space.")

    required_maps = (
        "busbar_id_to_node_row",
        "line_id_to_node_row",
        "gen_id_to_node_row",
        "load_id_to_node_row",
        "substation_busbar_node_rows",
    )
    missing_maps = [name for name in required_maps if name not in graph_spec]
    if missing_maps:
        raise ValueError(
            "heterogeneous_line graph spec is missing action row maps: "
            + ", ".join(missing_maps)
        )

    line_or_to_subid = np.asarray(line_or_to_subid, dtype=np.int64)
    line_ex_to_subid = np.asarray(line_ex_to_subid, dtype=np.int64)
    busbar_rows = np.asarray(graph_spec["substation_busbar_node_rows"], dtype=np.int64)
    line_rows = np.asarray(graph_spec["line_id_to_node_row"], dtype=np.int64)
    load_rows = np.asarray(graph_spec["load_id_to_node_row"], dtype=np.int64)
    generator_rows = np.asarray(graph_spec["gen_id_to_node_row"], dtype=np.int64)

    decoded_actions = [_decode_action(action) for action in actions]
    for decoded in decoded_actions:
        for line_id in decoded.line_status_lines:
            if 0 <= line_id < len(line_or_to_subid):
                decoded.substations.add(int(line_or_to_subid[line_id]))
                decoded.substations.add(int(line_ex_to_subid[line_id]))

    per_type_rows = {
        "busbar": [],
        "line": [],
        "load": [],
        "generator": [],
        "substation": [],
    }
    for action_id, decoded in enumerate(decoded_actions):
        substation_ids = sorted(int(value) for value in decoded.substations)
        action_busbar_rows = []
        missing_substations = []
        for substation_id in substation_ids:
            if 0 <= substation_id < len(busbar_rows):
                rows = busbar_rows[substation_id]
                action_busbar_rows.extend(int(row) for row in rows if row >= 0)
                if not np.any(rows >= 0):
                    missing_substations.append(substation_id)
            else:
                missing_substations.append(substation_id)
        if missing_substations and strict:
            raise ValueError(
                f"Action {action_id} touches substations {missing_substations}, "
                "but their busbar nodes are absent from the agent graph."
            )

        per_type_rows["busbar"].append(action_busbar_rows)
        per_type_rows["line"].append(
            _rows_for_ids(
                decoded.lines,
                line_rows,
                action_id=action_id,
                object_name="line",
                strict=strict,
            )
        )
        per_type_rows["load"].append(
            _rows_for_ids(
                decoded.loads,
                load_rows,
                action_id=action_id,
                object_name="load",
                strict=strict,
            )
        )
        per_type_rows["generator"].append(
            _rows_for_ids(
                decoded.generators,
                generator_rows,
                action_id=action_id,
                object_name="generator",
                strict=strict,
            )
        )
        per_type_rows["substation"].append(substation_ids)

        represented_count = sum(
            len(per_type_rows[name][-1])
            for name in ("busbar", "line", "load", "generator")
        )
        if action_id == 0 and represented_count:
            raise ValueError("Exposed action 0 is not a do-nothing action.")
        if action_id > 0 and represented_count == 0 and strict:
            raise ValueError(
                f"Action {action_id} has no graph-represented physical context."
            )

    busbar_indices, busbar_mask = _pad_rows(per_type_rows["busbar"])
    line_indices, line_mask = _pad_rows(per_type_rows["line"])
    load_indices, load_mask = _pad_rows(per_type_rows["load"])
    generator_indices, generator_mask = _pad_rows(per_type_rows["generator"])
    substation_ids, substation_mask = _pad_rows(per_type_rows["substation"])
    (
        delta_object_indices,
        delta_modification_mask,
        delta_target_busbar_indices,
        delta_target_busbar_mask,
        delta_context_busbar_indices,
        delta_context_busbar_mask,
        delta_operation_features,
    ) = _build_action_delta_tensors(
        decoded_actions,
        graph_spec,
        line_or_to_subid=line_or_to_subid,
        line_ex_to_subid=line_ex_to_subid,
        strict=strict,
    )

    if original_action_ids is None:
        original_action_ids = list(range(len(actions)))
    original_action_ids = [int(value) for value in original_action_ids]
    if len(original_action_ids) != len(actions):
        raise ValueError("original_action_ids must have one entry per exposed action.")

    metadata = ActionGraphMetadata(
        action_features=_scaled_action_features(
            decoded_actions, action_feature_scaling
        ),
        busbar_indices=busbar_indices,
        busbar_mask=busbar_mask,
        line_indices=line_indices,
        line_mask=line_mask,
        load_indices=load_indices,
        load_mask=load_mask,
        generator_indices=generator_indices,
        generator_mask=generator_mask,
        substation_ids=substation_ids,
        substation_mask=substation_mask,
        delta_object_indices=delta_object_indices,
        delta_modification_mask=delta_modification_mask,
        delta_target_busbar_indices=delta_target_busbar_indices,
        delta_target_busbar_mask=delta_target_busbar_mask,
        delta_context_busbar_indices=delta_context_busbar_indices,
        delta_context_busbar_mask=delta_context_busbar_mask,
        delta_operation_features=delta_operation_features,
        is_do_nothing=th.arange(len(actions), dtype=th.long) == 0,
        original_action_ids=th.tensor(original_action_ids, dtype=th.long),
    )
    metadata.validate(n_nodes=len(graph_spec["node_ids"]))
    return metadata
