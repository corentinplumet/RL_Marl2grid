from dataclasses import dataclass
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
    is_do_nothing: th.Tensor
    original_action_ids: th.Tensor
    action_feature_names: Tuple[str, ...] = ACTION_FEATURE_NAMES

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


def _as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    array = np.asarray(value)
    if array.size == 0:
        return []
    return [int(item) for item in array.reshape(-1).tolist()]


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
        decoded.substations.update(
            _as_int_list(section.get("modif_subs_id", []))
        )
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
                decoded.n_topology_changes += 1
                if object_type.startswith("line"):
                    decoded.lines.add(object_id)
                    decoded.n_line_endpoint_changes += 1
                elif object_type in {"generator", "gen", "production"}:
                    decoded.generators.add(object_id)
                    decoded.n_generator_changes += 1
                elif object_type == "load":
                    decoded.loads.add(object_id)
                    decoded.n_load_changes += 1
                else:
                    decoded.n_other_changes += 1

    changed_lines = set()
    change_line = payload.get("change_line_status", {}) or {}
    if isinstance(change_line, dict):
        changed_lines.update(_as_int_list(change_line.get("changed_id", [])))
    set_line = payload.get("set_line_status", {}) or {}
    if isinstance(set_line, dict):
        changed_lines.update(_as_int_list(set_line.get("connected_id", [])))
        changed_lines.update(_as_int_list(set_line.get("disconnected_id", [])))
    decoded.lines.update(changed_lines)
    decoded.line_status_lines.update(changed_lines)
    decoded.n_line_status_changes = len(changed_lines)
    return decoded


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
            "are absent from the agent graph. Candidate pooling requires "
            "gnn_include_neighbors=true for local heterogeneous-line graphs."
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


def _scaled_action_features(decoded_actions: Sequence[_DecodedAction]) -> th.Tensor:
    raw = np.zeros(
        (len(decoded_actions), len(ACTION_FEATURE_NAMES)), dtype=np.float32
    )
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
        scale = max(1.0, float(raw[:, column].max()))
        raw[:, column] /= scale
    return th.from_numpy(raw)


def build_action_graph_metadata(
    graph_spec: Dict[str, Any],
    actions: Sequence[Any],
    *,
    line_or_to_subid: Sequence[int],
    line_ex_to_subid: Sequence[int],
    original_action_ids: Optional[Sequence[int]] = None,
    strict: bool = True,
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
    busbar_rows = np.asarray(
        graph_spec["substation_busbar_node_rows"], dtype=np.int64
    )
    line_rows = np.asarray(graph_spec["line_id_to_node_row"], dtype=np.int64)
    load_rows = np.asarray(graph_spec["load_id_to_node_row"], dtype=np.int64)
    generator_rows = np.asarray(
        graph_spec["gen_id_to_node_row"], dtype=np.int64
    )

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

    if original_action_ids is None:
        original_action_ids = list(range(len(actions)))
    original_action_ids = [int(value) for value in original_action_ids]
    if len(original_action_ids) != len(actions):
        raise ValueError(
            "original_action_ids must have one entry per exposed action."
        )

    metadata = ActionGraphMetadata(
        action_features=_scaled_action_features(decoded_actions),
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
        is_do_nothing=th.arange(len(actions), dtype=th.long) == 0,
        original_action_ids=th.tensor(original_action_ids, dtype=th.long),
    )
    metadata.validate(n_nodes=len(graph_spec["node_ids"]))
    return metadata
