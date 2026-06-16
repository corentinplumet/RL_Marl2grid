from common.imports import *


EXPLAIN_INFO_KEY = "marl2grid_explain"


EXPLAIN_SCALAR_KEYS = (
    "pre_max_rho",
    "post_max_rho",
    "delta_max_rho",
    "topology_distance_before",
    "topology_distance_after",
    "topology_distance_delta",
)


EXPLAIN_LINE_KEYS = (
    "worst_line_before",
    "worst_line_after",
)


def transition_info(info: Any) -> Any:
    if isinstance(info, dict) and "final_info" in info:
        return info["final_info"]
    return info


def extract_transition_explain(info: Any) -> Optional[Dict[str, Any]]:
    info = transition_info(info)
    if not isinstance(info, dict):
        return None
    explain = info.get(EXPLAIN_INFO_KEY)
    return explain if isinstance(explain, dict) else None


def explain_arrays_from_infos(infos: Any) -> Dict[str, np.ndarray]:
    step_infos = list(infos) if isinstance(infos, (list, tuple)) else [infos]
    values = {key: [] for key in EXPLAIN_SCALAR_KEYS + EXPLAIN_LINE_KEYS}
    for info in step_infos:
        explain = extract_transition_explain(info)
        pre = explain.get("pre", {}) if explain else {}
        post = explain.get("post", {}) if explain else {}

        pre_max_rho = float(pre.get("max_rho", np.nan))
        post_max_rho = float(post.get("max_rho", np.nan))
        topology_before = float(pre.get("topology_distance", np.nan))
        topology_after = float(post.get("topology_distance", np.nan))

        values["pre_max_rho"].append(pre_max_rho)
        values["post_max_rho"].append(post_max_rho)
        values["delta_max_rho"].append(post_max_rho - pre_max_rho)
        values["worst_line_before"].append(float(pre.get("worst_line", -1)))
        values["worst_line_after"].append(float(post.get("worst_line", -1)))
        values["topology_distance_before"].append(topology_before)
        values["topology_distance_after"].append(topology_after)
        values["topology_distance_delta"].append(topology_after - topology_before)
    return {
        key: np.asarray(items, dtype=np.float32)
        for key, items in values.items()
    }


def finite_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(arr[finite].mean())


def finite_mode_int(values: Any) -> int:
    arr = np.asarray(values, dtype=np.float32)
    valid = arr[np.isfinite(arr) & (arr >= 0)].astype(np.int64)
    if valid.size == 0:
        return -1
    counts = np.bincount(valid)
    return int(np.argmax(counts))


def summarize_explain_arrays(
    arrays: Dict[str, Any],
    prefix: str = "train/explain",
) -> Dict[str, Any]:
    metrics = {
        f"{prefix}/{key}": finite_mean(arrays[key])
        for key in EXPLAIN_SCALAR_KEYS
        if key in arrays
    }
    for key in EXPLAIN_LINE_KEYS:
        if key in arrays:
            metrics[f"{prefix}/{key}"] = finite_mode_int(arrays[key])
    return metrics
