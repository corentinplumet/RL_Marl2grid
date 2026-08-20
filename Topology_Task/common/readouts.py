"""Names of the terminal graph readouts.

Shared by the encoders, the argument parser, and the configuration validators.
This module deliberately imports nothing, so a launcher can check a config
without loading torch or torch-geometric.

A readout name is composed as ``[controlled_][energized_]{mean,sum,max}``.
``controlled_`` restricts the aggregation to the agent's action domain and
``energized_`` to nodes incident to an active electrical relation. The two
restrictions are independent and may be combined, so the set is generated from
the composition rule rather than listed by hand: an option that can be named
here is one the pooling code can already evaluate. The two explicit
``energized_mean_max`` variants concatenate the complementary average and
extreme summaries before the learned readout projection.
"""

POOLING_AGGREGATIONS = tuple(
    f"{controlled}{energized}{base}"
    for controlled in ("", "controlled_")
    for energized in ("", "energized_")
    for base in ("mean", "sum", "max")
) + (
    "energized_mean_max",
    "controlled_energized_mean_max",
    "virtual_node",
)

#: Readouts provided only by the sparse graph transformer.
SPARSE_TRANSFORMER_ONLY_AGGREGATIONS = ("attention", "controlled_attention")
