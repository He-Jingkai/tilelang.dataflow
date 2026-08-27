"""Dataflow declarations exposed on the TileLang language surface."""

from tilelang.dataflow import (  # noqa: F401
    dataflow_finalize,
    dataflow_intermediate,
    dataflow_iter,
    dataflow_map,
    dataflow_program,
    dataflow_reduce,
)


finalize = dataflow_finalize
iter = dataflow_iter
map = dataflow_map
reduce = dataflow_reduce


def dataflow_lowering_only(name: str) -> None:
    raise RuntimeError(f"T.{name}() is only valid inside a lowered Dataflow handler body")


def dataflow_range_begin() -> int:
    """Marker for the scheduled Dataflow iter range begin in handler bodies."""

    return dataflow_lowering_only("dataflow_range_begin")


def dataflow_range_end() -> int:
    """Marker for the scheduled Dataflow iter range end in handler bodies."""

    return dataflow_lowering_only("dataflow_range_end")


def dataflow_range_tiles_per_handler() -> int:
    """Marker for the compile-time number of logical range tiles per handler."""

    return dataflow_lowering_only("dataflow_range_tiles_per_handler")


def dataflow_task_id() -> int:
    """Marker for the scheduler task id in handler bodies."""

    return dataflow_lowering_only("dataflow_task_id")


def dataflow_next_task_coord(axis: int) -> int:
    """Marker for a cross-handler handoff target task coordinate."""

    del axis
    return dataflow_lowering_only("dataflow_next_task_coord")


def dataflow_handoff_stage_count() -> int:
    """Marker for the number of shared stages handed across handlers."""

    return dataflow_lowering_only("dataflow_handoff_stage_count")


__all__ = [
    "dataflow_finalize",
    "dataflow_handoff_stage_count",
    "dataflow_intermediate",
    "dataflow_iter",
    "dataflow_map",
    "dataflow_next_task_coord",
    "dataflow_program",
    "dataflow_range_begin",
    "dataflow_range_end",
    "dataflow_range_tiles_per_handler",
    "dataflow_reduce",
    "dataflow_task_id",
    "finalize",
    "iter",
    "map",
    "reduce",
]
