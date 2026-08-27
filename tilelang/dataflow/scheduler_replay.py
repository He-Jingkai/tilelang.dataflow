"""Deterministic schedule recording and replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence

from tilelang.utils.target_capabilities import TargetCapabilitySnapshot

from .compile_config import canonical_fingerprint
from .program import DataflowProgram
from .scheduler_config import DataflowSchedulerConfig
from .topology import GPUTopology


DATAFLOW_SCHEDULE_REPLAY_SCHEMA_VERSION = 6


def target_capabilities_from_dict(
    value: Mapping[str, Any] | None,
) -> TargetCapabilitySnapshot | None:
    if value is None:
        return None
    return TargetCapabilitySnapshot.from_dict(dict(value))


def range_mapping(
    value: Mapping[Any, int | Sequence[int]] | None,
) -> tuple[tuple[Any, tuple[int, ...]], ...] | None:
    if value is None:
        return None
    result = []
    for axis, lengths in value.items():
        normalized = (int(lengths),) if isinstance(lengths, int) else tuple(int(item) for item in lengths)
        result.append((axis, normalized))
    return tuple(sorted(result, key=lambda item: repr(item[0])))


def serialized_range_mapping(
    value: tuple[tuple[Any, tuple[int, ...]], ...] | None,
) -> dict[str, list[int]] | None:
    if value is None:
        return None
    result = {str(axis): list(lengths) for axis, lengths in value}
    if len(result) != len(value):
        raise ValueError("Dataflow schedule replay range axes must have unique string identities")
    return result


def instruction_plan_fingerprint(plan: Any) -> str:
    """Return a name-independent fingerprint of a structured instruction plan."""

    return canonical_fingerprint(
        {
            "topology": plan.topology,
            "block_size": plan.block_size,
            "range_axis": str(plan.range_axis),
            "scheduler_policy": plan.scheduler_policy,
            "reduce_strategy": plan.reduce_strategy,
            "task_extents": plan.task_extents,
            "task_range_lengths": plan.task_range_lengths,
            "scheduler_config": plan.scheduler_config,
            "range_resource_budget_bytes": plan.range_resource_budget_bytes,
            "target_capabilities": plan.target_capabilities,
            "range_coarsening_plans": plan.range_coarsening_plans,
            "reshared_transport_plans": plan.reshared_transport_plans,
            "cross_handler_handoff_plans": plan.cross_handler_handoff_plans,
            "cross_handler_handoff_bindings": plan.cross_handler_handoff_bindings,
            "joint_execution_plan": plan.joint_execution_plan,
            "instructions": tuple(
                {
                    "id": instruction.instruction_id,
                    "opcode": instruction.opcode.value,
                    "task_id": instruction.task_id,
                    "task_coords": instruction.task_coords,
                    "sm_id": instruction.sm_id,
                    "task_range": instruction.task_range,
                    "input_slots": instruction.input_slots,
                    "output_slot": instruction.output_slot,
                    "attrs": instruction.attrs,
                    "handler_identity": instruction.handler_identity,
                    "handler_variant_key": instruction.handler_variant_key,
                    "value_forward": instruction.value_forward,
                }
                for instruction in plan.instructions
            ),
            "queues": {sm_id: tuple(instruction.instruction_id for instruction in queue) for sm_id, queue in plan.queues.items()},
            "slots": plan.slots,
            "comms": plan.comms,
        }
    )


@dataclass(frozen=True)
class DataflowScheduleReplay:
    program_fingerprint: str
    topology: GPUTopology
    range_lengths: tuple[tuple[Any, tuple[int, ...]], ...]
    range_offsets: tuple[tuple[Any, tuple[int, ...]], ...] | None
    block_size: int
    task_extents: tuple[int, ...] | None
    include_exit: bool
    scheduler_policy: str
    reduce_strategy: str
    force_hbm_comms: bool
    streaming_tree_consumer: str | None
    cluster_task_assignment: tuple[tuple[int, ...], ...] | str | None
    skip_tiny_root_fragment: bool | None
    skip_tiny_root_fragment_blocks: int | None
    partial_only: bool
    direct_leaf_acc: bool | None
    iter_range_buckets: tuple[int, ...]
    iter_range_bucket_size: int | None
    iter_range_exact_lengths: tuple[int, ...]
    task_coord_overrides: tuple[tuple[int, ...], ...] | None
    stage_graph_task_weights: tuple[float, ...] | None
    stage_graph_cluster_assignment: tuple[int, ...] | None
    range_resource_budget_bytes: int | None
    target_capabilities: TargetCapabilitySnapshot | None
    scheduler_config: DataflowSchedulerConfig
    expected_plan_fingerprint: str
    schema_version: int = DATAFLOW_SCHEDULE_REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_SCHEDULE_REPLAY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Dataflow schedule replay schema version {self.schema_version}")
        if not self.program_fingerprint or not self.expected_plan_fingerprint:
            raise ValueError("Dataflow schedule replay requires program and plan fingerprints")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "program_fingerprint": self.program_fingerprint,
            "topology": {
                "sm_count": self.topology.sm_count,
                "cluster_size": self.topology.cluster_size,
            },
            "range_lengths": serialized_range_mapping(self.range_lengths),
            "range_offsets": serialized_range_mapping(self.range_offsets),
            "block_size": self.block_size,
            "task_extents": None if self.task_extents is None else list(self.task_extents),
            "include_exit": self.include_exit,
            "scheduler_policy": self.scheduler_policy,
            "reduce_strategy": self.reduce_strategy,
            "force_hbm_comms": self.force_hbm_comms,
            "streaming_tree_consumer": self.streaming_tree_consumer,
            "cluster_task_assignment": self.cluster_task_assignment,
            "skip_tiny_root_fragment": self.skip_tiny_root_fragment,
            "skip_tiny_root_fragment_blocks": self.skip_tiny_root_fragment_blocks,
            "partial_only": self.partial_only,
            "direct_leaf_acc": self.direct_leaf_acc,
            "iter_range_buckets": list(self.iter_range_buckets),
            "iter_range_bucket_size": self.iter_range_bucket_size,
            "iter_range_exact_lengths": list(self.iter_range_exact_lengths),
            "task_coord_overrides": self.task_coord_overrides,
            "stage_graph_task_weights": self.stage_graph_task_weights,
            "stage_graph_cluster_assignment": self.stage_graph_cluster_assignment,
            "range_resource_budget_bytes": self.range_resource_budget_bytes,
            "target_capabilities": (None if self.target_capabilities is None else self.target_capabilities.to_dict()),
            "scheduler_config": self.scheduler_config.to_dict(),
            "expected_plan_fingerprint": self.expected_plan_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowScheduleReplay:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow schedule replay must be a mapping, got {type(value)!r}")
        topology = value.get("topology", {})
        if not isinstance(topology, Mapping):
            raise TypeError("Dataflow schedule replay topology must be a mapping")

        def ranges(name: str):
            raw = value.get(name)
            if raw is None:
                return None
            if not isinstance(raw, Mapping):
                raise TypeError(f"Dataflow schedule replay {name} must be a mapping")
            return tuple(
                (str(axis), tuple(int(item) for item in lengths)) for axis, lengths in sorted(raw.items(), key=lambda item: str(item[0]))
            )

        def optional_tuple(name: str, caster):
            raw = value.get(name)
            return None if raw is None else tuple(caster(item) for item in raw)

        raw_assignment = value.get("cluster_task_assignment")
        assignment = (
            raw_assignment
            if isinstance(raw_assignment, str) or raw_assignment is None
            else tuple(tuple(int(item) for item in group) for group in raw_assignment)
        )
        raw_coords = value.get("task_coord_overrides")
        coords = None if raw_coords is None else tuple(tuple(int(item) for item in coord) for coord in raw_coords)
        return cls(
            schema_version=int(value.get("schema_version", DATAFLOW_SCHEDULE_REPLAY_SCHEMA_VERSION)),
            program_fingerprint=str(value["program_fingerprint"]),
            topology=GPUTopology(
                sm_count=int(topology["sm_count"]),
                cluster_size=int(topology.get("cluster_size", 1)),
            ),
            range_lengths=ranges("range_lengths") or (),
            range_offsets=ranges("range_offsets"),
            block_size=int(value["block_size"]),
            task_extents=optional_tuple("task_extents", int),
            include_exit=bool(value.get("include_exit", True)),
            scheduler_policy=str(value["scheduler_policy"]),
            reduce_strategy=str(value["reduce_strategy"]),
            force_hbm_comms=bool(value.get("force_hbm_comms", False)),
            streaming_tree_consumer=value.get("streaming_tree_consumer"),
            cluster_task_assignment=assignment,
            skip_tiny_root_fragment=value.get("skip_tiny_root_fragment"),
            skip_tiny_root_fragment_blocks=value.get("skip_tiny_root_fragment_blocks"),
            partial_only=bool(value.get("partial_only", False)),
            direct_leaf_acc=value.get("direct_leaf_acc"),
            iter_range_buckets=tuple(int(item) for item in value.get("iter_range_buckets", ())),
            iter_range_bucket_size=value.get("iter_range_bucket_size"),
            iter_range_exact_lengths=tuple(int(item) for item in value.get("iter_range_exact_lengths", ())),
            task_coord_overrides=coords,
            stage_graph_task_weights=optional_tuple("stage_graph_task_weights", float),
            stage_graph_cluster_assignment=optional_tuple("stage_graph_cluster_assignment", int),
            range_resource_budget_bytes=(
                None if value.get("range_resource_budget_bytes") is None else int(value["range_resource_budget_bytes"])
            ),
            target_capabilities=target_capabilities_from_dict(value.get("target_capabilities")),
            scheduler_config=DataflowSchedulerConfig.from_dict(value.get("scheduler_config", {})),
            expected_plan_fingerprint=str(value["expected_plan_fingerprint"]),
        )


def record_schedule_replay(
    program: DataflowProgram,
    plan: Any,
    *,
    range_lengths: Mapping[Any, int | Sequence[int]],
    range_offsets: Mapping[Any, int | Sequence[int]] | None = None,
    include_exit: bool = True,
    force_hbm_comms: bool = False,
    streaming_tree_consumer: str | None = None,
    cluster_task_assignment: Sequence[Sequence[int]] | str | None = None,
    skip_tiny_root_fragment: bool | None = None,
    skip_tiny_root_fragment_blocks: int | None = None,
    partial_only: bool = False,
    direct_leaf_acc: bool | None = None,
    iter_range_buckets: Sequence[int] = (),
    iter_range_bucket_size: int | None = None,
    iter_range_exact_lengths: Sequence[int] = (),
    task_coord_overrides: Sequence[Sequence[int] | int] | None = None,
    stage_graph_task_weights: Sequence[int | float] | None = None,
    stage_graph_cluster_assignment: Sequence[int] | None = None,
) -> DataflowScheduleReplay:
    if not isinstance(program, DataflowProgram):
        raise TypeError(f"record_schedule_replay expects DataflowProgram, got {program!r}")
    normalized_assignment = (
        cluster_task_assignment
        if isinstance(cluster_task_assignment, str) or cluster_task_assignment is None
        else tuple(tuple(int(task_id) for task_id in cluster) for cluster in cluster_task_assignment)
    )
    normalized_coords = None
    if task_coord_overrides is not None:
        normalized_coords = tuple(
            (int(coords),) if isinstance(coords, int) else tuple(int(item) for item in coords) for coords in task_coord_overrides
        )
    return DataflowScheduleReplay(
        program_fingerprint=canonical_fingerprint(program),
        topology=plan.topology,
        range_lengths=range_mapping(range_lengths) or (),
        range_offsets=range_mapping(range_offsets),
        block_size=plan.block_size,
        task_extents=plan.task_extents,
        include_exit=bool(include_exit),
        scheduler_policy=plan.scheduler_policy,
        reduce_strategy=plan.reduce_strategy,
        force_hbm_comms=bool(force_hbm_comms),
        streaming_tree_consumer=streaming_tree_consumer,
        cluster_task_assignment=normalized_assignment,
        skip_tiny_root_fragment=skip_tiny_root_fragment,
        skip_tiny_root_fragment_blocks=skip_tiny_root_fragment_blocks,
        partial_only=bool(partial_only),
        direct_leaf_acc=direct_leaf_acc,
        iter_range_buckets=tuple(int(item) for item in iter_range_buckets),
        iter_range_bucket_size=iter_range_bucket_size,
        iter_range_exact_lengths=tuple(int(item) for item in iter_range_exact_lengths),
        task_coord_overrides=normalized_coords,
        stage_graph_task_weights=(None if stage_graph_task_weights is None else tuple(float(item) for item in stage_graph_task_weights)),
        stage_graph_cluster_assignment=(
            None if stage_graph_cluster_assignment is None else tuple(int(item) for item in stage_graph_cluster_assignment)
        ),
        range_resource_budget_bytes=plan.range_resource_budget_bytes,
        target_capabilities=plan.target_capabilities,
        scheduler_config=plan.scheduler_config,
        expected_plan_fingerprint=instruction_plan_fingerprint(plan),
    )


def replay_schedule(program: DataflowProgram, replay: DataflowScheduleReplay):
    if canonical_fingerprint(program) != replay.program_fingerprint:
        raise ValueError("Dataflow schedule replay program fingerprint does not match")
    from .scheduler import schedule

    range_lengths = dict(replay.range_lengths)
    range_offsets = None if replay.range_offsets is None else dict(replay.range_offsets)
    if program.partial_stage is not None and program.partial_stage.range_axis is not None:
        actual_axis = program.partial_stage.range_axis

        def restore_axis(values):
            if values is None or actual_axis in values:
                return values
            matches = [key for key in values if str(key) == str(actual_axis)]
            if len(matches) == 1:
                return {(actual_axis if key == matches[0] else key): item for key, item in values.items()}
            return values

        range_lengths = restore_axis(range_lengths)
        range_offsets = restore_axis(range_offsets)

    plan = schedule(
        program,
        topology=replay.topology,
        range_lengths=range_lengths,
        range_offsets=range_offsets,
        block_size=replay.block_size,
        task_extents=replay.task_extents,
        include_exit=replay.include_exit,
        scheduler_policy=replay.scheduler_policy,
        reduce_strategy=replay.reduce_strategy,
        force_hbm_comms=replay.force_hbm_comms,
        streaming_tree_consumer=replay.streaming_tree_consumer,
        cluster_task_assignment=replay.cluster_task_assignment,
        skip_tiny_root_fragment=replay.skip_tiny_root_fragment,
        skip_tiny_root_fragment_blocks=replay.skip_tiny_root_fragment_blocks,
        partial_only=replay.partial_only,
        direct_leaf_acc=replay.direct_leaf_acc,
        iter_range_buckets=replay.iter_range_buckets,
        iter_range_bucket_size=replay.iter_range_bucket_size,
        iter_range_exact_lengths=replay.iter_range_exact_lengths,
        task_coord_overrides=replay.task_coord_overrides,
        stage_graph_task_weights=replay.stage_graph_task_weights,
        stage_graph_cluster_assignment=replay.stage_graph_cluster_assignment,
        range_resource_budget_bytes=replay.range_resource_budget_bytes,
        target_capabilities=replay.target_capabilities,
        scheduler_config=replay.scheduler_config,
    )
    actual = instruction_plan_fingerprint(plan)
    if actual != replay.expected_plan_fingerprint:
        raise RuntimeError(
            f"Dataflow schedule replay produced a different structured plan: expected={replay.expected_plan_fingerprint}, actual={actual}"
        )
    return plan
