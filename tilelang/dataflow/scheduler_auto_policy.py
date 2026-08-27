"""Generic, replay-scored auto policy selection for Dataflow schedules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from .compile_config import canonical_fingerprint
from .ir import DataflowReducerContract
from .physical_contract import operator_physical_contract
from .program import DataflowProgram
from .scheduler import InstructionPlan, DataflowCommKind, DataflowOpcode, schedule
from .scheduler_config import DataflowSchedulerConfig, resolve_scheduler_config
from .scheduler_cost_model import (
    DATAFLOW_COST_MODEL_VERSION,
    DataflowCostModelConfig,
    estimate_communication_cost_us,
    estimate_finalize_cost_us,
    estimate_iter_cost_us,
    estimate_reduce_cost_us,
)
from .scheduler_policies import normalize_reduce_strategy, normalize_scheduler_policy
from .scheduler_replay import instruction_plan_fingerprint
from .topology import GPUTopology


DATAFLOW_SCHEDULER_AUTO_POLICY_SCHEMA_VERSION = 1
DATAFLOW_SCHEDULER_FEATURE_SCHEMA_VERSION = 1
DATAFLOW_SCHEDULER_POLICY_RECORD_SCHEMA_VERSION = 1
DATAFLOW_SCHEDULER_AUTO_POLICY_VERSION = "dataflow.scheduler_auto.v3"
DATAFLOW_SCHEDULER_AUTO = "auto"

_POLICY_RECORD_SCHEMA = "tilelang.dataflow.scheduler_policy_records"


class DataflowSchedulerAutoPolicyError(RuntimeError):
    """Raised when auto policy selection has no legal candidate."""


def serialize_canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {context} field(s): {', '.join(unknown)}")


def normalize_assignment(
    value: Sequence[Sequence[int]] | str | None,
) -> tuple[tuple[int, ...], ...] | str | None:
    if value is None or isinstance(value, str):
        return value
    return tuple(tuple(int(task_id) for task_id in group) for group in value)


def normalize_option_items(
    value: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
) -> tuple[tuple[str, Any], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    normalized = []
    for name, item in items:
        if isinstance(item, list):
            item = tuple(item)
        normalized.append((str(name), item))
    return tuple(sorted(normalized, key=lambda item: item[0]))


@dataclass(frozen=True)
class DataflowSchedulerFeatures:
    """Name-independent structured inputs consumed by the selector."""

    task_count: int
    task_extents: tuple[int, ...] | None
    range_length_min: int
    range_length_max: int
    range_length_sum: int
    range_length_mean: float
    range_length_p50: int
    range_length_p95: int
    range_length_aligned_count: int
    tile_count_min: int
    tile_count_max: int
    tile_count_sum: int
    tile_count_histogram: tuple[tuple[str, int], ...]
    sm_count: int
    cluster_size: int
    cluster_count: int
    associative_reducer: bool
    stage_kind_counts: tuple[tuple[str, int], ...]
    intermediate_bytes: int
    handler_costs_us: tuple[tuple[str, float], ...]
    cluster_comm_us: float
    hbm_comm_us: float
    target_compute_capability: tuple[int, int] | None
    target_supports_cluster_launch: bool | None
    target_max_cluster_size: int | None
    target_max_dynamic_shared_memory: int | None
    cost_model_version: str
    schema_version: int = DATAFLOW_SCHEDULER_FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_SCHEDULER_FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Dataflow scheduler feature schema version {self.schema_version}")
        if self.task_count <= 0:
            raise ValueError("Dataflow scheduler features require at least one task")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_count": self.task_count,
            "task_extents": (None if self.task_extents is None else list(self.task_extents)),
            "range_distribution": {
                "min": self.range_length_min,
                "max": self.range_length_max,
                "sum": self.range_length_sum,
                "mean": self.range_length_mean,
                "p50": self.range_length_p50,
                "p95": self.range_length_p95,
                "aligned_count": self.range_length_aligned_count,
            },
            "tile_distribution": {
                "min": self.tile_count_min,
                "max": self.tile_count_max,
                "sum": self.tile_count_sum,
                "histogram": dict(self.tile_count_histogram),
            },
            "topology": {
                "sm_count": self.sm_count,
                "cluster_size": self.cluster_size,
                "cluster_count": self.cluster_count,
            },
            "program": {
                "associative_reducer": self.associative_reducer,
                "stage_kind_counts": dict(self.stage_kind_counts),
                "intermediate_bytes": self.intermediate_bytes,
            },
            "costs": {
                "handler_us": dict(self.handler_costs_us),
                "cluster_comm_us": self.cluster_comm_us,
                "hbm_comm_us": self.hbm_comm_us,
                "cost_model_version": self.cost_model_version,
            },
            "target_resources": {
                "compute_capability": (None if self.target_compute_capability is None else list(self.target_compute_capability)),
                "supports_cluster_launch": self.target_supports_cluster_launch,
                "max_cluster_size": self.target_max_cluster_size,
                "max_dynamic_shared_memory": (self.target_max_dynamic_shared_memory),
            },
        }


@dataclass(frozen=True)
class DataflowSchedulerCandidate:
    """One fully replayable scheduler policy candidate."""

    candidate_id: str
    scheduler_policy: str
    reduce_strategy: str
    queue_policy: str
    tree_policy: str
    chunk_policy: str
    cluster_assignment_policy: str
    scheduler_config_options: tuple[tuple[str, Any], ...] = ()
    streaming_tree_consumer: str | None = None
    cluster_task_assignment: tuple[tuple[int, ...], ...] | str | None = None
    skip_tiny_root_fragment: bool | None = None
    skip_tiny_root_fragment_blocks: int | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("Dataflow scheduler candidate id must be non-empty")
        policy = normalize_scheduler_policy(self.scheduler_policy)
        strategy = normalize_reduce_strategy(self.reduce_strategy)
        object.__setattr__(self, "scheduler_policy", policy)
        object.__setattr__(self, "reduce_strategy", strategy)
        object.__setattr__(
            self,
            "scheduler_config_options",
            normalize_option_items(self.scheduler_config_options),
        )
        object.__setattr__(
            self,
            "cluster_task_assignment",
            normalize_assignment(self.cluster_task_assignment),
        )
        if self.streaming_tree_consumer not in {None, "left", "right"}:
            raise ValueError("Dataflow streaming tree consumer must be 'left', 'right', or None")

    def resolved_scheduler_config(
        self,
        base: DataflowSchedulerConfig,
    ) -> DataflowSchedulerConfig:
        options = dict(self.scheduler_config_options)
        if self.streaming_tree_consumer is not None:
            options["streaming_tree_consumer"] = self.streaming_tree_consumer
        if self.cluster_task_assignment is not None:
            assignment = self.cluster_task_assignment
            options["cluster_task_assignment"] = (
                assignment if isinstance(assignment, str) else "|".join(",".join(str(task_id) for task_id in group) for group in assignment)
            )
        if self.skip_tiny_root_fragment is not None:
            options["skip_tiny_root_fragment"] = self.skip_tiny_root_fragment
        if self.skip_tiny_root_fragment_blocks is not None:
            options["skip_tiny_root_fragment_blocks"] = self.skip_tiny_root_fragment_blocks
        return base.with_options(**options)

    def to_dict(self) -> dict[str, Any]:
        assignment = self.cluster_task_assignment
        return {
            "candidate_id": self.candidate_id,
            "scheduler_policy": self.scheduler_policy,
            "reduce_strategy": self.reduce_strategy,
            "queue_policy": self.queue_policy,
            "tree_policy": self.tree_policy,
            "chunk_policy": self.chunk_policy,
            "cluster_assignment_policy": self.cluster_assignment_policy,
            "scheduler_config_options": dict(self.scheduler_config_options),
            "streaming_tree_consumer": self.streaming_tree_consumer,
            "cluster_task_assignment": (
                assignment if assignment is None or isinstance(assignment, str) else [list(group) for group in assignment]
            ),
            "skip_tiny_root_fragment": self.skip_tiny_root_fragment,
            "skip_tiny_root_fragment_blocks": (self.skip_tiny_root_fragment_blocks),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowSchedulerCandidate:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow scheduler candidate must be a mapping, got {type(value)!r}")
        reject_unknown_fields(
            value,
            {
                "candidate_id",
                "scheduler_policy",
                "reduce_strategy",
                "queue_policy",
                "tree_policy",
                "chunk_policy",
                "cluster_assignment_policy",
                "scheduler_config_options",
                "streaming_tree_consumer",
                "cluster_task_assignment",
                "skip_tiny_root_fragment",
                "skip_tiny_root_fragment_blocks",
            },
            "Dataflow scheduler candidate",
        )
        return cls(
            candidate_id=str(value["candidate_id"]),
            scheduler_policy=str(value["scheduler_policy"]),
            reduce_strategy=str(value["reduce_strategy"]),
            queue_policy=str(value.get("queue_policy", "configured")),
            tree_policy=str(value.get("tree_policy", "none")),
            chunk_policy=str(value.get("chunk_policy", "configured")),
            cluster_assignment_policy=str(value.get("cluster_assignment_policy", "configured")),
            scheduler_config_options=normalize_option_items(value.get("scheduler_config_options")),
            streaming_tree_consumer=value.get("streaming_tree_consumer"),
            cluster_task_assignment=normalize_assignment(value.get("cluster_task_assignment")),
            skip_tiny_root_fragment=value.get("skip_tiny_root_fragment"),
            skip_tiny_root_fragment_blocks=value.get("skip_tiny_root_fragment_blocks"),
        )


@dataclass(frozen=True)
class DataflowSchedulerCandidateScore:
    objective_us: float
    makespan_us: float
    p95_finish_us: float
    finish_spread_us: float
    total_receive_wait_us: float
    max_receive_wait_us: float
    communication_cost_us: float
    hbm_edge_count: int
    instruction_count: int
    slot_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_us": self.objective_us,
            "makespan_us": self.makespan_us,
            "p95_finish_us": self.p95_finish_us,
            "finish_spread_us": self.finish_spread_us,
            "total_receive_wait_us": self.total_receive_wait_us,
            "max_receive_wait_us": self.max_receive_wait_us,
            "communication_cost_us": self.communication_cost_us,
            "hbm_edge_count": self.hbm_edge_count,
            "instruction_count": self.instruction_count,
            "slot_count": self.slot_count,
        }


@dataclass(frozen=True)
class DataflowSchedulerCandidateEvaluation:
    candidate: DataflowSchedulerCandidate
    legal: bool
    score: DataflowSchedulerCandidateScore | None
    plan_fingerprint: str | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if self.legal != (self.score is not None and self.plan_fingerprint is not None):
            raise ValueError("A legal Dataflow scheduler candidate requires a score and plan fingerprint")
        if self.legal and self.rejection_reason is not None:
            raise ValueError("A legal Dataflow scheduler candidate cannot have a rejection")
        if not self.legal and not self.rejection_reason:
            raise ValueError("An illegal Dataflow scheduler candidate requires a rejection")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "legal": self.legal,
            "score": None if self.score is None else self.score.to_dict(),
            "plan_fingerprint": self.plan_fingerprint,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class DataflowSchedulerPolicyRecord:
    record_id: str
    workload_fingerprint: str
    candidate: DataflowSchedulerCandidate
    evidence: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("Dataflow scheduler policy record id must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.workload_fingerprint):
            raise ValueError("Dataflow scheduler policy workload fingerprint must be a SHA-256 hex string")
        if not isinstance(self.candidate, DataflowSchedulerCandidate):
            raise TypeError("Dataflow scheduler policy record candidate must be DataflowSchedulerCandidate")
        object.__setattr__(self, "evidence", normalize_option_items(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "workload_fingerprint": self.workload_fingerprint,
            "candidate": self.candidate.to_dict(),
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowSchedulerPolicyRecord:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow scheduler policy record must be a mapping, got {type(value)!r}")
        reject_unknown_fields(
            value,
            {"record_id", "workload_fingerprint", "candidate", "evidence"},
            "Dataflow scheduler policy record",
        )
        return cls(
            record_id=str(value["record_id"]),
            workload_fingerprint=str(value["workload_fingerprint"]),
            candidate=DataflowSchedulerCandidate.from_dict(value["candidate"]),
            evidence=normalize_option_items(value.get("evidence")),
        )


@dataclass(frozen=True)
class DataflowSchedulerPolicyRecordSet:
    record_set_version: str
    records: tuple[DataflowSchedulerPolicyRecord, ...] = ()
    feature_schema_version: int = DATAFLOW_SCHEDULER_FEATURE_SCHEMA_VERSION
    cost_model_version: str = DATAFLOW_COST_MODEL_VERSION
    schema_version: int = DATAFLOW_SCHEDULER_POLICY_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_SCHEDULER_POLICY_RECORD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Dataflow scheduler policy record schema version {self.schema_version}")
        if not self.record_set_version:
            raise ValueError("Dataflow scheduler policy record set version must be non-empty")
        if self.feature_schema_version <= 0:
            raise ValueError("Dataflow scheduler policy feature schema version must be positive")
        if not self.cost_model_version:
            raise ValueError("Dataflow scheduler policy cost model version must be non-empty")
        records = tuple(self.records)
        if any(not isinstance(record, DataflowSchedulerPolicyRecord) for record in records):
            raise TypeError("Dataflow scheduler policy records must be DataflowSchedulerPolicyRecord")
        object.__setattr__(self, "records", records)
        ids = [record.record_id for record in records]
        fingerprints = [record.workload_fingerprint for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataflow scheduler policy record ids must be unique")
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("Dataflow scheduler policy workload fingerprints must be unique")

    @classmethod
    def empty(
        cls,
        *,
        record_set_version: str = "empty",
        cost_model_version: str = DATAFLOW_COST_MODEL_VERSION,
    ) -> DataflowSchedulerPolicyRecordSet:
        return cls(
            record_set_version=record_set_version,
            cost_model_version=cost_model_version,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    def lookup(self, workload_fingerprint: str) -> DataflowSchedulerPolicyRecord | None:
        for record in self.records:
            if record.workload_fingerprint == workload_fingerprint:
                return record
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _POLICY_RECORD_SCHEMA,
            "schema_version": self.schema_version,
            "record_set_version": self.record_set_version,
            "feature_schema_version": self.feature_schema_version,
            "cost_model_version": self.cost_model_version,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowSchedulerPolicyRecordSet:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow scheduler policy record set must be a mapping, got {type(value)!r}")
        reject_unknown_fields(
            value,
            {
                "schema",
                "schema_version",
                "record_set_version",
                "feature_schema_version",
                "cost_model_version",
                "records",
            },
            "Dataflow scheduler policy record set",
        )
        schema = value.get("schema", _POLICY_RECORD_SCHEMA)
        if schema != _POLICY_RECORD_SCHEMA:
            raise ValueError(f"Unsupported Dataflow scheduler policy record schema {schema!r}")
        raw_records = value.get("records", ())
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            raise TypeError("Dataflow scheduler policy records must be a sequence")
        return cls(
            schema_version=int(
                value.get(
                    "schema_version",
                    DATAFLOW_SCHEDULER_POLICY_RECORD_SCHEMA_VERSION,
                )
            ),
            record_set_version=str(value["record_set_version"]),
            feature_schema_version=int(
                value.get(
                    "feature_schema_version",
                    DATAFLOW_SCHEDULER_FEATURE_SCHEMA_VERSION,
                )
            ),
            cost_model_version=str(value.get("cost_model_version", DATAFLOW_COST_MODEL_VERSION)),
            records=tuple(DataflowSchedulerPolicyRecord.from_dict(record) for record in raw_records),
        )


@dataclass(frozen=True)
class DataflowSchedulerAutoPolicyDecision:
    requested_scheduler_policy: str
    requested_reduce_strategy: str
    workload_fingerprint: str
    features: DataflowSchedulerFeatures
    record_status: str
    record_set_version: str
    record_set_fingerprint: str
    record_id: str | None
    record_rejection_reason: str | None
    candidates: tuple[DataflowSchedulerCandidateEvaluation, ...]
    selected_candidate_id: str
    selection_reason: str
    cost_model_version: str
    selector_version: str = DATAFLOW_SCHEDULER_AUTO_POLICY_VERSION
    schema_version: int = DATAFLOW_SCHEDULER_AUTO_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_SCHEDULER_AUTO_POLICY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Dataflow scheduler auto-policy schema version {self.schema_version}")
        if self.record_status not in {"hit", "miss", "disabled", "incompatible"}:
            raise ValueError(f"Unsupported Dataflow scheduler record status {self.record_status!r}")
        selected = [item for item in self.candidates if item.candidate.candidate_id == self.selected_candidate_id]
        if len(selected) != 1 or not selected[0].legal:
            raise ValueError("Dataflow scheduler auto-policy selection must identify one legal candidate")

    @property
    def selected_evaluation(self) -> DataflowSchedulerCandidateEvaluation:
        return next(item for item in self.candidates if item.candidate.candidate_id == self.selected_candidate_id)

    @property
    def selected_candidate(self) -> DataflowSchedulerCandidate:
        return self.selected_evaluation.candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_version": self.schema_version,
            "selector_version": self.selector_version,
            "feature_schema_version": self.features.schema_version,
            "cost_model_version": self.cost_model_version,
            "requested": {
                "scheduler_policy": self.requested_scheduler_policy,
                "reduce_strategy": self.requested_reduce_strategy,
            },
            "workload_fingerprint": self.workload_fingerprint,
            "features": self.features.to_dict(),
            "record": {
                "status": self.record_status,
                "record_set_version": self.record_set_version,
                "record_set_fingerprint": self.record_set_fingerprint,
                "record_id": self.record_id,
                "rejection_reason": self.record_rejection_reason,
            },
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class DataflowSchedulerAutoPolicyResult:
    decision: DataflowSchedulerAutoPolicyDecision
    selected_plan: InstructionPlan
    selected_scheduler_config: DataflowSchedulerConfig

    def selected_compile_options(self) -> dict[str, Any]:
        candidate = self.decision.selected_candidate
        result = {
            "scheduler_policy": candidate.scheduler_policy,
            "reduce_strategy": candidate.reduce_strategy,
            "scheduler_config": self.selected_scheduler_config,
        }
        if candidate.streaming_tree_consumer is not None:
            result["streaming_tree_consumer"] = candidate.streaming_tree_consumer
        if candidate.cluster_task_assignment is not None:
            result["cluster_task_assignment"] = candidate.cluster_task_assignment
        if candidate.skip_tiny_root_fragment is not None:
            result["skip_tiny_root_fragment"] = candidate.skip_tiny_root_fragment
        if candidate.skip_tiny_root_fragment_blocks is not None:
            result["skip_tiny_root_fragment_blocks"] = candidate.skip_tiny_root_fragment_blocks
        return result


def load_scheduler_policy_records(
    value: DataflowSchedulerPolicyRecordSet | Mapping[str, Any] | str | Path | bool | None = None,
) -> DataflowSchedulerPolicyRecordSet:
    """Load an immutable record set; no workload records are bundled by default."""

    if value is None:
        return DataflowSchedulerPolicyRecordSet.empty(record_set_version="none")
    if value is False:
        return DataflowSchedulerPolicyRecordSet.empty(record_set_version="disabled")
    if value is True:
        raise TypeError("Dataflow scheduler policy records must be a record set, mapping, path, False, or None")
    if isinstance(value, DataflowSchedulerPolicyRecordSet):
        return value
    if isinstance(value, Mapping):
        return DataflowSchedulerPolicyRecordSet.from_dict(value)
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ValueError(f"Cannot read Dataflow scheduler policy records from {path}: {err}") from err
    except json.JSONDecodeError as err:
        raise ValueError(f"Dataflow scheduler policy records at {path} are not valid JSON") from err
    return DataflowSchedulerPolicyRecordSet.from_dict(payload)


def range_axis_and_lengths(
    program: DataflowProgram,
    range_lengths: Mapping[Any, int | Sequence[int]],
) -> tuple[Any, tuple[int, ...]]:
    if not range_lengths:
        raise ValueError("Dataflow scheduler auto policy requires concrete range lengths")
    if program.is_stage_graph:
        flattened = []
        for _, raw in sorted(range_lengths.items(), key=lambda item: repr(item[0])):
            values = (raw,) if isinstance(raw, int) else tuple(raw)
            flattened.extend(int(item) for item in values)
        lengths = tuple(flattened)
        if not lengths or any(item <= 0 for item in lengths):
            raise ValueError("Dataflow scheduler auto policy range lengths must be a non-empty positive sequence")
        return "stage_graph_ranges", lengths
    preferred = None if program.partial_stage is None else program.partial_stage.range_axis
    if preferred in range_lengths:
        axis = preferred
    elif len(range_lengths) == 1:
        axis = next(iter(range_lengths))
    else:
        matches = [axis for axis in range_lengths if str(axis) == str(preferred)]
        if len(matches) != 1:
            raise ValueError("Dataflow scheduler auto policy cannot resolve the dynamic range axis")
        axis = matches[0]
    raw = range_lengths[axis]
    values = (raw,) if isinstance(raw, int) else tuple(raw)
    lengths = tuple(int(item) for item in values)
    if not lengths or any(item <= 0 for item in lengths):
        raise ValueError("Dataflow scheduler auto policy range lengths must be a non-empty positive sequence")
    return axis, lengths


def nearest_rank(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def tile_histogram(tile_counts: Sequence[int]) -> tuple[tuple[str, int], ...]:
    buckets = (
        ("1", 1, 1),
        ("2", 2, 2),
        ("3_4", 3, 4),
        ("5_8", 5, 8),
        ("9_16", 9, 16),
        ("17_32", 17, 32),
        ("33_plus", 33, None),
    )
    return tuple(
        (
            name,
            sum(1 for count in tile_counts if count >= lower and (upper is None or count <= upper)),
        )
        for name, lower, upper in buckets
    )


def dtype_bytes(dtype: str | None) -> int:
    if dtype is None:
        return 0
    match = re.fullmatch(r"(?:float|int|uint|bfloat)(\d+)(?:x(\d+))?", dtype)
    if match is None:
        return 1 if dtype == "bool" else 0
    bits = int(match.group(1))
    lanes = 1 if match.group(2) is None else int(match.group(2))
    return math.ceil(bits * lanes / 8)


def intermediate_bytes(program: DataflowProgram) -> int:
    intermediate = None
    if program.partial_stage is not None:
        intermediate = program.partial_stage.output_type
    elif program.stages:
        intermediate = program.stages[0].output_type
    if intermediate is None:
        return 0
    total = 0
    for field in intermediate.fields:
        elements = 1
        for extent in field.shape or ():
            try:
                elements *= int(extent)
            except (TypeError, ValueError):
                elements = 0
                break
        total += elements * dtype_bytes(field.dtype)
    return total


def extract_scheduler_features(
    program: DataflowProgram,
    *,
    topology: GPUTopology,
    range_lengths: Mapping[Any, int | Sequence[int]],
    block_size: int,
    task_extents: Sequence[int] | None = None,
    scheduler_config: DataflowSchedulerConfig | Mapping[str, Any] | None = None,
    target_capabilities: Any | None = None,
) -> DataflowSchedulerFeatures:
    """Build the versioned feature vector without exact-list policy branches."""

    if not isinstance(program, DataflowProgram):
        raise TypeError(f"Expected DataflowProgram, got {program!r}")
    if not isinstance(topology, GPUTopology):
        raise TypeError(f"Expected GPUTopology, got {topology!r}")
    if block_size <= 0:
        raise ValueError("Dataflow scheduler auto-policy block size must be positive")
    if target_capabilities is not None and topology.cluster_size > 1:
        if not target_capabilities.supports_cluster_launch:
            raise ValueError("Dataflow scheduler auto-policy topology requests clusters but the target does not support cluster launch")
        if target_capabilities.max_cluster_size is not None and topology.cluster_size > target_capabilities.max_cluster_size:
            raise ValueError(
                "Dataflow scheduler auto-policy cluster size exceeds the target limit: "
                f"requested={topology.cluster_size}, "
                f"limit={target_capabilities.max_cluster_size}"
            )
    _, lengths = range_axis_and_lengths(program, range_lengths)
    config = resolve_scheduler_config(scheduler_config)
    model = config.cost_model
    tiles = tuple(math.ceil(length / block_size) for length in lengths)
    normalized_task_extents = None if task_extents is None else tuple(int(item) for item in task_extents)
    task_count = math.prod(normalized_task_extents) if program.is_stage_graph and normalized_task_extents is not None else len(lengths)
    kind_counts: dict[str, int] = {}
    for stage in program.stages:
        kind_counts[stage.kind.value] = kind_counts.get(stage.kind.value, 0) + 1
    associative = bool(
        program.reduce_stage is not None
        and program.reduce_stage.reduce_call.operator.reducer_contract is DataflowReducerContract.ASSOCIATIVE_BINARY
    )
    target_cc = None if target_capabilities is None else tuple(int(item) for item in target_capabilities.compute_capability)
    return DataflowSchedulerFeatures(
        task_count=task_count,
        task_extents=normalized_task_extents,
        range_length_min=min(lengths),
        range_length_max=max(lengths),
        range_length_sum=sum(lengths),
        range_length_mean=sum(lengths) / len(lengths),
        range_length_p50=nearest_rank(lengths, 0.50),
        range_length_p95=nearest_rank(lengths, 0.95),
        range_length_aligned_count=sum(1 for length in lengths if length % block_size == 0),
        tile_count_min=min(tiles),
        tile_count_max=max(tiles),
        tile_count_sum=sum(tiles),
        tile_count_histogram=tile_histogram(tiles),
        sm_count=topology.sm_count,
        cluster_size=topology.cluster_size,
        cluster_count=topology.cluster_count,
        associative_reducer=associative,
        stage_kind_counts=tuple(sorted(kind_counts.items())),
        intermediate_bytes=intermediate_bytes(program),
        handler_costs_us=(
            ("finalize", model.finalize_us),
            ("iter_base", model.iter_base_us),
            ("iter_per_block", model.iter_per_block_us),
            ("reduce_multi", model.reduce_multi_us),
            ("reduce_single", model.reduce_single_us),
        ),
        cluster_comm_us=model.cluster_comm_us,
        hbm_comm_us=model.hbm_comm_us,
        target_compute_capability=target_cc,
        target_supports_cluster_launch=(None if target_capabilities is None else bool(target_capabilities.supports_cluster_launch)),
        target_max_cluster_size=(None if target_capabilities is None else target_capabilities.max_cluster_size),
        target_max_dynamic_shared_memory=(None if target_capabilities is None else target_capabilities.max_dynamic_shared_memory),
        cost_model_version=model.version,
    )


def serialized_ranges(
    values: Mapping[Any, int | Sequence[int]] | None,
) -> dict[str, list[int]] | None:
    if values is None:
        return None
    result: dict[str, list[int]] = {}
    for axis, raw in sorted(values.items(), key=lambda item: repr(item[0])):
        key = str(axis)
        if key in result:
            raise ValueError("Dataflow scheduler auto-policy range axes must have unique string identities")
        normalized = (raw,) if isinstance(raw, int) else tuple(raw)
        result[key] = [int(item) for item in normalized]
    return result


def scheduler_workload_fingerprint(
    program: DataflowProgram,
    *,
    features: DataflowSchedulerFeatures,
    range_lengths: Mapping[Any, int | Sequence[int]],
    range_offsets: Mapping[Any, int | Sequence[int]] | None,
    block_size: int,
    task_extents: Sequence[int] | None,
    include_exit: bool,
    requested_scheduler_policy: str,
    requested_reduce_strategy: str,
    scheduler_config: DataflowSchedulerConfig,
    schedule_options: Mapping[str, Any],
) -> str:
    """Return the exact fingerprint used only for policy-record lookup."""

    return canonical_fingerprint(
        {
            "feature_schema_version": features.schema_version,
            "selector_version": DATAFLOW_SCHEDULER_AUTO_POLICY_VERSION,
            "program_fingerprint": canonical_fingerprint(program),
            "features": features.to_dict(),
            "range_lengths": serialized_ranges(range_lengths),
            "range_offsets": serialized_ranges(range_offsets),
            "block_size": int(block_size),
            "task_extents": (None if task_extents is None else tuple(int(item) for item in task_extents)),
            "include_exit": bool(include_exit),
            "requested_scheduler_policy": requested_scheduler_policy,
            "requested_reduce_strategy": requested_reduce_strategy,
            "scheduler_config": scheduler_config.to_dict(),
            "schedule_options": dict(schedule_options),
        }
    )


def round_robin_cluster_assignment(
    task_count: int,
    cluster_count: int,
) -> tuple[tuple[int, ...], ...]:
    groups = [[] for _ in range(cluster_count)]
    for task_id in range(task_count):
        groups[task_id % cluster_count].append(task_id)
    return tuple(tuple(group) for group in groups)


def generate_scheduler_candidates(
    features: DataflowSchedulerFeatures,
    *,
    requested_scheduler_policy: str = DATAFLOW_SCHEDULER_AUTO,
    requested_reduce_strategy: str = DATAFLOW_SCHEDULER_AUTO,
    scheduler_config: DataflowSchedulerConfig | None = None,
    streaming_tree_consumer: str | None = None,
    cluster_task_assignment: Sequence[Sequence[int]] | str | None = None,
    skip_tiny_root_fragment: bool | None = None,
    skip_tiny_root_fragment_blocks: int | None = None,
    stage_graph: bool = False,
    reduce_output_alias_input_indices: Sequence[int] = (),
    fused_reduce_finalize_arities: Sequence[int] = (),
    direct_leaf_acc: bool | None = None,
) -> tuple[DataflowSchedulerCandidate, ...]:
    """Generate candidates from schemas and distributions, never exact workloads."""

    config = scheduler_config or DataflowSchedulerConfig()
    policy_filter = (
        None if requested_scheduler_policy == DATAFLOW_SCHEDULER_AUTO else normalize_scheduler_policy(requested_scheduler_policy)
    )
    strategy_filter = None if requested_reduce_strategy == DATAFLOW_SCHEDULER_AUTO else normalize_reduce_strategy(requested_reduce_strategy)
    candidates: list[DataflowSchedulerCandidate] = []

    def allowed(policy: str, strategy: str) -> bool:
        return (policy_filter is None or policy == policy_filter) and (strategy_filter is None or strategy == strategy_filter)

    def add(
        candidate_id: str,
        policy: str,
        strategy: str,
        *,
        queue_policy: str = "not_applicable",
        tree_policy: str = "none",
        chunk_policy: str = "not_applicable",
        assignment_policy: str = "not_applicable",
        config_options: Mapping[str, Any] | None = None,
        consumer: str | None = None,
        assignment: Sequence[Sequence[int]] | str | None = None,
        skip_tiny: bool | None = None,
    ) -> None:
        if allowed(policy, strategy):
            candidates.append(
                DataflowSchedulerCandidate(
                    candidate_id=candidate_id,
                    scheduler_policy=policy,
                    reduce_strategy=strategy,
                    queue_policy=queue_policy,
                    tree_policy=tree_policy,
                    chunk_policy=chunk_policy,
                    cluster_assignment_policy=assignment_policy,
                    scheduler_config_options=normalize_option_items(config_options),
                    streaming_tree_consumer=consumer,
                    cluster_task_assignment=(cluster_task_assignment if assignment is None else assignment),
                    skip_tiny_root_fragment=(skip_tiny_root_fragment if skip_tiny is None else skip_tiny),
                    skip_tiny_root_fragment_blocks=skip_tiny_root_fragment_blocks,
                )
            )

    add("cyclic_all_at_once", "round_robin", "all_at_once")
    if stage_graph:
        if not candidates:
            raise DataflowSchedulerAutoPolicyError("Linear stage-graph scheduling only supports round_robin/all_at_once")
        return tuple(candidates)

    add(
        "cluster_local_all_at_once",
        "cluster_local",
        "all_at_once",
        assignment_policy="task_cluster_local",
    )

    queue_is_configured = config.contains("level0_queue_order")
    chunk_is_configured = config.contains("chunk_swap_search")
    configured_assignment = cluster_task_assignment is not None or config.contains("cluster_task_assignment")
    queue_variants = (
        (("configured", {}),)
        if queue_is_configured
        else (
            ("task", {"level0_queue_order": "task"}),
            ("long_first", {"level0_queue_order": "long_first"}),
            ("replay", {"level0_queue_order": "replay"}),
        )
    )
    assignments: tuple[tuple[str, Any], ...] = (("load_balanced", None),)
    if configured_assignment:
        assignments = (("configured", None),)
    elif features.cluster_count > 1 and features.task_count > 1:
        round_robin = round_robin_cluster_assignment(
            features.task_count,
            features.cluster_count,
        )
        assignments += (("round_robin", round_robin),)

    for queue_policy, queue_options in queue_variants:
        for assignment_policy, assignment in assignments:
            suffix = f"{queue_policy}_{assignment_policy}"
            add(
                f"cluster_streaming_{suffix}",
                "cluster_local",
                "streaming",
                queue_policy=queue_policy,
                chunk_policy=("configured" if chunk_is_configured else "balanced"),
                assignment_policy=assignment_policy,
                config_options=queue_options,
                assignment=assignment,
            )
    if not chunk_is_configured:
        add(
            "cluster_streaming_replay_chunk_search",
            "cluster_local",
            "streaming",
            queue_policy=("configured" if queue_is_configured else "replay"),
            chunk_policy="swap_search",
            assignment_policy=("configured" if configured_assignment else "load_balanced"),
            config_options={
                **({} if queue_is_configured else {"level0_queue_order": "replay"}),
                "chunk_swap_search": True,
            },
        )

    if features.associative_reducer:
        consumers = (streaming_tree_consumer,) if streaming_tree_consumer is not None else ("left", "right")
        tree_queue_policy, tree_queue_options = queue_variants[0]
        for consumer in consumers:
            for assignment_policy, assignment in assignments:
                add(
                    f"cluster_tree_{consumer}_{tree_queue_policy}_{assignment_policy}",
                    "cluster_local",
                    "streaming_tree",
                    queue_policy=tree_queue_policy,
                    tree_policy=f"binary_{consumer}",
                    chunk_policy=("configured" if chunk_is_configured else "balanced"),
                    assignment_policy=assignment_policy,
                    config_options=tree_queue_options,
                    consumer=consumer,
                    assignment=assignment,
                )
            if skip_tiny_root_fragment is None:
                add(
                    f"cluster_tree_{consumer}_skip_tiny_root",
                    "cluster_local",
                    "streaming_tree",
                    queue_policy=tree_queue_policy,
                    tree_policy=f"binary_{consumer}",
                    chunk_policy=("configured" if chunk_is_configured else "balanced"),
                    assignment_policy=("configured" if configured_assignment else "load_balanced"),
                    config_options=tree_queue_options,
                    consumer=consumer,
                    skip_tiny=True,
                )

        alias_consumers = tuple(
            dict.fromkeys(
                "right" if input_index == 0 else "left" for input_index in reduce_output_alias_input_indices if input_index in {0, 1}
            )
        )

        # A resident direct-leaf tree is a useful large-cluster anchor, but its
        # remote reducer input still needs an independent physical lifetime.
        # Keep it under the joint slot contract so auto scoring and lowering
        # agree about producer-push destination reuse.
        resident_required_flags = {
            "direct_leaf_acc": True,
            "ready_time_tree": True,
            "level_bucket_tree": True,
            "joint_schedule": True,
            "joint_hbm_async_receive_pipeline": True,
        }
        resident_flags_allowed = direct_leaf_acc is not False and all(
            not config.contains(name) or config.get(name) is expected for name, expected in resident_required_flags.items()
        )
        if resident_flags_allowed and alias_consumers and features.cluster_size >= 8:
            resident_defaults = {
                **resident_required_flags,
                "level0_queue_order": "long_first",
                "chunk_swap_search": True,
            }
            resident_options = {name: value for name, value in resident_defaults.items() if not config.contains(name)}
            for consumer in alias_consumers:
                if streaming_tree_consumer is not None and streaming_tree_consumer != consumer:
                    continue
                for assignment_policy, assignment in assignments:
                    common_candidate = {
                        "policy": "cluster_local",
                        "strategy": "streaming_tree",
                        "queue_policy": ("configured" if config.contains("level0_queue_order") else "long_first"),
                        "tree_policy": f"resident_binary_{consumer}",
                        "chunk_policy": ("configured" if config.contains("chunk_swap_search") else "swap_search"),
                        "assignment_policy": assignment_policy,
                        "config_options": resident_options,
                        "consumer": consumer,
                        "assignment": assignment,
                    }
                    add(
                        f"resident_cluster_tree_{consumer}_{assignment_policy}",
                        **common_candidate,
                    )
                    if skip_tiny_root_fragment is None:
                        add(
                            f"resident_cluster_tree_{consumer}_{assignment_policy}_skip_tiny",
                            skip_tiny=True,
                            **common_candidate,
                        )

        required_joint_flags = {
            "critical_path_split": True,
            "direct_leaf_acc": True,
            "hier_cross_cluster_split": True,
            "ready_time_tree": True,
            "joint_schedule": True,
            "joint_cluster_async_receive_pipeline": True,
            "joint_hbm_async_receive_pipeline": True,
            "joint_reuse_resident_handler_threads": True,
            "joint_scratch_release_queue_reorder": True,
        }
        joint_flags_allowed = all(
            not config.contains(name) or config.get(name) is expected for name, expected in required_joint_flags.items()
        )
        coordinated_tree_flags = {
            "ordered_tree_min_height": True,
            "balance_tree_leaf_pair_skew": True,
            "joint_hbm_spill_blocked_cluster_push": True,
        }
        coordinated_tree_flags_allowed = all(
            not config.contains(name) or config.get(name) is expected for name, expected in coordinated_tree_flags.items()
        )
        joint_requested = bool(
            config.get("joint_schedule", False)
            or config.get("joint_hbm_async_receive_pipeline", False)
            or config.get("joint_hbm_segmented_pipeline", False)
            or features.cluster_size >= 8
        )
        if joint_requested and joint_flags_allowed and features.cluster_count > 1 and features.tile_count_sum > features.sm_count:
            capacity_defaults = {
                **required_joint_flags,
                "critical_path_capacity_search_max_evals": 4,
                "critical_path_max_task_segments": features.cluster_count,
                "critical_path_max_extra_chunks": (features.sm_count + features.task_count),
                "critical_path_extra_chunk_penalty_us": 0.0,
                "critical_path_hbm_penalty_us": 0.0,
                "critical_path_p95_weight": 0.0,
                "critical_path_recv_weight": 0.0,
                "critical_path_total_recv_weight": 0.0,
                "critical_path_min_gain_us": 0.0,
                "level0_queue_order": "producer_ready",
            }
            capacity_options = {name: value for name, value in capacity_defaults.items() if not config.contains(name)}
            for consumer in alias_consumers:
                if streaming_tree_consumer is not None and streaming_tree_consumer != consumer:
                    continue
                common_candidate = {
                    "policy": "cluster_local",
                    "strategy": "streaming_tree",
                    "queue_policy": ("configured" if config.contains("level0_queue_order") else "producer_ready"),
                    "chunk_policy": "critical_path_capacity",
                    "assignment_policy": ("configured" if configured_assignment else "load_balanced"),
                    "consumer": consumer,
                }
                if not config.contains("ordered_interval_tree") or not config.get("ordered_interval_tree"):
                    add(
                        f"capacity_ordered_joint_{consumer}",
                        tree_policy=f"binary_{consumer}",
                        config_options=capacity_options,
                        **common_candidate,
                    )
                if not config.contains("ordered_interval_tree") or config.get("ordered_interval_tree"):
                    ordered_options = dict(capacity_options)
                    if not config.contains("ordered_interval_tree"):
                        ordered_options["ordered_interval_tree"] = True
                    add(
                        f"capacity_ordered_tree_joint_{consumer}",
                        tree_policy=f"ordered_interval_{consumer}",
                        config_options=ordered_options,
                        **common_candidate,
                    )
                    if coordinated_tree_flags_allowed and features.cluster_size >= 8:
                        coordinated_options = dict(ordered_options)
                        coordinated_options.update(
                            {name: value for name, value in coordinated_tree_flags.items() if not config.contains(name)}
                        )
                        add(
                            f"capacity_coordinated_tree_joint_{consumer}",
                            tree_policy=f"ordered_interval_coordinated_{consumer}",
                            config_options=coordinated_options,
                            **common_candidate,
                        )

            global_capacity_flags = {
                **required_joint_flags,
                "global_capacity_chunks": True,
                "level0_queue_order": "producer_ready",
                "ordered_interval_tree": True,
            }
            global_capacity_flags_allowed = all(
                not config.contains(name) or config.get(name) is expected for name, expected in global_capacity_flags.items()
            )
            if global_capacity_flags_allowed and coordinated_tree_flags_allowed:
                available_finalizer_arities = tuple(sorted({int(arity) for arity in fused_reduce_finalize_arities if int(arity) >= 2}))
                finalizer_limits: tuple[int | None, ...] = (
                    (None,) if not available_finalizer_arities else tuple(available_finalizer_arities)
                )
                root_ready_flags = {
                    "global_capacity_replay_refine": True,
                    "cross_cluster_root_global_candidates": True,
                    "root_reduce_cluster_candidates": True,
                    "ready_time_finalize": True,
                }

                def optional_variant_allowed(
                    variant: Mapping[str, Any],
                ) -> bool:
                    return all(not config.contains(name) or config.get(name) == value for name, value in variant.items())

                # Moving a tree root/finalizer away from its input owner only
                # has a chance to hide useful work once there are multiple
                # task waves per cluster and enough tiles per task to keep the
                # original root owner busy.  Below that topology-derived
                # threshold the extra handoff is pure fixed overhead.
                root_ready_search_useful = (
                    features.task_count > features.cluster_size
                    and features.tile_count_sum > features.task_count * 2 * features.cluster_size
                )
                for consumer in alias_consumers:
                    if streaming_tree_consumer is not None and streaming_tree_consumer != consumer:
                        continue
                    packing_variants = [
                        ("capacity", {}),
                        (
                            "minimax",
                            {"global_capacity_minimax_chunks": True},
                        ),
                    ]
                    if features.task_count <= features.sm_count:
                        packing_variants.append(
                            (
                                "copacked",
                                {"global_capacity_copacked_chunks": True},
                            )
                        )
                    for packing_suffix, packing_options in packing_variants:
                        if not optional_variant_allowed(packing_options):
                            continue
                        search_variants: list[tuple[str, dict[str, Any]]] = [
                            ("", {}),
                        ]
                        # Replay refinement can converge capacity and minimax
                        # seeds to the same partition.  Search it once from the
                        # capacity seed and combine it with the broader root
                        # placement domain whose readiness is included in the
                        # same replay objective.
                        if packing_suffix == "capacity" and root_ready_search_useful and optional_variant_allowed(root_ready_flags):
                            search_variants.append(("_replay_root_ready", dict(root_ready_flags)))
                        for max_arity in finalizer_limits:
                            for search_suffix, search_options in search_variants:
                                defaults = dict(global_capacity_flags)
                                defaults.update(coordinated_tree_flags)
                                defaults.update(packing_options)
                                defaults.update(search_options)
                                if max_arity is not None:
                                    defaults["fused_reduce_finalize_max_arity"] = max_arity
                                options = {name: value for name, value in defaults.items() if not config.contains(name)}
                                suffix = packing_suffix
                                if max_arity is not None:
                                    suffix += f"_fused{max_arity}"
                                suffix += search_suffix
                                add(
                                    f"global_{suffix}_tree_joint_{consumer}",
                                    "cluster_local",
                                    "streaming_tree",
                                    queue_policy="producer_ready",
                                    tree_policy=f"global_capacity_{consumer}",
                                    chunk_policy=(f"global_{packing_suffix}"),
                                    assignment_policy="global_capacity",
                                    config_options=options,
                                    consumer=consumer,
                                )

    if not candidates:
        raise DataflowSchedulerAutoPolicyError(
            "The requested scheduler policy/reduce strategy has no generic candidates: "
            f"policy={requested_scheduler_policy!r}, "
            f"strategy={requested_reduce_strategy!r}"
        )
    unique: dict[str, DataflowSchedulerCandidate] = {}
    for candidate in candidates:
        unique.setdefault(serialize_canonical_json(candidate.to_dict()), candidate)
    return tuple(unique.values())


def instruction_duration_us(
    instruction: Any,
    *,
    block_size: int,
    model: DataflowCostModelConfig,
) -> float:
    if instruction.opcode in {DataflowOpcode.ITER, DataflowOpcode.MAP}:
        range_length = block_size if instruction.task_range is None else instruction.task_range.length
        return estimate_iter_cost_us(
            model,
            range_length=range_length,
            block_size=block_size,
        )
    if instruction.opcode in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE}:
        input_count = len(instruction.input_slots)
        if input_count <= 2:
            return estimate_reduce_cost_us(model, input_count=input_count)
        return model.reduce_multi_us * (input_count - 1)
    if instruction.opcode is DataflowOpcode.FINALIZE:
        return estimate_finalize_cost_us(model)
    return 0.0


def score_instruction_plan(
    plan: InstructionPlan,
    *,
    force_hbm_comms: bool = False,
) -> DataflowSchedulerCandidateScore:
    """Replay one concrete plan through the versioned conservative cost model."""

    model = plan.scheduler_config.cost_model
    if plan.joint_execution_plan is not None:
        joint = plan.joint_execution_plan.require_valid(topology=plan.topology)
        schedule = joint.schedule
        cta_finish: dict[int, float] = {}
        for event in schedule.events:
            if event.cta_id is not None:
                cta_finish[event.cta_id] = max(
                    cta_finish.get(event.cta_id, 0.0),
                    event.end_us,
                )
        active = sorted(value for value in cta_finish.values() if value > 0.0)
        p95 = active[max(0, math.ceil(0.95 * len(active)) - 1)] if active else 0.0
        spread = (active[-1] - active[0]) if active else 0.0
        communication_cost = sum(
            model.cluster_comm_us if comm.kind is DataflowCommKind.CLUSTER_SEND else model.hbm_comm_us
            for comm in plan.comms
            if comm.kind in {DataflowCommKind.CLUSTER_SEND, DataflowCommKind.HBM_SEND}
        )
        return DataflowSchedulerCandidateScore(
            objective_us=schedule.makespan_us,
            makespan_us=schedule.makespan_us,
            p95_finish_us=p95,
            finish_spread_us=spread,
            total_receive_wait_us=schedule.total_guard_wait_us,
            max_receive_wait_us=schedule.critical_path_guard_wait_us,
            communication_cost_us=communication_cost,
            hbm_edge_count=sum(comm.kind is DataflowCommKind.HBM_SEND for comm in plan.comms),
            instruction_count=len(plan.instructions),
            slot_count=len(plan.slots),
        )
    instructions = {instruction.instruction_id: instruction for instruction in plan.instructions}
    slot_producers = {slot.slot_id: slot.producer_instruction_id for slot in plan.slots}
    edge_costs: dict[tuple[int, int, int], float] = {}
    communication_cost = 0.0
    hbm_edges = 0
    for comm in plan.comms:
        if comm.kind not in {DataflowCommKind.CLUSTER_SEND, DataflowCommKind.HBM_SEND}:
            continue
        cost = model.cluster_comm_us if comm.kind is DataflowCommKind.CLUSTER_SEND else model.hbm_comm_us
        edge_costs[
            (
                comm.source_instruction_id,
                comm.target_instruction_id,
                comm.target_slot_id,
            )
        ] = cost
        slot_producers[comm.target_slot_id] = comm.source_instruction_id
        communication_cost += cost
        hbm_edges += int(comm.kind is DataflowCommKind.HBM_SEND)

    queue_indices = {sm_id: 0 for sm_id in plan.queues}
    sm_ready = {sm_id: 0.0 for sm_id in range(plan.topology.sm_count)}
    instruction_finish: dict[int, float] = {}
    total_wait = 0.0
    max_wait = 0.0
    remaining = sum(len(queue) for queue in plan.queues.values())
    while remaining:
        progressed = False
        for sm_id in sorted(plan.queues):
            index = queue_indices[sm_id]
            queue = plan.queues[sm_id]
            if index >= len(queue):
                continue
            instruction = queue[index]
            input_ready_values = []
            blocked = False
            for slot_id in instruction.input_slots:
                producer_id = slot_producers.get(slot_id)
                if producer_id is None:
                    input_ready_values.append(0.0)
                    continue
                if producer_id not in instruction_finish:
                    blocked = True
                    break
                producer = instructions[producer_id]
                comm_cost = edge_costs.get((producer_id, instruction.instruction_id, slot_id))
                if comm_cost is None:
                    comm_cost = estimate_communication_cost_us(
                        model,
                        same_sm=producer.sm_id == instruction.sm_id,
                        same_cluster=(
                            producer.sm_id is not None
                            and instruction.sm_id is not None
                            and plan.topology.same_cluster(
                                producer.sm_id,
                                instruction.sm_id,
                            )
                        ),
                        force_hbm=force_hbm_comms,
                    )
                input_ready_values.append(instruction_finish[producer_id] + comm_cost)
            if blocked:
                continue
            queue_ready = sm_ready[sm_id]
            input_ready = max(input_ready_values, default=0.0)
            receive_wait = max(0.0, input_ready - queue_ready)
            finish = max(queue_ready, input_ready) + instruction_duration_us(
                instruction,
                block_size=plan.block_size,
                model=model,
            )
            sm_ready[sm_id] = finish
            instruction_finish[instruction.instruction_id] = finish
            total_wait += receive_wait
            max_wait = max(max_wait, receive_wait)
            queue_indices[sm_id] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            blocked_ids = [
                plan.queues[sm_id][queue_indices[sm_id]].instruction_id
                for sm_id in sorted(plan.queues)
                if queue_indices[sm_id] < len(plan.queues[sm_id])
            ]
            raise RuntimeError(f"Dataflow scheduler candidate replay could not resolve dependencies for instructions {blocked_ids[:8]}")

    active = sorted(value for value in sm_ready.values() if value > 0.0)
    makespan = active[-1] if active else 0.0
    p95 = active[max(0, math.ceil(0.95 * len(active)) - 1)] if active else 0.0
    spread = (active[-1] - active[0]) if active else 0.0
    objective = (
        makespan
        + 0.10 * p95
        + 0.05 * spread
        + communication_cost / max(1, plan.topology.sm_count)
        + 0.25 * max_wait
        + 0.01 * total_wait
        + 0.005 * len(plan.instructions)
    )
    return DataflowSchedulerCandidateScore(
        objective_us=objective,
        makespan_us=makespan,
        p95_finish_us=p95,
        finish_spread_us=spread,
        total_receive_wait_us=total_wait,
        max_receive_wait_us=max_wait,
        communication_cost_us=communication_cost,
        hbm_edge_count=hbm_edges,
        instruction_count=len(plan.instructions),
        slot_count=len(plan.slots),
    )


def evaluate_candidate(
    candidate: DataflowSchedulerCandidate,
    *,
    program: DataflowProgram,
    topology: GPUTopology,
    range_lengths: Mapping[Any, int | Sequence[int]],
    range_offsets: Mapping[Any, int | Sequence[int]] | None,
    block_size: int,
    task_extents: Sequence[int] | None,
    include_exit: bool,
    base_scheduler_config: DataflowSchedulerConfig,
    schedule_options: Mapping[str, Any],
    require_linear_slot_lifetime: bool,
    candidate_plan_validator: Callable[[InstructionPlan], str | None] | None,
) -> tuple[DataflowSchedulerCandidateEvaluation, InstructionPlan | None, DataflowSchedulerConfig]:
    scheduler_config = candidate.resolved_scheduler_config(base_scheduler_config)
    joint_slot_lifetime = bool(scheduler_config.get("joint_schedule", False))
    if require_linear_slot_lifetime and candidate.reduce_strategy != "streaming" and not joint_slot_lifetime:
        return (
            DataflowSchedulerCandidateEvaluation(
                candidate=candidate,
                legal=False,
                score=None,
                plan_fingerprint=None,
                rejection_reason=(
                    "resource constraint: scratch-backed slots in auto scheduling require a bounded linear accumulator lifetime"
                ),
            ),
            None,
            scheduler_config,
        )
    scheduler_options = dict(schedule_options)
    scheduler_options.pop("scratch_backed_slots", None)
    try:
        plan = schedule(
            program,
            topology=topology,
            range_lengths=dict(range_lengths),
            range_offsets=(None if range_offsets is None else dict(range_offsets)),
            block_size=block_size,
            task_extents=task_extents,
            include_exit=include_exit,
            scheduler_policy=candidate.scheduler_policy,
            reduce_strategy=candidate.reduce_strategy,
            streaming_tree_consumer=candidate.streaming_tree_consumer,
            cluster_task_assignment=candidate.cluster_task_assignment,
            skip_tiny_root_fragment=candidate.skip_tiny_root_fragment,
            skip_tiny_root_fragment_blocks=(candidate.skip_tiny_root_fragment_blocks),
            scheduler_config=scheduler_config,
            pic=False,
            **scheduler_options,
        )
        rejection_reason = None if candidate_plan_validator is None else candidate_plan_validator(plan)
        if rejection_reason is not None and (not isinstance(rejection_reason, str) or not rejection_reason):
            raise TypeError("Dataflow scheduler candidate plan validator must return a non-empty string or None")
        score = (
            None
            if rejection_reason is not None
            else score_instruction_plan(
                plan,
                force_hbm_comms=bool(schedule_options.get("force_hbm_comms", False)),
            )
        )
    except (TypeError, ValueError, RuntimeError) as err:
        return (
            DataflowSchedulerCandidateEvaluation(
                candidate=candidate,
                legal=False,
                score=None,
                plan_fingerprint=None,
                rejection_reason=f"{type(err).__name__}: {err}",
            ),
            None,
            scheduler_config,
        )
    if rejection_reason is not None:
        return (
            DataflowSchedulerCandidateEvaluation(
                candidate=candidate,
                legal=False,
                score=None,
                plan_fingerprint=None,
                rejection_reason=rejection_reason,
            ),
            None,
            scheduler_config,
        )
    assert score is not None
    return (
        DataflowSchedulerCandidateEvaluation(
            candidate=candidate,
            legal=True,
            score=score,
            plan_fingerprint=instruction_plan_fingerprint(plan),
            rejection_reason=None,
        ),
        plan,
        scheduler_config,
    )


def candidate_tie_break(
    candidate: DataflowSchedulerCandidate,
) -> tuple[int, int, int, int]:
    strategy_priority = {
        "streaming": 0,
        "streaming_tree": 1,
        "all_at_once": 2,
    }[candidate.reduce_strategy]
    policy_priority = 0 if candidate.scheduler_policy == "cluster_local" else 1
    tree_priority = 0 if candidate.tree_policy.startswith("ordered_interval_") else 1
    # Equal predicted makespans are common when two trees have the same
    # abstract event critical path.  Prefer the large-cluster coordinated
    # variant in that tie: it additionally bounds reduction height, balances
    # paired leaf readiness, and replaces producer-side head-of-line cluster
    # waits with the already modeled HBM fallback.  This is a topology/runtime
    # robustness rule, not a workload fingerprint or shape exception.
    coordination_priority = 0 if candidate.tree_policy.startswith("ordered_interval_coordinated_") else 1
    return (
        strategy_priority,
        policy_priority,
        tree_priority,
        coordination_priority,
    )


def select_scheduler_auto_policy(
    program: DataflowProgram,
    *,
    topology: GPUTopology,
    range_lengths: Mapping[Any, int | Sequence[int]],
    block_size: int,
    range_offsets: Mapping[Any, int | Sequence[int]] | None = None,
    task_extents: Sequence[int] | None = None,
    include_exit: bool = True,
    requested_scheduler_policy: str = DATAFLOW_SCHEDULER_AUTO,
    requested_reduce_strategy: str = DATAFLOW_SCHEDULER_AUTO,
    scheduler_config: DataflowSchedulerConfig | Mapping[str, Any] | None = None,
    target_capabilities: Any | None = None,
    policy_records: DataflowSchedulerPolicyRecordSet | Mapping[str, Any] | str | Path | bool | None = None,
    force_hbm_comms: bool = False,
    scratch_backed_slots: bool = False,
    partial_only: bool = False,
    direct_leaf_acc: bool | None = None,
    streaming_tree_consumer: str | None = None,
    cluster_task_assignment: Sequence[Sequence[int]] | str | None = None,
    skip_tiny_root_fragment: bool | None = None,
    skip_tiny_root_fragment_blocks: int | None = None,
    iter_range_buckets: Any = None,
    iter_range_bucket_size: int | None = None,
    iter_range_exact_lengths: Any = None,
    task_coord_overrides: Sequence[Sequence[int] | int] | None = None,
    stage_graph_task_weights: Sequence[int | float] | None = None,
    stage_graph_cluster_assignment: Sequence[int] | None = None,
    _candidate_plan_validator: Callable[[InstructionPlan], str | None] | None = None,
) -> DataflowSchedulerAutoPolicyResult:
    """Select, replay, and score a scheduler policy for one compile request."""

    requested_scheduler_policy = (
        DATAFLOW_SCHEDULER_AUTO
        if requested_scheduler_policy == DATAFLOW_SCHEDULER_AUTO
        else normalize_scheduler_policy(requested_scheduler_policy)
    )
    requested_reduce_strategy = (
        DATAFLOW_SCHEDULER_AUTO
        if requested_reduce_strategy == DATAFLOW_SCHEDULER_AUTO
        else normalize_reduce_strategy(requested_reduce_strategy)
    )
    base_config = resolve_scheduler_config(scheduler_config)
    records = load_scheduler_policy_records(policy_records)
    features = extract_scheduler_features(
        program,
        topology=topology,
        range_lengths=range_lengths,
        block_size=block_size,
        task_extents=task_extents,
        scheduler_config=base_config,
        target_capabilities=target_capabilities,
    )
    schedule_options = {
        "force_hbm_comms": bool(force_hbm_comms),
        "scratch_backed_slots": bool(scratch_backed_slots),
        "partial_only": bool(partial_only),
        "direct_leaf_acc": direct_leaf_acc,
        "iter_range_buckets": iter_range_buckets,
        "iter_range_bucket_size": iter_range_bucket_size,
        "iter_range_exact_lengths": iter_range_exact_lengths,
        "task_coord_overrides": task_coord_overrides,
        "stage_graph_task_weights": stage_graph_task_weights,
        "stage_graph_cluster_assignment": stage_graph_cluster_assignment,
    }
    workload_fingerprint = scheduler_workload_fingerprint(
        program,
        features=features,
        range_lengths=range_lengths,
        range_offsets=range_offsets,
        block_size=block_size,
        task_extents=task_extents,
        include_exit=include_exit,
        requested_scheduler_policy=requested_scheduler_policy,
        requested_reduce_strategy=requested_reduce_strategy,
        scheduler_config=base_config,
        schedule_options=schedule_options,
    )

    record_status = "disabled" if records.record_set_version == "disabled" else "miss"
    record_id = None
    record_rejection = None
    evaluations: list[DataflowSchedulerCandidateEvaluation] = []
    plans: dict[str, InstructionPlan] = {}
    configs: dict[str, DataflowSchedulerConfig] = {}
    record = None
    if records.feature_schema_version != features.schema_version:
        record_status = "incompatible"
        record_rejection = f"feature schema mismatch: record={records.feature_schema_version}, request={features.schema_version}"
    elif records.cost_model_version != base_config.cost_model.version:
        record_status = "incompatible"
        record_rejection = f"cost model mismatch: record={records.cost_model_version}, request={base_config.cost_model.version}"
    elif record_status != "disabled":
        record = records.lookup(workload_fingerprint)

    if record is not None:
        record_id = record.record_id
        evaluation, plan, selected_config = evaluate_candidate(
            record.candidate,
            program=program,
            topology=topology,
            range_lengths=range_lengths,
            range_offsets=range_offsets,
            block_size=block_size,
            task_extents=task_extents,
            include_exit=include_exit,
            base_scheduler_config=base_config,
            schedule_options=schedule_options,
            require_linear_slot_lifetime=False,
            candidate_plan_validator=_candidate_plan_validator,
        )
        evaluations.append(evaluation)
        if evaluation.legal:
            assert plan is not None
            decision = DataflowSchedulerAutoPolicyDecision(
                requested_scheduler_policy=requested_scheduler_policy,
                requested_reduce_strategy=requested_reduce_strategy,
                workload_fingerprint=workload_fingerprint,
                features=features,
                record_status="hit",
                record_set_version=records.record_set_version,
                record_set_fingerprint=records.fingerprint,
                record_id=record.record_id,
                record_rejection_reason=None,
                candidates=tuple(evaluations),
                selected_candidate_id=record.candidate.candidate_id,
                selection_reason=f"best_known_record:{record.record_id}",
                cost_model_version=base_config.cost_model.version,
            )
            return DataflowSchedulerAutoPolicyResult(
                decision=decision,
                selected_plan=plan,
                selected_scheduler_config=selected_config,
            )
        record_status = "miss"
        record_rejection = evaluation.rejection_reason

    candidates = generate_scheduler_candidates(
        features,
        requested_scheduler_policy=requested_scheduler_policy,
        requested_reduce_strategy=requested_reduce_strategy,
        scheduler_config=base_config,
        streaming_tree_consumer=streaming_tree_consumer,
        cluster_task_assignment=cluster_task_assignment,
        skip_tiny_root_fragment=skip_tiny_root_fragment,
        skip_tiny_root_fragment_blocks=skip_tiny_root_fragment_blocks,
        stage_graph=program.is_stage_graph,
        direct_leaf_acc=direct_leaf_acc,
        reduce_output_alias_input_indices=(
            ()
            if program.reduce_stage is None
            else operator_physical_contract(program.reduce_stage.reduce_call.operator.attrs).output_alias_input_indices
        ),
        fused_reduce_finalize_arities=(
            ()
            if program.finalize_stage is None
            else tuple(len(call.operator.input_types) for call in program.finalize_stage.fused_reduce_calls)
        ),
    )
    existing_ids = {item.candidate.candidate_id for item in evaluations}
    for candidate in candidates:
        if candidate.candidate_id in existing_ids:
            continue
        evaluation, plan, candidate_config = evaluate_candidate(
            candidate,
            program=program,
            topology=topology,
            range_lengths=range_lengths,
            range_offsets=range_offsets,
            block_size=block_size,
            task_extents=task_extents,
            include_exit=include_exit,
            base_scheduler_config=base_config,
            schedule_options=schedule_options,
            require_linear_slot_lifetime=(bool(scratch_backed_slots) and requested_reduce_strategy == DATAFLOW_SCHEDULER_AUTO),
            candidate_plan_validator=_candidate_plan_validator,
        )
        evaluations.append(evaluation)
        if plan is not None:
            plans[candidate.candidate_id] = plan
            configs[candidate.candidate_id] = candidate_config

    legal = [item for item in evaluations if item.legal]
    if not legal:
        reasons = "; ".join(f"{item.candidate.candidate_id}: {item.rejection_reason}" for item in evaluations)
        raise DataflowSchedulerAutoPolicyError("Dataflow scheduler auto policy found no legal candidates. " + reasons)
    # When every task can be partitioned over a unique CTA grid, global
    # capacity candidates guarantee at most one ITER producer per CTA.  This
    # is also the only candidate family that structurally removes producer
    # source-lifetime head-of-line waits when the permanent communication slot
    # aliases reusable handler scratch.  Prefer that stronger contract over a
    # slightly lower analytical makespan from a multi-producer CTA queue.
    global_capacity_legal = (
        [item for item in legal if item.candidate.cluster_assignment_policy == "global_capacity"]
        if features.task_count <= features.sm_count
        else []
    )
    if global_capacity_legal:
        best_global_objective_us = min(item.score.objective_us for item in global_capacity_legal if item.score is not None)
        # A co-packed partition may save a small modeled ITER tail while
        # opening several additional reduction and cross-cluster handoff
        # lifetimes.  The conservative replay cannot distinguish alternatives
        # inside one HBM handoff latency reliably.  In that uncertainty band,
        # retain candidates with the least aggregate communication before the
        # ordinary objective tie-break.  A material compute-imbalance gain
        # remains selectable outside the band.
        near_optimal_global = [
            item
            for item in global_capacity_legal
            if (item.score is not None and item.score.objective_us <= best_global_objective_us + base_config.cost_model.hbm_comm_us + 1e-9)
        ]
        minimum_communication_cost_us = min(item.score.communication_cost_us for item in near_optimal_global if item.score is not None)
        global_capacity_legal = [
            item
            for item in near_optimal_global
            if (item.score is not None and item.score.communication_cost_us <= minimum_communication_cost_us + 1e-9)
        ]
    overfull_robust_global: list[DataflowSchedulerCandidateEvaluation] = []
    if features.task_count > features.sm_count:
        best_objective_us = min(item.score.objective_us for item in legal if item.score is not None)
        # The conservative model cannot meaningfully distinguish candidates
        # inside one modeled cluster-transfer latency.  In that equivalence
        # band prefer the first reducer alias side exposed by the physical
        # contract, then the smallest fused finalizer arity.  This keeps a
        # co-packed CTA's accumulator chain shallow and deterministic without
        # excluding the other candidates from the decision artifact.
        near_optimal_global = [
            item
            for item in legal
            if (
                item.candidate.cluster_assignment_policy == "global_capacity"
                and item.score is not None
                and item.score.objective_us <= best_objective_us + base_config.cost_model.cluster_comm_us + 1e-9
            )
        ]
        if near_optimal_global:
            preferred_consumer = next(
                item.candidate.streaming_tree_consumer for item in near_optimal_global if item.candidate.streaming_tree_consumer is not None
            )
            preferred_side = [item for item in near_optimal_global if item.candidate.streaming_tree_consumer == preferred_consumer]

            def fused_finalizer_arity(
                item: DataflowSchedulerCandidateEvaluation,
            ) -> int:
                candidate_options = dict(item.candidate.scheduler_config_options)
                value = candidate_options.get(
                    "fused_reduce_finalize_max_arity",
                    base_config.get("fused_reduce_finalize_max_arity", 1 << 30),
                )
                return int(value)

            minimum_arity = min(map(fused_finalizer_arity, preferred_side))
            overfull_robust_global = [item for item in preferred_side if fused_finalizer_arity(item) == minimum_arity]
    selectable = overfull_robust_global or global_capacity_legal or legal
    selected = min(
        selectable,
        key=lambda item: (
            item.score.objective_us if item.score is not None else math.inf,
            item.score.p95_finish_us if item.score is not None else math.inf,
            item.score.max_receive_wait_us if item.score is not None else math.inf,
            item.score.total_receive_wait_us if item.score is not None else math.inf,
            item.score.communication_cost_us if item.score is not None else math.inf,
            item.score.instruction_count if item.score is not None else math.inf,
            *candidate_tie_break(item.candidate),
        ),
    )
    selected_id = selected.candidate.candidate_id
    decision = DataflowSchedulerAutoPolicyDecision(
        requested_scheduler_policy=requested_scheduler_policy,
        requested_reduce_strategy=requested_reduce_strategy,
        workload_fingerprint=workload_fingerprint,
        features=features,
        record_status=record_status,
        record_set_version=records.record_set_version,
        record_set_fingerprint=records.fingerprint,
        record_id=record_id,
        record_rejection_reason=record_rejection,
        candidates=tuple(evaluations),
        selected_candidate_id=selected_id,
        selection_reason=f"cost_model_minimum:{selected_id}",
        cost_model_version=base_config.cost_model.version,
    )
    return DataflowSchedulerAutoPolicyResult(
        decision=decision,
        selected_plan=plans[selected_id],
        selected_scheduler_config=configs[selected_id],
    )


__all__ = [
    "DATAFLOW_SCHEDULER_AUTO",
    "DATAFLOW_SCHEDULER_AUTO_POLICY_SCHEMA_VERSION",
    "DATAFLOW_SCHEDULER_AUTO_POLICY_VERSION",
    "DATAFLOW_SCHEDULER_FEATURE_SCHEMA_VERSION",
    "DATAFLOW_SCHEDULER_POLICY_RECORD_SCHEMA_VERSION",
    "DataflowSchedulerAutoPolicyDecision",
    "DataflowSchedulerAutoPolicyError",
    "DataflowSchedulerAutoPolicyResult",
    "DataflowSchedulerCandidate",
    "DataflowSchedulerCandidateEvaluation",
    "DataflowSchedulerCandidateScore",
    "DataflowSchedulerFeatures",
    "DataflowSchedulerPolicyRecord",
    "DataflowSchedulerPolicyRecordSet",
    "extract_scheduler_features",
    "generate_scheduler_candidates",
    "load_scheduler_policy_records",
    "scheduler_workload_fingerprint",
    "score_instruction_plan",
    "select_scheduler_auto_policy",
]
