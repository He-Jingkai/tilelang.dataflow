"""Target-aware execution candidate generation and resource validation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any
from collections.abc import Iterator, Mapping, Sequence

from tilelang._typing import DType
from tilelang.dtypes import dtype as TileLangDType
from tilelang.utils.target_capabilities import TargetCapabilitySnapshot

from .abi_schema import (
    DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES,
    DATAFLOW_SHARED_ALIGNMENT,
)
from .compile_config import current_target_capabilities
from .dtype_registry import require_dataflow_dtype
from .operation_contracts import (
    DATAFLOW_PIPELINE_EVICT_FIRST,
    DATAFLOW_PIPELINE_EVICTION_HINTS,
    DATAFLOW_TRANSPORT_ALL_GATHER,
    DATAFLOW_TRANSPORT_AUTO,
    DATAFLOW_TRANSPORT_FAMILIES,
    DATAFLOW_TRANSPORT_HBM,
    DATAFLOW_TRANSPORT_STREAMED,
    DataflowResharedTransportRequest,
)
from .reshared_transport import (
    DataflowResharedTransportPlanningError,
    plan_reshared_transport,
)
from .topology import GPUTopology


DATAFLOW_EXECUTION_PLAN_SCHEMA_VERSION = 1
DATAFLOW_EXECUTION_OVERRIDE_SCHEMA_VERSION = 1
DATAFLOW_EXECUTION_PLANNER_VERSION = "dataflow.execution.constraint.v2"
DATAFLOW_EXECUTION_PLANNER_IMPLEMENTATION = DATAFLOW_EXECUTION_PLANNER_VERSION
# Queue, slot, and barrier metadata share the dynamic allocation with handler
# scratch. The candidate planner already accounts for typed barriers below;
# keep one extra alignment quantum for records that are only known after queue
# materialization. The compiler performs the authoritative exact launch-layout
# check, so a large fixed reserve here would incorrectly reject legal tiles.
DATAFLOW_EXECUTION_SHARED_CONTROL_RESERVE_BYTES = DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES

# These are common-lowering implementation classes, not inherited MoE factory
# defaults. Their meaning is versioned by DATAFLOW_EXECUTION_PLANNER_VERSION and
# every instantiated value is recorded in the candidate plan.
_DATAFLOW_EXECUTION_MATRIX_TILE_CLASSES = (128, 64)
_DATAFLOW_EXECUTION_TASK_TILE_CLASSES = (64, 32, 16)

DATAFLOW_EXECUTION_GEMM_AUTO = "auto"
DATAFLOW_EXECUTION_GEMM_PORTABLE = "portable"
DATAFLOW_EXECUTION_GEMM_WARP_GROUP = "warp_group"
DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M = "warp_group_small_m"
DATAFLOW_EXECUTION_GEMM_FAMILIES = (
    DATAFLOW_EXECUTION_GEMM_AUTO,
    DATAFLOW_EXECUTION_GEMM_PORTABLE,
    DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
    DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M,
)

DATAFLOW_EXECUTION_TRANSFER_AUTO = "auto"
DATAFLOW_EXECUTION_TRANSFER_SYNCHRONOUS = "synchronous"
DATAFLOW_EXECUTION_TRANSFER_TMA = "tma"
DATAFLOW_EXECUTION_TRANSFER_FAMILIES = (
    DATAFLOW_EXECUTION_TRANSFER_AUTO,
    DATAFLOW_EXECUTION_TRANSFER_SYNCHRONOUS,
    DATAFLOW_EXECUTION_TRANSFER_TMA,
)

DATAFLOW_EXECUTION_INPUT_AUTO = "auto"
DATAFLOW_EXECUTION_INPUT_COOPERATIVE = "cooperative"
DATAFLOW_EXECUTION_INPUT_UNICAST = "unicast"
DATAFLOW_EXECUTION_INPUT_MULTICAST = "multicast"
DATAFLOW_EXECUTION_INPUT_DISTRIBUTIONS = (
    DATAFLOW_EXECUTION_INPUT_AUTO,
    DATAFLOW_EXECUTION_INPUT_COOPERATIVE,
    DATAFLOW_EXECUTION_INPUT_UNICAST,
    DATAFLOW_EXECUTION_INPUT_MULTICAST,
)

_CAPTURED_EXECUTION_PLANS: ContextVar[list[DataflowExecutionPlan] | None] = ContextVar(
    "tilelang_dataflow_captured_execution_plans", default=None
)


def serialize_canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(serialize_canonical_json(value).encode("utf-8")).hexdigest()


def positive_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def nonnegative_int(
    value: Any,
    name: str,
    *,
    allow_none: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def choice(value: Any, name: str, choices: Sequence[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"unsupported {name} {value!r}; expected one of {tuple(choices)!r}")
    return normalized


def reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields {sorted(unknown)!r}")


@dataclass(frozen=True)
class DataflowExecutionStageRequest:
    """Logical matrix stage facts used to build hardware candidates."""

    output_extent: int
    reduction_extent: int
    input_dtype: DType
    weight_dtype: DType
    output_dtype: DType
    projection_count: int = 1
    output_partitioned: bool = True
    input_from_previous_stage: bool = False

    def __post_init__(self) -> None:
        positive_int(self.output_extent, "stage output_extent")
        positive_int(self.reduction_extent, "stage reduction_extent")
        positive_int(self.projection_count, "stage projection_count")
        for name in ("output_partitioned", "input_from_previous_stage"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"execution stage {name} must be bool")
        for name in ("input_dtype", "weight_dtype", "output_dtype"):
            dtype_info = require_dataflow_dtype(getattr(self, name))
            object.__setattr__(self, name, TileLangDType(dtype_info.name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_extent": self.output_extent,
            "reduction_extent": self.reduction_extent,
            "input_dtype": require_dataflow_dtype(self.input_dtype).name,
            "weight_dtype": require_dataflow_dtype(self.weight_dtype).name,
            "output_dtype": require_dataflow_dtype(self.output_dtype).name,
            "projection_count": self.projection_count,
            "output_partitioned": self.output_partitioned,
            "input_from_previous_stage": self.input_from_previous_stage,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowExecutionStageRequest:
        if not isinstance(value, Mapping):
            raise TypeError("execution stage request must be a mapping")
        reject_unknown_fields(value, set(cls.__dataclass_fields__), "execution stage request")
        return cls(**value)


def is_native_fp8(dtype: DType) -> bool:
    return require_dataflow_dtype(dtype).is_float8


@dataclass(frozen=True)
class DataflowExecutionRequest:
    """Target-independent logical input to execution candidate planning."""

    task_extent: int
    stages: tuple[DataflowExecutionStageRequest, ...]
    linked_stage_pairs: tuple[tuple[int, int], ...] = ()
    transport_family: str = DATAFLOW_TRANSPORT_AUTO
    handoff_permitted: bool = True
    uniform_stage_implementation: bool = False
    minimum_tile_extent: int = 16
    # Independent ragged groups cannot share a matrix tile. Their extents are
    # scheduler facts, not model names or calibrated timing profiles.
    task_group_extents: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        positive_int(self.task_extent, "execution task_extent")
        positive_int(self.minimum_tile_extent, "execution minimum_tile_extent")
        groups = tuple(self.task_group_extents)
        for extent in groups:
            nonnegative_int(extent, "execution task_group_extent")
        if groups and sum(groups) != self.task_extent:
            raise ValueError("execution task_group_extents must sum to task_extent")
        object.__setattr__(self, "task_group_extents", groups)
        stages = tuple(self.stages)
        if not stages or any(not isinstance(stage, DataflowExecutionStageRequest) for stage in stages):
            raise TypeError("execution request stages must contain typed stage requests")
        object.__setattr__(self, "stages", stages)
        pairs = tuple(tuple(pair) for pair in self.linked_stage_pairs)
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("linked execution stage pairs must have two indices")
            producer, consumer = pair
            if (
                isinstance(producer, bool)
                or isinstance(consumer, bool)
                or not isinstance(producer, int)
                or not isinstance(consumer, int)
                or producer < 0
                or consumer < 0
                or producer >= len(stages)
                or consumer >= len(stages)
                or producer == consumer
            ):
                raise ValueError(f"invalid linked execution stage pair {pair!r}")
        if len(pairs) != len(set(pairs)):
            raise ValueError("linked execution stage pairs must be unique")
        object.__setattr__(self, "linked_stage_pairs", pairs)
        object.__setattr__(
            self,
            "transport_family",
            choice(
                self.transport_family,
                "execution transport family",
                DATAFLOW_TRANSPORT_FAMILIES,
            ),
        )
        for name in ("handoff_permitted", "uniform_stage_implementation"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"execution {name} must be bool")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "task_extent": self.task_extent,
            "stages": [stage.to_dict() for stage in self.stages],
            "linked_stage_pairs": [list(pair) for pair in self.linked_stage_pairs],
            "transport_family": self.transport_family,
            "handoff_permitted": self.handoff_permitted,
            "uniform_stage_implementation": self.uniform_stage_implementation,
            "minimum_tile_extent": self.minimum_tile_extent,
            # Preserve fingerprints of requests serialized before ragged task
            # facts were available.
            **({"task_group_extents": list(self.task_group_extents)} if self.task_group_extents else {}),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowExecutionRequest:
        if not isinstance(value, Mapping):
            raise TypeError("execution request must be a mapping")
        allowed = set(cls.__dataclass_fields__) | {"fingerprint"}
        reject_unknown_fields(value, allowed, "execution request")
        request = cls(
            task_extent=int(value["task_extent"]),
            stages=tuple(DataflowExecutionStageRequest.from_dict(stage) for stage in value["stages"]),
            linked_stage_pairs=tuple(tuple(int(index) for index in pair) for pair in value.get("linked_stage_pairs", ())),
            transport_family=str(value.get("transport_family", DATAFLOW_TRANSPORT_AUTO)),
            handoff_permitted=value.get("handoff_permitted", True),
            uniform_stage_implementation=value.get("uniform_stage_implementation", False),
            minimum_tile_extent=int(value.get("minimum_tile_extent", 16)),
            task_group_extents=tuple(value.get("task_group_extents", ())),
        )
        if value.get("fingerprint") not in {None, request.fingerprint}:
            raise ValueError("execution request fingerprint does not match its payload")
        return request


@dataclass(frozen=True)
class DataflowExecutionStageOverride:
    """Optional typed constraints for one logical compute stage."""

    tile_n: int | None = None
    tile_k: int | None = None
    handler_extent: int | None = None
    compute_threads: int | None = None
    consumer_threads: int | None = None
    loop_stages: int | None = None
    pipeline_stages: int | None = None
    max_outstanding: int | None = None
    gemm_family: str = DATAFLOW_EXECUTION_GEMM_AUTO
    transfer_family: str = DATAFLOW_EXECUTION_TRANSFER_AUTO
    input_distribution: str = DATAFLOW_EXECUTION_INPUT_AUTO
    split_producers: bool | None = None
    eviction_policy: str = DATAFLOW_PIPELINE_EVICT_FIRST
    wait_depth: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "tile_n",
            "tile_k",
            "handler_extent",
            "compute_threads",
            "consumer_threads",
            "loop_stages",
            "pipeline_stages",
            "max_outstanding",
        ):
            positive_int(getattr(self, name), name, allow_none=True)
        nonnegative_int(self.wait_depth, "wait_depth", allow_none=True)
        object.__setattr__(
            self,
            "gemm_family",
            choice(
                self.gemm_family,
                "execution GEMM family",
                DATAFLOW_EXECUTION_GEMM_FAMILIES,
            ),
        )
        object.__setattr__(
            self,
            "transfer_family",
            choice(
                self.transfer_family,
                "execution transfer family",
                DATAFLOW_EXECUTION_TRANSFER_FAMILIES,
            ),
        )
        object.__setattr__(
            self,
            "input_distribution",
            choice(
                self.input_distribution,
                "execution input distribution",
                DATAFLOW_EXECUTION_INPUT_DISTRIBUTIONS,
            ),
        )
        object.__setattr__(
            self,
            "eviction_policy",
            choice(
                self.eviction_policy,
                "execution eviction policy",
                DATAFLOW_PIPELINE_EVICTION_HINTS,
            ),
        )
        if self.split_producers is not None and not isinstance(self.split_producers, bool):
            raise TypeError("split_producers must be bool or None")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowExecutionStageOverride:
        if not isinstance(value, Mapping):
            raise TypeError("execution stage override must be a mapping")
        reject_unknown_fields(value, set(cls.__dataclass_fields__), "execution stage override")
        return cls(**value)


@dataclass(frozen=True)
class DataflowExecutionOverride:
    """One advanced typed override for execution candidate selection."""

    topology: GPUTopology | None = None
    task_tile_extent: int | None = None
    stages: tuple[DataflowExecutionStageOverride, ...] = ()
    transport_family: str = DATAFLOW_TRANSPORT_AUTO
    handoff_stages: int | None = None
    schema_version: int = DATAFLOW_EXECUTION_OVERRIDE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_EXECUTION_OVERRIDE_SCHEMA_VERSION:
            raise ValueError(f"unsupported Dataflow execution override schema version {self.schema_version}")
        if self.topology is not None and not isinstance(self.topology, GPUTopology):
            raise TypeError("execution override topology must be GPUTopology or None")
        positive_int(
            self.task_tile_extent,
            "execution task_tile_extent",
            allow_none=True,
        )
        stages = tuple(self.stages)
        if any(not isinstance(stage, DataflowExecutionStageOverride) for stage in stages):
            raise TypeError("execution override stages must be typed stage overrides")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(
            self,
            "transport_family",
            choice(
                self.transport_family,
                "execution override transport family",
                DATAFLOW_TRANSPORT_FAMILIES,
            ),
        )
        nonnegative_int(
            self.handoff_stages,
            "execution handoff_stages",
            allow_none=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "topology": (
                None
                if self.topology is None
                else {
                    "sm_count": self.topology.sm_count,
                    "cluster_size": self.topology.cluster_size,
                }
            ),
            "task_tile_extent": self.task_tile_extent,
            "stages": [stage.to_dict() for stage in self.stages],
            "transport_family": self.transport_family,
            "handoff_stages": self.handoff_stages,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowExecutionOverride:
        if not isinstance(value, Mapping):
            raise TypeError("execution override must be a mapping")
        reject_unknown_fields(value, set(cls.__dataclass_fields__), "execution override")
        topology = value.get("topology")
        return cls(
            topology=(
                None
                if topology is None
                else GPUTopology(
                    sm_count=int(topology["sm_count"]),
                    cluster_size=int(topology.get("cluster_size", 1)),
                )
            ),
            task_tile_extent=value.get("task_tile_extent"),
            stages=tuple(DataflowExecutionStageOverride.from_dict(stage) for stage in value.get("stages", ())),
            transport_family=value.get("transport_family", DATAFLOW_TRANSPORT_AUTO),
            handoff_stages=value.get("handoff_stages"),
            schema_version=int(
                value.get(
                    "schema_version",
                    DATAFLOW_EXECUTION_OVERRIDE_SCHEMA_VERSION,
                )
            ),
        )


def resolve_execution_override(
    value: DataflowExecutionOverride | Mapping[str, Any] | None,
) -> DataflowExecutionOverride:
    if value is None:
        return DataflowExecutionOverride()
    if isinstance(value, DataflowExecutionOverride):
        return value
    if isinstance(value, Mapping):
        return DataflowExecutionOverride.from_dict(value)
    raise TypeError(f"execution_override must be DataflowExecutionOverride, a serialized mapping, or None, got {type(value)!r}")


@dataclass(frozen=True)
class DataflowExecutionStageConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    handler_extent: int
    compute_threads: int
    consumer_threads: int
    loop_stages: int
    pipeline_stages: int
    max_outstanding: int
    gemm_family: str
    transfer_family: str
    input_distribution: str
    split_producers: bool
    eviction_policy: str
    wait_depth: int
    common_pipeline: bool
    fused_transfers: bool
    full_partition_pipeline: bool

    def __post_init__(self) -> None:
        for name in (
            "tile_m",
            "tile_n",
            "tile_k",
            "handler_extent",
            "compute_threads",
            "consumer_threads",
            "loop_stages",
            "pipeline_stages",
            "max_outstanding",
        ):
            positive_int(getattr(self, name), f"execution stage {name}")
        nonnegative_int(self.wait_depth, "execution stage wait_depth")
        object.__setattr__(
            self,
            "gemm_family",
            choice(
                self.gemm_family,
                "selected execution GEMM family",
                DATAFLOW_EXECUTION_GEMM_FAMILIES[1:],
            ),
        )
        object.__setattr__(
            self,
            "transfer_family",
            choice(
                self.transfer_family,
                "selected execution transfer family",
                DATAFLOW_EXECUTION_TRANSFER_FAMILIES[1:],
            ),
        )
        object.__setattr__(
            self,
            "input_distribution",
            choice(
                self.input_distribution,
                "selected execution input distribution",
                DATAFLOW_EXECUTION_INPUT_DISTRIBUTIONS[1:],
            ),
        )
        object.__setattr__(
            self,
            "eviction_policy",
            choice(
                self.eviction_policy,
                "selected execution eviction policy",
                DATAFLOW_PIPELINE_EVICTION_HINTS,
            ),
        )
        for name in (
            "split_producers",
            "common_pipeline",
            "fused_transfers",
            "full_partition_pipeline",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"execution stage {name} must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowExecutionStageConfig:
        if not isinstance(value, Mapping):
            raise TypeError("execution stage config must be a mapping")
        reject_unknown_fields(value, set(cls.__dataclass_fields__), "execution stage config")
        return cls(**value)


@dataclass(frozen=True)
class DataflowExecutionCandidate:
    topology: GPUTopology
    stages: tuple[DataflowExecutionStageConfig, ...]
    transport_family: str
    handoff_stages: int
    candidate_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.topology, GPUTopology):
            raise TypeError("execution candidate topology must be GPUTopology")
        stages = tuple(self.stages)
        if not stages or any(not isinstance(stage, DataflowExecutionStageConfig) for stage in stages):
            raise TypeError("execution candidate stages must be typed configs")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(
            self,
            "transport_family",
            choice(
                self.transport_family,
                "selected execution transport family",
                DATAFLOW_TRANSPORT_FAMILIES[1:],
            ),
        )
        nonnegative_int(self.handoff_stages, "execution candidate handoff_stages")
        expected = "execution." + fingerprint(self.payload())[:16]
        if self.candidate_id and self.candidate_id != expected:
            raise ValueError("execution candidate id does not match its payload")
        object.__setattr__(self, "candidate_id", expected)

    @property
    def task_tile_extent(self) -> int:
        return self.stages[0].tile_m

    def payload(self) -> dict[str, Any]:
        return {
            "topology": {
                "sm_count": self.topology.sm_count,
                "cluster_size": self.topology.cluster_size,
            },
            "stages": [stage.to_dict() for stage in self.stages],
            "transport_family": self.transport_family,
            "handoff_stages": self.handoff_stages,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **self.payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowExecutionCandidate:
        if not isinstance(value, Mapping):
            raise TypeError("execution candidate must be a mapping")
        allowed = {
            "candidate_id",
            "topology",
            "stages",
            "transport_family",
            "handoff_stages",
        }
        reject_unknown_fields(value, allowed, "execution candidate")
        topology = value["topology"]
        return cls(
            topology=GPUTopology(
                sm_count=int(topology["sm_count"]),
                cluster_size=int(topology["cluster_size"]),
            ),
            stages=tuple(DataflowExecutionStageConfig.from_dict(stage) for stage in value["stages"]),
            transport_family=str(value["transport_family"]),
            handoff_stages=int(value["handoff_stages"]),
            candidate_id=str(value.get("candidate_id", "")),
        )


@dataclass(frozen=True)
class DataflowExecutionResourceEstimate:
    shared_memory_bytes: int
    register_count: int
    barrier_count: int
    max_threads: int
    pipeline_shared_memory_budgets: tuple[int | None, ...]

    def __post_init__(self) -> None:
        for name in (
            "shared_memory_bytes",
            "register_count",
            "barrier_count",
            "max_threads",
        ):
            nonnegative_int(getattr(self, name), f"execution resource {name}")
        budgets = tuple(self.pipeline_shared_memory_budgets)
        for budget in budgets:
            positive_int(
                budget,
                "execution pipeline shared-memory budget",
                allow_none=True,
            )
        object.__setattr__(self, "pipeline_shared_memory_budgets", budgets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_memory_bytes": self.shared_memory_bytes,
            "register_count": self.register_count,
            "barrier_count": self.barrier_count,
            "max_threads": self.max_threads,
            "pipeline_shared_memory_budgets": list(self.pipeline_shared_memory_budgets),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowExecutionResourceEstimate:
        if not isinstance(value, Mapping):
            raise TypeError("execution resource estimate must be a mapping")
        reject_unknown_fields(value, set(cls.__dataclass_fields__), "execution resource estimate")
        return cls(
            shared_memory_bytes=int(value["shared_memory_bytes"]),
            register_count=int(value["register_count"]),
            barrier_count=int(value["barrier_count"]),
            max_threads=int(value["max_threads"]),
            pipeline_shared_memory_budgets=tuple(None if item is None else int(item) for item in value["pipeline_shared_memory_budgets"]),
        )


@dataclass(frozen=True)
class DataflowExecutionCandidateEvaluation:
    candidate: DataflowExecutionCandidate
    legal: bool
    resources: DataflowExecutionResourceEstimate
    rejection_reasons: tuple[str, ...] = ()
    score: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DataflowExecutionCandidate):
            raise TypeError("execution evaluation requires a typed candidate")
        if not isinstance(self.resources, DataflowExecutionResourceEstimate):
            raise TypeError("execution evaluation requires typed resources")
        reasons = tuple(str(reason) for reason in self.rejection_reasons if str(reason))
        object.__setattr__(self, "rejection_reasons", reasons)
        score = None if self.score is None else tuple(float(item) for item in self.score)
        object.__setattr__(self, "score", score)
        if self.legal:
            if reasons or score is None:
                raise ValueError("legal execution candidates require a score and no rejection")
        elif not reasons or score is not None:
            raise ValueError("illegal execution candidates require rejection reasons only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "legal": self.legal,
            "resources": self.resources.to_dict(),
            "rejection_reasons": list(self.rejection_reasons),
            "score": None if self.score is None else list(self.score),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowExecutionCandidateEvaluation:
        if not isinstance(value, Mapping):
            raise TypeError("execution candidate evaluation must be a mapping")
        reject_unknown_fields(value, set(cls.__dataclass_fields__), "execution candidate evaluation")
        score = value.get("score")
        return cls(
            candidate=DataflowExecutionCandidate.from_dict(value["candidate"]),
            legal=value["legal"],
            resources=DataflowExecutionResourceEstimate.from_dict(value["resources"]),
            rejection_reasons=tuple(value.get("rejection_reasons", ())),
            score=None if score is None else tuple(float(item) for item in score),
        )


@dataclass(frozen=True)
class DataflowExecutionPlan:
    request: DataflowExecutionRequest
    target_fingerprint: str
    target_compatibility_fingerprint: str
    planner_version: str
    candidates: tuple[DataflowExecutionCandidateEvaluation, ...]
    selected_candidate_id: str
    selection_reason: str
    used_fallback: bool
    explicit_override: bool
    schema_version: int = DATAFLOW_EXECUTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_EXECUTION_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported Dataflow execution plan schema {self.schema_version}")
        if self.planner_version != DATAFLOW_EXECUTION_PLANNER_VERSION:
            raise ValueError(f"unsupported execution planner {self.planner_version!r}")
        if not isinstance(self.request, DataflowExecutionRequest):
            raise TypeError("execution plan request must be typed")
        candidates = tuple(self.candidates)
        if not candidates or any(not isinstance(item, DataflowExecutionCandidateEvaluation) for item in candidates):
            raise TypeError("execution plan candidates must be typed evaluations")
        object.__setattr__(self, "candidates", candidates)
        ids = tuple(item.candidate.candidate_id for item in candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("execution plan candidate ids must be unique")
        selected = tuple(item for item in candidates if item.candidate.candidate_id == self.selected_candidate_id)
        if len(selected) != 1 or not selected[0].legal:
            raise ValueError("execution plan must select exactly one legal candidate")
        if not self.target_fingerprint or not self.target_compatibility_fingerprint:
            raise ValueError("execution plan target fingerprints must be non-empty")
        if not self.selection_reason:
            raise ValueError("execution plan selection_reason must be non-empty")
        for name in ("used_fallback", "explicit_override"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"execution plan {name} must be bool")

    @property
    def selected_evaluation(self) -> DataflowExecutionCandidateEvaluation:
        return next(item for item in self.candidates if item.candidate.candidate_id == self.selected_candidate_id)

    @property
    def selected_candidate(self) -> DataflowExecutionCandidate:
        return self.selected_evaluation.candidate

    @property
    def resources(self) -> DataflowExecutionResourceEstimate:
        return self.selected_evaluation.resources

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "target_fingerprint": self.target_fingerprint,
            "target_compatibility_fingerprint": (self.target_compatibility_fingerprint),
            "planner_version": self.planner_version,
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "selection_reason": self.selection_reason,
            "used_fallback": self.used_fallback,
            "explicit_override": self.explicit_override,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowExecutionPlan:
        if not isinstance(value, Mapping):
            raise TypeError("execution plan must be a mapping")
        allowed = set(cls.__dataclass_fields__) | {"fingerprint"}
        reject_unknown_fields(value, allowed, "execution plan")
        plan = cls(
            request=DataflowExecutionRequest.from_dict(value["request"]),
            target_fingerprint=str(value["target_fingerprint"]),
            target_compatibility_fingerprint=str(value["target_compatibility_fingerprint"]),
            planner_version=str(value["planner_version"]),
            candidates=tuple(DataflowExecutionCandidateEvaluation.from_dict(item) for item in value["candidates"]),
            selected_candidate_id=str(value["selected_candidate_id"]),
            selection_reason=str(value["selection_reason"]),
            used_fallback=value["used_fallback"],
            explicit_override=value["explicit_override"],
            schema_version=int(value["schema_version"]),
        )
        if value.get("fingerprint") not in {None, plan.fingerprint}:
            raise ValueError("execution plan fingerprint does not match its payload")
        return plan


class DataflowExecutionPlanningError(ValueError):
    """Raised when every typed execution candidate is rejected."""

    def __init__(
        self,
        message: str,
        *,
        request: DataflowExecutionRequest,
        candidates: Sequence[DataflowExecutionCandidateEvaluation],
    ) -> None:
        super().__init__(message)
        self.request = request
        self.candidates = tuple(candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_version": DATAFLOW_EXECUTION_PLANNER_VERSION,
            "request": self.request.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "message": str(self),
        }


@contextmanager
def capture_execution_plans() -> Iterator[list[DataflowExecutionPlan]]:
    plans: list[DataflowExecutionPlan] = []
    token = _CAPTURED_EXECUTION_PLANS.set(plans)
    try:
        yield plans
    finally:
        _CAPTURED_EXECUTION_PLANS.reset(token)


def record_execution_plan(plan: DataflowExecutionPlan) -> None:
    plans = _CAPTURED_EXECUTION_PLANS.get()
    if plans is not None:
        plans.append(plan)


def plan_execution(
    request: DataflowExecutionRequest,
    *,
    override: DataflowExecutionOverride | Mapping[str, Any] | None = None,
    target_capabilities: TargetCapabilitySnapshot | None = None,
) -> DataflowExecutionPlan:
    """Generate, validate, score, and record one execution plan."""

    if not isinstance(request, DataflowExecutionRequest):
        raise TypeError("execution planning requires a typed DataflowExecutionRequest")
    override = resolve_execution_override(override)
    target = target_capabilities or current_target_capabilities()
    if target is None:
        raise RuntimeError("Dataflow execution planning requires a compiler-owned target snapshot")
    if not isinstance(target, TargetCapabilitySnapshot):
        raise TypeError("execution planning target must be TargetCapabilitySnapshot")
    if override.stages and len(override.stages) != len(request.stages):
        raise ValueError(
            "execution override must provide one stage override per request stage: "
            f"got {len(override.stages)}, expected {len(request.stages)}"
        )
    stage_overrides = override.stages or tuple(DataflowExecutionStageOverride() for _ in request.stages)

    evaluations: list[DataflowExecutionCandidateEvaluation] = []
    for candidate in generate_candidates(
        request,
        override,
        stage_overrides,
        target,
    ):
        evaluation = evaluate_candidate(request, candidate, target)
        evaluations.append(evaluation)
    evaluations = deduplicate_evaluations(evaluations)
    legal = [item for item in evaluations if item.legal]
    if not legal:
        details = "; ".join(f"{item.candidate.candidate_id}: {', '.join(item.rejection_reasons)}" for item in evaluations)
        raise DataflowExecutionPlanningError(
            "no legal Dataflow execution candidate; " + details,
            request=request,
            candidates=evaluations,
        )
    selected = min(
        legal,
        key=lambda item: (item.score, item.candidate.candidate_id),
    )
    explicit = override_is_explicit(override)
    accelerated_available = target.supports_tma and target.supports_wgmma
    selected_portable = all(stage.gemm_family == DATAFLOW_EXECUTION_GEMM_PORTABLE for stage in selected.candidate.stages)
    used_fallback = bool(not explicit and not accelerated_available and selected_portable)
    selection_reason = (
        "explicit_typed_override"
        if explicit
        else "capability_portable_fallback"
        if used_fallback
        else "generic_cost_and_resource_selection"
    )
    plan = DataflowExecutionPlan(
        request=request,
        target_fingerprint=target.fingerprint,
        target_compatibility_fingerprint=target.compatibility_fingerprint,
        planner_version=DATAFLOW_EXECUTION_PLANNER_VERSION,
        candidates=tuple(evaluations),
        selected_candidate_id=selected.candidate.candidate_id,
        selection_reason=selection_reason,
        used_fallback=used_fallback,
        explicit_override=explicit,
    )
    record_execution_plan(plan)
    return plan


def override_is_explicit(override: DataflowExecutionOverride) -> bool:
    if (
        override.topology is not None
        or override.task_tile_extent is not None
        or override.handoff_stages is not None
        or override.transport_family != DATAFLOW_TRANSPORT_AUTO
    ):
        return True
    default_stage = DataflowExecutionStageOverride()
    return any(stage != default_stage for stage in override.stages)


def generate_candidates(
    request: DataflowExecutionRequest,
    override: DataflowExecutionOverride,
    stage_overrides: tuple[DataflowExecutionStageOverride, ...],
    target: TargetCapabilitySnapshot,
) -> tuple[DataflowExecutionCandidate, ...]:
    topologies = candidate_topologies(request, override, target)
    tile_widths: tuple[int | None, ...]
    if any(stage.tile_n is not None or stage.tile_k is not None for stage in stage_overrides):
        tile_widths = (None,)
    else:
        tile_widths = _DATAFLOW_EXECUTION_MATRIX_TILE_CLASSES
    task_tiles = (override.task_tile_extent,) if override.task_tile_extent is not None else _DATAFLOW_EXECUTION_TASK_TILE_CLASSES
    explicit_gemm = any(stage.gemm_family != DATAFLOW_EXECUTION_GEMM_AUTO for stage in stage_overrides)
    gemm_modes = (
        (None,)
        if explicit_gemm
        else (
            tuple(
                (
                    DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
                    DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M,
                    DATAFLOW_EXECUTION_GEMM_PORTABLE,
                )
                if all(is_native_fp8(stage.input_dtype) and is_native_fp8(stage.weight_dtype) for stage in request.stages)
                else (
                    DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
                    DATAFLOW_EXECUTION_GEMM_PORTABLE,
                )
            )
            if target.supports_tma and target.supports_wgmma
            else (DATAFLOW_EXECUTION_GEMM_PORTABLE,)
        )
    )
    candidates = []
    for topology in topologies:
        for task_tile in task_tiles:
            for tile_width in tile_widths:
                for default_gemm in gemm_modes:
                    stages = candidate_stages(
                        request,
                        stage_overrides,
                        topology,
                        target,
                        task_tile=task_tile,
                        tile_width=tile_width,
                        default_gemm=default_gemm,
                    )
                    transport = candidate_transport_family(
                        request,
                        override,
                        stages,
                        topology,
                        target,
                    )
                    # Pipeline ownership is a per-stage physical property.  A
                    # thread-limited multicast producer can require a
                    # synchronous first-stage fallback without disabling an
                    # otherwise legal TMA consumer pipeline in a linked stage.
                    # In particular, streamed producer-push transport relies
                    # on the consumer stage retaining its own receive/credit
                    # lifetime even when the producer stage falls back.
                    stages = tuple(
                        replace(
                            stage,
                            common_pipeline=bool(
                                stage.transfer_family == DATAFLOW_EXECUTION_TRANSFER_TMA
                                and not (
                                    stage_index == 0
                                    and stage.input_distribution == DATAFLOW_EXECUTION_INPUT_MULTICAST
                                    and stage.consumer_threads < stage.compute_threads
                                )
                            ),
                        )
                        for stage_index, stage in enumerate(stages)
                    )
                    last_request = request.stages[-1]
                    last_stage = stages[-1]
                    output_shard = last_request.output_extent // (topology.cluster_size if last_request.output_partitioned else 1)
                    full_partition_pipeline = bool(
                        last_stage.handler_extent == output_shard
                        and math.ceil(last_stage.handler_extent / last_stage.tile_n) == 2
                        and transport in {DATAFLOW_TRANSPORT_ALL_GATHER, DATAFLOW_TRANSPORT_STREAMED}
                        and last_stage.transfer_family == DATAFLOW_EXECUTION_TRANSFER_TMA
                        and last_stage.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP
                        and last_stage.pipeline_stages >= 2
                        and last_stage.max_outstanding >= 2
                        and last_stage.pipeline_stages % 2 == 0
                        and last_stage.max_outstanding % 2 == 0
                    )
                    stages = (
                        *stages[:-1],
                        replace(
                            last_stage,
                            full_partition_pipeline=full_partition_pipeline,
                        ),
                    )
                    candidates.append(
                        DataflowExecutionCandidate(
                            topology=topology,
                            stages=stages,
                            transport_family=transport,
                            handoff_stages=(override.handoff_stages or 0),
                        )
                    )
                    phase_specific = small_m_phase_specific_candidate(
                        request,
                        candidates[-1],
                        stage_overrides,
                        target,
                        transport_constraint=override.transport_family
                        if override.transport_family != DATAFLOW_TRANSPORT_AUTO
                        else request.transport_family,
                    )
                    if phase_specific is not None:
                        candidates.append(phase_specific)
    return tuple(candidates)


def small_m_reduction_chain(request: DataflowExecutionRequest) -> bool:
    """Long-reduction, low-row-count chains benefit from fewer TMA turns."""
    return bool(
        len(request.stages) == 2
        and request.linked_stage_pairs == ((0, 1),)
        and max(request.task_group_extents or (request.task_extent,)) <= 32
        and request.stages[0].reduction_extent >= 16 * 256
        and request.stages[0].projection_count == 2
        and request.stages[1].projection_count == 1
        and request.stages[1].input_from_previous_stage
        and all(is_native_fp8(stage.input_dtype) and is_native_fp8(stage.weight_dtype) for stage in request.stages)
    )


def small_m_phase_specific_candidate(
    request: DataflowExecutionRequest,
    base: DataflowExecutionCandidate,
    overrides: tuple[DataflowExecutionStageOverride, ...],
    target: TargetCapabilitySnapshot,
    *,
    transport_constraint: str,
) -> DataflowExecutionCandidate | None:
    """Joint K/N, handler and ring candidate, still checked by common planning.

    Widen the first reduction without widening its accumulator. The linked
    consumer uses the producer's N as K, so it needs a deeper, narrower weight
    ring. Full producer partitions amortize dispatch; consumer handlers retain
    the bounded seven-tile class. Explicit fields always win over these defaults.
    """
    if not small_m_reduction_chain(request) or transport_constraint not in {DATAFLOW_TRANSPORT_AUTO, DATAFLOW_TRANSPORT_ALL_GATHER}:
        return None
    if not all(stage.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M for stage in base.stages):
        return None
    if base.stages[0].input_distribution != DATAFLOW_EXECUTION_INPUT_UNICAST:
        return None
    shards = tuple(stage.output_extent // (base.topology.cluster_size if stage.output_partitioned else 1) for stage in request.stages)
    if shards[0] < 256 or shards[0] % 128 or shards[1] % 256 or request.stages[0].reduction_extent % 256:
        return None
    defaults = (
        dict(tile_n=128, tile_k=256, handler_extent=shards[0], pipeline_stages=3, max_outstanding=2),
        dict(tile_n=256, tile_k=128, handler_extent=min(shards[1], 7 * 256), pipeline_stages=4, max_outstanding=4),
    )
    merged = tuple(
        replace(override, **{key: value for key, value in values.items() if getattr(override, key) is None})
        for override, values in zip(overrides, defaults)
    )
    stages = candidate_stages(
        request,
        merged,
        base.topology,
        target,
        task_tile=base.task_tile_extent,
        tile_width=None,
        default_gemm=DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M,
    )
    return replace(
        base,
        transport_family=DATAFLOW_TRANSPORT_ALL_GATHER,
        candidate_id="",
        stages=tuple(replace(stage, common_pipeline=original.common_pipeline) for stage, original in zip(stages, base.stages)),
    )


def candidate_topologies(
    request: DataflowExecutionRequest,
    override: DataflowExecutionOverride,
    target: TargetCapabilitySnapshot,
) -> tuple[GPUTopology, ...]:
    if override.topology is not None:
        return (override.topology,)
    sm_count = target.multiprocessor_count or 1
    maximum_cluster = (target.max_cluster_size or 16) if target.supports_cluster_launch else 1
    cluster_sizes = tuple(
        size
        for size in range(min(maximum_cluster, sm_count), 0, -1)
        if sm_count % size == 0 and all(not stage.output_partitioned or stage.output_extent % size == 0 for stage in request.stages)
    )
    return tuple(GPUTopology(sm_count, size) for size in cluster_sizes) or (GPUTopology(sm_count, 1),)


def candidate_stages(
    request: DataflowExecutionRequest,
    overrides: tuple[DataflowExecutionStageOverride, ...],
    topology: GPUTopology,
    target: TargetCapabilitySnapshot,
    *,
    task_tile: int,
    tile_width: int | None,
    default_gemm: str | None,
) -> tuple[DataflowExecutionStageConfig, ...]:
    stages: list[DataflowExecutionStageConfig] = []
    linked_consumers = {consumer: producer for producer, consumer in request.linked_stage_pairs}
    for index, (stage_request, stage_override) in enumerate(zip(request.stages, overrides)):
        partition = topology.cluster_size if stage_request.output_partitioned else 1
        output_shard = max(1, stage_request.output_extent // partition)
        preferred = tile_width or 128
        tile_n = stage_override.tile_n or math.gcd(output_shard, preferred)
        if index in linked_consumers and stage_override.tile_k is None:
            tile_k = stages[linked_consumers[index]].tile_n
        else:
            tile_k = stage_override.tile_k or math.gcd(
                stage_request.reduction_extent,
                preferred,
            )
        handler_extent = stage_override.handler_extent or tile_n
        gemm = stage_override.gemm_family
        if gemm == DATAFLOW_EXECUTION_GEMM_AUTO:
            gemm = default_gemm or (
                DATAFLOW_EXECUTION_GEMM_WARP_GROUP if target.supports_tma and target.supports_wgmma else DATAFLOW_EXECUTION_GEMM_PORTABLE
            )
        transfer = stage_override.transfer_family
        if transfer == DATAFLOW_EXECUTION_TRANSFER_AUTO:
            transfer = (
                DATAFLOW_EXECUTION_TRANSFER_TMA if gemm != DATAFLOW_EXECUTION_GEMM_PORTABLE else DATAFLOW_EXECUTION_TRANSFER_SYNCHRONOUS
            )
        compute_threads = stage_override.compute_threads or (128 if gemm == DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M else 256)
        distribution = stage_override.input_distribution
        if distribution == DATAFLOW_EXECUTION_INPUT_AUTO:
            distribution = (
                DATAFLOW_EXECUTION_INPUT_UNICAST
                if index == 0 and transfer == DATAFLOW_EXECUTION_TRANSFER_TMA and is_native_fp8(stage_request.input_dtype)
                else DATAFLOW_EXECUTION_INPUT_COOPERATIVE
            )
        consumer_threads = stage_override.consumer_threads or (
            128 if index == 0 and distribution != DATAFLOW_EXECUTION_INPUT_COOPERATIVE else compute_threads
        )
        pipeline_stages = stage_override.pipeline_stages or 2
        max_outstanding = stage_override.max_outstanding or (pipeline_stages if index == 0 else min(2, pipeline_stages))
        split_producers = (
            stage_override.split_producers
            if stage_override.split_producers is not None
            else distribution != DATAFLOW_EXECUTION_INPUT_COOPERATIVE
        )
        fused_transfers = bool(
            transfer == DATAFLOW_EXECUTION_TRANSFER_TMA
            and stage_request.projection_count > 1
            and handler_extent == tile_n
            and target.max_tma_noninnermost_box_extent is not None
            and stage_request.projection_count * tile_n <= target.max_tma_noninnermost_box_extent
        )
        stages.append(
            DataflowExecutionStageConfig(
                tile_m=task_tile,
                tile_n=tile_n,
                tile_k=tile_k,
                handler_extent=handler_extent,
                compute_threads=compute_threads,
                consumer_threads=consumer_threads,
                loop_stages=stage_override.loop_stages or 1,
                pipeline_stages=pipeline_stages,
                max_outstanding=max_outstanding,
                gemm_family=gemm,
                transfer_family=transfer,
                input_distribution=distribution,
                split_producers=bool(split_producers),
                eviction_policy=stage_override.eviction_policy,
                wait_depth=stage_override.wait_depth or 0,
                common_pipeline=False,
                fused_transfers=fused_transfers,
                full_partition_pipeline=False,
            )
        )
    return tuple(stages)


def candidate_transport_family(
    request: DataflowExecutionRequest,
    override: DataflowExecutionOverride,
    stages: tuple[DataflowExecutionStageConfig, ...],
    topology: GPUTopology,
    target: TargetCapabilitySnapshot,
) -> str:
    family = override.transport_family if override.transport_family != DATAFLOW_TRANSPORT_AUTO else request.transport_family
    if family == DATAFLOW_TRANSPORT_AUTO and topology.cluster_size == 1:
        # A one-rank reshared value is already local. Keeping it in the shared
        # family avoids manufacturing HBM communication slots that have no
        # producer/consumer edge to own them.
        family = DATAFLOW_TRANSPORT_ALL_GATHER
    first_request = request.stages[0]
    first_stage = stages[0]
    logical_arity = max(
        1,
        math.ceil(first_request.output_extent / first_stage.tile_n),
    )
    physical_arity = max(
        1,
        math.ceil(first_request.output_extent / first_stage.handler_extent),
    )
    output_bytes = require_dataflow_dtype(first_request.output_dtype).element_bytes
    physical_slot_bytes = first_stage.tile_m * first_stage.handler_extent * output_bytes
    transport_request = DataflowResharedTransportRequest(
        family=family,
        logical_output_arity=logical_arity,
        physical_output_arity=physical_arity,
        max_temporary_bytes=target.max_dynamic_shared_memory,
    )
    try:
        plan = plan_reshared_transport(
            transport_request,
            cluster_size=topology.cluster_size,
            physical_slot_bytes=physical_slot_bytes,
            target_capabilities=target,
            available_threads=stages[-1].compute_threads,
        )
    except (DataflowResharedTransportPlanningError, ValueError):
        return DATAFLOW_TRANSPORT_HBM if family == DATAFLOW_TRANSPORT_AUTO else family
    return plan.family


def evaluate_candidate(
    request: DataflowExecutionRequest,
    candidate: DataflowExecutionCandidate,
    target: TargetCapabilitySnapshot,
) -> DataflowExecutionCandidateEvaluation:
    reasons: list[str] = []
    topology = candidate.topology
    if topology.sm_count % topology.cluster_size:
        reasons.append("topology_sm_count_not_divisible_by_cluster_size")
    if target.multiprocessor_count is not None and topology.sm_count > target.multiprocessor_count:
        reasons.append("target_multiprocessor_count_limit_exceeded")
    if topology.cluster_size > 1 and not target.supports_cluster_launch:
        reasons.append("target_cluster_launch_unsupported")
    if target.max_cluster_size is not None and topology.cluster_size > target.max_cluster_size:
        reasons.append("target_cluster_size_limit_exceeded")
    for stage_request in request.stages:
        if stage_request.output_partitioned and stage_request.output_extent % topology.cluster_size:
            reasons.append("partitioned_output_not_divisible_by_cluster_size")

    for index, (stage_request, stage) in enumerate(zip(request.stages, candidate.stages)):
        output_shard = stage_request.output_extent // (topology.cluster_size if stage_request.output_partitioned else 1)
        if stage.tile_n > output_shard or stage.tile_k > stage_request.reduction_extent:
            reasons.append(f"stage_{index}_tile_exceeds_logical_extent")
        if stage_request.reduction_extent % stage.tile_k:
            reasons.append(f"stage_{index}_reduction_extent_not_tile_divisible")
        if min(stage.tile_m, stage.tile_n, stage.tile_k) < request.minimum_tile_extent:
            reasons.append(f"stage_{index}_tile_below_common_mma_minimum")
        if stage.handler_extent > output_shard:
            reasons.append(f"stage_{index}_handler_extent_exceeds_partition")
        if stage.handler_extent % stage.tile_n and stage.handler_extent != output_shard:
            reasons.append(f"stage_{index}_handler_extent_not_tile_aligned")
        if stage.compute_threads % 32 or stage.consumer_threads % 32:
            reasons.append(f"stage_{index}_thread_count_not_warp_aligned")
        if stage.consumer_threads > stage.compute_threads:
            reasons.append(f"stage_{index}_consumer_threads_exceed_compute_threads")
        if stage.max_outstanding > stage.pipeline_stages:
            reasons.append(f"stage_{index}_outstanding_exceeds_pipeline_stages")
        if stage_request.input_from_previous_stage and stage_request.input_dtype != stage_request.weight_dtype:
            reasons.append(f"stage_{index}_linked_input_weight_dtype_mismatch")
        if stage.transfer_family == DATAFLOW_EXECUTION_TRANSFER_TMA:
            if not target.supports_tma:
                reasons.append(f"stage_{index}_target_tma_unsupported")
            if stage.gemm_family == DATAFLOW_EXECUTION_GEMM_PORTABLE:
                reasons.append(f"stage_{index}_tma_requires_accelerated_gemm")
            if stage.consumer_threads < 128 or stage.consumer_threads % 128:
                reasons.append(f"stage_{index}_tma_consumer_requires_complete_warp_group")
        if (
            stage.gemm_family
            in {
                DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
                DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M,
            }
            and not target.supports_wgmma
        ):
            reasons.append(f"stage_{index}_target_warp_group_gemm_unsupported")
        if stage.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP:
            if stage.tile_m < 64:
                reasons.append(f"stage_{index}_warp_group_m_tile_below_64")
            if stage.compute_threads != 256:
                reasons.append(f"stage_{index}_regular_warp_group_requires_256_threads")
        if stage.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M:
            if stage.tile_m >= 64 or stage.tile_m % 8:
                reasons.append(f"stage_{index}_small_m_tile_must_be_8_aligned_below_64")
            if stage.compute_threads != 128:
                reasons.append(f"stage_{index}_small_m_requires_128_threads")
            if stage.tile_n % 64 or stage.tile_k % 64:
                reasons.append(f"stage_{index}_small_m_tile_not_64_aligned")
            if stage.tile_n > 256:
                reasons.append(f"stage_{index}_small_m_tile_exceeds_256")
            if not all(
                is_native_fp8(dtype)
                for dtype in (
                    stage_request.input_dtype,
                    stage_request.weight_dtype,
                )
            ):
                reasons.append(f"stage_{index}_small_m_requires_native_fp8")
        if stage.input_distribution != DATAFLOW_EXECUTION_INPUT_COOPERATIVE:
            if stage.transfer_family != DATAFLOW_EXECUTION_TRANSFER_TMA:
                reasons.append(f"stage_{index}_distributed_input_requires_tma")
            if stage_request.input_dtype != stage_request.weight_dtype:
                reasons.append(f"stage_{index}_distributed_input_dtype_mismatch")
            if not is_native_fp8(stage_request.input_dtype):
                reasons.append(f"stage_{index}_distributed_input_requires_native_fp8")
        if stage.input_distribution == DATAFLOW_EXECUTION_INPUT_MULTICAST and topology.cluster_size > 16:
            reasons.append(f"stage_{index}_multicast_cluster_mask_limit_exceeded")
        if stage.split_producers and (stage.input_distribution == DATAFLOW_EXECUTION_INPUT_COOPERATIVE):
            reasons.append(f"stage_{index}_split_producer_without_distributed_input")
        if stage.pipeline_stages not in (1, 2, 3, 4, 8):
            reasons.append(f"stage_{index}_unsupported_pipeline_stage_count")
        if index == 0 and stage.pipeline_stages not in (2, 3):
            reasons.append("first_stage_pipeline_stage_count_unsupported")
        if index > 0 and stage.pipeline_stages == 3:
            reasons.append(f"stage_{index}_pipeline_stage_count_requires_power_of_two")
        if stage.wait_depth not in (0, 1):
            reasons.append(f"stage_{index}_unsupported_wait_depth")

    for producer, consumer in request.linked_stage_pairs:
        if candidate.stages[producer].tile_n != candidate.stages[consumer].tile_k:
            reasons.append(f"linked_stage_{producer}_{consumer}_tile_contract_mismatch")
    if request.uniform_stage_implementation:
        for attribute in (
            "compute_threads",
            "loop_stages",
            "gemm_family",
            "transfer_family",
            "eviction_policy",
        ):
            if len({getattr(stage, attribute) for stage in candidate.stages}) != 1:
                reasons.append(f"uniform_stage_{attribute}_contract_mismatch")

    first_request = request.stages[0]
    first = candidate.stages[0]
    first_output_shard = first_request.output_extent // (topology.cluster_size if first_request.output_partitioned else 1)
    first_handler_tiles = math.ceil(first.handler_extent / first.tile_n)
    if first.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP:
        warp_groups = first.consumer_threads // 128
        columns = first_request.projection_count * first.tile_n
        if warp_groups not in (1, 2) or columns % max(1, warp_groups) or (columns // max(1, warp_groups)) % 128:
            reasons.append("first_stage_warp_group_column_partition_illegal")
    if first.pipeline_stages == 3 and not (
        (
            first.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M
            and first.input_distribution != DATAFLOW_EXECUTION_INPUT_COOPERATIVE
            and first.split_producers
            and first.consumer_threads == 128
        )
        or (
            first.wait_depth == 1
            and first.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP
            and first.input_distribution != DATAFLOW_EXECUTION_INPUT_COOPERATIVE
            and first.split_producers
            and first_handler_tiles == 1
        )
    ):
        reasons.append("first_stage_three_stage_pipeline_contract_illegal")
    if first.wait_depth and not (
        first.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP
        and first_handler_tiles == 1
        and first.pipeline_stages >= 3
        and first.input_distribution != DATAFLOW_EXECUTION_INPUT_COOPERATIVE
    ):
        reasons.append("first_stage_wait_depth_contract_illegal")
    if first_handler_tiles > 1 and not (
        first.handler_extent == first_output_shard
        and first.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M
        and first.input_distribution == DATAFLOW_EXECUTION_INPUT_UNICAST
        and first.consumer_threads == 128
        and not first.fused_transfers
        and candidate.transport_family in {DATAFLOW_TRANSPORT_ALL_GATHER, DATAFLOW_TRANSPORT_STREAMED}
    ):
        reasons.append("first_stage_range_fusion_contract_illegal")

    last_request = request.stages[-1]
    last = candidate.stages[-1]
    if last.full_partition_pipeline and not (
        candidate.transport_family in {DATAFLOW_TRANSPORT_ALL_GATHER, DATAFLOW_TRANSPORT_STREAMED}
        and last.transfer_family == DATAFLOW_EXECUTION_TRANSFER_TMA
        and last.gemm_family == DATAFLOW_EXECUTION_GEMM_WARP_GROUP
        and last.pipeline_stages >= 2
        and last.max_outstanding >= 2
        and last.pipeline_stages % 2 == 0
        and last.max_outstanding % 2 == 0
    ):
        reasons.append("last_stage_full_partition_pipeline_contract_illegal")
    if candidate.transport_family == DATAFLOW_TRANSPORT_STREAMED:
        if last.transfer_family != DATAFLOW_EXECUTION_TRANSFER_TMA:
            reasons.append("streamed_transport_requires_tma_consumer")
        if last.max_outstanding > 2:
            reasons.append("streamed_transport_outstanding_limit_exceeded")
        if topology.cluster_size & (topology.cluster_size - 1):
            reasons.append("streamed_transport_requires_power_of_two_cluster")
        if not is_native_fp8(last_request.input_dtype):
            reasons.append("streamed_transport_requires_native_fp8_input")
    if candidate.handoff_stages and not request.handoff_permitted:
        reasons.append("cross_handler_handoff_not_permitted")

    resources = estimate_resources(request, candidate, target)
    if target.max_dynamic_shared_memory is not None and resources.shared_memory_bytes > target.max_dynamic_shared_memory:
        reasons.append("target_shared_memory_limit_exceeded")
    if target.max_threads_per_block is not None and resources.max_threads > target.max_threads_per_block:
        reasons.append("target_threads_per_block_limit_exceeded")
    if target.max_registers_per_block is not None and resources.register_count > target.max_registers_per_block:
        reasons.append("target_register_limit_exceeded")
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return DataflowExecutionCandidateEvaluation(
            candidate=candidate,
            legal=False,
            resources=resources,
            rejection_reasons=tuple(reasons),
        )
    score = score_candidate(request, candidate, resources, target)
    return DataflowExecutionCandidateEvaluation(
        candidate=candidate,
        legal=True,
        resources=resources,
        score=score,
    )


def estimate_resources(
    request: DataflowExecutionRequest,
    candidate: DataflowExecutionCandidate,
    target: TargetCapabilitySnapshot,
) -> DataflowExecutionResourceEstimate:
    first_request = request.stages[0]
    first = candidate.stages[0]
    output_bytes = require_dataflow_dtype(first_request.output_dtype).element_bytes
    full_intermediate_bytes = first.tile_m * first_request.output_extent * output_bytes
    physical_producer_bytes = first.tile_m * first.handler_extent * output_bytes
    physical_output_arity = max(
        1,
        math.ceil(first_request.output_extent / first.handler_extent),
    )
    producer_slots_per_rank = max(
        1,
        physical_output_arity // candidate.topology.cluster_size,
    )
    local_producer_bytes = producer_slots_per_rank * physical_producer_bytes
    if candidate.transport_family == DATAFLOW_TRANSPORT_ALL_GATHER:
        reshared_storage_bytes = full_intermediate_bytes
    elif candidate.transport_family == DATAFLOW_TRANSPORT_STREAMED:
        reshared_storage_bytes = local_producer_bytes
    else:
        reshared_storage_bytes = 0
    handoff_bytes = candidate.handoff_stages * first.tile_m * first.tile_k * require_dataflow_dtype(first_request.input_dtype).element_bytes
    stage_shared = []
    stage_pipeline_shared = []
    barrier_count = 0
    register_count = 0
    max_threads = 0
    for stage_index, (stage_request, stage) in enumerate(zip(request.stages, candidate.stages)):
        input_bytes = require_dataflow_dtype(stage_request.input_dtype).element_bytes
        weight_bytes = require_dataflow_dtype(stage_request.weight_dtype).element_bytes
        output_element_bytes = require_dataflow_dtype(stage_request.output_dtype).element_bytes
        if stage.transfer_family == DATAFLOW_EXECUTION_TRANSFER_SYNCHRONOUS:
            versions = 1
            input_versions = 1
        else:
            versions = min(stage.pipeline_stages, 2)
            input_versions = min(stage.pipeline_stages, 3)
        input_storage = stage.tile_m * stage.tile_k * input_bytes * input_versions
        if stage_request.input_from_previous_stage and candidate.transport_family == DATAFLOW_TRANSPORT_ALL_GATHER:
            input_storage = 0
        weight_storage = stage_request.projection_count * stage.tile_n * stage.tile_k * weight_bytes * versions
        stage_pipeline_shared.append(input_storage + weight_storage)
        output_storage = stage.tile_m * stage.handler_extent * output_element_bytes if stage_request.projection_count > 1 else 0
        if stage_index == 0 and candidate.transport_family != DATAFLOW_TRANSPORT_HBM:
            # All-gather materializes the complete logical intermediate in
            # every CTA. Streamed transport retains only this rank's physical
            # producer slots; its receive ring is the linked consumer input
            # storage accounted for above.
            output_storage = max(output_storage, reshared_storage_bytes)
        stage_shared.append(input_storage + weight_storage + output_storage)
        transfer_count = 1 + stage_request.projection_count
        barrier_count += transfer_count * stage.pipeline_stages * 2
        physical_threads = stage.compute_threads + (128 if stage.transfer_family == DATAFLOW_EXECUTION_TRANSFER_TMA else 0)
        max_threads = max(max_threads, physical_threads)
        if stage.transfer_family == DATAFLOW_EXECUTION_TRANSFER_TMA:
            consumer_groups = max(1, stage.consumer_threads // 128)
            register_count = max(
                register_count,
                128 * (40 + consumer_groups * 232),
            )
        else:
            register_count = max(register_count, stage.compute_threads * 128)
    barrier_count += candidate.handoff_stages * 2
    shared_memory_bytes = (
        DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES
        + max(
            (handoff_bytes + value for value in stage_shared),
            default=handoff_bytes,
        )
        + barrier_count * 8
        + DATAFLOW_EXECUTION_SHARED_CONTROL_RESERVE_BYTES
        + DATAFLOW_SHARED_ALIGNMENT
    )
    budgets: list[int | None] = [None for _ in candidate.stages]
    if first.common_pipeline:
        hardware_budget = (
            None
            if target.max_dynamic_shared_memory is None
            else max(
                1,
                target.max_dynamic_shared_memory
                - DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES
                - reshared_storage_bytes
                - DATAFLOW_EXECUTION_SHARED_CONTROL_RESERVE_BYTES
                - DATAFLOW_SHARED_ALIGNMENT,
            )
        )
        # A stage beyond the requested outstanding depth is a prefetch-only
        # stage. Preserve the deeper input ring while keeping weight rings at
        # the requested concurrency; the multi-transfer planner derives the
        # heterogeneous versions from this aggregate budget. A genuinely
        # deeper outstanding pipeline retains the full hardware budget.
        preferred_budget = stage_pipeline_shared[0] if first.split_producers and first.pipeline_stages > first.max_outstanding else None
        if hardware_budget is None:
            budgets[0] = preferred_budget
        elif preferred_budget is None:
            budgets[0] = hardware_budget
        else:
            budgets[0] = min(hardware_budget, preferred_budget)
    return DataflowExecutionResourceEstimate(
        shared_memory_bytes=shared_memory_bytes,
        register_count=register_count,
        barrier_count=barrier_count,
        max_threads=max_threads,
        pipeline_shared_memory_budgets=tuple(budgets),
    )


def score_candidate(
    request: DataflowExecutionRequest,
    candidate: DataflowExecutionCandidate,
    resources: DataflowExecutionResourceEstimate,
    target: TargetCapabilitySnapshot,
) -> tuple[float, ...]:
    portable = all(stage.gemm_family == DATAFLOW_EXECUTION_GEMM_PORTABLE for stage in candidate.stages)
    acceleration_penalty = 1.0 if portable and target.supports_wgmma else 0.0
    task_count = sum(math.ceil(extent / candidate.task_tile_extent) for extent in (request.task_group_extents or (request.task_extent,)))
    cluster_count = max(1, candidate.topology.cluster_count)
    waves = math.ceil(task_count / cluster_count)
    tile_penalty = sum(abs(128 - stage.tile_n) for stage in candidate.stages)
    shared_ratio = (
        0.0 if target.max_dynamic_shared_memory in {None, 0} else resources.shared_memory_bytes / target.max_dynamic_shared_memory
    )
    transport_penalty = {
        DATAFLOW_TRANSPORT_ALL_GATHER: 0.0,
        DATAFLOW_TRANSPORT_STREAMED: 0.25,
        DATAFLOW_TRANSPORT_HBM: 1.0,
    }[candidate.transport_family]
    if small_m_reduction_chain(request):
        # Count serialized reduction/transfer turns on the busiest cluster.
        # Waves alone overvalue narrow tiles; volume alone misses dispatch and
        # pipeline re-entry. This shape-only proxy jointly ranks both phases.
        turns = sum(
            math.ceil(
                stage_request.output_extent / (candidate.topology.cluster_size if stage_request.output_partitioned else 1) / stage.tile_n
            )
            * (stage_request.reduction_extent // stage.tile_k)
            for stage_request, stage in zip(request.stages, candidate.stages)
        )
        handlers = sum(
            math.ceil(
                stage_request.output_extent
                / (candidate.topology.cluster_size if stage_request.output_partitioned else 1)
                / stage.handler_extent
            )
            for stage_request, stage in zip(request.stages, candidate.stages)
        )
        return (
            acceleration_penalty,
            float(waves * (turns + handlers)),
            transport_penalty,
            float(waves),
            round(shared_ratio, 8),
            float(-candidate.topology.cluster_size),
        )
    return (
        acceleration_penalty,
        float(waves),
        float(tile_penalty),
        transport_penalty,
        round(shared_ratio, 8),
        float(-candidate.topology.cluster_size),
    )


def deduplicate_evaluations(
    evaluations: Sequence[DataflowExecutionCandidateEvaluation],
) -> list[DataflowExecutionCandidateEvaluation]:
    unique: dict[str, DataflowExecutionCandidateEvaluation] = {}
    for evaluation in evaluations:
        unique.setdefault(evaluation.candidate.candidate_id, evaluation)
    return [unique[candidate_id] for candidate_id in sorted(unique)]


__all__ = [
    "DATAFLOW_EXECUTION_GEMM_AUTO",
    "DATAFLOW_EXECUTION_GEMM_FAMILIES",
    "DATAFLOW_EXECUTION_GEMM_PORTABLE",
    "DATAFLOW_EXECUTION_GEMM_WARP_GROUP",
    "DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M",
    "DATAFLOW_EXECUTION_INPUT_AUTO",
    "DATAFLOW_EXECUTION_INPUT_COOPERATIVE",
    "DATAFLOW_EXECUTION_INPUT_DISTRIBUTIONS",
    "DATAFLOW_EXECUTION_INPUT_MULTICAST",
    "DATAFLOW_EXECUTION_INPUT_UNICAST",
    "DATAFLOW_EXECUTION_OVERRIDE_SCHEMA_VERSION",
    "DATAFLOW_EXECUTION_PLAN_SCHEMA_VERSION",
    "DATAFLOW_EXECUTION_PLANNER_IMPLEMENTATION",
    "DATAFLOW_EXECUTION_PLANNER_VERSION",
    "DATAFLOW_EXECUTION_SHARED_CONTROL_RESERVE_BYTES",
    "DATAFLOW_EXECUTION_TRANSFER_AUTO",
    "DATAFLOW_EXECUTION_TRANSFER_FAMILIES",
    "DATAFLOW_EXECUTION_TRANSFER_SYNCHRONOUS",
    "DATAFLOW_EXECUTION_TRANSFER_TMA",
    "DataflowExecutionCandidate",
    "DataflowExecutionCandidateEvaluation",
    "DataflowExecutionOverride",
    "DataflowExecutionPlan",
    "DataflowExecutionPlanningError",
    "DataflowExecutionRequest",
    "DataflowExecutionResourceEstimate",
    "DataflowExecutionStageConfig",
    "DataflowExecutionStageOverride",
    "DataflowExecutionStageRequest",
    "capture_execution_plans",
    "plan_execution",
    "resolve_execution_override",
]
