"""Generic logical multi-transfer pipeline planning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from collections.abc import Mapping

from tvm import ir, tir

from tilelang import _ffi_api

from .operation_contracts import (
    DATAFLOW_PIPELINE_COMPLETION_SEMANTICS,
    DATAFLOW_PIPELINE_EVICTION_HINTS,
    DATAFLOW_PIPELINE_MATERIALIZATIONS,
    DATAFLOW_PIPELINE_MATERIALIZE_COPY,
    DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT,
    DATAFLOW_PIPELINE_RELEASE_SEMANTICS,
    DATAFLOW_PIPELINE_SYNCHRONIZATION_OWNERS,
    DataflowOperationResourceEstimate,
    DataflowPipelineRequest,
)


DATAFLOW_PIPELINE_PLAN_SCHEMA_VERSION = 3
DATAFLOW_PIPELINE_MODE_PIPELINED = "pipelined"
DATAFLOW_PIPELINE_MODE_SYNCHRONOUS = "synchronous"
DATAFLOW_PIPELINE_MODES = (
    DATAFLOW_PIPELINE_MODE_PIPELINED,
    DATAFLOW_PIPELINE_MODE_SYNCHRONOUS,
)
DATAFLOW_PIPELINE_PLAN_FINGERPRINT_ATTR = "tl.pipeline_dataflow_plan_fingerprint"
DATAFLOW_PIPELINE_PLAN_SCHEMA_ATTR = "tl.pipeline_dataflow_plan_schema_version"
DATAFLOW_PIPELINE_MODE_ATTR = "tl.pipeline_dataflow_mode"
DATAFLOW_PIPELINE_BUFFER_VERSIONS_ATTR = "tl.pipeline_buffer_versions"
DATAFLOW_PIPELINE_PRODUCER_PARTITION_ATTR = "tl.pipeline_producer_partition"
DATAFLOW_PIPELINE_PRODUCER_THREADS_ATTR = "tl.pipeline_producer_threads"
DATAFLOW_PIPELINE_MATERIALIZATION_ATTR = "tl.pipeline_materialization"
_TRANSFER_PIPELINE_SYNC_ATTR = "tl.transfer_pipeline_sync_consumed"
_TRANSFER_PIPELINE_SYNC_MANAGED = 1
_TRANSFER_PIPELINE_SYNC_FALLBACK = 2
_PIPELINE_FALLBACK_REASONS = frozenset(
    {
        "asynchronous_not_permitted",
        "buffer_multiversion_not_permitted",
        "shared_memory_budget",
        "register_budget",
        "barrier_budget",
    }
)


class DataflowPipelinePlanningError(ValueError):
    """Raised when a typed dataflow has no legal logical pipeline plan."""


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataflowPipelinePlanningError(f"pipeline {name} must be a positive integer, got {value!r}")
    return value


def nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataflowPipelinePlanningError(f"pipeline {name} must be a non-negative integer, got {value!r}")
    return value


def fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_fingerprint(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a canonical SHA-256 fingerprint")
    return value


def strict_fields(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} fields do not match schema: missing={sorted(expected - set(value))!r}, unknown={sorted(set(value) - expected)!r}"
        )


@dataclass(frozen=True)
class DataflowPipelineImplementationRequirements:
    """Capabilities and ownership a concrete implementation must satisfy."""

    mode: str
    stage_count: int
    max_outstanding: int
    transfer_count: int
    gemm_count: int
    buffer_count: int
    async_transfer_indices: tuple[int, ...]
    multicast_permitted_transfer_indices: tuple[int, ...]
    buffer_versions: tuple[int, ...]
    additive_gemm_groups: tuple[tuple[int, ...], ...]
    release_after_gemm_indices: tuple[int, ...]
    transfer_materializations: tuple[str, ...]
    eviction_hints: tuple[str, ...]
    producer_partitions: tuple[int | None, ...]
    producer_threads: int | None
    consumer_threads: int | None
    synchronization_owner: str
    completion_semantics: str
    release_semantics: str

    def __post_init__(self) -> None:
        if self.mode not in DATAFLOW_PIPELINE_MODES:
            raise ValueError(f"unsupported pipeline mode {self.mode!r}")
        for name in (
            "stage_count",
            "max_outstanding",
            "transfer_count",
            "gemm_count",
            "buffer_count",
        ):
            positive_int(getattr(self, name), name)
        if self.max_outstanding > self.stage_count:
            raise ValueError("pipeline outstanding count cannot exceed stage count")
        if self.mode == DATAFLOW_PIPELINE_MODE_SYNCHRONOUS:
            if self.stage_count != 1 or self.max_outstanding != 1:
                raise ValueError("synchronous pipeline requirements use one stage")
        elif self.stage_count <= 1:
            raise ValueError("pipelined requirements need more than one stage")

        for name, limit in (
            ("async_transfer_indices", self.transfer_count),
            (
                "multicast_permitted_transfer_indices",
                self.transfer_count,
            ),
        ):
            values = tuple(getattr(self, name))
            if values != tuple(sorted(set(values))) or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= limit for index in values
            ):
                raise ValueError(f"pipeline {name} must be sorted unique in-range indices")
            object.__setattr__(self, name, values)
        if self.mode == DATAFLOW_PIPELINE_MODE_SYNCHRONOUS and (self.async_transfer_indices):
            raise ValueError("synchronous requirements cannot require async transfers")

        versions = tuple(self.buffer_versions)
        if len(versions) != self.buffer_count or any(
            isinstance(version, bool) or not isinstance(version, int) or version <= 0 or version > self.stage_count for version in versions
        ):
            raise ValueError("pipeline buffer versions do not match the plan")
        object.__setattr__(self, "buffer_versions", versions)

        groups = tuple(tuple(group) for group in self.additive_gemm_groups)
        flattened = tuple(index for group in groups for index in group)
        if (
            not groups
            or any(not group for group in groups)
            or tuple(sorted(flattened)) != tuple(range(self.gemm_count))
            or len(flattened) != len(set(flattened))
            or any(group != tuple(sorted(group)) for group in groups)
        ):
            raise ValueError("pipeline additive GEMM groups must partition consumer indices")
        object.__setattr__(self, "additive_gemm_groups", groups)

        releases = tuple(self.release_after_gemm_indices)
        if len(releases) != self.buffer_count or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= self.gemm_count for index in releases
        ):
            raise ValueError("pipeline release dependencies are invalid")
        object.__setattr__(self, "release_after_gemm_indices", releases)

        materializations = tuple(self.transfer_materializations)
        if len(materializations) != self.transfer_count or any(
            materialization not in DATAFLOW_PIPELINE_MATERIALIZATIONS for materialization in materializations
        ):
            raise ValueError("pipeline materializations do not match logical transfers")
        object.__setattr__(self, "transfer_materializations", materializations)

        eviction_hints = tuple(self.eviction_hints)
        if len(eviction_hints) != self.transfer_count or any(hint not in DATAFLOW_PIPELINE_EVICTION_HINTS for hint in eviction_hints):
            raise ValueError("pipeline eviction hints do not match transfers")
        object.__setattr__(self, "eviction_hints", eviction_hints)
        producer_partitions = tuple(self.producer_partitions)
        if len(producer_partitions) != self.transfer_count or any(
            partition is not None and (isinstance(partition, bool) or not isinstance(partition, int) or partition < 0)
            for partition in producer_partitions
        ):
            raise ValueError("pipeline producer partitions do not match transfers")
        if self.mode == DATAFLOW_PIPELINE_MODE_SYNCHRONOUS and any(partition is not None for partition in producer_partitions):
            raise ValueError("synchronous pipeline requirements cannot partition producers")
        object.__setattr__(self, "producer_partitions", producer_partitions)
        for name in ("producer_threads", "consumer_threads"):
            value = getattr(self, name)
            if value is not None:
                positive_int(value, name)
        if self.synchronization_owner not in DATAFLOW_PIPELINE_SYNCHRONIZATION_OWNERS:
            raise ValueError("unsupported pipeline synchronization owner")
        if self.completion_semantics not in DATAFLOW_PIPELINE_COMPLETION_SEMANTICS:
            raise ValueError("unsupported pipeline completion semantics")
        if self.release_semantics not in DATAFLOW_PIPELINE_RELEASE_SEMANTICS:
            raise ValueError("unsupported pipeline release semantics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "stage_count": self.stage_count,
            "max_outstanding": self.max_outstanding,
            "transfer_count": self.transfer_count,
            "gemm_count": self.gemm_count,
            "buffer_count": self.buffer_count,
            "async_transfer_indices": list(self.async_transfer_indices),
            "multicast_permitted_transfer_indices": list(self.multicast_permitted_transfer_indices),
            "buffer_versions": list(self.buffer_versions),
            "additive_gemm_groups": [list(group) for group in self.additive_gemm_groups],
            "release_after_gemm_indices": list(self.release_after_gemm_indices),
            "transfer_materializations": list(self.transfer_materializations),
            "eviction_hints": list(self.eviction_hints),
            "producer_partitions": list(self.producer_partitions),
            "producer_threads": self.producer_threads,
            "consumer_threads": self.consumer_threads,
            "synchronization_owner": self.synchronization_owner,
            "completion_semantics": self.completion_semantics,
            "release_semantics": self.release_semantics,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> DataflowPipelineImplementationRequirements:
        if not isinstance(value, Mapping):
            raise TypeError("pipeline implementation requirements must be a mapping")
        expected = set(cls.__dataclass_fields__)
        strict_fields(value, expected, "pipeline implementation requirements")
        for name in (
            "async_transfer_indices",
            "multicast_permitted_transfer_indices",
            "buffer_versions",
            "additive_gemm_groups",
            "release_after_gemm_indices",
            "transfer_materializations",
            "eviction_hints",
            "producer_partitions",
        ):
            if not isinstance(value[name], (tuple, list)):
                raise TypeError(f"pipeline requirements {name} must be a sequence")
        if any(not isinstance(group, (tuple, list)) for group in value["additive_gemm_groups"]):
            raise TypeError("pipeline additive GEMM groups must be sequences")
        return cls(
            mode=value["mode"],
            stage_count=value["stage_count"],
            max_outstanding=value["max_outstanding"],
            transfer_count=value["transfer_count"],
            gemm_count=value["gemm_count"],
            buffer_count=value["buffer_count"],
            async_transfer_indices=tuple(value["async_transfer_indices"]),
            multicast_permitted_transfer_indices=tuple(value["multicast_permitted_transfer_indices"]),
            buffer_versions=tuple(value["buffer_versions"]),
            additive_gemm_groups=tuple(tuple(group) for group in value["additive_gemm_groups"]),
            release_after_gemm_indices=tuple(value["release_after_gemm_indices"]),
            transfer_materializations=tuple(value["transfer_materializations"]),
            eviction_hints=tuple(value["eviction_hints"]),
            producer_partitions=tuple(value["producer_partitions"]),
            producer_threads=value["producer_threads"],
            consumer_threads=value["consumer_threads"],
            synchronization_owner=value["synchronization_owner"],
            completion_semantics=value["completion_semantics"],
            release_semantics=value["release_semantics"],
        )


@dataclass(frozen=True)
class DataflowPipelinePlan:
    """Selected logical stages, requirements, resources, and fallback state."""

    request_fingerprint: str
    requested_stage_budget: int | None
    auto_stage_budget: int
    selected_stages: int
    requested_max_outstanding: int | None
    selected_max_outstanding: int
    implementation_requirements: DataflowPipelineImplementationRequirements
    resources: DataflowOperationResourceEstimate
    fallback_reasons: tuple[str, ...]
    schema_version: int = DATAFLOW_PIPELINE_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != DATAFLOW_PIPELINE_PLAN_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported pipeline dataflow plan schema version {self.schema_version!r}")
        require_fingerprint(self.request_fingerprint, "pipeline request fingerprint")
        if self.requested_stage_budget is not None:
            positive_int(self.requested_stage_budget, "requested_stage_budget")
        positive_int(self.auto_stage_budget, "auto_stage_budget")
        positive_int(self.selected_stages, "selected_stages")
        desired_stages = self.requested_stage_budget or self.auto_stage_budget
        if self.selected_stages > desired_stages:
            raise ValueError("pipeline plan cannot exceed its stage budget")
        if self.requested_max_outstanding is not None:
            positive_int(
                self.requested_max_outstanding,
                "requested_max_outstanding",
            )
        positive_int(
            self.selected_max_outstanding,
            "selected_max_outstanding",
        )
        if self.selected_max_outstanding > self.selected_stages:
            raise ValueError("pipeline plan outstanding count exceeds selected stages")
        if not isinstance(
            self.implementation_requirements,
            DataflowPipelineImplementationRequirements,
        ):
            raise TypeError("pipeline plan requires typed implementation requirements")
        if (
            self.implementation_requirements.stage_count != self.selected_stages
            or self.implementation_requirements.max_outstanding != self.selected_max_outstanding
        ):
            raise ValueError("pipeline plan requirements changed selected stages")
        if not isinstance(self.resources, DataflowOperationResourceEstimate):
            raise TypeError("pipeline plan resources must be typed")
        reasons = tuple(self.fallback_reasons)
        if any(not isinstance(reason, str) or reason not in _PIPELINE_FALLBACK_REASONS for reason in reasons):
            raise ValueError("pipeline plan contains an unsupported fallback reason")
        if len(reasons) != len(set(reasons)):
            raise ValueError("pipeline fallback reasons must be unique")
        if bool(reasons) != (self.selected_stages < desired_stages):
            raise ValueError("pipeline fallback reasons do not match selected stage count")
        object.__setattr__(self, "fallback_reasons", reasons)

    @property
    def used_fallback(self) -> bool:
        return bool(self.fallback_reasons)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_fingerprint": self.request_fingerprint,
            "requested_stage_budget": self.requested_stage_budget,
            "auto_stage_budget": self.auto_stage_budget,
            "selected_stages": self.selected_stages,
            "requested_max_outstanding": self.requested_max_outstanding,
            "selected_max_outstanding": self.selected_max_outstanding,
            "implementation_requirements": (self.implementation_requirements.to_dict()),
            "resources": self.resources.to_dict(),
            "used_fallback": self.used_fallback,
            "fallback_reasons": list(self.fallback_reasons),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        result = self.canonical_payload()
        result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowPipelinePlan:
        if not isinstance(value, Mapping):
            raise TypeError("pipeline dataflow plan must be a mapping")
        expected = {
            "schema_version",
            "request_fingerprint",
            "requested_stage_budget",
            "auto_stage_budget",
            "selected_stages",
            "requested_max_outstanding",
            "selected_max_outstanding",
            "implementation_requirements",
            "resources",
            "used_fallback",
            "fallback_reasons",
            "fingerprint",
        }
        strict_fields(value, expected, "pipeline dataflow plan")
        if not isinstance(value["fallback_reasons"], (tuple, list)):
            raise TypeError("pipeline fallback_reasons must be a sequence")
        plan = cls(
            request_fingerprint=value["request_fingerprint"],
            requested_stage_budget=value["requested_stage_budget"],
            auto_stage_budget=value["auto_stage_budget"],
            selected_stages=value["selected_stages"],
            requested_max_outstanding=value["requested_max_outstanding"],
            selected_max_outstanding=value["selected_max_outstanding"],
            implementation_requirements=(DataflowPipelineImplementationRequirements.from_dict(value["implementation_requirements"])),
            resources=DataflowOperationResourceEstimate.from_dict(value["resources"]),
            fallback_reasons=tuple(value["fallback_reasons"]),
            schema_version=value["schema_version"],
        )
        if not isinstance(value["used_fallback"], bool):
            raise TypeError("pipeline used_fallback must be a bool")
        if value["used_fallback"] != plan.used_fallback:
            raise ValueError("pipeline fallback state does not match reasons")
        if value["fingerprint"] != plan.fingerprint:
            raise ValueError("pipeline plan fingerprint does not match payload")
        return plan


def additive_groups(
    request: DataflowPipelineRequest,
) -> tuple[tuple[int, ...], ...]:
    groups: dict[int, list[int]] = {}
    for index, gemm in enumerate(request.gemms):
        groups.setdefault(gemm.accumulator_index, []).append(index)
    return tuple(tuple(groups[key]) for key in sorted(groups))


def build_requirements(
    request: DataflowPipelineRequest,
    stages: int,
    *,
    buffer_versions: tuple[int, ...] | None = None,
) -> DataflowPipelineImplementationRequirements:
    pipelined = stages > 1 and any(transfer.async_permitted for transfer in request.transfers)
    mode = DATAFLOW_PIPELINE_MODE_PIPELINED if pipelined else DATAFLOW_PIPELINE_MODE_SYNCHRONOUS
    effective_stages = stages if pipelined else 1
    outstanding = 1 if not pipelined else min(request.max_outstanding or effective_stages, effective_stages)
    if buffer_versions is None:
        buffer_versions = tuple(
            (
                effective_stages
                if lifetime.allow_multiversion
                and request.transfers[lifetime.producer_transfer_index].materialization == DATAFLOW_PIPELINE_MATERIALIZE_COPY
                else 1
            )
            for lifetime in request.buffer_lifetimes
        )
    return DataflowPipelineImplementationRequirements(
        mode=mode,
        stage_count=effective_stages,
        max_outstanding=outstanding,
        transfer_count=len(request.transfers),
        gemm_count=len(request.gemms),
        buffer_count=len(request.buffer_lifetimes),
        async_transfer_indices=(
            tuple(index for index, transfer in enumerate(request.transfers) if transfer.async_permitted) if pipelined else ()
        ),
        multicast_permitted_transfer_indices=tuple(
            index for index, transfer in enumerate(request.transfers) if transfer.multicast_permitted
        ),
        buffer_versions=buffer_versions,
        additive_gemm_groups=additive_groups(request),
        release_after_gemm_indices=tuple(lifetime.release_after_gemm_index for lifetime in request.buffer_lifetimes),
        transfer_materializations=tuple(transfer.materialization for transfer in request.transfers),
        eviction_hints=tuple(transfer.eviction_hint for transfer in request.transfers),
        producer_partitions=(
            tuple(transfer.producer_partition for transfer in request.transfers) if pipelined else (None,) * len(request.transfers)
        ),
        producer_threads=request.producer_threads,
        consumer_threads=request.consumer_threads,
        synchronization_owner=request.synchronization_owner,
        completion_semantics=request.completion_semantics,
        release_semantics=request.release_semantics,
    )


def build_resources(
    request: DataflowPipelineRequest,
    requirements: DataflowPipelineImplementationRequirements,
) -> DataflowOperationResourceEstimate:
    bytes_by_buffer = {
        transfer.destination_buffer_index: transfer.bytes_per_stage
        for transfer in request.transfers
        if transfer.materialization == DATAFLOW_PIPELINE_MATERIALIZE_COPY
    }
    shared_memory_bytes = sum(bytes_by_buffer.get(index, 0) * versions for index, versions in enumerate(requirements.buffer_versions))
    register_bytes = request.fixed_register_bytes + request.register_bytes_per_stage * requirements.stage_count
    barrier_count = len(requirements.async_transfer_indices) * requirements.max_outstanding
    return DataflowOperationResourceEstimate(
        shared_memory_bytes=shared_memory_bytes,
        register_bytes=register_bytes,
        barrier_count=barrier_count,
        transaction_bytes=sum(
            transfer.bytes_per_stage for transfer in request.transfers if transfer.materialization == DATAFLOW_PIPELINE_MATERIALIZE_COPY
        ),
        temporary_bytes=0,
    )


def resource_rejections(
    request: DataflowPipelineRequest,
    resources: DataflowOperationResourceEstimate,
) -> tuple[str, ...]:
    limits = (
        (
            "shared_memory_budget",
            resources.shared_memory_bytes,
            request.max_shared_memory_bytes,
        ),
        (
            "register_budget",
            resources.register_bytes,
            request.max_register_bytes,
        ),
        (
            "barrier_budget",
            resources.barrier_count,
            request.max_barrier_count,
        ),
    )
    return tuple(reason for reason, selected, limit in limits if limit is not None and selected is not None and selected > limit)


def plan_incremental_buffer_versions(
    request: DataflowPipelineRequest,
    *,
    base_requirements: DataflowPipelineImplementationRequirements,
    target_stages: int,
) -> (
    tuple[
        DataflowPipelineImplementationRequirements,
        DataflowOperationResourceEstimate,
    ]
    | None
):
    """Raise individual copy rings in graph order while budgets permit."""

    if base_requirements.stage_count <= 1:
        return None
    versions = list(base_requirements.buffer_versions)
    requirements = build_requirements(
        request,
        target_stages,
        buffer_versions=tuple(versions),
    )
    resources = build_resources(request, requirements)
    if resource_rejections(request, resources):
        return None

    for version in range(base_requirements.stage_count + 1, target_stages + 1):
        for transfer in request.transfers:
            buffer_index = transfer.destination_buffer_index
            if versions[buffer_index] >= version:
                continue
            lifetime = request.buffer_lifetimes[buffer_index]
            if transfer.materialization != DATAFLOW_PIPELINE_MATERIALIZE_COPY or not lifetime.allow_multiversion:
                continue
            candidate_versions = list(versions)
            candidate_versions[buffer_index] = version
            candidate_requirements = build_requirements(
                request,
                target_stages,
                buffer_versions=tuple(candidate_versions),
            )
            candidate_resources = build_resources(request, candidate_requirements)
            if not resource_rejections(request, candidate_resources):
                versions = candidate_versions
                requirements = candidate_requirements
                resources = candidate_resources

    if max(versions) != target_stages:
        return None
    return requirements, resources


def plan_pipeline_dataflow(
    request: DataflowPipelineRequest,
    *,
    auto_stage_budget: int = 2,
) -> DataflowPipelinePlan:
    """Select logical stages without selecting transfer, GEMM, or sync codegen."""

    if not isinstance(request, DataflowPipelineRequest):
        raise TypeError(f"plan_pipeline_dataflow expects DataflowPipelineRequest, got {type(request).__name__}")
    auto_stage_budget = positive_int(auto_stage_budget, "auto_stage_budget")
    desired_stages = request.stage_budget or auto_stage_budget
    forced_reasons: list[str] = []
    maximum_stages = desired_stages
    if desired_stages > 1 and not any(transfer.async_permitted for transfer in request.transfers):
        maximum_stages = 1
        forced_reasons.append("asynchronous_not_permitted")
    if desired_stages > 1 and any(
        not lifetime.allow_multiversion
        for lifetime in request.buffer_lifetimes
        if request.transfers[lifetime.producer_transfer_index].materialization == DATAFLOW_PIPELINE_MATERIALIZE_COPY
    ):
        maximum_stages = 1
        forced_reasons.append("buffer_multiversion_not_permitted")

    desired_requirements = build_requirements(request, maximum_stages)
    desired_resources = build_resources(request, desired_requirements)
    resource_reasons = resource_rejections(request, desired_resources)
    selected_requirements = desired_requirements
    selected_resources = desired_resources
    selected_stages = selected_requirements.stage_count
    if resource_reasons:
        for candidate_stages in range(maximum_stages - 1, 0, -1):
            candidate_requirements = build_requirements(request, candidate_stages)
            candidate_resources = build_resources(request, candidate_requirements)
            if not resource_rejections(request, candidate_resources):
                selected_requirements = candidate_requirements
                selected_resources = candidate_resources
                selected_stages = candidate_requirements.stage_count
                break
        else:
            raise DataflowPipelinePlanningError("pipeline synchronous fallback exceeds resource limits: " + ", ".join(resource_reasons))

        for candidate_stages in range(maximum_stages, selected_stages, -1):
            incremental = plan_incremental_buffer_versions(
                request,
                base_requirements=selected_requirements,
                target_stages=candidate_stages,
            )
            if incremental is None:
                continue
            selected_requirements, selected_resources = incremental
            selected_stages = candidate_stages
            break

    fallback_reasons = tuple(dict.fromkeys((*forced_reasons, *resource_reasons)) if selected_stages < desired_stages else ())
    return DataflowPipelinePlan(
        request_fingerprint=request.fingerprint,
        requested_stage_budget=request.stage_budget,
        auto_stage_budget=auto_stage_budget,
        selected_stages=selected_stages,
        requested_max_outstanding=request.max_outstanding,
        selected_max_outstanding=selected_requirements.max_outstanding,
        implementation_requirements=selected_requirements,
        resources=selected_resources,
        fallback_reasons=fallback_reasons,
    )


def int_annotation(value: Any, context: str) -> int:
    raw = getattr(value, "value", value)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise DataflowPipelinePlanningError(f"{context} must be a static integer, got {value!r}")
    return raw


def group_pipeline_transfer_calls(
    copy_calls: list[tir.Call],
    transfer_count: int,
) -> list[list[tir.Call]]:
    if len(copy_calls) == transfer_count:
        return [[call] for call in copy_calls]

    groups: list[list[tir.Call]] = []
    for call in copy_calls:
        parsed = _ffi_api.ParseOperator(call)
        matching_group = next(
            (group for group in groups if _ffi_api.ParseOperator(group[0]).dst.data.same_as(parsed.dst.data)),
            None,
        )
        if matching_group is None:
            groups.append([call])
            continue
        representative = _ffi_api.ParseOperator(matching_group[0])
        if not ir.structural_equal(
            representative.transfer_contract,
            parsed.transfer_contract,
            map_free_vars=True,
        ):
            raise DataflowPipelinePlanningError("pipeline dispatch alternatives changed their transfer contract")
        matching_group.append(call)
    if len(groups) != transfer_count:
        raise DataflowPipelinePlanningError(
            "pipeline PrimFunc transfer count does not match typed request: "
            f"{len(groups)} logical ({len(copy_calls)} physical) != "
            f"{transfer_count}"
        )
    return groups


def plan_primfunc_pipeline_dataflow(
    prim_func: tir.PrimFunc,
    request: DataflowPipelineRequest,
    *,
    loop_index: int = 0,
) -> DataflowPipelinePlan:
    """Validate one ordinary PrimFunc loop against the typed dependency graph."""

    if not isinstance(prim_func, tir.PrimFunc):
        raise TypeError("pipeline PrimFunc planning requires a tir.PrimFunc")
    if isinstance(loop_index, bool) or not isinstance(loop_index, int) or loop_index < 0:
        raise ValueError("pipeline loop_index must be a non-negative integer")
    loops: list[tir.For] = []

    def collect_loop(node: Any) -> None:
        if not isinstance(node, tir.For):
            return
        if any(key in node.annotations for key in ("num_stages", "tl_pipelined_num_stages")):
            loops.append(node)

    tir.stmt_functor.post_order_visit(prim_func.body, collect_loop)
    if loop_index >= len(loops):
        raise DataflowPipelinePlanningError(f"pipeline PrimFunc has {len(loops)} annotated loops, cannot select index {loop_index}")
    loop = loops[loop_index]
    stage_value = next(loop.annotations[key] for key in ("num_stages", "tl_pipelined_num_stages") if key in loop.annotations)
    loop_stages = positive_int(
        int_annotation(stage_value, "pipeline loop stage count"),
        "loop stage count",
    )
    copy_calls: list[tir.Call] = []
    gemm_calls: list[tir.Call] = []

    def collect_call(node: Any) -> None:
        if not isinstance(node, tir.Call) or not isinstance(node.op, ir.Op):
            return
        if node.op.name == "tl.tileop.copy":
            copy_calls.append(node)
        elif node.op.name == "tl.tileop.gemm":
            gemm_calls.append(node)

    tir.stmt_functor.post_order_visit(loop.body, collect_call)
    copy_groups = group_pipeline_transfer_calls(
        copy_calls,
        len(request.transfers),
    )
    if len(gemm_calls) != len(request.gemms):
        raise DataflowPipelinePlanningError(
            f"pipeline PrimFunc GEMM count does not match typed request: {len(gemm_calls)} != {len(request.gemms)}"
        )

    destination_buffers = []
    for index, (calls, transfer) in enumerate(zip(copy_groups, request.transfers)):
        call = calls[0]
        parsed = _ffi_api.ParseOperator(call)
        cluster_mask = int_annotation(
            call.annotations.get("cluster_mask", 0),
            f"pipeline transfer {index} cluster mask",
        )
        if cluster_mask < 0:
            raise DataflowPipelinePlanningError(f"pipeline transfer {index} cluster mask must be non-negative")
        if cluster_mask and not transfer.multicast_permitted:
            raise DataflowPipelinePlanningError(f"pipeline transfer {index} requests multicast without permission")
        if transfer.materialization == DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT and (
            not parsed.src.data.same_as(parsed.dst.data)
            or not ir.structural_equal(
                parsed.src_range,
                parsed.dst_range,
                map_free_vars=True,
            )
        ):
            raise DataflowPipelinePlanningError(
                f"pipeline transfer {index} declares resident materialization but its source and destination regions differ"
            )
        if any(parsed.dst.data.same_as(destination.data) for destination in destination_buffers):
            raise DataflowPipelinePlanningError(f"pipeline transfer {index} aliases an earlier typed buffer")
        destination_buffers.append(parsed.dst)
        destination_region = tuple(parsed.dst_range)
        if len(destination_region) != len(transfer.logical_extent):
            raise DataflowPipelinePlanningError(f"pipeline transfer {index} rank does not match typed extent")
        for axis, (expected, region) in enumerate(zip(transfer.logical_extent, destination_region)):
            if expected is None:
                continue
            extent = getattr(region.extent, "value", None)
            if extent is None or int(extent) != expected:
                raise DataflowPipelinePlanningError(
                    f"pipeline transfer {index} axis {axis} extent does not match typed request: {extent!r} != {expected}"
                )
        for alternative in calls[1:]:
            alternative_copy = _ffi_api.ParseOperator(alternative)
            if not ir.structural_equal(
                alternative_copy.dst_range,
                parsed.dst_range,
                map_free_vars=True,
            ):
                raise DataflowPipelinePlanningError(f"pipeline transfer {index} dispatch destinations differ")

    accumulator_buffers: dict[int, Any] = {}
    for index, (call, gemm) in enumerate(zip(gemm_calls, request.gemms)):
        parsed = _ffi_api.ParseOperator(call)
        expected_inputs = tuple(destination_buffers[buffer_index] for buffer_index in gemm.input_buffer_indices)
        actual_inputs = (parsed.a, parsed.b)
        if any(not actual.data.same_as(expected.data) for actual, expected in zip(actual_inputs, expected_inputs)):
            raise DataflowPipelinePlanningError(f"pipeline GEMM {index} inputs do not match typed buffer edges")
        accumulator = accumulator_buffers.setdefault(
            gemm.accumulator_index,
            parsed.c,
        )
        if not parsed.c.data.same_as(accumulator.data):
            raise DataflowPipelinePlanningError(f"pipeline GEMM {index} changed its typed accumulator group")
        if gemm.accumulator_dependency and bool(parsed.clearAccum):
            raise DataflowPipelinePlanningError(f"pipeline GEMM {index} clears an additive accumulator")
    plan = plan_pipeline_dataflow(
        request,
        auto_stage_budget=(loop_stages if request.stage_budget is None else 2),
    )
    if plan.selected_stages != loop_stages:
        raise DataflowPipelinePlanningError(
            f"pipeline PrimFunc stage count does not match the selected plan: {loop_stages} != {plan.selected_stages}"
        )
    return plan


def bind_pipeline_dataflow_plan(
    prim_func: tir.PrimFunc,
    request: DataflowPipelineRequest,
    *,
    loop_index: int = 0,
) -> tuple[tir.PrimFunc, DataflowPipelinePlan]:
    """Validate a PrimFunc and bind its logical plan to physical tile ops."""

    plan = plan_primfunc_pipeline_dataflow(
        prim_func,
        request,
        loop_index=loop_index,
    )
    pipeline_loops: list[tir.For] = []

    def collect_loop(node: Any) -> None:
        if isinstance(node, tir.For) and any(key in node.annotations for key in ("num_stages", "tl_pipelined_num_stages")):
            pipeline_loops.append(node)

    tir.stmt_functor.post_order_visit(prim_func.body, collect_loop)
    selected_loop = pipeline_loops[loop_index]
    copy_calls: list[tir.Call] = []

    def collect_copy(node: Any) -> None:
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tileop.copy":
            copy_calls.append(node)

    tir.stmt_functor.post_order_visit(selected_loop.body, collect_copy)
    copy_groups = group_pipeline_transfer_calls(
        copy_calls,
        len(request.transfers),
    )
    async_indices = set(plan.implementation_requirements.async_transfer_indices)
    transfer_modes = [
        (_TRANSFER_PIPELINE_SYNC_MANAGED if graph_index in async_indices else _TRANSFER_PIPELINE_SYNC_FALLBACK)
        for graph_index in range(len(copy_groups))
    ]
    producer_partitions = plan.implementation_requirements.producer_partitions
    buffer_versions = plan.implementation_requirements.buffer_versions

    def annotate_transfer(node: Any) -> Any:
        if not isinstance(node, tir.Call):
            return node
        graph_index = next(
            (graph_index for graph_index, group in enumerate(copy_groups) if any(node.same_as(call) for call in group)),
            None,
        )
        if graph_index is None:
            return node
        if (
            _TRANSFER_PIPELINE_SYNC_ATTR in node.annotations
            or DATAFLOW_PIPELINE_BUFFER_VERSIONS_ATTR in node.annotations
            or DATAFLOW_PIPELINE_PRODUCER_PARTITION_ATTR in node.annotations
            or DATAFLOW_PIPELINE_MATERIALIZATION_ATTR in node.annotations
        ):
            raise DataflowPipelinePlanningError(
                "pipeline transfer synchronization, buffer versions, producer partition, and materialization are compiler-owned"
            )
        annotations = dict(node.annotations)
        annotations[_TRANSFER_PIPELINE_SYNC_ATTR] = tir.IntImm(
            "int32",
            transfer_modes[graph_index],
        )
        annotations[DATAFLOW_PIPELINE_BUFFER_VERSIONS_ATTR] = tir.IntImm(
            "int32",
            buffer_versions[request.transfers[graph_index].destination_buffer_index],
        )
        producer_partition = producer_partitions[graph_index]
        annotations[DATAFLOW_PIPELINE_MATERIALIZATION_ATTR] = tir.StringImm(
            plan.implementation_requirements.transfer_materializations[graph_index]
        )
        if producer_partition is not None:
            annotations[DATAFLOW_PIPELINE_PRODUCER_PARTITION_ATTR] = tir.IntImm(
                "int32",
                producer_partition,
            )
        return tir.Call(
            node.dtype,
            node.op,
            list(node.args),
            annotations=annotations,
            span=node.span,
        )

    def annotate_mode(node: Any) -> Any:
        if not isinstance(node, tir.For) or not node.same_as(selected_loop):
            return node
        annotations = dict(node.annotations)
        annotations[DATAFLOW_PIPELINE_MODE_ATTR] = tir.StringImm(plan.implementation_requirements.mode)
        producer_threads = plan.implementation_requirements.producer_threads
        if producer_threads is not None:
            annotations[DATAFLOW_PIPELINE_PRODUCER_THREADS_ATTR] = tir.IntImm(
                "int32",
                producer_threads,
            )
        return tir.For(
            node.loop_var,
            node.min,
            node.extent,
            node.kind,
            tir.stmt_functor.ir_transform(
                node.body,
                None,
                annotate_transfer,
                ["tir.Call"],
            ),
            node.thread_binding,
            annotations,
            node.step,
            getattr(node, "span", None),
        )

    prim_func = prim_func.with_body(
        tir.stmt_functor.ir_transform(
            prim_func.body,
            None,
            annotate_mode,
            ["tir.For"],
        )
    )
    prim_func = prim_func.with_attr(
        DATAFLOW_PIPELINE_PLAN_SCHEMA_ATTR,
        plan.schema_version,
    )
    prim_func = prim_func.with_attr(
        DATAFLOW_PIPELINE_PLAN_FINGERPRINT_ATTR,
        plan.fingerprint,
    )
    return prim_func, plan


__all__ = [
    "DATAFLOW_PIPELINE_BUFFER_VERSIONS_ATTR",
    "DATAFLOW_PIPELINE_MODE_ATTR",
    "DATAFLOW_PIPELINE_PLAN_FINGERPRINT_ATTR",
    "DATAFLOW_PIPELINE_PLAN_SCHEMA_ATTR",
    "DATAFLOW_PIPELINE_PLAN_SCHEMA_VERSION",
    "DATAFLOW_PIPELINE_MATERIALIZATION_ATTR",
    "DATAFLOW_PIPELINE_PRODUCER_PARTITION_ATTR",
    "DATAFLOW_PIPELINE_PRODUCER_THREADS_ATTR",
    "DATAFLOW_PIPELINE_MODES",
    "DATAFLOW_PIPELINE_MODE_PIPELINED",
    "DATAFLOW_PIPELINE_MODE_SYNCHRONOUS",
    "DataflowPipelinePlan",
    "DataflowPipelineImplementationRequirements",
    "DataflowPipelinePlanningError",
    "bind_pipeline_dataflow_plan",
    "plan_pipeline_dataflow",
    "plan_primfunc_pipeline_dataflow",
]
