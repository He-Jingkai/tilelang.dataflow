"""Lifecycle governance for selectable Dataflow and TileLang implementations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
import hashlib
import json
import re
from collections.abc import Iterable

from .operation_contracts import (
    DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
    DATAFLOW_LAYOUT_LINEAR_IMPLEMENTATION,
    DATAFLOW_LAYOUT_MATRIX_SWIZZLE_IMPLEMENTATION,
    DATAFLOW_OPERATION_CONTRACT_KINDS,
    DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
    DATAFLOW_OPERATION_CROSS_HANDLER_HANDOFF,
    DATAFLOW_OPERATION_PIPELINE,
    DATAFLOW_OPERATION_RANGE_COARSENING,
    DATAFLOW_OPERATION_RESHARED_TRANSPORT,
    DATAFLOW_OPERATION_TENSOR_LAYOUT,
    DATAFLOW_PIPELINE_PLAN_IMPLEMENTATION,
    DATAFLOW_PIPELINE_RESIDENT_IMPLEMENTATION,
    DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION,
    DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION,
    DATAFLOW_TRANSPORT_HBM_IMPLEMENTATION,
    DATAFLOW_TRANSPORT_STREAMED_IMPLEMENTATION,
    DataflowOperationRequest,
)
from .abi_schema import DATAFLOW_CLUSTER_LOAD_BALANCING_IMPLEMENTATION
from .handoff_planning import (
    DATAFLOW_CROSS_HANDLER_HANDOFF_DISABLED_IMPLEMENTATION,
    DATAFLOW_CROSS_HANDLER_HANDOFF_LOWERING_IMPLEMENTATION,
)
from .reshared_transport import (
    DATAFLOW_RESHARED_ALL_GATHER_LOWERING_IMPLEMENTATION,
    DATAFLOW_RESHARED_HBM_LOWERING_IMPLEMENTATION,
    DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION,
    DATAFLOW_RESHARED_STREAMED_PULL_IMPLEMENTATION,
)
from .physical_contract import (
    DATAFLOW_OPERATOR_PHYSICAL_IMPLEMENTATION,
    DATAFLOW_TERMINAL_SIDE_EFFECT_IMPLEMENTATION,
)
from .specialization import DATAFLOW_SPECIALIZATION_CAPTURE_IMPLEMENTATION
from .tensor_layout import DATAFLOW_TENSOR_ARGUMENT_LAYOUT_IMPLEMENTATION


DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION = 13
_DATAFLOW_EXECUTION_PLANNER_IMPLEMENTATION = "dataflow.execution.constraint.v2"


class DataflowImplementationState(str, Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    RETIRED = "retired"


@dataclass(frozen=True)
class DataflowImplementationSpec:
    implementation_id: str
    domain: str
    state: DataflowImplementationState
    owner: str
    legality_contract: str
    benchmark_evidence: tuple[str, ...]
    default_enabled: bool
    removal_date: str | None = None
    contract_kinds: tuple[str, ...] = ()
    contract_schema_versions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, DataflowImplementationState):
            raise TypeError(f"Dataflow implementation state must be DataflowImplementationState, got {self.state!r}")
        if not isinstance(self.default_enabled, bool):
            raise TypeError(f"Dataflow implementation default_enabled must be bool, got {self.default_enabled!r}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.:-]*", self.implementation_id):
            raise ValueError(f"Dataflow implementation id must be canonical lower-case text, got {self.implementation_id!r}")
        if "mla" in self.implementation_id:
            raise ValueError(f"Dataflow implementation ids cannot contain operator identities: {self.implementation_id!r}")
        if not self.domain or not self.owner or not self.legality_contract:
            raise ValueError(f"Dataflow implementation {self.implementation_id!r} requires domain, owner, and legality_contract")
        if not self.benchmark_evidence or any(not item for item in self.benchmark_evidence):
            raise ValueError(f"Dataflow implementation {self.implementation_id!r} requires evidence")
        kinds = tuple(self.contract_kinds)
        versions = tuple(self.contract_schema_versions)
        if len(kinds) != len(set(kinds)) or any(kind not in DATAFLOW_OPERATION_CONTRACT_KINDS for kind in kinds):
            raise ValueError(f"Dataflow implementation {self.implementation_id!r} has invalid contract kinds {kinds!r}")
        if len(versions) != len(set(versions)) or any(
            isinstance(version, bool) or not isinstance(version, int) or version <= 0 for version in versions
        ):
            raise ValueError(f"Dataflow implementation {self.implementation_id!r} has invalid contract schema versions {versions!r}")
        if bool(kinds) != bool(versions):
            raise ValueError(f"Dataflow implementation {self.implementation_id!r} must declare contract kinds and schema versions together")
        object.__setattr__(self, "contract_kinds", tuple(sorted(kinds)))
        object.__setattr__(self, "contract_schema_versions", tuple(sorted(versions)))
        if self.state is DataflowImplementationState.STABLE:
            if self.removal_date is not None:
                raise ValueError(f"stable implementation {self.implementation_id!r} cannot have a removal date")
            return
        if self.default_enabled:
            raise ValueError(f"non-stable implementation {self.implementation_id!r} must default off")
        if self.removal_date is None:
            raise ValueError(f"non-stable implementation {self.implementation_id!r} requires a removal date")
        if not isinstance(self.removal_date, str):
            raise TypeError(f"removal date for {self.implementation_id!r} must be an ISO date string")
        try:
            date.fromisoformat(self.removal_date)
        except ValueError as err:
            raise ValueError(f"invalid removal date for {self.implementation_id!r}: {self.removal_date!r}") from err

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation_id": self.implementation_id,
            "domain": self.domain,
            "state": self.state.value,
            "owner": self.owner,
            "legality_contract": self.legality_contract,
            "benchmark_evidence": list(self.benchmark_evidence),
            "default_enabled": self.default_enabled,
            "removal_date": self.removal_date,
            "contract_kinds": list(self.contract_kinds),
            "contract_schema_versions": list(self.contract_schema_versions),
        }


class DataflowImplementationRegistry:
    def __init__(self, specs: Iterable[DataflowImplementationSpec]):
        ordered = tuple(sorted(specs, key=lambda item: item.implementation_id))
        ids = tuple(item.implementation_id for item in ordered)
        if len(ids) != len(set(ids)):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            raise ValueError(f"duplicate Dataflow implementation ids: {duplicates!r}")
        self._specs = ordered
        self._by_id = {item.implementation_id: item for item in ordered}
        payload = {
            "schema_version": DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION,
            "implementations": [item.to_dict() for item in ordered],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def specs(self) -> tuple[DataflowImplementationSpec, ...]:
        return self._specs

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def require_selectable(
        self,
        implementation_id: str,
        *,
        selected_explicitly: bool,
        today: date | None = None,
    ) -> DataflowImplementationSpec:
        try:
            spec = self._by_id[implementation_id]
        except KeyError as err:
            raise ValueError(f"unregistered Dataflow implementation {implementation_id!r}") from err
        if spec.state is DataflowImplementationState.RETIRED:
            raise ValueError(f"retired Dataflow implementation {implementation_id!r} cannot be selected")
        if spec.state is DataflowImplementationState.EXPERIMENTAL:
            if not selected_explicitly:
                raise ValueError(f"experimental Dataflow implementation {implementation_id!r} requires an explicit typed selection")
            assert spec.removal_date is not None
            if date.fromisoformat(spec.removal_date) < (today or date.today()):
                raise ValueError(f"experimental Dataflow implementation {implementation_id!r} expired on {spec.removal_date}")
        return spec

    def require_contract_compatible(
        self,
        implementation_id: str,
        request: DataflowOperationRequest,
        *,
        selected_explicitly: bool,
        today: date | None = None,
    ) -> DataflowImplementationSpec:
        if not isinstance(request, DataflowOperationRequest):
            raise TypeError("Dataflow implementation compatibility requires a typed operation request")
        spec = self.require_selectable(
            implementation_id,
            selected_explicitly=selected_explicitly,
            today=today,
        )
        if request.KIND not in spec.contract_kinds:
            raise ValueError(f"Dataflow implementation {implementation_id!r} does not implement contract kind {request.KIND!r}")
        if request.schema_version not in spec.contract_schema_versions:
            raise ValueError(
                f"Dataflow implementation {implementation_id!r} does not support {request.KIND!r} schema version {request.schema_version}"
            )
        return spec

    def to_dict(self, *, include_catalog: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION,
            "fingerprint": self.fingerprint,
            "implementation_count": len(self._specs),
        }
        if include_catalog:
            result["implementations"] = [item.to_dict() for item in self._specs]
        return result


_REGISTERED_SPECS: dict[str, DataflowImplementationSpec] = {}


def register_dataflow_implementation(
    spec: DataflowImplementationSpec,
) -> DataflowImplementationSpec:
    previous = _REGISTERED_SPECS.get(spec.implementation_id)
    if previous is not None and previous != spec:
        raise ValueError(f"conflicting Dataflow implementation registration for {spec.implementation_id!r}")
    _REGISTERED_SPECS[spec.implementation_id] = spec
    return spec


def dataflow_implementation_registry() -> DataflowImplementationRegistry:
    return DataflowImplementationRegistry(_REGISTERED_SPECS.values())


def scheduler_option_implementation_id(option_name: str) -> str:
    return f"scheduler.option.{option_name}"


_STABLE_SCHEDULER_OPTIONS = frozenset(
    {
        "chunk_swap_search",
        "direct_leaf_acc",
        "level0_queue_order",
        "level_bucket_tree",
        "ready_time_tree",
        "schedule_pic",
        "schedule_pic_dir",
    }
)


def register_scheduler_option_implementation(
    option_name: str,
    *,
    category: str,
) -> DataflowImplementationSpec:
    stable = option_name in _STABLE_SCHEDULER_OPTIONS
    return register_dataflow_implementation(
        DataflowImplementationSpec(
            implementation_id=scheduler_option_implementation_id(option_name),
            domain=f"scheduler.{category}",
            state=(DataflowImplementationState.STABLE if stable else DataflowImplementationState.EXPERIMENTAL),
            owner="SCH",
            legality_contract=("typed scheduler config plus topology, plan, and replay validation"),
            benchmark_evidence=("testing/python/dataflow/test_dataflow_scheduler.py",),
            default_enabled=False,
            removal_date=None if stable else "2026-12-31",
        )
    )


def stable(
    implementation_id: str,
    domain: str,
    owner: str,
    legality_contract: str,
    evidence: str,
    *,
    default_enabled: bool = True,
    contract_kinds: tuple[str, ...] = (),
) -> None:
    register_dataflow_implementation(
        DataflowImplementationSpec(
            implementation_id=implementation_id,
            domain=domain,
            state=DataflowImplementationState.STABLE,
            owner=owner,
            legality_contract=legality_contract,
            benchmark_evidence=(evidence,),
            default_enabled=default_enabled,
            contract_kinds=contract_kinds,
            contract_schema_versions=((DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,) if contract_kinds else ()),
        )
    )


for _policy in ("round_robin", "cluster_local", "stage_graph"):
    stable(
        _policy,
        "scheduler.policy",
        "SCH",
        "registered policy/reduce-strategy compatibility and topology validation",
        "testing/python/dataflow/test_dataflow_scheduler_config.py",
    )

for _implementation in (
    "cuda.mma.sync",
    "cuda.scalar.sync",
    "cuda.tcgen05.sync",
    "cuda.wgmma.async",
    "cuda.wgmma.rs.shared_a",
):
    stable(
        _implementation,
        "tilelang.lowering.gemm",
        "TL-L",
        "structured GEMM request and target capability registry legality",
        "testing/python/dataflow/test_dataflow_gemm_lowering.py",
    )

for _implementation in (
    "common.simt",
    DATAFLOW_PIPELINE_RESIDENT_IMPLEMENTATION,
    "cuda.cp_async",
    "cuda.tma.load.full",
    "cuda.tma.load.tail_oob",
):
    stable(
        _implementation,
        "tilelang.lowering.transfer",
        "TL-L",
        "typed transfer contract, synchronization ownership, and target legality",
        "testing/python/dataflow/test_dataflow_primfunc_lowering.py",
    )

for _implementation in ("software_pipeline", "synchronous", "warp_specialized"):
    stable(
        _implementation,
        "tilelang.lowering.pipeline",
        "TL-L",
        "pipeline dependency, synchronization, and target capability validation",
        "testing/python/dataflow/test_dataflow_pipeline.py",
        contract_kinds=(DATAFLOW_OPERATION_PIPELINE,),
    )

stable(
    DATAFLOW_PIPELINE_PLAN_IMPLEMENTATION,
    "tilelang.pipeline.contract",
    "TL-L",
    "typed dependency graph, stage budget, lifetime, and resource validation",
    "testing/python/dataflow/test_dataflow_pipeline.py",
    contract_kinds=(DATAFLOW_OPERATION_PIPELINE,),
)

for _implementation, _kind, _domain, _owner, _legality, _evidence in (
    (
        DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION,
        DATAFLOW_OPERATION_RANGE_COARSENING,
        "dataflow.range",
        "SCH",
        "typed range extent and stage-graph task-range validation",
        "testing/python/dataflow/test_dataflow_reshared_graph.py",
    ),
    (
        DATAFLOW_TRANSPORT_HBM_IMPLEMENTATION,
        DATAFLOW_OPERATION_RESHARED_TRANSPORT,
        "dataflow.transport",
        "COMM",
        "typed HBM family constraint and scheduler communication legality",
        "testing/python/dataflow/test_dataflow_reshared_graph.py",
    ),
    (
        DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION,
        DATAFLOW_OPERATION_RESHARED_TRANSPORT,
        "dataflow.transport",
        "COMM",
        "typed all-gather family constraint and cluster topology legality",
        "testing/python/dataflow/test_dataflow_reshared_graph.py",
    ),
    (
        DATAFLOW_TRANSPORT_STREAMED_IMPLEMENTATION,
        DATAFLOW_OPERATION_RESHARED_TRANSPORT,
        "dataflow.transport",
        "COMM",
        "typed streamed family constraint and cluster topology legality",
        "testing/python/dataflow/test_dataflow_reshared_graph.py",
    ),
    (
        DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
        DATAFLOW_OPERATION_CROSS_HANDLER_HANDOFF,
        "dataflow.handoff",
        "SCH",
        "typed stage identity, queue ordering, and terminal-producer validation",
        "testing/python/dataflow/test_dataflow_reshared_graph.py",
    ),
    (
        DATAFLOW_LAYOUT_LINEAR_IMPLEMENTATION,
        DATAFLOW_OPERATION_TENSOR_LAYOUT,
        "tilelang.lowering.layout",
        "TL-L",
        "typed field rank and linear layout validation",
        "testing/python/dataflow/test_dataflow_primfunc_lowering.py",
    ),
    (
        DATAFLOW_LAYOUT_MATRIX_SWIZZLE_IMPLEMENTATION,
        DATAFLOW_OPERATION_TENSOR_LAYOUT,
        "tilelang.lowering.layout",
        "TL-L",
        "typed field rank, major axis, and matrix-layout lowering validation",
        "testing/python/dataflow/test_dataflow_primfunc_lowering.py",
    ),
):
    stable(
        _implementation,
        _domain,
        _owner,
        _legality,
        _evidence,
        contract_kinds=(_kind,),
    )

for _implementation, _legality in (
    (
        DATAFLOW_RESHARED_HBM_LOWERING_IMPLEMENTATION,
        "typed HBM plan, scheduler communication, and slot lifetime validation",
    ),
    (
        DATAFLOW_RESHARED_ALL_GATHER_LOWERING_IMPLEMENTATION,
        "typed cluster all-gather plan, barrier, and slot lifetime validation",
    ),
    (
        DATAFLOW_RESHARED_STREAMED_PULL_IMPLEMENTATION,
        "typed source-rank mapping, payload partition, receive credit, and cluster lifetime validation",
    ),
    (
        DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION,
        "typed rank-balanced producer push, remote publish, receive-slot credit, and cluster lifetime validation",
    ),
):
    stable(
        _implementation,
        "dataflow.transport.lowering",
        "COMM",
        _legality,
        "testing/python/dataflow/test_dataflow_reshared_transport.py",
    )

stable(
    _DATAFLOW_EXECUTION_PLANNER_IMPLEMENTATION,
    "dataflow.execution.planning",
    "SCH",
    "typed logical stage, topology, target capability, and resource constraints",
    "testing/python/dataflow/test_dataflow_execution_planning.py",
)

for _implementation, _domain, _owner, _legality, _evidence in (
    (
        DATAFLOW_CLUSTER_LOAD_BALANCING_IMPLEMENTATION,
        "dataflow.launch.cluster",
        "TL-L",
        "typed launch package selects CUDA cluster load-balancing residency policy",
        "testing/python/dataflow/test_dataflow_launch.py",
    ),
    (
        DATAFLOW_SPECIALIZATION_CAPTURE_IMPLEMENTATION,
        "dataflow.specialization",
        "DF-L",
        "source-referenced closure values with strict canonical serialization",
        "testing/python/dataflow/test_dataflow_execution_planning.py",
    ),
    (
        DATAFLOW_OPERATOR_PHYSICAL_IMPLEMENTATION,
        "dataflow.primfunc.contract",
        "DF-L",
        "typed input-slot, output-slot, and return warp-group semantics",
        "testing/python/dataflow/test_dataflow_operation_contracts.py",
    ),
    (
        DATAFLOW_TERMINAL_SIDE_EFFECT_IMPLEMENTATION,
        "dataflow.terminal",
        "DF-L",
        "terminal stage ordering, no-value body, and absent output-slot validation",
        "testing/python/dataflow/test_dataflow_primfunc_linking.py",
    ),
    (
        DATAFLOW_TENSOR_ARGUMENT_LAYOUT_IMPLEMENTATION,
        "dataflow.tensor.layout",
        "DF-L",
        "logical-to-physical axis mapping plus runtime shape, dtype, device, stride, and fingerprint validation",
        "testing/python/dataflow/test_dataflow_operation_contracts.py",
    ),
):
    stable(
        _implementation,
        _domain,
        _owner,
        _legality,
        _evidence,
    )

for _implementation, _legality in (
    (
        DATAFLOW_CROSS_HANDLER_HANDOFF_LOWERING_IMPLEMENTATION,
        "typed consumer pipeline transfers, queue peer binding, arena placement, and wait/release lifetime validation",
    ),
    (
        DATAFLOW_CROSS_HANDLER_HANDOFF_DISABLED_IMPLEMENTATION,
        "typed handoff resource or consumer-pipeline fallback with no physical transfer",
    ),
):
    stable(
        _implementation,
        "dataflow.handoff.lowering",
        "DF-L",
        _legality,
        "testing/python/dataflow/test_dataflow_handoff_planning.py",
    )

for _dtype in ("float16", "float32"):
    stable(
        f"dataflow.precision.accumulator.v1:{_dtype}",
        "dataflow.precision",
        "NUM",
        "operator accumulator contract intersected with typed precision policy",
        "testing/python/dataflow/test_dataflow_precision_memory_policy.py",
    )

for _placement in ("shared", "scratch_backed", "hbm_direct_global"):
    stable(
        f"dataflow.memory.v1:{_placement}",
        "dataflow.memory",
        "MEM",
        "compiler-owned placement candidate and target resource validation",
        "testing/python/dataflow/test_dataflow_precision_memory_policy.py",
    )

stable(
    "semantic.direct_slot_seed_reduce",
    "dataflow.semantic",
    "DF-L",
    "typed semantic config and value-forward ownership validation",
    "testing/python/dataflow/test_dataflow_compile.py",
    default_enabled=False,
)
stable(
    "semantic.fast_math",
    "dataflow.semantic",
    "NUM",
    "typed semantic permission and compiler-owned CUDA flag materialization",
    "testing/python/dataflow/test_dataflow_execution_planning.py",
    default_enabled=False,
)
register_dataflow_implementation(
    DataflowImplementationSpec(
        implementation_id="semantic.skip_finalize_post_sync",
        domain="dataflow.semantic",
        state=DataflowImplementationState.EXPERIMENTAL,
        owner="DF-L",
        legality_contract="typed semantic config plus CTA-wide finalize synchronization tests",
        benchmark_evidence=("testing/python/dataflow/test_dataflow_primfunc_linking.py",),
        default_enabled=False,
        removal_date="2026-12-31",
    )
)

for _provider in ("empty", "scalar_u32", "tensor_u32"):
    register_dataflow_implementation(
        DataflowImplementationSpec(
            implementation_id=f"experimental.{_provider}",
            domain="dataflow.debug_handler",
            state=DataflowImplementationState.EXPERIMENTAL,
            owner="QA",
            legality_contract="explicit debug mode and provider-specific ABI validation",
            benchmark_evidence=("testing/python/dataflow/test_dataflow_experimental_debug_handlers.py",),
            default_enabled=False,
            removal_date="2027-01-31",
        )
    )


__all__ = [
    "DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION",
    "DataflowImplementationRegistry",
    "DataflowImplementationSpec",
    "DataflowImplementationState",
    "dataflow_implementation_registry",
    "register_dataflow_implementation",
    "register_scheduler_option_implementation",
    "scheduler_option_implementation_id",
]
