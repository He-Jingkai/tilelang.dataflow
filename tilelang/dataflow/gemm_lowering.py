"""Capability-driven GEMM specialization for Dataflow PrimFunc handlers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from collections.abc import Callable

from tvm import tir
from tvm.target import Target

from tilelang import _ffi_api
from tilelang.utils.target_capabilities import TargetCapabilitySnapshot

from .implementation_registry import dataflow_implementation_registry

GEMM_LOWERING_REGISTRY_VERSION = 2


class UnsupportedTargetCapabilityError(RuntimeError):
    """Raised when no registered lowering can implement a GEMM request."""


@dataclass(frozen=True)
class GemmLoweringRequest:
    """Target-independent GEMM semantics used to select a legal implementation."""

    operation_id: str
    requested_primitive: str
    m: int | None
    n: int | None
    k: int | None
    a_dtype: str | None
    b_dtype: str | None
    c_dtype: str | None
    a_scope: str | None
    b_scope: str | None
    c_scope: str | None
    thread_count: int
    allow_padding: bool = False
    padding_value: int | float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "requested_primitive": self.requested_primitive,
            "shape": [self.m, self.n, self.k],
            "operand_dtypes": [self.a_dtype, self.b_dtype, self.c_dtype],
            "operand_scopes": [self.a_scope, self.b_scope, self.c_scope],
            "thread_count": self.thread_count,
            "allow_padding": self.allow_padding,
            "padding_value": self.padding_value,
        }


GemmLegalityPredicate = Callable[[GemmLoweringRequest], bool]


@dataclass(frozen=True)
class GemmLoweringImplementation:
    """One registry entry with feature requirements and semantic legality."""

    implementation_id: str
    primitive: str
    requires: frozenset[str]
    priority: int
    legal: GemmLegalityPredicate
    synchronous: bool
    supported_thread_counts: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_id": self.implementation_id,
            "primitive": self.primitive,
            "requires": sorted(self.requires),
            "priority": self.priority,
            "synchronous": self.synchronous,
            "supported_thread_counts": (None if self.supported_thread_counts is None else list(self.supported_thread_counts)),
        }


@dataclass(frozen=True)
class GemmLoweringResolution:
    """Stable selection metadata attached to a lowered Dataflow handler."""

    request: GemmLoweringRequest
    implementation_id: str
    primitive: str
    requires: tuple[str, ...]
    used_fallback: bool
    target_fingerprint: str
    registry_version: int = GEMM_LOWERING_REGISTRY_VERSION
    supported: bool = True
    synchronous: bool = True
    physical_shape: tuple[int, int, int] | None = None
    requires_padding: bool = False
    requires_materialization: bool = False
    temporary_requirements: tuple[dict[str, Any], ...] = ()
    additional_shared_memory_bytes: int = 0
    additional_fragment_bytes: int = 0
    estimated_resource_bytes: int = 0
    selection_reason: str = "legacy Dataflow registry selection"
    rejected_candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_version": self.registry_version,
            "request": self.request.to_dict(),
            "implementation_id": self.implementation_id,
            "primitive": self.primitive,
            "requires": list(self.requires),
            "used_fallback": self.used_fallback,
            "target_fingerprint": self.target_fingerprint,
            "supported": self.supported,
            "synchronous": self.synchronous,
            "physical_shape": (list(self.physical_shape) if self.physical_shape is not None else None),
            "requires_padding": self.requires_padding,
            "requires_materialization": self.requires_materialization,
            "temporary_requirements": [dict(item) for item in self.temporary_requirements],
            "additional_shared_memory_bytes": self.additional_shared_memory_bytes,
            "additional_fragment_bytes": self.additional_fragment_bytes,
            "estimated_resource_bytes": self.estimated_resource_bytes,
            "selection_reason": self.selection_reason,
            "rejected_candidates": list(self.rejected_candidates),
        }


class GemmLoweringRegistry:
    """Deterministic, capability-based GEMM implementation registry."""

    def __init__(self, implementations: tuple[GemmLoweringImplementation, ...] = ()):
        self._implementations = implementations

    @property
    def implementations(self) -> tuple[GemmLoweringImplementation, ...]:
        return self._implementations

    def register(
        self,
        *,
        implementation_id: str,
        primitive: str,
        requires: frozenset[str],
        priority: int,
        legal: GemmLegalityPredicate,
        synchronous: bool,
        supported_thread_counts: tuple[int, ...] | None = None,
    ) -> GemmLoweringRegistry:
        dataflow_implementation_registry().require_selectable(
            implementation_id,
            selected_explicitly=True,
        )
        if any(entry.implementation_id == implementation_id for entry in self._implementations):
            raise ValueError(f"duplicate Dataflow GEMM implementation id {implementation_id!r}")
        if supported_thread_counts is not None and any(int(count) <= 0 for count in supported_thread_counts):
            raise ValueError(f"Dataflow GEMM supported thread counts must be positive, got {supported_thread_counts!r}")
        entry = GemmLoweringImplementation(
            implementation_id=implementation_id,
            primitive=primitive,
            requires=frozenset(requires),
            priority=int(priority),
            legal=legal,
            synchronous=bool(synchronous),
            supported_thread_counts=(
                None if supported_thread_counts is None else tuple(sorted({int(count) for count in supported_thread_counts}))
            ),
        )
        return GemmLoweringRegistry(self._implementations + (entry,))

    def resolve(
        self,
        request: GemmLoweringRequest,
        target_capabilities: TargetCapabilitySnapshot,
    ) -> GemmLoweringResolution:
        available = target_feature_set(target_capabilities)
        legal_entries = tuple(
            entry
            for entry in self._implementations
            if entry.requires <= available
            and (entry.supported_thread_counts is None or request.thread_count in entry.supported_thread_counts)
            and entry.legal(request)
        )
        if not legal_entries:
            candidates = ", ".join(repr(entry.to_dict()) for entry in self._implementations)
            raise UnsupportedTargetCapabilityError(
                "no legal Dataflow GEMM lowering for "
                f"request={request.to_dict()!r}, target={target_capabilities.target!r}, "
                f"target_fingerprint={target_capabilities.fingerprint}, "
                f"available_capabilities={sorted(available)!r}, candidates=[{candidates}]"
            )
        selected = min(
            legal_entries,
            key=lambda entry: (-entry.priority, entry.implementation_id),
        )
        requested_impl = requested_implementation_id(request.requested_primitive)
        return GemmLoweringResolution(
            request=request,
            implementation_id=selected.implementation_id,
            primitive=selected.primitive,
            requires=tuple(sorted(selected.requires)),
            used_fallback=(requested_impl is not None and selected.implementation_id != requested_impl),
            target_fingerprint=target_capabilities.fingerprint,
            synchronous=selected.synchronous,
            physical_shape=(request.m, request.n, request.k) if known_shape(request) else None,
            selection_reason="selected by the capability-only Dataflow registry",
        )

    def recommend_thread_count(
        self,
        request: GemmLoweringRequest,
        target_capabilities: TargetCapabilitySnapshot,
        *,
        accelerated_only: bool = False,
    ) -> int:
        """Choose a legal handler size from declarative implementation constraints."""

        available = target_feature_set(target_capabilities)
        ordered = sorted(
            self._implementations,
            key=lambda entry: (-entry.priority, entry.implementation_id),
        )
        for entry in ordered:
            if not entry.requires <= available:
                continue
            if accelerated_only and entry.implementation_id == "cuda.scalar.sync":
                continue
            candidates = (
                (request.thread_count,)
                if entry.supported_thread_counts is None
                else tuple(count for count in reversed(entry.supported_thread_counts) if count <= request.thread_count)
            )
            for count in candidates:
                if entry.legal(replace(request, thread_count=count)):
                    return count
        return request.thread_count


def target_feature_set(snapshot: TargetCapabilitySnapshot) -> frozenset[str]:
    """Return registry feature names from one immutable target snapshot."""

    features = {snapshot.backend}
    if snapshot.supports_tensor_core_mma:
        features.add("tensor_core_mma")
    if snapshot.supports_tma:
        features.add("tma")
    if snapshot.supports_wgmma:
        features.add("wgmma")
    if snapshot.supports_tcgen05:
        features.add("tcgen05")
    if snapshot.supports_cluster_launch:
        features.add("cluster_launch")
    return frozenset(features)


def known_shape(request: GemmLoweringRequest) -> bool:
    return request.m is not None and request.n is not None and request.k is not None


def tcgen05_legal(request: GemmLoweringRequest) -> bool:
    return (
        request.requested_primitive in {"generic", "tcgen05"}
        and request.a_scope in {"shared", "shared.dyn", "shared.tmem"}
        and request.b_scope in {"shared", "shared.dyn"}
        and request.c_scope == "shared.tmem"
        and known_shape(request)
    )


def wgmma_legal(request: GemmLoweringRequest) -> bool:
    return (
        request.requested_primitive in {"generic", "wgmma"}
        and request.m is not None
        and request.m >= 64
        and request.b_scope in {"shared", "shared.dyn"}
        and request.c_scope == "local.fragment"
        and request.thread_count > 0
        and request.thread_count % 128 == 0
    )


def mma_legal(request: GemmLoweringRequest) -> bool:
    if not known_shape(request):
        return False
    assert request.m is not None and request.n is not None and request.k is not None
    return (
        request.requested_primitive in {"generic", "wgmma", "tcgen05"}
        and request.m >= 16
        and request.m % 16 == 0
        and request.n >= 8
        and request.n % 8 == 0
        and request.k > 0
        and request.c_scope == "local.fragment"
    )


def scalar_legal(request: GemmLoweringRequest) -> bool:
    return request.requested_primitive == "generic"


DEFAULT_GEMM_LOWERING_REGISTRY = (
    GemmLoweringRegistry()
    .register(
        implementation_id="cuda.tcgen05.sync",
        primitive="tcgen05",
        requires=frozenset(("cuda", "tcgen05")),
        priority=300,
        legal=tcgen05_legal,
        synchronous=True,
    )
    .register(
        implementation_id="cuda.wgmma.async",
        primitive="wgmma",
        requires=frozenset(("cuda", "wgmma")),
        priority=250,
        legal=wgmma_legal,
        synchronous=False,
    )
    .register(
        implementation_id="cuda.mma.sync",
        primitive="gemm",
        requires=frozenset(("cuda", "tensor_core_mma")),
        priority=200,
        legal=mma_legal,
        synchronous=True,
        supported_thread_counts=(32, 64, 128, 256),
    )
    .register(
        implementation_id="cuda.scalar.sync",
        primitive="gemm",
        requires=frozenset(("cuda",)),
        priority=0,
        legal=scalar_legal,
        synchronous=True,
    )
)


def specialize_primfunc_gemm_lowerings(
    prim_func: tir.PrimFunc,
    target_capabilities: TargetCapabilitySnapshot,
    *,
    handler_id: int,
    operator_name: str,
    thread_count: int,
    registry: GemmLoweringRegistry = DEFAULT_GEMM_LOWERING_REGISTRY,
) -> tuple[tir.PrimFunc, tuple[GemmLoweringResolution, ...]]:
    """Resolve GEMM calls through the common TileLang registry."""

    resolutions: list[GemmLoweringResolution] = []
    operation_index = 0
    common_plans = iter(
        _ffi_api.ResolvePrimFuncGemmLoweringPlans(
            prim_func,
            Target(target_capabilities.target),
            int(thread_count),
            int(target_capabilities.max_dynamic_shared_memory or -1),
        )
    )

    def post_order(node: Any) -> Any:
        nonlocal operation_index
        if not isinstance(node, tir.Call):
            return node
        primitive = gemm_primitive(node)
        if primitive is None:
            return node
        request = request_from_call(
            node,
            operation_id=f"handler:{handler_id}:{operator_name}:gemm:{operation_index}",
            requested_primitive=primitive,
            thread_count=thread_count,
        )
        operation_index += 1
        try:
            common_plan = next(common_plans)
        except StopIteration as err:
            raise RuntimeError("common GEMM planner returned fewer plans than GEMM calls") from err
        resolution = resolution_from_common_plan(
            request,
            common_plan,
            target_capabilities,
        )
        if not resolution.supported:
            raise UnsupportedTargetCapabilityError(
                f"no legal common GEMM lowering for request={request.to_dict()!r}, plan={resolution.to_dict()!r}"
            )
        resolutions.append(resolution)
        if resolution.primitive == primitive or (primitive == "generic" and resolution.primitive in {"gemm", "wgmma", "tcgen05"}):
            return node
        if resolution.primitive != "gemm":
            raise UnsupportedTargetCapabilityError(
                "Dataflow GEMM registry selected an implementation that requires an unavailable "
                f"IR rewrite: resolution={resolution.to_dict()!r}"
            )
        return tir.Call(
            node.dtype,
            tir.op.Op.get("tl.tileop.gemm"),
            list(node.args),
            annotations=None,
            span=getattr(node, "span", None),
        )

    body = tir.stmt_functor.ir_transform(prim_func.body, None, post_order)
    try:
        extra_plan = next(common_plans)
    except StopIteration:
        extra_plan = None
    if extra_plan is not None:
        raise RuntimeError("common GEMM planner returned more plans than GEMM calls")
    if not target_capabilities.supports_wgmma:
        body = tir.stmt_functor.ir_transform(body, None, remove_wgmma_wait)
    return prim_func.with_body(body), tuple(resolutions)


def remove_wgmma_wait(node: Any) -> Any:
    if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
        return node
    op_name = getattr(node.value.op, "name", str(node.value.op))
    if op_name != "tl.wait_wgmma":
        return node
    return tir.Evaluate(tir.const(0, "int32"), span=getattr(node, "span", None))


def gemm_primitive(call: tir.Call) -> str | None:
    op_name = getattr(call.op, "name", str(call.op))
    return {
        "tl.tileop.gemm": "generic",
        "tl.tileop.wgmma_gemm": "wgmma",
        "tl.tileop.tcgen05_gemm": "tcgen05",
    }.get(op_name)


def request_from_call(
    call: tir.Call,
    *,
    operation_id: str,
    requested_primitive: str,
    thread_count: int,
) -> GemmLoweringRequest:
    a_dtype, a_scope = region_dtype_scope(call.args[0])
    b_dtype, b_scope = region_dtype_scope(call.args[1])
    c_dtype, c_scope = region_dtype_scope(call.args[2])
    logical_m = constant_int(call.args[5])
    logical_n = constant_int(call.args[6])
    logical_k = constant_int(call.args[7])
    allow_padding = False
    padding_value: int | float = 0
    for arg in call.args[19:]:
        if not isinstance(arg, tir.Call) or getattr(arg.op, "name", None) != "tl.gemm_contract":
            continue
        logical_m = constant_int(arg.args[0])
        logical_n = constant_int(arg.args[1])
        logical_k = constant_int(arg.args[2])
        padding_value = constant_number(arg.args[3])
        allow_padding = bool(constant_int(arg.args[4]))
    return GemmLoweringRequest(
        operation_id=operation_id,
        requested_primitive=requested_primitive,
        m=logical_m,
        n=logical_n,
        k=logical_k,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        a_scope=a_scope,
        b_scope=b_scope,
        c_scope=c_scope,
        thread_count=int(thread_count),
        allow_padding=allow_padding,
        padding_value=padding_value,
    )


def resolution_from_common_plan(
    request: GemmLoweringRequest,
    plan: Any,
    target_capabilities: TargetCapabilitySnapshot,
) -> GemmLoweringResolution:
    implementation_id = str(plan.implementation_id)
    primitive = {
        "cuda.tcgen05.sync": "tcgen05",
        "cuda.wgmma.async": "wgmma",
        "cuda.mma.sync": "gemm",
        "cuda.scalar.sync": "gemm",
    }.get(implementation_id, "gemm")
    requires = {
        "cuda.tcgen05.sync": ("cuda", "tcgen05"),
        "cuda.wgmma.async": ("cuda", "wgmma"),
        "cuda.mma.sync": ("cuda", "tensor_core_mma"),
        "cuda.scalar.sync": ("cuda",),
    }.get(implementation_id, (target_capabilities.backend,))
    requested_impl = requested_implementation_id(request.requested_primitive)
    temporary_requirements = tuple(
        {
            "buffer_role": str(item.buffer_role),
            "storage_scope": str(item.storage_scope),
            "logical_shape": [int(value) for value in item.logical_shape],
            "physical_shape": [int(value) for value in item.physical_shape],
            "neutral_value": constant_number(item.neutral_value),
            "initialization_required": bool(item.initialization_required),
            "lifetime_start": str(item.lifetime_start),
            "lifetime_end": str(item.lifetime_end),
            "estimated_bytes": int(item.estimated_bytes),
            "additional_bytes": int(item.additional_bytes),
        }
        for item in plan.temporary_requirements
    )
    return GemmLoweringResolution(
        request=request,
        implementation_id=implementation_id,
        primitive=primitive,
        requires=tuple(sorted(requires)),
        used_fallback=(requested_impl is not None and implementation_id != requested_impl),
        target_fingerprint=target_capabilities.fingerprint,
        supported=bool(plan.supported),
        synchronous=bool(plan.synchronous),
        physical_shape=tuple(int(value) for value in plan.physical_shape),
        requires_padding=bool(plan.requires_padding),
        requires_materialization=bool(plan.requires_materialization),
        temporary_requirements=temporary_requirements,
        additional_shared_memory_bytes=int(plan.additional_shared_memory_bytes),
        additional_fragment_bytes=int(plan.additional_fragment_bytes),
        estimated_resource_bytes=int(plan.estimated_resource_bytes),
        selection_reason=str(plan.selection_reason),
        rejected_candidates=tuple(str(value) for value in plan.rejected_candidates),
    )


def region_dtype_scope(region: Any) -> tuple[str | None, str | None]:
    if not isinstance(region, tir.Call) or not region.args:
        return None, None
    load = region.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None, None
    return str(load.buffer.dtype), str(load.buffer.scope())


def constant_int(value: Any) -> int | None:
    if isinstance(value, tir.IntImm):
        return int(value.value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def constant_number(value: Any) -> int | float:
    if isinstance(value, (tir.IntImm, tir.FloatImm)):
        return value.value
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError) as err:
            raise TypeError(f"GEMM padding value must be a compile-time scalar, got {value!r}") from err


def requested_implementation_id(primitive: str) -> str | None:
    return {
        "wgmma": "cuda.wgmma.async",
        "tcgen05": "cuda.tcgen05.sync",
    }.get(primitive)


__all__ = [
    "DEFAULT_GEMM_LOWERING_REGISTRY",
    "GEMM_LOWERING_REGISTRY_VERSION",
    "GemmLoweringImplementation",
    "GemmLoweringRegistry",
    "GemmLoweringRequest",
    "GemmLoweringResolution",
    "UnsupportedTargetCapabilityError",
    "specialize_primfunc_gemm_lowerings",
    "target_feature_set",
]
