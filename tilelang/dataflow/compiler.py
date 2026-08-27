"""Compile entry for Dataflow programs.

The production path lowers Dataflow operator bodies to PrimFuncs, lowers them
through TileLang CUDA codegen, links handler adapters into the generated
wrapper, and launches the resulting queue-driven CUDA kernel. Inspection and
debug artifacts remain available with an explicit artifact contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from functools import partial, update_wrapper
import inspect
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .layout_validation import DataflowLayoutValidationReport

from tilelang import _ffi_api
from tilelang.utils.target_capabilities import (
    TargetCapabilitySnapshot,
    resolve_target_capabilities,
    target_capability_override,
)

from .architecture_contract import DATAFLOW_LOWERING_BOUNDARY_CONTRACT
from .compile_config import (
    DATAFLOW_COMPILE_MODE_DEBUG,
    DATAFLOW_COMPILE_MODE_EXECUTABLE,
    DATAFLOW_COMPILE_MODE_INSPECT,
    DataflowCompileConfig,
    canonical_fingerprint,
    capture_compile_environment,
    use_target_capabilities,
)
from .decision_artifact import (
    DataflowDecisionArtifact,
    collect_dataflow_decision_artifact,
)
from .cuda_contract import DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES
from .dtype_registry import require_dataflow_dtype
from .executor import DataflowExecutableKernel, DataflowExecutionResult, DataflowPersistentExecutable
from .execution_planning import (
    DataflowExecutionPlan,
    capture_execution_plans,
)
from .gemm_lowering import (
    GEMM_LOWERING_REGISTRY_VERSION,
    UnsupportedTargetCapabilityError,
)
from .handler import (
    PRIMFUNC_HANDLER_LOWERING,
    normalize_handler_lowering,
    validate_handler_lowering,
)
from .handler_abi import DataflowHandlerABI, build_handler_abi
from .handler_identity import DataflowHandlerVariantKey, build_handler_registry
from .handoff_planning import DataflowCrossHandlerHandoffPlan
from .iter_range_buckets import (
    normalize_iter_range_bucket_size,
    normalize_iter_range_buckets,
    normalize_iter_range_exact_lengths,
)
from .joint_schedule import (
    DataflowCommSlotMode,
    DataflowCommStorageKind,
    DataflowScheduleEventKind,
    DataflowTransportKind,
)
from .implementation_registry import (
    DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION,
    dataflow_implementation_registry,
)
from .launch import DATAFLOW_SHARED_ALIGNMENT, DataflowLaunchPackage, build_launch_package
from .memory_planner import (
    DATAFLOW_MEMORY_AUTO,
    DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL,
    DATAFLOW_MEMORY_PLANNER_VERSION,
    DATAFLOW_MEMORY_SCRATCH_BACKED,
    DATAFLOW_MEMORY_SHARED,
    DataflowMemoryCandidate,
    DataflowMemoryPlan,
    DataflowMemoryPlanningError,
    make_memory_candidate,
    resolve_memory_policy,
    select_memory_plan,
)
from .operation_contracts import (
    DATAFLOW_OPERATION_CONTRACT_FINGERPRINT,
    DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
)
from .physical_contract import operator_physical_contract
from .primfunc_linking import link_primfunc_handlers_for_wrapper
from .primfunc_lowering import (
    DataflowPrimFuncLoweringError,
    DataflowPrimFuncLoweringResult,
    lower_program_handlers_to_primfuncs,
)
from .precision import DataflowPrecisionPlan, resolve_program_precision
from .program import DataflowProgram
from .progress import DataflowProgressLogger, dataflow_progress_enabled
from .runtime import (
    DATAFLOW_SLOT_ALIGNMENT,
    DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
    DATAFLOW_SLOT_FLAG_SCRATCH_BACKED,
    PackedRuntimePlan,
    pack_instruction_plan,
)
from .scheduler import InstructionPlan, schedule
from .scheduler import (
    CommPlan,
    Instruction,
    DataflowCommKind,
    DataflowOpcode,
    DataflowValueForward,
    SlotPlan,
)
from .scheduler_auto_policy import (
    DATAFLOW_SCHEDULER_AUTO,
    select_scheduler_auto_policy,
)
from .scheduler_config import DataflowSchedulerConfig, resolve_scheduler_config
from .scheduler_policies import normalize_reduce_strategy, normalize_scheduler_policy
from .semantic_config import DataflowSemanticConfig, resolve_semantic_config
from .tensor_args import (
    DataflowTensorArgPlan,
    PackedRuntimeTensorArgs,
    collect_tensor_arg_plan,
    pack_tensor_args as _pack_tensor_args,
)
from .tma_descriptors import (
    build_tma_descriptor_handles,
    validate_tma_descriptor_runtime_tensors,
)
from .topology import GPUTopology
from .wrapper import (
    DataflowHandoffArenaSpec,
    DataflowWrapperSpec,
    build_wrapper_spec,
    generate_wrapper_source,
)


PRIMFUNC_DYNAMIC_SHARED_ALIGNMENT = DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES

DATAFLOW_ARTIFACT_PRODUCTION = "production"
DATAFLOW_ARTIFACT_INSPECTION = "inspection"
DATAFLOW_ARTIFACT_DEBUG = "debug"
_DATAFLOW_ARTIFACT_CONTRACTS = frozenset(
    (
        DATAFLOW_ARTIFACT_PRODUCTION,
        DATAFLOW_ARTIFACT_INSPECTION,
        DATAFLOW_ARTIFACT_DEBUG,
    )
)


def insert_cluster_destination_reuse_handoffs(
    plan: InstructionPlan,
) -> InstructionPlan:
    """Gate a later cluster push until the prior destination generation dies.

    Ordinary stage-graph all-gather plans reuse physical shared storage across
    logical tasks.  A receive mbarrier protects arrival, but cannot by itself
    stop a producer from starting the next generation while the consumer is
    still reading the previous one.  Insert the same point-to-point release
    contract used by joint communicate slots, derived solely from physical
    storage identity and queue lifetimes.
    """

    if plan.joint_execution_plan is not None:
        # Joint communicate-slot lowering derives releases from its explicit
        # interval model when physical slots are materialized.
        return plan

    slots_by_id = {slot.slot_id: slot for slot in plan.slots}

    def storage_key(slot_id: int) -> tuple[str, int] | None:
        slot = slots_by_id[slot_id]
        if slot.scratch_backed:
            return None
        if slot.shared_storage_id is None:
            return ("slot", slot.slot_id)
        return ("shared", slot.shared_storage_id)

    def transfer_key(comm: CommPlan) -> tuple[int, int, int, int, int]:
        return (
            comm.source_instruction_id,
            comm.target_instruction_id,
            comm.source_slot_id,
            comm.target_slot_id,
            comm.segment_id,
        )

    sends_by_key = {transfer_key(comm): comm for comm in plan.comms if comm.kind is DataflowCommKind.CLUSTER_SEND}
    existing_release_keys = {transfer_key(comm) for comm in plan.comms if comm.kind is DataflowCommKind.CLUSTER_RELEASE}
    queue_positions = {
        instruction.instruction_id: (cta_id, position)
        for cta_id, queue in plan.queues.items()
        for position, instruction in enumerate(queue)
    }
    receives_by_resource: dict[tuple[int, tuple[str, int]], list[tuple[int, CommPlan]]] = {}
    for comm in plan.comms:
        if comm.kind is not DataflowCommKind.CLUSTER_RECV:
            continue
        key = storage_key(comm.target_slot_id)
        position = queue_positions.get(comm.resolved_dispatch_instruction_id)
        if key is None or position is None:
            continue
        cta_id, queue_position = position
        if cta_id != comm.consumer_sm:
            continue
        receives_by_resource.setdefault((cta_id, key), []).append((queue_position, comm))

    gated_transfer_records: list[tuple[tuple[int, int, int], tuple[int, int, int, int, int], CommPlan]] = []
    gated_slot_ids: set[int] = set()
    for (consumer_cta, key), generations in receives_by_resource.items():
        generations.sort(key=lambda item: item[0])
        for (previous_position, previous), (next_position, subsequent) in zip(
            generations,
            generations[1:],
        ):
            subsequent_key = transfer_key(subsequent)
            if subsequent_key in existing_release_keys:
                continue
            send = sends_by_key.get(subsequent_key)
            if send is None:
                continue
            queue = plan.queue(consumer_cta)
            release_candidates = [
                (position, instruction.instruction_id)
                for position, instruction in enumerate(queue)
                if previous_position <= position < next_position and any(storage_key(slot_id) == key for slot_id in instruction.input_slots)
            ]
            release_node_id = previous.resolved_dispatch_instruction_id if not release_candidates else max(release_candidates)[1]
            gated_transfer_records.append(
                (
                    (consumer_cta, send.producer_sm, release_node_id),
                    subsequent_key,
                    send,
                )
            )

    if not gated_transfer_records:
        return plan

    gate_id_by_group: dict[tuple[int, int, int], int] = {}
    gate_id_by_transfer: dict[tuple[int, int, int, int, int], int] = {}
    release_template_by_group: dict[tuple[int, int, int], CommPlan] = {}
    for group, key, send in gated_transfer_records:
        gate_id = gate_id_by_group.setdefault(group, len(gate_id_by_group))
        gate_id_by_transfer[key] = gate_id
        release_template_by_group.setdefault(group, send)

    # Multiple payload tiles may share one generation gate.  Only the first
    # send in producer queue order needs to execute the wait; once it passes,
    # later sends covered by the same release contract are already safe.  Mark
    # that exact transfer rather than making every payload pay a redundant
    # pending-bit lookup on the hot path.
    first_gated_send_by_gate: dict[int, CommPlan] = {}
    for _, key, send in gated_transfer_records:
        gate_id = gate_id_by_transfer[key]
        current = first_gated_send_by_gate.get(gate_id)
        send_order = (
            queue_positions[send.resolved_dispatch_instruction_id][1],
            send.segment_id,
            send.target_slot_id,
        )
        if current is None:
            first_gated_send_by_gate[gate_id] = send
            continue
        current_order = (
            queue_positions[current.resolved_dispatch_instruction_id][1],
            current.segment_id,
            current.target_slot_id,
        )
        if send_order < current_order:
            first_gated_send_by_gate[gate_id] = send
    gated_slot_ids = {send.target_slot_id for send in first_gated_send_by_gate.values()}
    comms = tuple(
        replace(comm, cluster_gate_id=gate_id_by_transfer[transfer_key(comm)])
        if comm.kind
        in {
            DataflowCommKind.CLUSTER_SEND,
            DataflowCommKind.CLUSTER_RECV,
        }
        and transfer_key(comm) in gate_id_by_transfer
        else comm
        for comm in plan.comms
    )
    added_releases = tuple(
        replace(
            release_template_by_group[group],
            producer_sm=group[0],
            consumer_sm=group[1],
            kind=DataflowCommKind.CLUSTER_RELEASE,
            dispatch_instruction_id=group[2],
            peer_cta_rank=plan.topology.cluster_rank(group[1]),
            cluster_gate_id=gate_id,
        )
        for group, gate_id in gate_id_by_group.items()
    )
    slots = tuple(replace(slot, cluster_gated_push=True) if slot.slot_id in gated_slot_ids else slot for slot in plan.slots)
    return replace(
        plan,
        slots=slots,
        comms=comms + added_releases,
    )


class DataflowArtifactNotExecutableError(NotImplementedError):
    """Raised when an inspection-only Dataflow artifact is used as an executable."""


DataflowUnsupportedTargetCapabilityError = UnsupportedTargetCapabilityError


@dataclass(frozen=True)
class DataflowArtifactState:
    """Explicit compile-time contract for a Dataflow compiled artifact."""

    contract: str
    executable: bool
    non_executable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.contract not in _DATAFLOW_ARTIFACT_CONTRACTS:
            expected = ", ".join(sorted(_DATAFLOW_ARTIFACT_CONTRACTS))
            raise ValueError(f"Unsupported Dataflow artifact contract {self.contract!r}; expected one of: {expected}")
        if self.executable and self.non_executable_reason is not None:
            raise ValueError("executable Dataflow artifacts cannot have a non-executable reason")
        if not self.executable and not self.non_executable_reason:
            raise ValueError("non-executable Dataflow artifacts require a reason")
        if self.contract == DATAFLOW_ARTIFACT_PRODUCTION and not self.executable:
            raise ValueError("production Dataflow artifacts must be executable")
        if self.contract == DATAFLOW_ARTIFACT_INSPECTION and self.executable:
            raise ValueError("inspection Dataflow artifacts cannot be executable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "executable": self.executable,
            "non_executable_reason": self.non_executable_reason,
        }


@dataclass(frozen=True)
class DataflowKernelSpec:
    program: DataflowProgram
    topology: GPUTopology | tuple[int, int] | int
    range_lengths: dict[Any, int | Sequence[int]]
    block_size: int
    range_offsets: dict[Any, int | Sequence[int]] | None = None
    task_extents: Sequence[int] | None = None
    include_exit: bool = True
    block_dim: int | None = None
    wrapper_name: str = "dataflow_wrapper"
    mode: str | None = None
    inspection_stage: str = "ir"
    scheduler_policy: str = "round_robin"
    reduce_strategy: str = "all_at_once"
    scheduler_config: DataflowSchedulerConfig | Mapping[str, Any] | None = None
    semantic_config: DataflowSemanticConfig | Mapping[str, Any] | None = None
    force_hbm_comms: bool = False
    device_ordinal: int | None = None
    target_capabilities: TargetCapabilitySnapshot | None = None
    target_override: Any | None = None
    target: str | None = None
    arch: str | None = None
    pic: bool = False
    pic_dir: Any = "schedule_res"
    progress: bool | None = None
    options: dict[str, Any] = field(default_factory=dict)
    _factory_execution_plans: tuple[DataflowExecutionPlan, ...] = ()


def make_kernel_spec(
    program: DataflowProgram,
    *,
    topology: GPUTopology | tuple[int, int] | int,
    range_lengths: dict[Any, int | Sequence[int]],
    block_size: int,
    **compile_options: Any,
) -> DataflowKernelSpec:
    """Build a kernel spec while preserving compile options not owned by the dataclass."""

    direct_field_names = {
        item.name
        for item in dataclass_fields(DataflowKernelSpec)
        if item.name
        not in {
            "program",
            "topology",
            "range_lengths",
            "block_size",
            "options",
            "_factory_execution_plans",
        }
    }
    if "_factory_execution_plans" in compile_options:
        raise ValueError("factory execution plans are compiler-owned metadata")
    direct_fields = {name: compile_options.pop(name) for name in direct_field_names if name in compile_options}
    raw_options = compile_options.pop("options", None)
    if raw_options is not None and not isinstance(raw_options, Mapping):
        raise TypeError(f"Dataflow kernel spec options must be a mapping, got {type(raw_options)!r}")
    options = dict(raw_options or {})
    duplicates = options.keys() & compile_options.keys()
    if duplicates:
        raise ValueError(f"duplicate Dataflow kernel spec options in options mapping and keyword arguments: {sorted(duplicates)!r}")
    options.update(compile_options)
    return DataflowKernelSpec(
        program=program,
        topology=topology,
        range_lengths=range_lengths,
        block_size=block_size,
        options=options,
        **direct_fields,
    )


@dataclass(frozen=True)
class DataflowCompiledProgram:
    program: DataflowProgram
    plan: InstructionPlan
    packed_plan: PackedRuntimePlan
    launch_package: DataflowLaunchPackage
    tensor_arg_plan: DataflowTensorArgPlan
    wrapper_spec: DataflowWrapperSpec
    wrapper_source: str
    options: dict[str, Any]
    compile_config: DataflowCompileConfig
    artifact_state: DataflowArtifactState
    memory_plan: DataflowMemoryPlan
    primfunc_lowering: DataflowPrimFuncLoweringResult | None = None

    @property
    def artifact_contract(self) -> str:
        return self.artifact_state.contract

    @property
    def executable(self) -> bool:
        return self.artifact_state.executable

    @property
    def non_executable_reason(self) -> str | None:
        return self.artifact_state.non_executable_reason

    @property
    def target_capabilities(self) -> TargetCapabilitySnapshot:
        return self.compile_config.target_capabilities

    @property
    def target_fingerprint(self) -> str:
        return self.target_capabilities.fingerprint

    @property
    def artifact_fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "compile_config": self.compile_config.fingerprint,
                "handler_variant_keys": self.packed_plan.handler_variant_keys,
                "wrapper_source": self.wrapper_source,
                "launch_package": self.launch_package,
                "memory_plan": self.memory_plan,
            }
        )

    def decision_artifact(self) -> DataflowDecisionArtifact:
        """Return a read-only snapshot of decisions made during compilation."""

        return collect_dataflow_decision_artifact(self)

    def dump_plan(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact_state.to_dict(),
            "artifact_fingerprint": self.artifact_fingerprint,
            "target_capabilities": self.target_capabilities.to_dict(),
            "compile_config": self.compile_config.to_dict(),
            "options": dict(self.options),
            "plan": {
                "block_size": self.plan.block_size,
                "range_axis": self.plan.range_axis,
                "scheduler_policy": self.plan.scheduler_policy,
                "reduce_strategy": self.plan.reduce_strategy,
                "scheduler_config": self.plan.scheduler_config.to_dict(),
                "range_resource_budget_bytes": self.plan.range_resource_budget_bytes,
                "range_coarsening_plans": [
                    {"stage_id": stage_id, "plan": range_plan.to_dict()} for stage_id, range_plan in self.plan.range_coarsening_plans
                ],
                "reshared_transport_plans": [
                    {"stage_id": stage_id, "plan": transport_plan.to_dict()}
                    for stage_id, transport_plan in self.plan.reshared_transport_plans
                ],
                "cross_handler_handoff_plans": [handoff_plan.to_dict() for handoff_plan in self.plan.cross_handler_handoff_plans],
                "cross_handler_handoff_bindings": [binding.to_dict() for binding in self.plan.cross_handler_handoff_bindings],
                "joint_execution_plan": (None if self.plan.joint_execution_plan is None else self.plan.joint_execution_plan.to_dict()),
                "task_range_lengths": list(self.plan.task_range_lengths),
                "queue_lengths": {sm_id: len(queue) for sm_id, queue in self.plan.queues.items()},
                "slot_count": len(self.plan.slots),
                "slots": plan_slots_to_dict(self.plan, self.packed_plan),
                "comm_count": len(self.plan.comms),
                "instruction_count": len(self.plan.instructions),
            },
            "packed_plan": self.packed_plan.to_dict(),
            "launch_package": self.launch_package.to_dict(),
            "memory_plan": self.memory_plan.to_dict(),
            "tensor_args": self.tensor_arg_plan.to_dict(),
            "wrapper": self.wrapper_spec.to_dict(),
            "primfunc_lowering": (None if self.primfunc_lowering is None else self.primfunc_lowering.to_dict()),
            "decisions": self.decision_artifact().to_dict(),
        }

    def pack_tensor_args(self, *args: Any, **kwargs: Any) -> PackedRuntimeTensorArgs:
        return _pack_tensor_args(self.tensor_arg_plan, *args, **kwargs)

    def validate_memory_layout(self) -> DataflowLayoutValidationReport:
        """Return the structured memory-layout validation report for this artifact."""

        from .layout_validation import validate_dataflow_memory_layout

        handler_artifacts = () if self.primfunc_lowering is None else self.primfunc_lowering.codegen_artifacts
        return validate_dataflow_memory_layout(
            self.packed_plan,
            self.launch_package,
            self.wrapper_spec,
            plan=self.plan,
            handler_artifacts=handler_artifacts,
            target_capabilities=self.target_capabilities,
        )

    def persistent_executable(self, *args: Any, **kwargs: Any) -> DataflowPersistentExecutable:
        self.require_executable("build a persistent executable")
        runtime_tensor_args = _pack_tensor_args(
            self.tensor_arg_plan,
            *args,
            allow_missing=not args and not kwargs,
            **kwargs,
        )
        tma_descriptor_handles = build_tma_descriptor_handles_if_ready(
            self.wrapper_spec.tma_descriptors,
            runtime_tensor_args,
            expected_device_ordinal=self.target_capabilities.device_ordinal,
        )
        return DataflowPersistentExecutable(
            kernel_name=self.wrapper_spec.kernel_name,
            source=self.wrapper_source,
            launch_package=self.launch_package,
            tensor_args_bytes=runtime_tensor_args.tensor_args_bytes,
            tma_descriptor_specs=self.wrapper_spec.tma_descriptors,
            tma_descriptor_handles=tma_descriptor_handles,
            tma_tensor_indices=tensor_indices(runtime_tensor_args),
            tma_tensor_metadata=tensor_metadata(runtime_tensor_args),
            topology=self.plan.topology,
            target_capabilities=self.target_capabilities,
            artifact_fingerprint=self.artifact_fingerprint,
            options=self.options,
        )

    def get_profiler(self) -> Any:
        from .profiler import DataflowProfiler

        return DataflowProfiler(self)

    def profile_walltime(self, *args: Any, **kwargs: Any) -> Any:
        from .walltime import profile_compiled_walltime

        return profile_compiled_walltime(self, *args, **kwargs)

    def do_bench(self, *args: Any, **kwargs: Any) -> float | list[float]:
        return self.get_profiler().do_bench(*args, **kwargs)

    def __call__(self, *args: Any, stream: Any | None = None, **kwargs: Any) -> DataflowExecutionResult:
        self.require_executable("launch")
        runtime_tensor_args = _pack_tensor_args(
            self.tensor_arg_plan,
            *args,
            allow_missing=not args and not kwargs,
            **kwargs,
        )
        tma_descriptor_handles = build_tma_descriptor_handles_if_ready(
            self.wrapper_spec.tma_descriptors,
            runtime_tensor_args,
            expected_device_ordinal=self.target_capabilities.device_ordinal,
        )
        return DataflowExecutableKernel(
            kernel_name=self.wrapper_spec.kernel_name,
            source=self.wrapper_source,
            launch_package=self.launch_package,
            tensor_args_bytes=runtime_tensor_args.tensor_args_bytes,
            tma_descriptor_specs=self.wrapper_spec.tma_descriptors,
            tma_descriptor_handles=tma_descriptor_handles,
            tma_tensor_indices=tensor_indices(runtime_tensor_args),
            topology=self.plan.topology,
            target_capabilities=self.target_capabilities,
            artifact_fingerprint=self.artifact_fingerprint,
            options=self.options,
        ).launch(stream=stream)

    def require_executable(self, action: str) -> None:
        if self.executable:
            return
        raise DataflowArtifactNotExecutableError(
            f"cannot {action} Dataflow {self.artifact_contract} artifact: {self.non_executable_reason}"
        )


@dataclass(frozen=True)
class ScratchBackedSlotPlan:
    plan: InstructionPlan
    slot_count: int
    slot_bytes: int
    required_scratch_bytes: int
    handler_scratch_offsets: tuple[tuple[str, int], ...] = ()
    iter_symbols: tuple[str, ...] = ()
    cluster_inbox_offset: int = 0
    cluster_inbox_bytes: int = 0
    transient_prefetch_offset: int = 0
    transient_prefetch_bytes: int = 0


class DataflowJITFunction:
    """Callable Dataflow kernel factory with an internal compile cache."""

    def __init__(self, factory: Callable[..., DataflowKernelSpec], *, cache: bool = True):
        if not callable(factory):
            raise TypeError(f"tilelang.dataflow.jit expects a callable, got {factory!r}")
        self.factory = factory
        self.cache = bool(cache)
        self._signature = inspect.signature(factory)
        self._cache: dict[tuple[Any, ...], DataflowCompiledProgram] = {}
        update_wrapper(self, factory)

    def __call__(self, *args: Any, **kwargs: Any) -> DataflowCompiledProgram:
        environment = capture_compile_environment()
        factory_arguments = self.bound_factory_arguments(args, kwargs)
        target_capabilities = resolve_factory_target_capabilities(
            factory_arguments,
            environment,
        )
        with use_target_capabilities(target_capabilities), capture_execution_plans() as plans:
            spec = self.factory(*args, **kwargs)
        spec = bind_factory_target_capabilities(spec, target_capabilities, environment)
        spec = bind_factory_execution_plans(spec, plans)
        config = create_kernel_spec_compile_config(
            spec,
            environment=environment,
            factory_arguments=factory_arguments,
        )
        if not self.cache:
            return compile_kernel_spec_with_config(spec, config)

        key = self.make_cache_key(config)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        compiled = compile_kernel_spec_with_config(spec, config)
        self._cache[key] = compiled
        return compiled

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def bound_factory_arguments(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        bound = self._signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    @staticmethod
    def make_cache_key(config: DataflowCompileConfig) -> tuple[Any, ...]:
        return (config.schema_version, config.fingerprint)


def compile_kernel_spec(spec: DataflowKernelSpec) -> DataflowCompiledProgram:
    if not isinstance(spec, DataflowKernelSpec):
        raise TypeError(f"tilelang.dataflow.compile_kernel_spec expects DataflowKernelSpec, got {spec!r}")
    environment = capture_compile_environment()
    config = create_kernel_spec_compile_config(spec, environment=environment)
    return compile_kernel_spec_with_config(spec, config)


_FACTORY_TARGET_OPTION_NAMES = (
    "device_ordinal",
    "target_capabilities",
    "target_override",
    "target",
    "arch",
)


def resolve_factory_target_capabilities(
    factory_arguments: dict[str, Any],
    environment: dict[str, str],
) -> TargetCapabilitySnapshot:
    supplied: dict[str, Any] = {}
    nested_options = tuple(value for value in factory_arguments.values() if isinstance(value, dict))
    for name in _FACTORY_TARGET_OPTION_NAMES:
        direct = factory_arguments.get(name)
        nested = tuple(options[name] for options in nested_options if options.get(name) is not None)
        values = ((direct,) if direct is not None else ()) + nested
        if len(values) > 1 and any(value != values[0] for value in values[1:]):
            raise ValueError(f"conflicting Dataflow JIT factory target option {name!r}: {values!r}")
        if values:
            supplied[name] = values[0]
    snapshot, _ = resolve_target_capability_snapshot(supplied, environment)
    return snapshot


def bind_factory_target_capabilities(
    spec: DataflowKernelSpec,
    target_capabilities: TargetCapabilitySnapshot,
    environment: dict[str, str],
) -> DataflowKernelSpec:
    if not isinstance(spec, DataflowKernelSpec):
        raise TypeError(f"tilelang.dataflow.jit factory must return DataflowKernelSpec, got {spec!r}")
    target_options = {
        "target_capabilities": spec.target_capabilities,
        "target_override": spec.target_override,
        "target": spec.target,
        "arch": spec.arch,
    }
    returned_options = {name: value for name, value in target_options.items() if value is not None}
    if returned_options:
        returned_snapshot, _ = resolve_target_capability_snapshot(
            returned_options,
            environment,
        )
        if returned_snapshot.compatibility_fingerprint != target_capabilities.compatibility_fingerprint:
            raise ValueError(
                "Dataflow JIT factory returned target options that conflict with the compiler-owned "
                f"target snapshot: factory={returned_snapshot.to_dict()!r}, "
                f"compile_boundary={target_capabilities.to_dict()!r}"
            )
    if (
        spec.device_ordinal is not None
        and target_capabilities.device_ordinal is not None
        and int(spec.device_ordinal) != target_capabilities.device_ordinal
    ):
        raise ValueError(
            "Dataflow JIT factory returned a device ordinal that conflicts with the "
            f"compiler-owned target snapshot: {spec.device_ordinal} != "
            f"{target_capabilities.device_ordinal}"
        )
    return replace(
        spec,
        target_capabilities=target_capabilities,
        target_override=None,
        target=None,
        arch=None,
    )


def bind_factory_execution_plans(
    spec: DataflowKernelSpec,
    plans: Sequence[DataflowExecutionPlan],
) -> DataflowKernelSpec:
    if spec._factory_execution_plans:
        raise ValueError("Dataflow JIT factory cannot provide compiler-owned execution plans")
    typed_plans = tuple(plans)
    if any(not isinstance(plan, DataflowExecutionPlan) for plan in typed_plans):
        raise TypeError("captured Dataflow execution plans must be typed")
    return replace(spec, _factory_execution_plans=typed_plans)


def create_kernel_spec_compile_config(
    spec: DataflowKernelSpec,
    *,
    environment: dict[str, str],
    factory_arguments: dict[str, Any] | None = None,
) -> DataflowCompileConfig:
    if not isinstance(spec, DataflowKernelSpec):
        raise TypeError(f"tilelang.dataflow.compile_kernel_spec expects DataflowKernelSpec, got {spec!r}")
    options = dict(spec.options)
    if "execution_plans" in options:
        raise ValueError("execution_plans is compiler-owned decision metadata")
    options["execution_plans"] = tuple(plan.to_dict() for plan in spec._factory_execution_plans)
    options.update(
        {
            "wrapper_name": spec.wrapper_name,
            "inspection_stage": spec.inspection_stage,
            "scheduler_policy": spec.scheduler_policy,
            "reduce_strategy": spec.reduce_strategy,
            "force_hbm_comms": spec.force_hbm_comms,
            "pic": spec.pic,
            "pic_dir": spec.pic_dir,
        }
    )
    if spec.mode is not None:
        options["mode"] = spec.mode
    if spec.scheduler_config is not None:
        options["scheduler_config"] = spec.scheduler_config
    if spec.semantic_config is not None:
        options["semantic_config"] = spec.semantic_config
    if spec.block_dim is not None:
        options["block_dim"] = spec.block_dim
    if spec.device_ordinal is not None:
        options["device_ordinal"] = spec.device_ordinal
    if spec.target_capabilities is not None:
        options["target_capabilities"] = spec.target_capabilities
    if spec.target_override is not None:
        options["target_override"] = spec.target_override
    if spec.target is not None:
        options["target"] = spec.target
    if spec.arch is not None:
        options["arch"] = spec.arch
    if spec.progress is not None:
        options["progress"] = spec.progress
    return create_compile_config(
        program=spec.program,
        topology=normalize_kernel_spec_topology(spec.topology),
        range_lengths=spec.range_lengths,
        block_size=spec.block_size,
        range_offsets=spec.range_offsets,
        task_extents=spec.task_extents,
        include_exit=spec.include_exit,
        options=options,
        environment=environment,
        factory_arguments=factory_arguments,
    )


def compile_kernel_spec_with_config(
    spec: DataflowKernelSpec,
    config: DataflowCompileConfig,
) -> DataflowCompiledProgram:
    return compile_with_config(spec.program, config)


def jit(factory: Callable[..., DataflowKernelSpec] | None = None, *, cache: bool = True):
    if factory is None:
        return lambda wrapped: DataflowJITFunction(wrapped, cache=cache)

    return DataflowJITFunction(factory, cache=cache)


_COMPILE_OPTION_DEFAULTS: dict[str, Any] = {
    "wrapper_name": "dataflow_wrapper",
    "inspection_stage": "ir",
    "queue_indexing": "cta_rank",
    "scheduler_policy": "round_robin",
    "reduce_strategy": "all_at_once",
    "force_hbm_comms": False,
    "partial_only": False,
    "streaming_tree_consumer": None,
    "cluster_task_assignment": None,
    "task_coord_overrides": None,
    "stage_graph_task_weights": None,
    "stage_graph_cluster_assignment": None,
    "skip_tiny_root_fragment_blocks": None,
    "memory_policy": None,
    "reuse_hbm_flags": False,
    "async_safe_cluster_handoff": False,
    "primfunc_pass_configs": None,
    "block_dim": 32,
    "pic": False,
    "pic_dir": "schedule_res",
}
_LEGACY_TARGET_OVERRIDE_WARNING_EMITTED = False

_RETIRED_COMPILE_OPTIONS = frozenset(
    {
        "debug_handler",
        "hbm_direct_global",
        "handler_lowering",
        "lower_primfunc_handlers",
        "link_primfunc_handlers",
        "dataflow_scratch_backed_slots",
        "primfunc_reduce_staging",
    }
)


def create_compile_config(
    *,
    program: DataflowProgram,
    topology: GPUTopology,
    range_lengths: dict[Any, int | Sequence[int]],
    block_size: int,
    range_offsets: dict[Any, int | Sequence[int]] | None,
    task_extents: Sequence[int] | None,
    include_exit: bool,
    options: dict[str, Any],
    environment: dict[str, str],
    factory_arguments: dict[str, Any] | None = None,
) -> DataflowCompileConfig:
    if not isinstance(program, DataflowProgram):
        raise TypeError(f"tilelang.dataflow.compile expects DataflowProgram, got {program!r}")
    if not isinstance(topology, GPUTopology):
        raise TypeError(f"Dataflow compile topology must be a GPUTopology, got {topology!r}")

    supplied = dict(options)
    if "scheduler_auto_policy" in supplied:
        raise ValueError(
            "scheduler_auto_policy is compiler-owned decision metadata; request "
            "scheduler_policy='auto' and/or reduce_strategy='auto' instead"
        )
    if "precision_plan" in supplied:
        raise ValueError(
            "precision_plan is compiler-owned decision metadata; provide "
            "semantic_config.precision and operator accumulator_contracts instead"
        )
    if "memory_plan" in supplied or "memory_planner_version" in supplied:
        raise ValueError("memory_plan and memory_planner_version are compiler-owned decision metadata; provide memory_policy instead")
    governance_options = frozenset(
        {
            "implementation_registry_version",
            "implementation_registry_fingerprint",
            "lowering_boundary_schema_version",
            "lowering_boundary_fingerprint",
            "operation_contract_schema_version",
            "operation_contract_fingerprint",
        }
    )
    supplied_governance = tuple(sorted(governance_options.intersection(supplied)))
    if supplied_governance:
        raise ValueError(f"Dataflow registry, lowering boundary, and operation schema metadata are compiler-owned: {supplied_governance!r}")
    retired = tuple(sorted(_RETIRED_COMPILE_OPTIONS.intersection(supplied)))
    if retired:
        raise ValueError(
            "Dataflow compile options were retired: "
            f"{retired!r}; use mode='executable' or mode='inspect', typed "
            "memory_policy, scheduler_config, and semantic_config. Synthetic "
            "handlers are available only from tilelang.dataflow.experimental."
        )
    effective = dict(_COMPILE_OPTION_DEFAULTS)
    effective.update(supplied)
    provenance = {key: ("option" if key in supplied else "default") for key in effective}
    provenance.update({f"environment.{name}": "process-environment" for name in environment})
    effective["gemm_lowering_registry_version"] = GEMM_LOWERING_REGISTRY_VERSION
    provenance["gemm_lowering_registry_version"] = "compiler-registry"
    effective["transfer_lowering_registry_version"] = int(_ffi_api.TransferLoweringRegistryVersion())
    provenance["transfer_lowering_registry_version"] = "compiler-registry"
    effective["pipeline_lowering_registry_version"] = int(_ffi_api.PipelineLoweringVersion())
    provenance["pipeline_lowering_registry_version"] = "compiler-registry"
    implementation_registry = dataflow_implementation_registry()
    effective["implementation_registry_version"] = DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION
    effective["implementation_registry_fingerprint"] = implementation_registry.fingerprint
    provenance["implementation_registry_version"] = "compiler-registry"
    provenance["implementation_registry_fingerprint"] = "compiler-registry"
    effective["lowering_boundary_schema_version"] = DATAFLOW_LOWERING_BOUNDARY_CONTRACT.schema_version
    effective["lowering_boundary_fingerprint"] = DATAFLOW_LOWERING_BOUNDARY_CONTRACT.fingerprint
    provenance["lowering_boundary_schema_version"] = "architecture-contract"
    provenance["lowering_boundary_fingerprint"] = "architecture-contract"
    effective["operation_contract_schema_version"] = DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION
    effective["operation_contract_fingerprint"] = DATAFLOW_OPERATION_CONTRACT_FINGERPRINT
    provenance["operation_contract_schema_version"] = "operation-contract-schema"
    provenance["operation_contract_fingerprint"] = "operation-contract-schema"

    mode, handler_lowering, lower_handlers, link_handlers = resolve_compile_mode(supplied)
    effective.update(
        {
            "mode": mode,
            "handler_lowering": handler_lowering,
            "lower_primfunc_handlers": lower_handlers,
            "link_primfunc_handlers": link_handlers,
        }
    )
    provenance.update(
        {
            "mode": "option" if "mode" in supplied else "default",
            "handler_lowering": "mode",
            "lower_primfunc_handlers": "mode",
            "link_primfunc_handlers": "mode",
        }
    )

    for option_name, default in (
        ("iter_range_buckets", None),
        ("iter_range_exact_lengths", None),
    ):
        effective.setdefault(option_name, default)
        provenance.setdefault(
            option_name,
            "option" if option_name in supplied else "default",
        )
    effective.setdefault("iter_range_bucket_size", None)
    provenance.setdefault(
        "iter_range_bucket_size",
        "option" if "iter_range_bucket_size" in supplied else "default",
    )
    resolve_environment_option(
        effective,
        provenance,
        supplied,
        environment,
        option_name="progress",
        environment_names=("DATAFLOW_PROGRESS", "DATAFLOW_COMPILE_PROGRESS", "DATAFLOW_COMPILE_LOG"),
        default=False,
    )

    target_capabilities, target_source = resolve_target_capability_snapshot(
        supplied,
        environment,
    )
    validate_target_topology(target_capabilities, topology)
    for internal_name in ("target_override", "target_capabilities"):
        effective.pop(internal_name, None)
        provenance.pop(internal_name, None)
    effective.update(
        {
            "target": target_capabilities.target,
            "arch": target_capabilities.arch,
            "device_ordinal": target_capabilities.device_ordinal,
            "target_fingerprint": target_capabilities.fingerprint,
            "target_capabilities": target_capabilities.to_dict(),
        }
    )
    provenance.update(
        {
            "target": target_source,
            "arch": target_source,
            "device_ordinal": target_source,
            "target_fingerprint": "target-capability-snapshot",
            "target_capabilities": "target-capability-snapshot",
        }
    )

    raw_scheduler_policy = effective["scheduler_policy"]
    requested_scheduler_policy = (
        DATAFLOW_SCHEDULER_AUTO if raw_scheduler_policy == DATAFLOW_SCHEDULER_AUTO else normalize_scheduler_policy(raw_scheduler_policy)
    )
    raw_reduce_strategy = effective["reduce_strategy"]
    requested_reduce_strategy = (
        DATAFLOW_SCHEDULER_AUTO if raw_reduce_strategy == DATAFLOW_SCHEDULER_AUTO else normalize_reduce_strategy(raw_reduce_strategy)
    )
    scheduler_config = resolve_scheduler_config(supplied.get("scheduler_config"))
    semantic_config = resolve_semantic_config(supplied.get("semantic_config"))
    raw_compile_flags = effective.get("compile_flags", ()) or ()
    compile_flags = [raw_compile_flags] if isinstance(raw_compile_flags, str) else list(raw_compile_flags)
    if not semantic_config.fast_math and "--use_fast_math" in compile_flags:
        raise ValueError("--use_fast_math is compiler-owned; request it through DataflowSemanticConfig(fast_math=True)")
    if semantic_config.fast_math:
        if "--use_fast_math" not in compile_flags:
            compile_flags.append("--use_fast_math")
        effective["compile_flags"] = tuple(compile_flags)
        provenance["compile_flags"] = "option+semantic-config" if "compile_flags" in supplied else "semantic-config"
    precision_plan = resolve_program_precision(program, semantic_config.precision)
    effective["precision_plan"] = precision_plan.to_dict()
    provenance["precision_plan"] = "compiler-precision-policy"
    memory_policy = resolve_memory_policy(supplied.get("memory_policy"))
    memory_policy_source = "option" if supplied.get("memory_policy") is not None else "default"
    effective["memory_policy"] = memory_policy.to_dict()
    effective["memory_planner_version"] = DATAFLOW_MEMORY_PLANNER_VERSION
    provenance["memory_policy"] = memory_policy_source
    provenance["memory_planner_version"] = "compiler-memory-planner"
    auto_requested = requested_scheduler_policy == DATAFLOW_SCHEDULER_AUTO or requested_reduce_strategy == DATAFLOW_SCHEDULER_AUTO
    policy_records = supplied.get("scheduler_policy_records")
    effective.pop("scheduler_policy_records", None)
    provenance.pop("scheduler_policy_records", None)
    if policy_records is not None and not auto_requested:
        raise ValueError("scheduler_policy_records requires scheduler_policy='auto' or reduce_strategy='auto'")
    if auto_requested:
        scratch_backed_auto = memory_policy.considers_scratch and not program.is_stage_graph
        partial_only = bool_option(
            effective.get("partial_only", False),
            "partial_only",
        )
        auto_result = select_scheduler_auto_policy(
            program,
            topology=topology,
            range_lengths=range_lengths,
            range_offsets=range_offsets,
            block_size=block_size,
            task_extents=task_extents,
            include_exit=include_exit,
            requested_scheduler_policy=requested_scheduler_policy,
            requested_reduce_strategy=requested_reduce_strategy,
            scheduler_config=scheduler_config,
            target_capabilities=target_capabilities,
            policy_records=policy_records,
            force_hbm_comms=bool_option(
                effective.get("force_hbm_comms", False),
                "force_hbm_comms",
            ),
            scratch_backed_slots=scratch_backed_auto,
            partial_only=partial_only,
            direct_leaf_acc=(
                bool_option(effective["direct_leaf_acc"], "direct_leaf_acc") if effective.get("direct_leaf_acc") is not None else None
            ),
            streaming_tree_consumer=effective.get("streaming_tree_consumer"),
            cluster_task_assignment=effective.get("cluster_task_assignment"),
            skip_tiny_root_fragment=(
                bool_option(
                    effective["skip_tiny_root_fragment"],
                    "skip_tiny_root_fragment",
                )
                if effective.get("skip_tiny_root_fragment") is not None
                else None
            ),
            skip_tiny_root_fragment_blocks=effective.get("skip_tiny_root_fragment_blocks"),
            iter_range_buckets=effective.get("iter_range_buckets"),
            iter_range_bucket_size=effective.get("iter_range_bucket_size"),
            iter_range_exact_lengths=effective.get("iter_range_exact_lengths"),
            task_coord_overrides=effective.get("task_coord_overrides"),
            stage_graph_task_weights=effective.get("stage_graph_task_weights"),
            stage_graph_cluster_assignment=effective.get("stage_graph_cluster_assignment"),
            _candidate_plan_validator=(
                partial(
                    scratch_backed_candidate_rejection_reason,
                    partial_only=partial_only,
                )
                if scratch_backed_auto
                else None
            ),
        )
        selected_options = auto_result.selected_compile_options()
        scheduler_config = auto_result.selected_scheduler_config
        scheduler_policy = str(selected_options.pop("scheduler_policy"))
        reduce_strategy = str(selected_options.pop("reduce_strategy"))
        selected_options.pop("scheduler_config")
        effective.update(selected_options)
        effective["scheduler_policy"] = scheduler_policy
        effective["reduce_strategy"] = reduce_strategy
        effective["scheduler_auto_policy"] = auto_result.decision.to_dict()
        provenance.update(
            {
                "scheduler_policy": "scheduler-auto-policy",
                "reduce_strategy": "scheduler-auto-policy",
                "scheduler_auto_policy": "scheduler-auto-policy",
            }
        )
        for option_name in selected_options:
            provenance[option_name] = "scheduler-auto-policy"
    else:
        scheduler_policy = requested_scheduler_policy
        reduce_strategy = requested_reduce_strategy
        effective["scheduler_policy"] = scheduler_policy
        effective["reduce_strategy"] = reduce_strategy
    effective["scheduler_config"] = scheduler_config.to_dict()
    effective["semantic_config"] = semantic_config.to_dict()
    provenance["scheduler_config"] = (
        "scheduler-auto-policy" if auto_requested else "option" if "scheduler_config" in supplied else "default"
    )
    provenance["semantic_config"] = "option" if "semantic_config" in supplied else "default"
    return DataflowCompileConfig.create(
        mode=mode,
        handler_lowering=handler_lowering,
        lower_primfunc_handlers=lower_handlers,
        link_primfunc_handlers=link_handlers,
        topology=topology,
        range_lengths=range_lengths,
        range_offsets=range_offsets,
        block_size=block_size,
        task_extents=task_extents,
        include_exit=include_exit,
        target_capabilities=target_capabilities,
        scheduler_policy=scheduler_policy,
        reduce_strategy=reduce_strategy,
        scheduler_config=scheduler_config,
        semantic_config=semantic_config,
        options=effective,
        environment=environment,
        provenance=provenance,
        factory_arguments=factory_arguments,
        program_fingerprint=canonical_fingerprint(program),
    )


def resolve_target_capability_snapshot(
    options: dict[str, Any],
    environment: dict[str, str],
) -> tuple[TargetCapabilitySnapshot, str]:
    snapshot = options.get("target_capabilities")
    override = options.get("target_override")
    legacy_target = options.get("target")
    legacy_arch = options.get("arch")
    environment_arch = environment.get("TILELANG_DATAFLOW_WRAPPER_COMPILE_ARCH")
    device_ordinal = options.get("device_ordinal")

    if snapshot is not None and override is not None:
        raise ValueError("Dataflow compile accepts only one of target_capabilities and target_override")
    if snapshot is not None:
        if not isinstance(snapshot, TargetCapabilitySnapshot):
            raise TypeError(f"target_capabilities must be a TargetCapabilitySnapshot, got {snapshot!r}")
        if device_ordinal is not None and snapshot.device_ordinal != int(device_ordinal):
            raise ValueError(
                f"device_ordinal conflicts with injected target_capabilities: {device_ordinal!r} != {snapshot.device_ordinal!r}"
            )
        if legacy_target is not None or legacy_arch is not None:
            raise ValueError("target_capabilities cannot be combined with legacy target/arch options")
        return snapshot, "target-capability-snapshot"

    if override is not None:
        if legacy_target is not None or legacy_arch is not None:
            raise ValueError("target_override cannot be combined with legacy target/arch options")
        if isinstance(override, TargetCapabilitySnapshot):
            return override, "target-override-snapshot"
        return target_capability_override(override), "target-override"

    if legacy_target is not None or legacy_arch is not None or environment_arch:
        warn_legacy_target_override()
        resolved_arch = legacy_arch or environment_arch
        target_value = "cuda" if legacy_target is None else legacy_target
        if resolved_arch is None and str(target_value).strip() in {"auto", "cuda"}:
            return resolve_target_capabilities(device_ordinal), "auto-device"
        source = (
            "environment:TILELANG_DATAFLOW_WRAPPER_COMPILE_ARCH" if legacy_arch is None and environment_arch else "legacy-target-override"
        )
        return (
            target_capability_override(target_value, arch=resolved_arch),
            source,
        )

    return resolve_target_capabilities(device_ordinal), "auto-device"


def validate_target_topology(
    target_capabilities: TargetCapabilitySnapshot,
    topology: GPUTopology,
) -> None:
    if topology.cluster_size <= 1:
        return
    if not target_capabilities.supports_cluster_launch:
        raise DataflowUnsupportedTargetCapabilityError(
            "Dataflow topology requires CUDA cluster launch support: "
            f"cluster_size={topology.cluster_size}, target={target_capabilities.target}, "
            f"target_fingerprint={target_capabilities.fingerprint}"
        )
    max_cluster_size = target_capabilities.max_cluster_size
    if max_cluster_size is not None and topology.cluster_size > max_cluster_size:
        raise DataflowUnsupportedTargetCapabilityError(
            "Dataflow topology exceeds the resolved CUDA cluster limit: "
            f"cluster_size={topology.cluster_size}, max_cluster_size={max_cluster_size}, "
            f"target={target_capabilities.target}"
        )


def resolve_compile_mode(options: dict[str, Any]) -> tuple[str, str, bool, bool]:
    explicit_mode = options.get("mode")
    mode = DATAFLOW_COMPILE_MODE_EXECUTABLE if explicit_mode is None else str(explicit_mode)
    if mode not in {
        DATAFLOW_COMPILE_MODE_EXECUTABLE,
        DATAFLOW_COMPILE_MODE_INSPECT,
        DATAFLOW_COMPILE_MODE_DEBUG,
    }:
        expected = ", ".join((DATAFLOW_COMPILE_MODE_EXECUTABLE, DATAFLOW_COMPILE_MODE_INSPECT, DATAFLOW_COMPILE_MODE_DEBUG))
        raise ValueError(f"Unsupported Dataflow compile mode {mode!r}; expected one of: {expected}")

    if mode == DATAFLOW_COMPILE_MODE_EXECUTABLE:
        return mode, PRIMFUNC_HANDLER_LOWERING, True, True

    if mode == DATAFLOW_COMPILE_MODE_INSPECT:
        inspection_stage = str(options.get("inspection_stage", "ir"))
        if inspection_stage not in {"ir", "cuda"}:
            raise ValueError(f"Dataflow inspect mode inspection_stage must be 'ir' or 'cuda', got {inspection_stage!r}")
        lower_handlers = inspection_stage == "cuda"
        return mode, PRIMFUNC_HANDLER_LOWERING, lower_handlers, False

    provider_name = options.get("_experimental_debug_handler")
    if provider_name is None:
        raise ValueError(
            "Dataflow mode='debug' is reserved for tilelang.dataflow.experimental; "
            "production callers must use mode='executable' or mode='inspect'"
        )
    from .experimental.debug_handlers import resolve_debug_handler

    provider = resolve_debug_handler(provider_name)
    return mode, provider.name, False, False


def warn_legacy_target_override() -> None:
    global _LEGACY_TARGET_OVERRIDE_WARNING_EMITTED  # pylint: disable=global-statement
    if _LEGACY_TARGET_OVERRIDE_WARNING_EMITTED:
        return
    _LEGACY_TARGET_OVERRIDE_WARNING_EMITTED = True
    warnings.warn(
        "Dataflow target/arch compile options are deprecated; ordinary compilation now resolves "
        "the selected CUDA device automatically. Use device_ordinal for device selection or "
        "target_override/target_capabilities for explicit cross compilation.",
        DeprecationWarning,
        stacklevel=5,
    )


def resolve_environment_option(
    effective: dict[str, Any],
    provenance: dict[str, str],
    supplied: dict[str, Any],
    environment: dict[str, str],
    *,
    option_name: str,
    environment_names: tuple[str, ...],
    default: Any,
) -> None:
    if option_name in supplied and supplied[option_name] is not None:
        effective[option_name] = supplied[option_name]
        provenance[option_name] = "option"
        return
    for environment_name in environment_names:
        if environment_name in environment:
            effective[option_name] = environment[environment_name]
            provenance[option_name] = f"environment:{environment_name}"
            return
    effective[option_name] = default
    provenance[option_name] = "default"


def compile(
    program: DataflowProgram,
    *,
    topology: GPUTopology,
    range_lengths: dict[Any, int | Sequence[int]],
    block_size: int,
    range_offsets: dict[Any, int | Sequence[int]] | None = None,
    task_extents: Sequence[int] | None = None,
    include_exit: bool = True,
    **options: Any,
) -> DataflowCompiledProgram:
    if not isinstance(program, DataflowProgram):
        raise TypeError(f"tilelang.dataflow.compile expects DataflowProgram, got {program!r}")

    config = create_compile_config(
        program=program,
        topology=topology,
        range_lengths=range_lengths,
        range_offsets=range_offsets,
        block_size=block_size,
        task_extents=task_extents,
        include_exit=include_exit,
        options=options,
        environment=capture_compile_environment(),
    )
    return compile_with_config(program, config)


def compile_with_config(
    program: DataflowProgram,
    config: DataflowCompileConfig,
) -> DataflowCompiledProgram:
    topology = config.topology
    range_lengths = config.to_dict()["range_lengths"]
    range_offsets = config.to_dict()["range_offsets"]
    block_size = config.block_size
    task_extents = config.task_extents
    include_exit = config.include_exit
    options = config.options_dict()
    memory_policy = resolve_memory_policy(options.get("memory_policy"))
    memory_attempt = (
        DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL if memory_policy.mode == DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL else DATAFLOW_MEMORY_SHARED
    )

    return compile_snapshot(
        program,
        config=config,
        topology=topology,
        range_lengths=range_lengths,
        range_offsets=range_offsets,
        block_size=block_size,
        task_extents=task_extents,
        include_exit=include_exit,
        options=options,
        memory_attempt=memory_attempt,
    )


def joint_reduce_inplace_direct_output_is_safe(
    program: DataflowProgram,
    plan: InstructionPlan,
) -> bool:
    """Prove that joint inbox materialization leaves one alias-safe local input.

    Scratch-backed streaming accumulators intentionally reuse one physical
    window.  Direct-output lowering is nevertheless safe when every reduction
    aliases that window only with the operator-declared local input, while
    every other input is materialized in an independent joint inbox.  The
    proof deliberately covers both binary and generic associative frontiers;
    arity is not a storage-lifetime property.
    """

    if program.reduce_stage is None or plan.joint_execution_plan is None:
        return False
    alias_indices = set(operator_physical_contract(program.reduce_stage.reduce_call.operator.attrs).output_alias_input_indices)
    if not alias_indices:
        return False

    schedule = plan.joint_execution_plan.require_valid(topology=plan.topology).schedule
    incoming_by_consumer: dict[int, dict[int, Any]] = {}
    for transfer in schedule.transfers:
        incoming = incoming_by_consumer.setdefault(transfer.consumer_node_id, {})
        if transfer.consumer_input_index in incoming:
            return False
        incoming[transfer.consumer_input_index] = transfer

    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    reduce_count = 0
    for instruction in plan.instructions:
        if instruction.opcode not in {
            DataflowOpcode.REDUCE,
            DataflowOpcode.REDUCE_UPDATE,
        }:
            continue
        reduce_count += 1
        if instruction.output_slot is None or len(instruction.input_slots) < 2:
            return False
        incoming = incoming_by_consumer.get(instruction.instruction_id, {})
        remote_indices = set(incoming)
        local_indices = set(range(len(instruction.input_slots))) - remote_indices
        if len(local_indices) != 1 or len(remote_indices) != len(instruction.input_slots) - 1:
            return False
        local_index = next(iter(local_indices))
        if local_index not in alias_indices:
            return False
        for remote_index in remote_indices:
            transfer = incoming[remote_index]
            if (
                transfer.kind
                not in {
                    DataflowTransportKind.CLUSTER_PUSH,
                    DataflowTransportKind.HBM_STAGED,
                }
                or transfer.consumer_storage_kind
                not in {
                    DataflowCommStorageKind.PERMANENT,
                    DataflowCommStorageKind.TRANSIENT_PREFETCH,
                }
                or transfer.retained_input_copy_event_id is not None
            ):
                return False

        output = slots_by_id[instruction.output_slot]
        local_input = slots_by_id[instruction.input_slots[local_index]]
        if output.shared_storage_id is None or output.shared_storage_id != local_input.shared_storage_id:
            return False
    return reduce_count > 0


def compile_snapshot(
    program: DataflowProgram,
    *,
    config: DataflowCompileConfig,
    topology: GPUTopology,
    range_lengths: dict[Any, int | Sequence[int]],
    range_offsets: dict[Any, int | Sequence[int]] | None,
    block_size: int,
    task_extents: Sequence[int] | None,
    include_exit: bool,
    options: dict[str, Any],
    memory_attempt: str,
    prior_memory_candidates: tuple[DataflowMemoryCandidate, ...] = (),
    handoff_resource_fallback: bool = False,
) -> DataflowCompiledProgram:
    progress_enabled = dataflow_progress_enabled(options, environment=config.environment_dict())
    progress = DataflowProgressLogger(progress_enabled, prefix="[dataflow.compile]")
    compile_started = progress.start("compile")

    debug_handler = None
    if config.mode == DATAFLOW_COMPILE_MODE_DEBUG:
        from .experimental.debug_handlers import resolve_debug_handler

        debug_handler = resolve_debug_handler(config.handler_lowering)
        debug_handler.validate(program)
        handler_lowering = debug_handler.name
    else:
        handler_lowering = normalize_handler_lowering(config.handler_lowering)
        validate_handler_lowering(program, handler_lowering)
    queue_indexing = str(options.get("queue_indexing", "cta_rank"))
    scheduler_policy = str(options.get("scheduler_policy", "round_robin"))
    reduce_strategy = str(options.get("reduce_strategy", "all_at_once"))
    memory_policy = resolve_memory_policy(options.get("memory_policy"))
    if memory_attempt not in {DATAFLOW_MEMORY_SHARED, DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL}:
        raise ValueError(f"unsupported Dataflow memory planning attempt {memory_attempt!r}")
    force_hbm_comms = (
        True
        if memory_attempt == DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL
        else bool_option(options.get("force_hbm_comms", False), "force_hbm_comms")
    )
    partial_only = bool_option(options.get("partial_only", False), "partial_only")
    direct_leaf_acc = bool_option(options["direct_leaf_acc"], "direct_leaf_acc") if "direct_leaf_acc" in options else None
    streaming_tree_consumer = options.get("streaming_tree_consumer")
    cluster_task_assignment = options.get("cluster_task_assignment")
    task_coord_overrides = options.get("task_coord_overrides")
    stage_graph_task_weights = options.get("stage_graph_task_weights")
    stage_graph_cluster_assignment = options.get("stage_graph_cluster_assignment")
    skip_tiny_root_fragment = (
        bool_option(options["skip_tiny_root_fragment"], "skip_tiny_root_fragment") if "skip_tiny_root_fragment" in options else None
    )
    skip_tiny_root_fragment_blocks = options.get("skip_tiny_root_fragment_blocks")
    hbm_direct_global = memory_attempt == DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL
    consider_scratch_backed_slots = memory_attempt == DATAFLOW_MEMORY_SHARED and memory_policy.considers_scratch
    async_safe_cluster_handoff = bool_option(
        options.get("async_safe_cluster_handoff", False),
        "async_safe_cluster_handoff",
    )
    iter_range_bucket_option = options.get("iter_range_buckets")
    iter_range_buckets = normalize_iter_range_buckets(iter_range_bucket_option)
    iter_range_exact_option = options.get("iter_range_exact_lengths")
    iter_range_exact_lengths = normalize_iter_range_exact_lengths(iter_range_exact_option)
    iter_range_bucket_size = normalize_iter_range_bucket_size(
        options.get("iter_range_bucket_size"),
        fallback=block_size,
    )
    precision_plan = DataflowPrecisionPlan.from_dict(options["precision_plan"])
    expected_precision_plan = resolve_program_precision(
        program,
        config.semantic_config.precision,
    )
    if precision_plan != expected_precision_plan:
        raise RuntimeError("Dataflow precision plan does not match the current program contracts and semantic config")

    phase_started = progress.start("schedule")
    plan = schedule(
        program,
        topology=topology,
        range_lengths=range_lengths,
        range_offsets=range_offsets,
        block_size=block_size,
        task_extents=task_extents,
        include_exit=include_exit,
        scheduler_policy=scheduler_policy,
        reduce_strategy=reduce_strategy,
        force_hbm_comms=force_hbm_comms,
        partial_only=partial_only,
        direct_leaf_acc=direct_leaf_acc,
        streaming_tree_consumer=streaming_tree_consumer,
        cluster_task_assignment=cluster_task_assignment,
        skip_tiny_root_fragment=skip_tiny_root_fragment,
        skip_tiny_root_fragment_blocks=skip_tiny_root_fragment_blocks,
        iter_range_buckets=iter_range_buckets,
        iter_range_bucket_size=iter_range_bucket_size,
        iter_range_exact_lengths=iter_range_exact_lengths,
        task_coord_overrides=task_coord_overrides,
        stage_graph_task_weights=stage_graph_task_weights,
        stage_graph_cluster_assignment=stage_graph_cluster_assignment,
        range_resource_budget_bytes=(config.target_capabilities.max_dynamic_shared_memory),
        target_capabilities=config.target_capabilities,
        cross_handler_handoff_max_shared_memory_bytes=(
            0 if handoff_resource_fallback else config.target_capabilities.max_dynamic_shared_memory
        ),
        pic=bool_option(options.get("pic", False), "pic"),
        pic_dir=options.get("pic_dir", "schedule_res"),
        scheduler_config=config.scheduler_config,
    )
    if program.is_stage_graph:
        # Stage-graph all-gather keeps ordinary cluster destinations resident
        # across logical task generations and therefore needs an explicit
        # last-consumer release. Classic map/reduce schedules already resolve
        # reused inboxes in the memory planner (including HBM spill); adding a
        # release before that resolution would conflict with its lifetime
        # contract and create redundant ACK traffic.
        plan = insert_cluster_destination_reuse_handoffs(plan)
    progress.finish(
        "schedule",
        phase_started,
        (f"instructions={len(plan.instructions)} slots={len(plan.slots)} comms={len(plan.comms)} queues={len(plan.queues)}"),
    )
    joint_execution_required = plan.joint_execution_plan is not None
    joint_reduce_inplace_direct_output = consider_scratch_backed_slots and joint_reduce_inplace_direct_output_is_safe(program, plan)
    handoff_plans = tuple(handoff_plan for handoff_plan in plan.cross_handler_handoff_plans if handoff_plan.enabled)
    phase_started = progress.start("pack runtime plan")
    packed_plan = pack_instruction_plan(
        plan,
        reuse_hbm_flags=bool(options.get("reuse_hbm_flags", False)),
        hbm_direct_global=hbm_direct_global,
    )
    progress.finish("pack runtime plan", phase_started)
    phase_started = progress.start("build launch package")
    launch_package = build_launch_package(packed_plan)
    progress.finish(
        "build launch package",
        phase_started,
        f"shared_memory_bytes={launch_package.shared_memory_bytes}",
    )
    phase_started = progress.start("collect tensor args")
    tensor_arg_plan = collect_tensor_arg_plan(program)
    progress.finish("collect tensor args", phase_started, f"tensor_args={len(tensor_arg_plan.specs)}")
    phase_started = progress.start("build wrapper spec")
    wrapper_spec = build_wrapper_spec(
        packed_plan,
        kernel_name=options.get("wrapper_name", "dataflow_wrapper"),
        launch_package=launch_package,
        handler_lowering=handler_lowering,
        cluster_size=topology.cluster_size,
        tensor_arg_count=len(tensor_arg_plan.specs),
        queue_indexing=queue_indexing,
        launch_bound_threads=block_dim_x(options),
    )
    progress.finish("build wrapper spec", phase_started, f"kernel={wrapper_spec.kernel_name}")
    if debug_handler is not None:
        wrapper_spec = debug_handler.populate_wrapper_spec(wrapper_spec)
    primfunc_lowering = None
    memory_plan: DataflowMemoryPlan | None = None
    if handler_lowering == PRIMFUNC_HANDLER_LOWERING:
        lower_primfunc_handlers = config.lower_primfunc_handlers
        link_primfunc_handlers = config.link_primfunc_handlers
        handler_abi = None
        if link_primfunc_handlers:
            phase_started = progress.start("build PrimFunc handler ABI")
            handler_abi = build_handler_abi(program, wrapper_spec, tensor_arg_plan, plan)
            validate_linkable_primfunc_intermediate(handler_abi)
            progress.finish("build PrimFunc handler ABI", phase_started)
        legacy_cluster_communicate_slot_required = bool(
            consider_scratch_backed_slots
            and plan.joint_execution_plan is None
            and any(comm.kind is DataflowCommKind.CLUSTER_RECV for comm in plan.comms)
        )
        if legacy_cluster_communicate_slot_required:
            # A producer may push while the destination CTA is still running
            # the handler immediately before the receive.  Its internal
            # dynamic-shared allocations are not represented by logical slot
            # intervals, so a received value cannot safely alias the ordinary
            # handler arena.  Reserve one typed slot before GEMM resolution;
            # the common logical-GEMM planner can then select an unpadded
            # implementation when physical padding would consume that slot.
            async_safe_cluster_handoff = True
            options["async_safe_cluster_handoff"] = True
        primfunc_pass_configs = dict(options.get("primfunc_pass_configs") or {})
        if (
            legacy_cluster_communicate_slot_required
            and handler_abi is not None
            and config.target_capabilities.max_dynamic_shared_memory is not None
        ):
            scratch_base_offset = align_up(
                launch_package.shared_control_bytes,
                PRIMFUNC_DYNAMIC_SHARED_ALIGNMENT,
            )
            reserved_limit = int(config.target_capabilities.max_dynamic_shared_memory) - scratch_base_offset - handler_abi.slot_bytes
            if reserved_limit <= 0:
                raise DataflowMemoryPlanningError(
                    "Dataflow cluster communicate slot leaves no dynamic-shared "
                    "budget for PrimFunc handlers: "
                    f"target={config.target_capabilities.max_dynamic_shared_memory}, "
                    f"control={scratch_base_offset}, slot={handler_abi.slot_bytes}"
                )
            configured_limit = primfunc_pass_configs.get("tl.logical_gemm_max_shared_memory_bytes")
            primfunc_pass_configs["tl.logical_gemm_max_shared_memory_bytes"] = min(
                reserved_limit,
                reserved_limit if configured_limit is None else int(configured_limit),
            )
        phase_started = progress.start("lower PrimFunc handlers")
        primfunc_lowering = lower_program_handlers_to_primfuncs(
            program,
            wrapper_spec,
            tensor_arg_plan,
            plan=plan,
            target_capabilities=config.target_capabilities,
            lower_to_cuda=lower_primfunc_handlers,
            allow_reduce_direct_output=(not consider_scratch_backed_slots or joint_reduce_inplace_direct_output),
            allow_reduce_inplace_direct_output=(joint_reduce_inplace_direct_output),
            iter_range_bucket_size=iter_range_bucket_size,
            pass_configs=primfunc_pass_configs,
            thread_count_overrides=options.get("primfunc_thread_count_overrides"),
            precision_overrides=precision_plan.specialization_overrides(),
            map_input_scope=(None if hbm_direct_global and packed_plan_uses_hbm_direct_global(packed_plan) else "shared"),
        )
        resident_thread_overrides = joint_resident_handler_thread_overrides(
            program,
            plan,
            wrapper_spec,
            primfunc_lowering,
            explicit_overrides=options.get("primfunc_thread_count_overrides"),
            target_capabilities=config.target_capabilities,
        )
        if resident_thread_overrides:
            try:
                promoted_lowering = lower_program_handlers_to_primfuncs(
                    program,
                    wrapper_spec,
                    tensor_arg_plan,
                    plan=plan,
                    target_capabilities=config.target_capabilities,
                    lower_to_cuda=lower_primfunc_handlers,
                    allow_reduce_direct_output=(not consider_scratch_backed_slots or joint_reduce_inplace_direct_output),
                    allow_reduce_inplace_direct_output=(joint_reduce_inplace_direct_output),
                    iter_range_bucket_size=iter_range_bucket_size,
                    pass_configs=primfunc_pass_configs,
                    thread_count_overrides=resident_thread_overrides,
                    precision_overrides=precision_plan.specialization_overrides(),
                    map_input_scope=(None if hbm_direct_global and packed_plan_uses_hbm_direct_global(packed_plan) else "shared"),
                )
            except DataflowPrimFuncLoweringError:
                # A generated BodyIR may have a physical extent that is not
                # divisible by the resident CTA width.  The first lowering is
                # already a complete legal fallback; thread reuse is an
                # optimization, never an additional semantic requirement.
                pass
            else:
                if promoted_lowering.max_thread_count <= primfunc_lowering.max_thread_count:
                    primfunc_lowering = promoted_lowering
        progress.finish(
            "lower PrimFunc handlers",
            phase_started,
            f"dynamic_shared_bytes={primfunc_lowering.dynamic_shared_bytes}",
        )
        raise_block_dim_to_primfunc_thread_count(options, primfunc_lowering)
        wrapper_spec = replace(wrapper_spec, launch_bound_threads=block_dim_x(options))
        if link_primfunc_handlers:
            if not primfunc_lowering.cuda_source:
                raise ValueError("link_primfunc_handlers requires lower_primfunc_handlers=True")
            wrapper_spec = replace(
                wrapper_spec,
                tma_descriptors=primfunc_lowering.tma_descriptors,
            )
            phase_started = progress.start("plan memory placement")
            target_shared_limit = config.target_capabilities.max_dynamic_shared_memory
            baseline_shared_bytes = primfunc_launch_shared_memory_bytes(
                launch_package,
                primfunc_lowering.dynamic_shared_bytes,
                handoff_plans,
            )
            hbm_direct_slot_count = packed_slot_flag_count(
                packed_plan,
                DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
            )
            actual_placement = DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL if hbm_direct_global else DATAFLOW_MEMORY_SHARED
            placement_rejections: list[str] = []
            if hbm_direct_global and plan.slots and hbm_direct_slot_count == 0:
                placement_rejections.append("no_hbm_direct_global_slots")
            if joint_execution_required:
                placement_rejections.append("joint_execution_requires_materialized_communication_slots")
            baseline_candidate = make_memory_candidate(
                candidate_id=actual_placement,
                placement=actual_placement,
                shared_memory_bytes=baseline_shared_bytes,
                shared_slot_bytes=launch_package.shared_slot_bytes,
                primfunc_scratch_bytes=primfunc_lowering.dynamic_shared_bytes,
                hbm_direct_global_slot_count=hbm_direct_slot_count,
                target_shared_memory_limit=target_shared_limit,
                extra_rejection_reasons=placement_rejections,
            )
            candidates = [*prior_memory_candidates, baseline_candidate]
            scratch_plan = ScratchBackedSlotPlan(
                plan=plan,
                slot_count=0,
                slot_bytes=0,
                required_scratch_bytes=primfunc_lowering.dynamic_shared_bytes,
            )
            scratch_state: (
                tuple[
                    InstructionPlan,
                    PackedRuntimePlan,
                    DataflowLaunchPackage,
                ]
                | None
            ) = None
            if consider_scratch_backed_slots:
                scratch_plan = plan_scratch_backed_slots(
                    plan,
                    handler_abi,
                    primfunc_lowering,
                    wrapper_spec,
                    partial_only=partial_only,
                    async_safe_cluster_handoff=async_safe_cluster_handoff,
                    target_shared_memory_limit=target_shared_limit,
                )
                scratch_rejections: list[str] = []
                candidate_plan = scratch_plan.plan
                candidate_packed_plan = packed_plan
                candidate_launch_package = launch_package
                candidate_shared_bytes = baseline_shared_bytes
                if candidate_plan is plan or scratch_plan.slot_count == 0:
                    scratch_rejections.append("no_lifetime_safe_scratch_backed_slots")
                else:
                    selective_hbm_direct_slots = scratch_hbm_direct_global_slot_ids(candidate_plan)
                    candidate_packed_plan = pack_instruction_plan(
                        candidate_plan,
                        reuse_hbm_flags=bool(options.get("reuse_hbm_flags", False)),
                        hbm_direct_global=False,
                        hbm_direct_global_slot_ids=selective_hbm_direct_slots,
                    )
                    candidate_launch_package = build_launch_package(candidate_packed_plan)
                    candidate_shared_bytes = primfunc_launch_shared_memory_bytes(
                        candidate_launch_package,
                        scratch_plan.required_scratch_bytes,
                        handoff_plans,
                    )
                    if (
                        not joint_execution_required
                        and memory_policy.mode == DATAFLOW_MEMORY_AUTO
                        and candidate_shared_bytes >= baseline_shared_bytes
                    ):
                        scratch_rejections.append(
                            f"scratch_backed_layout_not_profitable: candidate={candidate_shared_bytes}, baseline={baseline_shared_bytes}"
                        )
                    scratch_state = (
                        candidate_plan,
                        candidate_packed_plan,
                        candidate_launch_package,
                    )
                candidates.append(
                    make_memory_candidate(
                        candidate_id=DATAFLOW_MEMORY_SCRATCH_BACKED,
                        placement=DATAFLOW_MEMORY_SCRATCH_BACKED,
                        shared_memory_bytes=candidate_shared_bytes,
                        shared_slot_bytes=candidate_launch_package.shared_slot_bytes,
                        primfunc_scratch_bytes=scratch_plan.required_scratch_bytes,
                        scratch_backed_slot_count=scratch_plan.slot_count,
                        scratch_backed_slot_bytes=(candidate_launch_package.scratch_backed_slot_bytes),
                        hbm_direct_global_slot_count=packed_slot_flag_count(
                            candidate_packed_plan,
                            DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
                        ),
                        target_shared_memory_limit=target_shared_limit,
                        extra_rejection_reasons=scratch_rejections,
                    )
                )

            try:
                memory_plan = select_memory_plan(
                    memory_policy,
                    candidates,
                    allow_deferred_hbm=(memory_policy.mode == DATAFLOW_MEMORY_AUTO and memory_attempt != DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL),
                )
            except DataflowMemoryPlanningError:
                if program.is_stage_graph and handoff_plans and not handoff_resource_fallback:
                    progress.finish(
                        "plan memory placement",
                        phase_started,
                        "retry=handoff_resource_fallback",
                    )
                    return compile_snapshot(
                        program,
                        config=config,
                        topology=topology,
                        range_lengths=range_lengths,
                        range_offsets=range_offsets,
                        block_size=block_size,
                        task_extents=task_extents,
                        include_exit=include_exit,
                        options=options,
                        memory_attempt=memory_attempt,
                        handoff_resource_fallback=True,
                    )
                raise
            if memory_plan is None:
                if program.is_stage_graph:
                    if handoff_plans and not handoff_resource_fallback:
                        progress.finish(
                            "plan memory placement",
                            phase_started,
                            "retry=handoff_resource_fallback",
                        )
                        return compile_snapshot(
                            program,
                            config=config,
                            topology=topology,
                            range_lengths=range_lengths,
                            range_offsets=range_offsets,
                            block_size=block_size,
                            task_extents=task_extents,
                            include_exit=include_exit,
                            options=options,
                            memory_attempt=memory_attempt,
                            handoff_resource_fallback=True,
                        )
                    raise DataflowMemoryPlanningError(
                        "Dataflow stage-graph memory placement exceeded the target limit and "
                        "cannot use the generic HBM direct-global fallback"
                    )
                progress.finish(
                    "plan memory placement",
                    phase_started,
                    "retry=hbm_direct_global",
                )
                return compile_snapshot(
                    program,
                    config=config,
                    topology=topology,
                    range_lengths=range_lengths,
                    range_offsets=range_offsets,
                    block_size=block_size,
                    task_extents=task_extents,
                    include_exit=include_exit,
                    options=options,
                    memory_attempt=DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL,
                    prior_memory_candidates=tuple(candidates),
                    handoff_resource_fallback=handoff_resource_fallback,
                )

            selected_placement = memory_plan.selected_candidate.placement
            scratch_backed_slots_enabled = selected_placement == DATAFLOW_MEMORY_SCRATCH_BACKED
            if scratch_backed_slots_enabled:
                if scratch_state is None:
                    raise RuntimeError("Dataflow memory planner selected scratch-backed placement without a materialized candidate")
                plan, packed_plan, launch_package = scratch_state
                wrapper_spec = build_wrapper_spec(
                    packed_plan,
                    kernel_name=options.get("wrapper_name", "dataflow_wrapper"),
                    launch_package=launch_package,
                    handler_lowering=handler_lowering,
                    cluster_size=topology.cluster_size,
                    tensor_arg_count=len(tensor_arg_plan.specs),
                    queue_indexing=queue_indexing,
                    launch_bound_threads=block_dim_x(options),
                )
                wrapper_spec = replace(
                    wrapper_spec,
                    tma_descriptors=primfunc_lowering.tma_descriptors,
                )
                handler_abi = build_handler_abi(
                    program,
                    wrapper_spec,
                    tensor_arg_plan,
                    plan,
                )
                primfunc_dynamic_shared_bytes = scratch_plan.required_scratch_bytes
            else:
                primfunc_dynamic_shared_bytes = primfunc_lowering.dynamic_shared_bytes
            progress.finish(
                "plan memory placement",
                phase_started,
                (f"selected={selected_placement} shared_memory_bytes={memory_plan.selected_candidate.shared_memory_bytes}"),
            )
            launch_package, wrapper_spec = extend_launch_for_primfunc_dynamic_shared(
                launch_package,
                wrapper_spec,
                primfunc_dynamic_shared_bytes,
                handoff_plans,
                cluster_inbox_offset=(scratch_plan.cluster_inbox_offset if scratch_backed_slots_enabled else 0),
                cluster_inbox_bytes=(scratch_plan.cluster_inbox_bytes if scratch_backed_slots_enabled else 0),
            )
            if scratch_backed_slots_enabled:
                wrapper_spec = replace(
                    wrapper_spec,
                    primfunc_handler_scratch_offsets=scratch_plan.handler_scratch_offsets,
                    primfunc_scratch_backed_iter_symbols=scratch_plan.iter_symbols,
                )
            wrapper_spec = replace(
                wrapper_spec,
                primfunc_use_global_slot_fields=(packed_plan_uses_hbm_direct_global(packed_plan)),
            )
            assert handler_abi is not None
            validate_contiguous_primfunc_map_inputs(
                primfunc_lowering,
                plan,
                packed_plan,
            )
            phase_started = progress.start("link PrimFunc handlers")
            linked_handlers = link_primfunc_handlers_for_wrapper(
                primfunc_lowering,
                wrapper_spec,
                handler_abi,
                semantic_config=config.semantic_config,
                terminal_finalize_variants=(queue_terminal_finalize_variants(plan)),
            )
            progress.finish("link PrimFunc handlers", phase_started)
            wrapper_spec = replace(
                wrapper_spec,
                handler_codegen_module=linked_handlers.codegen_module,
                handler_helper_source=linked_handlers.helper_source,
                handler_sources=linked_handlers.handler_sources,
            )
        elif lower_primfunc_handlers:
            # CUDA inspection still owns a physical handler scratch layout.
            # Plan and validate it exactly like the baseline executable
            # placement, while intentionally skipping ABI linking and wrapper
            # generation.
            phase_started = progress.start("plan inspection memory placement")
            target_shared_limit = config.target_capabilities.max_dynamic_shared_memory
            baseline_shared_bytes = primfunc_launch_shared_memory_bytes(
                launch_package,
                primfunc_lowering.dynamic_shared_bytes,
                handoff_plans,
            )
            hbm_direct_slot_count = packed_slot_flag_count(
                packed_plan,
                DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
            )
            actual_placement = DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL if hbm_direct_global else DATAFLOW_MEMORY_SHARED
            placement_rejections = (
                ("no_hbm_direct_global_slots",) if hbm_direct_global and plan.slots and hbm_direct_slot_count == 0 else ()
            )
            baseline_candidate = make_memory_candidate(
                candidate_id=actual_placement,
                placement=actual_placement,
                shared_memory_bytes=baseline_shared_bytes,
                shared_slot_bytes=launch_package.shared_slot_bytes,
                primfunc_scratch_bytes=primfunc_lowering.dynamic_shared_bytes,
                hbm_direct_global_slot_count=hbm_direct_slot_count,
                target_shared_memory_limit=target_shared_limit,
                extra_rejection_reasons=placement_rejections,
            )
            candidates = [*prior_memory_candidates, baseline_candidate]
            memory_plan = select_memory_plan(
                memory_policy,
                candidates,
                allow_deferred_hbm=(memory_policy.mode == DATAFLOW_MEMORY_AUTO and memory_attempt != DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL),
            )
            if memory_plan is None:
                if program.is_stage_graph:
                    if handoff_plans and not handoff_resource_fallback:
                        progress.finish(
                            "plan inspection memory placement",
                            phase_started,
                            "retry=handoff_resource_fallback",
                        )
                        return compile_snapshot(
                            program,
                            config=config,
                            topology=topology,
                            range_lengths=range_lengths,
                            range_offsets=range_offsets,
                            block_size=block_size,
                            task_extents=task_extents,
                            include_exit=include_exit,
                            options=options,
                            memory_attempt=memory_attempt,
                            handoff_resource_fallback=True,
                        )
                    raise DataflowMemoryPlanningError("Dataflow stage-graph inspection memory placement exceeded the target limit")
                progress.finish(
                    "plan inspection memory placement",
                    phase_started,
                    "retry=hbm_direct_global",
                )
                return compile_snapshot(
                    program,
                    config=config,
                    topology=topology,
                    range_lengths=range_lengths,
                    range_offsets=range_offsets,
                    block_size=block_size,
                    task_extents=task_extents,
                    include_exit=include_exit,
                    options=options,
                    memory_attempt=DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL,
                    prior_memory_candidates=tuple(candidates),
                    handoff_resource_fallback=handoff_resource_fallback,
                )
            launch_package, wrapper_spec = extend_launch_for_primfunc_dynamic_shared(
                launch_package,
                wrapper_spec,
                primfunc_lowering.dynamic_shared_bytes,
                handoff_plans,
            )
            wrapper_spec = replace(
                wrapper_spec,
                tma_descriptors=primfunc_lowering.tma_descriptors,
                primfunc_use_global_slot_fields=(hbm_direct_global and packed_plan_uses_hbm_direct_global(packed_plan)),
            )
            progress.finish(
                "plan inspection memory placement",
                phase_started,
                (
                    f"selected={memory_plan.selected_candidate.placement} "
                    f"shared_memory_bytes={memory_plan.selected_candidate.shared_memory_bytes}"
                ),
            )
        elif handoff_plans:
            # IR inspection does not yet know per-handler scratch bytes, but
            # the typed cross-handler arena is already a physical allocation.
            launch_package, wrapper_spec = extend_launch_for_primfunc_dynamic_shared(
                launch_package,
                wrapper_spec,
                0,
                handoff_plans,
            )
    if memory_plan is None:
        placement = DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL if hbm_direct_global else DATAFLOW_MEMORY_SHARED
        hbm_direct_slot_count = packed_slot_flag_count(
            packed_plan,
            DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
        )
        candidate = make_memory_candidate(
            candidate_id=placement,
            placement=placement,
            shared_memory_bytes=launch_package.shared_memory_bytes,
            shared_slot_bytes=launch_package.shared_slot_bytes,
            primfunc_scratch_bytes=0,
            hbm_direct_global_slot_count=hbm_direct_slot_count,
            target_shared_memory_limit=(config.target_capabilities.max_dynamic_shared_memory),
            extra_rejection_reasons=(
                ("no_hbm_direct_global_slots",) if hbm_direct_global and plan.slots and hbm_direct_slot_count == 0 else ()
            ),
        )
        memory_plan = select_memory_plan(memory_policy, (candidate,))
        assert memory_plan is not None
    if config.mode == DATAFLOW_COMPILE_MODE_INSPECT:
        wrapper_source = ""
    else:
        phase_started = progress.start("generate wrapper source")
        wrapper_source = generate_wrapper_source(wrapper_spec)
        progress.finish("generate wrapper source", phase_started, f"bytes={len(wrapper_source)}")
    compile_options = {
        "topology": {
            "sm_count": topology.sm_count,
            "cluster_size": topology.cluster_size,
        },
        "range_lengths": dict(range_lengths),
        "range_offsets": None if range_offsets is None else dict(range_offsets),
        "block_size": block_size,
        "task_extents": None if task_extents is None else tuple(task_extents),
        "include_exit": include_exit,
        "scheduler_policy": plan.scheduler_policy,
        "reduce_strategy": plan.reduce_strategy,
        "task_coord_overrides": task_coord_overrides,
        "stage_graph_task_weights": stage_graph_task_weights,
        "stage_graph_cluster_assignment": stage_graph_cluster_assignment,
        "partial_only": partial_only,
    }
    compile_options.update(options)
    compile_options["scheduler_policy"] = plan.scheduler_policy
    compile_options["reduce_strategy"] = plan.reduce_strategy
    compile_options["memory_policy"] = memory_policy.to_dict()
    compile_options["memory_plan"] = memory_plan.to_dict()
    compile_options["force_hbm_comms"] = force_hbm_comms
    compile_options["iter_range_buckets"] = iter_range_buckets
    compile_options["iter_range_bucket_size"] = iter_range_bucket_size if iter_range_buckets else None
    compile_options["iter_range_exact_lengths"] = iter_range_exact_lengths
    compile_options["progress"] = progress_enabled
    compile_options["handler_lowering"] = handler_lowering
    compile_options["queue_indexing"] = wrapper_spec.queue_indexing
    if handler_lowering == PRIMFUNC_HANDLER_LOWERING:
        compile_options["lower_primfunc_handlers"] = config.lower_primfunc_handlers
        compile_options["link_primfunc_handlers"] = config.link_primfunc_handlers
    progress.finish(
        "compile",
        compile_started,
        (
            f"kernel={wrapper_spec.kernel_name} instructions={len(plan.instructions)} "
            f"shared_memory_bytes={launch_package.shared_memory_bytes}"
        ),
    )
    return DataflowCompiledProgram(
        program=program,
        plan=plan,
        packed_plan=packed_plan,
        launch_package=launch_package,
        tensor_arg_plan=tensor_arg_plan,
        wrapper_spec=wrapper_spec,
        wrapper_source=wrapper_source,
        options=compile_options,
        compile_config=config,
        artifact_state=classify_artifact_state(config),
        memory_plan=memory_plan,
        primfunc_lowering=primfunc_lowering,
    )


def classify_artifact_state(config: DataflowCompileConfig) -> DataflowArtifactState:
    if config.mode == DATAFLOW_COMPILE_MODE_EXECUTABLE:
        return DataflowArtifactState(
            contract=DATAFLOW_ARTIFACT_PRODUCTION,
            executable=True,
        )

    if config.mode == DATAFLOW_COMPILE_MODE_DEBUG:
        from .experimental.debug_handlers import resolve_debug_handler

        debug_handler = resolve_debug_handler(config.handler_lowering)
        if not debug_handler.executable:
            return DataflowArtifactState(
                contract=DATAFLOW_ARTIFACT_DEBUG,
                executable=False,
                non_executable_reason=debug_handler.non_executable_reason,
            )
        return DataflowArtifactState(
            contract=DATAFLOW_ARTIFACT_DEBUG,
            executable=True,
        )

    assert config.mode == DATAFLOW_COMPILE_MODE_INSPECT
    lowered = config.lower_primfunc_handlers
    if not lowered:
        return DataflowArtifactState(
            contract=DATAFLOW_ARTIFACT_INSPECTION,
            executable=False,
            non_executable_reason=("PrimFunc handlers were retained as IR and were not lowered to CUDA"),
        )
    return DataflowArtifactState(
        contract=DATAFLOW_ARTIFACT_INSPECTION,
        executable=False,
        non_executable_reason=(
            "PrimFunc handlers were lowered for inspection; no executable wrapper was generated. compile with mode='executable'"
        ),
    )


def normalize_kernel_spec_topology(topology: GPUTopology | tuple[int, int] | int) -> GPUTopology:
    if isinstance(topology, GPUTopology):
        return topology
    if isinstance(topology, int):
        return GPUTopology(sm_count=topology)
    if isinstance(topology, tuple) and len(topology) == 2:
        return GPUTopology(sm_count=topology[0], cluster_size=topology[1])
    raise TypeError(f"DataflowKernelSpec.topology must be GPUTopology, sm_count int, or (sm_count, cluster_size) tuple, got {topology!r}")


def queue_slot_lifetimes_are_linear(
    plan: InstructionPlan,
    slots: Sequence[SlotPlan],
    *,
    physical_group: Callable[[SlotPlan, int], Any],
) -> bool:
    """Check queue-order lifetimes for slots mapped to one CTA-local address.

    A lifetime starts at its producer and ends at the last local consumer or
    source-dispatch boundary.  Equal boundaries are legal: an in-place handler
    may consume the old value and produce its alias at the same instruction.
    Asynchronous source reads are closed by the runtime physical-range fence
    after dispatch.
    """

    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    queue_position_by_instruction = {
        instruction.instruction_id: (cta_id, position)
        for cta_id, queue in plan.queues.items()
        for position, instruction in enumerate(queue)
    }
    consumers_by_slot: dict[int, list[int]] = {}
    for instruction in plan.instructions:
        for slot_id in instruction.input_slots:
            consumers_by_slot.setdefault(slot_id, []).append(instruction.instruction_id)
    source_dispatches_by_slot: dict[int, list[int]] = {}
    for comm in plan.comms:
        source_dispatches_by_slot.setdefault(comm.source_slot_id, []).append(comm.resolved_dispatch_instruction_id)

    intervals_by_group: dict[Any, list[tuple[int, int, int]]] = {}
    for slot in slots:
        if slot.producer_instruction_id is None:
            return False
        producer = instructions_by_id.get(slot.producer_instruction_id)
        producer_position = queue_position_by_instruction.get(slot.producer_instruction_id)
        if producer is None or producer.sm_id is None or producer_position is None or producer_position[0] != producer.sm_id:
            return False
        owner_cta = producer.sm_id
        begin = producer_position[1]
        end = begin
        for instruction_id in (
            *consumers_by_slot.get(slot.slot_id, ()),
            *source_dispatches_by_slot.get(slot.slot_id, ()),
        ):
            position = queue_position_by_instruction.get(instruction_id)
            if position is not None and position[0] == owner_cta:
                end = max(end, position[1])
        intervals_by_group.setdefault(physical_group(slot, owner_cta), []).append((begin, end, slot.slot_id))

    for intervals in intervals_by_group.values():
        previous_end = -1
        for begin, end, _slot_id in sorted(intervals):
            if begin < previous_end:
                return False
            previous_end = max(previous_end, end)
    return True


def lifetime_safe_scratch_storage_ids(
    plan: InstructionPlan,
    *,
    partial_only: bool,
) -> tuple[set[int], set[int]]:
    """Return scratch storage classes with a proven CTA-local linear lifetime."""

    producer_by_slot = {
        instruction.output_slot: instruction
        for instruction in plan.instructions
        if instruction.opcode is DataflowOpcode.ITER and instruction.output_slot is not None
    }
    consumers_by_slot: dict[int, list[Instruction]] = {}
    for instruction in plan.instructions:
        for slot_id in instruction.input_slots:
            consumers_by_slot.setdefault(slot_id, []).append(instruction)

    partial_storage_ids = {
        slot.shared_storage_id
        for slot in plan.slots
        if (
            slot.role == "partial"
            and slot.shared_storage_id is not None
            and (
                slot_is_immediately_reduced_streaming_partial(
                    slot,
                    producer_by_slot=producer_by_slot,
                    consumers_by_slot=consumers_by_slot,
                )
                or (partial_only and slot.slot_id in producer_by_slot and not consumers_by_slot.get(slot.slot_id))
            )
        )
    }
    streaming_acc_storage_ids = {
        slot.shared_storage_id for slot in plan.slots if slot.role == "streaming_acc" and slot.shared_storage_id is not None
    }
    partial_slots = tuple(slot for slot in plan.slots if slot.shared_storage_id in partial_storage_ids)
    if partial_slots and not queue_slot_lifetimes_are_linear(
        plan,
        partial_slots,
        physical_group=lambda _slot, owner_cta: (owner_cta, "partial"),
    ):
        partial_storage_ids = set()
    streaming_acc_slots = tuple(slot for slot in plan.slots if slot.shared_storage_id in streaming_acc_storage_ids)
    if streaming_acc_slots and not queue_slot_lifetimes_are_linear(
        plan,
        streaming_acc_slots,
        physical_group=lambda _slot, owner_cta: (
            owner_cta,
            "streaming_acc",
        ),
    ):
        streaming_acc_storage_ids = set()
    return partial_storage_ids, streaming_acc_storage_ids


def scratch_backed_candidate_rejection_reason(
    plan: InstructionPlan,
    *,
    partial_only: bool,
) -> str | None:
    partial_storage_ids, streaming_acc_storage_ids = lifetime_safe_scratch_storage_ids(
        plan,
        partial_only=partial_only,
    )
    if partial_storage_ids or streaming_acc_storage_ids:
        return None
    return "resource constraint: scratch-backed placement requires at least one CTA-local storage class with a bounded linear lifetime"


def joint_outbox_can_alias_ordinary_scratch(
    plan: InstructionPlan,
    *,
    communicate_slot_bytes: int,
) -> bool:
    """Prove that joint outboxes may reuse ordinary output storage.

    A producer already writes its value to scratch-backed ordinary storage.
    Inboxes deliberately use a disjoint interval, and lowering caps every push
    at the last queue boundary before a later output writer could reuse the
    source.  The runtime source-read fence then closes the physical lifetime.
    This verifier checks that every scheduled output has a valid value owner;
    it does not assume modeled CTA timings are exact.
    """

    joint_plan = plan.joint_execution_plan
    if joint_plan is None or communicate_slot_bytes <= 0:
        return False
    schedule = joint_plan.require_valid(topology=plan.topology).schedule
    scratch_slots = tuple(slot for slot in plan.slots if slot.scratch_backed)
    if not queue_slot_lifetimes_are_linear(
        plan,
        scratch_slots,
        physical_group=lambda slot, owner_cta: (
            owner_cta,
            slot.scratch_offset,
        ),
    ):
        return False
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    nodes_by_id = {node.node_id: node for node in joint_plan.nodes}
    saw_output = False
    for event in schedule.events:
        if event.node_id is None or event.cta_id is None:
            continue
        instruction = instructions_by_id.get(event.node_id)
        if instruction is None or instruction.output_slot is None:
            continue
        saw_output = True
        node = nodes_by_id.get(event.node_id)
        if node is None or node.output_value is None:
            return False
    return saw_output


def joint_outbox_ordinary_scratch_offset(
    plan: InstructionPlan,
    *,
    communicate_slot_bytes: int,
) -> int | None:
    """Return the common scratch offset of every joint send producer."""

    if not joint_outbox_can_alias_ordinary_scratch(
        plan,
        communicate_slot_bytes=communicate_slot_bytes,
    ):
        return None
    joint_plan = plan.joint_execution_plan
    assert joint_plan is not None
    producer_node_ids = {
        transfer.producer_node_id for transfer in joint_plan.schedule.transfers if transfer.producer_slot_epoch is not None
    }
    if not producer_node_ids:
        return None
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    offsets: set[int] = set()
    for node_id in producer_node_ids:
        instruction = instructions_by_id.get(node_id)
        if instruction is None or instruction.output_slot is None:
            return None
        output_slot = slots_by_id.get(instruction.output_slot)
        output_range = None if output_slot is None else scratch_slot_range(output_slot)
        if output_range is None or output_range[1] - output_range[0] > communicate_slot_bytes:
            return None
        offsets.add(output_range[0])
    if len(offsets) != 1:
        return None
    return offsets.pop()


def joint_permanent_inbox_needs_handler_disjoint_storage(
    plan: InstructionPlan,
) -> bool:
    """Return whether a planned permanent arrival overlaps ordinary compute.

    A permanent inbox may reuse otherwise idle handler scratch only when every
    push is ordered after the preceding non-reducer handlers on its destination
    CTA.  The joint schedule may intentionally issue sooner so communication
    overlaps those handlers.  Such a plan needs truly disjoint physical storage;
    a runtime acknowledgement would turn the planned push back into a blocking
    send and defeat the schedule.
    """

    joint_plan = plan.joint_execution_plan
    if joint_plan is None:
        return False
    schedule = joint_plan.require_valid(topology=plan.topology).schedule
    handler_event_by_node = {event.node_id: event for event in schedule.events if event.node_id is not None and event.cta_id is not None}
    queue_position_by_node = {
        instruction.instruction_id: (cta_id, position)
        for cta_id, queue in plan.queues.items()
        for position, instruction in enumerate(queue)
    }
    for transfer in schedule.transfers:
        if (
            transfer.kind is not DataflowTransportKind.CLUSTER_PUSH
            or transfer.consumer_storage_kind is not DataflowCommStorageKind.PERMANENT
        ):
            continue
        target_position = queue_position_by_node.get(transfer.consumer_node_id)
        if target_position is None:
            continue
        consumer_cta, target_index = target_position
        issue = schedule.event(transfer.producer_issue_event_id)
        for instruction in plan.queue(consumer_cta)[:target_index]:
            if instruction.opcode in {
                DataflowOpcode.REDUCE,
                DataflowOpcode.REDUCE_UPDATE,
                DataflowOpcode.FINALIZE,
            }:
                continue
            handler_event = handler_event_by_node.get(instruction.instruction_id)
            if handler_event is not None and issue.start_us + 1e-9 < handler_event.end_us:
                return True
    return False


def plan_scratch_backed_slots(
    plan: InstructionPlan,
    handler_abi: DataflowHandlerABI,
    primfunc_lowering: DataflowPrimFuncLoweringResult,
    wrapper_spec: DataflowWrapperSpec,
    *,
    partial_only: bool = False,
    async_safe_cluster_handoff: bool = False,
    target_shared_memory_limit: int | None = None,
) -> ScratchBackedSlotPlan:
    slot_bytes = handler_abi.slot_bytes
    if slot_bytes <= 0 or plan.reduce_strategy not in {"streaming", "streaming_tree"}:
        return ScratchBackedSlotPlan(
            plan=plan,
            slot_count=0,
            slot_bytes=0,
            required_scratch_bytes=primfunc_lowering.dynamic_shared_bytes,
        )

    iter_handler_ids = {handler.handler_id for handler in wrapper_spec.handlers if handler.operator_kind == "iter"}
    reduce_handler_ids = {handler.handler_id for handler in wrapper_spec.handlers if handler.operator_kind == "reduce"}
    finalize_handler_ids = {handler.handler_id for handler in wrapper_spec.handlers if handler.operator_kind == "finalize"}
    iter_handler_scratch = max(
        (handler.dynamic_shared_bytes for handler in primfunc_lowering.handlers if handler.handler_id in iter_handler_ids),
        default=0,
    )
    reduce_handler_scratch = max(
        (handler.dynamic_shared_bytes for handler in primfunc_lowering.handlers if handler.handler_id in reduce_handler_ids),
        default=0,
    )
    finalize_handler_scratch = max(
        (handler.dynamic_shared_bytes for handler in primfunc_lowering.handlers if handler.handler_id in finalize_handler_ids),
        default=0,
    )
    if iter_handler_scratch < slot_bytes:
        return ScratchBackedSlotPlan(
            plan=plan,
            slot_count=0,
            slot_bytes=0,
            required_scratch_bytes=primfunc_lowering.dynamic_shared_bytes,
        )

    partial_scratch_storage_ids, streaming_acc_scratch_storage_ids = lifetime_safe_scratch_storage_ids(
        plan,
        partial_only=partial_only,
    )
    if not partial_scratch_storage_ids and not streaming_acc_scratch_storage_ids:
        return ScratchBackedSlotPlan(
            plan=plan,
            slot_count=0,
            slot_bytes=0,
            required_scratch_bytes=primfunc_lowering.dynamic_shared_bytes,
        )

    planned_slots: list[SlotPlan] = []
    streaming_acc_scratch_offset = 0 if not partial_scratch_storage_ids else slot_bytes
    for slot in plan.slots:
        if slot.shared_storage_id in partial_scratch_storage_ids:
            planned_slots.append(
                replace(
                    slot,
                    scratch_backed=True,
                    scratch_offset=0,
                )
            )
        elif slot.shared_storage_id in streaming_acc_scratch_storage_ids:
            planned_slots.append(
                replace(
                    slot,
                    scratch_backed=True,
                    scratch_offset=streaming_acc_scratch_offset,
                )
            )
        else:
            planned_slots.append(slot)

    planned_plan = replace(plan, slots=tuple(planned_slots))
    output_shift = scratch_backed_iter_output_shift(handler_abi)
    scratch_slot_bytes = max(
        (scratch_range[1] for slot in planned_plan.slots if (scratch_range := scratch_slot_range(slot)) is not None),
        default=0,
    )
    joint_outbox_alias_offset = joint_outbox_ordinary_scratch_offset(
        planned_plan,
        communicate_slot_bytes=slot_bytes,
    )
    joint_outbox_scratch_alias = joint_outbox_alias_offset is not None
    disjoint_joint_permanent_inbox_requested = bool(
        joint_outbox_scratch_alias and joint_permanent_inbox_needs_handler_disjoint_storage(planned_plan)
    )
    # A disjoint permanent inbox begins no earlier than the end of the physical
    # handler arena.  If even this lower bound exceeds the target, retain the
    # compact overlap layout and its explicit destination-release gate.  The
    # gate is a legal fallback; an optimization-only inbox must not evict the
    # whole joint plan to HBM.
    disjoint_joint_permanent_inbox = bool(
        disjoint_joint_permanent_inbox_requested
        and (
            target_shared_memory_limit is None
            or align_up(
                primfunc_lowering.dynamic_shared_bytes,
                DATAFLOW_SLOT_ALIGNMENT,
            )
            + slot_bytes
            <= target_shared_memory_limit
        )
    )
    # When the schedule proves that an outgoing value stays in its ordinary
    # output slot until source-read completion, reserve only a disjoint inbox
    # immediately above the ordinary slots.  Reducer scratch starts after both
    # values so the consumer can read its accumulator and pushed input at once.
    joint_alias_inbox_offset = align_up(
        scratch_slot_bytes,
        DATAFLOW_SLOT_ALIGNMENT,
    )
    handler_slot_bytes = (
        max(scratch_slot_bytes, joint_alias_inbox_offset + slot_bytes)
        if joint_outbox_scratch_alias and not disjoint_joint_permanent_inbox
        else scratch_slot_bytes
    )
    reduce_scratch_offset = handler_slot_bytes
    has_multi_input_finalize = any(
        instruction.opcode is DataflowOpcode.FINALIZE and len(instruction.input_slots) > 1 for instruction in planned_plan.instructions
    )
    finalize_scratch_offset = align_up(
        (primfunc_lowering.dynamic_shared_bytes if has_multi_input_finalize else handler_slot_bytes),
        PRIMFUNC_DYNAMIC_SHARED_ALIGNMENT,
    )
    handler_offsets = []
    iter_symbols = []
    for handler in primfunc_lowering.handlers:
        if handler.handler_id in iter_handler_ids:
            handler_offsets.append((handler.device_symbol, output_shift))
            iter_symbols.append(handler.device_symbol)
        elif handler.handler_id in reduce_handler_ids:
            handler_offsets.append((handler.device_symbol, reduce_scratch_offset))
        elif handler.handler_id in finalize_handler_ids and finalize_handler_scratch > 0:
            handler_offsets.append((handler.device_symbol, finalize_scratch_offset))
    handler_scratch_bytes = max(
        primfunc_lowering.dynamic_shared_bytes,
        handler_slot_bytes,
        output_shift + iter_handler_scratch,
        reduce_scratch_offset + reduce_handler_scratch,
        finalize_scratch_offset + finalize_handler_scratch,
    )
    transient_prefetch_offset = 0
    transient_prefetch_bytes = 0
    transient_storage_lanes = (
        ()
        if planned_plan.joint_execution_plan is None
        else tuple(
            transfer.consumer_storage_lane
            for transfer in planned_plan.joint_execution_plan.schedule.transfers
            if transfer.consumer_storage_kind is DataflowCommStorageKind.TRANSIENT_PREFETCH
        )
    )
    if transient_storage_lanes:
        transient_prefetch_offset = align_up(
            reduce_scratch_offset + reduce_handler_scratch,
            DATAFLOW_SLOT_ALIGNMENT,
        )
        transient_prefetch_bytes = (max(transient_storage_lanes) + 1) * slot_bytes
    cluster_inbox_offset = 0
    cluster_inbox_bytes = 0
    planned_slots_by_id = {slot.slot_id: slot for slot in planned_plan.slots}
    planned_instructions_by_id = {instruction.instruction_id: instruction for instruction in planned_plan.instructions}
    scratch_cluster_inbox_required = any(
        comm.kind is DataflowCommKind.CLUSTER_RECV
        and (received_slot := planned_slots_by_id.get(comm.target_slot_id)) is not None
        and (received_range := scratch_slot_range(received_slot)) is not None
        and (target := planned_instructions_by_id.get(comm.target_instruction_id)) is not None
        and any(
            input_slot_id != comm.target_slot_id
            and (local_slot := planned_slots_by_id.get(input_slot_id)) is not None
            and (local_range := scratch_slot_range(local_slot)) is not None
            and ranges_overlap(received_range, local_range)
            for input_slot_id in target.input_slots
        )
        for comm in planned_plan.comms
    )
    joint_slot_required = bool(planned_plan.joint_execution_plan is not None and planned_plan.joint_execution_plan.schedule.transfers)
    standalone_cluster_slot_required = bool(async_safe_cluster_handoff or scratch_cluster_inbox_required)
    if standalone_cluster_slot_required or joint_slot_required:
        alias_joint_outbox = bool(joint_slot_required and joint_outbox_scratch_alias)
        candidate_offset = (
            align_up(
                max(
                    handler_scratch_bytes,
                    transient_prefetch_offset + transient_prefetch_bytes,
                ),
                DATAFLOW_SLOT_ALIGNMENT,
            )
            if alias_joint_outbox and disjoint_joint_permanent_inbox
            else joint_alias_inbox_offset
            if alias_joint_outbox
            else align_up(handler_scratch_bytes, DATAFLOW_SLOT_ALIGNMENT)
        )
        if transient_prefetch_bytes > 0 and ranges_overlap(
            (
                transient_prefetch_offset,
                transient_prefetch_offset + transient_prefetch_bytes,
            ),
            (candidate_offset, candidate_offset + slot_bytes),
        ):
            raise DataflowMemoryPlanningError(
                "Dataflow joint HBM prefetch storage overlaps the permanent "
                "communicate slot: "
                f"prefetch=[{transient_prefetch_offset}, "
                f"{transient_prefetch_offset + transient_prefetch_bytes}), "
                f"permanent=[{candidate_offset}, "
                f"{candidate_offset + slot_bytes})"
            )
        if not joint_slot_required:
            planned_plan = materialize_cluster_inbox_slots(
                planned_plan,
                cluster_inbox_offset=candidate_offset,
            )
        else:
            planned_plan = materialize_joint_communication_slots(
                planned_plan,
                communicate_slot_offset=candidate_offset,
                communicate_slot_bytes=slot_bytes,
                outbox_slot_offset=(joint_outbox_alias_offset if alias_joint_outbox else None),
                permanent_inbox_overlaps_handler_scratch=(alias_joint_outbox and not disjoint_joint_permanent_inbox),
                transient_prefetch_offset=(None if transient_prefetch_bytes == 0 else transient_prefetch_offset),
            )
        inbox_slots = tuple(
            slot
            for slot in planned_plan.slots
            if slot.role
            in {
                "cluster_inbox",
                "cluster_gated_inbox",
                "hbm_spill_inbox",
                "joint_comm_inbox",
                "joint_comm_outbox",
            }
        )
        if inbox_slots:
            cluster_inbox_offset = candidate_offset
            cluster_inbox_bytes = max(
                scratch_range[1] - candidate_offset
                for slot in inbox_slots
                if (scratch_range := scratch_slot_range(slot)) is not None and scratch_range[0] == candidate_offset
            )
            if not joint_slot_required:
                verify_cluster_inbox_contract(
                    planned_plan,
                    handler_scratch_bytes=handler_scratch_bytes,
                    cluster_inbox_offset=cluster_inbox_offset,
                    cluster_inbox_bytes=cluster_inbox_bytes,
                )
            else:
                verify_joint_communication_slot_contract(
                    planned_plan,
                    handler_scratch_bytes=handler_scratch_bytes,
                    communicate_slot_offset=cluster_inbox_offset,
                    communicate_slot_bytes=cluster_inbox_bytes,
                    outbox_slot_offset=(joint_outbox_alias_offset if alias_joint_outbox else None),
                )
    required_scratch_bytes = max(
        handler_scratch_bytes,
        cluster_inbox_offset + cluster_inbox_bytes,
        transient_prefetch_offset + transient_prefetch_bytes,
    )
    slot_count = sum(slot.scratch_backed for slot in planned_plan.slots)

    return ScratchBackedSlotPlan(
        plan=planned_plan,
        slot_count=slot_count,
        slot_bytes=slot_bytes,
        required_scratch_bytes=required_scratch_bytes,
        handler_scratch_offsets=tuple(handler_offsets),
        iter_symbols=tuple(iter_symbols),
        cluster_inbox_offset=cluster_inbox_offset,
        cluster_inbox_bytes=cluster_inbox_bytes,
        transient_prefetch_offset=transient_prefetch_offset,
        transient_prefetch_bytes=transient_prefetch_bytes,
    )


def materialize_joint_communication_slots(
    plan: InstructionPlan,
    *,
    communicate_slot_offset: int,
    communicate_slot_bytes: int,
    outbox_slot_offset: int | None = None,
    permanent_inbox_overlaps_handler_scratch: bool = False,
    transient_prefetch_offset: int | None = None,
) -> InstructionPlan:
    """Bind joint schedule epochs to per-CTA outbox/inbox storage."""

    joint_plan = plan.joint_execution_plan
    if joint_plan is None:
        return plan
    schedule = joint_plan.require_valid(topology=plan.topology).schedule
    materialized_transfers = tuple(schedule.transfers)
    if not materialized_transfers:
        return plan
    retained_transfers = tuple(transfer for transfer in materialized_transfers if transfer.retained_input_copy_event_id is not None)
    if communicate_slot_offset < 0 or communicate_slot_offset % DATAFLOW_SLOT_ALIGNMENT:
        raise DataflowMemoryPlanningError("Dataflow joint communicate slot must use a non-negative aligned offset")
    if outbox_slot_offset is not None and (outbox_slot_offset < 0 or outbox_slot_offset % DATAFLOW_SLOT_ALIGNMENT):
        raise DataflowMemoryPlanningError("Dataflow joint outbox alias must use a non-negative aligned offset")
    requires_transient_prefetch = any(
        transfer.consumer_storage_kind is DataflowCommStorageKind.TRANSIENT_PREFETCH for transfer in materialized_transfers
    )
    if requires_transient_prefetch and transient_prefetch_offset is None:
        raise DataflowMemoryPlanningError("joint execution schedule requires transient HBM prefetch storage")
    if transient_prefetch_offset is not None and (transient_prefetch_offset < 0 or transient_prefetch_offset % DATAFLOW_SLOT_ALIGNMENT):
        raise DataflowMemoryPlanningError("Dataflow transient prefetch storage must use a non-negative aligned offset")

    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    if set(slots_by_id) != set(range(len(plan.slots))):
        raise DataflowMemoryPlanningError("Dataflow joint communicate-slot lowering requires dense logical slot ids")

    if requires_transient_prefetch:
        handler_events = tuple(event for event in schedule.events if event.node_id is not None)
        for interval in schedule.comm_slot_intervals:
            if interval.storage_kind is not DataflowCommStorageKind.TRANSIENT_PREFETCH:
                continue
            for event in handler_events:
                if event.cta_id != interval.cta_id:
                    continue
                if event.end_us <= interval.begin_us + 1e-9 or event.start_us + 1e-9 >= interval.end_us:
                    continue
                instruction = instructions_by_id[event.node_id]
                if instruction.opcode not in {
                    DataflowOpcode.REDUCE,
                    DataflowOpcode.REDUCE_UPDATE,
                    DataflowOpcode.FINALIZE,
                }:
                    raise DataflowMemoryPlanningError(
                        "transient HBM prefetch lifetime crosses a handler whose "
                        f"scratch is not reusable: instruction={instruction.instruction_id}, "
                        f"opcode={instruction.opcode.value}"
                    )

    materialized_edges = {
        (
            transfer.producer_node_id,
            transfer.consumer_node_id,
            transfer.value_id,
        )
        for transfer in materialized_transfers
    }
    async_send_kinds = {
        DataflowCommKind.CLUSTER_SEND,
        DataflowCommKind.HBM_SEND,
    }
    async_recv_kinds = {
        DataflowCommKind.CLUSTER_RECV,
        DataflowCommKind.HBM_RECV,
    }
    sends_by_edge: dict[tuple[int, int, int], CommPlan] = {}
    recvs_by_edge: dict[tuple[int, int, int], CommPlan] = {}
    retained_comms: list[CommPlan] = []
    for comm in plan.comms:
        edge = (
            comm.source_instruction_id,
            comm.target_instruction_id,
            comm.source_slot_id,
        )
        if edge not in materialized_edges:
            if comm.kind is not DataflowCommKind.CLUSTER_RELEASE:
                retained_comms.append(comm)
        elif comm.kind in async_send_kinds:
            if edge in sends_by_edge:
                raise DataflowMemoryPlanningError(f"joint execution edge {edge!r} has duplicate sends")
            sends_by_edge[edge] = comm
        elif comm.kind in async_recv_kinds:
            if edge in recvs_by_edge:
                raise DataflowMemoryPlanningError(f"joint execution edge {edge!r} has duplicate receives")
            recvs_by_edge[edge] = comm
        elif comm.kind is not DataflowCommKind.CLUSTER_RELEASE:
            retained_comms.append(comm)

    intervals_by_resource: dict[tuple[int, DataflowCommStorageKind, int], list[Any]] = {}
    inbox_interval_by_transfer: dict[int, Any] = {}
    materialized_transfer_ids = {transfer.transfer_id for transfer in materialized_transfers}
    for interval in schedule.comm_slot_intervals:
        if interval.transfer_id is not None and interval.transfer_id not in materialized_transfer_ids:
            continue
        intervals_by_resource.setdefault(
            (
                interval.cta_id,
                interval.storage_kind,
                interval.storage_lane,
            ),
            [],
        ).append(interval)
        if interval.mode is DataflowCommSlotMode.INBOX:
            if interval.transfer_id is None:
                raise DataflowMemoryPlanningError("joint inbox interval is missing its transfer id")
            inbox_interval_by_transfer[interval.transfer_id] = interval
    for intervals in intervals_by_resource.values():
        intervals.sort(key=lambda item: item.epoch)

    previous_interval_by_transfer: dict[int, Any] = {}
    for (
        _cta_id,
        _storage_kind,
        _storage_lane,
    ), intervals in intervals_by_resource.items():
        previous = None
        for interval in intervals:
            if interval.mode is DataflowCommSlotMode.INBOX:
                assert interval.transfer_id is not None
                if previous is not None:
                    previous_interval_by_transfer[interval.transfer_id] = previous
            previous = interval

    queue_position_by_node = {
        instruction.instruction_id: (cta_id, position)
        for cta_id, queue in plan.queues.items()
        for position, instruction in enumerate(queue)
    }
    handler_event_by_node = {
        event.node_id: event
        for event in schedule.events
        if event.node_id is not None and event.cta_id is not None and event.node_id in instructions_by_id
    }
    retained_copy_dispatch_node_by_transfer: dict[int, int] = {}
    for transfer in retained_transfers:
        assert transfer.retained_input_copy_event_id is not None
        copy_event = schedule.event(transfer.retained_input_copy_event_id)
        if copy_event.kind is not DataflowScheduleEventKind.INBOX_RETAIN_COPY:
            raise DataflowMemoryPlanningError(f"joint retained transfer {transfer.transfer_id} has an invalid copy event")
        target_position = queue_position_by_node.get(transfer.consumer_node_id)
        if target_position is None:
            raise DataflowMemoryPlanningError(f"joint retained transfer {transfer.transfer_id} has an unqueued consumer")
        consumer_cta, target_index = target_position
        candidates = tuple(
            (
                handler_event_by_node[instruction.instruction_id].end_us,
                position,
                instruction.instruction_id,
            )
            for position, instruction in enumerate(plan.queue(consumer_cta)[:target_index])
            if instruction.instruction_id in handler_event_by_node
            and handler_event_by_node[instruction.instruction_id].end_us <= copy_event.start_us + 1e-9
        )
        if not candidates:
            raise DataflowMemoryPlanningError(
                f"joint retained transfer {transfer.transfer_id} has no preceding CTA queue boundary for its modeled copy"
            )
        retained_copy_dispatch_node_by_transfer[transfer.transfer_id] = max(candidates)[2]

    def interval_release_node_id(interval: Any) -> int | None:
        release = schedule.event(interval.end_event_id)
        if release.node_id is not None:
            return release.node_id
        if release.kind is DataflowScheduleEventKind.INBOX_RETAIN_COPY and release.transfer_id is not None:
            return retained_copy_dispatch_node_by_transfer.get(release.transfer_id)
        return None

    cluster_release_gate_node_by_transfer: dict[int, int] = {}
    for transfer in materialized_transfers:
        if transfer.kind is not DataflowTransportKind.CLUSTER_PUSH:
            continue
        gate_node_ids: list[int] = []
        previous_interval = previous_interval_by_transfer.get(transfer.transfer_id)
        if previous_interval is not None:
            previous_release_node_id = interval_release_node_id(previous_interval)
            if previous_release_node_id is None:
                raise DataflowMemoryPlanningError(f"joint transfer {transfer.transfer_id} has no handler-owned communicate-slot release")
            gate_node_ids.append(previous_release_node_id)
        if transfer.consumer_storage_kind is DataflowCommStorageKind.TRANSIENT_PREFETCH or permanent_inbox_overlaps_handler_scratch:
            target_position = queue_position_by_node.get(transfer.consumer_node_id)
            if target_position is None:
                raise DataflowMemoryPlanningError(f"joint scratch-aliased cluster transfer {transfer.transfer_id} has an unqueued consumer")
            consumer_cta, target_index = target_position
            for instruction in plan.queue(consumer_cta)[:target_index]:
                if instruction.opcode not in {
                    DataflowOpcode.REDUCE,
                    DataflowOpcode.REDUCE_UPDATE,
                    DataflowOpcode.FINALIZE,
                }:
                    gate_node_ids.append(instruction.instruction_id)
        if gate_node_ids:
            cluster_release_gate_node_by_transfer[transfer.transfer_id] = max(
                gate_node_ids,
                key=lambda node_id: queue_position_by_node[node_id][1],
            )

    next_slot_id = len(plan.slots)
    staged_slots: list[SlotPlan] = []
    inbox_slot_by_transfer: dict[int, int] = {}
    for transfer in materialized_transfers:
        template = slots_by_id.get(transfer.value_id)
        if template is None:
            raise DataflowMemoryPlanningError(f"joint transfer {transfer.transfer_id} has no value slot {transfer.value_id}")
        previous_interval = previous_interval_by_transfer.get(transfer.transfer_id)
        if previous_interval is not None and previous_interval.mode is DataflowCommSlotMode.OUTBOX:
            raise DataflowMemoryPlanningError(
                "joint outbox-to-inbox transition requires an explicit source-complete "
                f"phase action; cta={transfer.consumer_cta}, "
                f"transfer={transfer.transfer_id}"
            )
        gated_cluster_push = transfer.transfer_id in cluster_release_gate_node_by_transfer
        if transfer.consumer_storage_kind is DataflowCommStorageKind.TRANSIENT_PREFETCH:
            if transient_prefetch_offset is None:
                raise DataflowMemoryPlanningError("transient prefetch transfer has no physical scratch offset")
            role = "cluster_gated_inbox" if gated_cluster_push else "joint_prefetch_inbox"
            scratch_offset = transient_prefetch_offset + transfer.consumer_storage_lane * communicate_slot_bytes
        else:
            role = "cluster_gated_inbox" if gated_cluster_push else "joint_comm_inbox"
            scratch_offset = communicate_slot_offset
        inbox_slot = replace(
            template,
            slot_id=next_slot_id,
            task_id=(
                instructions_by_id[transfer.consumer_node_id].task_id
                if transfer.consumer_node_id in instructions_by_id and instructions_by_id[transfer.consumer_node_id].task_id is not None
                else template.task_id
            ),
            role=role,
            producer_instruction_id=transfer.producer_node_id,
            shared_storage_id=None,
            global_storage_id=None,
            barrier_storage_id=None,
            scratch_backed=True,
            scratch_offset=scratch_offset,
            allocation_owner_slot_id=None,
            alias_of_slot_id=None,
            physical_owner_cta=transfer.consumer_cta,
        )
        staged_slots.append(inbox_slot)
        if transfer.consumer_storage_kind is DataflowCommStorageKind.TRANSIENT_PREFETCH:
            prefetch_range = scratch_slot_range(inbox_slot)
            if prefetch_range is None or ranges_overlap(
                prefetch_range,
                (
                    communicate_slot_offset,
                    communicate_slot_offset + communicate_slot_bytes,
                ),
            ):
                raise DataflowMemoryPlanningError(
                    f"joint transient prefetch slot {inbox_slot.slot_id} overlaps the permanent communicate slot"
                )
        inbox_slot_by_transfer[transfer.transfer_id] = next_slot_id
        next_slot_id += 1

        consumer = instructions_by_id.get(transfer.consumer_node_id)
        if consumer is None:
            raise DataflowMemoryPlanningError(f"joint transfer {transfer.transfer_id} has no consumer instruction")
        if (
            transfer.consumer_input_index >= len(consumer.input_slots)
            or consumer.input_slots[transfer.consumer_input_index] != transfer.value_id
        ):
            raise DataflowMemoryPlanningError(f"joint transfer {transfer.transfer_id} input binding is stale")
        if transfer.retained_input_copy_event_id is None:
            rewritten_inputs = tuple(inbox_slot.slot_id if slot_id == transfer.value_id else slot_id for slot_id in consumer.input_slots)
            instructions_by_id[consumer.instruction_id] = replace(
                consumer,
                input_slots=rewritten_inputs,
            )

    producer_epoch_by_node: dict[int, int] = {}
    for transfer in materialized_transfers:
        if transfer.producer_slot_epoch is None:
            continue
        previous = producer_epoch_by_node.setdefault(
            transfer.producer_node_id,
            transfer.producer_slot_epoch,
        )
        if previous != transfer.producer_slot_epoch:
            raise DataflowMemoryPlanningError(f"joint producer {transfer.producer_node_id} uses multiple outbox epochs")

    outbox_slot_by_node: dict[int, int] = {}
    for node_id, _epoch in sorted(producer_epoch_by_node.items()):
        producer = instructions_by_id.get(node_id)
        if producer is None or producer.output_slot is None:
            raise DataflowMemoryPlanningError(f"joint outbox producer {node_id} has no output slot")
        template = slots_by_id.get(next(node.output_value for node in joint_plan.nodes if node.node_id == node_id))
        if template is None:
            raise DataflowMemoryPlanningError(f"joint outbox producer {node_id} has no logical value slot")
        outbox_slot = replace(
            template,
            slot_id=next_slot_id,
            role="joint_comm_outbox",
            producer_instruction_id=node_id,
            shared_storage_id=None,
            global_storage_id=None,
            barrier_storage_id=None,
            scratch_backed=True,
            scratch_offset=(communicate_slot_offset if outbox_slot_offset is None else outbox_slot_offset),
            allocation_owner_slot_id=(None if outbox_slot_offset is None else producer.output_slot),
            alias_of_slot_id=(None if outbox_slot_offset is None else producer.output_slot),
            physical_owner_cta=producer.sm_id,
        )
        staged_slots.append(outbox_slot)
        outbox_slot_by_node[node_id] = next_slot_id
        value_forward = producer.value_forward
        if value_forward is not None:
            if len(producer.input_slots) != 1:
                raise DataflowMemoryPlanningError(f"joint ValueForward producer {node_id} does not have one input")
            value_forward = DataflowValueForward(
                source_slot_id=producer.input_slots[0],
                output_slot_id=next_slot_id,
                copy_owner_slot_id=next_slot_id,
                alias_owner_slot_id=producer.input_slots[0],
                storage_policy=value_forward.storage_policy,
            )
        instructions_by_id[node_id] = replace(
            producer,
            output_slot=next_slot_id,
            value_forward=value_forward,
        )
        next_slot_id += 1

    rewritten_comms = list(retained_comms)
    rewritten_send_by_transfer: dict[int, CommPlan] = {}
    joint_nodes_by_id = {node.node_id: node for node in joint_plan.nodes}

    def hbm_issue_dispatch_instruction_ids(
        transfer: Any,
        load_issue_event_id: int,
    ) -> tuple[int, ...]:
        issue = schedule.event(load_issue_event_id)
        modeled_dispatch_node_id = None
        for node_id in schedule.queue(transfer.consumer_cta):
            handler_event = handler_event_by_node[node_id]
            if handler_event.start_us + 1e-9 >= issue.end_us:
                modeled_dispatch_node_id = node_id
                break
        if modeled_dispatch_node_id is None:
            raise DataflowMemoryPlanningError(
                f"joint HBM transfer {transfer.transfer_id} has no queue boundary for load issue event {load_issue_event_id}"
            )
        queue = schedule.queue(transfer.consumer_cta)
        queue_positions = {node_id: position for position, node_id in enumerate(queue)}
        target_position = queue_positions.get(transfer.consumer_node_id)
        if target_position is None:
            raise DataflowMemoryPlanningError(f"joint HBM transfer {transfer.transfer_id} has an unqueued consumer")
        required_bytes = sum(segment.byte_count for segment in transfer.segments)
        previous_interval = previous_interval_by_transfer.get(transfer.transfer_id)
        if transfer.consumer_storage_kind is DataflowCommStorageKind.PERMANENT:
            earliest_position = 0
            if previous_interval is not None:
                previous_release = schedule.event(previous_interval.end_event_id)
                if previous_release.node_id is None:
                    return (modeled_dispatch_node_id,)
                earliest_position = max(
                    earliest_position,
                    queue_positions.get(previous_release.node_id, target_position),
                )
            if permanent_inbox_overlaps_handler_scratch:
                earliest_position = max(
                    earliest_position,
                    max(
                        (
                            position
                            for position, node_id in enumerate(queue[:target_position])
                            if instructions_by_id[node_id].opcode
                            not in {
                                DataflowOpcode.REDUCE,
                                DataflowOpcode.REDUCE_UPDATE,
                                DataflowOpcode.FINALIZE,
                            }
                        ),
                        default=-1,
                    ),
                )
            speculative_dispatches = tuple(queue[position] for position in range(earliest_position, target_position))
            if speculative_dispatches:
                return speculative_dispatches
            return (modeled_dispatch_node_id,)

        # Transient-prefetch storage is a handler-level lifetime contract, not
        # merely a modeled timestamp.  Attach nonblocking retries to compatible
        # handler boundaries before the consumer; a miss falls back at the
        # target without blocking useful local work.
        previous_release_position = -1
        if previous_interval is not None:
            previous_release = schedule.event(previous_interval.end_event_id)
            if previous_release.node_id is None:
                return (modeled_dispatch_node_id,)
            previous_release_position = queue_positions.get(
                previous_release.node_id,
                target_position,
            )

        earliest_position = target_position
        while earliest_position > previous_release_position + 1:
            candidate_position = earliest_position - 1
            candidate_node_id = queue[candidate_position]
            candidate_node = joint_nodes_by_id.get(candidate_node_id)
            if candidate_node is None or candidate_node.transient_prefetch_bytes < (transfer.consumer_storage_lane + 1) * required_bytes:
                break
            earliest_position = candidate_position
        latest_speculative_position = target_position - 1
        selected_positions: list[int] = []
        if latest_speculative_position >= earliest_position:
            # Retry at every compatible handler boundary.  Each probe is a
            # leader-only nonblocking flag load guarded by the issued bit, so
            # an early prediction miss cannot stall useful work.  Retrying is
            # important when the producer becomes ready between two reducers:
            # the later boundary can arm TMA and overlap the load with the
            # remaining local queue instead of falling back at the consumer.
            selected_positions.extend(range(earliest_position, latest_speculative_position + 1))
        speculative_dispatches = tuple(queue[position] for position in selected_positions)
        if speculative_dispatches:
            return speculative_dispatches
        return (modeled_dispatch_node_id,)

    def cluster_issue_dispatch_instruction_id(transfer: Any) -> int:
        """Lower a modeled push at the latest preceding CTA boundary.

        A communicate outbox may remain live while its CTA executes handlers
        that use ordinary scratch.  Attaching every cluster push directly to
        the producer would turn the modeled destination-slot delay into a
        blocking producer-post wait and destroy that overlap.  The joint
        schedule already proves the outbox lifetime, so dispatch at the last
        handler boundary no later than the modeled issue event.
        """

        issue = schedule.event(transfer.producer_issue_event_id)
        candidates = []
        for queue_index, node_id in enumerate(schedule.queue(transfer.producer_cta)):
            handler_event = handler_event_by_node[node_id]
            if handler_event.end_us <= issue.start_us + 1e-9:
                candidates.append((handler_event.end_us, queue_index, node_id))
        if not candidates:
            raise DataflowMemoryPlanningError(
                f"joint cluster transfer {transfer.transfer_id} has no queue "
                f"boundary preceding issue event {transfer.producer_issue_event_id}"
            )
        if outbox_slot_offset is not None:
            producer_position = queue_position_by_node.get(transfer.producer_node_id)
            if producer_position is None:
                raise DataflowMemoryPlanningError(f"joint cluster transfer {transfer.transfer_id} has an unqueued aliased-outbox producer")
            producer_queue = plan.queue(transfer.producer_cta)
            next_writer_index = next(
                (
                    queue_index
                    for queue_index, instruction in enumerate(producer_queue)
                    if queue_index > producer_position[1] and instruction.output_slot is not None
                ),
                None,
            )
            if next_writer_index is not None:
                safe_candidates = [candidate for candidate in candidates if candidate[1] < next_writer_index]
                if safe_candidates:
                    candidates = safe_candidates
                else:
                    producer_event = handler_event_by_node[transfer.producer_node_id]
                    candidates = [
                        (
                            producer_event.end_us,
                            producer_position[1],
                            transfer.producer_node_id,
                        )
                    ]
        dispatch_node_id = max(candidates)[2]
        producer_position = queue_position_by_node.get(transfer.producer_node_id)
        dispatch_position = queue_position_by_node.get(dispatch_node_id)
        if (
            producer_position is None
            or dispatch_position is None
            or producer_position[0] != transfer.producer_cta
            or dispatch_position[0] != transfer.producer_cta
            or dispatch_position[1] < producer_position[1]
        ):
            raise DataflowMemoryPlanningError(f"joint cluster transfer {transfer.transfer_id} dispatches before its producer")
        return dispatch_node_id

    for transfer in materialized_transfers:
        edge = (
            transfer.producer_node_id,
            transfer.consumer_node_id,
            transfer.value_id,
        )
        send = sends_by_edge.pop(edge, None)
        recv = recvs_by_edge.pop(edge, None)
        expected_send_kind = (
            DataflowCommKind.CLUSTER_SEND if transfer.kind is DataflowTransportKind.CLUSTER_PUSH else DataflowCommKind.HBM_SEND
        )
        expected_recv_kind = (
            DataflowCommKind.CLUSTER_RECV if transfer.kind is DataflowTransportKind.CLUSTER_PUSH else DataflowCommKind.HBM_RECV
        )
        if send is None or recv is None:
            raise DataflowMemoryPlanningError(f"joint transfer {transfer.transfer_id} does not match logical comms")
        source_slot_id = outbox_slot_by_node.get(
            transfer.producer_node_id,
            transfer.value_id,
        )
        target_slot_id = inbox_slot_by_transfer[transfer.transfer_id]
        segment_descriptors = (
            tuple((segment.segment_id, segment.byte_offset, segment.byte_count) for segment in transfer.segments)
            if transfer.kind is DataflowTransportKind.HBM_STAGED
            else ((0, 0, 0),)
        )
        if not segment_descriptors:
            raise DataflowMemoryPlanningError(f"joint HBM transfer {transfer.transfer_id} has no runtime segments")
        segment_count = len(segment_descriptors)
        grouped_hbm_issue_dispatch_instruction_ids = (
            hbm_issue_dispatch_instruction_ids(
                transfer,
                transfer.segments[-1].load_issue_event_id,
            )
            if transfer.kind is DataflowTransportKind.HBM_STAGED
            else ()
        )
        for segment_id, byte_offset, byte_count in segment_descriptors:
            segment_epoch = transfer.consumer_slot_epoch + segment_id
            barrier_phase = (segment_epoch - 1) & 1
            rewritten_send = replace(
                send,
                kind=expected_send_kind,
                source_slot_id=source_slot_id,
                target_slot_id=target_slot_id,
                peer_cta_rank=(send.peer_cta_rank if transfer.kind is DataflowTransportKind.CLUSTER_PUSH else None),
                dispatch_instruction_id=(
                    cluster_issue_dispatch_instruction_id(transfer)
                    if transfer.kind is DataflowTransportKind.CLUSTER_PUSH
                    else send.dispatch_instruction_id
                ),
                flag_epoch=segment_epoch,
                barrier_phase=barrier_phase,
                byte_offset=byte_offset,
                byte_count=byte_count,
                segment_id=segment_id,
                segment_count=segment_count,
            )
            rewritten_recv = replace(
                recv,
                kind=expected_recv_kind,
                source_slot_id=source_slot_id,
                target_slot_id=target_slot_id,
                peer_cta_rank=(recv.peer_cta_rank if transfer.kind is DataflowTransportKind.CLUSTER_PUSH else None),
                flag_epoch=segment_epoch,
                barrier_phase=barrier_phase,
                byte_offset=byte_offset,
                byte_count=byte_count,
                segment_id=segment_id,
                segment_count=segment_count,
            )
            issue_dispatch_instruction_ids = (
                (transfer.consumer_node_id,)
                if transfer.kind is DataflowTransportKind.CLUSTER_PUSH
                else grouped_hbm_issue_dispatch_instruction_ids
            )
            if transfer.kind is DataflowTransportKind.HBM_STAGED and any(
                dispatch_instruction_id != transfer.consumer_node_id for dispatch_instruction_id in issue_dispatch_instruction_ids
            ):
                rewritten_issues = tuple(
                    replace(
                        rewritten_recv,
                        kind=DataflowCommKind.HBM_RECV_ISSUE,
                        dispatch_instruction_id=dispatch_instruction_id,
                    )
                    for dispatch_instruction_id in issue_dispatch_instruction_ids
                    if dispatch_instruction_id != transfer.consumer_node_id
                )
                rewritten_wait = replace(
                    rewritten_recv,
                    kind=DataflowCommKind.HBM_RECV_WAIT,
                    dispatch_instruction_id=transfer.consumer_node_id,
                )
                rewritten_comms.append(rewritten_send)
                rewritten_comms.extend(rewritten_issues)
                rewritten_comms.append(rewritten_wait)
            else:
                rewritten_comms.extend((rewritten_send, rewritten_recv))
            if transfer.kind is DataflowTransportKind.CLUSTER_PUSH:
                rewritten_send_by_transfer[transfer.transfer_id] = rewritten_send

    if sends_by_edge or recvs_by_edge:
        raise DataflowMemoryPlanningError("joint execution plan does not cover every logical async communication edge")

    # A retain event breaks a communicate-slot resource cycle by moving the
    # arrived value into its ordinary CTA-local input storage.  Materialize it
    # with the existing shared::cluster TMA path targeted at the issuing CTA
    # itself.  The modeled copy boundary is a post-handler dispatch; the
    # consumer's normal pre-handler receive supplies the completion/fence.
    for transfer in retained_transfers:
        dispatch_node_id = retained_copy_dispatch_node_by_transfer[transfer.transfer_id]
        source_slot_id = inbox_slot_by_transfer[transfer.transfer_id]
        target_slot_id = transfer.value_id
        consumer_rank = plan.topology.cluster_rank(transfer.consumer_cta)
        copy_send = CommPlan(
            source_instruction_id=dispatch_node_id,
            target_instruction_id=transfer.consumer_node_id,
            source_slot_id=source_slot_id,
            target_slot_id=target_slot_id,
            producer_sm=transfer.consumer_cta,
            consumer_sm=transfer.consumer_cta,
            kind=DataflowCommKind.CLUSTER_SEND,
            dispatch_instruction_id=dispatch_node_id,
            peer_cta_rank=consumer_rank,
            flag_epoch=transfer.consumer_slot_epoch,
            barrier_phase=(transfer.consumer_slot_epoch - 1) & 1,
        )
        copy_recv = replace(
            copy_send,
            kind=DataflowCommKind.CLUSTER_RECV,
            dispatch_instruction_id=transfer.consumer_node_id,
        )
        rewritten_comms.extend((copy_send, copy_recv))

    for transfer in materialized_transfers:
        release_node_id = cluster_release_gate_node_by_transfer.get(transfer.transfer_id)
        if release_node_id is None:
            continue
        send = rewritten_send_by_transfer[transfer.transfer_id]
        rewritten_comms.append(
            replace(
                send,
                producer_sm=transfer.consumer_cta,
                consumer_sm=send.producer_sm,
                kind=DataflowCommKind.CLUSTER_RELEASE,
                dispatch_instruction_id=release_node_id,
                peer_cta_rank=plan.topology.cluster_rank(send.producer_sm),
            )
        )

    instructions = tuple(instructions_by_id[instruction.instruction_id] for instruction in plan.instructions)
    queues = {
        cta_id: tuple(instructions_by_id[instruction.instruction_id] for instruction in queue) for cta_id, queue in plan.queues.items()
    }
    return replace(
        plan,
        instructions=instructions,
        queues=queues,
        slots=plan.slots + tuple(staged_slots),
        comms=tuple(rewritten_comms),
    )


def verify_joint_communication_slot_contract(
    plan: InstructionPlan,
    *,
    handler_scratch_bytes: int,
    communicate_slot_offset: int,
    communicate_slot_bytes: int,
    outbox_slot_offset: int | None = None,
) -> None:
    joint_plan = plan.joint_execution_plan
    if joint_plan is None:
        raise DataflowMemoryPlanningError("joint communicate-slot validation requires a joint execution plan")
    joint_plan.require_valid(topology=plan.topology)
    if communicate_slot_offset < handler_scratch_bytes and outbox_slot_offset is None:
        raise DataflowMemoryPlanningError("Dataflow joint communicate slot overlaps ordinary handler scratch")
    if communicate_slot_bytes <= 0:
        raise DataflowMemoryPlanningError("Dataflow joint communicate slot must reserve a positive byte extent")
    permanent_communicate_roles = {
        "joint_comm_inbox",
        "joint_comm_outbox",
        "joint_comm_relay",
    }
    communicate_roles = permanent_communicate_roles | {"cluster_gated_inbox"}
    communicate_slots = tuple(slot for slot in plan.slots if slot.role in communicate_roles)
    if not communicate_slots:
        raise DataflowMemoryPlanningError("joint execution plan has no materialized communicate slots")
    communicate_end = communicate_slot_offset + communicate_slot_bytes
    communicate_range = (communicate_slot_offset, communicate_end)
    outbox_offset = communicate_slot_offset if outbox_slot_offset is None else outbox_slot_offset
    outbox_range = (outbox_offset, outbox_offset + communicate_slot_bytes)
    if outbox_slot_offset is not None and ranges_overlap(communicate_range, outbox_range):
        raise DataflowMemoryPlanningError("joint aliased outbox must remain disjoint from the permanent inbox")
    for slot in plan.slots:
        scratch_range = scratch_slot_range(slot)
        if scratch_range is None:
            continue
        if slot.role == "joint_comm_inbox":
            if scratch_range[0] != communicate_slot_offset or scratch_range[1] > communicate_end:
                raise DataflowMemoryPlanningError(f"joint communicate slot {slot.slot_id} escapes its permanent extent")
        elif slot.role == "joint_comm_outbox":
            if scratch_range[0] != outbox_offset or scratch_range[1] > outbox_range[1]:
                raise DataflowMemoryPlanningError(f"joint outbox slot {slot.slot_id} escapes its aliased extent")
        elif slot.role == "cluster_gated_inbox":
            if scratch_range[0] == communicate_slot_offset:
                if scratch_range[1] > communicate_end:
                    raise DataflowMemoryPlanningError(f"joint gated slot {slot.slot_id} escapes its permanent extent")
            elif ranges_overlap(scratch_range, communicate_range):
                raise DataflowMemoryPlanningError(f"joint gated transient slot {slot.slot_id} overlaps the permanent communicate slot")
        else:
            if ranges_overlap(scratch_range, communicate_range):
                raise DataflowMemoryPlanningError(f"ordinary scratch slot {slot.slot_id} overlaps the communicate slot")
            if ranges_overlap(scratch_range, outbox_range):
                lifetime_alias = bool(
                    outbox_slot_offset is not None
                    and scratch_range[0] == outbox_offset
                    and scratch_range[1] <= outbox_range[1]
                    and slot.role in {"partial", "streaming_acc"}
                )
                if not lifetime_alias:
                    raise DataflowMemoryPlanningError(f"ordinary scratch slot {slot.slot_id} overlaps the joint outbox")


def materialize_cluster_inbox_slots(
    plan: InstructionPlan,
    *,
    cluster_inbox_offset: int,
) -> InstructionPlan:
    """Bind scratch-backed cluster destinations to one permanent per-CTA inbox."""

    if cluster_inbox_offset < 0 or cluster_inbox_offset % DATAFLOW_SLOT_ALIGNMENT:
        raise DataflowMemoryPlanningError(
            f"Dataflow cluster inbox offset must be non-negative and {DATAFLOW_SLOT_ALIGNMENT}-byte aligned, got {cluster_inbox_offset}"
        )

    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    if set(slots_by_id) != set(range(len(plan.slots))):
        raise DataflowMemoryPlanningError("Dataflow cluster inbox planning requires dense slot ids before runtime packing")

    scratch_recvs = tuple(
        comm
        for comm in plan.comms
        if comm.kind is DataflowCommKind.CLUSTER_RECV and (slot := slots_by_id.get(comm.target_slot_id)) is not None and slot.scratch_backed
    )
    inbound_by_cta: dict[int, list[CommPlan]] = {}
    for comm in scratch_recvs:
        inbound_by_cta.setdefault(comm.consumer_sm, []).append(comm)
    queue_position = {
        instruction.instruction_id: (sm_id, index) for sm_id, queue in plan.queues.items() for index, instruction in enumerate(queue)
    }
    spilled_transfer_predecessors: dict[tuple[int, int, int, int], CommPlan] = {}
    for sm_id, comms in inbound_by_cta.items():
        target_ids = [comm.target_instruction_id for comm in comms]
        if len(target_ids) != len(set(target_ids)):
            raise DataflowMemoryPlanningError(
                "Dataflow reusable cluster inbox supports at most one scratch-backed "
                f"receive per target instruction; cta={sm_id}, targets={target_ids!r}"
            )
        try:
            ordered = sorted(
                comms,
                key=lambda comm: (
                    queue_position[comm.target_instruction_id][1],
                    comm.target_instruction_id,
                    comm.source_instruction_id,
                ),
            )
        except KeyError as err:
            raise DataflowMemoryPlanningError(
                f"Dataflow reusable cluster inbox references an unqueued target instruction {err.args[0]}"
            ) from err
        if any(queue_position[comm.target_instruction_id][0] != sm_id for comm in ordered):
            raise DataflowMemoryPlanningError(f"Dataflow reusable cluster inbox CTA {sm_id} does not own every target")
        for previous, spilled in zip(ordered, ordered[1:]):
            spilled_transfer_predecessors[comm_transfer_key(spilled)] = previous

    next_slot_id = len(plan.slots)
    staged_slots: list[SlotPlan] = []
    staged_target_by_transfer: dict[tuple[int, int, int, int], int] = {}
    for comm in scratch_recvs:
        received_slot = slots_by_id.get(comm.target_slot_id)
        target = instructions_by_id.get(comm.target_instruction_id)
        if received_slot is None or target is None:
            continue
        received_range = scratch_slot_range(received_slot)
        if received_range is None:
            continue
        target_positions = [index for index, slot_id in enumerate(target.input_slots) if slot_id == comm.target_slot_id]
        if len(target_positions) != 1:
            raise DataflowMemoryPlanningError(
                "Dataflow async receive target must appear exactly once in its handler inputs; "
                f"instruction={target.instruction_id}, slot={comm.target_slot_id}, "
                f"occurrences={len(target_positions)}"
            )
        staging_slot = replace(
            received_slot,
            slot_id=next_slot_id,
            task_id=target.task_id if target.task_id is not None else received_slot.task_id,
            role=("hbm_spill_inbox" if comm_transfer_key(comm) in spilled_transfer_predecessors else "cluster_inbox"),
            producer_instruction_id=comm.source_instruction_id,
            shared_storage_id=None,
            global_storage_id=None,
            barrier_storage_id=None,
            scratch_offset=cluster_inbox_offset,
            allocation_owner_slot_id=None,
            alias_of_slot_id=None,
            physical_owner_cta=comm.consumer_sm,
        )
        staged_slots.append(staging_slot)
        slots_by_id[next_slot_id] = staging_slot
        updated_inputs = list(target.input_slots)
        updated_inputs[target_positions[0]] = next_slot_id
        instructions_by_id[target.instruction_id] = replace(
            target,
            input_slots=tuple(updated_inputs),
        )
        staged_target_by_transfer[
            (
                comm.source_instruction_id,
                comm.target_instruction_id,
                comm.source_slot_id,
                comm.target_slot_id,
            )
        ] = next_slot_id
        next_slot_id += 1

    if not staged_slots:
        return plan

    instructions = tuple(instructions_by_id[instruction.instruction_id] for instruction in plan.instructions)
    queues = {sm_id: tuple(instructions_by_id[instruction.instruction_id] for instruction in queue) for sm_id, queue in plan.queues.items()}
    comms: list[CommPlan] = []
    for comm in plan.comms:
        transfer_key = (
            comm.source_instruction_id,
            comm.target_instruction_id,
            comm.source_slot_id,
            comm.target_slot_id,
        )
        target_slot_id = staged_target_by_transfer.get(transfer_key)
        rewritten = comm if target_slot_id is None else replace(comm, target_slot_id=target_slot_id)
        if transfer_key in spilled_transfer_predecessors:
            if rewritten.kind is DataflowCommKind.CLUSTER_SEND:
                rewritten = replace(
                    rewritten,
                    kind=DataflowCommKind.HBM_SEND,
                    peer_cta_rank=None,
                )
            elif rewritten.kind is DataflowCommKind.CLUSTER_RECV:
                rewritten = replace(
                    rewritten,
                    kind=DataflowCommKind.HBM_RECV,
                    peer_cta_rank=None,
                )
        comms.append(rewritten)
    return replace(
        plan,
        instructions=instructions,
        queues=queues,
        slots=plan.slots + tuple(staged_slots),
        comms=tuple(comms),
    )


def comm_transfer_key(comm: CommPlan) -> tuple[int, int, int, int]:
    return (
        comm.source_instruction_id,
        comm.target_instruction_id,
        comm.source_slot_id,
        comm.target_slot_id,
    )


def verify_cluster_inbox_contract(
    plan: InstructionPlan,
    *,
    handler_scratch_bytes: int,
    cluster_inbox_offset: int,
    cluster_inbox_bytes: int,
) -> None:
    """Prove physical disjointness, ordered inbox reuse, and no wait cycle."""

    if cluster_inbox_bytes <= 0:
        raise DataflowMemoryPlanningError("Dataflow cluster inbox must reserve a positive byte extent")
    if cluster_inbox_offset < handler_scratch_bytes:
        raise DataflowMemoryPlanningError(
            f"Dataflow cluster inbox overlaps ordinary handler scratch: offset={cluster_inbox_offset}, handler_peak={handler_scratch_bytes}"
        )

    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    inbox_roles = {"cluster_inbox", "hbm_spill_inbox"}
    inbox_slot_ids = {slot.slot_id for slot in plan.slots if slot.role in inbox_roles}
    if not inbox_slot_ids:
        raise DataflowMemoryPlanningError("Dataflow cluster inbox metadata has no bound slots")

    inbox_end = cluster_inbox_offset + cluster_inbox_bytes
    for slot in plan.slots:
        scratch_range = scratch_slot_range(slot)
        if scratch_range is None:
            continue
        if slot.slot_id in inbox_slot_ids:
            if scratch_range[0] != cluster_inbox_offset or scratch_range[1] > inbox_end:
                raise DataflowMemoryPlanningError(
                    f"Dataflow cluster inbox slot {slot.slot_id} escapes [{cluster_inbox_offset}, {inbox_end})"
                )
        elif scratch_range[1] > cluster_inbox_offset:
            raise DataflowMemoryPlanningError(f"Dataflow ordinary scratch slot {slot.slot_id} overlaps the cluster inbox")

    send_by_key: dict[tuple[int, int, int, int], list[CommPlan]] = {}
    recv_by_key: dict[tuple[int, int, int, int], list[CommPlan]] = {}
    release_by_key: dict[tuple[int, int, int, int], list[CommPlan]] = {}
    for comm in plan.comms:
        key = comm_transfer_key(comm)
        if comm.kind in {DataflowCommKind.CLUSTER_SEND, DataflowCommKind.HBM_SEND}:
            send_by_key.setdefault(key, []).append(comm)
        elif comm.kind in {
            DataflowCommKind.CLUSTER_RECV,
            DataflowCommKind.HBM_RECV,
            DataflowCommKind.HBM_RECV_WAIT,
        }:
            recv_by_key.setdefault(key, []).append(comm)
        elif comm.kind is DataflowCommKind.CLUSTER_RELEASE:
            release_by_key.setdefault(key, []).append(comm)

    queue_position = {
        instruction.instruction_id: (sm_id, index) for sm_id, queue in plan.queues.items() for index, instruction in enumerate(queue)
    }
    inbound_by_cta: dict[int, list[tuple[int, int, str, tuple[int, int, int, int]]]] = {}
    for key, recvs in recv_by_key.items():
        target_slot = slots_by_id.get(key[3])
        if target_slot is None or target_slot.slot_id not in inbox_slot_ids:
            if target_slot is not None and target_slot.scratch_backed:
                raise DataflowMemoryPlanningError(f"Dataflow scratch-backed cluster receive {key!r} bypasses the permanent inbox")
            continue
        if len(recvs) != 1 or len(send_by_key.get(key, ())) != 1:
            raise DataflowMemoryPlanningError(f"Dataflow reusable inbox transfer {key!r} requires exactly one send/recv pair")
        recv = recvs[0]
        target = instructions_by_id.get(recv.target_instruction_id)
        if target is None or target.sm_id != recv.consumer_sm or target.input_slots.count(target_slot.slot_id) != 1:
            raise DataflowMemoryPlanningError(
                f"Dataflow cluster inbox slot {target_slot.slot_id} is not uniquely consumed by instruction {recv.target_instruction_id}"
            )
        target_position = queue_position.get(recv.target_instruction_id)
        if target_position is None or target_position[0] != recv.consumer_sm:
            raise DataflowMemoryPlanningError(
                f"Dataflow cluster inbox target {recv.target_instruction_id} is not queued on CTA {recv.consumer_sm}"
            )
        inbound_by_cta.setdefault(recv.consumer_sm, []).append(
            (
                target_position[1],
                recv.target_instruction_id,
                target_slot.role,
                key,
            )
        )

    bound_slot_count = sum(len(records) for records in inbound_by_cta.values())
    if bound_slot_count != len(inbox_slot_ids):
        raise DataflowMemoryPlanningError("Dataflow cluster inbox slots and staged receives are not one-to-one")
    for sm_id, records in inbound_by_cta.items():
        ordered = sorted(records)
        target_ids = [target_id for _, target_id, _, _ in ordered]
        if len(target_ids) != len(set(target_ids)):
            raise DataflowMemoryPlanningError(f"Dataflow CTA {sm_id} reuses its inbox more than once in one instruction")
        roles = [role for _, _, role, _ in ordered]
        if not roles or roles[0] != "cluster_inbox" or any(role != "hbm_spill_inbox" for role in roles[1:]):
            raise DataflowMemoryPlanningError(
                "Dataflow reusable cluster inbox requires one initial cluster push "
                f"followed only by HBM-spilled pushes; cta={sm_id}, roles={roles!r}"
            )
        for index, (_, _, _, transfer_key) in enumerate(ordered):
            releases = release_by_key.get(transfer_key, ())
            send = send_by_key[transfer_key][0]
            recv = recv_by_key[transfer_key][0]
            if index == 0:
                if send.kind is not DataflowCommKind.CLUSTER_SEND or recv.kind is not DataflowCommKind.CLUSTER_RECV or releases:
                    raise DataflowMemoryPlanningError(
                        f"Dataflow initial inbox transfer {transfer_key!r} must be one ungated cluster send/receive pair"
                    )
                continue
            if send.kind is not DataflowCommKind.HBM_SEND or recv.kind is not DataflowCommKind.HBM_RECV or releases:
                raise DataflowMemoryPlanningError(
                    f"Dataflow reused inbox transfer {transfer_key!r} must spill through "
                    "one HBM send/receive pair without a consumer release"
                )
    verify_cluster_source_completion_order(plan)


def verify_cluster_source_completion_order(plan: InstructionPlan) -> None:
    """Verify that recv-before-source-wait queue execution has no causal cycle."""

    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    active_queues = {
        sm_id: tuple(instruction for instruction in queue if instruction.opcode is not DataflowOpcode.EXIT)
        for sm_id, queue in plan.queues.items()
    }
    position_by_instruction: dict[int, tuple[int, int]] = {}
    for sm_id, queue in active_queues.items():
        for index, instruction in enumerate(queue):
            if instruction.instruction_id in position_by_instruction:
                raise DataflowMemoryPlanningError(f"Dataflow instruction {instruction.instruction_id} appears in multiple queues")
            position_by_instruction[instruction.instruction_id] = (sm_id, index)

    Event = tuple[int, int, str]
    edges: dict[Event, set[Event]] = {}

    def add_edge(source: Event, target: Event) -> None:
        edges.setdefault(source, set()).add(target)
        edges.setdefault(target, set())

    for sm_id, queue in active_queues.items():
        for index in range(len(queue)):
            recv_event = (sm_id, index, "recv")
            wait_event = (sm_id, index, "wait")
            handler_event = (sm_id, index, "handler")
            send_event = (sm_id, index, "send")
            add_edge(recv_event, wait_event)
            add_edge(wait_event, handler_event)
            add_edge(handler_event, send_event)
            next_recv = (sm_id, index + 1, "recv")
            add_edge(send_event, next_recv)
        edges.setdefault((sm_id, len(queue), "recv"), set())
        edges.setdefault((sm_id, len(queue), "wait"), set())
        add_edge((sm_id, len(queue), "recv"), (sm_id, len(queue), "wait"))

    sends_by_instruction: dict[int, list[CommPlan]] = {}
    recv_event_by_key: dict[tuple[int, int, int, int], Event] = {}
    release_comms: list[CommPlan] = []
    for comm in plan.comms:
        key = comm_transfer_key(comm)
        if comm.kind is DataflowCommKind.CLUSTER_SEND:
            sends_by_instruction.setdefault(comm.source_instruction_id, []).append(comm)
            source_position = position_by_instruction.get(comm.source_instruction_id)
            target_position = position_by_instruction.get(comm.target_instruction_id)
            if source_position is None or target_position is None:
                raise DataflowMemoryPlanningError(f"Dataflow cluster transfer {key!r} references an unqueued instruction")
            add_edge(
                (source_position[0], source_position[1], "send"),
                (target_position[0], target_position[1], "recv"),
            )
        elif comm.kind is DataflowCommKind.CLUSTER_RECV:
            target_position = position_by_instruction.get(comm.target_instruction_id)
            if target_position is None:
                raise DataflowMemoryPlanningError(f"Dataflow cluster receive {key!r} references an unqueued instruction")
            recv_event_by_key[key] = (
                target_position[0],
                target_position[1],
                "recv",
            )
        elif comm.kind is DataflowCommKind.CLUSTER_RELEASE:
            release_comms.append(comm)

    send_by_key = {comm_transfer_key(comm): comm for comms in sends_by_instruction.values() for comm in comms}
    for release in release_comms:
        key = comm_transfer_key(release)
        gated_send = send_by_key.get(key)
        release_position = position_by_instruction.get(release.resolved_dispatch_instruction_id)
        source_position = None if gated_send is None else position_by_instruction.get(gated_send.source_instruction_id)
        if gated_send is None or release_position is None or source_position is None:
            raise DataflowMemoryPlanningError(f"Dataflow cluster release {key!r} has no queued gated send")
        add_edge(
            (release_position[0], release_position[1], "send"),
            (source_position[0], source_position[1], "send"),
        )

    for sm_id, queue in active_queues.items():
        pending: list[CommPlan] = []
        scratch_pending = False
        for index, instruction in enumerate(queue):
            if scratch_pending:
                wait_event = (sm_id, index, "wait")
                for comm in pending:
                    recv_event = recv_event_by_key.get(comm_transfer_key(comm))
                    if recv_event is None:
                        raise DataflowMemoryPlanningError(f"Dataflow cluster send {comm_transfer_key(comm)!r} has no receive")
                    add_edge(recv_event, wait_event)
                pending.clear()
                scratch_pending = False
            instruction_sends = sends_by_instruction.get(instruction.instruction_id, ())
            pending.extend(instruction_sends)
            scratch_pending = scratch_pending or any(
                (slot := slots_by_id.get(comm.source_slot_id)) is not None and slot.scratch_backed for comm in instruction_sends
            )
        terminal_wait = (sm_id, len(queue), "wait")
        for comm in pending:
            recv_event = recv_event_by_key.get(comm_transfer_key(comm))
            if recv_event is None:
                raise DataflowMemoryPlanningError(f"Dataflow cluster send {comm_transfer_key(comm)!r} has no receive")
            add_edge(recv_event, terminal_wait)

    indegree = {event: 0 for event in edges}
    for targets in edges.values():
        for target in targets:
            indegree[target] += 1
    ready = [event for event, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        event = ready.pop()
        visited += 1
        for target in edges[event]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(edges):
        cycle_events = sorted(event for event, degree in indegree.items() if degree)
        raise DataflowMemoryPlanningError(f"Dataflow cluster inbox source-completion waits form a queue cycle: events={cycle_events[:8]!r}")


def scratch_backed_iter_output_shift(handler_abi: DataflowHandlerABI) -> int:
    if not handler_abi.field_layouts:
        return 0
    largest_field = max(
        handler_abi.field_layouts,
        key=lambda field: (field.numel * field.element_bytes, -field.offset),
    )
    if largest_field.offset < 0:
        return 0
    return align_up(largest_field.offset, PRIMFUNC_DYNAMIC_SHARED_ALIGNMENT)


def slot_is_immediately_reduced_streaming_partial(
    slot: SlotPlan,
    *,
    producer_by_slot: dict[int, Instruction],
    consumers_by_slot: dict[int, list[Instruction]],
) -> bool:
    producer = producer_by_slot.get(slot.slot_id)
    if producer is None or producer.sm_id is None:
        return False
    consumers = consumers_by_slot.get(slot.slot_id, [])
    if len(consumers) != 1:
        return False
    consumer = consumers[0]
    if consumer.opcode is not DataflowOpcode.REDUCE_UPDATE:
        return False
    if consumer.sm_id != producer.sm_id:
        return False
    return bool(consumer.input_slots and consumer.input_slots[-1] == slot.slot_id)


def plan_slots_to_dict(
    plan: InstructionPlan,
    packed_plan: PackedRuntimePlan,
) -> list[dict[str, Any]]:
    result = []
    for slot, packed_slot in zip(plan.slots, packed_plan.slots):
        result.append(
            {
                "slot_id": slot.slot_id,
                "task_id": slot.task_id,
                "role": slot.role,
                "shared_storage_id": slot.shared_storage_id,
                "global_storage_id": slot.global_storage_id,
                "barrier_storage_id": slot.barrier_storage_id,
                "storage": "scratch" if packed_slot.is_scratch_backed else "shared",
                "scratch_backed": packed_slot.is_scratch_backed,
                "shared_offset": packed_slot.shared_offset,
                "global_offset": packed_slot.global_offset,
                "bytes": packed_slot.bytes,
                "field_offsets": slot_field_offsets(slot),
            }
        )
    return result


def slot_field_offsets(slot: SlotPlan) -> dict[str, int]:
    offset = 0
    result: dict[str, int] = {}
    for field_spec in slot.intermediate_type.fields:
        element_bytes = field_element_bytes(field_spec.dtype)
        offset = align_up(offset, min(element_bytes, DATAFLOW_SLOT_ALIGNMENT))
        result[str(field_spec.name)] = offset
        offset += element_bytes * field_numel(field_spec.shape)
    return result


def field_element_bytes(dtype: Any) -> int:
    return require_dataflow_dtype(dtype).element_bytes


def field_numel(shape: Any) -> int:
    if shape is None:
        return 1
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def bool_option(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    raise TypeError(f"Dataflow compile option {name!r} must be a bool, got {value!r}")


def queue_terminal_finalize_variants(
    plan: InstructionPlan,
) -> frozenset[DataflowHandlerVariantKey]:
    """Return finalizer variants whose every use is immediately before EXIT.

    A CTA-wide finalizer adapter normally ends with a completion barrier so a
    following handler cannot reuse its scratch while another warp is still
    writing the final output.  When every scheduled use of one physical
    variant is queue-terminal, kernel exit already provides that lifetime
    boundary and the adapter barrier is redundant.  The proof is deliberately
    plan-local: a variant with even one non-terminal use keeps the barrier for
    all of its calls.
    """

    terminal_instruction_ids = {
        queue[index].instruction_id
        for queue in plan.queues.values()
        for index, instruction in enumerate(queue)
        if instruction.opcode is DataflowOpcode.FINALIZE and all(trailing.opcode is DataflowOpcode.EXIT for trailing in queue[index + 1 :])
    }
    used_variants: set[DataflowHandlerVariantKey] = set()
    nonterminal_variants: set[DataflowHandlerVariantKey] = set()
    for instruction in plan.instructions:
        variant = instruction.handler_variant_key
        if instruction.opcode is not DataflowOpcode.FINALIZE or variant is None:
            continue
        used_variants.add(variant)
        if instruction.instruction_id not in terminal_instruction_ids:
            nonterminal_variants.add(variant)
    return frozenset(used_variants - nonterminal_variants)


def joint_resident_handler_thread_overrides(
    program: DataflowProgram,
    plan: InstructionPlan,
    wrapper_spec: DataflowWrapperSpec,
    primfunc_lowering: DataflowPrimFuncLoweringResult,
    *,
    explicit_overrides: Mapping[int, int] | None,
    target_capabilities: TargetCapabilitySnapshot,
) -> dict[int, int]:
    """Reuse an already-resident CTA width for compiler-generated tail work.

    Warp-specialized ITER lowering can require more physical threads than the
    logical reduce/finalize bodies declare.  Those threads are resident for the
    whole persistent kernel regardless, so leaving them idle in tail handlers
    increases latency without improving occupancy.  Promote only compiler-
    generated BodyIR handlers, never source factories, and never grow the
    launch width discovered by the first physical lowering.

    The returned mapping includes caller overrides unchanged.  An empty result
    means no second lowering is needed.
    """

    if not bool(
        plan.scheduler_config.get(
            "joint_reuse_resident_handler_threads",
            False,
        )
    ):
        return {}
    resident_threads = primfunc_lowering.max_thread_count
    if resident_threads <= 1:
        return {}
    target_limit = target_capabilities.max_threads_per_block
    if target_limit is not None and resident_threads > int(target_limit):
        return {}

    overrides = {int(handler_id): int(thread_count) for handler_id, thread_count in (explicit_overrides or {}).items()}
    explicit_ids = frozenset(overrides)
    lowered_by_id = {handler.handler_id: handler for handler in primfunc_lowering.handlers}
    registry = build_handler_registry(program)
    promoted = False
    for handler in wrapper_spec.handlers:
        if handler.handler_id in explicit_ids or handler.operator_kind not in {"reduce", "finalize"} or handler.handler_identity is None:
            continue
        lowered = lowered_by_id.get(handler.handler_id)
        if lowered is None or lowered.thread_count >= resident_threads:
            continue
        binding = registry.resolve(handler.handler_identity)
        if binding.call.operator.attrs.get("primfunc_source_factory") is not None:
            continue
        overrides[handler.handler_id] = resident_threads
        promoted = True
    return overrides if promoted else {}


def raise_block_dim_to_primfunc_thread_count(
    options: dict[str, Any],
    primfunc_lowering: DataflowPrimFuncLoweringResult,
) -> None:
    required_threads = primfunc_lowering.max_thread_count
    if required_threads <= 0:
        return
    current = options.get("block_dim", (32, 1, 1))
    if isinstance(current, int):
        if current < required_threads:
            options["block_dim"] = required_threads
        return
    block_dim = tuple(int(item) for item in current)
    if len(block_dim) != 3:
        raise ValueError(f"Dataflow executable block_dim must be a positive int or 3-tuple, got {current!r}")
    if block_dim[0] < required_threads:
        options["block_dim"] = (required_threads, block_dim[1], block_dim[2])


def block_dim_x(options: dict[str, Any]) -> int:
    block_dim = options.get("block_dim", 32)
    if isinstance(block_dim, int):
        return int(block_dim)
    block_dim_tuple = tuple(int(item) for item in block_dim)
    if len(block_dim_tuple) != 3:
        raise ValueError(f"Dataflow executable block_dim must be a positive int or 3-tuple, got {block_dim!r}")
    return block_dim_tuple[0]


def collect_tensor_data_pointers(runtime_tensor_args: PackedRuntimeTensorArgs) -> dict[str, int]:
    return {spec.name: runtime_tensor_args.records[spec.index].data_ptr for spec in runtime_tensor_args.plan.specs}


def build_tma_descriptor_handles_if_ready(
    descriptors: Any,
    runtime_tensor_args: PackedRuntimeTensorArgs,
    *,
    expected_device_ordinal: int | None,
) -> Any:
    tensor_data_ptrs = collect_tensor_data_pointers(runtime_tensor_args)
    if descriptors and any(tensor_data_ptrs.values()):
        validate_tma_descriptor_runtime_tensors(
            descriptors,
            tensor_data_ptrs=tensor_data_ptrs,
            tensor_metadata=tensor_metadata(runtime_tensor_args),
            expected_device_ordinal=expected_device_ordinal,
        )
        return build_tma_descriptor_handles(descriptors, tensor_data_ptrs=tensor_data_ptrs)
    return ()


def tensor_indices(runtime_tensor_args: PackedRuntimeTensorArgs) -> dict[str, int]:
    return {spec.name: spec.index for spec in runtime_tensor_args.plan.specs}


def tensor_metadata(runtime_tensor_args: PackedRuntimeTensorArgs) -> dict[str, Any]:
    return {spec.name: runtime_tensor_args.metadata[spec.index] for spec in runtime_tensor_args.plan.specs}


def validate_contiguous_primfunc_map_inputs(
    lowering: DataflowPrimFuncLoweringResult,
    plan: InstructionPlan,
    packed_plan: PackedRuntimePlan,
) -> None:
    packed_slots_by_id = {slot.slot_id: packed_slot for slot, packed_slot in zip(plan.slots, packed_plan.slots)}
    for handler in lowering.handlers:
        input_count = handler.contiguous_map_input_count
        if input_count <= 0:
            continue
        instructions = [
            instruction
            for instruction in plan.instructions
            if instruction.opcode is DataflowOpcode.MAP
            and instruction.handler_identity == handler.handler_identity
            and instruction.handler_variant_key == handler.handler_variant_key
        ]
        if not instructions:
            raise ValueError(f"Dataflow contiguous map inputs found no scheduled instructions for {handler.operator_name!r}")
        for instruction in instructions:
            if len(instruction.input_slots) != input_count:
                raise ValueError(
                    "Dataflow contiguous map inputs require every instruction to have the compiled "
                    f"slot count: operator={handler.operator_name!r}, "
                    f"got={len(instruction.input_slots)}, expected={input_count}"
                )
            slots = [packed_slots_by_id[slot_id] for slot_id in instruction.input_slots]
            slot_bytes = slots[0].bytes
            slot_stride = align_up(max(slot_bytes, DATAFLOW_SLOT_ALIGNMENT), DATAFLOW_SLOT_ALIGNMENT)
            base_offset = slots[0].shared_offset
            for slot_index, slot in enumerate(slots):
                if slot.flags & (DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL | DATAFLOW_SLOT_FLAG_SCRATCH_BACKED):
                    raise ValueError(
                        "Dataflow contiguous map inputs require ordinary shared slots, "
                        f"operator={handler.operator_name!r}, slot_index={slot_index}"
                    )
                expected_offset = base_offset + slot_index * slot_stride
                if slot.bytes != slot_bytes or slot.shared_offset != expected_offset:
                    raise ValueError(
                        "Dataflow contiguous map inputs require equal-size, adjacent shared slots: "
                        f"operator={handler.operator_name!r}, slot_index={slot_index}, "
                        f"offset={slot.shared_offset}, expected_offset={expected_offset}, "
                        f"bytes={slot.bytes}, expected_bytes={slot_bytes}"
                    )


def extend_launch_for_primfunc_dynamic_shared(
    launch_package: DataflowLaunchPackage,
    wrapper_spec: DataflowWrapperSpec,
    dynamic_shared_bytes: int,
    handoff_plans: tuple[DataflowCrossHandlerHandoffPlan, ...] = (),
    *,
    cluster_inbox_offset: int = 0,
    cluster_inbox_bytes: int = 0,
) -> tuple[DataflowLaunchPackage, DataflowWrapperSpec]:
    if cluster_inbox_offset < 0 or cluster_inbox_bytes < 0:
        raise ValueError(f"Dataflow cluster inbox offset/bytes must be non-negative, got {cluster_inbox_offset}/{cluster_inbox_bytes}")
    if cluster_inbox_bytes and (
        cluster_inbox_offset % DATAFLOW_SLOT_ALIGNMENT or cluster_inbox_offset + cluster_inbox_bytes > max(0, dynamic_shared_bytes)
    ):
        raise ValueError(
            "Dataflow cluster inbox must be aligned and contained in PrimFunc scratch: "
            f"offset={cluster_inbox_offset}, bytes={cluster_inbox_bytes}, "
            f"scratch_bytes={dynamic_shared_bytes}"
        )
    if dynamic_shared_bytes <= 0 and not handoff_plans and not cluster_inbox_bytes:
        return launch_package, wrapper_spec
    scratch_offset = align_up(
        launch_package.shared_control_bytes,
        PRIMFUNC_DYNAMIC_SHARED_ALIGNMENT,
    )
    handoff_arenas, handoff_arena_offset, handoff_arena_bytes, arena_end = place_handoff_arenas(
        handoff_plans,
        scratch_offset + max(0, dynamic_shared_bytes),
    )
    shared_slot_base_offset = align_up(
        arena_end,
        DATAFLOW_SHARED_ALIGNMENT,
    )
    shared_memory_bytes = shared_slot_base_offset + launch_package.shared_slot_bytes
    launch_package = replace(
        launch_package,
        shared_memory_bytes=shared_memory_bytes,
        shared_slot_base_offset=shared_slot_base_offset,
        cluster_inbox_offset=cluster_inbox_offset,
        cluster_inbox_bytes=cluster_inbox_bytes,
    )
    wrapper_spec = replace(
        wrapper_spec,
        shared_memory_bytes=shared_memory_bytes,
        shared_slot_base_offset=shared_slot_base_offset,
        primfunc_scratch_offset=scratch_offset,
        primfunc_scratch_bytes=max(0, dynamic_shared_bytes),
        cluster_inbox_offset=cluster_inbox_offset,
        cluster_inbox_bytes=cluster_inbox_bytes,
        handoff_arena_offset=handoff_arena_offset,
        handoff_arena_bytes=handoff_arena_bytes,
        handoff_plan_arenas=handoff_arenas,
    )
    return launch_package, wrapper_spec


def primfunc_launch_shared_memory_bytes(
    launch_package: DataflowLaunchPackage,
    dynamic_shared_bytes: int,
    handoff_plans: tuple[DataflowCrossHandlerHandoffPlan, ...] = (),
) -> int:
    if dynamic_shared_bytes <= 0 and not handoff_plans:
        return launch_package.shared_memory_bytes
    scratch_offset = align_up(
        launch_package.shared_control_bytes,
        PRIMFUNC_DYNAMIC_SHARED_ALIGNMENT,
    )
    _, _, _, arena_end = place_handoff_arenas(
        handoff_plans,
        scratch_offset + max(0, dynamic_shared_bytes),
    )
    shared_slot_base_offset = align_up(
        arena_end,
        DATAFLOW_SHARED_ALIGNMENT,
    )
    return shared_slot_base_offset + launch_package.shared_slot_bytes


def place_handoff_arenas(
    handoff_plans: tuple[DataflowCrossHandlerHandoffPlan, ...],
    begin: int,
) -> tuple[tuple[DataflowHandoffArenaSpec, ...], int, int, int]:
    cursor = int(begin)
    arenas: list[DataflowHandoffArenaSpec] = []
    fingerprints: set[str] = set()
    for plan in handoff_plans:
        if not isinstance(plan, DataflowCrossHandlerHandoffPlan) or not plan.enabled:
            raise ValueError("physical handoff arena placement requires enabled typed plans")
        if plan.fingerprint in fingerprints:
            raise ValueError("handoff arena placement contains duplicate plans")
        fingerprints.add(plan.fingerprint)
        cursor = align_up(cursor, plan.arena_alignment)
        arenas.append(
            DataflowHandoffArenaSpec(
                plan_fingerprint=plan.fingerprint,
                offset=cursor,
                bytes=plan.arena_bytes,
                alignment=plan.arena_alignment,
            )
        )
        cursor += plan.arena_bytes
    if not arenas:
        return (), 0, 0, cursor
    arena_offset = arenas[0].offset
    return tuple(arenas), arena_offset, cursor - arena_offset, cursor


def packed_slot_flag_count(packed_plan: PackedRuntimePlan, flag: int) -> int:
    return sum(bool(slot.flags & flag) for slot in packed_plan.slots)


def scratch_slot_range(slot: SlotPlan) -> tuple[int, int] | None:
    if not slot.scratch_backed or slot.scratch_offset is None:
        return None
    size = 0
    for field_spec in slot.intermediate_type.fields:
        element_bytes = field_element_bytes(field_spec.dtype)
        size = align_up(size, min(element_bytes, DATAFLOW_SLOT_ALIGNMENT))
        size += element_bytes * field_numel(field_spec.shape)
    size = align_up(size, DATAFLOW_SLOT_ALIGNMENT)
    return (slot.scratch_offset, slot.scratch_offset + size)


def ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def slots_share_global_storage(left: SlotPlan, right: SlotPlan) -> bool:
    if left.slot_id == right.slot_id:
        return True
    return left.global_storage_id is not None and left.global_storage_id == right.global_storage_id


def scratch_hbm_direct_global_slot_ids(plan: InstructionPlan) -> frozenset[int]:
    """Route HBM inputs around scratch aliases that would overwrite live local inputs."""

    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    direct_slots: set[int] = set()
    for comm in plan.comms:
        if comm.kind is not DataflowCommKind.HBM_RECV:
            continue
        received_slot = slots_by_id.get(comm.target_slot_id)
        target = instructions_by_id.get(comm.target_instruction_id)
        if received_slot is None or target is None or (received_range := scratch_slot_range(received_slot)) is None:
            continue
        aliases_live_input = any(
            input_slot_id != comm.target_slot_id
            and (other_slot := slots_by_id.get(input_slot_id)) is not None
            and (other_range := scratch_slot_range(other_slot)) is not None
            and ranges_overlap(received_range, other_range)
            for input_slot_id in target.input_slots
        )
        if not aliases_live_input:
            continue
        source_slot = slots_by_id.get(comm.source_slot_id)
        if source_slot is None or not slots_share_global_storage(
            source_slot,
            received_slot,
        ):
            raise DataflowMemoryPlanningError(
                "Dataflow scratch-backed HBM receive aliases a live input but uses "
                "different global storage; selective direct-global handoff requires "
                f"one storage identity, got {comm.source_slot_id} -> "
                f"{comm.target_slot_id}"
            )
        direct_slots.update((comm.source_slot_id, comm.target_slot_id))
    return frozenset(direct_slots)


def packed_plan_uses_hbm_direct_global(packed_plan: PackedRuntimePlan) -> bool:
    return any(bool(slot.flags & DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL) for slot in packed_plan.slots)


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + alignment - 1) // alignment) * alignment


def validate_linkable_primfunc_intermediate(handler_abi: Any) -> None:
    field_layout_sets = list(getattr(handler_abi, "field_layouts_by_handler", {}).values())
    if not field_layout_sets:
        field_layout_sets = [handler_abi.field_layouts]
    for field_layouts in field_layout_sets:
        for field_layout in field_layouts:
            if field_layout.numel <= 0:
                raise NotImplementedError(
                    "linked primfunc handlers require positive intermediate field element counts, "
                    f"got field {field_layout.name!r} with shape {field_layout.shape!r}"
                )
