"""CUDA wrapper source skeleton generation for Dataflow programs."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .abi_schema import (
    DATAFLOW_COMM_KIND_ABI_VALUES,
    DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH,
)
from .cuda_contract import DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES
from .handler_codegen import DataflowHandlerCodegenModule
from .handler_identity import DataflowHandlerIdentity, DataflowHandlerVariantKey
from .tma_descriptors import DataflowTMADescriptorSpec

from .handler import PRIMFUNC_HANDLER_LOWERING
from .runtime import (
    DATAFLOW_CLUSTER_SOURCE_WAIT_ALL,
    DATAFLOW_CLUSTER_SOURCE_WAIT_PREVIOUS,
    UINT32_SENTINEL,
    PackedRuntimePlan,
)


_IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z_]")


@dataclass(frozen=True)
class DataflowHandlerSpec:
    handler_id: int
    operator_name: str
    operator_kind: str
    symbol_name: str
    handler_identity: DataflowHandlerIdentity | None = None
    handler_variant_key: DataflowHandlerVariantKey | None = None

    def __post_init__(self) -> None:
        if (self.handler_identity is None) != (self.handler_variant_key is None):
            raise ValueError(f"Dataflow wrapper handler {self.handler_id} requires both identity and variant key, or neither")
        if (
            self.handler_identity is not None
            and self.handler_variant_key is not None
            and self.handler_variant_key.base_identity != self.handler_identity
        ):
            raise ValueError(f"Dataflow wrapper handler {self.handler_id} identity does not match its variant base")

    def to_dict(self) -> dict[str, Any]:
        return {
            "handler_id": self.handler_id,
            "operator_name": self.operator_name,
            "operator_kind": self.operator_kind,
            "symbol_name": self.symbol_name,
            "identity": (None if self.handler_identity is None else self.handler_identity.to_dict()),
            "variant_key": (None if self.handler_variant_key is None else self.handler_variant_key.to_dict()),
        }


@dataclass(frozen=True)
class DataflowHandlerSource:
    handler_id: int
    body_source: str
    completion_synchronized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "handler_id": self.handler_id,
            "body_bytes": len(self.body_source.encode("utf-8")),
            "completion_synchronized": self.completion_synchronized,
        }


@dataclass(frozen=True)
class DataflowHandoffArenaSpec:
    """One typed handoff plan's absolute interval in dynamic shared memory."""

    plan_fingerprint: str
    offset: int
    bytes: int
    alignment: int

    def __post_init__(self) -> None:
        if len(self.plan_fingerprint) != 64:
            raise ValueError("handoff arena requires a plan fingerprint")
        for name in ("offset", "bytes", "alignment"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"handoff arena {name} must be non-negative")
        if self.bytes <= 0 or self.alignment <= 0:
            raise ValueError("handoff arena requires positive bytes and alignment")
        if self.offset % self.alignment:
            raise ValueError(f"handoff arena offset {self.offset} is not {self.alignment}-byte aligned")

    @property
    def end(self) -> int:
        return self.offset + self.bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_fingerprint": self.plan_fingerprint,
            "offset": self.offset,
            "bytes": self.bytes,
            "end": self.end,
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class DataflowWrapperKernelParam:
    """One extra kernel parameter registered by wrapper instrumentation."""

    name: str
    c_type: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z_]\w*", self.name) is None:
            raise ValueError(f"Dataflow wrapper instrumentation parameter has invalid name {self.name!r}")
        if not self.c_type.strip():
            raise ValueError(f"Dataflow wrapper instrumentation parameter {self.name!r} requires a C type")

    @property
    def declaration(self) -> str:
        return f"{self.c_type.strip()} {self.name}"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "c_type": self.c_type.strip()}


@dataclass(frozen=True)
class DataflowWrapperInstrumentation:
    """Structured source fragments attached at semantic wrapper generation points."""

    name: str
    extra_kernel_params: tuple[DataflowWrapperKernelParam, ...] = ()
    namespace_source: str = ""
    kernel_prologue: str = ""
    before_barrier_init: str = ""
    after_barrier_init: str = ""
    after_cluster_sync: str = ""
    before_loop: str = ""
    before_recv: str = ""
    after_recv: str = ""
    after_handler: str = ""
    after_send: str = ""
    kernel_epilogue: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Dataflow wrapper instrumentation requires a non-empty name")
        invalid_params = tuple(param for param in self.extra_kernel_params if not isinstance(param, DataflowWrapperKernelParam))
        if invalid_params:
            raise TypeError(
                f"Dataflow wrapper instrumentation parameters must be DataflowWrapperKernelParam instances, got {invalid_params!r}"
            )
        parameter_names = tuple(param.name for param in self.extra_kernel_params)
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError(f"Dataflow wrapper instrumentation has duplicate kernel parameters: {parameter_names!r}")

    def to_dict(self) -> dict[str, Any]:
        hook_names = (
            "namespace_source",
            "kernel_prologue",
            "before_barrier_init",
            "after_barrier_init",
            "after_cluster_sync",
            "before_loop",
            "before_recv",
            "after_recv",
            "after_handler",
            "after_send",
            "kernel_epilogue",
        )
        return {
            "name": self.name,
            "extra_kernel_params": [param.to_dict() for param in self.extra_kernel_params],
            "hook_bytes": {hook_name: len(getattr(self, hook_name).encode("utf-8")) for hook_name in hook_names},
        }


@dataclass(frozen=True)
class DataflowWrapperLaunchPacing:
    """Typed persistent-kernel entry pacing shared by all wrapper variants."""

    delay_ns: int = 0
    cluster_stagger_ns: int = 0
    cluster_stagger_group_size: int = 1

    def __post_init__(self) -> None:
        if self.delay_ns < 0:
            raise ValueError("Dataflow wrapper launch delay must be non-negative")
        if self.cluster_stagger_ns < 0:
            raise ValueError("Dataflow wrapper cluster stagger must be non-negative")
        if self.cluster_stagger_group_size <= 0:
            raise ValueError("Dataflow wrapper cluster stagger group size must be positive")

    @property
    def enabled(self) -> bool:
        return self.delay_ns != 0 or self.cluster_stagger_ns != 0

    def to_dict(self) -> dict[str, int]:
        return {
            "delay_ns": self.delay_ns,
            "cluster_stagger_ns": self.cluster_stagger_ns,
            "cluster_stagger_group_size": self.cluster_stagger_group_size,
        }


@dataclass(frozen=True)
class DataflowWrapperSpec:
    kernel_name: str
    handlers: tuple[DataflowHandlerSpec, ...]
    launch_bound_threads: int = 0
    shared_memory_bytes: int = 0
    shared_slot_base_offset: int = 0
    primfunc_scratch_offset: int = 0
    primfunc_scratch_bytes: int = 0
    handoff_arena_offset: int = 0
    handoff_arena_bytes: int = 0
    handoff_plan_arenas: tuple[DataflowHandoffArenaSpec, ...] = ()
    primfunc_use_global_slot_fields: bool = False
    primfunc_all_slots_scratch_backed: bool = False
    primfunc_scratch_backed_iter_symbols: tuple[str, ...] = ()
    primfunc_handler_scratch_offsets: tuple[tuple[str, int], ...] = ()
    cluster_inbox_offset: int = 0
    cluster_inbox_bytes: int = 0
    barrier_count: int = 0
    cluster_ack_count: int = 0
    comm_count: int = 0
    has_hbm_comms: bool = False
    has_cluster_send_comms: bool = False
    has_cluster_release_comms: bool = False
    has_gated_cluster_push: bool = False
    unique_gated_cluster_waits: bool = False
    precomputed_cluster_source_lifetime: bool = False
    has_cluster_source_reuse_wait_actions: bool = False
    segmented_hbm_tma: bool = False
    cluster_size: int = 1
    tensor_arg_count: int = 0
    handler_lowering: str = PRIMFUNC_HANDLER_LOWERING
    handler_codegen_module: DataflowHandlerCodegenModule | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    handler_helper_source: str = ""
    handler_sources: tuple[DataflowHandlerSource, ...] = ()
    queue_indexing: str = "cta_rank"
    tma_descriptors: tuple[DataflowTMADescriptorSpec, ...] = ()
    launch_pacing: DataflowWrapperLaunchPacing = field(default_factory=DataflowWrapperLaunchPacing)

    @property
    def tma_descriptor_names(self) -> tuple[str, ...]:
        return tuple(descriptor.name for descriptor in self.tma_descriptors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "handlers": [handler.to_dict() for handler in self.handlers],
            "launch_bound_threads": self.launch_bound_threads,
            "shared_memory_bytes": self.shared_memory_bytes,
            "shared_slot_base_offset": self.shared_slot_base_offset,
            "primfunc_scratch_offset": self.primfunc_scratch_offset,
            "primfunc_scratch_bytes": self.primfunc_scratch_bytes,
            "handoff_arena_offset": self.handoff_arena_offset,
            "handoff_arena_bytes": self.handoff_arena_bytes,
            "handoff_plan_arenas": [arena.to_dict() for arena in self.handoff_plan_arenas],
            "primfunc_use_global_slot_fields": self.primfunc_use_global_slot_fields,
            "primfunc_all_slots_scratch_backed": self.primfunc_all_slots_scratch_backed,
            "primfunc_scratch_backed_iter_symbols": list(self.primfunc_scratch_backed_iter_symbols),
            "primfunc_handler_scratch_offsets": [
                {"symbol": symbol, "offset": offset} for symbol, offset in self.primfunc_handler_scratch_offsets
            ],
            "cluster_inbox_offset": self.cluster_inbox_offset,
            "cluster_inbox_bytes": self.cluster_inbox_bytes,
            "barrier_count": self.barrier_count,
            "cluster_ack_count": self.cluster_ack_count,
            "comm_count": self.comm_count,
            "has_hbm_comms": self.has_hbm_comms,
            "has_cluster_send_comms": self.has_cluster_send_comms,
            "has_cluster_release_comms": self.has_cluster_release_comms,
            "has_gated_cluster_push": self.has_gated_cluster_push,
            "unique_gated_cluster_waits": self.unique_gated_cluster_waits,
            "precomputed_cluster_source_lifetime": self.precomputed_cluster_source_lifetime,
            "has_cluster_source_reuse_wait_actions": (self.has_cluster_source_reuse_wait_actions),
            "segmented_hbm_tma": self.segmented_hbm_tma,
            "cluster_size": self.cluster_size,
            "tensor_arg_count": self.tensor_arg_count,
            "handler_lowering": self.handler_lowering,
            "handler_codegen_module": (None if self.handler_codegen_module is None else self.handler_codegen_module.to_dict()),
            "handler_helper_bytes": len(self.handler_helper_source.encode("utf-8")),
            "handler_sources": [source.to_dict() for source in self.handler_sources],
            "queue_indexing": self.queue_indexing,
            "tma_descriptors": [descriptor.to_dict() for descriptor in self.tma_descriptors],
            "launch_pacing": self.launch_pacing.to_dict(),
        }


def sanitize_identifier(value: str, *, prefix: str) -> str:
    identifier = _IDENTIFIER_RE.sub("_", value).strip("_")
    if not identifier:
        identifier = prefix
    if identifier[0].isdigit():
        identifier = f"{prefix}_{identifier}"
    return identifier


def collect_handler_kinds(packed_plan: PackedRuntimePlan) -> dict[int, str]:
    kinds: dict[int, str] = {}
    operator_kinds = getattr(packed_plan, "operator_kinds", {})
    for operator_name, handler_id in packed_plan.operator_table.items():
        kind = operator_kinds.get(operator_name)
        if kind is not None:
            kinds[handler_id] = kind
    for instruction in packed_plan.instructions:
        if instruction.handler_id == UINT32_SENTINEL:
            continue
        if instruction.handler_id in kinds:
            continue
        kind = {
            1: "iter",
            2: "reduce",
            3: "finalize",
        }.get(instruction.opcode)
        if kind is None:
            continue
        kinds.setdefault(instruction.handler_id, kind)
    return kinds


def build_wrapper_spec(
    packed_plan: PackedRuntimePlan,
    *,
    kernel_name: str = "dataflow_wrapper",
    launch_package: Any | None = None,
    handler_lowering: Any | None = None,
    cluster_size: int = 1,
    tensor_arg_count: int = 0,
    queue_indexing: str = "cta_rank",
    launch_bound_threads: int = 0,
) -> DataflowWrapperSpec:
    if not isinstance(packed_plan, PackedRuntimePlan):
        raise TypeError(f"build_wrapper_spec expects PackedRuntimePlan, got {packed_plan!r}")

    sanitized_kernel_name = sanitize_identifier(kernel_name, prefix="dataflow_wrapper")
    normalized_handler_lowering = PRIMFUNC_HANDLER_LOWERING if handler_lowering is None else str(handler_lowering).strip()
    if not normalized_handler_lowering:
        raise ValueError("Dataflow wrapper handler lowering label must be non-empty")
    normalized_queue_indexing = str(queue_indexing)
    if normalized_queue_indexing not in ("cta_rank", "smid"):
        raise ValueError(f"Dataflow queue_indexing must be 'cta_rank' or 'smid', got {queue_indexing!r}")
    handler_kinds = collect_handler_kinds(packed_plan)
    handler_names = packed_plan.handler_names
    handler_identities = packed_plan.handler_identities
    handler_variant_keys = packed_plan.handler_variant_keys
    handlers = tuple(
        DataflowHandlerSpec(
            handler_id=handler_id,
            operator_name=operator_name,
            operator_kind=(identity.operator_kind if identity is not None else handler_kinds.get(handler_id, "unknown")),
            symbol_name=f"dataflow_handler_{handler_id}_{sanitize_identifier(operator_name, prefix='handler')}",
            handler_identity=identity,
            handler_variant_key=variant_key,
        )
        for handler_id, (operator_name, identity, variant_key) in enumerate(zip(handler_names, handler_identities, handler_variant_keys))
    )
    cluster_send_comms = tuple(comm for comm in packed_plan.comms if comm.kind == DATAFLOW_COMM_KIND_ABI_VALUES["cluster_send"])
    has_cluster_release_comms = any(comm.kind == DATAFLOW_COMM_KIND_ABI_VALUES["cluster_release"] for comm in packed_plan.comms)
    gated_cluster_send_comms = tuple(
        comm for comm in cluster_send_comms if bool(packed_plan.slots[comm.dst_slot_id].flags & DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH)
    )
    gated_wait_keys = tuple(
        (
            packed_plan.slots[comm.src_slot_id].owner_cta,
            comm.flag_index,
        )
        for comm in gated_cluster_send_comms
    )
    return DataflowWrapperSpec(
        kernel_name=sanitized_kernel_name,
        handlers=handlers,
        launch_bound_threads=int(launch_bound_threads),
        shared_memory_bytes=0 if launch_package is None else launch_package.shared_memory_bytes,
        shared_slot_base_offset=0 if launch_package is None else launch_package.shared_slot_base_offset,
        primfunc_all_slots_scratch_backed=(
            False
            if launch_package is None
            else launch_package.slot_count > 0 and launch_package.scratch_backed_slot_count == launch_package.slot_count
        ),
        barrier_count=0 if launch_package is None else launch_package.barrier_count,
        cluster_ack_count=(0 if launch_package is None else launch_package.cluster_ack_count),
        comm_count=0 if launch_package is None else launch_package.comm_count,
        has_hbm_comms=any(
            comm.kind
            in {
                DATAFLOW_COMM_KIND_ABI_VALUES["hbm_send"],
                DATAFLOW_COMM_KIND_ABI_VALUES["hbm_recv"],
                DATAFLOW_COMM_KIND_ABI_VALUES["hbm_recv_issue"],
                DATAFLOW_COMM_KIND_ABI_VALUES["hbm_recv_wait"],
            }
            for comm in packed_plan.comms
        ),
        has_cluster_send_comms=bool(cluster_send_comms),
        has_cluster_release_comms=has_cluster_release_comms,
        has_gated_cluster_push=bool(gated_cluster_send_comms),
        unique_gated_cluster_waits=(bool(gated_wait_keys) and len(gated_wait_keys) == len(set(gated_wait_keys))),
        precomputed_cluster_source_lifetime=(
            bool(cluster_send_comms) and all(not packed_plan.slots[comm.src_slot_id].is_scratch_backed for comm in cluster_send_comms)
        ),
        has_cluster_source_reuse_wait_actions=any(args.reserved0 != 0 for args in packed_plan.args),
        segmented_hbm_tma=any(
            comm.segment_count > 1
            or comm.kind
            in {
                DATAFLOW_COMM_KIND_ABI_VALUES["hbm_recv_issue"],
                DATAFLOW_COMM_KIND_ABI_VALUES["hbm_recv_wait"],
            }
            for comm in packed_plan.comms
        ),
        cluster_size=int(cluster_size),
        tensor_arg_count=int(tensor_arg_count),
        handler_lowering=normalized_handler_lowering,
        queue_indexing=normalized_queue_indexing,
    )


def format_slot_shared_accessor(spec: DataflowWrapperSpec) -> str:
    if spec.primfunc_all_slots_scratch_backed:
        return """TL_DEVICE void *dataflow_slot_shared_ptr(
    void *shared_base,
    const tl::DataflowSlot &slot) {
  return tl::dataflow_byte_ptr(dataflow_primfunc_scratch_base(shared_base), slot.shared_offset);
}

TL_DEVICE const void *dataflow_slot_shared_ptr(
    const void *shared_base,
    const tl::DataflowSlot &slot) {
  return tl::dataflow_byte_ptr(dataflow_primfunc_scratch_base(shared_base), slot.shared_offset);
}"""
    return """static constexpr uint32_t kDataflowSlotFlagScratchBacked = tl::kDataflowSlotFlagScratchBacked;

TL_DEVICE bool dataflow_slot_is_scratch_backed(const tl::DataflowSlot &slot) {
  return (slot.flags & kDataflowSlotFlagScratchBacked) != 0u;
}

TL_DEVICE void *dataflow_slot_shared_ptr(
    void *shared_base,
    const tl::DataflowSlot &slot) {
  return tl::dataflow_slot_shared_ptr(
      shared_base, dataflow_primfunc_scratch_base(shared_base), slot);
}

TL_DEVICE const void *dataflow_slot_shared_ptr(
    const void *shared_base,
    const tl::DataflowSlot &slot) {
  return tl::dataflow_slot_shared_ptr(
      shared_base, dataflow_primfunc_scratch_base(shared_base), slot);
}"""


def generate_wrapper_source(
    spec: DataflowWrapperSpec,
    *,
    instrumentation: DataflowWrapperInstrumentation | None = None,
) -> str:
    if not isinstance(spec, DataflowWrapperSpec):
        raise TypeError(f"generate_wrapper_source expects DataflowWrapperSpec, got {spec!r}")
    if instrumentation is not None and not isinstance(instrumentation, DataflowWrapperInstrumentation):
        raise TypeError(f"generate_wrapper_source instrumentation must be DataflowWrapperInstrumentation, got {instrumentation!r}")
    if instrumentation is not None:
        reserved_kernel_params = {
            "instructions",
            "queue_offsets",
            "queue_lengths",
            "slots",
            "comm_plans",
            "barrier_init_offsets",
            "barrier_init_lengths",
            "barrier_init_indices",
            "arg_base",
            "tensor_args",
            "input_slots",
            "task_coords",
            "global_base",
            "flags",
            *spec.tma_descriptor_names,
        }
        collisions = tuple(param.name for param in instrumentation.extra_kernel_params if param.name in reserved_kernel_params)
        if collisions:
            raise ValueError(f"Dataflow wrapper instrumentation kernel parameters collide with the wrapper ABI: {collisions!r}")

    handler_sources_by_id = {source.handler_id: source for source in spec.handler_sources}
    custom_handler_bodies = {handler_id: source.body_source for handler_id, source in handler_sources_by_id.items()}
    missing = [handler.handler_id for handler in spec.handlers if handler.handler_id not in custom_handler_bodies]
    if missing:
        raise ValueError(
            f"Dataflow wrapper generation requires concrete handler bodies; {spec.handler_lowering!r} is missing handler ids: {missing}"
        )

    handler_helpers = spec.handler_helper_source
    handler_stubs = "\n\n".join(
        format_handler_stub(
            handler,
            body_source=custom_handler_bodies[handler.handler_id],
            tma_descriptors=spec.tma_descriptors,
        )
        for handler in spec.handlers
    )
    handler_cases = "\n".join(
        format_handler_case(
            handler,
            completion_synchronized=handler_sources_by_id[handler.handler_id].completion_synchronized,
            tma_descriptors=spec.tma_descriptors,
        )
        for handler in spec.handlers
    )
    if not handler_cases:
        handler_cases = (
            "  (void)inst;\n  (void)handler_args;\n  (void)slots;\n"
            "  (void)tensor_args;\n"
            "  (void)input_slots;\n  (void)task_coords;\n"
            "  (void)shared_base;\n  (void)global_base;"
        )

    queue_rank_expr = {
        "cta_rank": "tl::dataflow_cta_rank_in_grid()",
        "smid": "tl::dataflow_smid()",
    }[spec.queue_indexing]

    tma_global_kernel_params = "".join(
        f",\n    __grid_constant__ const CUtensorMap {descriptor.name}" for descriptor in spec.tma_descriptors
    )
    tma_device_params = "".join(f",\n    const CUtensorMap &{descriptor.name}" for descriptor in spec.tma_descriptors)
    tma_dispatch_args = "".join(f", {descriptor.name}" for descriptor in spec.tma_descriptors)
    instrumentation_kernel_params = ""
    if instrumentation is not None:
        instrumentation_kernel_params = "".join(f",\n    {param.declaration}" for param in instrumentation.extra_kernel_params)
    launch_bounds = format_launch_bounds(spec.launch_bound_threads)
    launch_pacing = format_launch_pacing(spec)
    startup_collective = format_startup_collective(spec, instrumentation)

    slot_shared_accessor = format_slot_shared_accessor(spec)
    instrumentation_namespace_source = instrumentation_hook(
        instrumentation,
        "namespace_source",
        indent=0,
    )
    instrumentation_kernel_prologue = instrumentation_hook(
        instrumentation,
        "kernel_prologue",
        indent=2,
    )
    instrumentation_before_barrier_init = instrumentation_hook(
        instrumentation,
        "before_barrier_init",
        indent=2,
    )
    instrumentation_after_barrier_init = instrumentation_hook(
        instrumentation,
        "after_barrier_init",
        indent=2,
    )
    instrumentation_after_cluster_sync = instrumentation_hook(
        instrumentation,
        "after_cluster_sync",
        indent=2,
    )
    instrumentation_before_loop = instrumentation_hook(
        instrumentation,
        "before_loop",
        indent=2,
    )
    instrumentation_before_recv = instrumentation_hook(
        instrumentation,
        "before_recv",
        indent=4,
    )
    instrumentation_after_recv = instrumentation_hook(
        instrumentation,
        "after_recv",
        indent=4,
    )
    instrumentation_after_handler = instrumentation_hook(
        instrumentation,
        "after_handler",
        indent=4,
    )
    instrumentation_after_send = instrumentation_hook(
        instrumentation,
        "after_send",
        indent=4,
    )
    instrumentation_kernel_epilogue = instrumentation_hook(
        instrumentation,
        "kernel_epilogue",
        indent=2,
    )

    hbm_tma_define = "#define TILELANG_DATAFLOW_HBM_USE_TMA 1\n" if spec.segmented_hbm_tma else ""
    if spec.precomputed_cluster_source_lifetime:
        cluster_source_state_declarations = ""
        cluster_source_before_receive = (
            """  const uint32_t cluster_source_wait_action =
      handler_args.reserved0;
  if (cluster_source_wait_action ==
      tl_dataflow_generated::kDataflowClusterSourceWaitAll) {
    tl_dataflow_generated::dataflow_wait_cluster_send_source_reads(
        cluster_ack_barriers, cluster_ack_pending);
  } else if (cluster_source_wait_action ==
             tl_dataflow_generated::kDataflowClusterSourceWaitPrevious) {
    tl_dataflow_generated::dataflow_wait_cluster_send_source_reads(
        cluster_ack_barriers, cluster_ack_pending, 1u);
  }"""
            if spec.has_cluster_source_reuse_wait_actions
            else ""
        )
        cluster_source_before_release = ""
        cluster_source_dispatch = """      (void)tl_dataflow_generated::dataflow_dispatch_send_comms(
          inst, handler_completion_synchronized, comm_plans, slots,
          shared_base, scratch_base, global_base, flags,
          barriers, cluster_ack_barriers, cluster_ack_pending,
          hbm_recv_issued, nullptr, nullptr);"""
        cluster_source_epilogue = """  if (tl_dataflow_generated::kDataflowHasClusterSendComms) {
    tl_dataflow_generated::dataflow_finish_cluster_send_source_reads();
  }"""
    else:
        cluster_source_state_declarations = """  bool cluster_send_source_pending = false;
  uint32_t cluster_send_source_begin = 0u;
  uint32_t cluster_send_source_end = 0u;"""
        cluster_source_before_receive = """  if (cluster_send_source_pending &&
      tl_dataflow_generated::dataflow_instruction_writes_pending_cluster_source(
          inst, comm_plans, slots, cluster_send_source_begin,
          cluster_send_source_end)) {
    tl_dataflow_generated::dataflow_wait_cluster_send_source_reads(
        cluster_ack_barriers, cluster_ack_pending);
    cluster_send_source_pending = false;
    cluster_send_source_begin = 0u;
    cluster_send_source_end = 0u;
  }"""
        cluster_source_before_release = """  if (tl_dataflow_generated::kDataflowHasClusterReleaseComms &&
      cluster_send_source_pending && instruction_has_comms &&
      tl_dataflow_generated::dataflow_instruction_releases_cluster_slot(
          inst, comm_plans)) {
    tl_dataflow_generated::dataflow_wait_cluster_send_source_reads(
        cluster_ack_barriers, cluster_ack_pending);
    cluster_send_source_pending = false;
    cluster_send_source_begin = 0u;
    cluster_send_source_end = 0u;
  }"""
        cluster_source_dispatch = """      cluster_send_source_pending =
          tl_dataflow_generated::dataflow_dispatch_send_comms(
              inst, handler_completion_synchronized, comm_plans, slots,
              shared_base, scratch_base, global_base, flags,
              barriers, cluster_ack_barriers, cluster_ack_pending,
              hbm_recv_issued, &cluster_send_source_begin,
              &cluster_send_source_end) || cluster_send_source_pending;"""
        cluster_source_epilogue = """  if (cluster_send_source_pending) {
    tl_dataflow_generated::dataflow_finish_cluster_send_source_reads();
  }"""
    source = f"""{hbm_tma_define}#include <stdint.h>
#include <tl_templates/cuda/cluster.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/dataflow_comm.h>
#include <tl_templates/cuda/reduce.h>

namespace tl_dataflow_generated {{

static constexpr uint32_t kDataflowSharedSlotBaseOffset = {spec.shared_slot_base_offset}u;
static constexpr uint32_t kDataflowPrimFuncScratchOffset = {spec.primfunc_scratch_offset}u;
static constexpr uint32_t kDataflowHandoffArenaOffset = {spec.handoff_arena_offset}u;
static constexpr uint32_t kDataflowHandoffArenaBytes = {spec.handoff_arena_bytes}u;
static constexpr bool kDataflowPrimFuncUseGlobalSlotFields = {str(spec.primfunc_use_global_slot_fields).lower()};
static constexpr uint32_t kDataflowBarrierCount = {spec.barrier_count}u;
static constexpr uint32_t kDataflowBarrierWordCount =
    (kDataflowBarrierCount + 31u) / 32u;
static constexpr uint32_t kDataflowClusterAckCount = {spec.cluster_ack_count}u;
static constexpr uint32_t kDataflowClusterAckWordCount =
    (kDataflowClusterAckCount + 31u) / 32u;
static constexpr uint32_t kDataflowClusterInboxOffset = {spec.cluster_inbox_offset}u;
static constexpr uint32_t kDataflowClusterInboxBytes = {spec.cluster_inbox_bytes}u;
static constexpr uint32_t kDataflowClusterSize = {spec.cluster_size}u;
static constexpr uint32_t kDataflowTensorArgCount = {spec.tensor_arg_count}u;
static constexpr bool kDataflowHasHBMComms = {str(spec.has_hbm_comms).lower()};
static constexpr bool kDataflowHasClusterSendComms = {str(spec.has_cluster_send_comms).lower()};
static constexpr bool kDataflowHasClusterReleaseComms = {str(spec.has_cluster_release_comms).lower()};
static constexpr bool kDataflowHasGatedClusterPush = {str(spec.has_gated_cluster_push).lower()};
static constexpr bool kDataflowUniqueGatedClusterWaits = {str(spec.unique_gated_cluster_waits).lower()};
static constexpr bool kDataflowPrecomputedClusterSourceLifetime = {str(spec.precomputed_cluster_source_lifetime).lower()};
static constexpr bool kDataflowHasClusterSourceReuseWaitActions = {str(spec.has_cluster_source_reuse_wait_actions).lower()};
static constexpr uint32_t kDataflowClusterSourceWaitAll = {DATAFLOW_CLUSTER_SOURCE_WAIT_ALL}u;
static constexpr uint32_t kDataflowClusterSourceWaitPrevious = {DATAFLOW_CLUSTER_SOURCE_WAIT_PREVIOUS}u;
{instrumentation_namespace_source}

TL_DEVICE void *dataflow_primfunc_scratch_base(void *shared_base) {{
  return reinterpret_cast<uint8_t *>(shared_base) -
         kDataflowSharedSlotBaseOffset + kDataflowPrimFuncScratchOffset;
}}

TL_DEVICE const void *dataflow_primfunc_scratch_base(const void *shared_base) {{
  return reinterpret_cast<const uint8_t *>(shared_base) -
         kDataflowSharedSlotBaseOffset + kDataflowPrimFuncScratchOffset;
}}
{slot_shared_accessor}

template <typename BarrierType>
TL_DEVICE void dataflow_dispatch_comm(
    const tl::DataflowCommPlan &comm,
    const tl::DataflowSlot *slots,
    void *shared_base,
    void *scratch_base,
    void *global_base,
    uint32_t *flags,
    BarrierType *barriers,
    uint32_t *hbm_recv_issued,
    bool hbm_flag_ready = false) {{
  const tl::DataflowSlot &src_slot = slots[comm.src_slot_id];
  const tl::DataflowSlot &dst_slot = slots[comm.dst_slot_id];
  const bool direct_hbm_slot =
      tl::dataflow_slot_uses_hbm_direct_global(src_slot) &&
      tl::dataflow_slot_uses_hbm_direct_global(dst_slot) &&
      src_slot.global_offset == dst_slot.global_offset;
  if (kDataflowPrimFuncUseGlobalSlotFields && direct_hbm_slot &&
      comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kHBMSend)) {{
    tl::dataflow_send_hbm_direct_global(
        flags,
        tl::dataflow_select_index(comm.flag_index, dst_slot.flag_index),
        comm.flag_epoch);
    return;
  }}
  if (kDataflowPrimFuncUseGlobalSlotFields && direct_hbm_slot &&
      comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kHBMRecv)) {{
    tl::dataflow_recv_hbm_direct_global(
        flags,
        tl::dataflow_select_index(comm.flag_index, dst_slot.flag_index),
        comm.flag_epoch);
    return;
  }}
  if (kDataflowPrimFuncUseGlobalSlotFields && direct_hbm_slot &&
      comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kHBMRecvIssue)) {{
    return;
  }}
  if (kDataflowPrimFuncUseGlobalSlotFields && direct_hbm_slot &&
      comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kHBMRecvWait)) {{
    tl::dataflow_recv_hbm_direct_global(
        flags,
        tl::dataflow_select_index(comm.flag_index, dst_slot.flag_index),
        comm.flag_epoch);
    return;
  }}
  tl::dataflow_comm_dispatch(comm, slots, shared_base, scratch_base, global_base,
                          flags, barriers, hbm_recv_issued, hbm_flag_ready);
}}

template <typename BarrierType>
TL_DEVICE void dataflow_dispatch_cluster_recv_relaxed(
    const tl::DataflowCommPlan &comm,
    const tl::DataflowSlot *slots,
    void *shared_base,
    void *scratch_base,
    BarrierType *barriers,
    BarrierType *cluster_ack_barriers) {{
  const tl::DataflowSlot &src_slot = slots[comm.src_slot_id];
  const tl::DataflowSlot &dst_slot = slots[comm.dst_slot_id];
  const uint32_t bytes = tl::dataflow_comm_bytes(comm, src_slot, dst_slot);
  const uint32_t barrier_index =
      tl::dataflow_select_index(comm.barrier_index, dst_slot.barrier_index);
  (void)src_slot;
  (void)bytes;
  (void)shared_base;
  (void)scratch_base;
  (void)cluster_ack_barriers;
  tl::dataflow_recv_cluster_relaxed(barriers[barrier_index], comm.flag_epoch);
}}

template <typename BarrierType>
TL_DEVICE void dataflow_dispatch_cluster_send_relaxed(
    const tl::DataflowCommPlan &comm,
    const tl::DataflowSlot *slots,
    void *shared_base,
    void *scratch_base,
    BarrierType *barriers,
    BarrierType *cluster_ack_barriers,
    uint32_t *cluster_ack_pending) {{
  if (kDataflowHasClusterReleaseComms && comm.kind ==
      static_cast<uint32_t>(tl::DataflowCommKind::kClusterRelease)) {{
    if (tl::dataflow_is_leader_thread()) {{
      tl::mbarrier_arrive(
          cluster_ack_barriers[comm.flag_index], comm.peer_cta_rank, 1u);
    }}
    return;
  }}
  const tl::DataflowSlot &src_slot = slots[comm.src_slot_id];
  const tl::DataflowSlot &dst_slot = slots[comm.dst_slot_id];
  const uint32_t bytes = tl::dataflow_comm_bytes(comm, src_slot, dst_slot);
  const uint32_t barrier_index =
      tl::dataflow_select_index(comm.barrier_index, dst_slot.barrier_index);
  if (tl::dataflow_is_leader_thread()) {{
    if (kDataflowHasGatedClusterPush &&
        tl::dataflow_slot_uses_cluster_gated_push(dst_slot)) {{
      if (kDataflowUniqueGatedClusterWaits) {{
        tl::mbarrier_wait(cluster_ack_barriers[comm.flag_index], 0);
      }} else {{
        const uint32_t word = comm.flag_index >> 5u;
        const uint32_t mask = 1u << (comm.flag_index & 31u);
        if ((cluster_ack_pending[word] & mask) == 0u) {{
          tl::mbarrier_wait(cluster_ack_barriers[comm.flag_index], 0);
          cluster_ack_pending[word] |= mask;
        }}
      }}
    }}
  }}
  tl::dataflow_send_cluster_relaxed(
      tl::dataflow_slot_shared_ptr(shared_base, scratch_base, dst_slot),
      tl::dataflow_slot_shared_ptr(shared_base, scratch_base, src_slot),
      comm.peer_cta_rank, bytes, barriers[barrier_index]);
}}

template <typename BarrierType>
TL_DEVICE void dataflow_init_queue_recv_barriers(
    uint32_t queue_rank,
    const uint32_t *barrier_init_offsets,
    const uint32_t *barrier_init_lengths,
    const uint32_t *barrier_init_indices,
    BarrierType *barriers) {{
  uint32_t offset = barrier_init_offsets[queue_rank];
  uint32_t length = barrier_init_lengths[queue_rank];
  if (length == 0u) {{
    return;
  }}
  if (tl::dataflow_is_leader_thread()) {{
    for (uint32_t i = 0; i < length; ++i) {{
      const uint32_t barrier_index = barrier_init_indices[offset + i];
      tl::mbarrier_init(barriers[barrier_index], 1u);
    }}
    // Publish the complete per-queue barrier set once.  Initializing each
    // barrier through dataflow_init_cluster_barrier would put a CTA barrier in
    // every loop iteration, making queues with more inbound edges start later
    // even though all initialization is leader-thread work.
    tl::fence_barrier_init();
  }}
}}

template <typename BarrierType>
TL_DEVICE void dataflow_init_cluster_ack_state(
    const tl::DataflowQueue &queue,
    uint32_t queue_rank,
    const tl::DataflowCommPlan *comm_plans,
    const tl::DataflowSlot *slots,
    BarrierType *cluster_ack_barriers,
    uint32_t *cluster_ack_pending) {{
  if (tl::dataflow_is_leader_thread()) {{
    bool initialized_gated_barrier = false;
    if (!kDataflowUniqueGatedClusterWaits) {{
      for (uint32_t i = 0; i < kDataflowClusterAckWordCount; ++i) {{
        cluster_ack_pending[i] = 0u;
      }}
    }}
    const uint32_t queue_length = tl::dataflow_queue_length(queue, queue_rank);
    for (uint32_t pc = 0; pc < queue_length; ++pc) {{
      const tl::DataflowInstruction inst =
          tl::dataflow_queue_load(queue, queue_rank, pc);
      const uint32_t cluster_recv_count =
          tl::dataflow_instruction_cluster_recv_count(inst);
      const uint32_t cluster_send_count =
          tl::dataflow_instruction_cluster_send_count(inst);
      for (uint32_t j = 0; j < cluster_send_count; ++j) {{
        const tl::DataflowCommPlan &comm =
            comm_plans[inst.comm_offset + cluster_recv_count + j];
        if (comm.kind ==
                static_cast<uint32_t>(tl::DataflowCommKind::kClusterSend) &&
            tl::dataflow_slot_uses_cluster_gated_push(slots[comm.dst_slot_id])) {{
          const uint32_t word = comm.flag_index >> 5u;
          const uint32_t mask = 1u << (comm.flag_index & 31u);
          if (kDataflowUniqueGatedClusterWaits ||
              (cluster_ack_pending[word] & mask) == 0u) {{
            tl::mbarrier_init(cluster_ack_barriers[comm.flag_index], 1u);
            if (!kDataflowUniqueGatedClusterWaits) {{
              cluster_ack_pending[word] |= mask;
            }}
            initialized_gated_barrier = true;
          }}
        }}
      }}
    }}
    if (initialized_gated_barrier) {{
      tl::fence_barrier_init();
    }}
    if (!kDataflowUniqueGatedClusterWaits) {{
      for (uint32_t i = 0; i < kDataflowClusterAckWordCount; ++i) {{
        cluster_ack_pending[i] = 0u;
      }}
    }}
  }}
}}

TL_DEVICE void dataflow_init_hbm_recv_state(uint32_t *hbm_recv_issued) {{
  if (tl::dataflow_is_leader_thread()) {{
    for (uint32_t i = 0; i < kDataflowBarrierWordCount; ++i) {{
      hbm_recv_issued[i] = 0u;
    }}
  }}
}}

template <typename BarrierType>
TL_DEVICE void dataflow_wait_cluster_send_source_reads(
    BarrierType *cluster_ack_barriers,
    uint32_t *cluster_ack_pending,
    uint32_t keep_groups = 0u) {{
  if (tl::dataflow_is_leader_thread()) {{
    // A shared::cta -> shared::cluster bulk copy is also tracked by the
    // issuing thread's bulk async group.  The .read wait is precisely the
    // source-lifetime fence: it permits the producer to reuse its outbox as
    // soon as the async engine has consumed the source, without waiting for
    // the destination CTA to reach the eventual reduce handler.  Destination
    // readiness remains independently guarded by its receive mbarrier.
    if (keep_groups == 0u) {{
      tl::tma_store_wait<0>();
    }} else {{
      // The tracker only requests the one-group form: it proves that the
      // newest group does not overlap the storage about to be written.
      tl::tma_store_wait<1>();
    }}
  }}
  (void)cluster_ack_barriers;
  (void)cluster_ack_pending;
  __syncthreads();
}}

TL_DEVICE void dataflow_finish_cluster_send_source_reads() {{
  // At an in-kernel reuse boundary all threads must observe source-read
  // completion before any writer proceeds, hence the synchronized helper
  // above.  At kernel exit the issuing leader only needs to keep the CTA alive
  // until its own bulk-async read group is complete; block teardown preserves
  // shared memory until every thread (including that leader) has exited.
  if (tl::dataflow_is_leader_thread()) {{
    tl::tma_store_wait<0>();
  }}
}}

TL_DEVICE uint32_t dataflow_slot_shared_arena_offset(
    const tl::DataflowSlot &slot) {{
  return (tl::dataflow_slot_is_scratch_backed(slot)
              ? kDataflowPrimFuncScratchOffset
              : kDataflowSharedSlotBaseOffset) +
         slot.shared_offset;
}}

TL_DEVICE bool dataflow_slot_overlaps_pending_cluster_source(
    const tl::DataflowSlot &slot,
    uint32_t pending_begin,
    uint32_t pending_end) {{
  if (pending_begin >= pending_end) {{
    return false;
  }}
  const uint32_t slot_begin = dataflow_slot_shared_arena_offset(slot);
  const uint32_t slot_end = slot_begin + slot.bytes;
  return slot_begin < pending_end && pending_begin < slot_end;
}}

TL_DEVICE bool dataflow_instruction_writes_pending_cluster_source(
    const tl::DataflowInstruction &inst,
    const tl::DataflowCommPlan *comm_plans,
    const tl::DataflowSlot *slots,
    uint32_t pending_begin,
    uint32_t pending_end) {{
  if (inst.slot_id != tl::kDataflowInvalidIndex &&
      dataflow_slot_overlaps_pending_cluster_source(
          slots[inst.slot_id], pending_begin, pending_end)) {{
    return true;
  }}
  for (uint32_t j = 0; j < inst.comm_count; ++j) {{
    const tl::DataflowCommPlan &comm = comm_plans[inst.comm_offset + j];
    const bool writes_destination =
        comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kClusterRecv) ||
        comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kHBMRecv) ||
        comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kHBMRecvIssue) ||
        comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kHBMRecvWait);
    if (writes_destination &&
        dataflow_slot_overlaps_pending_cluster_source(
            slots[comm.dst_slot_id], pending_begin, pending_end)) {{
      return true;
    }}
  }}
  return false;
}}

TL_DEVICE bool dataflow_instruction_releases_cluster_slot(
    const tl::DataflowInstruction &inst,
    const tl::DataflowCommPlan *comm_plans) {{
  const uint32_t cluster_recv_count =
      tl::dataflow_instruction_cluster_recv_count(inst);
  const uint32_t cluster_send_count =
      tl::dataflow_instruction_cluster_send_count(inst);
  for (uint32_t j = 0; j < cluster_send_count; ++j) {{
    const tl::DataflowCommPlan &comm =
        comm_plans[inst.comm_offset + cluster_recv_count + j];
    if (comm.kind ==
        static_cast<uint32_t>(tl::DataflowCommKind::kClusterRelease)) {{
      return true;
    }}
  }}
  return false;
}}

template <typename BarrierType>
TL_DEVICE void dataflow_dispatch_cluster_recv_comms(
    const tl::DataflowInstruction &inst,
    const tl::DataflowCommPlan *comm_plans,
    const tl::DataflowSlot *slots,
    void *shared_base,
    void *scratch_base,
    BarrierType *barriers,
    BarrierType *cluster_ack_barriers) {{
  const uint32_t cluster_recv_count =
      tl::dataflow_instruction_cluster_recv_count(inst);
  const uint32_t cluster_send_count =
      tl::dataflow_instruction_cluster_send_count(inst);
  for (uint32_t j = 0; j < cluster_recv_count; ++j) {{
    const tl::DataflowCommPlan &comm = comm_plans[inst.comm_offset + j];
    dataflow_dispatch_cluster_recv_relaxed(
        comm, slots, shared_base, scratch_base, barriers,
        cluster_ack_barriers);
  }}
  if (cluster_recv_count != 0u) {{
    tl::dataflow_complete_cluster_recv_batch();
  }}
}}

template <typename BarrierType>
TL_DEVICE void dataflow_dispatch_hbm_recv_issue_comms(
    const tl::DataflowInstruction &inst,
    const tl::DataflowCommPlan *comm_plans,
    const tl::DataflowSlot *slots,
    void *shared_base,
    void *scratch_base,
    void *global_base,
    uint32_t *flags,
    BarrierType *barriers,
    BarrierType *cluster_ack_barriers,
    uint32_t *hbm_recv_issued) {{
  const uint32_t cluster_recv_count =
      tl::dataflow_instruction_cluster_recv_count(inst);
  const uint32_t cluster_send_count =
      tl::dataflow_instruction_cluster_send_count(inst);
  (void)cluster_ack_barriers;
  if (!tl::dataflow_is_leader_thread()) {{
    return;
  }}
  for (uint32_t j = cluster_recv_count + cluster_send_count;
       j < inst.comm_count; ++j) {{
    const tl::DataflowCommPlan &comm = comm_plans[inst.comm_offset + j];
    if (comm.kind !=
        static_cast<uint32_t>(tl::DataflowCommKind::kHBMRecvIssue)) {{
      continue;
    }}
    const uint32_t remaining = inst.comm_count - j;
    const uint32_t segment_count = comm.segment_count;
    const tl::DataflowSlot &first_src = slots[comm.src_slot_id];
    const tl::DataflowSlot &first_dst = slots[comm.dst_slot_id];
    const bool direct_hbm_slot =
        kDataflowPrimFuncUseGlobalSlotFields &&
        tl::dataflow_slot_uses_hbm_direct_global(first_src) &&
        tl::dataflow_slot_uses_hbm_direct_global(first_dst) &&
        first_src.global_offset == first_dst.global_offset;
    bool grouped = segment_count > 1u && comm.segment_id == 0u &&
                   segment_count <= remaining && !direct_hbm_slot;
    if (!grouped) {{
      dataflow_dispatch_comm(comm, slots, shared_base, scratch_base, global_base,
                          flags, barriers, hbm_recv_issued);
      continue;
    }}
    const tl::DataflowCommPlan &last =
        comm_plans[inst.comm_offset + j + segment_count - 1u];
    const tl::DataflowSlot &last_dst = slots[last.dst_slot_id];
    const uint32_t group_barrier =
        tl::dataflow_select_index(comm.barrier_index, first_dst.barrier_index);
#if defined(TILELANG_DATAFLOW_HBM_USE_TMA) && defined(__CUDA_ARCH__) &&          \
    (__CUDA_ARCH__ >= 900)
    if (!tl::dataflow_hbm_recv_is_issued(
            hbm_recv_issued, group_barrier) &&
        tl::dataflow_load_flag_acquire(
            flags,
            tl::dataflow_select_index(last.flag_index, last_dst.flag_index)) >=
            last.flag_epoch) {{
      uint32_t total_bytes = 0u;
      for (uint32_t segment_id = 0; segment_id < segment_count;
           ++segment_id) {{
        const tl::DataflowCommPlan &segment =
            comm_plans[inst.comm_offset + j + segment_id];
        total_bytes += tl::dataflow_comm_bytes(
            segment, slots[segment.src_slot_id], slots[segment.dst_slot_id]);
      }}
      tl::mbarrier_arrive_expect_tx(barriers[group_barrier], total_bytes);
      for (uint32_t segment_id = 0; segment_id < segment_count;
           ++segment_id) {{
        const tl::DataflowCommPlan &segment =
            comm_plans[inst.comm_offset + j + segment_id];
        const tl::DataflowSlot &segment_src = slots[segment.src_slot_id];
        const tl::DataflowSlot &segment_dst = slots[segment.dst_slot_id];
        const uint32_t bytes = tl::dataflow_comm_bytes(
            segment, segment_src, segment_dst);
        tl::tma_load(
            tl::dataflow_comm_byte_ptr(
                tl::dataflow_slot_shared_ptr(
                    shared_base, scratch_base, segment_dst),
                segment),
            tl::dataflow_comm_byte_ptr(
                tl::dataflow_slot_global_ptr(global_base, segment_dst),
                segment),
            barriers[group_barrier], bytes);
      }}
      tl::dataflow_hbm_recv_mark_issued(hbm_recv_issued, group_barrier);
    }}
#else
    (void)last;
    (void)last_dst;
    (void)group_barrier;
#endif
    j += segment_count - 1u;
  }}
}}

template <typename BarrierType>
TL_DEVICE void dataflow_dispatch_hbm_recv_wait_comms(
    const tl::DataflowInstruction &inst,
    const tl::DataflowCommPlan *comm_plans,
    const tl::DataflowSlot *slots,
    void *shared_base,
    void *scratch_base,
    void *global_base,
    uint32_t *flags,
    BarrierType *barriers,
    uint32_t *hbm_recv_issued) {{
  const uint32_t cluster_recv_count =
      tl::dataflow_instruction_cluster_recv_count(inst);
  const uint32_t cluster_send_count =
      tl::dataflow_instruction_cluster_send_count(inst);
  for (uint32_t j = cluster_recv_count + cluster_send_count;
       j < inst.comm_count; ++j) {{
    const tl::DataflowCommPlan &comm = comm_plans[inst.comm_offset + j];
    const bool is_hbm_wait =
        comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kHBMRecv) ||
        comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kHBMRecvWait);
    if (!is_hbm_wait) {{
      continue;
    }}
    const uint32_t remaining = inst.comm_count - j;
    const uint32_t segment_count = comm.segment_count;
    const tl::DataflowSlot &first_src = slots[comm.src_slot_id];
    const tl::DataflowSlot &first_dst = slots[comm.dst_slot_id];
    const bool direct_hbm_slot =
        kDataflowPrimFuncUseGlobalSlotFields &&
        tl::dataflow_slot_uses_hbm_direct_global(first_src) &&
        tl::dataflow_slot_uses_hbm_direct_global(first_dst) &&
        first_src.global_offset == first_dst.global_offset;
    const bool grouped = segment_count > 1u && comm.segment_id == 0u &&
                         segment_count <= remaining && !direct_hbm_slot;
    if (!grouped) {{
      dataflow_dispatch_comm(comm, slots, shared_base, scratch_base, global_base,
                          flags, barriers, hbm_recv_issued);
      continue;
    }}
#if defined(TILELANG_DATAFLOW_HBM_USE_TMA) && defined(__CUDA_ARCH__) &&          \
    (__CUDA_ARCH__ >= 900)
    const tl::DataflowCommPlan &last =
        comm_plans[inst.comm_offset + j + segment_count - 1u];
    const tl::DataflowSlot &last_dst = slots[last.dst_slot_id];
    const uint32_t group_barrier =
        tl::dataflow_select_index(comm.barrier_index, first_dst.barrier_index);
    if (tl::dataflow_is_leader_thread()) {{
      if (!tl::dataflow_hbm_recv_is_issued(
              hbm_recv_issued, group_barrier)) {{
        while (tl::dataflow_load_flag_acquire(
                   flags,
                   tl::dataflow_select_index(
                       last.flag_index, last_dst.flag_index)) <
               last.flag_epoch) {{
          __nanosleep(64);
        }}
        uint32_t total_bytes = 0u;
        for (uint32_t segment_id = 0; segment_id < segment_count;
             ++segment_id) {{
          const tl::DataflowCommPlan &segment =
              comm_plans[inst.comm_offset + j + segment_id];
          total_bytes += tl::dataflow_comm_bytes(
              segment, slots[segment.src_slot_id],
              slots[segment.dst_slot_id]);
        }}
        tl::mbarrier_arrive_expect_tx(
            barriers[group_barrier], total_bytes);
        for (uint32_t segment_id = 0; segment_id < segment_count;
             ++segment_id) {{
          const tl::DataflowCommPlan &segment =
              comm_plans[inst.comm_offset + j + segment_id];
          const tl::DataflowSlot &segment_src = slots[segment.src_slot_id];
          const tl::DataflowSlot &segment_dst = slots[segment.dst_slot_id];
          const uint32_t bytes = tl::dataflow_comm_bytes(
              segment, segment_src, segment_dst);
          tl::tma_load(
              tl::dataflow_comm_byte_ptr(
                  tl::dataflow_slot_shared_ptr(
                      shared_base, scratch_base, segment_dst),
                  segment),
              tl::dataflow_comm_byte_ptr(
                  tl::dataflow_slot_global_ptr(global_base, segment_dst),
                  segment),
              barriers[group_barrier], bytes);
        }}
      }}
      tl::mbarrier_wait(
          barriers[group_barrier], static_cast<int>(comm.barrier_phase & 1u));
      tl::dataflow_hbm_recv_clear_issued(
          hbm_recv_issued, group_barrier);
      tl::fence_proxy_async();
    }}
    __syncthreads();
#else
    for (uint32_t segment_id = 0; segment_id < segment_count;
         ++segment_id) {{
      dataflow_dispatch_comm(
          comm_plans[inst.comm_offset + j + segment_id], slots,
          shared_base, scratch_base, global_base, flags, barriers,
          hbm_recv_issued);
    }}
#endif
    j += segment_count - 1u;
  }}
}}

template <typename BarrierType>
TL_DEVICE bool dataflow_dispatch_send_comms(
    const tl::DataflowInstruction &inst,
    bool handler_completion_synchronized,
    const tl::DataflowCommPlan *comm_plans,
    const tl::DataflowSlot *slots,
    void *shared_base,
    void *scratch_base,
    void *global_base,
    uint32_t *flags,
    BarrierType *barriers,
    BarrierType *cluster_ack_barriers,
    uint32_t *cluster_ack_pending,
    uint32_t *hbm_recv_issued,
    uint32_t *batch_source_begin,
    uint32_t *batch_source_end) {{
  const uint32_t cluster_recv_count =
      tl::dataflow_instruction_cluster_recv_count(inst);
  const uint32_t cluster_send_count =
      tl::dataflow_instruction_cluster_send_count(inst);
  if (cluster_send_count != 0u && !handler_completion_synchronized) {{
    tl::dataflow_begin_cluster_send_batch();
  }}
  bool issued_cluster_copy = false;
  for (uint32_t j = 0; j < cluster_send_count; ++j) {{
    const tl::DataflowCommPlan &comm =
        comm_plans[inst.comm_offset + cluster_recv_count + j];
    // The packed-plan summary makes the common cluster-only path entirely
    // branch-free while preserving mixed send/release plans.  If the plan has
    // no release records, every entry in this ABI partition is a cluster copy.
    const bool is_cluster_copy = !kDataflowHasClusterReleaseComms ||
        comm.kind == static_cast<uint32_t>(tl::DataflowCommKind::kClusterSend);
    issued_cluster_copy = issued_cluster_copy || is_cluster_copy;
    if (is_cluster_copy && batch_source_begin != nullptr &&
        batch_source_end != nullptr) {{
      const tl::DataflowSlot &source_slot = slots[comm.src_slot_id];
      const uint32_t source_begin =
          dataflow_slot_shared_arena_offset(source_slot);
      const uint32_t source_end = source_begin + source_slot.bytes;
      if (*batch_source_begin >= *batch_source_end) {{
        *batch_source_begin = source_begin;
        *batch_source_end = source_end;
      }} else {{
        *batch_source_begin = min(*batch_source_begin, source_begin);
        *batch_source_end = max(*batch_source_end, source_end);
      }}
    }}
    dataflow_dispatch_cluster_send_relaxed(
        comm, slots, shared_base, scratch_base, barriers,
        cluster_ack_barriers, cluster_ack_pending);
  }}
  if (issued_cluster_copy && tl::dataflow_is_leader_thread()) {{
    tl::tma_store_arrive();
  }}
  if (kDataflowHasHBMComms) {{
    for (uint32_t j = cluster_recv_count + cluster_send_count;
         j < inst.comm_count; ++j) {{
    const tl::DataflowCommPlan &comm = comm_plans[inst.comm_offset + j];
    if (comm.kind != static_cast<uint32_t>(tl::DataflowCommKind::kHBMSend)) {{
      continue;
    }}
    const uint32_t remaining = inst.comm_count - j;
    const uint32_t segment_count = comm.segment_count;
    const tl::DataflowSlot &first_src = slots[comm.src_slot_id];
    const tl::DataflowSlot &first_dst = slots[comm.dst_slot_id];
    const bool direct_hbm_slot =
        kDataflowPrimFuncUseGlobalSlotFields &&
        tl::dataflow_slot_uses_hbm_direct_global(first_src) &&
        tl::dataflow_slot_uses_hbm_direct_global(first_dst) &&
        first_src.global_offset == first_dst.global_offset;
    const bool grouped = segment_count > 1u && comm.segment_id == 0u &&
                         segment_count <= remaining && !direct_hbm_slot;
    if (!grouped) {{
      dataflow_dispatch_comm(comm, slots, shared_base, scratch_base, global_base,
                          flags, barriers, hbm_recv_issued);
      continue;
    }}
#if defined(TILELANG_DATAFLOW_HBM_USE_TMA) && defined(__CUDA_ARCH__) &&          \
    (__CUDA_ARCH__ >= 900)
    __syncthreads();
    if (tl::dataflow_is_leader_thread()) {{
      tl::fence_proxy_async();
      for (uint32_t segment_id = 0; segment_id < segment_count;
           ++segment_id) {{
        const tl::DataflowCommPlan &segment =
            comm_plans[inst.comm_offset + j + segment_id];
        const tl::DataflowSlot &segment_src = slots[segment.src_slot_id];
        const tl::DataflowSlot &segment_dst = slots[segment.dst_slot_id];
        tl::tma_store<tl::CacheHintSm90::EVICT_LAST>(
            tl::dataflow_comm_byte_ptr(
                tl::dataflow_slot_global_ptr(global_base, segment_dst),
                segment),
            tl::dataflow_comm_byte_ptr(
                tl::dataflow_slot_shared_ptr(
                    shared_base, scratch_base, segment_src),
                segment),
            tl::dataflow_comm_bytes(segment, segment_src, segment_dst));
      }}
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      const tl::DataflowCommPlan &last =
          comm_plans[inst.comm_offset + j + segment_count - 1u];
      const tl::DataflowSlot &last_dst = slots[last.dst_slot_id];
      tl::dataflow_store_flag_release(
          flags,
          tl::dataflow_select_index(last.flag_index, last_dst.flag_index),
          last.flag_epoch);
    }}
    __syncthreads();
#else
    for (uint32_t segment_id = 0; segment_id < segment_count;
         ++segment_id) {{
      dataflow_dispatch_comm(
          comm_plans[inst.comm_offset + j + segment_id], slots,
          shared_base, scratch_base, global_base, flags, barriers,
          hbm_recv_issued);
    }}
#endif
      j += segment_count - 1u;
    }}
  }}
  // Keep the source-read group live until the next source reuse (or kernel
  // epilogue).  Even a dedicated source must be complete before its CTA exits.
  return issued_cluster_copy;
}}

{handler_helpers}

{handler_stubs}

TL_DEVICE bool dataflow_dispatch_handler(
    const tl::DataflowInstruction &inst,
    const tl::DataflowHandlerArgs &handler_args,
    const void *arg_base,
    const tl::DataflowSlot *slots,
    const tl::DataflowTensorArg *tensor_args,
    const uint32_t *input_slots,
    const uint32_t *task_coords,
    void *shared_base,
    void *global_base{tma_device_params}) {{
  if (tl::dataflow_opcode_is_cluster_sync(inst)) {{
    if (kDataflowClusterSize > 1u) {{
      tl::cluster_sync();
    }}
    return true;
  }}
  switch (inst.handler_id) {{
{handler_cases}
  default:
    return false;
  }}
  return false;
}}

}}  // namespace tl_dataflow_generated

extern "C" __global__ void {launch_bounds}{spec.kernel_name}(
    const tl::DataflowInstruction *instructions,
    const uint32_t *queue_offsets,
    const uint32_t *queue_lengths,
    const tl::DataflowSlot *slots,
    const tl::DataflowCommPlan *comm_plans,
    const uint32_t *barrier_init_offsets,
    const uint32_t *barrier_init_lengths,
    const uint32_t *barrier_init_indices,
    void *arg_base,
    const tl::DataflowTensorArg *tensor_args,
    const uint32_t *input_slots,
    const uint32_t *task_coords,
    void *global_base,
    uint32_t *flags{instrumentation_kernel_params}{tma_global_kernel_params}) {{
{instrumentation_kernel_prologue}
  extern __shared__ __align__({DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES}) uint8_t dataflow_shared[];
  auto *barriers = reinterpret_cast<uint64_t *>(dataflow_shared);
  auto *cluster_ack_barriers =
      barriers + tl_dataflow_generated::kDataflowBarrierCount;
  auto *cluster_ack_pending = reinterpret_cast<uint32_t *>(
      cluster_ack_barriers + tl_dataflow_generated::kDataflowClusterAckCount);
  auto *hbm_recv_issued =
      cluster_ack_pending + tl_dataflow_generated::kDataflowClusterAckWordCount;
  void *shared_base = dataflow_shared + tl_dataflow_generated::kDataflowSharedSlotBaseOffset;
  void *scratch_base = dataflow_shared + tl_dataflow_generated::kDataflowPrimFuncScratchOffset;
{launch_pacing}
  tl::DataflowQueue queue{{instructions, queue_offsets, queue_lengths}};
  uint32_t queue_rank = {queue_rank_expr};
{instrumentation_before_barrier_init}
  tl_dataflow_generated::dataflow_init_queue_recv_barriers(
      queue_rank, barrier_init_offsets, barrier_init_lengths, barrier_init_indices, barriers);
  if (tl_dataflow_generated::kDataflowClusterAckCount != 0u) {{
    tl_dataflow_generated::dataflow_init_cluster_ack_state(
        queue, queue_rank, comm_plans, slots, cluster_ack_barriers,
        cluster_ack_pending);
  }}
  if (tl_dataflow_generated::kDataflowHasHBMComms) {{
    tl_dataflow_generated::dataflow_init_hbm_recv_state(hbm_recv_issued);
  }}
  // The initialization helpers only perform leader-thread writes to disjoint
  // control regions.  Cluster transport publishes them once before a remote
  // sender can arrive.  In an HBM-only production wrapper the same leader
  // initializes and first consumes every control word, and the receive wait
  // supplies the CTA barrier before payload use, so no startup rendezvous is
  // required.  Instrumented wrappers retain local visibility for semantic
  // hooks that inspect initialized state.
{startup_collective}
{instrumentation_after_barrier_init}
{instrumentation_after_cluster_sync}

  uint32_t length = tl::dataflow_queue_length(queue, queue_rank);
{cluster_source_state_declarations}
{instrumentation_before_loop}

  for (uint32_t pc = 0; pc < length; ++pc) {{
    tl::DataflowInstruction inst = tl::dataflow_queue_load(queue, queue_rank, pc);
    if (tl::dataflow_opcode_is_exit(inst)) {{
      break;
    }}
    const bool instruction_has_comms = inst.comm_count != 0u;
    const tl::DataflowHandlerArgs &handler_args =
        tl::dataflow_handler_args(arg_base, inst.arg_offset);
{instrumentation_before_recv}

{cluster_source_before_receive}
    if (instruction_has_comms) {{
      tl_dataflow_generated::dataflow_dispatch_cluster_recv_comms(
          inst, comm_plans, slots, shared_base, scratch_base, barriers,
          cluster_ack_barriers);
      if (tl_dataflow_generated::kDataflowHasHBMComms) {{
        tl_dataflow_generated::dataflow_dispatch_hbm_recv_wait_comms(
            inst, comm_plans, slots, shared_base, scratch_base, global_base, flags,
            barriers, hbm_recv_issued);
      }}
    }}
{instrumentation_after_recv}

    const bool handler_completion_synchronized =
        tl_dataflow_generated::dataflow_dispatch_handler(
        inst, handler_args, arg_base, slots, tensor_args, input_slots, task_coords, shared_base, global_base{tma_dispatch_args});
    // The joint schedule attaches a nonblocking receive issue to a handler
    // whose scratch lifetime is compatible with the transient destination.
    // Issue after that handler so producer readiness advances concurrently
    // with useful compute and a prediction miss falls back at the target.
    if (instruction_has_comms &&
        tl_dataflow_generated::kDataflowHasHBMComms) {{
      tl_dataflow_generated::dataflow_dispatch_hbm_recv_issue_comms(
          inst, comm_plans, slots, shared_base, scratch_base, global_base, flags,
          barriers, cluster_ack_barriers, hbm_recv_issued);
    }}
{instrumentation_after_handler}

{cluster_source_before_release}
    if (instruction_has_comms) {{
{cluster_source_dispatch}
    }}
{instrumentation_after_send}
  }}
{cluster_source_epilogue}
{instrumentation_kernel_epilogue}
}}
"""
    if spec.handler_codegen_module is not None:
        return spec.handler_codegen_module.compose(source, kernel_name=spec.kernel_name)
    return source


def instrumentation_hook(
    instrumentation: DataflowWrapperInstrumentation | None,
    hook_name: str,
    *,
    indent: int,
) -> str:
    if instrumentation is None:
        return ""
    source = getattr(instrumentation, hook_name).strip("\n")
    if not source:
        return ""
    prefix = " " * indent
    return "\n".join(prefix + line if line else "" for line in source.splitlines())


def format_launch_bounds(threads: int) -> str:
    if threads <= 0:
        return ""
    return f"__launch_bounds__({int(threads)}, 1) "


def format_startup_collective(
    spec: DataflowWrapperSpec,
    instrumentation: DataflowWrapperInstrumentation | None,
) -> str:
    if spec.has_cluster_send_comms:
        if spec.cluster_size > 1:
            return "  tl::cluster_sync();"
        return "  __syncthreads();"
    if instrumentation is not None:
        return "  __syncthreads();"
    return ""


def format_launch_pacing(spec: DataflowWrapperSpec) -> str:
    pacing = spec.launch_pacing
    if not pacing.enabled:
        return ""
    if spec.cluster_size <= 0:
        raise ValueError("Dataflow wrapper launch pacing requires a positive cluster size")
    return f"""  if (threadIdx.x == 0) {{
    unsigned long long dataflow_delay_start;
    unsigned long long dataflow_delay_now;
    const unsigned long long dataflow_launch_delay_ns =
        {pacing.delay_ns}ull +
        ((blockIdx.x / {spec.cluster_size}u) /
         {pacing.cluster_stagger_group_size}u) *
        {pacing.cluster_stagger_ns}ull;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(dataflow_delay_start));
    do {{
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(dataflow_delay_now));
    }} while (dataflow_delay_now - dataflow_delay_start <
             dataflow_launch_delay_ns);
  }}
  __syncthreads();"""


def format_handler_stub(
    handler: DataflowHandlerSpec,
    *,
    body_source: str,
    tma_descriptors: tuple[DataflowTMADescriptorSpec, ...] = (),
) -> str:
    tma_params = "".join(f",\n    const CUtensorMap &{descriptor.name}" for descriptor in tma_descriptors)
    reduce_arity = (
        None
        if handler.handler_variant_key is None or handler.handler_variant_key.reduce_arity is None
        else handler.handler_variant_key.reduce_arity.arity_class
    )
    device_qualifier = "TL_DEVICE_NOINLINE" if handler.operator_kind == "reduce" and reduce_arity == "generic" else "TL_DEVICE"
    return f"""{device_qualifier} void {handler.symbol_name}(
    const tl::DataflowInstruction &inst,
    const tl::DataflowHandlerArgs &handler_args,
    const void *arg_base,
    const tl::DataflowSlot *slots,
    const tl::DataflowTensorArg *tensor_args,
    const uint32_t *input_slots,
    const uint32_t *task_coords,
    void *shared_base,
    void *global_base{tma_params}) {{
{body_source}
}}"""


def format_handler_case(
    handler: DataflowHandlerSpec,
    *,
    completion_synchronized: bool = False,
    tma_descriptors: tuple[DataflowTMADescriptorSpec, ...] = (),
) -> str:
    tma_args = "".join(f", {descriptor.name}" for descriptor in tma_descriptors)
    completion = "true" if completion_synchronized else "false"
    return f"""  case {handler.handler_id}u:
    {handler.symbol_name}(inst, handler_args, arg_base, slots, tensor_args, input_slots, task_coords, shared_base, global_base{tma_args});
    return {completion};"""
