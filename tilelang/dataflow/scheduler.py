"""Queue, slot, communication, and barrier planning for Dataflow programs.

The scheduler produces the structured instruction plan consumed by runtime
packing and the production PrimFunc handler compilation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import itertools
import math
import operator
from typing import Any
from collections.abc import Mapping, Sequence

from tilelang.utils.target_capabilities import TargetCapabilitySnapshot

from .abi_schema import DATAFLOW_SLOT_ALIGNMENT
from .handler_identity import (
    REDUCE_ARITY_PASSTHROUGH,
    DataflowHandlerIdentity,
    DataflowHandlerVariantKey,
    ReduceAritySpecialization,
    build_handler_registry,
)
from .handoff_planning import (
    DataflowCrossHandlerHandoffPlan,
    DataflowHandoffQueueBinding,
    fallback_cross_handler_handoff_plan,
    plan_cross_handler_handoff,
)
from .iter_range_buckets import (
    iter_range_specialization_for_length,
    normalize_iter_range_bucket_size,
    normalize_iter_range_buckets,
    normalize_iter_range_exact_lengths,
)
from .ir import IntermediateType, DataflowReducerContract
from .joint_schedule import (
    DataflowJointExecutionPlan,
    DataflowJointScheduleNode,
    DataflowTransportKind,
    schedule_joint_compute_communication,
)
from .implementation_registry import dataflow_implementation_registry
from .operation_contracts import (
    DATAFLOW_HANDOFF_CONTRACT_ATTR,
    DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
    DATAFLOW_PIPELINE_CONTRACT_ATTR,
    DATAFLOW_RANGE_CONTRACT_ATTR,
    DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION,
    DATAFLOW_TRANSPORT_ALL_GATHER,
    DATAFLOW_TRANSPORT_CONTRACT_ATTR,
    DATAFLOW_TRANSPORT_HBM,
    DATAFLOW_TRANSPORT_STREAMED,
    DataflowCrossHandlerHandoffRequest,
    DataflowPipelineRequest,
    DataflowRangeCoarseningRequest,
    DataflowResharedTransportRequest,
)
from .pipeline_planning import plan_pipeline_dataflow
from .physical_contract import (
    DATAFLOW_OUTPUT_SLOT_NONE,
    operator_physical_contract,
)
from .program import DataflowProgram, DataflowStage, DataflowStageKind
from .range_coarsening import (
    DataflowRangeCoarseningPlan,
    estimate_range_output_bytes_per_tile,
    plan_range_coarsening,
)
from .reshared_transport import (
    DataflowResharedTransportPlan,
    plan_reshared_transport,
)
from .scheduler_config import (
    DataflowSchedulerConfig,
    activate_scheduler_config,
    current_scheduler_config,
)
from .scheduler_cost_model import (
    estimate_communication_cost_us,
    estimate_finalize_cost_us,
    estimate_iter_cost_us,
    estimate_reduce_cost_us,
)
from .scheduler_policies import (
    validate_scheduler_policy as _validate_scheduler_policy,
)
from .topology import GPUTopology


DATAFLOW_HBM_BULK_COPY_MAX_BYTES = 16 * 1024


class DataflowOpcode(str, Enum):
    EXIT = "exit"
    CLUSTER_SYNC = "cluster_sync"
    ITER = "iter"
    MAP = "map"
    RESHARED = "reshared"
    REDUCE = "reduce"
    REDUCE_UPDATE = "reduce_update"
    FINALIZE = "finalize"


class DataflowCommKind(str, Enum):
    NONE = "none"
    CLUSTER_SEND = "cluster_send"
    CLUSTER_RECV = "cluster_recv"
    CLUSTER_RELEASE = "cluster_release"
    HBM_SEND = "hbm_send"
    HBM_RECV = "hbm_recv"
    HBM_RECV_ISSUE = "hbm_recv_issue"
    HBM_RECV_WAIT = "hbm_recv_wait"


class DataflowValueForwardStoragePolicy(str, Enum):
    """Ownership rule for forwarding one logical intermediate value."""

    COPY_IF_DISTINCT = "copy_if_distinct"


@dataclass(frozen=True)
class DataflowValueForward:
    """Semantic passthrough with a copy fallback and non-owning alias fast path."""

    source_slot_id: int
    output_slot_id: int
    copy_owner_slot_id: int
    alias_owner_slot_id: int
    storage_policy: DataflowValueForwardStoragePolicy = DataflowValueForwardStoragePolicy.COPY_IF_DISTINCT

    def __post_init__(self) -> None:
        for name in (
            "source_slot_id",
            "output_slot_id",
            "copy_owner_slot_id",
            "alias_owner_slot_id",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Dataflow ValueForward {name} must be non-negative, got {value!r}")
        if self.source_slot_id == self.output_slot_id:
            raise ValueError("Dataflow ValueForward source and output slots must be distinct")
        if self.copy_owner_slot_id != self.output_slot_id:
            raise ValueError("Dataflow ValueForward copy must be owned by its output slot")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_slot_id": self.source_slot_id,
            "output_slot_id": self.output_slot_id,
            "storage_policy": self.storage_policy.value,
            "copy_owner_slot_id": self.copy_owner_slot_id,
            "alias_owner_slot_id": self.alias_owner_slot_id,
        }


@dataclass(frozen=True)
class TaskRange:
    axis: Any
    begin: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class SlotPlan:
    slot_id: int
    task_id: int
    intermediate_type: IntermediateType
    role: str
    producer_instruction_id: int | None = None
    shared_storage_id: int | None = None
    global_storage_id: int | None = None
    barrier_storage_id: int | None = None
    scratch_backed: bool = False
    scratch_offset: int | None = None
    allocation_owner_slot_id: int | None = None
    alias_of_slot_id: int | None = None
    physical_owner_cta: int | None = None
    cluster_gated_push: bool = False


@dataclass(frozen=True)
class CommPlan:
    source_instruction_id: int
    target_instruction_id: int
    source_slot_id: int
    target_slot_id: int
    producer_sm: int
    consumer_sm: int
    kind: DataflowCommKind
    dispatch_instruction_id: int | None = None
    peer_cta_rank: int | None = None
    flag_epoch: int = 1
    barrier_phase: int = 0
    byte_offset: int = 0
    byte_count: int = 0
    segment_id: int = 0
    segment_count: int = 1
    cluster_gate_id: int | None = None

    def __post_init__(self) -> None:
        for name in ("byte_offset", "byte_count", "segment_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Dataflow CommPlan {name} must be non-negative")
        if isinstance(self.segment_count, bool) or not isinstance(self.segment_count, int) or self.segment_count <= 0:
            raise ValueError("Dataflow CommPlan segment_count must be positive")
        if self.segment_id >= self.segment_count:
            raise ValueError("Dataflow CommPlan segment_id must be smaller than segment_count")
        if self.cluster_gate_id is not None and (
            isinstance(self.cluster_gate_id, bool) or not isinstance(self.cluster_gate_id, int) or self.cluster_gate_id < 0
        ):
            raise ValueError("Dataflow CommPlan cluster_gate_id must be non-negative")

    @property
    def resolved_dispatch_instruction_id(self) -> int:
        if self.dispatch_instruction_id is not None:
            return self.dispatch_instruction_id
        if self.kind in (
            DataflowCommKind.CLUSTER_RECV,
            DataflowCommKind.HBM_RECV,
            DataflowCommKind.HBM_RECV_ISSUE,
            DataflowCommKind.HBM_RECV_WAIT,
        ):
            return self.target_instruction_id
        return self.source_instruction_id

    @property
    def dispatch_phase(self) -> str:
        if self.kind in (
            DataflowCommKind.CLUSTER_RECV,
            DataflowCommKind.HBM_RECV,
            DataflowCommKind.HBM_RECV_ISSUE,
            DataflowCommKind.HBM_RECV_WAIT,
        ):
            return "pre"
        return "post"


@dataclass(frozen=True)
class Instruction:
    instruction_id: int
    opcode: DataflowOpcode
    operator_name: str
    task_id: int | None
    task_coords: tuple[int, ...] = ()
    sm_id: int | None = None
    task_range: TaskRange | None = None
    input_slots: tuple[int, ...] = ()
    output_slot: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    handler_identity: DataflowHandlerIdentity | None = None
    handler_variant_key: DataflowHandlerVariantKey | None = None
    value_forward: DataflowValueForward | None = None

    def __post_init__(self) -> None:
        if (
            self.handler_identity is not None
            and self.handler_variant_key is not None
            and self.handler_variant_key.base_identity != self.handler_identity
        ):
            raise ValueError(
                f"Dataflow instruction handler identity and variant base identity do not match for instruction {self.instruction_id}"
            )
        if self.value_forward is not None:
            if self.input_slots != (self.value_forward.source_slot_id,):
                raise ValueError(f"Dataflow instruction {self.instruction_id} ValueForward source does not match its single input slot")
            if self.output_slot != self.value_forward.output_slot_id:
                raise ValueError(f"Dataflow instruction {self.instruction_id} ValueForward output does not match its output slot")


@dataclass(frozen=True)
class InstructionPlan:
    topology: GPUTopology
    block_size: int
    range_axis: Any
    scheduler_policy: str
    reduce_strategy: str
    task_extents: tuple[int, ...] | None
    task_range_lengths: tuple[int, ...]
    instructions: tuple[Instruction, ...]
    queues: dict[int, tuple[Instruction, ...]]
    slots: tuple[SlotPlan, ...]
    comms: tuple[CommPlan, ...]
    scheduler_config: DataflowSchedulerConfig = field(default_factory=DataflowSchedulerConfig)
    range_coarsening_plans: tuple[tuple[int, DataflowRangeCoarseningPlan], ...] = ()
    reshared_transport_plans: tuple[tuple[int, DataflowResharedTransportPlan], ...] = ()
    cross_handler_handoff_plans: tuple[DataflowCrossHandlerHandoffPlan, ...] = ()
    cross_handler_handoff_bindings: tuple[DataflowHandoffQueueBinding, ...] = ()
    range_resource_budget_bytes: int | None = None
    target_capabilities: TargetCapabilitySnapshot | None = None
    joint_execution_plan: DataflowJointExecutionPlan | None = None

    def queue(self, sm_id: int) -> tuple[Instruction, ...]:
        return self.queues.get(sm_id, ())


_FUSED_REDUCE_FINALIZE_ATTR = "fused_reduce_finalize"


def apply_fused_reduce_finalize_contract(
    program: DataflowProgram,
    plan: InstructionPlan,
) -> InstructionPlan:
    """Replace a terminal reduce plus finalize with a declared multi-input finalizer."""

    finalize_stage = program.finalize_stage
    scheduler_config = current_scheduler_config()
    if finalize_stage is None or not finalize_stage.fused_reduce_calls or not scheduler_config.contains("fused_reduce_finalize_max_arity"):
        return plan
    available_calls_by_arity = {len(call.operator.input_types): call for call in finalize_stage.fused_reduce_calls}
    configured_max_arity = int(scheduler_config.get("fused_reduce_finalize_max_arity"))
    fused_calls_by_arity = {arity: call for arity, call in available_calls_by_arity.items() if arity <= configured_max_arity}
    if not fused_calls_by_arity:
        return plan
    max_fused_input_count = max(fused_calls_by_arity)
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    consumers_by_slot: dict[int, set[int]] = {}
    for instruction in plan.instructions:
        for slot_id in instruction.input_slots:
            consumers_by_slot.setdefault(slot_id, set()).add(instruction.instruction_id)

    replacements: dict[int, Instruction] = {}
    removed_instruction_ids: set[int] = set()
    removed_slot_ids: set[int] = set()
    retarget_instruction_ids: dict[int, int] = {}
    for finalize in plan.instructions:
        if (
            finalize.opcode is not DataflowOpcode.FINALIZE
            or finalize.attrs.get(_FUSED_REDUCE_FINALIZE_ATTR, False)
            or len(finalize.input_slots) != 1
        ):
            continue
        root_slot_id = finalize.input_slots[0]
        root_slot = slots_by_id.get(root_slot_id)
        if root_slot is None or root_slot.producer_instruction_id is None:
            continue
        root = instructions_by_id.get(root_slot.producer_instruction_id)
        if (
            root is None
            or root.opcode not in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE}
            or root.output_slot != root_slot_id
            or root.task_id != finalize.task_id
            or consumers_by_slot.get(root_slot_id, set()) != {finalize.instruction_id}
        ):
            continue
        frontier = [(slot_id, root.instruction_id) for slot_id in root.input_slots]
        expanded_instruction_ids: list[int] = []
        selected_frontier: tuple[int, ...] | None = None
        selected_expanded_instruction_ids: tuple[int, ...] = ()
        if len(frontier) in fused_calls_by_arity:
            selected_frontier = tuple(slot_id for slot_id, _ in frontier)
        while len(frontier) < max_fused_input_count:
            expandable = None
            for index, (slot_id, parent_instruction_id) in enumerate(frontier):
                slot = slots_by_id.get(slot_id)
                producer = (
                    None if slot is None or slot.producer_instruction_id is None else instructions_by_id.get(slot.producer_instruction_id)
                )
                if (
                    producer is None
                    or producer.opcode not in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE}
                    or producer.sm_id != root.sm_id
                    or producer.task_id != root.task_id
                    or producer.output_slot != slot_id
                    or len(producer.input_slots) < 2
                    or consumers_by_slot.get(slot_id, set()) != {parent_instruction_id}
                    or len(frontier) - 1 + len(producer.input_slots) > max_fused_input_count
                ):
                    continue
                expandable = (index, producer)
                break
            if expandable is None:
                break
            index, producer = expandable
            frontier[index : index + 1] = [(slot_id, producer.instruction_id) for slot_id in producer.input_slots]
            expanded_instruction_ids.append(producer.instruction_id)
            if len(frontier) in fused_calls_by_arity:
                selected_frontier = tuple(slot_id for slot_id, _ in frontier)
                selected_expanded_instruction_ids = tuple(expanded_instruction_ids)
        if selected_frontier is None:
            continue
        fused_call = fused_calls_by_arity[len(selected_frontier)]
        replacements[root.instruction_id] = replace(
            root,
            opcode=DataflowOpcode.FINALIZE,
            operator_name=fused_call.name,
            task_coords=finalize.task_coords,
            input_slots=selected_frontier,
            output_slot=None,
            attrs={
                **root.attrs,
                _FUSED_REDUCE_FINALIZE_ATTR: True,
                "fused_reduce_finalize_arity": len(selected_frontier),
                "replaced_finalize_instruction_id": finalize.instruction_id,
                "absorbed_reduce_instruction_ids": (selected_expanded_instruction_ids),
            },
            handler_identity=None,
            handler_variant_key=None,
            value_forward=None,
        )
        removed_instruction_ids.add(finalize.instruction_id)
        removed_instruction_ids.update(selected_expanded_instruction_ids)
        removed_slot_ids.add(root_slot_id)
        for instruction_id in selected_expanded_instruction_ids:
            removed_instruction = instructions_by_id[instruction_id]
            assert removed_instruction.output_slot is not None
            removed_slot_ids.add(removed_instruction.output_slot)
            retarget_instruction_ids[instruction_id] = root.instruction_id

    if not replacements:
        return plan
    updated_by_id = {
        instruction_id: replacements.get(instruction_id, instruction)
        for instruction_id, instruction in instructions_by_id.items()
        if instruction_id not in removed_instruction_ids
    }
    remaining_slots = tuple(slot for slot in plan.slots if slot.slot_id not in removed_slot_ids)
    compact_slot_id = {slot.slot_id: index for index, slot in enumerate(remaining_slots)}

    def map_optional_slot(slot_id: int | None) -> int | None:
        if slot_id is None:
            return None
        return compact_slot_id[slot_id]

    def compact_instruction(instruction: Instruction) -> Instruction:
        value_forward = instruction.value_forward
        if value_forward is not None:
            value_forward = replace(
                value_forward,
                source_slot_id=compact_slot_id[value_forward.source_slot_id],
                output_slot_id=compact_slot_id[value_forward.output_slot_id],
                copy_owner_slot_id=compact_slot_id[value_forward.copy_owner_slot_id],
                alias_owner_slot_id=compact_slot_id[value_forward.alias_owner_slot_id],
            )
        return replace(
            instruction,
            input_slots=tuple(compact_slot_id[slot_id] for slot_id in instruction.input_slots),
            output_slot=map_optional_slot(instruction.output_slot),
            value_forward=value_forward,
        )

    updated_by_id = {instruction_id: compact_instruction(instruction) for instruction_id, instruction in updated_by_id.items()}
    compacted_slots = tuple(
        replace(
            slot,
            slot_id=compact_slot_id[slot.slot_id],
            allocation_owner_slot_id=map_optional_slot(slot.allocation_owner_slot_id),
            alias_of_slot_id=map_optional_slot(slot.alias_of_slot_id),
        )
        for slot in remaining_slots
    )
    compacted_comms_list: list[CommPlan] = []
    for comm in plan.comms:
        if (
            comm.source_slot_id in removed_slot_ids
            or comm.target_slot_id in removed_slot_ids
            or comm.source_instruction_id in removed_instruction_ids
        ):
            continue
        target_instruction_id = retarget_instruction_ids.get(
            comm.target_instruction_id,
            comm.target_instruction_id,
        )
        if target_instruction_id in removed_instruction_ids:
            continue
        dispatch_instruction_id = comm.dispatch_instruction_id
        if dispatch_instruction_id is not None:
            dispatch_instruction_id = retarget_instruction_ids.get(
                dispatch_instruction_id,
                dispatch_instruction_id,
            )
            if dispatch_instruction_id in removed_instruction_ids:
                continue
        compacted_comms_list.append(
            replace(
                comm,
                target_instruction_id=target_instruction_id,
                dispatch_instruction_id=dispatch_instruction_id,
                source_slot_id=compact_slot_id[comm.source_slot_id],
                target_slot_id=compact_slot_id[comm.target_slot_id],
            )
        )
    compacted_comms = tuple(compacted_comms_list)
    return replace(
        plan,
        instructions=tuple(
            updated_by_id[instruction.instruction_id] for instruction in plan.instructions if instruction.instruction_id in updated_by_id
        ),
        queues={
            sm_id: tuple(updated_by_id[instruction.instruction_id] for instruction in queue if instruction.instruction_id in updated_by_id)
            for sm_id, queue in plan.queues.items()
        },
        slots=compacted_slots,
        comms=compacted_comms,
        joint_execution_plan=None,
    )


def attach_handler_identities(
    program: DataflowProgram,
    plan: InstructionPlan,
) -> InstructionPlan:
    """Attach canonical program identities without consulting diagnostic names."""

    plan = apply_fused_reduce_finalize_contract(program, plan)
    registry = build_handler_registry(program)

    def resolve_identity(instruction: Instruction) -> DataflowHandlerIdentity | None:
        if instruction.handler_variant_key is not None:
            registry.resolve_variant(instruction.handler_variant_key)
            if instruction.handler_identity is not None and instruction.handler_identity != instruction.handler_variant_key.base_identity:
                raise ValueError(
                    f"Dataflow instruction handler variant base does not match its identity: instruction_id={instruction.instruction_id}"
                )
            return instruction.handler_variant_key.base_identity
        if instruction.handler_identity is not None:
            registry.resolve(instruction.handler_identity)
            return instruction.handler_identity
        if (
            instruction.opcode is DataflowOpcode.FINALIZE
            and instruction.attrs.get(_FUSED_REDUCE_FINALIZE_ATTR, False)
            and program.finalize_stage is not None
        ):
            call = program.finalize_stage.fused_reduce_call_for_arity(len(instruction.input_slots))
            if call is None:
                raise ValueError(
                    f"Dataflow fused reduce-finalize instruction has no declared handler for arity {len(instruction.input_slots)}"
                )
            return registry.identity_for_call(call, "finalize")
        stage_id = instruction.attrs.get("stage_id")
        if isinstance(stage_id, int) and instruction.opcode is not DataflowOpcode.RESHARED:
            return registry.identity_for_stage_id(stage_id)
        if instruction.opcode is DataflowOpcode.ITER and program.partial_stage is not None:
            return registry.identity_for_call(program.partial_stage.iter_call, "iter")
        if instruction.opcode in (DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE) and program.reduce_stage is not None:
            return registry.identity_for_call(program.reduce_stage.reduce_call, "reduce")
        if instruction.opcode is DataflowOpcode.FINALIZE and program.finalize_stage is not None:
            return registry.identity_for_call(
                program.finalize_stage.finalize_call,
                "finalize",
            )
        return None

    def resolve_variant(
        instruction: Instruction,
        identity: DataflowHandlerIdentity | None,
    ) -> DataflowHandlerVariantKey | None:
        if identity is None:
            if instruction.handler_variant_key is not None:
                raise ValueError(
                    f"Dataflow instruction without a handler cannot carry a variant key: instruction_id={instruction.instruction_id}"
                )
            return None
        variant_key = instruction.handler_variant_key
        if variant_key is None:
            variant_key = DataflowHandlerVariantKey.default_for_identity(
                identity,
                reduce_arity=(len(instruction.input_slots) if identity.operator_kind == "reduce" else None),
            )
        if variant_key.base_identity != identity:
            raise ValueError(
                "Dataflow handler variant base identity mismatch: "
                f"instruction_id={instruction.instruction_id}, "
                f"identity={identity.to_dict()!r}, "
                f"variant={variant_key.to_dict()!r}"
            )
        if identity.operator_kind == "reduce":
            expected = ReduceAritySpecialization.for_arity(len(instruction.input_slots))
            if variant_key.reduce_arity != expected:
                raise ValueError(
                    "Dataflow reduce handler variant does not match instruction arity: "
                    f"instruction_id={instruction.instruction_id}, "
                    f"input_slots={len(instruction.input_slots)}, "
                    f"variant={variant_key.to_dict()!r}"
                )
        registry.resolve_variant(variant_key)
        return variant_key

    instructions = []
    for instruction in plan.instructions:
        identity = resolve_identity(instruction)
        variant_key = resolve_variant(instruction, identity)
        value_forward = instruction.value_forward
        if (
            identity is not None
            and identity.operator_kind == "reduce"
            and variant_key is not None
            and variant_key.reduce_arity is not None
            and variant_key.reduce_arity.arity_class == REDUCE_ARITY_PASSTHROUGH
        ):
            binding = registry.resolve(identity)
            if binding.call.operator.reducer_contract is DataflowReducerContract.ASSOCIATIVE_BINARY:
                if len(instruction.input_slots) != 1 or instruction.output_slot is None:
                    raise ValueError(
                        f"associative passthrough instruction {instruction.instruction_id} requires one input and one output slot"
                    )
                expected_forward = DataflowValueForward(
                    source_slot_id=instruction.input_slots[0],
                    output_slot_id=instruction.output_slot,
                    copy_owner_slot_id=instruction.output_slot,
                    alias_owner_slot_id=instruction.input_slots[0],
                )
                if value_forward is not None and value_forward != expected_forward:
                    raise ValueError(
                        f"associative passthrough instruction {instruction.instruction_id} has inconsistent ValueForward metadata"
                    )
                value_forward = expected_forward
        elif value_forward is not None:
            raise ValueError(
                f"instruction {instruction.instruction_id} carries ValueForward metadata without an associative passthrough variant"
            )
        instructions.append(
            replace(
                instruction,
                handler_identity=identity,
                handler_variant_key=variant_key,
                value_forward=value_forward,
            )
        )
    instructions = tuple(instructions)
    by_id = {instruction.instruction_id: instruction for instruction in instructions}
    queues = {sm_id: tuple(by_id[instruction.instruction_id] for instruction in queue) for sm_id, queue in plan.queues.items()}
    return plan_value_forward_storage(replace(plan, instructions=instructions, queues=queues))


def reorder_one_joint_jit_push_queue(
    plan: InstructionPlan,
) -> InstructionPlan | None:
    """Move storage-closed work ahead of an intentionally delayed push.

    The joint scheduler represents a cluster push as a first-class operation,
    so it may keep a produced value live while the producer CTA executes
    independent handlers and only issue the copy when the destination inbox is
    available.  Scratch-backed lowering cannot realize that order when the
    source outbox aliases the next handler output: the compiler must attach the
    send to the producer boundary and the runtime blocks on the destination
    gate instead.

    If the intervening queue prefix is dependency-independent and consumes all
    values it creates, executing that closed prefix *before* the producer keeps
    the modeled push time unchanged while making the source lifetime physical.
    This is a queue/lifetime transformation; it does not depend on an operator,
    workload, shape, or measured trace.
    """

    joint_plan = plan.joint_execution_plan
    if joint_plan is None:
        return None
    schedule = joint_plan.require_valid(topology=plan.topology).schedule
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    producer_by_value = {
        instruction.output_slot: instruction.instruction_id for instruction in plan.instructions if instruction.output_slot is not None
    }
    consumers_by_value: dict[int, set[int]] = {}
    for instruction in plan.instructions:
        for value_id in instruction.input_slots:
            consumers_by_value.setdefault(value_id, set()).add(instruction.instruction_id)
    handler_event_by_node = {event.node_id: event for event in schedule.events if event.node_id is not None and event.cta_id is not None}

    candidates = []
    for transfer in schedule.transfers:
        if transfer.kind is not DataflowTransportKind.CLUSTER_PUSH:
            continue
        producer_event = handler_event_by_node.get(transfer.producer_node_id)
        if producer_event is None:
            continue
        issue_event = schedule.event(transfer.producer_issue_event_id)
        delayed_us = issue_event.start_us - producer_event.end_us
        if delayed_us <= 1e-9:
            continue
        candidates.append(
            (
                -delayed_us,
                transfer.producer_cta,
                transfer.producer_node_id,
                issue_event.start_us,
            )
        )

    for _, cta_id, producer_node_id, issue_start_us in sorted(candidates):
        queue = list(plan.queue(cta_id))
        queue_position = {instruction.instruction_id: index for index, instruction in enumerate(queue)}
        producer_index = queue_position.get(producer_node_id)
        if producer_index is None:
            continue

        movable_limit = producer_index
        for index in range(producer_index + 1, len(queue)):
            instruction = queue[index]
            event = handler_event_by_node.get(instruction.instruction_id)
            if event is None or event.end_us > issue_start_us + 1e-9:
                break
            movable_limit = index
        if movable_limit == producer_index:
            continue

        best_end = None
        for end_index in range(producer_index + 1, movable_limit + 1):
            segment = queue[producer_index + 1 : end_index + 1]
            segment_ids = {instruction.instruction_id for instruction in segment}
            movable = True
            for instruction in segment:
                for value_id in instruction.input_slots:
                    dependency_id = producer_by_value.get(value_id)
                    if dependency_id is None or dependency_id in segment_ids:
                        continue
                    dependency = instructions_by_id[dependency_id]
                    dependency_position = queue_position.get(dependency_id)
                    if (
                        dependency.sm_id != cta_id
                        or dependency_id == producer_node_id
                        or dependency_position is None
                        or dependency_position > producer_index
                    ):
                        movable = False
                        break
                if not movable:
                    break
                if instruction.output_slot is None:
                    continue
                if not consumers_by_value.get(instruction.output_slot, set()).issubset(segment_ids):
                    movable = False
                    break
            if movable:
                best_end = end_index

        if best_end is None:
            continue
        closed_prefix = queue[producer_index + 1 : best_end + 1]
        reordered = queue[:producer_index] + closed_prefix + [queue[producer_index]] + queue[best_end + 1 :]
        queues = dict(plan.queues)
        queues[cta_id] = tuple(reordered)
        return replace(
            plan,
            queues=queues,
            joint_execution_plan=None,
        )
    return None


def reorder_one_joint_scratch_release_queue(
    plan: InstructionPlan,
) -> InstructionPlan | None:
    """Cover a scratch-backed destination gate with ready source work.

    A communicate slot has its own logical lifetime, but compact permanent and
    transient physical placement may reuse scratch owned by non-reducer
    handlers.  When a producer reaches its push before that destination handler
    completes, a runtime acknowledgement preserves correctness but creates
    head-of-line waiting.  Move a dependency-closed source-queue prefix before
    the producer when its modeled duration fits inside that release gap.

    This transformation is driven only by queue dependencies, modeled handler
    durations, transport storage kind, and the physical overlap contract.  It
    never assumes a task id, shape, model, or measured trace.
    """

    joint_plan = plan.joint_execution_plan
    if joint_plan is None:
        return None
    schedule = joint_plan.require_valid(topology=plan.topology).schedule
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    producer_by_value = {
        instruction.output_slot: instruction.instruction_id for instruction in plan.instructions if instruction.output_slot is not None
    }
    consumers_by_value: dict[int, set[int]] = {}
    for instruction in plan.instructions:
        for value_id in instruction.input_slots:
            consumers_by_value.setdefault(value_id, set()).add(instruction.instruction_id)
    handler_event_by_node = {event.node_id: event for event in schedule.events if event.node_id is not None and event.cta_id is not None}
    queue_position_by_node = {
        instruction.instruction_id: (cta_id, position)
        for cta_id, queue in plan.queues.items()
        for position, instruction in enumerate(queue)
    }

    candidates: list[tuple[float, int, int]] = []
    for transfer in schedule.transfers:
        if transfer.kind is not DataflowTransportKind.CLUSTER_PUSH:
            continue
        producer_event = handler_event_by_node.get(transfer.producer_node_id)
        target_position = queue_position_by_node.get(transfer.consumer_node_id)
        if producer_event is None or target_position is None:
            continue
        consumer_cta, target_index = target_position
        issue_event = schedule.event(transfer.producer_issue_event_id)
        scratch_release_us = max(
            (
                handler_event.end_us
                for instruction in plan.queue(consumer_cta)[:target_index]
                if instruction.opcode not in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE}
                and (handler_event := handler_event_by_node.get(instruction.instruction_id)) is not None
            ),
            default=0.0,
        )
        blocked_us = scratch_release_us - issue_event.start_us
        if blocked_us > 1e-9:
            candidates.append((-blocked_us, transfer.producer_cta, transfer.producer_node_id))

    for negative_gap_us, cta_id, producer_node_id in sorted(candidates):
        gap_us = -negative_gap_us
        queue = list(plan.queue(cta_id))
        queue_position = {instruction.instruction_id: index for index, instruction in enumerate(queue)}
        producer_index = queue_position.get(producer_node_id)
        if producer_index is None:
            continue

        best_end = None
        accumulated_duration_us = 0.0
        for end_index in range(producer_index + 1, len(queue)):
            instruction = queue[end_index]
            if instruction.opcode in {DataflowOpcode.EXIT, DataflowOpcode.CLUSTER_SYNC}:
                break
            event = handler_event_by_node.get(instruction.instruction_id)
            if event is None:
                break
            accumulated_duration_us += event.end_us - event.start_us
            if accumulated_duration_us > gap_us + 1e-9:
                break

            segment = queue[producer_index + 1 : end_index + 1]
            segment_ids = {item.instruction_id for item in segment}
            movable = True
            for item in segment:
                for value_id in item.input_slots:
                    dependency_id = producer_by_value.get(value_id)
                    if dependency_id is None or dependency_id in segment_ids:
                        continue
                    dependency = instructions_by_id[dependency_id]
                    dependency_position = queue_position.get(dependency_id)
                    if (
                        dependency.sm_id != cta_id
                        or dependency_id == producer_node_id
                        or dependency_position is None
                        or dependency_position > producer_index
                    ):
                        movable = False
                        break
                if not movable:
                    break
                if item.output_slot is not None and not consumers_by_value.get(
                    item.output_slot,
                    set(),
                ).issubset(segment_ids):
                    movable = False
                    break
            if movable:
                best_end = end_index

        if best_end is None:
            continue
        closed_prefix = queue[producer_index + 1 : best_end + 1]
        reordered = queue[:producer_index] + closed_prefix + [queue[producer_index]] + queue[best_end + 1 :]
        queues = dict(plan.queues)
        queues[cta_id] = tuple(reordered)
        return replace(plan, queues=queues, joint_execution_plan=None)
    return None


def attach_joint_execution_plan(
    program: DataflowProgram,
    plan: InstructionPlan,
    *,
    _apply_jit_push_reorder: bool = True,
) -> InstructionPlan:
    if not config_flag("joint_schedule"):
        return plan

    executable_instructions = tuple(
        instruction for instruction in plan.instructions if instruction.opcode not in {DataflowOpcode.EXIT, DataflowOpcode.CLUSTER_SYNC}
    )
    producer_by_value = {
        instruction.output_slot: instruction for instruction in executable_instructions if instruction.output_slot is not None
    }
    outgoing_instruction_ids = {
        comm.source_instruction_id for comm in plan.comms if comm.kind in {DataflowCommKind.CLUSTER_SEND, DataflowCommKind.HBM_SEND}
    }
    reduce_physical = None if program.reduce_stage is None else operator_physical_contract(program.reduce_stage.reduce_call.operator.attrs)
    async_receive_pipeline = config_flag("joint_hbm_async_receive_pipeline") or config_flag("joint_cluster_async_receive_pipeline")
    transient_prefetch_lane_count = max(
        1,
        max(2, config_int("ordered_tree_max_reduce_arity", 2)) - 1,
    )
    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    value_nbytes = {
        value_id: byte_count
        for value_id in producer_by_value
        if (slot := slots_by_id.get(value_id)) is not None
        and (
            byte_count := estimate_range_output_bytes_per_tile(
                slot.intermediate_type,
                output_tile_arity=1,
            )
        )
        > 0
    }

    nodes: list[DataflowJointScheduleNode] = []
    for instruction in executable_instructions:
        if instruction.sm_id is None:
            raise ValueError("Dataflow joint scheduling requires fixed CTA ownership for every handler")
        if instruction.opcode is DataflowOpcode.ITER:
            if instruction.task_range is None:
                raise ValueError(f"Dataflow ITER instruction {instruction.instruction_id} has no range")
            duration_us = estimated_streaming_iter_cost_us(
                instruction.task_range,
                plan.block_size,
            )
        elif instruction.opcode in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE}:
            duration_us = estimated_streaming_reduce_cost_us(len(instruction.input_slots))
        elif instruction.opcode is DataflowOpcode.FINALIZE:
            duration_us = estimated_streaming_finalize_cost_us()
        else:
            duration_us = current_scheduler_config().cost_model.iter_base_us

        remote_input_count = sum(
            1
            for value_id in instruction.input_slots
            if ((producer := producer_by_value.get(value_id)) is not None and producer.sm_id != instruction.sm_id)
        )
        alias_input_index = None
        if (
            reduce_physical is not None
            and instruction.instruction_id in outgoing_instruction_ids
            and instruction.opcode in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE}
        ):
            remote_input_indices = {
                input_index
                for input_index, value_id in enumerate(instruction.input_slots)
                if ((producer := producer_by_value.get(value_id)) is not None and producer.sm_id != instruction.sm_id)
            }
            local_input_indices = set(range(len(instruction.input_slots))) - (remote_input_indices)
            preferred_groups = (
                (local_input_indices, remote_input_indices) if async_receive_pipeline else (remote_input_indices, local_input_indices)
            )
            alias_input_index = next(
                (
                    input_index
                    for candidate_group in preferred_groups
                    for input_index in reduce_physical.output_alias_input_indices
                    if input_index in candidate_group
                ),
                None,
            )
        transient_prefetch_bytes = 0
        if async_receive_pipeline and (
            instruction.opcode in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE}
            or (instruction.opcode is DataflowOpcode.FINALIZE and remote_input_count > 1)
        ):
            transient_prefetch_bytes = max(
                (value_nbytes.get(value_id, 0) for value_id in instruction.input_slots),
                default=0,
            ) * max(
                transient_prefetch_lane_count,
                max(0, remote_input_count - 1),
            )
        nodes.append(
            DataflowJointScheduleNode(
                node_id=instruction.instruction_id,
                cta_id=instruction.sm_id,
                duration_us=duration_us,
                input_values=instruction.input_slots,
                output_value=instruction.output_slot,
                output_alias_input_index=alias_input_index,
                transient_prefetch_bytes=transient_prefetch_bytes,
                sort_key=(
                    -1 if instruction.task_id is None else instruction.task_id,
                    instruction.instruction_id,
                ),
            )
        )

    cross_cluster_value_bytes = tuple(
        value_nbytes.get(comm.source_slot_id, 0) for comm in plan.comms if comm.kind is DataflowCommKind.HBM_SEND
    )
    max_cross_cluster_bytes = max(cross_cluster_value_bytes, default=0)
    hbm_segment_bytes = None
    if config_flag("joint_hbm_segmented_pipeline") and max_cross_cluster_bytes > DATAFLOW_SLOT_ALIGNMENT:
        half_bytes = math.ceil(max_cross_cluster_bytes / 2)
        hbm_segment_bytes = min(
            DATAFLOW_HBM_BULK_COPY_MAX_BYTES,
            math.ceil(half_bytes / DATAFLOW_SLOT_ALIGNMENT) * DATAFLOW_SLOT_ALIGNMENT,
        )
    fixed_queues = {
        cta_id: tuple(
            instruction.instruction_id
            for instruction in plan.queue(cta_id)
            if instruction.opcode not in {DataflowOpcode.EXIT, DataflowOpcode.CLUSTER_SYNC}
        )
        for cta_id in range(plan.topology.sm_count)
    }
    model = current_scheduler_config().cost_model
    execution_schedule = schedule_joint_compute_communication(
        plan.topology,
        nodes,
        cluster_transfer_us=model.cluster_comm_us,
        hbm_transfer_us=model.hbm_comm_us,
        transfer_issue_us=model.transfer_issue_us,
        retained_copy_us=model.retained_copy_us,
        value_nbytes=value_nbytes,
        hbm_segment_bytes=hbm_segment_bytes,
        fixed_queue_node_ids=fixed_queues,
        cluster_async_receive_pipeline=config_flag("joint_cluster_async_receive_pipeline"),
        critical_hbm_permanent_prefetch=config_flag("joint_critical_hbm_permanent_prefetch"),
        serialize_permanent_inbox_reuse=config_flag("joint_serialize_permanent_inbox_reuse"),
        hbm_spill_blocked_cluster_push=config_flag("joint_hbm_spill_blocked_cluster_push"),
    )
    joint_plan = DataflowJointExecutionPlan(
        nodes=tuple(nodes),
        schedule=execution_schedule,
        hbm_segment_bytes=hbm_segment_bytes,
    ).require_valid(topology=plan.topology)
    result = replace(plan, joint_execution_plan=joint_plan)
    if not _apply_jit_push_reorder:
        return result

    # A delayed push can expose more than one aliased producer in the same
    # queue.  Rebuild after every monotonic reorder so subsequent decisions use
    # the updated fixed-queue timing.  Signatures make this fail-safe even if a
    # future scheduler adds a transformation with an opposing preference.
    seen_queue_signatures = {
        tuple(tuple(instruction.instruction_id for instruction in result.queue(cta_id)) for cta_id in range(result.topology.sm_count))
    }
    for _ in range(len(executable_instructions)):
        reordered = None
        if config_flag("joint_scratch_release_queue_reorder"):
            reordered = reorder_one_joint_scratch_release_queue(result)
        if reordered is None:
            reordered = reorder_one_joint_jit_push_queue(result)
        if reordered is None:
            break
        signature = tuple(
            tuple(instruction.instruction_id for instruction in reordered.queue(cta_id)) for cta_id in range(reordered.topology.sm_count)
        )
        if signature in seen_queue_signatures:
            break
        seen_queue_signatures.add(signature)
        result = attach_joint_execution_plan(
            program,
            reordered,
            _apply_jit_push_reorder=False,
        )
    return result


def plan_value_forward_storage(plan: InstructionPlan) -> InstructionPlan:
    """Select non-owning aliases only when the source lifetime is unambiguous."""

    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    consumers: dict[int, set[int]] = {slot.slot_id: set() for slot in plan.slots}
    for instruction in plan.instructions:
        for slot_id in instruction.input_slots:
            consumers.setdefault(slot_id, set()).add(instruction.instruction_id)
    communicated_slots = {slot_id for comm in plan.comms for slot_id in (comm.source_slot_id, comm.target_slot_id)}
    shared_storage_members: dict[int, set[int]] = {}
    global_storage_members: dict[int, set[int]] = {}
    for slot in plan.slots:
        if slot.shared_storage_id is not None:
            shared_storage_members.setdefault(slot.shared_storage_id, set()).add(slot.slot_id)
        if slot.global_storage_id is not None:
            global_storage_members.setdefault(slot.global_storage_id, set()).add(slot.slot_id)

    updated_instructions: dict[int, Instruction] = {}
    for instruction in plan.instructions:
        forward = instruction.value_forward
        if forward is None:
            updated_instructions[instruction.instruction_id] = instruction
            continue
        source = slots_by_id[forward.source_slot_id]
        output = slots_by_id[forward.output_slot_id]
        producer = None if source.producer_instruction_id is None else instructions_by_id.get(source.producer_instruction_id)
        alias_slot_ids = {source.slot_id, output.slot_id}
        shared_addresses_match = (
            source.scratch_backed
            and output.scratch_backed
            and source.scratch_offset is not None
            and source.scratch_offset == output.scratch_offset
        ) or (
            not source.scratch_backed
            and not output.scratch_backed
            and source.shared_storage_id is not None
            and source.shared_storage_id == output.shared_storage_id
        )
        global_addresses_match = source.global_storage_id is not None and source.global_storage_id == output.global_storage_id
        source_shared_members = set() if source.shared_storage_id is None else shared_storage_members[source.shared_storage_id]
        source_global_members = set() if source.global_storage_id is None else global_storage_members[source.global_storage_id]
        eligible = (
            producer is not None
            and producer.sm_id == instruction.sm_id
            and consumers.get(source.slot_id, set()) == {instruction.instruction_id}
            and source.slot_id not in communicated_slots
            and source.intermediate_type is output.intermediate_type
            and shared_addresses_match
            and global_addresses_match
            and source_shared_members <= alias_slot_ids
            and source_global_members <= alias_slot_ids
        )
        if not eligible:
            updated_instructions[instruction.instruction_id] = instruction
            continue

        owner_slot_id = source.slot_id if source.allocation_owner_slot_id is None else source.allocation_owner_slot_id
        source = replace(
            source,
            allocation_owner_slot_id=owner_slot_id,
        )
        output = replace(
            output,
            allocation_owner_slot_id=owner_slot_id,
            alias_of_slot_id=source.slot_id,
        )
        slots_by_id[source.slot_id] = source
        slots_by_id[output.slot_id] = output
        updated_instructions[instruction.instruction_id] = replace(
            instruction,
            value_forward=replace(
                forward,
                alias_owner_slot_id=owner_slot_id,
            ),
        )

    instructions = tuple(updated_instructions[instruction.instruction_id] for instruction in plan.instructions)
    queues = {
        sm_id: tuple(updated_instructions[instruction.instruction_id] for instruction in queue) for sm_id, queue in plan.queues.items()
    }
    slots = tuple(slots_by_id[slot.slot_id] for slot in plan.slots)
    return replace(plan, instructions=instructions, queues=queues, slots=slots)


@dataclass(frozen=True)
class StreamingChunk:
    task_id: int
    sm_id: int
    task_range: TaskRange
    part_index: int
    part_count: int


def logical_streaming_chunk_order(
    chunks: Sequence[StreamingChunk],
) -> tuple[StreamingChunk, ...]:
    """Restore the reducer's logical range order after placement searches."""

    return tuple(
        sorted(
            chunks,
            key=lambda chunk: (
                chunk.part_index,
                chunk.task_range.begin,
                chunk.task_range.end,
                chunk.sm_id,
            ),
        )
    )


def balance_streaming_cluster_segment_chunks(
    topology: GPUTopology,
    chunks_by_task: tuple[tuple[StreamingChunk, ...], ...],
    *,
    block_size: int,
    critical_tasks_only: bool = False,
) -> tuple[tuple[StreamingChunk, ...], ...]:
    """Evenly divide each contiguous task/cluster segment across its CTAs."""

    maximum_task_span = max(
        (task_chunks[-1].task_range.end - task_chunks[0].task_range.begin for task_chunks in chunks_by_task if task_chunks),
        default=0,
    )
    balanced_tasks: list[tuple[StreamingChunk, ...]] = []
    for task_chunks in chunks_by_task:
        ordered = list(logical_streaming_chunk_order(task_chunks))
        task_span = 0 if not ordered else ordered[-1].task_range.end - ordered[0].task_range.begin
        if critical_tasks_only and task_span < maximum_task_span:
            balanced_tasks.append(tuple(ordered))
            continue
        runs = ordered_cluster_run_bounds(
            topology,
            [chunk.sm_id for chunk in ordered],
        )
        for _cluster_id, run_begin, run_end in runs:
            run = ordered[run_begin:run_end]
            if len(run) <= 1 or any(left.task_range.end != right.task_range.begin for left, right in zip(run, run[1:])):
                continue
            segment_begin = run[0].task_range.begin
            segment_end = run[-1].task_range.end
            segment_blocks = math.ceil((segment_end - segment_begin) / block_size)
            if segment_blocks < len(run):
                continue
            base_blocks, extra_blocks = divmod(segment_blocks, len(run))
            cursor = segment_begin
            replacements: list[StreamingChunk] = []
            for index, chunk in enumerate(run):
                chunk_blocks = base_blocks + (1 if index < extra_blocks else 0)
                end = min(segment_end, cursor + chunk_blocks * block_size)
                replacements.append(
                    replace(
                        chunk,
                        task_range=TaskRange(
                            axis=chunk.task_range.axis,
                            begin=cursor,
                            end=end,
                        ),
                    )
                )
                cursor = end
            if cursor == segment_end:
                ordered[run_begin:run_end] = replacements
        balanced_tasks.append(tuple(ordered))
    return tuple(balanced_tasks)


def balance_streaming_tree_owner_work_chunks(
    topology: GPUTopology,
    chunks_by_task: tuple[tuple[StreamingChunk, ...], ...],
    *,
    block_size: int,
    consumer_side: str,
    hierarchical_cross_cluster: bool,
) -> tuple[tuple[StreamingChunk, ...], ...]:
    """Trade leaf blocks for the reductions owned by the same CTA."""

    maximum_task_span = max(
        (task_chunks[-1].task_range.end - task_chunks[0].task_range.begin for task_chunks in chunks_by_task if task_chunks),
        default=0,
    )
    cost_model = current_scheduler_config().cost_model
    reduce_us = estimated_streaming_reduce_cost_us(2)
    iter_per_block_us = cost_model.iter_per_block_us
    if iter_per_block_us <= 0.0:
        return chunks_by_task

    def count_reducers(
        plan: OrderedReductionTreePlan,
        counts: dict[int, int],
    ) -> None:
        if plan.is_leaf:
            return
        counts[plan.consumer_sm] = counts.get(plan.consumer_sm, 0) + 1
        assert plan.left is not None and plan.right is not None
        count_reducers(plan.left, counts)
        count_reducers(plan.right, counts)

    def tree_optimal_block_counts(
        plan: OrderedReductionTreePlan,
        total_blocks: int,
    ) -> list[int] | None:
        """Minimize one fixed reduction tree's modeled completion time.

        Reduction work on a consumer spine overlaps computation in its remote
        subtrees, so charging every owned reduction as an additive leaf cost
        systematically under-fills tree roots.  This integer DP preserves the
        selected ordered tree and assigns at least one block to every leaf.
        """

        plans_by_interval: dict[tuple[int, int], OrderedReductionTreePlan] = {}

        def index_plan(subtree: OrderedReductionTreePlan) -> None:
            plans_by_interval[(subtree.begin, subtree.end)] = subtree
            if subtree.is_leaf:
                return
            assert subtree.left is not None and subtree.right is not None
            index_plan(subtree.left)
            index_plan(subtree.right)

        index_plan(plan)
        memo: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {}

        def solve(
            begin: int,
            end: int,
            block_count: int,
        ) -> tuple[float, tuple[int, ...]] | None:
            key = (begin, end, block_count)
            if key in memo:
                return memo[key]
            subtree = plans_by_interval[(begin, end)]
            leaf_count = end - begin
            if block_count < leaf_count:
                return None
            if subtree.is_leaf:
                result = (
                    cost_model.iter_base_us + iter_per_block_us * block_count,
                    (block_count,),
                )
                memo[key] = result
                return result

            assert subtree.left is not None and subtree.right is not None
            left_leaf_count = subtree.left.end - subtree.left.begin
            right_leaf_count = subtree.right.end - subtree.right.begin
            best: (
                tuple[
                    tuple[float, int, int, tuple[int, ...]],
                    tuple[float, tuple[int, ...]],
                ]
                | None
            ) = None
            for left_blocks in range(
                left_leaf_count,
                block_count - right_leaf_count + 1,
            ):
                left_result = solve(
                    subtree.left.begin,
                    subtree.left.end,
                    left_blocks,
                )
                right_result = solve(
                    subtree.right.begin,
                    subtree.right.end,
                    block_count - left_blocks,
                )
                if left_result is None or right_result is None:
                    continue
                left_ready, left_counts = left_result
                right_ready, right_counts = right_result
                if subtree.consumer_sm == subtree.left.consumer_sm:
                    consumer_ready = left_ready
                    remote_ready = right_ready
                    remote_sm = subtree.right.consumer_sm
                else:
                    consumer_ready = right_ready
                    remote_ready = left_ready
                    remote_sm = subtree.left.consumer_sm
                remote_ready += estimated_streaming_comm_cost_us(
                    topology,
                    remote_sm,
                    subtree.consumer_sm,
                )
                ready = max(consumer_ready, remote_ready) + reduce_us
                counts_tuple = left_counts + right_counts
                spread = max(counts_tuple) - min(counts_tuple)
                squared_load = sum(value * value for value in counts_tuple)
                candidate_key = (
                    ready,
                    spread,
                    squared_load,
                    counts_tuple,
                )
                candidate = (ready, counts_tuple)
                if best is None or candidate_key < best[0]:
                    best = (candidate_key, candidate)
            if best is None:
                return None
            memo[key] = best[1]
            return best[1]

        result = solve(plan.begin, plan.end, total_blocks)
        return None if result is None else list(result[1])

    result: list[tuple[StreamingChunk, ...]] = []
    for task_chunks in chunks_by_task:
        ordered = list(logical_streaming_chunk_order(task_chunks))
        task_span = 0 if not ordered else ordered[-1].task_range.end - ordered[0].task_range.begin
        if len(ordered) <= 1 or task_span < maximum_task_span:
            result.append(tuple(ordered))
            continue
        runs = ordered_cluster_run_bounds(
            topology,
            [chunk.sm_id for chunk in ordered],
        )
        counts: dict[int, int] = {}
        local_plans: list[OrderedReductionTreePlan] = []
        for _cluster_id, run_begin, run_end in runs:
            run = ordered[run_begin:run_end]
            plan = ordered_reduction_tree_plan(
                topology,
                tuple(
                    (
                        chunk.sm_id,
                        estimated_streaming_iter_cost_us(
                            chunk.task_range,
                            block_size,
                        ),
                    )
                    for chunk in run
                ),
                consumer_side=consumer_side,
                adaptive_consumer=config_flag("ordered_tree_adaptive_consumer"),
            )
            local_plans.append(plan)
            count_reducers(plan, counts)
        if hierarchical_cross_cluster and len(local_plans) > 1:
            global_plan = ordered_reduction_tree_plan(
                topology,
                tuple((plan.consumer_sm, plan.ready_us) for plan in local_plans),
                consumer_side=consumer_side,
                adaptive_consumer=config_flag("ordered_tree_adaptive_consumer"),
            )
            count_reducers(global_plan, counts)

        for run_index, (_cluster_id, run_begin, run_end) in enumerate(runs):
            run = ordered[run_begin:run_end]
            if len(run) <= 1 or any(left.task_range.end != right.task_range.begin for left, right in zip(run, run[1:])):
                continue
            segment_begin = run[0].task_range.begin
            segment_end = run[-1].task_range.end
            total_blocks = math.ceil((segment_end - segment_begin) / block_size)
            if total_blocks < len(run):
                continue
            block_counts = (
                tree_optimal_block_counts(
                    local_plans[run_index],
                    total_blocks,
                )
                if config_flag("balance_tree_owner_work_dp")
                else None
            )
            if block_counts is None:
                block_counts = [1 for _ in run]
                for _ in range(total_blocks - len(run)):
                    owner_index = min(
                        range(len(run)),
                        key=lambda index: (
                            block_counts[index] * iter_per_block_us + counts.get(run[index].sm_id, 0) * reduce_us,
                            block_counts[index],
                            index,
                        ),
                    )
                    block_counts[owner_index] += 1

            if config_flag("balance_tree_leaf_pair_skew"):
                run_index_by_sm = {chunk.sm_id: index for index, chunk in enumerate(run)}

                def close_leaf_pair_skew(
                    subtree: OrderedReductionTreePlan,
                    run_index_by_sm: dict[int, int] = run_index_by_sm,
                    counts: dict[int, int] = counts,
                    block_counts: list[int] = block_counts,
                ) -> None:
                    if subtree.is_leaf:
                        return
                    assert subtree.left is not None and subtree.right is not None
                    if subtree.left.is_leaf and subtree.right.is_leaf:
                        consumer_index = run_index_by_sm.get(subtree.consumer_sm)
                        remote_sm = (
                            subtree.right.consumer_sm if subtree.left.consumer_sm == subtree.consumer_sm else subtree.left.consumer_sm
                        )
                        remote_index = run_index_by_sm.get(remote_sm)
                        if consumer_index is None or remote_index is None:
                            return
                        downstream_reductions = max(
                            0,
                            counts.get(subtree.consumer_sm, 0) - 1,
                        )
                        target_skew_blocks = round(
                            downstream_reductions
                            * cost_model.reduce_single_us
                            * max(
                                0.0,
                                config_float(
                                    "balance_tree_leaf_pair_reduction_weight",
                                    1.0,
                                ),
                            )
                            / iter_per_block_us
                        )
                        while block_counts[remote_index] > 1:
                            current_skew = block_counts[remote_index] - block_counts[consumer_index]
                            shifted_skew = current_skew - 2
                            if abs(shifted_skew - target_skew_blocks) >= abs(current_skew - target_skew_blocks):
                                break
                            block_counts[remote_index] -= 1
                            block_counts[consumer_index] += 1
                        return
                    close_leaf_pair_skew(subtree.left)
                    close_leaf_pair_skew(subtree.right)

                close_leaf_pair_skew(local_plans[run_index])

            cursor = segment_begin
            replacements: list[StreamingChunk] = []
            for chunk, chunk_blocks in zip(run, block_counts):
                end = min(segment_end, cursor + chunk_blocks * block_size)
                replacements.append(
                    replace(
                        chunk,
                        task_range=TaskRange(
                            axis=chunk.task_range.axis,
                            begin=cursor,
                            end=end,
                        ),
                    )
                )
                cursor = end
            if cursor == segment_end:
                ordered[run_begin:run_end] = replacements
        result.append(tuple(ordered))
    return tuple(result)


def ordered_cluster_run_bounds(
    topology: GPUTopology,
    sm_ids: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    """Return contiguous ``(cluster_id, begin, end)`` runs in logical order."""

    runs: list[tuple[int, int, int]] = []
    for index, sm_id in enumerate(sm_ids):
        cluster_id = topology.cluster_id(sm_id)
        if runs and runs[-1][0] == cluster_id:
            previous_cluster, begin, _ = runs[-1]
            runs[-1] = (previous_cluster, begin, index + 1)
        else:
            runs.append((cluster_id, index, index + 1))
    return tuple(runs)


@dataclass(frozen=True)
class StreamingSegment:
    task_id: int
    cluster_id: int
    begin: int
    end: int


@dataclass(frozen=True)
class CapacityItem:
    task_id: int
    begin: int
    end: int
    block_count: int
    split_task: bool


@dataclass(frozen=True)
class StreamingDagReplayScore:
    max_finish_us: float
    p95_finish_us: float
    finish_spread_us: float
    total_recv_wait_us: float
    max_recv_wait_us: float
    hbm_edges: int
    instruction_count: int
    slot_count: int
    max_local_root_skew_us: float = 0.0
    total_local_root_skew_us: float = 0.0
    split_task_count: int = 0
    max_recv_instruction_id: int | None = None
    max_recv_task_id: int | None = None
    max_recv_sm_id: int | None = None
    max_recv_slot_id: int | None = None
    # Descending CTA finish times form a minimax-fair load profile.  Keeping
    # the complete profile lets local placement searches make progress across
    # a plateau with several equally critical CTAs: first minimize the slowest
    # CTA, then the second slowest, and so on.
    finish_profile_us: tuple[float, ...] = ()
    finish_by_sm_us: tuple[float, ...] = ()
    critical_leaf_owners: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class StreamingReplayInstruction:
    instruction_id: int
    task_id: int
    sm_id: int
    input_slots: tuple[int, ...]
    output_slot: int | None
    duration_us: float
    queue_key: tuple[int, int, int, int]
    level0_sort_us: float = 0.0


@dataclass(frozen=True)
class QueueGroupReplayScore:
    objective_us: float
    max_finish_us: float
    p95_finish_us: float
    finish_spread_us: float
    total_recv_wait_us: float
    max_recv_wait_us: float


@dataclass(frozen=True)
class ProducerReadyQueueNode:
    node_id: int
    sm_id: int
    input_slots: tuple[int, ...]
    output_slots: tuple[int, ...]
    duration_us: float
    sort_key: tuple[int, ...]


@dataclass(frozen=True)
class OrderedReductionTreePlan:
    """One order-preserving binary reduction interval."""

    begin: int
    end: int
    split: int | None
    consumer_sm: int
    ready_us: float
    height: int
    critical_guard_wait_us: float
    total_guard_wait_us: float
    left: OrderedReductionTreePlan | None = None
    right: OrderedReductionTreePlan | None = None

    @property
    def is_leaf(self) -> bool:
        return self.split is None


def ordered_reduction_tree_plan(
    topology: GPUTopology,
    leaves: Sequence[tuple[int, float]],
    *,
    consumer_side: str,
    adaptive_consumer: bool = False,
    force_hbm_comms: bool = False,
) -> OrderedReductionTreePlan:
    """Find the lowest-latency alphabetic tree for fixed leaf owners/readiness.

    The dynamic program only combines contiguous intervals, so it never relies
    on commutativity.  A parent remains on the selected consumer child's CTA;
    the other child can push while that consumer child is still computing.
    """

    leaves = tuple((int(sm_id), float(ready_us)) for sm_id, ready_us in leaves)
    if not leaves:
        raise ValueError("Dataflow ordered reduction tree requires at least one leaf")
    if consumer_side not in {"left", "right"}:
        raise ValueError(f"Dataflow ordered reduction tree consumer_side must be 'left' or 'right', got {consumer_side!r}")
    for index, (sm_id, ready_us) in enumerate(leaves):
        if sm_id < 0 or sm_id >= topology.sm_count:
            raise ValueError(f"Dataflow ordered reduction tree leaf {index} uses invalid SM {sm_id}")
        if not math.isfinite(ready_us) or ready_us < 0.0:
            raise ValueError(f"Dataflow ordered reduction tree leaf {index} has invalid readiness {ready_us}")

    plans: dict[tuple[int, int], OrderedReductionTreePlan] = {
        (index, index + 1): OrderedReductionTreePlan(
            begin=index,
            end=index + 1,
            split=None,
            consumer_sm=sm_id,
            ready_us=ready_us,
            height=0,
            critical_guard_wait_us=0.0,
            total_guard_wait_us=0.0,
        )
        for index, (sm_id, ready_us) in enumerate(leaves)
    }
    reduce_us = estimated_streaming_reduce_cost_us(2)
    for width in range(2, len(leaves) + 1):
        for begin in range(0, len(leaves) - width + 1):
            end = begin + width
            candidates: list[
                tuple[
                    tuple[float, float, float, int, int],
                    OrderedReductionTreePlan,
                ]
            ] = []
            for split in range(begin + 1, end):
                left = plans[(begin, split)]
                right = plans[(split, end)]
                consumer_choices = (
                    (("left", left, right), ("right", right, left))
                    if adaptive_consumer
                    else (
                        (
                            consumer_side,
                            right if consumer_side == "right" else left,
                            left if consumer_side == "right" else right,
                        ),
                    )
                )
                for selected_side, consumer, remote in consumer_choices:
                    comm_us = estimated_streaming_comm_cost_us(
                        topology,
                        remote.consumer_sm,
                        consumer.consumer_sm,
                        force_hbm_comms=force_hbm_comms,
                    )
                    remote_ready_us = remote.ready_us + comm_us
                    guard_wait_us = max(0.0, remote_ready_us - consumer.ready_us)
                    plan = OrderedReductionTreePlan(
                        begin=begin,
                        end=end,
                        split=split,
                        consumer_sm=consumer.consumer_sm,
                        ready_us=max(remote_ready_us, consumer.ready_us) + reduce_us,
                        height=max(left.height, right.height) + 1,
                        critical_guard_wait_us=max(
                            left.critical_guard_wait_us,
                            right.critical_guard_wait_us,
                            guard_wait_us,
                        ),
                        total_guard_wait_us=(left.total_guard_wait_us + right.total_guard_wait_us + guard_wait_us),
                        left=left,
                        right=right,
                    )
                    candidates.append(
                        (
                            (
                                plan.ready_us,
                                plan.critical_guard_wait_us,
                                plan.total_guard_wait_us,
                                plan.height,
                                0 if selected_side == consumer_side else 1,
                                split if consumer_side == "right" else -split,
                            ),
                            plan,
                        )
                    )
            if config_flag("ordered_tree_min_height"):
                minimum_height = (width - 1).bit_length()
                height_optimal = [candidate for candidate in candidates if candidate[1].height == minimum_height]
                if height_optimal:
                    candidates = height_optimal
            plans[(begin, end)] = min(candidates, key=lambda item: item[0])[1]
    return plans[(0, len(leaves))]


def ordered_reduction_fused_frontier(
    plan: OrderedReductionTreePlan,
    *,
    max_reduce_arity: int,
    max_resident_remote_inputs: int,
    output_alias_input_indices: Sequence[int] = (),
) -> tuple[OrderedReductionTreePlan, ...]:
    """Expose an executable same-CTA multi-input reduction frontier.

    Expanding a same-CTA child removes one local reduction, but may add another
    remote value that must remain live until the fused handler runs.  Bound the
    expansion by the number of independently lifetimed inbox resources that the
    joint lowering can materialize; larger requested arities remain valid and
    simply stop at the largest executable frontier.
    """

    if plan.is_leaf:
        return (plan,)
    if max_reduce_arity < 2:
        raise ValueError(f"ordered_tree_max_reduce_arity must be at least 2, got {max_reduce_arity}")
    if max_resident_remote_inputs < 1:
        raise ValueError(f"max_resident_remote_inputs must be at least 1, got {max_resident_remote_inputs}")
    assert plan.left is not None and plan.right is not None
    alias_indices = frozenset(int(index) for index in output_alias_input_indices)
    frontier = [plan.left, plan.right]
    while len(frontier) < max_reduce_arity:
        expandable = []
        for index, child in enumerate(frontier):
            if child.is_leaf or child.consumer_sm != plan.consumer_sm:
                continue
            assert child.left is not None and child.right is not None
            candidate = frontier[:index] + [child.left, child.right] + frontier[index + 1 :]
            remote_inputs = sum(item.consumer_sm != plan.consumer_sm for item in candidate)
            local_input_indices = tuple(
                candidate_index for candidate_index, item in enumerate(candidate) if item.consumer_sm == plan.consumer_sm
            )
            if remote_inputs <= max_resident_remote_inputs and (
                not alias_indices or (len(local_input_indices) == 1 and local_input_indices[0] in alias_indices)
            ):
                expandable.append((index, child))
        if not expandable:
            break
        index, child = max(
            expandable,
            key=lambda item: (
                item[1].height,
                item[1].end - item[1].begin,
                -item[0],
            ),
        )
        assert child.left is not None and child.right is not None
        frontier[index : index + 1] = [child.left, child.right]
    return tuple(frontier)


def producer_ready_queue_order(
    topology: GPUTopology,
    nodes: Sequence[ProducerReadyQueueNode],
    *,
    force_hbm_comms: bool = False,
) -> dict[int, tuple[int, ...]]:
    """Order fixed-SM DAG nodes while filling remote-input wait with ready work."""

    nodes_by_id = {node.node_id: node for node in nodes}
    if len(nodes_by_id) != len(nodes):
        raise ValueError("Dataflow producer-ready queue nodes must have unique ids")

    producer_by_slot: dict[int, int] = {}
    for node in nodes:
        if node.sm_id < 0 or node.sm_id >= topology.sm_count:
            raise ValueError(f"Dataflow producer-ready queue node {node.node_id} uses invalid SM {node.sm_id}")
        for slot_id in node.output_slots:
            previous = producer_by_slot.setdefault(slot_id, node.node_id)
            if previous != node.node_id:
                raise ValueError(f"Dataflow producer-ready queue slot {slot_id} has multiple producers")

    predecessors: dict[int, set[int]] = {node.node_id: set() for node in nodes}
    successors: dict[int, set[int]] = {node.node_id: set() for node in nodes}
    for node in nodes:
        for slot_id in node.input_slots:
            producer_id = producer_by_slot.get(slot_id)
            if producer_id is None or producer_id == node.node_id:
                continue
            predecessors[node.node_id].add(producer_id)
            successors[producer_id].add(node.node_id)

    rank_cache: dict[int, float] = {}
    rank_stack: set[int] = set()

    def upward_rank(node_id: int) -> float:
        cached = rank_cache.get(node_id)
        if cached is not None:
            return cached
        if node_id in rank_stack:
            raise RuntimeError("Dataflow producer-ready queue dependency graph contains a cycle")
        rank_stack.add(node_id)
        node = nodes_by_id[node_id]
        tail = 0.0
        for successor_id in successors[node_id]:
            successor = nodes_by_id[successor_id]
            comm_cost = estimated_streaming_comm_cost_us(
                topology,
                node.sm_id,
                successor.sm_id,
                force_hbm_comms=force_hbm_comms,
            )
            tail = max(tail, comm_cost + upward_rank(successor_id))
        rank_stack.remove(node_id)
        result = node.duration_us + tail
        rank_cache[node_id] = result
        return result

    for node_id in nodes_by_id:
        upward_rank(node_id)
    remote_unlock_rank = {
        node_id: max(
            (
                estimated_streaming_comm_cost_us(
                    topology,
                    nodes_by_id[node_id].sm_id,
                    nodes_by_id[successor_id].sm_id,
                    force_hbm_comms=force_hbm_comms,
                )
                + rank_cache[successor_id]
                for successor_id in successors[node_id]
                if nodes_by_id[successor_id].sm_id != nodes_by_id[node_id].sm_id
            ),
            default=0.0,
        )
        for node_id in nodes_by_id
    }

    remaining_predecessors = {node_id: len(node_predecessors) for node_id, node_predecessors in predecessors.items()}
    ready = {node_id for node_id, predecessor_count in remaining_predecessors.items() if predecessor_count == 0}
    finish_by_node: dict[int, float] = {}
    sm_ready = {sm_id: 0.0 for sm_id in range(topology.sm_count)}
    ordered: dict[int, list[int]] = {sm_id: [] for sm_id in range(topology.sm_count)}

    while ready:

        def candidate_key(
            node_id: int,
        ) -> tuple[
            bool,
            float,
            int,
            bool,
            float,
            float,
            float,
            float,
            float,
            tuple[int, ...],
        ]:
            node = nodes_by_id[node_id]
            input_ready = max(
                (
                    finish_by_node[predecessor_id]
                    + estimated_streaming_comm_cost_us(
                        topology,
                        nodes_by_id[predecessor_id].sm_id,
                        node.sm_id,
                        force_hbm_comms=force_hbm_comms,
                    )
                    for predecessor_id in predecessors[node_id]
                ),
                default=0.0,
            )
            wait = max(0.0, input_ready - sm_ready[node.sm_id])
            local_release_count = sum(
                1
                for predecessor_id in predecessors[node_id]
                if (predecessor_id in finish_by_node and nodes_by_id[predecessor_id].sm_id == node.sm_id)
            )
            candidate_finish = max(sm_ready[node.sm_id], input_ready) + node.duration_us
            local_retention_wait = 0.0
            for successor_id in successors[node_id]:
                successor = nodes_by_id[successor_id]
                if successor.sm_id != node.sm_id:
                    continue
                other_predecessors = predecessors[successor_id] - {node_id}
                if any(predecessor_id not in finish_by_node for predecessor_id in other_predecessors):
                    local_retention_wait = math.inf
                    break
                successor_input_ready = max(
                    (
                        finish_by_node[predecessor_id]
                        + estimated_streaming_comm_cost_us(
                            topology,
                            nodes_by_id[predecessor_id].sm_id,
                            successor.sm_id,
                            force_hbm_comms=force_hbm_comms,
                        )
                        for predecessor_id in other_predecessors
                    ),
                    default=candidate_finish,
                )
                local_retention_wait = max(
                    local_retention_wait,
                    successor_input_ready - candidate_finish,
                )
            ready_unlock_rank = max(
                (rank_cache[successor_id] for successor_id in successors[node_id] if remaining_predecessors[successor_id] == 1),
                default=0.0,
            )
            return (
                wait > 1e-9,
                wait,
                -local_release_count,
                local_retention_wait > 1e-9,
                local_retention_wait,
                -remote_unlock_rank[node_id],
                -ready_unlock_rank,
                -rank_cache[node_id],
                input_ready,
                node.sort_key,
            )

        node_id = min(ready, key=candidate_key)
        ready.remove(node_id)
        node = nodes_by_id[node_id]
        input_ready = max(
            (
                finish_by_node[predecessor_id]
                + estimated_streaming_comm_cost_us(
                    topology,
                    nodes_by_id[predecessor_id].sm_id,
                    node.sm_id,
                    force_hbm_comms=force_hbm_comms,
                )
                for predecessor_id in predecessors[node_id]
            ),
            default=0.0,
        )
        finish = max(sm_ready[node.sm_id], input_ready) + node.duration_us
        sm_ready[node.sm_id] = finish
        finish_by_node[node_id] = finish
        ordered[node.sm_id].append(node_id)
        for successor_id in successors[node_id]:
            remaining_predecessors[successor_id] -= 1
            if remaining_predecessors[successor_id] == 0:
                ready.add(successor_id)

    if len(finish_by_node) != len(nodes_by_id):
        blocked = sorted(set(nodes_by_id) - set(finish_by_node))
        raise RuntimeError(f"Dataflow producer-ready queue could not resolve dependencies for nodes {blocked[:8]}")
    return {sm_id: tuple(node_ids) for sm_id, node_ids in ordered.items()}


def as_lengths(value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, int):
        values = (value,)
    else:
        values = tuple(value)
    if not values:
        raise ValueError("Dataflow scheduler range_lengths must contain at least one task length")
    for item in values:
        if not isinstance(item, int):
            raise TypeError(f"Dataflow scheduler range length must be an int, got {item!r}")
        if item <= 0:
            raise ValueError(f"Dataflow scheduler range length must be positive, got {item}")
    return values


def as_offsets(value: int | Sequence[int], *, task_count: int) -> tuple[int, ...]:
    if isinstance(value, int):
        values = (value,) * task_count
    else:
        values = tuple(value)
    if len(values) != task_count:
        raise ValueError(f"Dataflow scheduler range_offsets must contain one offset per task; expected {task_count}, got {len(values)}")
    for item in values:
        if not isinstance(item, int):
            raise TypeError(f"Dataflow scheduler range offset must be an int, got {item!r}")
        if item < 0:
            raise ValueError(f"Dataflow scheduler range offset must be non-negative, got {item}")
    return values


def resolve_range_axis(program: DataflowProgram, range_lengths: dict[Any, int | Sequence[int]]) -> Any:
    program.validate()
    assert program.partial_stage is not None

    axis = program.partial_stage.range_axis
    if axis is not None:
        if axis not in range_lengths:
            raise ValueError(f"Dataflow scheduler missing range_lengths entry for range_axis {axis!r}")
        return axis

    if len(range_lengths) != 1:
        raise ValueError("Dataflow scheduler requires range_axis on the partial stage when multiple range_lengths are provided")
    return next(iter(range_lengths.keys()))


def normalize_task_extents(task_extents: Sequence[int] | None, task_count: int) -> tuple[int, ...] | None:
    if task_extents is None:
        return None
    extents = tuple(task_extents)
    if not extents:
        raise ValueError("Dataflow scheduler task_extents must contain at least one dimension")
    for extent in extents:
        if not isinstance(extent, int):
            raise TypeError(f"Dataflow scheduler task extent must be an int, got {extent!r}")
        if extent <= 0:
            raise ValueError(f"Dataflow scheduler task extent must be positive, got {extent}")
    product = math.prod(extents)
    if product != task_count:
        raise ValueError(f"Dataflow scheduler task_extents product must match task count: got product {product}, task count {task_count}")
    return extents


def task_coords(task_id: int, task_extents: tuple[int, ...] | None) -> tuple[int, ...]:
    if task_extents is None:
        return (task_id,)
    coords = []
    remainder = task_id
    for extent in reversed(task_extents):
        coords.append(remainder % extent)
        remainder //= extent
    return tuple(reversed(coords))


def normalize_task_coord_overrides(
    value: Sequence[Sequence[int] | int] | None,
    *,
    task_count: int,
    task_rank: int,
) -> tuple[tuple[int, ...], ...] | None:
    if value is None:
        return None
    if len(value) != task_count:
        raise ValueError(f"Dataflow scheduler task_coord_overrides length must match task count: got {len(value)}, expected {task_count}")
    normalized: list[tuple[int, ...]] = []
    for task_index, item in enumerate(value):
        coords = (item,) if isinstance(item, int) else tuple(item)
        if len(coords) != task_rank:
            raise ValueError(
                "Dataflow scheduler task_coord_overrides entries must match task_domain rank: "
                f"task {task_index} has rank {len(coords)}, expected {task_rank}"
            )
        for coord in coords:
            if not isinstance(coord, int):
                raise TypeError(f"Dataflow scheduler task_coord_overrides entries must contain ints, got {coord!r}")
            if coord < 0:
                raise ValueError(f"Dataflow scheduler task_coord_overrides entries must be non-negative, got {coord}")
        normalized.append(coords)
    return tuple(normalized)


def normalize_stage_graph_task_weights(
    value: Sequence[int | float] | None,
    *,
    task_count: int,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if len(value) != task_count:
        raise ValueError(f"Dataflow stage_graph_task_weights length must match task count: got {len(value)}, expected {task_count}")
    weights: list[float] = []
    for index, raw_weight in enumerate(value):
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Dataflow stage_graph_task_weights entries must be finite non-negative values: task {index} has {raw_weight!r}"
            )
        weights.append(weight)
    return tuple(weights)


def normalize_stage_graph_cluster_assignment(
    value: Sequence[int] | None,
    *,
    task_count: int,
    cluster_count: int,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if len(value) != task_count:
        raise ValueError(f"Dataflow stage_graph_cluster_assignment length must match task count: got {len(value)}, expected {task_count}")
    assignment = tuple(int(item) for item in value)
    for task_id, cluster_id in enumerate(assignment):
        if cluster_id < 0 or cluster_id >= cluster_count:
            raise ValueError(
                "Dataflow stage_graph_cluster_assignment contains an out-of-range cluster: "
                f"task {task_id} -> cluster {cluster_id}, cluster_count={cluster_count}"
            )
    return assignment


def partition_range(length: int, block_size: int, axis: Any, *, offset: int = 0) -> tuple[TaskRange, ...]:
    return tuple(
        TaskRange(axis=axis, begin=offset + begin, end=offset + min(begin + block_size, length)) for begin in range(0, length, block_size)
    )


def linear_reshared_stages(program: DataflowProgram) -> tuple[DataflowStage, DataflowStage, DataflowStage, DataflowStage | None]:
    program.validate()
    stages = tuple(program.stages)
    kinds = tuple(stage.kind for stage in stages)
    expected_with_finalize = (
        DataflowStageKind.MAP,
        DataflowStageKind.RESHARED,
        DataflowStageKind.MAP,
        DataflowStageKind.FINALIZE,
    )
    expected_terminal_map = (
        DataflowStageKind.MAP,
        DataflowStageKind.RESHARED,
        DataflowStageKind.MAP,
    )
    if kinds == expected_with_finalize:
        map1, reshared, map2, finalize = stages
    elif kinds == expected_terminal_map:
        map1, reshared, map2 = stages
        finalize = None
    else:
        actual = " -> ".join(kind.value for kind in kinds)
        raise ValueError(
            "Dataflow stage graph scheduler currently supports exactly "
            f"map -> reshared -> map or map -> reshared -> map -> finalize, got {actual}"
        )
    if map1.call is None or map2.call is None or (finalize is not None and finalize.call is None):
        raise ValueError("Dataflow stage graph compute stages require operator calls")
    if reshared.output_arity <= 0:
        raise ValueError(f"Dataflow reshared stage {reshared.name!r} output_arity must be positive")
    return map1, reshared, map2, finalize


def stage_graph_range_lengths(
    stage: DataflowStage,
    range_lengths: dict[Any, int | Sequence[int]],
) -> tuple[int, ...]:
    axis = stage.range_axis
    if axis is None:
        raise ValueError(f"Dataflow stage {stage.name!r} requires range_axis for graph scheduling")
    if axis not in range_lengths:
        raise ValueError(f"Dataflow scheduler missing range_lengths entry for range_axis {axis!r}")
    return as_lengths(range_lengths[axis])


def cluster_rank_ranges(
    *,
    length: int,
    axis: Any,
    topology: GPUTopology,
) -> tuple[TaskRange, ...]:
    if length % topology.cluster_size != 0:
        raise ValueError(
            f"Dataflow reshared graph range length must be divisible by cluster_size: length={length}, cluster_size={topology.cluster_size}"
        )
    shard = length // topology.cluster_size
    if shard <= 0:
        raise ValueError(f"Dataflow reshared graph shard length must be positive, got {shard}")
    return tuple(TaskRange(axis=axis, begin=rank * shard, end=(rank + 1) * shard) for rank in range(topology.cluster_size))


def stage_graph_range_request(
    stage: DataflowStage,
) -> DataflowRangeCoarseningRequest | None:
    request = stage.attrs.get(DATAFLOW_RANGE_CONTRACT_ATTR)
    if request is not None:
        legacy = {"range_tile", "range_tile_size"}.intersection(stage.attrs)
        if legacy:
            raise ValueError(f"Dataflow stage {stage.name!r} cannot combine range_contract with legacy range attrs {sorted(legacy)!r}")
        if not isinstance(request, DataflowRangeCoarseningRequest):
            raise TypeError(f"Dataflow stage {stage.name!r} has an unnormalized range contract")
        dataflow_implementation_registry().require_contract_compatible(
            DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION,
            request,
            selected_explicitly=False,
        )
        return request
    return None


def stage_graph_legacy_range_tile_size(stage: DataflowStage) -> int | None:
    raw_tile_size = stage.attrs.get("range_tile")
    if raw_tile_size is None:
        raw_tile_size = stage.attrs.get("range_tile_size")
    if raw_tile_size is None:
        return None
    if isinstance(raw_tile_size, bool):
        raise ValueError(f"Dataflow stage {stage.name!r} range_tile must be a positive integer")
    try:
        tile_size = int(raw_tile_size)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Dataflow stage {stage.name!r} range_tile must be a positive integer") from err
    if tile_size <= 0:
        raise ValueError(f"Dataflow stage {stage.name!r} range_tile must be positive, got {tile_size}")
    return tile_size


def stage_graph_range_plan(
    stage: DataflowStage,
    request: DataflowRangeCoarseningRequest,
    *,
    logical_range_extent: int,
    resource_budget_bytes: int | None,
) -> DataflowRangeCoarseningPlan:
    return plan_range_coarsening(
        request,
        logical_range_extent=logical_range_extent,
        resource_bytes_per_tile=(
            0
            if stage.output_type is None
            else estimate_range_output_bytes_per_tile(
                stage.output_type,
                output_tile_arity=request.output_tile_arity,
            )
        ),
        resource_budget_bytes=resource_budget_bytes,
    )


@dataclass(frozen=True)
class CrossHandlerHandoffSpec:
    producer_stage_id: int
    consumer_stage_id: int
    stage_count: int
    lookahead_distance: int = 1
    plan: DataflowCrossHandlerHandoffPlan | None = None


def cross_handler_handoff_spec(
    program: DataflowProgram,
    *,
    max_shared_memory_bytes: int | None = None,
) -> CrossHandlerHandoffSpec | None:
    specs: list[CrossHandlerHandoffSpec] = []
    for producer in program.stages:
        request = producer.attrs.get(DATAFLOW_HANDOFF_CONTRACT_ATTR)
        if request is None:
            raw_spec = producer.attrs.get("cross_handler_handoff")
            if raw_spec is None:
                continue
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"Dataflow stage {producer.name!r} cross_handler_handoff must be a mapping")
            unknown = set(raw_spec) - {"consumer_stage", "stages"}
            if unknown:
                raise ValueError(f"Dataflow stage {producer.name!r} cross_handler_handoff has unknown keys {sorted(unknown)!r}")
            consumer_name = raw_spec.get("consumer_stage")
            if not isinstance(consumer_name, str) or not consumer_name:
                raise ValueError(f"Dataflow stage {producer.name!r} cross_handler_handoff consumer_stage must name a stage")
            try:
                consumer = program.stage(consumer_name)
            except KeyError as err:
                raise ValueError(
                    f"Dataflow stage {producer.name!r} cross_handler_handoff references unknown consumer stage {consumer_name!r}"
                ) from err
            raw_stage_count = raw_spec.get("stages")
            try:
                if isinstance(raw_stage_count, bool):
                    raise TypeError
                stage_count = operator.index(raw_stage_count)
            except (TypeError, ValueError) as err:
                raise ValueError("cross_handler_handoff stages must be a positive integer") from err
            if stage_count <= 0:
                raise ValueError(f"cross_handler_handoff stages must be positive, got {stage_count}")
        else:
            if "cross_handler_handoff" in producer.attrs:
                raise ValueError(f"Dataflow stage {producer.name!r} cannot combine handoff_contract with legacy cross_handler_handoff")
            if not isinstance(request, DataflowCrossHandlerHandoffRequest):
                raise TypeError(f"Dataflow stage {producer.name!r} has an unnormalized handoff contract")
            dataflow_implementation_registry().require_contract_compatible(
                DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
                request,
                selected_explicitly=False,
            )
            try:
                consumer = program.stage(request.consumer_stage_id)
            except KeyError as err:
                raise ValueError(
                    f"Dataflow stage {producer.name!r} handoff_contract references unknown consumer stage id {request.consumer_stage_id}"
                ) from err
            stage_count = request.buffer_stages
        if producer.kind is not DataflowStageKind.MAP or consumer.kind is not DataflowStageKind.MAP:
            raise ValueError("cross_handler_handoff currently requires map producer and consumer stages")
        if consumer.stage_id >= producer.stage_id:
            raise ValueError(
                "cross_handler_handoff consumer_stage must be an earlier, distinct "
                "stage so the producer can hand shared state to the next task"
            )
        if producer.stage_id != program.stages[-1].stage_id:
            raise ValueError(
                "cross_handler_handoff producer must be the terminal stage so no compute handler intervenes before the next task"
            )
        handoff_plan = None
        lookahead_distance = 1
        if isinstance(request, DataflowCrossHandlerHandoffRequest):
            consumer_pipeline_request = consumer.attrs.get(DATAFLOW_PIPELINE_CONTRACT_ATTR)
            if consumer_pipeline_request is not None and not isinstance(
                consumer_pipeline_request,
                DataflowPipelineRequest,
            ):
                raise TypeError(f"Dataflow stage {consumer.name!r} has an unnormalized pipeline contract")
            handoff_plan = plan_cross_handler_handoff(
                request,
                producer_stage_id=producer.stage_id,
                consumer_pipeline_request=consumer_pipeline_request,
                task_coord_rank=len(program.task_domain.axes),
                max_shared_memory_bytes=max_shared_memory_bytes,
            )
            stage_count = handoff_plan.selected_buffer_stages
            lookahead_distance = handoff_plan.lookahead_distance
        specs.append(
            CrossHandlerHandoffSpec(
                producer_stage_id=producer.stage_id,
                consumer_stage_id=consumer.stage_id,
                stage_count=stage_count,
                lookahead_distance=lookahead_distance,
                plan=handoff_plan,
            )
        )
    if len(specs) > 1:
        raise ValueError("Dataflow stage graph currently supports one cross_handler_handoff declaration")
    return specs[0] if specs else None


def annotate_cross_handler_handoff(
    instructions: list[Instruction],
    queues: dict[int, list[Instruction]],
    spec: CrossHandlerHandoffSpec | None,
) -> tuple[
    list[Instruction],
    dict[int, list[Instruction]],
    tuple[DataflowHandoffQueueBinding, ...],
]:
    if spec is None:
        return instructions, queues, ()

    if spec.plan is None:
        # Legacy queue metadata remains readable until the compatibility
        # surface is retired, but only typed contracts create PR-7 physical
        # plans and bindings.
        updated_instructions, updated_queues = annotate_legacy_cross_handler_handoff(
            instructions,
            queues,
            spec,
        )
        return updated_instructions, updated_queues, ()

    updates: dict[int, dict[str, Any]] = {}
    bindings: list[DataflowHandoffQueueBinding] = []
    for queue in queues.values():
        for instruction in queue:
            stage_id = instruction.attrs.get("stage_id")
            if stage_id not in {spec.producer_stage_id, spec.consumer_stage_id}:
                continue
            updates[instruction.instruction_id] = {
                **instruction.attrs,
                "cross_handler_handoff_stage_count": 0,
                "cross_handler_handoff_plan_fingerprint": spec.plan.fingerprint,
                "cross_handler_handoff_state": ("unpaired" if spec.plan.enabled else "disabled"),
            }

        terminal_producers = []
        first_consumers_by_task: dict[int, Instruction] = {}
        task_order: list[int] = []
        for index, instruction in enumerate(queue):
            stage_id = instruction.attrs.get("stage_id")
            if (
                stage_id == spec.consumer_stage_id
                and instruction.task_id is not None
                and instruction.task_id not in first_consumers_by_task
            ):
                first_consumers_by_task[instruction.task_id] = instruction
                task_order.append(instruction.task_id)
            if (
                stage_id == spec.producer_stage_id
                and instruction.task_id is not None
                and not any(
                    later.task_id == instruction.task_id and later.attrs.get("stage_id") == spec.producer_stage_id
                    for later in queue[index + 1 :]
                )
            ):
                terminal_producers.append(instruction)

        task_position = {task_id: index for index, task_id in enumerate(task_order)}
        queue_active_binding = 0
        for producer in terminal_producers:
            assert producer.sm_id is not None and producer.task_id is not None
            producer_position = task_position.get(producer.task_id)
            target_position = None if producer_position is None else producer_position + spec.lookahead_distance
            consumer = (
                None
                if target_position is None or target_position >= len(task_order)
                else first_consumers_by_task[task_order[target_position]]
            )
            state = "disabled" if not spec.plan.enabled else "tail" if consumer is None else "active"
            binding_id = len(bindings)
            arena_slot = queue_active_binding % spec.lookahead_distance if state == "active" else None
            if state == "active":
                queue_active_binding += 1
            active_consumer = consumer if state == "active" else None
            consumer_range = None if active_consumer is None else active_consumer.task_range
            bindings.append(
                DataflowHandoffQueueBinding(
                    binding_id=binding_id,
                    plan_fingerprint=spec.plan.fingerprint,
                    sm_id=producer.sm_id,
                    producer_instruction_id=producer.instruction_id,
                    consumer_instruction_id=(None if active_consumer is None else active_consumer.instruction_id),
                    producer_task_id=producer.task_id,
                    consumer_task_id=(None if active_consumer is None else active_consumer.task_id),
                    consumer_task_coords=(None if active_consumer is None else active_consumer.task_coords),
                    consumer_range_begin=(None if consumer_range is None else consumer_range.begin),
                    consumer_range_end=(None if consumer_range is None else consumer_range.end),
                    stage_count=(spec.stage_count if state == "active" else 0),
                    arena_slot=arena_slot,
                    state=state,
                )
            )
            producer_attrs = updates[producer.instruction_id]
            producer_attrs.update(
                {
                    "cross_handler_handoff_role": "producer",
                    "cross_handler_handoff_state": state,
                    "cross_handler_handoff_binding_id": binding_id,
                    "cross_handler_handoff_stage_count": (spec.stage_count if state == "active" else 0),
                    "cross_handler_handoff_arena_slot": arena_slot,
                }
            )
            if active_consumer is not None:
                producer_attrs["cross_handler_handoff_peer_instruction_id"] = active_consumer.instruction_id
                consumer_attrs = updates[active_consumer.instruction_id]
                if consumer_attrs.get("cross_handler_handoff_role") is not None:
                    raise ValueError(
                        f"cross_handler_handoff queue assigns multiple producers to consumer instruction {active_consumer.instruction_id}"
                    )
                consumer_attrs.update(
                    {
                        "cross_handler_handoff_role": "consumer",
                        "cross_handler_handoff_state": "active",
                        "cross_handler_handoff_binding_id": binding_id,
                        "cross_handler_handoff_stage_count": spec.stage_count,
                        "cross_handler_handoff_arena_slot": arena_slot,
                        "cross_handler_handoff_peer_instruction_id": (producer.instruction_id),
                    }
                )

    def updated(instruction: Instruction) -> Instruction:
        attrs = updates.get(instruction.instruction_id)
        return instruction if attrs is None else replace(instruction, attrs=attrs)

    updated_by_id = {instruction.instruction_id: updated(instruction) for instruction in instructions}
    return (
        [updated_by_id[instruction.instruction_id] for instruction in instructions],
        {sm_id: [updated_by_id[instruction.instruction_id] for instruction in queue] for sm_id, queue in queues.items()},
        tuple(bindings),
    )


def annotate_legacy_cross_handler_handoff(
    instructions: list[Instruction],
    queues: dict[int, list[Instruction]],
    spec: CrossHandlerHandoffSpec,
) -> tuple[list[Instruction], dict[int, list[Instruction]]]:
    """Annotate compatibility-only handoff metadata without a physical plan."""

    updates: dict[int, dict[str, Any]] = {}
    for queue in queues.values():
        for instruction in queue:
            stage_id = instruction.attrs.get("stage_id")
            if stage_id not in {spec.producer_stage_id, spec.consumer_stage_id}:
                continue
            updates[instruction.instruction_id] = {
                **instruction.attrs,
                "cross_handler_handoff_stage_count": 0,
                "cross_handler_handoff_target_task_coords": tuple(0 for _ in instruction.task_coords),
            }
        for index, producer in enumerate(queue):
            if producer.attrs.get("stage_id") != spec.producer_stage_id:
                continue
            if any(
                later.task_id == producer.task_id and later.attrs.get("stage_id") == spec.producer_stage_id for later in queue[index + 1 :]
            ):
                continue
            consumer = next(
                (
                    later
                    for later in queue[index + 1 :]
                    if later.attrs.get("stage_id") == spec.consumer_stage_id and later.task_id != producer.task_id
                ),
                None,
            )
            if consumer is None:
                continue
            producer_attrs = updates[producer.instruction_id]
            producer_attrs.update(
                {
                    "cross_handler_handoff_role": "producer",
                    "cross_handler_handoff_stage_count": spec.stage_count,
                    "cross_handler_handoff_target_task_coords": consumer.task_coords,
                    "cross_handler_handoff_peer_instruction_id": consumer.instruction_id,
                }
            )
            updates[consumer.instruction_id].update(
                {
                    "cross_handler_handoff_role": "consumer",
                    "cross_handler_handoff_stage_count": spec.stage_count,
                    "cross_handler_handoff_target_task_coords": consumer.task_coords,
                    "cross_handler_handoff_peer_instruction_id": producer.instruction_id,
                }
            )
    updated_by_id = {
        instruction.instruction_id: (
            instruction if instruction.instruction_id not in updates else replace(instruction, attrs=updates[instruction.instruction_id])
        )
        for instruction in instructions
    }
    return (
        [updated_by_id[item.instruction_id] for item in instructions],
        {sm_id: [updated_by_id[item.instruction_id] for item in queue] for sm_id, queue in queues.items()},
    )


def split_task_range(task_range: TaskRange, tile_size: int | None) -> tuple[TaskRange, ...]:
    if tile_size is None or task_range.length <= tile_size:
        return (task_range,)
    return tuple(
        TaskRange(axis=task_range.axis, begin=begin, end=min(begin + tile_size, task_range.end))
        for begin in range(task_range.begin, task_range.end, tile_size)
    )


def weighted_stage_graph_cluster_assignment(
    topology: GPUTopology,
    task_weights: tuple[float, ...],
) -> tuple[int, ...]:
    cluster_loads = [0.0 for _ in range(topology.cluster_count)]
    assignment = [-1 for _ in task_weights]
    for task_id in sorted(range(len(task_weights)), key=lambda idx: (-task_weights[idx], idx)):
        cluster_id = min(
            range(topology.cluster_count),
            key=lambda idx: (
                cluster_loads[idx] / max(1, len(get_cluster_sms(topology, idx))),
                cluster_loads[idx],
                idx,
            ),
        )
        assignment[task_id] = cluster_id
        cluster_loads[cluster_id] += task_weights[task_id]
    return tuple(assignment)


def schedule_linear_stage_graph(
    program: DataflowProgram,
    *,
    topology: GPUTopology,
    range_lengths: dict[Any, int | Sequence[int]],
    block_size: int,
    task_extents: Sequence[int] | None,
    include_exit: bool,
    force_hbm_comms: bool,
    task_coord_overrides: Sequence[Sequence[int] | int] | None = None,
    stage_graph_task_weights: Sequence[int | float] | None = None,
    stage_graph_cluster_assignment: Sequence[int] | None = None,
    range_resource_budget_bytes: int | None = None,
    target_capabilities: TargetCapabilitySnapshot | None = None,
    cross_handler_handoff_max_shared_memory_bytes: int | None = None,
) -> InstructionPlan:
    map1_stage, reshared_stage, map2_stage, finalize_stage = linear_reshared_stages(program)
    cross_handler_handoff = cross_handler_handoff_spec(
        program,
        max_shared_memory_bytes=(
            cross_handler_handoff_max_shared_memory_bytes
            if cross_handler_handoff_max_shared_memory_bytes is not None
            else None
            if target_capabilities is None
            else target_capabilities.max_dynamic_shared_memory
        ),
    )
    if topology.sm_count % topology.cluster_size != 0:
        raise ValueError(
            "Dataflow reshared graph scheduling requires sm_count to be divisible by cluster_size, "
            f"got sm_count={topology.sm_count}, cluster_size={topology.cluster_size}"
        )

    map1_axis = map1_stage.range_axis
    map2_axis = map2_stage.range_axis
    map1_lengths = stage_graph_range_lengths(map1_stage, range_lengths)
    map2_lengths = stage_graph_range_lengths(map2_stage, range_lengths)
    map1_range_request = stage_graph_range_request(map1_stage)
    map2_range_request = stage_graph_range_request(map2_stage)
    map1_legacy_range_tile_size = stage_graph_legacy_range_tile_size(map1_stage)
    map2_legacy_range_tile_size = stage_graph_legacy_range_tile_size(map2_stage)
    if len(map1_lengths) != len(map2_lengths):
        raise ValueError(
            f"Dataflow reshared graph map stages must have the same task count: map1 has {len(map1_lengths)}, map2 has {len(map2_lengths)}"
        )
    task_count = len(map1_lengths)
    normalized_task_extents = normalize_task_extents(task_extents, task_count)
    task_coord_overrides_normalized = normalize_task_coord_overrides(
        task_coord_overrides,
        task_count=task_count,
        task_rank=len(program.task_domain.axes),
    )
    normalized_cluster_assignment = normalize_stage_graph_cluster_assignment(
        stage_graph_cluster_assignment,
        task_count=task_count,
        cluster_count=topology.cluster_count,
    )
    normalized_task_weights = normalize_stage_graph_task_weights(
        stage_graph_task_weights,
        task_count=task_count,
    )
    if normalized_cluster_assignment is None and normalized_task_weights is not None:
        normalized_cluster_assignment = weighted_stage_graph_cluster_assignment(
            topology,
            normalized_task_weights,
        )

    next_instruction_id = 0
    next_slot_id = 0
    all_instructions: list[Instruction] = []
    queues: dict[int, list[Instruction]] = {sm_id: [] for sm_id in range(topology.sm_count)}
    slots: list[SlotPlan] = []
    comms: list[CommPlan] = []
    range_coarsening_plans: dict[tuple[int, str], tuple[int, DataflowRangeCoarseningPlan]] = {}
    next_flag_epoch = 1

    def emit(instruction: Instruction) -> Instruction:
        all_instructions.append(instruction)
        if instruction.sm_id is not None:
            queues[instruction.sm_id].append(instruction)
        return instruction

    assert map1_axis is not None
    assert map2_axis is not None
    assert map1_stage.output_type is not None
    assert reshared_stage.physical_output_type is not None
    map2_outputless = map2_stage.output_type is None
    if map2_outputless:
        assert map2_stage.call is not None
        if (
            finalize_stage is not None
            or operator_physical_contract(map2_stage.call.operator.attrs).output_slot != DATAFLOW_OUTPUT_SLOT_NONE
        ):
            raise ValueError("an outputless stage-graph map must be the typed terminal map")
    transport_request = reshared_stage.attrs.get(DATAFLOW_TRANSPORT_CONTRACT_ATTR)
    reshared_transport_plan = None
    if transport_request is None:
        reshared_policy = str(reshared_stage.attrs.get("policy", "hbm_all_gather"))
    else:
        if "policy" in reshared_stage.attrs:
            raise ValueError(f"Dataflow stage {reshared_stage.name!r} cannot combine transport_contract with legacy policy")
        if not isinstance(transport_request, DataflowResharedTransportRequest):
            raise TypeError(f"Dataflow stage {reshared_stage.name!r} has an unnormalized transport contract")
        physical_slot_bytes = estimate_range_output_bytes_per_tile(
            reshared_stage.physical_output_type,
            output_tile_arity=1,
        )
        if physical_slot_bytes <= 0:
            raise ValueError("Dataflow reshared transport requires a concrete physical slot layout")
        receive_stage_count = None
        map2_pipeline_request = map2_stage.attrs.get(DATAFLOW_PIPELINE_CONTRACT_ATTR)
        if isinstance(map2_pipeline_request, DataflowPipelineRequest):
            pipeline_plan = plan_pipeline_dataflow(map2_pipeline_request)
            matching_buffers = tuple(
                transfer.destination_buffer_index
                for transfer in map2_pipeline_request.transfers
                if transfer.bytes_per_stage
                == physical_slot_bytes // (transport_request.logical_output_arity // transport_request.physical_output_arity)
            )
            if len(set(matching_buffers)) == 1:
                receive_stage_count = pipeline_plan.implementation_requirements.buffer_versions[matching_buffers[0]]
        reshared_transport_plan = plan_reshared_transport(
            transport_request,
            cluster_size=topology.cluster_size,
            physical_slot_bytes=physical_slot_bytes,
            target_capabilities=target_capabilities,
            receive_stage_count=receive_stage_count,
            available_threads=int(
                map2_stage.call.operator.attrs.get("threads", 1)  # type: ignore[union-attr]
            ),
        )
        dataflow_implementation_registry().require_contract_compatible(
            reshared_transport_plan.implementation_id,
            transport_request,
            selected_explicitly=transport_request.explicit,
        )
        selected_family = reshared_transport_plan.family
        reshared_policy = {
            DATAFLOW_TRANSPORT_HBM: "hbm_all_gather",
            DATAFLOW_TRANSPORT_ALL_GATHER: "cluster_shared_all_gather",
            DATAFLOW_TRANSPORT_STREAMED: "cluster_shared_pull_ring",
        }[selected_family]
        if selected_family == DATAFLOW_TRANSPORT_HBM:
            force_hbm_comms = True
    if reshared_policy not in {
        "hbm_all_gather",
        "cluster_shared_all_gather",
        "cluster_shared_pull_ring",
    }:
        raise ValueError(
            "Dataflow reshared graph scheduling supports policy='hbm_all_gather' "
            "policy='cluster_shared_all_gather', or policy='cluster_shared_pull_ring', got "
            f"{reshared_policy!r}"
        )
    cluster_shared_all_gather = reshared_policy == "cluster_shared_all_gather"
    cluster_shared_pull_ring = reshared_policy == "cluster_shared_pull_ring"
    if (cluster_shared_all_gather or cluster_shared_pull_ring) and force_hbm_comms:
        raise ValueError(f"Dataflow policy={reshared_policy!r} requires force_hbm_comms=False")
    cluster_shared_barrier_phases: dict[tuple[int, int], int] = {}
    publish_sync_required = cluster_shared_pull_ring and (reshared_transport_plan is None or reshared_transport_plan.publish_sync_required)
    release_sync_required = cluster_shared_pull_ring and (reshared_transport_plan is None or reshared_transport_plan.release_sync_required)

    for task_id, (map1_length, map2_length) in enumerate(zip(map1_lengths, map2_lengths)):
        coords = (
            task_coord_overrides_normalized[task_id]
            if task_coord_overrides_normalized is not None
            else task_coords(task_id, normalized_task_extents)
        )
        cluster_id = (
            normalized_cluster_assignment[task_id] if normalized_cluster_assignment is not None else task_id % topology.cluster_count
        )
        cluster_sms = get_cluster_sms(topology, cluster_id)
        if len(cluster_sms) != topology.cluster_size:
            raise ValueError(
                f"Dataflow reshared graph found incomplete cluster {cluster_id}: "
                f"expected {topology.cluster_size} SMs, got {len(cluster_sms)}"
            )
        map1_ranges = cluster_rank_ranges(length=map1_length, axis=map1_axis, topology=topology)
        map2_ranges = cluster_rank_ranges(length=map2_length, axis=map2_axis, topology=topology)
        map1_range_plans_by_rank = (
            tuple(
                stage_graph_range_plan(
                    map1_stage,
                    map1_range_request,
                    logical_range_extent=task_range.length,
                    resource_budget_bytes=range_resource_budget_bytes,
                )
                for task_range in map1_ranges
            )
            if map1_range_request is not None
            else None
        )
        map2_range_plans_by_rank = (
            tuple(
                stage_graph_range_plan(
                    map2_stage,
                    map2_range_request,
                    logical_range_extent=task_range.length,
                    resource_budget_bytes=range_resource_budget_bytes,
                )
                for task_range in map2_ranges
            )
            if map2_range_request is not None
            else None
        )
        for stage, plans in (
            (map1_stage, map1_range_plans_by_rank),
            (map2_stage, map2_range_plans_by_rank),
        ):
            for range_plan in plans or ():
                range_coarsening_plans.setdefault(
                    (stage.stage_id, range_plan.fingerprint),
                    (stage.stage_id, range_plan),
                )
        map1_range_tiles_by_rank = tuple(
            split_task_range(
                task_range,
                (
                    map1_range_plans_by_rank[rank].selected_handler_range_extent
                    if map1_range_plans_by_rank is not None
                    else map1_legacy_range_tile_size
                ),
            )
            for rank, task_range in enumerate(map1_ranges)
        )
        map2_range_tiles_by_rank = tuple(
            split_task_range(
                task_range,
                (
                    map2_range_plans_by_rank[rank].selected_handler_range_extent
                    if map2_range_plans_by_rank is not None
                    else map2_legacy_range_tile_size
                ),
            )
            for rank, task_range in enumerate(map2_ranges)
        )
        map1_tile_offsets_by_rank: list[int] = []
        map1_output_arity = 0
        for rank_tiles in map1_range_tiles_by_rank:
            map1_tile_offsets_by_rank.append(map1_output_arity)
            map1_output_arity += len(rank_tiles)
        if reshared_stage.output_arity != map1_output_arity:
            raise ValueError(
                "Dataflow reshared stage output_arity must match the number of map1 output tiles "
                "for reshared all-gather v1: "
                f"output_arity={reshared_stage.output_arity}, map1_output_tiles={map1_output_arity}"
            )

        map1_instructions_by_rank: list[list[Instruction]] = []
        map1_slots_by_rank: list[list[int]] = []
        for rank, sm_id in enumerate(cluster_sms):
            rank_instructions: list[Instruction] = []
            rank_slots: list[int] = []
            for tile_index, task_range in enumerate(map1_range_tiles_by_rank[rank]):
                range_plan = None if map1_range_plans_by_rank is None else map1_range_plans_by_rank[rank]
                output_mapping = None if range_plan is None else range_plan.output_mappings[tile_index]
                slot_id = next_slot_id
                next_slot_id += 1
                instruction = emit(
                    Instruction(
                        instruction_id=next_instruction_id,
                        opcode=DataflowOpcode.MAP,
                        operator_name=map1_stage.call.name,  # type: ignore[union-attr]
                        task_id=task_id,
                        task_coords=coords,
                        sm_id=sm_id,
                        task_range=task_range,
                        output_slot=slot_id,
                        attrs={
                            "stage_id": map1_stage.stage_id,
                            "cluster_rank": rank,
                            "range_tile_index": tile_index,
                            **(
                                {}
                                if range_plan is None or output_mapping is None
                                else {
                                    "range_coarsening_plan_fingerprint": range_plan.fingerprint,
                                    "range_output_mapping": output_mapping.to_dict(),
                                    "range_origin": map1_ranges[rank].begin,
                                }
                            ),
                        },
                    )
                )
                next_instruction_id += 1
                rank_instructions.append(instruction)
                rank_slots.append(slot_id)
                physical_tile_index = map1_tile_offsets_by_rank[rank] + tile_index
                shared_storage_id = (
                    tile_index if cluster_shared_pull_ring else (physical_tile_index if cluster_shared_all_gather else slot_id)
                )
                slots.append(
                    SlotPlan(
                        slot_id=slot_id,
                        task_id=task_id,
                        intermediate_type=map1_stage.output_type,
                        role=f"{map1_stage.name}_output",
                        producer_instruction_id=instruction.instruction_id,
                        shared_storage_id=shared_storage_id,
                        global_storage_id=shared_storage_id,
                    )
                )
            map1_instructions_by_rank.append(rank_instructions)
            map1_slots_by_rank.append(rank_slots)

        reshared_instructions: list[Instruction] = []
        reshared_slots_by_rank: list[tuple[int, ...]] = []
        for consumer_rank, consumer_sm in enumerate(cluster_sms):
            if cluster_shared_pull_ring:
                # Every consumer receives producer-local slot addresses. The
                # transport binding maps logical accesses to peer ranks.
                consumer_slots = list(map1_slots_by_rank[consumer_rank])
            else:
                consumer_slots = []
                for producer_rank, producer_slots in enumerate(map1_slots_by_rank):
                    for producer_tile_index, producer_slot_id in enumerate(producer_slots):
                        target_slot_id = next_slot_id
                        next_slot_id += 1
                        is_local = producer_rank == consumer_rank
                        physical_tile_index = map1_tile_offsets_by_rank[producer_rank] + producer_tile_index
                        consumer_slots.append(target_slot_id)
                        slots.append(
                            SlotPlan(
                                slot_id=target_slot_id,
                                task_id=task_id,
                                intermediate_type=reshared_stage.physical_output_type,
                                role="reshared_full",
                                producer_instruction_id=(
                                    map1_instructions_by_rank[producer_rank][producer_tile_index].instruction_id if is_local else None
                                ),
                                shared_storage_id=(
                                    physical_tile_index if cluster_shared_all_gather else (producer_slot_id if is_local else None)
                                ),
                                global_storage_id=(physical_tile_index if cluster_shared_all_gather else producer_slot_id),
                                barrier_storage_id=physical_tile_index if cluster_shared_all_gather else None,
                            )
                        )
            reshared_instruction = emit(
                Instruction(
                    instruction_id=next_instruction_id,
                    opcode=DataflowOpcode.RESHARED,
                    operator_name=reshared_stage.name,
                    task_id=task_id,
                    task_coords=coords,
                    sm_id=consumer_sm,
                    input_slots=tuple(consumer_slots),
                    attrs={
                        "stage_id": reshared_stage.stage_id,
                        "cluster_rank": consumer_rank,
                        **dict(reshared_stage.attrs),
                    },
                )
            )
            next_instruction_id += 1
            reshared_instructions.append(reshared_instruction)
            reshared_slots_by_rank.append(tuple(consumer_slots))

        if publish_sync_required:
            for rank, sm_id in enumerate(cluster_sms):
                emit(
                    Instruction(
                        instruction_id=next_instruction_id,
                        opcode=DataflowOpcode.CLUSTER_SYNC,
                        operator_name="reshared_stream_publish",
                        task_id=task_id,
                        task_coords=coords,
                        sm_id=sm_id,
                        attrs={
                            "cluster_rank": rank,
                            "transport_family": DATAFLOW_TRANSPORT_STREAMED,
                            "transport_sync": "publish",
                            **(
                                {}
                                if reshared_transport_plan is None
                                else {"transport_plan_fingerprint": (reshared_transport_plan.fingerprint)}
                            ),
                        },
                    )
                )
                next_instruction_id += 1
        elif not cluster_shared_pull_ring:
            for consumer_rank, consumer_sm in enumerate(cluster_sms):
                target_instruction = reshared_instructions[consumer_rank]
                for producer_rank, producer_sm in enumerate(cluster_sms):
                    if producer_rank == consumer_rank:
                        continue
                    producer_slots = map1_slots_by_rank[producer_rank]
                    for producer_tile_index, source_slot_id in enumerate(producer_slots):
                        source_instruction = map1_instructions_by_rank[producer_rank][producer_tile_index]
                        producer_slot_offset = sum(len(slots_for_rank) for slots_for_rank in map1_slots_by_rank[:producer_rank])
                        target_slot_id = reshared_slots_by_rank[consumer_rank][producer_slot_offset + producer_tile_index]
                        physical_tile_index = map1_tile_offsets_by_rank[producer_rank] + producer_tile_index
                        same_cluster = topology.same_cluster(producer_sm, consumer_sm) and (
                            cluster_shared_all_gather or not force_hbm_comms
                        )
                        if cluster_shared_all_gather and not same_cluster:
                            raise ValueError(
                                "Dataflow policy='cluster_shared_all_gather' requires all reshared "
                                "producer/consumer pairs to be placed in the same cluster"
                            )
                        send_kind = DataflowCommKind.CLUSTER_SEND if same_cluster else DataflowCommKind.HBM_SEND
                        recv_kind = DataflowCommKind.CLUSTER_RECV if same_cluster else DataflowCommKind.HBM_RECV
                        flag_epoch = next_flag_epoch
                        next_flag_epoch += 1
                        barrier_phase = 0
                        if cluster_shared_all_gather and recv_kind is DataflowCommKind.CLUSTER_RECV:
                            phase_key = (consumer_sm, physical_tile_index)
                            barrier_phase = cluster_shared_barrier_phases.get(phase_key, 0)
                            cluster_shared_barrier_phases[phase_key] = barrier_phase ^ 1
                        comms.append(
                            CommPlan(
                                source_instruction_id=source_instruction.instruction_id,
                                target_instruction_id=target_instruction.instruction_id,
                                source_slot_id=source_slot_id,
                                target_slot_id=target_slot_id,
                                producer_sm=producer_sm,
                                consumer_sm=consumer_sm,
                                kind=send_kind,
                                dispatch_instruction_id=source_instruction.instruction_id,
                                peer_cta_rank=topology.cluster_rank(consumer_sm) if same_cluster else None,
                                flag_epoch=flag_epoch,
                                barrier_phase=barrier_phase,
                            )
                        )
                        comms.append(
                            CommPlan(
                                source_instruction_id=source_instruction.instruction_id,
                                target_instruction_id=target_instruction.instruction_id,
                                source_slot_id=source_slot_id,
                                target_slot_id=target_slot_id,
                                producer_sm=producer_sm,
                                consumer_sm=consumer_sm,
                                kind=recv_kind,
                                dispatch_instruction_id=target_instruction.instruction_id,
                                peer_cta_rank=(topology.cluster_rank(producer_sm) if same_cluster else None),
                                flag_epoch=flag_epoch,
                                barrier_phase=barrier_phase,
                            )
                        )

        map2_shared_storage_id = None if map2_outputless else next_slot_id
        map2_tile_wave_count = max(len(rank_tiles) for rank_tiles in map2_range_tiles_by_rank)
        for tile_index in range(map2_tile_wave_count):
            for rank, sm_id in enumerate(cluster_sms):
                if tile_index >= len(map2_range_tiles_by_rank[rank]):
                    continue
                task_range = map2_range_tiles_by_rank[rank][tile_index]
                range_plan = None if map2_range_plans_by_rank is None else map2_range_plans_by_rank[rank]
                output_mapping = None if range_plan is None else range_plan.output_mappings[tile_index]
                slot_id = None if map2_outputless else next_slot_id
                if slot_id is not None:
                    next_slot_id += 1
                instruction = emit(
                    Instruction(
                        instruction_id=next_instruction_id,
                        opcode=DataflowOpcode.MAP,
                        operator_name=map2_stage.call.name,  # type: ignore[union-attr]
                        task_id=task_id,
                        task_coords=coords,
                        sm_id=sm_id,
                        task_range=task_range,
                        input_slots=reshared_slots_by_rank[rank],
                        output_slot=slot_id,
                        attrs={
                            "stage_id": map2_stage.stage_id,
                            "cluster_rank": rank,
                            "range_tile_index": tile_index,
                            **(
                                {}
                                if range_plan is None or output_mapping is None
                                else {
                                    "range_coarsening_plan_fingerprint": range_plan.fingerprint,
                                    "range_output_mapping": output_mapping.to_dict(),
                                    "range_origin": map2_ranges[rank].begin,
                                }
                            ),
                        },
                    )
                )
                next_instruction_id += 1
                if slot_id is not None:
                    assert map2_stage.output_type is not None
                    slots.append(
                        SlotPlan(
                            slot_id=slot_id,
                            task_id=task_id,
                            intermediate_type=map2_stage.output_type,
                            role=f"{map2_stage.name}_output",
                            producer_instruction_id=instruction.instruction_id,
                            shared_storage_id=map2_shared_storage_id,
                        )
                    )
                if finalize_stage is not None:
                    assert slot_id is not None
                    emit(
                        Instruction(
                            instruction_id=next_instruction_id,
                            opcode=DataflowOpcode.FINALIZE,
                            operator_name=finalize_stage.call.name,  # type: ignore[union-attr]
                            task_id=task_id,
                            task_coords=coords,
                            sm_id=sm_id,
                            task_range=task_range,
                            input_slots=(slot_id,),
                            attrs={
                                "stage_id": finalize_stage.stage_id,
                                "cluster_rank": rank,
                                "range_tile_index": tile_index,
                            },
                        )
                    )
                    next_instruction_id += 1

            if cluster_shared_pull_ring and tile_index + 1 < map2_tile_wave_count:
                for rank, sm_id in enumerate(cluster_sms):
                    emit(
                        Instruction(
                            instruction_id=next_instruction_id,
                            opcode=DataflowOpcode.CLUSTER_SYNC,
                            operator_name="reshared_stream_advance",
                            task_id=task_id,
                            task_coords=coords,
                            sm_id=sm_id,
                            attrs={
                                "cluster_rank": rank,
                                "range_tile_index": tile_index,
                                "transport_family": DATAFLOW_TRANSPORT_STREAMED,
                                "transport_sync": "consumer_advance",
                            },
                        )
                    )
                    next_instruction_id += 1

        if release_sync_required:
            for rank, sm_id in enumerate(get_cluster_sms(topology, cluster_id)):
                emit(
                    Instruction(
                        instruction_id=next_instruction_id,
                        opcode=DataflowOpcode.CLUSTER_SYNC,
                        operator_name="reshared_stream_release",
                        task_id=task_id,
                        task_coords=coords,
                        sm_id=sm_id,
                        attrs={
                            "cluster_rank": rank,
                            "transport_family": DATAFLOW_TRANSPORT_STREAMED,
                            "transport_sync": "release",
                            **(
                                {}
                                if reshared_transport_plan is None
                                else {"transport_plan_fingerprint": (reshared_transport_plan.fingerprint)}
                            ),
                        },
                    )
                )
                next_instruction_id += 1

    if include_exit:
        for sm_id in range(topology.sm_count):
            emit(
                Instruction(
                    instruction_id=next_instruction_id,
                    opcode=DataflowOpcode.EXIT,
                    operator_name="exit",
                    task_id=None,
                    sm_id=sm_id,
                )
            )
            next_instruction_id += 1

    annotated_instructions, annotated_queues, handoff_bindings = annotate_cross_handler_handoff(
        all_instructions,
        queues,
        cross_handler_handoff,
    )
    if (
        cross_handler_handoff is not None
        and cross_handler_handoff.plan is not None
        and cross_handler_handoff.plan.enabled
        and not any(binding.state == "active" for binding in handoff_bindings)
    ):
        fallback_plan = fallback_cross_handler_handoff_plan(
            cross_handler_handoff.plan,
            reason="handoff_queue_has_no_active_edges",
        )
        cross_handler_handoff = replace(
            cross_handler_handoff,
            stage_count=0,
            plan=fallback_plan,
        )
        annotated_instructions, annotated_queues, handoff_bindings = annotate_cross_handler_handoff(
            all_instructions,
            queues,
            cross_handler_handoff,
        )
    all_instructions, queues = annotated_instructions, annotated_queues

    return InstructionPlan(
        topology=topology,
        block_size=block_size,
        range_axis=map1_axis,
        scheduler_policy="stage_graph",
        reduce_strategy="none",
        task_extents=normalized_task_extents,
        task_range_lengths=map1_lengths,
        instructions=tuple(all_instructions),
        queues={sm_id: tuple(items) for sm_id, items in queues.items()},
        slots=tuple(slots),
        comms=tuple(comms),
        scheduler_config=current_scheduler_config(),
        range_coarsening_plans=tuple(range_coarsening_plans.values()),
        reshared_transport_plans=(() if reshared_transport_plan is None else ((reshared_stage.stage_id, reshared_transport_plan),)),
        cross_handler_handoff_plans=(
            () if cross_handler_handoff is None or cross_handler_handoff.plan is None else (cross_handler_handoff.plan,)
        ),
        cross_handler_handoff_bindings=handoff_bindings,
        range_resource_budget_bytes=range_resource_budget_bytes,
        target_capabilities=target_capabilities,
    )


def streaming_tree_consumer_side(value: str | None = None) -> str:
    raw_value = config_string("streaming_tree_consumer", "left") if value is None else value
    normalized = str(raw_value).strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"Dataflow streaming_tree_consumer must be 'left' or 'right', got {raw_value!r}")
    return normalized


def get_cluster_sms(topology: GPUTopology, cluster_id: int) -> tuple[int, ...]:
    begin = cluster_id * topology.cluster_size
    end = min(begin + topology.cluster_size, topology.sm_count)
    return tuple(range(begin, end))


def cluster_local_sm_id(
    topology: GPUTopology,
    *,
    task_id: int,
    range_index: int,
) -> int:
    cluster_id = task_id % topology.cluster_count
    sms = get_cluster_sms(topology, cluster_id)
    if not sms:
        raise ValueError(f"Dataflow scheduler found empty cluster {cluster_id} in topology {topology!r}")
    return sms[range_index % len(sms)]


def load_balanced_cluster_tasks(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    cluster_loads = [0 for _ in range(topology.cluster_count)]
    cluster_tasks: list[list[int]] = [[] for _ in range(topology.cluster_count)]
    for task_id in sorted(range(len(task_range_lengths)), key=lambda index: (-task_range_lengths[index], index)):
        cluster_id = min(
            range(topology.cluster_count),
            key=lambda index: (cluster_loads[index] / max(1, len(get_cluster_sms(topology, index))), index),
        )
        cluster_tasks[cluster_id].append(task_id)
        cluster_loads[cluster_id] += task_range_lengths[task_id]
    return tuple(tuple(tasks) for tasks in cluster_tasks)


def append_streaming_range(
    ranges: list[tuple[int, TaskRange]],
    *,
    sm_id: int,
    axis: Any,
    begin: int,
    end: int,
) -> None:
    if begin >= end:
        return
    task_range = TaskRange(axis=axis, begin=begin, end=end)
    if ranges and ranges[-1][0] == sm_id and ranges[-1][1].end == begin:
        previous = ranges[-1][1]
        ranges[-1] = (sm_id, TaskRange(axis=axis, begin=previous.begin, end=end))
        return
    ranges.append((sm_id, task_range))


def offset_task_range(task_range: TaskRange, offset: int) -> TaskRange:
    if offset == 0:
        return task_range
    return TaskRange(
        axis=task_range.axis,
        begin=task_range.begin + offset,
        end=task_range.end + offset,
    )


def estimated_streaming_comm_cost_us(
    topology: GPUTopology,
    producer_sm: int,
    consumer_sm: int,
    *,
    force_hbm_comms: bool = False,
) -> float:
    cost_model = current_scheduler_config().cost_model
    return estimate_communication_cost_us(
        cost_model,
        same_sm=producer_sm == consumer_sm,
        same_cluster=topology.same_cluster(producer_sm, consumer_sm),
        force_hbm=force_hbm_comms,
    )


def estimated_streaming_iter_cost_us(task_range: TaskRange | None, block_size: int) -> float:
    if task_range is None:
        return 0.0
    cost_model = current_scheduler_config().cost_model
    tail_penalty = max(0.0, config_float("iter_tail_penalty_us", 0.0))
    tail_penalty_min_blocks = max(1, config_int("iter_tail_penalty_min_blocks", 8))
    return estimate_iter_cost_us(
        cost_model,
        range_length=task_range.length,
        block_size=block_size,
        tail_penalty_us=tail_penalty,
        tail_penalty_min_blocks=tail_penalty_min_blocks,
    )


def estimated_streaming_reduce_cost_us(input_count: int) -> float:
    cost = estimate_reduce_cost_us(
        current_scheduler_config().cost_model,
        input_count=input_count,
    )
    return cost * max(1, input_count - 1)


def estimated_streaming_finalize_cost_us() -> float:
    return estimate_finalize_cost_us(current_scheduler_config().cost_model)


def replace_streaming_chunk_sm(chunk: StreamingChunk, sm_id: int) -> StreamingChunk:
    return StreamingChunk(
        task_id=chunk.task_id,
        sm_id=sm_id,
        task_range=chunk.task_range,
        part_index=chunk.part_index,
        part_count=chunk.part_count,
    )


def streaming_chunk_swap_score(
    topology: GPUTopology,
    chunks_by_task: Sequence[Sequence[StreamingChunk]],
    *,
    block_size: int,
) -> tuple[float, float, float, int]:
    mix_penalty_us = config_float("chunk_swap_mix_penalty_us", 8.0)
    sm_costs: list[float] = []
    max_queue_len = 0
    for sm_id in range(topology.sm_count):
        chunks = [chunk for task_chunks in chunks_by_task for chunk in task_chunks if chunk.sm_id == sm_id]
        if not chunks:
            continue
        max_queue_len = max(max_queue_len, len(chunks))
        iter_cost = sum(estimated_streaming_iter_cost_us(chunk.task_range, block_size) for chunk in chunks)
        sm_costs.append(iter_cost + mix_penalty_us * max(0, len(chunks) - 1))
    if not sm_costs:
        return (0.0, 0.0, 0.0, 0)
    sm_costs.sort()
    p95_index = min(len(sm_costs) - 1, math.ceil(0.95 * len(sm_costs)) - 1)
    return (
        sm_costs[-1],
        sum(cost * cost for cost in sm_costs),
        sm_costs[p95_index],
        max_queue_len,
    )


def swap_streaming_chunk_sms(
    chunks_by_task: Sequence[Sequence[StreamingChunk]],
    left: StreamingChunk,
    right: StreamingChunk,
) -> tuple[tuple[StreamingChunk, ...], ...]:
    swapped: list[tuple[StreamingChunk, ...]] = []
    for task_chunks in chunks_by_task:
        next_task_chunks: list[StreamingChunk] = []
        for chunk in task_chunks:
            if chunk is left:
                next_task_chunks.append(replace_streaming_chunk_sm(chunk, right.sm_id))
            elif chunk is right:
                next_task_chunks.append(replace_streaming_chunk_sm(chunk, left.sm_id))
            else:
                next_task_chunks.append(chunk)
        swapped.append(tuple(next_task_chunks))
    return tuple(swapped)


def task_has_duplicate_sm(chunks_by_task: Sequence[Sequence[StreamingChunk]]) -> bool:
    for task_chunks in chunks_by_task:
        seen: set[int] = set()
        for chunk in task_chunks:
            if chunk.sm_id in seen:
                return True
            seen.add(chunk.sm_id)
    return False


def optimize_streaming_chunk_swaps(
    topology: GPUTopology,
    chunks_by_task: tuple[tuple[StreamingChunk, ...], ...],
    *,
    block_size: int,
) -> tuple[tuple[StreamingChunk, ...], ...]:
    if not config_flag("chunk_swap_search"):
        return chunks_by_task

    score_mode = config_string("chunk_swap_score", "load").strip().lower()
    if score_mode not in {"load", "replay"}:
        raise ValueError(f"chunk_swap_score must be 'load' or 'replay', got {score_mode!r}")

    def swap_score(
        candidate_chunks: Sequence[Sequence[StreamingChunk]],
    ) -> tuple[Any, ...]:
        if score_mode == "load":
            return streaming_chunk_swap_score(
                topology,
                candidate_chunks,
                block_size=block_size,
            )
        replay = replay_streaming_tree_score(
            topology,
            candidate_chunks,
            block_size=block_size,
            streaming_tree_consumer=streaming_tree_consumer_side(),
            hierarchical_cross_cluster=config_flag("hier_cross_cluster_split"),
            ready_time_tree=config_flag("ready_time_tree"),
        )
        return (
            replay.max_finish_us,
            replay.p95_finish_us,
            replay.max_recv_wait_us,
            replay.total_recv_wait_us,
            replay.hbm_edges,
            replay.instruction_count,
            replay.slot_count,
        )

    max_swaps = max(0, config_int("chunk_swap_max_swaps", 3))
    tail_only = not config_flag("chunk_swap_all_chunks")
    current = chunks_by_task
    for _ in range(max_swaps):
        current_score = swap_score(current)
        best_candidate: (
            tuple[
                tuple[Any, ...],
                tuple[tuple[StreamingChunk, ...], ...],
            ]
            | None
        ) = None
        flat_chunks = tuple(chunk for task_chunks in current for chunk in task_chunks)
        for index, left in enumerate(flat_chunks):
            if tail_only and left.part_index != left.part_count - 1:
                continue
            for right in flat_chunks[index + 1 :]:
                if tail_only and right.part_index != right.part_count - 1:
                    continue
                if left.task_id == right.task_id or left.sm_id == right.sm_id:
                    continue
                if topology.cluster_id(left.sm_id) != topology.cluster_id(right.sm_id):
                    continue
                candidate = swap_streaming_chunk_sms(current, left, right)
                if task_has_duplicate_sm(candidate):
                    continue
                candidate_score = swap_score(candidate)
                if candidate_score >= current_score:
                    continue
                if best_candidate is None or candidate_score < best_candidate[0]:
                    best_candidate = (candidate_score, candidate)
        if best_candidate is None:
            break
        current = best_candidate[1]
    return current


def balanced_dag_replay_objective(
    replay: StreamingDagReplayScore,
    *,
    hbm_penalty_blocks: int,
) -> float:
    root_skew_weight = max(0.0, config_float("balanced_dag_root_skew_weight", 0.0))
    total_root_skew_weight = max(
        0.0,
        config_float("balanced_dag_total_root_skew_weight", 0.0),
    )
    return (
        replay.max_finish_us
        + 0.20 * replay.p95_finish_us
        + 0.30 * replay.max_recv_wait_us
        + 0.02 * replay.total_recv_wait_us
        + 0.70 * hbm_penalty_blocks * replay.hbm_edges
        + root_skew_weight * replay.max_local_root_skew_us
        + total_root_skew_weight * replay.total_local_root_skew_us
        + 0.01 * replay.instruction_count
        + 0.005 * replay.slot_count
    )


def streaming_tree_replay_has_chunk_balance_gain(
    candidate: StreamingDagReplayScore,
    incumbent: StreamingDagReplayScore,
) -> bool:
    """Return whether a range move improves the minimax CTA finish profile.

    A strict makespan-only coordinate descent stalls when several CTAs share
    the critical finish time: reducing one of them leaves the global maximum
    unchanged.  Lexicographically comparing descending finish times is the
    standard minimax-fair continuation of that objective and never accepts a
    slower highest-ranked CTA merely to improve a lower-ranked one.
    """

    epsilon_us = max(
        0.0,
        config_float("tree_leaf_order_critical_eps_us", 0.05),
    )
    for candidate_finish, incumbent_finish in zip(
        candidate.finish_profile_us,
        incumbent.finish_profile_us,
    ):
        if candidate_finish < incumbent_finish - epsilon_us:
            return True
        if candidate_finish > incumbent_finish + epsilon_us:
            return False
    if len(candidate.finish_profile_us) != len(incumbent.finish_profile_us):
        return len(candidate.finish_profile_us) < len(incumbent.finish_profile_us)
    return False


def streaming_tree_chunk_balance_score_key(
    score: StreamingDagReplayScore,
) -> tuple[Any, ...]:
    return (
        score.finish_profile_us,
        score.max_recv_wait_us,
        score.total_recv_wait_us,
        score.hbm_edges,
        score.instruction_count,
        score.slot_count,
    )


def streaming_tree_reduce_candidate_sms(
    topology: GPUTopology,
    *,
    left_sm: int,
    right_sm: int,
    prefer_late_input: bool,
) -> tuple[int, ...]:
    if prefer_late_input and config_flag("cross_cluster_root_global_candidates") and not topology.same_cluster(left_sm, right_sm):
        return tuple(range(topology.sm_count))
    if prefer_late_input and config_flag("root_reduce_cluster_candidates") and topology.same_cluster(left_sm, right_sm):
        return tuple(sorted(get_cluster_sms(topology, topology.cluster_id(left_sm))))
    return tuple(sorted({left_sm, right_sm}))


def prune_ready_time_reduce_candidates(
    candidates: Sequence[int],
    *,
    ready_by_sm: Mapping[int, float],
    left_sm: int,
    right_sm: int,
) -> tuple[int, ...]:
    """Keep input owners plus the globally lightest reducer candidates."""

    candidates = tuple(dict.fromkeys(int(sm_id) for sm_id in candidates))
    limit = max(
        2,
        config_int("cross_cluster_root_candidate_limit", 8),
    )
    if len(candidates) <= limit:
        return candidates
    selected = {left_sm, right_sm}
    for sm_id in sorted(
        candidates,
        key=lambda candidate: (ready_by_sm.get(candidate, 0.0), candidate),
    ):
        selected.add(sm_id)
        if len(selected) >= limit:
            break
    return tuple(sm_id for sm_id in candidates if sm_id in selected)


def replace_task_streaming_chunks(
    chunks_by_task: Sequence[Sequence[StreamingChunk]],
    task_id: int,
    task_chunks: Sequence[StreamingChunk],
) -> tuple[tuple[StreamingChunk, ...], ...]:
    return tuple(tuple(task_chunks) if index == task_id else tuple(chunks) for index, chunks in enumerate(chunks_by_task))


def capacity_balanced_streaming_chunks(
    topology: GPUTopology,
    incumbent_chunks: Sequence[Sequence[StreamingChunk]],
    *,
    block_size: int,
    capacity_blocks: int,
    min_segment_blocks: int,
    min_chunk_blocks: int,
    max_task_segments: int,
    assignment_beam: int,
) -> tuple[tuple[StreamingChunk, ...], ...] | None:
    """Repack tasks under a per-SM capacity and the active inbox contract."""

    if capacity_blocks <= 0 or topology.cluster_count <= 0:
        return None
    cluster_sms = tuple(get_cluster_sms(topology, cluster_id) for cluster_id in range(topology.cluster_count))
    cluster_capacities = tuple(capacity_blocks * len(sms) for sms in cluster_sms)
    max_cluster_capacity = max(cluster_capacities, default=0)
    if max_cluster_capacity <= 0:
        return None

    split_items: list[CapacityItem] = []
    whole_items: list[CapacityItem] = []
    for task_id, chunks in enumerate(incumbent_chunks):
        if not chunks:
            return None
        ordered_chunks = sorted(
            chunks,
            key=lambda chunk: (
                chunk.task_range.begin,
                chunk.task_range.end,
                chunk.sm_id,
            ),
        )
        task_begin = ordered_chunks[0].task_range.begin
        task_end = ordered_chunks[-1].task_range.end
        if task_end <= task_begin:
            return None
        axis = ordered_chunks[0].task_range.axis
        if any(chunk.task_range.axis != axis for chunk in ordered_chunks):
            return None
        expected_begin = task_begin
        for chunk in ordered_chunks:
            if chunk.task_range.begin != expected_begin:
                return None
            expected_begin = chunk.task_range.end
        if expected_begin != task_end:
            return None

        task_blocks = math.ceil((task_end - task_begin) / block_size)
        if task_blocks <= max_cluster_capacity:
            whole_items.append(
                CapacityItem(
                    task_id=task_id,
                    begin=task_begin,
                    end=task_end,
                    block_count=task_blocks,
                    split_task=False,
                )
            )
            continue

        part_count = math.ceil(task_blocks / max_cluster_capacity)
        if part_count > max_task_segments:
            return None
        part_blocks = [max_cluster_capacity for _ in range(part_count - 1)]
        part_blocks.append(task_blocks - sum(part_blocks))
        if part_blocks[-1] < min_segment_blocks:
            borrowed = min_segment_blocks - part_blocks[-1]
            if part_blocks[-2] - borrowed < min_segment_blocks:
                return None
            part_blocks[-2] -= borrowed
            part_blocks[-1] += borrowed

        cursor = task_begin
        for part_index, blocks in enumerate(part_blocks):
            end = task_end if part_index == part_count - 1 else min(task_end, cursor + blocks * block_size)
            if end <= cursor:
                return None
            split_items.append(
                CapacityItem(
                    task_id=task_id,
                    begin=cursor,
                    end=end,
                    block_count=blocks,
                    split_task=True,
                )
            )
            cursor = end
        if cursor != task_end:
            return None
    items = tuple(
        sorted(
            split_items + whole_items,
            key=lambda item: (
                -item.block_count,
                not item.split_task,
                item.task_id,
                item.begin,
            ),
        )
    )
    total_blocks = sum(item.block_count for item in items)
    if total_blocks > sum(cluster_capacities):
        return None

    # State is (cluster loads, item-to-cluster assignment, split-task clusters).
    states: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, tuple[int, ...]], ...],
        ]
    ] = [
        (
            tuple(0 for _ in range(topology.cluster_count)),
            (),
            (),
        )
    ]

    def optimistic_score(
        state: tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, tuple[int, ...]], ...],
        ],
        remaining_blocks: int,
    ) -> tuple[float, float, float, tuple[int, ...], tuple[int, ...]]:
        loads, assignments, _ = state
        sm_counts = tuple(max(1, len(sms)) for sms in cluster_sms)

        def fill_at(level: float) -> float:
            return sum(
                max(
                    0.0,
                    min(float(cluster_capacities[index]), level * sm_counts[index]) - loads[index],
                )
                for index in range(topology.cluster_count)
            )

        low = min(
            (loads[index] / sm_counts[index] for index in range(topology.cluster_count)),
            default=0.0,
        )
        high = float(capacity_blocks)
        if remaining_blocks > 0:
            for _ in range(32):
                middle = (low + high) * 0.5
                if fill_at(middle) < remaining_blocks:
                    low = middle
                else:
                    high = middle
        level = high if remaining_blocks > 0 else low
        projected = tuple(
            max(
                loads[index] / sm_counts[index],
                min(float(capacity_blocks), level),
            )
            for index in range(topology.cluster_count)
        )
        max_load = max(projected, default=0.0)
        min_load = min(projected, default=0.0)
        mean_load = total_blocks / max(1, topology.sm_count)
        variance = sum((load - mean_load) ** 2 for load in projected)
        return (max_load, max_load - min_load, variance, loads, assignments)

    remaining_blocks = total_blocks
    beam_width = max(1, assignment_beam)
    for item in items:
        remaining_blocks -= item.block_count
        next_states: dict[
            tuple[tuple[int, ...], tuple[tuple[int, tuple[int, ...]], ...]],
            tuple[
                tuple[int, ...],
                tuple[int, ...],
                tuple[tuple[int, tuple[int, ...]], ...],
            ],
        ] = {}
        for loads, assignments, split_clusters_tuple in states:
            split_clusters = {task_id: set(cluster_ids) for task_id, cluster_ids in split_clusters_tuple}
            used_clusters = split_clusters.get(item.task_id, set())
            equivalent_clusters: set[tuple[int, int, bool]] = set()
            for cluster_id in range(topology.cluster_count):
                signature = (
                    loads[cluster_id],
                    cluster_capacities[cluster_id],
                    cluster_id in used_clusters,
                )
                if signature in equivalent_clusters:
                    continue
                equivalent_clusters.add(signature)
                if cluster_id in used_clusters:
                    continue
                if loads[cluster_id] + item.block_count > cluster_capacities[cluster_id]:
                    continue
                updated_loads = list(loads)
                updated_loads[cluster_id] += item.block_count
                updated_split_clusters = {task_id: set(cluster_ids) for task_id, cluster_ids in split_clusters.items()}
                if item.split_task:
                    updated_split_clusters.setdefault(item.task_id, set()).add(cluster_id)
                normalized_split_clusters = tuple(
                    (task_id, tuple(sorted(cluster_ids))) for task_id, cluster_ids in sorted(updated_split_clusters.items())
                )
                candidate = (
                    tuple(updated_loads),
                    assignments + (cluster_id,),
                    normalized_split_clusters,
                )
                key = (candidate[0], candidate[2])
                previous = next_states.get(key)
                if previous is None or candidate[1] < previous[1]:
                    next_states[key] = candidate
        if not next_states:
            return None
        states = sorted(
            next_states.values(),
            key=lambda state: optimistic_score(state, remaining_blocks),
        )[:beam_width]

    def assign_cluster_items(
        cluster_id: int,
        cluster_items: Sequence[CapacityItem],
        *,
        short_root_first: bool,
    ) -> list[StreamingChunk] | None:
        sms = cluster_sms[cluster_id]
        if not sms and cluster_items:
            return None
        sm_loads = {sm_id: 0 for sm_id in sms}
        cluster_inbox_owners: set[int] = set()
        require_unique_inbox_owners = not config_flag("joint_schedule")
        cluster_chunks: list[StreamingChunk] = []
        for item in sorted(
            cluster_items,
            key=lambda candidate: (-candidate.block_count, candidate.task_id, candidate.begin),
        ):
            chunk_count = math.ceil(item.block_count / capacity_blocks)
            chunk_blocks_by_index = [capacity_blocks for _ in range(chunk_count - 1)]
            chunk_blocks_by_index.append(item.block_count - sum(chunk_blocks_by_index))
            if chunk_count > 1 and item.block_count >= min_chunk_blocks and chunk_blocks_by_index[-1] < min_chunk_blocks:
                borrowed = min_chunk_blocks - chunk_blocks_by_index[-1]
                if chunk_blocks_by_index[-2] - borrowed < min_chunk_blocks:
                    return None
                chunk_blocks_by_index[-2] -= borrowed
                chunk_blocks_by_index[-1] += borrowed
            if short_root_first and chunk_count > 1:
                chunk_blocks_by_index = chunk_blocks_by_index[-1:] + chunk_blocks_by_index[:-1]
            cursor = item.begin
            axis = incumbent_chunks[item.task_id][0].task_range.axis
            used_sms: set[int] = set()
            for chunk_index, chunk_blocks in enumerate(chunk_blocks_by_index):
                needs_cluster_inbox = chunk_index > 0
                available_sms = [
                    sm_id
                    for sm_id in sms
                    if sm_id not in used_sms
                    and sm_loads[sm_id] + chunk_blocks <= capacity_blocks
                    and (not needs_cluster_inbox or not require_unique_inbox_owners or sm_id not in cluster_inbox_owners)
                ]
                if not available_sms:
                    return None
                sm_id = min(
                    available_sms,
                    key=lambda candidate: (
                        (candidate not in cluster_inbox_owners if short_root_first and not needs_cluster_inbox else False),
                        (sm_loads[candidate] == 0 if short_root_first and not needs_cluster_inbox else False),
                        (-sm_loads[candidate] if short_root_first and not needs_cluster_inbox else sm_loads[candidate]),
                        candidate,
                    ),
                )
                end = item.end if chunk_index == chunk_count - 1 else min(item.end, cursor + chunk_blocks * block_size)
                if end <= cursor:
                    return None
                cluster_chunks.append(
                    StreamingChunk(
                        task_id=item.task_id,
                        sm_id=sm_id,
                        task_range=TaskRange(axis=axis, begin=cursor, end=end),
                        part_index=0,
                        part_count=0,
                    )
                )
                sm_loads[sm_id] += chunk_blocks
                used_sms.add(sm_id)
                if needs_cluster_inbox and require_unique_inbox_owners:
                    cluster_inbox_owners.add(sm_id)
                cursor = end
            if cursor != item.end:
                return None
        return cluster_chunks

    def materialize_cluster_assignment(
        items_by_cluster: Sequence[Sequence[CapacityItem]],
    ) -> list[list[StreamingChunk]] | None:
        candidate_chunks: list[list[StreamingChunk]] = [[] for _ in incumbent_chunks]
        for cluster_id, cluster_items in enumerate(items_by_cluster):
            cluster_chunks = assign_cluster_items(
                cluster_id,
                cluster_items,
                short_root_first=False,
            )
            if cluster_chunks is None:
                # A short first chunk consumes no cluster inbox and can share a
                # CTA with a receiver, reserving unique destinations for all
                # later accumulator handoffs in the segment.
                cluster_chunks = assign_cluster_items(
                    cluster_id,
                    cluster_items,
                    short_root_first=True,
                )
            if cluster_chunks is None:
                return None
            for chunk in cluster_chunks:
                candidate_chunks[chunk.task_id].append(chunk)
        return candidate_chunks

    assigned_chunks: list[list[StreamingChunk]] | None = None
    for _, assignments, _ in sorted(
        states,
        key=lambda state: optimistic_score(state, 0),
    ):
        items_by_cluster: list[list[CapacityItem]] = [[] for _ in range(topology.cluster_count)]
        for item, cluster_id in zip(items, assignments):
            items_by_cluster[cluster_id].append(item)
        candidate_chunks = materialize_cluster_assignment(items_by_cluster)
        if candidate_chunks is not None:
            assigned_chunks = candidate_chunks
            break

    if assigned_chunks is None:
        return None

    result: list[tuple[StreamingChunk, ...]] = []
    for _task_id, chunks in enumerate(assigned_chunks):
        ordered = sorted(
            chunks,
            key=lambda chunk: (
                chunk.task_range.begin,
                chunk.task_range.end,
                chunk.sm_id,
            ),
        )
        if not ordered:
            return None
        result.append(
            tuple(
                replace(
                    chunk,
                    part_index=part_index,
                    part_count=len(ordered),
                )
                for part_index, chunk in enumerate(ordered)
            )
        )
    return tuple(result)


def streaming_task_chunk_block_counts(
    chunks: Sequence[StreamingChunk],
    *,
    block_size: int,
) -> tuple[int, ...]:
    return tuple(max(1, math.ceil(chunk.task_range.length / block_size)) for chunk in chunks)


def rebuild_streaming_task_chunks_with_block_counts(
    chunks: Sequence[StreamingChunk],
    *,
    block_counts: Sequence[int],
    block_size: int,
) -> tuple[StreamingChunk, ...] | None:
    if not chunks or len(chunks) != len(block_counts):
        return None
    axis = chunks[0].task_range.axis
    begin = min(chunk.task_range.begin for chunk in chunks)
    task_end = max(chunk.task_range.end for chunk in chunks)
    part_count = len(chunks)
    rebuilt: list[StreamingChunk] = []
    cursor = begin
    for part_index, (chunk, block_count) in enumerate(zip(chunks, block_counts)):
        if block_count <= 0:
            return None
        if part_index == part_count - 1:
            end = task_end
        else:
            end = min(task_end, cursor + block_count * block_size)
        if end <= cursor:
            return None
        rebuilt.append(
            StreamingChunk(
                task_id=chunk.task_id,
                sm_id=chunk.sm_id,
                task_range=TaskRange(axis=axis, begin=cursor, end=end),
                part_index=part_index,
                part_count=part_count,
            )
        )
        cursor = end
    if cursor != task_end:
        return None
    return tuple(rebuilt)


def estimate_streaming_tree_root_sms(
    topology: GPUTopology,
    chunks_by_task: Sequence[Sequence[StreamingChunk]],
    *,
    block_size: int,
    streaming_tree_consumer: str,
    hierarchical_cross_cluster: bool,
    ready_time_tree: bool,
) -> tuple[int | None, ...]:
    sm_ready = {sm_id: 0.0 for sm_id in range(topology.sm_count)}
    slot_ready: dict[int, float] = {}
    next_slot_id = 0
    ready_time_recv_weight = config_float("ready_time_recv_weight", 0.6)
    ready_time_queue_weight = config_float("ready_time_queue_weight", 0.1)
    ready_time_balance_weight = config_float("ready_time_balance_weight", 0.0)
    ready_time_cluster_comm_weight = config_float("ready_time_cluster_comm_weight", 0.25)
    direct_leaf_acc = config_flag("direct_leaf_acc")
    root_sms: list[int | None] = []

    def reserve_instruction(
        *,
        sm_id: int,
        input_sources: Sequence[tuple[int, int]],
        output_slot: int | None,
        duration_us: float,
    ) -> None:
        input_ready = max(
            (
                slot_ready.get(slot_id, 0.0) + estimated_streaming_comm_cost_us(topology, producer_sm, sm_id)
                for slot_id, producer_sm in input_sources
            ),
            default=0.0,
        )
        finish = max(sm_ready.get(sm_id, 0.0), input_ready) + duration_us
        sm_ready[sm_id] = finish
        if output_slot is not None:
            slot_ready[output_slot] = finish

    def choose_reduce_sm(
        *,
        left_slot: int,
        left_sm: int,
        right_slot: int,
        right_sm: int,
        prefer_late_input: bool,
    ) -> int:
        fixed_sm = right_sm if streaming_tree_consumer == "right" else left_sm
        if not ready_time_tree:
            return fixed_sm
        candidates = streaming_tree_reduce_candidate_sms(
            topology,
            left_sm=left_sm,
            right_sm=right_sm,
            prefer_late_input=prefer_late_input,
        )
        candidates = prune_ready_time_reduce_candidates(
            candidates,
            ready_by_sm=sm_ready,
            left_sm=left_sm,
            right_sm=right_sm,
        )

        def candidate_score(sm_id: int) -> tuple[float, float, float, int]:
            left_comm = estimated_streaming_comm_cost_us(topology, left_sm, sm_id)
            right_comm = estimated_streaming_comm_cost_us(topology, right_sm, sm_id)
            left_ready = slot_ready.get(left_slot, 0.0) + left_comm
            right_ready = slot_ready.get(right_slot, 0.0) + right_comm
            input_ready = max(left_ready, right_ready)
            queue_ready = sm_ready.get(sm_id, 0.0)
            recv_wait = max(0.0, input_ready - queue_ready)
            finish = max(queue_ready, input_ready) + estimated_streaming_reduce_cost_us(2)
            projected_max = max(finish, max(sm_ready.values(), default=0.0))
            remote_comm_cost = (left_comm if left_sm != sm_id else 0.0) + (right_comm if right_sm != sm_id else 0.0)
            score = (
                finish
                + ready_time_recv_weight * recv_wait
                + ready_time_queue_weight * queue_ready
                + ready_time_balance_weight * projected_max
                + ready_time_cluster_comm_weight * remote_comm_cost
            )
            if prefer_late_input and (
                (config_flag("cross_cluster_root_global_candidates") and not topology.same_cluster(left_sm, right_sm))
                or (config_flag("root_reduce_cluster_candidates") and topology.same_cluster(left_sm, right_sm))
            ):
                return (score, queue_ready, recv_wait, -sm_id)
            return (score, recv_wait, queue_ready, -sm_id)

        return min(candidates, key=candidate_score)

    def emit_tree(nodes: list[tuple[int, int]]) -> tuple[int, int]:
        nonlocal next_slot_id
        if config_flag("ordered_interval_tree") and len(nodes) > 1:
            tree_plan = ordered_reduction_tree_plan(
                topology,
                tuple((sm_id, slot_ready.get(slot_id, 0.0)) for slot_id, sm_id in nodes),
                consumer_side=streaming_tree_consumer,
                adaptive_consumer=config_flag("ordered_tree_adaptive_consumer"),
            )

            def emit_ordered(
                plan: OrderedReductionTreePlan,
            ) -> tuple[int, int]:
                nonlocal next_slot_id
                if plan.is_leaf:
                    return nodes[plan.begin]
                frontier = ordered_reduction_fused_frontier(
                    plan,
                    max_reduce_arity=max(
                        2,
                        config_int("ordered_tree_max_reduce_arity", 2),
                    ),
                    max_resident_remote_inputs=joint_max_resident_remote_inputs(),
                )
                inputs = tuple(emit_ordered(child) for child in frontier)
                acc_slot_id = next_slot_id
                next_slot_id += 1
                reserve_instruction(
                    sm_id=plan.consumer_sm,
                    input_sources=inputs,
                    output_slot=acc_slot_id,
                    duration_us=estimated_streaming_reduce_cost_us(len(inputs)),
                )
                return acc_slot_id, plan.consumer_sm

            return emit_ordered(tree_plan)

        current_nodes = nodes
        while len(current_nodes) > 1:
            next_nodes: list[tuple[int, int]] = []
            for index in range(0, len(current_nodes), 2):
                if index + 1 >= len(current_nodes):
                    next_nodes.append(current_nodes[index])
                    continue
                left_slot, left_sm = current_nodes[index]
                right_slot, right_sm = current_nodes[index + 1]
                reduce_sm = choose_reduce_sm(
                    left_slot=left_slot,
                    left_sm=left_sm,
                    right_slot=right_slot,
                    right_sm=right_sm,
                    prefer_late_input=len(current_nodes) == 2,
                )
                acc_slot_id = next_slot_id
                next_slot_id += 1
                reserve_instruction(
                    sm_id=reduce_sm,
                    input_sources=((left_slot, left_sm), (right_slot, right_sm)),
                    output_slot=acc_slot_id,
                    duration_us=estimated_streaming_reduce_cost_us(2),
                )
                next_nodes.append((acc_slot_id, reduce_sm))
            current_nodes = next_nodes
        return current_nodes[0]

    for unordered_chunks in chunks_by_task:
        chunks = logical_streaming_chunk_order(unordered_chunks)
        if not chunks:
            root_sms.append(None)
            continue
        leaf_nodes: list[tuple[int, int]] = []
        for chunk in chunks:
            partial_slot_id = next_slot_id
            next_slot_id += 1
            reserve_instruction(
                sm_id=chunk.sm_id,
                input_sources=(),
                output_slot=partial_slot_id,
                duration_us=estimated_streaming_iter_cost_us(chunk.task_range, block_size),
            )
            if direct_leaf_acc:
                leaf_nodes.append((partial_slot_id, chunk.sm_id))
                continue
            acc_slot_id = next_slot_id
            next_slot_id += 1
            reserve_instruction(
                sm_id=chunk.sm_id,
                input_sources=((partial_slot_id, chunk.sm_id),),
                output_slot=acc_slot_id,
                duration_us=estimated_streaming_reduce_cost_us(1),
            )
            leaf_nodes.append((acc_slot_id, chunk.sm_id))

        cluster_runs = ordered_cluster_run_bounds(
            topology,
            tuple(sm_id for _, sm_id in leaf_nodes),
        )
        if hierarchical_cross_cluster and len(cluster_runs) > 1:
            local_roots = [emit_tree(leaf_nodes[begin:end]) for _, begin, end in cluster_runs]
            root_slot, root_sm = emit_tree(local_roots)
        else:
            root_slot, root_sm = emit_tree(leaf_nodes)
        reserve_instruction(
            sm_id=root_sm,
            input_sources=((root_slot, root_sm),),
            output_slot=None,
            duration_us=estimated_streaming_finalize_cost_us(),
        )
        root_sms.append(root_sm)
    return tuple(root_sms)


def optimize_streaming_tree_chunk_lengths_by_replay(
    topology: GPUTopology,
    chunks_by_task: tuple[tuple[StreamingChunk, ...], ...],
    *,
    block_size: int,
    streaming_tree_consumer: str,
    hierarchical_cross_cluster: bool,
    ready_time_tree: bool,
) -> tuple[tuple[StreamingChunk, ...], ...]:
    if not config_flag("chunk_length_search"):
        return chunks_by_task

    max_passes = max(1, config_int("chunk_length_search_passes", 2))
    max_evals = max(1, config_int("chunk_length_search_max_evals", 4096))
    step_blocks = max(1, config_int("chunk_length_search_step_blocks", 2))
    min_blocks = max(1, config_int("chunk_length_search_min_blocks", 1))
    all_donors = config_flag("chunk_length_search_all_donors")

    current = chunks_by_task
    current_score = replay_streaming_tree_score(
        topology,
        current,
        block_size=block_size,
        streaming_tree_consumer=streaming_tree_consumer,
        hierarchical_cross_cluster=hierarchical_cross_cluster,
        ready_time_tree=ready_time_tree,
    )

    eval_count = 0
    for _ in range(max_passes):
        current_root_sms = estimate_streaming_tree_root_sms(
            topology,
            current,
            block_size=block_size,
            streaming_tree_consumer=streaming_tree_consumer,
            hierarchical_cross_cluster=hierarchical_cross_cluster,
            ready_time_tree=ready_time_tree,
        )
        critical_leaf_owners = set(current_score.critical_leaf_owners)
        best_candidate: tuple[StreamingDagReplayScore, tuple[tuple[StreamingChunk, ...], ...]] | None = None
        for task_id, task_chunks in enumerate(current):
            if len(task_chunks) <= 1:
                continue
            root_sm = current_root_sms[task_id] if task_id < len(current_root_sms) else None
            if root_sm is None:
                continue
            base_counts = streaming_task_chunk_block_counts(
                task_chunks,
                block_size=block_size,
            )
            for donor_index, donor_blocks in enumerate(base_counts):
                if all_donors and critical_leaf_owners and (task_id, task_chunks[donor_index].sm_id) not in critical_leaf_owners:
                    continue
                if not all_donors and task_chunks[donor_index].sm_id != root_sm:
                    continue
                if donor_blocks <= min_blocks:
                    continue
                shift_blocks = min(step_blocks, donor_blocks - min_blocks)
                if shift_blocks <= 0:
                    continue
                for receiver_index in range(len(task_chunks)):
                    if receiver_index == donor_index:
                        continue
                    if eval_count >= max_evals:
                        break
                    candidate_counts = list(base_counts)
                    candidate_counts[donor_index] -= shift_blocks
                    candidate_counts[receiver_index] += shift_blocks
                    rebuilt_task_chunks = rebuild_streaming_task_chunks_with_block_counts(
                        task_chunks,
                        block_counts=candidate_counts,
                        block_size=block_size,
                    )
                    if rebuilt_task_chunks is None:
                        continue
                    candidate = replace_task_streaming_chunks(
                        current,
                        task_id,
                        rebuilt_task_chunks,
                    )
                    candidate_root_sms = estimate_streaming_tree_root_sms(
                        topology,
                        candidate,
                        block_size=block_size,
                        streaming_tree_consumer=streaming_tree_consumer,
                        hierarchical_cross_cluster=hierarchical_cross_cluster,
                        ready_time_tree=ready_time_tree,
                    )
                    if not all_donors and (task_id >= len(candidate_root_sms) or candidate_root_sms[task_id] != root_sm):
                        continue
                    eval_count += 1
                    candidate_score = replay_streaming_tree_score(
                        topology,
                        candidate,
                        block_size=block_size,
                        streaming_tree_consumer=streaming_tree_consumer,
                        hierarchical_cross_cluster=hierarchical_cross_cluster,
                        ready_time_tree=ready_time_tree,
                    )
                    if not streaming_tree_replay_has_chunk_balance_gain(
                        candidate_score,
                        current_score,
                    ):
                        continue
                    if best_candidate is None or streaming_tree_chunk_balance_score_key(
                        candidate_score
                    ) < streaming_tree_chunk_balance_score_key(best_candidate[0]):
                        best_candidate = (candidate_score, candidate)
                if eval_count >= max_evals:
                    break
            if eval_count >= max_evals:
                break
        if best_candidate is None:
            break
        current_score, current = best_candidate
        if eval_count >= max_evals:
            break
    return current


def optimize_streaming_tree_leaf_orders(
    topology: GPUTopology,
    chunks_by_task: tuple[tuple[StreamingChunk, ...], ...],
    *,
    block_size: int,
    streaming_tree_consumer: str,
    hierarchical_cross_cluster: bool,
    ready_time_tree: bool,
) -> tuple[tuple[StreamingChunk, ...], ...]:
    del topology, block_size, streaming_tree_consumer
    del hierarchical_cross_cluster, ready_time_tree

    # Associativity permits a different parenthesization, but it does not permit
    # a permutation.  Placement searches may move ranges between CTAs; the tree
    # itself must still consume those ranges in part order.
    return tuple(logical_streaming_chunk_order(task_chunks) for task_chunks in chunks_by_task)


def load_balanced_streaming_chunks(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
    axis: Any,
    task_range_offsets: tuple[int, ...] | None = None,
    cluster_task_assignment: Sequence[Sequence[int]] | str | None = None,
    skip_tiny_root_fragment: bool | None = None,
    skip_tiny_root_fragment_blocks: int | None = None,
    reduce_strategy: str = "streaming_tree",
) -> tuple[tuple[StreamingChunk, ...], ...]:
    resolved_task_range_offsets = task_range_offsets or (0,) * len(task_range_lengths)

    def postprocess_chunks(
        chunks_tuple: tuple[tuple[StreamingChunk, ...], ...],
        *,
        cluster_tasks: Sequence[Sequence[int]],
    ) -> tuple[tuple[StreamingChunk, ...], ...]:
        """Run topology-independent chunk refinements for every seed policy.

        Seed construction determines the initial ownership graph.  Critical
        path replay, tree-owner range balancing, and placement swaps are
        deliberately downstream operations: bypassing them for a global seed
        prevents communication readiness from feeding back into the compute
        partition even though the resulting tree uses the same contracts.
        """

        if config_flag("critical_path_split") and not config_flag("global_capacity_chunks") and topology.cluster_count > 1:
            chunks_tuple = optimize_critical_path_split_streaming_chunks(
                topology,
                task_range_lengths,
                block_size=block_size,
                axis=axis,
                task_range_offsets=resolved_task_range_offsets,
                cluster_task_assignment=cluster_tasks,
                incumbent_chunks=chunks_tuple,
                reduce_strategy=reduce_strategy,
            )
        if config_flag("balance_cluster_segment_chunks") or config_flag("balance_critical_cluster_segment_chunks"):
            chunks_tuple = balance_streaming_cluster_segment_chunks(
                topology,
                chunks_tuple,
                block_size=block_size,
                critical_tasks_only=(
                    not config_flag("balance_cluster_segment_chunks") and config_flag("balance_critical_cluster_segment_chunks")
                ),
            )
        if config_flag("balance_tree_owner_work_chunks"):
            chunks_tuple = balance_streaming_tree_owner_work_chunks(
                topology,
                chunks_tuple,
                block_size=block_size,
                consumer_side=streaming_tree_consumer_side(),
                hierarchical_cross_cluster=config_flag("hier_cross_cluster_split"),
            )
        return optimize_streaming_chunk_swaps(
            topology,
            chunks_tuple,
            block_size=block_size,
        )

    manual_segments = manual_cluster_segments_from_config(
        topology,
        task_range_lengths,
        block_size=block_size,
    )
    if manual_segments is not None:
        return segments_to_streaming_chunks(
            topology,
            task_range_lengths,
            block_size=block_size,
            axis=axis,
            task_range_offsets=resolved_task_range_offsets,
            segments=manual_segments,
            require_all_tasks=True,
            error_prefix="Dataflow manual cluster segment scheduler",
        )
    resolved_cluster_tasks = normalize_cluster_task_assignment(
        cluster_task_assignment,
        task_count=len(task_range_lengths),
        cluster_count=topology.cluster_count,
    )
    cluster_tasks = load_balanced_cluster_tasks(topology, task_range_lengths) if resolved_cluster_tasks is None else resolved_cluster_tasks
    if config_flag("global_capacity_chunks"):
        return postprocess_chunks(
            global_capacity_streaming_chunks(
                topology,
                task_range_lengths,
                block_size=block_size,
                axis=axis,
                task_range_offsets=resolved_task_range_offsets,
            ),
            cluster_tasks=cluster_tasks,
        )
    if config_flag("balanced_dag_sched") and topology.cluster_count > 1:
        return load_balanced_dag_streaming_chunks(
            topology,
            task_range_lengths,
            block_size=block_size,
            axis=axis,
            task_range_offsets=resolved_task_range_offsets,
            cluster_task_assignment=resolved_cluster_tasks,
        )
    if resolved_cluster_tasks is None and config_flag("cross_cluster_long_split") and topology.cluster_count > 1:
        return load_balanced_cross_cluster_streaming_chunks(
            topology,
            task_range_lengths,
            block_size=block_size,
            axis=axis,
            task_range_offsets=resolved_task_range_offsets,
        )

    task_ranges: list[list[tuple[int, TaskRange]]] = [[] for _ in task_range_lengths]
    heap_chunks = config_flag("heap_streaming_chunks")

    for cluster_id, tasks in enumerate(cluster_tasks):
        if not tasks:
            continue
        sms = get_cluster_sms(topology, cluster_id)
        if not sms:
            raise ValueError(f"Dataflow scheduler found empty cluster {cluster_id} in topology {topology!r}")

        cluster_work_blocks = sum(math.ceil(task_range_lengths[task_id] / block_size) for task_id in tasks)
        target_sm_work_blocks = max(1, math.ceil(cluster_work_blocks / len(sms)))
        if heap_chunks:
            sm_work_blocks_by_id = {sm_id: 0 for sm_id in sms}
            for task_id in tasks:
                length = task_range_lengths[task_id]
                begin = 0
                while begin < length:
                    remaining = length - begin
                    remaining_blocks = math.ceil(remaining / block_size)
                    chunk_blocks = min(remaining_blocks, target_sm_work_blocks)
                    end = (
                        length
                        if remaining_blocks <= target_sm_work_blocks
                        else min(
                            length,
                            begin + chunk_blocks * block_size,
                        )
                    )
                    sm_id = min(sms, key=lambda item: (sm_work_blocks_by_id[item], item))
                    append_streaming_range(
                        task_ranges[task_id],
                        sm_id=sm_id,
                        axis=axis,
                        begin=begin,
                        end=end,
                    )
                    sm_work_blocks_by_id[sm_id] += math.ceil((end - begin) / block_size)
                    begin = end
            continue

        sm_index = 0
        sm_work_blocks = 0

        resolved_skip_tiny_root_fragment = (
            config_flag("skip_tiny_root_fragment")
            if skip_tiny_root_fragment is None
            else bool_value(skip_tiny_root_fragment, "skip_tiny_root_fragment")
        )

        for task_index, task_id in enumerate(tasks):
            length = task_range_lengths[task_id]
            if resolved_skip_tiny_root_fragment and task_index > 0 and sm_work_blocks > 0 and sm_index < len(sms) - 1:
                remaining_capacity_blocks = target_sm_work_blocks - sm_work_blocks
                skip_threshold_blocks = (
                    config_int(
                        "skip_tiny_root_fragment_blocks",
                        max(2, target_sm_work_blocks // 3),
                    )
                    if skip_tiny_root_fragment_blocks is None
                    else int(skip_tiny_root_fragment_blocks)
                )
                if skip_threshold_blocks > 0 and remaining_capacity_blocks <= skip_threshold_blocks:
                    sm_index += 1
                    sm_work_blocks = 0
            begin = 0
            while begin < length:
                if sm_index >= len(sms):
                    sm_index = len(sms) - 1
                sm_id = sms[sm_index]
                remaining = length - begin
                remaining_blocks = math.ceil(remaining / block_size)

                if sm_index == len(sms) - 1:
                    end = length
                else:
                    capacity_blocks = target_sm_work_blocks - sm_work_blocks
                    if capacity_blocks <= 0:
                        sm_index += 1
                        sm_work_blocks = 0
                        continue
                    if remaining_blocks <= capacity_blocks:
                        end = length
                    else:
                        chunk_blocks = max(1, min(remaining_blocks, capacity_blocks))
                        end = min(length, begin + chunk_blocks * block_size)

                append_streaming_range(
                    task_ranges[task_id],
                    sm_id=sm_id,
                    axis=axis,
                    begin=begin,
                    end=end,
                )
                sm_work_blocks += math.ceil((end - begin) / block_size)
                begin = end

                if begin < length and sm_index < len(sms) - 1 and sm_work_blocks >= target_sm_work_blocks:
                    sm_index += 1
                    sm_work_blocks = 0

    merge_tiny_final_streaming_ranges(
        task_ranges,
        topology=topology,
        block_size=block_size,
    )

    chunks_by_task: list[tuple[StreamingChunk, ...]] = []
    for task_id, ranges in enumerate(task_ranges):
        part_count = len(ranges)
        if part_count == 0:
            raise ValueError(f"Dataflow streaming scheduler produced no chunks for task {task_id}")
        range_offset = resolved_task_range_offsets[task_id]
        chunks_by_task.append(
            tuple(
                StreamingChunk(
                    task_id=task_id,
                    sm_id=sm_id,
                    task_range=offset_task_range(task_range, range_offset),
                    part_index=part_index,
                    part_count=part_count,
                )
                for part_index, (sm_id, task_range) in enumerate(ranges)
            )
        )
    chunks_tuple = tuple(chunks_by_task)
    return postprocess_chunks(
        chunks_tuple,
        cluster_tasks=cluster_tasks,
    )


def materialize_global_capacity_streaming_chunks(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
    axis: Any,
    task_range_offsets: tuple[int, ...],
    task_blocks: tuple[int, ...],
    part_counts: Sequence[int],
) -> tuple[tuple[StreamingChunk, ...], ...]:
    if len(part_counts) != len(task_blocks):
        raise ValueError("global capacity chunking requires one part count per task")
    if any(part_count <= 0 or part_count > blocks for blocks, part_count in zip(task_blocks, part_counts)):
        raise ValueError("global capacity chunking requires 1 <= parts <= task blocks")
    if sum(part_counts) > topology.sm_count:
        raise ValueError(
            f"global capacity chunking cannot assign more than one ITER chunk per CTA: parts={sum(part_counts)}, ctas={topology.sm_count}"
        )

    task_block_partitions: dict[int, tuple[int, ...]] = {}
    consumer_side = streaming_tree_consumer_side()
    for task_id, blocks in enumerate(task_blocks):
        part_count = part_counts[task_id]
        base, larger_parts = divmod(blocks, part_count)
        task_block_partitions[task_id] = tuple(
            base + (part_index >= part_count - larger_parts if consumer_side == "left" else part_index < larger_parts)
            for part_index in range(part_count)
        )

    available_sms = {cluster_id: list(get_cluster_sms(topology, cluster_id)) for cluster_id in range(topology.cluster_count)}
    cluster_loads = [0 for _ in range(topology.cluster_count)]
    task_ranges: list[list[tuple[int, TaskRange]]] = [[] for _ in task_range_lengths]
    for task_id in sorted(
        range(len(task_range_lengths)),
        key=lambda item: (-task_blocks[item], item),
    ):
        partitions = task_block_partitions[task_id]
        begin_block = 0
        part_index = 0
        while part_index < len(partitions):
            remaining_parts = len(partitions) - part_index
            candidates = tuple(cluster_id for cluster_id, sms in available_sms.items() if sms)
            if not candidates:
                raise RuntimeError("global capacity chunking exhausted CTA ownership")
            cluster_id = min(
                candidates,
                key=lambda item: (
                    0 if len(available_sms[item]) >= remaining_parts else 1,
                    cluster_loads[item] / max(1, len(get_cluster_sms(topology, item))),
                    -len(available_sms[item]),
                    item,
                ),
            )
            take = min(remaining_parts, len(available_sms[cluster_id]))
            for _ in range(take):
                chunk_blocks = partitions[part_index]
                chunk_begin = begin_block * block_size
                begin_block += chunk_blocks
                chunk_end = min(
                    task_range_lengths[task_id],
                    begin_block * block_size,
                )
                sm_id = available_sms[cluster_id].pop(0)
                task_ranges[task_id].append(
                    (
                        sm_id,
                        TaskRange(
                            axis=axis,
                            begin=chunk_begin,
                            end=chunk_end,
                        ),
                    )
                )
                cluster_loads[cluster_id] += chunk_blocks
                part_index += 1

    result: list[tuple[StreamingChunk, ...]] = []
    for task_id, ranges in enumerate(task_ranges):
        ordered = sorted(
            ranges,
            key=lambda item: (item[1].begin, item[1].end, item[0]),
        )
        if not ordered or ordered[0][1].begin != 0:
            raise RuntimeError(f"global capacity chunking left task {task_id} uncovered")
        if ordered[-1][1].end != task_range_lengths[task_id] or any(
            left[1].end != right[1].begin for left, right in zip(ordered, ordered[1:])
        ):
            raise RuntimeError(f"global capacity chunking does not tile task {task_id}")
        part_count = len(ordered)
        range_offset = task_range_offsets[task_id]
        result.append(
            tuple(
                StreamingChunk(
                    task_id=task_id,
                    sm_id=sm_id,
                    task_range=offset_task_range(task_range, range_offset),
                    part_index=part_index,
                    part_count=part_count,
                )
                for part_index, (sm_id, task_range) in enumerate(ordered)
            )
        )
    return tuple(result)


def refine_global_capacity_part_counts_by_replay(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
    axis: Any,
    task_range_offsets: tuple[int, ...],
    task_blocks: tuple[int, ...],
    initial_part_counts: Sequence[int],
) -> tuple[tuple[StreamingChunk, ...], ...]:
    """Exchange CTA partitions using the complete reduction/transport replay."""

    def materialize(
        part_counts: Sequence[int],
    ) -> tuple[tuple[StreamingChunk, ...], ...]:
        return materialize_global_capacity_streaming_chunks(
            topology,
            task_range_lengths,
            block_size=block_size,
            axis=axis,
            task_range_offsets=task_range_offsets,
            task_blocks=task_blocks,
            part_counts=part_counts,
        )

    current_counts = tuple(int(value) for value in initial_part_counts)
    current = materialize(current_counts)
    if not config_flag("global_capacity_replay_refine"):
        return current

    consumer_side = streaming_tree_consumer_side()
    hierarchical_cross_cluster = config_flag("hier_cross_cluster_split")
    ready_time_tree = config_flag("ready_time_tree")

    def replay(
        chunks: Sequence[Sequence[StreamingChunk]],
    ) -> StreamingDagReplayScore:
        return replay_streaming_tree_score(
            topology,
            chunks,
            block_size=block_size,
            streaming_tree_consumer=consumer_side,
            hierarchical_cross_cluster=hierarchical_cross_cluster,
            ready_time_tree=ready_time_tree,
        )

    current_score = replay(current)
    max_steps = max(
        0,
        config_int("global_capacity_replay_max_steps", 8),
    )
    max_evals = max(
        0,
        config_int("global_capacity_replay_max_evals", 4096),
    )
    eval_count = 0
    for _ in range(max_steps):
        critical_tasks = tuple(dict.fromkeys(task_id for task_id, _sm_id in current_score.critical_leaf_owners))
        receivers = critical_tasks or tuple(range(len(current_counts)))
        proposal_counts: set[tuple[int, ...]] = set()
        free_ctas = topology.sm_count - sum(current_counts)
        if free_ctas > 0:
            # A free CTA has no donor-side regression, so also evaluate tasks
            # outside the current critical chain.  The minimax finish profile
            # decides whether relieving a plateau is useful.
            for receiver in range(len(current_counts)):
                if current_counts[receiver] >= task_blocks[receiver]:
                    continue
                candidate = list(current_counts)
                candidate[receiver] += 1
                proposal_counts.add(tuple(candidate))
        else:
            for receiver in receivers:
                if current_counts[receiver] >= task_blocks[receiver]:
                    continue
                for donor in range(len(current_counts)):
                    if donor == receiver or current_counts[donor] <= 1:
                        continue
                    candidate = list(current_counts)
                    candidate[donor] -= 1
                    candidate[receiver] += 1
                    proposal_counts.add(tuple(candidate))

        best: (
            tuple[
                StreamingDagReplayScore,
                tuple[int, ...],
                tuple[tuple[StreamingChunk, ...], ...],
            ]
            | None
        ) = None
        for candidate_counts in sorted(proposal_counts):
            if eval_count >= max_evals:
                break
            candidate = materialize(candidate_counts)
            candidate_score = replay(candidate)
            eval_count += 1
            if not streaming_tree_replay_has_chunk_balance_gain(
                candidate_score,
                current_score,
            ):
                continue
            if best is None or streaming_tree_chunk_balance_score_key(candidate_score) < streaming_tree_chunk_balance_score_key(best[0]):
                best = (candidate_score, candidate_counts, candidate)
        if best is None:
            break
        current_score, current_counts, current = best
        if eval_count >= max_evals:
            break
    return current


def global_capacity_copacked_streaming_chunks(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
    axis: Any,
    task_range_offsets: tuple[int, ...],
    task_blocks: tuple[int, ...],
) -> tuple[tuple[StreamingChunk, ...], ...]:
    """Pack independent ITER chunks by modeled CTA work, not chunk count.

    A one-chunk-per-CTA partition strands the CTAs assigned to short tasks
    while long-task trees are still producing leaves.  This construction lets
    adjacent task ranges share a CTA queue.  It retains a single owner for
    every chunk, never places two live chunks of the same task on one CTA, and
    uses only topology, range, and cost-model contracts.

    The longest modeled trees receive a one-block tighter leaf cap when that
    does not increase their tree height.  The remaining work is co-packed into
    the same CTA budget, so reducing the critical leaf time does not require
    dropping a task or oversubscribing the persistent grid.
    """

    if not task_blocks:
        return ()
    model = current_scheduler_config().cost_model
    if model.iter_per_block_us <= 0.0:
        raise ValueError("global capacity co-packing requires a positive ITER per-block cost")
    total_blocks = sum(task_blocks)
    average_capacity_blocks = max(
        1,
        math.ceil(total_blocks / topology.sm_count),
    )

    def modeled_root_tail(blocks: int) -> float:
        part_count = max(1, math.ceil(blocks / average_capacity_blocks))
        leaf_blocks = max(1, math.ceil(blocks / part_count))
        return (
            model.iter_base_us
            + leaf_blocks * model.iter_per_block_us
            + (part_count - 1).bit_length() * model.reduce_single_us
            + ((part_count - 1) // topology.cluster_size) * model.hbm_comm_us
        )

    tails = tuple(modeled_root_tail(blocks) for blocks in task_blocks)
    critical_tail = max(tails)
    critical_tasks = frozenset(task_id for task_id, tail in enumerate(tails) if tail >= critical_tail - model.iter_per_block_us - 1e-9)
    tighter_capacity_blocks = max(1, average_capacity_blocks - 1)

    def task_chunk_cap(task_id: int) -> int | None:
        if task_id not in critical_tasks or tighter_capacity_blocks <= 0:
            return None
        blocks = task_blocks[task_id]
        ordinary_parts = math.ceil(blocks / average_capacity_blocks)
        tighter_parts = math.ceil(blocks / tighter_capacity_blocks)
        if (tighter_parts - 1).bit_length() > (ordinary_parts - 1).bit_length():
            return None
        return tighter_capacity_blocks

    task_order = tuple(
        sorted(
            range(len(task_blocks)),
            key=lambda task_id: (-task_blocks[task_id], task_id),
        )
    )

    def pack(
        target_us: float,
        *,
        materialize: bool,
    ) -> tuple[int, list[list[tuple[int, int]]]]:
        bins: list[list[tuple[int, int]]] = []
        bin_costs_us: list[float] = []
        current: list[tuple[int, int]] = []
        current_us = 0.0

        def close_current() -> None:
            nonlocal current, current_us
            if current:
                bins.append(current)
                bin_costs_us.append(current_us)
                current = []
                current_us = 0.0

        for task_id in task_order:
            remaining_blocks = task_blocks[task_id]
            chunk_cap = task_chunk_cap(task_id)
            while remaining_blocks > 0:
                if any(item_task_id == task_id for item_task_id, _ in current):
                    close_current()
                whole_task_fits_one_cta = (
                    remaining_blocks == task_blocks[task_id]
                    and (chunk_cap is None or remaining_blocks <= chunk_cap)
                    and (model.iter_base_us + remaining_blocks * model.iter_per_block_us + model.finalize_us <= target_us + 1e-9)
                )
                if whole_task_fits_one_cta:
                    whole_task_us = model.iter_base_us + remaining_blocks * model.iter_per_block_us + model.finalize_us
                    reusable_bin = max(
                        (bin_id for bin_id, bin_us in enumerate(bin_costs_us) if bin_us + whole_task_us <= target_us + 1e-9),
                        key=lambda bin_id: (bin_costs_us[bin_id], -bin_id),
                        default=None,
                    )
                    if reusable_bin is not None:
                        # A complete one-owner task has no partial to hand off,
                        # so it can safely fill an earlier CTA's modeled slack.
                        # Best-fit preserves larger holes for later tasks while
                        # avoiding the artificial hot tail produced by a
                        # strictly forward-only bin packer.
                        bins[reusable_bin].append((task_id, remaining_blocks))
                        bin_costs_us[reusable_bin] += whole_task_us
                        remaining_blocks = 0
                        continue
                if current and whole_task_fits_one_cta:
                    remaining_current_blocks = math.floor(
                        (target_us - current_us - model.iter_base_us - model.finalize_us + 1e-9) / model.iter_per_block_us
                    )
                    if remaining_current_blocks < remaining_blocks:
                        # Do not create a fixed-size partial handoff merely to
                        # fill the tail of this CTA bin.  Keeping a task whole
                        # trades at most the modeled bin slack for eliminating
                        # one reduce and, potentially, one cross-cluster HBM
                        # transfer.  The outer capacity search raises the bin
                        # target if whole-task packing otherwise needs too many
                        # CTAs.
                        close_current()
                        continue
                available_blocks = math.floor(
                    (target_us - current_us - model.iter_base_us - (model.finalize_us if whole_task_fits_one_cta else 0.0) + 1e-9)
                    / model.iter_per_block_us
                )
                if chunk_cap is not None:
                    available_blocks = min(available_blocks, chunk_cap)
                if available_blocks <= 0:
                    close_current()
                    if len(bins) >= topology.sm_count:
                        return len(bins) + 1, bins
                    continue
                chunk_blocks = min(remaining_blocks, available_blocks)
                current.append((task_id, chunk_blocks))
                current_us += (
                    model.iter_base_us + chunk_blocks * model.iter_per_block_us + (model.finalize_us if whole_task_fits_one_cta else 0.0)
                )
                remaining_blocks -= chunk_blocks
                if len(bins) + (1 if current else 0) > topology.sm_count:
                    return len(bins) + 1, bins
        close_current()
        if not materialize:
            return len(bins), []
        return len(bins), bins

    lower_us = max(
        model.iter_base_us + model.iter_per_block_us,
        sum(model.iter_base_us + blocks * model.iter_per_block_us + model.finalize_us for blocks in task_blocks) / topology.sm_count,
        max(
            model.iter_base_us
            + min(
                blocks,
                task_chunk_cap(task_id) or average_capacity_blocks,
            )
            * model.iter_per_block_us
            + (model.finalize_us if blocks <= (task_chunk_cap(task_id) or average_capacity_blocks) else 0.0)
            for task_id, blocks in enumerate(task_blocks)
        ),
    )
    upper_us = sum(model.iter_base_us + blocks * model.iter_per_block_us + model.finalize_us for blocks in task_blocks)
    for _ in range(64):
        middle_us = (lower_us + upper_us) * 0.5
        bin_count, _ = pack(middle_us, materialize=False)
        if bin_count <= topology.sm_count:
            upper_us = middle_us
        else:
            lower_us = middle_us
    bin_count, packed_bins = pack(upper_us + 1e-7, materialize=True)
    if bin_count > topology.sm_count:
        raise RuntimeError(
            f"global capacity co-packing could not fit the modeled work into the CTA grid: bins={bin_count}, ctas={topology.sm_count}"
        )

    task_ranges: list[list[tuple[int, TaskRange]]] = [[] for _ in task_range_lengths]
    task_block_cursors = [0 for _ in task_range_lengths]
    for sm_id, packed_items in enumerate(packed_bins):
        for task_id, chunk_blocks in packed_items:
            begin_block = task_block_cursors[task_id]
            end_block = begin_block + chunk_blocks
            begin = min(task_range_lengths[task_id], begin_block * block_size)
            end = min(task_range_lengths[task_id], end_block * block_size)
            if begin >= end:
                raise RuntimeError(f"global capacity co-packing produced an empty task range: task={task_id}, begin={begin}, end={end}")
            task_ranges[task_id].append(
                (
                    sm_id,
                    TaskRange(axis=axis, begin=begin, end=end),
                )
            )
            task_block_cursors[task_id] = end_block

    result: list[tuple[StreamingChunk, ...]] = []
    for task_id, ranges in enumerate(task_ranges):
        if (
            not ranges
            or ranges[0][1].begin != 0
            or ranges[-1][1].end != task_range_lengths[task_id]
            or any(left[1].end != right[1].begin for left, right in zip(ranges, ranges[1:]))
        ):
            raise RuntimeError(f"global capacity co-packing does not exactly tile task {task_id}")
        range_offset = task_range_offsets[task_id]
        part_count = len(ranges)
        result.append(
            tuple(
                StreamingChunk(
                    task_id=task_id,
                    sm_id=sm_id,
                    task_range=offset_task_range(task_range, range_offset),
                    part_index=part_index,
                    part_count=part_count,
                )
                for part_index, (sm_id, task_range) in enumerate(ranges)
            )
        )
    return tuple(result)


def global_capacity_streaming_chunks(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
    axis: Any,
    task_range_offsets: tuple[int, ...],
) -> tuple[tuple[StreamingChunk, ...], ...]:
    """Place at most one balanced ITER chunk on each CTA.

    This is the low-task-count dual of the ordinary cluster-local packer.  It
    first finds the smallest per-chunk tile capacity whose rounded task
    partitions fit the whole CTA grid.  It then keeps each task in as few
    clusters as cluster capacity permits.  The construction uses only range,
    topology, and block-size contracts; reduction placement and transport are
    still decided by the normal streaming-tree and joint schedulers.
    """

    task_blocks = tuple(max(1, math.ceil(length / block_size)) for length in task_range_lengths)
    if len(task_range_lengths) > topology.sm_count or config_flag("global_capacity_copacked_chunks"):
        return global_capacity_copacked_streaming_chunks(
            topology,
            task_range_lengths,
            block_size=block_size,
            axis=axis,
            task_range_offsets=task_range_offsets,
            task_blocks=task_blocks,
        )
    total_blocks = sum(task_blocks)
    minimum_capacity = max(1, math.ceil(total_blocks / topology.sm_count))
    capacity_blocks = next(
        capacity
        for capacity in range(minimum_capacity, max(task_blocks) + 1)
        if sum(math.ceil(blocks / capacity) for blocks in task_blocks) <= topology.sm_count
    )

    part_counts = [math.ceil(blocks / capacity_blocks) for blocks in task_blocks]
    if config_flag("global_capacity_minimax_chunks"):
        model = current_scheduler_config().cost_model

        def root_tail_us(blocks: int, part_count: int) -> float:
            ready_blocks = max(1, math.ceil(blocks / part_count))
            tree_height = (part_count - 1).bit_length()
            crossed_cluster_boundaries = (part_count - 1) // topology.cluster_size
            return (
                model.iter_base_us
                + ready_blocks * model.iter_per_block_us
                + tree_height * model.reduce_single_us
                + crossed_cluster_boundaries * model.hbm_comm_us
            )

        scores_by_task = tuple(
            tuple(root_tail_us(blocks, part_count) for part_count in range(1, min(blocks, topology.sm_count) + 1)) for blocks in task_blocks
        )
        thresholds = sorted({score for task_scores in scores_by_task for score in task_scores})
        low = 0
        high = len(thresholds) - 1
        best_counts: list[int] | None = None
        while low <= high:
            middle = (low + high) // 2
            threshold = thresholds[middle]
            candidate_counts: list[int] = []
            for task_scores in scores_by_task:
                part_count = next(
                    (index + 1 for index, score in enumerate(task_scores) if score <= threshold + 1e-9),
                    None,
                )
                if part_count is None:
                    candidate_counts = []
                    break
                candidate_counts.append(part_count)
            if candidate_counts and sum(candidate_counts) <= topology.sm_count:
                best_counts = candidate_counts
                high = middle - 1
            else:
                low = middle + 1
        if best_counts is None:
            raise RuntimeError("global capacity minimax chunking found no CTA-feasible partition")
        part_counts = best_counts

    return refine_global_capacity_part_counts_by_replay(
        topology,
        task_range_lengths,
        block_size=block_size,
        axis=axis,
        task_range_offsets=task_range_offsets,
        task_blocks=task_blocks,
        initial_part_counts=part_counts,
    )


def load_balanced_cross_cluster_streaming_chunks(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
    axis: Any,
    task_range_offsets: tuple[int, ...] | None = None,
) -> tuple[tuple[StreamingChunk, ...], ...]:
    resolved_task_range_offsets = task_range_offsets or (0,) * len(task_range_lengths)
    total_blocks = sum(math.ceil(length / block_size) for length in task_range_lengths)
    target_cluster_blocks = max(1, math.ceil(total_blocks / topology.cluster_count))
    cluster_loads = [0 for _ in range(topology.cluster_count)]
    segments_by_cluster: list[list[StreamingSegment]] = [[] for _ in range(topology.cluster_count)]

    for task_id in sorted(range(len(task_range_lengths)), key=lambda index: (-task_range_lengths[index], index)):
        length = task_range_lengths[task_id]
        task_blocks = math.ceil(length / block_size)
        split_count = min(
            topology.cluster_count,
            max(1, math.ceil(task_blocks / target_cluster_blocks)),
        )
        begin_block = 0
        for part_index in range(split_count):
            remaining_parts = split_count - part_index
            remaining_blocks = task_blocks - begin_block
            segment_blocks = max(1, math.ceil(remaining_blocks / remaining_parts))
            end_block = min(task_blocks, begin_block + segment_blocks)
            begin = min(length, begin_block * block_size)
            end = min(length, end_block * block_size)
            cluster_id = min(
                range(topology.cluster_count),
                key=lambda index: (
                    cluster_loads[index] / max(1, len(get_cluster_sms(topology, index))),
                    cluster_loads[index],
                    index,
                ),
            )
            segments_by_cluster[cluster_id].append(
                StreamingSegment(
                    task_id=task_id,
                    cluster_id=cluster_id,
                    begin=begin,
                    end=end,
                )
            )
            cluster_loads[cluster_id] += end_block - begin_block
            begin_block = end_block

    task_ranges: list[list[tuple[int, TaskRange]]] = [[] for _ in task_range_lengths]
    for cluster_id, segments in enumerate(segments_by_cluster):
        if not segments:
            continue
        sms = get_cluster_sms(topology, cluster_id)
        if not sms:
            raise ValueError(f"Dataflow scheduler found empty cluster {cluster_id} in topology {topology!r}")

        cluster_work_blocks = sum(math.ceil((segment.end - segment.begin) / block_size) for segment in segments)
        target_sm_work_blocks = max(1, math.ceil(cluster_work_blocks / len(sms)))
        sm_index = 0
        sm_work_blocks = 0

        for segment in sorted(
            segments,
            key=lambda item: (
                -math.ceil((item.end - item.begin) / block_size),
                item.task_id,
                item.begin,
                item.end,
            ),
        ):
            begin = segment.begin
            while begin < segment.end:
                if sm_index >= len(sms):
                    sm_index = len(sms) - 1
                sm_id = sms[sm_index]
                remaining = segment.end - begin
                remaining_blocks = math.ceil(remaining / block_size)

                if sm_index == len(sms) - 1:
                    end = segment.end
                else:
                    capacity_blocks = target_sm_work_blocks - sm_work_blocks
                    if capacity_blocks <= 0:
                        sm_index += 1
                        sm_work_blocks = 0
                        continue
                    if remaining_blocks <= capacity_blocks:
                        end = segment.end
                    else:
                        chunk_blocks = max(1, min(remaining_blocks, capacity_blocks))
                        end = min(segment.end, begin + chunk_blocks * block_size)

                append_streaming_range(
                    task_ranges[segment.task_id],
                    sm_id=sm_id,
                    axis=axis,
                    begin=begin,
                    end=end,
                )
                sm_work_blocks += math.ceil((end - begin) / block_size)
                begin = end

                if begin < segment.end and sm_index < len(sms) - 1 and sm_work_blocks >= target_sm_work_blocks:
                    sm_index += 1
                    sm_work_blocks = 0

    chunks_by_task: list[tuple[StreamingChunk, ...]] = []
    for task_id, ranges in enumerate(task_ranges):
        if not ranges:
            raise ValueError(f"Dataflow cross-cluster streaming scheduler produced no chunks for task {task_id}")
        merged_ranges: list[tuple[int, TaskRange]] = []
        for sm_id, task_range in sorted(ranges, key=lambda item: (item[1].begin, item[1].end, item[0])):
            append_streaming_range(
                merged_ranges,
                sm_id=sm_id,
                axis=axis,
                begin=task_range.begin,
                end=task_range.end,
            )
        part_count = len(merged_ranges)
        range_offset = resolved_task_range_offsets[task_id]
        chunks_by_task.append(
            tuple(
                StreamingChunk(
                    task_id=task_id,
                    sm_id=sm_id,
                    task_range=offset_task_range(task_range, range_offset),
                    part_index=part_index,
                    part_count=part_count,
                )
                for part_index, (sm_id, task_range) in enumerate(merged_ranges)
            )
        )
    return tuple(chunks_by_task)


def manual_cluster_segments_from_config(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
) -> tuple[StreamingSegment, ...] | None:
    value = config_string("manual_cluster_segments", None)
    if value is None or value.strip() == "":
        return None

    segments_by_task: dict[int, list[StreamingSegment]] = {}
    try:
        task_entries = [entry.strip() for entry in value.split(";") if entry.strip()]
        for task_entry in task_entries:
            task_text, segment_text = task_entry.split(":", 1)
            task_id = int(task_text.strip())
            if task_id < 0 or task_id >= len(task_range_lengths):
                raise ValueError(f"task id {task_id} out of range")
            if task_id in segments_by_task:
                raise ValueError(f"duplicate task id {task_id}")
            task_segments: list[StreamingSegment] = []
            for segment_entry in segment_text.split(","):
                stripped_segment = segment_entry.strip()
                if not stripped_segment:
                    continue
                cluster_text, range_text = stripped_segment.split(":", 1)
                begin_text, end_text = range_text.split("-", 1)
                cluster_id = int(cluster_text.strip())
                begin_block = int(begin_text.strip())
                end_block = int(end_text.strip())
                if cluster_id < 0 or cluster_id >= topology.cluster_count:
                    raise ValueError(f"cluster id {cluster_id} out of range")
                if begin_block < 0 or end_block <= begin_block:
                    raise ValueError(f"invalid block range {begin_block}-{end_block} for task {task_id}")
                task_length = task_range_lengths[task_id]
                begin = min(task_length, begin_block * block_size)
                end = min(task_length, end_block * block_size)
                if end <= begin:
                    raise ValueError(f"empty token range from block range {begin_block}-{end_block} for task {task_id}")
                task_segments.append(
                    StreamingSegment(
                        task_id=task_id,
                        cluster_id=cluster_id,
                        begin=begin,
                        end=end,
                    )
                )
            segments_by_task[task_id] = task_segments
    except ValueError as exc:
        raise ValueError(
            "manual_cluster_segments must use "
            "'task:cluster:begin_block-end_block[,cluster:begin_block-end_block];...' "
            f"syntax, got {value!r}"
        ) from exc

    if sorted(segments_by_task) != list(range(len(task_range_lengths))):
        raise ValueError(
            "manual_cluster_segments must cover every task exactly once; "
            f"got task ids {sorted(segments_by_task)}, expected {list(range(len(task_range_lengths)))}"
        )

    segments: list[StreamingSegment] = []
    for task_id in range(len(task_range_lengths)):
        task_segments = sorted(
            segments_by_task[task_id],
            key=lambda segment: (segment.begin, segment.end, segment.cluster_id),
        )
        expected_begin = 0
        task_length = task_range_lengths[task_id]
        for segment in task_segments:
            if segment.begin != expected_begin:
                raise ValueError(
                    "manual_cluster_segments must exactly tile each task range; "
                    f"task {task_id} expected begin {expected_begin}, got {segment.begin}"
                )
            expected_begin = segment.end
            segments.append(segment)
        if expected_begin != task_length:
            raise ValueError(
                "manual_cluster_segments must exactly tile each task range; "
                f"task {task_id} ended at {expected_begin}, expected {task_length}"
            )
    return tuple(segments)


def segments_to_streaming_chunks(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
    axis: Any,
    task_range_offsets: tuple[int, ...],
    segments: Sequence[StreamingSegment],
    require_all_tasks: bool,
    error_prefix: str,
) -> tuple[tuple[StreamingChunk, ...], ...]:
    merged_segments = merge_adjacent_streaming_segments(segments)
    segments_by_cluster: list[list[StreamingSegment]] = [[] for _ in range(topology.cluster_count)]
    for segment in merged_segments:
        segments_by_cluster[segment.cluster_id].append(segment)

    task_ranges: list[list[tuple[int, TaskRange]]] = [[] for _ in task_range_lengths]
    avoid_tiny_copack = config_flag("avoid_tiny_segment_copack")
    tiny_copack_blocks = max(0, config_int("tiny_segment_copack_blocks", 2))
    even_segment_chunks = config_flag("even_segment_chunks")
    for cluster_id, cluster_segments in enumerate(segments_by_cluster):
        if not cluster_segments:
            continue
        sms = get_cluster_sms(topology, cluster_id)
        if not sms:
            raise ValueError(f"Dataflow scheduler found empty cluster {cluster_id} in topology {topology!r}")

        cluster_work_blocks = sum(math.ceil((segment.end - segment.begin) / block_size) for segment in cluster_segments)
        target_sm_work_blocks = max(1, math.ceil(cluster_work_blocks / len(sms)))
        sm_index = 0
        sm_work_blocks = 0
        for segment in sorted(
            cluster_segments,
            key=lambda item: (
                -math.ceil((item.end - item.begin) / block_size),
                item.task_id,
                item.begin,
                item.end,
            ),
        ):
            if even_segment_chunks:
                segment_blocks = math.ceil((segment.end - segment.begin) / block_size)
                desired_chunks = max(1, math.ceil(segment_blocks / target_sm_work_blocks))
                begin_block = 0
                for chunk_index in range(desired_chunks):
                    if sm_index >= len(sms):
                        sm_index = len(sms) - 1
                    sm_id = sms[sm_index]
                    remaining_blocks = segment_blocks - begin_block
                    remaining_chunks = desired_chunks - chunk_index
                    chunk_blocks = max(1, math.ceil(remaining_blocks / remaining_chunks))
                    end_block = min(segment_blocks, begin_block + chunk_blocks)
                    begin = min(segment.end, segment.begin + begin_block * block_size)
                    end = min(segment.end, segment.begin + end_block * block_size)
                    append_streaming_range(
                        task_ranges[segment.task_id],
                        sm_id=sm_id,
                        axis=axis,
                        begin=begin,
                        end=end,
                    )
                    begin_block = end_block
                    sm_index += 1
                    sm_work_blocks = 0
                continue

            begin = segment.begin
            while begin < segment.end:
                if sm_index >= len(sms):
                    sm_index = len(sms) - 1
                sm_id = sms[sm_index]
                remaining = segment.end - begin
                remaining_blocks = math.ceil(remaining / block_size)
                if sm_index == len(sms) - 1:
                    end = segment.end
                else:
                    capacity_blocks = target_sm_work_blocks - sm_work_blocks
                    if capacity_blocks <= 0:
                        sm_index += 1
                        sm_work_blocks = 0
                        continue
                    if remaining_blocks <= capacity_blocks:
                        end = segment.end
                    else:
                        chunk_blocks = max(1, min(remaining_blocks, capacity_blocks))
                        end = min(segment.end, begin + chunk_blocks * block_size)

                append_streaming_range(
                    task_ranges[segment.task_id],
                    sm_id=sm_id,
                    axis=axis,
                    begin=begin,
                    end=end,
                )
                chunk_blocks = math.ceil((end - begin) / block_size)
                chunk_begin = begin
                sm_work_blocks += chunk_blocks
                begin = end
                if (
                    avoid_tiny_copack
                    and tiny_copack_blocks > 0
                    and chunk_begin > segment.begin
                    and begin == segment.end
                    and chunk_blocks <= tiny_copack_blocks
                    and sm_index < len(sms) - 1
                ):
                    sm_index += 1
                    sm_work_blocks = 0
                    continue
                if begin < segment.end and sm_index < len(sms) - 1 and sm_work_blocks >= target_sm_work_blocks:
                    sm_index += 1
                    sm_work_blocks = 0

    merge_tiny_final_streaming_ranges(
        task_ranges,
        topology=topology,
        block_size=block_size,
    )

    chunks_by_task: list[tuple[StreamingChunk, ...]] = []
    for task_id, ranges in enumerate(task_ranges):
        if not ranges:
            if require_all_tasks:
                raise ValueError(f"{error_prefix} produced no chunks for task {task_id}")
            chunks_by_task.append(())
            continue
        merged_ranges: list[tuple[int, TaskRange]] = []
        for sm_id, task_range in sorted(ranges, key=lambda item: (item[1].begin, item[1].end, item[0])):
            append_streaming_range(
                merged_ranges,
                sm_id=sm_id,
                axis=axis,
                begin=task_range.begin,
                end=task_range.end,
            )
        part_count = len(merged_ranges)
        range_offset = task_range_offsets[task_id]
        chunks_by_task.append(
            tuple(
                StreamingChunk(
                    task_id=task_id,
                    sm_id=sm_id,
                    task_range=offset_task_range(task_range, range_offset),
                    part_index=part_index,
                    part_count=part_count,
                )
                for part_index, (sm_id, task_range) in enumerate(merged_ranges)
            )
        )
    return tuple(chunks_by_task)


def merge_tiny_final_streaming_ranges(
    task_ranges: list[list[tuple[int, TaskRange]]],
    *,
    topology: GPUTopology,
    block_size: int,
) -> None:
    if not config_flag("merge_tiny_final_chunk"):
        return
    tiny_blocks = max(0, config_int("tiny_final_chunk_blocks", 4))
    if tiny_blocks == 0:
        return
    max_merged_blocks = max(0, config_int("tiny_final_chunk_max_merged_blocks", 0))

    for ranges in task_ranges:
        if len(ranges) < 2:
            continue
        ranges.sort(key=lambda item: (item[1].begin, item[1].end, item[0]))
        prev_sm, prev_range = ranges[-2]
        tail_sm, tail_range = ranges[-1]
        if prev_range.end != tail_range.begin:
            continue
        if not topology.same_cluster(prev_sm, tail_sm):
            continue
        tail_blocks = math.ceil(tail_range.length / block_size)
        if tail_blocks > tiny_blocks:
            continue
        merged_blocks = math.ceil((tail_range.end - prev_range.begin) / block_size)
        if max_merged_blocks > 0 and merged_blocks > max_merged_blocks:
            continue
        ranges[-2] = (
            prev_sm,
            TaskRange(axis=prev_range.axis, begin=prev_range.begin, end=tail_range.end),
        )
        ranges.pop()


def merge_adjacent_streaming_segments(
    segments: Sequence[StreamingSegment],
) -> tuple[StreamingSegment, ...]:
    merged: list[StreamingSegment] = []
    for segment in sorted(
        segments,
        key=lambda item: (item.task_id, item.cluster_id, item.begin, item.end),
    ):
        if (
            merged
            and merged[-1].task_id == segment.task_id
            and merged[-1].cluster_id == segment.cluster_id
            and merged[-1].end == segment.begin
        ):
            previous = merged[-1]
            merged[-1] = StreamingSegment(
                task_id=previous.task_id,
                cluster_id=previous.cluster_id,
                begin=previous.begin,
                end=segment.end,
            )
            continue
        merged.append(segment)
    return tuple(merged)


def replay_streaming_tree_score(
    topology: GPUTopology,
    chunks_by_task: Sequence[Sequence[StreamingChunk]],
    *,
    block_size: int,
    streaming_tree_consumer: str,
    hierarchical_cross_cluster: bool,
    ready_time_tree: bool,
) -> StreamingDagReplayScore:
    construction_sm_ready = {sm_id: 0.0 for sm_id in range(topology.sm_count)}
    construction_slot_ready: dict[int, float] = {}
    construction_slot_tree_level: dict[int, int] = {}
    next_slot_id = 0
    next_instruction_id = 0
    replay_instructions: list[StreamingReplayInstruction] = []
    split_local_root_slots: list[tuple[int, tuple[int, ...]]] = []
    ready_time_recv_weight = config_float("ready_time_recv_weight", 0.6)
    ready_time_queue_weight = config_float("ready_time_queue_weight", 0.1)
    ready_time_balance_weight = config_float("ready_time_balance_weight", 0.0)
    ready_time_cluster_comm_weight = config_float("ready_time_cluster_comm_weight", 0.25)
    direct_leaf_acc = config_flag("direct_leaf_acc")

    def emit_replay_instruction(
        *,
        task_id: int,
        sm_id: int,
        input_slots: Sequence[int],
        output_slot: int | None,
        duration_us: float,
        queue_key: tuple[int, int, int, int],
        level0_sort_us: float = 0.0,
    ) -> None:
        nonlocal next_instruction_id
        replay_instructions.append(
            StreamingReplayInstruction(
                instruction_id=next_instruction_id,
                task_id=task_id,
                sm_id=sm_id,
                input_slots=tuple(input_slots),
                output_slot=output_slot,
                duration_us=duration_us,
                queue_key=queue_key,
                level0_sort_us=level0_sort_us,
            )
        )
        next_instruction_id += 1

    def reserve_construction_instruction(
        *,
        sm_id: int,
        input_sources: Sequence[tuple[int, int]],
        output_slot: int | None,
        duration_us: float,
    ) -> float:
        input_ready = max(
            (
                construction_slot_ready.get(slot_id, 0.0) + estimated_streaming_comm_cost_us(topology, producer_sm, sm_id)
                for slot_id, producer_sm in input_sources
            ),
            default=0.0,
        )
        queue_ready = construction_sm_ready.get(sm_id, 0.0)
        finish = max(queue_ready, input_ready) + duration_us
        construction_sm_ready[sm_id] = finish
        if output_slot is not None:
            construction_slot_ready[output_slot] = finish
        return finish

    def choose_reduce_sm(
        *,
        left_slot: int,
        left_sm: int,
        right_slot: int,
        right_sm: int,
        prefer_late_input: bool = False,
    ) -> int:
        fixed_sm = right_sm if streaming_tree_consumer == "right" else left_sm
        if not ready_time_tree:
            return fixed_sm
        candidates = streaming_tree_reduce_candidate_sms(
            topology,
            left_sm=left_sm,
            right_sm=right_sm,
            prefer_late_input=prefer_late_input,
        )
        candidates = prune_ready_time_reduce_candidates(
            candidates,
            ready_by_sm=construction_sm_ready,
            left_sm=left_sm,
            right_sm=right_sm,
        )
        if config_flag("reduce_higher_level_producer"):
            left_level = construction_slot_tree_level.get(left_slot, 0)
            right_level = construction_slot_tree_level.get(right_slot, 0)
            if left_level != right_level:
                return left_sm if left_level > right_level else right_sm

        def candidate_score(sm_id: int) -> tuple[float, float, float, int]:
            left_comm = estimated_streaming_comm_cost_us(topology, left_sm, sm_id)
            right_comm = estimated_streaming_comm_cost_us(topology, right_sm, sm_id)
            left_ready = construction_slot_ready.get(left_slot, 0.0) + left_comm
            right_ready = construction_slot_ready.get(right_slot, 0.0) + right_comm
            input_ready = max(left_ready, right_ready)
            queue_ready = construction_sm_ready.get(sm_id, 0.0)
            recv_wait = max(0.0, input_ready - queue_ready)
            finish = max(queue_ready, input_ready) + estimated_streaming_reduce_cost_us(2)
            projected_max = max(finish, max(construction_sm_ready.values(), default=0.0))
            remote_comm_cost = (left_comm if left_sm != sm_id else 0.0) + (right_comm if right_sm != sm_id else 0.0)
            score = (
                finish
                + ready_time_recv_weight * recv_wait
                + ready_time_queue_weight * queue_ready
                + ready_time_balance_weight * projected_max
                + ready_time_cluster_comm_weight * remote_comm_cost
            )
            if prefer_late_input and (
                (config_flag("cross_cluster_root_global_candidates") and not topology.same_cluster(left_sm, right_sm))
                or (config_flag("root_reduce_cluster_candidates") and topology.same_cluster(left_sm, right_sm))
            ):
                return (score, queue_ready, recv_wait, -sm_id)
            return (score, recv_wait, queue_ready, -sm_id)

        return min(candidates, key=candidate_score)

    def emit_tree(
        *,
        task_id: int,
        nodes: list[tuple[int, int]],
        start_level: int,
    ) -> tuple[tuple[int, int], int]:
        nonlocal next_slot_id
        if config_flag("ordered_interval_tree") and len(nodes) > 1:
            tree_plan = ordered_reduction_tree_plan(
                topology,
                tuple((sm_id, construction_slot_ready.get(slot_id, 0.0)) for slot_id, sm_id in nodes),
                consumer_side=streaming_tree_consumer,
                adaptive_consumer=config_flag("ordered_tree_adaptive_consumer"),
            )

            def emit_ordered(
                plan: OrderedReductionTreePlan,
            ) -> tuple[int, int]:
                nonlocal next_slot_id
                if plan.is_leaf:
                    return nodes[plan.begin]
                max_reduce_arity = max(
                    2,
                    config_int("ordered_tree_max_reduce_arity", 2),
                )
                frontier = ordered_reduction_fused_frontier(
                    plan,
                    max_reduce_arity=max_reduce_arity,
                    max_resident_remote_inputs=joint_max_resident_remote_inputs(),
                )
                inputs = tuple(emit_ordered(child) for child in frontier)
                reduce_sm = plan.consumer_sm
                acc_slot_id = next_slot_id
                next_slot_id += 1
                level = start_level + plan.height - 1
                emit_replay_instruction(
                    task_id=task_id,
                    sm_id=reduce_sm,
                    input_slots=tuple(slot_id for slot_id, _ in inputs),
                    output_slot=acc_slot_id,
                    duration_us=estimated_streaming_reduce_cost_us(len(inputs)),
                    queue_key=(level, task_id, plan.begin, 0),
                )
                reserve_construction_instruction(
                    sm_id=reduce_sm,
                    input_sources=inputs,
                    output_slot=acc_slot_id,
                    duration_us=estimated_streaming_reduce_cost_us(len(inputs)),
                )
                construction_slot_tree_level[acc_slot_id] = level
                return acc_slot_id, reduce_sm

            return emit_ordered(tree_plan), start_level + tree_plan.height

        level = start_level
        current_nodes = nodes
        while len(current_nodes) > 1:
            next_nodes: list[tuple[int, int]] = []
            pair_index = 0
            for index in range(0, len(current_nodes), 2):
                if index + 1 >= len(current_nodes):
                    next_nodes.append(current_nodes[index])
                    continue

                left_slot, left_sm = current_nodes[index]
                right_slot, right_sm = current_nodes[index + 1]
                reduce_sm = choose_reduce_sm(
                    left_slot=left_slot,
                    left_sm=left_sm,
                    right_slot=right_slot,
                    right_sm=right_sm,
                    prefer_late_input=len(current_nodes) == 2,
                )
                acc_slot_id = next_slot_id
                next_slot_id += 1
                emit_replay_instruction(
                    task_id=task_id,
                    sm_id=reduce_sm,
                    input_slots=(left_slot, right_slot),
                    output_slot=acc_slot_id,
                    duration_us=estimated_streaming_reduce_cost_us(2),
                    queue_key=(level, task_id, pair_index, 0),
                )
                reserve_construction_instruction(
                    sm_id=reduce_sm,
                    input_sources=((left_slot, left_sm), (right_slot, right_sm)),
                    output_slot=acc_slot_id,
                    duration_us=estimated_streaming_reduce_cost_us(2),
                )
                construction_slot_tree_level[acc_slot_id] = level
                next_nodes.append((acc_slot_id, reduce_sm))
                pair_index += 1
            current_nodes = next_nodes
            level += 1
        return current_nodes[0], level

    for task_id, unordered_chunks in enumerate(chunks_by_task):
        chunks = logical_streaming_chunk_order(unordered_chunks)
        if not chunks:
            continue
        leaf_nodes: list[tuple[int, int]] = []
        for chunk in chunks:
            partial_slot_id = next_slot_id
            next_slot_id += 1
            iter_duration = estimated_streaming_iter_cost_us(chunk.task_range, block_size)
            emit_replay_instruction(
                task_id=task_id,
                sm_id=chunk.sm_id,
                input_slots=(),
                output_slot=partial_slot_id,
                duration_us=iter_duration,
                queue_key=(0, task_id, chunk.part_index, 0),
                level0_sort_us=iter_duration,
            )
            reserve_construction_instruction(
                sm_id=chunk.sm_id,
                input_sources=(),
                output_slot=partial_slot_id,
                duration_us=iter_duration,
            )
            construction_slot_tree_level[partial_slot_id] = 0
            if direct_leaf_acc:
                leaf_nodes.append((partial_slot_id, chunk.sm_id))
                continue

            acc_slot_id = next_slot_id
            next_slot_id += 1
            reduce_leaf_duration = estimated_streaming_reduce_cost_us(1)
            emit_replay_instruction(
                task_id=task_id,
                sm_id=chunk.sm_id,
                input_slots=(partial_slot_id,),
                output_slot=acc_slot_id,
                duration_us=reduce_leaf_duration,
                queue_key=(0, task_id, chunk.part_index, 1),
                level0_sort_us=iter_duration,
            )
            reserve_construction_instruction(
                sm_id=chunk.sm_id,
                input_sources=((partial_slot_id, chunk.sm_id),),
                output_slot=acc_slot_id,
                duration_us=reduce_leaf_duration,
            )
            construction_slot_tree_level[acc_slot_id] = 0
            leaf_nodes.append((acc_slot_id, chunk.sm_id))

        cluster_runs = ordered_cluster_run_bounds(
            topology,
            tuple(sm_id for _, sm_id in leaf_nodes),
        )
        if hierarchical_cross_cluster and len(cluster_runs) > 1:
            local_roots: list[tuple[int, int]] = []
            local_next_levels: list[int] = []
            for _, begin, end in cluster_runs:
                local_root, local_next_level = emit_tree(
                    task_id=task_id,
                    nodes=leaf_nodes[begin:end],
                    start_level=1,
                )
                local_roots.append(local_root)
                local_next_levels.append(local_next_level)
            split_local_root_slots.append((task_id, tuple(root_slot for root_slot, _ in local_roots)))
            (root_slot, root_sm), level = emit_tree(
                task_id=task_id,
                nodes=local_roots,
                start_level=max(local_next_levels, default=1),
            )
        else:
            (root_slot, root_sm), level = emit_tree(
                task_id=task_id,
                nodes=leaf_nodes,
                start_level=1,
            )

        finalize_duration = estimated_streaming_finalize_cost_us()
        emit_replay_instruction(
            task_id=task_id,
            sm_id=root_sm,
            input_slots=(root_slot,),
            output_slot=None,
            duration_us=finalize_duration,
            queue_key=(level, task_id, 0, 0),
        )
        reserve_construction_instruction(
            sm_id=root_sm,
            input_sources=((root_slot, root_sm),),
            output_slot=None,
            duration_us=finalize_duration,
        )

    queues_by_sm: dict[int, list[StreamingReplayInstruction]] = {}
    output_slot_to_sm: dict[int, int] = {}
    output_slot_to_instruction: dict[int, int] = {}
    replay_by_instruction_id = {instruction.instruction_id: instruction for instruction in replay_instructions}
    for instruction in replay_instructions:
        queues_by_sm.setdefault(instruction.sm_id, []).append(instruction)
        if instruction.output_slot is not None:
            output_slot_to_sm[instruction.output_slot] = instruction.sm_id
            output_slot_to_instruction[instruction.output_slot] = instruction.instruction_id

    level0_queue_order = config_string("level0_queue_order", "task").strip().lower()

    def replay_queue_sort_key(instruction: StreamingReplayInstruction) -> tuple[float, int, int, int, int]:
        level, task_id, part_index, sub_index = instruction.queue_key
        if level0_queue_order == "task_major":
            return (task_id, level, part_index, sub_index, 0)
        if level == 0 and level0_queue_order == "long_first":
            return (level, -instruction.level0_sort_us, task_id, part_index, sub_index)
        return (level, task_id, part_index, sub_index, 0)

    if level0_queue_order == "producer_ready":
        replay_by_id = {instruction.instruction_id: instruction for instruction in replay_instructions}
        ordered_node_ids = producer_ready_queue_order(
            topology,
            tuple(
                ProducerReadyQueueNode(
                    node_id=instruction.instruction_id,
                    sm_id=instruction.sm_id,
                    input_slots=instruction.input_slots,
                    output_slots=(() if instruction.output_slot is None else (instruction.output_slot,)),
                    duration_us=instruction.duration_us,
                    sort_key=instruction.queue_key + (instruction.instruction_id,),
                )
                for instruction in replay_instructions
            ),
        )
        queues_by_sm = {sm_id: [replay_by_id[node_id] for node_id in node_ids] for sm_id, node_ids in ordered_node_ids.items() if node_ids}
    else:
        for queue in queues_by_sm.values():
            queue.sort(key=replay_queue_sort_key)

    queue_indices = {sm_id: 0 for sm_id in queues_by_sm}
    sm_ready = {sm_id: 0.0 for sm_id in queues_by_sm}
    slot_ready: dict[int, float] = {}
    total_recv_wait_us = 0.0
    max_recv_wait_us = 0.0
    max_recv_instruction_id: int | None = None
    max_recv_task_id: int | None = None
    max_recv_sm_id: int | None = None
    max_recv_slot_id: int | None = None
    hbm_edges = 0
    last_instruction_by_sm: dict[int, int] = {}
    critical_predecessor: dict[int, int | None] = {}
    remaining = sum(len(queue) for queue in queues_by_sm.values())

    while remaining:
        progressed = False
        for sm_id in sorted(queues_by_sm):
            index = queue_indices[sm_id]
            queue = queues_by_sm[sm_id]
            if index >= len(queue):
                continue
            instruction = queue[index]
            input_ready_values: list[float] = []
            input_ready_slots: list[int] = []
            missing_dependency = False
            instruction_hbm_edges = 0
            for slot_id in instruction.input_slots:
                producer_sm = output_slot_to_sm.get(slot_id)
                if producer_sm is None:
                    input_ready_values.append(0.0)
                    input_ready_slots.append(slot_id)
                    continue
                if slot_id not in slot_ready:
                    missing_dependency = True
                    break
                input_ready_values.append(slot_ready[slot_id] + estimated_streaming_comm_cost_us(topology, producer_sm, sm_id))
                input_ready_slots.append(slot_id)
                if producer_sm != sm_id and not topology.same_cluster(producer_sm, sm_id):
                    instruction_hbm_edges += 1
            if missing_dependency:
                continue

            queue_ready = sm_ready.get(sm_id, 0.0)
            input_ready = max(input_ready_values, default=0.0)
            recv_wait = max(0.0, input_ready - queue_ready) if instruction.input_slots else 0.0
            finish = max(queue_ready, input_ready) + instruction.duration_us
            queue_predecessor = last_instruction_by_sm.get(sm_id)
            input_predecessor = None
            if input_ready_values:
                critical_input_index = max(
                    range(len(input_ready_values)),
                    key=input_ready_values.__getitem__,
                )
                input_predecessor = output_slot_to_instruction.get(input_ready_slots[critical_input_index])
            critical_predecessor[instruction.instruction_id] = (
                input_predecessor if input_predecessor is not None and input_ready > queue_ready + 1e-9 else queue_predecessor
            )
            sm_ready[sm_id] = finish
            last_instruction_by_sm[sm_id] = instruction.instruction_id
            if instruction.output_slot is not None:
                slot_ready[instruction.output_slot] = finish
            total_recv_wait_us += recv_wait
            if recv_wait > max_recv_wait_us:
                max_recv_wait_us = recv_wait
                max_recv_instruction_id = instruction.instruction_id
                max_recv_task_id = instruction.task_id
                max_recv_sm_id = instruction.sm_id
                max_recv_slot_id = (
                    input_ready_slots[
                        max(
                            range(len(input_ready_values)),
                            key=input_ready_values.__getitem__,
                        )
                    ]
                    if input_ready_values
                    else None
                )
            hbm_edges += instruction_hbm_edges
            queue_indices[sm_id] = index + 1
            remaining -= 1
            progressed = True
        if not progressed:
            blocked = [
                queues_by_sm[sm_id][queue_indices[sm_id]].instruction_id
                for sm_id in sorted(queues_by_sm)
                if queue_indices[sm_id] < len(queues_by_sm[sm_id])
            ]
            raise RuntimeError(f"Dataflow streaming-tree replay could not resolve dependencies for instructions {blocked[:8]}")

    active_finishes = sorted(finish for finish in sm_ready.values() if finish > 0.0)
    finish_by_sm_us = tuple(sm_ready.get(sm_id, 0.0) for sm_id in range(topology.sm_count))
    critical_leaf_owners: list[tuple[int, int]] = []
    if last_instruction_by_sm:
        terminal_sm = max(
            last_instruction_by_sm,
            key=lambda sm_id: (sm_ready.get(sm_id, 0.0), -sm_id),
        )
        current_instruction_id: int | None = last_instruction_by_sm[terminal_sm]
        visited_critical: set[int] = set()
        while current_instruction_id is not None and current_instruction_id not in visited_critical:
            visited_critical.add(current_instruction_id)
            critical_instruction = replay_by_instruction_id[current_instruction_id]
            if not critical_instruction.input_slots:
                critical_leaf_owners.append(
                    (
                        critical_instruction.task_id,
                        critical_instruction.sm_id,
                    )
                )
            current_instruction_id = critical_predecessor.get(current_instruction_id)
    instruction_count = len(replay_instructions)
    slot_count = sum(1 for instruction in replay_instructions if instruction.output_slot is not None)
    local_root_skews: list[float] = []
    for _task_id, root_slots in split_local_root_slots:
        ready_times = [slot_ready[slot_id] for slot_id in root_slots if slot_id in slot_ready]
        if len(ready_times) >= 2:
            local_root_skews.append(max(ready_times) - min(ready_times))
    if not active_finishes:
        return StreamingDagReplayScore(
            max_finish_us=0.0,
            p95_finish_us=0.0,
            finish_spread_us=0.0,
            total_recv_wait_us=0.0,
            max_recv_wait_us=0.0,
            hbm_edges=0,
            instruction_count=0,
            slot_count=0,
            max_local_root_skew_us=0.0,
            total_local_root_skew_us=0.0,
            split_task_count=0,
        )
    p95_index = min(len(active_finishes) - 1, math.ceil(0.95 * len(active_finishes)) - 1)
    return StreamingDagReplayScore(
        max_finish_us=active_finishes[-1],
        p95_finish_us=active_finishes[p95_index],
        finish_spread_us=active_finishes[-1] - active_finishes[0],
        total_recv_wait_us=total_recv_wait_us,
        max_recv_wait_us=max_recv_wait_us,
        hbm_edges=hbm_edges,
        instruction_count=instruction_count,
        slot_count=slot_count,
        max_local_root_skew_us=max(local_root_skews, default=0.0),
        total_local_root_skew_us=sum(local_root_skews),
        split_task_count=len(local_root_skews),
        max_recv_instruction_id=max_recv_instruction_id,
        max_recv_task_id=max_recv_task_id,
        max_recv_sm_id=max_recv_sm_id,
        max_recv_slot_id=max_recv_slot_id,
        finish_profile_us=tuple(reversed(active_finishes)),
        finish_by_sm_us=finish_by_sm_us,
        critical_leaf_owners=tuple(critical_leaf_owners),
    )


def replay_streaming_linear_score(
    topology: GPUTopology,
    chunks_by_task: Sequence[Sequence[StreamingChunk]],
    *,
    block_size: int,
) -> StreamingDagReplayScore:
    """Replay the migrating accumulator chain at its indivisible queue-group boundary."""

    nodes: list[ProducerReadyQueueNode] = []
    node_iter_lengths: dict[int, int] = {}
    stages_by_node: dict[int, tuple[StreamingReplayInstruction, ...]] = {}
    next_node_id = 0
    next_instruction_id = 0
    next_slot_id = 0
    iter_chunk_count = 0
    for task_id, chunks in enumerate(chunks_by_task):
        previous_acc_slot: int | None = None
        for chunk_index, chunk in enumerate(chunks):
            partial_slot = next_slot_id
            acc_slot = next_slot_id + 1
            next_slot_id += 2
            iter_duration = estimated_streaming_iter_cost_us(
                chunk.task_range,
                block_size,
            )
            reduce_inputs = (partial_slot,) if previous_acc_slot is None else (previous_acc_slot, partial_slot)
            reduce_duration = estimated_streaming_reduce_cost_us(len(reduce_inputs))
            group_stages = [
                StreamingReplayInstruction(
                    instruction_id=next_instruction_id,
                    task_id=task_id,
                    sm_id=chunk.sm_id,
                    input_slots=(),
                    output_slot=partial_slot,
                    duration_us=iter_duration,
                    queue_key=(0, task_id, chunk.part_index, 0),
                    level0_sort_us=iter_duration,
                ),
                StreamingReplayInstruction(
                    instruction_id=next_instruction_id + 1,
                    task_id=task_id,
                    sm_id=chunk.sm_id,
                    input_slots=reduce_inputs,
                    output_slot=acc_slot,
                    duration_us=reduce_duration,
                    queue_key=(0, task_id, chunk.part_index, 1),
                    level0_sort_us=iter_duration,
                ),
            ]
            next_instruction_id += 2
            if chunk_index + 1 == len(chunks):
                group_stages.append(
                    StreamingReplayInstruction(
                        instruction_id=next_instruction_id,
                        task_id=task_id,
                        sm_id=chunk.sm_id,
                        input_slots=(acc_slot,),
                        output_slot=None,
                        duration_us=estimated_streaming_finalize_cost_us(),
                        queue_key=(0, task_id, chunk.part_index, 2),
                        level0_sort_us=iter_duration,
                    )
                )
                next_instruction_id += 1
            node = ProducerReadyQueueNode(
                node_id=next_node_id,
                sm_id=chunk.sm_id,
                input_slots=(() if previous_acc_slot is None else (previous_acc_slot,)),
                output_slots=(acc_slot,),
                duration_us=sum(stage.duration_us for stage in group_stages),
                sort_key=(
                    streaming_queue_priority(chunk),
                    task_id,
                    chunk.part_index,
                    next_node_id,
                ),
            )
            nodes.append(node)
            node_iter_lengths[next_node_id] = chunk.task_range.length
            stages_by_node[next_node_id] = tuple(group_stages)
            previous_acc_slot = acc_slot
            next_node_id += 1
            iter_chunk_count += 1

    nodes_by_id = {node.node_id: node for node in nodes}
    queue_order = config_string("level0_queue_order", "task").strip().lower()
    if queue_order == "producer_ready":
        ordered_by_sm = producer_ready_queue_order(topology, nodes)
    else:
        priority_task_ids = config_int_set("level0_priority_tasks")

        def sort_key(node_id: int) -> tuple[int, int, int, int, int]:
            node = nodes_by_id[node_id]
            priority, task_id, part_index, _ = node.sort_key
            priority_task = 0 if task_id in priority_task_ids else 1
            if queue_order == "task_major":
                return (priority_task, task_id, priority, part_index, node_id)
            if queue_order == "long_first" and priority == 0:
                return (
                    priority,
                    priority_task,
                    -node_iter_lengths[node_id],
                    task_id,
                    part_index,
                )
            return (priority, priority_task, task_id, part_index, node_id)

        ordered_by_sm = {
            sm_id: tuple(
                sorted(
                    (node.node_id for node in nodes if node.sm_id == sm_id),
                    key=sort_key,
                )
            )
            for sm_id in range(topology.sm_count)
        }

    queues_by_sm = {
        sm_id: tuple(stage for node_id in ordered_by_sm[sm_id] for stage in stages_by_node[node_id]) for sm_id in range(topology.sm_count)
    }
    output_slot_to_instruction = {
        stage.output_slot: stage for queue in queues_by_sm.values() for stage in queue if stage.output_slot is not None
    }
    queue_indices = {sm_id: 0 for sm_id in range(topology.sm_count)}
    sm_ready = {sm_id: 0.0 for sm_id in range(topology.sm_count)}
    slot_ready: dict[int, float] = {}
    total_recv_wait_us = 0.0
    max_recv_wait_us = 0.0
    max_recv_instruction: StreamingReplayInstruction | None = None
    max_recv_slot_id: int | None = None
    hbm_edges = 0
    remaining = sum(len(queue) for queue in queues_by_sm.values())
    while remaining:
        progressed = False
        for sm_id in range(topology.sm_count):
            queue = queues_by_sm[sm_id]
            index = queue_indices[sm_id]
            if index >= len(queue):
                continue
            instruction = queue[index]
            input_ready_values: list[tuple[float, int]] = []
            missing_dependency = False
            instruction_hbm_edges = 0
            for slot_id in instruction.input_slots:
                producer = output_slot_to_instruction.get(slot_id)
                if producer is None:
                    continue
                producer_finish = slot_ready.get(slot_id)
                if producer_finish is None:
                    missing_dependency = True
                    break
                input_ready_values.append(
                    (
                        producer_finish
                        + estimated_streaming_comm_cost_us(
                            topology,
                            producer.sm_id,
                            instruction.sm_id,
                        ),
                        slot_id,
                    )
                )
                if producer.sm_id != instruction.sm_id and not topology.same_cluster(
                    producer.sm_id,
                    instruction.sm_id,
                ):
                    instruction_hbm_edges += 1
            if missing_dependency:
                continue
            input_ready, recv_slot_id = max(
                input_ready_values,
                default=(0.0, -1),
            )
            recv_wait = max(0.0, input_ready - sm_ready[sm_id])
            finish = max(sm_ready[sm_id], input_ready) + instruction.duration_us
            sm_ready[sm_id] = finish
            if instruction.output_slot is not None:
                slot_ready[instruction.output_slot] = finish
            total_recv_wait_us += recv_wait
            hbm_edges += instruction_hbm_edges
            if recv_wait > max_recv_wait_us:
                max_recv_wait_us = recv_wait
                max_recv_instruction = instruction
                max_recv_slot_id = None if recv_slot_id < 0 else recv_slot_id
            queue_indices[sm_id] = index + 1
            remaining -= 1
            progressed = True
        if not progressed:
            blocked = [
                queues_by_sm[sm_id][queue_indices[sm_id]].instruction_id
                for sm_id in range(topology.sm_count)
                if queue_indices[sm_id] < len(queues_by_sm[sm_id])
            ]
            raise RuntimeError(f"Dataflow streaming-linear replay could not resolve dependencies for queue groups {blocked[:8]}")

    active_finishes = sorted(value for value in sm_ready.values() if value > 0.0)
    if not active_finishes:
        return StreamingDagReplayScore(
            max_finish_us=0.0,
            p95_finish_us=0.0,
            finish_spread_us=0.0,
            total_recv_wait_us=0.0,
            max_recv_wait_us=0.0,
            hbm_edges=0,
            instruction_count=0,
            slot_count=0,
        )
    p95_index = min(
        len(active_finishes) - 1,
        math.ceil(0.95 * len(active_finishes)) - 1,
    )
    return StreamingDagReplayScore(
        max_finish_us=active_finishes[-1],
        p95_finish_us=active_finishes[p95_index],
        finish_spread_us=active_finishes[-1] - active_finishes[0],
        total_recv_wait_us=total_recv_wait_us,
        max_recv_wait_us=max_recv_wait_us,
        hbm_edges=hbm_edges,
        instruction_count=next_instruction_id,
        slot_count=2 * iter_chunk_count,
        max_recv_instruction_id=(None if max_recv_instruction is None else max_recv_instruction.instruction_id),
        max_recv_task_id=(None if max_recv_instruction is None else max_recv_instruction.task_id),
        max_recv_sm_id=(None if max_recv_instruction is None else max_recv_instruction.sm_id),
        max_recv_slot_id=max_recv_slot_id,
        finish_profile_us=tuple(reversed(active_finishes)),
        finish_by_sm_us=tuple(sm_ready.get(sm_id, 0.0) for sm_id in range(topology.sm_count)),
    )


def load_balanced_dag_streaming_chunks(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
    axis: Any,
    task_range_offsets: tuple[int, ...] | None = None,
    cluster_task_assignment: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[StreamingChunk, ...], ...]:
    resolved_task_range_offsets = task_range_offsets or (0,) * len(task_range_lengths)
    task_blocks = tuple(math.ceil(length / block_size) for length in task_range_lengths)
    score_mode = config_string("balanced_dag_score", "replay").strip().lower()
    if score_mode not in {"load", "replay"}:
        raise ValueError(f"balanced_dag_score must be 'load' or 'replay', got {score_mode!r}")
    beam_width = config_int("balanced_dag_beam", 48)
    max_split = max(1, config_int("balanced_dag_max_split", min(4, topology.cluster_count)))
    hbm_penalty_blocks = max(0, config_int("balanced_dag_hbm_penalty_blocks", 2))
    min_segment_blocks = max(1, config_int("balanced_dag_min_segment_blocks", 2))
    split_gain_blocks = max(
        0,
        config_int(
            "balanced_dag_split_gain_blocks",
            0 if score_mode == "replay" else 4,
        ),
    )
    tree_consumer_side = streaming_tree_consumer_side()
    hierarchical_cross_cluster = config_flag("hier_cross_cluster_split")
    ready_time_tree = config_flag("ready_time_tree")

    @dataclass(frozen=True)
    class BeamState:
        cluster_loads: tuple[int, ...]
        segments: tuple[StreamingSegment, ...]

    def state_from_cluster_tasks(cluster_tasks: Sequence[Sequence[int]]) -> BeamState:
        cluster_loads = [0 for _ in range(topology.cluster_count)]
        segments: list[StreamingSegment] = []
        for cluster_id, tasks in enumerate(cluster_tasks):
            for task_id in tasks:
                cluster_loads[cluster_id] += task_blocks[task_id]
                segments.append(
                    StreamingSegment(
                        task_id=task_id,
                        cluster_id=cluster_id,
                        begin=0,
                        end=task_range_lengths[task_id],
                    )
                )
        return BeamState(
            cluster_loads=tuple(cluster_loads),
            segments=tuple(segments),
        )

    def state_from_segments(segments: Sequence[StreamingSegment]) -> BeamState:
        cluster_loads = [0 for _ in range(topology.cluster_count)]
        for segment in segments:
            cluster_loads[segment.cluster_id] += math.ceil((segment.end - segment.begin) / block_size)
        return BeamState(
            cluster_loads=tuple(cluster_loads),
            segments=tuple(segments),
        )

    def candidate_segments_for_task(
        task_id: int,
        blocks: int,
        cluster_loads: tuple[int, ...],
    ) -> list[tuple[tuple[StreamingSegment, ...], tuple[int, ...]]]:
        max_k = min(max_split, topology.cluster_count, blocks)
        candidates: list[tuple[tuple[StreamingSegment, ...], tuple[int, ...]]] = []
        best_single_max_load = min(
            max(
                (
                    (cluster_loads[cluster_id] + (blocks if cluster_id == candidate_cluster else 0))
                    / max(1, len(get_cluster_sms(topology, cluster_id)))
                )
                for cluster_id in range(topology.cluster_count)
            )
            for candidate_cluster in range(topology.cluster_count)
        )
        ranked_clusters = sorted(
            range(topology.cluster_count),
            key=lambda cluster_id: (
                cluster_loads[cluster_id] / max(1, len(get_cluster_sms(topology, cluster_id))),
                cluster_loads[cluster_id],
                cluster_id,
            ),
        )
        for split_count in range(1, max_k + 1):
            if split_count > 1 and blocks < split_count * min_segment_blocks:
                continue
            cluster_pool = ranked_clusters[: max(split_count, min(topology.cluster_count, split_count + 3))]
            for cluster_ids in itertools.combinations(cluster_pool, split_count):
                remaining_blocks = blocks
                begin_block = 0
                segments: list[StreamingSegment] = []
                updated_loads = list(cluster_loads)
                ordered_clusters = tuple(
                    sorted(
                        cluster_ids,
                        key=lambda cluster_id: (
                            cluster_loads[cluster_id] / max(1, len(get_cluster_sms(topology, cluster_id))),
                            cluster_loads[cluster_id],
                            cluster_id,
                        ),
                    )
                )
                for index, cluster_id in enumerate(ordered_clusters):
                    remaining_parts = split_count - index
                    avg_blocks = math.ceil(remaining_blocks / remaining_parts)
                    desired_blocks = max(min_segment_blocks, avg_blocks)
                    segment_blocks = min(remaining_blocks - (remaining_parts - 1) * min_segment_blocks, desired_blocks)
                    segment_blocks = max(1, segment_blocks)
                    end_block = begin_block + segment_blocks
                    begin = min(task_range_lengths[task_id], begin_block * block_size)
                    end = min(task_range_lengths[task_id], end_block * block_size)
                    segments.append(
                        StreamingSegment(
                            task_id=task_id,
                            cluster_id=cluster_id,
                            begin=begin,
                            end=end,
                        )
                    )
                    updated_loads[cluster_id] += segment_blocks
                    begin_block = end_block
                    remaining_blocks -= segment_blocks
                if split_count > 1:
                    split_max_load = max(
                        updated_loads[cluster_id] / max(1, len(get_cluster_sms(topology, cluster_id)))
                        for cluster_id in range(topology.cluster_count)
                    )
                    split_gain = best_single_max_load - split_max_load
                    required_gain = hbm_penalty_blocks * (split_count - 1) + split_gain_blocks
                    if split_gain < required_gain:
                        continue
                candidates.append((tuple(segments), tuple(updated_loads)))
        return candidates

    def load_state_score(state: BeamState) -> tuple[float, float, int, int, tuple[int, ...]]:
        per_sm_loads = tuple(
            load / max(1, len(get_cluster_sms(topology, cluster_id))) for cluster_id, load in enumerate(state.cluster_loads)
        )
        max_load = max(per_sm_loads, default=0.0)
        spread = max_load - min(per_sm_loads, default=0.0)
        hbm_edges = 0
        tiny_segments = 0
        by_task: dict[int, int] = {}
        for segment in state.segments:
            by_task[segment.task_id] = by_task.get(segment.task_id, 0) + 1
            segment_blocks = math.ceil((segment.end - segment.begin) / block_size)
            if segment_blocks < min_segment_blocks:
                tiny_segments += 1
        for split_count in by_task.values():
            hbm_edges += max(0, split_count - 1)
        score = max_load + 0.20 * spread + hbm_penalty_blocks * hbm_edges + 4.0 * tiny_segments
        return (score, max_load, hbm_edges, tiny_segments, state.cluster_loads)

    def replay_state_score(state: BeamState) -> tuple[float, float, float, float, int, int, tuple[int, ...]]:
        chunks_by_task = segments_to_streaming_chunks(
            topology,
            task_range_lengths,
            block_size=block_size,
            axis=axis,
            task_range_offsets=resolved_task_range_offsets,
            segments=state.segments,
            require_all_tasks=False,
            error_prefix="Dataflow balanced DAG replay scorer",
        )
        try:
            replay = replay_streaming_tree_score(
                topology,
                chunks_by_task,
                block_size=block_size,
                streaming_tree_consumer=tree_consumer_side,
                hierarchical_cross_cluster=hierarchical_cross_cluster,
                ready_time_tree=ready_time_tree,
            )
        except RuntimeError:
            return (
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                math.inf,
                math.inf,
                state.cluster_loads,
            )
        objective = balanced_dag_replay_objective(
            replay,
            hbm_penalty_blocks=hbm_penalty_blocks,
        )
        return (
            objective,
            replay.max_finish_us,
            replay.max_recv_wait_us,
            replay.total_recv_wait_us,
            replay.hbm_edges,
            replay.instruction_count,
            state.cluster_loads,
        )

    def state_score(state: BeamState):
        if score_mode == "replay":
            return replay_state_score(state)
        return load_state_score(state)

    def no_split_incumbent_state() -> BeamState:
        cluster_tasks = (
            cluster_task_assignment if cluster_task_assignment is not None else load_balanced_cluster_tasks(topology, task_range_lengths)
        )
        return state_from_cluster_tasks(cluster_tasks)

    def local_suffix_split_search(initial_state: BeamState) -> BeamState:
        max_steps = max(0, config_int("balanced_dag_local_steps", 3))
        max_task_segments = max(1, config_int("balanced_dag_local_max_task_segments", 2))
        suffix_block_candidates = (16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160)
        current = initial_state
        current_score = state_score(current)

        for _ in range(max_steps):
            best_candidate: tuple[Any, BeamState] | None = None
            task_segment_counts: dict[int, int] = {}
            for segment in current.segments:
                task_segment_counts[segment.task_id] = task_segment_counts.get(segment.task_id, 0) + 1

            for segment_index, segment in enumerate(current.segments):
                if task_segment_counts.get(segment.task_id, 0) >= max_task_segments:
                    continue
                segment_blocks = math.ceil((segment.end - segment.begin) / block_size)
                if segment_blocks < 2 * min_segment_blocks:
                    continue
                existing_task_clusters = {item.cluster_id for item in current.segments if item.task_id == segment.task_id}
                for target_cluster in range(topology.cluster_count):
                    if target_cluster == segment.cluster_id or target_cluster in existing_task_clusters:
                        continue
                    for suffix_blocks in suffix_block_candidates:
                        if suffix_blocks >= segment_blocks:
                            continue
                        prefix_blocks = segment_blocks - suffix_blocks
                        if prefix_blocks < min_segment_blocks or suffix_blocks < min_segment_blocks:
                            continue
                        split_at = segment.end - suffix_blocks * block_size
                        if split_at <= segment.begin or split_at >= segment.end:
                            continue
                        candidate_segments = list(current.segments)
                        candidate_segments[segment_index] = StreamingSegment(
                            task_id=segment.task_id,
                            cluster_id=segment.cluster_id,
                            begin=segment.begin,
                            end=split_at,
                        )
                        candidate_segments.append(
                            StreamingSegment(
                                task_id=segment.task_id,
                                cluster_id=target_cluster,
                                begin=split_at,
                                end=segment.end,
                            )
                        )
                        candidate_state = state_from_segments(candidate_segments)
                        try:
                            candidate_score = state_score(candidate_state)
                        except RuntimeError:
                            continue
                        if candidate_score >= current_score:
                            continue
                        if best_candidate is None or candidate_score < best_candidate[0]:
                            best_candidate = (candidate_score, candidate_state)

            if best_candidate is None:
                break
            current_score, current = best_candidate

        return current

    if config_flag("balanced_dag_local_search"):
        best_state = local_suffix_split_search(no_split_incumbent_state())
        return segments_to_streaming_chunks(
            topology,
            task_range_lengths,
            block_size=block_size,
            axis=axis,
            task_range_offsets=resolved_task_range_offsets,
            segments=best_state.segments,
            require_all_tasks=True,
            error_prefix="Dataflow balanced DAG local-search scheduler",
        )

    beam = (
        BeamState(
            cluster_loads=tuple(0 for _ in range(topology.cluster_count)),
            segments=(),
        ),
    )
    for task_id in sorted(range(len(task_blocks)), key=lambda index: (-task_blocks[index], index)):
        next_beam: list[BeamState] = []
        for state in beam:
            for segments, updated_loads in candidate_segments_for_task(
                task_id,
                task_blocks[task_id],
                state.cluster_loads,
            ):
                next_beam.append(
                    BeamState(
                        cluster_loads=updated_loads,
                        segments=state.segments + segments,
                    )
                )
        if not next_beam:
            raise ValueError(f"Dataflow balanced DAG scheduler found no candidates for task {task_id}")
        beam = tuple(sorted(next_beam, key=state_score)[:beam_width])

    final_candidates = list(beam)
    if score_mode == "replay":
        final_candidates.append(no_split_incumbent_state())
    best_state = min(final_candidates, key=state_score)
    return segments_to_streaming_chunks(
        topology,
        task_range_lengths,
        block_size=block_size,
        axis=axis,
        task_range_offsets=resolved_task_range_offsets,
        segments=best_state.segments,
        require_all_tasks=True,
        error_prefix="Dataflow balanced DAG scheduler",
    )


def optimize_critical_path_split_streaming_chunks(
    topology: GPUTopology,
    task_range_lengths: tuple[int, ...],
    *,
    block_size: int,
    axis: Any,
    task_range_offsets: tuple[int, ...] | None = None,
    cluster_task_assignment: Sequence[Sequence[int]],
    incumbent_chunks: tuple[tuple[StreamingChunk, ...], ...],
    reduce_strategy: str = "streaming_tree",
) -> tuple[tuple[StreamingChunk, ...], ...]:
    del axis, task_range_offsets, cluster_task_assignment
    min_segment_blocks = max(1, config_int("critical_path_min_segment_blocks", 8))
    min_chunk_blocks = max(1, config_int("critical_path_min_chunk_blocks", 4))
    max_steps = max(0, config_int("critical_path_max_steps", 2))
    max_task_segments = max(1, config_int("critical_path_max_task_segments", 2))
    max_extra_chunks = max(0, config_int("critical_path_max_extra_chunks", 16))
    beam_width = max(1, config_int("critical_path_beam", 8))
    task_limit = max(1, config_int("critical_path_task_limit", 6))
    target_cluster_limit = max(
        1,
        config_int("critical_path_target_cluster_limit", 3),
    )
    suffix_candidate_limit = max(
        1,
        config_int("critical_path_suffix_candidate_limit", 4),
    )
    capacity_search_max_evals = max(
        0,
        config_int("critical_path_capacity_search_max_evals", 4),
    )
    allowed_max_block_regression = max(
        0,
        config_int("critical_path_allowed_max_block_regression", 0),
    )
    allowed_extra_tiny_chunks = max(
        0,
        config_int("critical_path_allow_extra_tiny_chunks", 0),
    )
    min_gain_us = max(0.0, config_float("critical_path_min_gain_us", 0.25))
    p95_weight = config_float("critical_path_p95_weight", 0.20)
    recv_weight = config_float("critical_path_recv_weight", 0.20)
    total_recv_weight = config_float("critical_path_total_recv_weight", 0.01)
    hbm_penalty_us = config_float("critical_path_hbm_penalty_us", 8.0)
    extra_chunk_penalty_us = config_float(
        "critical_path_extra_chunk_penalty_us",
        25.0,
    )
    tiny_chunk_penalty_us = config_float(
        "critical_path_tiny_chunk_penalty_us",
        100.0,
    )
    max_block_regression_penalty_us = config_float(
        "critical_path_max_block_regression_penalty_us",
        100.0,
    )
    tree_consumer_side = streaming_tree_consumer_side()
    hierarchical_cross_cluster = config_flag("hier_cross_cluster_split")
    ready_time_tree = config_flag("ready_time_tree")

    def chunk_block_count(chunk: StreamingChunk) -> int:
        return max(1, math.ceil(chunk.task_range.length / block_size))

    def sm_block_loads(
        chunks_by_task: Sequence[Sequence[StreamingChunk]],
    ) -> tuple[int, ...]:
        loads = [0 for _ in range(topology.sm_count)]
        for task_chunks in chunks_by_task:
            for chunk in task_chunks:
                loads[chunk.sm_id] += chunk_block_count(chunk)
        return tuple(loads)

    def count_tiny_chunks(
        chunks_by_task: Sequence[Sequence[StreamingChunk]],
    ) -> int:
        count = 0
        for task_id, task_chunks in enumerate(chunks_by_task):
            task_blocks = math.ceil(task_range_lengths[task_id] / block_size)
            if task_blocks < min_chunk_blocks:
                continue
            for chunk in task_chunks:
                if chunk_block_count(chunk) < min_chunk_blocks:
                    count += 1
        return count

    def score_chunks(
        chunks_by_task: tuple[tuple[StreamingChunk, ...], ...],
        *,
        base_iter_chunks: int,
        base_tiny_chunks: int,
        base_max_sm_blocks: int,
    ) -> tuple[float, float, float, float, int, int, int, int]:
        if reduce_strategy == "streaming":
            replay = replay_streaming_linear_score(
                topology,
                chunks_by_task,
                block_size=block_size,
            )
        else:
            replay = replay_streaming_tree_score(
                topology,
                chunks_by_task,
                block_size=block_size,
                streaming_tree_consumer=tree_consumer_side,
                hierarchical_cross_cluster=hierarchical_cross_cluster,
                ready_time_tree=ready_time_tree,
            )
        iter_chunks = sum(len(task_chunks) for task_chunks in chunks_by_task)
        tiny_chunks = count_tiny_chunks(chunks_by_task)
        max_sm_blocks = max(sm_block_loads(chunks_by_task), default=0)
        extra_iter_chunks = max(0, iter_chunks - base_iter_chunks)
        extra_tiny_chunks = max(0, tiny_chunks - base_tiny_chunks)
        max_block_regression = max(0, max_sm_blocks - base_max_sm_blocks)
        objective = (
            replay.max_finish_us
            + p95_weight * replay.p95_finish_us
            + recv_weight * replay.max_recv_wait_us
            + total_recv_weight * replay.total_recv_wait_us
            + hbm_penalty_us * replay.hbm_edges
            + extra_chunk_penalty_us * extra_iter_chunks
            + tiny_chunk_penalty_us * extra_tiny_chunks
            + max_block_regression_penalty_us * max_block_regression
        )
        return (
            objective,
            replay.max_finish_us,
            replay.p95_finish_us,
            replay.max_recv_wait_us,
            replay.hbm_edges,
            iter_chunks,
            tiny_chunks,
            max_sm_blocks,
        )

    base_iter_chunks = sum(len(task_chunks) for task_chunks in incumbent_chunks)
    base_tiny_chunks = count_tiny_chunks(incumbent_chunks)
    base_max_sm_blocks = max(sm_block_loads(incumbent_chunks), default=0)
    incumbent_score = score_chunks(
        incumbent_chunks,
        base_iter_chunks=base_iter_chunks,
        base_tiny_chunks=base_tiny_chunks,
        base_max_sm_blocks=base_max_sm_blocks,
    )

    best_score = incumbent_score
    best_chunks = incumbent_chunks
    explicit_capacity = config_int("critical_path_capacity_blocks", 0)
    if explicit_capacity < 0:
        raise ValueError("critical_path_capacity_blocks must be non-negative")
    explicit_capacity_applied = False
    if capacity_search_max_evals > 0:
        total_blocks = sum(math.ceil(length / block_size) for length in task_range_lengths)
        minimum_capacity = max(
            min_chunk_blocks,
            math.ceil(total_blocks / max(1, topology.sm_count)),
        )
        available_capacities = [explicit_capacity] if explicit_capacity > 0 else list(range(minimum_capacity, base_max_sm_blocks))
        if explicit_capacity == 0 and len(available_capacities) > capacity_search_max_evals:
            frontier_count = min(
                capacity_search_max_evals,
                max(1, (capacity_search_max_evals + 1) // 2),
            )
            sampled_indices = set(range(frontier_count))
            remaining_budget = capacity_search_max_evals - len(sampled_indices)
            tail_begin = frontier_count
            tail_length = len(available_capacities) - tail_begin
            if remaining_budget == 1 and tail_length > 0:
                sampled_indices.add(len(available_capacities) - 1)
            elif remaining_budget > 1 and tail_length > 0:
                sampled_indices.update(
                    tail_begin + round(index * (tail_length - 1) / (remaining_budget - 1)) for index in range(remaining_budget)
                )
            available_capacities = [available_capacities[index] for index in sorted(sampled_indices)]
        for capacity_blocks in available_capacities:
            candidate_chunks = capacity_balanced_streaming_chunks(
                topology,
                incumbent_chunks,
                block_size=block_size,
                capacity_blocks=capacity_blocks,
                min_segment_blocks=min_segment_blocks,
                min_chunk_blocks=min_chunk_blocks,
                max_task_segments=max_task_segments,
                assignment_beam=max(16, beam_width * 4),
            )
            if candidate_chunks is None:
                continue
            iter_chunks = sum(len(chunks) for chunks in candidate_chunks)
            if iter_chunks > base_iter_chunks + max_extra_chunks:
                continue
            tiny_chunks = count_tiny_chunks(candidate_chunks)
            if tiny_chunks > base_tiny_chunks + allowed_extra_tiny_chunks:
                continue
            max_sm_blocks = max(sm_block_loads(candidate_chunks), default=0)
            if max_sm_blocks > base_max_sm_blocks + allowed_max_block_regression:
                continue
            try:
                candidate_score = score_chunks(
                    candidate_chunks,
                    base_iter_chunks=base_iter_chunks,
                    base_tiny_chunks=base_tiny_chunks,
                    base_max_sm_blocks=base_max_sm_blocks,
                )
            except RuntimeError:
                continue
            if explicit_capacity > 0:
                # An explicit capacity is a controlled scheduling constraint,
                # not merely a hint to a cost model.  Silently returning the
                # incumbent makes the generated ranges contradict the caller's
                # request and prevents reproducible capacity exploration.
                best_score = candidate_score
                best_chunks = candidate_chunks
                explicit_capacity_applied = True
                break
            if candidate_score < best_score:
                best_score = candidate_score
                best_chunks = candidate_chunks
    if explicit_capacity > 0:
        if not explicit_capacity_applied:
            raise ValueError(f"Dataflow critical-path scheduler cannot satisfy explicit capacity {explicit_capacity} blocks per CTA")
        return best_chunks

    def state_key(
        chunks_by_task: Sequence[Sequence[StreamingChunk]],
    ) -> tuple[tuple[tuple[int, int, int], ...], ...]:
        return tuple(
            tuple((chunk.sm_id, chunk.task_range.begin, chunk.task_range.end) for chunk in task_chunks) for task_chunks in chunks_by_task
        )

    def moved_suffix_candidate(
        chunks_by_task: tuple[tuple[StreamingChunk, ...], ...],
        *,
        task_id: int,
        target_cluster: int,
        suffix_blocks: int,
        current_sm_loads: tuple[int, ...],
    ) -> tuple[tuple[StreamingChunk, ...], ...] | None:
        task_chunks = tuple(
            sorted(
                chunks_by_task[task_id],
                key=lambda chunk: (
                    chunk.task_range.begin,
                    chunk.task_range.end,
                    chunk.sm_id,
                ),
            )
        )
        if len(task_chunks) < 1:
            return None
        source_cluster = topology.cluster_id(task_chunks[0].sm_id)
        if any(topology.cluster_id(chunk.sm_id) != source_cluster for chunk in task_chunks):
            return None
        existing_clusters = {source_cluster}
        if target_cluster in existing_clusters or len(existing_clusters) >= max_task_segments:
            return None
        task_begin = task_chunks[0].task_range.begin
        task_end = task_chunks[-1].task_range.end
        task_blocks = math.ceil((task_end - task_begin) / block_size)
        prefix_blocks = task_blocks - suffix_blocks
        if prefix_blocks < min_segment_blocks or suffix_blocks < min_segment_blocks:
            return None
        source_sms = tuple(chunk.sm_id for chunk in task_chunks)
        if len(source_sms) != len(set(source_sms)):
            return None
        source_chunk_count = len(source_sms)
        if prefix_blocks < source_chunk_count * min_chunk_blocks:
            return None
        target_chunk_blocks = max(
            min_chunk_blocks,
            math.ceil(prefix_blocks / source_chunk_count),
        )
        target_chunk_count = max(1, math.ceil(suffix_blocks / target_chunk_blocks))
        if target_chunk_count > max_extra_chunks:
            return None

        target_sms = list(get_cluster_sms(topology, target_cluster))
        if target_chunk_count > len(target_sms):
            return None
        projected_loads = list(current_sm_loads)
        for chunk in task_chunks:
            projected_loads[chunk.sm_id] -= chunk_block_count(chunk)

        def partition_range(
            begin: int,
            end: int,
            sms: Sequence[int],
        ) -> list[StreamingChunk]:
            total_blocks = math.ceil((end - begin) / block_size)
            begin_block = 0
            result: list[StreamingChunk] = []
            for index, sm_id in enumerate(sms):
                remaining_blocks = total_blocks - begin_block
                remaining_parts = len(sms) - index
                chunk_blocks = math.ceil(remaining_blocks / remaining_parts)
                end_block = begin_block + chunk_blocks
                chunk_begin = min(end, begin + begin_block * block_size)
                chunk_end = end if index == len(sms) - 1 else min(end, begin + end_block * block_size)
                if chunk_end <= chunk_begin:
                    return []
                result.append(
                    StreamingChunk(
                        task_id=task_id,
                        sm_id=sm_id,
                        task_range=TaskRange(
                            axis=task_chunks[0].task_range.axis,
                            begin=chunk_begin,
                            end=chunk_end,
                        ),
                        part_index=0,
                        part_count=0,
                    )
                )
                projected_loads[sm_id] += math.ceil((chunk_end - chunk_begin) / block_size)
                begin_block = end_block
            return result

        split_at = min(task_end, task_begin + prefix_blocks * block_size)
        source_chunks = partition_range(task_begin, split_at, source_sms)
        if len(source_chunks) != source_chunk_count:
            return None
        selected_target_sms: list[int] = []
        for _ in range(target_chunk_count):
            available_sms = [sm_id for sm_id in target_sms if sm_id not in selected_target_sms]
            sm_id = min(available_sms, key=lambda item: (projected_loads[item], item))
            selected_target_sms.append(sm_id)
        target_chunks = partition_range(split_at, task_end, selected_target_sms)
        if len(target_chunks) != target_chunk_count:
            return None
        repartitioned = source_chunks + target_chunks
        updated_task_chunks = tuple(
            replace(
                chunk,
                part_index=part_index,
                part_count=len(repartitioned),
            )
            for part_index, chunk in enumerate(repartitioned)
        )
        return replace_task_streaming_chunks(
            chunks_by_task,
            task_id,
            updated_task_chunks,
        )

    def suffix_chunk_candidates(
        task_chunks: Sequence[StreamingChunk],
        *,
        desired_blocks: float,
    ) -> tuple[int, ...]:
        explicit_blocks = config_int_set("critical_path_suffix_blocks")
        if explicit_blocks and not any(value > 0 for value in explicit_blocks):
            raise ValueError("critical_path_suffix_blocks must contain positive integers")
        task_blocks = sum(chunk_block_count(chunk) for chunk in task_chunks)
        if explicit_blocks:
            return tuple(value for value in explicit_blocks if min_segment_blocks <= value <= task_blocks - min_segment_blocks)
        candidates: set[int] = set()
        source_chunk_count = len(task_chunks)
        max_target_chunks = min(max_extra_chunks, len(get_cluster_sms(topology, 0)))
        for target_chunk_count in range(1, max_target_chunks + 1):
            balanced_chunk_blocks = math.ceil(task_blocks / (source_chunk_count + target_chunk_count))
            suffix_blocks = task_blocks - source_chunk_count * balanced_chunk_blocks
            if min_segment_blocks <= suffix_blocks <= task_blocks - min_segment_blocks:
                candidates.add(suffix_blocks)
        rounded_desired = int(round(desired_blocks))
        if min_segment_blocks <= rounded_desired <= task_blocks - min_segment_blocks:
            candidates.add(rounded_desired)
        return tuple(
            suffix_blocks
            for suffix_blocks in sorted(
                candidates,
                key=lambda value: (
                    abs(value - desired_blocks),
                    value,
                ),
            )[:suffix_candidate_limit]
        )

    beam: list[
        tuple[
            tuple[float, float, float, float, int, int, int, int],
            tuple[tuple[StreamingChunk, ...], ...],
        ]
    ] = [(incumbent_score, incumbent_chunks)]
    seen = {state_key(incumbent_chunks)}

    for _ in range(max_steps):
        candidates = list(beam)
        generated = False
        for _, chunks_by_task in beam:
            current_sm_loads = sm_block_loads(chunks_by_task)
            cluster_loads = tuple(
                sum(current_sm_loads[sm_id] for sm_id in get_cluster_sms(topology, cluster_id))
                for cluster_id in range(topology.cluster_count)
            )
            ranked_tasks = sorted(
                range(len(chunks_by_task)),
                key=lambda task_id: (
                    -max(
                        (estimated_streaming_iter_cost_us(chunk.task_range, block_size) for chunk in chunks_by_task[task_id]),
                        default=0.0,
                    ),
                    -sum(chunk_block_count(chunk) for chunk in chunks_by_task[task_id]),
                    task_id,
                ),
            )[:task_limit]
            for task_id in ranked_tasks:
                task_chunks = tuple(
                    sorted(
                        chunks_by_task[task_id],
                        key=lambda chunk: (
                            chunk.task_range.begin,
                            chunk.task_range.end,
                            chunk.sm_id,
                        ),
                    )
                )
                if len(task_chunks) < 2:
                    continue
                source_cluster = topology.cluster_id(task_chunks[-1].sm_id)
                existing_clusters = {topology.cluster_id(chunk.sm_id) for chunk in task_chunks}
                if len(existing_clusters) >= max_task_segments:
                    continue
                source_sms = max(1, len(get_cluster_sms(topology, source_cluster)))
                target_clusters = sorted(
                    (cluster_id for cluster_id in range(topology.cluster_count) if cluster_id not in existing_clusters),
                    key=lambda cluster_id: (
                        cluster_loads[cluster_id] / max(1, len(get_cluster_sms(topology, cluster_id))),
                        cluster_id,
                    ),
                )[:target_cluster_limit]
                for target_cluster in target_clusters:
                    target_sms = max(1, len(get_cluster_sms(topology, target_cluster)))
                    desired_blocks = max(
                        0.0,
                        (cluster_loads[source_cluster] * target_sms - cluster_loads[target_cluster] * source_sms)
                        / (source_sms + target_sms),
                    )
                    if desired_blocks < min_segment_blocks:
                        continue
                    for suffix_blocks in suffix_chunk_candidates(
                        task_chunks,
                        desired_blocks=desired_blocks,
                    ):
                        candidate_chunks = moved_suffix_candidate(
                            chunks_by_task,
                            task_id=task_id,
                            target_cluster=target_cluster,
                            suffix_blocks=suffix_blocks,
                            current_sm_loads=current_sm_loads,
                        )
                        if candidate_chunks is None:
                            continue
                        if sum(len(chunks) for chunks in candidate_chunks) > (base_iter_chunks + max_extra_chunks):
                            continue
                        key = state_key(candidate_chunks)
                        if key in seen:
                            continue
                        seen.add(key)
                        tiny_chunks = count_tiny_chunks(candidate_chunks)
                        if tiny_chunks > base_tiny_chunks + allowed_extra_tiny_chunks:
                            continue
                        max_sm_blocks = max(sm_block_loads(candidate_chunks), default=0)
                        if max_sm_blocks > base_max_sm_blocks + allowed_max_block_regression:
                            continue
                        try:
                            candidate_score = score_chunks(
                                candidate_chunks,
                                base_iter_chunks=base_iter_chunks,
                                base_tiny_chunks=base_tiny_chunks,
                                base_max_sm_blocks=base_max_sm_blocks,
                            )
                        except RuntimeError:
                            continue
                        candidates.append((candidate_score, candidate_chunks))
                        generated = True
                        if candidate_score < best_score:
                            best_score = candidate_score
                            best_chunks = candidate_chunks
        beam = sorted(candidates, key=lambda item: item[0])[:beam_width]
        if not generated:
            break

    if best_score[0] >= incumbent_score[0] - min_gain_us:
        return incumbent_chunks
    return best_chunks


def streaming_queue_priority(chunk: StreamingChunk) -> int:
    if chunk.part_count == 1:
        return 1
    if chunk.part_index == 0:
        return 0
    return 2


def streaming_group_iter_length(group_instructions: Sequence[Instruction]) -> int:
    for instruction in group_instructions:
        if instruction.opcode is DataflowOpcode.ITER and instruction.task_range is not None:
            return instruction.task_range.length
    return 0


def streaming_instruction_duration_us(instruction: Instruction, block_size: int) -> float:
    if instruction.opcode is DataflowOpcode.ITER:
        return estimated_streaming_iter_cost_us(instruction.task_range, block_size)
    if instruction.opcode in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE}:
        return estimated_streaming_reduce_cost_us(len(instruction.input_slots))
    if instruction.opcode is DataflowOpcode.FINALIZE:
        return estimated_streaming_finalize_cost_us()
    return 0.0


def score_streaming_queue_group_order(
    topology: GPUTopology,
    ordered_groups: dict[int, tuple[tuple[int, int, int, tuple[Instruction, ...]], ...]],
    *,
    block_size: int,
    force_hbm_comms: bool,
) -> QueueGroupReplayScore:
    queues_by_sm = {
        sm_id: tuple(instruction for _, _, _, group_instructions in groups for instruction in group_instructions)
        for sm_id, groups in ordered_groups.items()
    }
    output_slot_to_sm: dict[int, int] = {}
    for queue in queues_by_sm.values():
        for instruction in queue:
            if instruction.output_slot is not None and instruction.sm_id is not None:
                output_slot_to_sm[instruction.output_slot] = instruction.sm_id

    queue_indices = {sm_id: 0 for sm_id in queues_by_sm}
    sm_ready = {sm_id: 0.0 for sm_id in range(topology.sm_count)}
    slot_ready: dict[int, float] = {}
    total_recv_wait_us = 0.0
    max_recv_wait_us = 0.0
    remaining = sum(len(queue) for queue in queues_by_sm.values())

    while remaining:
        progressed = False
        for sm_id in sorted(queues_by_sm):
            index = queue_indices[sm_id]
            queue = queues_by_sm[sm_id]
            if index >= len(queue):
                continue
            instruction = queue[index]

            input_ready_values: list[float] = []
            missing_dependency = False
            for slot_id in instruction.input_slots:
                producer_sm = output_slot_to_sm.get(slot_id)
                if producer_sm is None:
                    input_ready_values.append(0.0)
                    continue
                if slot_id not in slot_ready:
                    missing_dependency = True
                    break
                input_ready_values.append(
                    slot_ready[slot_id]
                    + estimated_streaming_comm_cost_us(
                        topology,
                        producer_sm,
                        sm_id,
                        force_hbm_comms=force_hbm_comms,
                    )
                )
            if missing_dependency:
                continue

            queue_ready = sm_ready.get(sm_id, 0.0)
            input_ready = max(input_ready_values, default=0.0)
            recv_wait = max(0.0, input_ready - queue_ready) if instruction.input_slots else 0.0
            finish = max(queue_ready, input_ready) + streaming_instruction_duration_us(instruction, block_size)
            sm_ready[sm_id] = finish
            if instruction.output_slot is not None:
                slot_ready[instruction.output_slot] = finish
            total_recv_wait_us += recv_wait
            max_recv_wait_us = max(max_recv_wait_us, recv_wait)
            queue_indices[sm_id] = index + 1
            remaining -= 1
            progressed = True
        if not progressed:
            blocked = [
                queues_by_sm[sm_id][queue_indices[sm_id]].instruction_id
                for sm_id in sorted(queues_by_sm)
                if queue_indices[sm_id] < len(queues_by_sm[sm_id])
            ]
            raise RuntimeError(f"Dataflow level0 queue replay could not resolve dependencies for instructions {blocked[:8]}")

    active_finishes = sorted(finish for finish in sm_ready.values() if finish > 0.0)
    if not active_finishes:
        return QueueGroupReplayScore(
            objective_us=0.0,
            max_finish_us=0.0,
            p95_finish_us=0.0,
            finish_spread_us=0.0,
            total_recv_wait_us=0.0,
            max_recv_wait_us=0.0,
        )
    p95_index = min(len(active_finishes) - 1, math.ceil(0.95 * len(active_finishes)) - 1)
    max_finish = active_finishes[-1]
    p95_finish = active_finishes[p95_index]
    finish_spread = active_finishes[-1] - active_finishes[0]
    objective = max_finish + 0.20 * p95_finish + 0.10 * finish_spread + 0.50 * max_recv_wait_us + 0.02 * total_recv_wait_us
    return QueueGroupReplayScore(
        objective_us=objective,
        max_finish_us=max_finish,
        p95_finish_us=p95_finish,
        finish_spread_us=finish_spread,
        total_recv_wait_us=total_recv_wait_us,
        max_recv_wait_us=max_recv_wait_us,
    )


def queue_group_replay_score_key(score: QueueGroupReplayScore) -> tuple[float, float, float, float, float]:
    return (
        score.objective_us,
        score.max_finish_us,
        score.max_recv_wait_us,
        score.total_recv_wait_us,
        score.finish_spread_us,
    )


def replace_ordered_sm_groups(
    ordered_groups: dict[int, tuple[tuple[int, int, int, tuple[Instruction, ...]], ...]],
    sm_id: int,
    groups: tuple[tuple[int, int, int, tuple[Instruction, ...]], ...],
) -> dict[int, tuple[tuple[int, int, int, tuple[Instruction, ...]], ...]]:
    candidate = dict(ordered_groups)
    candidate[sm_id] = groups
    return candidate


def move_level0_group(
    groups: tuple[tuple[int, int, int, tuple[Instruction, ...]], ...],
    *,
    source_index: int,
    target_index: int,
) -> tuple[tuple[int, int, int, tuple[Instruction, ...]], ...]:
    if source_index == target_index:
        return groups
    level0 = [group for group in groups if group[0] == 0]
    tail = tuple(group for group in groups if group[0] != 0)
    moved = level0.pop(source_index)
    level0.insert(target_index, moved)
    return tuple(level0) + tail


def optimize_level0_queue_groups_by_replay(
    topology: GPUTopology,
    task_ordered_groups: dict[int, tuple[tuple[int, int, int, tuple[Instruction, ...]], ...]],
    long_first_groups: dict[int, tuple[tuple[int, int, int, tuple[Instruction, ...]], ...]],
    *,
    block_size: int,
    force_hbm_comms: bool,
) -> dict[int, tuple[tuple[int, int, int, tuple[Instruction, ...]], ...]]:
    task_score = score_streaming_queue_group_order(
        topology,
        task_ordered_groups,
        block_size=block_size,
        force_hbm_comms=force_hbm_comms,
    )
    long_score = score_streaming_queue_group_order(
        topology,
        long_first_groups,
        block_size=block_size,
        force_hbm_comms=force_hbm_comms,
    )
    if queue_group_replay_score_key(long_score) < queue_group_replay_score_key(task_score):
        current_groups = dict(long_first_groups)
        current_score = long_score
    else:
        current_groups = dict(task_ordered_groups)
        current_score = task_score

    max_passes = max(1, config_int("level0_replay_passes", 2))
    for _ in range(max_passes):
        improved = False
        for sm_id in sorted(current_groups):
            groups = current_groups[sm_id]
            level0_count = sum(1 for group in groups if group[0] == 0)
            if level0_count <= 1:
                continue

            best_groups = groups
            best_score = current_score
            for source_index in range(level0_count):
                for target_index in range(level0_count):
                    if source_index == target_index:
                        continue
                    candidate_sm_groups = move_level0_group(
                        groups,
                        source_index=source_index,
                        target_index=target_index,
                    )
                    if candidate_sm_groups == groups:
                        continue
                    candidate_groups = replace_ordered_sm_groups(
                        current_groups,
                        sm_id,
                        candidate_sm_groups,
                    )
                    candidate_score = score_streaming_queue_group_order(
                        topology,
                        candidate_groups,
                        block_size=block_size,
                        force_hbm_comms=force_hbm_comms,
                    )
                    if queue_group_replay_score_key(candidate_score) < queue_group_replay_score_key(best_score):
                        best_groups = candidate_sm_groups
                        best_score = candidate_score

            if best_groups != groups:
                current_groups = replace_ordered_sm_groups(current_groups, sm_id, best_groups)
                current_score = best_score
                improved = True
        if not improved:
            break

    return current_groups


def ordered_streaming_queue_groups(
    topology: GPUTopology,
    queue_groups: dict[int, list[tuple[int, int, int, tuple[Instruction, ...]]]],
    *,
    order: str,
    block_size: int,
    force_hbm_comms: bool,
) -> dict[int, tuple[tuple[int, int, int, tuple[Instruction, ...]], ...]]:
    priority_task_ids = config_int_set("level0_priority_tasks")

    if order == "producer_ready":
        groups_by_node_id: dict[
            int,
            tuple[int, int, int, tuple[Instruction, ...]],
        ] = {}
        nodes: list[ProducerReadyQueueNode] = []
        next_node_id = 0
        for sm_id in sorted(queue_groups):
            for group in queue_groups[sm_id]:
                level, task_id, part_index, group_instructions = group
                produced_slots = {instruction.output_slot for instruction in group_instructions if instruction.output_slot is not None}
                input_slots = tuple(
                    dict.fromkeys(
                        slot_id
                        for instruction in group_instructions
                        for slot_id in instruction.input_slots
                        if slot_id not in produced_slots
                    )
                )
                output_slots = tuple(sorted(produced_slots))
                nodes.append(
                    ProducerReadyQueueNode(
                        node_id=next_node_id,
                        sm_id=sm_id,
                        input_slots=input_slots,
                        output_slots=output_slots,
                        duration_us=sum(
                            streaming_instruction_duration_us(
                                instruction,
                                block_size,
                            )
                            for instruction in group_instructions
                        ),
                        sort_key=(level, task_id, part_index, next_node_id),
                    )
                )
                groups_by_node_id[next_node_id] = group
                next_node_id += 1
        ordered_node_ids = producer_ready_queue_order(
            topology,
            nodes,
            force_hbm_comms=force_hbm_comms,
        )
        return {sm_id: tuple(groups_by_node_id[node_id] for node_id in ordered_node_ids[sm_id]) for sm_id in range(topology.sm_count)}

    def priority_sort_key(level: int, task_id: int) -> int:
        return 0 if level == 0 and task_id in priority_task_ids else 1

    def task_sort_key(
        item: tuple[int, int, int, tuple[Instruction, ...]],
    ) -> tuple[int, int, int, int, int]:
        level, task_id, part_index, _ = item
        return (level, priority_sort_key(level, task_id), task_id, part_index, 0)

    def task_major_sort_key(
        item: tuple[int, int, int, tuple[Instruction, ...]],
    ) -> tuple[int, int, int, int, int]:
        level, task_id, part_index, _ = item
        return (priority_sort_key(level, task_id), task_id, level, part_index, 0)

    def long_first_sort_key(
        item: tuple[int, int, int, tuple[Instruction, ...]],
    ) -> tuple[int, int, int, int, int]:
        level, task_id, part_index, group_instructions = item
        if level == 0:
            return (
                level,
                priority_sort_key(level, task_id),
                -streaming_group_iter_length(group_instructions),
                task_id,
                part_index,
            )
        return (level, priority_sort_key(level, task_id), task_id, part_index, 0)

    slot_consumer_tree_level: dict[int, int] = {}
    for groups in queue_groups.values():
        for _, _, _, group_instructions in groups:
            for instruction in group_instructions:
                if instruction.opcode is not DataflowOpcode.REDUCE_UPDATE:
                    continue
                tree_level = instruction.attrs.get("tree_level")
                if not isinstance(tree_level, int):
                    continue
                for slot_id in instruction.input_slots:
                    slot_consumer_tree_level[slot_id] = max(
                        slot_consumer_tree_level.get(slot_id, 0),
                        tree_level,
                    )

    def group_output_slot(group_instructions: Sequence[Instruction]) -> int | None:
        for instruction in reversed(group_instructions):
            if instruction.output_slot is not None:
                return instruction.output_slot
        return None

    def group_consumer_level(
        item: tuple[int, int, int, tuple[Instruction, ...]],
    ) -> int:
        output_slot = group_output_slot(item[3])
        return 0 if output_slot is None else slot_consumer_tree_level.get(output_slot, 0)

    def critical_dominates(
        left: tuple[int, int, int, tuple[Instruction, ...]],
        right: tuple[int, int, int, tuple[Instruction, ...]],
    ) -> bool:
        return streaming_group_iter_length(left[3]) > streaming_group_iter_length(right[3]) and group_consumer_level(
            left
        ) > group_consumer_level(right)

    def critical_order_groups(
        groups: Sequence[tuple[int, int, int, tuple[Instruction, ...]]],
    ) -> tuple[tuple[int, int, int, tuple[Instruction, ...]], ...]:
        ordered = list(sorted(groups, key=task_sort_key))
        level0_count = sum(1 for group in ordered if group[0] == 0)
        if level0_count <= 1:
            return tuple(ordered)
        changed = True
        while changed:
            changed = False
            for index in range(1, level0_count):
                if critical_dominates(ordered[index], ordered[index - 1]):
                    ordered[index - 1], ordered[index] = ordered[index], ordered[index - 1]
                    changed = True
        return tuple(ordered)

    task_ordered_groups = {sm_id: tuple(sorted(groups, key=task_sort_key)) for sm_id, groups in queue_groups.items()}
    if order == "task":
        return task_ordered_groups
    if order == "task_major":
        return {sm_id: tuple(sorted(groups, key=task_major_sort_key)) for sm_id, groups in queue_groups.items()}

    long_first_groups = {sm_id: tuple(sorted(groups, key=long_first_sort_key)) for sm_id, groups in queue_groups.items()}
    if order == "long_first":
        return long_first_groups

    critical_groups = {sm_id: critical_order_groups(groups) for sm_id, groups in queue_groups.items()}
    if order == "critical":
        return critical_groups

    return optimize_level0_queue_groups_by_replay(
        topology,
        task_ordered_groups,
        long_first_groups,
        block_size=block_size,
        force_hbm_comms=force_hbm_comms,
    )


@activate_scheduler_config
def schedule(
    program: DataflowProgram,
    *,
    topology: GPUTopology,
    range_lengths: dict[Any, int | Sequence[int]],
    range_offsets: dict[Any, int | Sequence[int]] | None = None,
    block_size: int,
    task_extents: Sequence[int] | None = None,
    include_exit: bool = True,
    scheduler_policy: str = "round_robin",
    reduce_strategy: str = "all_at_once",
    force_hbm_comms: bool = False,
    streaming_tree_consumer: str | None = None,
    cluster_task_assignment: Sequence[Sequence[int]] | str | None = None,
    skip_tiny_root_fragment: bool | None = None,
    skip_tiny_root_fragment_blocks: int | None = None,
    partial_only: bool = False,
    direct_leaf_acc: bool | None = None,
    iter_range_buckets: Any = None,
    iter_range_bucket_size: int | None = None,
    iter_range_exact_lengths: Any = None,
    task_coord_overrides: Sequence[Sequence[int] | int] | None = None,
    stage_graph_task_weights: Sequence[int | float] | None = None,
    stage_graph_cluster_assignment: Sequence[int] | None = None,
    range_resource_budget_bytes: int | None = None,
    target_capabilities: TargetCapabilitySnapshot | None = None,
    cross_handler_handoff_max_shared_memory_bytes: int | None = None,
    pic: bool = False,
    pic_dir: Any = "schedule_res",
    scheduler_config: DataflowSchedulerConfig | Mapping[str, Any] | None = None,
) -> InstructionPlan:
    """Create a deterministic instruction plan for a complete Dataflow program."""

    del scheduler_config  # Bound by ``activate_scheduler_config`` for helper access.

    if not isinstance(topology, GPUTopology):
        raise TypeError(f"Dataflow scheduler expects a GPUTopology, got {topology!r}")
    if block_size <= 0:
        raise ValueError(f"Dataflow scheduler block_size must be positive, got {block_size}")
    if not range_lengths:
        raise ValueError("Dataflow scheduler requires concrete range_lengths")
    if program.is_stage_graph:
        plan = schedule_linear_stage_graph(
            program,
            topology=topology,
            range_lengths=range_lengths,
            block_size=block_size,
            task_extents=task_extents,
            include_exit=include_exit,
            force_hbm_comms=force_hbm_comms,
            task_coord_overrides=task_coord_overrides,
            stage_graph_task_weights=stage_graph_task_weights,
            stage_graph_cluster_assignment=stage_graph_cluster_assignment,
            range_resource_budget_bytes=range_resource_budget_bytes,
            target_capabilities=target_capabilities,
            cross_handler_handoff_max_shared_memory_bytes=(cross_handler_handoff_max_shared_memory_bytes),
        )
        plan = attach_handler_identities(program, plan)
        plan = attach_joint_execution_plan(program, plan)
        maybe_dump_schedule_visualization(plan, pic=pic, pic_dir=pic_dir)
        return plan

    range_axis = resolve_range_axis(program, range_lengths)
    task_range_lengths = as_lengths(range_lengths[range_axis])
    task_range_offsets = (
        (0,) * len(task_range_lengths)
        if range_offsets is None or range_axis not in range_offsets
        else as_offsets(range_offsets[range_axis], task_count=len(task_range_lengths))
    )
    normalized_task_extents = normalize_task_extents(task_extents, len(task_range_lengths))
    normalized_scheduler_policy, normalized_reduce_strategy = _validate_scheduler_policy(
        scheduler_policy,
        reduce_strategy,
    )
    normalized_iter_range_buckets = normalize_iter_range_buckets(iter_range_buckets)
    normalized_iter_range_bucket_size = normalize_iter_range_bucket_size(
        iter_range_bucket_size,
        fallback=block_size,
    )
    normalized_iter_range_exact_lengths = normalize_iter_range_exact_lengths(iter_range_exact_lengths)

    assert program.partial_stage is not None
    assert program.reduce_stage is not None
    assert program.finalize_stage is not None
    if (
        normalized_reduce_strategy == "streaming_tree"
        and program.reduce_stage.reduce_call.operator.reducer_contract is not DataflowReducerContract.ASSOCIATIVE_BINARY
    ):
        raise ValueError(
            "Dataflow streaming_tree scheduling requires an explicitly associative "
            "binary reducer; legacy n-ary reducers cannot be reordered"
        )
    intermediate_type = program.partial_stage.output_type
    handler_registry = build_handler_registry(program)
    iter_base_identity = handler_registry.identity_for_call(
        program.partial_stage.iter_call,
        "iter",
    )

    next_instruction_id = 0
    next_slot_id = 0
    next_iter_sm = 0
    all_instructions: list[Instruction] = []
    queues: dict[int, list[Instruction]] = {sm_id: [] for sm_id in range(topology.sm_count)}
    slots: list[SlotPlan] = []
    comms: list[CommPlan] = []

    def iter_handler_variant_key(task_range: TaskRange) -> DataflowHandlerVariantKey:
        specialization = iter_range_specialization_for_length(
            task_range.length,
            bucket_size=normalized_iter_range_bucket_size,
            buckets=normalized_iter_range_buckets,
            exact_lengths=normalized_iter_range_exact_lengths,
        )
        return DataflowHandlerVariantKey(
            base_identity=iter_base_identity,
            typed_specializations=(specialization,),
        )

    def iter_operator_name(task_range: TaskRange) -> str:
        variant_key = iter_handler_variant_key(task_range)
        specialization = variant_key.iter_range
        assert specialization is not None
        return specialization.display_name(program.partial_stage.iter_call.name)

    def emit(instruction: Instruction) -> Instruction:
        all_instructions.append(instruction)
        if instruction.sm_id is not None:
            queues[instruction.sm_id].append(instruction)
        return instruction

    if normalized_reduce_strategy in {"streaming", "streaming_tree"}:
        chunks_by_task = load_balanced_streaming_chunks(
            topology,
            task_range_lengths,
            block_size=block_size,
            axis=range_axis,
            task_range_offsets=task_range_offsets,
            cluster_task_assignment=cluster_task_assignment,
            skip_tiny_root_fragment=skip_tiny_root_fragment,
            skip_tiny_root_fragment_blocks=skip_tiny_root_fragment_blocks,
            reduce_strategy=normalized_reduce_strategy,
        )
        queue_groups: dict[int, list[tuple[int, int, int, tuple[Instruction, ...]]]] = {sm_id: [] for sm_id in range(topology.sm_count)}
        level0_queue_order = config_string("level0_queue_order", "task").strip().lower()
        if level0_queue_order not in {
            "task",
            "task_major",
            "long_first",
            "critical",
            "replay",
            "producer_ready",
        }:
            raise ValueError(
                "level0_queue_order must be 'task', 'task_major', 'long_first', "
                "'critical', 'replay', or 'producer_ready', "
                f"got {level0_queue_order!r}"
            )

        def emit_streaming(instruction: Instruction) -> Instruction:
            all_instructions.append(instruction)
            return instruction

        def append_streaming_comm(
            *,
            source_instruction: Instruction,
            target_instruction: Instruction,
            slot_id: int,
            producer_sm: int,
            consumer_sm: int,
            task_id: int,
        ) -> None:
            if producer_sm == consumer_sm:
                return
            same_cluster = topology.same_cluster(producer_sm, consumer_sm) and not force_hbm_comms
            send_kind = DataflowCommKind.CLUSTER_SEND if same_cluster else DataflowCommKind.HBM_SEND
            recv_kind = DataflowCommKind.CLUSTER_RECV if same_cluster else DataflowCommKind.HBM_RECV
            flag_epoch = task_id + 1 if not same_cluster else 1
            barrier_phase = 0
            comms.append(
                CommPlan(
                    source_instruction_id=source_instruction.instruction_id,
                    target_instruction_id=target_instruction.instruction_id,
                    source_slot_id=slot_id,
                    target_slot_id=slot_id,
                    producer_sm=producer_sm,
                    consumer_sm=consumer_sm,
                    kind=send_kind,
                    dispatch_instruction_id=source_instruction.instruction_id,
                    peer_cta_rank=topology.cluster_rank(consumer_sm) if same_cluster else None,
                    flag_epoch=flag_epoch,
                    barrier_phase=barrier_phase,
                )
            )
            comms.append(
                CommPlan(
                    source_instruction_id=source_instruction.instruction_id,
                    target_instruction_id=target_instruction.instruction_id,
                    source_slot_id=slot_id,
                    target_slot_id=slot_id,
                    producer_sm=producer_sm,
                    consumer_sm=consumer_sm,
                    kind=recv_kind,
                    dispatch_instruction_id=target_instruction.instruction_id,
                    peer_cta_rank=(topology.cluster_rank(producer_sm) if same_cluster else None),
                    flag_epoch=flag_epoch,
                    barrier_phase=barrier_phase,
                )
            )

        if normalized_reduce_strategy == "streaming_tree":
            tree_consumer_side = streaming_tree_consumer_side(streaming_tree_consumer)
            hierarchical_cross_cluster = config_flag("hier_cross_cluster_split")
            skip_cross_cluster_reduce = config_flag("skip_cross_cluster_reduce")
            ready_time_tree = config_flag("ready_time_tree")
            chunks_by_task = optimize_streaming_tree_leaf_orders(
                topology,
                chunks_by_task,
                block_size=block_size,
                streaming_tree_consumer=tree_consumer_side,
                hierarchical_cross_cluster=hierarchical_cross_cluster,
                ready_time_tree=ready_time_tree,
            )
            chunks_by_task = optimize_streaming_tree_chunk_lengths_by_replay(
                topology,
                chunks_by_task,
                block_size=block_size,
                streaming_tree_consumer=tree_consumer_side,
                hierarchical_cross_cluster=hierarchical_cross_cluster,
                ready_time_tree=ready_time_tree,
            )
            chunks_by_task = tuple(logical_streaming_chunk_order(task_chunks) for task_chunks in chunks_by_task)
            estimated_sm_ready = {sm_id: 0.0 for sm_id in range(topology.sm_count)}
            estimated_slot_ready: dict[int, float] = {}
            estimated_slot_producer_sm: dict[int, int] = {}
            estimated_slot_tree_level: dict[int, int] = {}
            estimated_slot_level0_position: dict[int, int] = {}
            ready_time_recv_weight = config_float("ready_time_recv_weight", 0.6)
            ready_time_queue_weight = config_float("ready_time_queue_weight", 0.1)
            ready_time_balance_weight = config_float("ready_time_balance_weight", 0.0)
            ready_time_cluster_comm_weight = config_float("ready_time_cluster_comm_weight", 0.25)
            ready_time_finalize = config_flag("ready_time_finalize")
            ready_time_finalize_comm_penalty = config_float(
                "ready_time_finalize_comm_penalty_us",
                0.5,
            )

            def reserve_estimated_instruction(
                *,
                sm_id: int,
                input_sources: Sequence[tuple[int, int]],
                output_slot: int | None,
                duration_us: float,
            ) -> float:
                input_ready = max(
                    (
                        estimated_slot_ready.get(slot_id, 0.0)
                        + estimated_streaming_comm_cost_us(
                            topology,
                            producer_sm,
                            sm_id,
                            force_hbm_comms=force_hbm_comms,
                        )
                        for slot_id, producer_sm in input_sources
                    ),
                    default=0.0,
                )
                start = max(estimated_sm_ready.get(sm_id, 0.0), input_ready)
                finish = start + duration_us
                estimated_sm_ready[sm_id] = finish
                if output_slot is not None:
                    estimated_slot_ready[output_slot] = finish
                return finish

            def reserve_estimated_streaming_instruction(
                instruction: Instruction,
                *,
                level0_position: int | None = None,
            ) -> None:
                if instruction.sm_id is None:
                    return
                input_sources = tuple(
                    (slot_id, estimated_slot_producer_sm[slot_id])
                    for slot_id in instruction.input_slots
                    if slot_id in estimated_slot_producer_sm
                )
                reserve_estimated_instruction(
                    sm_id=instruction.sm_id,
                    input_sources=input_sources,
                    output_slot=instruction.output_slot,
                    duration_us=streaming_instruction_duration_us(instruction, block_size),
                )
                if instruction.output_slot is not None:
                    estimated_slot_producer_sm[instruction.output_slot] = instruction.sm_id
                    tree_level = instruction.attrs.get("tree_level")
                    if isinstance(tree_level, int):
                        estimated_slot_tree_level[instruction.output_slot] = tree_level
                        if tree_level == 0 and level0_position is not None:
                            estimated_slot_level0_position[instruction.output_slot] = level0_position

            def choose_reduce_sm(
                *,
                left_slot: int,
                left_sm: int,
                right_slot: int,
                right_sm: int,
                prefer_late_input: bool = False,
            ) -> int:
                fixed_sm = right_sm if tree_consumer_side == "right" else left_sm
                if not ready_time_tree:
                    return fixed_sm
                candidates = streaming_tree_reduce_candidate_sms(
                    topology,
                    left_sm=left_sm,
                    right_sm=right_sm,
                    prefer_late_input=prefer_late_input,
                )
                candidates = prune_ready_time_reduce_candidates(
                    candidates,
                    ready_by_sm=estimated_sm_ready,
                    left_sm=left_sm,
                    right_sm=right_sm,
                )
                if config_flag("reduce_higher_level_producer"):
                    left_level = estimated_slot_tree_level.get(left_slot, 0)
                    right_level = estimated_slot_tree_level.get(right_slot, 0)
                    if left_level != right_level:
                        return left_sm if left_level > right_level else right_sm
                if prefer_late_input and len(candidates) == 2 and config_flag("root_reduce_late_producer"):
                    left_level = estimated_slot_tree_level.get(left_slot, 0)
                    right_level = estimated_slot_tree_level.get(right_slot, 0)
                    if left_level != right_level:
                        lower_slot = left_slot if left_level < right_level else right_slot
                        lower_sm = left_sm if left_level < right_level else right_sm
                        if estimated_slot_level0_position.get(lower_slot, 0) > 0:
                            return lower_sm
                    left_ready = estimated_slot_ready.get(left_slot, 0.0)
                    right_ready = estimated_slot_ready.get(right_slot, 0.0)
                    return right_sm if right_ready > left_ready else left_sm

                def candidate_score(sm_id: int) -> tuple[float, float, float, int]:
                    left_comm = estimated_streaming_comm_cost_us(
                        topology,
                        left_sm,
                        sm_id,
                        force_hbm_comms=force_hbm_comms,
                    )
                    right_comm = estimated_streaming_comm_cost_us(
                        topology,
                        right_sm,
                        sm_id,
                        force_hbm_comms=force_hbm_comms,
                    )
                    left_ready = estimated_slot_ready.get(left_slot, 0.0) + left_comm
                    right_ready = estimated_slot_ready.get(right_slot, 0.0) + right_comm
                    input_ready = max(left_ready, right_ready)
                    queue_ready = estimated_sm_ready.get(sm_id, 0.0)
                    recv_wait = max(0.0, input_ready - queue_ready)
                    finish = max(queue_ready, input_ready) + estimated_streaming_reduce_cost_us(2)
                    projected_max = max(finish, max(estimated_sm_ready.values(), default=0.0))
                    remote_comm_cost = (left_comm if left_sm != sm_id else 0.0) + (right_comm if right_sm != sm_id else 0.0)
                    score = (
                        finish
                        + ready_time_recv_weight * recv_wait
                        + ready_time_queue_weight * queue_ready
                        + ready_time_balance_weight * projected_max
                        + ready_time_cluster_comm_weight * remote_comm_cost
                    )
                    if prefer_late_input and (
                        (config_flag("cross_cluster_root_global_candidates") and not topology.same_cluster(left_sm, right_sm))
                        or (config_flag("root_reduce_cluster_candidates") and topology.same_cluster(left_sm, right_sm))
                    ):
                        return (score, queue_ready, recv_wait, -sm_id)
                    return (score, recv_wait, queue_ready, -sm_id)

                return min(candidates, key=candidate_score)

            def choose_finalize_sm(*, root_slot: int, root_sm: int) -> int:
                if not ready_time_finalize:
                    return root_sm
                root_ready = estimated_slot_ready.get(root_slot, estimated_sm_ready.get(root_sm, 0.0))
                candidates = get_cluster_sms(topology, topology.cluster_id(root_sm))
                finalize_duration = estimated_streaming_finalize_cost_us()

                def candidate_score(sm_id: int) -> tuple[float, float, float, bool, int]:
                    comm_cost = estimated_streaming_comm_cost_us(
                        topology,
                        root_sm,
                        sm_id,
                        force_hbm_comms=force_hbm_comms,
                    )
                    input_ready = root_ready + comm_cost
                    queue_ready = estimated_sm_ready.get(sm_id, 0.0)
                    recv_wait = max(0.0, input_ready - queue_ready)
                    finish = max(queue_ready, input_ready) + finalize_duration
                    score = finish + 0.30 * recv_wait + 0.05 * queue_ready + (ready_time_finalize_comm_penalty if sm_id != root_sm else 0.0)
                    return (score, finish, recv_wait, sm_id != root_sm, sm_id)

                return min(candidates, key=candidate_score)

            def append_finalize_comm(
                *,
                root_instruction: Instruction,
                finalize_instruction: Instruction,
                root_slot: int,
                root_sm: int,
                finalize_sm: int,
                task_id: int,
            ) -> None:
                if root_sm == finalize_sm:
                    return
                append_streaming_comm(
                    source_instruction=root_instruction,
                    target_instruction=finalize_instruction,
                    slot_id=root_slot,
                    producer_sm=root_sm,
                    consumer_sm=finalize_sm,
                    task_id=task_id,
                )

            def emit_streaming_tree_nodes(
                *,
                task_id: int,
                coords: tuple[int, ...],
                nodes: list[tuple[int, Instruction, int]],
                start_level: int,
                scope: str | None = None,
            ) -> tuple[tuple[int, Instruction, int], int]:
                nonlocal next_instruction_id, next_slot_id
                if config_flag("ordered_interval_tree") and len(nodes) > 1:
                    tree_plan = ordered_reduction_tree_plan(
                        topology,
                        tuple((sm_id, estimated_slot_ready.get(slot_id, 0.0)) for slot_id, _, sm_id in nodes),
                        consumer_side=tree_consumer_side,
                        adaptive_consumer=config_flag("ordered_tree_adaptive_consumer"),
                        force_hbm_comms=force_hbm_comms,
                    )

                    def emit_ordered(
                        plan: OrderedReductionTreePlan,
                    ) -> tuple[int, Instruction, int]:
                        nonlocal next_instruction_id, next_slot_id
                        if plan.is_leaf:
                            return nodes[plan.begin]
                        max_reduce_arity = max(
                            2,
                            config_int("ordered_tree_max_reduce_arity", 2),
                        )
                        frontier = ordered_reduction_fused_frontier(
                            plan,
                            max_reduce_arity=max_reduce_arity,
                            max_resident_remote_inputs=joint_max_resident_remote_inputs(),
                            output_alias_input_indices=(
                                operator_physical_contract(program.reduce_stage.reduce_call.operator.attrs).output_alias_input_indices
                                if config_flag("direct_leaf_acc") and config_flag("joint_schedule") and program.reduce_stage is not None
                                else ()
                            ),
                        )
                        inputs = tuple(emit_ordered(child) for child in frontier)
                        reduce_sm = plan.consumer_sm
                        if (
                            plan.begin == 0
                            and plan.end == len(nodes)
                            and len(inputs) == 2
                            and (config_flag("root_reduce_cluster_candidates") or config_flag("cross_cluster_root_global_candidates"))
                        ):
                            reduce_sm = choose_reduce_sm(
                                left_slot=inputs[0][0],
                                left_sm=inputs[0][2],
                                right_slot=inputs[1][0],
                                right_sm=inputs[1][2],
                                prefer_late_input=True,
                            )
                        level = start_level + plan.height - 1
                        acc_slot_id = next_slot_id
                        next_slot_id += 1
                        attrs: dict[str, Any] = {
                            "reduce_strategy": "streaming_tree",
                            "tree_level": level,
                            "placement_policy": "ordered_interval_ready",
                            "ordered_interval_begin": plan.begin,
                            "ordered_interval_end": plan.end,
                        }
                        if len(inputs) > 2:
                            attrs["fused_reduce_arity"] = len(inputs)
                        if scope is not None:
                            attrs["hierarchical_scope"] = scope
                            attrs["cluster_id"] = topology.cluster_id(reduce_sm)
                        reduce_instruction = emit_streaming(
                            Instruction(
                                instruction_id=next_instruction_id,
                                opcode=DataflowOpcode.REDUCE_UPDATE,
                                operator_name=program.reduce_stage.reduce_call.name,
                                task_id=task_id,
                                task_coords=coords,
                                sm_id=reduce_sm,
                                input_slots=tuple(slot_id for slot_id, _, _ in inputs),
                                output_slot=acc_slot_id,
                                attrs=attrs,
                            )
                        )
                        next_instruction_id += 1
                        slots.append(
                            SlotPlan(
                                slot_id=acc_slot_id,
                                task_id=task_id,
                                intermediate_type=intermediate_type,
                                role="streaming_acc",
                                producer_instruction_id=reduce_instruction.instruction_id,
                                shared_storage_id=0,
                            )
                        )
                        for input_slot, input_instruction, input_sm in inputs:
                            append_streaming_comm(
                                source_instruction=input_instruction,
                                target_instruction=reduce_instruction,
                                slot_id=input_slot,
                                producer_sm=input_sm,
                                consumer_sm=reduce_sm,
                                task_id=task_id,
                            )
                        reserve_estimated_instruction(
                            sm_id=reduce_sm,
                            input_sources=tuple((slot_id, input_sm) for slot_id, _, input_sm in inputs),
                            output_slot=acc_slot_id,
                            duration_us=estimated_streaming_reduce_cost_us(len(inputs)),
                        )
                        estimated_slot_producer_sm[acc_slot_id] = reduce_sm
                        estimated_slot_tree_level[acc_slot_id] = level
                        estimated_slot_level0_position[acc_slot_id] = max(
                            (estimated_slot_level0_position.get(slot_id, 0) for slot_id, _, _ in inputs),
                            default=0,
                        )
                        queue_groups[reduce_sm].append(
                            (
                                level,
                                task_id,
                                plan.begin,
                                (reduce_instruction,),
                            )
                        )
                        return acc_slot_id, reduce_instruction, reduce_sm

                    return emit_ordered(tree_plan), start_level + tree_plan.height

                level = start_level
                current_nodes = nodes
                while len(current_nodes) > 1:
                    next_nodes: list[tuple[int, Instruction, int]] = []
                    pair_index = 0
                    for index in range(0, len(current_nodes), 2):
                        if index + 1 >= len(current_nodes):
                            next_nodes.append(current_nodes[index])
                            continue

                        left_slot, left_instruction, left_sm = current_nodes[index]
                        right_slot, right_instruction, right_sm = current_nodes[index + 1]
                        reduce_sm = choose_reduce_sm(
                            left_slot=left_slot,
                            left_sm=left_sm,
                            right_slot=right_slot,
                            right_sm=right_sm,
                            prefer_late_input=len(current_nodes) == 2,
                        )
                        acc_slot_id = next_slot_id
                        next_slot_id += 1
                        attrs: dict[str, Any] = {
                            "reduce_strategy": "streaming_tree",
                            "tree_level": level,
                        }
                        if ready_time_tree:
                            attrs["placement_policy"] = "ready_time"
                        if scope is not None:
                            attrs["hierarchical_scope"] = scope
                            attrs["cluster_id"] = topology.cluster_id(reduce_sm)
                        reduce_instruction = emit_streaming(
                            Instruction(
                                instruction_id=next_instruction_id,
                                opcode=DataflowOpcode.REDUCE_UPDATE,
                                operator_name=program.reduce_stage.reduce_call.name,
                                task_id=task_id,
                                task_coords=coords,
                                sm_id=reduce_sm,
                                input_slots=(left_slot, right_slot),
                                output_slot=acc_slot_id,
                                attrs=attrs,
                            )
                        )
                        next_instruction_id += 1
                        slots.append(
                            SlotPlan(
                                slot_id=acc_slot_id,
                                task_id=task_id,
                                intermediate_type=intermediate_type,
                                role="streaming_acc",
                                producer_instruction_id=reduce_instruction.instruction_id,
                                shared_storage_id=0,
                            )
                        )
                        append_streaming_comm(
                            source_instruction=left_instruction,
                            target_instruction=reduce_instruction,
                            slot_id=left_slot,
                            producer_sm=left_sm,
                            consumer_sm=reduce_sm,
                            task_id=task_id,
                        )
                        append_streaming_comm(
                            source_instruction=right_instruction,
                            target_instruction=reduce_instruction,
                            slot_id=right_slot,
                            producer_sm=right_sm,
                            consumer_sm=reduce_sm,
                            task_id=task_id,
                        )
                        reserve_estimated_instruction(
                            sm_id=reduce_sm,
                            input_sources=((left_slot, left_sm), (right_slot, right_sm)),
                            output_slot=acc_slot_id,
                            duration_us=estimated_streaming_reduce_cost_us(2),
                        )
                        estimated_slot_producer_sm[acc_slot_id] = reduce_sm
                        estimated_slot_tree_level[acc_slot_id] = level
                        estimated_slot_level0_position[acc_slot_id] = max(
                            estimated_slot_level0_position.get(left_slot, 0),
                            estimated_slot_level0_position.get(right_slot, 0),
                        )
                        queue_groups[reduce_sm].append(
                            (
                                level,
                                task_id,
                                pair_index,
                                (reduce_instruction,),
                            )
                        )
                        next_nodes.append((acc_slot_id, reduce_instruction, reduce_sm))
                        pair_index += 1
                    current_nodes = next_nodes
                    level += 1
                return current_nodes[0], level

            level_bucket_tree = config_flag("level_bucket_tree") and not config_flag("ordered_interval_tree") and not partial_only
            direct_leaf_acc = (config_flag("direct_leaf_acc") if direct_leaf_acc is None else bool(direct_leaf_acc)) and not partial_only
            if level_bucket_tree and hierarchical_cross_cluster:
                for chunks in chunks_by_task:
                    leaf_clusters = {topology.cluster_id(chunk.sm_id) for chunk in chunks}
                    if len(leaf_clusters) > 1:
                        level_bucket_tree = False
                        break

            if level_bucket_tree:
                coords_by_task = [task_coords(task_id, normalized_task_extents) for task_id in range(len(chunks_by_task))]
                current_nodes_by_task: list[list[tuple[int, Instruction, int]]] = []
                defer_level0_ready_estimate = level0_queue_order == "long_first"

                for task_id, chunks in enumerate(chunks_by_task):
                    coords = coords_by_task[task_id]
                    leaf_nodes: list[tuple[int, Instruction, int]] = []
                    for chunk in chunks:
                        output_slot_id = next_slot_id
                        next_slot_id += 1
                        group_instructions: list[Instruction] = []
                        iter_instruction = emit_streaming(
                            Instruction(
                                instruction_id=next_instruction_id,
                                opcode=DataflowOpcode.ITER,
                                operator_name=iter_operator_name(chunk.task_range),
                                task_id=task_id,
                                task_coords=coords,
                                sm_id=chunk.sm_id,
                                task_range=chunk.task_range,
                                output_slot=output_slot_id,
                                handler_identity=iter_base_identity,
                                handler_variant_key=iter_handler_variant_key(chunk.task_range),
                            )
                        )
                        group_instructions.append(iter_instruction)
                        next_instruction_id += 1
                        if direct_leaf_acc:
                            slots.append(
                                SlotPlan(
                                    slot_id=output_slot_id,
                                    task_id=task_id,
                                    intermediate_type=intermediate_type,
                                    role="streaming_acc",
                                    producer_instruction_id=iter_instruction.instruction_id,
                                    shared_storage_id=0,
                                )
                            )
                            leaf_nodes.append((output_slot_id, iter_instruction, chunk.sm_id))
                            queue_groups[chunk.sm_id].append(
                                (
                                    0,
                                    task_id,
                                    chunk.part_index,
                                    tuple(group_instructions),
                                )
                            )
                            if not defer_level0_ready_estimate:
                                reserve_estimated_streaming_instruction(
                                    iter_instruction,
                                    level0_position=0,
                                )
                                estimated_slot_tree_level[output_slot_id] = 0
                                estimated_slot_level0_position[output_slot_id] = 0
                            continue

                        partial_slot_id = output_slot_id
                        slots.append(
                            SlotPlan(
                                slot_id=partial_slot_id,
                                task_id=task_id,
                                intermediate_type=intermediate_type,
                                role="partial",
                                producer_instruction_id=iter_instruction.instruction_id,
                                shared_storage_id=1,
                            )
                        )

                        acc_slot_id = next_slot_id
                        next_slot_id += 1
                        reduce_attrs: dict[str, Any] = {
                            "reduce_strategy": "streaming_tree",
                            "tree_level": 0,
                        }
                        if hierarchical_cross_cluster:
                            reduce_attrs["hierarchical_scope"] = "leaf"
                            reduce_attrs["cluster_id"] = topology.cluster_id(chunk.sm_id)
                        reduce_instruction = emit_streaming(
                            Instruction(
                                instruction_id=next_instruction_id,
                                opcode=DataflowOpcode.REDUCE_UPDATE,
                                operator_name=program.reduce_stage.reduce_call.name,
                                task_id=task_id,
                                task_coords=coords,
                                sm_id=chunk.sm_id,
                                input_slots=(partial_slot_id,),
                                output_slot=acc_slot_id,
                                attrs=reduce_attrs,
                            )
                        )
                        group_instructions.append(reduce_instruction)
                        next_instruction_id += 1
                        slots.append(
                            SlotPlan(
                                slot_id=acc_slot_id,
                                task_id=task_id,
                                intermediate_type=intermediate_type,
                                role="streaming_acc",
                                producer_instruction_id=reduce_instruction.instruction_id,
                                shared_storage_id=0,
                            )
                        )
                        leaf_nodes.append((acc_slot_id, reduce_instruction, chunk.sm_id))
                        queue_groups[chunk.sm_id].append(
                            (
                                0,
                                task_id,
                                chunk.part_index,
                                tuple(group_instructions),
                            )
                        )
                        if not defer_level0_ready_estimate:
                            for instruction in group_instructions:
                                reserve_estimated_streaming_instruction(instruction, level0_position=0)
                    if not leaf_nodes:
                        raise ValueError(f"Dataflow streaming tree scheduler produced no chunks for task {task_id}")
                    current_nodes_by_task.append(leaf_nodes)

                if defer_level0_ready_estimate:
                    level0_groups = {sm_id: [group for group in groups if group[0] == 0] for sm_id, groups in queue_groups.items()}
                    ordered_level0_groups = ordered_streaming_queue_groups(
                        topology,
                        level0_groups,
                        order=level0_queue_order,
                        block_size=block_size,
                        force_hbm_comms=force_hbm_comms,
                    )
                    for sm_id in sorted(ordered_level0_groups):
                        for level0_position, (_, _, _, group_instructions) in enumerate(ordered_level0_groups[sm_id]):
                            for instruction in group_instructions:
                                reserve_estimated_streaming_instruction(
                                    instruction,
                                    level0_position=level0_position,
                                )
                                if direct_leaf_acc and instruction.output_slot is not None:
                                    estimated_slot_tree_level[instruction.output_slot] = 0
                                    estimated_slot_level0_position[instruction.output_slot] = level0_position

                finalized_task_ids: set[int] = set()
                level = 1
                while len(finalized_task_ids) < len(current_nodes_by_task):
                    made_progress = False
                    for task_id, current_nodes in enumerate(current_nodes_by_task):
                        if task_id in finalized_task_ids:
                            continue
                        coords = coords_by_task[task_id]
                        if len(current_nodes) == 1:
                            root_slot, root_instruction, root_sm = current_nodes[0]
                            finalize_sm = choose_finalize_sm(root_slot=root_slot, root_sm=root_sm)
                            finalize_attrs: dict[str, Any] = {}
                            if ready_time_finalize and finalize_sm != root_sm:
                                finalize_attrs["placement_policy"] = "ready_time_finalize"
                            finalize_instruction = emit_streaming(
                                Instruction(
                                    instruction_id=next_instruction_id,
                                    opcode=DataflowOpcode.FINALIZE,
                                    operator_name=program.finalize_stage.finalize_call.name,
                                    task_id=task_id,
                                    task_coords=coords,
                                    sm_id=finalize_sm,
                                    input_slots=(root_slot,),
                                    attrs=finalize_attrs,
                                )
                            )
                            append_finalize_comm(
                                root_instruction=root_instruction,
                                finalize_instruction=finalize_instruction,
                                root_slot=root_slot,
                                root_sm=root_sm,
                                finalize_sm=finalize_sm,
                                task_id=task_id,
                            )
                            reserve_estimated_instruction(
                                sm_id=finalize_sm,
                                input_sources=((root_slot, root_sm),),
                                output_slot=None,
                                duration_us=estimated_streaming_finalize_cost_us(),
                            )
                            queue_groups[finalize_sm].append(
                                (
                                    level,
                                    task_id,
                                    0,
                                    (finalize_instruction,),
                                )
                            )
                            next_instruction_id += 1
                            finalized_task_ids.add(task_id)
                            made_progress = True
                            continue

                        next_nodes: list[tuple[int, Instruction, int]] = []
                        pair_index = 0
                        for index in range(0, len(current_nodes), 2):
                            if index + 1 >= len(current_nodes):
                                next_nodes.append(current_nodes[index])
                                continue

                            left_slot, left_instruction, left_sm = current_nodes[index]
                            right_slot, right_instruction, right_sm = current_nodes[index + 1]
                            reduce_sm = choose_reduce_sm(
                                left_slot=left_slot,
                                left_sm=left_sm,
                                right_slot=right_slot,
                                right_sm=right_sm,
                                prefer_late_input=len(current_nodes) == 2,
                            )
                            acc_slot_id = next_slot_id
                            next_slot_id += 1
                            attrs: dict[str, Any] = {
                                "reduce_strategy": "streaming_tree",
                                "tree_level": level,
                            }
                            if ready_time_tree:
                                attrs["placement_policy"] = "ready_time_level_bucket"
                            reduce_instruction = emit_streaming(
                                Instruction(
                                    instruction_id=next_instruction_id,
                                    opcode=DataflowOpcode.REDUCE_UPDATE,
                                    operator_name=program.reduce_stage.reduce_call.name,
                                    task_id=task_id,
                                    task_coords=coords,
                                    sm_id=reduce_sm,
                                    input_slots=(left_slot, right_slot),
                                    output_slot=acc_slot_id,
                                    attrs=attrs,
                                )
                            )
                            next_instruction_id += 1
                            slots.append(
                                SlotPlan(
                                    slot_id=acc_slot_id,
                                    task_id=task_id,
                                    intermediate_type=intermediate_type,
                                    role="streaming_acc",
                                    producer_instruction_id=reduce_instruction.instruction_id,
                                    shared_storage_id=0,
                                )
                            )
                            append_streaming_comm(
                                source_instruction=left_instruction,
                                target_instruction=reduce_instruction,
                                slot_id=left_slot,
                                producer_sm=left_sm,
                                consumer_sm=reduce_sm,
                                task_id=task_id,
                            )
                            append_streaming_comm(
                                source_instruction=right_instruction,
                                target_instruction=reduce_instruction,
                                slot_id=right_slot,
                                producer_sm=right_sm,
                                consumer_sm=reduce_sm,
                                task_id=task_id,
                            )
                            reserve_estimated_instruction(
                                sm_id=reduce_sm,
                                input_sources=((left_slot, left_sm), (right_slot, right_sm)),
                                output_slot=acc_slot_id,
                                duration_us=estimated_streaming_reduce_cost_us(2),
                            )
                            estimated_slot_producer_sm[acc_slot_id] = reduce_sm
                            estimated_slot_tree_level[acc_slot_id] = level
                            estimated_slot_level0_position[acc_slot_id] = max(
                                estimated_slot_level0_position.get(left_slot, 0),
                                estimated_slot_level0_position.get(right_slot, 0),
                            )
                            queue_groups[reduce_sm].append(
                                (
                                    level,
                                    task_id,
                                    pair_index,
                                    (reduce_instruction,),
                                )
                            )
                            next_nodes.append((acc_slot_id, reduce_instruction, reduce_sm))
                            pair_index += 1
                        current_nodes_by_task[task_id] = next_nodes
                        made_progress = True
                    if not made_progress:
                        raise RuntimeError("Dataflow level-bucket streaming tree made no scheduling progress")
                    level += 1
            else:
                for task_id, chunks in enumerate(chunks_by_task):
                    coords = task_coords(task_id, normalized_task_extents)
                    leaf_nodes: list[tuple[int, Instruction, int]] = []

                    for chunk in chunks:
                        output_slot_id = next_slot_id
                        next_slot_id += 1
                        group_instructions: list[Instruction] = []
                        iter_instruction = emit_streaming(
                            Instruction(
                                instruction_id=next_instruction_id,
                                opcode=DataflowOpcode.ITER,
                                operator_name=iter_operator_name(chunk.task_range),
                                task_id=task_id,
                                task_coords=coords,
                                sm_id=chunk.sm_id,
                                task_range=chunk.task_range,
                                output_slot=output_slot_id,
                                handler_identity=iter_base_identity,
                                handler_variant_key=iter_handler_variant_key(chunk.task_range),
                            )
                        )
                        group_instructions.append(iter_instruction)
                        next_instruction_id += 1
                        if direct_leaf_acc:
                            slots.append(
                                SlotPlan(
                                    slot_id=output_slot_id,
                                    task_id=task_id,
                                    intermediate_type=intermediate_type,
                                    role="streaming_acc",
                                    producer_instruction_id=iter_instruction.instruction_id,
                                    shared_storage_id=0,
                                )
                            )
                            leaf_nodes.append((output_slot_id, iter_instruction, chunk.sm_id))
                            reserve_estimated_instruction(
                                sm_id=chunk.sm_id,
                                input_sources=(),
                                output_slot=output_slot_id,
                                duration_us=estimated_streaming_iter_cost_us(chunk.task_range, block_size),
                            )
                            estimated_slot_producer_sm[output_slot_id] = chunk.sm_id
                            estimated_slot_tree_level[output_slot_id] = 0
                            estimated_slot_level0_position[output_slot_id] = 0
                            queue_groups[chunk.sm_id].append(
                                (
                                    0,
                                    task_id,
                                    chunk.part_index,
                                    tuple(group_instructions),
                                )
                            )
                            continue

                        partial_slot_id = output_slot_id
                        slots.append(
                            SlotPlan(
                                slot_id=partial_slot_id,
                                task_id=task_id,
                                intermediate_type=intermediate_type,
                                role="partial",
                                producer_instruction_id=iter_instruction.instruction_id,
                                shared_storage_id=1,
                            )
                        )
                        if partial_only:
                            queue_groups[chunk.sm_id].append(
                                (
                                    0,
                                    task_id,
                                    chunk.part_index,
                                    tuple(group_instructions),
                                )
                            )
                            continue

                        acc_slot_id = next_slot_id
                        next_slot_id += 1
                        reduce_attrs: dict[str, Any] = {
                            "reduce_strategy": "streaming_tree",
                            "tree_level": 0,
                        }
                        if hierarchical_cross_cluster:
                            reduce_attrs["hierarchical_scope"] = "leaf"
                            reduce_attrs["cluster_id"] = topology.cluster_id(chunk.sm_id)
                        reduce_instruction = emit_streaming(
                            Instruction(
                                instruction_id=next_instruction_id,
                                opcode=DataflowOpcode.REDUCE_UPDATE,
                                operator_name=program.reduce_stage.reduce_call.name,
                                task_id=task_id,
                                task_coords=coords,
                                sm_id=chunk.sm_id,
                                input_slots=(partial_slot_id,),
                                output_slot=acc_slot_id,
                                attrs=reduce_attrs,
                            )
                        )
                        group_instructions.append(reduce_instruction)
                        next_instruction_id += 1
                        slots.append(
                            SlotPlan(
                                slot_id=acc_slot_id,
                                task_id=task_id,
                                intermediate_type=intermediate_type,
                                role="streaming_acc",
                                producer_instruction_id=reduce_instruction.instruction_id,
                                shared_storage_id=0,
                            )
                        )
                        leaf_nodes.append((acc_slot_id, reduce_instruction, chunk.sm_id))
                        partial_ready = reserve_estimated_instruction(
                            sm_id=chunk.sm_id,
                            input_sources=(),
                            output_slot=partial_slot_id,
                            duration_us=estimated_streaming_iter_cost_us(chunk.task_range, block_size),
                        )
                        estimated_slot_ready[partial_slot_id] = partial_ready
                        reserve_estimated_instruction(
                            sm_id=chunk.sm_id,
                            input_sources=((partial_slot_id, chunk.sm_id),),
                            output_slot=acc_slot_id,
                            duration_us=estimated_streaming_reduce_cost_us(1),
                        )
                        queue_groups[chunk.sm_id].append(
                            (
                                0,
                                task_id,
                                chunk.part_index,
                                tuple(group_instructions),
                            )
                        )

                    if partial_only:
                        continue
                    if not leaf_nodes:
                        raise ValueError(f"Dataflow streaming tree scheduler produced no chunks for task {task_id}")

                    cluster_runs = ordered_cluster_run_bounds(
                        topology,
                        tuple(sm_id for _, _, sm_id in leaf_nodes),
                    )
                    if hierarchical_cross_cluster and len(cluster_runs) > 1:
                        local_roots: list[tuple[int, Instruction, int]] = []
                        local_next_levels: list[int] = []
                        for run_index, (cluster_id, begin, end) in enumerate(cluster_runs):
                            local_root, local_next_level = emit_streaming_tree_nodes(
                                task_id=task_id,
                                coords=coords,
                                nodes=leaf_nodes[begin:end],
                                start_level=1,
                                scope="local",
                            )
                            local_root_instruction = local_root[1]
                            local_root_instruction.attrs["hierarchical_role"] = "local_root"
                            local_root_instruction.attrs["hierarchical_scope"] = "local"
                            local_root_instruction.attrs["cluster_id"] = cluster_id
                            local_root_instruction.attrs["hierarchical_run_index"] = run_index
                            local_roots.append(local_root)
                            local_next_levels.append(local_next_level)
                        if skip_cross_cluster_reduce:
                            local_finalize_level = max(local_next_levels, default=1)
                            for local_index, (root_slot, root_instruction, root_sm) in enumerate(local_roots):
                                finalize_sm = choose_finalize_sm(root_slot=root_slot, root_sm=root_sm)
                                finalize_attrs: dict[str, Any] = {
                                    "diagnostic_skip_cross_cluster_reduce": True,
                                    "local_root_cluster_id": topology.cluster_id(root_sm),
                                }
                                if ready_time_finalize and finalize_sm != root_sm:
                                    finalize_attrs["placement_policy"] = "ready_time_finalize"
                                finalize_instruction = emit_streaming(
                                    Instruction(
                                        instruction_id=next_instruction_id,
                                        opcode=DataflowOpcode.FINALIZE,
                                        operator_name=program.finalize_stage.finalize_call.name,
                                        task_id=task_id,
                                        task_coords=coords,
                                        sm_id=finalize_sm,
                                        input_slots=(root_slot,),
                                        attrs=finalize_attrs,
                                    )
                                )
                                append_finalize_comm(
                                    root_instruction=root_instruction,
                                    finalize_instruction=finalize_instruction,
                                    root_slot=root_slot,
                                    root_sm=root_sm,
                                    finalize_sm=finalize_sm,
                                    task_id=task_id,
                                )
                                reserve_estimated_instruction(
                                    sm_id=finalize_sm,
                                    input_sources=((root_slot, root_sm),),
                                    output_slot=None,
                                    duration_us=estimated_streaming_finalize_cost_us(),
                                )
                                queue_groups[finalize_sm].append(
                                    (
                                        local_finalize_level,
                                        task_id,
                                        local_index,
                                        (finalize_instruction,),
                                    )
                                )
                                next_instruction_id += 1
                            continue
                        global_root, level = emit_streaming_tree_nodes(
                            task_id=task_id,
                            coords=coords,
                            nodes=local_roots,
                            start_level=max(local_next_levels, default=1),
                            scope="global",
                        )
                        global_root[1].attrs["hierarchical_role"] = "global_root"
                        root_slot, root_instruction, root_sm = global_root
                    else:
                        root_node, level = emit_streaming_tree_nodes(
                            task_id=task_id,
                            coords=coords,
                            nodes=leaf_nodes,
                            start_level=1,
                        )
                        root_slot, root_instruction, root_sm = root_node
                    finalize_sm = choose_finalize_sm(root_slot=root_slot, root_sm=root_sm)
                    finalize_attrs: dict[str, Any] = {}
                    if ready_time_finalize and finalize_sm != root_sm:
                        finalize_attrs["placement_policy"] = "ready_time_finalize"
                    finalize_instruction = emit_streaming(
                        Instruction(
                            instruction_id=next_instruction_id,
                            opcode=DataflowOpcode.FINALIZE,
                            operator_name=program.finalize_stage.finalize_call.name,
                            task_id=task_id,
                            task_coords=coords,
                            sm_id=finalize_sm,
                            input_slots=(root_slot,),
                            attrs=finalize_attrs,
                        )
                    )
                    append_finalize_comm(
                        root_instruction=root_instruction,
                        finalize_instruction=finalize_instruction,
                        root_slot=root_slot,
                        root_sm=root_sm,
                        finalize_sm=finalize_sm,
                        task_id=task_id,
                    )
                    reserve_estimated_instruction(
                        sm_id=finalize_sm,
                        input_sources=((root_slot, root_sm),),
                        output_slot=None,
                        duration_us=estimated_streaming_finalize_cost_us(),
                    )
                    queue_groups[finalize_sm].append(
                        (
                            level,
                            task_id,
                            0,
                            (finalize_instruction,),
                        )
                    )
                    next_instruction_id += 1

        else:
            for task_id, chunks in enumerate(chunks_by_task):
                coords = task_coords(task_id, normalized_task_extents)
                previous_acc_slot: int | None = None
                previous_reduce_instruction: Instruction | None = None
                previous_reduce_sm: int | None = None
                last_group: list[Instruction] | None = None

                for chunk in chunks:
                    partial_slot_id = next_slot_id
                    next_slot_id += 1
                    group_instructions: list[Instruction] = []
                    iter_instruction = emit_streaming(
                        Instruction(
                            instruction_id=next_instruction_id,
                            opcode=DataflowOpcode.ITER,
                            operator_name=iter_operator_name(chunk.task_range),
                            task_id=task_id,
                            task_coords=coords,
                            sm_id=chunk.sm_id,
                            task_range=chunk.task_range,
                            output_slot=partial_slot_id,
                            handler_identity=iter_base_identity,
                            handler_variant_key=iter_handler_variant_key(chunk.task_range),
                        )
                    )
                    group_instructions.append(iter_instruction)
                    next_instruction_id += 1
                    slots.append(
                        SlotPlan(
                            slot_id=partial_slot_id,
                            task_id=task_id,
                            intermediate_type=intermediate_type,
                            role="partial",
                            producer_instruction_id=iter_instruction.instruction_id,
                            shared_storage_id=1,
                        )
                    )
                    if partial_only:
                        queue_groups[chunk.sm_id].append(
                            (
                                streaming_queue_priority(chunk),
                                task_id,
                                chunk.part_index,
                                tuple(group_instructions),
                            )
                        )
                        continue

                    acc_slot_id = next_slot_id
                    next_slot_id += 1
                    input_slots = (partial_slot_id,) if previous_acc_slot is None else (previous_acc_slot, partial_slot_id)
                    reduce_instruction = emit_streaming(
                        Instruction(
                            instruction_id=next_instruction_id,
                            opcode=DataflowOpcode.REDUCE_UPDATE,
                            operator_name=program.reduce_stage.reduce_call.name,
                            task_id=task_id,
                            task_coords=coords,
                            sm_id=chunk.sm_id,
                            input_slots=input_slots,
                            output_slot=acc_slot_id,
                            attrs={"reduce_strategy": "streaming"},
                        )
                    )
                    group_instructions.append(reduce_instruction)
                    next_instruction_id += 1
                    slots.append(
                        SlotPlan(
                            slot_id=acc_slot_id,
                            task_id=task_id,
                            intermediate_type=intermediate_type,
                            role="streaming_acc",
                            producer_instruction_id=reduce_instruction.instruction_id,
                            shared_storage_id=0,
                        )
                    )

                    if previous_acc_slot is not None and previous_reduce_instruction is not None and previous_reduce_sm is not None:
                        append_streaming_comm(
                            source_instruction=previous_reduce_instruction,
                            target_instruction=reduce_instruction,
                            slot_id=previous_acc_slot,
                            producer_sm=previous_reduce_sm,
                            consumer_sm=chunk.sm_id,
                            task_id=task_id,
                        )

                    previous_acc_slot = acc_slot_id
                    previous_reduce_instruction = reduce_instruction
                    previous_reduce_sm = chunk.sm_id
                    last_group = group_instructions
                    queue_groups[chunk.sm_id].append(
                        (
                            streaming_queue_priority(chunk),
                            task_id,
                            chunk.part_index,
                            tuple(group_instructions),
                        )
                    )

                if partial_only:
                    continue
                if previous_acc_slot is None or previous_reduce_sm is None or last_group is None:
                    raise ValueError(f"Dataflow streaming scheduler produced no chunks for task {task_id}")
                finalize_instruction = emit_streaming(
                    Instruction(
                        instruction_id=next_instruction_id,
                        opcode=DataflowOpcode.FINALIZE,
                        operator_name=program.finalize_stage.finalize_call.name,
                        task_id=task_id,
                        task_coords=coords,
                        sm_id=previous_reduce_sm,
                        input_slots=(previous_acc_slot,),
                    )
                )
                assert last_group is not None
                last_group.append(finalize_instruction)
                group_priority, group_task_id, group_part_index, _ = queue_groups[previous_reduce_sm][-1]
                queue_groups[previous_reduce_sm][-1] = (
                    group_priority,
                    group_task_id,
                    group_part_index,
                    tuple(last_group),
                )
                next_instruction_id += 1

        ordered_queue_groups = ordered_streaming_queue_groups(
            topology,
            queue_groups,
            order=level0_queue_order,
            block_size=block_size,
            force_hbm_comms=force_hbm_comms,
        )
        for sm_id, groups in ordered_queue_groups.items():
            for _, _, _, group_instructions in groups:
                queues[sm_id].extend(group_instructions)

        if include_exit:
            for sm_id in range(topology.sm_count):
                emit(
                    Instruction(
                        instruction_id=next_instruction_id,
                        opcode=DataflowOpcode.EXIT,
                        operator_name="exit",
                        task_id=None,
                        sm_id=sm_id,
                    )
                )
                next_instruction_id += 1

        frozen_queues = {sm_id: tuple(items) for sm_id, items in queues.items()}
        plan = InstructionPlan(
            topology=topology,
            block_size=block_size,
            range_axis=range_axis,
            scheduler_policy=normalized_scheduler_policy,
            reduce_strategy=normalized_reduce_strategy,
            task_extents=normalized_task_extents,
            task_range_lengths=task_range_lengths,
            instructions=tuple(all_instructions),
            queues=frozen_queues,
            slots=tuple(slots),
            comms=tuple(comms),
            scheduler_config=current_scheduler_config(),
            target_capabilities=target_capabilities,
        )
        plan = attach_handler_identities(program, plan)
        plan = attach_joint_execution_plan(program, plan)
        maybe_dump_schedule_visualization(plan, pic=pic, pic_dir=pic_dir)
        return plan

    for task_id, length in enumerate(task_range_lengths):
        coords = task_coords(task_id, normalized_task_extents)
        partial_slots: list[int] = []
        iter_instructions: list[Instruction] = []

        for range_index, task_range in enumerate(partition_range(length, block_size, range_axis, offset=task_range_offsets[task_id])):
            if normalized_scheduler_policy == "round_robin":
                sm_id = next_iter_sm % topology.sm_count
                next_iter_sm += 1
            else:
                sm_id = cluster_local_sm_id(
                    topology,
                    task_id=task_id,
                    range_index=range_index,
                )

            slot_id = next_slot_id
            next_slot_id += 1
            partial_slots.append(slot_id)

            instruction = emit(
                Instruction(
                    instruction_id=next_instruction_id,
                    opcode=DataflowOpcode.ITER,
                    operator_name=iter_operator_name(task_range),
                    task_id=task_id,
                    task_coords=coords,
                    sm_id=sm_id,
                    task_range=task_range,
                    output_slot=slot_id,
                    handler_identity=iter_base_identity,
                    handler_variant_key=iter_handler_variant_key(task_range),
                )
            )
            next_instruction_id += 1
            iter_instructions.append(instruction)
            slots.append(
                SlotPlan(
                    slot_id=slot_id,
                    task_id=task_id,
                    intermediate_type=intermediate_type,
                    role="partial",
                    producer_instruction_id=instruction.instruction_id,
                )
            )

        if partial_only:
            continue

        reduce_sm = iter_instructions[0].sm_id
        assert reduce_sm is not None
        reduced_slot_id = next_slot_id
        next_slot_id += 1

        reduce_instruction = emit(
            Instruction(
                instruction_id=next_instruction_id,
                opcode=DataflowOpcode.REDUCE,
                operator_name=program.reduce_stage.reduce_call.name,
                task_id=task_id,
                task_coords=coords,
                sm_id=reduce_sm,
                input_slots=tuple(partial_slots),
                output_slot=reduced_slot_id,
            )
        )
        next_instruction_id += 1
        slots.append(
            SlotPlan(
                slot_id=reduced_slot_id,
                task_id=task_id,
                intermediate_type=intermediate_type,
                role="reduced",
                producer_instruction_id=reduce_instruction.instruction_id,
            )
        )

        for iter_instruction, slot_id in zip(iter_instructions, partial_slots):
            assert iter_instruction.sm_id is not None
            if iter_instruction.sm_id == reduce_sm:
                continue

            same_cluster = topology.same_cluster(iter_instruction.sm_id, reduce_sm) and not force_hbm_comms
            send_kind = DataflowCommKind.CLUSTER_SEND if same_cluster else DataflowCommKind.HBM_SEND
            recv_kind = DataflowCommKind.CLUSTER_RECV if same_cluster else DataflowCommKind.HBM_RECV
            flag_epoch = task_id + 1 if not same_cluster else 1
            barrier_phase = 0
            comms.append(
                CommPlan(
                    source_instruction_id=iter_instruction.instruction_id,
                    target_instruction_id=reduce_instruction.instruction_id,
                    source_slot_id=slot_id,
                    target_slot_id=slot_id,
                    producer_sm=iter_instruction.sm_id,
                    consumer_sm=reduce_sm,
                    kind=send_kind,
                    dispatch_instruction_id=iter_instruction.instruction_id,
                    peer_cta_rank=topology.cluster_rank(reduce_sm) if same_cluster else None,
                    flag_epoch=flag_epoch,
                    barrier_phase=barrier_phase,
                )
            )
            comms.append(
                CommPlan(
                    source_instruction_id=iter_instruction.instruction_id,
                    target_instruction_id=reduce_instruction.instruction_id,
                    source_slot_id=slot_id,
                    target_slot_id=slot_id,
                    producer_sm=iter_instruction.sm_id,
                    consumer_sm=reduce_sm,
                    kind=recv_kind,
                    dispatch_instruction_id=reduce_instruction.instruction_id,
                    peer_cta_rank=(topology.cluster_rank(iter_instruction.sm_id) if same_cluster else None),
                    flag_epoch=flag_epoch,
                    barrier_phase=barrier_phase,
                )
            )

        emit(
            Instruction(
                instruction_id=next_instruction_id,
                opcode=DataflowOpcode.FINALIZE,
                operator_name=program.finalize_stage.finalize_call.name,
                task_id=task_id,
                task_coords=coords,
                sm_id=reduce_sm,
                input_slots=(reduced_slot_id,),
            )
        )
        next_instruction_id += 1

    if include_exit:
        for sm_id in range(topology.sm_count):
            emit(
                Instruction(
                    instruction_id=next_instruction_id,
                    opcode=DataflowOpcode.EXIT,
                    operator_name="exit",
                    task_id=None,
                    sm_id=sm_id,
                )
            )
            next_instruction_id += 1

    frozen_queues = {sm_id: tuple(items) for sm_id, items in queues.items()}
    plan = InstructionPlan(
        topology=topology,
        block_size=block_size,
        range_axis=range_axis,
        scheduler_policy=normalized_scheduler_policy,
        reduce_strategy=normalized_reduce_strategy,
        task_extents=normalized_task_extents,
        task_range_lengths=task_range_lengths,
        instructions=tuple(all_instructions),
        queues=frozen_queues,
        slots=tuple(slots),
        comms=tuple(comms),
        scheduler_config=current_scheduler_config(),
        target_capabilities=target_capabilities,
    )
    plan = attach_handler_identities(program, plan)
    plan = attach_joint_execution_plan(program, plan)
    maybe_dump_schedule_visualization(plan, pic=pic, pic_dir=pic_dir)
    return plan


def maybe_dump_schedule_visualization(plan: InstructionPlan, *, pic: bool, pic_dir: Any) -> None:
    if not (pic or config_flag("schedule_pic")):
        return
    from .scheduler_viz import dump_schedule_visualization

    dump_schedule_visualization(
        plan,
        output_dir=config_string("schedule_pic_dir", str(pic_dir)),
    )


def config_flag(name: str) -> bool:
    return bool(current_scheduler_config().get(name, False))


def joint_max_resident_remote_inputs() -> int:
    """Return the number of independently lifetimed joint inbox resources."""

    if not (
        config_flag("joint_schedule")
        and (config_flag("joint_hbm_async_receive_pipeline") or config_flag("joint_cluster_async_receive_pipeline"))
    ):
        return 2
    return max(2, config_int("ordered_tree_max_reduce_arity", 2))


def bool_value(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"Dataflow scheduler option {name!r} must be a bool, got {value!r}")


def normalize_cluster_task_assignment(
    value: Sequence[Sequence[int]] | str | None,
    *,
    task_count: int,
    cluster_count: int,
) -> tuple[tuple[int, ...], ...] | None:
    if value is None:
        value = config_string("cluster_task_assignment", None)
    if value is None or value == "":
        return None

    if isinstance(value, str):
        clusters: list[tuple[int, ...]] = []
        for cluster_text in value.split("|"):
            stripped = cluster_text.strip()
            if not stripped:
                clusters.append(())
                continue
            try:
                clusters.append(tuple(int(item.strip()) for item in stripped.split(",") if item.strip()))
            except ValueError as exc:
                raise ValueError(f"cluster_task_assignment must contain integer task ids, got {value!r}") from exc
        assignment = tuple(clusters)
    else:
        assignment = tuple(tuple(int(task_id) for task_id in cluster) for cluster in value)

    if len(assignment) != cluster_count:
        raise ValueError(
            f"Dataflow cluster_task_assignment must provide one task-id group per cluster; expected {cluster_count}, got {len(assignment)}"
        )

    seen: list[int] = []
    for cluster in assignment:
        for task_id in cluster:
            if task_id < 0 or task_id >= task_count:
                raise ValueError(f"Dataflow cluster_task_assignment task id out of range: {task_id}; expected 0 <= task_id < {task_count}")
            seen.append(task_id)

    expected = list(range(task_count))
    if sorted(seen) != expected:
        raise ValueError(f"Dataflow cluster_task_assignment must cover each task exactly once; expected {expected}, got {seen}")

    return assignment


def config_int(name: str, default: int) -> int:
    return int(current_scheduler_config().get(name, default))


def config_int_set(name: str) -> frozenset[int]:
    return frozenset(int(item) for item in current_scheduler_config().get(name, ()))


def config_float(name: str, default: float) -> float:
    return float(current_scheduler_config().get(name, default))


def config_string(name: str, default: str | None) -> str | None:
    value = current_scheduler_config().get(name, default)
    return None if value is None else str(value)
