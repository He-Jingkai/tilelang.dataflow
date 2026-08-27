"""Versioned, read-only provenance for Dataflow compile decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from tilelang import _ffi_api, tvm
from tilelang.utils.target_capabilities import TargetCapabilitySnapshot
from tvm import ir, tir

from .architecture_contract import DATAFLOW_LOWERING_BOUNDARY_CONTRACT
from .compile_config import (
    DATAFLOW_COMPILE_MODE_DEBUG,
    DATAFLOW_COMPILE_MODE_EXECUTABLE,
    DATAFLOW_COMPILE_MODE_INSPECT,
)
from .gemm_lowering import GEMM_LOWERING_REGISTRY_VERSION
from .execution_planning import (
    DATAFLOW_EXECUTION_PLANNER_IMPLEMENTATION,
    DataflowExecutionPlan,
)
from .handoff_planning import (
    DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR,
    DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_VERSION,
    DATAFLOW_CROSS_HANDLER_HANDOFF_PLANNER_VERSION,
    DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_STAGE_ATTR,
    DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_TRANSFER_ATTR,
    DataflowCrossHandlerHandoffPlan,
    DataflowHandoffQueueBinding,
    fallback_cross_handler_handoff_plan,
    plan_cross_handler_handoff,
)
from .handler import PRIMFUNC_HANDLER_LOWERING
from .implementation_registry import (
    DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION,
    dataflow_implementation_registry,
    scheduler_option_implementation_id,
)
from .operation_contracts import (
    DATAFLOW_HANDOFF_CONTRACT_ATTR,
    DATAFLOW_LAYOUT_CONTRACTS_ATTR,
    DATAFLOW_OPERATION_CONTRACT_FINGERPRINT,
    DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
    DATAFLOW_PIPELINE_CONTRACT_ATTR,
    DATAFLOW_PIPELINE_PLAN_IMPLEMENTATION,
    DATAFLOW_PIPELINE_MATERIALIZATIONS,
    DATAFLOW_PIPELINE_MATERIALIZE_COPY,
    DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT,
    DATAFLOW_PIPELINE_RESIDENT_IMPLEMENTATION,
    DATAFLOW_RANGE_CONTRACT_ATTR,
    DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION,
    DATAFLOW_TRANSPORT_CONTRACT_ATTR,
    DATAFLOW_TRANSPORT_STREAMED,
    DataflowOperationDecision,
    DataflowOperationRequest,
    DataflowOperationResourceEstimate,
    DataflowCrossHandlerHandoffRequest,
    DataflowPipelineRequest,
    DataflowRangeCoarseningRequest,
    DataflowResharedTransportRequest,
    DataflowTensorLayoutRequest,
    layout_implementation_id,
    dataflow_operation_contract_schema_dict,
    dataflow_operation_request_from_dict,
    selected_operation_decision,
)
from .runtime import (
    DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
    DATAFLOW_SLOT_FLAG_SCRATCH_BACKED,
)
from .range_coarsening import DataflowRangeCoarseningPlan
from .pipeline_planning import (
    DATAFLOW_PIPELINE_MATERIALIZATION_ATTR,
    DATAFLOW_PIPELINE_MODE_SYNCHRONOUS,
    DataflowPipelinePlan,
    plan_pipeline_dataflow,
)
from .physical_contract import (
    DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_ATTR,
    DATAFLOW_OPERATOR_PHYSICAL_IMPLEMENTATION,
    DATAFLOW_OUTPUT_SLOT_NONE,
    DATAFLOW_TERMINAL_SIDE_EFFECT_IMPLEMENTATION,
    DataflowOperatorPhysicalContract,
    operator_physical_contract,
)
from .scheduler_replay import instruction_plan_fingerprint
from .reshared_transport import (
    DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_VERSION,
    DATAFLOW_RESHARED_TRANSPORT_PLANNER_VERSION,
    DataflowResharedTransportPlan,
    plan_reshared_transport,
)
from .specialization import (
    DATAFLOW_SPECIALIZATION_CAPTURE_IMPLEMENTATION,
    DataflowSpecializationSnapshot,
    specialization_snapshot_from_attrs,
)
from .tensor_layout import (
    DATAFLOW_TENSOR_ARGUMENT_LAYOUT_IMPLEMENTATION,
    DataflowTensorArgumentLayout,
)


DATAFLOW_DECISION_ARTIFACT_SCHEMA_VERSION = 20
DATAFLOW_PIPELINE_BINDING_SCHEMA_VERSION = 3
DATAFLOW_RESHARED_TRANSPORT_BINDING_SCHEMA_VERSION = 1
DATAFLOW_CROSS_HANDLER_HANDOFF_BINDING_SCHEMA_VERSION = 2


_TRANSFER_SYNC_OWNER_NAMES = {
    0: "transfer",
    1: "pipeline",
    2: "caller",
}
_TRANSFER_PIPELINE_SYNC_MANAGED = 1
_TRANSFER_PIPELINE_SYNC_FALLBACK = 2


@dataclass(frozen=True)
class DataflowDecisionArtifact:
    """Canonical snapshot derived from an already compiled Dataflow program.

    The artifact is observational: it never feeds compilation, scheduling, or
    cache keys.  Its fingerprint therefore identifies the recorded decisions
    without becoming part of the compiled program's artifact fingerprint.
    """

    schema_version: int
    canonical_json: str
    fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_DECISION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Dataflow decision artifact schema version "
                f"{self.schema_version}; expected "
                f"{DATAFLOW_DECISION_ARTIFACT_SCHEMA_VERSION}"
            )
        try:
            payload = json.loads(self.canonical_json)
        except (TypeError, json.JSONDecodeError) as err:
            raise ValueError("Dataflow decision artifact canonical_json must be valid JSON") from err
        if not isinstance(payload, dict):
            raise TypeError("Dataflow decision artifact payload must be a JSON object")
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("Dataflow decision artifact payload schema version does not match")
        canonical_json = serialize_canonical_json(payload)
        if canonical_json != self.canonical_json:
            raise ValueError("Dataflow decision artifact JSON must use canonical serialization")
        if sha256(canonical_json) != self.fingerprint:
            raise ValueError("Dataflow decision artifact fingerprint does not match its payload")
        validate_dataflow_decision_artifact_payload(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DataflowDecisionArtifact:
        if not isinstance(payload, Mapping):
            raise TypeError(f"Dataflow decision artifact payload must be a mapping, got {type(payload)!r}")
        if "fingerprint" in payload:
            raise ValueError("Dataflow decision artifact payload cannot contain its fingerprint")
        normalized = dict(payload)
        schema_version = int(
            normalized.setdefault(
                "schema_version",
                DATAFLOW_DECISION_ARTIFACT_SCHEMA_VERSION,
            )
        )
        canonical_json = serialize_canonical_json(normalized)
        return cls(
            schema_version=schema_version,
            canonical_json=canonical_json,
            fingerprint=sha256(canonical_json),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = json.loads(self.canonical_json)
        payload["fingerprint"] = self.fingerprint
        return payload


def collect_dataflow_decision_artifact(compiled_program: Any) -> DataflowDecisionArtifact:
    """Collect generic selection and placement metadata after compilation."""

    lowering = compiled_program.primfunc_lowering
    lowered_handlers = {} if lowering is None else {handler.handler_id: handler for handler in lowering.handlers}
    handler_calls = tuple(
        stage.call for stage in compiled_program.program.stages if stage.call is not None and stage.kind.value != "reshared"
    )
    handlers = []
    for handler_id, identity in enumerate(compiled_program.packed_plan.handler_identities):
        variant_key = compiled_program.packed_plan.handler_variant_keys[handler_id]
        lowered = lowered_handlers.get(handler_id)
        call = None if identity is None or identity.operator_id >= len(handler_calls) else handler_calls[identity.operator_id]
        specialization = DataflowSpecializationSnapshot() if call is None else specialization_snapshot_from_attrs(call.operator.attrs)
        physical = DataflowOperatorPhysicalContract() if call is None else operator_physical_contract(call.operator.attrs)
        handlers.append(
            {
                "handler_id": handler_id,
                "binding_key": (None if identity is None else [identity.operator_id, identity.operator_kind]),
                "operator_kind": (None if identity is None else identity.operator_kind),
                "base_identity": (None if identity is None else identity.to_dict()),
                "variant_key": (None if variant_key is None else variant_key.to_dict()),
                "specialization": specialization.to_dict(),
                "physical_contract": physical.to_dict(),
                "lowering_state": "not_lowered" if lowered is None else "lowered",
                "thread_count": None if lowered is None else lowered.thread_count,
                "dynamic_shared_bytes": (None if lowered is None else lowered.dynamic_shared_bytes),
                "reducer_contract": (None if lowered is None or lowered.reducer_contract is None else lowered.reducer_contract.value),
            }
        )

    handler_id_by_variant = {
        handler.handler_variant_key: handler.handler_id
        for handler in compiled_program.wrapper_spec.handlers
        if handler.handler_variant_key is not None
    }
    packed_slots_by_id = {
        slot.slot_id: packed
        for slot, packed in zip(
            compiled_program.plan.slots,
            compiled_program.packed_plan.slots,
        )
    }
    reductions = []
    for instruction in compiled_program.plan.instructions:
        variant_key = instruction.handler_variant_key
        if variant_key is None or variant_key.reduce_arity is None:
            continue
        arity_class = variant_key.reduce_arity.arity_class
        forward = instruction.value_forward
        forward_decision = None
        if forward is not None:
            source = packed_slots_by_id[forward.source_slot_id]
            output = packed_slots_by_id[forward.output_slot_id]

            def location(packed_slot):
                if (
                    compiled_program.wrapper_spec.primfunc_use_global_slot_fields
                    and packed_slot.flags & DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL
                ):
                    return ("global", packed_slot.global_offset)
                if packed_slot.flags & DATAFLOW_SLOT_FLAG_SCRATCH_BACKED:
                    return ("scratch", packed_slot.shared_offset)
                return ("shared", packed_slot.shared_offset)

            source_space, source_offset = location(source)
            output_space, output_offset = location(output)
            selected_storage = (
                "non_owning_alias"
                if source_space == output_space and source_offset == output_offset and source.bytes == output.bytes
                else "copy"
            )
            forward_decision = {
                **forward.to_dict(),
                "selected_storage": selected_storage,
                "allocation_owner_slot_id": (
                    forward.alias_owner_slot_id if selected_storage == "non_owning_alias" else forward.copy_owner_slot_id
                ),
                "memory_space": source_space,
            }
        reductions.append(
            {
                "instruction_id": instruction.instruction_id,
                "handler_id": handler_id_by_variant.get(variant_key),
                "arity_class": arity_class,
                "input_slots": list(instruction.input_slots),
                "reduction_order": (
                    "value_forward"
                    if arity_class == "passthrough"
                    else "binary_pair"
                    if arity_class == "binary"
                    else "serial_left_fold_plan_input_order"
                ),
                "value_forward": forward_decision,
            }
        )

    gemm_resolutions = []
    pipeline_resolutions = []
    if lowering is not None:
        for handler in lowering.handlers:
            for operation_index, resolution in enumerate(handler.gemm_lowerings):
                resolution_dict = resolution.to_dict()
                request = dict(resolution_dict["request"])
                request.pop("operation_id", None)
                resolution_dict["request"] = request
                gemm_resolutions.append(
                    {
                        "handler_id": handler.handler_id,
                        "operation_index": operation_index,
                        **resolution_dict,
                    }
                )
            for resolution in handler.pipeline_lowerings:
                pipeline_resolutions.append(
                    {
                        "handler_id": handler.handler_id,
                        **dict(resolution),
                    }
                )

    pass_configs = compiled_program.options.get("primfunc_pass_configs") or {}
    with tvm.transform.PassContext(opt_level=3, config=pass_configs):
        transfer_requests = collect_transfer_requests(
            lowering,
            target=transfer_lowering_target(compiled_program.compile_config),
        )
    pipeline_dataflow_bindings = collect_pipeline_dataflow_bindings(
        lowering,
        transfer_requests=transfer_requests,
        gemm_resolutions=gemm_resolutions,
        pipeline_resolutions=pipeline_resolutions,
    )
    reshared_transport_bindings = collect_reshared_transport_bindings(
        compiled_program,
        lowering,
    )
    cross_handler_handoff_bindings = collect_cross_handler_handoff_bindings(
        compiled_program,
        lowering,
        transfer_requests=transfer_requests,
    )

    layout_report = compiled_program.validate_memory_layout()
    layout_regions = tuple(layout_report.regions)
    slot_placements = []
    for slot, packed_slot in zip(
        compiled_program.plan.slots,
        compiled_program.packed_plan.slots,
    ):
        scratch_backed = bool(packed_slot.flags & DATAFLOW_SLOT_FLAG_SCRATCH_BACKED)
        hbm_direct_global = bool(packed_slot.flags & DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL)
        placement = "hbm_direct_global" if hbm_direct_global else "scratch_backed" if scratch_backed else "shared"
        selected_memory_space = "global" if hbm_direct_global else "shared"
        selected_region = next(
            (region for region in layout_regions if region.slot_id == slot.slot_id and region.memory_space == selected_memory_space),
            None,
        )
        slot_placements.append(
            {
                "slot_id": slot.slot_id,
                "role": slot.role,
                "bytes": packed_slot.bytes,
                "placement": placement,
                "offset": (
                    packed_slot.global_offset
                    if hbm_direct_global
                    else compiled_program.wrapper_spec.primfunc_scratch_offset + packed_slot.shared_offset
                    if scratch_backed
                    else compiled_program.launch_package.shared_slot_base_offset + packed_slot.shared_offset
                ),
                "alignment": (None if selected_region is None else selected_region.alignment),
                "live_range": (
                    None if selected_region is None or selected_region.live_range is None else selected_region.live_range.to_dict()
                ),
                "placement_reason": (None if selected_region is None else selected_region.placement_reason),
                "scratch_backed": scratch_backed,
                "hbm_direct_global": hbm_direct_global,
                "shared_storage_id": slot.shared_storage_id,
                "global_storage_id": slot.global_storage_id,
                "barrier_storage_id": slot.barrier_storage_id,
                "allocation_owner_slot_id": (slot.slot_id if slot.allocation_owner_slot_id is None else slot.allocation_owner_slot_id),
                "alias_of_slot_id": slot.alias_of_slot_id,
            }
        )

    launch = compiled_program.launch_package
    scheduler_config = compiled_program.plan.scheduler_config.to_dict()
    compile_options = compiled_program.compile_config.options_dict()
    execution_plans = [DataflowExecutionPlan.from_dict(plan).to_dict() for plan in compile_options.get("execution_plans", ())]
    auto_policy = compile_options.get("scheduler_auto_policy")
    if auto_policy is None:
        auto_policy = {
            "enabled": False,
            "requested": {
                "scheduler_policy": compiled_program.plan.scheduler_policy,
                "reduce_strategy": compiled_program.plan.reduce_strategy,
            },
            "selected_candidate_id": None,
            "selection_reason": "explicit_compile_options",
            "cost_model_version": scheduler_config["cost_model"]["version"],
        }
    tma_descriptors = [descriptor.to_dict() for descriptor in compiled_program.wrapper_spec.tma_descriptors]
    implementation_registry = dataflow_implementation_registry()
    if (
        compile_options["implementation_registry_version"] != DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION
        or compile_options["implementation_registry_fingerprint"] != implementation_registry.fingerprint
    ):
        raise ValueError("Dataflow implementation registry changed after the compile snapshot was created")
    if (
        compile_options["lowering_boundary_schema_version"] != DATAFLOW_LOWERING_BOUNDARY_CONTRACT.schema_version
        or compile_options["lowering_boundary_fingerprint"] != DATAFLOW_LOWERING_BOUNDARY_CONTRACT.fingerprint
    ):
        raise ValueError("Dataflow lowering boundary changed after the compile snapshot was created")
    if (
        compile_options["operation_contract_schema_version"] != DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION
        or compile_options["operation_contract_fingerprint"] != DATAFLOW_OPERATION_CONTRACT_FINGERPRINT
    ):
        raise ValueError("Dataflow operation contract schema changed after the compile snapshot was created")
    operation_contracts = operation_contract_records(compiled_program)
    selected_implementations = selected_implementation_records(
        compiled_program,
        scheduler_config=compiled_program.plan.scheduler_config,
        gemm_resolutions=gemm_resolutions,
        transfer_requests=transfer_requests,
        pipeline_resolutions=pipeline_resolutions,
        compile_options=compile_options,
        operation_contracts=operation_contracts,
        reshared_transport_bindings=reshared_transport_bindings,
        cross_handler_handoff_bindings=cross_handler_handoff_bindings,
    )
    payload = {
        "schema_version": DATAFLOW_DECISION_ARTIFACT_SCHEMA_VERSION,
        "artifact": {
            "artifact_fingerprint": compiled_program.artifact_fingerprint,
            "compile_config_fingerprint": compiled_program.compile_config.fingerprint,
            "program_fingerprint": compiled_program.compile_config.program_fingerprint,
            "target_fingerprint": compiled_program.target_fingerprint,
            "compile_mode": compiled_program.compile_config.mode,
            "handler_lowering": compiled_program.compile_config.handler_lowering,
        },
        "target": compiled_program.target_capabilities.to_dict(),
        "governance": {
            "implementation_registry": implementation_registry.to_dict(),
            "lowering_boundary": DATAFLOW_LOWERING_BOUNDARY_CONTRACT.to_dict(),
            "selected_implementations": selected_implementations,
        },
        "scheduler": {
            "plan_fingerprint": instruction_plan_fingerprint(compiled_program.plan),
            "policy": compiled_program.plan.scheduler_policy,
            "reduce_strategy": compiled_program.plan.reduce_strategy,
            "topology": {
                "sm_count": compiled_program.plan.topology.sm_count,
                "cluster_size": compiled_program.plan.topology.cluster_size,
            },
            "config": scheduler_config,
            "cost_model_version": scheduler_config["cost_model"]["version"],
            "auto_policy": auto_policy,
            "execution_plans": execution_plans,
            "range_resource_budget_bytes": (compiled_program.plan.range_resource_budget_bytes),
        },
        "handlers": {
            "coverage": "base_identity_and_typed_variant_key",
            "items": handlers,
        },
        "tensor_arguments": compiled_program.tensor_arg_plan.to_dict(),
        "operation_contracts": {
            "schema": dataflow_operation_contract_schema_dict(),
            "records": operation_contracts,
        },
        "lowerings": {
            "reduction": {
                "coverage": "contract_variant_order_and_value_forward_ownership",
                "instructions": reductions,
            },
            "gemm": {
                "coverage": "common_logical_physical_plan_and_resource_requirements",
                "registry_version": GEMM_LOWERING_REGISTRY_VERSION,
                "resolutions": gemm_resolutions,
            },
            "copy": {
                "coverage": "selected_transfer_lowering_and_tma_descriptor_metadata",
                "transfer_contract_selection_state": "selected",
                "registry_version": int(_ffi_api.TransferLoweringRegistryVersion()),
                "requests": transfer_requests,
                "tma_descriptors": tma_descriptors,
            },
            "pipeline": {
                "coverage": ("typed_graph_to_transfer_gemm_pipeline_physical_binding"),
                "registry_version": int(compile_options["pipeline_lowering_registry_version"]),
                "resolutions": pipeline_resolutions,
                "dataflow_bindings": pipeline_dataflow_bindings,
            },
            "transport": {
                "coverage": ("typed_family_plan_to_scheduler_and_primfunc_physical_binding"),
                "plan_schema_version": (DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_VERSION),
                "planner_version": DATAFLOW_RESHARED_TRANSPORT_PLANNER_VERSION,
                "binding_schema_version": (DATAFLOW_RESHARED_TRANSPORT_BINDING_SCHEMA_VERSION),
                "bindings": reshared_transport_bindings,
            },
            "handoff": {
                "coverage": ("typed_graph_pipeline_plan_to_queue_runtime_and_primfunc_binding"),
                "plan_schema_version": (DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_VERSION),
                "planner_version": DATAFLOW_CROSS_HANDLER_HANDOFF_PLANNER_VERSION,
                "binding_schema_version": (DATAFLOW_CROSS_HANDLER_HANDOFF_BINDING_SCHEMA_VERSION),
                "bindings": cross_handler_handoff_bindings,
            },
            "precision": {
                "coverage": ("typed_policy_operator_contract_selected_dtype_error_budget_reduction_order_and_implementation"),
                "semantic_config": (compiled_program.compile_config.semantic_config.to_dict()),
                "plan": compile_options["precision_plan"],
            },
        },
        "memory": {
            "planner": compiled_program.memory_plan.to_dict(),
            "shared_memory_bytes": launch.shared_memory_bytes,
            "shared_slot_bytes": launch.shared_slot_bytes,
            "scratch_backed_slot_count": launch.scratch_backed_slot_count,
            "scratch_backed_slot_bytes": launch.scratch_backed_slot_bytes,
            "global_staging_bytes": len(launch.global_staging_bytes),
            "barrier_count": launch.barrier_count,
            "barrier_bytes": launch.barrier_bytes,
            "cluster_ack_count": launch.cluster_ack_count,
            "cluster_ack_barrier_bytes": launch.cluster_ack_barrier_bytes,
            "cluster_ack_pending_bytes": launch.cluster_ack_pending_bytes,
            "cluster_inbox_offset": launch.cluster_inbox_offset,
            "cluster_inbox_bytes": launch.cluster_inbox_bytes,
            "shared_control_bytes": launch.shared_control_bytes,
            "primfunc_scratch_offset": (compiled_program.wrapper_spec.primfunc_scratch_offset),
            "primfunc_scratch_bytes": (compiled_program.wrapper_spec.primfunc_scratch_bytes),
            "handoff_arena_offset": (compiled_program.wrapper_spec.handoff_arena_offset),
            "handoff_arena_bytes": (compiled_program.wrapper_spec.handoff_arena_bytes),
            "handoff_plan_arenas": [arena.to_dict() for arena in compiled_program.wrapper_spec.handoff_plan_arenas],
            "slot_placements": slot_placements,
            "allocation_regions": [region.to_dict() for region in layout_regions],
            "validation": {
                "valid": layout_report.valid,
                "errors": list(layout_report.errors),
                "conflicts": [conflict.to_dict() for conflict in layout_report.conflicts],
                "target_shared_memory_limit": (layout_report.target_shared_memory_limit),
                "shared_memory_within_target_limit": (layout_report.shared_memory_within_target_limit),
            },
        },
    }
    return DataflowDecisionArtifact.from_payload(payload)


def operation_contract_records(
    compiled_program: Any,
) -> list[dict[str, Any]]:
    registry = dataflow_implementation_registry()
    packed_slots = {
        slot.slot_id: packed
        for slot, packed in zip(
            compiled_program.plan.slots,
            compiled_program.packed_plan.slots,
        )
    }

    def stage_slot_bytes(stage_id: int, *, inputs: bool = False) -> int:
        slot_ids: set[int] = set()
        for instruction in compiled_program.plan.instructions:
            if instruction.attrs.get("stage_id") != stage_id:
                continue
            if inputs:
                slot_ids.update(instruction.input_slots)
            elif instruction.output_slot is not None:
                slot_ids.add(instruction.output_slot)
        return sum(packed_slots[slot_id].bytes for slot_id in slot_ids)

    def select(
        request: DataflowOperationRequest,
        implementation_id: str,
        *,
        resources: DataflowOperationResourceEstimate,
        selection_reason: str,
        used_fallback: bool = False,
        selected_explicitly: bool = False,
    ) -> DataflowOperationDecision:
        registry.require_contract_compatible(
            implementation_id,
            request,
            selected_explicitly=selected_explicitly,
        )
        return selected_operation_decision(
            request,
            implementation_id,
            resources=resources,
            selection_reason=selection_reason,
            used_fallback=used_fallback,
        )

    def stage_decisions(stage: Any, request: DataflowOperationRequest):
        selected_explicitly = False
        decisions: list[DataflowOperationDecision] = []
        lowering_plans: list[dict[str, Any]] = []
        if isinstance(request, DataflowRangeCoarseningRequest):
            if compiled_program.program.is_stage_graph:
                range_plans = tuple(
                    range_plan for stage_id, range_plan in compiled_program.plan.range_coarsening_plans if stage_id == stage.stage_id
                )
                if not range_plans:
                    raise ValueError(f"selected range stage {stage.stage_id} lacks a coarsening plan")
                for range_plan in range_plans:
                    if range_plan.request_fingerprint != request.fingerprint:
                        raise ValueError(f"range stage {stage.stage_id} plan changed its typed request")
                lowering_plans.extend(range_plan.to_dict() for range_plan in range_plans)
                used_fallback = any(range_plan.used_fallback for range_plan in range_plans)
                decisions.append(
                    select(
                        request,
                        DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION,
                        resources=DataflowOperationResourceEstimate(
                            slot_bytes=max(range_plan.selected_resource_bytes for range_plan in range_plans)
                        ),
                        selection_reason=("range_resource_fallback" if used_fallback else "typed_range_coarsening_plan"),
                        used_fallback=used_fallback,
                    )
                )
        elif isinstance(request, DataflowPipelineRequest):
            pipeline_plan = plan_pipeline_dataflow(request)
            lowering_plans.append(pipeline_plan.to_dict())
            decisions.append(
                select(
                    request,
                    DATAFLOW_PIPELINE_PLAN_IMPLEMENTATION,
                    resources=pipeline_plan.resources,
                    selection_reason=("pipeline_constraint_fallback" if pipeline_plan.used_fallback else "typed_pipeline_dataflow_plan"),
                    used_fallback=pipeline_plan.used_fallback,
                )
            )
        elif isinstance(request, DataflowResharedTransportRequest):
            selected_explicitly = request.explicit
            transport_plans = tuple(
                transport_plan
                for stage_id, transport_plan in (compiled_program.plan.reshared_transport_plans)
                if stage_id == stage.stage_id
            )
            if len(transport_plans) != 1:
                raise ValueError(f"selected reshared stage {stage.stage_id} requires one transport plan")
            transport_plan = transport_plans[0]
            if transport_plan.request_fingerprint != request.fingerprint:
                raise ValueError(f"reshared stage {stage.stage_id} plan changed its typed request")
            registry.require_contract_compatible(
                transport_plan.implementation_id,
                request,
                selected_explicitly=selected_explicitly,
            )
            lowering_plans.append(transport_plan.to_dict())
            decisions.append(transport_plan.decision)
        elif isinstance(request, DataflowCrossHandlerHandoffRequest):
            handoff_plans = tuple(
                handoff_plan
                for handoff_plan in (compiled_program.plan.cross_handler_handoff_plans)
                if handoff_plan.producer_stage_id == stage.stage_id
            )
            if len(handoff_plans) != 1:
                raise ValueError(f"selected handoff stage {stage.stage_id} requires one plan")
            handoff_plan = handoff_plans[0]
            if handoff_plan.request_fingerprint != request.fingerprint:
                raise ValueError(f"handoff stage {stage.stage_id} plan changed its typed request")
            lowering_plans.append(handoff_plan.to_dict())
            decisions.append(handoff_plan.decision)
        return decisions, selected_explicitly, lowering_plans

    records: list[dict[str, Any]] = []
    stage_contract_attrs = (
        DATAFLOW_RANGE_CONTRACT_ATTR,
        DATAFLOW_PIPELINE_CONTRACT_ATTR,
        DATAFLOW_TRANSPORT_CONTRACT_ATTR,
        DATAFLOW_HANDOFF_CONTRACT_ATTR,
    )
    for stage in compiled_program.program.stages:
        for attr_name in stage_contract_attrs:
            request = stage.attrs.get(attr_name)
            if request is None:
                continue
            if not isinstance(request, DataflowOperationRequest):
                raise TypeError(f"stage {stage.stage_id} has an unnormalized operation contract")
            decisions, selected_explicitly, lowering_plans = stage_decisions(stage, request)
            records.append(
                {
                    "request_id": f"stage.{stage.stage_id}.{request.KIND}",
                    "owner": {
                        "kind": "stage",
                        "stage_id": stage.stage_id,
                        "attribute": attr_name,
                    },
                    "request": request.to_dict(),
                    "selection_state": "selected" if decisions else "declared",
                    "selected_explicitly": selected_explicitly,
                    "decisions": [decision.to_dict() for decision in decisions],
                    "lowering_plans": lowering_plans,
                }
            )

    intermediates: list[Any] = []
    seen_intermediates: set[int] = set()
    lowered_intermediates: set[int] = set()
    for stage in compiled_program.program.stages:
        values = [
            stage.input_type,
            stage.output_type,
            stage.physical_output_type,
        ]
        if stage.call is not None:
            values.extend(stage.call.operator.input_types)
            values.append(stage.call.operator.output_type)
            lowered_intermediates.update(
                id(intermediate)
                for intermediate in (
                    *stage.call.operator.input_types,
                    stage.call.operator.output_type,
                )
                if intermediate is not None
            )
        for intermediate in values:
            if intermediate is None or id(intermediate) in seen_intermediates:
                continue
            seen_intermediates.add(id(intermediate))
            intermediates.append(intermediate)
    for intermediate_index, intermediate in enumerate(intermediates):
        for request in intermediate.attrs.get(DATAFLOW_LAYOUT_CONTRACTS_ATTR, ()):
            if not isinstance(request, DataflowTensorLayoutRequest):
                raise TypeError(f"intermediate {intermediate_index} has an unnormalized layout contract")
            decisions = []
            if compiled_program.primfunc_lowering is not None and id(intermediate) in lowered_intermediates:
                decisions.append(
                    select(
                        request,
                        layout_implementation_id(request),
                        resources=DataflowOperationResourceEstimate(),
                        selection_reason="typed_intermediate_layout",
                        selected_explicitly=True,
                    )
                )
            records.append(
                {
                    "request_id": (f"intermediate.{intermediate_index}.tensor_layout.field.{request.field_index}"),
                    "owner": {
                        "kind": "intermediate",
                        "intermediate_index": intermediate_index,
                        "field_index": request.field_index,
                    },
                    "request": request.to_dict(),
                    "selection_state": "selected" if decisions else "declared",
                    "selected_explicitly": True,
                    "decisions": [decision.to_dict() for decision in decisions],
                    "lowering_plans": [],
                }
            )
    return records


def selected_implementation_records(
    compiled_program: Any,
    *,
    scheduler_config: Any,
    gemm_resolutions: list[dict[str, Any]],
    transfer_requests: list[dict[str, Any]],
    pipeline_resolutions: list[dict[str, Any]],
    compile_options: Mapping[str, Any],
    operation_contracts: list[dict[str, Any]],
    reshared_transport_bindings: list[dict[str, Any]],
    cross_handler_handoff_bindings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selections: dict[str, dict[str, Any]] = {}

    def select(
        implementation_id: str,
        source: str,
        *,
        explicit: bool = False,
    ) -> None:
        record = selections.setdefault(
            implementation_id,
            {"sources": set(), "selected_explicitly": False},
        )
        record["sources"].add(source)
        record["selected_explicitly"] = bool(record["selected_explicitly"] or explicit)

    select(compiled_program.plan.scheduler_policy, "scheduler.policy")
    for stage in compiled_program.program.stages:
        if stage.call is None:
            continue
        snapshot = specialization_snapshot_from_attrs(stage.call.operator.attrs)
        if snapshot.entries:
            select(
                DATAFLOW_SPECIALIZATION_CAPTURE_IMPLEMENTATION,
                f"handler.specialization.stage:{stage.stage_id}",
            )
        if DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_ATTR in stage.call.operator.attrs:
            physical = operator_physical_contract(stage.call.operator.attrs)
            select(
                DATAFLOW_OPERATOR_PHYSICAL_IMPLEMENTATION,
                f"handler.physical.stage:{stage.stage_id}",
                explicit=True,
            )
            if physical.output_slot == DATAFLOW_OUTPUT_SLOT_NONE:
                select(
                    DATAFLOW_TERMINAL_SIDE_EFFECT_IMPLEMENTATION,
                    f"handler.terminal.stage:{stage.stage_id}",
                    explicit=True,
                )
    for spec in compiled_program.tensor_arg_plan.specs:
        if spec.layout is not None:
            select(
                DATAFLOW_TENSOR_ARGUMENT_LAYOUT_IMPLEMENTATION,
                f"tensor_argument:{spec.index}",
                explicit=True,
            )
    execution_plans = tuple(DataflowExecutionPlan.from_dict(plan) for plan in compile_options.get("execution_plans", ()))
    for index, plan in enumerate(execution_plans):
        select(
            DATAFLOW_EXECUTION_PLANNER_IMPLEMENTATION,
            f"scheduler.execution_plan:{index}",
            explicit=plan.explicit_override,
        )
    for option_name in scheduler_config.explicit_option_names():
        select(
            scheduler_option_implementation_id(option_name),
            f"scheduler.config.{option_name}",
            explicit=True,
        )
    semantic_config = compiled_program.compile_config.semantic_config
    if semantic_config.direct_slot_seed_reduce:
        select(
            "semantic.direct_slot_seed_reduce",
            "semantic_config.direct_slot_seed_reduce",
            explicit=True,
        )
    if semantic_config.skip_finalize_post_sync:
        select(
            "semantic.skip_finalize_post_sync",
            "semantic_config.skip_finalize_post_sync",
            explicit=True,
        )
    if semantic_config.fast_math:
        select(
            "semantic.fast_math",
            "semantic_config.fast_math",
            explicit=True,
        )
    for resolution in gemm_resolutions:
        select(
            str(resolution["implementation_id"]),
            f"lowerings.gemm.handler:{resolution['handler_id']}",
        )
    for request in transfer_requests:
        select(
            str(request["selected_implementation"]),
            f"lowerings.copy.handler:{request['handler_id']}",
        )
    for resolution in pipeline_resolutions:
        select(
            str(resolution["selected_implementation"]),
            f"lowerings.pipeline.handler:{resolution['handler_id']}",
        )
    for record in operation_contracts:
        for decision in record["decisions"]:
            implementation_id = decision.get("selected_implementation")
            if implementation_id is not None:
                select(
                    str(implementation_id),
                    f"operation_contracts.{record['request_id']}",
                    explicit=bool(record["selected_explicitly"]),
                )
    for binding in reshared_transport_bindings:
        select(
            str(binding["lowering_implementation_id"]),
            f"lowerings.transport.stage:{binding['stage_id']}",
        )
    for binding in cross_handler_handoff_bindings:
        select(
            str(binding["lowering_implementation_id"]),
            "lowerings.handoff",
        )
    precision_plan = compile_options["precision_plan"]
    for resolution in precision_plan["resolutions"]:
        select(
            str(resolution["implementation_id"]),
            f"lowerings.precision.operator:{resolution['operator_id']}",
        )
    memory_plan = compiled_program.memory_plan
    select(
        f"{memory_plan.planner_version}:{memory_plan.selected_candidate.placement}",
        "memory.selected_candidate",
    )
    if compiled_program.compile_config.mode == "debug":
        select(
            compiled_program.compile_config.handler_lowering,
            "debug.handler_provider",
            explicit=True,
        )

    registry = dataflow_implementation_registry()
    records = []
    for implementation_id, selection in sorted(selections.items()):
        spec = registry.require_selectable(
            implementation_id,
            selected_explicitly=selection["selected_explicitly"],
        )
        records.append(
            {
                **spec.to_dict(),
                "selection_sources": sorted(selection["sources"]),
                "selected_explicitly": selection["selected_explicitly"],
            }
        )
    return records


def validate_dataflow_decision_artifact_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed when required decision provenance is absent or stale."""

    if int(payload.get("schema_version", -1)) != DATAFLOW_DECISION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Dataflow decision artifact payload has an unsupported schema version")
    required_sections = (
        "artifact",
        "target",
        "governance",
        "scheduler",
        "handlers",
        "operation_contracts",
        "lowerings",
        "memory",
    )
    for section in required_sections:
        if not isinstance(payload.get(section), Mapping):
            raise ValueError(f"Dataflow decision artifact is missing mapping section {section!r}")

    artifact = payload["artifact"]
    for name in (
        "artifact_fingerprint",
        "compile_config_fingerprint",
        "program_fingerprint",
        "target_fingerprint",
        "compile_mode",
        "handler_lowering",
    ):
        if not isinstance(artifact.get(name), str) or not artifact[name]:
            raise ValueError(f"Dataflow decision artifact is missing artifact.{name}")
    compile_mode = artifact["compile_mode"]
    handler_lowering = artifact["handler_lowering"]
    if compile_mode not in {
        DATAFLOW_COMPILE_MODE_EXECUTABLE,
        DATAFLOW_COMPILE_MODE_INSPECT,
        DATAFLOW_COMPILE_MODE_DEBUG,
    }:
        raise ValueError(f"Dataflow decision artifact has unsupported compile mode {compile_mode!r}")
    required_selected: set[str] = set()
    if compile_mode == DATAFLOW_COMPILE_MODE_DEBUG:
        required_selected.add(str(handler_lowering))
    elif handler_lowering != PRIMFUNC_HANDLER_LOWERING:
        raise ValueError("production and inspection Dataflow artifacts must use PrimFunc handler lowering")

    target = payload["target"]
    if target.get("fingerprint") != artifact["target_fingerprint"]:
        raise ValueError("Dataflow decision artifact target fingerprint does not match the compile artifact")
    target_snapshot = TargetCapabilitySnapshot.from_dict(dict(target))

    governance = payload["governance"]
    registry_record = governance.get("implementation_registry")
    boundary_record = governance.get("lowering_boundary")
    selected_records = governance.get("selected_implementations")
    if not isinstance(registry_record, Mapping):
        raise ValueError("Dataflow decision artifact lacks implementation registry provenance")
    if not isinstance(boundary_record, Mapping):
        raise ValueError("Dataflow decision artifact lacks lowering boundary provenance")
    if not isinstance(selected_records, list) or not selected_records:
        raise ValueError("Dataflow decision artifact lacks selected implementation provenance")
    registry = dataflow_implementation_registry()
    if (
        registry_record.get("schema_version") != DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION
        or registry_record.get("fingerprint") != registry.fingerprint
    ):
        raise ValueError("Dataflow decision artifact implementation registry is stale")
    if (
        boundary_record.get("schema_version") != DATAFLOW_LOWERING_BOUNDARY_CONTRACT.schema_version
        or boundary_record.get("fingerprint") != DATAFLOW_LOWERING_BOUNDARY_CONTRACT.fingerprint
    ):
        raise ValueError("Dataflow decision artifact lowering boundary is stale")

    selected_by_id: dict[str, Mapping[str, Any]] = {}
    for record in selected_records:
        if not isinstance(record, Mapping):
            raise TypeError("selected Dataflow implementations must be mappings")
        implementation_id = str(record.get("implementation_id", ""))
        if implementation_id in selected_by_id:
            raise ValueError(f"duplicate selected Dataflow implementation {implementation_id!r}")
        sources = record.get("selection_sources")
        if not isinstance(sources, list) or not sources or any(not isinstance(source, str) or not source for source in sources):
            raise ValueError(f"selected Dataflow implementation {implementation_id!r} lacks sources")
        selected_explicitly = record.get("selected_explicitly")
        if not isinstance(selected_explicitly, bool):
            raise TypeError(f"selected Dataflow implementation {implementation_id!r} lacks explicit state")
        spec = registry.require_selectable(
            implementation_id,
            selected_explicitly=selected_explicitly,
        )
        for name, expected in spec.to_dict().items():
            if record.get(name) != expected:
                raise ValueError(f"selected Dataflow implementation {implementation_id!r} has stale {name}")
        selected_by_id[implementation_id] = record

    scheduler = payload["scheduler"]
    for name in (
        "plan_fingerprint",
        "policy",
        "reduce_strategy",
        "cost_model_version",
    ):
        if not scheduler.get(name):
            raise ValueError(f"Dataflow decision artifact is missing scheduler.{name}")
    required_selected.add(str(scheduler["policy"]))
    scheduler_config = scheduler.get("config")
    if not isinstance(scheduler_config, Mapping):
        raise ValueError("Dataflow decision artifact scheduler config must be a mapping")
    for group_name in ("semantic", "policy", "search", "cost_options", "debug"):
        group = scheduler_config.get(group_name)
        if not isinstance(group, Mapping):
            raise ValueError(f"Dataflow decision artifact scheduler config lacks {group_name!r}")
        required_selected.update(scheduler_option_implementation_id(str(option_name)) for option_name in group)
    auto_policy = scheduler.get("auto_policy")
    if not isinstance(auto_policy, Mapping) or not auto_policy.get("selection_reason"):
        raise ValueError("Dataflow decision artifact lacks scheduler selection provenance")
    execution_plans = scheduler.get("execution_plans")
    if not isinstance(execution_plans, list):
        raise ValueError("Dataflow decision artifact execution plans must be a list")
    for plan_payload in execution_plans:
        plan = DataflowExecutionPlan.from_dict(plan_payload)
        if (
            plan.target_fingerprint != target_snapshot.fingerprint
            or plan.target_compatibility_fingerprint != target_snapshot.compatibility_fingerprint
        ):
            raise ValueError("Dataflow execution plan target provenance is stale")
        required_selected.add(DATAFLOW_EXECUTION_PLANNER_IMPLEMENTATION)

    handlers = payload["handlers"]
    if handlers.get("coverage") != "base_identity_and_typed_variant_key":
        raise ValueError("Dataflow decision artifact handler coverage is incomplete")
    handler_items = handlers.get("items")
    if not isinstance(handler_items, list) or not handler_items:
        raise ValueError("Dataflow decision artifact lacks handler identities")
    for handler in handler_items:
        if not isinstance(handler, Mapping):
            raise TypeError("Dataflow decision artifact handlers must be mappings")
        identity = handler.get("base_identity")
        variant = handler.get("variant_key")
        if identity is None or not isinstance(variant, Mapping):
            raise ValueError("Dataflow decision artifact handler lacks typed identity")
        if variant.get("base_identity") != identity:
            raise ValueError("Dataflow handler variant does not match its base identity")
        specialization = DataflowSpecializationSnapshot.from_dict(handler.get("specialization", {}))
        physical = DataflowOperatorPhysicalContract.from_dict(handler.get("physical_contract", {}))
        if specialization.entries:
            required_selected.add(DATAFLOW_SPECIALIZATION_CAPTURE_IMPLEMENTATION)
        if physical != DataflowOperatorPhysicalContract():
            required_selected.add(DATAFLOW_OPERATOR_PHYSICAL_IMPLEMENTATION)
        if physical.output_slot == DATAFLOW_OUTPUT_SLOT_NONE:
            required_selected.add(DATAFLOW_TERMINAL_SIDE_EFFECT_IMPLEMENTATION)

    tensor_arguments = payload.get("tensor_arguments")
    if not isinstance(tensor_arguments, Mapping):
        raise ValueError("Dataflow decision artifact lacks tensor argument metadata")
    tensor_specs = tensor_arguments.get("specs")
    if not isinstance(tensor_specs, list):
        raise ValueError("Dataflow tensor argument metadata lacks specs")
    for spec in tensor_specs:
        if not isinstance(spec, Mapping):
            raise TypeError("Dataflow tensor argument specs must be mappings")
        layout_payload = spec.get("layout")
        if layout_payload is not None:
            DataflowTensorArgumentLayout.from_dict(layout_payload)
            required_selected.add(DATAFLOW_TENSOR_ARGUMENT_LAYOUT_IMPLEMENTATION)

    operation_contracts = payload["operation_contracts"]
    if operation_contracts.get("schema") != dataflow_operation_contract_schema_dict():
        raise ValueError("Dataflow decision artifact operation contract schema is stale")
    operation_records = operation_contracts.get("records")
    if not isinstance(operation_records, list):
        raise ValueError("Dataflow decision artifact operation contracts lack records")
    request_ids: set[str] = set()
    pipeline_contracts: dict[
        tuple[str, str],
        tuple[DataflowPipelineRequest, DataflowPipelinePlan],
    ] = {}
    transport_contracts: dict[
        str,
        tuple[DataflowResharedTransportRequest, DataflowResharedTransportPlan],
    ] = {}
    handoff_contracts: dict[
        str,
        tuple[DataflowCrossHandlerHandoffRequest, DataflowCrossHandlerHandoffPlan],
    ] = {}
    for record in operation_records:
        if not isinstance(record, Mapping):
            raise TypeError("Dataflow operation contract records must be mappings")
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Dataflow operation contract record lacks request_id")
        if request_id in request_ids:
            raise ValueError(f"duplicate Dataflow operation contract request_id {request_id!r}")
        request_ids.add(request_id)
        owner = record.get("owner")
        if not isinstance(owner, Mapping) or owner.get("kind") not in {
            "stage",
            "intermediate",
        }:
            raise ValueError(f"Dataflow operation contract {request_id!r} lacks structural owner")
        request_payload = record.get("request")
        if not isinstance(request_payload, Mapping):
            raise ValueError(f"Dataflow operation contract {request_id!r} lacks typed request")
        if "diagnostic_name" in request_payload:
            raise ValueError(f"Dataflow operation contract {request_id!r} leaked a diagnostic name")
        request = dataflow_operation_request_from_dict(request_payload)
        selection_state = record.get("selection_state")
        decisions = record.get("decisions")
        lowering_plan_payloads = record.get("lowering_plans")
        selected_explicitly = record.get("selected_explicitly")
        if not isinstance(selected_explicitly, bool):
            raise TypeError(f"Dataflow operation contract {request_id!r} lacks explicit state")
        if not isinstance(decisions, list):
            raise ValueError(f"Dataflow operation contract {request_id!r} lacks decisions")
        if not isinstance(lowering_plan_payloads, list):
            raise ValueError(f"Dataflow operation contract {request_id!r} lacks lowering plans")
        range_plans: tuple[DataflowRangeCoarseningPlan, ...] = ()
        pipeline_plans: tuple[DataflowPipelinePlan, ...] = ()
        transport_plans: tuple[DataflowResharedTransportPlan, ...] = ()
        handoff_plans: tuple[DataflowCrossHandlerHandoffPlan, ...] = ()
        if isinstance(request, DataflowRangeCoarseningRequest):
            range_plans = tuple(DataflowRangeCoarseningPlan.from_dict(item) for item in lowering_plan_payloads)
            if selection_state == "selected" and not range_plans:
                raise ValueError(f"selected Dataflow range contract {request_id!r} lacks a plan")
            if any(range_plan.request_fingerprint != request.fingerprint for range_plan in range_plans):
                raise ValueError(f"Dataflow range contract {request_id!r} plan changed its request")
            plan_fingerprints = tuple(range_plan.fingerprint for range_plan in range_plans)
            if len(plan_fingerprints) != len(set(plan_fingerprints)):
                raise ValueError(f"Dataflow range contract {request_id!r} has duplicate plans")
        elif isinstance(request, DataflowPipelineRequest):
            pipeline_plans = tuple(DataflowPipelinePlan.from_dict(item) for item in lowering_plan_payloads)
            if selection_state == "selected" and len(pipeline_plans) != 1:
                raise ValueError(f"selected Dataflow pipeline contract {request_id!r} requires exactly one plan")
            if any(plan.request_fingerprint != request.fingerprint for plan in pipeline_plans):
                raise ValueError(f"Dataflow pipeline contract {request_id!r} plan changed its request")
            for plan in pipeline_plans:
                pipeline_contracts[(request.fingerprint, plan.fingerprint)] = (
                    request,
                    plan,
                )
        elif isinstance(request, DataflowResharedTransportRequest):
            transport_plans = tuple(DataflowResharedTransportPlan.from_dict(item) for item in lowering_plan_payloads)
            if selection_state == "selected" and len(transport_plans) != 1:
                raise ValueError(f"selected Dataflow transport contract {request_id!r} requires exactly one plan")
            if any(plan.request_fingerprint != request.fingerprint for plan in transport_plans):
                raise ValueError(f"Dataflow transport contract {request_id!r} plan changed its request")
            for plan in transport_plans:
                if plan.fingerprint in transport_contracts:
                    raise ValueError(f"duplicate Dataflow transport plan {plan.fingerprint!r}")
                transport_contracts[plan.fingerprint] = (request, plan)
        elif isinstance(request, DataflowCrossHandlerHandoffRequest):
            handoff_plans = tuple(DataflowCrossHandlerHandoffPlan.from_dict(item) for item in lowering_plan_payloads)
            if selection_state == "selected" and len(handoff_plans) != 1:
                raise ValueError(f"selected Dataflow handoff contract {request_id!r} requires exactly one plan")
            if any(plan.request_fingerprint != request.fingerprint for plan in handoff_plans):
                raise ValueError(f"Dataflow handoff contract {request_id!r} plan changed its request")
            for plan in handoff_plans:
                if plan.fingerprint in handoff_contracts:
                    raise ValueError(f"duplicate Dataflow handoff plan {plan.fingerprint!r}")
                handoff_contracts[plan.fingerprint] = (request, plan)
        elif lowering_plan_payloads:
            raise ValueError(f"Dataflow operation contract {request_id!r} has unexpected lowering plans")
        if selection_state not in {"declared", "selected"} or (selection_state == "selected") != bool(decisions):
            raise ValueError(f"Dataflow operation contract {request_id!r} has inconsistent selection state")
        parsed_decisions: list[DataflowOperationDecision] = []
        for decision_payload in decisions:
            decision = DataflowOperationDecision.from_dict(decision_payload)
            parsed_decisions.append(decision)
            if decision.request.fingerprint != request.fingerprint:
                raise ValueError(f"Dataflow operation decision {request_id!r} changed its request")
            implementation_id = decision.selected_implementation
            if implementation_id is None:
                continue
            registry.require_contract_compatible(
                implementation_id,
                request,
                selected_explicitly=selected_explicitly,
            )
            required_selected.add(implementation_id)
        if isinstance(request, DataflowRangeCoarseningRequest):
            if (selection_state == "selected") != bool(range_plans):
                raise ValueError(f"Dataflow range contract {request_id!r} has inconsistent plan state")
            if selection_state == "selected":
                if len(parsed_decisions) != 1:
                    raise ValueError(f"selected Dataflow range contract {request_id!r} must have one decision")
                decision = parsed_decisions[0]
                expected_fallback = any(plan.used_fallback for plan in range_plans)
                expected_reason = "range_resource_fallback" if expected_fallback else "typed_range_coarsening_plan"
                expected_slot_bytes = max(plan.selected_resource_bytes for plan in range_plans)
                if (
                    decision.selected_implementation != DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION
                    or decision.used_fallback != expected_fallback
                    or decision.selection_reason != expected_reason
                    or decision.resources is None
                    or decision.resources.slot_bytes != expected_slot_bytes
                ):
                    raise ValueError(f"Dataflow range contract {request_id!r} decision changed its plan")
                scheduler_budget = payload["scheduler"].get("range_resource_budget_bytes")
                if any(plan.resource_budget_bytes != scheduler_budget for plan in range_plans):
                    raise ValueError(f"Dataflow range contract {request_id!r} changed its resource budget")
        elif isinstance(request, DataflowPipelineRequest):
            if (selection_state == "selected") != bool(pipeline_plans):
                raise ValueError(f"Dataflow pipeline contract {request_id!r} has inconsistent plan state")
            if selection_state == "selected":
                if len(parsed_decisions) != 1:
                    raise ValueError(f"selected Dataflow pipeline contract {request_id!r} must have one decision")
                plan = pipeline_plans[0]
                expected_plan = plan_pipeline_dataflow(request)
                if plan != expected_plan:
                    raise ValueError(f"Dataflow pipeline contract {request_id!r} changed its typed plan")
                decision = parsed_decisions[0]
                expected_reason = "pipeline_constraint_fallback" if plan.used_fallback else "typed_pipeline_dataflow_plan"
                if (
                    decision.selected_implementation != DATAFLOW_PIPELINE_PLAN_IMPLEMENTATION
                    or decision.used_fallback != plan.used_fallback
                    or decision.selection_reason != expected_reason
                    or decision.resources != plan.resources
                ):
                    raise ValueError(f"Dataflow pipeline contract {request_id!r} decision changed its plan")
        elif isinstance(request, DataflowResharedTransportRequest):
            if (selection_state == "selected") != bool(transport_plans):
                raise ValueError(f"Dataflow transport contract {request_id!r} has inconsistent plan state")
            if selection_state == "selected":
                if len(parsed_decisions) != 1:
                    raise ValueError(f"selected Dataflow transport contract {request_id!r} must have one decision")
                plan = transport_plans[0]
                expected_plan = plan_reshared_transport(
                    request,
                    cluster_size=plan.cluster_size,
                    physical_slot_bytes=plan.physical_slot_bytes,
                    target_capabilities=target_snapshot,
                    receive_stage_count=plan.receive_stage_count or None,
                    available_threads=(plan.transfer_threads or None),
                )
                if plan != expected_plan or parsed_decisions[0] != plan.decision:
                    raise ValueError(f"Dataflow transport contract {request_id!r} changed its typed plan")
        elif isinstance(request, DataflowCrossHandlerHandoffRequest):
            if (selection_state == "selected") != bool(handoff_plans):
                raise ValueError(f"Dataflow handoff contract {request_id!r} has inconsistent plan state")
            if selection_state == "selected":
                if len(parsed_decisions) != 1:
                    raise ValueError(f"selected Dataflow handoff contract {request_id!r} must have one decision")
                plan = handoff_plans[0]
                matching_pipeline_requests = tuple(
                    pipeline_request
                    for (_, pipeline_fingerprint), (
                        pipeline_request,
                        _,
                    ) in pipeline_contracts.items()
                    if pipeline_fingerprint == plan.pipeline_plan_fingerprint
                )
                if len(matching_pipeline_requests) > 1:
                    raise ValueError(f"Dataflow handoff contract {request_id!r} has ambiguous consumer pipeline provenance")
                expected_plan = plan_cross_handler_handoff(
                    request,
                    producer_stage_id=plan.producer_stage_id,
                    consumer_pipeline_request=(None if not matching_pipeline_requests else matching_pipeline_requests[0]),
                    task_coord_rank=plan.task_coord_rank,
                    max_shared_memory_bytes=plan.resource_budget_bytes,
                    arena_alignment=plan.arena_alignment,
                )
                if plan.fallback_reason == "handoff_queue_has_no_active_edges":
                    expected_plan = fallback_cross_handler_handoff_plan(
                        expected_plan,
                        reason=plan.fallback_reason,
                    )
                if plan != expected_plan or parsed_decisions[0] != plan.decision:
                    raise ValueError(f"Dataflow handoff contract {request_id!r} changed its typed plan")

    lowerings = payload["lowerings"]
    for name in (
        "reduction",
        "gemm",
        "copy",
        "pipeline",
        "transport",
        "handoff",
        "precision",
    ):
        if not isinstance(lowerings.get(name), Mapping):
            raise ValueError(f"Dataflow decision artifact lacks lowering section {name!r}")
    for resolution in lowerings["gemm"].get("resolutions", ()):
        require_decision_fields(
            resolution,
            "GEMM",
            (
                "request",
                "implementation_id",
                "used_fallback",
                "selection_reason",
                "rejected_candidates",
            ),
        )
        required_selected.add(str(resolution["implementation_id"]))
    for request in lowerings["copy"].get("requests", ()):
        require_decision_fields(
            request,
            "transfer",
            (
                "selected_implementation",
                "selection_supported",
                "selection_reason",
                "rejected_candidates",
                "handoff_producer",
            ),
        )
        required_selected.add(str(request["selected_implementation"]))
    for resolution in lowerings["pipeline"].get("resolutions", ()):
        require_decision_fields(
            resolution,
            "pipeline",
            (
                "requested_stages",
                "selected_implementation",
                "fallback",
                "selection_reason",
            ),
        )
        required_selected.add(str(resolution["selected_implementation"]))
    validate_pipeline_dataflow_bindings(
        lowerings["pipeline"].get("dataflow_bindings"),
        pipeline_contracts=pipeline_contracts,
        transfer_requests=lowerings["copy"].get("requests", ()),
        gemm_resolutions=lowerings["gemm"].get("resolutions", ()),
        pipeline_resolutions=lowerings["pipeline"].get("resolutions", ()),
        target_fingerprint=artifact["target_fingerprint"],
        handler_ids={int(item["handler_id"]) for item in handler_items},
    )
    transport_lowerings = lowerings["transport"]
    validate_reshared_transport_bindings(
        transport_lowerings,
        transport_contracts=transport_contracts,
        handlers={int(item["handler_id"]): item for item in handler_items},
    )
    for binding in transport_lowerings.get("bindings", ()):
        required_selected.add(str(binding["lowering_implementation_id"]))
    handoff_lowerings = lowerings["handoff"]
    validate_cross_handler_handoff_bindings(
        handoff_lowerings,
        handoff_contracts=handoff_contracts,
        transfer_requests=lowerings["copy"].get("requests", ()),
        handler_ids={int(item["handler_id"]) for item in handler_items},
    )
    for binding in handoff_lowerings.get("bindings", ()):
        required_selected.add(str(binding["lowering_implementation_id"]))
    precision_plan = lowerings["precision"].get("plan")
    if not isinstance(precision_plan, Mapping):
        raise ValueError("Dataflow decision artifact lacks precision plan")
    semantic_config = lowerings["precision"].get("semantic_config")
    if not isinstance(semantic_config, Mapping):
        raise ValueError("Dataflow decision artifact lacks semantic config")
    if semantic_config.get("direct_slot_seed_reduce") is True:
        required_selected.add("semantic.direct_slot_seed_reduce")
    if semantic_config.get("skip_finalize_post_sync") is True:
        required_selected.add("semantic.skip_finalize_post_sync")
    if semantic_config.get("fast_math") is True:
        required_selected.add("semantic.fast_math")
    for resolution in precision_plan.get("resolutions", ()):
        require_decision_fields(
            resolution,
            "precision",
            (
                "selected_dtype",
                "selection_reason",
                "reduction_order",
                "implementation_id",
            ),
        )
        required_selected.add(str(resolution["implementation_id"]))

    memory = payload["memory"]
    memory_plan = memory.get("planner")
    if not isinstance(memory_plan, Mapping):
        raise ValueError("Dataflow decision artifact lacks memory planner provenance")
    selected_candidate_id = memory_plan.get("selected_candidate_id")
    candidates = memory_plan.get("candidates")
    if not memory_plan.get("selection_reason") or not isinstance(candidates, list):
        raise ValueError("Dataflow decision artifact memory selection is incomplete")
    selected_candidates = [
        candidate for candidate in candidates if isinstance(candidate, Mapping) and candidate.get("candidate_id") == selected_candidate_id
    ]
    if len(selected_candidates) != 1 or selected_candidates[0].get("legal") is not True:
        raise ValueError("Dataflow decision artifact memory candidate is not uniquely legal")
    required_selected.add(f"{memory_plan['planner_version']}:{selected_candidates[0]['placement']}")
    validation = memory.get("validation")
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise ValueError("Dataflow decision artifact memory validation did not pass")

    missing = sorted(required_selected.difference(selected_by_id))
    if missing:
        raise ValueError(f"Dataflow decision artifact lacks lifecycle records for {missing!r}")


def validate_reshared_transport_bindings(
    section: Any,
    *,
    transport_contracts: Mapping[
        str,
        tuple[DataflowResharedTransportRequest, DataflowResharedTransportPlan],
    ],
    handlers: Mapping[int, Mapping[str, Any]],
) -> None:
    if not isinstance(section, Mapping):
        raise ValueError("Dataflow decision artifact lacks transport lowerings")
    expected_metadata = {
        "coverage": "typed_family_plan_to_scheduler_and_primfunc_physical_binding",
        "plan_schema_version": DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_VERSION,
        "planner_version": DATAFLOW_RESHARED_TRANSPORT_PLANNER_VERSION,
        "binding_schema_version": (DATAFLOW_RESHARED_TRANSPORT_BINDING_SCHEMA_VERSION),
    }
    for name, expected in expected_metadata.items():
        if section.get(name) != expected:
            raise ValueError(f"Dataflow transport lowering has stale {name!r} metadata")
    bindings = section.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("Dataflow transport lowering bindings must be a list")

    bound_plans: dict[str, set[int | None]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise TypeError("Dataflow transport bindings must be mappings")
        if binding.get("schema_version") != DATAFLOW_RESHARED_TRANSPORT_BINDING_SCHEMA_VERSION:
            raise ValueError("Dataflow transport binding schema is stale")
        plan_fingerprint = str(binding.get("plan_fingerprint", ""))
        try:
            request, plan = transport_contracts[plan_fingerprint]
        except KeyError as err:
            raise ValueError("Dataflow transport binding references an unknown typed plan") from err
        handler_id_value = binding.get("handler_id")
        handler_id = None if handler_id_value is None else int(handler_id_value)
        plan_handlers = bound_plans.setdefault(plan_fingerprint, set())
        if handler_id in plan_handlers:
            raise ValueError("duplicate Dataflow transport handler binding")
        plan_handlers.add(handler_id)

        locality_counts = {locality: sum(step.locality == locality for step in plan.steps) for locality in ("local", "remote", "global")}
        expected = {
            "request_fingerprint": request.fingerprint,
            "family": plan.family,
            "implementation_id": plan.implementation_id,
            "lowering_implementation_id": plan.lowering_implementation_id,
            "cluster_size": plan.cluster_size,
            "consumer_access_order": plan.consumer_access_order,
            "logical_tile_count": plan.logical_tile_count,
            "physical_slot_count": plan.physical_slot_count,
            "logical_tile_bytes": plan.logical_tile_bytes,
            "transport_bytes_per_consumer": (plan.logical_tile_count * plan.logical_tile_bytes),
            "transaction_bytes": plan.transaction_bytes,
            "payload_partitions": [item.to_dict() for item in plan.payload_partitions],
            "transfer_threads": plan.transfer_threads,
            "receive_stage_count": plan.receive_stage_count,
            "credit_count": plan.credit_count,
            "credit_lifetimes": [item.to_dict() for item in plan.credit_lifetimes],
            "remote_barrier_count": plan.remote_barrier_count,
            "publish_sync_required": plan.publish_sync_required,
            "release_sync_required": plan.release_sync_required,
            "locality_counts": locality_counts,
        }
        for name, expected_value in expected.items():
            if binding.get(name) != expected_value:
                raise ValueError(f"Dataflow transport binding changed plan field {name!r}")

        if handler_id is None:
            if binding.get("handler_thread_count") is not None:
                raise ValueError("scheduler-only transport binding has handler state")
            continue
        try:
            handler = handlers[handler_id]
        except KeyError as err:
            raise ValueError(f"Dataflow transport binding references unknown handler {handler_id}") from err
        handler_threads = int(handler["thread_count"])
        if binding.get("handler_thread_count") != handler_threads:
            raise ValueError("Dataflow transport handler thread count changed")
        if plan.transfer_threads > handler_threads:
            raise ValueError("Dataflow transport exceeds consumer handler threads")
        if plan.family == DATAFLOW_TRANSPORT_STREAMED:
            if (
                binding.get("primfunc_plan_fingerprint") != plan.fingerprint
                or binding.get("primfunc_plan_schema_version") != plan.schema_version
            ):
                raise ValueError("streamed Dataflow transport lacks its PrimFunc physical binding")
        elif binding.get("primfunc_plan_fingerprint") is not None or binding.get("primfunc_plan_schema_version") is not None:
            raise ValueError("non-streamed Dataflow transport cannot carry a PrimFunc DSM binding")

    missing = set(transport_contracts).difference(bound_plans)
    if missing:
        raise ValueError(f"Dataflow transport plans lack physical bindings {sorted(missing)!r}")


def validate_cross_handler_handoff_bindings(
    section: Any,
    *,
    handoff_contracts: Mapping[
        str,
        tuple[DataflowCrossHandlerHandoffRequest, DataflowCrossHandlerHandoffPlan],
    ],
    transfer_requests: Any,
    handler_ids: set[int],
) -> None:
    if not isinstance(section, Mapping):
        raise ValueError("Dataflow decision artifact lacks handoff lowerings")
    expected_metadata = {
        "coverage": ("typed_graph_pipeline_plan_to_queue_runtime_and_primfunc_binding"),
        "plan_schema_version": DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_VERSION,
        "planner_version": DATAFLOW_CROSS_HANDLER_HANDOFF_PLANNER_VERSION,
        "binding_schema_version": (DATAFLOW_CROSS_HANDLER_HANDOFF_BINDING_SCHEMA_VERSION),
    }
    for name, expected in expected_metadata.items():
        if section.get(name) != expected:
            raise ValueError(f"Dataflow handoff lowering has stale {name!r} metadata")
    bindings = section.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("Dataflow handoff lowering bindings must be a list")

    physical_transfers: dict[tuple[int, int], Mapping[str, Any]] = {}
    for request in transfer_requests:
        handoff = request.get("handoff_producer")
        if handoff is None:
            continue
        if not isinstance(handoff, Mapping) or set(handoff) != {
            "plan_fingerprint",
            "transfer_index",
            "stage_index",
        }:
            raise ValueError("Dataflow handoff producer copy has invalid metadata")
        key = (int(request["handler_id"]), int(request["operation_index"]))
        if key in physical_transfers:
            raise ValueError("duplicate Dataflow handoff producer copy operation")
        physical_transfers[key] = request

    observed_plans: set[str] = set()
    bound_physical_transfers: set[tuple[int, int]] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise TypeError("Dataflow handoff bindings must be mappings")
        if binding.get("schema_version") != DATAFLOW_CROSS_HANDLER_HANDOFF_BINDING_SCHEMA_VERSION:
            raise ValueError("Dataflow handoff binding schema is stale")
        recorded_fingerprint = binding.get("fingerprint")
        canonical_binding = dict(binding)
        canonical_binding.pop("fingerprint", None)
        if recorded_fingerprint != sha256(serialize_canonical_json(canonical_binding)):
            raise ValueError("Dataflow handoff binding fingerprint mismatch")
        plan_fingerprint = str(binding.get("plan_fingerprint", ""))
        try:
            request, plan = handoff_contracts[plan_fingerprint]
        except KeyError as err:
            raise ValueError("Dataflow handoff binding references an unknown typed plan") from err
        if plan_fingerprint in observed_plans:
            raise ValueError("duplicate Dataflow handoff physical binding")
        observed_plans.add(plan_fingerprint)
        expected = {
            "request_fingerprint": request.fingerprint,
            "producer_stage_id": plan.producer_stage_id,
            "consumer_stage_id": plan.consumer_stage_id,
            "lookahead_distance": plan.lookahead_distance,
            "selected_buffer_stages": plan.selected_buffer_stages,
            "tail_policy": plan.tail_policy,
            "enabled": plan.enabled,
            "fallback_reason": plan.fallback_reason,
            "implementation_id": plan.implementation_id,
            "lowering_implementation_id": plan.lowering_implementation_id,
            "arena_alignment": plan.arena_alignment,
            "resource_budget_bytes": plan.resource_budget_bytes,
            "required_arena_bytes": plan.required_arena_bytes,
            "arena_bytes": plan.arena_bytes,
            "barrier_count": plan.barrier_count,
            "wait_required": plan.wait_required,
            "release_required": plan.release_required,
            "transfer_plans": [item.to_dict() for item in plan.transfer_plans],
            "buffer_lifetimes": [item.to_dict() for item in plan.buffer_lifetimes],
        }
        for name, expected_value in expected.items():
            if binding.get(name) != expected_value:
                raise ValueError(f"Dataflow handoff binding changed plan field {name!r}")

        producer_handler_ids = binding.get("producer_handler_ids")
        consumer_handler_ids = binding.get("consumer_handler_ids")
        if (
            not isinstance(producer_handler_ids, list)
            or producer_handler_ids != sorted(set(producer_handler_ids))
            or not isinstance(consumer_handler_ids, list)
            or consumer_handler_ids != sorted(set(consumer_handler_ids))
            or not set(producer_handler_ids).issubset(handler_ids)
            or not set(consumer_handler_ids).issubset(handler_ids)
            or set(producer_handler_ids).intersection(consumer_handler_ids)
        ):
            raise ValueError("Dataflow handoff has invalid PrimFunc role bindings")
        producer_transfers = binding.get("producer_transfers")
        if not isinstance(producer_transfers, list):
            raise ValueError("Dataflow handoff producer transfers must be a list")
        actual_physical_transfers: set[tuple[int, int, int]] = set()
        for producer_transfer in producer_transfers:
            require_decision_fields(
                producer_transfer,
                "handoff producer transfer",
                (
                    "handler_id",
                    "operation_index",
                    "transfer_index",
                    "stage_index",
                    "contract_fingerprint",
                    "selected_implementation",
                    "asynchronous",
                    "uses_tma_descriptor",
                ),
            )
            operation_key = (
                int(producer_transfer["handler_id"]),
                int(producer_transfer["operation_index"]),
            )
            physical = physical_transfers.get(operation_key)
            if physical is None or operation_key in bound_physical_transfers:
                raise ValueError("Dataflow handoff producer transfer lacks a unique physical copy")
            handoff = physical["handoff_producer"]
            transfer_index = int(producer_transfer["transfer_index"])
            stage_index = int(producer_transfer["stage_index"])
            if (
                handoff["plan_fingerprint"] != plan.fingerprint
                or int(handoff["transfer_index"]) != transfer_index
                or int(handoff["stage_index"]) != stage_index
                or producer_transfer["contract_fingerprint"] != physical["contract_fingerprint"]
                or producer_transfer["selected_implementation"] != physical["selected_implementation"]
                or producer_transfer["asynchronous"] != physical["asynchronous"]
                or producer_transfer["uses_tma_descriptor"] != physical["uses_tma_descriptor"]
            ):
                raise ValueError("Dataflow handoff producer transfer changed its physical copy")
            bound_physical_transfers.add(operation_key)
            actual_physical_transfers.add((operation_key[0], transfer_index, stage_index))
        expected_physical_transfers = {
            (handler_id, transfer.binding_index, stage_index)
            for handler_id in producer_handler_ids
            for transfer in plan.transfer_plans
            for stage_index in range(transfer.buffer_stages)
        }
        if plan.enabled:
            if not producer_handler_ids or not consumer_handler_ids:
                raise ValueError("enabled Dataflow handoff lacks producer or consumer PrimFunc roles")
            if len(actual_physical_transfers) != len(producer_transfers) or actual_physical_transfers != expected_physical_transfers:
                raise ValueError("Dataflow handoff producer transfer coverage changed")
        elif producer_transfers:
            raise ValueError("disabled Dataflow handoff contains producer transfers")

        raw_queue_bindings = binding.get("queue_bindings")
        if not isinstance(raw_queue_bindings, list):
            raise ValueError("Dataflow handoff queue bindings must be a list")
        queue_bindings = tuple(DataflowHandoffQueueBinding.from_dict(item) for item in raw_queue_bindings)
        producer_ids = tuple(item.producer_instruction_id for item in queue_bindings)
        if len(producer_ids) != len(set(producer_ids)):
            raise ValueError("Dataflow handoff queue repeats a producer instruction")
        consumer_ids = tuple(item.consumer_instruction_id for item in queue_bindings if item.consumer_instruction_id is not None)
        if len(consumer_ids) != len(set(consumer_ids)):
            raise ValueError("Dataflow handoff queue repeats a consumer instruction")
        if any(item.plan_fingerprint != plan.fingerprint for item in queue_bindings):
            raise ValueError("Dataflow handoff queue binding changed its typed plan")
        if plan.enabled:
            if any(item.state == "disabled" for item in queue_bindings):
                raise ValueError("enabled Dataflow handoff contains disabled queue edges")
            if not any(item.state == "active" for item in queue_bindings):
                raise ValueError("enabled Dataflow handoff lacks an active queue edge")
            if any(
                item.state == "active"
                and (
                    item.stage_count != plan.selected_buffer_stages or item.arena_slot is None or item.arena_slot >= plan.lookahead_distance
                )
                for item in queue_bindings
            ):
                raise ValueError("Dataflow handoff queue changed stage or arena binding")
        elif any(item.state != "disabled" for item in queue_bindings):
            raise ValueError("disabled Dataflow handoff contains active queue edges")

    missing = set(handoff_contracts).difference(observed_plans)
    if missing:
        raise ValueError(f"Dataflow handoff plans lack physical bindings {sorted(missing)!r}")
    if bound_physical_transfers != set(physical_transfers):
        raise ValueError("Dataflow handoff producer copies lack typed plan bindings")


def validate_pipeline_dataflow_bindings(
    bindings: Any,
    *,
    pipeline_contracts: Mapping[
        tuple[str, str],
        tuple[DataflowPipelineRequest, DataflowPipelinePlan],
    ],
    transfer_requests: Any,
    gemm_resolutions: Any,
    pipeline_resolutions: Any,
    target_fingerprint: str,
    handler_ids: set[int],
) -> None:
    if not isinstance(bindings, list):
        raise ValueError("Dataflow decision artifact lacks pipeline dataflow bindings")
    transfer_by_operation = {(int(item["handler_id"]), int(item["operation_index"])): item for item in transfer_requests}
    gemm_by_operation = {(int(item["handler_id"]), int(item["operation_index"])): item for item in gemm_resolutions}
    pipeline_by_operation = {(int(item["handler_id"]), int(item["loop_index"])): item for item in pipeline_resolutions}
    seen_handlers: set[int] = set()
    for binding in bindings:
        require_decision_fields(
            binding,
            "pipeline dataflow binding",
            (
                "schema_version",
                "handler_id",
                "request_fingerprint",
                "plan_fingerprint",
                "target_fingerprint",
                "selected_stages",
                "selected_max_outstanding",
                "buffer_versions",
                "additive_gemm_groups",
                "release_after_gemm_indices",
                "transfers",
                "gemms",
                "pipeline",
                "used_fallback",
                "fallback_reasons",
                "fingerprint",
            ),
        )
        if binding["schema_version"] != DATAFLOW_PIPELINE_BINDING_SCHEMA_VERSION:
            raise ValueError("unsupported pipeline dataflow binding schema")
        canonical = dict(binding)
        fingerprint = canonical.pop("fingerprint")
        if fingerprint != sha256(serialize_canonical_json(canonical)):
            raise ValueError("pipeline dataflow binding fingerprint changed")
        handler_id = int(binding["handler_id"])
        if handler_id not in handler_ids or handler_id in seen_handlers:
            raise ValueError("pipeline dataflow binding has an unknown or duplicate handler")
        seen_handlers.add(handler_id)
        if binding["target_fingerprint"] != target_fingerprint:
            raise ValueError("pipeline dataflow binding changed its target")
        try:
            request, plan = pipeline_contracts[
                (
                    str(binding["request_fingerprint"]),
                    str(binding["plan_fingerprint"]),
                )
            ]
        except KeyError as err:
            raise ValueError("pipeline dataflow binding does not reference a typed plan") from err
        requirements = plan.implementation_requirements
        if (
            int(binding["selected_stages"]) != plan.selected_stages
            or int(binding["selected_max_outstanding"]) != plan.selected_max_outstanding
            or binding["buffer_versions"] != list(requirements.buffer_versions)
            or binding["additive_gemm_groups"] != [list(group) for group in requirements.additive_gemm_groups]
            or binding["release_after_gemm_indices"] != list(requirements.release_after_gemm_indices)
        ):
            raise ValueError("pipeline dataflow binding changed its typed plan")

        transfers = binding["transfers"]
        gemms = binding["gemms"]
        pipeline = binding["pipeline"]
        if (
            not isinstance(transfers, list)
            or len(transfers) != requirements.transfer_count
            or not isinstance(gemms, list)
            or len(gemms) != requirements.gemm_count
            or not isinstance(pipeline, Mapping)
        ):
            raise ValueError("pipeline dataflow binding cardinality changed")
        bound_transfer_operations: set[tuple[int, int]] = set()
        for graph_index, transfer in enumerate(transfers):
            if int(transfer.get("graph_index", -1)) != graph_index:
                raise ValueError("pipeline transfer binding indices are not canonical")
            if transfer.get("producer_partition") != (requirements.producer_partitions[graph_index]):
                raise ValueError("pipeline transfer binding changed its producer partition")
            if transfer.get("materialization") != (requirements.transfer_materializations[graph_index]):
                raise ValueError("pipeline transfer binding changed its materialization")
            alternatives = transfer.get("alternatives")
            if not isinstance(alternatives, list) or not alternatives:
                raise ValueError("pipeline transfer binding lacks physical alternatives")
            operation_indices = [int(alternative.get("operation_index", -1)) for alternative in alternatives]
            if operation_indices != sorted(set(operation_indices)):
                raise ValueError("pipeline transfer alternatives are not canonical")
            for alternative in alternatives:
                key = (
                    handler_id,
                    int(alternative.get("operation_index", -1)),
                )
                physical = transfer_by_operation.get(key)
                if (
                    key in bound_transfer_operations
                    or physical is None
                    or (
                        alternative.get("selected_implementation") != physical["selected_implementation"]
                        or alternative.get("asynchronous") != physical["asynchronous"]
                        or alternative.get("cluster_mask") != physical["cluster_mask"]
                        or transfer.get("materialization") != physical["materialization"]
                    )
                ):
                    raise ValueError("pipeline transfer binding changed physical selection")
                bound_transfer_operations.add(key)
                cluster_mask = alternative.get("cluster_mask")
                if (
                    isinstance(cluster_mask, bool)
                    or not isinstance(cluster_mask, int)
                    or cluster_mask < 0
                    or (cluster_mask and not request.transfers[graph_index].multicast_permitted)
                ):
                    raise ValueError("pipeline transfer binding has illegal multicast")
                if transfer["materialization"] == DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT and (
                    alternative.get("selected_implementation") != DATAFLOW_PIPELINE_RESIDENT_IMPLEMENTATION
                    or alternative.get("asynchronous") is not False
                    or cluster_mask != 0
                ):
                    raise ValueError("resident pipeline transfer binding changed physical ownership")
        expected_transfer_operations = {
            (handler_id, int(item["operation_index"]))
            for item in transfer_requests
            if int(item["handler_id"]) == handler_id
            and item.get("synchronization_owner") == "pipeline"
            and item.get("handoff_producer") is None
        }
        if bound_transfer_operations != expected_transfer_operations:
            raise ValueError("pipeline transfer binding coverage changed")
        for graph_index, gemm in enumerate(gemms):
            if int(gemm.get("graph_index", -1)) != graph_index:
                raise ValueError("pipeline GEMM binding indices are not canonical")
            key = (handler_id, int(gemm.get("operation_index", -1)))
            physical = gemm_by_operation.get(key)
            if physical is None or (
                gemm.get("selected_implementation") != physical["implementation_id"]
                or gemm.get("physical_shape") != physical["physical_shape"]
            ):
                raise ValueError("pipeline GEMM binding changed physical selection")
        pipeline_key = (handler_id, int(pipeline.get("loop_index", -1)))
        physical_pipeline = pipeline_by_operation.get(pipeline_key)
        if physical_pipeline is None or (
            pipeline.get("selected_implementation") != physical_pipeline["selected_implementation"]
            or pipeline.get("fallback") != physical_pipeline["fallback"]
            or int(physical_pipeline["requested_stages"]) != plan.selected_stages
        ):
            raise ValueError("pipeline binding changed physical pipeline selection")

        fallback_reasons = list(plan.fallback_reasons)
        if bool(physical_pipeline["fallback"]):
            fallback_reasons.append("physical_pipeline_fallback")
        fallback_reasons.extend(
            f"gemm_{index}_fallback"
            for index, gemm in enumerate(gemms)
            if gemm_by_operation[(handler_id, int(gemm["operation_index"]))]["used_fallback"]
        )
        async_indices = set(requirements.async_transfer_indices)
        fallback_reasons.extend(
            f"transfer_{index}_synchronous_fallback"
            for index, transfer in enumerate(transfers)
            if index in async_indices and any(not alternative["asynchronous"] for alternative in transfer["alternatives"])
        )
        if binding["fallback_reasons"] != fallback_reasons or binding["used_fallback"] != bool(fallback_reasons):
            raise ValueError("pipeline dataflow fallback provenance changed")


def require_decision_fields(
    decision: Any,
    kind: str,
    fields: tuple[str, ...],
) -> None:
    if not isinstance(decision, Mapping):
        raise TypeError(f"Dataflow {kind} decisions must be mappings")
    missing = tuple(name for name in fields if name not in decision)
    if missing:
        raise ValueError(f"Dataflow {kind} decision lacks required provenance {missing!r}")


def collect_transfer_requests(
    lowering: Any,
    *,
    target: tvm.target.Target,
) -> list[dict[str, Any]]:
    if lowering is None:
        return []

    requests: list[dict[str, Any]] = []
    for handler in lowering.handlers:
        copy_operation_index = 0
        destination_buffers: list[Any] = []
        synchronous_pipeline = (
            handler.pipeline_dataflow_plan is not None
            and handler.pipeline_dataflow_plan.implementation_requirements.mode == DATAFLOW_PIPELINE_MODE_SYNCHRONOUS
        ) or any(
            str(resolution.get("selected_implementation")) == DATAFLOW_PIPELINE_MODE_SYNCHRONOUS
            for resolution in handler.pipeline_lowerings
        )

        def visit(
            node: Any,
            handler: Any = handler,
            synchronous_pipeline: bool = synchronous_pipeline,
            destination_buffers: list[Any] = destination_buffers,
        ) -> None:
            nonlocal copy_operation_index
            if not (isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tileop.copy"):
                return
            operation_index = copy_operation_index
            copy_operation_index += 1
            parsed = _ffi_api.ParseOperator(node)
            contract = parsed.transfer_contract
            if contract is None:
                return
            try:
                synchronization_owner = _TRANSFER_SYNC_OWNER_NAMES[int(contract.sync_owner)]
            except KeyError as err:
                raise ValueError(f"unsupported transfer synchronization owner {contract.sync_owner!r}") from err
            resolution_call = node
            if synchronization_owner == "pipeline":
                bound_mode = node.annotations.get("tl.transfer_pipeline_sync_consumed")
                if bound_mode is None:
                    annotations = dict(node.annotations)
                    annotations["tl.transfer_pipeline_sync_consumed"] = tir.IntImm(
                        "int32",
                        (_TRANSFER_PIPELINE_SYNC_FALLBACK if synchronous_pipeline else _TRANSFER_PIPELINE_SYNC_MANAGED),
                    )
                    resolution_call = tir.Call(
                        node.dtype,
                        node.op,
                        list(node.args),
                        annotations=annotations,
                        span=node.span,
                    )
                elif int(bound_mode) not in (
                    _TRANSFER_PIPELINE_SYNC_MANAGED,
                    _TRANSFER_PIPELINE_SYNC_FALLBACK,
                ):
                    raise ValueError("pipeline transfer has invalid bound synchronization mode")
            materialization_value = node.annotations.get(
                DATAFLOW_PIPELINE_MATERIALIZATION_ATTR,
                DATAFLOW_PIPELINE_MATERIALIZE_COPY,
            )
            materialization = str(getattr(materialization_value, "value", materialization_value))
            if materialization not in DATAFLOW_PIPELINE_MATERIALIZATIONS:
                raise ValueError(f"pipeline transfer has unsupported materialization {materialization!r}")
            if materialization == DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT:
                lowering_plan_schema_version = None
                selected_implementation = DATAFLOW_PIPELINE_RESIDENT_IMPLEMENTATION
                selection_supported = True
                asynchronous = False
                uses_tma_descriptor = False
                requires_post_fill = False
                selection_reason = "typed pipeline buffer is already resident"
                rejected_candidates: list[str] = []
            else:
                plan = _ffi_api.ResolveTransferLowering(resolution_call, target)
                lowering_plan_schema_version = int(plan.schema_version)
                selected_implementation = str(plan.implementation_id)
                selection_supported = bool(plan.supported)
                asynchronous = bool(plan.asynchronous)
                uses_tma_descriptor = bool(plan.uses_tma_descriptor)
                requires_post_fill = bool(plan.requires_post_fill)
                selection_reason = str(plan.selection_reason)
                rejected_candidates = [str(candidate) for candidate in plan.rejected_candidates]
            handoff_producer = None
            handoff_transfer_value = node.annotations.get(DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_TRANSFER_ATTR)
            handoff_stage_value = node.annotations.get(DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_STAGE_ATTR)
            handoff_fingerprint_value = node.annotations.get(DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR)
            producer_metadata = (
                handoff_transfer_value,
                handoff_stage_value,
            )
            if any(item is not None for item in producer_metadata):
                if any(item is None for item in producer_metadata):
                    raise ValueError("handoff producer copy has incomplete plan metadata")
                handoff_plan = handler.cross_handler_handoff_plan
                if handoff_plan is None or not handoff_plan.enabled or handler.cross_handler_handoff_role != "producer":
                    raise ValueError("handoff producer copy is not owned by an enabled producer plan")
                if handoff_fingerprint_value is not None:
                    annotated_fingerprint = str(
                        getattr(
                            handoff_fingerprint_value,
                            "value",
                            handoff_fingerprint_value,
                        )
                    )
                    if annotated_fingerprint != handoff_plan.fingerprint:
                        raise ValueError("handoff producer copy changed its typed plan")
                handoff_transfer_index = int(handoff_transfer_value)
                handoff_stage_index = int(handoff_stage_value)
                if not 0 <= handoff_transfer_index < len(handoff_plan.transfer_plans):
                    raise ValueError("handoff producer copy has an invalid transfer index")
                transfer_plan = handoff_plan.transfer_plans[handoff_transfer_index]
                if not 0 <= handoff_stage_index < transfer_plan.buffer_stages:
                    raise ValueError("handoff producer copy has an invalid stage index")
                handoff_producer = {
                    "plan_fingerprint": handoff_plan.fingerprint,
                    "transfer_index": handoff_transfer_index,
                    "stage_index": handoff_stage_index,
                }
            destination_buffer_index = next(
                (index for index, buffer_data in enumerate(destination_buffers) if buffer_data.same_as(parsed.dst.data)),
                -1,
            )
            if destination_buffer_index < 0:
                destination_buffer_index = len(destination_buffers)
                destination_buffers.append(parsed.dst.data)
            requests.append(
                {
                    "handler_id": handler.handler_id,
                    "operation_index": operation_index,
                    "contract_schema_version": int(contract.schema_version),
                    "contract_fingerprint": tir_object_fingerprint(contract),
                    "source": {
                        "dtype": str(parsed.src.dtype),
                        "scope": parsed.src.scope(),
                        "logical_region": range_records(parsed.src_range),
                    },
                    "destination": {
                        "buffer_index": destination_buffer_index,
                        "dtype": str(parsed.dst.dtype),
                        "scope": parsed.dst.scope(),
                        "logical_region": range_records(parsed.dst_range),
                    },
                    "valid_source_region": range_records(contract.src_valid_region.region),
                    "oob_fill": prim_expr_record(contract.oob_fill),
                    "allow_async": bool(contract.allow_async),
                    "synchronization_owner": synchronization_owner,
                    "handoff_producer": handoff_producer,
                    "materialization": materialization,
                    "cluster_mask": int(getattr(node.annotations.get("cluster_mask", 0), "value", 0)),
                    "lowering_plan_schema_version": lowering_plan_schema_version,
                    "selected_implementation": selected_implementation,
                    "selection_supported": selection_supported,
                    "asynchronous": asynchronous,
                    "uses_tma_descriptor": uses_tma_descriptor,
                    "requires_post_fill": requires_post_fill,
                    "selection_reason": selection_reason,
                    "rejected_candidates": rejected_candidates,
                }
            )

        tir.stmt_functor.post_order_visit(handler.prim_func.body, visit)
    return requests


def collect_reshared_transport_bindings(
    compiled_program: Any,
    lowering: Any,
) -> list[dict[str, Any]]:
    """Record scheduler transport plans and their consumer PrimFunc bindings."""

    lowered_handlers = () if lowering is None else lowering.handlers
    bindings: list[dict[str, Any]] = []
    for stage_id, plan in compiled_program.plan.reshared_transport_plans:
        handlers = tuple(
            handler
            for handler in lowered_handlers
            if handler.reshared_transport_plan is not None and handler.reshared_transport_plan.fingerprint == plan.fingerprint
        )
        if lowering is not None and not handlers:
            raise ValueError(f"reshared transport stage {stage_id} lacks a consumer PrimFunc binding")
        for handler in handlers or (None,):
            primfunc_plan_fingerprint = None
            primfunc_plan_schema_version = None
            if handler is not None and handler.prim_func.attrs:
                fingerprint_attr = handler.prim_func.attrs.get("tl.reshared_transport_plan_fingerprint")
                schema_attr = handler.prim_func.attrs.get("tl.reshared_transport_plan_schema_version")
                if fingerprint_attr is not None:
                    primfunc_plan_fingerprint = str(fingerprint_attr)
                if schema_attr is not None:
                    primfunc_plan_schema_version = int(schema_attr)
            locality_counts = {
                locality: sum(step.locality == locality for step in plan.steps) for locality in ("local", "remote", "global")
            }
            bindings.append(
                {
                    "schema_version": (DATAFLOW_RESHARED_TRANSPORT_BINDING_SCHEMA_VERSION),
                    "stage_id": stage_id,
                    "request_fingerprint": plan.request_fingerprint,
                    "plan_fingerprint": plan.fingerprint,
                    "family": plan.family,
                    "implementation_id": plan.implementation_id,
                    "lowering_implementation_id": (plan.lowering_implementation_id),
                    "cluster_size": plan.cluster_size,
                    "consumer_access_order": plan.consumer_access_order,
                    "logical_tile_count": plan.logical_tile_count,
                    "physical_slot_count": plan.physical_slot_count,
                    "logical_tile_bytes": plan.logical_tile_bytes,
                    "transport_bytes_per_consumer": (plan.logical_tile_count * plan.logical_tile_bytes),
                    "transaction_bytes": plan.transaction_bytes,
                    "payload_partitions": [item.to_dict() for item in plan.payload_partitions],
                    "transfer_threads": plan.transfer_threads,
                    "receive_stage_count": plan.receive_stage_count,
                    "credit_count": plan.credit_count,
                    "credit_lifetimes": [item.to_dict() for item in plan.credit_lifetimes],
                    "remote_barrier_count": plan.remote_barrier_count,
                    "publish_sync_required": plan.publish_sync_required,
                    "release_sync_required": plan.release_sync_required,
                    "locality_counts": locality_counts,
                    "handler_id": None if handler is None else handler.handler_id,
                    "handler_thread_count": (None if handler is None else handler.thread_count),
                    "primfunc_plan_fingerprint": primfunc_plan_fingerprint,
                    "primfunc_plan_schema_version": (primfunc_plan_schema_version),
                }
            )
    return bindings


def collect_cross_handler_handoff_bindings(
    compiled_program: Any,
    lowering: Any,
    *,
    transfer_requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Record graph, queue, and PrimFunc ownership for every handoff plan."""

    bindings_by_plan: dict[str, list[DataflowHandoffQueueBinding]] = {}
    for binding in compiled_program.plan.cross_handler_handoff_bindings:
        bindings_by_plan.setdefault(binding.plan_fingerprint, []).append(binding)
    handler_roles_by_plan: dict[str, dict[str, list[int]]] = {}
    if lowering is not None:
        for handler in lowering.handlers:
            plan = handler.cross_handler_handoff_plan
            role = handler.cross_handler_handoff_role
            if plan is None or role is None:
                continue
            handler_roles_by_plan.setdefault(
                plan.fingerprint,
                {"producer": [], "consumer": []},
            )[role].append(handler.handler_id)
    transfers_by_plan: dict[str, list[dict[str, Any]]] = {}
    for request in transfer_requests:
        handoff = request["handoff_producer"]
        if handoff is None:
            continue
        transfers_by_plan.setdefault(
            str(handoff["plan_fingerprint"]),
            [],
        ).append(request)

    records = []
    for plan in compiled_program.plan.cross_handler_handoff_plans:
        queue_bindings = tuple(bindings_by_plan.pop(plan.fingerprint, ()))
        handler_roles = handler_roles_by_plan.pop(
            plan.fingerprint,
            {"producer": [], "consumer": []},
        )
        producer_handler_ids = tuple(sorted(handler_roles["producer"]))
        consumer_handler_ids = tuple(sorted(handler_roles["consumer"]))
        physical_requests = sorted(
            transfers_by_plan.pop(plan.fingerprint, ()),
            key=lambda item: (
                int(item["handler_id"]),
                int(item["handoff_producer"]["transfer_index"]),
                int(item["handoff_producer"]["stage_index"]),
            ),
        )
        expected_physical_transfers = {
            (handler_id, transfer.binding_index, stage_index)
            for handler_id in producer_handler_ids
            for transfer in plan.transfer_plans
            for stage_index in range(transfer.buffer_stages)
        }
        actual_physical_transfers = {
            (
                int(request["handler_id"]),
                int(request["handoff_producer"]["transfer_index"]),
                int(request["handoff_producer"]["stage_index"]),
            )
            for request in physical_requests
        }
        if plan.enabled:
            if not producer_handler_ids or not consumer_handler_ids:
                raise ValueError("enabled handoff plan lacks producer or consumer PrimFunc bindings")
            if len(actual_physical_transfers) != len(physical_requests) or actual_physical_transfers != expected_physical_transfers:
                raise ValueError("handoff producer copies do not cover every typed transfer stage")
        elif physical_requests:
            raise ValueError("disabled handoff plan owns producer copies")
        producer_ids = tuple(binding.producer_instruction_id for binding in queue_bindings)
        if len(producer_ids) != len(set(producer_ids)):
            raise ValueError("handoff plan contains duplicate producer bindings")
        active_consumers = tuple(binding.consumer_instruction_id for binding in queue_bindings if binding.state == "active")
        if len(active_consumers) != len(set(active_consumers)):
            raise ValueError("handoff plan contains duplicate consumer bindings")
        record = {
            "schema_version": DATAFLOW_CROSS_HANDLER_HANDOFF_BINDING_SCHEMA_VERSION,
            "request_fingerprint": plan.request_fingerprint,
            "plan_fingerprint": plan.fingerprint,
            "producer_stage_id": plan.producer_stage_id,
            "consumer_stage_id": plan.consumer_stage_id,
            "lookahead_distance": plan.lookahead_distance,
            "selected_buffer_stages": plan.selected_buffer_stages,
            "tail_policy": plan.tail_policy,
            "enabled": plan.enabled,
            "fallback_reason": plan.fallback_reason,
            "implementation_id": plan.implementation_id,
            "lowering_implementation_id": plan.lowering_implementation_id,
            "arena_alignment": plan.arena_alignment,
            "resource_budget_bytes": plan.resource_budget_bytes,
            "required_arena_bytes": plan.required_arena_bytes,
            "arena_bytes": plan.arena_bytes,
            "barrier_count": plan.barrier_count,
            "wait_required": plan.wait_required,
            "release_required": plan.release_required,
            "transfer_plans": [item.to_dict() for item in plan.transfer_plans],
            "buffer_lifetimes": [item.to_dict() for item in plan.buffer_lifetimes],
            "producer_handler_ids": list(producer_handler_ids),
            "consumer_handler_ids": list(consumer_handler_ids),
            "producer_transfers": [
                {
                    "handler_id": int(request["handler_id"]),
                    "operation_index": int(request["operation_index"]),
                    "transfer_index": int(request["handoff_producer"]["transfer_index"]),
                    "stage_index": int(request["handoff_producer"]["stage_index"]),
                    "contract_fingerprint": request["contract_fingerprint"],
                    "selected_implementation": request["selected_implementation"],
                    "asynchronous": request["asynchronous"],
                    "uses_tma_descriptor": request["uses_tma_descriptor"],
                }
                for request in physical_requests
            ],
            "queue_bindings": [item.to_dict() for item in queue_bindings],
        }
        record["fingerprint"] = sha256(serialize_canonical_json(record))
        records.append(record)
    if bindings_by_plan:
        raise ValueError("handoff queue bindings reference plans absent from the instruction plan")
    if handler_roles_by_plan:
        raise ValueError("handoff PrimFunc roles reference absent typed plans")
    if transfers_by_plan:
        raise ValueError("handoff producer copies reference absent typed plans")
    return records


def collect_pipeline_dataflow_bindings(
    lowering: Any,
    *,
    transfer_requests: list[dict[str, Any]],
    gemm_resolutions: list[dict[str, Any]],
    pipeline_resolutions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind one typed graph to the common physical lowering decisions."""

    if lowering is None:
        return []
    bindings: list[dict[str, Any]] = []
    for handler in lowering.handlers:
        plan = handler.pipeline_dataflow_plan
        if plan is None:
            continue
        handler_pipeline = sorted(
            (item for item in pipeline_resolutions if item["handler_id"] == handler.handler_id),
            key=lambda item: item["loop_index"],
        )
        if not handler_pipeline:
            # IR-only inspection has a validated logical plan but no physical pass result.
            continue
        handler_transfers = sorted(
            (
                item
                for item in transfer_requests
                if item["handler_id"] == handler.handler_id
                and item["synchronization_owner"] == "pipeline"
                and item["handoff_producer"] is None
            ),
            key=lambda item: item["operation_index"],
        )
        transfer_groups: list[list[dict[str, Any]]] = []
        for item in handler_transfers:
            destination_buffer_index = int(item["destination"]["buffer_index"])
            matching_group = next(
                (group for group in transfer_groups if int(group[0]["destination"]["buffer_index"]) == destination_buffer_index),
                None,
            )
            if matching_group is None:
                transfer_groups.append([item])
                continue
            representative = matching_group[0]
            if (
                representative["contract_fingerprint"] != item["contract_fingerprint"]
                or representative["destination"]["logical_region"] != item["destination"]["logical_region"]
            ):
                raise ValueError("pipeline dispatch alternatives changed their transfer contract")
            matching_group.append(item)
        handler_gemms = sorted(
            (item for item in gemm_resolutions if item["handler_id"] == handler.handler_id),
            key=lambda item: item["operation_index"],
        )
        requirements = plan.implementation_requirements
        expected = (
            requirements.transfer_count,
            requirements.gemm_count,
            1,
        )
        actual = (
            len(transfer_groups),
            len(handler_gemms),
            len(handler_pipeline),
        )
        if actual != expected:
            raise ValueError(
                "typed pipeline graph does not match physical lowering cardinality: "
                f"handler_id={handler.handler_id}, expected={expected!r}, "
                f"actual={actual!r}"
            )
        pipeline = handler_pipeline[0]
        if int(pipeline["requested_stages"]) != plan.selected_stages:
            raise ValueError(
                f"typed pipeline selected stages do not match physical lowering: {plan.selected_stages} != {pipeline['requested_stages']}"
            )
        if any(not item["selection_supported"] for item in handler_transfers):
            raise ValueError("typed pipeline graph contains an unsupported physical transfer")

        transfer_bindings = [
            {
                "graph_index": graph_index,
                "producer_partition": requirements.producer_partitions[graph_index],
                "materialization": requirements.transfer_materializations[graph_index],
                "alternatives": [
                    {
                        "operation_index": item["operation_index"],
                        "selected_implementation": item["selected_implementation"],
                        "asynchronous": item["asynchronous"],
                        "cluster_mask": item["cluster_mask"],
                    }
                    for item in group
                ],
            }
            for graph_index, group in enumerate(transfer_groups)
        ]
        gemm_bindings = [
            {
                "graph_index": graph_index,
                "operation_index": item["operation_index"],
                "selected_implementation": item["implementation_id"],
                "physical_shape": item["physical_shape"],
            }
            for graph_index, item in enumerate(handler_gemms)
        ]
        fallback_reasons = list(plan.fallback_reasons)
        if bool(pipeline["fallback"]):
            fallback_reasons.append("physical_pipeline_fallback")
        fallback_reasons.extend(f"gemm_{index}_fallback" for index, item in enumerate(handler_gemms) if item["used_fallback"])
        async_indices = set(requirements.async_transfer_indices)
        fallback_reasons.extend(
            f"transfer_{index}_synchronous_fallback"
            for index, group in enumerate(transfer_groups)
            if index in async_indices and any(not item["asynchronous"] for item in group)
        )
        record = {
            "schema_version": DATAFLOW_PIPELINE_BINDING_SCHEMA_VERSION,
            "handler_id": handler.handler_id,
            "request_fingerprint": plan.request_fingerprint,
            "plan_fingerprint": plan.fingerprint,
            "target_fingerprint": lowering.target_fingerprint,
            "selected_stages": plan.selected_stages,
            "selected_max_outstanding": plan.selected_max_outstanding,
            "buffer_versions": list(requirements.buffer_versions),
            "additive_gemm_groups": [list(group) for group in requirements.additive_gemm_groups],
            "release_after_gemm_indices": list(requirements.release_after_gemm_indices),
            "transfers": transfer_bindings,
            "gemms": gemm_bindings,
            "pipeline": {
                "loop_index": pipeline["loop_index"],
                "selected_implementation": pipeline["selected_implementation"],
                "fallback": pipeline["fallback"],
            },
            "used_fallback": bool(fallback_reasons),
            "fallback_reasons": fallback_reasons,
        }
        record["fingerprint"] = sha256(serialize_canonical_json(record))
        bindings.append(record)
    return bindings


def transfer_lowering_target(compile_config: Any) -> tvm.target.Target:
    target = str(compile_config.target)
    arch = str(compile_config.arch)
    if target == "cuda" and arch:
        return tvm.target.Target(f"cuda -arch={arch}")
    return tvm.target.Target(target)


def range_records(ranges: Any) -> list[dict[str, Any]]:
    return [
        {
            "min": prim_expr_record(region.min),
            "extent": prim_expr_record(region.extent),
        }
        for region in ranges
    ]


def prim_expr_record(expr: tir.PrimExpr) -> dict[str, Any]:
    record: dict[str, Any] = {
        "dtype": str(expr.dtype),
        "structural_fingerprint": tir_object_fingerprint(expr),
    }
    if isinstance(expr, tir.IntImm):
        record.update(kind="integer", value=int(expr.value))
    elif isinstance(expr, tir.FloatImm):
        record.update(kind="float", value=float(expr.value))
    elif isinstance(expr, tir.Var):
        record.update(kind="symbolic")
    else:
        record.update(kind="expression")
    return record


def tir_object_fingerprint(value: Any) -> str:
    structural_hash = int(tvm.ir.structural_hash(value, map_free_vars=True))
    return f"{structural_hash & ((1 << 64) - 1):016x}"


def serialize_canonical_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as err:
        raise TypeError("Dataflow decision artifact payload must contain only finite JSON values") from err


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "DATAFLOW_DECISION_ARTIFACT_SCHEMA_VERSION",
    "DATAFLOW_PIPELINE_BINDING_SCHEMA_VERSION",
    "DATAFLOW_RESHARED_TRANSPORT_BINDING_SCHEMA_VERSION",
    "DataflowDecisionArtifact",
    "collect_dataflow_decision_artifact",
    "validate_dataflow_decision_artifact_payload",
]
