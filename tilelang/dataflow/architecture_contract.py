"""Versioned ownership boundary between Dataflow specialization and TileLang lowering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .operation_contracts import (
    DATAFLOW_OPERATION_CONTRACT_FINGERPRINT,
    DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
)

DATAFLOW_LOWERING_BOUNDARY_SCHEMA_VERSION = 10


@dataclass(frozen=True)
class DataflowLoweringBoundaryContract:
    schema_version: int = DATAFLOW_LOWERING_BOUNDARY_SCHEMA_VERSION
    operation_contract_schema_version: int = DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION
    operation_contract_fingerprint: str = DATAFLOW_OPERATION_CONTRACT_FINGERPRINT
    dataflow_owned_inputs: tuple[str, ...] = (
        "instruction_slots",
        "handler_arity",
        "range_specialization",
        "plan_topology",
        "handler_runtime_binding",
        "typed_operation_requests",
        "typed_execution_request",
        "typed_execution_override",
        "operator_closure_values",
        "typed_operator_physical_contract",
        "typed_tensor_argument_layout",
    )
    dataflow_outputs: tuple[str, ...] = (
        "typed_handler_variant",
        "ordinary_primfunc",
        "typed_operation_contracts",
        "pipeline_dataflow_plan",
        "pipeline_dataflow_physical_binding",
        "reshared_transport_plan",
        "reshared_transport_physical_binding",
        "cross_handler_handoff_plan",
        "cross_handler_handoff_queue_binding",
        "cross_handler_handoff_runtime_binding",
        "execution_candidate_plan",
        "execution_resource_estimate",
        "target_capability_snapshot",
        "canonical_specialization_snapshot",
        "inferred_reshared_logical_type",
        "terminal_no_value_completion",
        "tensor_argument_layout_validation",
    )
    tilelang_owned_inputs: tuple[str, ...] = (
        "dtype",
        "scope",
        "logical_shape",
        "alignment",
        "bounds",
        "synchronization_contract",
        "pipeline_implementation_requirements",
        "pipeline_buffer_materialization",
        "transport_source_rank_mapping",
        "transport_payload_partition",
        "transport_receive_credit_lifetime",
        "handoff_pipeline_transfer_binding",
        "handoff_buffer_lifetime",
        "handoff_arena_placement",
        "target_capabilities",
    )
    tilelang_outputs: tuple[str, ...] = (
        "selected_implementation",
        "fallback_state",
        "rejected_candidates",
        "resource_requirements",
        "graph_operation_binding",
        "transport_family_and_physical_implementation",
        "handoff_transfer_wait_release_implementation",
    )
    forbidden_cross_layer_facts: tuple[str, ...] = (
        "trace_identity",
        "model_identity",
        "operator_diagnostic_name_selector",
        "diagnostic_name_in_operation_selection",
        "unversioned_operation_mapping",
        "scheduler_private_state_in_tilelang_lowering",
        "hardware_instruction_codegen_in_dataflow_specialization",
    )

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_LOWERING_BOUNDARY_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Dataflow lowering boundary schema version "
                f"{self.schema_version}; expected "
                f"{DATAFLOW_LOWERING_BOUNDARY_SCHEMA_VERSION}"
            )
        if (
            self.operation_contract_schema_version != DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION
            or self.operation_contract_fingerprint != DATAFLOW_OPERATION_CONTRACT_FINGERPRINT
        ):
            raise ValueError("Dataflow lowering boundary operation contract metadata is stale")
        for name in (
            "dataflow_owned_inputs",
            "dataflow_outputs",
            "tilelang_owned_inputs",
            "tilelang_outputs",
            "forbidden_cross_layer_facts",
        ):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"Dataflow lowering boundary {name} must be non-empty and unique")

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(include_fingerprint=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "operation_contract_schema_version": (self.operation_contract_schema_version),
            "operation_contract_fingerprint": (self.operation_contract_fingerprint),
            "dataflow_owned_inputs": list(self.dataflow_owned_inputs),
            "dataflow_outputs": list(self.dataflow_outputs),
            "tilelang_owned_inputs": list(self.tilelang_owned_inputs),
            "tilelang_outputs": list(self.tilelang_outputs),
            "forbidden_cross_layer_facts": list(self.forbidden_cross_layer_facts),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result


DATAFLOW_LOWERING_BOUNDARY_CONTRACT = DataflowLoweringBoundaryContract()


__all__ = [
    "DATAFLOW_LOWERING_BOUNDARY_CONTRACT",
    "DATAFLOW_LOWERING_BOUNDARY_SCHEMA_VERSION",
    "DataflowLoweringBoundaryContract",
]
