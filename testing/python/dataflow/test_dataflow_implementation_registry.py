from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

import tilelang.dataflow as df
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from testing.python.dataflow.test_dataflow_compile import make_program


def spec(
    implementation_id: str,
    *,
    state: df.DataflowImplementationState = df.DataflowImplementationState.STABLE,
    default_enabled: bool = True,
    removal_date: str | None = None,
) -> df.DataflowImplementationSpec:
    return df.DataflowImplementationSpec(
        implementation_id=implementation_id,
        domain="test.domain",
        state=state,
        owner="QA",
        legality_contract="test legality contract",
        benchmark_evidence=("testing/python/dataflow/test_dataflow_implementation_registry.py",),
        default_enabled=default_enabled,
        removal_date=removal_date,
    )


def test_implementation_registry_is_order_independent_and_complete():
    first = spec("test.first")
    second = spec("test.second", default_enabled=False)

    forward = df.DataflowImplementationRegistry((first, second))
    reversed_registry = df.DataflowImplementationRegistry((second, first))

    assert forward.fingerprint == reversed_registry.fingerprint
    assert forward.to_dict() == {
        "schema_version": df.DATAFLOW_IMPLEMENTATION_REGISTRY_VERSION,
        "fingerprint": forward.fingerprint,
        "implementation_count": 2,
    }
    assert forward.to_dict(include_catalog=True)["implementations"] == [
        first.to_dict(),
        second.to_dict(),
    ]


def test_global_implementation_registry_covers_selectable_domains():
    registry = df.dataflow_implementation_registry()
    specs = registry.specs

    assert {
        "scheduler.policy",
        "tilelang.lowering.gemm",
        "tilelang.lowering.transfer",
        "tilelang.lowering.pipeline",
        "dataflow.precision",
        "dataflow.memory",
        "dataflow.semantic",
        "dataflow.debug_handler",
    } <= {spec.domain for spec in specs}
    assert all(spec.owner and spec.legality_contract for spec in specs)
    assert all(spec.benchmark_evidence for spec in specs)
    assert all("mla" not in spec.implementation_id for spec in specs)
    assert all(
        spec.state is df.DataflowImplementationState.STABLE or (spec.default_enabled is False and spec.removal_date is not None)
        for spec in specs
    )


def test_registry_rejects_unregistered_retired_and_implicit_experimental():
    experimental = spec(
        "test.experimental",
        state=df.DataflowImplementationState.EXPERIMENTAL,
        default_enabled=False,
        removal_date="2026-12-31",
    )
    retired = spec(
        "test.retired",
        state=df.DataflowImplementationState.RETIRED,
        default_enabled=False,
        removal_date="2026-01-31",
    )
    registry = df.DataflowImplementationRegistry((experimental, retired))

    with pytest.raises(ValueError, match="unregistered Dataflow implementation"):
        registry.require_selectable("test.unknown", selected_explicitly=True)
    with pytest.raises(ValueError, match="retired Dataflow implementation"):
        registry.require_selectable("test.retired", selected_explicitly=True)
    with pytest.raises(ValueError, match="requires an explicit typed selection"):
        registry.require_selectable(
            "test.experimental",
            selected_explicitly=False,
            today=date(2026, 7, 29),
        )
    assert (
        registry.require_selectable(
            "test.experimental",
            selected_explicitly=True,
            today=date(2026, 7, 29),
        )
        is experimental
    )


def test_registry_rejects_expired_experimental_implementation():
    registry = df.DataflowImplementationRegistry(
        (
            spec(
                "test.expired",
                state=df.DataflowImplementationState.EXPERIMENTAL,
                default_enabled=False,
                removal_date="2026-06-30",
            ),
        )
    )

    with pytest.raises(ValueError, match="expired on 2026-06-30"):
        registry.require_selectable(
            "test.expired",
            selected_explicitly=True,
            today=date(2026, 7, 1),
        )


def test_non_stable_implementation_metadata_is_fail_closed():
    with pytest.raises(ValueError, match="must default off"):
        spec(
            "test.default_on",
            state=df.DataflowImplementationState.EXPERIMENTAL,
            removal_date="2026-12-31",
        )
    with pytest.raises(ValueError, match="requires a removal date"):
        spec(
            "test.no_removal",
            state=df.DataflowImplementationState.EXPERIMENTAL,
            default_enabled=False,
        )
    with pytest.raises(ValueError, match="cannot have a removal date"):
        spec("test.stable_removal", removal_date="2026-12-31")
    with pytest.raises(TypeError, match="state must be DataflowImplementationState"):
        replace(spec("test.invalid_state"), state="stable")


def test_lowering_boundary_contract_is_versioned_and_fingerprinted():
    contract = df.DATAFLOW_LOWERING_BOUNDARY_CONTRACT
    serialized = contract.to_dict()

    assert serialized["schema_version"] == (df.DATAFLOW_LOWERING_BOUNDARY_SCHEMA_VERSION)
    assert serialized["fingerprint"] == contract.fingerprint
    assert len(contract.fingerprint) == 64
    assert {
        "trace_identity",
        "model_identity",
        "scheduler_private_state_in_tilelang_lowering",
        "hardware_instruction_codegen_in_dataflow_specialization",
    } <= set(contract.forbidden_cross_layer_facts)
    changed = replace(
        contract,
        tilelang_owned_inputs=(*contract.tilelang_owned_inputs, "new_contract"),
    )
    assert changed.fingerprint != contract.fingerprint

    with pytest.raises(ValueError, match="Unsupported Dataflow lowering boundary"):
        replace(contract, schema_version=contract.schema_version + 1)
    with pytest.raises(ValueError, match="must be non-empty and unique"):
        replace(
            contract,
            dataflow_owned_inputs=(
                contract.dataflow_owned_inputs[0],
                contract.dataflow_owned_inputs[0],
            ),
        )


@pytest.mark.parametrize(
    "retired_option",
    ["dataflow_scratch_backed_slots", "hbm_direct_global"],
)
def test_compile_rejects_retired_memory_compatibility_options(
    retired_option,
):
    with pytest.raises(ValueError, match="compile options were retired"):
        df.compile(
            make_program(),
            topology=df.GPUTopology(sm_count=1, cluster_size=1),
            range_lengths={"kv": [64]},
            block_size=32,
            mode="debug",
            target_override=df.TargetCapabilitySnapshot.for_cuda(
                (9, 0),
                compiler_version=(12, 8),
            ),
            _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
            **{retired_option: True},
        )
