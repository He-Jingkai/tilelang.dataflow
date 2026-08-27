from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tilelang import _ffi_api
import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.decision_artifact import tir_object_fingerprint
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from testing.python.dataflow.test_dataflow_compile import make_program
from testing.python.dataflow.test_dataflow_primfunc_linking import make_scalar_program
from tvm import ir, tir


_REPO_ROOT = Path(__file__).resolve().parents[3]


def compile_debug_decision_program(**options):
    target = options.pop(
        "target_override",
        df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=232_448,
        ),
    )
    return df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [256]},
        block_size=128,
        task_extents=(1,),
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        target_override=target,
        **options,
    )


def test_dataflow_decision_artifact_is_canonical_and_reproducible():
    first = compile_debug_decision_program()
    repeated = compile_debug_decision_program()

    first_decisions = first.decision_artifact()
    repeated_decisions = repeated.decision_artifact()

    assert isinstance(first_decisions, df.DataflowDecisionArtifact)
    assert first_decisions.schema_version == df.DATAFLOW_DECISION_ARTIFACT_SCHEMA_VERSION
    assert first_decisions.fingerprint == repeated_decisions.fingerprint
    assert first_decisions.canonical_json == repeated_decisions.canonical_json
    serialized = first_decisions.to_dict()
    fingerprint = serialized.pop("fingerprint")
    assert fingerprint == first_decisions.fingerprint
    assert (
        json.dumps(
            serialized,
            sort_keys=True,
            separators=(",", ":"),
        )
        == first_decisions.canonical_json
    )
    assert first.dump_plan()["decisions"] == first_decisions.to_dict()


def test_dataflow_artifact_and_decisions_are_reproducible_across_processes(
    tmp_path,
):
    script = """
import json
from testing.python.dataflow.test_dataflow_decision_artifact import compile_debug_decision_program

compiled = compile_debug_decision_program()
print(json.dumps({
    "artifact_fingerprint": compiled.artifact_fingerprint,
    "decision_artifact": compiled.decision_artifact().to_dict(),
}, sort_keys=True))
"""
    results = []
    for index, hash_seed in enumerate(("1", "8675309")):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        environment["TILELANG_CACHE_DIR"] = str(tmp_path / f"cache-{index}")
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(_REPO_ROOT),
                environment.get("PYTHONPATH", ""),
            ]
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            cwd=_REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        results.append(json.loads(completed.stdout.splitlines()[-1]))

    assert results[0] == results[1]


def test_typed_transfer_request_is_reproducible_across_processes(tmp_path):
    script = """
from dataclasses import replace
import json

import tilelang.dataflow as df
from testing.python.dataflow.test_dataflow_primfunc_lowering import make_bounded_transfer_program
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug

compiled = df.compile(
    make_bounded_transfer_program(),
    topology=df.GPUTopology(sm_count=1, cluster_size=1),
    range_lengths={"kv": [13, 29]},
    task_extents=(2,),
    block_size=128,
    mode="debug",
    target_override=df.TargetCapabilitySnapshot.for_cuda(
        (9, 0), compiler_version=(12, 8)
    ),
    _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
)
lowering = df.lower_program_handlers_to_primfuncs(
    compiled.program,
    compiled.wrapper_spec,
    compiled.tensor_arg_plan,
    plan=compiled.plan,
    lower_to_cuda=False,
)
requests = replace(
    compiled,
    primfunc_lowering=lowering,
).decision_artifact().to_dict()["lowerings"]["copy"]["requests"]
print(json.dumps(requests, sort_keys=True))
"""
    results = []
    for index, hash_seed in enumerate(("17", "2309")):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        environment["TILELANG_CACHE_DIR"] = str(tmp_path / f"transfer-cache-{index}")
        environment["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT), environment.get("PYTHONPATH", "")])
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            cwd=_REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        results.append(json.loads(completed.stdout.splitlines()[-1]))

    assert results[0] == results[1]
    assert len(results[0]) == 1


def test_transfer_contract_fingerprint_ignores_symbol_and_buffer_names():
    def make_contract(source_name, destination_name, bound_name):
        source = tir.decl_buffer((32,), "float16", name=source_name)
        destination = tir.decl_buffer((16,), "float16", name=destination_name)
        valid_bound = tir.Var(bound_name, "int32")
        valid_region = tir.BufferRegion(
            source,
            [ir.Range.from_min_extent(0, valid_bound)],
        )
        return _ffi_api.ParseOperator(T.copy(source[0:16], destination, valid_region=valid_region)).transfer_contract

    first = make_contract("Source", "Destination", "valid_bound")
    renamed = make_contract("RenamedSource", "RenamedDestination", "renamed_bound")

    assert ir.structural_equal(first, renamed, map_free_vars=True)
    assert tir_object_fingerprint(first) == tir_object_fingerprint(renamed)


def test_dataflow_decision_artifact_records_current_generic_coverage():
    compiled = compile_debug_decision_program()

    decisions = compiled.decision_artifact().to_dict()

    assert decisions["artifact"]["artifact_fingerprint"] == compiled.artifact_fingerprint
    assert decisions["artifact"]["compile_mode"] == "debug"
    assert decisions["artifact"]["handler_lowering"] == (dataflow_debug.EMPTY_HANDLER.name)
    assert decisions["governance"]["implementation_registry"] == (df.dataflow_implementation_registry().to_dict())
    assert decisions["governance"]["lowering_boundary"] == (df.DATAFLOW_LOWERING_BOUNDARY_CONTRACT.to_dict())
    selected_implementations = {record["implementation_id"]: record for record in decisions["governance"]["selected_implementations"]}
    assert selected_implementations[dataflow_debug.EMPTY_HANDLER.name]["selected_explicitly"] is True
    assert selected_implementations[dataflow_debug.EMPTY_HANDLER.name]["state"] == ("experimental")
    assert decisions["scheduler"]["policy"] == compiled.plan.scheduler_policy
    assert decisions["scheduler"]["reduce_strategy"] == compiled.plan.reduce_strategy
    assert decisions["scheduler"]["topology"] == {
        "sm_count": 2,
        "cluster_size": 1,
    }
    assert decisions["handlers"]["coverage"] == "base_identity_and_typed_variant_key"
    assert decisions["handlers"]["items"]
    assert all(handler["binding_key"] is not None for handler in decisions["handlers"]["items"])
    assert all(
        handler["base_identity"] is not None
        and handler["variant_key"] is not None
        and handler["variant_key"]["base_identity"] == handler["base_identity"]
        for handler in decisions["handlers"]["items"]
    )
    assert decisions["lowerings"]["gemm"]["registry_version"] == (df.GEMM_LOWERING_REGISTRY_VERSION)
    assert decisions["lowerings"]["copy"]["coverage"] == ("selected_transfer_lowering_and_tma_descriptor_metadata")
    assert decisions["lowerings"]["copy"]["transfer_contract_selection_state"] == "selected"
    assert decisions["lowerings"]["copy"]["registry_version"] == 2
    assert compiled.compile_config.options_dict()["transfer_lowering_registry_version"] == 2
    assert decisions["lowerings"]["copy"]["requests"] == []
    assert decisions["lowerings"]["pipeline"]["coverage"] == ("typed_graph_to_transfer_gemm_pipeline_physical_binding")
    assert decisions["lowerings"]["pipeline"]["registry_version"] == 1
    assert compiled.compile_config.options_dict()["pipeline_lowering_registry_version"] == 1
    assert decisions["lowerings"]["pipeline"]["resolutions"] == []
    assert decisions["lowerings"]["pipeline"]["dataflow_bindings"] == []
    assert decisions["lowerings"]["precision"]["coverage"] == (
        "typed_policy_operator_contract_selected_dtype_error_budget_reduction_order_and_implementation"
    )
    assert decisions["lowerings"]["precision"]["plan"] == (compiled.compile_config.options_dict()["precision_plan"])
    assert decisions["memory"]["planner"] == compiled.memory_plan.to_dict()
    assert decisions["memory"]["validation"]["valid"]
    assert decisions["memory"]["shared_memory_bytes"] == (compiled.launch_package.shared_memory_bytes)
    assert len(decisions["memory"]["slot_placements"]) == len(compiled.plan.slots)
    assert all(
        placement["alignment"] == df.DATAFLOW_SLOT_ALIGNMENT and placement["live_range"] is not None and placement["placement_reason"]
        for placement in decisions["memory"]["slot_placements"]
    )


def test_dataflow_decision_artifact_changes_with_owned_compile_inputs():
    baseline = compile_debug_decision_program()
    changed_scheduler = compile_debug_decision_program(
        scheduler_policy="cluster_local",
    )
    changed_semantics = compile_debug_decision_program(
        semantic_config=df.DataflowSemanticConfig(direct_slot_seed_reduce=True),
    )
    changed_target = compile_debug_decision_program(
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (12, 0),
            compiler_version=(13, 0),
            max_dynamic_shared_memory=232_448,
        ),
    )

    baseline_decisions = baseline.decision_artifact().to_dict()
    scheduler_decisions = changed_scheduler.decision_artifact().to_dict()
    semantic_decisions = changed_semantics.decision_artifact().to_dict()
    target_decisions = changed_target.decision_artifact().to_dict()

    assert (
        len(
            {
                baseline_decisions["fingerprint"],
                scheduler_decisions["fingerprint"],
                semantic_decisions["fingerprint"],
                target_decisions["fingerprint"],
            }
        )
        == 4
    )
    assert scheduler_decisions["scheduler"]["policy"] == "cluster_local"
    assert semantic_decisions["lowerings"]["precision"]["semantic_config"]["direct_slot_seed_reduce"] is True
    assert target_decisions["target"]["fingerprint"] != (baseline_decisions["target"]["fingerprint"])


def test_collecting_decisions_does_not_change_existing_compiled_artifact():
    compiled = compile_debug_decision_program()
    artifact_fingerprint = compiled.artifact_fingerprint
    wrapper_source = compiled.wrapper_source
    plan = compiled.plan
    packed_plan = compiled.packed_plan
    launch_package = compiled.launch_package

    compiled.decision_artifact()
    compiled.dump_plan()

    assert compiled.artifact_fingerprint == artifact_fingerprint
    assert compiled.wrapper_source == wrapper_source
    assert compiled.plan == plan
    assert compiled.packed_plan == packed_plan
    assert compiled.launch_package == launch_package


def test_dataflow_decision_artifact_rejects_noncanonical_or_stale_payload():
    artifact = compile_debug_decision_program().decision_artifact()

    with pytest.raises(ValueError, match="canonical serialization"):
        replace(artifact, canonical_json=artifact.canonical_json + " ")
    with pytest.raises(ValueError, match="fingerprint does not match"):
        replace(artifact, fingerprint="0" * 64)


@pytest.mark.parametrize(
    ("mode", "mode_options", "expected_handler"),
    [
        ("executable", {}, df.PRIMFUNC_HANDLER_LOWERING),
        (
            "inspect",
            {"inspection_stage": "ir"},
            df.PRIMFUNC_HANDLER_LOWERING,
        ),
        (
            "debug",
            {"_experimental_debug_handler": dataflow_debug.EMPTY_HANDLER.name},
            dataflow_debug.EMPTY_HANDLER.name,
        ),
    ],
)
def test_decision_governance_covers_every_compile_mode(
    mode,
    mode_options,
    expected_handler,
):
    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode=mode,
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=232_448,
        ),
        **mode_options,
    )

    decisions = compiled.decision_artifact().to_dict()
    selected = {record["implementation_id"]: record for record in decisions["governance"]["selected_implementations"]}

    assert decisions["artifact"]["compile_mode"] == mode
    assert decisions["artifact"]["handler_lowering"] == expected_handler
    assert (
        compiled.compile_config.options_dict()["implementation_registry_fingerprint"]
        == decisions["governance"]["implementation_registry"]["fingerprint"]
    )
    assert "round_robin" in selected
    assert "dataflow.memory.v1:shared" in selected
    if mode == "debug":
        assert selected[dataflow_debug.EMPTY_HANDLER.name]["selection_sources"] == ["debug.handler_provider"]
    else:
        assert dataflow_debug.EMPTY_HANDLER.name not in selected


def test_decision_artifact_rejects_incomplete_governance_provenance():
    artifact = compile_debug_decision_program().decision_artifact()
    original = artifact.to_dict()
    original.pop("fingerprint")

    missing_registry = deepcopy(original)
    missing_registry["governance"].pop("implementation_registry")
    with pytest.raises(ValueError, match="lacks implementation registry"):
        df.DataflowDecisionArtifact.from_payload(missing_registry)

    stale_registry = deepcopy(original)
    stale_registry["governance"]["implementation_registry"]["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="implementation registry is stale"):
        df.DataflowDecisionArtifact.from_payload(stale_registry)

    stale_boundary = deepcopy(original)
    stale_boundary["governance"]["lowering_boundary"]["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="lowering boundary is stale"):
        df.DataflowDecisionArtifact.from_payload(stale_boundary)

    missing_debug_provider = deepcopy(original)
    missing_debug_provider["governance"]["selected_implementations"] = [
        record
        for record in missing_debug_provider["governance"]["selected_implementations"]
        if record["implementation_id"] != dataflow_debug.EMPTY_HANDLER.name
    ]
    with pytest.raises(ValueError, match="lacks lifecycle records.*experimental.empty"):
        df.DataflowDecisionArtifact.from_payload(missing_debug_provider)

    stale_owner = deepcopy(original)
    stale_owner["governance"]["selected_implementations"][0]["owner"] = "unknown"
    with pytest.raises(ValueError, match="has stale owner"):
        df.DataflowDecisionArtifact.from_payload(stale_owner)
