from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from testing.python.dataflow.test_dataflow_scheduler_config import (
    ReplayIntermediate,
    make_replay_program,
    replay_finalize,
    replay_iter,
)
from testing.python.dataflow.test_dataflow_compile import make_reshared_program
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug


_REPO_ROOT = Path(__file__).resolve().parents[3]


@T.dataflow.reduce(
    associative=True,
    physical_contract=df.DataflowOperatorPhysicalContract(
        output_alias_input_indices=(0, 1),
    ),
)
def alias_replay_reduce(
    left: ReplayIntermediate,
    right: ReplayIntermediate,
) -> ReplayIntermediate:
    raise AssertionError("Dataflow reduce body should not execute during scheduling")


def make_alias_replay_program():
    return (
        T.dataflow_program(task_domain=("task",), dynamic_ranges={"kv": "lengths"})
        .partial(replay_iter(Input="Input"), task_args=("task",), range_axis="kv")
        .reduce(alias_replay_reduce())
        .finalize(replay_finalize(Output="Output"))
    )


def make_target(*, cluster: bool = True):
    return df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
        supports_cluster_launch=cluster,
        max_cluster_size=16 if cluster else 1,
        max_dynamic_shared_memory=232_448,
    )


def select(
    lengths,
    *,
    topology=None,
    policy_records=False,
    scheduler_config=None,
    **options,
):
    if topology is None:
        topology = df.GPUTopology(sm_count=4, cluster_size=2)
    return df.select_scheduler_auto_policy(
        make_replay_program(),
        topology=topology,
        range_lengths={"kv": tuple(lengths)},
        block_size=64,
        task_extents=(len(lengths),),
        target_capabilities=make_target(),
        policy_records=policy_records,
        scheduler_config=scheduler_config,
        **options,
    )


def test_scheduler_auto_policy_features_use_distributions_not_exact_lists():
    program = make_replay_program()
    topology = df.GPUTopology(sm_count=8, cluster_size=4)
    first = df.extract_scheduler_features(
        program,
        topology=topology,
        range_lengths={"kv": (64, 192, 320, 512)},
        block_size=64,
        task_extents=(4,),
        target_capabilities=make_target(),
    )
    permuted = df.extract_scheduler_features(
        program,
        topology=topology,
        range_lengths={"kv": (512, 64, 320, 192)},
        block_size=64,
        task_extents=(4,),
        target_capabilities=make_target(),
    )

    assert first.to_dict() == permuted.to_dict()
    assert first.to_dict()["tile_distribution"]["histogram"] == {
        "1": 1,
        "2": 0,
        "3_4": 1,
        "5_8": 2,
        "9_16": 0,
        "17_32": 0,
        "33_plus": 0,
    }

    common = dict(
        program=program,
        features=first,
        range_offsets=None,
        block_size=64,
        task_extents=(4,),
        include_exit=True,
        requested_scheduler_policy="auto",
        requested_reduce_strategy="auto",
        scheduler_config=df.DataflowSchedulerConfig(),
        schedule_options={
            "force_hbm_comms": False,
            "partial_only": False,
        },
    )
    first_key = df.scheduler_workload_fingerprint(
        range_lengths={"kv": (64, 192, 320, 512)},
        **common,
    )
    permuted_key = df.scheduler_workload_fingerprint(
        range_lengths={"kv": (512, 64, 320, 192)},
        **common,
    )
    assert first_key != permuted_key


@pytest.mark.parametrize(
    ("seed", "topology"),
    (
        (7, df.GPUTopology(sm_count=4, cluster_size=2)),
        (19, df.GPUTopology(sm_count=8, cluster_size=4)),
        (31, df.GPUTopology(sm_count=6, cluster_size=3)),
    ),
)
def test_scheduler_auto_policy_randomized_distributions_are_legal_and_replayable(
    seed,
    topology,
):
    rng = random.Random(seed)
    lengths = tuple(rng.randint(1, 24) * 64 - rng.randint(0, 31) for _ in range(7))
    result = select(lengths, topology=topology)
    decision = result.decision

    assert decision.record_status == "disabled"
    assert decision.features.task_count == len(lengths)
    assert len(decision.candidates) >= 4
    assert decision.selected_evaluation.legal
    assert any(item.legal for item in decision.candidates)
    assert all(item.legal or item.rejection_reason for item in decision.candidates)

    replay = df.record_schedule_replay(
        make_replay_program(),
        result.selected_plan,
        range_lengths={"kv": lengths},
    )
    restored = df.DataflowScheduleReplay.from_dict(json.loads(json.dumps(replay.to_dict())))
    replayed = df.replay_schedule(make_replay_program(), restored)
    assert df.instruction_plan_fingerprint(replayed) == (decision.selected_evaluation.plan_fingerprint)


def test_scheduler_auto_policy_handles_multi_axis_stage_graph():
    program = make_reshared_program()
    result = df.select_scheduler_auto_policy(
        program,
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": (256,), "hidden_tile": (384,)},
        block_size=64,
        task_extents=(1, 1),
        force_hbm_comms=True,
        policy_records=False,
        target_capabilities=make_target(),
    )

    assert result.decision.features.task_count == 1
    assert result.decision.features.range_length_min == 256
    assert result.decision.features.range_length_max == 384
    assert result.selected_plan.scheduler_policy == "stage_graph"
    assert result.selected_plan.reduce_strategy == "none"
    assert len(result.decision.candidates) == 1
    assert result.decision.selected_evaluation.legal

    compiled = df.compile(
        program,
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": (256,), "hidden_tile": (384,)},
        block_size=64,
        task_extents=(1, 1),
        force_hbm_comms=True,
        scheduler_policy="auto",
        reduce_strategy="auto",
        scheduler_policy_records=False,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        target_override=make_target(),
    )
    assert compiled.plan.scheduler_policy == "stage_graph"
    assert compiled.decision_artifact().to_dict()["scheduler"]["auto_policy"]["selected_candidate_id"] == "cyclic_all_at_once"


def test_scheduler_auto_policy_record_hit_bypasses_generic_candidate_generation():
    miss = select((512, 128, 320))
    record = df.DataflowSchedulerPolicyRecord(
        record_id="synthetic-best-known",
        workload_fingerprint=miss.decision.workload_fingerprint,
        candidate=miss.decision.selected_candidate,
        evidence=(("source", "unit-test"),),
    )
    records = df.DataflowSchedulerPolicyRecordSet(
        record_set_version="synthetic.v1",
        records=(record,),
    )

    hit = select((512, 128, 320), policy_records=records)

    assert hit.decision.record_status == "hit"
    assert hit.decision.record_id == "synthetic-best-known"
    assert hit.decision.selection_reason == "best_known_record:synthetic-best-known"
    assert len(hit.decision.candidates) == 1
    assert hit.decision.selected_candidate == miss.decision.selected_candidate
    assert df.instruction_plan_fingerprint(hit.selected_plan) == (df.instruction_plan_fingerprint(miss.selected_plan))


def test_scheduler_auto_policy_record_file_round_trip_and_exact_miss(tmp_path):
    miss = select((256, 384))
    records = df.DataflowSchedulerPolicyRecordSet(
        record_set_version="file.v1",
        records=(
            df.DataflowSchedulerPolicyRecord(
                record_id="file-record",
                workload_fingerprint=miss.decision.workload_fingerprint,
                candidate=miss.decision.selected_candidate,
            ),
        ),
    )
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records.to_dict()), encoding="utf-8")

    restored = df.load_scheduler_policy_records(path)
    hit = select((256, 384), policy_records=path)
    exact_miss = select((384, 256), policy_records=path)

    assert restored == records
    assert restored.fingerprint == records.fingerprint
    assert hit.decision.record_status == "hit"
    assert exact_miss.decision.record_status == "miss"
    assert exact_miss.decision.record_id is None


def test_default_scheduler_policy_records_do_not_embed_workloads():
    records = df.load_scheduler_policy_records()

    assert records.record_set_version == "none"
    assert records.feature_schema_version == df.DATAFLOW_SCHEDULER_FEATURE_SCHEMA_VERSION
    assert records.cost_model_version == df.DATAFLOW_COST_MODEL_VERSION
    assert records.records == ()
    with pytest.raises(ValueError, match="Unknown Dataflow scheduler policy record set"):
        df.load_scheduler_policy_records({**records.to_dict(), "typo": True})
    with pytest.raises(TypeError, match="record set, mapping, path"):
        df.load_scheduler_policy_records(True)


def test_scheduler_auto_policy_rejects_stale_or_illegal_records_and_falls_back():
    miss = select((128, 512, 192))
    illegal_candidate = replace(
        miss.decision.selected_candidate,
        candidate_id="illegal-record-candidate",
        cluster_task_assignment=((0,),),
        cluster_assignment_policy="invalid-test-record",
    )
    illegal_records = df.DataflowSchedulerPolicyRecordSet(
        record_set_version="illegal.v1",
        records=(
            df.DataflowSchedulerPolicyRecord(
                record_id="illegal-record",
                workload_fingerprint=miss.decision.workload_fingerprint,
                candidate=illegal_candidate,
            ),
        ),
    )
    recovered = select((128, 512, 192), policy_records=illegal_records)

    assert recovered.decision.record_status == "miss"
    assert recovered.decision.record_id == "illegal-record"
    assert "cluster_task_assignment" in recovered.decision.record_rejection_reason
    assert recovered.decision.selected_evaluation.legal

    stale_records = replace(illegal_records, feature_schema_version=999)
    incompatible = select((128, 512, 192), policy_records=stale_records)
    assert incompatible.decision.record_status == "incompatible"
    assert "feature schema mismatch" in incompatible.decision.record_rejection_reason


def test_scheduler_auto_policy_honors_explicit_policy_and_typed_config_overrides():
    config = df.DataflowSchedulerConfig.from_options(level0_queue_order="long_first")
    result = select(
        (128, 640, 256),
        scheduler_config=config,
        requested_scheduler_policy="cluster_local",
        requested_reduce_strategy="streaming",
    )

    assert result.selected_plan.scheduler_policy == "cluster_local"
    assert result.selected_plan.reduce_strategy == "streaming"
    assert result.selected_scheduler_config.get("level0_queue_order") == "long_first"
    assert all(
        item.candidate.scheduler_policy == "cluster_local"
        and item.candidate.reduce_strategy == "streaming"
        and item.candidate.queue_policy == "configured"
        for item in result.decision.candidates
    )


def test_scheduler_auto_policy_rejects_topology_outside_target_resources():
    with pytest.raises(ValueError, match="does not support cluster launch"):
        df.extract_scheduler_features(
            make_replay_program(),
            topology=df.GPUTopology(sm_count=4, cluster_size=2),
            range_lengths={"kv": (128, 256)},
            block_size=64,
            target_capabilities=make_target(cluster=False),
        )

    limited = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        supports_cluster_launch=True,
        max_cluster_size=2,
    )
    with pytest.raises(ValueError, match="exceeds the target limit"):
        df.extract_scheduler_features(
            make_replay_program(),
            topology=df.GPUTopology(sm_count=8, cluster_size=4),
            range_lengths={"kv": (128, 256)},
            block_size=64,
            target_capabilities=limited,
        )


def test_scheduler_auto_policy_filters_unbounded_reduce_for_scratch_backed_slots():
    result = select(
        (70,),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        scratch_backed_slots=True,
    )

    assert result.selected_plan.reduce_strategy == "streaming"
    rejected = [item for item in result.decision.candidates if item.candidate.reduce_strategy != "streaming"]
    assert rejected
    assert all(not item.legal for item in rejected)
    assert all("scratch-backed slots" in item.rejection_reason for item in rejected)


def test_scheduler_auto_policy_derives_joint_capacity_candidates_from_generic_contracts():
    program = make_alias_replay_program()
    topology = df.GPUTopology(sm_count=4, cluster_size=2)
    lengths = (512, 384)
    features = df.extract_scheduler_features(
        program,
        topology=topology,
        range_lengths={"kv": lengths},
        block_size=64,
        task_extents=(2,),
        target_capabilities=make_target(),
    )

    without_alias = df.generate_scheduler_candidates(features)
    assert not any(candidate.candidate_id.startswith("capacity_ordered_joint_") for candidate in without_alias)

    async_config = df.DataflowSchedulerConfig.from_options(
        joint_hbm_async_receive_pipeline=True,
    )
    candidates = df.generate_scheduler_candidates(
        features,
        scheduler_config=async_config,
        reduce_output_alias_input_indices=(0, 1),
    )
    binary_joint = {
        candidate.candidate_id: candidate for candidate in candidates if candidate.candidate_id.startswith("capacity_ordered_joint_")
    }
    assert set(binary_joint) == {
        "capacity_ordered_joint_left",
        "capacity_ordered_joint_right",
    }
    ordered_joint = {
        candidate.candidate_id: candidate for candidate in candidates if candidate.candidate_id.startswith("capacity_ordered_tree_joint_")
    }
    assert set(ordered_joint) == {
        "capacity_ordered_tree_joint_left",
        "capacity_ordered_tree_joint_right",
    }
    for candidate in (*binary_joint.values(), *ordered_joint.values()):
        options = dict(candidate.scheduler_config_options)
        assert options["critical_path_max_task_segments"] == topology.cluster_count
        assert options["critical_path_max_extra_chunks"] == (topology.sm_count + features.task_count)
        assert options["joint_schedule"] is True
        assert "joint_hbm_segmented_pipeline" not in options

    result = df.select_scheduler_auto_policy(
        program,
        topology=topology,
        range_lengths={"kv": lengths},
        block_size=64,
        task_extents=(2,),
        target_capabilities=make_target(),
        policy_records=False,
        scratch_backed_slots=True,
        scheduler_config=async_config,
    )
    evaluations = {item.candidate.candidate_id: item for item in result.decision.candidates}
    assert result.decision.selected_candidate_id.startswith("global_capacity_tree_joint_")
    assert all(evaluations[candidate_id].legal for candidate_id in (*binary_joint, *ordered_joint))


def test_scheduler_auto_policy_derives_global_capacity_and_finalizer_variants():
    program = make_alias_replay_program()
    topology = df.GPUTopology(sm_count=16, cluster_size=8)
    features = df.extract_scheduler_features(
        program,
        topology=topology,
        range_lengths={"kv": (1024, 896)},
        block_size=64,
        task_extents=(2,),
        target_capabilities=make_target(),
    )

    candidates = df.generate_scheduler_candidates(
        features,
        reduce_output_alias_input_indices=(0, 1),
        fused_reduce_finalize_arities=(2, 3),
    )
    global_candidates = {candidate.candidate_id: candidate for candidate in candidates if candidate.candidate_id.startswith("global_")}

    assert {
        "global_capacity_fused2_tree_joint_left",
        "global_capacity_fused3_tree_joint_left",
        "global_minimax_fused2_tree_joint_left",
        "global_minimax_fused3_tree_joint_left",
        "global_copacked_fused2_tree_joint_left",
        "global_copacked_fused3_tree_joint_left",
        "global_capacity_fused2_tree_joint_right",
        "global_capacity_fused3_tree_joint_right",
        "global_minimax_fused2_tree_joint_right",
        "global_minimax_fused3_tree_joint_right",
        "global_copacked_fused2_tree_joint_right",
        "global_copacked_fused3_tree_joint_right",
    } <= global_candidates.keys()
    for candidate in global_candidates.values():
        options = dict(candidate.scheduler_config_options)
        assert options["global_capacity_chunks"] is True
        assert options["joint_schedule"] is True
        assert options["ordered_interval_tree"] is True
        assert options["fused_reduce_finalize_max_arity"] in {2, 3}
        if candidate.candidate_id.startswith("global_copacked_"):
            assert options["global_capacity_copacked_chunks"] is True


def test_scheduler_auto_policy_keeps_global_capacity_search_when_tasks_exceed_ctas():
    program = make_alias_replay_program()
    topology = df.GPUTopology(sm_count=16, cluster_size=8)
    features = df.extract_scheduler_features(
        program,
        topology=topology,
        range_lengths={"kv": tuple(256 + 64 * index for index in range(20))},
        block_size=64,
        task_extents=(20,),
        target_capabilities=make_target(),
    )

    candidates = df.generate_scheduler_candidates(
        features,
        reduce_output_alias_input_indices=(0, 1),
        fused_reduce_finalize_arities=(2, 3),
    )

    assert any(candidate.cluster_assignment_policy == "global_capacity" for candidate in candidates)

    selected = df.select_scheduler_auto_policy(
        program,
        topology=topology,
        range_lengths={"kv": tuple(256 + 64 * index for index in range(20))},
        block_size=64,
        task_extents=(20,),
        target_capabilities=make_target(),
        policy_records=False,
        scratch_backed_slots=True,
        scheduler_config=df.DataflowSchedulerConfig.from_options(
            global_capacity_replay_refine=False,
            chunk_length_search=False,
        ),
    )
    assert selected.decision.selected_candidate_id.endswith("_right")
    assert selected.selected_plan.scheduler_config.get("streaming_tree_consumer") == "right"


def test_scheduler_auto_policy_derives_bounded_resident_large_cluster_tree():
    program = make_alias_replay_program()
    topology = df.GPUTopology(sm_count=16, cluster_size=8)
    lengths = (1024, 896)
    features = df.extract_scheduler_features(
        program,
        topology=topology,
        range_lengths={"kv": lengths},
        block_size=64,
        task_extents=(2,),
        target_capabilities=make_target(),
    )

    candidates = df.generate_scheduler_candidates(
        features,
        reduce_output_alias_input_indices=(0, 1),
    )
    resident = {
        candidate.candidate_id: candidate for candidate in candidates if candidate.candidate_id.startswith("resident_cluster_tree_")
    }
    joint = {
        candidate.candidate_id: candidate for candidate in candidates if candidate.candidate_id.startswith("capacity_ordered_tree_joint_")
    }
    coordinated = {
        candidate.candidate_id: candidate
        for candidate in candidates
        if candidate.candidate_id.startswith("capacity_coordinated_tree_joint_")
    }

    assert resident
    assert joint
    assert coordinated
    for candidate in resident.values():
        options = dict(candidate.scheduler_config_options)
        assert options["direct_leaf_acc"] is True
        assert options["ready_time_tree"] is True
        assert options["level_bucket_tree"] is True
        assert options["joint_schedule"] is True
        assert options["joint_hbm_async_receive_pipeline"] is True
        assert options["level0_queue_order"] == "long_first"
    for candidate in joint.values():
        options = dict(candidate.scheduler_config_options)
        assert options["joint_schedule"] is True
        assert options["joint_hbm_async_receive_pipeline"] is True
        assert options["ordered_interval_tree"] is True
    for candidate in coordinated.values():
        options = dict(candidate.scheduler_config_options)
        assert options["ordered_interval_tree"] is True
        assert options["ordered_tree_min_height"] is True
        assert options["balance_tree_leaf_pair_skew"] is True
        assert options["joint_hbm_spill_blocked_cluster_push"] is True

    result = df.select_scheduler_auto_policy(
        program,
        topology=topology,
        range_lengths={"kv": lengths},
        block_size=64,
        task_extents=(2,),
        target_capabilities=make_target(),
        policy_records=False,
        scratch_backed_slots=True,
    )
    evaluations = {item.candidate.candidate_id: item for item in result.decision.candidates}
    assert any(evaluations[candidate_id].legal for candidate_id in resident)
    assert any(evaluations[candidate_id].legal for candidate_id in joint)
    assert any(evaluations[candidate_id].legal for candidate_id in coordinated)


def test_scheduler_auto_policy_compile_artifact_and_cache_fingerprint():
    common = dict(
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": (256, 128)},
        block_size=64,
        task_extents=(2,),
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        target_override=make_target(),
        scheduler_policy="auto",
        reduce_strategy="auto",
    )
    first = df.compile(
        make_replay_program(),
        scheduler_policy_records=False,
        **common,
    )
    repeated = df.compile(
        make_replay_program(),
        scheduler_policy_records=False,
        **common,
    )
    decisions = first.decision_artifact().to_dict()["scheduler"]

    assert first.compile_config.fingerprint == repeated.compile_config.fingerprint
    assert decisions["policy"] == first.plan.scheduler_policy
    assert decisions["reduce_strategy"] == first.plan.reduce_strategy
    assert decisions["auto_policy"]["feature_schema_version"] == 1
    assert decisions["auto_policy"]["record"]["status"] == "disabled"
    assert decisions["auto_policy"]["candidates"]
    assert decisions["auto_policy"]["selection_reason"].startswith("cost_model_minimum:")
    assert first.compile_config.options_dict()["scheduler_auto_policy"] == (decisions["auto_policy"])

    changed_model = df.DataflowSchedulerConfig(
        cost_model=replace(
            df.DataflowCostModelConfig(),
            version="test.calibrated.v2",
            hbm_comm_us=32.0,
        )
    )
    changed = df.compile(
        make_replay_program(),
        scheduler_policy_records=False,
        scheduler_config=changed_model,
        **common,
    )
    assert changed.compile_config.fingerprint != first.compile_config.fingerprint
    assert changed.decision_artifact().to_dict()["scheduler"]["auto_policy"]["cost_model_version"] == "test.calibrated.v2"


def test_scheduler_auto_policy_is_reproducible_across_processes(tmp_path):
    script = """
import json
import tilelang.dataflow as df
from testing.python.dataflow.test_dataflow_scheduler_config import make_replay_program
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug

compiled = df.compile(
    make_replay_program(),
    topology=df.GPUTopology(sm_count=4, cluster_size=2),
    range_lengths={"kv": (320, 128, 576)},
    block_size=64,
    task_extents=(3,),
    scheduler_policy="auto",
    reduce_strategy="auto",
    scheduler_policy_records=False,
    mode="debug",
    _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    target_override=df.TargetCapabilitySnapshot.for_cuda(
        (9, 0), max_dynamic_shared_memory=232448
    ),
)
print(json.dumps({
    "compile_config_fingerprint": compiled.compile_config.fingerprint,
    "plan_fingerprint": df.instruction_plan_fingerprint(compiled.plan),
    "auto_policy": compiled.decision_artifact().to_dict()["scheduler"]["auto_policy"],
}, sort_keys=True))
"""
    results = []
    for index, hash_seed in enumerate(("1", "314159")):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        environment["TILELANG_CACHE_DIR"] = str(tmp_path / f"cache-{index}")
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


def test_scheduler_config_with_options_is_immutable_and_typed():
    base = df.DataflowSchedulerConfig.from_options(level0_queue_order="task")
    changed = base.with_options(
        level0_queue_order="replay",
        chunk_swap_search=True,
    )

    assert base.get("level0_queue_order") == "task"
    assert not base.contains("chunk_swap_search")
    assert changed.get("level0_queue_order") == "replay"
    assert changed.get("chunk_swap_search") is True
    with pytest.raises(ValueError, match="must be a bool"):
        base.with_options(chunk_swap_search="perhaps")
