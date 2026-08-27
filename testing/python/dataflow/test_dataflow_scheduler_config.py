from __future__ import annotations

from dataclasses import replace
import json

import pytest

import tilelang.language as T
import tilelang.dataflow as df


@T.dataflow_intermediate
class ReplayIntermediate:
    value: T.float32


@T.dataflow.iter(range=("range_begin", "range_end"))
def replay_iter(task: T.int32, Input) -> ReplayIntermediate:
    raise AssertionError("Dataflow iter body should not execute during scheduling")


@T.dataflow.reduce(associative=True)
def replay_reduce(left: ReplayIntermediate, right: ReplayIntermediate) -> ReplayIntermediate:
    raise AssertionError("Dataflow reduce body should not execute during scheduling")


@T.dataflow.finalize
def replay_finalize(item: ReplayIntermediate, task: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during scheduling")


def make_replay_program():
    return (
        T.dataflow_program(task_domain=("task",), dynamic_ranges={"kv": "lengths"})
        .partial(replay_iter(Input="Input"), task_args=("task",), range_axis="kv")
        .reduce(replay_reduce())
        .finalize(replay_finalize(Output="Output"))
    )


def schedule_replay_program(*, config: df.DataflowSchedulerConfig):
    return df.schedule(
        make_replay_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": (256, 128)},
        block_size=64,
        task_extents=(2,),
        scheduler_policy=df.DataflowSchedulerPolicy.CLUSTER_LOCAL,
        reduce_strategy=df.DataflowReduceStrategy.STREAMING_TREE,
        scheduler_config=config,
    )


def test_scheduler_option_registry_is_typed_complete_and_unique():
    specs = df.DATAFLOW_SCHEDULER_OPTION_SPECS

    assert specs
    assert len({spec.option_name for spec in specs}) == len(specs)
    assert len({spec.implementation_id for spec in specs}) == len(specs)
    assert all("DATAFLOW_" not in spec.option_name for spec in specs)
    assert {spec.category for spec in specs} == set(df.DataflowSchedulerOptionCategory)
    assert {spec.value_type for spec in specs} == set(df.DataflowSchedulerOptionType)
    registered = {spec.implementation_id: spec for spec in df.dataflow_implementation_registry().specs}
    assert all(spec.implementation_id in registered for spec in specs)
    assert all(
        item.owner == "SCH"
        and item.benchmark_evidence
        and (
            item.state is df.DataflowImplementationState.STABLE
            or (
                item.state is df.DataflowImplementationState.EXPERIMENTAL
                and item.default_enabled is False
                and item.removal_date is not None
            )
        )
        for item in (registered[spec.implementation_id] for spec in specs)
    )


def test_scheduler_policy_registry_declares_supported_reduce_strategies():
    specs = {spec.policy: spec for spec in df.DATAFLOW_SCHEDULER_POLICY_SPECS}

    assert set(specs) == set(df.DataflowSchedulerPolicy)
    assert specs[df.DataflowSchedulerPolicy.ROUND_ROBIN].reduce_strategies == (df.DataflowReduceStrategy.ALL_AT_ONCE,)
    assert set(specs[df.DataflowSchedulerPolicy.CLUSTER_LOCAL].reduce_strategies) == set(df.DataflowReduceStrategy)

    with pytest.raises(ValueError, match="supports reduce strategies"):
        df.schedule(
            make_replay_program(),
            topology=df.GPUTopology(sm_count=2, cluster_size=2),
            range_lengths={"kv": (64, 64)},
            block_size=64,
            task_extents=(2,),
            scheduler_policy=df.DataflowSchedulerPolicy.ROUND_ROBIN,
            reduce_strategy=df.DataflowReduceStrategy.STREAMING,
        )
    with pytest.raises(ValueError, match="must be one of"):
        df.schedule(
            make_replay_program(),
            topology=df.GPUTopology(sm_count=2, cluster_size=2),
            range_lengths={"kv": (64, 64)},
            block_size=64,
            task_extents=(2,),
            scheduler_policy="mla_cluster_local",
        )


def test_scheduler_config_captures_an_immutable_typed_snapshot():
    options = {
        "direct_leaf_acc": "1",
        "balanced_dag_beam": "12",
        "critical_path_suffix_blocks": "8,4,8",
        "cluster_comm_cost_us": "0.75",
    }
    config = df.DataflowSchedulerConfig.from_options(**options)
    options["direct_leaf_acc"] = "0"
    options["cluster_comm_cost_us"] = "9.0"

    assert config.get("direct_leaf_acc") is True
    assert config.get("balanced_dag_beam") == 12
    assert config.get("critical_path_suffix_blocks") == (4, 8)
    assert config.cost_model.cluster_comm_us == pytest.approx(0.75)
    assert df.DataflowSchedulerConfig.from_dict(config.to_dict()) == config


def test_scheduler_config_canonicalizes_option_insertion_order():
    first = df.DataflowSchedulerConfig.from_options(
        ordered_tree_min_height=True,
        balance_tree_leaf_pair_skew=True,
        joint_hbm_spill_blocked_cluster_push=True,
    )
    second = df.DataflowSchedulerConfig.from_options(
        joint_hbm_spill_blocked_cluster_push=True,
        balance_tree_leaf_pair_skew=True,
        ordered_tree_min_height=True,
    )

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_scheduler_config_rejects_unknown_or_invalid_options():
    with pytest.raises(TypeError, match="Unknown Dataflow scheduler option"):
        df.DataflowSchedulerConfig.from_options(not_a_scheduler_option=True)
    with pytest.raises(ValueError, match="must be a bool"):
        df.DataflowSchedulerConfig.from_options(direct_leaf_acc="sometimes")
    with pytest.raises(ValueError, match="finite non-negative"):
        df.DataflowCostModelConfig(iter_base_us=-1.0)


def test_schedule_binds_explicit_config_and_cost_model_into_fingerprint():
    default_config = df.DataflowSchedulerConfig()
    calibrated_config = df.DataflowSchedulerConfig(cost_model=replace(default_config.cost_model, version="lab.v2", iter_per_block_us=3.1))

    default_plan = schedule_replay_program(config=default_config)
    calibrated_plan = schedule_replay_program(config=calibrated_config)

    assert default_plan.scheduler_config is default_config
    assert calibrated_plan.scheduler_config is calibrated_config
    assert df.instruction_plan_fingerprint(default_plan) != df.instruction_plan_fingerprint(calibrated_plan)


def test_schedule_replay_round_trips_through_json_and_detects_drift():
    program = make_replay_program()
    config = df.DataflowSchedulerConfig.from_options(
        direct_leaf_acc=True,
        level0_queue_order="long_first",
    )
    plan = df.schedule(
        program,
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": (256, 128)},
        block_size=64,
        task_extents=(2,),
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        scheduler_config=config,
    )
    replay = df.record_schedule_replay(
        program,
        plan,
        range_lengths={"kv": (256, 128)},
    )
    restored = df.DataflowScheduleReplay.from_dict(json.loads(json.dumps(replay.to_dict())))

    replayed_plan = df.replay_schedule(program, restored)
    assert df.instruction_plan_fingerprint(replayed_plan) == replay.expected_plan_fingerprint

    drifted = replace(restored, expected_plan_fingerprint="0" * 64)
    with pytest.raises(RuntimeError, match="produced a different structured plan"):
        df.replay_schedule(program, drifted)


def test_semantic_config_is_serializable_and_strictly_parsed():
    config = df.DataflowSemanticConfig.from_dict(
        {
            "direct_slot_seed_reduce": "true",
            "skip_finalize_post_sync": "0",
        }
    )

    assert config.direct_slot_seed_reduce is True
    assert config.skip_finalize_post_sync is False
    assert df.DataflowSemanticConfig.from_dict(config.to_dict()) == config
    legacy = df.DataflowSemanticConfig.from_dict(
        {
            "schema_version": 2,
            "direct_slot_seed_reduce": False,
            "skip_finalize_post_sync": False,
            "precision": {"schema_version": 1, "mode": "strict"},
        }
    )
    assert legacy.schema_version == df.DATAFLOW_SEMANTIC_CONFIG_SCHEMA_VERSION
    assert legacy.fast_math is False
    with pytest.raises(ValueError, match="must be a bool"):
        df.DataflowSemanticConfig.from_dict({"skip_finalize_post_sync": "maybe"})
    with pytest.raises(ValueError, match="schema 2 cannot encode fast_math"):
        df.DataflowSemanticConfig.from_dict({"schema_version": 2, "fast_math": True})
