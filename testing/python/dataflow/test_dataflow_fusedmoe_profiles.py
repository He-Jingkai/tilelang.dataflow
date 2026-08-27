"""Automatic routed planning and reproducible historical execution controls."""

import pytest
from dataclasses import replace
import inspect
from types import SimpleNamespace

import tilelang.dataflow as df
from examples.dataflow.fusedmoe import benchmark
from testing.python.dataflow.test_dataflow_execution_planning import make_request, make_target


@pytest.mark.parametrize("workload", benchmark.PUBLISHED_H100_WORKLOADS)
def test_published_moe_profile_keeps_original_tiles(workload):
    args = benchmark.parse_args_from_list(
        [
            "--published-h100-workload",
            workload,
            "--dataflow-walltime-dir",
            "unused",
        ]
    )
    profile = benchmark.PUBLISHED_H100_WORKLOADS[workload]
    assert args.token_num == [profile["token_num"]]
    assert (args.block_dhidden, args.block_dexpert) == (128, 128)
    assert (args.dataflow_cold_block_dhidden, args.dataflow_cold_block_dexpert) == (128, 256)
    assert args.dataflow_hot_cluster_size is None
    assert args.dataflow_cold_cluster_size == 2
    assert args.dataflow_hot_map1_handler_dexpert is None
    assert args.dataflow_map2_weight_stages == profile.get("map2_weight_stages")
    assert args.dataflow_map2_weight_max_outstanding == 2
    assert (args.dataflow_cold_map2_weight_stages, args.dataflow_cold_map2_weight_max_outstanding) == (2, 2)
    assert (args.warmups, args.repeats, args.dataflow_walltime_repeats) == (5, 30, 30)
    assert args.dataflow_execution_policy == "manual"


@pytest.mark.parametrize("workload", benchmark.PUBLISHED_H100_WORKLOADS)
def test_generic_moe_workload_selects_shape_not_optimized_profile(workload):
    args = benchmark.parse_args_from_list(["--h100-workload", workload, "--dataflow-walltime-dir", "unused"])
    assert args.dataflow_execution_policy == "auto"
    assert args.dataflow_hot_cluster_size is None
    assert (args.block_dhidden, args.block_dexpert) == (128, 128)
    assert args.verify_output
    assert (args.warmups, args.repeats, args.dataflow_walltime_repeats) == (5, 30, 30)
    counts = benchmark.scale_expert_heat(benchmark.EXPERT_HEAT, total_assignments=args.token_num[0] * args.top_k)
    assert (
        benchmark.automatic_dataflow_token_tile(counts, d_hidden=args.d_hidden)
        == benchmark.PUBLISHED_H100_WORKLOADS[workload]["block_token"]
    )


@pytest.mark.parametrize("hidden,expert,assignments", [(7168, 2048, 256), (6144, 1536, 192), (8192, 2048, 224)])
def test_generic_moe_joint_schedule_uses_routing_and_target(hidden, expert, assignments):
    counts = benchmark.scale_expert_heat(benchmark.EXPERT_HEAT, total_assignments=assignments)
    kwargs = dict(
        d_hidden=hidden,
        d_expert=expert,
        hot_sm_count=112,
        cold_sm_count=20,
        cold_expert_count=None,
        target_capabilities=make_target(sm_count=132),
    )
    plans = benchmark.select_dataflow_small_batch_execution(counts, **kwargs)
    assert plans is not None
    assert all(plan.selected_candidate.topology.cluster_size == 4 for plan in plans)
    assert all(plan.selected_candidate.stages[0].tile_k == 256 for plan in plans)
    assert all(plan.selected_evaluation.score[3] == 1 for plan in plans)
    # Relabeling experts changes neither matrix choices nor joint topology.
    shuffled = benchmark.select_dataflow_small_batch_execution(tuple(reversed(counts)), **kwargs)
    assert [plan.selected_candidate for plan in shuffled] == [plan.selected_candidate for plan in plans]


@pytest.mark.parametrize("target", [make_target(sm_count=132, shared_memory=196_608), make_target((8, 0), sm_count=132)])
def test_generic_moe_small_batch_candidate_falls_back_on_target_limits(target):
    counts = benchmark.scale_expert_heat(benchmark.EXPERT_HEAT, total_assignments=256)
    assert (
        benchmark.select_dataflow_small_batch_execution(
            counts,
            d_hidden=7168,
            d_expert=2048,
            hot_sm_count=112,
            cold_sm_count=20,
            cold_expert_count=None,
            target_capabilities=target,
        )
        is None
    )


def test_generic_moe_default_and_manual_controls():
    assert benchmark.parse_args_from_list([]).dataflow_execution_policy == "auto"
    assert inspect.signature(benchmark.profile_dataflow_dual_routed_moe).parameters["execution_policy"].default == "manual"
    for flag in (["--block-dhidden", "128"], ["--dataflow-hot-cluster-size=4"], ["--no-dataflow-fuse-map2-hidden-tiles"]):
        assert benchmark.parse_args_from_list(flag).dataflow_execution_policy == "manual"
        with pytest.raises(SystemExit):
            benchmark.parse_args_from_list(["--dataflow-execution-policy", "auto", *flag])
    with pytest.raises(SystemExit):
        benchmark.parse_args_from_list(["--h100-workload", "deepseek-128"])


def test_generic_moe_unknown_driver_cluster_limit_is_not_single_rank():
    target = replace(make_target(sm_count=132), max_cluster_size=None)
    counts = benchmark.scale_expert_heat(benchmark.EXPERT_HEAT, total_assignments=256)
    plans = benchmark.select_dataflow_small_batch_execution(
        counts,
        d_hidden=7168,
        d_expert=2048,
        hot_sm_count=112,
        cold_sm_count=20,
        cold_expert_count=None,
        target_capabilities=target,
    )
    assert plans is not None
    assert all(plan.selected_candidate.topology.cluster_size == 4 for plan in plans)


def test_generic_moe_cold_count_constraint_prunes_only_incompatible_topologies():
    counts = benchmark.scale_expert_heat(benchmark.EXPERT_HEAT, total_assignments=256)
    plans = benchmark.select_dataflow_small_batch_execution(
        counts,
        d_hidden=7168,
        d_expert=2048,
        hot_sm_count=112,
        cold_sm_count=20,
        cold_expert_count=12,
        target_capabilities=make_target(sm_count=132),
    )
    assert plans is not None
    assert plans[1].selected_candidate.topology.cluster_size == 1


def test_generic_moe_128_keeps_the_verified_execution_contract():
    # Pin the measured execution contract without depending on local benchmark
    # result archives, which are intentionally not part of the source tree.
    common_stage = {
        "compute_threads": 128,
        "consumer_threads": 128,
        "loop_stages": 1,
        "gemm_family": "warp_group_small_m",
        "transfer_family": "tma",
        "eviction_policy": "evict_first",
        "wait_depth": 0,
        "common_pipeline": True,
        "fused_transfers": False,
        "full_partition_pipeline": False,
    }
    expected_candidates = [
        {
            "candidate_id": candidate_id,
            "topology": {"sm_count": sm_count, "cluster_size": 4},
            "stages": [
                {
                    **common_stage,
                    "tile_m": token_tile,
                    "tile_n": 128,
                    "tile_k": 256,
                    "handler_extent": 512,
                    "pipeline_stages": 3,
                    "max_outstanding": first_outstanding,
                    "input_distribution": "unicast",
                    "split_producers": True,
                },
                {
                    **common_stage,
                    "tile_m": token_tile,
                    "tile_n": 256,
                    "tile_k": 128,
                    "handler_extent": 1792,
                    "pipeline_stages": 4,
                    "max_outstanding": 4,
                    "input_distribution": "cooperative",
                    "split_producers": False,
                },
            ],
            "transport_family": "all_gather",
            "handoff_stages": 0,
        }
        for candidate_id, sm_count, token_tile, first_outstanding in (
            ("execution.de359091a409f509", 112, 32, 2),
            ("execution.1232d6f4abe51257", 20, 16, 3),
        )
    ]
    counts = benchmark.scale_expert_heat(benchmark.EXPERT_HEAT, total_assignments=256)
    plans = benchmark.select_dataflow_small_batch_execution(
        counts,
        d_hidden=7168,
        d_expert=2048,
        hot_sm_count=112,
        cold_sm_count=20,
        cold_expert_count=None,
        target_capabilities=make_target(sm_count=132),
    )
    assert [plan.selected_candidate.to_dict() for plan in plans] == expected_candidates
    assert benchmark.NONREGRESSION_H100_P50_US == {
        "deepseek-128": 459.552,
        "deepseek-256": 496.128,
        "deepseek-512": 522.016,
        "deepseek-1024": 573.504,
        "qwen-128": 76.976,
        "qwen-256": 79.712,
        "qwen-512": 78.368,
        "qwen-1024": 84.064,
    }


@pytest.mark.parametrize("workload", [name for name in benchmark.PUBLISHED_H100_WORKLOADS if name != "deepseek-128"])
def test_generic_moe_other_published_shapes_stay_on_existing_candidate_path(workload):
    shape = benchmark.PUBLISHED_H100_WORKLOADS[workload]
    counts = benchmark.scale_expert_heat(benchmark.EXPERT_HEAT, total_assignments=shape["token_num"] * 2)
    assert (
        benchmark.select_dataflow_small_batch_execution(
            counts,
            d_hidden=shape["d_hidden"],
            d_expert=shape["d_expert"],
            hot_sm_count=112,
            cold_sm_count=20,
            cold_expert_count=None,
            target_capabilities=make_target(sm_count=132),
        )
        is None
    )


@pytest.mark.parametrize("workload", benchmark.PUBLISHED_H100_WORKLOADS)
def test_generic_moe_performance_guard_is_enforced_at_every_token_count(workload):
    reference = benchmark.NONREGRESSION_H100_P50_US[workload]
    tolerance = max(1.0, reference * 0.005)
    assert benchmark.h100_latency_guard(workload, reference, automatic=True)["passes"]
    assert benchmark.h100_latency_guard(workload, reference + tolerance, automatic=True)["passes"]
    assert not benchmark.h100_latency_guard(workload, reference + tolerance + 0.001, automatic=True)["passes"]
    historical = benchmark.PUBLISHED_H100_P50_US[workload]
    assert benchmark.h100_latency_guard(workload, historical * 1.03, automatic=False)["passes"]


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), 0, -1])
def test_generic_moe_performance_guard_rejects_invalid_measurements(latency):
    with pytest.raises(ValueError, match="finite and positive"):
        benchmark.h100_latency_guard("deepseek-128", latency, automatic=True)


@pytest.mark.parametrize(
    "flag,tokens",
    [("--h100-workload", tokens) for tokens in (128, 256, 512, 1024)] + [("--optimized-h100-workload", 128)],
)
def test_generic_moe_cli_fails_on_performance_regression(monkeypatch, flag, tokens):
    import tilelang

    workload = f"deepseek-{tokens}"
    args = benchmark.parse_args_from_list([flag, workload, "--dataflow-walltime-dir", "unused"])
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark.os, "environ", {})
    monkeypatch.setattr(tilelang, "disable_cache", lambda: None)
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="test", multi_processor_count=132, major=9, minor=0),
    )
    monkeypatch.setattr(benchmark, "print_result", lambda _: None)
    monkeypatch.setattr(
        benchmark,
        "profile_dataflow_dual_routed_moe",
        lambda *a, **kw: {"dual_walltime_p50_us": benchmark.NONREGRESSION_H100_P50_US[workload] + 10.0},
    )
    with pytest.raises(RuntimeError, match="regressed"):
        benchmark.main()


@pytest.mark.parametrize("cold", [False, True])
def test_optimized_moe_profile_has_legal_linked_tiles_and_on_chip_transport(cold):
    args = benchmark.parse_args_from_list(
        [
            "--optimized-h100-workload",
            "deepseek-128",
            "--dataflow-walltime-dir",
            "unused",
        ]
    )
    assert (args.token_num, args.top_k, args.d_hidden, args.d_expert) == ([128], 2, 7168, 2048)
    assert not args.no_dataflow_fp8 and not args.no_dataflow_tma_weights
    assert (args.warmups, args.repeats, args.dataflow_walltime_repeats) == (5, 30, 30)
    assert args.dataflow_hot_cluster_size == args.dataflow_cold_cluster_size == 4
    assert (args.dataflow_hot_sm_count, args.dataflow_cold_sm_count) == (112, 20)
    assert args.dataflow_hot_map1_handler_dexpert == 512

    token_tile = args.dataflow_cold_block_token if cold else args.dataflow_block_token
    tile_k = args.dataflow_cold_block_dhidden if cold else args.block_dhidden
    tile_n = args.dataflow_cold_block_dexpert if cold else args.block_dexpert
    weight_stages = args.dataflow_cold_map2_weight_stages if cold else args.dataflow_map2_weight_stages
    outstanding = args.dataflow_cold_map2_weight_max_outstanding if cold else args.dataflow_map2_weight_max_outstanding
    assert (tile_k, tile_n, weight_stages, outstanding) == (256, 128, 4, 4)
    override = benchmark.dataflow_execution_override(
        d_hidden=args.d_hidden,
        d_expert=args.d_expert,
        block_token=token_tile,
        block_dhidden=tile_k,
        block_dexpert=tile_n,
        map2_block_dhidden=256,
        map1_handler_dexpert=512,
        map2_handler_dhidden=1792,
        sm_count=args.dataflow_cold_sm_count if cold else args.dataflow_hot_sm_count,
        cluster_size=4,
        threads=128,
        use_tma_weights=True,
        use_small_token_wgmma=True,
        map1_input_mode="unicast",
        map1_gate_stages=3 if cold else 2,
        map2_weight_stages=weight_stages,
        map2_weight_max_outstanding=outstanding,
        reshared_policy=args.dataflow_reshared_policy,
    )
    plan = df.plan_execution(
        make_request(task_extent=256, first_output=2048, second_output=7168),
        override=override,
        target_capabilities=make_target(sm_count=132),
    )
    assert plan.selected_evaluation.legal
    candidate = plan.selected_candidate
    assert candidate.transport_family == df.DATAFLOW_TRANSPORT_ALL_GATHER
    assert candidate.stages[0].tile_n == candidate.stages[1].tile_k == 128
    assert candidate.stages[0].tile_k == 256
    assert all(stage.common_pipeline for stage in candidate.stages)
    assert candidate.handoff_stages == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["--optimized-h100-workload", "deepseek-128"],
        ["--published-h100-workload", "deepseek-128"],
        ["--optimized-h100-workload", "deepseek-128", "--published-h100-workload", "deepseek-128", "--dataflow-walltime-dir", "unused"],
    ],
)
def test_moe_profile_rejects_ambiguous_or_uninstrumented_protocol(argv):
    with pytest.raises(SystemExit) as exc:
        benchmark.parse_args_from_list(argv)
    assert exc.value.code == 2
