from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pytest

import tilelang.dataflow as df
from tilelang.dataflow.die_sm_probe import DieSmGroup, DieSmGroups, classify_die_groups
from tilelang.dataflow.gpc_sm_probe import (
    ClusterSmSample,
    GpcSmGroup,
    GpuGpcSmGroups,
    KernelSmRecord,
    infer_gpc_sm_groups_from_cluster_samples,
    records_to_cluster_samples,
    solve_gpc_sizes_from_active_clusters,
)
from tilelang.dataflow.hardware_topology import GpuHardwareTopology, main, render_gpu_topology


def make_die_samples(anchor_fast_group, *, sm_count=8, base_fast=100, base_slow=400):
    samples = []
    fast = set(anchor_fast_group)
    for smid in range(sm_count):
        base = base_fast if smid in fast else base_slow
        samples.extend((smid, base + jitter) for jitter in (-2, 0, 2))
    return samples


def die_groups() -> DieSmGroups:
    return DieSmGroups(
        dies=(
            DieSmGroup(die_id=0, sm_ids=(0, 1, 2, 3)),
            DieSmGroup(die_id=1, sm_ids=(4, 5, 6, 7)),
        ),
        median_gap_cycles=123.5,
        anchor_count=3,
        sm_count=8,
        device_id=0,
        device_name="Fake GPU",
    )


def gpc_groups() -> GpuGpcSmGroups:
    return GpuGpcSmGroups(
        device_id=0,
        sm_count=8,
        max_cluster_size=4,
        active_clusters_by_size=((1, 8), (2, 4), (3, 2), (4, 2)),
        inferred_gpc_sizes=(2, 2, 4),
        gpcs=(
            GpcSmGroup(gpc_id=0, size=4, sm_ids=(4, 5, 6, 7)),
            GpcSmGroup(gpc_id=1, size=2, sm_ids=(0, 1)),
            GpcSmGroup(gpc_id=2, size=2, sm_ids=(2, 3)),
        ),
        cluster_sample_count=12,
        kernel_record_count=48,
        device_name="Fake GPU",
        shared_mem_bytes=131072,
        thread_per_block=1024,
        cuda_arch="native",
    )


def test_gpc_solver_and_cluster_sample_inference_are_available_under_dataflow():
    active_clusters = {
        1: 148,
        2: 74,
        3: 45,
        4: 33,
        5: 26,
        6: 22,
        7: 15,
        8: 15,
        9: 15,
        10: 11,
        11: 7,
        12: 7,
        13: 7,
        14: 7,
        15: 7,
        16: 7,
    }

    assert solve_gpc_sizes_from_active_clusters(active_clusters, sm_count=148) == (
        2,
        2,
        2,
        10,
        18,
        18,
        18,
        18,
        20,
        20,
        20,
    )

    records = (
        KernelSmRecord(0, 2, 0, 0, 0, 11),
        KernelSmRecord(0, 2, 0, 1, 1, 12),
        KernelSmRecord(0, 2, 1, 0, 2, 21),
        KernelSmRecord(0, 2, 1, 1, 3, 22),
    )
    assert records_to_cluster_samples(records) == (
        ClusterSmSample(2, 0, 0, (11, 12)),
        ClusterSmSample(2, 0, 1, (21, 22)),
    )

    samples = [
        ClusterSmSample(cluster_size=2, launch_round=0, cluster_rank=0, sm_ids=(0, 1)),
        ClusterSmSample(cluster_size=2, launch_round=0, cluster_rank=1, sm_ids=(2, 3)),
        ClusterSmSample(cluster_size=2, launch_round=1, cluster_rank=0, sm_ids=(3, 4)),
        ClusterSmSample(cluster_size=2, launch_round=2, cluster_rank=0, sm_ids=(4, 5)),
        ClusterSmSample(cluster_size=2, launch_round=0, cluster_rank=2, sm_ids=(6, 7)),
        ClusterSmSample(cluster_size=2, launch_round=1, cluster_rank=1, sm_ids=(7, 8)),
        ClusterSmSample(cluster_size=2, launch_round=2, cluster_rank=1, sm_ids=(8, 9)),
    ]
    assert infer_gpc_sm_groups_from_cluster_samples(
        samples,
        expected_gpc_sizes=(2, 4, 4),
        sm_count=10,
    ) == ((2, 3, 4, 5), (6, 7, 8, 9), (0, 1))


def test_die_classifier_and_combined_topology_render_are_available_under_dataflow():
    groups = classify_die_groups(
        [
            make_die_samples({0, 1, 2, 3}),
            make_die_samples({4, 5, 6, 7}),
            make_die_samples({0, 1, 2, 3}),
        ],
        sm_count=8,
    )
    assert groups.as_lists() == [[0, 1, 2, 3], [4, 5, 6, 7]]

    topology = GpuHardwareTopology(die_groups=die_groups(), gpc_groups=gpc_groups())
    payload = topology.to_dict()
    assert payload["device_name"] == "Fake GPU"
    assert payload["gpc_distribution_by_die"][0]["gpc_count"] == 2
    assert payload["gpc_distribution_by_die"][1]["gpcs"][0]["sm_ids"] == [4, 5, 6, 7]

    rendered = render_gpu_topology(die_groups(), gpc_groups())
    assert "Device 0 (Fake GPU): 8 SMs" in rendered
    assert "GPC count: 3" in rendered
    assert "Die -> GPC distribution:" in rendered


def test_hardware_topology_cli_keeps_gpc_only_probe_independent_of_die_probe():
    with (
        patch("tilelang.dataflow.hardware_topology.get_gpu_die_sm_groups") as die_probe,
        patch(
            "tilelang.dataflow.hardware_topology.get_gpu_gpc_sm_groups",
            return_value=gpc_groups(),
        ) as gpc_probe,
        redirect_stdout(StringIO()) as stdout,
    ):
        exit_code = main(["--gpc-only", "--device", "0", "--json"])

    assert exit_code == 0
    die_probe.assert_not_called()
    gpc_probe.assert_called_once()
    assert gpc_probe.call_args.kwargs["cuda_arch"] == "native"
    payload = json.loads(stdout.getvalue())
    assert payload["gpc_count"] == 3


def test_dataflow_programming_topology_remains_logical_cuda_cluster_only():
    topology = df.GPUTopology(sm_count=8, cluster_size=2)

    assert topology.cluster_count == 4
    assert topology.cluster_id(4) == 2
    assert topology.cluster_rank(5) == 1
    assert topology.same_cluster(0, 1)
    assert not topology.same_cluster(0, 4)

    with pytest.raises(TypeError, match="cluster_groups"):
        df.GPUTopology(sm_count=4, cluster_size=2, cluster_groups=((0, 1),))


if __name__ == "__main__":
    unittest.main()
