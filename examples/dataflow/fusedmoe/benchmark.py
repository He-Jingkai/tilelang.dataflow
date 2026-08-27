"""Profile routed MoE kernels with group sizes derived from expert heat."""

from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import statistics
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Literal, Mapping

import torch


EXPERT_HEAT: list[int] = [
    1180466,
    858509,
    723223,
    680851,
    625081,
    550310,
    485283,
    476994,
    433631,
    378858,
    346996,
    336053,
    324312,
    293756,
    290547,
    269086,
    263293,
    232033,
    212633,
    197280,
    194094,
    169803,
    166134,
    145841,
    134740,
    114794,
    89319,
    65466,
    30561,
    25534,
    21173,
    1236,
]

DEFAULT_TOKEN_NUMS = (128,)
DEFAULT_BACKEND = "dataflow-dual"
DEFAULT_TOP_K = 2
DEFAULT_D_HIDDEN = 7168
DEFAULT_D_EXPERT = 2048
DEFAULT_BLOCK_DHIDDEN = 128
DEFAULT_BLOCK_DEXPERT = 128
DEFAULT_DATAFLOW_BLOCK_TOKEN = 64
DEFAULT_DATAFLOW_MAP2_BLOCK_DHIDDEN = 256
DEFAULT_DATAFLOW_SM_COUNT = 132
DEFAULT_DATAFLOW_CLUSTER_SIZE = 2
DEFAULT_DATAFLOW_HOT_SM_COUNT = 112
DEFAULT_DATAFLOW_HOT_CLUSTER_SIZE = 16
DEFAULT_DATAFLOW_COLD_SM_COUNT = 20
DEFAULT_DATAFLOW_COLD_CLUSTER_SIZE = 2
DEFAULT_DATAFLOW_COLD_EXPERT_COUNT = None
DEFAULT_DATAFLOW_COLD_BLOCK_TOKEN = 16
DEFAULT_DATAFLOW_DUAL_TIMING_BATCH_SIZE = 10
DEFAULT_DATAFLOW_COLD_BLOCK_DHIDDEN = 128
DEFAULT_DATAFLOW_COLD_MAP2_BLOCK_DHIDDEN = 256
DEFAULT_DATAFLOW_COLD_BLOCK_DEXPERT = 256
DEFAULT_DATAFLOW_COLD_FUSE_MAP1_EXPERT_SHARDS = True
DEFAULT_DATAFLOW_HOT_MAP1_CONSUMER_THREADS = None
DEFAULT_DATAFLOW_MAP1_INPUT_MODE = "unicast"
DEFAULT_DATAFLOW_COLD_MAP1_INPUT_MODE = None
DEFAULT_DATAFLOW_MAP2_MAP1_LOOKAHEAD_STAGES = 2
DEFAULT_DATAFLOW_MAP2_WEIGHT_STAGES = None
DEFAULT_DATAFLOW_MAP2_WEIGHT_MAX_OUTSTANDING = 2
DEFAULT_DATAFLOW_COLD_MAP2_WEIGHT_STAGES = 2
DEFAULT_DATAFLOW_COLD_MAP2_WEIGHT_MAX_OUTSTANDING = 2
DEFAULT_DATAFLOW_FUSE_MAP2_HIDDEN_TILES = True
DATAFLOW_MAP1_PIPELINE_MIN_K_STEPS = 8
DEFAULT_WARMUPS = 5
DEFAULT_REPEATS = 20
DATAFLOW_ROUTED_TASK_FIXED_US = 135.250
DATAFLOW_ROUTED_TASK_ROW_US = 4.913
DATAFLOW_ROUTED_TASK_ROW_CAP = 8


Backend = Literal["dataflow", "dataflow-dual"]

PUBLISHED_H100_WORKLOADS: dict[str, dict[str, Any]] = {
    "deepseek-128": {"d_hidden": 7168, "d_expert": 2048, "token_num": 128, "block_token": 32, "wait_depth": 0},
    "deepseek-256": {"d_hidden": 7168, "d_expert": 2048, "token_num": 256, "block_token": 32, "wait_depth": 0},
    "deepseek-512": {
        "d_hidden": 7168,
        "d_expert": 2048,
        "token_num": 512,
        "block_token": 64,
        "fuse_map2_hidden_tiles": True,
    },
    "deepseek-1024": {
        "d_hidden": 7168,
        "d_expert": 2048,
        "token_num": 1024,
        "block_token": 128,
        "reshared_policy": "cluster_shared_pull_ring",
        "cold_reshared_policy": "cluster_shared_all_gather",
        "map2_weight_stages": 2,
        "map2_map1_lookahead_stages": 0,
        "cold_map1_gate_stages": 2,
    },
    "qwen-128": {"d_hidden": 2048, "d_expert": 768, "token_num": 128, "block_token": 32, "wait_depth": 0},
    "qwen-256": {"d_hidden": 2048, "d_expert": 768, "token_num": 256, "block_token": 32, "wait_depth": 0},
    "qwen-512": {"d_hidden": 2048, "d_expert": 768, "token_num": 512, "block_token": 32, "wait_depth": 0},
    "qwen-1024": {"d_hidden": 2048, "d_expert": 768, "token_num": 1024, "block_token": 64},
}
PUBLISHED_H100_P50_US = {
    "deepseek-128": 483.808,
    "deepseek-256": 482.464,
    "deepseek-512": 501.664,
    "deepseek-1024": 558.992,
    "qwen-128": 77.616,
    "qwen-256": 79.696,
    "qwen-512": 76.224,
    "qwen-1024": 81.424,
}
PUBLISHED_PER_POINT_GATE_PERCENT = 3.0


@dataclass(frozen=True)
class RoutedGroupMetadata:
    group_sizes: torch.Tensor
    group_offsets: torch.Tensor
    group_padded_offsets: torch.Tensor
    group_idx_for_bx: torch.Tensor
    group_blocks: int


@dataclass(frozen=True)
class RoutedGroupBlock:
    group_block: int
    expert_id: int
    actual_rows: int


@dataclass(frozen=True)
class DualKernelBlockSplit:
    hot_blocks: tuple[RoutedGroupBlock, ...]
    cold_blocks: tuple[RoutedGroupBlock, ...]


@dataclass(frozen=True)
class DualKernelSchedule:
    split: DualKernelBlockSplit
    hot_task_costs_us: tuple[float, ...]
    cold_task_costs_us: tuple[float, ...]
    hot_cluster_assignment: tuple[int, ...]
    cold_cluster_assignment: tuple[int, ...]
    cold_expert_count: int
    selection_score: tuple[float, ...]
    candidate_scores: tuple[tuple[int, tuple[float, ...]], ...]


@dataclass(frozen=True)
class DataflowHotTopology:
    cluster_size: int
    map2_handler_dhidden: int | None
    map2_map1_lookahead_stages: int
    selection_score: tuple[float, ...]
    candidate_scores: tuple[tuple[int, int, tuple[float, ...]], ...]


@dataclass(frozen=True)
class StageGraphAssignmentSummary:
    task_cluster_assignment: tuple[int, ...]
    cluster_task_counts: tuple[int, ...]
    cluster_task_costs_us: tuple[float, ...]


@dataclass(frozen=True)
class DualConcurrentWalltimeResult:
    detail_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    solo_summary_rows: tuple[dict[str, Any], ...]
    report: str
    detail_csv: Path | None = None
    summary_csv: Path | None = None


def dataflow_torch_dtypes(*, use_fp8: bool) -> tuple[torch.dtype, torch.dtype]:
    if use_fp8:
        return torch.float8_e4m3fn, torch.float8_e4m3fn
    return torch.float16, torch.float16


def routed_moe_reference(
    input_tensor: torch.Tensor,
    routed_expert_gate: torch.Tensor,
    routed_expert_up: torch.Tensor,
    routed_expert_down: torch.Tensor,
    routed_expert_weights: torch.Tensor,
    counts: list[int],
    group_offsets: torch.Tensor,
    *,
    intermediate_dtype: torch.dtype,
) -> torch.Tensor:
    output = torch.zeros(
        (input_tensor.shape[0], routed_expert_down.shape[1]),
        device=input_tensor.device,
        dtype=torch.float32,
    )
    offsets = group_offsets.tolist()
    for expert_idx, group_size in enumerate(counts):
        if group_size <= 0:
            continue
        start = offsets[expert_idx]
        end = start + group_size
        expert_input = input_tensor[start:end].float()
        gate = expert_input @ routed_expert_gate[expert_idx].float().T
        up = expert_input @ routed_expert_up[expert_idx].float().T
        intermediate = (gate * torch.sigmoid(gate) * up).to(intermediate_dtype).float()
        hidden = intermediate @ routed_expert_down[expert_idx].float().T
        output[start:end] = hidden * routed_expert_weights[start:end].float().view(-1, 1)
    return output


def dataflow_map1_codegen_options(
    *,
    use_fp8: bool,
    use_tma_weights: bool,
    cluster_size: int,
    input_mode: str = DEFAULT_DATAFLOW_MAP1_INPUT_MODE,
) -> dict[str, bool]:
    if input_mode not in ("multicast", "unicast", "cooperative"):
        raise ValueError(f"map1 input mode must be 'multicast', 'unicast', or 'cooperative', got {input_mode!r}")
    use_tma_input = bool(input_mode != "cooperative" and use_fp8 and use_tma_weights and cluster_size <= 16)
    return {
        "use_tma_map1_input_multicast": (use_tma_input and input_mode == "multicast"),
        "use_tma_map1_input_unicast": (use_tma_input and input_mode == "unicast"),
        "split_map1_tma_producers": use_tma_input,
    }


def dataflow_execution_override(
    *,
    d_hidden: int,
    d_expert: int,
    block_token: int,
    block_dhidden: int,
    map2_block_dhidden: int,
    block_dexpert: int,
    sm_count: int,
    cluster_size: int,
    use_tma_weights: bool,
    reshared_policy: str,
    map2_handler_dhidden: int | None = None,
    fuse_map2_hidden_tiles: bool = False,
    map1_handler_dexpert: int | None = None,
    threads: int = 256,
    map1_consumer_threads: int | None = None,
    num_stages: int = 1,
    weight_eviction_policy: str = "evict_first",
    use_small_token_wgmma: bool = False,
    map1_gate_stages: int = 2,
    map1_wgmma_wait_depth: int = 0,
    map2_weight_stages: int | None = None,
    map2_weight_max_outstanding: int = 2,
    map1_input_mode: str = "cooperative",
    split_map1_tma_producers: bool | None = None,
    map2_map1_lookahead_stages: int = 0,
):
    """Build the single typed advanced override accepted by the operator."""

    import tilelang.dataflow as df

    expert_shard = d_expert // cluster_size
    hidden_shard = d_hidden // cluster_size
    map1_tile_n = math.gcd(expert_shard, block_dexpert)
    map1_tile_k = math.gcd(d_hidden, block_dhidden)
    map2_tile_n = min(hidden_shard, map2_block_dhidden)
    map1_handler = map1_tile_n if map1_handler_dexpert is None else map1_handler_dexpert
    if fuse_map2_hidden_tiles:
        map2_handler = hidden_shard
    else:
        map2_handler = map2_tile_n if map2_handler_dhidden is None else min(hidden_shard, map2_handler_dhidden)
    if map2_weight_stages is None:
        map2_weight_stages = 4 if reshared_policy == "cluster_shared_pull_ring" else 2
    if map1_input_mode not in {"cooperative", "unicast", "multicast"}:
        raise ValueError(f"unsupported typed map1 input mode {map1_input_mode!r}")
    distributed_input = use_tma_weights and map1_input_mode != "cooperative"
    if map1_consumer_threads is None:
        map1_consumer_threads = 128 if distributed_input else threads
    if split_map1_tma_producers is None:
        split_map1_tma_producers = distributed_input
    gemm_family = (
        df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M
        if use_small_token_wgmma
        else df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP
        if use_tma_weights
        else df.DATAFLOW_EXECUTION_GEMM_PORTABLE
    )
    transfer_family = df.DATAFLOW_EXECUTION_TRANSFER_TMA if use_tma_weights else df.DATAFLOW_EXECUTION_TRANSFER_SYNCHRONOUS
    transport_family = {
        "hbm_all_gather": df.DATAFLOW_TRANSPORT_HBM,
        "cluster_shared_all_gather": df.DATAFLOW_TRANSPORT_ALL_GATHER,
        "cluster_shared_pull_ring": df.DATAFLOW_TRANSPORT_STREAMED,
    }[reshared_policy]
    common = dict(
        compute_threads=threads,
        loop_stages=num_stages,
        gemm_family=gemm_family,
        transfer_family=transfer_family,
        eviction_policy=weight_eviction_policy,
    )
    return df.DataflowExecutionOverride(
        topology=df.GPUTopology(sm_count=sm_count, cluster_size=cluster_size),
        task_tile_extent=block_token,
        stages=(
            df.DataflowExecutionStageOverride(
                tile_n=map1_tile_n,
                tile_k=map1_tile_k,
                handler_extent=map1_handler,
                consumer_threads=map1_consumer_threads,
                # Independent TMA producer partitions can profit from one
                # additional stage on the smallest transfer.  The generic
                # pipeline planner assigns that version only when the shared
                # budget permits it, producing heterogeneous rings rather
                # than inflating every transfer.
                pipeline_stages=max(
                    map1_gate_stages,
                    (3 if split_map1_tma_producers and (use_small_token_wgmma or map1_wgmma_wait_depth) else map1_gate_stages),
                ),
                max_outstanding=map1_gate_stages,
                input_distribution=(map1_input_mode if distributed_input else df.DATAFLOW_EXECUTION_INPUT_COOPERATIVE),
                split_producers=bool(split_map1_tma_producers),
                wait_depth=map1_wgmma_wait_depth,
                **common,
            ),
            df.DataflowExecutionStageOverride(
                tile_n=map2_tile_n,
                tile_k=map1_tile_n,
                handler_extent=map2_handler,
                consumer_threads=threads,
                pipeline_stages=map2_weight_stages,
                max_outstanding=map2_weight_max_outstanding,
                input_distribution=df.DATAFLOW_EXECUTION_INPUT_COOPERATIVE,
                split_producers=False,
                wait_depth=0,
                **common,
            ),
        ),
        transport_family=transport_family,
        handoff_stages=map2_map1_lookahead_stages,
    )


def dataflow_dual_tma_modes(
    *,
    use_tma_weights: bool,
    cold_block_token: int,
    use_small_token_wgmma: bool = False,
) -> tuple[bool, bool]:
    if cold_block_token <= 0:
        raise ValueError(f"cold_block_token must be positive, got {cold_block_token}")
    hot_use_tma = bool(use_tma_weights)
    cold_use_tma = hot_use_tma and (cold_block_token >= 64 or use_small_token_wgmma)
    return hot_use_tma, cold_use_tma


def dual_source_paths(path: str | Path) -> tuple[Path, Path]:
    source_path = Path(path).expanduser()
    suffix = source_path.suffix
    stem = source_path.stem if suffix else source_path.name
    parent = source_path.parent
    return (
        parent / f"{stem}_hot{suffix}",
        parent / f"{stem}_cold{suffix}",
    )


def scale_expert_heat(expert_heat: Iterable[int], *, total_assignments: int) -> list[int]:
    """Scale trace expert heat into integer per-expert routed-token counts."""

    heat = [int(value) for value in expert_heat]
    if not heat:
        raise ValueError("expert_heat must not be empty")
    if total_assignments <= 0:
        raise ValueError(f"total_assignments must be positive, got {total_assignments}")
    if any(value < 0 for value in heat):
        raise ValueError("expert_heat values must be non-negative")
    heat_sum = sum(heat)
    if heat_sum <= 0:
        raise ValueError("expert_heat must contain at least one positive value")

    scaled = [value * total_assignments / heat_sum for value in heat]
    counts = [int(math.floor(value)) for value in scaled]
    remaining = total_assignments - sum(counts)
    order = sorted(
        range(len(scaled)),
        key=lambda idx: scaled[idx] - counts[idx],
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts


def build_group_metadata(
    group_sizes: Iterable[int],
    *,
    block_token: int,
    device: str | torch.device = "cuda",
) -> RoutedGroupMetadata:
    """Build grouped GEMM metadata used by both TileLang and Dataflow routed kernels."""

    counts = [int(value) for value in group_sizes]
    if not counts:
        raise ValueError("group_sizes must not be empty")
    if block_token <= 0:
        raise ValueError(f"block_token must be positive, got {block_token}")
    if any(value < 0 for value in counts):
        raise ValueError("group_sizes values must be non-negative")

    group_sizes_host = torch.tensor(counts, dtype=torch.int32)
    group_offsets_host = (torch.cumsum(group_sizes_host, dim=0) - group_sizes_host).to(torch.int32)
    group_padded_offsets = [0 for _ in counts]
    for idx in range(1, len(counts)):
        previous = counts[idx - 1]
        group_padded_offsets[idx] = group_padded_offsets[idx - 1] + math.ceil((previous + 1) / block_token) * block_token

    group_sum = sum(counts)
    group_blocks = math.ceil(group_sum / block_token) + len(counts)
    group_idx_for_bx = [0 for _ in range(group_blocks)]
    for bx in range(group_blocks):
        m_start_padded = bx * block_token
        for expert_id, padded_offset in enumerate(group_padded_offsets):
            if m_start_padded >= padded_offset:
                group_idx_for_bx[bx] = expert_id

    return RoutedGroupMetadata(
        group_sizes=group_sizes_host.to(device),
        group_offsets=group_offsets_host.to(device),
        group_padded_offsets=torch.tensor(group_padded_offsets, dtype=torch.int32, device=device),
        group_idx_for_bx=torch.tensor(group_idx_for_bx, dtype=torch.int32, device=device),
        group_blocks=group_blocks,
    )


def build_group_block_plan(
    group_sizes: Iterable[int],
    *,
    group_padded_offsets: Iterable[int],
    group_idx_for_bx: Iterable[int],
    block_token: int,
) -> tuple[RoutedGroupBlock, ...]:
    counts = [int(value) for value in group_sizes]
    padded_offsets = [int(value) for value in group_padded_offsets]
    group_indices = [int(value) for value in group_idx_for_bx]
    if len(padded_offsets) != len(counts):
        raise ValueError(f"group_padded_offsets must contain one entry per expert: got {len(padded_offsets)}, expected {len(counts)}")
    if block_token <= 0:
        raise ValueError(f"block_token must be positive, got {block_token}")

    blocks: list[RoutedGroupBlock] = []
    for group_block, expert_id in enumerate(group_indices):
        if expert_id < 0 or expert_id >= len(counts):
            raise ValueError(f"group_idx_for_bx[{group_block}]={expert_id} is outside {len(counts)} experts")
        m_start_padded = group_block * block_token
        row_in_group = m_start_padded - padded_offsets[expert_id]
        actual_rows = max(0, min(block_token, counts[expert_id] - row_in_group))
        blocks.append(
            RoutedGroupBlock(
                group_block=group_block,
                expert_id=expert_id,
                actual_rows=actual_rows,
            )
        )
    return tuple(blocks)


def build_active_group_block_plan(
    group_sizes: Iterable[int],
    *,
    metadata: RoutedGroupMetadata,
    block_token: int,
) -> tuple[RoutedGroupBlock, ...]:
    blocks = build_group_block_plan(
        group_sizes,
        group_padded_offsets=metadata.group_padded_offsets.tolist(),
        group_idx_for_bx=metadata.group_idx_for_bx.tolist(),
        block_token=block_token,
    )
    return tuple(block for block in blocks if block.actual_rows > 0)


def estimate_dataflow_routed_task_costs(
    blocks: Iterable[RoutedGroupBlock],
) -> tuple[float, ...]:
    active = tuple(blocks)
    if not active:
        raise ValueError("Dataflow routed task cost model requires at least one active block")
    if any(block.actual_rows <= 0 for block in active):
        raise ValueError("Dataflow routed task cost model only accepts blocks with actual_rows > 0")
    return tuple(
        round(
            DATAFLOW_ROUTED_TASK_FIXED_US + DATAFLOW_ROUTED_TASK_ROW_US * min(block.actual_rows, DATAFLOW_ROUTED_TASK_ROW_CAP),
            3,
        )
        for block in active
    )


def balance_stage_graph_task_costs(
    task_costs: Iterable[float],
    *,
    cluster_count: int,
) -> tuple[int, ...]:
    costs = tuple(float(cost) for cost in task_costs)
    if cluster_count <= 0:
        raise ValueError(f"cluster_count must be positive, got {cluster_count}")
    if not costs:
        raise ValueError("stage-graph task balancing requires at least one task cost")
    if any(not math.isfinite(cost) or cost < 0.0 for cost in costs):
        raise ValueError("stage-graph task costs must contain finite non-negative values")

    cluster_loads = [0.0 for _ in range(cluster_count)]
    assignment = [-1 for _ in costs]
    for task_id in sorted(range(len(costs)), key=lambda idx: (-costs[idx], idx)):
        cluster_id = min(range(cluster_count), key=lambda idx: (cluster_loads[idx], idx))
        assignment[task_id] = cluster_id
        cluster_loads[cluster_id] += costs[task_id]

    def load_score(loads: Iterable[float]) -> tuple[float, ...]:
        return tuple(sorted((round(load, 9) for load in loads), reverse=True))

    for _ in range(min(len(costs), 2 * cluster_count)):
        best_score = load_score(cluster_loads)
        best_swap: tuple[int, int, list[float]] | None = None
        for left in range(len(costs)):
            left_cluster = assignment[left]
            for right in range(left + 1, len(costs)):
                right_cluster = assignment[right]
                if left_cluster == right_cluster:
                    continue
                candidate_loads = cluster_loads.copy()
                candidate_loads[left_cluster] += costs[right] - costs[left]
                candidate_loads[right_cluster] += costs[left] - costs[right]
                candidate_score = load_score(candidate_loads)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_swap = (left, right, candidate_loads)
        if best_swap is None:
            break
        left, right, cluster_loads = best_swap
        assignment[left], assignment[right] = assignment[right], assignment[left]

    return tuple(assignment)


def summarize_stage_graph_assignment(
    plan: Any,
    *,
    task_costs: Iterable[float] | None = None,
) -> StageGraphAssignmentSummary:
    task_clusters: dict[int, int] = {}
    for inst in plan.instructions:
        if inst.attrs.get("stage_id") != 0 or inst.attrs.get("cluster_rank") != 0:
            continue
        task_id = int(inst.task_id)
        cluster_id = int(plan.topology.cluster_id(inst.sm_id))
        previous = task_clusters.setdefault(task_id, cluster_id)
        if previous != cluster_id:
            raise ValueError(f"stage-graph task {task_id} is assigned to multiple clusters: {previous} and {cluster_id}")

    task_ids = tuple(sorted(task_clusters))
    if task_ids != tuple(range(len(task_ids))):
        raise ValueError(f"stage-graph task ids must be compact, got {task_ids}")
    assignment = tuple(task_clusters[task_id] for task_id in task_ids)
    cluster_task_counts = [0 for _ in range(plan.topology.cluster_count)]
    for cluster_id in assignment:
        cluster_task_counts[cluster_id] += 1
    cluster_task_costs_us: tuple[float, ...] = ()
    if task_costs is not None:
        costs = tuple(float(cost) for cost in task_costs)
        if len(costs) != len(assignment):
            raise ValueError(f"task_costs must contain one value per stage-graph task: got {len(costs)}, expected {len(assignment)}")
        if any(not math.isfinite(cost) or cost < 0.0 for cost in costs):
            raise ValueError("task_costs must contain finite non-negative values")
        cluster_costs = [0.0 for _ in range(plan.topology.cluster_count)]
        for cluster_id, cost in zip(assignment, costs):
            cluster_costs[cluster_id] += cost
        cluster_task_costs_us = tuple(round(cost, 6) for cost in cluster_costs)
    return StageGraphAssignmentSummary(
        task_cluster_assignment=assignment,
        cluster_task_counts=tuple(cluster_task_counts),
        cluster_task_costs_us=cluster_task_costs_us,
    )


def partition_dual_kernel_blocks(
    blocks: Iterable[RoutedGroupBlock],
    *,
    hot_cluster_count: int,
    cold_cluster_count: int,
    cold_expert_count: int | None = None,
) -> DualKernelBlockSplit:
    if hot_cluster_count <= 0:
        raise ValueError(f"hot_cluster_count must be positive, got {hot_cluster_count}")
    if cold_cluster_count <= 0:
        raise ValueError(f"cold_cluster_count must be positive, got {cold_cluster_count}")
    if cold_expert_count is not None and cold_expert_count <= 0:
        raise ValueError(f"cold_expert_count must be positive when provided, got {cold_expert_count}")
    if cold_expert_count is not None and cold_expert_count > cold_cluster_count:
        raise ValueError(
            "cold_expert_count must not exceed cold_cluster_count for the one-wave cold plan: "
            f"got {cold_expert_count}, cold_cluster_count={cold_cluster_count}"
        )
    active = tuple(block for block in blocks if block.actual_rows > 0)
    rows_by_expert: dict[int, int] = {}
    for block in active:
        rows_by_expert[block.expert_id] = rows_by_expert.get(block.expert_id, 0) + block.actual_rows
    cold_target = min(
        len(rows_by_expert),
        cold_cluster_count if cold_expert_count is None else cold_expert_count,
    )
    cold_expert_ids = {
        expert_id
        for expert_id, _ in sorted(
            rows_by_expert.items(),
            key=lambda item: (item[1], item[0]),
        )[:cold_target]
    }
    hot = tuple(block for block in active if block.expert_id not in cold_expert_ids)
    cold = tuple(block for block in active if block.expert_id in cold_expert_ids)
    return DualKernelBlockSplit(hot_blocks=hot, cold_blocks=cold)


def build_dual_kernel_schedule(
    blocks: Iterable[RoutedGroupBlock],
    *,
    cold_blocks: Iterable[RoutedGroupBlock] | None = None,
    hot_cluster_count: int,
    cold_cluster_count: int,
    cold_expert_count: int | None,
) -> DualKernelSchedule:
    hot_plan = tuple(block for block in blocks if block.actual_rows > 0)
    cold_plan = hot_plan if cold_blocks is None else tuple(block for block in cold_blocks if block.actual_rows > 0)
    active_experts = {block.expert_id for block in hot_plan}
    if len(active_experts) < 2:
        raise ValueError("dual-kernel scheduling requires at least two active experts")

    max_cold_experts = min(cold_cluster_count, len(active_experts) - 1)
    candidate_counts = tuple(range(1, max_cold_experts + 1)) if cold_expert_count is None else (cold_expert_count,)
    candidates: list[DualKernelSchedule] = []

    for candidate_count in candidate_counts:
        hot_granularity_split = partition_dual_kernel_blocks(
            hot_plan,
            hot_cluster_count=hot_cluster_count,
            cold_cluster_count=cold_cluster_count,
            cold_expert_count=candidate_count,
        )
        cold_expert_ids = {block.expert_id for block in hot_granularity_split.cold_blocks}
        candidate_cold_blocks = tuple(block for block in cold_plan if block.expert_id in cold_expert_ids)
        split = DualKernelBlockSplit(
            hot_blocks=hot_granularity_split.hot_blocks,
            cold_blocks=candidate_cold_blocks,
        )
        if not split.hot_blocks or not split.cold_blocks:
            raise ValueError("dual-kernel scheduling requires non-empty hot and cold block sets")

        hot_task_costs = estimate_dataflow_routed_task_costs(split.hot_blocks)
        cold_task_costs = estimate_dataflow_routed_task_costs(split.cold_blocks)
        hot_assignment = balance_stage_graph_task_costs(
            hot_task_costs,
            cluster_count=hot_cluster_count,
        )
        cold_assignment = balance_stage_graph_task_costs(
            cold_task_costs,
            cluster_count=cold_cluster_count,
        )

        hot_task_counts = [0 for _ in range(hot_cluster_count)]
        cold_task_counts = [0 for _ in range(cold_cluster_count)]
        hot_cluster_costs = [0.0 for _ in range(hot_cluster_count)]
        cold_cluster_costs = [0.0 for _ in range(cold_cluster_count)]
        for cluster_id, cost in zip(hot_assignment, hot_task_costs):
            hot_task_counts[cluster_id] += 1
            hot_cluster_costs[cluster_id] += cost
        for cluster_id, cost in zip(cold_assignment, cold_task_costs):
            cold_task_counts[cluster_id] += 1
            cold_cluster_costs[cluster_id] += cost

        # Task waves are robust across differently specialized hot/cold handlers;
        # the calibrated row-cost model breaks ties within the same wave count.
        hot_waves = max(hot_task_counts)
        cold_waves = max(cold_task_counts)
        hot_peak_cost = max(hot_cluster_costs)
        cold_peak_cost = max(cold_cluster_costs)
        selection_score = (
            float(max(hot_waves, cold_waves)),
            float(hot_waves + cold_waves),
            round(max(hot_peak_cost, cold_peak_cost), 6),
            round(hot_peak_cost + cold_peak_cost, 6),
            float(candidate_count),
        )
        candidates.append(
            DualKernelSchedule(
                split=split,
                hot_task_costs_us=hot_task_costs,
                cold_task_costs_us=cold_task_costs,
                hot_cluster_assignment=hot_assignment,
                cold_cluster_assignment=cold_assignment,
                cold_expert_count=candidate_count,
                selection_score=selection_score,
                candidate_scores=(),
            )
        )

    candidate_scores = tuple((candidate.cold_expert_count, candidate.selection_score) for candidate in candidates)
    selected = min(candidates, key=lambda candidate: candidate.selection_score)
    return replace(selected, candidate_scores=candidate_scores)


def dataflow_schedule_task_waves(
    assignment: tuple[int, ...],
    *,
    cluster_count: int,
) -> int:
    task_counts = [0 for _ in range(cluster_count)]
    for cluster_id in assignment:
        task_counts[cluster_id] += 1
    return max(task_counts, default=0)


def resolve_dataflow_cold_map1_fusion(
    *,
    d_expert: int,
    cold_cluster_size: int,
    cold_block_dexpert: int,
    requested: bool | None,
) -> bool:
    if requested is not None:
        return requested
    return (d_expert // cold_cluster_size) % cold_block_dexpert == 0


def resolve_dataflow_cold_map1_gate_stages(
    *,
    use_small_token_wgmma: bool,
    fuse_map1_expert_shards: bool,
    requested: int | None = None,
) -> int:
    if requested is not None:
        if requested not in (2, 3):
            raise ValueError(f"cold map1 gate stages must be 2 or 3, got {requested}")
        return requested
    # Small-token handlers need the deeper ring to overlap their split input
    # and gate/up TMA producers. Expert-shard fusion does not change that need.
    return 3 if use_small_token_wgmma else 2


def resolve_dataflow_hot_map1_wgmma_wait_depth(
    *,
    d_hidden: int,
    d_expert: int,
    block_dhidden: int,
    block_dexpert: int,
    block_token: int,
    cluster_size: int,
    map1_handler_dexpert: int | None,
    map1_consumer_threads: int | None,
    use_fp8: bool,
    use_tma_weights: bool,
    map1_codegen_options: dict[str, bool],
    map2_map1_lookahead_stages: int,
    requested: int | None,
) -> int:
    if requested is not None:
        if requested not in (0, 1):
            raise ValueError(f"hot map1 WGMMA wait depth must be 0, 1, or auto, got {requested}")
        return requested
    if cluster_size <= 0 or d_expert % cluster_size:
        return 0

    expert_shard = d_expert // cluster_size
    map_block_dhidden = math.gcd(d_hidden, block_dhidden)
    map_block_dexpert = math.gcd(expert_shard, block_dexpert)
    resolved_handler_dexpert = map_block_dexpert if map1_handler_dexpert is None else map1_handler_dexpert
    handler_is_one_tile = resolved_handler_dexpert == map_block_dexpert
    regular_tma_input = bool(
        use_fp8
        and use_tma_weights
        and block_token >= 64
        and map1_codegen_options["split_map1_tma_producers"]
        and (map1_codegen_options["use_tma_map1_input_multicast"] or map1_codegen_options["use_tma_map1_input_unicast"])
    )
    prospective_consumer_threads = 256 if map1_consumer_threads is None else map1_consumer_threads
    consumer_warp_groups = prospective_consumer_threads // 128
    gate_up_columns = 2 * map_block_dexpert
    consumer_layout_is_eligible = bool(
        prospective_consumer_threads == 256
        and gate_up_columns % consumer_warp_groups == 0
        and (gate_up_columns // consumer_warp_groups) % 128 == 0
    )
    k_steps = d_hidden // map_block_dhidden
    return int(
        regular_tma_input
        and handler_is_one_tile
        and consumer_layout_is_eligible
        and map2_map1_lookahead_stages == 0
        and k_steps >= DATAFLOW_MAP1_PIPELINE_MIN_K_STEPS
    )


def select_dataflow_hot_topology(
    hot_plan_blocks: Iterable[RoutedGroupBlock],
    *,
    cold_plan_blocks: Iterable[RoutedGroupBlock],
    d_hidden: int,
    d_expert: int,
    block_token: int,
    block_dhidden: int,
    map2_block_dhidden: int,
    block_dexpert: int,
    hot_sm_count: int,
    cold_cluster_count: int,
    cold_expert_count: int | None,
    hot_cluster_size: int | None,
    hot_map2_handler_dhidden: int | None,
    map2_map1_lookahead_stages: int | None,
    fuse_map2_hidden_tiles: bool,
) -> tuple[DataflowHotTopology, DualKernelSchedule]:
    hot_plan = tuple(block for block in hot_plan_blocks if block.actual_rows > 0)
    cold_plan = tuple(block for block in cold_plan_blocks if block.actual_rows > 0)
    if not hot_plan:
        raise ValueError("hot topology selection requires active routed blocks")

    candidate_cluster_sizes = (
        (hot_cluster_size,)
        if hot_cluster_size is not None
        else tuple(
            cluster_size
            for cluster_size in (16, 8, 4, 2, 1)
            if hot_sm_count % cluster_size == 0 and d_hidden % cluster_size == 0 and d_expert % cluster_size == 0
        )
    )
    if not candidate_cluster_sizes:
        raise ValueError("no legal hot Dataflow cluster size for the requested shape")

    candidates: list[tuple[tuple[float, ...], int, DualKernelSchedule]] = []
    for cluster_size in candidate_cluster_sizes:
        if cluster_size is None or cluster_size <= 0:
            raise ValueError(f"hot_cluster_size must be positive or auto, got {cluster_size}")
        if hot_sm_count % cluster_size != 0:
            raise ValueError(
                f"hot_sm_count must be divisible by hot_cluster_size, got hot_sm_count={hot_sm_count}, hot_cluster_size={cluster_size}"
            )
        if d_hidden % cluster_size or d_expert % cluster_size:
            raise ValueError(
                "d_hidden and d_expert must be divisible by hot_cluster_size, got "
                f"d_hidden={d_hidden}, d_expert={d_expert}, "
                f"hot_cluster_size={cluster_size}"
            )

        hot_cluster_count = hot_sm_count // cluster_size
        schedule = build_dual_kernel_schedule(
            hot_plan,
            cold_blocks=cold_plan,
            hot_cluster_count=hot_cluster_count,
            cold_cluster_count=cold_cluster_count,
            cold_expert_count=cold_expert_count,
        )
        hot_waves = dataflow_schedule_task_waves(
            schedule.hot_cluster_assignment,
            cluster_count=hot_cluster_count,
        )
        expert_shard = d_expert // cluster_size
        map_block_dexpert = math.gcd(expert_shard, block_dexpert)
        active_hot_sms = min(len(schedule.split.hot_blocks), hot_cluster_count) * cluster_size

        # Normalize waves by both cluster fanout and the fraction of the
        # requested expert tile retained after shape legalization. This favors
        # full-SM first waves without selecting very narrow WGMMA N tiles.
        normalized_waves = (hot_waves * block_dexpert) / (cluster_size * map_block_dexpert)
        score = (
            round(normalized_waves, 6),
            float(hot_waves),
            float(-active_hot_sms),
            schedule.selection_score[0],
            schedule.selection_score[2],
            float(cluster_size),
        )
        candidates.append((score, cluster_size, schedule))

    selection_score, selected_cluster_size, selected_schedule = min(
        candidates,
        key=lambda candidate: candidate[0],
    )
    hidden_shard = d_hidden // selected_cluster_size
    map2_tile_dhidden = min(hidden_shard, map2_block_dhidden)
    can_fuse_map2 = bool(fuse_map2_hidden_tiles and block_token >= 64 and math.ceil(hidden_shard / map2_tile_dhidden) == 2)

    resolved_handler_dhidden = hot_map2_handler_dhidden
    if resolved_handler_dhidden is None and not can_fuse_map2:
        aligned_hidden_shard = (hidden_shard // map2_block_dhidden) * map2_block_dhidden
        if aligned_hidden_shard > 0:
            resolved_handler_dhidden = min(
                aligned_hidden_shard,
                7 * map2_block_dhidden,
            )

    map_block_dhidden = math.gcd(d_hidden, block_dhidden)
    expert_shard = d_expert // selected_cluster_size
    map_block_dexpert = math.gcd(expert_shard, block_dexpert)
    expert_chunks = expert_shard // map_block_dexpert
    handler_tiles = 1 if resolved_handler_dhidden is None else resolved_handler_dhidden // map2_block_dhidden
    lookahead_eligible = bool(
        (can_fuse_map2 or handler_tiles == 1)
        and map2_block_dhidden == 2 * map_block_dexpert
        and map_block_dexpert == map_block_dhidden
        and (selected_cluster_size * expert_chunks) % 2 == 0
    )
    if map2_map1_lookahead_stages is None:
        resolved_lookahead_stages = 2 if lookahead_eligible else 0
    else:
        resolved_lookahead_stages = map2_map1_lookahead_stages

    candidate_scores = tuple((cluster_size, schedule.cold_expert_count, score) for score, cluster_size, schedule in candidates)
    topology = DataflowHotTopology(
        cluster_size=selected_cluster_size,
        map2_handler_dhidden=resolved_handler_dhidden,
        map2_map1_lookahead_stages=resolved_lookahead_stages,
        selection_score=selection_score,
        candidate_scores=candidate_scores,
    )
    return topology, selected_schedule


def profile_dataflow_routed_moe(
    token_num: int,
    *,
    d_hidden: int,
    d_expert: int,
    top_k: int,
    block_token: int,
    block_dhidden: int,
    map2_block_dhidden: int,
    map2_handler_dhidden: int | None = None,
    block_dexpert: int,
    sm_count: int,
    cluster_size: int,
    warmups: int,
    repeats: int,
    use_fp8: bool,
    use_tma_weights: bool,
    reshared_policy: str,
    map2_weight_stages: int | None = DEFAULT_DATAFLOW_MAP2_WEIGHT_STAGES,
    map2_weight_max_outstanding: int = DEFAULT_DATAFLOW_MAP2_WEIGHT_MAX_OUTSTANDING,
    fuse_map2_hidden_tiles: bool = DEFAULT_DATAFLOW_FUSE_MAP2_HIDDEN_TILES,
    active_only: bool = False,
    walltime_dir: str | None = None,
    walltime_repeats: int = 1,
    dump_dataflow_source: str | None = None,
    map1_input_mode: str = DEFAULT_DATAFLOW_MAP1_INPUT_MODE,
    map2_map1_lookahead_stages: int | None = DEFAULT_DATAFLOW_MAP2_MAP1_LOOKAHEAD_STAGES,
) -> dict[str, Any]:
    import tilelang.language as T
    import tilelang.dataflow as df

    from examples.dataflow.fusedmoe import example_fusedmoe_dataflow as dataflow_moe

    if map2_weight_stages is None:
        map2_weight_stages = 4 if reshared_policy == "cluster_shared_pull_ring" else 2
    if map2_map1_lookahead_stages is None:
        map2_map1_lookahead_stages = DEFAULT_DATAFLOW_MAP2_MAP1_LOOKAHEAD_STAGES

    counts = scale_expert_heat(EXPERT_HEAT, total_assignments=token_num * top_k)
    group_sum = token_num * top_k
    metadata = build_group_metadata(counts, block_token=block_token, device="cuda")
    active_blocks = build_active_group_block_plan(
        counts,
        metadata=metadata,
        block_token=block_token,
    )
    task_costs = estimate_dataflow_routed_task_costs(active_blocks)
    if not active_only:
        task_costs = None
    cluster_assignment = (
        balance_stage_graph_task_costs(
            task_costs,
            cluster_count=sm_count // cluster_size,
        )
        if task_costs is not None
        else None
    )
    if walltime_repeats <= 0:
        raise ValueError(f"walltime_repeats must be positive, got {walltime_repeats}")
    n_experts = len(EXPERT_HEAT)
    input_torch_dtype, weight_torch_dtype = dataflow_torch_dtypes(use_fp8=use_fp8)
    input_tl_dtype = T.float8_e4m3fn if use_fp8 else T.float16
    weight_tl_dtype = T.float8_e4m3fn if use_fp8 else T.float16
    intermediate_tl_dtype = T.float8_e4m3fn if use_fp8 else T.float16
    map1_codegen_options = dataflow_map1_codegen_options(
        use_fp8=use_fp8,
        use_tma_weights=use_tma_weights,
        cluster_size=cluster_size,
        input_mode=map1_input_mode,
    )
    map2_tile_dhidden = min(d_hidden // cluster_size, map2_block_dhidden)
    fuse_map2_hidden_tiles = bool(
        fuse_map2_hidden_tiles and use_tma_weights and block_token >= 64 and math.ceil((d_hidden // cluster_size) / map2_tile_dhidden) == 2
    )

    input_tensor = torch.empty((group_sum, d_hidden), dtype=input_torch_dtype, device="cuda")
    routed_expert_gate_up = df.mark_tensor_layout(
        torch.empty(
            (n_experts, 2 * d_expert, d_hidden),
            dtype=weight_torch_dtype,
            device="cuda",
        ),
        dataflow_moe.gate_up_weight_layout(n_experts, d_expert, d_hidden),
    )
    routed_expert_down = torch.empty((n_experts, d_hidden, d_expert), dtype=weight_torch_dtype, device="cuda")
    routed_expert_weights = torch.empty((group_sum,), dtype=torch.float16, device="cuda")
    output = torch.empty((group_sum, d_hidden), dtype=torch.float16, device="cuda")

    compiled = dataflow_moe.compile_dataflow_routed_moe(
        d_hidden=d_hidden,
        d_expert=d_expert,
        n_routed_experts=n_experts,
        group_sum=group_sum,
        input_dtype=input_tl_dtype,
        weight_dtype=weight_tl_dtype,
        intermediate_dtype=intermediate_tl_dtype,
        semantic_config=df.DataflowSemanticConfig(fast_math=True),
        execution_override=dataflow_execution_override(
            d_hidden=d_hidden,
            d_expert=d_expert,
            block_token=block_token,
            block_dhidden=block_dhidden,
            map2_block_dhidden=map2_block_dhidden,
            map2_handler_dhidden=map2_handler_dhidden,
            fuse_map2_hidden_tiles=fuse_map2_hidden_tiles,
            block_dexpert=block_dexpert,
            sm_count=sm_count,
            cluster_size=cluster_size,
            use_tma_weights=use_tma_weights,
            reshared_policy=reshared_policy,
            map2_weight_stages=map2_weight_stages,
            map2_weight_max_outstanding=map2_weight_max_outstanding,
            map1_input_mode=(
                "multicast"
                if map1_codegen_options["use_tma_map1_input_multicast"]
                else "unicast"
                if map1_codegen_options["use_tma_map1_input_unicast"]
                else "cooperative"
            ),
            split_map1_tma_producers=map1_codegen_options["split_map1_tma_producers"],
            map2_map1_lookahead_stages=map2_map1_lookahead_stages,
        ),
        active_group_blocks=(tuple(block.group_block for block in active_blocks) if active_only else None),
        stage_graph_task_weights=task_costs,
        stage_graph_cluster_assignment=cluster_assignment,
    )
    assignment = summarize_stage_graph_assignment(compiled.plan, task_costs=task_costs)

    source_path = None
    if dump_dataflow_source is not None:
        source_file = Path(dump_dataflow_source).expanduser()
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(compiled.wrapper_source, encoding="utf-8")
        source_path = str(source_file)

    kwargs = {
        "input": input_tensor,
        "routed_expert_gate_up": routed_expert_gate_up,
        "routed_expert_down": routed_expert_down,
        "routed_expert_weights": routed_expert_weights,
        "group_sizes": metadata.group_sizes,
        "group_offsets": metadata.group_offsets,
        "group_padded_offsets": metadata.group_padded_offsets,
        "group_idx_for_bx": metadata.group_idx_for_bx,
        "output": output,
    }

    torch.cuda.synchronize()
    result = compiled.get_profiler().profile(
        **kwargs,
        metric="pure_kernel_event_ms",
        n_warmup=warmups,
        n_repeat=repeats,
        flush_l2_cache=False,
    )
    torch.cuda.synchronize()

    walltime_detail_csv = None
    walltime_summary_csv = None
    walltime_report = None
    if walltime_dir is not None:
        walltime = compiled.profile_walltime(
            **kwargs,
            repeat=walltime_repeats,
            warmup=0,
            output_dir=walltime_dir,
            prefix=f"dataflow_moe_{sm_count}_{cluster_size}_{'active' if active_only else 'full'}",
        )
        walltime_detail_csv = None if walltime.detail_csv is None else str(walltime.detail_csv)
        walltime_summary_csv = None if walltime.summary_csv is None else str(walltime.summary_csv)
        walltime_report = walltime.report

    samples = list(result.samples_ms)
    row = base_result(
        "dataflow",
        token_num=token_num,
        top_k=top_k,
        block_token=block_token,
        counts=counts,
        metadata=metadata,
        samples=samples,
    )
    row.update(
        {
            "sm_count": sm_count,
            "cluster_size": cluster_size,
            "reshared_policy": reshared_policy,
            "map2_weight_stages": map2_weight_stages,
            "map2_weight_max_outstanding": map2_weight_max_outstanding,
            "fuse_map2_hidden_tiles": fuse_map2_hidden_tiles,
            "map2_block_dhidden": map2_block_dhidden,
            "map2_handler_dhidden": map2_handler_dhidden,
            "active_only": active_only,
            "active_blocks": len(active_blocks),
            "scheduled_blocks": len(assignment.task_cluster_assignment),
            "task_cluster_assignment": assignment.task_cluster_assignment,
            "cluster_task_counts": assignment.cluster_task_counts,
            "task_cost_model": (
                {
                    "fixed_us": DATAFLOW_ROUTED_TASK_FIXED_US,
                    "row_us": DATAFLOW_ROUTED_TASK_ROW_US,
                    "row_cap": DATAFLOW_ROUTED_TASK_ROW_CAP,
                }
                if task_costs is not None
                else None
            ),
            "task_costs_us": () if task_costs is None else task_costs,
            "cluster_task_costs_us": assignment.cluster_task_costs_us,
            "dtype": "input_fp8_weights_fp8_intermediate_fp8" if use_fp8 else "fp16",
            "map1_input_multicast": map1_codegen_options["use_tma_map1_input_multicast"],
            "map1_input_unicast": map1_codegen_options["use_tma_map1_input_unicast"],
            "map1_input_mode": map1_input_mode,
            "map2_map1_lookahead_stages": map2_map1_lookahead_stages,
            "map1_consumer_threads": (
                128 if (map1_codegen_options["use_tma_map1_input_multicast"] or map1_codegen_options["use_tma_map1_input_unicast"]) else 256
            ),
            "instructions": len(compiled.plan.instructions),
            "comms": len(compiled.plan.comms),
            "cluster_send": sum(1 for comm in compiled.plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_SEND),
            "cluster_recv": sum(1 for comm in compiled.plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_RECV),
            "shared_memory_bytes": compiled.launch_package.shared_memory_bytes,
            "shared_slot_bytes": compiled.launch_package.shared_slot_bytes,
            "barrier_count": compiled.launch_package.barrier_count,
            "source_path": source_path,
            "walltime_detail_csv": walltime_detail_csv,
            "walltime_summary_csv": walltime_summary_csv,
            "walltime_report": walltime_report,
            "setup_timings_ms": dict(result.setup_timings_ms),
        }
    )

    del compiled
    del input_tensor, routed_expert_gate_up, routed_expert_down
    del routed_expert_weights, output, metadata
    gc.collect()
    torch.cuda.empty_cache()
    return row


def launch_persistent_executable(executable: Any, stream: Any) -> None:
    from tilelang.dataflow.executor import launch_block_dim, launch_wrapper

    driver = executable._driver
    if driver is None or executable._function is None:
        raise RuntimeError("Dataflow persistent executable must be open before launch")
    launch_wrapper(
        driver,
        executable._function,
        executable.launch_package,
        executable._ptrs,
        launch_block_dim(executable.options),
        executable.topology,
        stream,
        executable.tma_descriptor_specs,
        executable.tma_descriptor_handles,
        synchronize=False,
    )


def walltime_percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile from no walltime samples")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def dual_concurrent_global_span_p50_us(
    result: DualConcurrentWalltimeResult,
) -> float:
    return walltime_percentile(
        (float(row["global_span_us"]) for row in result.summary_rows if row["kernel"] == "hot"),
        50.0,
    )


def moe_walltime_stage(operator_name: str) -> str:
    lowered = operator_name.lower()
    if "map1" in lowered:
        return "map1"
    if "reshared" in lowered or "gather" in lowered:
        return "reshared"
    if "map2" in lowered:
        return "map2"
    if "finalize" in lowered:
        return "finalize"
    return "other"


def group_walltime_rows_by_kernel_sample(
    detail_rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in detail_rows:
        key = (str(row["kernel"]), int(row["sample"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def dual_concurrent_summary_rows(
    detail_rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    grouped = group_walltime_rows_by_kernel_sample(detail_rows)
    summaries: list[dict[str, Any]] = []
    samples = sorted(sample for kernel, sample in grouped if kernel in ("hot", "cold"))
    for sample in sorted(set(samples)):
        by_kernel = {kernel: grouped.get((kernel, sample), []) for kernel in ("hot", "cold")}
        missing = [kernel for kernel, rows in by_kernel.items() if not rows]
        if missing:
            raise RuntimeError(f"concurrent Dataflow walltime sample {sample} has no rows for {missing}")
        sample_rows = by_kernel["hot"] + by_kernel["cold"]

        global_start = min(int(row["start_time_ns"]) for row in sample_rows)
        global_end = max(int(row["end_time_ns"]) for row in sample_rows)
        kernel_bounds = {
            kernel: (
                min(int(row["start_time_ns"]) for row in rows),
                max(int(row["end_time_ns"]) for row in rows),
            )
            for kernel, rows in by_kernel.items()
        }
        hot_start, hot_end = kernel_bounds["hot"]
        cold_start, cold_end = kernel_bounds["cold"]
        overlap_ns = max(0, min(hot_end, cold_end) - max(hot_start, cold_start))
        first_kernel_end = min(hot_end, cold_end)
        tail_ns = global_end - first_kernel_end
        real_smids = {kernel: {int(row["real_smid"]) for row in rows} for kernel, rows in by_kernel.items()}
        real_smid_overlap_count = len(real_smids["hot"] & real_smids["cold"])

        for kernel, rows in by_kernel.items():
            kernel_start, kernel_end = kernel_bounds[kernel]
            per_sm: dict[int, list[dict[str, Any]]] = {}
            for row in rows:
                per_sm.setdefault(int(row["logical_sm"]), []).append(row)
            sm_bounds = [
                (
                    min(int(row["start_time_ns"]) for row in sm_rows),
                    max(int(row["end_time_ns"]) for row in sm_rows),
                )
                for sm_rows in per_sm.values()
            ]
            sm_start_offsets_us = [(start - global_start) / 1000.0 for start, _ in sm_bounds]
            sm_finish_offsets_us = [(end - global_start) / 1000.0 for _, end in sm_bounds]
            sm_active_spans_us = [(end - start) / 1000.0 for start, end in sm_bounds]
            summary: dict[str, Any] = {
                "sample": sample,
                "kernel": kernel,
                "queue_count": len(per_sm),
                "real_sm_count": len(real_smids[kernel]),
                "real_smids": " ".join(str(smid) for smid in sorted(real_smids[kernel])),
                "real_smid_overlap_count": real_smid_overlap_count,
                "global_span_us": (global_end - global_start) / 1000.0,
                "kernel_start_offset_us": (kernel_start - global_start) / 1000.0,
                "kernel_completion_us": (kernel_end - global_start) / 1000.0,
                "kernel_span_us": (kernel_end - kernel_start) / 1000.0,
                "overlap_us": overlap_ns / 1000.0,
                "tail_us": tail_ns / 1000.0,
                "sm_start_min_us": min(sm_start_offsets_us),
                "sm_start_p50_us": walltime_percentile(sm_start_offsets_us, 50.0),
                "sm_start_p95_us": walltime_percentile(sm_start_offsets_us, 95.0),
                "sm_start_max_us": max(sm_start_offsets_us),
                "sm_start_spread_us": max(sm_start_offsets_us) - min(sm_start_offsets_us),
                "sm_finish_min_us": min(sm_finish_offsets_us),
                "sm_finish_p50_us": walltime_percentile(sm_finish_offsets_us, 50.0),
                "sm_finish_p95_us": walltime_percentile(sm_finish_offsets_us, 95.0),
                "sm_finish_max_us": max(sm_finish_offsets_us),
                "sm_finish_spread_us": max(sm_finish_offsets_us) - min(sm_finish_offsets_us),
                "sm_active_p50_us": walltime_percentile(sm_active_spans_us, 50.0),
                "sm_active_p95_us": walltime_percentile(sm_active_spans_us, 95.0),
                "sm_active_max_us": max(sm_active_spans_us),
                "active_sms_at_peer_finish": (
                    sum(int(start <= first_kernel_end < end) for start, end in sm_bounds) if kernel_end > first_kernel_end else 0
                ),
            }
            for stage in ("map1", "reshared", "map2", "finalize", "other"):
                stage_rows = [row for row in rows if moe_walltime_stage(str(row["operator_name"])) == stage]
                if stage_rows:
                    stage_start = min(int(row["start_time_ns"]) for row in stage_rows)
                    stage_end = max(int(row["end_time_ns"]) for row in stage_rows)
                    summary[f"{stage}_start_us"] = (stage_start - global_start) / 1000.0
                    summary[f"{stage}_completion_us"] = (stage_end - global_start) / 1000.0
                    summary[f"{stage}_span_us"] = (stage_end - stage_start) / 1000.0
                    per_sm_stage_total_us = [
                        sum(
                            int(
                                row.get(
                                    "total_ns",
                                    int(row["end_time_ns"]) - int(row["start_time_ns"]),
                                )
                            )
                            for row in sm_rows
                            if moe_walltime_stage(str(row["operator_name"])) == stage
                        )
                        / 1000.0
                        for sm_rows in per_sm.values()
                    ]
                    instruction_us = [
                        int(
                            row.get(
                                "total_ns",
                                int(row["end_time_ns"]) - int(row["start_time_ns"]),
                            )
                        )
                        / 1000.0
                        for row in stage_rows
                    ]
                    summary[f"{stage}_sm_total_p50_us"] = walltime_percentile(per_sm_stage_total_us, 50.0)
                    summary[f"{stage}_sm_total_p95_us"] = walltime_percentile(per_sm_stage_total_us, 95.0)
                    summary[f"{stage}_sm_total_max_us"] = max(per_sm_stage_total_us)
                    summary[f"{stage}_instruction_p50_us"] = walltime_percentile(instruction_us, 50.0)
                    summary[f"{stage}_instruction_p95_us"] = walltime_percentile(instruction_us, 95.0)
                else:
                    summary[f"{stage}_start_us"] = ""
                    summary[f"{stage}_completion_us"] = ""
                    summary[f"{stage}_span_us"] = ""
                    summary[f"{stage}_sm_total_p50_us"] = ""
                    summary[f"{stage}_sm_total_p95_us"] = ""
                    summary[f"{stage}_sm_total_max_us"] = ""
                    summary[f"{stage}_instruction_p50_us"] = ""
                    summary[f"{stage}_instruction_p95_us"] = ""
            summaries.append(summary)
    return tuple(summaries)


def solo_walltime_summary_rows(
    detail_rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    grouped = group_walltime_rows_by_kernel_sample(detail_rows)
    summaries: list[dict[str, Any]] = []
    for kernel in ("hot_solo", "cold_solo"):
        samples = sorted(sample for grouped_kernel, sample in grouped if grouped_kernel == kernel)
        for sample in samples:
            rows = grouped[(kernel, sample)]
            kernel_start = min(int(row["start_time_ns"]) for row in rows)
            kernel_end = max(int(row["end_time_ns"]) for row in rows)
            per_sm: dict[int, list[dict[str, Any]]] = {}
            for row in rows:
                per_sm.setdefault(int(row["logical_sm"]), []).append(row)
            sm_bounds = [
                (
                    min(int(row["start_time_ns"]) for row in sm_rows),
                    max(int(row["end_time_ns"]) for row in sm_rows),
                )
                for sm_rows in per_sm.values()
            ]
            sm_start_offsets_us = [(start - kernel_start) / 1000.0 for start, _ in sm_bounds]
            sm_finish_offsets_us = [(end - kernel_start) / 1000.0 for _, end in sm_bounds]
            sm_active_spans_us = [(end - start) / 1000.0 for start, end in sm_bounds]
            real_smids = {int(row["real_smid"]) for row in rows}
            summary: dict[str, Any] = {
                "sample": sample,
                "kernel": kernel,
                "queue_count": len(per_sm),
                "real_sm_count": len(real_smids),
                "real_smids": " ".join(str(smid) for smid in sorted(real_smids)),
                "real_smid_overlap_count": 0,
                "global_span_us": (kernel_end - kernel_start) / 1000.0,
                "kernel_start_offset_us": 0.0,
                "kernel_completion_us": (kernel_end - kernel_start) / 1000.0,
                "kernel_span_us": (kernel_end - kernel_start) / 1000.0,
                "overlap_us": 0.0,
                "tail_us": 0.0,
                "sm_start_min_us": min(sm_start_offsets_us),
                "sm_start_p50_us": walltime_percentile(sm_start_offsets_us, 50.0),
                "sm_start_p95_us": walltime_percentile(sm_start_offsets_us, 95.0),
                "sm_start_max_us": max(sm_start_offsets_us),
                "sm_start_spread_us": max(sm_start_offsets_us) - min(sm_start_offsets_us),
                "sm_finish_min_us": min(sm_finish_offsets_us),
                "sm_finish_p50_us": walltime_percentile(sm_finish_offsets_us, 50.0),
                "sm_finish_p95_us": walltime_percentile(sm_finish_offsets_us, 95.0),
                "sm_finish_max_us": max(sm_finish_offsets_us),
                "sm_finish_spread_us": max(sm_finish_offsets_us) - min(sm_finish_offsets_us),
                "sm_active_p50_us": walltime_percentile(sm_active_spans_us, 50.0),
                "sm_active_p95_us": walltime_percentile(sm_active_spans_us, 95.0),
                "sm_active_max_us": max(sm_active_spans_us),
                "active_sms_at_peer_finish": 0,
            }
            for stage in ("map1", "reshared", "map2", "finalize", "other"):
                stage_rows = [row for row in rows if moe_walltime_stage(str(row["operator_name"])) == stage]
                if stage_rows:
                    stage_start = min(int(row["start_time_ns"]) for row in stage_rows)
                    stage_end = max(int(row["end_time_ns"]) for row in stage_rows)
                    per_sm_stage_total_us = [
                        sum(
                            int(
                                row.get(
                                    "total_ns",
                                    int(row["end_time_ns"]) - int(row["start_time_ns"]),
                                )
                            )
                            for row in sm_rows
                            if moe_walltime_stage(str(row["operator_name"])) == stage
                        )
                        / 1000.0
                        for sm_rows in per_sm.values()
                    ]
                    instruction_us = [
                        int(
                            row.get(
                                "total_ns",
                                int(row["end_time_ns"]) - int(row["start_time_ns"]),
                            )
                        )
                        / 1000.0
                        for row in stage_rows
                    ]
                    summary[f"{stage}_start_us"] = (stage_start - kernel_start) / 1000.0
                    summary[f"{stage}_completion_us"] = (stage_end - kernel_start) / 1000.0
                    summary[f"{stage}_span_us"] = (stage_end - stage_start) / 1000.0
                    summary[f"{stage}_sm_total_p50_us"] = walltime_percentile(per_sm_stage_total_us, 50.0)
                    summary[f"{stage}_sm_total_p95_us"] = walltime_percentile(per_sm_stage_total_us, 95.0)
                    summary[f"{stage}_sm_total_max_us"] = max(per_sm_stage_total_us)
                    summary[f"{stage}_instruction_p50_us"] = walltime_percentile(instruction_us, 50.0)
                    summary[f"{stage}_instruction_p95_us"] = walltime_percentile(instruction_us, 95.0)
                else:
                    for suffix in (
                        "start_us",
                        "completion_us",
                        "span_us",
                        "sm_total_p50_us",
                        "sm_total_p95_us",
                        "sm_total_max_us",
                        "instruction_p50_us",
                        "instruction_p95_us",
                    ):
                        summary[f"{stage}_{suffix}"] = ""
            summaries.append(summary)
    return tuple(summaries)


def format_dual_concurrent_walltime_report(
    summary_rows: Iterable[dict[str, Any]],
    solo_summary_rows: Iterable[dict[str, Any]] = (),
) -> str:
    summaries = tuple(summary_rows)
    solo_summaries = tuple(solo_summary_rows)
    all_summaries = summaries + solo_summaries
    samples = sorted({int(row["sample"]) for row in summaries})
    lines = [
        "Dataflow dual-kernel concurrent walltime report (%globaltimer)",
        f"samples={len(samples)}",
    ]

    def values_for(kernel: str, field: str) -> list[float]:
        return [float(row[field]) for row in all_summaries if row["kernel"] == kernel and row[field] != ""]

    def stats(values: Iterable[float]) -> str:
        numbers = tuple(float(value) for value in values)
        return (
            f"min/p50/p95/max={min(numbers):.3f}/"
            f"{walltime_percentile(numbers, 50.0):.3f}/"
            f"{walltime_percentile(numbers, 95.0):.3f}/"
            f"{max(numbers):.3f} us"
        )

    hot_rows = [row for row in summaries if row["kernel"] == "hot"]
    lines.append(f"global span: {stats(row['global_span_us'] for row in hot_rows)}")
    lines.append(f"overlap: {stats(row['overlap_us'] for row in hot_rows)}")
    lines.append(f"tail after first kernel: {stats(row['tail_us'] for row in hot_rows)}")
    lines.append(f"physical SM overlap hot/cold: max={max(int(row['real_smid_overlap_count']) for row in hot_rows)}")
    for kernel in ("hot", "cold"):
        solo_kernel = f"{kernel}_solo"
        lines.append(f"{kernel}:")
        lines.append(f"  kernel span: {stats(values_for(kernel, 'kernel_span_us'))}")
        if values_for(solo_kernel, "kernel_span_us"):
            concurrent_span = statistics.median(values_for(kernel, "kernel_span_us"))
            solo_span = statistics.median(values_for(solo_kernel, "kernel_span_us"))
            lines.append(
                f"  solo kernel span p50={solo_span:.3f} us; "
                f"co-resident delta={concurrent_span - solo_span:.3f} us "
                f"({concurrent_span / solo_span:.3f}x)"
            )
        lines.append(f"  placement offset: {stats(values_for(kernel, 'kernel_start_offset_us'))}")
        start_fields = (
            "sm_start_min_us",
            "sm_start_p50_us",
            "sm_start_p95_us",
            "sm_start_max_us",
            "sm_start_spread_us",
        )
        finish_fields = (
            "sm_finish_min_us",
            "sm_finish_p50_us",
            "sm_finish_p95_us",
            "sm_finish_max_us",
            "sm_finish_spread_us",
        )
        lines.append(
            "  SM start median min/p50/p95/max/spread="
            + "/".join(f"{statistics.median(values_for(kernel, field)):.3f}" for field in start_fields)
            + " us"
        )
        lines.append(
            "  SM finish median min/p50/p95/max/spread="
            + "/".join(f"{statistics.median(values_for(kernel, field)):.3f}" for field in finish_fields)
            + " us"
        )
        lines.append(
            "  active SMs at peer finish: "
            f"median={statistics.median(values_for(kernel, 'active_sms_at_peer_finish')):.1f} "
            f"max={max(values_for(kernel, 'active_sms_at_peer_finish')):.0f}"
        )
        for stage in ("map1", "reshared", "map2", "finalize", "other"):
            stage_spans = values_for(kernel, f"{stage}_span_us")
            if stage_spans:
                lines.append(f"  {stage} span: {stats(stage_spans)}")
                concurrent_stage = statistics.median(values_for(kernel, f"{stage}_sm_total_max_us"))
                solo_stage_values = values_for(
                    solo_kernel,
                    f"{stage}_sm_total_max_us",
                )
                if solo_stage_values:
                    solo_stage = statistics.median(solo_stage_values)
                    lines.append(
                        f"    critical-SM total p50 concurrent/solo="
                        f"{concurrent_stage:.3f}/{solo_stage:.3f} us; "
                        f"delta={concurrent_stage - solo_stage:.3f} us "
                        f"({concurrent_stage / solo_stage:.3f}x)"
                    )
    return "\n".join(lines)


def profile_persistent_dual_walltime(
    hot_compiled: Any,
    cold_compiled: Any,
    *,
    hot_kwargs: dict[str, Any],
    cold_kwargs: dict[str, Any],
    warmups: int,
    repeats: int,
    output_dir: str | Path,
) -> DualConcurrentWalltimeResult:
    from tilelang.dataflow import walltime as dataflow_walltime
    from tilelang.dataflow.executor import check_cuda

    if warmups < 0:
        raise ValueError(f"warmups must be non-negative, got {warmups}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    hot_timed = dataflow_walltime.build_persistent_walltime_executable(
        hot_compiled,
        **hot_kwargs,
    )
    cold_timed = dataflow_walltime.build_persistent_walltime_executable(
        cold_compiled,
        **cold_kwargs,
    )
    hot_executable = hot_timed.executable
    cold_executable = cold_timed.executable
    detail_rows: list[dict[str, Any]] = []

    def synchronize(driver: Any, stream: Any) -> None:
        check_cuda(
            driver,
            driver.cuStreamSynchronize(stream)[0],
            "cuStreamSynchronize",
        )

    with hot_executable.open(), cold_executable.open():
        driver = hot_executable._driver
        if driver is None or cold_executable._driver is not driver:
            raise RuntimeError("timed hot and cold Dataflow executables must share one CUDA driver context")
        result, least_priority, greatest_priority = driver.cuCtxGetStreamPriorityRange()
        check_cuda(driver, result, "cuCtxGetStreamPriorityRange")
        result, hot_stream = driver.cuStreamCreateWithPriority(
            driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
            greatest_priority,
        )
        check_cuda(driver, result, "cuStreamCreateWithPriority(hot)")
        result, cold_stream = driver.cuStreamCreateWithPriority(
            driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
            least_priority,
        )
        check_cuda(driver, result, "cuStreamCreateWithPriority(cold)")

        def enqueue_pair() -> None:
            launch_persistent_executable(hot_executable, hot_stream)
            launch_persistent_executable(cold_executable, cold_stream)

        def append_timing_rows(
            kernel: str,
            compiled: Any,
            timed: Any,
            sample: int,
        ) -> None:
            rows = dataflow_walltime.timing_rows(
                compiled,
                timed.read_timing_records(),
                clock_khz=0,
                sample=sample,
            )
            detail_rows.extend({"kernel": kernel, **row} for row in rows)

        try:
            print("profiling instrumented solo launches", flush=True)
            for _ in range(warmups):
                launch_persistent_executable(hot_executable, hot_stream)
                synchronize(driver, hot_stream)
                launch_persistent_executable(cold_executable, cold_stream)
                synchronize(driver, cold_stream)
            for sample in range(repeats):
                hot_timed.clear_timing_records()
                launch_persistent_executable(hot_executable, hot_stream)
                synchronize(driver, hot_stream)
                append_timing_rows(
                    "hot_solo",
                    hot_compiled,
                    hot_timed,
                    sample,
                )
                cold_timed.clear_timing_records()
                launch_persistent_executable(cold_executable, cold_stream)
                synchronize(driver, cold_stream)
                append_timing_rows(
                    "cold_solo",
                    cold_compiled,
                    cold_timed,
                    sample,
                )

            print("profiling instrumented concurrent launches", flush=True)
            for _ in range(warmups):
                enqueue_pair()
                synchronize(driver, hot_stream)
                synchronize(driver, cold_stream)
            for sample in range(repeats):
                hot_timed.clear_timing_records()
                cold_timed.clear_timing_records()
                enqueue_pair()
                synchronize(driver, hot_stream)
                synchronize(driver, cold_stream)
                for kernel, compiled, timed in (
                    ("hot", hot_compiled, hot_timed),
                    ("cold", cold_compiled, cold_timed),
                ):
                    append_timing_rows(
                        kernel,
                        compiled,
                        timed,
                        sample,
                    )
        finally:
            synchronize(driver, hot_stream)
            synchronize(driver, cold_stream)
            check_cuda(driver, driver.cuStreamDestroy(hot_stream)[0], "cuStreamDestroy(hot_timing)")
            check_cuda(driver, driver.cuStreamDestroy(cold_stream)[0], "cuStreamDestroy(cold_timing)")

    print("profiled instrumented concurrent launches", flush=True)
    summaries = dual_concurrent_summary_rows(detail_rows)
    solo_summaries = solo_walltime_summary_rows(detail_rows)
    report = format_dual_concurrent_walltime_report(
        summaries,
        solo_summaries,
    )
    output_path = Path(output_dir)
    detail_csv = output_path / "dataflow_moe_dual_concurrent_details.csv"
    summary_csv = output_path / "dataflow_moe_dual_concurrent_summary.csv"
    dataflow_walltime._write_csv_rows(detail_csv, detail_rows)
    dataflow_walltime._write_csv_rows(summary_csv, summaries + solo_summaries)
    return DualConcurrentWalltimeResult(
        detail_rows=tuple(detail_rows),
        summary_rows=summaries,
        solo_summary_rows=solo_summaries,
        report=report,
        detail_csv=detail_csv,
        summary_csv=summary_csv,
    )


def profile_persistent_dual_event_free(
    hot_executable: Any,
    cold_executable: Any,
    *,
    warmups: int,
    repeats: int,
    timing_batch_size: int,
) -> tuple[list[float], list[float], list[float], dict[str, float], dict[str, float]]:
    from tilelang.dataflow.executor import check_cuda

    if warmups < 0:
        raise ValueError(f"warmups must be non-negative, got {warmups}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if timing_batch_size <= 0:
        raise ValueError(f"timing_batch_size must be positive, got {timing_batch_size}")

    def create_stream(driver: Any, priority: int) -> Any:
        result, stream = driver.cuStreamCreateWithPriority(
            driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
            priority,
        )
        check_cuda(driver, result, "cuStreamCreateWithPriority")
        return stream

    def synchronize(driver: Any, stream: Any) -> None:
        check_cuda(
            driver,
            driver.cuStreamSynchronize(stream)[0],
            "cuStreamSynchronize",
        )

    def allocate_epoch(driver: Any) -> Any:
        result, pointer = driver.cuMemAlloc(8)
        check_cuda(driver, result, "cuMemAlloc(epoch)")
        check_cuda(driver, driver.cuMemsetD8(pointer, 0, 8)[0], "cuMemsetD8(epoch)")
        return pointer

    def write_epoch(driver: Any, stream: Any, pointer: Any, epoch: int) -> None:
        check_cuda(
            driver,
            driver.cuStreamWriteValue64(
                stream,
                pointer,
                epoch,
                driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )[0],
            "cuStreamWriteValue64",
        )

    def wait_epoch(driver: Any, stream: Any, pointer: Any, epoch: int) -> None:
        check_cuda(
            driver,
            driver.cuStreamWaitValue64(
                stream,
                pointer,
                epoch,
                driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
            )[0],
            "cuStreamWaitValue64",
        )

    def profile_single(
        driver: Any,
        executable: Any,
        stream: Any,
    ) -> list[float]:
        for _ in range(warmups):
            launch_persistent_executable(executable, stream)
            synchronize(driver, stream)

        samples: list[float] = []
        for _ in range(repeats):
            started_ns = time.perf_counter_ns()
            for _ in range(timing_batch_size):
                launch_persistent_executable(executable, stream)
            synchronize(driver, stream)
            samples.append((time.perf_counter_ns() - started_ns) / (1_000_000.0 * timing_batch_size))
        return samples

    def profile_solo(executable: Any, *, label: str, high_priority: bool):
        print(f"profiling {label} solo event-free launches", flush=True)
        with executable.open():
            driver = executable._driver
            if driver is None:
                raise RuntimeError("Dataflow persistent executable did not open a CUDA driver")
            result, least_priority, greatest_priority = driver.cuCtxGetStreamPriorityRange()
            check_cuda(driver, result, "cuCtxGetStreamPriorityRange")
            stream = create_stream(
                driver,
                greatest_priority if high_priority else least_priority,
            )
            try:
                samples = profile_single(driver, executable, stream)
                setup = dict(executable.setup_timings_ms)
            finally:
                synchronize(driver, stream)
                check_cuda(
                    driver,
                    driver.cuStreamDestroy(stream)[0],
                    "cuStreamDestroy(solo)",
                )
        print(f"profiled {label} solo event-free launches", flush=True)
        return samples, setup

    hot_samples, hot_setup = profile_solo(
        hot_executable,
        label="hot",
        high_priority=True,
    )
    cold_samples, cold_setup = profile_solo(
        cold_executable,
        label="cold",
        high_priority=False,
    )

    print("profiling concurrent event-free launches", flush=True)
    with hot_executable.open(), cold_executable.open():
        driver = hot_executable._driver
        if driver is None or cold_executable._driver is not driver:
            raise RuntimeError("hot and cold Dataflow executables must share one CUDA driver context")
        result, least_priority, greatest_priority = driver.cuCtxGetStreamPriorityRange()
        check_cuda(driver, result, "cuCtxGetStreamPriorityRange")
        hot_stream = create_stream(driver, greatest_priority)
        cold_stream = create_stream(driver, least_priority)
        hot_done = allocate_epoch(driver)
        cold_done = allocate_epoch(driver)
        try:
            epoch = 0

            def enqueue_pair() -> None:
                nonlocal epoch
                epoch += 1
                launch_persistent_executable(hot_executable, hot_stream)
                launch_persistent_executable(cold_executable, cold_stream)
                write_epoch(driver, hot_stream, hot_done, epoch)
                write_epoch(driver, cold_stream, cold_done, epoch)
                wait_epoch(driver, hot_stream, cold_done, epoch)
                wait_epoch(driver, cold_stream, hot_done, epoch)

            for _ in range(warmups):
                enqueue_pair()
                synchronize(driver, hot_stream)
                synchronize(driver, cold_stream)

            total_samples: list[float] = []
            for _ in range(repeats):
                started_ns = time.perf_counter_ns()
                for _ in range(timing_batch_size):
                    enqueue_pair()
                synchronize(driver, hot_stream)
                total_samples.append((time.perf_counter_ns() - started_ns) / (1_000_000.0 * timing_batch_size))
            synchronize(driver, cold_stream)
        finally:
            synchronize(driver, hot_stream)
            synchronize(driver, cold_stream)
            check_cuda(driver, driver.cuMemFree(hot_done)[0], "cuMemFree(hot_epoch)")
            check_cuda(driver, driver.cuMemFree(cold_done)[0], "cuMemFree(cold_epoch)")
            check_cuda(driver, driver.cuStreamDestroy(hot_stream)[0], "cuStreamDestroy(hot)")
            check_cuda(driver, driver.cuStreamDestroy(cold_stream)[0], "cuStreamDestroy(cold)")

    print("profiled concurrent event-free launches", flush=True)
    return hot_samples, cold_samples, total_samples, hot_setup, cold_setup


def profile_dataflow_dual_routed_moe(
    token_num: int,
    *,
    d_hidden: int,
    d_expert: int,
    top_k: int,
    block_token: int,
    cold_block_token: int,
    block_dhidden: int,
    map2_block_dhidden: int,
    block_dexpert: int,
    cold_block_dhidden: int,
    cold_map2_block_dhidden: int,
    cold_map2_handler_dhidden: int | None,
    cold_block_dexpert: int,
    cold_fuse_map1_expert_shards: bool | None = (DEFAULT_DATAFLOW_COLD_FUSE_MAP1_EXPERT_SHARDS),
    cold_map1_gate_stages: int | None = None,
    cold_weight_eviction_policy: str = "evict_first",
    hot_sm_count: int,
    hot_cluster_size: int | None,
    cold_sm_count: int,
    cold_cluster_size: int,
    cold_expert_count: int | None,
    hot_map1_consumer_threads: int | None = DEFAULT_DATAFLOW_HOT_MAP1_CONSUMER_THREADS,
    hot_map1_handler_dexpert: int | None = None,
    hot_map1_wgmma_wait_depth: int | None = None,
    hot_map2_handler_dhidden: int | None = None,
    warmups: int,
    repeats: int,
    timing_batch_size: int,
    use_fp8: bool,
    use_tma_weights: bool,
    reshared_policy: str,
    cold_reshared_policy: str | None = None,
    map2_weight_stages: int | None = DEFAULT_DATAFLOW_MAP2_WEIGHT_STAGES,
    map2_weight_max_outstanding: int = DEFAULT_DATAFLOW_MAP2_WEIGHT_MAX_OUTSTANDING,
    cold_map2_weight_stages: int = DEFAULT_DATAFLOW_COLD_MAP2_WEIGHT_STAGES,
    cold_map2_weight_max_outstanding: int = (DEFAULT_DATAFLOW_COLD_MAP2_WEIGHT_MAX_OUTSTANDING),
    fuse_map2_hidden_tiles: bool = DEFAULT_DATAFLOW_FUSE_MAP2_HIDDEN_TILES,
    walltime_dir: str | None = None,
    walltime_repeats: int = 1,
    dump_dataflow_source: str | None = None,
    verify_output: bool = False,
    map1_input_mode: str = DEFAULT_DATAFLOW_MAP1_INPUT_MODE,
    cold_map1_input_mode: str | None = DEFAULT_DATAFLOW_COLD_MAP1_INPUT_MODE,
    map2_map1_lookahead_stages: int | None = DEFAULT_DATAFLOW_MAP2_MAP1_LOOKAHEAD_STAGES,
    cold_launch_delay_us: float = 0.0,
    cold_launch_stagger_us: float = 0.0,
    cold_launch_stagger_group_clusters: int = 1,
    compile_record_builder: Callable[[Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    import tilelang.language as T
    import tilelang.dataflow as df

    from examples.dataflow.fusedmoe import example_fusedmoe_dataflow as dataflow_moe

    if cold_reshared_policy is None:
        cold_reshared_policy = reshared_policy
    if not math.isfinite(cold_launch_delay_us) or cold_launch_delay_us < 0.0:
        raise ValueError(f"cold_launch_delay_us must be finite and non-negative, got {cold_launch_delay_us}")
    cold_launch_delay_ns = int(round(cold_launch_delay_us * 1000.0))
    if not math.isfinite(cold_launch_stagger_us) or cold_launch_stagger_us < 0.0:
        raise ValueError(f"cold_launch_stagger_us must be finite and non-negative, got {cold_launch_stagger_us}")
    cold_launch_stagger_ns = int(round(cold_launch_stagger_us * 1000.0))
    if cold_launch_stagger_group_clusters <= 0:
        raise ValueError(f"cold_launch_stagger_group_clusters must be positive, got {cold_launch_stagger_group_clusters}")
    requested_cold_fuse_map1_expert_shards = cold_fuse_map1_expert_shards
    cold_fuse_map1_expert_shards = resolve_dataflow_cold_map1_fusion(
        d_expert=d_expert,
        cold_cluster_size=cold_cluster_size,
        cold_block_dexpert=cold_block_dexpert,
        requested=cold_fuse_map1_expert_shards,
    )
    if map2_weight_stages is None:
        map2_weight_stages = 4 if reshared_policy == "cluster_shared_pull_ring" else 2
    supported_cluster_policies = {
        "cluster_shared_all_gather",
        "cluster_shared_pull_ring",
    }
    if reshared_policy not in supported_cluster_policies or cold_reshared_policy not in supported_cluster_policies:
        raise ValueError(
            "event-free dual-kernel profiling currently requires "
            "a cluster-shared reshared policy so repeated launches do not reuse HBM flags"
        )
    if walltime_repeats <= 0:
        raise ValueError(f"walltime_repeats must be positive, got {walltime_repeats}")
    cold_cluster_count = cold_sm_count // cold_cluster_size
    counts = scale_expert_heat(EXPERT_HEAT, total_assignments=token_num * top_k)
    group_sum = token_num * top_k
    hot_metadata = build_group_metadata(counts, block_token=block_token, device="cuda")
    hot_plan_blocks = build_group_block_plan(
        counts,
        group_padded_offsets=hot_metadata.group_padded_offsets.tolist(),
        group_idx_for_bx=hot_metadata.group_idx_for_bx.tolist(),
        block_token=block_token,
    )
    if cold_block_token == block_token:
        cold_metadata = hot_metadata
        cold_plan_blocks = hot_plan_blocks
    else:
        cold_metadata = build_group_metadata(counts, block_token=cold_block_token, device="cuda")
        cold_plan_blocks = build_group_block_plan(
            counts,
            group_padded_offsets=cold_metadata.group_padded_offsets.tolist(),
            group_idx_for_bx=cold_metadata.group_idx_for_bx.tolist(),
            block_token=cold_block_token,
        )
    requested_hot_cluster_size = hot_cluster_size
    hot_topology, dual_schedule = select_dataflow_hot_topology(
        hot_plan_blocks,
        cold_plan_blocks=cold_plan_blocks,
        d_hidden=d_hidden,
        d_expert=d_expert,
        block_token=block_token,
        block_dhidden=block_dhidden,
        map2_block_dhidden=map2_block_dhidden,
        block_dexpert=block_dexpert,
        hot_sm_count=hot_sm_count,
        cold_cluster_count=cold_cluster_count,
        cold_expert_count=cold_expert_count,
        hot_cluster_size=hot_cluster_size,
        hot_map2_handler_dhidden=hot_map2_handler_dhidden,
        map2_map1_lookahead_stages=map2_map1_lookahead_stages,
        fuse_map2_hidden_tiles=fuse_map2_hidden_tiles,
    )
    hot_cluster_size = hot_topology.cluster_size
    hot_map2_handler_dhidden = hot_topology.map2_handler_dhidden
    map2_map1_lookahead_stages = hot_topology.map2_map1_lookahead_stages
    if cold_map2_handler_dhidden is None:
        cold_map2_handler_dhidden = min(
            d_hidden // cold_cluster_size,
            7 * cold_map2_block_dhidden,
        )
    split = dual_schedule.split
    cold_expert_ids = {block.expert_id for block in split.cold_blocks}
    hot_blocks = split.hot_blocks
    cold_blocks = split.cold_blocks
    hot_task_costs = dual_schedule.hot_task_costs_us
    cold_task_costs = dual_schedule.cold_task_costs_us
    hot_cluster_assignment = dual_schedule.hot_cluster_assignment
    cold_cluster_assignment = dual_schedule.cold_cluster_assignment
    selected_cold_expert_count = dual_schedule.cold_expert_count

    n_experts = len(EXPERT_HEAT)
    input_torch_dtype, weight_torch_dtype = dataflow_torch_dtypes(use_fp8=use_fp8)
    input_tl_dtype = T.float8_e4m3fn if use_fp8 else T.float16
    weight_tl_dtype = T.float8_e4m3fn if use_fp8 else T.float16
    intermediate_tl_dtype = T.float8_e4m3fn if use_fp8 else T.float16
    hot_use_small_token_wgmma = bool(use_fp8 and block_token < 64)
    cold_use_small_token_wgmma = bool(use_fp8 and cold_block_token < 64)
    requested_cold_map1_gate_stages = cold_map1_gate_stages
    cold_map1_gate_stages = resolve_dataflow_cold_map1_gate_stages(
        use_small_token_wgmma=cold_use_small_token_wgmma,
        fuse_map1_expert_shards=cold_fuse_map1_expert_shards,
        requested=requested_cold_map1_gate_stages,
    )
    hot_use_tma_weights, cold_use_tma_weights = dataflow_dual_tma_modes(
        use_tma_weights=use_tma_weights,
        cold_block_token=cold_block_token,
        use_small_token_wgmma=cold_use_small_token_wgmma,
    )
    hot_threads = 128 if hot_use_small_token_wgmma else 256
    cold_threads = 128 if cold_use_small_token_wgmma else 256
    hot_map2_tile_dhidden = min(
        d_hidden // hot_cluster_size,
        map2_block_dhidden,
    )
    fuse_map2_hidden_tiles = bool(
        fuse_map2_hidden_tiles
        and hot_use_tma_weights
        and not hot_use_small_token_wgmma
        and math.ceil((d_hidden // hot_cluster_size) / hot_map2_tile_dhidden) == 2
    )
    hot_map1_codegen_options = dataflow_map1_codegen_options(
        use_fp8=use_fp8,
        use_tma_weights=hot_use_tma_weights,
        cluster_size=hot_cluster_size,
        input_mode=map1_input_mode,
    )
    if cold_map1_input_mode is None:
        cold_map1_input_mode = map1_input_mode
    cold_map1_codegen_options = dataflow_map1_codegen_options(
        use_fp8=use_fp8,
        use_tma_weights=cold_use_tma_weights,
        cluster_size=cold_cluster_size,
        input_mode=cold_map1_input_mode,
    )
    requested_hot_map1_wgmma_wait_depth = hot_map1_wgmma_wait_depth
    hot_map1_wgmma_wait_depth = resolve_dataflow_hot_map1_wgmma_wait_depth(
        d_hidden=d_hidden,
        d_expert=d_expert,
        block_dhidden=block_dhidden,
        block_dexpert=block_dexpert,
        block_token=block_token,
        cluster_size=hot_cluster_size,
        map1_handler_dexpert=hot_map1_handler_dexpert,
        map1_consumer_threads=hot_map1_consumer_threads,
        use_fp8=use_fp8,
        use_tma_weights=hot_use_tma_weights,
        map1_codegen_options=hot_map1_codegen_options,
        map2_map1_lookahead_stages=map2_map1_lookahead_stages,
        requested=hot_map1_wgmma_wait_depth,
    )
    hot_map1_pipeline_selection = (
        "auto_pipeline"
        if requested_hot_map1_wgmma_wait_depth is None and hot_map1_wgmma_wait_depth
        else ("auto_baseline" if requested_hot_map1_wgmma_wait_depth is None else "explicit")
    )
    if hot_map1_consumer_threads is None:
        hot_map1_consumer_threads = (
            256
            if hot_map1_wgmma_wait_depth
            else (
                128
                if hot_map1_codegen_options["use_tma_map1_input_multicast"] or hot_map1_codegen_options["use_tma_map1_input_unicast"]
                else hot_threads
            )
        )

    expected_output = None
    if verify_output:
        generator = torch.Generator(device="cuda").manual_seed(17)

        def random_tensor(shape: tuple[int, ...], scale: float, dtype: torch.dtype):
            return (
                torch.randn(
                    shape,
                    dtype=torch.float16,
                    device="cuda",
                    generator=generator,
                )
                * scale
            ).to(dtype)

        input_tensor = random_tensor(
            (group_sum, d_hidden),
            0.25,
            input_torch_dtype,
        )
        routed_expert_gate = random_tensor(
            (n_experts, d_expert, d_hidden),
            0.10,
            weight_torch_dtype,
        )
        routed_expert_up = random_tensor(
            (n_experts, d_expert, d_hidden),
            0.10,
            weight_torch_dtype,
        )
        routed_expert_gate_up = dataflow_moe.pack_gate_up_weights(
            routed_expert_gate,
            routed_expert_up,
        )
        routed_expert_down = random_tensor(
            (n_experts, d_hidden, d_expert),
            0.05,
            weight_torch_dtype,
        )
        routed_expert_weights = torch.rand(
            (group_sum,),
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        output = torch.zeros(
            (group_sum, d_hidden),
            dtype=torch.float16,
            device="cuda",
        )
        expected_output = routed_moe_reference(
            input_tensor,
            routed_expert_gate,
            routed_expert_up,
            routed_expert_down,
            routed_expert_weights,
            counts,
            hot_metadata.group_offsets,
            intermediate_dtype=weight_torch_dtype,
        )
    else:
        input_tensor = torch.empty((group_sum, d_hidden), dtype=input_torch_dtype, device="cuda")
        routed_expert_gate = None
        routed_expert_up = None
        routed_expert_gate_up = df.mark_tensor_layout(
            torch.empty(
                (n_experts, 2 * d_expert, d_hidden),
                dtype=weight_torch_dtype,
                device="cuda",
            ),
            dataflow_moe.gate_up_weight_layout(n_experts, d_expert, d_hidden),
        )
        routed_expert_down = torch.empty((n_experts, d_hidden, d_expert), dtype=weight_torch_dtype, device="cuda")
        routed_expert_weights = torch.empty((group_sum,), dtype=torch.float16, device="cuda")
        output = torch.empty((group_sum, d_hidden), dtype=torch.float16, device="cuda")

    tensor_kwargs = {
        "input": input_tensor,
        "routed_expert_gate_up": routed_expert_gate_up,
        "routed_expert_down": routed_expert_down,
        "routed_expert_weights": routed_expert_weights,
        "output": output,
    }

    def metadata_kwargs(metadata: RoutedGroupMetadata) -> dict[str, Any]:
        return {
            **tensor_kwargs,
            "group_sizes": metadata.group_sizes,
            "group_offsets": metadata.group_offsets,
            "group_padded_offsets": metadata.group_padded_offsets,
            "group_idx_for_bx": metadata.group_idx_for_bx,
        }

    def compile_bucket(
        blocks: tuple[RoutedGroupBlock, ...],
        *,
        task_costs: tuple[float, ...],
        cluster_assignment: tuple[int, ...],
        bucket_block_token: int,
        bucket_block_dhidden: int,
        bucket_map2_block_dhidden: int,
        bucket_map2_handler_dhidden: int | None,
        bucket_block_dexpert: int,
        bucket_map1_handler_dexpert: int | None,
        bucket_use_tma_weights: bool,
        bucket_weight_eviction_policy: str,
        bucket_use_small_token_wgmma: bool,
        bucket_map1_gate_stages: int,
        bucket_map1_wgmma_wait_depth: int,
        bucket_map2_weight_stages: int,
        bucket_map2_weight_max_outstanding: int,
        bucket_fuse_map2_hidden_tiles: bool,
        bucket_map2_map1_lookahead_stages: int,
        bucket_threads: int,
        bucket_map1_consumer_threads: int | None,
        map1_codegen_options: dict[str, bool],
        sm_count: int,
        cluster_size: int,
        bucket_reshared_policy: str,
    ):
        if not blocks:
            return None
        try:
            return dataflow_moe.compile_dataflow_routed_moe(
                d_hidden=d_hidden,
                d_expert=d_expert,
                n_routed_experts=n_experts,
                group_sum=group_sum,
                input_dtype=input_tl_dtype,
                weight_dtype=weight_tl_dtype,
                intermediate_dtype=intermediate_tl_dtype,
                semantic_config=df.DataflowSemanticConfig(fast_math=True),
                execution_override=dataflow_execution_override(
                    d_hidden=d_hidden,
                    d_expert=d_expert,
                    block_token=bucket_block_token,
                    block_dhidden=bucket_block_dhidden,
                    map2_block_dhidden=bucket_map2_block_dhidden,
                    map2_handler_dhidden=bucket_map2_handler_dhidden,
                    fuse_map2_hidden_tiles=bucket_fuse_map2_hidden_tiles,
                    block_dexpert=bucket_block_dexpert,
                    map1_handler_dexpert=bucket_map1_handler_dexpert,
                    cluster_size=cluster_size,
                    sm_count=sm_count,
                    threads=bucket_threads,
                    map1_consumer_threads=bucket_map1_consumer_threads,
                    use_tma_weights=bucket_use_tma_weights,
                    weight_eviction_policy=bucket_weight_eviction_policy,
                    use_small_token_wgmma=bucket_use_small_token_wgmma,
                    map1_gate_stages=bucket_map1_gate_stages,
                    map1_wgmma_wait_depth=bucket_map1_wgmma_wait_depth,
                    map2_weight_stages=bucket_map2_weight_stages,
                    map2_weight_max_outstanding=(bucket_map2_weight_max_outstanding),
                    map1_input_mode=(
                        "multicast"
                        if map1_codegen_options["use_tma_map1_input_multicast"]
                        else "unicast"
                        if map1_codegen_options["use_tma_map1_input_unicast"]
                        else "cooperative"
                    ),
                    split_map1_tma_producers=map1_codegen_options["split_map1_tma_producers"],
                    map2_map1_lookahead_stages=(bucket_map2_map1_lookahead_stages),
                    reshared_policy=bucket_reshared_policy,
                ),
                active_group_blocks=tuple(block.group_block for block in blocks),
                stage_graph_task_weights=task_costs,
                stage_graph_cluster_assignment=cluster_assignment,
            )
        except Exception as err:
            if bucket_use_tma_weights and bucket_block_token < 64:
                raise RuntimeError(
                    "Dataflow cold block_token < 64 cannot use the token-major TMA/WGMMA path; "
                    "enable the transposed small-token WGMMA path or keep "
                    "--dataflow-cold-block-token 64."
                ) from err
            raise

    print("compiling hot routed-MoE bucket", flush=True)
    hot_compiled = compile_bucket(
        hot_blocks,
        task_costs=hot_task_costs,
        cluster_assignment=hot_cluster_assignment,
        bucket_block_token=block_token,
        bucket_block_dhidden=block_dhidden,
        bucket_map2_block_dhidden=map2_block_dhidden,
        bucket_map2_handler_dhidden=hot_map2_handler_dhidden,
        bucket_block_dexpert=block_dexpert,
        bucket_map1_handler_dexpert=hot_map1_handler_dexpert,
        bucket_use_tma_weights=hot_use_tma_weights,
        bucket_weight_eviction_policy="evict_first",
        bucket_use_small_token_wgmma=hot_use_small_token_wgmma,
        bucket_map1_gate_stages=(3 if hot_map1_wgmma_wait_depth else 2),
        bucket_map1_wgmma_wait_depth=hot_map1_wgmma_wait_depth,
        bucket_map2_weight_stages=map2_weight_stages,
        bucket_map2_weight_max_outstanding=map2_weight_max_outstanding,
        bucket_fuse_map2_hidden_tiles=fuse_map2_hidden_tiles,
        bucket_map2_map1_lookahead_stages=map2_map1_lookahead_stages,
        bucket_threads=hot_threads,
        bucket_map1_consumer_threads=hot_map1_consumer_threads,
        map1_codegen_options=hot_map1_codegen_options,
        sm_count=hot_sm_count,
        cluster_size=hot_cluster_size,
        bucket_reshared_policy=reshared_policy,
    )
    print("compiled hot routed-MoE bucket", flush=True)
    print("compiling cold routed-MoE bucket", flush=True)
    cold_compiled = compile_bucket(
        cold_blocks,
        task_costs=cold_task_costs,
        cluster_assignment=cold_cluster_assignment,
        bucket_block_token=cold_block_token,
        bucket_block_dhidden=cold_block_dhidden,
        bucket_map2_block_dhidden=cold_map2_block_dhidden,
        bucket_map2_handler_dhidden=cold_map2_handler_dhidden,
        bucket_block_dexpert=cold_block_dexpert,
        bucket_map1_handler_dexpert=(d_expert // cold_cluster_size if cold_fuse_map1_expert_shards else None),
        bucket_use_tma_weights=cold_use_tma_weights,
        bucket_weight_eviction_policy=cold_weight_eviction_policy,
        bucket_use_small_token_wgmma=cold_use_small_token_wgmma,
        bucket_map1_gate_stages=cold_map1_gate_stages,
        bucket_map1_wgmma_wait_depth=0,
        bucket_map2_weight_stages=cold_map2_weight_stages,
        bucket_map2_weight_max_outstanding=(cold_map2_weight_max_outstanding),
        bucket_fuse_map2_hidden_tiles=False,
        bucket_map2_map1_lookahead_stages=0,
        bucket_threads=cold_threads,
        bucket_map1_consumer_threads=None,
        map1_codegen_options=cold_map1_codegen_options,
        sm_count=cold_sm_count,
        cluster_size=cold_cluster_size,
        bucket_reshared_policy=cold_reshared_policy,
    )
    print("compiled cold routed-MoE bucket", flush=True)
    hot_kwargs = metadata_kwargs(hot_metadata)
    cold_kwargs = metadata_kwargs(cold_metadata)
    if hot_compiled is None or cold_compiled is None:
        raise ValueError("dual-kernel profiling requires non-empty hot and cold compiled plans")

    if cold_launch_delay_ns or cold_launch_stagger_ns:
        launch_pacing = df.DataflowWrapperLaunchPacing(
            delay_ns=cold_launch_delay_ns,
            cluster_stagger_ns=cold_launch_stagger_ns,
            cluster_stagger_group_size=cold_launch_stagger_group_clusters,
        )
        paced_wrapper_spec = replace(
            cold_compiled.wrapper_spec,
            launch_pacing=launch_pacing,
        )
        cold_compiled = replace(
            cold_compiled,
            wrapper_spec=paced_wrapper_spec,
            wrapper_source=df.generate_wrapper_source(paced_wrapper_spec),
        )

    compile_records = (
        None
        if compile_record_builder is None
        else {
            "hot": dict(compile_record_builder(hot_compiled)),
            "cold": dict(compile_record_builder(cold_compiled)),
        }
    )

    hot_source_path = None
    cold_source_path = None
    if dump_dataflow_source is not None:
        hot_source_file, cold_source_file = dual_source_paths(dump_dataflow_source)
        hot_source_file.parent.mkdir(parents=True, exist_ok=True)
        hot_source_file.write_text(hot_compiled.wrapper_source, encoding="utf-8")
        cold_source_file.write_text(cold_compiled.wrapper_source, encoding="utf-8")
        hot_source_path = str(hot_source_file)
        cold_source_path = str(cold_source_file)

    hot_assignment_summary = summarize_stage_graph_assignment(
        hot_compiled.plan,
        task_costs=hot_task_costs,
    )
    cold_assignment_summary = summarize_stage_graph_assignment(
        cold_compiled.plan,
        task_costs=cold_task_costs,
    )
    hot_executable = hot_compiled.persistent_executable(**hot_kwargs)
    cold_executable = cold_compiled.persistent_executable(**cold_kwargs)
    hot_samples, cold_samples, samples, hot_setup_timings_ms, cold_setup_timings_ms = profile_persistent_dual_event_free(
        hot_executable,
        cold_executable,
        warmups=warmups,
        repeats=repeats,
        timing_batch_size=timing_batch_size,
    )
    concurrent_output = None
    if expected_output is not None:
        torch.cuda.synchronize()
        concurrent_output = output.float().clone()

    dual_walltime = None
    instrumented_concurrent_output = None
    if walltime_dir is not None:
        dual_walltime = profile_persistent_dual_walltime(
            hot_compiled,
            cold_compiled,
            hot_kwargs=hot_kwargs,
            cold_kwargs=cold_kwargs,
            warmups=warmups,
            repeats=walltime_repeats,
            output_dir=walltime_dir,
        )
        if expected_output is not None:
            torch.cuda.synchronize()
            instrumented_concurrent_output = output.float().clone()
    verification_max_abs_error = None
    verification_mean_abs_error = None
    verification_rmse = None
    verification_relative_l2 = None
    verification_concurrent_max_abs_error = None
    verification_instrumented_max_abs_error = None
    if expected_output is not None:
        if concurrent_output is None:
            raise RuntimeError("production concurrent output was not captured")
        output.zero_()
        torch.cuda.synchronize()
        hot_compiled(**hot_kwargs)
        cold_compiled(**cold_kwargs)
        torch.cuda.synchronize()
        sequential_output = output.float()
        verification_concurrent_max_abs_error = float((concurrent_output - sequential_output).abs().max().item())
        if instrumented_concurrent_output is not None:
            verification_instrumented_max_abs_error = float((instrumented_concurrent_output - concurrent_output).abs().max().item())
        output_float = concurrent_output
        error = output_float - expected_output
        abs_error = error.abs()
        verification_max_abs_error = float(abs_error.max().item())
        verification_mean_abs_error = float(abs_error.mean().item())
        verification_rmse = float(error.square().mean().sqrt().item())
        verification_relative_l2 = float(error.norm().div(expected_output.norm().clamp_min(1.0e-12)).item())
    row = base_result(
        "dataflow-dual",
        token_num=token_num,
        top_k=top_k,
        block_token=block_token,
        counts=counts,
        metadata=hot_metadata,
        samples=samples,
    )
    row.update(
        {
            "dtype": "input_fp8_weights_fp8_intermediate_fp8" if use_fp8 else "fp16",
            "hot_sm_count": hot_sm_count,
            "hot_cluster_size": hot_cluster_size,
            "hot_cluster_size_requested": ("auto" if requested_hot_cluster_size is None else requested_hot_cluster_size),
            "hot_topology_selection_score": hot_topology.selection_score,
            "hot_topology_candidate_scores": hot_topology.candidate_scores,
            "cold_sm_count": cold_sm_count,
            "cold_cluster_size": cold_cluster_size,
            "cold_expert_count": selected_cold_expert_count,
            "cold_expert_count_requested": ("auto" if cold_expert_count is None else cold_expert_count),
            "dual_schedule_selection_score": dual_schedule.selection_score,
            "dual_schedule_candidate_scores": dual_schedule.candidate_scores,
            "cold_expert_ids": tuple(sorted(cold_expert_ids)),
            "hot_reshared_policy": reshared_policy,
            "cold_reshared_policy": cold_reshared_policy,
            "map2_weight_stages": map2_weight_stages,
            "map2_weight_max_outstanding": map2_weight_max_outstanding,
            "cold_map2_weight_stages": cold_map2_weight_stages,
            "cold_map2_weight_max_outstanding": (cold_map2_weight_max_outstanding),
            "fuse_map2_hidden_tiles": fuse_map2_hidden_tiles,
            "cold_launch_delay_us": cold_launch_delay_us,
            "cold_launch_delay_ns": cold_launch_delay_ns,
            "cold_launch_stagger_us": cold_launch_stagger_us,
            "cold_launch_stagger_ns": cold_launch_stagger_ns,
            "cold_launch_stagger_group_clusters": (cold_launch_stagger_group_clusters),
            "timing_metric": "persistent_stream_host_batch_ms",
            "timing_batch_size": timing_batch_size,
            "samples_ms": tuple(samples),
            "hot_samples_ms": tuple(hot_samples),
            "cold_samples_ms": tuple(cold_samples),
            "compile_records": compile_records,
            "hot_use_tma_weights": hot_use_tma_weights,
            "cold_use_tma_weights": cold_use_tma_weights,
            "hot_use_small_token_wgmma": hot_use_small_token_wgmma,
            "hot_threads": hot_threads,
            "hot_map1_consumer_threads": hot_map1_consumer_threads,
            "hot_map1_handler_dexpert": hot_map1_handler_dexpert,
            "hot_map1_wgmma_wait_depth": hot_map1_wgmma_wait_depth,
            "hot_map1_wgmma_wait_depth_requested": (
                "auto" if requested_hot_map1_wgmma_wait_depth is None else requested_hot_map1_wgmma_wait_depth
            ),
            "hot_map1_pipeline_selection": hot_map1_pipeline_selection,
            "hot_map2_handler_dhidden": hot_map2_handler_dhidden,
            "cold_threads": cold_threads,
            "cold_use_small_token_wgmma": cold_use_small_token_wgmma,
            "cold_map1_gate_stages": cold_map1_gate_stages,
            "cold_weight_eviction_policy": cold_weight_eviction_policy,
            "cold_map1_gate_stages_requested": ("auto" if requested_cold_map1_gate_stages is None else requested_cold_map1_gate_stages),
            "hot_map1_input_multicast": hot_map1_codegen_options["use_tma_map1_input_multicast"],
            "cold_map1_input_multicast": cold_map1_codegen_options["use_tma_map1_input_multicast"],
            "hot_map1_input_unicast": hot_map1_codegen_options["use_tma_map1_input_unicast"],
            "cold_map1_input_unicast": cold_map1_codegen_options["use_tma_map1_input_unicast"],
            "map1_input_mode": map1_input_mode,
            "hot_map1_input_mode": map1_input_mode,
            "cold_map1_input_mode": cold_map1_input_mode,
            "hot_map2_map1_lookahead_stages": map2_map1_lookahead_stages,
            "hot_block_token": block_token,
            "cold_block_token": cold_block_token,
            "cold_block_dhidden": cold_block_dhidden,
            "cold_map2_block_dhidden": cold_map2_block_dhidden,
            "cold_map2_handler_dhidden": cold_map2_handler_dhidden,
            "cold_block_dexpert": cold_block_dexpert,
            "cold_fuse_map1_expert_shards": (cold_fuse_map1_expert_shards),
            "cold_fuse_map1_expert_shards_requested": (
                "auto" if requested_cold_fuse_map1_expert_shards is None else requested_cold_fuse_map1_expert_shards
            ),
            "hot_blocks": len(hot_blocks),
            "cold_blocks": len(cold_blocks),
            "hot_task_cluster_assignment": hot_assignment_summary.task_cluster_assignment,
            "cold_task_cluster_assignment": cold_assignment_summary.task_cluster_assignment,
            "hot_cluster_task_counts": hot_assignment_summary.cluster_task_counts,
            "cold_cluster_task_counts": cold_assignment_summary.cluster_task_counts,
            "hot_cluster_task_costs_us": hot_assignment_summary.cluster_task_costs_us,
            "cold_cluster_task_costs_us": cold_assignment_summary.cluster_task_costs_us,
            "cold_group_blocks": cold_metadata.group_blocks,
            "hot_instructions": 0 if hot_compiled is None else len(hot_compiled.plan.instructions),
            "cold_instructions": 0 if cold_compiled is None else len(cold_compiled.plan.instructions),
            "hot_comms": 0 if hot_compiled is None else len(hot_compiled.plan.comms),
            "cold_comms": 0 if cold_compiled is None else len(cold_compiled.plan.comms),
            "hot_cluster_send": 0
            if hot_compiled is None
            else sum(1 for comm in hot_compiled.plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_SEND),
            "hot_cluster_recv": 0
            if hot_compiled is None
            else sum(1 for comm in hot_compiled.plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_RECV),
            "cold_cluster_send": 0
            if cold_compiled is None
            else sum(1 for comm in cold_compiled.plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_SEND),
            "cold_cluster_recv": 0
            if cold_compiled is None
            else sum(1 for comm in cold_compiled.plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_RECV),
            "hot_shared_memory_bytes": 0 if hot_compiled is None else hot_compiled.launch_package.shared_memory_bytes,
            "cold_shared_memory_bytes": 0 if cold_compiled is None else cold_compiled.launch_package.shared_memory_bytes,
            "hot_shared_slot_bytes": 0 if hot_compiled is None else hot_compiled.launch_package.shared_slot_bytes,
            "cold_shared_slot_bytes": 0 if cold_compiled is None else cold_compiled.launch_package.shared_slot_bytes,
            "hot_barrier_count": 0 if hot_compiled is None else hot_compiled.launch_package.barrier_count,
            "cold_barrier_count": 0 if cold_compiled is None else cold_compiled.launch_package.barrier_count,
            "hot_source_path": hot_source_path,
            "cold_source_path": cold_source_path,
            "dual_walltime_detail_csv": (
                None if dual_walltime is None or dual_walltime.detail_csv is None else str(dual_walltime.detail_csv)
            ),
            "dual_walltime_summary_csv": (
                None if dual_walltime is None or dual_walltime.summary_csv is None else str(dual_walltime.summary_csv)
            ),
            "dual_walltime_report": (None if dual_walltime is None else dual_walltime.report),
            "dual_walltime_p50_us": (None if dual_walltime is None else dual_concurrent_global_span_p50_us(dual_walltime)),
            "hot_median_ms": statistics.median(hot_samples),
            "cold_median_ms": statistics.median(cold_samples),
            "hot_solo_median_ms": statistics.median(hot_samples),
            "cold_solo_median_ms": statistics.median(cold_samples),
            "hot_setup_timings_ms": hot_setup_timings_ms,
            "cold_setup_timings_ms": cold_setup_timings_ms,
            "verified_output": expected_output is not None,
            "verification_max_abs_error": verification_max_abs_error,
            "verification_mean_abs_error": verification_mean_abs_error,
            "verification_rmse": verification_rmse,
            "verification_relative_l2": verification_relative_l2,
            "verification_concurrent_max_abs_error": (verification_concurrent_max_abs_error),
            "verification_instrumented_max_abs_error": (verification_instrumented_max_abs_error),
        }
    )

    del hot_executable, cold_executable, hot_compiled, cold_compiled
    del input_tensor, routed_expert_gate_up, routed_expert_down
    del routed_expert_gate, routed_expert_up
    del routed_expert_weights, output, hot_metadata, cold_metadata
    gc.collect()
    torch.cuda.empty_cache()
    return row


def base_result(
    backend: Backend,
    *,
    token_num: int,
    top_k: int,
    block_token: int,
    counts: list[int],
    metadata: RoutedGroupMetadata,
    samples: list[float],
) -> dict[str, Any]:
    return {
        "backend": backend,
        "token_num": token_num,
        "group_sum": token_num * top_k,
        "nonzero_experts": sum(1 for count in counts if count > 0),
        "max_group_size": max(counts),
        "min_nonzero_group_size": min(count for count in counts if count > 0),
        "group_blocks": metadata.group_blocks,
        "block_token": block_token,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stddev_ms": statistics.pstdev(samples),
    }


def selected_backends(value: str) -> tuple[Backend, ...]:
    if value in ("both", "all"):
        return ("dataflow", "dataflow-dual")
    if value in ("dataflow", "dataflow-dual"):
        return (value,)  # type: ignore[return-value]
    raise ValueError(f"unsupported backend {value!r}")


def parse_positive_int_or_auto(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    try:
        count = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'auto'") from err
    if count <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'auto'")
    return count


def parse_cold_expert_count(value: str) -> int | None:
    return parse_positive_int_or_auto(value)


def parse_lookahead_stages(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    try:
        stages = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("lookahead stages must be 0, 2, or 'auto'") from err
    if stages not in (0, 2):
        raise argparse.ArgumentTypeError("lookahead stages must be 0, 2, or 'auto'")
    return stages


def parse_wgmma_wait_depth(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    try:
        depth = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("WGMMA wait depth must be 0, 1, or 'auto'") from err
    if depth not in (0, 1):
        raise argparse.ArgumentTypeError("WGMMA wait depth must be 0, 1, or 'auto'")
    return depth


def apply_published_h100_workload(args: argparse.Namespace) -> None:
    workload_name = args.published_h100_workload
    if workload_name is None:
        return
    profile = PUBLISHED_H100_WORKLOADS[workload_name]
    args.backend = "dataflow-dual"
    args.token_num = [profile["token_num"]]
    args.d_hidden = profile["d_hidden"]
    args.d_expert = profile["d_expert"]
    args.top_k = 2
    args.block_dhidden = 128
    args.block_dexpert = 128
    args.warmups = 5
    args.repeats = 30
    args.disable_cache = True
    args.dataflow_hot_sm_count = 112
    args.dataflow_hot_cluster_size = None
    args.dataflow_cold_sm_count = 20
    args.dataflow_cold_cluster_size = 2
    args.dataflow_cold_expert_count = None
    args.dataflow_dual_timing_batch_size = 10
    args.dataflow_block_token = profile["block_token"]
    args.dataflow_cold_block_token = 16
    args.dataflow_cold_block_dhidden = 128
    args.dataflow_cold_map2_block_dhidden = 256
    args.dataflow_cold_map2_handler_dhidden = None
    args.dataflow_cold_block_dexpert = 256
    args.dataflow_cold_fuse_map1_expert_shards = None
    args.dataflow_cold_map1_gate_stages = profile.get("cold_map1_gate_stages")
    args.dataflow_cold_weight_eviction_policy = "evict_first"
    args.dataflow_hot_map1_consumer_threads = None
    args.dataflow_hot_map1_wgmma_wait_depth = profile.get("wait_depth")
    args.dataflow_hot_map2_handler_dhidden = None
    args.dataflow_hot_map1_handler_dexpert = None
    args.dataflow_map2_block_dhidden = 256
    args.dataflow_map2_weight_stages = profile.get("map2_weight_stages")
    args.dataflow_map2_weight_max_outstanding = 2
    args.dataflow_cold_map2_weight_stages = 2
    args.dataflow_cold_map2_weight_max_outstanding = 2
    args.dataflow_fuse_map2_hidden_tiles = profile.get("fuse_map2_hidden_tiles", False)
    args.dataflow_reshared_policy = profile.get("reshared_policy", "cluster_shared_all_gather")
    args.dataflow_cold_reshared_policy = profile.get("cold_reshared_policy")
    args.dataflow_cold_launch_delay_us = 0.0
    args.dataflow_cold_launch_stagger_us = 0.0
    args.dataflow_cold_launch_stagger_group_clusters = 1
    args.dataflow_active_only = False
    args.dataflow_walltime_repeats = 30
    args.dataflow_map1_input_mode = "unicast"
    args.dataflow_cold_map1_input_mode = None
    args.dataflow_map2_map1_lookahead_stages = profile.get("map2_map1_lookahead_stages")
    args.no_dataflow_fp8 = False
    args.no_dataflow_tma_weights = False


def parse_args_from_list(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile routed MoE kernels with expert heat scaled into grouped-GEMM inputs.",
    )
    parser.add_argument(
        "--backend",
        choices=("dataflow", "dataflow-dual", "both", "all"),
        default=DEFAULT_BACKEND,
    )
    parser.add_argument(
        "--published-h100-workload",
        choices=tuple(PUBLISHED_H100_WORKLOADS),
        help=("Apply one exact workload and 5-warmup/30-sample configuration from examples/dataflow/current_results.md."),
    )
    parser.add_argument("--token-num", type=int, nargs="+", default=list(DEFAULT_TOKEN_NUMS))
    parser.add_argument("--d-hidden", type=int, default=DEFAULT_D_HIDDEN)
    parser.add_argument("--d-expert", type=int, default=DEFAULT_D_EXPERT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--block-dhidden", type=int, default=DEFAULT_BLOCK_DHIDDEN)
    parser.add_argument("--block-dexpert", type=int, default=DEFAULT_BLOCK_DEXPERT)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--disable-cache", action="store_true")

    parser.add_argument("--dataflow-sm-count", type=int, default=DEFAULT_DATAFLOW_SM_COUNT)
    parser.add_argument("--dataflow-cluster-size", type=int, default=DEFAULT_DATAFLOW_CLUSTER_SIZE)
    parser.add_argument("--dataflow-hot-sm-count", type=int, default=DEFAULT_DATAFLOW_HOT_SM_COUNT)
    parser.add_argument(
        "--dataflow-hot-cluster-size",
        type=parse_positive_int_or_auto,
        default=None,
        metavar="SIZE|auto",
        help=(
            "Hot cluster size. Default 'auto' scores legal topologies from task waves, expert-tile efficiency, and active SM utilization."
        ),
    )
    parser.add_argument("--dataflow-cold-sm-count", type=int, default=DEFAULT_DATAFLOW_COLD_SM_COUNT)
    parser.add_argument("--dataflow-cold-cluster-size", type=int, default=DEFAULT_DATAFLOW_COLD_CLUSTER_SIZE)
    parser.add_argument(
        "--dataflow-cold-expert-count",
        type=parse_cold_expert_count,
        default=DEFAULT_DATAFLOW_COLD_EXPERT_COUNT,
        metavar="COUNT|auto",
        help=(
            "Number of least-loaded experts assigned to the cold kernel. "
            "Default 'auto' minimizes hot/cold task waves and modeled cluster load."
        ),
    )
    parser.add_argument(
        "--dataflow-dual-timing-batch-size",
        type=int,
        default=DEFAULT_DATAFLOW_DUAL_TIMING_BATCH_SIZE,
    )
    parser.add_argument("--dataflow-block-token", type=int, default=DEFAULT_DATAFLOW_BLOCK_TOKEN)
    parser.add_argument(
        "--dataflow-cold-block-token",
        type=int,
        default=DEFAULT_DATAFLOW_COLD_BLOCK_TOKEN,
    )
    parser.add_argument(
        "--dataflow-cold-block-dhidden",
        type=int,
        default=DEFAULT_DATAFLOW_COLD_BLOCK_DHIDDEN,
    )
    parser.add_argument(
        "--dataflow-cold-map2-block-dhidden",
        type=int,
        default=DEFAULT_DATAFLOW_COLD_MAP2_BLOCK_DHIDDEN,
    )
    parser.add_argument(
        "--dataflow-cold-map2-handler-dhidden",
        type=int,
        help="Fuse this many hidden columns into each cold map2 handler; defaults to at most seven tiles.",
    )
    parser.add_argument(
        "--dataflow-cold-block-dexpert",
        type=int,
        default=DEFAULT_DATAFLOW_COLD_BLOCK_DEXPERT,
    )
    parser.add_argument(
        "--dataflow-cold-fuse-map1-expert-shards",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fuse all cold map1 expert tiles owned by one CTA. The default selects fusion only when the expert shard has equal-width tiles."
        ),
    )
    parser.add_argument(
        "--dataflow-cold-map1-gate-stages",
        type=int,
        choices=(2, 3),
        help=("Cold map1 gate/up shared-memory ring depth. The default keeps three stages for fused small-token handlers."),
    )
    parser.add_argument(
        "--dataflow-cold-weight-eviction-policy",
        choices=("evict_normal", "evict_first", "evict_last"),
        default="evict_first",
        help="TMA L2 eviction hint for cold gate/up/down weights.",
    )
    parser.add_argument(
        "--dataflow-hot-map1-consumer-threads",
        type=int,
        choices=(128, 256),
        default=DEFAULT_DATAFLOW_HOT_MAP1_CONSUMER_THREADS,
        help=(
            "Use one or two hot map1 WGMMA consumer warp groups; by default, "
            "TMA input uses one group and other input paths preserve their existing layout."
        ),
    )
    parser.add_argument(
        "--dataflow-hot-map1-wgmma-wait-depth",
        type=parse_wgmma_wait_depth,
        default=None,
        metavar="DEPTH|auto",
        help=(
            "Hot map1 WGMMA groups kept in flight. Default 'auto' enables the "
            "pipeline only when its path, work, and shared-memory constraints hold."
        ),
    )
    parser.add_argument(
        "--dataflow-hot-map2-handler-dhidden",
        type=int,
        help="Fuse this many hidden columns into each hot map2 handler.",
    )
    parser.add_argument(
        "--dataflow-hot-map1-handler-dexpert",
        type=int,
        help="Fuse this many expert columns into each hot map1 handler.",
    )
    parser.add_argument(
        "--dataflow-map2-block-dhidden",
        type=int,
        default=DEFAULT_DATAFLOW_MAP2_BLOCK_DHIDDEN,
    )
    parser.add_argument(
        "--dataflow-map2-weight-stages",
        type=int,
        choices=(1, 2, 4, 8),
        default=DEFAULT_DATAFLOW_MAP2_WEIGHT_STAGES,
    )
    parser.add_argument(
        "--dataflow-map2-weight-max-outstanding",
        type=int,
        default=DEFAULT_DATAFLOW_MAP2_WEIGHT_MAX_OUTSTANDING,
        help=("Limit map2 weight TMA producer lead independently of the physical shared-memory ring depth."),
    )
    parser.add_argument(
        "--dataflow-cold-map2-weight-stages",
        type=int,
        choices=(1, 2, 4, 8),
        default=DEFAULT_DATAFLOW_COLD_MAP2_WEIGHT_STAGES,
    )
    parser.add_argument(
        "--dataflow-cold-map2-weight-max-outstanding",
        type=int,
        default=DEFAULT_DATAFLOW_COLD_MAP2_WEIGHT_MAX_OUTSTANDING,
        help="Use a smaller cold-kernel TMA lead to preserve hot-kernel memory-service progress.",
    )
    parser.add_argument(
        "--dataflow-fuse-map2-hidden-tiles",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DATAFLOW_FUSE_MAP2_HIDDEN_TILES,
        help=("Fuse the two hot map2 hidden tiles so one activation traversal feeds the complete per-SM output shard."),
    )
    parser.add_argument("--dataflow-reshared-policy", default="cluster_shared_all_gather")
    parser.add_argument(
        "--dataflow-cold-reshared-policy",
        help="Override the reshared policy for the dual-kernel cold bucket.",
    )
    parser.add_argument(
        "--dataflow-cold-launch-delay-us",
        type=float,
        default=0.0,
        help="Delay the cold persistent kernel at entry using the GPU global timer.",
    )
    parser.add_argument(
        "--dataflow-cold-launch-stagger-us",
        type=float,
        default=0.0,
        help="Add this per-cluster cold launch offset using the GPU global timer.",
    )
    parser.add_argument(
        "--dataflow-cold-launch-stagger-group-clusters",
        type=int,
        default=1,
        help="Apply each cold launch stagger offset to this many clusters.",
    )
    parser.add_argument("--dataflow-active-only", action="store_true")
    parser.add_argument("--dataflow-walltime-dir")
    parser.add_argument("--dataflow-walltime-repeats", type=int, default=1)
    parser.add_argument(
        "--dataflow-map1-input-mode",
        choices=("multicast", "unicast", "cooperative"),
        default=DEFAULT_DATAFLOW_MAP1_INPUT_MODE,
    )
    parser.add_argument(
        "--dataflow-cold-map1-input-mode",
        choices=("multicast", "unicast", "cooperative"),
        default=DEFAULT_DATAFLOW_COLD_MAP1_INPUT_MODE,
        help="Override map1 input loading for the dual-kernel cold bucket.",
    )
    parser.add_argument(
        "--dataflow-map2-map1-lookahead-stages",
        type=parse_lookahead_stages,
        default=None,
        metavar="STAGES|auto",
        help=("Prefetch stages for the next hot task. Default 'auto' enables two stages only for a compatible single/fused map2 handler."),
    )
    parser.add_argument("--dump-dataflow-source")
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--no-dataflow-fp8", action="store_true")
    parser.add_argument("--no-dataflow-tma-weights", action="store_true")

    args = parser.parse_args(argv)
    if args.published_h100_workload is not None and args.dataflow_walltime_dir is None:
        parser.error("--published-h100-workload requires --dataflow-walltime-dir")
    apply_published_h100_workload(args)
    return args


def parse_args() -> argparse.Namespace:
    return parse_args_from_list()


def print_result(row: dict[str, Any]) -> None:
    common = (
        f"RESULT backend={row['backend']} token_num={row['token_num']} "
        f"group_sum={row['group_sum']} nonzero_experts={row['nonzero_experts']} "
        f"max_group={row['max_group_size']} min_nonzero_group={row['min_nonzero_group_size']} "
        f"group_blocks={row['group_blocks']} block_token={row['block_token']} "
        f"median_ms={row['median_ms']:.6f} mean_ms={row['mean_ms']:.6f} "
        f"min_ms={row['min_ms']:.6f} max_ms={row['max_ms']:.6f} "
        f"stddev_ms={row['stddev_ms']:.6f} dtype={row['dtype']}"
    )
    if row["backend"] == "dataflow":
        print(
            common
            + f" sm_count={row['sm_count']} cluster_size={row['cluster_size']} "
            + f"reshared_policy={row.get('reshared_policy')} "
            + f"map2_weight_stages={row.get('map2_weight_stages')} "
            + f"map2_weight_max_outstanding={row.get('map2_weight_max_outstanding')} "
            + f"fuse_map2_hidden_tiles={row.get('fuse_map2_hidden_tiles')} "
            + f"map1_input_mode={row.get('map1_input_mode')} "
            + f"map2_map1_lookahead_stages={row.get('map2_map1_lookahead_stages', 0)} "
            + f"map2_block_dhidden={row.get('map2_block_dhidden', DEFAULT_DATAFLOW_MAP2_BLOCK_DHIDDEN)} "
            + f"map2_handler_dhidden={row.get('map2_handler_dhidden')} "
            + f"active_only={row.get('active_only', False)} "
            + f"active_blocks={row.get('active_blocks', row['group_blocks'])} "
            + f"scheduled_blocks={row.get('scheduled_blocks', row['group_blocks'])} "
            + f"task_cluster_assignment={row.get('task_cluster_assignment', ())} "
            + f"cluster_task_counts={row.get('cluster_task_counts', ())} "
            + f"task_cost_model={row.get('task_cost_model')} "
            + f"task_costs_us={row.get('task_costs_us', ())} "
            + f"cluster_task_costs_us={row.get('cluster_task_costs_us', ())} "
            + f"instructions={row['instructions']} comms={row['comms']} "
            + f"cluster_send={row['cluster_send']} cluster_recv={row['cluster_recv']} "
            + f"shared_memory_bytes={row['shared_memory_bytes']} "
            + f"shared_slot_bytes={row['shared_slot_bytes']} barrier_count={row['barrier_count']}"
        )
        print(f"SETUP_NOT_COUNTED backend=dataflow token_num={row['token_num']} {row['setup_timings_ms']}")
        if row.get("source_path"):
            print(f"ARTIFACT backend=dataflow source={row['source_path']}")
        if row.get("walltime_detail_csv") or row.get("walltime_summary_csv"):
            print(f"WALLTIME_CSV backend=dataflow details={row.get('walltime_detail_csv')} summary={row.get('walltime_summary_csv')}")
        if row.get("walltime_report"):
            print(row["walltime_report"])
    elif row["backend"] == "dataflow-dual":
        print(
            common
            + f" hot={row['hot_sm_count']}/{row['hot_cluster_size']} "
            + f"hot_cluster_size_requested={row.get('hot_cluster_size_requested')} "
            + f"hot_topology_selection_score={row.get('hot_topology_selection_score')} "
            + f"cold={row['cold_sm_count']}/{row['cold_cluster_size']} "
            + f"hot_reshared_policy={row.get('hot_reshared_policy')} "
            + f"cold_reshared_policy={row.get('cold_reshared_policy')} "
            + f"map2_weight_stages={row.get('map2_weight_stages')} "
            + f"map2_weight_max_outstanding={row.get('map2_weight_max_outstanding')} "
            + f"cold_map2_weight_stages={row.get('cold_map2_weight_stages')} "
            + "cold_map2_weight_max_outstanding="
            + f"{row.get('cold_map2_weight_max_outstanding')} "
            + f"fuse_map2_hidden_tiles={row.get('fuse_map2_hidden_tiles')} "
            + f"cold_launch_delay_us={row.get('cold_launch_delay_us', 0.0)} "
            + f"cold_launch_stagger_us={row.get('cold_launch_stagger_us', 0.0)} "
            + "cold_launch_stagger_group_clusters="
            + f"{row.get('cold_launch_stagger_group_clusters', 1)} "
            + f"hot_map1_input_mode={row.get('hot_map1_input_mode')} "
            + f"cold_map1_input_mode={row.get('cold_map1_input_mode')} "
            + f"hot_map1_consumer_threads={row.get('hot_map1_consumer_threads')} "
            + f"hot_map1_handler_dexpert={row.get('hot_map1_handler_dexpert')} "
            + "hot_map1_wgmma_wait_depth="
            + f"{row.get('hot_map1_wgmma_wait_depth', 0)} "
            + "hot_map1_wgmma_wait_depth_requested="
            + f"{row.get('hot_map1_wgmma_wait_depth_requested')} "
            + "hot_map1_pipeline_selection="
            + f"{row.get('hot_map1_pipeline_selection')} "
            + f"hot_map2_handler_dhidden={row.get('hot_map2_handler_dhidden')} "
            + f"hot_map2_map1_lookahead_stages={row.get('hot_map2_map1_lookahead_stages', 0)} "
            + f"cold_expert_count={row['cold_expert_count']} "
            + "cold_expert_count_requested="
            + f"{row.get('cold_expert_count_requested')} "
            + "dual_schedule_selection_score="
            + f"{row.get('dual_schedule_selection_score')} "
            + f"cold_expert_ids={row['cold_expert_ids']} "
            + f"timing_metric={row['timing_metric']} "
            + f"timing_batch_size={row['timing_batch_size']} "
            + f"hot_solo_median_ms={row['hot_solo_median_ms']:.6f} "
            + f"cold_solo_median_ms={row['cold_solo_median_ms']:.6f} "
            + f"hot_block_token={row['hot_block_token']} cold_block_token={row['cold_block_token']} "
            + f"cold_block_dhidden={row['cold_block_dhidden']} "
            + f"cold_map2_block_dhidden={row['cold_map2_block_dhidden']} "
            + f"cold_map2_handler_dhidden={row['cold_map2_handler_dhidden']} "
            + f"cold_block_dexpert={row['cold_block_dexpert']} "
            + "cold_fuse_map1_expert_shards="
            + f"{row.get('cold_fuse_map1_expert_shards', False)} "
            + f"cold_map1_gate_stages={row.get('cold_map1_gate_stages')} "
            + "cold_weight_eviction_policy="
            + f"{row.get('cold_weight_eviction_policy')} "
            + f"hot_blocks={row['hot_blocks']} cold_blocks={row['cold_blocks']} "
            + f"hot_cluster_task_counts={row['hot_cluster_task_counts']} "
            + f"cold_cluster_task_counts={row['cold_cluster_task_counts']} "
            + f"cold_group_blocks={row['cold_group_blocks']} "
            + f"hot_instructions={row['hot_instructions']} cold_instructions={row['cold_instructions']} "
            + f"hot_comms={row['hot_comms']} cold_comms={row['cold_comms']} "
            + f"hot_cluster_send={row['hot_cluster_send']} hot_cluster_recv={row['hot_cluster_recv']} "
            + f"cold_cluster_send={row['cold_cluster_send']} cold_cluster_recv={row['cold_cluster_recv']} "
            + f"hot_shared_memory_bytes={row['hot_shared_memory_bytes']} "
            + f"cold_shared_memory_bytes={row['cold_shared_memory_bytes']} "
            + f"hot_shared_slot_bytes={row['hot_shared_slot_bytes']} "
            + f"cold_shared_slot_bytes={row['cold_shared_slot_bytes']} "
            + f"hot_barrier_count={row['hot_barrier_count']} "
            + f"cold_barrier_count={row['cold_barrier_count']}"
        )
        print(
            f"SETUP_NOT_COUNTED backend=dataflow-dual token_num={row['token_num']} "
            f"hot={row['hot_setup_timings_ms']} cold={row['cold_setup_timings_ms']}"
        )
        if row.get("hot_source_path") or row.get("cold_source_path"):
            print(f"ARTIFACT backend=dataflow-dual hot_source={row.get('hot_source_path')} cold_source={row.get('cold_source_path')}")
        if row.get("dual_walltime_detail_csv"):
            print(
                f"WALLTIME_CSV backend=dataflow-dual "
                f"details={row.get('dual_walltime_detail_csv')} "
                f"summary={row.get('dual_walltime_summary_csv')}"
            )
        if row.get("dual_walltime_report"):
            print(row["dual_walltime_report"])
        if row.get("verified_output"):
            print(
                f"VERIFIED backend=dataflow-dual token_num={row['token_num']} "
                f"max_abs_error={row['verification_max_abs_error']:.6f} "
                f"mean_abs_error={row['verification_mean_abs_error']:.6f} "
                f"rmse={row['verification_rmse']:.6f} "
                f"relative_l2={row['verification_relative_l2']:.6f} "
                f"concurrent_max_abs_error="
                f"{row['verification_concurrent_max_abs_error']:.6f} "
                f"instrumented_max_abs_error="
                f"{row.get('verification_instrumented_max_abs_error')}"
            )


def main() -> None:
    args = parse_args()
    if args.published_h100_workload is not None:
        for name in tuple(os.environ):
            if name.startswith("DATAFLOW_"):
                os.environ.pop(name)
    if args.disable_cache:
        import tilelang

        tilelang.disable_cache()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    props = torch.cuda.get_device_properties(0)
    print(f"DEVICE name={props.name} sm_count={props.multi_processor_count} capability={props.major}.{props.minor}")
    print(
        "CONFIG "
        f"backend={args.backend} token_num={args.token_num} n_experts={len(EXPERT_HEAT)} top_k={args.top_k} "
        f"d_hidden={args.d_hidden} d_expert={args.d_expert} "
        f"warmups={args.warmups} repeats={args.repeats} metric=cuda_event_kernel_only flush_l2_cache=False"
    )

    for token_num in args.token_num:
        for backend in selected_backends(args.backend):
            if backend == "dataflow":
                row = profile_dataflow_routed_moe(
                    token_num,
                    d_hidden=args.d_hidden,
                    d_expert=args.d_expert,
                    top_k=args.top_k,
                    block_token=args.dataflow_block_token,
                    block_dhidden=args.block_dhidden,
                    map2_block_dhidden=args.dataflow_map2_block_dhidden,
                    map2_handler_dhidden=(args.dataflow_hot_map2_handler_dhidden),
                    block_dexpert=args.block_dexpert,
                    sm_count=args.dataflow_sm_count,
                    cluster_size=args.dataflow_cluster_size,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    use_fp8=not args.no_dataflow_fp8,
                    use_tma_weights=not args.no_dataflow_tma_weights,
                    reshared_policy=args.dataflow_reshared_policy,
                    map2_weight_stages=args.dataflow_map2_weight_stages,
                    map2_weight_max_outstanding=(args.dataflow_map2_weight_max_outstanding),
                    fuse_map2_hidden_tiles=(args.dataflow_fuse_map2_hidden_tiles),
                    active_only=args.dataflow_active_only,
                    walltime_dir=args.dataflow_walltime_dir,
                    walltime_repeats=args.dataflow_walltime_repeats,
                    dump_dataflow_source=args.dump_dataflow_source,
                    map1_input_mode=args.dataflow_map1_input_mode,
                    map2_map1_lookahead_stages=(args.dataflow_map2_map1_lookahead_stages),
                )
            elif backend == "dataflow-dual":
                row = profile_dataflow_dual_routed_moe(
                    token_num,
                    d_hidden=args.d_hidden,
                    d_expert=args.d_expert,
                    top_k=args.top_k,
                    block_token=args.dataflow_block_token,
                    cold_block_token=args.dataflow_cold_block_token,
                    block_dhidden=args.block_dhidden,
                    map2_block_dhidden=args.dataflow_map2_block_dhidden,
                    block_dexpert=args.block_dexpert,
                    cold_block_dhidden=args.dataflow_cold_block_dhidden,
                    cold_map2_block_dhidden=args.dataflow_cold_map2_block_dhidden,
                    cold_map2_handler_dhidden=args.dataflow_cold_map2_handler_dhidden,
                    cold_block_dexpert=args.dataflow_cold_block_dexpert,
                    cold_fuse_map1_expert_shards=(args.dataflow_cold_fuse_map1_expert_shards),
                    cold_map1_gate_stages=(args.dataflow_cold_map1_gate_stages),
                    cold_weight_eviction_policy=(args.dataflow_cold_weight_eviction_policy),
                    hot_sm_count=args.dataflow_hot_sm_count,
                    hot_cluster_size=args.dataflow_hot_cluster_size,
                    cold_sm_count=args.dataflow_cold_sm_count,
                    cold_cluster_size=args.dataflow_cold_cluster_size,
                    cold_expert_count=args.dataflow_cold_expert_count,
                    hot_map1_consumer_threads=(args.dataflow_hot_map1_consumer_threads),
                    hot_map1_handler_dexpert=(args.dataflow_hot_map1_handler_dexpert),
                    hot_map1_wgmma_wait_depth=(args.dataflow_hot_map1_wgmma_wait_depth),
                    hot_map2_handler_dhidden=(args.dataflow_hot_map2_handler_dhidden),
                    warmups=args.warmups,
                    repeats=args.repeats,
                    timing_batch_size=args.dataflow_dual_timing_batch_size,
                    use_fp8=not args.no_dataflow_fp8,
                    use_tma_weights=not args.no_dataflow_tma_weights,
                    reshared_policy=args.dataflow_reshared_policy,
                    cold_reshared_policy=args.dataflow_cold_reshared_policy,
                    map2_weight_stages=args.dataflow_map2_weight_stages,
                    map2_weight_max_outstanding=(args.dataflow_map2_weight_max_outstanding),
                    cold_map2_weight_stages=(args.dataflow_cold_map2_weight_stages),
                    cold_map2_weight_max_outstanding=(args.dataflow_cold_map2_weight_max_outstanding),
                    fuse_map2_hidden_tiles=(args.dataflow_fuse_map2_hidden_tiles),
                    walltime_dir=args.dataflow_walltime_dir,
                    walltime_repeats=args.dataflow_walltime_repeats,
                    dump_dataflow_source=args.dump_dataflow_source,
                    verify_output=args.verify_output,
                    map1_input_mode=args.dataflow_map1_input_mode,
                    cold_map1_input_mode=(args.dataflow_cold_map1_input_mode),
                    map2_map1_lookahead_stages=(args.dataflow_map2_map1_lookahead_stages),
                    cold_launch_delay_us=(args.dataflow_cold_launch_delay_us),
                    cold_launch_stagger_us=(args.dataflow_cold_launch_stagger_us),
                    cold_launch_stagger_group_clusters=(args.dataflow_cold_launch_stagger_group_clusters),
                )
            print_result(row)
            if args.published_h100_workload is not None:
                p50_us = float(row["dual_walltime_p50_us"])
                reference_p50_us = PUBLISHED_H100_P50_US[args.published_h100_workload]
                regression_percent = (p50_us / reference_p50_us - 1.0) * 100.0
                print(
                    "PUBLISHED_RESULT "
                    f"workload={args.published_h100_workload} "
                    "metric=percent_globaltimer_concurrent_global_span_us "
                    f"p50_us={p50_us:.3f} "
                    f"reference_p50_us={reference_p50_us:.3f} "
                    f"regression_percent={regression_percent:+.3f} "
                    f"gate_percent={PUBLISHED_PER_POINT_GATE_PERCENT:.1f} "
                    f"passes={regression_percent <= PUBLISHED_PER_POINT_GATE_PERCENT} "
                    f"warmups={args.warmups} samples={args.dataflow_walltime_repeats}"
                )
                if regression_percent > PUBLISHED_PER_POINT_GATE_PERCENT:
                    raise RuntimeError(
                        f"{args.published_h100_workload} regressed "
                        f"{regression_percent:.3f}% against the published "
                        f"{reference_p50_us:.3f} us result"
                    )


if __name__ == "__main__":
    main()
