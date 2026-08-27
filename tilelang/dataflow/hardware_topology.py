"""Print GPU SM die topology and GPC topology."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal
from collections.abc import Sequence

from .die_sm_probe import DieSmGroups, get_gpu_die_sm_groups
from .gpc_sm_probe import GpcSmGroup, GpuGpcSmGroups, get_gpu_gpc_sm_groups


@dataclass(frozen=True)
class GpuHardwareTopology:
    die_groups: DieSmGroups
    gpc_groups: GpuGpcSmGroups

    def __post_init__(self) -> None:
        gpc_distribution_by_die(self.die_groups, self.gpc_groups)

    def to_dict(self) -> dict[str, object]:
        gpc_distribution = gpc_distribution_by_die(
            self.die_groups,
            self.gpc_groups,
        )
        return {
            "device_id": self.gpc_groups.device_id,
            "device_name": self.gpc_groups.device_name or self.die_groups.device_name,
            "sm_count": self.gpc_groups.sm_count,
            "die_topology": self.die_groups.to_dict(),
            "gpc_topology": self.gpc_groups.to_dict(),
            "gpc_die_assignment": [
                {
                    "gpc_id": group.gpc_id,
                    "die": die.die_id,
                    "size": group.size,
                    "sm_ids": list(group.sm_ids),
                }
                for die, groups in zip(self.die_groups.dies, gpc_distribution)
                for group in groups
            ],
            "gpc_distribution_by_die": [
                {
                    "die_id": die.die_id,
                    "gpc_count": len(groups),
                    "sm_count": sum(group.size for group in groups),
                    "gpcs": [
                        {
                            "gpc_id": group.gpc_id,
                            "size": group.size,
                            "sm_ids": list(group.sm_ids),
                        }
                        for group in groups
                    ],
                }
                for die, groups in zip(self.die_groups.dies, gpc_distribution)
            ],
        }


def probe_gpu_topology(
    device_id: int = 0,
    *,
    die_anchor_count: int = 8,
    die_rounds: int = 5,
    die_evict_bytes: int = 512 * 1024 * 1024,
    die_anchor_stride_bytes: int = 2 * 1024 * 1024,
    die_shared_mem_bytes: int | None = None,
    die_min_gap_cycles: float = 80.0,
    die_mode: Literal["auto", "split", "single"] = "auto",
    die_print_latency: bool = False,
    gpc_rounds_per_cluster_size: int = 16,
    gpc_waves_per_cluster_size: int = 4,
    gpc_thread_per_block: int | None = None,
    gpc_shared_mem_bytes: int | None = None,
    gpc_spin_cycles: int = 10_000,
    expected_sm_count: int | None = None,
    nvcc: str = "nvcc",
    cuda_arch: str = "native",
) -> GpuHardwareTopology:
    """Measure both die and GPC topology views and return structured metadata."""

    die_groups = get_gpu_die_sm_groups(
        device_id=device_id,
        anchor_count=die_anchor_count,
        rounds=die_rounds,
        evict_bytes=die_evict_bytes,
        anchor_stride_bytes=die_anchor_stride_bytes,
        shared_mem_bytes=die_shared_mem_bytes,
        min_gap_cycles=die_min_gap_cycles,
        expected_sm_count=expected_sm_count,
        die_mode=die_mode,
        print_latency=die_print_latency,
    )
    gpc_groups = get_gpu_gpc_sm_groups(
        device_id=device_id,
        rounds_per_cluster_size=gpc_rounds_per_cluster_size,
        waves_per_cluster_size=gpc_waves_per_cluster_size,
        thread_per_block=gpc_thread_per_block,
        shared_mem_bytes=gpc_shared_mem_bytes,
        spin_cycles=gpc_spin_cycles,
        expected_sm_count=expected_sm_count,
        nvcc=nvcc,
        cuda_arch=cuda_arch,
    )
    return GpuHardwareTopology(die_groups=die_groups, gpc_groups=gpc_groups)


def render_gpu_topology(
    die_groups: DieSmGroups,
    gpc_groups: GpuGpcSmGroups,
    *,
    verbose: bool = False,
) -> str:
    """Render a readable die/GPC topology report."""

    gpc_distribution = gpc_distribution_by_die(die_groups, gpc_groups)
    lines = [
        "=== GPU SM Die Topology ===",
        device_summary(gpc_groups, die_groups),
    ]
    if verbose:
        lines.append(f"Die classification: {die_groups.anchor_count} anchors, median gap {die_groups.median_gap_cycles:.1f} cycles")

    for die in die_groups.dies:
        lines.append(f"Die {die.die_id}: {die.size} SMs")
        lines.append(f"  SM IDs: {format_sm_ids(die.sm_ids)}")

    lines.append("")
    lines.extend(render_gpc_lines(gpc_groups, die_groups=die_groups, verbose=verbose))
    lines.extend(render_gpc_distribution_lines(die_groups, gpc_distribution))
    return "\n".join(lines)


def render_gpu_gpc_topology(
    gpc_groups: GpuGpcSmGroups,
    *,
    verbose: bool = False,
) -> str:
    """Render only the GPC topology report."""

    return "\n".join(render_gpc_lines(gpc_groups, die_groups=None, verbose=verbose))


def render_gpc_lines(
    gpc_groups: GpuGpcSmGroups,
    *,
    die_groups: DieSmGroups | None,
    verbose: bool,
) -> list[str]:
    lines = [
        "=== GPU / GPC Topology ===",
        device_summary(gpc_groups, die_groups),
        f"GPC count: {gpc_groups.gpc_count}",
        f"GPC sizes: {' '.join(str(size) for size in gpc_groups.gpc_sizes)}",
    ]

    if verbose:
        lines.extend(
            [
                f"Max cluster size: {gpc_groups.max_cluster_size}",
                "Cluster occupancy curve:",
            ]
        )
        for cluster_size, active_clusters in gpc_groups.active_clusters_by_size:
            lines.append(f"  cluster_size {cluster_size} -> active_clusters {active_clusters}")
        lines.append(f"Kernel samples: {gpc_groups.cluster_sample_count} clusters, {gpc_groups.kernel_record_count} block records")

    lines.append("GPC SM groups:")
    for group in gpc_groups.gpcs:
        if die_groups is None:
            lines.append(f"  GPC {group.gpc_id}: {group.size} SMs")
        else:
            die_label = die_label_for_sm_ids(group.sm_ids, die_groups)
            lines.append(f"  GPC {group.gpc_id}: {group.size} SMs, {die_label}")
        lines.append(f"    SM IDs: {format_sm_ids(group.sm_ids)}")

    return lines


def device_summary(
    gpc_groups: GpuGpcSmGroups,
    die_groups: DieSmGroups | None,
) -> str:
    device_name = gpc_groups.device_name
    if device_name is None and die_groups is not None:
        device_name = die_groups.device_name
    if device_name:
        return f"Device {gpc_groups.device_id} ({device_name}): {gpc_groups.sm_count} SMs"
    return f"Device {gpc_groups.device_id}: {gpc_groups.sm_count} SMs"


def format_sm_ids(sm_ids: Sequence[int]) -> str:
    return " ".join(str(smid) for smid in sm_ids)


def render_gpc_distribution_lines(
    die_groups: DieSmGroups,
    gpc_distribution: Sequence[Sequence[GpcSmGroup]],
) -> list[str]:
    lines = ["", "Die -> GPC distribution:"]
    for die, groups in zip(die_groups.dies, gpc_distribution):
        gpc_word = "GPC" if len(groups) == 1 else "GPCs"
        sm_count = sum(group.size for group in groups)
        lines.append(f"  Die {die.die_id}: {len(groups)} {gpc_word}, {sm_count} SMs")
        for group in groups:
            lines.append(f"    GPC {group.gpc_id}: {group.size} SMs, SM IDs: {format_sm_ids(group.sm_ids)}")
    return lines


def gpc_distribution_by_die(
    die_groups: DieSmGroups,
    gpc_groups: GpuGpcSmGroups,
) -> tuple[tuple[GpcSmGroup, ...], ...]:
    sm_to_die: dict[int, int] = {}
    distribution: dict[int, list[GpcSmGroup]] = {die.die_id: [] for die in die_groups.dies}

    for die in die_groups.dies:
        for smid in die.sm_ids:
            if smid in sm_to_die:
                raise RuntimeError(f"SM {smid} appears in both die {sm_to_die[smid]} and die {die.die_id}")
            sm_to_die[smid] = die.die_id

    for group in gpc_groups.gpcs:
        smid_set = set(group.sm_ids)
        unknown_hits = sorted(smid for smid in smid_set if smid not in sm_to_die)
        if unknown_hits:
            raise RuntimeError(f"GPC {group.gpc_id} has SM IDs outside the die topology: {format_sm_ids(tuple(unknown_hits))}")

        hits_by_die: dict[int, list[int]] = {}
        for smid in sorted(smid_set):
            hits_by_die.setdefault(sm_to_die[smid], []).append(smid)

        if len(hits_by_die) > 1:
            die_ids = sorted(hits_by_die)
            details = ", ".join(f"die {die_id} SM IDs {format_sm_ids(tuple(hits_by_die[die_id]))}" for die_id in die_ids)
            die_pair = " and ".join(f"die {die_id}" for die_id in die_ids)
            raise RuntimeError(f"GPC {group.gpc_id} spans {die_pair}: {details}")
        if not hits_by_die:
            raise RuntimeError(f"GPC {group.gpc_id} has no SM IDs")

        die_id = next(iter(hits_by_die))
        distribution[die_id].append(group)

    return tuple(tuple(distribution[die.die_id]) for die in die_groups.dies)


def die_label_for_sm_ids(sm_ids: Sequence[int], die_groups: DieSmGroups) -> str:
    smid_set = set(sm_ids)
    hits: list[int] = []
    for die in die_groups.dies:
        if smid_set & set(die.sm_ids):
            hits.append(die.die_id)

    if len(hits) == 1:
        return f"die {hits[0]}"
    if len(hits) > 1:
        return "mixed " + "/".join(f"die {die_id}" for die_id in hits)
    return "die unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gpc-only", action="store_true")
    parser.add_argument("--expected-sm-count", type=int)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuda-arch", default="native")
    parser.add_argument("--die-mode", choices=("auto", "split", "single"), default="auto")
    parser.add_argument("--die-anchors", type=int, default=8)
    parser.add_argument("--die-rounds", type=int, default=5)
    parser.add_argument("--die-evict-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--die-anchor-stride-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--die-shared-mem-bytes", type=int)
    parser.add_argument("--die-min-gap-cycles", type=float, default=80.0)
    parser.add_argument("--die-print-latency", action="store_true")
    parser.add_argument("--gpc-rounds", type=int, default=16)
    parser.add_argument("--gpc-waves", type=int, default=4)
    parser.add_argument("--gpc-thread-per-block", type=int)
    parser.add_argument("--gpc-shared-mem-bytes", type=int)
    parser.add_argument("--gpc-spin-cycles", type=int, default=10_000)
    args = parser.parse_args(argv)

    if args.gpc_only:
        gpc_groups = get_gpu_gpc_sm_groups(
            device_id=args.device,
            rounds_per_cluster_size=args.gpc_rounds,
            waves_per_cluster_size=args.gpc_waves,
            thread_per_block=args.gpc_thread_per_block,
            shared_mem_bytes=args.gpc_shared_mem_bytes,
            spin_cycles=args.gpc_spin_cycles,
            expected_sm_count=args.expected_sm_count,
            nvcc=args.nvcc,
            cuda_arch=args.cuda_arch,
        )
        if args.json:
            print(json.dumps(gpc_groups.to_dict(), indent=2))
        else:
            print(render_gpu_gpc_topology(gpc_groups, verbose=args.verbose))
        return 0

    topology = probe_gpu_topology(
        device_id=args.device,
        die_anchor_count=args.die_anchors,
        die_rounds=args.die_rounds,
        die_evict_bytes=args.die_evict_bytes,
        die_anchor_stride_bytes=args.die_anchor_stride_bytes,
        die_shared_mem_bytes=args.die_shared_mem_bytes,
        die_min_gap_cycles=args.die_min_gap_cycles,
        die_mode=args.die_mode,
        die_print_latency=args.die_print_latency,
        gpc_rounds_per_cluster_size=args.gpc_rounds,
        gpc_waves_per_cluster_size=args.gpc_waves,
        gpc_thread_per_block=args.gpc_thread_per_block,
        gpc_shared_mem_bytes=args.gpc_shared_mem_bytes,
        gpc_spin_cycles=args.gpc_spin_cycles,
        expected_sm_count=args.expected_sm_count,
        nvcc=args.nvcc,
        cuda_arch=args.cuda_arch,
    )

    if args.json:
        print(json.dumps(topology.to_dict(), indent=2))
    else:
        print(
            render_gpu_topology(
                topology.die_groups,
                topology.gpc_groups,
                verbose=args.verbose,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
