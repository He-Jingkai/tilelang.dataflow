"""Find the SM IDs that belong to each GPU die.

Some modern NVIDIA packages expose one logical CUDA device while placing SMs
on multiple dies.  The concrete ``%smid`` labels can vary across cards, so the
mapping should be measured per device instead of hard-coded from a SKU name.

This module does that measurement with ordinary CUDA kernels JIT-compiled by
CuPy.  The public function is ``get_gpu_die_sm_ids``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from statistics import median
from typing import Literal, TextIO
from collections.abc import Iterable, Sequence


_MIN_DIE_GROUP_FRACTION = 0.25


_CUDA_SOURCE = r"""
extern "C" __device__ __forceinline__ unsigned int read_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

extern "C" __device__ __forceinline__ unsigned int load_global_cv_u32(
    const unsigned int* ptr
) {
    unsigned int value;
    asm volatile("ld.global.cv.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(ptr)
                 : "memory");
    return value;
}

extern "C" __device__ __forceinline__ unsigned long long timed_cold_load_u32(
    const unsigned int* ptr,
    unsigned int* out_value
) {
    unsigned long long t0;
    unsigned long long t1;
    unsigned int value;

    asm volatile(
        "{ .reg .pred p;\n\t"
        "discard.global.L2 [%3], 128;\n\t"
        "membar.gl;\n\t"
        "mov.u64 %0, %%clock64;\n\t"
        "ld.global.cv.u32 %2, [%3];\n\t"
        "setp.ge.s32 p, %2, 0;\n\t"
        "@p mov.u64 %1, %%clock64;\n\t"
        "@!p mov.u64 %1, %%clock64;\n\t"
        "}"
        : "=l"(t0), "=l"(t1), "=r"(value)
        : "l"(ptr)
        : "memory");

    *out_value = value;
    return t1 - t0;
}

extern "C" __device__ __forceinline__ unsigned int load_global_cg_u32(
    const unsigned int* ptr
) {
    unsigned int value;
    asm volatile("ld.global.cg.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(ptr)
                 : "memory");
    return value;
}

extern "C" __global__ void flush_l2(
    const unsigned int* __restrict__ buf,
    unsigned long long n_words,
    unsigned long long* __restrict__ sink
) {
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    unsigned int acc = 0;

    for (unsigned long long i = tid; i < n_words; i += stride) {
        acc ^= load_global_cg_u32(buf + i) + (unsigned int)i;
    }

    if ((tid & 255ULL) == 0ULL) {
        atomicAdd(sink, (unsigned long long)acc);
    }
}

extern "C" __global__ void collect_smids(
    unsigned int* __restrict__ smids
) {
    extern __shared__ unsigned char reserved_smem[];
    (void)reserved_smem;

    if (threadIdx.x == 0) {
        smids[blockIdx.x] = read_smid();
    }
}

extern "C" __global__ void measure_target_smid_latency(
    const unsigned int* __restrict__ probe,
    unsigned long long anchor_word,
    unsigned int target_smid,
    unsigned int* __restrict__ seen,
    unsigned int* __restrict__ smid_out,
    unsigned long long* __restrict__ cycles_out,
    unsigned int* __restrict__ value_out
) {
    extern __shared__ unsigned char reserved_smem[];
    (void)reserved_smem;

    if (threadIdx.x != 0) {
        return;
    }

    const unsigned int* ptr = probe + anchor_word;
    unsigned int smid = read_smid();
    if (smid != target_smid) {
        return;
    }
    if (atomicCAS(seen, 0U, 1U) != 0U) {
        return;
    }

    unsigned int value = 0;
    unsigned long long cycles = timed_cold_load_u32(ptr, &value);

    smid_out[0] = smid;
    cycles_out[0] = cycles;
    value_out[0] = value;
}
"""


@dataclass(frozen=True)
class DieSmGroup:
    """One die and the ``%smid`` values assigned to it."""

    die_id: int
    sm_ids: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.sm_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "die_id": self.die_id,
            "size": self.size,
            "sm_ids": list(self.sm_ids),
        }


@dataclass(frozen=True)
class DieSmGroups:
    """SM IDs grouped by die."""

    dies: tuple[DieSmGroup, ...]
    median_gap_cycles: float
    anchor_count: int
    sm_count: int
    device_id: int | None = None
    device_name: str | None = None

    @property
    def die_count(self) -> int:
        return len(self.dies)

    @property
    def die0_sms(self) -> tuple[int, ...]:
        return self.dies[0].sm_ids

    @property
    def die1_sms(self) -> tuple[int, ...]:
        if len(self.dies) < 2:
            return ()
        return self.dies[1].sm_ids

    def as_lists(self) -> list[list[int]]:
        return [list(die.sm_ids) for die in self.dies]

    def to_dict(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "sm_count": self.sm_count,
            "die_count": self.die_count,
            "median_gap_cycles": self.median_gap_cycles,
            "anchor_count": self.anchor_count,
            "dies": [die.to_dict() for die in self.dies],
        }


def get_gpu_die_sm_ids(
    device_id: int = 0,
    *,
    anchor_count: int = 8,
    rounds: int = 5,
    evict_bytes: int = 512 * 1024 * 1024,
    anchor_stride_bytes: int = 2 * 1024 * 1024,
    shared_mem_bytes: int | None = None,
    min_gap_cycles: float = 80.0,
    expected_sm_count: int | None = None,
    die_mode: Literal["auto", "split", "single"] = "auto",
    scheduler_waves: int = 4,
    target_retries: int = 4,
    print_latency: bool = False,
) -> list[list[int]]:
    """Return one list of ``%smid`` values per detected die.

    This must be run on the target GPU.  It launches one large-dynamic-shared-
    memory block per SM, measures a single cold global load to several anchor
    addresses, and classifies the two non-overlapping latency bands.

    Requirements on the remote machine:
      * NVIDIA driver/CUDA runtime
      * CuPy matching the CUDA runtime, for example ``cupy-cuda12x``

    Die labels are arbitrary and may flip between processes.  The contents of
    each list are the useful part.
    """

    groups = get_gpu_die_sm_groups(
        device_id=device_id,
        anchor_count=anchor_count,
        rounds=rounds,
        evict_bytes=evict_bytes,
        anchor_stride_bytes=anchor_stride_bytes,
        shared_mem_bytes=shared_mem_bytes,
        min_gap_cycles=min_gap_cycles,
        expected_sm_count=expected_sm_count,
        die_mode=die_mode,
        scheduler_waves=scheduler_waves,
        target_retries=target_retries,
        print_latency=print_latency,
    )
    return groups.as_lists()


def get_gpu_die_sm_groups(
    device_id: int = 0,
    *,
    anchor_count: int = 8,
    rounds: int = 5,
    evict_bytes: int = 512 * 1024 * 1024,
    anchor_stride_bytes: int = 2 * 1024 * 1024,
    shared_mem_bytes: int | None = None,
    min_gap_cycles: float = 80.0,
    expected_sm_count: int | None = None,
    die_mode: Literal["auto", "split", "single"] = "auto",
    scheduler_waves: int = 4,
    target_retries: int = 4,
    print_latency: bool = False,
    latency_stream: TextIO | None = None,
) -> DieSmGroups:
    """Measure and return detailed SM-to-die grouping metadata."""

    cp = import_cupy()
    import numpy as np

    if anchor_count < 1:
        raise ValueError("anchor_count must be at least 1")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if anchor_stride_bytes % 4:
        raise ValueError("anchor_stride_bytes must be 4-byte aligned")
    if evict_bytes < 64 * 1024 * 1024:
        raise ValueError("evict_bytes should be at least 64 MiB to evict L2")
    if shared_mem_bytes is not None and shared_mem_bytes < 0:
        raise ValueError("shared_mem_bytes must be non-negative")
    if expected_sm_count is not None and expected_sm_count <= 0:
        raise ValueError("expected_sm_count must be positive")
    if die_mode not in {"auto", "split", "single"}:
        raise ValueError("die_mode must be one of: auto, split, single")
    if scheduler_waves < 1:
        raise ValueError("scheduler_waves must be at least 1")
    if target_retries < 1:
        raise ValueError("target_retries must be at least 1")

    with cp.cuda.Device(device_id):
        attrs = cp.cuda.Device(device_id).attributes
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        sm_count = int(attrs.get("MultiProcessorCount") or props.get("multiProcessorCount") or props.get(b"multiProcessorCount"))
        device_name = device_property_text(props, "name")
        if expected_sm_count is not None and sm_count != expected_sm_count:
            raise RuntimeError(f"device {device_id} reports {sm_count} SMs, not the {expected_sm_count} SMs requested by expected_sm_count")

        selected_shared_mem_bytes = select_shared_mem_bytes(
            attrs,
            requested=shared_mem_bytes,
        )
        max_optin_smem = max_dynamic_shared_mem_bytes(attrs)
        if max_optin_smem and selected_shared_mem_bytes > max_optin_smem:
            raise RuntimeError(
                f"requested {selected_shared_mem_bytes} bytes of dynamic shared memory, "
                f"but device reports a per-block opt-in limit of "
                f"{max_optin_smem} bytes"
            )

        module = cp.RawModule(code=_CUDA_SOURCE, options=("--std=c++11",))
        flush_l2 = module.get_function("flush_l2")
        collect_smids = module.get_function("collect_smids")
        measure_target = module.get_function("measure_target_smid_latency")
        for kernel in (collect_smids, measure_target):
            try:
                kernel.max_dynamic_shared_size_bytes = selected_shared_mem_bytes
            except Exception as exc:
                raise RuntimeError(
                    "CuPy could not set the opt-in dynamic shared memory size. Try a newer CuPy build or lower shared_mem_bytes."
                ) from exc

        anchor_stride_words = anchor_stride_bytes // 4
        probe_words = anchor_stride_words * anchor_count + 1
        probe = cp.empty(probe_words, dtype=cp.uint32)

        evict_words = evict_bytes // 4
        evict = cp.empty(evict_words, dtype=cp.uint32)
        evict.fill(np.uint32(0x9E3779B9))

        sink = cp.zeros(1, dtype=cp.uint64)
        launch_blocks = sm_count * scheduler_waves
        discovered_slots = cp.empty(launch_blocks, dtype=cp.uint32)
        discovered_smids: set[int] = set()

        for _attempt in range(target_retries):
            discovered_slots.fill(np.uint32(0xFFFFFFFF))
            collect_smids(
                (launch_blocks,),
                (32,),
                (discovered_slots,),
                shared_mem=selected_shared_mem_bytes,
            )
            cp.cuda.runtime.deviceSynchronize()
            for smid in cp.asnumpy(discovered_slots):
                if smid != 0xFFFFFFFF:
                    discovered_smids.add(int(smid))
            if len(discovered_smids) == sm_count:
                break

        if len(discovered_smids) != sm_count:
            raise RuntimeError(
                f"discovered {len(discovered_smids)} unique SM IDs, expected "
                f"{sm_count}. Increase scheduler_waves/target_retries or check "
                "whether the device is partitioned."
            )

        if die_mode == "single":
            return single_die_groups(
                discovered_smids,
                sm_count=sm_count,
                device_id=device_id,
                device_name=device_name,
            )

        smid_out = cp.empty(1, dtype=cp.uint32)
        cycles_out = cp.empty(1, dtype=cp.uint64)
        value_out = cp.empty(1, dtype=cp.uint32)
        seen = cp.empty(1, dtype=cp.uint32)

        samples_by_anchor: list[list[tuple[int, int]]] = [[] for _ in range(anchor_count)]
        flush_threads = 256
        flush_blocks = min(
            65535,
            max(1, (evict_words + flush_threads - 1) // flush_threads),
        )

        for _round, anchor_idx, target_smid in targeted_probe_order(
            discovered_smids,
            anchor_count=anchor_count,
            rounds=rounds,
        ):
            captured = False
            for _retry in range(target_retries):
                flush_l2_cache(
                    cp,
                    flush_l2,
                    evict,
                    np.uint64(evict_words),
                    sink,
                    flush_blocks,
                    flush_threads,
                )

                seen.fill(np.uint32(0))
                smid_out.fill(np.uint32(0xFFFFFFFF))
                cycles_out.fill(np.uint64(0))
                value_out.fill(np.uint32(0))

                measure_target(
                    (launch_blocks,),
                    (32,),
                    (
                        probe,
                        np.uint64(anchor_idx * anchor_stride_words),
                        np.uint32(target_smid),
                        seen,
                        smid_out,
                        cycles_out,
                        value_out,
                    ),
                    shared_mem=selected_shared_mem_bytes,
                )
                cp.cuda.runtime.deviceSynchronize()

                if int(cp.asnumpy(seen)[0]) == 1:
                    smid = int(cp.asnumpy(smid_out)[0])
                    cycle_count = int(cp.asnumpy(cycles_out)[0])
                    if smid == target_smid and cycle_count > 0:
                        samples_by_anchor[anchor_idx].append((smid, cycle_count))
                        captured = True
                        break

            if not captured:
                raise RuntimeError(
                    f"could not capture target SM {target_smid} for anchor "
                    f"{anchor_idx} in round {_round}. Increase scheduler_waves "
                    "or target_retries."
                )

        if print_latency:
            print_latency_diagnostics(
                samples_by_anchor,
                sm_count=sm_count,
                stream=latency_stream or sys.stderr,
            )

        result = classify_die_groups(
            samples_by_anchor,
            sm_count=sm_count,
            min_gap_cycles=min_gap_cycles,
            allow_single_die=(die_mode == "auto"),
        )
        return with_device_metadata(
            result,
            device_id=device_id,
            device_name=device_name,
        )


def device_property_text(props, key: str) -> str | None:
    value = props.get(key)
    if value is None:
        value = props.get(key.encode("utf-8"))
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def max_dynamic_shared_mem_bytes(attrs: dict[str, object]) -> int:
    return int(attrs.get("MaxSharedMemoryPerBlockOptin") or attrs.get("MaxSharedMemoryPerBlock") or 0)


def select_shared_mem_bytes(
    attrs: dict[str, object],
    *,
    requested: int | None,
) -> int:
    if requested is not None:
        return int(requested)
    return max_dynamic_shared_mem_bytes(attrs)


def single_die_groups(
    smids: Iterable[int],
    *,
    sm_count: int,
    device_id: int | None = None,
    device_name: str | None = None,
) -> DieSmGroups:
    ordered = tuple(sorted(int(smid) for smid in smids))
    if len(ordered) != sm_count:
        raise RuntimeError(f"saw {len(ordered)} total SM IDs, expected {sm_count}; missing SM IDs")
    return DieSmGroups(
        dies=(DieSmGroup(die_id=0, sm_ids=ordered),),
        median_gap_cycles=0.0,
        anchor_count=0,
        sm_count=sm_count,
        device_id=device_id,
        device_name=device_name,
    )


def with_device_metadata(
    groups: DieSmGroups,
    *,
    device_id: int | None,
    device_name: str | None,
) -> DieSmGroups:
    return DieSmGroups(
        dies=groups.dies,
        median_gap_cycles=groups.median_gap_cycles,
        anchor_count=groups.anchor_count,
        sm_count=groups.sm_count,
        device_id=device_id,
        device_name=device_name,
    )


def flush_l2_cache(cp, flush_kernel, evict, evict_words, sink, blocks, threads) -> None:
    flush_kernel((blocks,), (threads,), (evict, evict_words, sink))
    cp.cuda.runtime.deviceSynchronize()


def import_cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "get_gpu_die_sm_ids requires CuPy on the remote GPU machine. Install the CUDA-matching wheel, e.g. `pip install cupy-cuda12x`."
        ) from exc
    return cp


def targeted_probe_order(
    smids: Iterable[int],
    *,
    anchor_count: int,
    rounds: int,
) -> tuple[tuple[int, int, int], ...]:
    unique_smids = tuple(sorted({int(smid) for smid in smids}))
    return tuple(
        (round_idx, anchor_idx, smid) for round_idx in range(rounds) for anchor_idx in range(anchor_count) for smid in unique_smids
    )


def print_latency_diagnostics(
    samples_by_anchor: Sequence[Iterable[tuple[int, int]]],
    *,
    sm_count: int,
    stream: TextIO,
) -> None:
    print("=== GPU die latency diagnostics ===", file=stream)
    print(
        json.dumps(
            latency_diagnostics_payload(samples_by_anchor, sm_count=sm_count),
            indent=2,
            sort_keys=True,
        ),
        file=stream,
    )


def latency_diagnostics_payload(
    samples_by_anchor: Sequence[Iterable[tuple[int, int]]],
    *,
    sm_count: int,
) -> dict[str, object]:
    anchors: list[dict[str, object]] = []

    for anchor_idx, samples in enumerate(samples_by_anchor):
        medians = median_cycles_by_sm(samples)
        ordered = sorted(medians.items(), key=lambda item: (item[1], item[0]))
        if len(ordered) >= 2:
            largest_idx, largest_gap = largest_latency_gap(ordered)
            usable_idx, usable_gap = largest_latency_gap(
                ordered,
                min_group_size=minimum_die_group_size(sm_count),
            )
        else:
            largest_idx, largest_gap = -1, 0.0
            usable_idx, usable_gap = -1, 0.0
        split_after = largest_idx + 1 if largest_idx >= 0 else None
        candidate_sizes = [split_after, len(ordered) - split_after] if split_after is not None else None
        usable_split_after = usable_idx + 1 if usable_idx >= 0 else None
        usable_candidate_sizes = [usable_split_after, len(ordered) - usable_split_after] if usable_split_after is not None else None

        anchors.append(
            {
                "anchor": anchor_idx,
                "observed_sm_count": len(ordered),
                "largest_gap_after": split_after,
                "largest_gap_cycles": largest_gap,
                "candidate_group_sizes": candidate_sizes,
                "usable_gap_after": usable_split_after,
                "usable_gap_cycles": usable_gap,
                "usable_group_sizes": usable_candidate_sizes,
                "sorted_by_latency": [[smid, cycles] for smid, cycles in ordered],
                "by_smid": {str(smid): cycles for smid, cycles in sorted(medians.items())},
            }
        )

    return {
        "sm_count": sm_count,
        "anchors": anchors,
    }


def classify_die_groups(
    samples_by_anchor: Sequence[Iterable[tuple[int, int]]],
    *,
    sm_count: int,
    min_gap_cycles: float = 50.0,
    allow_single_die: bool = False,
) -> DieSmGroups:
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if not samples_by_anchor:
        raise RuntimeError("no latency samples were collected")

    min_group_size = minimum_die_group_size(sm_count)
    all_anchor_splits: list[tuple[set[int], set[int]]] = []
    gaps: list[float] = []
    observed_any: set[int] = set()
    rejected_anchors: list[str] = []

    for anchor_idx, samples in enumerate(samples_by_anchor):
        medians = median_cycles_by_sm(samples)
        observed = set(medians)
        observed_any.update(observed)

        if len(observed) != sm_count:
            raise RuntimeError(f"anchor {anchor_idx} saw {len(observed)} SMs, expected {sm_count}; missing SM IDs or too few samples")

        ordered = sorted(medians.items(), key=lambda item: (item[1], item[0]))
        boundary_idx, usable_gap = largest_latency_gap(
            ordered,
            min_group_size=min_group_size,
        )
        if boundary_idx < 0 or usable_gap < min_gap_cycles:
            largest_idx, largest_gap = largest_latency_gap(ordered)
            usable_after = str(boundary_idx + 1) if boundary_idx >= 0 else "none"
            rejected_anchors.append(
                f"anchor {anchor_idx}: best usable gap "
                f"{usable_gap:.1f} cycles after {usable_after} SMs; "
                f"largest raw gap {largest_gap:.1f} cycles after "
                f"{largest_idx + 1} SMs"
            )
            continue

        fast_group = {smid for smid, _cycles in ordered[: boundary_idx + 1]}
        slow_group = observed - fast_group
        all_anchor_splits.append((fast_group, slow_group))
        gaps.append(usable_gap)

    if len(observed_any) != sm_count:
        raise RuntimeError(f"saw {len(observed_any)} total SM IDs, expected {sm_count}; missing SM IDs")
    if not all_anchor_splits:
        if allow_single_die:
            return single_die_groups(observed_any, sm_count=sm_count)
        details = "; ".join(rejected_anchors) or "no anchor had enough samples"
        raise RuntimeError(
            "no latency anchor produced a usable two-large-cluster split; "
            f"{details}. Increase rounds/evict_bytes or force die_mode=single "
            "on a known single-die GPU"
        )

    all_sms = set(observed_any)
    reference = set(all_anchor_splits[0][0])
    reference_other = all_sms - reference
    votes = {smid: 0 for smid in all_sms}
    anchor_count = 0

    for fast_group, slow_group in all_anchor_splits:
        agreement_as_is = len(reference & fast_group) + len(reference_other & slow_group)
        agreement_flipped = len(reference & slow_group) + len(reference_other & fast_group)
        if agreement_as_is == agreement_flipped:
            raise RuntimeError("ambiguous anchor alignment between die groups")

        aligned = fast_group if agreement_as_is > agreement_flipped else slow_group
        for smid in aligned:
            votes[smid] += 1
        anchor_count += 1

    if anchor_count == 0:
        raise RuntimeError("no valid latency anchors were collected")

    ties = [smid for smid, vote in votes.items() if vote * 2 == anchor_count]
    if ties:
        raise RuntimeError(f"ambiguous die votes for SM IDs {sorted(ties)}; increase anchor_count/rounds")

    die0_set = {smid for smid, vote in votes.items() if vote * 2 > anchor_count}
    die1_set = all_sms - die0_set
    min_side = min(len(die0_set), len(die1_set))
    if min_side < min_group_size:
        raise RuntimeError(
            "consensus die split is imbalanced: "
            f"{len(die0_set)} / {len(die1_set)} SMs; refusing to infer die "
            "groups from unstable latency signals"
        )

    die0 = tuple(sorted(die0_set))
    die1 = tuple(sorted(die1_set))
    return DieSmGroups(
        dies=(
            DieSmGroup(die_id=0, sm_ids=die0),
            DieSmGroup(die_id=1, sm_ids=die1),
        ),
        median_gap_cycles=float(median(gaps)),
        anchor_count=len(all_anchor_splits),
        sm_count=sm_count,
    )


def median_cycles_by_sm(samples: Iterable[tuple[int, int]]) -> dict[int, float]:
    per_sm: dict[int, list[int]] = {}
    for smid, cycles in samples:
        if cycles <= 0:
            continue
        per_sm.setdefault(int(smid), []).append(int(cycles))
    return {smid: float(median(values)) for smid, values in per_sm.items()}


def minimum_die_group_size(sm_count: int) -> int:
    return max(1, int(sm_count * _MIN_DIE_GROUP_FRACTION))


def largest_latency_gap(
    ordered: Sequence[tuple[int, float]],
    *,
    min_group_size: int = 1,
) -> tuple[int, float]:
    if len(ordered) < 2:
        raise RuntimeError("need at least two SM latency medians")
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")

    gaps = [ordered[idx + 1][1] - ordered[idx][1] for idx in range(len(ordered) - 1)]
    candidates = [idx for idx in range(len(gaps)) if idx + 1 >= min_group_size and len(ordered) - (idx + 1) >= min_group_size]
    if not candidates:
        return -1, 0.0

    idx = max(candidates, key=lambda i: gaps[i])
    return idx, float(gaps[idx])


if __name__ == "__main__":
    import json

    groups = get_gpu_die_sm_groups()
    print(json.dumps(groups.to_dict(), indent=2))
