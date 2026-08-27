"""Measure the SM IDs that belong to each GPC on an NVIDIA GPU.

The public entry point is ``get_gpu_gpc_sm_groups``.  It measures the CUDA
thread-block-cluster occupancy curve, solves the implied GPC-size multiset,
then launches real cluster kernels and groups the ``%smid`` values that appear
inside the same cluster.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence


_DEFAULT_SHARED_MEM_BYTES: int | None = None
_DEFAULT_THREAD_PER_BLOCK: int | None = None
_DEFAULT_CUDA_ARCH = "native"
_ERROR_BUFFER_BYTES = 4096


_CUDA_CPP_SOURCE = r"""
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace cg = cooperative_groups;

extern "C" {
struct GpuGpcKernelRecord {
    uint32_t launch_round;
    uint32_t cluster_size;
    uint32_t cluster_rank;
    uint32_t block_rank;
    uint32_t block_idx;
    uint32_t smid;
};
}

__device__ __forceinline__ uint32_t gpu_read_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void gpu_gpc_record_kernel(
    GpuGpcKernelRecord* __restrict__ records,
    uint32_t records_per_round,
    uint32_t launch_round,
    uint32_t requested_cluster_size,
    unsigned long long spin_cycles
) {
    extern __shared__ unsigned char reserved_smem[];
    (void)reserved_smem;

    cg::cluster_group cluster = cg::this_cluster();
    uint32_t block_rank = cluster.block_rank();
    uint32_t actual_cluster_size = cluster.num_blocks();

    if (threadIdx.x == 0) {
        unsigned long long start = clock64();
        while (clock64() - start < spin_cycles) {
        }

        uint32_t idx = launch_round * records_per_round + blockIdx.x;
        uint32_t cluster_rank = blockIdx.x / actual_cluster_size;
        records[idx].launch_round = launch_round;
        records[idx].cluster_size = requested_cluster_size;
        records[idx].cluster_rank = cluster_rank;
        records[idx].block_rank = block_rank;
        records[idx].block_idx = blockIdx.x;
        records[idx].smid = gpu_read_smid();
    }

    cluster.sync();
}

static int gpu_set_error(char* err, size_t err_len, const char* message) {
    if (err != nullptr && err_len > 0) {
        snprintf(err, err_len, "%s", message);
    }
    return 1;
}

static int gpu_set_cuda_error(
    char* err,
    size_t err_len,
    const char* call,
    cudaError_t status
) {
    if (err != nullptr && err_len > 0) {
        snprintf(
            err,
            err_len,
            "%s failed: %s",
            call,
            cudaGetErrorString(status)
        );
    }
    return 1;
}

static unsigned int gpu_round_up_to_cluster_multiple(
    unsigned int blocks,
    unsigned int cluster_size
) {
    return ((blocks + cluster_size - 1U) / cluster_size) * cluster_size;
}

static int gpu_select_thread_per_block(
    const cudaDeviceProp& props,
    int requested_thread_per_block
) {
    if (requested_thread_per_block > 0) {
        return requested_thread_per_block;
    }
    return props.maxThreadsPerBlock;
}

static int gpu_select_dynamic_smem_bytes(
    const cudaDeviceProp& props,
    int requested_dynamic_smem_bytes
) {
    if (requested_dynamic_smem_bytes > 0) {
        return requested_dynamic_smem_bytes;
    }
    if (props.sharedMemPerBlockOptin > 0) {
        return static_cast<int>(props.sharedMemPerBlockOptin);
    }
    return static_cast<int>(props.sharedMemPerBlock);
}

static cudaError_t gpu_configure_kernel(int dynamic_smem_bytes) {
    cudaError_t status = cudaFuncSetAttribute(
        gpu_gpc_record_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaFuncSetAttribute(
        gpu_gpc_record_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );
    if (status != cudaSuccess) {
        return status;
    }

    return cudaFuncSetAttribute(
        gpu_gpc_record_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_smem_bytes
    );
}

extern "C" int gpu_gpc_probe(
    int device_id,
    int thread_per_block,
    int dynamic_smem_bytes,
    int active_clusters_capacity,
    int* out_sm_count,
    int* out_max_cluster_size,
    int* out_thread_per_block,
    int* out_dynamic_smem_bytes,
    int* out_active_clusters,
    char* out_device_name,
    size_t out_device_name_len,
    char* err,
    size_t err_len
) {
    if (out_sm_count == nullptr || out_max_cluster_size == nullptr ||
        out_thread_per_block == nullptr || out_dynamic_smem_bytes == nullptr ||
        out_active_clusters == nullptr) {
        return gpu_set_error(err, err_len, "null output pointer");
    }
    if (thread_per_block < 0 || dynamic_smem_bytes < 0 ||
        active_clusters_capacity < 1) {
        return gpu_set_error(err, err_len, "invalid probe argument");
    }

    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return gpu_set_cuda_error(err, err_len, "cudaSetDevice", status);
    }

    cudaDeviceProp props;
    status = cudaGetDeviceProperties(&props, device_id);
    if (status != cudaSuccess) {
        return gpu_set_cuda_error(
            err,
            err_len,
            "cudaGetDeviceProperties",
            status
        );
    }

    int selected_thread_per_block =
        gpu_select_thread_per_block(props, thread_per_block);
    int selected_dynamic_smem_bytes =
        gpu_select_dynamic_smem_bytes(props, dynamic_smem_bytes);
    if (selected_thread_per_block <= 0 ||
        selected_thread_per_block > props.maxThreadsPerBlock) {
        return gpu_set_error(err, err_len, "invalid selected thread_per_block");
    }
    if (selected_dynamic_smem_bytes < 0 ||
        (props.sharedMemPerBlockOptin > 0 &&
         selected_dynamic_smem_bytes > props.sharedMemPerBlockOptin)) {
        return gpu_set_error(err, err_len, "invalid selected dynamic shared memory");
    }

    status = gpu_configure_kernel(selected_dynamic_smem_bytes);
    if (status != cudaSuccess) {
        return gpu_set_cuda_error(
            err,
            err_len,
            "cudaFuncSetAttribute",
            status
        );
    }

    for (int idx = 0; idx <= active_clusters_capacity; ++idx) {
        out_active_clusters[idx] = 0;
    }

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(props.multiProcessorCount);
    config.blockDim = dim3(selected_thread_per_block);
    config.dynamicSmemBytes = selected_dynamic_smem_bytes;
    config.attrs = attr;
    config.numAttrs = 1;

    int max_cluster_size = 0;
    status = cudaOccupancyMaxPotentialClusterSize(
        &max_cluster_size,
        reinterpret_cast<void*>(gpu_gpc_record_kernel),
        &config
    );
    if (status != cudaSuccess) {
        return gpu_set_cuda_error(
            err,
            err_len,
            "cudaOccupancyMaxPotentialClusterSize",
            status
        );
    }
    if (max_cluster_size > active_clusters_capacity) {
        return gpu_set_error(
            err,
            err_len,
            "active cluster output capacity is too small"
        );
    }

    for (int cluster_size = 1; cluster_size <= max_cluster_size; ++cluster_size) {
        attr[0].val.clusterDim.x = cluster_size;
        config.gridDim = dim3(
            gpu_round_up_to_cluster_multiple(
                static_cast<unsigned int>(props.multiProcessorCount + cluster_size),
                static_cast<unsigned int>(cluster_size)
            )
        );

        int max_active_clusters = 0;
        status = cudaOccupancyMaxActiveClusters(
            &max_active_clusters,
            reinterpret_cast<void*>(gpu_gpc_record_kernel),
            &config
        );
        if (status != cudaSuccess) {
            return gpu_set_cuda_error(
                err,
                err_len,
                "cudaOccupancyMaxActiveClusters",
                status
            );
        }
        out_active_clusters[cluster_size] = max_active_clusters;
    }

    *out_sm_count = props.multiProcessorCount;
    *out_max_cluster_size = max_cluster_size;
    *out_thread_per_block = selected_thread_per_block;
    *out_dynamic_smem_bytes = selected_dynamic_smem_bytes;
    if (out_device_name != nullptr && out_device_name_len > 0) {
        snprintf(out_device_name, out_device_name_len, "%s", props.name);
    }
    return 0;
}

extern "C" int gpu_gpc_collect(
    int device_id,
    int cluster_size,
    int clusters_per_launch,
    int rounds,
    int thread_per_block,
    int dynamic_smem_bytes,
    unsigned long long spin_cycles,
    GpuGpcKernelRecord* host_records,
    size_t record_capacity,
    size_t* out_record_count,
    char* err,
    size_t err_len
) {
    if (host_records == nullptr || out_record_count == nullptr) {
        return gpu_set_error(err, err_len, "null output pointer");
    }
    if (cluster_size < 1 || clusters_per_launch < 1 || rounds < 1 ||
        thread_per_block <= 0 || dynamic_smem_bytes < 0) {
        return gpu_set_error(err, err_len, "invalid collect argument");
    }

    size_t records_per_round =
        static_cast<size_t>(cluster_size) * static_cast<size_t>(clusters_per_launch);
    size_t total_records = records_per_round * static_cast<size_t>(rounds);
    if (record_capacity < total_records) {
        return gpu_set_error(err, err_len, "host record buffer is too small");
    }
    if (records_per_round > 0xFFFFFFFFULL || rounds > 0xFFFFFFFFULL) {
        return gpu_set_error(err, err_len, "collect launch is too large");
    }

    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return gpu_set_cuda_error(err, err_len, "cudaSetDevice", status);
    }

    status = gpu_configure_kernel(dynamic_smem_bytes);
    if (status != cudaSuccess) {
        return gpu_set_cuda_error(
            err,
            err_len,
            "cudaFuncSetAttribute",
            status
        );
    }

    GpuGpcKernelRecord* device_records = nullptr;
    status = cudaMalloc(
        reinterpret_cast<void**>(&device_records),
        total_records * sizeof(GpuGpcKernelRecord)
    );
    if (status != cudaSuccess) {
        return gpu_set_cuda_error(err, err_len, "cudaMalloc", status);
    }

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cluster_size;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(static_cast<unsigned int>(records_per_round));
    config.blockDim = dim3(thread_per_block);
    config.dynamicSmemBytes = dynamic_smem_bytes;
    config.attrs = attr;
    config.numAttrs = 1;

    for (int round = 0; round < rounds; ++round) {
        status = cudaLaunchKernelEx(
            &config,
            gpu_gpc_record_kernel,
            device_records,
            static_cast<uint32_t>(records_per_round),
            static_cast<uint32_t>(round),
            static_cast<uint32_t>(cluster_size),
            spin_cycles
        );
        if (status != cudaSuccess) {
            cudaFree(device_records);
            return gpu_set_cuda_error(
                err,
                err_len,
                "cudaLaunchKernelEx",
                status
            );
        }

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            cudaFree(device_records);
            return gpu_set_cuda_error(
                err,
                err_len,
                "cudaDeviceSynchronize",
                status
            );
        }
    }

    status = cudaMemcpy(
        host_records,
        device_records,
        total_records * sizeof(GpuGpcKernelRecord),
        cudaMemcpyDeviceToHost
    );
    cudaFree(device_records);
    if (status != cudaSuccess) {
        return gpu_set_cuda_error(err, err_len, "cudaMemcpy", status);
    }

    *out_record_count = total_records;
    return 0;
}
"""


@dataclass(frozen=True)
class ClusterSmSample:
    """SM IDs observed inside one launched CUDA block cluster."""

    cluster_size: int
    launch_round: int
    cluster_rank: int
    sm_ids: tuple[int, ...]


@dataclass(frozen=True)
class GpcSmGroup:
    """One inferred GPC and the actual ``%smid`` values assigned to it."""

    gpc_id: int
    size: int
    sm_ids: tuple[int, ...]


@dataclass(frozen=True)
class GpuGpcSmGroups:
    """Complete GPU GPC grouping result."""

    device_id: int
    sm_count: int
    max_cluster_size: int
    active_clusters_by_size: tuple[tuple[int, int], ...]
    inferred_gpc_sizes: tuple[int, ...]
    gpcs: tuple[GpcSmGroup, ...]
    cluster_sample_count: int
    kernel_record_count: int
    cluster_samples: tuple[ClusterSmSample, ...] = ()
    device_name: str | None = None
    shared_mem_bytes: int | None = None
    thread_per_block: int | None = None
    cuda_arch: str | None = None

    @property
    def gpc_count(self) -> int:
        return len(self.gpcs)

    @property
    def gpc_sizes(self) -> tuple[int, ...]:
        return tuple(group.size for group in self.gpcs)

    def as_lists(self) -> list[list[int]]:
        return [list(group.sm_ids) for group in self.gpcs]

    def to_dict(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "sm_count": self.sm_count,
            "gpc_count": self.gpc_count,
            "max_cluster_size": self.max_cluster_size,
            "shared_mem_bytes": self.shared_mem_bytes,
            "thread_per_block": self.thread_per_block,
            "cuda_arch": self.cuda_arch,
            "active_clusters_by_size": dict(self.active_clusters_by_size),
            "inferred_gpc_sizes": list(self.inferred_gpc_sizes),
            "gpc_sizes": list(self.gpc_sizes),
            "gpcs": [
                {
                    "gpc_id": group.gpc_id,
                    "size": group.size,
                    "sm_ids": list(group.sm_ids),
                }
                for group in self.gpcs
            ],
            "cluster_sample_count": self.cluster_sample_count,
            "kernel_record_count": self.kernel_record_count,
        }


@dataclass(frozen=True)
class KernelSmRecord:
    launch_round: int
    cluster_size: int
    cluster_rank: int
    block_rank: int
    block_idx: int
    smid: int


@dataclass(frozen=True)
class ProbeResult:
    sm_count: int
    max_cluster_size: int
    thread_per_block: int
    shared_mem_bytes: int
    device_name: str | None
    active_clusters_by_size: dict[int, int]


class CRecord(ctypes.Structure):
    _fields_ = [
        ("launch_round", ctypes.c_uint32),
        ("cluster_size", ctypes.c_uint32),
        ("cluster_rank", ctypes.c_uint32),
        ("block_rank", ctypes.c_uint32),
        ("block_idx", ctypes.c_uint32),
        ("smid", ctypes.c_uint32),
    ]


def get_gpu_gpc_sm_ids(
    device_id: int = 0,
    **kwargs,
) -> list[list[int]]:
    """Return one list of actual ``%smid`` values per inferred GPU GPC."""

    return get_gpu_gpc_sm_groups(device_id=device_id, **kwargs).as_lists()


def get_gpu_gpc_sm_groups(
    device_id: int = 0,
    *,
    rounds_per_cluster_size: int = 16,
    waves_per_cluster_size: int = 4,
    thread_per_block: int | None = _DEFAULT_THREAD_PER_BLOCK,
    shared_mem_bytes: int | None = _DEFAULT_SHARED_MEM_BYTES,
    spin_cycles: int = 10_000,
    expected_sm_count: int | None = None,
    nvcc: str = "nvcc",
    cuda_arch: str = _DEFAULT_CUDA_ARCH,
    cache_dir: str | Path | None = None,
    cluster_sizes: Sequence[int] | None = None,
    include_samples: bool = False,
    verbose: bool = False,
) -> GpuGpcSmGroups:
    """Probe, solve, and return GPU GPC sizes plus per-GPC SM IDs.

    This function must run on the target GPU machine.  It requires ``nvcc`` and
    a CUDA runtime new enough for thread-block-cluster launch attributes.
    """

    if rounds_per_cluster_size < 1:
        raise ValueError("rounds_per_cluster_size must be at least 1")
    if waves_per_cluster_size < 1:
        raise ValueError("waves_per_cluster_size must be at least 1")
    if thread_per_block is not None and thread_per_block < 1:
        raise ValueError("thread_per_block must be positive")
    if shared_mem_bytes is not None and shared_mem_bytes < 0:
        raise ValueError("shared_mem_bytes must be non-negative")
    if expected_sm_count is not None and expected_sm_count <= 0:
        raise ValueError("expected_sm_count must be positive")
    if spin_cycles < 0:
        raise ValueError("spin_cycles must be non-negative")

    lib = load_probe_library(nvcc=nvcc, cuda_arch=cuda_arch, cache_dir=cache_dir)
    probe = probe_active_clusters(
        lib,
        device_id=device_id,
        thread_per_block=thread_per_block,
        shared_mem_bytes=shared_mem_bytes,
    )
    if expected_sm_count is not None and probe.sm_count != expected_sm_count:
        raise RuntimeError(
            f"device {device_id} reports {probe.sm_count} SMs, not the {expected_sm_count} SMs requested by expected_sm_count"
        )

    inferred_sizes = solve_gpc_sizes_from_active_clusters(
        probe.active_clusters_by_size,
        sm_count=probe.sm_count,
    )
    selected_cluster_sizes = select_cluster_sizes(
        cluster_sizes,
        max_cluster_size=probe.max_cluster_size,
    )

    records: list[KernelSmRecord] = []
    for cluster_size in selected_cluster_sizes:
        active_clusters = probe.active_clusters_by_size[cluster_size]
        clusters_per_launch = active_clusters * waves_per_cluster_size
        records.extend(
            collect_kernel_sm_records(
                lib,
                device_id=device_id,
                cluster_size=cluster_size,
                clusters_per_launch=clusters_per_launch,
                rounds=rounds_per_cluster_size,
                thread_per_block=probe.thread_per_block,
                shared_mem_bytes=probe.shared_mem_bytes,
                spin_cycles=spin_cycles,
            )
        )

    samples = records_to_cluster_samples(records)
    grouped_sms = infer_gpc_sm_groups_from_cluster_samples(
        samples,
        expected_gpc_sizes=inferred_sizes,
        sm_count=probe.sm_count,
    )
    gpcs = tuple(GpcSmGroup(gpc_id=idx, size=len(sm_ids), sm_ids=sm_ids) for idx, sm_ids in enumerate(grouped_sms))
    result = GpuGpcSmGroups(
        device_id=device_id,
        sm_count=probe.sm_count,
        max_cluster_size=probe.max_cluster_size,
        active_clusters_by_size=tuple(sorted(probe.active_clusters_by_size.items())),
        inferred_gpc_sizes=inferred_sizes,
        gpcs=gpcs,
        cluster_sample_count=len(samples),
        kernel_record_count=len(records),
        cluster_samples=samples if include_samples else (),
        device_name=probe.device_name,
        shared_mem_bytes=probe.shared_mem_bytes,
        thread_per_block=probe.thread_per_block,
        cuda_arch=cuda_arch,
    )

    if verbose:
        print(json.dumps(result.to_dict(), indent=2))
    return result


def select_cluster_sizes(
    requested: Sequence[int] | None,
    *,
    max_cluster_size: int,
) -> tuple[int, ...]:
    if max_cluster_size < 2:
        raise RuntimeError("device reported max cluster size < 2")
    if requested is None:
        return tuple(range(max_cluster_size, 1, -1))

    selected = tuple(dict.fromkeys(int(size) for size in requested))
    bad = [size for size in selected if size < 2 or size > max_cluster_size]
    if bad:
        raise ValueError(f"cluster_sizes must be in [2, {max_cluster_size}], got {bad}")
    return selected


def load_probe_library(
    *,
    nvcc: str,
    cuda_arch: str,
    cache_dir: str | Path | None,
) -> ctypes.CDLL:
    nvcc_path = shutil.which(nvcc)
    if nvcc_path is None:
        raise RuntimeError(f"could not find {nvcc!r}; install CUDA nvcc or pass nvcc=/path/to/nvcc")

    cache_root = Path(cache_dir) if cache_dir is not None else (Path(tempfile.gettempdir()) / "gpu_gpc_sms")
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((_CUDA_CPP_SOURCE + "\n" + cuda_arch).encode("utf-8")).hexdigest()[:16]
    cu_path = cache_root / f"gpu_gpc_probe_{digest}.cu"
    so_path = cache_root / f"gpu_gpc_probe_{digest}.so"

    if not so_path.exists():
        cu_path.write_text(_CUDA_CPP_SOURCE, encoding="utf-8")
        cmd = [
            nvcc_path,
            "--shared",
            "-Xcompiler",
            "-fPIC",
            "--std=c++17",
            f"-arch={cuda_arch}",
            str(cu_path),
            "-o",
            str(so_path),
        ]
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to compile GPU GPC probe library with nvcc\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

    lib = ctypes.CDLL(str(so_path))
    configure_ctypes(lib)
    return lib


def configure_ctypes(lib: ctypes.CDLL) -> None:
    lib.gpu_gpc_probe.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.gpu_gpc_probe.restype = ctypes.c_int

    lib.gpu_gpc_collect.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_ulonglong,
        ctypes.POINTER(CRecord),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.gpu_gpc_collect.restype = ctypes.c_int


def probe_active_clusters(
    lib: ctypes.CDLL,
    *,
    device_id: int,
    thread_per_block: int | None,
    shared_mem_bytes: int | None,
) -> ProbeResult:
    capacity = 64
    active_array_type = ctypes.c_int * (capacity + 1)
    active = active_array_type()
    sm_count = ctypes.c_int()
    max_cluster_size = ctypes.c_int()
    selected_thread_per_block = ctypes.c_int()
    selected_shared_mem_bytes = ctypes.c_int()
    device_name = ctypes.create_string_buffer(256)
    err = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)

    status = lib.gpu_gpc_probe(
        ctypes.c_int(device_id),
        ctypes.c_int(0 if thread_per_block is None else thread_per_block),
        ctypes.c_int(0 if shared_mem_bytes is None else shared_mem_bytes),
        ctypes.c_int(capacity),
        ctypes.byref(sm_count),
        ctypes.byref(max_cluster_size),
        ctypes.byref(selected_thread_per_block),
        ctypes.byref(selected_shared_mem_bytes),
        active,
        device_name,
        ctypes.c_size_t(len(device_name)),
        err,
        ctypes.c_size_t(len(err)),
    )
    if status != 0:
        raise RuntimeError(err.value.decode("utf-8", errors="replace"))

    active_clusters = {cluster_size: int(active[cluster_size]) for cluster_size in range(1, int(max_cluster_size.value) + 1)}
    return ProbeResult(
        sm_count=int(sm_count.value),
        max_cluster_size=int(max_cluster_size.value),
        thread_per_block=int(selected_thread_per_block.value),
        shared_mem_bytes=int(selected_shared_mem_bytes.value),
        device_name=decode_c_string(device_name.value),
        active_clusters_by_size=active_clusters,
    )


def decode_c_string(value: bytes) -> str | None:
    if not value:
        return None
    return value.decode("utf-8", errors="replace").rstrip("\x00")


def collect_kernel_sm_records(
    lib: ctypes.CDLL,
    *,
    device_id: int,
    cluster_size: int,
    clusters_per_launch: int,
    rounds: int,
    thread_per_block: int,
    shared_mem_bytes: int,
    spin_cycles: int,
) -> tuple[KernelSmRecord, ...]:
    total_records = cluster_size * clusters_per_launch * rounds
    record_array_type = CRecord * total_records
    record_array = record_array_type()
    out_count = ctypes.c_size_t()
    err = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)

    status = lib.gpu_gpc_collect(
        ctypes.c_int(device_id),
        ctypes.c_int(cluster_size),
        ctypes.c_int(clusters_per_launch),
        ctypes.c_int(rounds),
        ctypes.c_int(thread_per_block),
        ctypes.c_int(shared_mem_bytes),
        ctypes.c_ulonglong(spin_cycles),
        record_array,
        ctypes.c_size_t(total_records),
        ctypes.byref(out_count),
        err,
        ctypes.c_size_t(len(err)),
    )
    if status != 0:
        raise RuntimeError(err.value.decode("utf-8", errors="replace"))

    return tuple(
        KernelSmRecord(
            launch_round=int(record.launch_round),
            cluster_size=int(record.cluster_size),
            cluster_rank=int(record.cluster_rank),
            block_rank=int(record.block_rank),
            block_idx=int(record.block_idx),
            smid=int(record.smid),
        )
        for record in record_array[: int(out_count.value)]
    )


def solve_gpc_sizes_from_active_clusters(
    active_clusters_by_size: Mapping[int, int],
    *,
    sm_count: int,
    candidate_step: int = 2,
) -> tuple[int, ...]:
    """Solve ``sum(floor(gpc_size / cluster_size)) == active_clusters``."""

    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if candidate_step <= 0:
        raise ValueError("candidate_step must be positive")
    if not active_clusters_by_size:
        raise ValueError("active_clusters_by_size must not be empty")

    targets = {int(size): int(value) for size, value in active_clusters_by_size.items()}
    targets[1] = sm_count
    max_cluster_size = max(targets)
    expected = tuple(targets[size] for size in range(1, max_cluster_size + 1))

    try:
        result = solve_gpc_sizes_with_scipy(
            expected,
            sm_count=sm_count,
            max_cluster_size=max_cluster_size,
            candidate_step=candidate_step,
        )
    except (ImportError, RuntimeError):
        result = solve_gpc_sizes_with_search(
            expected,
            sm_count=sm_count,
            max_cluster_size=max_cluster_size,
            candidate_step=candidate_step,
        )

    validate_gpc_size_solution(result, expected)
    return result


def solve_gpc_sizes_with_scipy(
    expected: Sequence[int],
    *,
    sm_count: int,
    max_cluster_size: int,
    candidate_step: int,
) -> tuple[int, ...]:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    sizes = np.arange(candidate_step, sm_count + 1, candidate_step)
    matrix = np.zeros((max_cluster_size, len(sizes)), dtype=np.float64)
    for row, cluster_size in enumerate(range(1, max_cluster_size + 1)):
        matrix[row, :] = sizes // cluster_size

    constraints = LinearConstraint(
        matrix,
        lb=np.array(expected, dtype=np.float64),
        ub=np.array(expected, dtype=np.float64),
    )
    result = milp(
        c=np.ones(len(sizes), dtype=np.float64),
        constraints=constraints,
        integrality=np.ones(len(sizes), dtype=np.float64),
        bounds=Bounds(0, np.inf),
    )
    if not result.success:
        raise RuntimeError(f"could not solve GPC sizes: {result.message}")

    solved: list[int] = []
    for size, count in zip(sizes, result.x):
        rounded = int(round(float(count)))
        if rounded > 0:
            solved.extend([int(size)] * rounded)
    return tuple(sorted(solved))


def solve_gpc_sizes_with_search(
    expected: Sequence[int],
    *,
    sm_count: int,
    max_cluster_size: int,
    candidate_step: int,
) -> tuple[int, ...]:
    candidate_sizes = tuple(range(candidate_step, sm_count + 1, candidate_step))
    contributions = {size: tuple(size // cluster_size for cluster_size in range(1, max_cluster_size + 1)) for size in candidate_sizes}
    expected_tuple = tuple(int(value) for value in expected)

    @cache
    def floor_bounds(
        group_count: int,
        min_size: int,
        total_sm: int,
        cluster_size: int,
    ) -> tuple[int, int]:
        if group_count == 0:
            return (0, 0) if total_sm == 0 else (10**9, -(10**9))
        if total_sm < group_count * min_size:
            return (10**9, -(10**9))

        best_min = 10**9
        best_max = -(10**9)
        max_first = min(sm_count, total_sm - (group_count - 1) * min_size)
        for size in candidate_sizes:
            if size < min_size:
                continue
            if size > max_first:
                break
            child_min, child_max = floor_bounds(
                group_count - 1,
                size,
                total_sm - size,
                cluster_size,
            )
            if child_min <= child_max:
                value = size // cluster_size
                best_min = min(best_min, value + child_min)
                best_max = max(best_max, value + child_max)
        return best_min, best_max

    def feasible(group_count: int, min_size: int, remaining: tuple[int, ...]) -> bool:
        total_sm = remaining[0]
        if total_sm < group_count * min_size:
            return False
        for idx, cluster_size in enumerate(range(1, max_cluster_size + 1)):
            lower, upper = floor_bounds(
                group_count,
                min_size,
                total_sm,
                cluster_size,
            )
            if remaining[idx] < lower or remaining[idx] > upper:
                return False
        return True

    lower_bound = minimum_group_count_lower_bound(expected_tuple)
    for group_count in range(lower_bound, sm_count // candidate_step + 1):

        @cache
        def search(
            remaining_groups: int,
            min_size: int,
            remaining: tuple[int, ...],
        ) -> tuple[int, ...] | None:
            if remaining_groups == 0:
                return () if all(value == 0 for value in remaining) else None
            if not feasible(remaining_groups, min_size, remaining):
                return None

            max_first = min(
                sm_count,
                remaining[0] - (remaining_groups - 1) * min_size,
            )
            for size in candidate_sizes:
                if size < min_size:
                    continue
                if size > max_first:
                    break
                contribution = contributions[size]
                next_remaining = tuple(remaining[idx] - contribution[idx] for idx in range(max_cluster_size))
                if any(value < 0 for value in next_remaining):
                    continue
                if not feasible(remaining_groups - 1, size, next_remaining):
                    continue

                child = search(remaining_groups - 1, size, next_remaining)
                if child is not None:
                    return (size,) + child
            return None

        solved = search(group_count, candidate_step, expected_tuple)
        if solved is not None:
            return tuple(sorted(solved))

    raise RuntimeError("could not solve a GPC-size multiset from the measured cluster occupancy curve")


def minimum_group_count_lower_bound(expected: Sequence[int]) -> int:
    total_sm = int(expected[0])
    lower_bound = 1
    for cluster_size, active_clusters in enumerate(expected, start=1):
        if cluster_size == 1:
            continue
        numerator = total_sm - cluster_size * int(active_clusters)
        if numerator > 0:
            lower_bound = max(
                lower_bound,
                (numerator + cluster_size - 2) // (cluster_size - 1),
            )
    return lower_bound


def validate_gpc_size_solution(
    sizes: Sequence[int],
    expected: Sequence[int],
) -> None:
    for cluster_size, expected_active in enumerate(expected, start=1):
        actual = sum(size // cluster_size for size in sizes)
        if actual != expected_active:
            raise RuntimeError(
                "GPC size solution does not match measured occupancy: "
                f"cluster_size={cluster_size}, expected {expected_active}, got {actual}"
            )


def records_to_cluster_samples(
    records: Sequence[KernelSmRecord],
) -> tuple[ClusterSmSample, ...]:
    grouped: dict[tuple[int, int, int], set[int]] = {}
    for record in records:
        key = (record.cluster_size, record.launch_round, record.cluster_rank)
        grouped.setdefault(key, set()).add(record.smid)

    samples: list[ClusterSmSample] = []
    for (cluster_size, launch_round, cluster_rank), smids in sorted(grouped.items()):
        if len(smids) != cluster_size:
            continue
        samples.append(
            ClusterSmSample(
                cluster_size=cluster_size,
                launch_round=launch_round,
                cluster_rank=cluster_rank,
                sm_ids=tuple(sorted(smids)),
            )
        )
    return tuple(samples)


def infer_gpc_sm_groups_from_cluster_samples(
    samples: Iterable[ClusterSmSample],
    *,
    expected_gpc_sizes: Sequence[int],
    sm_count: int,
) -> tuple[tuple[int, ...], ...]:
    expected_sizes = tuple(sorted(int(size) for size in expected_gpc_sizes))
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if sum(expected_sizes) != sm_count:
        raise RuntimeError(f"expected GPC sizes sum to {sum(expected_sizes)}, not sm_count={sm_count}")

    observed_sms: set[int] = set()
    sample_list = list(samples)
    for sample in sample_list:
        observed_sms.update(sample.sm_ids)
    if len(observed_sms) != sm_count:
        raise RuntimeError(
            f"kernel samples saw {len(observed_sms)} SM IDs, expected {sm_count}; "
            "increase rounds_per_cluster_size or waves_per_cluster_size"
        )

    parent = {smid: smid for smid in observed_sms}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for sample in sample_list:
        if len(sample.sm_ids) < 2:
            continue
        first = sample.sm_ids[0]
        for smid in sample.sm_ids[1:]:
            union(first, smid)

    components_by_root: dict[int, list[int]] = {}
    for smid in observed_sms:
        components_by_root.setdefault(find(smid), []).append(smid)

    components = tuple(
        tuple(sorted(component))
        for component in sorted(
            components_by_root.values(),
            key=lambda values: (-len(values), min(values)),
        )
    )
    component_sizes = tuple(sorted(len(component) for component in components))
    if component_sizes != expected_sizes:
        raise RuntimeError(
            "kernel cluster samples did not connect SM IDs into the solved GPC "
            f"sizes; got component sizes {component_sizes}, expected {expected_sizes}. "
            "Increase rounds_per_cluster_size/waves_per_cluster_size, or inspect "
            "include_samples=True output."
        )
    return components


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--shared-mem-bytes", type=int, default=None)
    parser.add_argument("--thread-per-block", type=int, default=None)
    parser.add_argument("--spin-cycles", type=int, default=10_000)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuda-arch", default=_DEFAULT_CUDA_ARCH)
    parser.add_argument("--expected-sm-count", type=int)
    args = parser.parse_args()

    measured = get_gpu_gpc_sm_groups(
        device_id=args.device,
        rounds_per_cluster_size=args.rounds,
        waves_per_cluster_size=args.waves,
        thread_per_block=args.thread_per_block,
        shared_mem_bytes=args.shared_mem_bytes,
        spin_cycles=args.spin_cycles,
        expected_sm_count=args.expected_sm_count,
        nvcc=args.nvcc,
        cuda_arch=args.cuda_arch,
    )
    print(json.dumps(measured.to_dict(), indent=2))
