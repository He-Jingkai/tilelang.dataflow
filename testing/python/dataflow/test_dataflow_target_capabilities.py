from __future__ import annotations

from types import SimpleNamespace

import pytest

import tilelang.dataflow as df
from tilelang.dataflow import executor
from tilelang.dataflow.executor import DataflowExecutableKernel
from tilelang.utils.target_capabilities import resolve_target_capabilities_from_device


class FakeCudaDriver:
    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUdevice_attribute = SimpleNamespace(
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR="major",
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR="minor",
        CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH="cluster_launch",
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN="max_shared",
        CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED="tensor_map",
    )

    def __init__(self, *, major: int, minor: int, cluster_launch: int, max_shared: int):
        self.attributes = {
            "major": major,
            "minor": minor,
            "cluster_launch": cluster_launch,
            "max_shared": max_shared,
            "tensor_map": 1,
        }
        self.primary_context_releases = []

    def cuDeviceGetAttribute(self, attribute, device):
        return 0, self.attributes[attribute]

    def cuDriverGetVersion(self):
        return 0, 13000

    def cuDeviceGetName(self, length, device):
        del length, device
        return 0, b"mock CUDA device\0"

    def cuDevicePrimaryCtxRelease(self, device):
        self.primary_context_releases.append(device)
        return (0,)


def resolve_mock_snapshot(
    *,
    major: int,
    minor: int,
    device_ordinal: int = 0,
) -> df.TargetCapabilitySnapshot:
    return resolve_target_capabilities_from_device(
        FakeCudaDriver(
            major=major,
            minor=minor,
            cluster_launch=1,
            max_shared=100_000,
        ),
        device_ordinal,
        device_ordinal=device_ordinal,
        compiler_version=(13, 0),
    )


def test_target_capability_snapshot_distinguishes_hopper_and_blackwell_features():
    hopper = resolve_mock_snapshot(major=9, minor=0)
    blackwell = resolve_mock_snapshot(major=12, minor=0)

    assert hopper.arch == "sm_90a"
    assert hopper.supports_tma is True
    assert hopper.supports_tensor_core_mma is True
    assert hopper.supports_wgmma is True
    assert hopper.supports_tcgen05 is False
    assert blackwell.arch == "sm_120a"
    assert blackwell.supports_tma is True
    assert blackwell.supports_tensor_core_mma is True
    assert blackwell.supports_wgmma is False
    assert blackwell.supports_tcgen05 is True
    assert hopper.fingerprint != blackwell.fingerprint
    assert hopper.compatibility_mismatches(blackwell) == (
        "compute_capability artifact=(9, 0) device=(12, 0)",
        "supports_wgmma required by artifact but unavailable on device",
    )


def test_target_capability_snapshot_device_ordinal_changes_identity_not_compatibility():
    device0 = resolve_mock_snapshot(major=12, minor=0, device_ordinal=0)
    device1 = resolve_mock_snapshot(major=12, minor=0, device_ordinal=1)

    assert device0.fingerprint != device1.fingerprint
    assert device0.compatibility_fingerprint == device1.compatibility_fingerprint
    assert device0.compatibility_mismatches(device1) == ()
    assert device0.to_dict()["device_ordinal"] == 0
    assert device1.to_dict()["device_ordinal"] == 1


def test_executor_compiles_exactly_the_snapshot_arch_without_fallback(monkeypatch):
    target = resolve_mock_snapshot(major=12, minor=0)
    compile_arches = []

    def fake_compile_cuda(source, *, target_format, arch, options, verbose):
        del source, target_format, options, verbose
        compile_arches.append(arch)
        return b"cubin"

    monkeypatch.setattr(executor.nvcc, "compile_cuda", fake_compile_cuda)
    kernel = DataflowExecutableKernel(
        kernel_name="mock_kernel",
        source="source",
        launch_package=None,
        topology=None,
        target_capabilities=target,
    )

    assert kernel.compile_cubin() == (b"cubin", "sm_120a")
    assert kernel.compile_cubin() == (b"cubin", "sm_120a")
    assert compile_arches == ["sm_120a"]


def test_cross_target_artifact_is_rejected_before_compile_or_module_load(monkeypatch):
    artifact_target = resolve_mock_snapshot(major=9, minor=0)
    actual_driver = FakeCudaDriver(
        major=12,
        minor=0,
        cluster_launch=1,
        max_shared=100_000,
    )
    compile_calls = []
    module_load_calls = []

    monkeypatch.setattr(
        executor,
        "retain_cuda_context",
        lambda device_ordinal=None: (actual_driver, 0),
    )
    monkeypatch.setattr(
        executor.DataflowExecutableKernel,
        "compile_cubin",
        lambda self: compile_calls.append(self) or (b"cubin", artifact_target.arch),
    )
    monkeypatch.setattr(
        executor,
        "load_function",
        lambda *args: module_load_calls.append(args) or (0, "module", "function"),
    )
    kernel = DataflowExecutableKernel(
        kernel_name="cross_target_kernel",
        source="source",
        launch_package=None,
        topology=None,
        target_capabilities=artifact_target,
        artifact_fingerprint="artifact-test-fingerprint",
    )

    with pytest.raises(df.DataflowTargetCompatibilityError, match="before|incompatible"):
        kernel.launch()

    assert compile_calls == []
    assert module_load_calls == []
    assert actual_driver.primary_context_releases == [0]


def test_kernel_specific_cluster_limit_is_validated_before_launch():
    class ClusterDriver:
        CUresult = SimpleNamespace(CUDA_SUCCESS=0)
        CUfunction_attribute = SimpleNamespace(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES="max_shared",
            CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED="non_portable_cluster",
        )

        class CUlaunchConfig:
            pass

        @staticmethod
        def CUstream(value):
            return value

        @staticmethod
        def cuFuncSetAttribute(function, attribute, value):
            del function, attribute, value
            return (0,)

        @staticmethod
        def cuOccupancyMaxPotentialClusterSize(function, config):
            del function, config
            return 0, 4

    package = SimpleNamespace(
        queue=SimpleNamespace(queue_count=8),
        shared_memory_bytes=64 * 1024,
    )

    with pytest.raises(df.DataflowTargetCompatibilityError, match="kernel-specific"):
        executor.configure_function(
            ClusterDriver(),
            "function",
            package,
            df.GPUTopology(sm_count=8, cluster_size=8),
            block_dim=(128, 1, 1),
        )
