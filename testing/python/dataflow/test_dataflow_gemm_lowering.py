from __future__ import annotations

from dataclasses import replace

import pytest

import tilelang.language as T
import tilelang.dataflow as df


def make_request(**kwargs) -> df.GemmLoweringRequest:
    values = dict(
        operation_id="test.gemm",
        requested_primitive="wgmma",
        m=64,
        n=64,
        k=64,
        a_dtype="float16",
        b_dtype="float16",
        c_dtype="float32",
        a_scope="local.fragment",
        b_scope="shared.dyn",
        c_scope="local.fragment",
        thread_count=256,
    )
    values.update(kwargs)
    return df.GemmLoweringRequest(**values)


def test_gemm_registry_selects_wgmma_from_hopper_capabilities():
    target = df.TargetCapabilitySnapshot.for_cuda((9, 0), compiler_version=(12, 8))

    resolution = df.DEFAULT_GEMM_LOWERING_REGISTRY.resolve(make_request(), target)

    assert resolution.implementation_id == "cuda.wgmma.async"
    assert resolution.used_fallback is False
    assert resolution.target_fingerprint == target.fingerprint
    assert set(resolution.requires) <= df.target_feature_set(target)


def test_gemm_registry_falls_back_from_wgmma_to_mma_on_blackwell():
    target = df.TargetCapabilitySnapshot.for_cuda((12, 0), compiler_version=(13, 0))

    resolution = df.DEFAULT_GEMM_LOWERING_REGISTRY.resolve(make_request(), target)

    assert resolution.implementation_id == "cuda.mma.sync"
    assert resolution.used_fallback is True
    assert "wgmma" not in df.target_feature_set(target)
    assert set(resolution.requires) <= df.target_feature_set(target)


def test_gemm_registry_selects_tcgen05_for_tmem_accumulator():
    target = df.TargetCapabilitySnapshot.for_cuda((12, 0), compiler_version=(13, 0))
    request = make_request(
        requested_primitive="tcgen05",
        a_scope="shared.dyn",
        c_scope="shared.tmem",
    )

    resolution = df.DEFAULT_GEMM_LOWERING_REGISTRY.resolve(request, target)

    assert resolution.implementation_id == "cuda.tcgen05.sync"
    assert resolution.used_fallback is False


def test_gemm_registry_recommends_declared_portable_handler_size():
    target = df.TargetCapabilitySnapshot.for_cuda((12, 0), compiler_version=(13, 0))
    request = make_request(requested_primitive="generic", thread_count=384)

    selected_threads = df.DEFAULT_GEMM_LOWERING_REGISTRY.recommend_thread_count(
        request,
        target,
        accelerated_only=True,
    )
    resolution = df.DEFAULT_GEMM_LOWERING_REGISTRY.resolve(
        replace(request, thread_count=selected_threads),
        target,
    )

    assert selected_threads == 256
    assert resolution.implementation_id == "cuda.mma.sync"


def test_gemm_registry_reports_structured_unsupported_request():
    target = df.TargetCapabilitySnapshot.for_cuda(
        (6, 0),
        supports_tensor_core_mma=False,
        compiler_version=(12, 8),
    )

    with pytest.raises(df.UnsupportedTargetCapabilityError) as error:
        df.DEFAULT_GEMM_LOWERING_REGISTRY.resolve(make_request(), target)

    message = str(error.value)
    assert "request={'operation_id': 'test.gemm'" in message
    assert f"target_fingerprint={target.fingerprint}" in message
    assert "available_capabilities=['cuda']" in message
    assert "cuda.wgmma.async" in message


def test_actual_primfunc_resolution_records_logical_physical_plan():
    @T.prim_func
    def main(
        A: T.Tensor((32, 64), T.float16),
        B: T.Tensor((64, 64), T.float16),
        C: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 64), T.float16)
            B_shared = T.alloc_shared((64, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float32)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                clear_accum=True,
                logical_shape=(32, 64, 64),
            )
            T.copy(C_local, C)

    target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
        max_dynamic_shared_memory=232_448,
    )
    _, resolutions = df.specialize_primfunc_gemm_lowerings(
        main,
        target,
        handler_id=7,
        operator_name="renamable_operator",
        thread_count=128,
    )

    assert len(resolutions) == 1
    resolution = resolutions[0]
    assert resolution.request.m == 32
    assert resolution.physical_shape == (64, 64, 64)
    assert resolution.requires_padding is True
    assert resolution.requires_materialization is True
    assert resolution.implementation_id == "cuda.wgmma.rs.shared_a"
    assert resolution.additional_shared_memory_bytes == 0
    assert resolution.additional_fragment_bytes == 8192
    assert resolution.estimated_resource_bytes == 8192
    assert {requirement["buffer_role"] for requirement in resolution.temporary_requirements} == {"C"}
    assert resolution.selection_reason == ("WGMMA selected with logical-M shared A streamed through a bounded register-source ring")

    single_residency_target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
        max_dynamic_shared_memory=20_000,
    )
    _, single_residency_resolutions = df.specialize_primfunc_gemm_lowerings(
        main,
        single_residency_target,
        handler_id=7,
        operator_name="renamable_operator",
        thread_count=128,
    )
    single_residency = single_residency_resolutions[0]
    assert single_residency.implementation_id == "cuda.wgmma.async"
    assert single_residency.additional_shared_memory_bytes == 4096
    assert single_residency.selection_reason == ("WGMMA selected with compiler-owned neutral logical-M padding")


def test_actual_primfunc_prefers_ss_when_rs_must_recycle_source_ring():
    @T.prim_func
    def main(
        A: T.Tensor((32, 80), T.float16),
        B: T.Tensor((80, 64), T.float16),
        C: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 80), T.float16)
            B_shared = T.alloc_shared((80, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float32)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                clear_accum=True,
                logical_shape=(32, 64, 80),
            )
            T.copy(C_local, C)

    target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
        max_dynamic_shared_memory=232_448,
    )
    _, resolutions = df.specialize_primfunc_gemm_lowerings(
        main,
        target,
        handler_id=7,
        operator_name="long_reduction_operator",
        thread_count=128,
    )
    resolution = resolutions[0]
    assert resolution.implementation_id == "cuda.wgmma.async"
    assert resolution.physical_shape == (64, 64, 80)
    assert resolution.additional_shared_memory_bytes == 5120
    assert resolution.additional_fragment_bytes == 8192
    assert resolution.estimated_resource_bytes == 13312
    assert {requirement["buffer_role"] for requirement in resolution.temporary_requirements} == {"A", "C"}
    assert resolution.selection_reason == ("WGMMA selected with compiler-owned neutral logical-M padding")

    constrained_target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
        max_dynamic_shared_memory=15_360,
    )
    _, constrained_resolutions = df.specialize_primfunc_gemm_lowerings(
        main,
        constrained_target,
        handler_id=7,
        operator_name="long_reduction_operator",
        thread_count=128,
    )
    fallback = constrained_resolutions[0]
    assert fallback.implementation_id == "cuda.wgmma.rs.shared_a"
    assert fallback.additional_shared_memory_bytes == 0
    assert fallback.additional_fragment_bytes == 8192
    assert {requirement["buffer_role"] for requirement in fallback.temporary_requirements} == {"C"}
    assert fallback.selection_reason == (
        "WGMMA selected with logical-M shared A streamed through a bounded "
        "register-source ring because SS materialization exceeds the "
        "available resource budget"
    )
