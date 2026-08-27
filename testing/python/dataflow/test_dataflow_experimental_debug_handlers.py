from __future__ import annotations

import ctypes
import os
import struct

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.contrib import nvcc
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug


@T.dataflow_intermediate
class ScalarInter:
    value: T.int32


@T.dataflow.iter(range=("begin", "end"))
def split_scalar(batch: T.int32, Source) -> ScalarInter:
    raise AssertionError("synthetic debug handlers must not execute the operator body")


@T.dataflow.reduce
def reduce_scalar(items: list[ScalarInter]) -> ScalarInter:
    raise AssertionError("synthetic debug handlers must not execute the operator body")


@T.dataflow.finalize
def finalize_scalar(inter: ScalarInter, Output) -> None:
    raise AssertionError("synthetic debug handlers must not execute the operator body")


@T.dataflow_intermediate
class PairInter:
    a: T.int32
    b: T.int32


@T.dataflow.iter(range=("begin", "end"))
def split_pair(batch: T.int32, Source) -> PairInter:
    raise AssertionError("synthetic debug handlers must not execute the operator body")


@T.dataflow.reduce
def reduce_pair(items: list[PairInter]) -> PairInter:
    raise AssertionError("synthetic debug handlers must not execute the operator body")


@T.dataflow.finalize
def finalize_pair(inter: PairInter, Output) -> None:
    raise AssertionError("synthetic debug handlers must not execute the operator body")


def make_scalar_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_scalar(Source="S"),
            task_args=("batch",),
            range_axis="kv",
        )
        .reduce(reduce_scalar())
        .finalize(finalize_scalar(Output="O"))
    )


def make_pair_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_pair(Source="S"),
            task_args=("batch",),
            range_axis="kv",
        )
        .reduce(reduce_pair())
        .finalize(finalize_pair(Output="O"))
    )


def require_executable_cuda() -> None:
    try:
        compiler = nvcc.get_nvcc_compiler()
    except Exception as err:
        pytest.skip(f"CUDA toolkit/NVCC unavailable: {err}")
    if not os.path.exists(compiler):
        pytest.skip(f"NVCC not found at {compiler}")

    try:
        from cuda.bindings import driver
    except Exception as err:
        pytest.skip(f"CUDA driver bindings unavailable: {err}")

    result = driver.cuInit(0)[0]
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA driver unavailable: {result}")
    result, count = driver.cuDeviceGetCount()
    if result != driver.CUresult.CUDA_SUCCESS or count == 0:
        pytest.skip(f"CUDA device unavailable: {result}, count={count}")

    result, device = driver.cuDeviceGet(0)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA device 0 unavailable: {result}")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA primary context unavailable: {result}")
    result = driver.cuCtxSetCurrent(context)[0]
    driver.cuDevicePrimaryCtxRelease(device)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA context activation failed: {result}")


def require_cluster_launch_cuda() -> None:
    require_executable_cuda()

    from cuda.bindings import driver

    result, device = driver.cuDeviceGet(0)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA device 0 unavailable: {result}")
    cluster_launch_attr = getattr(
        driver.CUdevice_attribute,
        "CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH",
        None,
    )
    if cluster_launch_attr is None:
        pytest.skip("CUDA driver bindings do not expose cluster launch support attribute")
    result, supported = driver.cuDeviceGetAttribute(cluster_launch_attr, device)
    if result != driver.CUresult.CUDA_SUCCESS or not supported:
        pytest.skip("CUDA device does not support cluster launch")


def check_cuda(driver, result, action: str) -> None:
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{action} failed: {result}")


def finalized_u32_values(
    compiled: df.DataflowCompiledProgram,
    result: df.DataflowExecutionResult,
) -> dict[int, int]:
    values: dict[int, int] = {}
    for instruction in compiled.plan.instructions:
        if instruction.opcode is not df.DataflowOpcode.FINALIZE:
            continue
        assert instruction.task_id is not None
        slot_id = instruction.input_slots[0]
        offset = compiled.packed_plan.slots[slot_id].global_offset
        values[instruction.task_id] = struct.unpack_from("<I", result.global_staging_bytes, offset)[0]
    return values


def test_scalar_debug_provider_generates_concrete_handler_source():
    compiled = dataflow_debug.compile(
        make_scalar_program(),
        handler=dataflow_debug.SCALAR_U32_HANDLER,
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [384]},
        block_size=128,
        wrapper_name="dataflow_scalar_debug_source",
    )

    assert compiled.wrapper_spec.handler_lowering == dataflow_debug.SCALAR_U32_HANDLER.name
    assert [handler.operator_kind for handler in compiled.wrapper_spec.handlers] == [
        "iter",
        "reduce",
        "finalize",
    ]
    assert tuple(source.handler_id for source in compiled.wrapper_spec.handler_sources) == tuple(
        handler.handler_id for handler in compiled.wrapper_spec.handlers
    )
    assert all(source.body_source for source in compiled.wrapper_spec.handler_sources)
    assert compiled.wrapper_spec.handler_helper_source
    assert compiled.dump_plan()["wrapper"]["handler_lowering"] == dataflow_debug.SCALAR_U32_HANDLER.name


def test_scalar_debug_provider_rejects_non_scalar_intermediate():
    with pytest.raises(NotImplementedError, match="exactly one scalar intermediate field"):
        dataflow_debug.compile(
            make_pair_program(),
            handler=dataflow_debug.SCALAR_U32_HANDLER,
            topology=df.GPUTopology(sm_count=1, cluster_size=1),
            range_lengths={"kv": [128]},
            block_size=128,
        )


def test_tensor_debug_provider_generates_tensor_handler_source():
    compiled = dataflow_debug.compile(
        make_scalar_program(),
        handler=dataflow_debug.TENSOR_U32_HANDLER,
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [384]},
        block_size=128,
        wrapper_name="dataflow_tensor_debug_source",
    )

    assert compiled.tensor_arg_plan.names == ("S", "O")
    assert tuple(source.handler_id for source in compiled.wrapper_spec.handler_sources) == tuple(
        handler.handler_id for handler in compiled.wrapper_spec.handlers
    )
    assert all(source.body_source for source in compiled.wrapper_spec.handler_sources)
    assert compiled.wrapper_spec.handler_helper_source


def test_scalar_debug_provider_executes_minimal_dataflow_on_cuda():
    require_executable_cuda()
    compiled = dataflow_debug.compile(
        make_scalar_program(),
        handler=dataflow_debug.SCALAR_U32_HANDLER,
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [384]},
        block_size=128,
        wrapper_name="dataflow_scalar_debug_executable",
    )

    assert finalized_u32_values(compiled, compiled()) == {0: 384}


def test_tensor_debug_provider_executes_tensor_dataflow_on_cuda():
    require_executable_cuda()
    from cuda.bindings import driver

    result, device = driver.cuDeviceGet(0)
    check_cuda(driver, result, "cuDeviceGet")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    check_cuda(driver, result, "cuDevicePrimaryCtxRetain")
    check_cuda(driver, driver.cuCtxSetCurrent(context)[0], "cuCtxSetCurrent")

    input_values = list(range(1, 513))
    input_bytes = len(input_values) * ctypes.sizeof(ctypes.c_uint32)
    output_bytes = 2 * ctypes.sizeof(ctypes.c_uint32)
    input_host = (ctypes.c_uint32 * len(input_values))(*input_values)
    output_zero = (ctypes.c_uint32 * 2)()
    output_host = (ctypes.c_uint32 * 2)()
    allocations = []
    try:
        result, input_ptr = driver.cuMemAlloc(input_bytes)
        check_cuda(driver, result, "cuMemAlloc(input)")
        allocations.append(input_ptr)
        result, output_ptr = driver.cuMemAlloc(output_bytes)
        check_cuda(driver, result, "cuMemAlloc(output)")
        allocations.append(output_ptr)
        check_cuda(driver, driver.cuMemcpyHtoD(input_ptr, input_host, input_bytes)[0], "cuMemcpyHtoD(input)")
        check_cuda(driver, driver.cuMemcpyHtoD(output_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(output)")

        compiled = dataflow_debug.compile(
            make_scalar_program(),
            handler=dataflow_debug.TENSOR_U32_HANDLER,
            topology=df.GPUTopology(sm_count=4, cluster_size=1),
            range_lengths={"kv": [512, 384]},
            task_extents=(2,),
            block_size=128,
            wrapper_name="dataflow_tensor_debug_executable",
        )
        compiled(S=int(input_ptr), O=int(output_ptr))
        check_cuda(driver, driver.cuMemcpyDtoH(output_host, output_ptr, output_bytes)[0], "cuMemcpyDtoH(output)")

        assert list(output_host) == [sum(input_values[:512]), sum(input_values[:384])]
    finally:
        for ptr in reversed(allocations):
            driver.cuMemFree(ptr)
        driver.cuDevicePrimaryCtxRelease(device)


@pytest.mark.parametrize(
    "sm_count,cluster_size,range_lengths,expected",
    (
        (2, 1, [384], {0: 384}),
        (4, 1, [512, 384], {0: 512, 1: 384}),
        (2, 1, [256, 256], {0: 256, 1: 256}),
    ),
)
def test_scalar_debug_provider_executes_hbm_dataflow_on_cuda(
    sm_count,
    cluster_size,
    range_lengths,
    expected,
):
    require_executable_cuda()
    compiled = dataflow_debug.compile(
        make_scalar_program(),
        handler=dataflow_debug.SCALAR_U32_HANDLER,
        topology=df.GPUTopology(sm_count=sm_count, cluster_size=cluster_size),
        range_lengths={"kv": range_lengths},
        task_extents=(len(range_lengths),),
        block_size=128,
        reuse_hbm_flags=len(range_lengths) > 1 and len(set(range_lengths)) == 1,
    )

    assert finalized_u32_values(compiled, compiled()) == expected
    assert any(comm.kind is df.DataflowCommKind.HBM_SEND for comm in compiled.plan.comms)


@pytest.mark.parametrize(
    "sm_count,range_lengths,expected",
    (
        (2, [384], {0: 384}),
        (4, [256, 512], {0: 256, 1: 512}),
    ),
)
def test_scalar_debug_provider_executes_cluster_dataflow_on_cuda(
    sm_count,
    range_lengths,
    expected,
):
    require_cluster_launch_cuda()
    compiled = dataflow_debug.compile(
        make_scalar_program(),
        handler=dataflow_debug.SCALAR_U32_HANDLER,
        topology=df.GPUTopology(sm_count=sm_count, cluster_size=2),
        range_lengths={"kv": range_lengths},
        task_extents=(len(range_lengths),),
        block_size=128,
    )

    result = compiled()
    assert result.cluster_dim == (2, 1, 1)
    assert any(comm.kind is df.DataflowCommKind.CLUSTER_SEND for comm in compiled.plan.comms)
    assert finalized_u32_values(compiled, result) == expected
