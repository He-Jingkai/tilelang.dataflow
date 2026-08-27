from __future__ import annotations

import ctypes
import os

import pytest

import tilelang.language as T
import tilelang.dataflow as df


@T.dataflow_intermediate
class ScalarShard:
    value: T.int32


@T.dataflow_intermediate
class ScalarFull:
    value: T.int32


@T.dataflow_intermediate
class HiddenShard:
    value: T.int32


@T.dataflow.map(range=("expert_begin", "expert_end"))
def map_expert(expert: T.int32, token: T.int32, Input: T.Tensor((64,), T.int32)) -> ScalarShard:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Input[i]
    return ScalarShard(value=value)


@T.dataflow.map(range=("hidden_begin", "hidden_end"))
def map_hidden(parts: list[ScalarShard], expert: T.int32, token: T.int32, Bias: T.Tensor((64,), T.int32)) -> HiddenShard:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += parts[0].value + parts[1].value + Bias[i]
    return HiddenShard(value=value)


@T.dataflow.finalize
def finalize_hidden(hidden: HiddenShard, expert: T.int32, token: T.int32, Output: T.Tensor((64,), T.int32)) -> None:
    Output[T.dataflow_range_begin()] = hidden.value + expert + token


def make_program(*, cross_handler_handoff=None, include_finalize=True):
    map2_attrs = {}
    if cross_handler_handoff is not None:
        map2_attrs["cross_handler_handoff"] = cross_handler_handoff
        map2_attrs["range_tile"] = 64
    program = (
        T.dataflow_program(
            task_domain=("expert", "token"),
            dynamic_ranges={"expert_tile": "expert_tiles", "hidden_tile": "hidden_tiles"},
        )
        .map(map_expert(Input="Input"), name="map1", task_args=("expert", "token"), range_axis="expert_tile")
        .reshared(
            input="map1",
            name="gather",
            output_type=ScalarFull,
            physical_output_type=ScalarShard,
            output_arity=2,
            policy="hbm_all_gather",
        )
        .map(
            map_hidden(Bias="Bias"),
            name="map2",
            input="gather",
            task_args=("expert", "token"),
            range_axis="hidden_tile",
            **map2_attrs,
        )
    )
    if include_finalize:
        program.finalize(finalize_hidden(Output="Output"), input="map2")
    return program


def test_reshared_graph_uses_task_coord_overrides_for_compact_tasks():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": [64, 64], "hidden_tile": [64, 64]},
        block_size=64,
        include_exit=False,
        task_coord_overrides=((3, 0), (9, 0)),
    )

    compute = [
        inst for inst in plan.instructions if inst.opcode in {df.DataflowOpcode.MAP, df.DataflowOpcode.RESHARED, df.DataflowOpcode.FINALIZE}
    ]
    assert {inst.task_id for inst in compute} == {0, 1}
    assert {inst.task_coords for inst in compute} == {(3, 0), (9, 0)}


def test_reshared_graph_balances_weighted_tasks_across_clusters():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": [64, 64, 64], "hidden_tile": [64, 64, 64]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        stage_graph_task_weights=(10, 1, 1),
    )

    map1_clusters = {
        inst.task_id: plan.topology.cluster_id(inst.sm_id)
        for inst in plan.instructions
        if inst.attrs.get("stage_id") == 0 and inst.attrs["cluster_rank"] == 0
    }
    assert map1_clusters[0] == 0
    assert map1_clusters[1] == 1
    assert map1_clusters[2] == 1


def test_reshared_graph_accepts_explicit_cluster_assignment():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": [64, 64], "hidden_tile": [64, 64]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        stage_graph_cluster_assignment=(1, 0),
    )

    map1_clusters = {
        inst.task_id: plan.topology.cluster_id(inst.sm_id)
        for inst in plan.instructions
        if inst.attrs.get("stage_id") == 0 and inst.attrs["cluster_rank"] == 0
    }
    assert map1_clusters == {0: 1, 1: 0}


def test_reshared_graph_pairs_cross_handler_handoff_with_next_task_in_same_queue():
    plan = df.schedule(
        make_program(
            cross_handler_handoff={"consumer_stage": "map1", "stages": 2},
            include_finalize=False,
        ),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"expert_tile": [64, 64], "hidden_tile": [256, 256]},
        block_size=64,
        include_exit=False,
        task_coord_overrides=((3, 0), (9, 0)),
        stage_graph_cluster_assignment=(0, 0),
    )

    for queue in plan.queues.values():
        task0_map2 = [inst for inst in queue if inst.attrs.get("stage_id") == 2 and inst.task_id == 0]
        task1_map1 = [inst for inst in queue if inst.attrs.get("stage_id") == 0 and inst.task_id == 1]
        assert len(task0_map2) == 2
        assert len(task1_map1) == 1
        assert task0_map2[0].attrs["cross_handler_handoff_stage_count"] == 0
        assert task0_map2[1].attrs["cross_handler_handoff_stage_count"] == 2
        assert task0_map2[1].attrs["cross_handler_handoff_target_task_coords"] == (9, 0)
        assert task1_map1[0].attrs["cross_handler_handoff_stage_count"] == 2

        task1_map2 = [inst for inst in queue if inst.attrs.get("stage_id") == 2 and inst.task_id == 1]
        assert task1_map2[-1].attrs["cross_handler_handoff_stage_count"] == 0

    packed = df.pack_instruction_plan(plan)
    ordered = [inst for sm_id in range(2) for inst in plan.queue(sm_id)]
    producer_index = next(
        index
        for index, inst in enumerate(ordered)
        if inst.attrs.get("stage_id") == 2 and inst.task_id == 0 and inst.attrs.get("range_tile_index") == 1
    )
    producer_args = packed.args[packed.instructions[producer_index].arg_offset // df.ARG_STRUCT.size]
    assert producer_args.handoff_stage_count == 2
    assert producer_args.handoff_peer_arg_offset != df.UINT32_SENTINEL
    peer_args = packed.args[producer_args.handoff_peer_arg_offset // df.ARG_STRUCT.size]
    assert packed.task_coords[peer_args.task_coord_offset : peer_args.task_coord_offset + peer_args.task_coord_count] == (9, 0)


@pytest.mark.parametrize(
    ("handoff", "error"),
    [
        ({"consumer_stage": "map1", "stages": 0}, "stages must be positive"),
        (
            {"consumer_stage": "map1", "stages": 2.5},
            "stages must be a positive integer",
        ),
        (
            {"consumer_stage": "map2", "stages": 2},
            "consumer_stage must be an earlier, distinct stage",
        ),
    ],
)
def test_reshared_graph_rejects_invalid_cross_handler_handoff(handoff, error):
    with pytest.raises(ValueError, match=error):
        df.schedule(
            make_program(
                cross_handler_handoff=handoff,
            ),
            topology=df.GPUTopology(sm_count=2, cluster_size=2),
            range_lengths={"expert_tile": [64], "hidden_tile": [64]},
            block_size=64,
            include_exit=False,
        )


def test_reshared_graph_rejects_handoff_before_finalize_handler():
    with pytest.raises(ValueError, match="producer must be the terminal stage"):
        df.schedule(
            make_program(
                cross_handler_handoff={"consumer_stage": "map1", "stages": 2},
                include_finalize=True,
            ),
            topology=df.GPUTopology(sm_count=2, cluster_size=2),
            range_lengths={"expert_tile": [64], "hidden_tile": [64]},
            block_size=64,
            include_exit=False,
        )


def require_cluster_launch_cuda():
    try:
        from tilelang.contrib import nvcc
    except Exception as err:
        pytest.skip(f"CUDA toolkit/NVCC unavailable: {err}")
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
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA cluster launch support query failed: {result}")
    if not supported:
        pytest.skip("CUDA device does not support cluster launch")


def check_cuda(driver, result, action: str):
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{action} failed: {result}")


def test_hbm_reshared_graph_compiles_linked_handlers():
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"expert_tile": [64], "hidden_tile": [64]},
        block_size=32,
        task_extents=(1, 1),
        include_exit=False,
        force_hbm_comms=True,
        wrapper_name="dataflow_reshared_graph_compile",
    )

    assert compiled.plan.scheduler_policy == "stage_graph"
    assert [handler.operator_kind for handler in compiled.wrapper_spec.handlers] == ["map", "map", "finalize"]
    assert "dataflow_primfunc_map_hidden_device_kernel" in compiled.wrapper_source
    assert "handler_args.input_slot_count < 2u" in compiled.wrapper_source
    assert "handler_args.range_begin" in compiled.wrapper_source


def test_hbm_reshared_graph_executes_all_gather_on_cuda():
    if os.environ.get("TILELANG_DATAFLOW_RUN_RESHARED_CUDA") != "1":
        pytest.skip("set TILELANG_DATAFLOW_RUN_RESHARED_CUDA=1 to run the CUDA reshared graph launch test")
    require_cluster_launch_cuda()

    from cuda.bindings import driver

    result, device = driver.cuDeviceGet(0)
    check_cuda(driver, result, "cuDeviceGet")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    check_cuda(driver, result, "cuDevicePrimaryCtxRetain")
    result, previous_context = driver.cuCtxGetCurrent()
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, result, "cuCtxGetCurrent")

    allocations = []
    try:
        result = driver.cuCtxSetCurrent(context)[0]
        check_cuda(driver, result, "cuCtxSetCurrent")

        input_values = list(range(1, 65))
        bias_values = list(range(101, 165))
        output_values = [0] * 64
        byte_count = len(input_values) * ctypes.sizeof(ctypes.c_int32)
        input_host = (ctypes.c_int32 * 64)(*input_values)
        bias_host = (ctypes.c_int32 * 64)(*bias_values)
        output_zero = (ctypes.c_int32 * 64)(*output_values)
        output_host = (ctypes.c_int32 * 64)()

        result, input_ptr = driver.cuMemAlloc(byte_count)
        check_cuda(driver, result, "cuMemAlloc(input)")
        allocations.append(input_ptr)
        result, bias_ptr = driver.cuMemAlloc(byte_count)
        check_cuda(driver, result, "cuMemAlloc(bias)")
        allocations.append(bias_ptr)
        result, output_ptr = driver.cuMemAlloc(byte_count)
        check_cuda(driver, result, "cuMemAlloc(output)")
        allocations.append(output_ptr)

        check_cuda(driver, driver.cuMemcpyHtoD(input_ptr, input_host, byte_count)[0], "cuMemcpyHtoD(input)")
        check_cuda(driver, driver.cuMemcpyHtoD(bias_ptr, bias_host, byte_count)[0], "cuMemcpyHtoD(bias)")
        check_cuda(driver, driver.cuMemcpyHtoD(output_ptr, output_zero, byte_count)[0], "cuMemcpyHtoD(output)")

        compiled = df.compile(
            make_program(),
            topology=df.GPUTopology(sm_count=2, cluster_size=2),
            range_lengths={"expert_tile": [64], "hidden_tile": [64]},
            block_size=32,
            task_extents=(1, 1),
            include_exit=False,
            force_hbm_comms=True,
            wrapper_name="dataflow_reshared_graph_correctness",
        )

        compiled(Input=int(input_ptr), Bias=int(bias_ptr), Output=int(output_ptr))
        check_cuda(driver, driver.cuMemcpyDtoH(output_host, output_ptr, byte_count)[0], "cuMemcpyDtoH(output)")

        gathered = sum(input_values[:32]) + sum(input_values[32:])
        expected = [0] * 64
        expected[0] = 32 * gathered + sum(bias_values[:32])
        expected[32] = 32 * gathered + sum(bias_values[32:])
        assert list(output_host) == expected
    finally:
        for ptr in reversed(allocations):
            driver.cuMemFree(ptr)
        restore_result = driver.cuCtxSetCurrent(previous_context)[0]
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, restore_result, "cuCtxSetCurrent(previous)")
