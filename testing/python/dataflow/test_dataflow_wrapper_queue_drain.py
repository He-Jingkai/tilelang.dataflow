from __future__ import annotations

import ctypes
import os

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from tilelang.contrib import nvcc
from tilelang.env import env


@T.dataflow_intermediate
class AttnInter:
    lse: T.float32
    o: T.Tensor((2, 8), T.float16)


@T.dataflow.iter(range=("kv_begin", "kv_end"))
def split_kv(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
    raise AssertionError("Dataflow iter body should not execute during queue-drain test")


@T.dataflow.reduce
def combine(items: list[AttnInter]) -> AttnInter:
    raise AssertionError("Dataflow reduce body should not execute during queue-drain test")


@T.dataflow.finalize
def finalize(inter: AttnInter, seq: T.int32, head: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during queue-drain test")


_DEBUG_KERNEL = "dataflow_queue_drain_debug"
_HANDLER_DISPATCH_KERNEL = "dataflow_handler_dispatch_debug"
_DEBUG_FIELD_COUNT = 8
_HANDLER_DEBUG_FIELD_COUNT = 18
_SENTINEL = 0xFFFFFFFF


def make_program():
    return (
        T.dataflow_program(task_domain=("batch", "head"), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
            range_axis="kv",
        )
        .reduce(combine())
        .finalize(finalize(Output="O"))
    )


def require_nvcc() -> None:
    try:
        compiler = nvcc.get_nvcc_compiler()
    except Exception as err:
        pytest.skip(f"CUDA toolkit/NVCC unavailable: {err}")
    if not os.path.exists(compiler):
        pytest.skip(f"NVCC not found at {compiler}")


def require_cuda_context():
    try:
        from cuda.bindings import driver
    except Exception as err:
        pytest.skip(f"CUDA driver bindings unavailable: {err}")

    result = driver.cuInit(0)[0]
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA driver unavailable: {result}")

    result, count = driver.cuDeviceGetCount()
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA device enumeration failed: {result}")
    if count == 0:
        pytest.skip("CUDA driver has no executable device")

    result, device = driver.cuDeviceGet(0)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA device 0 unavailable: {result}")

    result, context = driver.cuDevicePrimaryCtxRetain(device)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA primary context unavailable: {result}")

    result = driver.cuCtxSetCurrent(context)[0]
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuDevicePrimaryCtxRelease(device)
        pytest.skip(f"CUDA context activation failed: {result}")

    return driver, device


def queue_drain_source() -> str:
    return f"""#include <stdint.h>
#include <tl_templates/cuda/dataflow_runtime.h>

extern "C" __global__ void {_DEBUG_KERNEL}(
    const tl::DataflowInstruction *instructions,
    const uint32_t *queue_offsets,
    const uint32_t *queue_lengths,
    const tl::DataflowSlot *slots,
    const tl::DataflowCommPlan *comm_plans,
    void *arg_base,
    void *global_base,
    uint32_t *flags,
    uint32_t *debug,
    uint32_t max_queue_len) {{
  (void)slots;
  (void)comm_plans;
  (void)arg_base;
  (void)global_base;
  (void)flags;

  if (!tl::dataflow_is_leader_thread()) {{
    return;
  }}

  tl::DataflowQueue queue{{instructions, queue_offsets, queue_lengths}};
  uint32_t cta_rank = tl::dataflow_cta_rank_in_grid();
  uint32_t length = tl::dataflow_queue_length(queue, cta_rank);
  uint32_t clamped_length = tl::dataflow_min_u32(length, max_queue_len);

  for (uint32_t pc = 0; pc < clamped_length; ++pc) {{
    tl::DataflowInstruction inst = tl::dataflow_queue_load(queue, cta_rank, pc);
    uint32_t base = (cta_rank * max_queue_len + pc) * {_DEBUG_FIELD_COUNT}u;
    debug[base + 0] = inst.opcode;
    debug[base + 1] = inst.handler_id;
    debug[base + 2] = inst.task_id;
    debug[base + 3] = inst.arg_offset;
    debug[base + 4] = inst.comm_offset;
    debug[base + 5] = inst.comm_count;
    debug[base + 6] = inst.slot_id;
    debug[base + 7] = inst.flags;

    if (tl::dataflow_opcode_is_exit(inst)) {{
      break;
    }}
  }}
}}
"""


def handler_dispatch_source(handler_ids: tuple[int, ...]) -> str:
    cases = "\n".join(
        f"""  case {handler_id}u:
    record_empty_handler({handler_id}u, inst, handler_args, input_slots, task_coords,
                         cta_rank, local_pc, flat_index, debug, debug_record_count);
    break;"""
        for handler_id in handler_ids
    )
    return f"""#include <stdint.h>
#include <tl_templates/cuda/dataflow_runtime.h>

namespace tl_dataflow_dispatch_test {{

TL_DEVICE uint32_t first_or_sentinel(
    const uint32_t *values,
    uint32_t offset,
    uint32_t count) {{
  return count == 0 ? tl::kDataflowInvalidIndex : values[offset];
}}

TL_DEVICE void record_empty_handler(
    uint32_t dispatched_handler_id,
    const tl::DataflowInstruction &inst,
    const tl::DataflowHandlerArgs &handler_args,
    const uint32_t *input_slots,
    const uint32_t *task_coords,
    uint32_t cta_rank,
    uint32_t local_pc,
    uint32_t flat_index,
    uint32_t *debug,
    uint32_t debug_record_count) {{
  if (flat_index >= debug_record_count) {{
    return;
  }}

  uint32_t base = flat_index * {_HANDLER_DEBUG_FIELD_COUNT}u;
  debug[base + 0] = cta_rank;
  debug[base + 1] = local_pc;
  debug[base + 2] = inst.opcode;
  debug[base + 3] = inst.handler_id;
  debug[base + 4] = dispatched_handler_id;
  debug[base + 5] = inst.arg_offset;
  debug[base + 6] = handler_args.task_id;
  debug[base + 7] = handler_args.task_coord_offset;
  debug[base + 8] = handler_args.task_coord_count;
  debug[base + 9] = handler_args.range_begin;
  debug[base + 10] = handler_args.range_end;
  debug[base + 11] = handler_args.input_slot_offset;
  debug[base + 12] = handler_args.input_slot_count;
  debug[base + 13] = handler_args.output_slot;
  debug[base + 14] = first_or_sentinel(
      task_coords, handler_args.task_coord_offset, handler_args.task_coord_count);
  debug[base + 15] = first_or_sentinel(
      input_slots, handler_args.input_slot_offset, handler_args.input_slot_count);
  debug[base + 16] = inst.comm_count;
  debug[base + 17] = inst.slot_id;
}}

TL_DEVICE void dispatch_empty_handler(
    const tl::DataflowInstruction &inst,
    const tl::DataflowHandlerArgs &handler_args,
    const uint32_t *input_slots,
    const uint32_t *task_coords,
    uint32_t cta_rank,
    uint32_t local_pc,
    uint32_t flat_index,
    uint32_t *debug,
    uint32_t debug_record_count) {{
  switch (inst.handler_id) {{
{cases}
  default:
    break;
  }}
}}

}}  // namespace tl_dataflow_dispatch_test

extern "C" __global__ void {_HANDLER_DISPATCH_KERNEL}(
    const tl::DataflowInstruction *instructions,
    const uint32_t *queue_offsets,
    const uint32_t *queue_lengths,
    const tl::DataflowSlot *slots,
    const tl::DataflowCommPlan *comm_plans,
    void *arg_base,
    void *global_base,
    uint32_t *flags,
    const uint32_t *input_slots,
    const uint32_t *task_coords,
    uint32_t *debug,
    uint32_t debug_record_count) {{
  (void)slots;
  (void)comm_plans;
  (void)global_base;
  (void)flags;

  if (!tl::dataflow_is_leader_thread()) {{
    return;
  }}

  tl::DataflowQueue queue{{instructions, queue_offsets, queue_lengths}};
  uint32_t cta_rank = tl::dataflow_cta_rank_in_grid();
  uint32_t length = tl::dataflow_queue_length(queue, cta_rank);
  uint32_t flat_offset = tl::dataflow_queue_offset(queue, cta_rank);

  for (uint32_t pc = 0; pc < length; ++pc) {{
    tl::DataflowInstruction inst = tl::dataflow_queue_load(queue, cta_rank, pc);
    if (tl::dataflow_opcode_is_exit(inst)) {{
      break;
    }}

    uint32_t flat_index = flat_offset + pc;
    const tl::DataflowHandlerArgs &handler_args =
        tl::dataflow_handler_args(arg_base, inst.arg_offset);
    tl_dataflow_dispatch_test::dispatch_empty_handler(
        inst, handler_args, input_slots, task_coords, cta_rank, pc, flat_index,
        debug, debug_record_count);
  }}
}}
"""


def compile_source_cubin(source: str, arch: str) -> bytes:
    env.TILELANG_CLEANUP_TEMP_FILES = "1"
    return bytes(
        nvcc.compile_cuda(
            source,
            target_format="cubin",
            arch=arch,
            options=nvcc.default_compile_options(),
            verbose=True,
        )
    )


def compile_debug_cubin(arch: str) -> bytes:
    return compile_source_cubin(queue_drain_source(), arch)


def compile_handler_dispatch_cubin(
    arch: str,
    handler_ids: tuple[int, ...],
) -> bytes:
    return compile_source_cubin(handler_dispatch_source(handler_ids), arch)


def check_cuda(driver, result, action: str) -> None:
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{action} failed: {result}")


def device_alloc(driver, allocations: list, data: bytes):
    byte_count = max(len(data), 1)
    result, ptr = driver.cuMemAlloc(byte_count)
    check_cuda(driver, result, "cuMemAlloc")
    allocations.append(ptr)
    if data:
        host = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        check_cuda(driver, driver.cuMemcpyHtoD(ptr, host, len(data))[0], "cuMemcpyHtoD")
    return ptr


def device_alloc_u32(driver, allocations: list, values: list[int]):
    host = (ctypes.c_uint32 * len(values))(*values)
    result, ptr = driver.cuMemAlloc(ctypes.sizeof(host))
    check_cuda(driver, result, "cuMemAlloc")
    allocations.append(ptr)
    check_cuda(driver, driver.cuMemcpyHtoD(ptr, host, ctypes.sizeof(host))[0], "cuMemcpyHtoD")
    return ptr, ctypes.sizeof(host)


def load_function(driver, cubin: bytes, kernel_name: str = _DEBUG_KERNEL):
    image = (ctypes.c_ubyte * len(cubin)).from_buffer_copy(cubin)
    result, module = driver.cuModuleLoadData(image)
    if result != driver.CUresult.CUDA_SUCCESS:
        return result, None, None
    result, function = driver.cuModuleGetFunction(module, kernel_name.encode())
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuModuleUnload(module)
        return result, None, None
    return result, module, function


def launch_queue_drain(driver, function, package: df.DataflowLaunchPackage, ptrs: dict[str, object], max_queue_len: int) -> None:
    config = driver.CUlaunchConfig()
    config.gridDimX = package.queue.queue_count
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = 32
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 0
    config.hStream = driver.CUstream(0)

    arg_values = [
        int(ptrs["instructions"]),
        int(ptrs["queue_offsets"]),
        int(ptrs["queue_lengths"]),
        int(ptrs["slots"]),
        int(ptrs["comms"]),
        int(ptrs["args"]),
        int(ptrs["global_staging"]),
        int(ptrs["flags"]),
        int(ptrs["debug"]),
        max_queue_len,
    ]
    arg_types = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    check_cuda(
        driver,
        driver.cuLaunchKernelEx(config, function, (tuple(arg_values), tuple(arg_types)), 0)[0],
        "cuLaunchKernelEx",
    )
    check_cuda(driver, driver.cuCtxSynchronize()[0], "cuCtxSynchronize")


def launch_handler_dispatch(
    driver,
    function,
    package: df.DataflowLaunchPackage,
    ptrs: dict[str, object],
    debug_record_count: int,
) -> None:
    config = driver.CUlaunchConfig()
    config.gridDimX = package.queue.queue_count
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = 32
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 0
    config.hStream = driver.CUstream(0)

    arg_values = [
        int(ptrs["instructions"]),
        int(ptrs["queue_offsets"]),
        int(ptrs["queue_lengths"]),
        int(ptrs["slots"]),
        int(ptrs["comms"]),
        int(ptrs["args"]),
        int(ptrs["global_staging"]),
        int(ptrs["flags"]),
        int(ptrs["input_slots"]),
        int(ptrs["task_coords"]),
        int(ptrs["debug"]),
        debug_record_count,
    ]
    arg_types = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    check_cuda(
        driver,
        driver.cuLaunchKernelEx(config, function, (tuple(arg_values), tuple(arg_types)), 0)[0],
        "cuLaunchKernelEx",
    )
    check_cuda(driver, driver.cuCtxSynchronize()[0], "cuCtxSynchronize")


def assert_debug_matches_plan(packed: df.PackedRuntimePlan, debug_values: list[int], max_queue_len: int) -> None:
    for queue_index, (offset, length) in enumerate(zip(packed.queue_offsets, packed.queue_lengths)):
        for local_index in range(max_queue_len):
            base = (queue_index * max_queue_len + local_index) * _DEBUG_FIELD_COUNT
            actual = tuple(debug_values[base : base + _DEBUG_FIELD_COUNT])

            if local_index >= length:
                assert actual == (_SENTINEL,) * _DEBUG_FIELD_COUNT
                continue

            expected = packed.instructions[offset + local_index].to_tuple()
            assert actual == expected

            if expected[0] == 0:
                for remaining_index in range(local_index + 1, max_queue_len):
                    remaining_base = (queue_index * max_queue_len + remaining_index) * _DEBUG_FIELD_COUNT
                    remaining = tuple(debug_values[remaining_base : remaining_base + _DEBUG_FIELD_COUNT])
                    assert remaining == (_SENTINEL,) * _DEBUG_FIELD_COUNT
                break


def assert_handler_debug_matches_plan(packed: df.PackedRuntimePlan, debug_values: list[int]) -> None:
    handler_counts = dict.fromkeys(range(len(packed.handler_names)), 0)
    for queue_index, (offset, length) in enumerate(zip(packed.queue_offsets, packed.queue_lengths)):
        for local_index in range(length):
            instruction = packed.instructions[offset + local_index]
            if instruction.opcode == 0:
                break

            arg_index = instruction.arg_offset // df.ARG_STRUCT.size
            handler_args = packed.args[arg_index]
            base = (offset + local_index) * _HANDLER_DEBUG_FIELD_COUNT
            first_task_coord = _SENTINEL if handler_args.task_coord_count == 0 else packed.task_coords[handler_args.task_coord_offset]
            first_input_slot = _SENTINEL if handler_args.input_slot_count == 0 else packed.input_slots[handler_args.input_slot_offset]

            assert tuple(debug_values[base : base + _HANDLER_DEBUG_FIELD_COUNT]) == (
                queue_index,
                local_index,
                instruction.opcode,
                instruction.handler_id,
                instruction.handler_id,
                instruction.arg_offset,
                handler_args.task_id,
                handler_args.task_coord_offset,
                handler_args.task_coord_count,
                handler_args.range_begin,
                handler_args.range_end,
                handler_args.input_slot_offset,
                handler_args.input_slot_count,
                handler_args.output_slot,
                first_task_coord,
                first_input_slot,
                instruction.comm_count,
                instruction.slot_id,
            )
            handler_counts[instruction.handler_id] += 1

    for index, instruction in enumerate(packed.instructions):
        if instruction.opcode != 0:
            continue
        base = index * _HANDLER_DEBUG_FIELD_COUNT
        assert tuple(debug_values[base : base + _HANDLER_DEBUG_FIELD_COUNT]) == (_SENTINEL,) * _HANDLER_DEBUG_FIELD_COUNT

    assert all(count > 0 for count in handler_counts.values())


def test_executable_dataflow_wrapper_queue_drain_reads_packed_runtime_queue():
    require_nvcc()
    driver, device = require_cuda_context()

    compiled = dataflow_debug.compile(
        make_program(),
        handler=dataflow_debug.EMPTY_HANDLER,
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        block_size=128,
        task_extents=(2,),
        wrapper_name="dataflow_queue_drain_contract",
    )
    packed = compiled.packed_plan
    package = compiled.launch_package
    max_queue_len = max(packed.queue_lengths)
    debug_u32_count = package.queue.queue_count * max_queue_len * _DEBUG_FIELD_COUNT

    arch = compiled.target_capabilities.arch
    module = None
    allocations = []
    try:
        cubin = compile_debug_cubin(arch)
        result, module, function = load_function(driver, cubin)
        check_cuda(driver, result, f"cuModuleLoadData/cuModuleGetFunction for {arch}")

        debug_init = [_SENTINEL] * debug_u32_count
        debug_ptr, debug_bytes = device_alloc_u32(driver, allocations, debug_init)
        ptrs = {
            "instructions": device_alloc(driver, allocations, package.queue.instructions_bytes),
            "queue_offsets": device_alloc(driver, allocations, package.queue.offsets_bytes),
            "queue_lengths": device_alloc(driver, allocations, package.queue.lengths_bytes),
            "slots": device_alloc(driver, allocations, package.slots_bytes),
            "comms": device_alloc(driver, allocations, package.comms_bytes),
            "args": device_alloc(driver, allocations, package.args_bytes),
            "global_staging": device_alloc(driver, allocations, package.global_staging_bytes),
            "flags": device_alloc(driver, allocations, package.flags_bytes),
            "debug": debug_ptr,
        }

        launch_queue_drain(driver, function, package, ptrs, max_queue_len)

        out = (ctypes.c_uint32 * debug_u32_count)()
        check_cuda(driver, driver.cuMemcpyDtoH(out, debug_ptr, debug_bytes)[0], "cuMemcpyDtoH")
        assert_debug_matches_plan(packed, list(out), max_queue_len)
        assert arch == compiled.target_capabilities.arch
    finally:
        for ptr in reversed(allocations):
            driver.cuMemFree(ptr)
        if module is not None:
            driver.cuModuleUnload(module)
        driver.cuDevicePrimaryCtxRelease(device)


def test_executable_dataflow_wrapper_dispatches_empty_handlers_and_decodes_args():
    require_nvcc()
    driver, device = require_cuda_context()

    compiled = dataflow_debug.compile(
        make_program(),
        handler=dataflow_debug.EMPTY_HANDLER,
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [256, 128, 256, 128]},
        block_size=128,
        task_extents=(2, 2),
        wrapper_name="dataflow_handler_dispatch_contract",
    )
    packed = compiled.packed_plan
    package = compiled.launch_package
    debug_u32_count = len(packed.instructions) * _HANDLER_DEBUG_FIELD_COUNT

    arch = compiled.target_capabilities.arch
    module = None
    allocations = []
    try:
        cubin = compile_handler_dispatch_cubin(
            arch,
            tuple(handler.handler_id for handler in compiled.wrapper_spec.handlers),
        )
        result, module, function = load_function(driver, cubin, _HANDLER_DISPATCH_KERNEL)
        check_cuda(driver, result, f"cuModuleLoadData/cuModuleGetFunction for {arch}")

        debug_init = [_SENTINEL] * debug_u32_count
        debug_ptr, debug_bytes = device_alloc_u32(driver, allocations, debug_init)
        ptrs = {
            "instructions": device_alloc(driver, allocations, package.queue.instructions_bytes),
            "queue_offsets": device_alloc(driver, allocations, package.queue.offsets_bytes),
            "queue_lengths": device_alloc(driver, allocations, package.queue.lengths_bytes),
            "slots": device_alloc(driver, allocations, package.slots_bytes),
            "comms": device_alloc(driver, allocations, package.comms_bytes),
            "args": device_alloc(driver, allocations, package.args_bytes),
            "global_staging": device_alloc(driver, allocations, package.global_staging_bytes),
            "flags": device_alloc(driver, allocations, package.flags_bytes),
            "input_slots": device_alloc(driver, allocations, package.input_slots_bytes),
            "task_coords": device_alloc(driver, allocations, package.task_coords_bytes),
            "debug": debug_ptr,
        }

        launch_handler_dispatch(driver, function, package, ptrs, len(packed.instructions))

        out = (ctypes.c_uint32 * debug_u32_count)()
        check_cuda(driver, driver.cuMemcpyDtoH(out, debug_ptr, debug_bytes)[0], "cuMemcpyDtoH")
        assert_handler_debug_matches_plan(packed, list(out))
        assert arch == compiled.target_capabilities.arch
    finally:
        for ptr in reversed(allocations):
            driver.cuMemFree(ptr)
        if module is not None:
            driver.cuModuleUnload(module)
        driver.cuDevicePrimaryCtxRelease(device)
