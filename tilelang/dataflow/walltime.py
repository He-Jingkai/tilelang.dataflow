"""Per-SM wall-time diagnostics for Dataflow compiled programs."""

from __future__ import annotations

import ctypes
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import statistics
from typing import Any
from collections.abc import Sequence

from .executor import (
    DataflowExecutableKernel,
    DataflowPersistentExecutable,
    check_cuda,
    configure_function,
    device_alloc,
    device_alloc_empty,
    device_to_host,
    launch_attrs,
    launch_block_dim,
    load_function,
    retain_cuda_context,
    validate_cluster_launch,
    validate_target_compatibility,
)
from .tensor_args import TENSOR_ARG_STRUCT
from .tma_descriptors import build_tma_descriptor_handles
from .tma_descriptors import validate_tma_descriptor_runtime_tensors
from .wrapper import (
    DataflowWrapperInstrumentation,
    DataflowWrapperKernelParam,
    DataflowWrapperSpec,
    generate_wrapper_source,
)


TIMING_RECORD_WORDS = 8
TIMING_RECORD_ALIGNMENT = 8


@dataclass(frozen=True)
class DataflowWalltimeProfileResult:
    """Instruction-level per-SM timing diagnostics for one compiled Dataflow kernel."""

    detail_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    clock_khz: int
    report: str
    target_fingerprint: str | None = None
    artifact_fingerprint: str | None = None
    detail_csv: Path | None = None
    summary_csv: Path | None = None

    def write_csvs(self, output_dir: str | Path, prefix: str) -> DataflowWalltimeProfileResult:
        output_path = Path(output_dir)
        detail_csv = output_path / f"{prefix}_details.csv"
        summary_csv = output_path / f"{prefix}_summary.csv"
        write_csv(detail_csv, self.detail_rows)
        write_csv(summary_csv, self.summary_rows)
        return DataflowWalltimeProfileResult(
            detail_rows=self.detail_rows,
            summary_rows=self.summary_rows,
            clock_khz=self.clock_khz,
            report=self.report,
            target_fingerprint=self.target_fingerprint,
            artifact_fingerprint=self.artifact_fingerprint,
            detail_csv=detail_csv,
            summary_csv=summary_csv,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detail_rows": list(self.detail_rows),
            "summary_rows": list(self.summary_rows),
            "clock_khz": self.clock_khz,
            "report": self.report,
            "target_fingerprint": self.target_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "detail_csv": None if self.detail_csv is None else str(self.detail_csv),
            "summary_csv": None if self.summary_csv is None else str(self.summary_csv),
        }


@dataclass(frozen=True)
class DataflowPersistentWalltimeExecutable:
    """An ABI-preserving instrumented persistent executable.

    Timing records live in an aligned suffix of global staging. The production
    wrapper source and kernel parameter list remain unchanged.
    """

    executable: DataflowPersistentExecutable
    timing_global_offset: int
    timing_bytes: int
    max_queue_len: int

    def clear_timing_records(self) -> None:
        executable = self.executable
        driver = executable._driver
        if driver is None or "global_staging" not in executable._ptrs:
            raise RuntimeError("Dataflow persistent walltime executable must be open before reset")
        if self.timing_bytes <= 0:
            return
        check_cuda(
            driver,
            driver.cuMemsetD8(self.timing_device_pointer(), 0, self.timing_bytes)[0],
            "cuMemsetD8(timing_records)",
        )

    def read_timing_records(self) -> bytes:
        executable = self.executable
        driver = executable._driver
        if driver is None or "global_staging" not in executable._ptrs:
            raise RuntimeError("Dataflow persistent walltime executable must be open before readback")
        return device_to_host(driver, self.timing_device_pointer(), self.timing_bytes)

    def timing_device_pointer(self) -> int:
        return int(self.executable._ptrs["global_staging"]) + self.timing_global_offset


def profile_compiled_walltime(
    compiled: Any,
    *args: Any,
    repeat: int = 1,
    warmup: int = 0,
    span_only: bool = False,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    top_k: int = 12,
    stream: Any | None = None,
    **kwargs: Any,
) -> DataflowWalltimeProfileResult:
    """Run an instrumented copy of ``compiled`` and return per-SM timing rows.

    The production wrapper source is not modified. This regenerates an
    instrumented wrapper from its structured spec and registers the timing
    buffer through the wrapper instrumentation ABI. ``span_only`` records one
    begin/end pair per CTA, avoiding instruction-level probe overhead when the
    metric is the concurrent global kernel span.
    """

    if repeat <= 0:
        raise ValueError(f"repeat must be > 0, got {repeat}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    runtime_tensor_args = compiled.pack_tensor_args(*args, **kwargs)
    samples, clock_khz = run_timed_samples(
        compiled,
        tensor_args_bytes=runtime_tensor_args.tensor_args_bytes,
        tensor_metadata={spec.name: runtime_tensor_args.metadata[spec.index] for spec in runtime_tensor_args.plan.specs},
        sample_count=repeat,
        warmup_count=warmup,
        stream=stream,
        span_only=span_only,
    )
    detail_rows = tuple(
        row for sample, timing in enumerate(samples) for row in timing_rows(compiled, timing, clock_khz=clock_khz, sample=sample)
    )
    summary = tuple(summary_rows(detail_rows, clock_khz=clock_khz))
    report = format_walltime_balance_report(detail_rows, summary, top_k=top_k)
    result = DataflowWalltimeProfileResult(
        detail_rows=detail_rows,
        summary_rows=summary,
        clock_khz=clock_khz,
        report=report,
        target_fingerprint=getattr(compiled, "target_fingerprint", None),
        artifact_fingerprint=getattr(compiled, "artifact_fingerprint", None),
    )
    if output_dir is not None:
        label = prefix or getattr(compiled.wrapper_spec, "kernel_name", "dataflow_walltime")
        result = result.write_csvs(output_dir, label)
    return result


def build_walltime_instrumentation(
    *,
    max_queue_len: int,
    timing_global_offset: int | None = None,
    hbm_issue_probe_barrier_index: int | None = None,
    span_only: bool = False,
) -> DataflowWrapperInstrumentation:
    """Build instruction-level timing hooks without parsing generated CUDA."""

    if max_queue_len < 0:
        raise ValueError(f"max_queue_len must be non-negative, got {max_queue_len}")
    if timing_global_offset is not None:
        if timing_global_offset < 0:
            raise ValueError(f"timing_global_offset must be non-negative, got {timing_global_offset}")
        if timing_global_offset % TIMING_RECORD_ALIGNMENT:
            raise ValueError(f"timing_global_offset must be aligned to {TIMING_RECORD_ALIGNMENT} bytes, got {timing_global_offset}")
    if hbm_issue_probe_barrier_index is not None and hbm_issue_probe_barrier_index < 0:
        raise ValueError(f"hbm_issue_probe_barrier_index must be non-negative, got {hbm_issue_probe_barrier_index}")
    if span_only and hbm_issue_probe_barrier_index is not None:
        raise ValueError("span-only walltime does not support an HBM issue probe")
    namespace_source = (
        f"static constexpr uint32_t kDataflowTimingRecordStride = {max_queue_len}u;\n"
        f"static constexpr uint32_t kDataflowTimingRecordWords = {TIMING_RECORD_WORDS}u;\n"
    )
    if timing_global_offset is not None:
        namespace_source += f"static constexpr uint64_t kDataflowTimingGlobalOffset = {timing_global_offset}ull;\n"
    namespace_source += (
        "TL_DEVICE uint64_t dataflow_read_globaltimer() {\n"
        "  unsigned long long value;\n"
        '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));\n'
        "  return static_cast<uint64_t>(value);\n"
        "}"
    )
    extra_kernel_params = (DataflowWrapperKernelParam(name="timing_records", c_type="uint64_t *"),) if timing_global_offset is None else ()
    kernel_prologue = ""
    if timing_global_offset is not None:
        kernel_prologue = (
            "auto *timing_records = reinterpret_cast<uint64_t *>(\n"
            "    reinterpret_cast<uint8_t *>(global_base) +\n"
            "    tl_dataflow_generated::kDataflowTimingGlobalOffset);"
        )
    if span_only:
        return DataflowWrapperInstrumentation(
            name="cta_span_walltime",
            extra_kernel_params=extra_kernel_params,
            namespace_source=namespace_source,
            kernel_prologue=kernel_prologue,
            before_loop=(
                "uint64_t timing_span_start = 0;\n"
                "uint64_t timing_span_end = 0;\n"
                "if (tl::dataflow_is_leader_thread()) {\n"
                "  timing_span_start = tl_dataflow_generated::dataflow_read_globaltimer();\n"
                "  timing_span_end = timing_span_start;\n"
                "}"
            ),
            after_send=(
                "if (tl::dataflow_is_leader_thread()) {\n  timing_span_end = tl_dataflow_generated::dataflow_read_globaltimer();\n}"
            ),
            kernel_epilogue=(
                "if (tl::dataflow_is_leader_thread()) {\n"
                "  uint64_t timing_base =\n"
                "      static_cast<uint64_t>(queue_rank) *\n"
                "      tl_dataflow_generated::kDataflowTimingRecordStride *\n"
                "      tl_dataflow_generated::kDataflowTimingRecordWords;\n"
                "  timing_records[timing_base + 0] = timing_span_start;\n"
                "  timing_records[timing_base + 1] = timing_span_start;\n"
                "  timing_records[timing_base + 2] = timing_span_end;\n"
                "  timing_records[timing_base + 3] = timing_span_end;\n"
                "  timing_records[timing_base + 4] = static_cast<uint64_t>(tl::dataflow_smid());\n"
                "}"
            ),
        )
    probe_before_recv = ""
    probe_after_handler = ""
    recorded_slot_expression = "static_cast<uint64_t>(inst.slot_id)"
    if hbm_issue_probe_barrier_index is not None:
        probe_word = hbm_issue_probe_barrier_index >> 5
        probe_mask = 1 << (hbm_issue_probe_barrier_index & 31)
        probe_before_recv = (
            "uint32_t timing_hbm_issued_before_recv = 0u;\n"
            "uint32_t timing_hbm_issued_after_handler = 0u;\n"
            "if (tl::dataflow_is_leader_thread()) {\n"
            f"  timing_hbm_issued_before_recv = "
            f"(hbm_recv_issued[{probe_word}u] & {probe_mask}u) != 0u;\n"
            "}"
        )
        probe_after_handler = (
            "if (tl::dataflow_is_leader_thread()) {\n"
            f"  timing_hbm_issued_after_handler = "
            f"(hbm_recv_issued[{probe_word}u] & {probe_mask}u) != 0u;\n"
            "}"
        )
        recorded_slot_expression = (
            "static_cast<uint64_t>(timing_hbm_issued_before_recv) | (static_cast<uint64_t>(timing_hbm_issued_after_handler) << 1u)"
        )
    return DataflowWrapperInstrumentation(
        name="instruction_walltime",
        extra_kernel_params=extra_kernel_params,
        namespace_source=namespace_source,
        kernel_prologue=kernel_prologue,
        before_recv=(
            "uint64_t timing_start = 0;\n"
            "uint64_t timing_after_recv = 0;\n"
            "uint64_t timing_after_handler = 0;\n"
            "uint64_t timing_after_send = 0;\n"
            "if (tl::dataflow_is_leader_thread()) {\n"
            "  timing_start = tl_dataflow_generated::dataflow_read_globaltimer();\n"
            "}\n"
            f"{probe_before_recv}"
        ),
        after_recv=("if (tl::dataflow_is_leader_thread()) {\n  timing_after_recv = tl_dataflow_generated::dataflow_read_globaltimer();\n}"),
        after_handler=(
            "if (tl::dataflow_is_leader_thread()) {\n  timing_after_handler = tl_dataflow_generated::dataflow_read_globaltimer();\n}\n"
            f"{probe_after_handler}"
        ),
        after_send=(
            "if (tl::dataflow_is_leader_thread()) {\n"
            "  timing_after_send = tl_dataflow_generated::dataflow_read_globaltimer();\n"
            "  uint64_t timing_base =\n"
            "      (static_cast<uint64_t>(queue_rank) * "
            "tl_dataflow_generated::kDataflowTimingRecordStride + pc) *\n"
            "      tl_dataflow_generated::kDataflowTimingRecordWords;\n"
            "  timing_records[timing_base + 0] = timing_start;\n"
            "  timing_records[timing_base + 1] = timing_after_recv;\n"
            "  timing_records[timing_base + 2] = timing_after_handler;\n"
            "  timing_records[timing_base + 3] = timing_after_send;\n"
            "  timing_records[timing_base + 4] = static_cast<uint64_t>(tl::dataflow_smid());\n"
            "  timing_records[timing_base + 5] = static_cast<uint64_t>(inst.opcode);\n"
            "  timing_records[timing_base + 6] = static_cast<uint64_t>(inst.task_id);\n"
            f"  timing_records[timing_base + 7] = {recorded_slot_expression};\n"
            "}"
        ),
    )


def generate_walltime_wrapper_source(
    spec: DataflowWrapperSpec,
    *,
    max_queue_len: int,
    timing_global_offset: int | None = None,
    hbm_issue_probe_barrier_index: int | None = None,
    span_only: bool = False,
) -> str:
    instrumentation = build_walltime_instrumentation(
        max_queue_len=max_queue_len,
        timing_global_offset=timing_global_offset,
        hbm_issue_probe_barrier_index=hbm_issue_probe_barrier_index,
        span_only=span_only,
    )
    return generate_wrapper_source(spec, instrumentation=instrumentation)


def build_persistent_walltime_executable(
    compiled: Any,
    *args: Any,
    **kwargs: Any,
) -> DataflowPersistentWalltimeExecutable:
    """Build an unopened persistent executable with ABI-preserving timing."""

    max_queue_len = max((len(queue) for queue in compiled.plan.queues.values()), default=0)
    timing_bytes = compiled.launch_package.queue.queue_count * max_queue_len * TIMING_RECORD_WORDS * 8
    original_global_bytes = compiled.launch_package.global_staging_bytes
    timing_global_offset = align_up(
        len(original_global_bytes),
        TIMING_RECORD_ALIGNMENT,
    )
    padding = bytes(timing_global_offset - len(original_global_bytes))
    timed_package = replace(
        compiled.launch_package,
        global_staging_bytes=original_global_bytes + padding + bytes(timing_bytes),
    )
    production_executable = compiled.persistent_executable(*args, **kwargs)
    timed_executable = replace(
        production_executable,
        source=generate_walltime_wrapper_source(
            compiled.wrapper_spec,
            max_queue_len=max_queue_len,
            timing_global_offset=timing_global_offset,
        ),
        launch_package=timed_package,
    )
    return DataflowPersistentWalltimeExecutable(
        executable=timed_executable,
        timing_global_offset=timing_global_offset,
        timing_bytes=timing_bytes,
        max_queue_len=max_queue_len,
    )


def run_timed_samples(
    compiled: Any,
    *,
    tensor_args_bytes: bytes,
    tensor_metadata: dict[str, Any],
    sample_count: int,
    warmup_count: int,
    stream: Any | None = None,
    hbm_issue_probe_barrier_index: int | None = None,
    span_only: bool = False,
) -> tuple[list[bytes], int]:
    max_queue_len = max((len(queue) for queue in compiled.plan.queues.values()), default=0)
    timing_bytes = compiled.launch_package.queue.queue_count * max_queue_len * TIMING_RECORD_WORDS * 8
    instrumented_source = generate_walltime_wrapper_source(
        compiled.wrapper_spec,
        max_queue_len=max_queue_len,
        hbm_issue_probe_barrier_index=hbm_issue_probe_barrier_index,
        span_only=span_only,
    )

    driver, device = retain_cuda_context(compiled.target_capabilities.device_ordinal)
    module = None
    allocations: list[Any] = []
    try:
        clock_khz = query_clock_khz(driver, device)
        actual_target = validate_target_compatibility(
            compiled.target_capabilities,
            driver,
            device,
            artifact_fingerprint=compiled.artifact_fingerprint,
        )
        validate_cluster_launch(
            driver,
            device,
            compiled.plan.topology,
            compiled.launch_package,
            target_capabilities=actual_target,
        )
        cubin, _arch = DataflowExecutableKernel(
            kernel_name=compiled.wrapper_spec.kernel_name,
            source=instrumented_source,
            launch_package=compiled.launch_package,
            topology=compiled.plan.topology,
            tensor_args_bytes=tensor_args_bytes,
            target_capabilities=compiled.target_capabilities,
            artifact_fingerprint=compiled.artifact_fingerprint,
            options=compiled.options,
        ).compile_cubin()
        result, module, function = load_function(driver, cubin, compiled.wrapper_spec.kernel_name)
        check_cuda(driver, result, "cuModuleLoadData/cuModuleGetFunction")
        configure_function(
            driver,
            function,
            compiled.launch_package,
            compiled.plan.topology,
            block_dim=launch_block_dim(compiled.options),
        )
        tma_descriptor_specs = tuple(getattr(compiled.wrapper_spec, "tma_descriptors", ()))
        tensor_data_ptrs = tensor_data_ptrs_from_bytes(
            tensor_args_bytes,
            compiled_tensor_indices(compiled),
        )
        if tma_descriptor_specs:
            validate_tma_descriptor_runtime_tensors(
                tma_descriptor_specs,
                tensor_data_ptrs=tensor_data_ptrs,
                tensor_metadata=tensor_metadata,
                expected_device_ordinal=(compiled.target_capabilities.device_ordinal),
            )
        tma_descriptor_handles = (
            build_tma_descriptor_handles(
                tma_descriptor_specs,
                tensor_data_ptrs=tensor_data_ptrs,
            )
            if tma_descriptor_specs
            else ()
        )

        ptrs = {
            "instructions": device_alloc(driver, allocations, compiled.launch_package.queue.instructions_bytes),
            "queue_offsets": device_alloc(driver, allocations, compiled.launch_package.queue.offsets_bytes),
            "queue_lengths": device_alloc(driver, allocations, compiled.launch_package.queue.lengths_bytes),
            "slots": device_alloc(driver, allocations, compiled.launch_package.slots_bytes),
            "comms": device_alloc(driver, allocations, compiled.launch_package.comms_bytes),
            "barrier_init_offsets": device_alloc(
                driver,
                allocations,
                compiled.launch_package.queue.barrier_init_offsets_bytes,
            ),
            "barrier_init_lengths": device_alloc(
                driver,
                allocations,
                compiled.launch_package.queue.barrier_init_lengths_bytes,
            ),
            "barrier_init_indices": device_alloc(
                driver,
                allocations,
                compiled.launch_package.queue.barrier_init_indices_bytes,
            ),
            "args": device_alloc(driver, allocations, compiled.launch_package.args_bytes),
            "tensor_args": device_alloc(driver, allocations, tensor_args_bytes),
            "input_slots": device_alloc(driver, allocations, compiled.launch_package.input_slots_bytes),
            "task_coords": device_alloc(driver, allocations, compiled.launch_package.task_coords_bytes),
            "global_staging": device_alloc_empty(driver, allocations, len(compiled.launch_package.global_staging_bytes)),
            "flags": device_alloc_empty(driver, allocations, len(compiled.launch_package.flags_bytes)),
            "timing_records": device_alloc_empty(driver, allocations, timing_bytes),
        }

        config = driver.CUlaunchConfig()
        config.gridDimX = compiled.launch_package.queue.queue_count
        config.gridDimY = 1
        config.gridDimZ = 1
        block_dim = launch_block_dim(compiled.options)
        config.blockDimX, config.blockDimY, config.blockDimZ = block_dim
        config.sharedMemBytes = compiled.launch_package.shared_memory_bytes
        config.hStream = driver.CUstream(0) if stream is None else stream
        attrs = launch_attrs(
            driver,
            compiled.plan.topology.cluster_size,
            compiled.launch_package.cluster_scheduling_policy,
        )
        if attrs:
            config.numAttrs = len(attrs)
            config.attrs = attrs

        arg_values = [
            int(ptrs["instructions"]),
            int(ptrs["queue_offsets"]),
            int(ptrs["queue_lengths"]),
            int(ptrs["slots"]),
            int(ptrs["comms"]),
            int(ptrs["barrier_init_offsets"]),
            int(ptrs["barrier_init_lengths"]),
            int(ptrs["barrier_init_indices"]),
            int(ptrs["args"]),
            int(ptrs["tensor_args"]),
            int(ptrs["input_slots"]),
            int(ptrs["task_coords"]),
            int(ptrs["global_staging"]),
            int(ptrs["flags"]),
            int(ptrs["timing_records"]),
        ]
        arg_types = [ctypes.c_void_p] * len(arg_values)
        if tma_descriptor_specs:
            expected_names = [spec.name for spec in tma_descriptor_specs]
            handle_names = [handle.spec.name for handle in tma_descriptor_handles]
            if handle_names != expected_names:
                raise RuntimeError(f"Dataflow TMA descriptor launch order mismatch: expected {expected_names!r}, got {handle_names!r}")
            arg_values.extend(handle.handle for handle in tma_descriptor_handles)
            arg_types.extend([None] * len(tma_descriptor_handles))
        kernel_params = (tuple(arg_values), tuple(arg_types))

        samples: list[bytes] = []
        for sample_index in range(warmup_count + sample_count):
            if len(compiled.launch_package.flags_bytes):
                check_cuda(
                    driver,
                    driver.cuMemsetD8(ptrs["flags"], 0, len(compiled.launch_package.flags_bytes))[0],
                    "cuMemsetD8(flags)",
                )
            check_cuda(
                driver,
                driver.cuMemsetD8(ptrs["timing_records"], 0, timing_bytes)[0],
                "cuMemsetD8(timing_records)",
            )
            check_cuda(
                driver,
                driver.cuLaunchKernelEx(config, function, kernel_params, 0)[0],
                "cuLaunchKernelEx",
            )
            check_cuda(driver, driver.cuCtxSynchronize()[0], "cuCtxSynchronize")
            if sample_index >= warmup_count:
                samples.append(device_to_host(driver, ptrs["timing_records"], timing_bytes))
        return samples, clock_khz
    finally:
        for ptr in reversed(allocations):
            driver.cuMemFree(ptr)
        if module is not None:
            driver.cuModuleUnload(module)
        driver.cuDevicePrimaryCtxRelease(device)


def timing_rows(
    compiled: Any,
    timing: bytes,
    *,
    clock_khz: int,
    sample: int,
    hbm_issue_probe_barrier_index: int | None = None,
) -> list[dict[str, Any]]:
    del clock_khz
    max_queue_len = max((len(queue) for queue in compiled.plan.queues.values()), default=0)
    queue_count = compiled.launch_package.queue.queue_count
    values = memoryview(timing).cast("Q")
    rows: list[dict[str, Any]] = []
    for logical_sm in range(queue_count):
        queue = compiled.plan.queue(logical_sm)
        for pc, inst in enumerate(queue):
            base = (logical_sm * max_queue_len + pc) * TIMING_RECORD_WORDS
            start, after_recv, after_handler, after_send, real_smid, opcode, task_id, slot_id = values[base : base + TIMING_RECORD_WORDS]
            if after_send == 0:
                continue
            recv_ns = max(0, after_recv - start)
            handler_ns = max(0, after_handler - after_recv)
            send_ns = max(0, after_send - after_handler)
            total_ns = max(0, after_send - start)

            task_range = inst.task_range
            row = {
                "sample": sample,
                "logical_sm": logical_sm,
                "real_smid": int(real_smid),
                "pc": pc,
                "instruction_id": inst.instruction_id,
                "opcode": inst.opcode.value,
                "operator_name": inst.operator_name,
                "recorded_opcode": int(opcode),
                "task_id": inst.task_id,
                "recorded_task_id": int(task_id),
                "range_begin": "" if task_range is None else task_range.begin,
                "range_end": "" if task_range is None else task_range.end,
                "input_slots": " ".join(str(slot) for slot in inst.input_slots),
                "output_slot": "" if inst.output_slot is None else inst.output_slot,
                "recorded_slot_id": (int(slot_id) if hbm_issue_probe_barrier_index is None else ""),
                "recv_us": recv_ns / 1000.0,
                "handler_us": handler_ns / 1000.0,
                "send_us": send_ns / 1000.0,
                "total_us": total_ns / 1000.0,
                "start_time_ns": int(start),
                "recv_end_time_ns": int(after_recv),
                "handler_end_time_ns": int(after_handler),
                "end_time_ns": int(after_send),
                "recv_ns": int(recv_ns),
                "handler_ns": int(handler_ns),
                "send_ns": int(send_ns),
                "total_ns": int(total_ns),
            }
            if hbm_issue_probe_barrier_index is not None:
                row["hbm_issued_before_recv"] = int(slot_id) & 1
                row["hbm_issued_after_handler"] = (int(slot_id) >> 1) & 1
            rows.append(row)
    return rows


def summary_rows(detail_rows: Sequence[dict[str, Any]], *, clock_khz: int | None = None) -> list[dict[str, Any]]:
    del clock_khz
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    sample_bounds: dict[int, tuple[int, int]] = {}
    for row in detail_rows:
        sample = int(row["sample"])
        grouped.setdefault((sample, int(row["logical_sm"])), []).append(row)
        start_time_ns = int(row["start_time_ns"])
        end_time_ns = int(row["end_time_ns"])
        if sample not in sample_bounds:
            sample_bounds[sample] = (start_time_ns, end_time_ns)
        else:
            sample_first_start_time_ns, sample_last_end_time_ns = sample_bounds[sample]
            sample_bounds[sample] = (
                min(sample_first_start_time_ns, start_time_ns),
                max(sample_last_end_time_ns, end_time_ns),
            )

    rows: list[dict[str, Any]] = []
    for (sample, logical_sm), items in sorted(grouped.items()):
        first_start = min(int(row["start_time_ns"]) for row in items)
        last_end = max(int(row["end_time_ns"]) for row in items)
        sample_first_start, sample_last_end = sample_bounds[sample]
        active_span_us = (last_end - first_start) / 1000.0
        start_offset_us = (first_start - sample_first_start) / 1000.0
        finish_time_us = (last_end - sample_first_start) / 1000.0
        finish_slack_us = (sample_last_end - last_end) / 1000.0
        total_us_sum = sum(float(row["total_us"]) for row in items)
        opcode_counts = {opcode: sum(1 for row in items if row["opcode"] == opcode) for opcode in ("iter", "reduce_update", "finalize")}
        rows.append(
            {
                "sample": sample,
                "logical_sm": logical_sm,
                "real_smids": " ".join(str(smid) for smid in sorted({int(row["real_smid"]) for row in items})),
                "count": len(items),
                "active_span_us": active_span_us,
                "start_offset_us": start_offset_us,
                "finish_time_us": finish_time_us,
                "finish_slack_us": finish_slack_us,
                "total_us_sum": total_us_sum,
                "idle_gap_us": active_span_us - total_us_sum,
                "handler_us_sum": sum(float(row["handler_us"]) for row in items),
                "recv_us_sum": sum(float(row["recv_us"]) for row in items),
                "send_us_sum": sum(float(row["send_us"]) for row in items),
                "max_task_us": max(float(row["total_us"]) for row in items),
                "iter_count": opcode_counts["iter"],
                "reduce_count": opcode_counts["reduce_update"],
                "finalize_count": opcode_counts["finalize"],
                "first_start_time_ns": first_start,
                "last_end_time_ns": last_end,
                "sample_first_start_time_ns": sample_first_start,
                "sample_last_end_time_ns": sample_last_end,
            }
        )
    return rows


def format_walltime_balance_report(
    detail_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    *,
    top_k: int = 12,
) -> str:
    lines: list[str] = ["Dataflow per-SM walltime balance report"]

    by_opcode: dict[str, list[float]] = {}
    for row in detail_rows:
        by_opcode.setdefault(str(row["opcode"]), []).append(float(row["total_us"]))
    lines.append("opcode total_us stats:")
    for opcode, values in sorted(by_opcode.items()):
        stats = compute_statistics(values)
        lines.append(
            f"  {opcode}: n={len(values)} mean={stats['mean']:.3f}us "
            f"median={stats['median']:.3f}us min={stats['min']:.3f}us max={stats['max']:.3f}us"
        )

    finish_by_sample: dict[int, list[float]] = {}
    for row in summary_rows:
        finish_by_sample.setdefault(int(row["sample"]), []).append(float(row["finish_time_us"]))
    lines.append("SM finish_time_us stats by sample:")
    for sample, values in sorted(finish_by_sample.items()):
        min_finish = min(values)
        max_finish = max(values)
        lines.append(
            f"  sample={sample}: n={len(values)} finish min/p50/p95/max/spread: "
            f"{min_finish:.3f} / {percentile(values, 50.0):.3f} / "
            f"{percentile(values, 95.0):.3f} / {max_finish:.3f} / "
            f"{max_finish - min_finish:.3f} us"
        )

    top_finish_rows = sorted(summary_rows, key=lambda row: float(row["finish_time_us"]), reverse=True)[:top_k]
    lines.append("top logical SMs by finish_time_us:")
    for row in top_finish_rows:
        lines.append(
            f"  sample={row['sample']} logical_sm={int(row['logical_sm']):3d} "
            f"real_smids=[{row['real_smids']}] count={int(row['count']):2d} "
            f"finish={float(row['finish_time_us']):.3f}us "
            f"slack={float(row.get('finish_slack_us', 0.0)):.3f}us "
            f"start_offset={float(row.get('start_offset_us', 0.0)):.3f}us "
            f"active_span={float(row['active_span_us']):.3f}us "
            f"recv={float(row['recv_us_sum']):.3f}us "
            f"handler={float(row['handler_us_sum']):.3f}us "
            f"send={float(row['send_us_sum']):.3f}us"
        )

    top_active_rows = sorted(summary_rows, key=lambda row: float(row["active_span_us"]), reverse=True)[:top_k]
    lines.append("top logical SMs by active_span_us:")
    for row in top_active_rows:
        lines.append(
            f"  sample={row['sample']} logical_sm={int(row['logical_sm']):3d} "
            f"real_smids=[{row['real_smids']}] count={int(row['count']):2d} "
            f"active_span={float(row['active_span_us']):.3f}us "
            f"sum_tasks={float(row.get('total_us_sum', 0.0)):.3f}us "
            f"idle_gap={float(row['idle_gap_us']):.3f}us "
            f"max_task={float(row.get('max_task_us', 0.0)):.3f}us"
        )

    return "\n".join(lines)


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tensor_data_ptrs_from_bytes(tensor_args_bytes: bytes, tensor_indices: dict[str, int]) -> dict[str, int]:
    ptrs: dict[str, int] = {}
    for tensor_name, index in tensor_indices.items():
        offset = int(index) * TENSOR_ARG_STRUCT.size
        if offset + TENSOR_ARG_STRUCT.size > len(tensor_args_bytes):
            ptrs[tensor_name] = 0
            continue
        ptrs[tensor_name] = int(TENSOR_ARG_STRUCT.unpack_from(tensor_args_bytes, offset)[0])
    return ptrs


def compiled_tensor_indices(compiled: Any) -> dict[str, int]:
    return {spec.name: spec.index for spec in compiled.tensor_arg_plan.specs}


def query_clock_khz(driver: Any, device: Any) -> int:
    from cuda.bindings import driver as cuda_driver

    result, value = driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        device,
    )
    check_cuda(driver, result, "cuDeviceGetAttribute(CLOCK_RATE)")
    return int(value)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


_run_timed_samples = run_timed_samples
_timing_rows = timing_rows
_summary_rows = summary_rows
_write_csv_rows = write_csv
