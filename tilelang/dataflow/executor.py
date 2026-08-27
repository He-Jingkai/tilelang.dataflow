"""Minimal CUDA execution support for Dataflow wrapper skeletons."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import time
from typing import Any

from tilelang.contrib import nvcc
from tilelang.utils.target_capabilities import (
    TargetCapabilityResolutionError,
    TargetCapabilitySnapshot,
    resolve_target_capabilities_from_device,
)

from .launch import (
    DATAFLOW_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING,
    DataflowLaunchPackage,
)
from .progress import DataflowProgressLogger, dataflow_progress_enabled
from .tensor_args import DataflowRuntimeTensorMetadata, TENSOR_ARG_STRUCT
from .runtime import COMM_STRUCT
from .tma_descriptors import (
    DataflowTMADescriptorHandle,
    DataflowTMADescriptorSpec,
    build_tma_descriptor_handles,
    validate_tma_descriptor_runtime_tensors,
)
from .topology import GPUTopology


@dataclass(frozen=True)
class DataflowExecutionResult:
    kernel_name: str
    arch: str
    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]
    shared_memory_bytes: int
    instruction_count: int
    global_staging_bytes: bytes = b""
    cluster_dim: tuple[int, int, int] = (1, 1, 1)
    tensor_arg_count: int = 0
    target_fingerprint: str = ""
    artifact_fingerprint: str = ""


class DataflowTargetCompatibilityError(RuntimeError):
    """Raised before module load when an artifact targets another device."""


@dataclass(frozen=True)
class DataflowPersistentProfileResult:
    execution: DataflowExecutionResult
    timings_ms: dict[str, float]


@dataclass
class DataflowExecutableKernel:
    kernel_name: str
    source: str
    launch_package: DataflowLaunchPackage
    topology: GPUTopology
    tensor_args_bytes: bytes = b""
    tma_descriptor_specs: tuple[DataflowTMADescriptorSpec, ...] = ()
    tma_descriptor_handles: tuple[DataflowTMADescriptorHandle, ...] = ()
    tma_tensor_indices: dict[str, int] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    target_capabilities: TargetCapabilitySnapshot | None = None
    artifact_fingerprint: str | None = None
    _cubin: bytes | None = field(default=None, init=False, repr=False)
    _arch: str | None = field(default=None, init=False, repr=False)

    def launch(self, *, stream: Any | None = None) -> DataflowExecutionResult:
        progress = progress_logger(self.options, "[dataflow.launch]")
        launch_started = progress.start("launch")
        phase_started = progress.start("retain CUDA context")
        target_capabilities = require_target_capabilities(self.target_capabilities)
        driver, device = retain_cuda_context(target_capabilities.device_ordinal)
        progress.finish("retain CUDA context", phase_started)
        module = None
        allocations: list[Any] = []
        try:
            phase_started = progress.start("validate target and cluster launch")
            actual_capabilities = validate_target_compatibility(
                target_capabilities,
                driver,
                device,
                artifact_fingerprint=self.artifact_fingerprint,
            )
            validate_cluster_launch(
                driver,
                device,
                self.topology,
                self.launch_package,
                target_capabilities=actual_capabilities,
            )
            progress.finish("validate target and cluster launch", phase_started)
            cubin, arch = self.compile_cubin()
            phase_started = progress.start("module load")
            result, module, function = load_function(driver, cubin, self.kernel_name)
            check_cuda(driver, result, f"cuModuleLoadData/cuModuleGetFunction for {self.kernel_name}")
            progress.finish("module load", phase_started, f"arch={arch}")
            phase_started = progress.start("configure function")
            block_dim = launch_block_dim(self.options)
            configure_function(
                driver,
                function,
                self.launch_package,
                self.topology,
                block_dim=block_dim,
            )
            progress.finish("configure function", phase_started)

            phase_started = progress.start("static runtime plan H2D")
            ptrs = {
                "instructions": device_alloc(driver, allocations, self.launch_package.queue.instructions_bytes),
                "queue_offsets": device_alloc(driver, allocations, self.launch_package.queue.offsets_bytes),
                "queue_lengths": device_alloc(driver, allocations, self.launch_package.queue.lengths_bytes),
                "slots": device_alloc(driver, allocations, self.launch_package.slots_bytes),
                "comms": device_alloc(driver, allocations, self.launch_package.comms_bytes),
                "barrier_init_offsets": device_alloc(
                    driver,
                    allocations,
                    self.launch_package.queue.barrier_init_offsets_bytes,
                ),
                "barrier_init_lengths": device_alloc(
                    driver,
                    allocations,
                    self.launch_package.queue.barrier_init_lengths_bytes,
                ),
                "barrier_init_indices": device_alloc(
                    driver,
                    allocations,
                    self.launch_package.queue.barrier_init_indices_bytes,
                ),
                "args": device_alloc(driver, allocations, self.launch_package.args_bytes),
                "tensor_args": device_alloc(driver, allocations, self.tensor_args_bytes),
                "input_slots": device_alloc(driver, allocations, self.launch_package.input_slots_bytes),
                "task_coords": device_alloc(driver, allocations, self.launch_package.task_coords_bytes),
                "global_staging": device_alloc(driver, allocations, self.launch_package.global_staging_bytes),
                "flags": device_alloc(driver, allocations, self.launch_package.flags_bytes),
            }
            progress.finish("static runtime plan H2D", phase_started)

            phase_started = progress.start("kernel launch")
            launch_wrapper(
                driver,
                function,
                self.launch_package,
                ptrs,
                block_dim,
                self.topology,
                stream,
                self.tma_descriptor_specs,
                self.tma_descriptor_handles,
                synchronize=True,
            )
            progress.finish("kernel launch", phase_started)
            phase_started = progress.start("global staging D2H")
            global_staging_bytes = device_to_host(
                driver,
                ptrs["global_staging"],
                len(self.launch_package.global_staging_bytes),
            )
            progress.finish("global staging D2H", phase_started, f"bytes={len(global_staging_bytes)}")
            progress.finish("launch", launch_started, f"kernel={self.kernel_name}")
            return DataflowExecutionResult(
                kernel_name=self.kernel_name,
                arch=arch,
                grid_dim=(self.launch_package.queue.queue_count, 1, 1),
                block_dim=block_dim,
                shared_memory_bytes=self.launch_package.shared_memory_bytes,
                instruction_count=self.launch_package.queue.instruction_count,
                global_staging_bytes=global_staging_bytes,
                cluster_dim=(self.topology.cluster_size, 1, 1),
                tensor_arg_count=len(self.tensor_args_bytes) // TENSOR_ARG_STRUCT.size,
                target_fingerprint=target_capabilities.fingerprint,
                artifact_fingerprint=self.artifact_fingerprint or "",
            )
        finally:
            cleanup_started = progress.start("cleanup")
            for ptr in reversed(allocations):
                driver.cuMemFree(ptr)
            if module is not None:
                driver.cuModuleUnload(module)
            driver.cuDevicePrimaryCtxRelease(device)
            progress.finish("cleanup", cleanup_started)

    def compile_cubin(self) -> tuple[bytes, str]:
        progress = progress_logger(self.options, "[dataflow.launch]")
        if self._cubin is not None and self._arch is not None:
            progress.message(f"wrapper cubin compile skipped; cached arch={self._arch}")
            return self._cubin, self._arch

        target_capabilities = require_target_capabilities(self.target_capabilities)
        arch = target_capabilities.arch
        phase_started = progress.start("wrapper cubin compile")
        try:
            cubin = bytes(
                nvcc.compile_cuda(
                    self.source,
                    target_format="cubin",
                    arch=arch,
                    options=nvcc.default_compile_options(self.options.get("compile_flags")),
                    verbose=bool(self.options.get("verbose", False)),
                )
            )
        except RuntimeError as err:
            raise TargetCapabilityResolutionError(
                "NVCC failed to compile the Dataflow wrapper for the resolved target; "
                f"arch={arch}, target_fingerprint={target_capabilities.fingerprint}, "
                f"compiler_version={target_capabilities.compiler_version}: {err}"
            ) from err
        self._cubin = cubin
        self._arch = arch
        progress.finish("wrapper cubin compile", phase_started, f"arch={arch} bytes={len(cubin)}")
        return cubin, arch


@dataclass
class DataflowPersistentExecutable:
    kernel_name: str
    source: str
    launch_package: DataflowLaunchPackage
    topology: GPUTopology
    tensor_args_bytes: bytes = b""
    tma_descriptor_specs: tuple[DataflowTMADescriptorSpec, ...] = ()
    tma_descriptor_handles: tuple[DataflowTMADescriptorHandle, ...] = ()
    tma_tensor_indices: dict[str, int] = field(default_factory=dict)
    tma_tensor_metadata: dict[str, DataflowRuntimeTensorMetadata] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    target_capabilities: TargetCapabilitySnapshot | None = None
    artifact_fingerprint: str | None = None
    setup_timings_ms: dict[str, float] = field(default_factory=dict, init=False)
    _driver: Any | None = field(default=None, init=False, repr=False)
    _device: Any | None = field(default=None, init=False, repr=False)
    _module: Any | None = field(default=None, init=False, repr=False)
    _function: Any | None = field(default=None, init=False, repr=False)
    _ptrs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _allocations: list[Any] = field(default_factory=list, init=False, repr=False)
    _arch: str | None = field(default=None, init=False, repr=False)
    _tensor_args_capacity: int = field(default=0, init=False, repr=False)
    _tensor_args_uploaded_bytes: bytes | None = field(default=None, init=False, repr=False)
    _tma_tensor_data_ptrs: dict[str, int] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __enter__(self) -> DataflowPersistentExecutable:
        return self.open()

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._driver is not None

    def open(self) -> DataflowPersistentExecutable:
        if self.is_open:
            return self

        progress = progress_logger(self.options, "[dataflow.persistent]")
        open_started = progress.start("open")
        timings: dict[str, float] = {}
        phase_started = progress.start("retain CUDA context")
        start = time.perf_counter()
        target_capabilities = require_target_capabilities(self.target_capabilities)
        driver, device = retain_cuda_context(target_capabilities.device_ordinal)
        timings["context_retain_ms"] = elapsed_ms(start)
        progress.finish("retain CUDA context", phase_started)
        self._driver = driver
        self._device = device

        try:
            phase_started = progress.start("validate target and cluster launch")
            actual_capabilities = validate_target_compatibility(
                target_capabilities,
                driver,
                device,
                artifact_fingerprint=self.artifact_fingerprint,
            )
            validate_cluster_launch(
                driver,
                device,
                self.topology,
                self.launch_package,
                target_capabilities=actual_capabilities,
            )
            progress.finish("validate target and cluster launch", phase_started)

            compile_start = time.perf_counter()
            cubin, arch = DataflowExecutableKernel(
                kernel_name=self.kernel_name,
                source=self.source,
                launch_package=self.launch_package,
                topology=self.topology,
                tensor_args_bytes=self.tensor_args_bytes,
                tma_descriptor_specs=self.tma_descriptor_specs,
                tma_descriptor_handles=self.tma_descriptor_handles,
                tma_tensor_indices=self.tma_tensor_indices,
                options=self.options,
                target_capabilities=target_capabilities,
                artifact_fingerprint=self.artifact_fingerprint,
            ).compile_cubin()
            timings["wrapper_cubin_compile_ms"] = elapsed_ms(compile_start)
            self._arch = arch

            phase_started = progress.start("module load")
            module_start = time.perf_counter()
            result, module, function = load_function(driver, cubin, self.kernel_name)
            check_cuda(driver, result, f"cuModuleLoadData/cuModuleGetFunction for {self.kernel_name}")
            self._module = module
            self._function = function
            configure_function(
                driver,
                function,
                self.launch_package,
                self.topology,
                block_dim=launch_block_dim(self.options),
            )
            timings["module_load_ms"] = elapsed_ms(module_start)
            progress.finish("module load", phase_started, f"arch={arch}")

            phase_started = progress.start("static runtime plan H2D")
            static_start = time.perf_counter()
            self._ptrs = {
                "instructions": device_alloc(driver, self._allocations, self.launch_package.queue.instructions_bytes),
                "queue_offsets": device_alloc(driver, self._allocations, self.launch_package.queue.offsets_bytes),
                "queue_lengths": device_alloc(driver, self._allocations, self.launch_package.queue.lengths_bytes),
                "slots": device_alloc(driver, self._allocations, self.launch_package.slots_bytes),
                "comms": device_alloc(driver, self._allocations, self.launch_package.comms_bytes),
                "barrier_init_offsets": device_alloc(
                    driver,
                    self._allocations,
                    self.launch_package.queue.barrier_init_offsets_bytes,
                ),
                "barrier_init_lengths": device_alloc(
                    driver,
                    self._allocations,
                    self.launch_package.queue.barrier_init_lengths_bytes,
                ),
                "barrier_init_indices": device_alloc(
                    driver,
                    self._allocations,
                    self.launch_package.queue.barrier_init_indices_bytes,
                ),
                "args": device_alloc(driver, self._allocations, self.launch_package.args_bytes),
                "input_slots": device_alloc(driver, self._allocations, self.launch_package.input_slots_bytes),
                "task_coords": device_alloc(driver, self._allocations, self.launch_package.task_coords_bytes),
                "global_staging": device_alloc_empty(
                    driver,
                    self._allocations,
                    len(self.launch_package.global_staging_bytes),
                ),
                "flags": device_alloc_empty(driver, self._allocations, len(self.launch_package.flags_bytes)),
            }
            timings["static_plan_h2d_ms"] = elapsed_ms(static_start)
            progress.finish("static runtime plan H2D", phase_started)

            phase_started = progress.start("tensor args H2D")
            tensor_start = time.perf_counter()
            self.allocate_or_update_tensor_args(self.tensor_args_bytes)
            timings["tensor_args_h2d_ms"] = elapsed_ms(tensor_start)
            progress.finish("tensor args H2D", phase_started)

            self.setup_timings_ms = timings
            progress.finish("open", open_started, f"kernel={self.kernel_name}")
            return self
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        driver = self._driver
        if driver is None:
            return
        try:
            for ptr in reversed(self._allocations):
                driver.cuMemFree(ptr)
            self._allocations.clear()
            self._ptrs.clear()
            if self._module is not None:
                driver.cuModuleUnload(self._module)
        finally:
            if self._device is not None:
                driver.cuDevicePrimaryCtxRelease(self._device)
            self._driver = None
            self._device = None
            self._module = None
            self._function = None
            self._tensor_args_capacity = 0
            self._tensor_args_uploaded_bytes = None

    def launch(
        self,
        *,
        tensor_args_bytes: bytes | None = None,
        stream: Any | None = None,
        copy_global_staging: bool = False,
    ) -> DataflowExecutionResult:
        profile = self.profile_launch(
            tensor_args_bytes=tensor_args_bytes,
            stream=stream,
            copy_global_staging=copy_global_staging,
            use_cuda_events=False,
        )
        return profile.execution

    def profile_launch(
        self,
        *,
        tensor_args_bytes: bytes | None = None,
        stream: Any | None = None,
        copy_global_staging: bool = False,
        use_cuda_events: bool = True,
        event_scope: str = "kernel",
    ) -> DataflowPersistentProfileResult:
        if not self.is_open:
            self.open()
        assert self._driver is not None
        assert self._function is not None
        if tensor_args_bytes is None:
            tensor_args_bytes = self.tensor_args_bytes
        if event_scope not in ("kernel", "launch"):
            raise ValueError(f"event_scope must be 'kernel' or 'launch', got {event_scope!r}")

        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        progress = progress_logger(self.options, "[dataflow.persistent]")
        launch_started = progress.start("launch")

        block_dim = launch_block_dim(self.options)

        def launch_body(*, profile_kernel_event: bool, synchronize_kernel: bool) -> None:
            phase_started = progress.start("tensor args H2D")
            tensor_start = time.perf_counter()
            self.allocate_or_update_tensor_args(tensor_args_bytes)
            timings["tensor_args_h2d_ms"] = elapsed_ms(tensor_start)
            progress.finish("tensor args H2D", phase_started)

            phase_started = progress.start("flags reset")
            flags_start = time.perf_counter()
            self.reset_flags()
            timings["flags_reset_ms"] = elapsed_ms(flags_start)
            progress.finish("flags reset", phase_started)

            phase_started = progress.start("kernel launch")
            kernel_start = time.perf_counter()
            if profile_kernel_event:
                timings["pure_kernel_event_ms"] = profile_cuda_event_ms(
                    self._driver,
                    stream,
                    lambda: launch_wrapper(
                        self._driver,
                        self._function,
                        self.launch_package,
                        self._ptrs,
                        block_dim,
                        self.topology,
                        stream,
                        self.tma_descriptor_specs,
                        self.tma_descriptor_handles,
                        synchronize=False,
                    ),
                )
            else:
                launch_wrapper(
                    self._driver,
                    self._function,
                    self.launch_package,
                    self._ptrs,
                    block_dim,
                    self.topology,
                    stream,
                    self.tma_descriptor_specs,
                    self.tma_descriptor_handles,
                    synchronize=synchronize_kernel,
                )
            timings["kernel_host_ms"] = elapsed_ms(kernel_start)
            progress.finish("kernel launch", phase_started)

        if use_cuda_events and event_scope == "launch":
            timings["launch_event_ms"] = profile_cuda_event_ms(
                self._driver,
                stream,
                lambda: launch_body(profile_kernel_event=False, synchronize_kernel=False),
            )
        else:
            launch_body(
                profile_kernel_event=use_cuda_events and event_scope == "kernel",
                synchronize_kernel=not use_cuda_events,
            )

        global_staging_bytes = b""
        phase_started = progress.start("global staging D2H")
        d2h_start = time.perf_counter()
        if copy_global_staging:
            global_staging_bytes = device_to_host(
                self._driver,
                self._ptrs["global_staging"],
                len(self.launch_package.global_staging_bytes),
            )
        timings["global_staging_d2h_ms"] = elapsed_ms(d2h_start)
        timings["launch_total_host_ms"] = elapsed_ms(total_start)
        progress.finish("global staging D2H", phase_started, f"bytes={len(global_staging_bytes)}")
        progress.finish("launch", launch_started, f"kernel={self.kernel_name}")

        return DataflowPersistentProfileResult(
            execution=self.execution_result(global_staging_bytes),
            timings_ms=timings,
        )

    def execution_result(self, global_staging_bytes: bytes) -> DataflowExecutionResult:
        return DataflowExecutionResult(
            kernel_name=self.kernel_name,
            arch="" if self._arch is None else self._arch,
            grid_dim=(self.launch_package.queue.queue_count, 1, 1),
            block_dim=launch_block_dim(self.options),
            shared_memory_bytes=self.launch_package.shared_memory_bytes,
            instruction_count=self.launch_package.queue.instruction_count,
            global_staging_bytes=global_staging_bytes,
            cluster_dim=(self.topology.cluster_size, 1, 1),
            tensor_arg_count=len(self.tensor_args_bytes) // TENSOR_ARG_STRUCT.size,
            target_fingerprint=("" if self.target_capabilities is None else self.target_capabilities.fingerprint),
            artifact_fingerprint=self.artifact_fingerprint or "",
        )

    def allocate_or_update_tensor_args(self, tensor_args_bytes: bytes) -> None:
        assert self._driver is not None
        tensor_data_ptrs = tensor_data_ptrs_from_bytes(
            tensor_args_bytes,
            self.tma_tensor_indices,
        )
        if self.tma_descriptor_specs:
            if self._tma_tensor_data_ptrs is not None and tensor_data_ptrs != self._tma_tensor_data_ptrs:
                raise ValueError(
                    "persistent Dataflow TMA tensor pointers cannot be changed "
                    "through pointer-only tensor_args_bytes; create a new "
                    "executable from CUDA tensor-like arguments so shape, "
                    "stride, and device metadata can be revalidated"
                )
            if any(tensor_data_ptrs.values()) and not self.tma_descriptor_handles:
                expected_device_ordinal = None if self.target_capabilities is None else self.target_capabilities.device_ordinal
                validate_tma_descriptor_runtime_tensors(
                    self.tma_descriptor_specs,
                    tensor_data_ptrs=tensor_data_ptrs,
                    tensor_metadata=self.tma_tensor_metadata,
                    expected_device_ordinal=expected_device_ordinal,
                )
                self.tma_descriptor_handles = build_tma_descriptor_handles(
                    self.tma_descriptor_specs,
                    tensor_data_ptrs=tensor_data_ptrs,
                )

        byte_count = max(len(tensor_args_bytes), 1)
        if "tensor_args" not in self._ptrs or byte_count > self._tensor_args_capacity:
            if "tensor_args" in self._ptrs:
                self._driver.cuMemFree(self._ptrs["tensor_args"])
                self._allocations.remove(self._ptrs["tensor_args"])
            self._ptrs["tensor_args"] = device_alloc_empty(self._driver, self._allocations, byte_count)
            self._tensor_args_capacity = byte_count
            self._tensor_args_uploaded_bytes = None
        if tensor_args_bytes == self._tensor_args_uploaded_bytes:
            return
        if tensor_args_bytes:
            device_copy_to(self._driver, self._ptrs["tensor_args"], tensor_args_bytes)
        self.tensor_args_bytes = tensor_args_bytes
        self._tensor_args_uploaded_bytes = bytes(tensor_args_bytes)
        if self.tma_descriptor_specs and any(tensor_data_ptrs.values()):
            self._tma_tensor_data_ptrs = tensor_data_ptrs

    def reset_flags(self) -> None:
        assert self._driver is not None
        if not launch_package_needs_flag_reset(self.launch_package):
            return
        byte_count = len(self.launch_package.flags_bytes)
        if byte_count <= 0:
            return
        check_cuda(
            self._driver,
            self._driver.cuMemsetD8(self._ptrs["flags"], 0, byte_count)[0],
            "cuMemsetD8(flags)",
        )


def launch_package_needs_flag_reset(package: DataflowLaunchPackage) -> bool:
    hbm_send = 3
    hbm_recv = 4
    record_size = COMM_STRUCT.size
    for offset in range(0, len(package.comms_bytes), record_size):
        kind = COMM_STRUCT.unpack_from(package.comms_bytes, offset)[0]
        if kind in (hbm_send, hbm_recv):
            return True
    return False


def retain_cuda_context(device_ordinal: int | None = None):
    try:
        from cuda.bindings import driver
    except Exception as err:  # pragma: no cover - environment dependent
        raise RuntimeError(f"CUDA driver bindings are required for executable Dataflow launch: {err}") from err

    result = driver.cuInit(0)[0]
    check_cuda(driver, result, "cuInit")
    result, count = driver.cuDeviceGetCount()
    check_cuda(driver, result, "cuDeviceGetCount")
    if count == 0:
        raise RuntimeError("CUDA driver reports no executable devices")

    selected_ordinal = 0 if device_ordinal is None else int(device_ordinal)
    if selected_ordinal < 0 or selected_ordinal >= count:
        raise RuntimeError(f"CUDA device ordinal {selected_ordinal} is outside [0, {count})")
    result, device = driver.cuDeviceGet(selected_ordinal)
    check_cuda(driver, result, "cuDeviceGet")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    check_cuda(driver, result, "cuDevicePrimaryCtxRetain")
    result = driver.cuCtxSetCurrent(context)[0]
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, result, "cuCtxSetCurrent")
    return driver, device


def validate_cluster_launch(
    driver,
    device: Any,
    topology: GPUTopology,
    package: DataflowLaunchPackage,
    *,
    target_capabilities: TargetCapabilitySnapshot | None = None,
) -> None:
    cluster_size = topology.cluster_size
    if cluster_size <= 0:
        raise ValueError(f"Dataflow executable cluster_size must be positive, got {cluster_size}")
    if package.queue.queue_count % cluster_size != 0:
        raise ValueError(
            "Dataflow executable queue_count must be divisible by cluster_size for cluster launch: "
            f"queue_count={package.queue.queue_count}, cluster_size={cluster_size}"
        )
    if (
        target_capabilities is not None
        and target_capabilities.max_dynamic_shared_memory is not None
        and package.shared_memory_bytes > target_capabilities.max_dynamic_shared_memory
    ):
        raise DataflowTargetCompatibilityError(
            "Dataflow artifact exceeds the selected CUDA device dynamic shared-memory limit: "
            f"requested={package.shared_memory_bytes}, "
            f"limit={target_capabilities.max_dynamic_shared_memory}, "
            f"target_fingerprint={target_capabilities.fingerprint}"
        )
    if cluster_size == 1:
        return

    if target_capabilities is None:
        cluster_launch_attr = getattr(
            driver.CUdevice_attribute,
            "CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH",
            None,
        )
        if cluster_launch_attr is None:
            raise RuntimeError("CUDA driver bindings do not expose CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH")
        result, supported = driver.cuDeviceGetAttribute(cluster_launch_attr, device)
        check_cuda(driver, result, "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH)")
    else:
        supported = target_capabilities.supports_cluster_launch
    if not supported:
        raise RuntimeError("CUDA device does not support cluster launch required by Dataflow topology")


def require_target_capabilities(
    target_capabilities: TargetCapabilitySnapshot | None,
) -> TargetCapabilitySnapshot:
    if target_capabilities is None:
        raise TargetCapabilityResolutionError("Dataflow executable is missing the compile-time target capability snapshot")
    return target_capabilities


def validate_target_compatibility(
    artifact_target: TargetCapabilitySnapshot,
    driver: Any,
    device: Any,
    *,
    artifact_fingerprint: str | None,
) -> TargetCapabilitySnapshot:
    actual_target = resolve_target_capabilities_from_device(
        driver,
        device,
        device_ordinal=int(device),
        compiler_version=artifact_target.compiler_version,
    )
    mismatches = artifact_target.compatibility_mismatches(actual_target)
    if mismatches:
        detail = "; ".join(mismatches)
        raise DataflowTargetCompatibilityError(
            "Dataflow artifact target is incompatible with the selected CUDA device; "
            f"artifact_fingerprint={artifact_fingerprint or '<unknown>'}, "
            f"artifact_target_fingerprint={artifact_target.fingerprint}, "
            f"device_target_fingerprint={actual_target.fingerprint}: {detail}"
        )
    return actual_target


def device_alloc(driver, allocations: list[Any], data: bytes):
    byte_count = max(len(data), 1)
    ptr = device_alloc_empty(driver, allocations, byte_count)
    if data:
        device_copy_to(driver, ptr, data)
    return ptr


def device_alloc_empty(driver, allocations: list[Any], byte_count: int):
    byte_count = max(byte_count, 1)
    result, ptr = driver.cuMemAlloc(byte_count)
    check_cuda(driver, result, "cuMemAlloc")
    allocations.append(ptr)
    return ptr


def device_copy_to(driver, ptr: Any, data: bytes) -> None:
    if not data:
        return
    host = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    check_cuda(driver, driver.cuMemcpyHtoD(ptr, host, len(data))[0], "cuMemcpyHtoD")


def device_to_host(driver, ptr: Any, byte_count: int) -> bytes:
    if byte_count <= 0:
        return b""
    host = (ctypes.c_ubyte * byte_count)()
    check_cuda(driver, driver.cuMemcpyDtoH(host, ptr, byte_count)[0], "cuMemcpyDtoH")
    return bytes(host)


def load_function(driver, cubin: bytes, kernel_name: str):
    image = (ctypes.c_ubyte * len(cubin)).from_buffer_copy(cubin)
    result, module = driver.cuModuleLoadData(image)
    if result != driver.CUresult.CUDA_SUCCESS:
        return result, None, None
    result, function = driver.cuModuleGetFunction(module, kernel_name.encode())
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuModuleUnload(module)
        return result, None, None
    return result, module, function


def configure_function(
    driver,
    function: Any,
    package: DataflowLaunchPackage,
    topology: GPUTopology,
    *,
    block_dim: tuple[int, int, int],
) -> None:
    if package.shared_memory_bytes:
        check_cuda(
            driver,
            driver.cuFuncSetAttribute(
                function,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                package.shared_memory_bytes,
            )[0],
            "cuFuncSetAttribute",
        )
    if topology.cluster_size > 1:
        check_cuda(
            driver,
            driver.cuFuncSetAttribute(
                function,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            )[0],
            "cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED)",
        )
        config = driver.CUlaunchConfig()
        config.gridDimX = package.queue.queue_count
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = block_dim[0]
        config.blockDimY = block_dim[1]
        config.blockDimZ = block_dim[2]
        config.sharedMemBytes = package.shared_memory_bytes
        config.hStream = driver.CUstream(0)
        query_max_cluster_size = getattr(
            driver,
            "cuOccupancyMaxPotentialClusterSize",
            None,
        )
        if query_max_cluster_size is None:
            raise DataflowTargetCompatibilityError("CUDA driver bindings do not expose kernel-specific cluster limit validation")
        result, max_cluster_size = query_max_cluster_size(
            function,
            config,
        )
        check_cuda(driver, result, "cuOccupancyMaxPotentialClusterSize")
        if topology.cluster_size > int(max_cluster_size):
            raise DataflowTargetCompatibilityError(
                "Dataflow topology exceeds the kernel-specific CUDA cluster limit: "
                f"cluster_size={topology.cluster_size}, "
                f"max_cluster_size={int(max_cluster_size)}"
            )


def launch_block_dim(options: dict[str, Any]) -> tuple[int, int, int]:
    block_dim = options.get("block_dim", (32, 1, 1))
    if isinstance(block_dim, int):
        block_dim = (block_dim, 1, 1)
    block_dim = tuple(int(item) for item in block_dim)
    if len(block_dim) != 3 or any(item <= 0 for item in block_dim):
        raise ValueError(f"Dataflow executable block_dim must be a positive int or 3-tuple, got {block_dim!r}")
    return block_dim


def launch_wrapper(
    driver,
    function,
    package: DataflowLaunchPackage,
    ptrs: dict[str, Any],
    block_dim: tuple[int, int, int],
    topology: GPUTopology,
    stream: Any | None,
    tma_descriptor_specs: tuple[DataflowTMADescriptorSpec, ...],
    tma_descriptor_handles: tuple[DataflowTMADescriptorHandle, ...],
    *,
    synchronize: bool,
) -> None:
    config = driver.CUlaunchConfig()
    config.gridDimX = package.queue.queue_count
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = block_dim[0]
    config.blockDimY = block_dim[1]
    config.blockDimZ = block_dim[2]
    config.sharedMemBytes = package.shared_memory_bytes
    config.hStream = driver.CUstream(0) if stream is None else stream
    attrs = launch_attrs(
        driver,
        topology.cluster_size,
        package.cluster_scheduling_policy,
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
    ]
    arg_types = [ctypes.c_void_p] * len(arg_values)
    if tma_descriptor_specs:
        if len(tma_descriptor_handles) != len(tma_descriptor_specs):
            raise RuntimeError(
                f"Dataflow wrapper expects TMA descriptors, but initialized {len(tma_descriptor_handles)} of {len(tma_descriptor_specs)}"
            )
        expected_names = [handle.spec.name for handle in tma_descriptor_handles]
        spec_names = [spec.name for spec in tma_descriptor_specs]
        if expected_names != spec_names:
            raise RuntimeError(f"Dataflow TMA descriptor launch order mismatch: expected {spec_names!r}, got {expected_names!r}")
        if len(expected_names) != len(set(expected_names)):
            raise ValueError(f"duplicate Dataflow TMA descriptor names in launch args: {expected_names!r}")
        arg_values.extend(handle.handle for handle in tma_descriptor_handles)
        arg_types.extend([None] * len(tma_descriptor_handles))
    kernel_params = (tuple(arg_values), tuple(arg_types))
    check_cuda(driver, driver.cuLaunchKernelEx(config, function, kernel_params, 0)[0], "cuLaunchKernelEx")
    if synchronize:
        check_cuda(driver, driver.cuCtxSynchronize()[0], "cuCtxSynchronize")


def launch_attrs(
    driver,
    cluster_size: int,
    cluster_scheduling_policy: str = (DATAFLOW_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING),
) -> list[Any]:
    if cluster_scheduling_policy != DATAFLOW_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING:
        raise ValueError(f"unsupported Dataflow cluster scheduling policy {cluster_scheduling_policy!r}")
    if cluster_size == 1:
        return []
    cluster_dim = driver.CUlaunchAttribute()
    cluster_dim.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    cluster_dim.value.clusterDim.x = cluster_size
    cluster_dim.value.clusterDim.y = 1
    cluster_dim.value.clusterDim.z = 1
    scheduling = driver.CUlaunchAttribute()
    scheduling.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
    scheduling.value.clusterSchedulingPolicyPreference = driver.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING
    return [cluster_dim, scheduling]


def tensor_data_ptrs_from_bytes(
    tensor_args_bytes: bytes,
    tensor_indices: dict[str, int],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, index in tensor_indices.items():
        offset = int(index) * TENSOR_ARG_STRUCT.size
        if offset + TENSOR_ARG_STRUCT.size > len(tensor_args_bytes):
            result[name] = 0
            continue
        data_ptr, *_ = TENSOR_ARG_STRUCT.unpack_from(tensor_args_bytes, offset)
        result[name] = int(data_ptr)
    return result


def check_cuda(driver, result: Any, action: str) -> None:
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{action} failed: {result}")


def profile_cuda_event_ms(driver, stream: Any | None, launch: Any) -> float:
    launch_stream = driver.CUstream(0) if stream is None else stream
    result, start = driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DEFAULT)
    check_cuda(driver, result, "cuEventCreate(start)")
    result, end = driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DEFAULT)
    check_cuda(driver, result, "cuEventCreate(end)")
    try:
        check_cuda(driver, driver.cuEventRecord(start, launch_stream)[0], "cuEventRecord(start)")
        launch()
        check_cuda(driver, driver.cuEventRecord(end, launch_stream)[0], "cuEventRecord(end)")
        check_cuda(driver, driver.cuEventSynchronize(end)[0], "cuEventSynchronize(end)")
        result, elapsed_ms = driver.cuEventElapsedTime(start, end)
        check_cuda(driver, result, "cuEventElapsedTime")
        return float(elapsed_ms)
    finally:
        driver.cuEventDestroy(start)
        driver.cuEventDestroy(end)


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def progress_logger(options: dict[str, Any], prefix: str) -> DataflowProgressLogger:
    return DataflowProgressLogger(dataflow_progress_enabled(options), prefix=prefix)
