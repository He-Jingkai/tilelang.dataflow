"""Structured codegen metadata for Dataflow PrimFunc handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .dtype_registry import require_dataflow_dtype
from .tma_descriptors import DataflowTMADescriptorSpec


DATAFLOW_HANDLER_PARAM_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DataflowHandlerCodegenModule:
    """A specialized handler IRModule that composes with one raw wrapper kernel."""

    module: Any = field(repr=False, compare=False)
    target: Any = field(repr=False, compare=False)
    handler_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.module is None or self.target is None:
            raise ValueError("handler codegen module requires module and target")
        if not self.handler_symbols or len(self.handler_symbols) != len(set(self.handler_symbols)):
            raise ValueError(f"handler codegen module requires unique handler symbols, got {self.handler_symbols!r}")

    def compose(self, wrapper_source: str, *, kernel_name: str) -> str:
        """Codegen handlers and a source-kernel wrapper as one CUDA IRModule."""

        import tvm
        from tvm import tir
        from tvm.ir import CallingConv

        from tilelang.engine.lower import device_codegen_without_compile

        if not wrapper_source or not kernel_name:
            raise ValueError("handler module composition requires wrapper source and kernel name")
        if any(str(global_var) == kernel_name for global_var in self.module.functions):
            raise ValueError(f"wrapper kernel symbol {kernel_name!r} collides with a handler")
        wrapper_func = tir.PrimFunc([], tir.Evaluate(0))
        for attr_name, attr_value in (
            ("global_symbol", kernel_name),
            ("calling_conv", CallingConv.DEVICE_KERNEL_LAUNCH),
            ("target", self.target),
            ("code_block_source", wrapper_source),
            ("code_block_entry_name", kernel_name),
        ):
            wrapper_func = wrapper_func.with_attr(attr_name, attr_value)
        composed = tvm.IRModule(self.module.functions, attrs=self.module.attrs)
        composed.update(tvm.IRModule({kernel_name: wrapper_func}))
        with tvm.transform.PassContext(opt_level=3), self.target:
            codegen_mod = device_codegen_without_compile(composed, self.target)
        return codegen_mod.inspect_source()

    def to_dict(self) -> dict[str, Any]:
        return {
            "handler_symbols": list(self.handler_symbols),
            "target": str(self.target),
        }


def dataflow_buffer_cuda_c_type(dtype: str) -> str:
    """Return the canonical CUDA element type recorded in handler metadata."""

    info = require_dataflow_dtype(dtype)
    if not info.primfunc_intermediate_supported:
        raise NotImplementedError(f"Dataflow handler metadata does not support tensor dtype {dtype!r}")
    return info.cuda_type


def dataflow_scalar_cuda_c_type(dtype: str) -> str:
    """Return the canonical CUDA scalar type recorded in handler metadata."""

    info = require_dataflow_dtype(dtype)
    if info.scalar_cuda_type is None:
        raise NotImplementedError(f"Dataflow handler metadata does not support scalar dtype {dtype!r}")
    return info.scalar_cuda_type


class DataflowHandlerParamRole(str, Enum):
    """Semantic role of a parameter in a lowered handler ABI."""

    TENSOR_ARG = "tensor_arg"
    TASK_COORD = "task_coord"
    RANGE_BEGIN = "range_begin"
    RANGE_END = "range_end"
    INPUT_SLOT_FIELD = "input_slot_field"
    OUTPUT_SLOT_FIELD = "output_slot_field"
    INPUT_COUNT = "input_count"
    TASK_ID = "task_id"
    TMA_DESCRIPTOR = "tma_descriptor"
    NEXT_TASK_COORD = "next_task_coord"
    HANDOFF_STAGE_COUNT = "handoff_stage_count"
    HANDOFF_ARENA_SLOT = "handoff_arena_slot"
    HANDOFF_PEER_RANGE_BEGIN = "handoff_peer_range_begin"
    HANDOFF_PEER_RANGE_END = "handoff_peer_range_end"
    HANDOFF_PEER_TASK_ID = "handoff_peer_task_id"
    HANDOFF_TRANSFER_BUFFER = "handoff_transfer_buffer"


@dataclass(frozen=True)
class DataflowHandlerParam:
    """One ordered, semantically classified handler parameter."""

    ordinal: int
    name: str
    role: DataflowHandlerParamRole
    dtype: str
    c_type: str
    is_pointer: bool = False
    is_const: bool = False
    tensor_arg_index: int | None = None
    task_coord_axis: int | None = None
    slot_index: int | None = None
    slot_extent: int = 1
    field_name: str | None = None
    descriptor_name: str | None = None
    handoff_transfer_index: int | None = None
    handoff_stage_index: int | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError(f"handler parameter ordinal must be non-negative, got {self.ordinal}")
        if not self.name:
            raise ValueError("handler parameter name must be non-empty")
        if not isinstance(self.role, DataflowHandlerParamRole):
            raise TypeError(f"handler parameter role must be DataflowHandlerParamRole, got {self.role!r}")
        if not self.dtype or not self.c_type:
            raise ValueError(f"handler parameter {self.name!r} requires non-empty dtype and c_type")
        if self.slot_extent <= 0:
            raise ValueError(f"handler parameter {self.name!r} slot_extent must be positive, got {self.slot_extent}")
        if self.role is DataflowHandlerParamRole.TENSOR_ARG:
            if self.tensor_arg_index is None or self.tensor_arg_index < 0:
                raise ValueError(f"tensor parameter {self.name!r} requires a non-negative tensor_arg_index")
            if not self.is_pointer:
                raise ValueError(f"tensor parameter {self.name!r} must be a pointer")
        elif self.tensor_arg_index is not None:
            raise ValueError(f"non-tensor parameter {self.name!r} cannot define tensor_arg_index")
        if self.role in {
            DataflowHandlerParamRole.TASK_COORD,
            DataflowHandlerParamRole.NEXT_TASK_COORD,
        }:
            if self.task_coord_axis is None or self.task_coord_axis < 0:
                raise ValueError(f"task-coordinate parameter {self.name!r} requires a non-negative axis")
        elif self.task_coord_axis is not None:
            raise ValueError(f"parameter {self.name!r} with role {self.role.value!r} cannot define a task axis")
        if self.role in {
            DataflowHandlerParamRole.INPUT_SLOT_FIELD,
            DataflowHandlerParamRole.OUTPUT_SLOT_FIELD,
        }:
            if not self.is_pointer or not self.field_name:
                raise ValueError(f"slot-field parameter {self.name!r} requires pointer and field metadata")
            if self.slot_index is not None and self.slot_index < 0:
                raise ValueError(f"slot-field parameter {self.name!r} has negative slot_index")
        elif self.slot_index is not None or self.field_name is not None:
            raise ValueError(f"parameter {self.name!r} with role {self.role.value!r} cannot define slot metadata")
        if self.role is DataflowHandlerParamRole.TMA_DESCRIPTOR:
            if self.descriptor_name != self.name:
                raise ValueError(f"TMA parameter {self.name!r} must use its descriptor name as metadata")
        elif self.descriptor_name is not None:
            raise ValueError(f"non-TMA parameter {self.name!r} cannot define descriptor_name")
        if self.role is DataflowHandlerParamRole.HANDOFF_TRANSFER_BUFFER:
            if not self.is_pointer:
                raise ValueError(f"handoff transfer parameter {self.name!r} must be a pointer")
            if self.handoff_transfer_index is None or self.handoff_transfer_index < 0:
                raise ValueError(f"handoff transfer parameter {self.name!r} requires a non-negative transfer index")
            if self.handoff_stage_index is not None and self.handoff_stage_index < 0:
                raise ValueError(f"handoff transfer parameter {self.name!r} has a negative stage index")
        elif self.handoff_transfer_index is not None or self.handoff_stage_index is not None:
            raise ValueError(f"parameter {self.name!r} with role {self.role.value!r} cannot define handoff transfer metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "role": self.role.value,
            "dtype": self.dtype,
            "c_type": self.c_type,
            "is_pointer": self.is_pointer,
            "is_const": self.is_const,
            "tensor_arg_index": self.tensor_arg_index,
            "task_coord_axis": self.task_coord_axis,
            "slot_index": self.slot_index,
            "slot_extent": self.slot_extent,
            "field_name": self.field_name,
            "descriptor_name": self.descriptor_name,
            "handoff_transfer_index": self.handoff_transfer_index,
            "handoff_stage_index": self.handoff_stage_index,
        }


@dataclass(frozen=True)
class DataflowHandlerCodegenArtifact:
    """Stable metadata emitted with one lowered CUDA handler."""

    handler_id: int
    global_symbol: str
    device_symbol: str
    params: tuple[DataflowHandlerParam, ...]
    thread_count: int
    dynamic_shared_bytes: int
    tma_descriptors: tuple[DataflowTMADescriptorSpec, ...]
    module: Any = field(repr=False, compare=False)
    target_fingerprint: str | None = None
    param_schema_version: int = DATAFLOW_HANDLER_PARAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.handler_id < 0:
            raise ValueError(f"handler_id must be non-negative, got {self.handler_id}")
        if not self.global_symbol or not self.device_symbol:
            raise ValueError("handler codegen artifact requires global and device symbols")
        if self.thread_count <= 0:
            raise ValueError(f"handler {self.handler_id} thread_count must be positive, got {self.thread_count}")
        if self.dynamic_shared_bytes < 0:
            raise ValueError(f"handler dynamic_shared_bytes must be non-negative, got {self.dynamic_shared_bytes}")
        if self.param_schema_version != DATAFLOW_HANDLER_PARAM_SCHEMA_VERSION:
            raise ValueError(f"unsupported Dataflow handler parameter schema version {self.param_schema_version}")
        ordinals = tuple(param.ordinal for param in self.params)
        if ordinals != tuple(range(len(self.params))):
            raise ValueError(f"handler {self.handler_id} parameter ordinals must be contiguous, got {ordinals!r}")
        names = tuple(param.name for param in self.params)
        if len(names) != len(set(names)):
            raise ValueError(f"handler {self.handler_id} has duplicate parameter names: {names!r}")
        descriptor_names = tuple(spec.name for spec in self.tma_descriptors)
        parameter_descriptor_names = tuple(
            param.descriptor_name for param in self.params if param.role is DataflowHandlerParamRole.TMA_DESCRIPTOR
        )
        if parameter_descriptor_names != descriptor_names:
            raise ValueError(
                f"handler {self.handler_id} TMA parameters {parameter_descriptor_names!r} do not match descriptors {descriptor_names!r}"
            )
        if self.module is None:
            raise ValueError(f"handler {self.handler_id} codegen artifact requires a lowered module")

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(param.name for param in self.params)

    def params_for_role(self, role: DataflowHandlerParamRole) -> tuple[DataflowHandlerParam, ...]:
        return tuple(param for param in self.params if param.role is role)

    def param(self, name: str) -> DataflowHandlerParam:
        for param in self.params:
            if param.name == name:
                return param
        raise KeyError(f"handler {self.handler_id} has no parameter {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "handler_id": self.handler_id,
            "global_symbol": self.global_symbol,
            "device_symbol": self.device_symbol,
            "param_schema_version": self.param_schema_version,
            "params": [param.to_dict() for param in self.params],
            "thread_count": self.thread_count,
            "dynamic_shared_bytes": self.dynamic_shared_bytes,
            "tma_descriptors": [descriptor.to_dict() for descriptor in self.tma_descriptors],
            "target_fingerprint": self.target_fingerprint,
        }
