"""Tensor argument ABI planning and launch-time packing for Dataflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from tvm import tir

from .abi_schema import (
    DATAFLOW_TENSOR_ARG_FLAG_CUDA_ARRAY_INTERFACE,
    DATAFLOW_TENSOR_ARG_FLAG_RAW_POINTER,
    DATAFLOW_TENSOR_ARG_FLAG_TORCH,
    DATAFLOW_TENSOR_DTYPE_BFLOAT as DATAFLOW_TENSOR_DTYPE_BFLOAT,
    DATAFLOW_TENSOR_DTYPE_BOOL as DATAFLOW_TENSOR_DTYPE_BOOL,
    DATAFLOW_TENSOR_DTYPE_FLOAT as DATAFLOW_TENSOR_DTYPE_FLOAT,
    DATAFLOW_TENSOR_DTYPE_INT as DATAFLOW_TENSOR_DTYPE_INT,
    DATAFLOW_TENSOR_DTYPE_UINT as DATAFLOW_TENSOR_DTYPE_UINT,
    DATAFLOW_TENSOR_DTYPE_UNKNOWN,
    TENSOR_ARG_STRUCT,
)
from .dtype_registry import (
    tensor_dtype_metadata_from_name,
    tensor_dtype_metadata_from_typestr,
)
from .handler_identity import DataflowHandlerIdentity, build_handler_registry
from .ir import OperatorCall
from .program import DataflowProgram, DataflowStage, DataflowStageKind
from .scheduler import DataflowOpcode
from .tensor_layout import (
    DATAFLOW_TENSOR_LAYOUT_FINGERPRINT_ATTR,
    DataflowTensorArgumentLayout,
    tensor_argument_layout,
)


@dataclass(frozen=True)
class DataflowTensorArgSpec:
    index: int
    name: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    layout: DataflowTensorArgumentLayout | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "shape": None if self.shape is None else list(self.shape),
            "dtype": self.dtype,
            "layout": None if self.layout is None else self.layout.to_dict(),
        }


@dataclass(frozen=True)
class DataflowTensorArgBinding:
    tensor_index: int
    tensor_name: str
    operator_name: str
    operator_kind: str
    parameter_name: str
    handler_identity: DataflowHandlerIdentity

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_index": self.tensor_index,
            "tensor_name": self.tensor_name,
            "operator_name": self.operator_name,
            "operator_kind": self.operator_kind,
            "parameter_name": self.parameter_name,
            "identity": self.handler_identity.to_dict(),
        }


@dataclass(frozen=True)
class DataflowTensorArgPlan:
    specs: tuple[DataflowTensorArgSpec, ...]
    bindings: tuple[DataflowTensorArgBinding, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.specs),
            "names": list(self.names),
            "specs": [spec.to_dict() for spec in self.specs],
            "bindings": [binding.to_dict() for binding in self.bindings],
        }


@dataclass(frozen=True)
class PackedTensorArg:
    data_ptr: int
    ndim: int = 0
    dtype_code: int = DATAFLOW_TENSOR_DTYPE_UNKNOWN
    dtype_bits: int = 0
    flags: int = 0
    reserved: int = 0

    def to_tuple(self) -> tuple[int, ...]:
        return (
            self.data_ptr,
            self.ndim,
            self.dtype_code,
            self.dtype_bits,
            self.flags,
            self.reserved,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "data_ptr": self.data_ptr,
            "ndim": self.ndim,
            "dtype_code": self.dtype_code,
            "dtype_bits": self.dtype_bits,
            "flags": self.flags,
            "reserved": self.reserved,
        }


@dataclass(frozen=True)
class DataflowRuntimeTensorMetadata:
    source: str
    shape: tuple[int, ...] | None = None
    strides_bytes: tuple[int, ...] | None = None
    device_type: str | None = None
    device_index: int | None = None
    dtype_code: int = DATAFLOW_TENSOR_DTYPE_UNKNOWN
    dtype_bits: int = 0
    layout_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "shape": None if self.shape is None else list(self.shape),
            "strides_bytes": (None if self.strides_bytes is None else list(self.strides_bytes)),
            "device_type": self.device_type,
            "device_index": self.device_index,
            "dtype_code": self.dtype_code,
            "dtype_bits": self.dtype_bits,
            "layout_fingerprint": self.layout_fingerprint,
        }


@dataclass(frozen=True)
class PackedRuntimeTensorArgs:
    plan: DataflowTensorArgPlan
    records: tuple[PackedTensorArg, ...]
    metadata: tuple[DataflowRuntimeTensorMetadata, ...]
    tensor_args_bytes: bytes

    @property
    def record_size(self) -> int:
        return TENSOR_ARG_STRUCT.size

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "record_size": self.record_size,
            "records": [record.to_dict() for record in self.records],
            "metadata": [item.to_dict() for item in self.metadata],
            "bytes": len(self.tensor_args_bytes),
        }


def collect_tensor_arg_plan(program: DataflowProgram) -> DataflowTensorArgPlan:
    if not isinstance(program, DataflowProgram):
        raise TypeError(f"collect_tensor_arg_plan expects DataflowProgram, got {program!r}")
    program.validate()

    specs: list[DataflowTensorArgSpec] = []
    bindings: list[DataflowTensorArgBinding] = []
    indices: dict[str, int] = {}
    handler_registry = build_handler_registry(program)

    def add_binding(
        call: OperatorCall,
        operator_kind: str,
        handler_identity: DataflowHandlerIdentity,
    ) -> None:
        for parameter_name, value in call.bound_arguments.items():
            if not call.operator.is_external_tensor_parameter(parameter_name):
                continue
            tensor_name = resolve_tensor_name(value, call, parameter_name)
            shape, dtype, layout = tensor_parameter_contract(
                call,
                parameter_name,
            )
            index = indices.get(tensor_name)
            if index is None:
                index = len(specs)
                indices[tensor_name] = index
                specs.append(
                    DataflowTensorArgSpec(
                        index=index,
                        name=tensor_name,
                        shape=shape,
                        dtype=dtype,
                        layout=layout,
                    )
                )
            else:
                specs[index] = merge_tensor_arg_spec(
                    specs[index],
                    shape=shape,
                    dtype=dtype,
                    layout=layout,
                    operator_name=call.name,
                    parameter_name=parameter_name,
                )
            bindings.append(
                DataflowTensorArgBinding(
                    tensor_index=index,
                    tensor_name=tensor_name,
                    operator_name=call.name,
                    operator_kind=operator_kind,
                    parameter_name=parameter_name,
                    handler_identity=handler_identity,
                )
            )

    for stage in program.stages:
        operator_kind = tensor_operator_kind(stage)
        if operator_kind is None:
            continue
        assert stage.call is not None
        add_binding(
            stage.call,
            operator_kind,
            handler_registry.identity_for_stage_id(stage.stage_id),
        )
    if program.finalize_stage is not None and program.finalize_stage.fused_reduce_calls:
        for fused_reduce_call in program.finalize_stage.fused_reduce_calls:
            add_binding(
                fused_reduce_call,
                DataflowOpcode.FINALIZE.value,
                handler_registry.identity_for_call(
                    fused_reduce_call,
                    DataflowOpcode.FINALIZE.value,
                ),
            )
    return DataflowTensorArgPlan(specs=tuple(specs), bindings=tuple(bindings))


def tensor_operator_kind(stage: DataflowStage) -> str | None:
    if stage.call is None or stage.kind == DataflowStageKind.RESHARED:
        return None
    if stage.kind == DataflowStageKind.MAP:
        return stage.call.kind.value
    if stage.kind == DataflowStageKind.REDUCE:
        return DataflowOpcode.REDUCE.value
    if stage.kind == DataflowStageKind.FINALIZE:
        return DataflowOpcode.FINALIZE.value
    raise ValueError(f"unsupported Dataflow stage kind {stage.kind!r}")


def pack_tensor_args(
    plan: DataflowTensorArgPlan,
    *args: Any,
    allow_missing: bool = False,
    **kwargs: Any,
) -> PackedRuntimeTensorArgs:
    if not isinstance(plan, DataflowTensorArgPlan):
        raise TypeError(f"pack_tensor_args expects DataflowTensorArgPlan, got {plan!r}")
    if allow_missing and (args or kwargs):
        raise ValueError("allow_missing is only valid when no tensor arguments are provided")

    names = plan.names
    if allow_missing:
        records = tuple(PackedTensorArg(data_ptr=0) for _ in names)
        metadata = tuple(DataflowRuntimeTensorMetadata(source="missing") for _ in names)
        return pack_records(plan, records, metadata)

    if len(args) > len(names):
        raise TypeError(f"Dataflow expected at most {len(names)} tensor arguments, got {len(args)}")

    values: dict[str, Any] = {}
    for name, value in zip(names, args):
        values[name] = value

    unknown = sorted(set(kwargs) - set(names))
    if unknown:
        raise TypeError(f"Unknown Dataflow tensor argument(s): {', '.join(unknown)}")

    duplicate = sorted(set(kwargs) & set(values))
    if duplicate:
        raise TypeError(f"Duplicate Dataflow tensor argument(s): {', '.join(duplicate)}")
    values.update(kwargs)

    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"Missing Dataflow tensor argument(s): {', '.join(missing)}")

    packed_values = tuple(validated_tensor_value(spec, values[spec.name]) for spec in plan.specs)
    records = tuple(item[0] for item in packed_values)
    metadata = tuple(item[1] for item in packed_values)
    return pack_records(plan, records, metadata)


def pack_records(
    plan: DataflowTensorArgPlan,
    records: tuple[PackedTensorArg, ...],
    metadata: tuple[DataflowRuntimeTensorMetadata, ...],
) -> PackedRuntimeTensorArgs:
    if len(records) != len(metadata):
        raise ValueError("Dataflow tensor records and host metadata must align")
    data = b"".join(TENSOR_ARG_STRUCT.pack(*record.to_tuple()) for record in records)
    return PackedRuntimeTensorArgs(
        plan=plan,
        records=records,
        metadata=metadata,
        tensor_args_bytes=data,
    )


def tensor_parameter_contract(
    call: OperatorCall,
    parameter_name: str,
) -> tuple[
    tuple[int, ...] | None,
    str | None,
    DataflowTensorArgumentLayout | None,
]:
    parameter = call.operator.signature.parameters[parameter_name]
    annotation = call.operator.annotations.get(parameter_name, parameter.annotation)
    shape = None
    dtype = None
    if isinstance(annotation, tir.Buffer):
        try:
            shape = tuple(int(extent) for extent in annotation.shape)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Dataflow tensor argument {parameter_name!r} requires fixed shape metadata") from err
        dtype = str(annotation.dtype)
    layout = tensor_argument_layout(call.operator.attrs, parameter_name)
    if layout is not None:
        if shape is None or dtype is None:
            raise TypeError(f"Dataflow tensor layout parameter {parameter_name!r} must be a typed tensor argument")
        if layout.physical_shape != shape:
            raise ValueError(
                f"Dataflow tensor layout for {parameter_name!r} has physical shape {layout.physical_shape!r}, expected {shape!r}"
            )
    return shape, dtype, layout


def merge_tensor_arg_spec(
    spec: DataflowTensorArgSpec,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    layout: DataflowTensorArgumentLayout | None,
    operator_name: str,
    parameter_name: str,
) -> DataflowTensorArgSpec:
    context = f"tensor {spec.name!r} at operator {operator_name!r} parameter {parameter_name!r}"
    for name, previous, current in (
        ("shape", spec.shape, shape),
        ("dtype", spec.dtype, dtype),
    ):
        if previous is not None and current is not None and previous != current:
            raise ValueError(f"Dataflow {context} has conflicting {name}: {previous!r} != {current!r}")
    if spec.layout is not None and layout is not None and spec.layout.fingerprint != layout.fingerprint:
        raise ValueError(f"Dataflow {context} has conflicting layout fingerprints")
    return replace(
        spec,
        shape=spec.shape if spec.shape is not None else shape,
        dtype=spec.dtype if spec.dtype is not None else dtype,
        layout=spec.layout if spec.layout is not None else layout,
    )


def validated_tensor_value(
    spec: DataflowTensorArgSpec,
    value: Any,
) -> tuple[PackedTensorArg, DataflowRuntimeTensorMetadata]:
    record, metadata = pack_tensor_value(value, spec.name)
    layout = spec.layout
    if layout is None:
        return record, metadata
    if metadata.source == "raw_pointer":
        raise TypeError(f"Dataflow tensor argument {spec.name!r} has a typed layout and cannot be validated from a raw pointer")
    if metadata.shape != layout.physical_shape:
        raise ValueError(
            f"Dataflow tensor argument {spec.name!r} shape {metadata.shape!r} does "
            f"not match layout physical shape {layout.physical_shape!r}"
        )
    if spec.dtype is not None:
        expected_code, expected_bits = tensor_dtype_metadata_from_name(spec.dtype)
        if (metadata.dtype_code, metadata.dtype_bits) != (
            expected_code,
            expected_bits,
        ):
            raise TypeError(f"Dataflow tensor argument {spec.name!r} dtype metadata does not match {spec.dtype!r}")
    if metadata.device_type != "cuda":
        raise TypeError(f"Dataflow tensor argument {spec.name!r} with a typed layout must be CUDA")
    if metadata.layout_fingerprint != layout.fingerprint:
        raise ValueError(f"Dataflow tensor argument {spec.name!r} layout fingerprint does not match its compile-time contract")
    if layout.contiguous:
        expected_strides = contiguous_strides_bytes(
            layout.physical_shape,
            metadata.dtype_bits,
        )
        if metadata.strides_bytes != expected_strides:
            raise ValueError(f"Dataflow tensor argument {spec.name!r} must use contiguous physical layout")
    return record, metadata


def contiguous_strides_bytes(
    shape: tuple[int, ...],
    dtype_bits: int,
) -> tuple[int, ...]:
    if dtype_bits <= 0 or dtype_bits % 8:
        raise TypeError("Dataflow typed tensor layout requires a byte-aligned dtype")
    strides = [0] * len(shape)
    stride = dtype_bits // 8
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


def tensor_layout_fingerprint(value: Any) -> str | None:
    fingerprint = getattr(value, DATAFLOW_TENSOR_LAYOUT_FINGERPRINT_ATTR, None)
    if fingerprint is None:
        return None
    if not isinstance(fingerprint, str) or not fingerprint:
        raise TypeError("Dataflow tensor layout fingerprint metadata must be a string")
    return fingerprint


def resolve_tensor_name(value: Any, call: OperatorCall, parameter_name: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise TypeError(
        "Dataflow external tensor bindings must use non-empty string names: "
        f"operator={call.name!r}, parameter={parameter_name!r}, value={value!r}"
    )


def pack_tensor_value(
    value: Any,
    name: str,
) -> tuple[PackedTensorArg, DataflowRuntimeTensorMetadata]:
    if isinstance(value, int):
        return (
            PackedTensorArg(
                data_ptr=u64(
                    value,
                    f"Dataflow tensor argument {name!r} pointer",
                ),
                flags=DATAFLOW_TENSOR_ARG_FLAG_RAW_POINTER,
            ),
            DataflowRuntimeTensorMetadata(source="raw_pointer"),
        )

    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return record_from_data_ptr(value, data_ptr, name)

    cuda_array_interface = getattr(value, "__cuda_array_interface__", None)
    if cuda_array_interface is not None:
        return record_from_cuda_array_interface(
            value,
            cuda_array_interface,
            name,
        )

    raise TypeError(
        f"Dataflow tensor argument {name!r} must be a CUDA tensor-like object or a raw device pointer integer, got {type(value).__name__}"
    )


def record_from_cuda_array_interface(
    value: Any,
    interface: Any,
    name: str,
) -> tuple[PackedTensorArg, DataflowRuntimeTensorMetadata]:
    if not isinstance(interface, dict):
        raise TypeError(f"Dataflow tensor argument {name!r} has invalid __cuda_array_interface__ metadata")
    data = interface.get("data")
    if not isinstance(data, tuple) or not data:
        raise TypeError(f"Dataflow tensor argument {name!r} has invalid CUDA data pointer metadata")
    data_ptr = u64(int(data[0]), f"Dataflow tensor argument {name!r} pointer")
    shape = shape_tuple(interface.get("shape", ()), name)
    dtype_code, dtype_bits = tensor_dtype_metadata_from_typestr(interface.get("typestr"))
    strides_bytes = runtime_strides_bytes(
        shape,
        interface.get("strides"),
        dtype_bits,
        name,
        strides_are_bytes=True,
    )
    return (
        PackedTensorArg(
            data_ptr=data_ptr,
            ndim=len(shape),
            dtype_code=dtype_code,
            dtype_bits=dtype_bits,
            flags=DATAFLOW_TENSOR_ARG_FLAG_CUDA_ARRAY_INTERFACE,
        ),
        DataflowRuntimeTensorMetadata(
            source="cuda_array_interface",
            shape=shape,
            strides_bytes=strides_bytes,
            device_type="cuda",
            device_index=cuda_device_index(value),
            dtype_code=dtype_code,
            dtype_bits=dtype_bits,
            layout_fingerprint=tensor_layout_fingerprint(value),
        ),
    )


def record_from_data_ptr(
    value: Any,
    data_ptr: Any,
    name: str,
) -> tuple[PackedTensorArg, DataflowRuntimeTensorMetadata]:
    is_cuda = getattr(value, "is_cuda", None)
    if is_cuda is not None and not bool(is_cuda):
        raise TypeError(f"Dataflow tensor argument {name!r} must be a CUDA tensor")
    pointer = u64(int(data_ptr()), f"Dataflow tensor argument {name!r} pointer")
    shape = shape_tuple(getattr(value, "shape", ()), name)
    dtype_code, dtype_bits = tensor_dtype_metadata_from_name(getattr(value, "dtype", ""))
    stride = getattr(value, "stride", None)
    strides = stride() if callable(stride) else stride
    strides_bytes = runtime_strides_bytes(
        shape,
        strides,
        dtype_bits,
        name,
        strides_are_bytes=False,
        element_size=getattr(value, "element_size", None),
    )
    return (
        PackedTensorArg(
            data_ptr=pointer,
            ndim=len(shape),
            dtype_code=dtype_code,
            dtype_bits=dtype_bits,
            flags=DATAFLOW_TENSOR_ARG_FLAG_TORCH,
        ),
        DataflowRuntimeTensorMetadata(
            source="data_ptr",
            shape=shape,
            strides_bytes=strides_bytes,
            device_type="cuda",
            device_index=cuda_device_index(value),
            dtype_code=dtype_code,
            dtype_bits=dtype_bits,
            layout_fingerprint=tensor_layout_fingerprint(value),
        ),
    )


def shape_tuple(shape: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in shape)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Dataflow tensor argument {name!r} has invalid shape metadata") from err
    if any(item < 0 for item in result):
        raise ValueError(f"Dataflow tensor argument {name!r} shape must be non-negative")
    return result


def runtime_strides_bytes(
    shape: tuple[int, ...],
    strides: Any,
    dtype_bits: int,
    name: str,
    *,
    strides_are_bytes: bool,
    element_size: Any = None,
) -> tuple[int, ...] | None:
    if callable(element_size):
        item_bytes = int(element_size())
    elif dtype_bits > 0 and dtype_bits % 8 == 0:
        item_bytes = dtype_bits // 8
    else:
        return None
    if item_bytes <= 0:
        return None
    if strides is None:
        result = [0] * len(shape)
        stride = item_bytes
        for axis in range(len(shape) - 1, -1, -1):
            result[axis] = stride
            stride *= shape[axis]
        return tuple(result)
    try:
        result = tuple(int(item) for item in strides)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Dataflow tensor argument {name!r} has invalid stride metadata") from err
    if len(result) != len(shape) or any(item < 0 for item in result):
        raise ValueError(f"Dataflow tensor argument {name!r} strides must be non-negative and match its rank")
    if strides_are_bytes:
        return result
    return tuple(item * item_bytes for item in result)


def cuda_device_index(value: Any) -> int | None:
    device = getattr(value, "device", None)
    if device is not None:
        device_type = getattr(device, "type", None)
        if device_type is not None and str(device_type) != "cuda":
            return None
        index = getattr(device, "index", None)
        if index is None:
            index = getattr(device, "id", None)
        if index is not None:
            return int(index)
    dlpack_device = getattr(value, "__dlpack_device__", None)
    if callable(dlpack_device):
        try:
            device_type, index = dlpack_device()
        except Exception:
            return None
        if int(device_type) in (2, 13):
            return int(index)
    return None


def u64(value: int, context: str) -> int:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{context} must fit uint64, got {value}")
    return value
