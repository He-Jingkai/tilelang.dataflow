"""Canonical Dataflow dtype registry shared by ABI, layout, and lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .abi_schema import (
    DATAFLOW_TENSOR_DTYPE_BFLOAT,
    DATAFLOW_TENSOR_DTYPE_BOOL,
    DATAFLOW_TENSOR_DTYPE_FLOAT,
    DATAFLOW_TENSOR_DTYPE_INT,
    DATAFLOW_TENSOR_DTYPE_UINT,
    DATAFLOW_TENSOR_DTYPE_UNKNOWN,
)


@dataclass(frozen=True)
class DataflowDTypeInfo:
    name: str
    element_bytes: int
    cuda_type: str
    tensor_dtype_code: int
    dtype_bits: int
    primfunc_intermediate_supported: bool
    primfunc_scalar_supported: bool = False
    scalar_cuda_type: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.element_bytes <= 0 or self.dtype_bits <= 0:
            raise ValueError(f"Invalid Dataflow dtype registry entry {self!r}")
        if self.element_bytes * 8 < self.dtype_bits:
            raise ValueError(f"Dataflow dtype bits exceed storage size for {self.name!r}")
        if self.primfunc_scalar_supported and self.scalar_cuda_type is None:
            raise ValueError(f"Dataflow scalar dtype {self.name!r} requires a scalar CUDA type")

    @property
    def is_float8(self) -> bool:
        return self.tensor_dtype_code == DATAFLOW_TENSOR_DTYPE_FLOAT and self.dtype_bits == 8


def dtype(
    name: str,
    element_bytes: int,
    cuda_type: str,
    tensor_dtype_code: int,
    *,
    intermediate: bool = False,
    scalar: bool = False,
    scalar_cuda_type: str | None = None,
) -> DataflowDTypeInfo:
    return DataflowDTypeInfo(
        name=name,
        element_bytes=element_bytes,
        cuda_type=cuda_type,
        tensor_dtype_code=tensor_dtype_code,
        dtype_bits=1 if name == "bool" else element_bytes * 8,
        primfunc_intermediate_supported=intermediate,
        primfunc_scalar_supported=scalar,
        scalar_cuda_type=scalar_cuda_type,
    )


DATAFLOW_DTYPE_REGISTRY = tuple(
    (
        dtype("bool", 1, "bool", DATAFLOW_TENSOR_DTYPE_BOOL),
        dtype("int8", 1, "int8_t", DATAFLOW_TENSOR_DTYPE_INT),
        dtype("int16", 2, "int16_t", DATAFLOW_TENSOR_DTYPE_INT),
        dtype(
            "int32",
            4,
            "int32_t",
            DATAFLOW_TENSOR_DTYPE_INT,
            intermediate=True,
            scalar=True,
            scalar_cuda_type="int",
        ),
        dtype("int64", 8, "int64_t", DATAFLOW_TENSOR_DTYPE_INT, scalar_cuda_type="int64_t"),
        dtype("uint8", 1, "uint8_t", DATAFLOW_TENSOR_DTYPE_UINT),
        dtype("uint16", 2, "uint16_t", DATAFLOW_TENSOR_DTYPE_UINT),
        dtype(
            "uint32",
            4,
            "uint32_t",
            DATAFLOW_TENSOR_DTYPE_UINT,
            intermediate=True,
            scalar=True,
            scalar_cuda_type="uint",
        ),
        dtype("uint64", 8, "uint64_t", DATAFLOW_TENSOR_DTYPE_UINT, scalar_cuda_type="uint64_t"),
        dtype(
            "float16",
            2,
            "half_t",
            DATAFLOW_TENSOR_DTYPE_FLOAT,
            intermediate=True,
            scalar=True,
            scalar_cuda_type="half_t",
        ),
        dtype("bfloat16", 2, "__nv_bfloat16", DATAFLOW_TENSOR_DTYPE_BFLOAT),
        dtype(
            "float32",
            4,
            "float",
            DATAFLOW_TENSOR_DTYPE_FLOAT,
            intermediate=True,
            scalar=True,
            scalar_cuda_type="float",
        ),
        dtype("float64", 8, "double", DATAFLOW_TENSOR_DTYPE_FLOAT, scalar_cuda_type="double"),
        dtype("float8_e4m3", 1, "fp8_e4_t", DATAFLOW_TENSOR_DTYPE_FLOAT, intermediate=True),
        dtype("float8_e4m3fn", 1, "fp8_e4_t", DATAFLOW_TENSOR_DTYPE_FLOAT, intermediate=True),
        dtype("float8_e4m3fnuz", 1, "fp8_e4_t", DATAFLOW_TENSOR_DTYPE_FLOAT, intermediate=True),
        dtype("float8_e5m2", 1, "fp8_e5_t", DATAFLOW_TENSOR_DTYPE_FLOAT, intermediate=True),
        dtype("float8_e5m2fnuz", 1, "fp8_e5_t", DATAFLOW_TENSOR_DTYPE_FLOAT, intermediate=True),
    )
)
_DTYPES_BY_NAME = {info.name: info for info in DATAFLOW_DTYPE_REGISTRY}
_ALIASES = {
    "half": "float16",
    "fp16": "float16",
    "bf16": "bfloat16",
    "float": "float32",
    "double": "float64",
}


def normalize_dtype_name(dtype: Any | None) -> str | None:
    if dtype is None:
        return None
    normalized = str(dtype).strip().lower().rsplit(".", 1)[-1]
    return _ALIASES.get(normalized, normalized)


def dataflow_dtype_info(dtype: Any | None) -> DataflowDTypeInfo | None:
    normalized = normalize_dtype_name(dtype)
    return None if normalized is None else _DTYPES_BY_NAME.get(normalized)


def require_dataflow_dtype(dtype: Any | None) -> DataflowDTypeInfo:
    info = dataflow_dtype_info(dtype)
    if info is None:
        raise NotImplementedError(f"Dataflow does not support dtype {dtype!r}")
    return info


def primfunc_intermediate_dtype_names() -> tuple[str, ...]:
    return tuple(info.name for info in DATAFLOW_DTYPE_REGISTRY if info.primfunc_intermediate_supported)


def tensor_dtype_metadata_from_name(dtype: Any | None) -> tuple[int, int]:
    info = dataflow_dtype_info(dtype)
    if info is None:
        return DATAFLOW_TENSOR_DTYPE_UNKNOWN, 0
    return info.tensor_dtype_code, info.dtype_bits


def tensor_dtype_metadata_from_typestr(typestr: Any) -> tuple[int, int]:
    if not isinstance(typestr, str) or len(typestr) < 2:
        return DATAFLOW_TENSOR_DTYPE_UNKNOWN, 0
    kind = typestr[-2]
    try:
        bits = int(typestr[-1]) * 8
    except ValueError:
        return DATAFLOW_TENSOR_DTYPE_UNKNOWN, 0
    code = {
        "i": DATAFLOW_TENSOR_DTYPE_INT,
        "u": DATAFLOW_TENSOR_DTYPE_UINT,
        "f": DATAFLOW_TENSOR_DTYPE_FLOAT,
        "b": DATAFLOW_TENSOR_DTYPE_BOOL,
    }.get(kind, DATAFLOW_TENSOR_DTYPE_UNKNOWN)
    return code, bits if code != DATAFLOW_TENSOR_DTYPE_UNKNOWN else 0
