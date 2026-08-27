"""TMA descriptor metadata and host-side initialization for Dataflow wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .abi_schema import (
    DATAFLOW_TENSOR_DTYPE_BFLOAT,
    DATAFLOW_TENSOR_DTYPE_FLOAT,
    DATAFLOW_TENSOR_DTYPE_INT,
    DATAFLOW_TENSOR_DTYPE_UINT,
)
from .tensor_args import DataflowRuntimeTensorMetadata


_TENSOR_MAP_RUNTIME_DTYPES = {
    0: (
        {
            DATAFLOW_TENSOR_DTYPE_INT,
            DATAFLOW_TENSOR_DTYPE_UINT,
            DATAFLOW_TENSOR_DTYPE_FLOAT,
        },
        8,
    ),
    1: ({DATAFLOW_TENSOR_DTYPE_INT, DATAFLOW_TENSOR_DTYPE_UINT}, 16),
    2: ({DATAFLOW_TENSOR_DTYPE_UINT}, 32),
    3: ({DATAFLOW_TENSOR_DTYPE_INT}, 32),
    4: ({DATAFLOW_TENSOR_DTYPE_UINT}, 64),
    5: ({DATAFLOW_TENSOR_DTYPE_INT}, 64),
    6: ({DATAFLOW_TENSOR_DTYPE_FLOAT}, 16),
    7: ({DATAFLOW_TENSOR_DTYPE_FLOAT}, 32),
    8: ({DATAFLOW_TENSOR_DTYPE_FLOAT}, 64),
    9: ({DATAFLOW_TENSOR_DTYPE_BFLOAT}, 16),
}


@dataclass(frozen=True)
class DataflowTMADescriptorSpec:
    name: str
    tensor_name: str
    dtype: int
    tensor_rank: int
    global_dim: tuple[int, ...]
    global_stride: tuple[int, ...]
    box_dim: tuple[int, ...]
    element_strides: tuple[int, ...]
    interleave: int
    swizzle: int
    l2_promotion: int
    oob_fill: int
    is_im2col: bool = False
    lower_corner: tuple[int, ...] = ()
    upper_corner: tuple[int, ...] = ()
    smem_box_channel: int = 0
    smem_box_pixel: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tensor_name": self.tensor_name,
            "dtype": self.dtype,
            "tensor_rank": self.tensor_rank,
            "global_dim": list(self.global_dim),
            "global_stride": list(self.global_stride),
            "box_dim": list(self.box_dim),
            "element_strides": list(self.element_strides),
            "interleave": self.interleave,
            "swizzle": self.swizzle,
            "l2_promotion": self.l2_promotion,
            "oob_fill": self.oob_fill,
            "is_im2col": self.is_im2col,
            "lower_corner": list(self.lower_corner),
            "upper_corner": list(self.upper_corner),
            "smem_box_channel": self.smem_box_channel,
            "smem_box_pixel": self.smem_box_pixel,
        }


@dataclass(frozen=True)
class DataflowTMADescriptorHandle:
    spec: DataflowTMADescriptorSpec
    handle: Any


def validate_tma_descriptor_runtime_tensors(
    specs: tuple[DataflowTMADescriptorSpec, ...],
    *,
    tensor_data_ptrs: dict[str, int],
    tensor_metadata: dict[str, DataflowRuntimeTensorMetadata],
    expected_device_ordinal: int | None,
) -> None:
    """Validate host tensor facts against compiler-produced descriptor specs."""

    for spec in specs:
        try:
            data_ptr = int(tensor_data_ptrs[spec.tensor_name])
        except KeyError as err:
            raise ValueError(f"Dataflow TMA descriptor {spec.name!r} references missing tensor argument {spec.tensor_name!r}") from err
        if data_ptr == 0:
            raise ValueError(f"Dataflow TMA descriptor {spec.name!r} cannot use a null tensor pointer")
        if data_ptr % 16:
            raise ValueError(f"Dataflow TMA descriptor {spec.name!r} requires a 16-byte aligned tensor pointer, got 0x{data_ptr:x}")
        try:
            metadata = tensor_metadata[spec.tensor_name]
        except KeyError as err:
            raise ValueError(f"Dataflow TMA descriptor {spec.name!r} has no runtime shape, stride, and device metadata") from err
        if metadata.shape is None or metadata.strides_bytes is None:
            raise ValueError(
                f"Dataflow TMA descriptor {spec.name!r} cannot validate tensor "
                f"{spec.tensor_name!r} from {metadata.source}; pass a CUDA "
                "tensor-like object with shape and stride metadata instead "
                "of a raw pointer"
            )
        expected_dtype = _TENSOR_MAP_RUNTIME_DTYPES.get(spec.dtype)
        if expected_dtype is not None and (metadata.dtype_code not in expected_dtype[0] or metadata.dtype_bits != expected_dtype[1]):
            raise ValueError(
                f"Dataflow TMA descriptor {spec.name!r} runtime dtype mismatch: "
                f"expected TensorMap dtype {spec.dtype}, got code "
                f"{metadata.dtype_code} with {metadata.dtype_bits} bits"
            )
        expected_shape = tuple(reversed(spec.global_dim))
        expected_strides = tuple(reversed(spec.global_stride))
        if len(metadata.shape) != spec.tensor_rank:
            raise ValueError(f"Dataflow TMA descriptor {spec.name!r} expects rank {spec.tensor_rank}, got {len(metadata.shape)}")
        if metadata.shape != expected_shape:
            raise ValueError(
                f"Dataflow TMA descriptor {spec.name!r} runtime shape mismatch: expected {expected_shape}, got {metadata.shape}"
            )
        if metadata.strides_bytes != expected_strides:
            raise ValueError(
                f"Dataflow TMA descriptor {spec.name!r} runtime byte-stride "
                f"mismatch: expected {expected_strides}, got "
                f"{metadata.strides_bytes}"
            )
        if metadata.device_type != "cuda" or metadata.device_index is None:
            raise ValueError(f"Dataflow TMA descriptor {spec.name!r} requires explicit CUDA device metadata")
        if expected_device_ordinal is not None and metadata.device_index != expected_device_ordinal:
            raise ValueError(
                f"Dataflow TMA descriptor {spec.name!r} device mismatch: "
                f"expected cuda:{expected_device_ordinal}, got "
                f"cuda:{metadata.device_index}"
            )


def build_tma_descriptor_handles(
    specs: tuple[DataflowTMADescriptorSpec, ...],
    *,
    tensor_data_ptrs: dict[str, int],
) -> tuple[DataflowTMADescriptorHandle, ...]:
    if not specs:
        return ()

    try:
        from cuda.bindings import driver
    except Exception as err:  # pragma: no cover - environment dependent
        raise RuntimeError(f"CUDA driver bindings are required for Dataflow TMA descriptors: {err}") from err

    handles: list[DataflowTMADescriptorHandle] = []
    for spec in specs:
        try:
            global_address = int(tensor_data_ptrs[spec.tensor_name])
        except KeyError as err:
            raise ValueError(f"Dataflow TMA descriptor {spec.name!r} references missing tensor argument {spec.tensor_name!r}") from err
        if global_address == 0:
            raise ValueError(f"Dataflow TMA descriptor {spec.name!r} cannot be initialized with a null tensor pointer")
        if spec.is_im2col:
            result, handle = driver.cuTensorMapEncodeIm2col(
                driver.CUtensorMapDataType(spec.dtype),
                spec.tensor_rank,
                global_address,
                cuuint64_list(driver, spec.global_dim),
                cuuint64_list(driver, spec.global_stride[1:]),
                int_list(spec.lower_corner),
                int_list(spec.upper_corner),
                spec.smem_box_channel,
                spec.smem_box_pixel,
                cuuint32_list(driver, spec.element_strides),
                driver.CUtensorMapInterleave(spec.interleave),
                driver.CUtensorMapSwizzle(spec.swizzle),
                driver.CUtensorMapL2promotion(spec.l2_promotion),
                driver.CUtensorMapFloatOOBfill(spec.oob_fill),
            )
        else:
            result, handle = driver.cuTensorMapEncodeTiled(
                driver.CUtensorMapDataType(spec.dtype),
                spec.tensor_rank,
                global_address,
                cuuint64_list(driver, spec.global_dim),
                cuuint64_list(driver, spec.global_stride[1:]),
                cuuint32_list(driver, spec.box_dim),
                cuuint32_list(driver, spec.element_strides),
                driver.CUtensorMapInterleave(spec.interleave),
                driver.CUtensorMapSwizzle(spec.swizzle),
                driver.CUtensorMapL2promotion(spec.l2_promotion),
                driver.CUtensorMapFloatOOBfill(spec.oob_fill),
            )
        if result != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to initialize Dataflow TMA descriptor {spec.name}: {result}")
        handles.append(DataflowTMADescriptorHandle(spec=spec, handle=handle))
    return tuple(handles)


def cuuint64_list(driver: Any, values: tuple[int, ...]) -> list[Any]:
    return [driver.cuuint64_t(value) for value in values]


def cuuint32_list(driver: Any, values: tuple[int, ...]) -> list[Any]:
    return [driver.cuuint32_t(value) for value in values]


def int_list(values: tuple[int, ...]) -> list[int]:
    return [int(value) for value in values]
