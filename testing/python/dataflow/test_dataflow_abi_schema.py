from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

import tilelang.dataflow as df
from tilelang.dataflow.abi_schema import (
    COMM_RECORD,
    HANDLER_ARG_RECORD,
    INSTRUCTION_RECORD,
    SLOT_RECORD,
    TENSOR_ARG_RECORD,
)
from tilelang.dataflow.handler_codegen import (
    dataflow_buffer_cuda_c_type,
    dataflow_scalar_cuda_c_type,
)


def test_generated_cpp_header_matches_canonical_abi_schema():
    repository_root = Path(__file__).resolve().parents[3]
    generated_header = repository_root / "src/tl_templates/cuda/dataflow_abi_generated.h"

    assert generated_header.read_text() == df.render_dataflow_abi_cpp_header()


@pytest.mark.parametrize(
    ("record", "python_type"),
    (
        (INSTRUCTION_RECORD, df.PackedInstruction),
        (HANDLER_ARG_RECORD, df.PackedHandlerArgs),
        (TENSOR_ARG_RECORD, df.PackedTensorArg),
        (SLOT_RECORD, df.PackedSlot),
        (COMM_RECORD, df.PackedComm),
    ),
)
def test_python_record_order_size_offsets_and_packing_follow_schema(record, python_type):
    field_names = tuple(field.name for field in record.fields)
    expected_sizes = {
        HANDLER_ARG_RECORD: 64,
        COMM_RECORD: 48,
    }

    assert tuple(field.name for field in fields(python_type)) == field_names
    assert record.size == expected_sizes.get(record, 32)
    assert record.alignment == 16
    assert tuple(record.field_offsets.values()) == tuple(range(0, record.size, 4)) or (
        record is TENSOR_ARG_RECORD and tuple(record.field_offsets.values()) == (0, 8, 12, 16, 20, 24)
    )

    value = python_type(*range(1, len(field_names) + 1))
    packed = record.struct.pack(*value.to_tuple())
    assert len(packed) == record.size
    assert record.struct.unpack(packed) == value.to_tuple()


def test_dtype_registry_drives_layout_tensor_metadata_and_codegen_types():
    float16 = df.require_dataflow_dtype("torch.float16")
    bfloat16 = df.require_dataflow_dtype("bf16")

    assert float16.element_bytes == 2
    assert float16.dtype_bits == 16
    assert float16.primfunc_intermediate_supported is True
    assert dataflow_buffer_cuda_c_type("float16") == "half_t"
    assert dataflow_scalar_cuda_c_type("float16") == "half_t"
    assert bfloat16.cuda_type == "__nv_bfloat16"
    assert bfloat16.primfunc_intermediate_supported is False
    assert len({info.name for info in df.DATAFLOW_DTYPE_REGISTRY}) == len(df.DATAFLOW_DTYPE_REGISTRY)

    with pytest.raises(NotImplementedError, match="does not support tensor dtype"):
        dataflow_buffer_cuda_c_type("bfloat16")
    with pytest.raises(NotImplementedError, match="does not support dtype"):
        df.require_dataflow_dtype("complex64")


def test_abi_schema_exports_versioned_layout_and_enum_contracts():
    schema = df.dataflow_abi_schema_dict()

    assert schema["abi_version"] == df.ABI_VERSION == 7
    assert schema["sentinel"] == df.UINT32_SENTINEL
    assert schema["layout"] == {
        "slot_alignment": df.DATAFLOW_SLOT_ALIGNMENT,
        "shared_alignment": df.DATAFLOW_SHARED_ALIGNMENT,
        "barrier_bytes": df.DATAFLOW_BARRIER_BYTES,
        "cuda_dynamic_shared_alignment": df.DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES,
    }
    assert {record["size"] for record in schema["records"]} == {32, 48, 64}
    assert {record["alignment"] for record in schema["records"]} == {16}
    assert "cluster_pull" not in schema["slot_flags"]
    assert schema["slot_flags"]["cluster_gated_push"] == 4
    assert schema["slot_flags"]["communicate"] == 8
