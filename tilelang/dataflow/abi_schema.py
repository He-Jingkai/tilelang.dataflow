"""Single source of truth for the Dataflow host/device runtime ABI."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any


ABI_VERSION = 7
UINT32_SENTINEL = 0xFFFFFFFF

DATAFLOW_SLOT_ALIGNMENT = 16
DATAFLOW_SHARED_ALIGNMENT = 16
DATAFLOW_BARRIER_BYTES = 8
DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES = 1024
DATAFLOW_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = "load_balancing"
DATAFLOW_CLUSTER_LOAD_BALANCING_IMPLEMENTATION = "dataflow.launch.cluster.load_balancing.v1"

DATAFLOW_SLOT_FLAG_SCRATCH_BACKED = 1 << 0
DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL = 1 << 1
DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH = 1 << 2
DATAFLOW_SLOT_FLAG_COMMUNICATE = 1 << 3
DATAFLOW_INSTRUCTION_CLUSTER_COMM_COUNT_MASK = 0xFFFF
DATAFLOW_INSTRUCTION_CLUSTER_SEND_COUNT_SHIFT = 16
DATAFLOW_HANDOFF_FLAG_PRODUCER = 1 << 0
DATAFLOW_HANDOFF_FLAG_CONSUMER = 1 << 1
DATAFLOW_HANDOFF_FLAG_TAIL = 1 << 2
DATAFLOW_HANDOFF_FLAG_DISABLED = 1 << 3

DATAFLOW_TENSOR_DTYPE_UNKNOWN = 0
DATAFLOW_TENSOR_DTYPE_INT = 1
DATAFLOW_TENSOR_DTYPE_UINT = 2
DATAFLOW_TENSOR_DTYPE_FLOAT = 3
DATAFLOW_TENSOR_DTYPE_BFLOAT = 4
DATAFLOW_TENSOR_DTYPE_BOOL = 5

DATAFLOW_TENSOR_ARG_FLAG_RAW_POINTER = 1 << 0
DATAFLOW_TENSOR_ARG_FLAG_CUDA_ARRAY_INTERFACE = 1 << 1
DATAFLOW_TENSOR_ARG_FLAG_TORCH = 1 << 2


@dataclass(frozen=True)
class DataflowABIEnumMember:
    cpp_name: str
    python_names: tuple[str, ...]
    value: int


DATAFLOW_OPCODE_ABI_MEMBERS = (
    DataflowABIEnumMember("kExit", ("exit",), 0),
    DataflowABIEnumMember("kIter", ("iter", "map", "reshared"), 1),
    DataflowABIEnumMember("kReduce", ("reduce", "reduce_update"), 2),
    DataflowABIEnumMember("kFinalize", ("finalize",), 3),
    DataflowABIEnumMember("kClusterSync", ("cluster_sync",), 4),
)
DATAFLOW_COMM_KIND_ABI_MEMBERS = (
    DataflowABIEnumMember("kNone", ("none",), 0),
    DataflowABIEnumMember("kClusterSend", ("cluster_send",), 1),
    DataflowABIEnumMember("kClusterRecv", ("cluster_recv",), 2),
    DataflowABIEnumMember("kHBMSend", ("hbm_send",), 3),
    DataflowABIEnumMember("kHBMRecv", ("hbm_recv",), 4),
    DataflowABIEnumMember("kClusterRelease", ("cluster_release",), 5),
    DataflowABIEnumMember("kHBMRecvIssue", ("hbm_recv_issue",), 6),
    DataflowABIEnumMember("kHBMRecvWait", ("hbm_recv_wait",), 7),
)
DATAFLOW_TENSOR_DTYPE_ABI_MEMBERS = (
    DataflowABIEnumMember("kUnknown", ("unknown",), DATAFLOW_TENSOR_DTYPE_UNKNOWN),
    DataflowABIEnumMember("kInt", ("int",), DATAFLOW_TENSOR_DTYPE_INT),
    DataflowABIEnumMember("kUInt", ("uint",), DATAFLOW_TENSOR_DTYPE_UINT),
    DataflowABIEnumMember("kFloat", ("float",), DATAFLOW_TENSOR_DTYPE_FLOAT),
    DataflowABIEnumMember("kBFloat", ("bfloat",), DATAFLOW_TENSOR_DTYPE_BFLOAT),
    DataflowABIEnumMember("kBool", ("bool",), DATAFLOW_TENSOR_DTYPE_BOOL),
)


def enum_values(members: tuple[DataflowABIEnumMember, ...]) -> dict[str, int]:
    return {python_name: member.value for member in members for python_name in member.python_names}


DATAFLOW_OPCODE_ABI_VALUES = enum_values(DATAFLOW_OPCODE_ABI_MEMBERS)
DATAFLOW_COMM_KIND_ABI_VALUES = enum_values(DATAFLOW_COMM_KIND_ABI_MEMBERS)


@dataclass(frozen=True)
class DataflowABIField:
    name: str
    struct_code: str
    c_type: str


@dataclass(frozen=True)
class DataflowABIRecord:
    python_name: str
    cpp_name: str
    fields: tuple[DataflowABIField, ...]
    alignment: int = 16

    @property
    def struct_format(self) -> str:
        return "<" + "".join(field.struct_code for field in self.fields)

    @property
    def struct(self) -> struct.Struct:
        return struct.Struct(self.struct_format)

    @property
    def size(self) -> int:
        return self.struct.size

    @property
    def field_offsets(self) -> dict[str, int]:
        result: dict[str, int] = {}
        prefix = "<"
        for field in self.fields:
            result[field.name] = struct.calcsize(prefix)
            prefix += field.struct_code
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_name": self.python_name,
            "cpp_name": self.cpp_name,
            "struct_format": self.struct_format,
            "size": self.size,
            "alignment": self.alignment,
            "fields": [
                {
                    "name": field.name,
                    "c_type": field.c_type,
                    "offset": self.field_offsets[field.name],
                }
                for field in self.fields
            ],
        }


def u32(name: str) -> DataflowABIField:
    return DataflowABIField(name, "I", "uint32_t")


def u64(name: str) -> DataflowABIField:
    return DataflowABIField(name, "Q", "uint64_t")


INSTRUCTION_RECORD = DataflowABIRecord(
    "instruction",
    "DataflowInstruction",
    tuple(
        u32(name)
        for name in (
            "opcode",
            "handler_id",
            "task_id",
            "arg_offset",
            "comm_offset",
            "comm_count",
            "slot_id",
            "flags",
        )
    ),
)
HANDLER_ARG_RECORD = DataflowABIRecord(
    "handler_args",
    "DataflowHandlerArgs",
    tuple(
        u32(name)
        for name in (
            "task_id",
            "task_coord_offset",
            "task_coord_count",
            "range_begin",
            "range_end",
            "input_slot_offset",
            "input_slot_count",
            "output_slot",
            "handoff_peer_arg_offset",
            "handoff_plan_index",
            "handoff_binding_index",
            "handoff_stage_count",
            "handoff_arena_slot",
            "handoff_flags",
            "reserved0",
            "reserved1",
        )
    ),
)
TENSOR_ARG_RECORD = DataflowABIRecord(
    "tensor_arg",
    "DataflowTensorArg",
    (
        u64("data_ptr"),
        u32("ndim"),
        u32("dtype_code"),
        u32("dtype_bits"),
        u32("flags"),
        u64("reserved"),
    ),
)
SLOT_RECORD = DataflowABIRecord(
    "slot",
    "DataflowSlot",
    tuple(
        u32(name)
        for name in (
            "shared_offset",
            "global_offset",
            "bytes",
            "flag_index",
            "barrier_index",
            "owner_cta",
            "flags",
            "reserved",
        )
    ),
)
COMM_RECORD = DataflowABIRecord(
    "comm",
    "DataflowCommPlan",
    tuple(
        u32(name)
        for name in (
            "kind",
            "src_slot_id",
            "dst_slot_id",
            "peer_cta_rank",
            "barrier_phase",
            "flag_index",
            "flag_epoch",
            "barrier_index",
            "byte_offset",
            "byte_count",
            "segment_id",
            "segment_count",
        )
    ),
)

DATAFLOW_ABI_RECORDS = (
    INSTRUCTION_RECORD,
    HANDLER_ARG_RECORD,
    TENSOR_ARG_RECORD,
    SLOT_RECORD,
    COMM_RECORD,
)

INSTRUCTION_STRUCT = INSTRUCTION_RECORD.struct
ARG_STRUCT = HANDLER_ARG_RECORD.struct
TENSOR_ARG_STRUCT = TENSOR_ARG_RECORD.struct
SLOT_STRUCT = SLOT_RECORD.struct
COMM_STRUCT = COMM_RECORD.struct
UINT32_STRUCT = struct.Struct("<I")


def dataflow_abi_schema_dict() -> dict[str, Any]:
    return {
        "abi_version": ABI_VERSION,
        "sentinel": UINT32_SENTINEL,
        "layout": {
            "slot_alignment": DATAFLOW_SLOT_ALIGNMENT,
            "shared_alignment": DATAFLOW_SHARED_ALIGNMENT,
            "barrier_bytes": DATAFLOW_BARRIER_BYTES,
            "cuda_dynamic_shared_alignment": DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES,
        },
        "opcodes": dict(DATAFLOW_OPCODE_ABI_VALUES),
        "comm_kinds": dict(DATAFLOW_COMM_KIND_ABI_VALUES),
        "slot_flags": {
            "scratch_backed": DATAFLOW_SLOT_FLAG_SCRATCH_BACKED,
            "hbm_direct_global": DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
            "cluster_gated_push": DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH,
            "communicate": DATAFLOW_SLOT_FLAG_COMMUNICATE,
        },
        "handoff_flags": {
            "producer": DATAFLOW_HANDOFF_FLAG_PRODUCER,
            "consumer": DATAFLOW_HANDOFF_FLAG_CONSUMER,
            "tail": DATAFLOW_HANDOFF_FLAG_TAIL,
            "disabled": DATAFLOW_HANDOFF_FLAG_DISABLED,
        },
        "tensor_arg_flags": {
            "raw_pointer": DATAFLOW_TENSOR_ARG_FLAG_RAW_POINTER,
            "cuda_array_interface": DATAFLOW_TENSOR_ARG_FLAG_CUDA_ARRAY_INTERFACE,
            "torch": DATAFLOW_TENSOR_ARG_FLAG_TORCH,
        },
        "tensor_dtype_codes": enum_values(DATAFLOW_TENSOR_DTYPE_ABI_MEMBERS),
        "records": [record.to_dict() for record in DATAFLOW_ABI_RECORDS],
    }


def render_cpp_static_assert(condition: str, message: str) -> list[str]:
    """Render a stable clang-format-compatible assertion at column limit 80."""

    single_line = f'static_assert({condition}, "{message}");'
    if len(single_line) <= 80:
        return [single_line]
    condition_line = f"static_assert({condition},"
    message_line = f'              "{message}");'
    if len(condition_line) <= 80 and len(message_line) <= 80:
        return [condition_line, message_line]
    return [
        "static_assert(",
        f"    {condition},",
        f'    "{message}");',
    ]


def render_dataflow_abi_cpp_header() -> str:
    """Render the generated declarations included by ``dataflow_runtime.h``."""

    lines = [
        "// Generated by tilelang.dataflow.abi_schema; do not edit manually.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace tl {",
        "",
        f"static constexpr uint32_t kDataflowABIVersion = {ABI_VERSION}u;",
        f"static constexpr uint32_t kDataflowInvalidIndex = 0x{UINT32_SENTINEL:08x}u;",
        f"static constexpr uint32_t kDataflowSlotFlagScratchBacked = {DATAFLOW_SLOT_FLAG_SCRATCH_BACKED}u;",
        f"static constexpr uint32_t kDataflowSlotFlagHBMDirectGlobal = {DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL}u;",
        f"static constexpr uint32_t kDataflowSlotFlagClusterGatedPush = {DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH}u;",
        f"static constexpr uint32_t kDataflowSlotFlagCommunicate = {DATAFLOW_SLOT_FLAG_COMMUNICATE}u;",
        f"static constexpr uint32_t kDataflowHandoffFlagProducer = {DATAFLOW_HANDOFF_FLAG_PRODUCER}u;",
        f"static constexpr uint32_t kDataflowHandoffFlagConsumer = {DATAFLOW_HANDOFF_FLAG_CONSUMER}u;",
        f"static constexpr uint32_t kDataflowHandoffFlagTail = {DATAFLOW_HANDOFF_FLAG_TAIL}u;",
        f"static constexpr uint32_t kDataflowHandoffFlagDisabled = {DATAFLOW_HANDOFF_FLAG_DISABLED}u;",
        "",
    ]
    for enum_name, members in (
        ("DataflowOpcode", DATAFLOW_OPCODE_ABI_MEMBERS),
        ("DataflowCommKind", DATAFLOW_COMM_KIND_ABI_MEMBERS),
        ("DataflowTensorDTypeCode", DATAFLOW_TENSOR_DTYPE_ABI_MEMBERS),
    ):
        lines.append(f"enum class {enum_name} : uint32_t {{")
        lines.extend(f"  {member.cpp_name} = {member.value}," for member in members)
        lines.extend(("};", ""))
    for record in DATAFLOW_ABI_RECORDS:
        lines.append(f"struct alignas({record.alignment}) {record.cpp_name} {{")
        lines.extend(f"  {field.c_type} {field.name};" for field in record.fields)
        lines.append("};")
        lines.append("")
        lines.extend(
            render_cpp_static_assert(
                f"sizeof({record.cpp_name}) == {record.size}",
                f"{record.cpp_name} ABI size mismatch",
            )
        )
        lines.extend(
            render_cpp_static_assert(
                f"alignof({record.cpp_name}) == {record.alignment}",
                f"{record.cpp_name} ABI alignment mismatch",
            )
        )
        for field in record.fields:
            offset = record.field_offsets[field.name]
            lines.extend(
                render_cpp_static_assert(
                    f"offsetof({record.cpp_name}, {field.name}) == {offset}",
                    f"{record.cpp_name}::{field.name} ABI offset mismatch",
                )
            )
        lines.append("")
    lines.extend(
        [
            f"static constexpr uint32_t kDataflowInstructionClusterCommCountMask = 0x{DATAFLOW_INSTRUCTION_CLUSTER_COMM_COUNT_MASK:04x}u;",
            f"static constexpr uint32_t kDataflowInstructionClusterSendCountShift = {DATAFLOW_INSTRUCTION_CLUSTER_SEND_COUNT_SHIFT}u;",
            "",
            "} // namespace tl",
            "",
        ]
    )
    return "\n".join(lines)
