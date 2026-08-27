"""Dataflow handler ABI metadata shared by PrimFunc lowering and wrapper linking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any
from collections.abc import Mapping

from .abi_schema import DATAFLOW_SLOT_ALIGNMENT
from .dtype_registry import require_dataflow_dtype
from .handler_identity import DataflowHandlerIdentity, build_handler_registry
from .ir import IntermediateType
from .program import DataflowProgram, DataflowStage, DataflowStageKind
from .scheduler import InstructionPlan, DataflowOpcode
from .tensor_args import DataflowTensorArgPlan
from .wrapper import DataflowHandlerSpec, DataflowWrapperSpec


@dataclass(frozen=True)
class DataflowFieldLayout:
    name: str
    dtype: str
    c_type: str
    element_bytes: int
    offset: int
    shape: tuple[int, ...] | None
    numel: int


@dataclass(frozen=True)
class DataflowHandlerABI:
    handlers: tuple[DataflowHandlerSpec, ...]
    field_layouts: tuple[DataflowFieldLayout, ...]
    field_layouts_by_handler: Mapping[tuple[str, str], tuple[DataflowFieldLayout, ...]]
    input_field_layouts_by_handler: Mapping[tuple[str, str], tuple[DataflowFieldLayout, ...]]
    field_layouts_by_identity: Mapping[tuple[int, str], tuple[DataflowFieldLayout, ...]]
    input_field_layouts_by_identity: Mapping[tuple[int, str], tuple[DataflowFieldLayout, ...]]
    slot_bytes: int
    max_reduce_input_slots: int
    tensor_arg_plan: DataflowTensorArgPlan
    tensor_binding_indices: Mapping[tuple[str, str, str], int]
    tensor_binding_indices_by_handler: Mapping[tuple[str, str], Mapping[str, int]]
    tensor_binding_indices_by_identity: Mapping[tuple[int, str], Mapping[str, int]]

    def tensor_index(self, operator_kind: str, operator_name: str, parameter_name: str) -> int:
        key = (operator_kind, operator_name, parameter_name)
        try:
            return self.tensor_binding_indices[key]
        except KeyError as err:
            raise KeyError(
                "Missing Dataflow tensor binding for "
                f"operator_kind={operator_kind!r}, operator_name={operator_name!r}, "
                f"parameter_name={parameter_name!r}"
            ) from err

    def tensor_indices_for_handler(self, operator_kind: str, operator_name: str) -> dict[str, int]:
        result = self.tensor_binding_indices_by_handler.get((operator_kind, operator_name))
        return dict(result or {})

    def tensor_index_for_identity(
        self,
        identity: DataflowHandlerIdentity,
        parameter_name: str,
    ) -> int:
        indices = self.tensor_binding_indices_by_identity.get(identity.binding_key)
        if indices is None or parameter_name not in indices:
            raise KeyError(f"Missing Dataflow tensor binding for identity={identity.to_dict()!r}, parameter_name={parameter_name!r}")
        return indices[parameter_name]

    def tensor_indices_for_identity(
        self,
        identity: DataflowHandlerIdentity,
    ) -> dict[str, int]:
        return dict(self.tensor_binding_indices_by_identity.get(identity.binding_key, {}))

    def layouts_for_identity(
        self,
        identity: DataflowHandlerIdentity,
    ) -> tuple[DataflowFieldLayout, ...]:
        return self.field_layouts_by_identity.get(identity.binding_key, self.field_layouts)

    def input_layouts_for_identity(
        self,
        identity: DataflowHandlerIdentity,
    ) -> tuple[DataflowFieldLayout, ...]:
        return self.input_field_layouts_by_identity.get(identity.binding_key, ())

    def layouts_for_handler(self, operator_kind: str, operator_name: str) -> tuple[DataflowFieldLayout, ...]:
        return self.field_layouts_by_handler.get((operator_kind, operator_name), self.field_layouts)

    def input_layouts_for_handler(self, operator_kind: str, operator_name: str) -> tuple[DataflowFieldLayout, ...]:
        return self.input_field_layouts_by_handler.get((operator_kind, operator_name), ())


def field_layouts_for_intermediate(intermediate: IntermediateType) -> tuple[tuple[DataflowFieldLayout, ...], int]:
    offset = 0
    layouts: list[DataflowFieldLayout] = []
    for field in intermediate.fields:
        dtype_info = require_dataflow_dtype(field.dtype)
        if not dtype_info.primfunc_intermediate_supported:
            raise NotImplementedError(f"Dataflow PrimFunc handler ABI does not support dtype {field.dtype!r}")
        c_type, element_bytes = dtype_info.cuda_type, dtype_info.element_bytes
        shape = fixed_shape(field.shape, field.name)
        numel = math.prod(shape) if shape is not None else 1
        offset = align_up(offset, min(element_bytes, DATAFLOW_SLOT_ALIGNMENT))
        layouts.append(
            DataflowFieldLayout(
                name=field.name,
                dtype=str(field.dtype),
                c_type=c_type,
                element_bytes=element_bytes,
                offset=offset,
                shape=shape,
                numel=numel,
            )
        )
        offset += element_bytes * numel
    return tuple(layouts), align_up(offset, DATAFLOW_SLOT_ALIGNMENT)


def build_handler_abi(
    program: DataflowProgram,
    wrapper_spec: DataflowWrapperSpec,
    tensor_arg_plan: DataflowTensorArgPlan,
    plan: InstructionPlan,
) -> DataflowHandlerABI:
    program.validate()
    (
        field_layouts_by_handler,
        input_field_layouts_by_handler,
        field_layouts_by_identity,
        input_field_layouts_by_identity,
        fallback_field_layouts,
        slot_bytes,
    ) = handler_field_layouts(program)
    max_reduce_input_slots = max(
        (len(inst.input_slots) for inst in plan.instructions if inst.opcode in (DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE)),
        default=0,
    )
    return DataflowHandlerABI(
        handlers=wrapper_spec.handlers,
        field_layouts=fallback_field_layouts,
        field_layouts_by_handler=field_layouts_by_handler,
        input_field_layouts_by_handler=input_field_layouts_by_handler,
        field_layouts_by_identity=field_layouts_by_identity,
        input_field_layouts_by_identity=input_field_layouts_by_identity,
        slot_bytes=slot_bytes,
        max_reduce_input_slots=max_reduce_input_slots,
        tensor_arg_plan=tensor_arg_plan,
        tensor_binding_indices=tensor_binding_indices(tensor_arg_plan),
        tensor_binding_indices_by_handler=tensor_binding_indices_by_handler(tensor_arg_plan),
        tensor_binding_indices_by_identity=tensor_binding_indices_by_identity(tensor_arg_plan),
    )


def handler_field_layouts(
    program: DataflowProgram,
) -> tuple[
    Mapping[tuple[str, str], tuple[DataflowFieldLayout, ...]],
    Mapping[tuple[str, str], tuple[DataflowFieldLayout, ...]],
    Mapping[tuple[int, str], tuple[DataflowFieldLayout, ...]],
    Mapping[tuple[int, str], tuple[DataflowFieldLayout, ...]],
    tuple[DataflowFieldLayout, ...],
    int,
]:
    handler_registry = build_handler_registry(program)
    if not program.is_stage_graph:
        intermediate = program.intermediate_type
        if intermediate is None:
            raise ValueError("Dataflow PrimFunc handler ABI requires an intermediate type")
        field_layouts, slot_bytes = field_layouts_for_intermediate(intermediate)
        layouts_by_identity = {binding.identity.binding_key: field_layouts for binding in handler_registry.bindings_by_key.values()}
        return (
            MappingProxyType({}),
            MappingProxyType({}),
            MappingProxyType(layouts_by_identity),
            MappingProxyType({}),
            field_layouts,
            slot_bytes,
        )

    layouts_by_handler: dict[tuple[str, str], tuple[DataflowFieldLayout, ...]] = {}
    input_layouts_by_handler: dict[tuple[str, str], tuple[DataflowFieldLayout, ...]] = {}
    layouts_by_identity: dict[tuple[int, str], tuple[DataflowFieldLayout, ...]] = {}
    input_layouts_by_identity: dict[tuple[int, str], tuple[DataflowFieldLayout, ...]] = {}
    slot_bytes = 0
    fallback_field_layouts: tuple[DataflowFieldLayout, ...] = ()
    for stage in program.stages:
        if stage.call is None or stage.kind is DataflowStageKind.RESHARED:
            continue
        intermediate = handler_intermediate(stage)
        if intermediate is None:
            if stage.kind is not DataflowStageKind.MAP or stage.input_type is None:
                continue
            field_layouts = ()
            stage_slot_bytes = 0
        else:
            field_layouts, stage_slot_bytes = field_layouts_for_intermediate(intermediate)
        key = (handler_operator_kind(stage), stage.call.name)
        layouts_by_handler[key] = field_layouts
        identity_key = handler_registry.identity_for_stage_id(stage.stage_id).binding_key
        layouts_by_identity[identity_key] = field_layouts
        if stage.kind is DataflowStageKind.MAP and stage.input_type is not None:
            input_field_layouts, _ = field_layouts_for_intermediate(stage.input_type)
            input_layouts_by_handler[key] = input_field_layouts
            input_layouts_by_identity[identity_key] = input_field_layouts
        slot_bytes = max(slot_bytes, stage_slot_bytes)
        if not fallback_field_layouts:
            fallback_field_layouts = field_layouts
    if not layouts_by_handler:
        raise ValueError("Dataflow PrimFunc handler ABI requires an intermediate type")
    return (
        MappingProxyType(layouts_by_handler),
        MappingProxyType(input_layouts_by_handler),
        MappingProxyType(layouts_by_identity),
        MappingProxyType(input_layouts_by_identity),
        fallback_field_layouts,
        slot_bytes,
    )


def handler_intermediate(stage: DataflowStage) -> IntermediateType | None:
    if stage.kind is DataflowStageKind.FINALIZE:
        return stage.input_type
    return stage.physical_output_type or stage.output_type


def handler_operator_kind(stage: DataflowStage) -> str:
    if stage.call is None:
        raise ValueError(f"Dataflow stage {stage.name!r} has no operator call")
    if stage.kind is DataflowStageKind.MAP:
        return stage.call.kind.value
    return stage.kind.value


def fixed_shape(shape: tuple[Any, ...] | None, field_name: str) -> tuple[int, ...] | None:
    if shape is None:
        return None
    result = []
    for extent in shape:
        try:
            value = int(extent)
        except (TypeError, ValueError) as err:
            raise NotImplementedError(
                f"Dataflow PrimFunc handler ABI requires positive fixed extents for field {field_name!r}, got dynamic extent {extent!r}"
            ) from err
        if value <= 0:
            raise NotImplementedError(f"Dataflow PrimFunc handler ABI requires positive fixed extents for field {field_name!r}")
        result.append(value)
    return tuple(result)


def tensor_binding_indices(tensor_arg_plan: DataflowTensorArgPlan) -> Mapping[tuple[str, str, str], int]:
    return MappingProxyType(
        {
            (binding.operator_kind, binding.operator_name, binding.parameter_name): binding.tensor_index
            for binding in tensor_arg_plan.bindings
        }
    )


def tensor_binding_indices_by_handler(
    tensor_arg_plan: DataflowTensorArgPlan,
) -> Mapping[tuple[str, str], Mapping[str, int]]:
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for binding in tensor_arg_plan.bindings:
        key = (binding.operator_kind, binding.operator_name)
        grouped.setdefault(key, {})[binding.parameter_name] = binding.tensor_index
    return MappingProxyType({key: MappingProxyType(value) for key, value in grouped.items()})


def tensor_binding_indices_by_identity(
    tensor_arg_plan: DataflowTensorArgPlan,
) -> Mapping[tuple[int, str], Mapping[str, int]]:
    grouped: dict[tuple[int, str], dict[str, int]] = {}
    for binding in tensor_arg_plan.bindings:
        key = binding.handler_identity.binding_key
        grouped.setdefault(key, {})[binding.parameter_name] = binding.tensor_index
    return MappingProxyType({key: MappingProxyType(value) for key, value in grouped.items()})


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
