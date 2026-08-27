"""Dataflow program graph builder and validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any
from collections.abc import Mapping

from .ir import (
    IntermediateType,
    DATAFLOW_INTERMEDIATE_ATTR,
    DataflowOperatorKind,
    OperatorCall,
    get_intermediate_type,
)
from .operation_contracts import (
    DATAFLOW_HANDOFF_CONTRACT_ATTR,
    DATAFLOW_PIPELINE_CONTRACT_ATTR,
    DATAFLOW_RANGE_CONTRACT_ATTR,
    DATAFLOW_TRANSPORT_CONTRACT_ATTR,
    DataflowCrossHandlerHandoffRequest,
    DataflowOperationRequest,
    DataflowPipelineRequest,
    DataflowRangeCoarseningRequest,
    DataflowResharedFieldMapping,
    DataflowResharedTransportRequest,
    normalize_operation_request,
)
from .physical_contract import (
    DATAFLOW_OUTPUT_SLOT_NONE,
    operator_physical_contract,
)


OperationRequestInput = DataflowOperationRequest | Mapping[str, Any]
_DATAFLOW_OPERATOR_NAME_PREFIX = "dataflow_"


def as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def require_operator_call(value: Any, kind: DataflowOperatorKind, context: str) -> OperatorCall:
    if not isinstance(value, OperatorCall):
        raise TypeError(f"{context} expects a Dataflow OperatorCall, got {value!r}")
    if value.kind is not kind:
        raise TypeError(f"{context} expects a Dataflow {kind.value} call, got {value.kind.value} call {value.name!r}")
    return value


def require_operator_call_one_of(value: Any, kinds: tuple[DataflowOperatorKind, ...], context: str) -> OperatorCall:
    if not isinstance(value, OperatorCall):
        raise TypeError(f"{context} expects a Dataflow OperatorCall, got {value!r}")
    if value.kind not in kinds:
        expected = ", ".join(kind.value for kind in kinds)
        raise TypeError(f"{context} expects a Dataflow {expected} call, got {value.kind.value} call {value.name!r}")
    return value


def as_intermediate_type(value: Any, context: str) -> IntermediateType:
    intermediate = get_intermediate_type(value)
    if intermediate is None:
        raise TypeError(f"{context} expects a @T.dataflow_intermediate type, got {value!r}")
    return intermediate


def first_input_type(call: OperatorCall, context: str) -> IntermediateType:
    input_types = call.operator.input_types
    if not input_types:
        raise TypeError(f"{context} call {call.name!r} does not consume a Dataflow intermediate")
    return input_types[0]


def default_map_stage_name(call: OperatorCall) -> str:
    if call.name.startswith(_DATAFLOW_OPERATOR_NAME_PREFIX):
        return call.name[len(_DATAFLOW_OPERATOR_NAME_PREFIX) :]
    return call.name


def with_operation_contracts(
    attrs: Mapping[str, Any],
    *contracts: tuple[str, OperationRequestInput | None, type[DataflowOperationRequest]],
) -> dict[str, Any]:
    normalized = dict(attrs)
    for attr_name, value, expected_type in contracts:
        if value is None:
            continue
        normalized[attr_name] = normalize_operation_request(value, expected_type)
    return normalized


@dataclass(frozen=True)
class TaskDomain:
    axes: tuple[Any, ...]


class DataflowStageKind(str, Enum):
    MAP = "map"
    REDUCE = "reduce"
    RESHARED = "reshared"
    FINALIZE = "finalize"


@dataclass(frozen=True)
class DataflowStage:
    stage_id: int
    name: str
    kind: DataflowStageKind
    call: OperatorCall | None
    input_type: IntermediateType | None
    output_type: IntermediateType | None
    deps: tuple[int, ...]
    physical_output_type: IntermediateType | None = None
    output_arity: int = 1
    task_args: tuple[Any, ...] = ()
    range_axis: Any | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PartialStage:
    iter_call: OperatorCall
    task_args: tuple[Any, ...] = ()
    range_axis: Any | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def output_type(self) -> IntermediateType:
        output_type = self.iter_call.output_type
        assert output_type is not None
        return output_type


@dataclass(frozen=True)
class ReduceStage:
    reduce_call: OperatorCall
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def input_type(self) -> IntermediateType:
        return first_input_type(self.reduce_call, "Reduce")

    @property
    def output_type(self) -> IntermediateType:
        output_type = self.reduce_call.output_type
        assert output_type is not None
        return output_type


@dataclass(frozen=True)
class FinalizeStage:
    finalize_call: OperatorCall
    fused_reduce_calls: tuple[OperatorCall, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def input_type(self) -> IntermediateType:
        return first_input_type(self.finalize_call, "Finalize")

    def fused_reduce_call_for_arity(self, arity: int) -> OperatorCall | None:
        return next(
            (call for call in self.fused_reduce_calls if len(call.operator.input_types) == arity),
            None,
        )


@dataclass
class DataflowProgram:
    task_domain: TaskDomain
    dynamic_ranges: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    partial_stage: PartialStage | None = None
    reduce_stage: ReduceStage | None = None
    finalize_stage: FinalizeStage | None = None
    stages: list[DataflowStage] = field(default_factory=list)

    @property
    def intermediate_type(self) -> IntermediateType | None:
        if self.partial_stage is None:
            return None
        return self.partial_stage.output_type

    @property
    def is_complete(self) -> bool:
        if not self.stages:
            return False
        if self.is_stage_graph:
            return self.stages[-1].kind in (DataflowStageKind.MAP, DataflowStageKind.FINALIZE)
        return self.stages[-1].kind is DataflowStageKind.FINALIZE

    @property
    def is_stage_graph(self) -> bool:
        return any(stage.kind is DataflowStageKind.RESHARED for stage in self.stages)

    def stage(self, name_or_id: str | int) -> DataflowStage:
        if isinstance(name_or_id, int):
            for stage in self.stages:
                if stage.stage_id == name_or_id:
                    return stage
            raise KeyError(f"Unknown Dataflow stage id {name_or_id}")
        for stage in self.stages:
            if stage.name == name_or_id:
                return stage
        raise KeyError(f"Unknown Dataflow stage {name_or_id!r}")

    def append_stage(
        self,
        *,
        name: str,
        kind: DataflowStageKind,
        call: OperatorCall | None,
        input_type: IntermediateType | None,
        output_type: IntermediateType | None,
        deps: tuple[int, ...],
        physical_output_type: IntermediateType | None = None,
        output_arity: int = 1,
        task_args: tuple[Any, ...] = (),
        range_axis: Any | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> DataflowStage:
        if any(stage.name == name for stage in self.stages):
            raise ValueError(f"Dataflow program already has a stage named {name!r}")
        if output_arity <= 0:
            raise ValueError(f"Dataflow stage {name!r} output_arity must be positive, got {output_arity}")
        stage = DataflowStage(
            stage_id=len(self.stages),
            name=name,
            kind=kind,
            call=call,
            input_type=input_type,
            output_type=output_type,
            deps=deps,
            physical_output_type=physical_output_type if physical_output_type is not None else output_type,
            output_arity=output_arity,
            task_args=task_args,
            range_axis=range_axis,
            attrs=dict(attrs or {}),
        )
        self.stages.append(stage)
        return stage

    def resolve_upstream_stage(self, name_or_id: str | int, context: str) -> DataflowStage:
        try:
            return self.stage(name_or_id)
        except KeyError as err:
            raise ValueError(f"{context} requires an upstream stage, got {name_or_id!r}") from err

    def normalize_task_args(self, task_args: Any, context: str) -> tuple[Any, ...]:
        normalized_task_args = as_tuple(task_args)
        if normalized_task_args and len(normalized_task_args) != len(self.task_domain.axes):
            raise ValueError(
                f"{context} task_args rank must match task_domain rank: "
                f"got {len(normalized_task_args)} task args for {len(self.task_domain.axes)} task axes"
            )
        return normalized_task_args

    def ensure_not_finalized(self, action: str) -> None:
        if self.finalize_stage is not None:
            raise ValueError(f"Cannot {action}: Dataflow program is already finalized")

    def partial(
        self,
        iter_call: OperatorCall,
        *,
        task_args: Any = (),
        range_axis: Any | None = None,
        range_contract: OperationRequestInput | None = None,
        pipeline_contract: OperationRequestInput | None = None,
        **attrs: Any,
    ) -> DataflowProgram:
        self.ensure_not_finalized("add partial stage")
        if self.partial_stage is not None:
            raise ValueError("Dataflow program already has a partial stage")
        call = require_operator_call(iter_call, DataflowOperatorKind.ITER, "DataflowProgram.partial")
        normalized_task_args = self.normalize_task_args(task_args, "DataflowProgram.partial")
        normalized_attrs = with_operation_contracts(
            attrs,
            (DATAFLOW_RANGE_CONTRACT_ATTR, range_contract, DataflowRangeCoarseningRequest),
            (DATAFLOW_PIPELINE_CONTRACT_ATTR, pipeline_contract, DataflowPipelineRequest),
        )
        if DATAFLOW_RANGE_CONTRACT_ATTR in normalized_attrs and range_axis is None:
            raise ValueError("Dataflow range_contract requires range_axis")
        self.partial_stage = PartialStage(
            iter_call=call,
            task_args=normalized_task_args,
            range_axis=range_axis,
            attrs=normalized_attrs,
        )
        self.append_stage(
            name="partial",
            kind=DataflowStageKind.MAP,
            call=call,
            input_type=None,
            output_type=self.partial_stage.output_type,
            deps=(),
            task_args=normalized_task_args,
            range_axis=range_axis,
            attrs=normalized_attrs,
        )
        return self

    def map(
        self,
        map_call: OperatorCall,
        *,
        name: str | None = None,
        input: str | int | None = None,  # pylint: disable=redefined-builtin
        task_args: Any = (),
        range_axis: Any | None = None,
        range_contract: OperationRequestInput | None = None,
        pipeline_contract: OperationRequestInput | None = None,
        handoff_contract: OperationRequestInput | None = None,
        **attrs: Any,
    ) -> DataflowProgram:
        self.ensure_not_finalized("add map stage")
        call = require_operator_call_one_of(
            map_call,
            (DataflowOperatorKind.MAP, DataflowOperatorKind.ITER),
            "DataflowProgram.map",
        )
        output_type = call.output_type
        input_type = call.operator.input_types[0] if call.operator.input_types else None
        deps: tuple[int, ...] = ()
        if input is not None:
            upstream = self.resolve_upstream_stage(input, "DataflowProgram.map")
            deps = (upstream.stage_id,)
            upstream_type = upstream.physical_output_type or upstream.output_type
            if input_type is None:
                input_type = upstream_type
            elif upstream_type is not None and input_type is not upstream_type:
                raise TypeError(
                    "DataflowProgram.map intermediate type mismatch: "
                    f"upstream stage {upstream.name!r} produces {upstream_type.name}, "
                    f"map consumes {input_type.name}"
                )
        normalized_task_args = self.normalize_task_args(task_args, "DataflowProgram.map")
        normalized_attrs = with_operation_contracts(
            attrs,
            (DATAFLOW_RANGE_CONTRACT_ATTR, range_contract, DataflowRangeCoarseningRequest),
            (DATAFLOW_PIPELINE_CONTRACT_ATTR, pipeline_contract, DataflowPipelineRequest),
            (
                DATAFLOW_HANDOFF_CONTRACT_ATTR,
                handoff_contract,
                DataflowCrossHandlerHandoffRequest,
            ),
        )
        if DATAFLOW_RANGE_CONTRACT_ATTR in normalized_attrs and range_axis is None:
            raise ValueError("Dataflow range_contract requires range_axis")
        self.append_stage(
            name=name or default_map_stage_name(call),
            kind=DataflowStageKind.MAP,
            call=call,
            input_type=input_type,
            output_type=output_type,
            deps=deps,
            task_args=normalized_task_args,
            range_axis=range_axis,
            attrs=normalized_attrs,
        )
        return self

    def reshared(
        self,
        *,
        input: str | int,  # pylint: disable=redefined-builtin
        output_type: Any | None = None,
        name: str = "reshared",
        physical_output_type: Any | None = None,
        output_arity: int | None = None,
        transport_contract: OperationRequestInput | None = None,
        **attrs: Any,
    ) -> DataflowProgram:
        self.ensure_not_finalized("add reshared stage")
        upstream = self.resolve_upstream_stage(input, "DataflowProgram.reshared")
        normalized_attrs = with_operation_contracts(
            attrs,
            (
                DATAFLOW_TRANSPORT_CONTRACT_ATTR,
                transport_contract,
                DataflowResharedTransportRequest,
            ),
        )
        request = normalized_attrs.get(DATAFLOW_TRANSPORT_CONTRACT_ATTR)
        normalized_physical_type = (
            upstream.physical_output_type or upstream.output_type
            if physical_output_type is None
            else as_intermediate_type(
                physical_output_type,
                "DataflowProgram.reshared physical_output_type",
            )
        )
        if normalized_physical_type is None:
            raise TypeError("DataflowProgram.reshared requires a physical output intermediate type")
        resolved_output_arity = (
            request.physical_output_arity
            if output_arity is None and isinstance(request, DataflowResharedTransportRequest)
            else 1
            if output_arity is None
            else output_arity
        )
        if isinstance(resolved_output_arity, bool) or not isinstance(resolved_output_arity, int) or resolved_output_arity <= 0:
            raise ValueError("Dataflow reshared output_arity must be a positive integer")
        if isinstance(request, DataflowResharedTransportRequest) and request.physical_output_arity != resolved_output_arity:
            raise ValueError(
                "Dataflow reshared transport physical_output_arity must match "
                "stage output_arity: "
                f"{request.physical_output_arity} != {resolved_output_arity}"
            )
        if output_type is not None:
            normalized_output_type = as_intermediate_type(
                output_type,
                "DataflowProgram.reshared output_type",
            )
        elif isinstance(request, DataflowResharedTransportRequest) and request.field_mappings:
            normalized_output_type = infer_reshared_logical_type(
                normalized_physical_type,
                request,
            )
        else:
            normalized_output_type = normalized_physical_type
        self.append_stage(
            name=name,
            kind=DataflowStageKind.RESHARED,
            call=None,
            input_type=upstream.physical_output_type or upstream.output_type,
            output_type=normalized_output_type,
            deps=(upstream.stage_id,),
            physical_output_type=normalized_physical_type,
            output_arity=resolved_output_arity,
            attrs=normalized_attrs,
        )
        return self

    def reduce(
        self,
        reduce_call: OperatorCall,
        *,
        pipeline_contract: OperationRequestInput | None = None,
        **attrs: Any,
    ) -> DataflowProgram:
        self.ensure_not_finalized("add reduce stage")
        if self.partial_stage is None:
            raise ValueError("DataflowProgram.reduce requires a partial stage first")
        if self.reduce_stage is not None:
            raise ValueError("Dataflow program already has a reduce stage")
        call = require_operator_call(reduce_call, DataflowOperatorKind.REDUCE, "DataflowProgram.reduce")
        reduce_input_type = first_input_type(call, "DataflowProgram.reduce")
        if reduce_input_type is not self.partial_stage.output_type:
            raise TypeError(
                "DataflowProgram.reduce intermediate type mismatch: "
                f"partial produces {self.partial_stage.output_type.name}, reduce consumes {reduce_input_type.name}"
            )
        if call.output_type is not self.partial_stage.output_type:
            output_name = call.output_type.name if call.output_type is not None else "None"
            raise TypeError(
                "DataflowProgram.reduce intermediate type mismatch: "
                f"partial produces {self.partial_stage.output_type.name}, reduce returns {output_name}"
            )
        normalized_attrs = with_operation_contracts(
            attrs,
            (
                DATAFLOW_PIPELINE_CONTRACT_ATTR,
                pipeline_contract,
                DataflowPipelineRequest,
            ),
        )
        self.reduce_stage = ReduceStage(reduce_call=call, attrs=normalized_attrs)
        partial_stage = self.stage("partial")
        self.append_stage(
            name="reduce",
            kind=DataflowStageKind.REDUCE,
            call=call,
            input_type=self.reduce_stage.input_type,
            output_type=self.reduce_stage.output_type,
            deps=(partial_stage.stage_id,),
            attrs=normalized_attrs,
        )
        return self

    def finalize(
        self,
        finalize_call: OperatorCall,
        *,
        fused_reduce: OperatorCall | tuple[OperatorCall, ...] | None = None,
        input: str | int | None = None,  # pylint: disable=redefined-builtin
        name: str = "finalize",
        pipeline_contract: OperationRequestInput | None = None,
        **attrs: Any,
    ) -> DataflowProgram:
        self.ensure_not_finalized("add finalize stage")
        call = require_operator_call(finalize_call, DataflowOperatorKind.FINALIZE, "DataflowProgram.finalize")
        finalize_input_type = first_input_type(call, "DataflowProgram.finalize")
        if input is None:
            if self.partial_stage is None:
                raise ValueError("DataflowProgram.finalize requires a partial stage first")
            if self.reduce_stage is None:
                raise ValueError("DataflowProgram.finalize requires a reduce stage first")
            upstream_type = self.reduce_stage.output_type
            deps = (self.stage("reduce").stage_id,)
        else:
            upstream = self.resolve_upstream_stage(input, "DataflowProgram.finalize")
            upstream_type = upstream.physical_output_type or upstream.output_type
            deps = (upstream.stage_id,)
        if finalize_input_type is not upstream_type:
            raise TypeError(
                "DataflowProgram.finalize intermediate type mismatch: "
                f"upstream produces {upstream_type.name if upstream_type is not None else 'None'}, "
                f"finalize consumes {finalize_input_type.name}"
            )
        fused_reduce_calls: tuple[OperatorCall, ...] = ()
        if fused_reduce is not None:
            if input is not None:
                raise ValueError(
                    "DataflowProgram.finalize fused_reduce is only supported for the canonical partial/reduce/finalize pipeline"
                )
            raw_fused_reduce_calls = fused_reduce if isinstance(fused_reduce, tuple) else (fused_reduce,)
            fused_reduce_calls = tuple(
                require_operator_call(
                    item,
                    DataflowOperatorKind.FINALIZE,
                    "DataflowProgram.finalize fused_reduce",
                )
                for item in raw_fused_reduce_calls
            )
            arities: set[int] = set()
            for fused_reduce_call in fused_reduce_calls:
                fused_input_types = fused_reduce_call.operator.input_types
                if len(fused_input_types) < 2:
                    raise TypeError("DataflowProgram.finalize fused_reduce must consume at least two intermediate inputs")
                if len(fused_input_types) in arities:
                    raise ValueError(f"DataflowProgram.finalize fused_reduce declares duplicate input arity {len(fused_input_types)}")
                arities.add(len(fused_input_types))
                if any(item is not upstream_type for item in fused_input_types):
                    names = ", ".join(item.name for item in fused_input_types)
                    raise TypeError(
                        "DataflowProgram.finalize fused_reduce intermediate type mismatch: "
                        f"upstream produces {upstream_type.name}, fused finalizer consumes {names}"
                    )
        normalized_attrs = with_operation_contracts(
            attrs,
            (
                DATAFLOW_PIPELINE_CONTRACT_ATTR,
                pipeline_contract,
                DataflowPipelineRequest,
            ),
        )
        self.finalize_stage = FinalizeStage(
            finalize_call=call,
            fused_reduce_calls=fused_reduce_calls,
            attrs=normalized_attrs,
        )
        self.append_stage(
            name=name,
            kind=DataflowStageKind.FINALIZE,
            call=call,
            input_type=finalize_input_type,
            output_type=None,
            deps=deps,
            attrs=normalized_attrs,
        )
        return self

    def validate(self) -> DataflowProgram:
        if not self.stages:
            raise ValueError("Dataflow program is missing a partial stage")
        seen_stage_ids: set[int] = set()
        for expected_id, stage in enumerate(self.stages):
            if stage.stage_id != expected_id:
                raise ValueError(f"Dataflow program stage ids must be contiguous: expected {expected_id}, got {stage.stage_id}")
            if stage.output_arity <= 0:
                raise ValueError(f"Dataflow stage {stage.name!r} output_arity must be positive")
            for dep in stage.deps:
                if dep not in seen_stage_ids:
                    raise ValueError(f"Dataflow stage {stage.name!r} depends on unknown or later stage id {dep}")
            seen_stage_ids.add(stage.stage_id)
            if stage.kind is DataflowStageKind.MAP and stage.call is not None and stage.call.output_type is None:
                physical = operator_physical_contract(stage.call.operator.attrs)
                if (
                    stage.kind is not DataflowStageKind.MAP
                    or physical.output_slot != DATAFLOW_OUTPUT_SLOT_NONE
                    or expected_id != len(self.stages) - 1
                ):
                    raise ValueError("an outputless Dataflow map must be the terminal program stage")
        if self.is_stage_graph:
            if self.stages[-1].kind not in (DataflowStageKind.MAP, DataflowStageKind.FINALIZE):
                raise ValueError("Dataflow stage graph must end with a map or finalize stage")
            if self.stages[-1].kind is DataflowStageKind.FINALIZE and self.finalize_stage is None:
                raise ValueError("Dataflow program is missing a finalize stage")
            return self
        if self.partial_stage is None:
            raise ValueError("Dataflow program is missing a partial stage")
        if self.reduce_stage is None:
            raise ValueError("Dataflow program is missing a reduce stage")
        if self.finalize_stage is None:
            raise ValueError("Dataflow program is missing a finalize stage")
        return self


def dataflow_program(
    task_domain: Any,
    *,
    dynamic_ranges: dict[str, Any] | None = None,
    **attrs: Any,
) -> DataflowProgram:
    axes = as_tuple(task_domain)
    if not axes:
        raise ValueError("dataflow_program task_domain must contain at least one axis")
    return DataflowProgram(
        task_domain=TaskDomain(axes=axes),
        dynamic_ranges=dict(dynamic_ranges or {}),
        attrs=dict(attrs),
    )


def infer_reshared_logical_type(
    physical_type: IntermediateType,
    request: DataflowResharedTransportRequest,
) -> IntermediateType:
    if request.logical_output_arity % request.physical_output_arity:
        raise ValueError("reshared logical_output_arity must be divisible by physical_output_arity when inferring a logical type")
    mapping_by_field = {mapping.field_index: mapping for mapping in request.field_mappings}
    if any(index >= len(physical_type.fields) for index in mapping_by_field):
        raise ValueError("reshared field mapping index is outside the physical type")
    logical_per_physical = request.logical_output_arity // request.physical_output_arity
    fields = []
    for field_index, physical_field in enumerate(physical_type.fields):
        mapping = mapping_by_field.get(field_index)
        if mapping is None:
            fields.append(physical_field)
            continue
        if physical_field.shape is None:
            raise ValueError("reshared field mapping requires a fixed-shape tensor field")
        shape = [int(extent) for extent in physical_field.shape]
        validate_reshared_mapping_axes(mapping, len(shape))
        value_axis = mapping.physical_value_axis
        tile_axis = mapping.physical_tile_axis
        if logical_per_physical > 1:
            if tile_axis is None:
                raise ValueError("reshared physical output with multiple logical tiles requires physical_tile_axis")
            if shape[tile_axis] != logical_per_physical:
                raise ValueError(
                    f"reshared physical tile axis does not match logical/physical arity ratio: {shape[tile_axis]} != {logical_per_physical}"
                )
        elif tile_axis is not None and shape[tile_axis] != 1:
            raise ValueError("reshared physical_tile_axis must have extent one when arities match")
        shape[value_axis] *= request.logical_output_arity
        if tile_axis is not None:
            shape.pop(tile_axis)
        fields.append(replace(physical_field, shape=tuple(shape)))
    name = f"{physical_type.name}LogicalReshared"
    python_class = type(name, (), {"__annotations__": {}})
    logical_type = IntermediateType(
        name=name,
        fields=tuple(fields),
        python_class=python_class,
        attrs={
            "inferred_reshared_field_mappings": tuple(request.field_mappings),
            "transport_request_fingerprint": request.fingerprint,
        },
    )
    setattr(python_class, DATAFLOW_INTERMEDIATE_ATTR, logical_type)
    return logical_type


def validate_reshared_mapping_axes(
    mapping: DataflowResharedFieldMapping,
    rank: int,
) -> None:
    for name, axis in (
        ("physical_value_axis", mapping.physical_value_axis),
        ("physical_tile_axis", mapping.physical_tile_axis),
    ):
        if axis is not None and axis >= rank:
            raise ValueError(f"reshared {name} is outside tensor rank {rank}")
