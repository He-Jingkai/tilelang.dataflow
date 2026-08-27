"""Proof-of-concept lowering from Dataflow Body IR to TVM PrimFuncs."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, replace
import collections.abc
import inspect
import linecache
import math
import re
import textwrap
from typing import Any, get_args, get_origin
from collections.abc import Mapping

import tvm
from tvm import tir

import tilelang.language as T
from tilelang import _ffi_api
from tilelang.utils.target_capabilities import TargetCapabilitySnapshot

from .body_ir import (
    AssignIR,
    AugAssignIR,
    BinaryOpIR,
    CallIR,
    CastIR,
    ExprIR,
    FieldElementAccessIR,
    FieldAccessIR,
    LiteralIR,
    DataflowBodyIR,
    DataflowLoopIR,
    ScalarVarIR,
    SliceIR,
    TensorLoadIR,
    TensorStoreIR,
    TupleExprIR,
    lower_operator_call_to_body_ir,
)
from .cuda_contract import (
    DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES,
)
from .dtype_registry import dataflow_dtype_info, normalize_dtype_name
from .handler_identity import (
    REDUCE_ARITY_BINARY,
    REDUCE_ARITY_GENERIC,
    REDUCE_ARITY_PASSTHROUGH,
    DataflowHandlerIdentity,
    DataflowHandlerRegistry,
    DataflowHandlerVariantKey,
    build_handler_registry,
)
from .gemm_lowering import (
    GemmLoweringResolution,
    specialize_primfunc_gemm_lowerings,
)
from .handoff_planning import (
    DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR,
    DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_STAGE_ATTR,
    DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_TRANSFER_ATTR,
    DATAFLOW_CROSS_HANDLER_HANDOFF_TRANSFER_INDEX_ATTR,
    DataflowCrossHandlerHandoffPlan,
    bind_cross_handler_handoff_plan,
)
from .handler_codegen import (
    DATAFLOW_HANDLER_PARAM_SCHEMA_VERSION,
    DataflowHandlerCodegenArtifact,
    DataflowHandlerCodegenModule,
    DataflowHandlerParam,
    DataflowHandlerParamRole,
    dataflow_buffer_cuda_c_type,
    dataflow_scalar_cuda_c_type,
)
from .range_coarsening import (
    DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION,
)
from .ir import (
    IntermediateType,
    DataflowReducerContract,
    OperatorCall,
    get_intermediate_type,
)
from .implementation_registry import dataflow_implementation_registry
from .operation_contracts import (
    DATAFLOW_LAYOUT_CONTRACTS_ATTR,
    DATAFLOW_LAYOUT_LINEAR,
    DATAFLOW_LAYOUT_MATRIX_SWIZZLE,
    DATAFLOW_PIPELINE_CONTRACT_ATTR,
    DATAFLOW_TRANSPORT_STREAMED,
    DataflowPipelineRequest,
    DataflowTensorLayoutRequest,
    layout_implementation_id,
)
from .pipeline_planning import (
    DataflowPipelinePlan,
    bind_pipeline_dataflow_plan,
)
from .physical_contract import (
    DATAFLOW_INPUT_SLOTS_CONTIGUOUS,
    DATAFLOW_OUTPUT_SLOT_DIRECT,
    operator_physical_contract,
)
from .program import DataflowProgram, DataflowStage, DataflowStageKind
from .reshared_transport import (
    DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION,
    DataflowResharedTransportPlan,
    bind_reshared_transport_plan,
)
from .scheduler import InstructionPlan, DataflowOpcode
from .tensor_args import DataflowTensorArgPlan
from .tma_descriptors import DataflowTMADescriptorSpec
from .wrapper import DataflowHandlerSpec, DataflowWrapperSpec


class DataflowPrimFuncLoweringError(ValueError):
    """Raised when Dataflow Body IR cannot be lowered to a PrimFunc."""


@dataclass(frozen=True)
class DataflowPrimFuncHandler:
    handler_id: int
    operator_kind: str
    operator_name: str
    global_symbol: str
    device_symbol: str
    prim_func: tir.PrimFunc
    handler_identity: DataflowHandlerIdentity
    handler_variant_key: DataflowHandlerVariantKey
    params: tuple[DataflowHandlerParam, ...] = ()
    thread_count: int = 1
    task_param_names: tuple[str, ...] = ()
    task_param_dtypes: tuple[str, ...] = ()
    dynamic_shared_bytes: int = 0
    contiguous_map_input_count: int = 0
    reduce_input_slot_count: int = 0
    reducer_contract: DataflowReducerContract | None = None
    gemm_lowerings: tuple[GemmLoweringResolution, ...] = ()
    pipeline_lowerings: tuple[dict[str, Any], ...] = ()
    pipeline_dataflow_plan: DataflowPipelinePlan | None = None
    reshared_transport_plan: DataflowResharedTransportPlan | None = None
    cross_handler_handoff_plan: DataflowCrossHandlerHandoffPlan | None = None
    cross_handler_handoff_role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "handler_id": self.handler_id,
            "operator_kind": self.operator_kind,
            "operator_name": self.operator_name,
            "identity": self.handler_identity.to_dict(),
            "variant_key": self.handler_variant_key.to_dict(),
            "global_symbol": self.global_symbol,
            "device_symbol": self.device_symbol,
            "params": [param.to_dict() for param in self.params],
            "thread_count": self.thread_count,
            "dynamic_shared_bytes": self.dynamic_shared_bytes,
            "contiguous_map_input_count": self.contiguous_map_input_count,
            "reduce_input_slot_count": self.reduce_input_slot_count,
            "reducer_contract": (None if self.reducer_contract is None else self.reducer_contract.value),
            "gemm_lowerings": [lowering.to_dict() for lowering in self.gemm_lowerings],
            "pipeline_lowerings": [dict(lowering) for lowering in self.pipeline_lowerings],
            "pipeline_dataflow_plan": (None if self.pipeline_dataflow_plan is None else self.pipeline_dataflow_plan.to_dict()),
            "reshared_transport_plan": (None if self.reshared_transport_plan is None else self.reshared_transport_plan.to_dict()),
            "cross_handler_handoff_plan": (None if self.cross_handler_handoff_plan is None else self.cross_handler_handoff_plan.to_dict()),
            "cross_handler_handoff_role": self.cross_handler_handoff_role,
        }


@dataclass(frozen=True)
class DataflowPrimFuncLoweringResult:
    handlers: tuple[DataflowPrimFuncHandler, ...]
    ir_module: tvm.IRModule
    cuda_source: str = ""
    dynamic_shared_bytes: int = 0
    tma_descriptors: tuple[DataflowTMADescriptorSpec, ...] = ()
    codegen_artifacts: tuple[DataflowHandlerCodegenArtifact, ...] = ()
    target_fingerprint: str | None = None

    @property
    def max_thread_count(self) -> int:
        return max((handler.thread_count for handler in self.handlers), default=1)

    def codegen_artifact(self, handler_id: int) -> DataflowHandlerCodegenArtifact:
        for artifact in self.codegen_artifacts:
            if artifact.handler_id == handler_id:
                return artifact
        raise KeyError(f"Dataflow PrimFunc lowering has no codegen artifact for handler {handler_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "handlers": [handler.to_dict() for handler in self.handlers],
            "cuda_source_bytes": len(self.cuda_source.encode("utf-8")),
            "dynamic_shared_bytes": self.dynamic_shared_bytes,
            "tma_descriptors": [descriptor.to_dict() for descriptor in self.tma_descriptors],
            "codegen_artifacts": [artifact.to_dict() for artifact in self.codegen_artifacts],
            "target_fingerprint": self.target_fingerprint,
            "gemm_lowerings": [lowering.to_dict() for handler in self.handlers for lowering in handler.gemm_lowerings],
        }


@dataclass(frozen=True)
class PrimFuncField:
    name: str
    dtype: str
    shape: tuple[int, ...] | None
    numel: int


@dataclass(frozen=True)
class TaskParam:
    name: str
    dtype: str


@dataclass(frozen=True)
class ParamRoleBinding:
    role: DataflowHandlerParamRole
    tensor_arg_index: int | None = None
    task_coord_axis: int | None = None
    slot_index: int | None = None
    slot_extent: int = 1
    field_name: str | None = None


def build_logical_handler_params(
    prim_func: tir.PrimFunc,
    call: OperatorCall,
    body_ir: DataflowBodyIR | None,
    operator_kind: str,
    identity: DataflowHandlerIdentity,
    tensor_arg_plan: DataflowTensorArgPlan,
    task_params: tuple[TaskParam, ...],
    *,
    max_reduce_input_slots: int,
    max_input_slots: int,
    contiguous_map_input_count: int,
) -> tuple[DataflowHandlerParam, ...]:
    """Classify the logical PrimFunc ABI once, before CUDA text exists."""

    bindings: dict[str, ParamRoleBinding] = {}

    def bind(name: str, binding: ParamRoleBinding) -> None:
        previous = bindings.get(name)
        if previous is not None and previous != binding:
            raise DataflowPrimFuncLoweringError(
                f"handler {identity.to_dict()!r} assigns conflicting roles to parameter {name!r}: {previous!r} vs {binding!r}"
            )
        bindings[name] = binding

    for tensor_binding in tensor_arg_plan.bindings:
        if tensor_binding.handler_identity.binding_key != identity.binding_key:
            continue
        bind(
            tensor_binding.parameter_name,
            ParamRoleBinding(
                DataflowHandlerParamRole.TENSOR_ARG,
                tensor_arg_index=tensor_binding.tensor_index,
            ),
        )

    if operator_kind in {"iter", "map", "reduce"}:
        output_fields = resolve_output_fields(call)
        output_names = field_buffer_names("Out", (field.name for field in output_fields))
        for field in output_fields:
            bind(
                output_names[field.name],
                ParamRoleBinding(
                    DataflowHandlerParamRole.OUTPUT_SLOT_FIELD,
                    field_name=field.name,
                ),
            )

    if operator_kind == "map":
        input_fields = resolve_input_fields(call) if call.operator.input_types else ()
        if contiguous_map_input_count:
            input_names = field_buffer_names("Items", (field.name for field in input_fields))
            for field in input_fields:
                bind(
                    input_names[field.name],
                    ParamRoleBinding(
                        DataflowHandlerParamRole.INPUT_SLOT_FIELD,
                        slot_index=0,
                        slot_extent=contiguous_map_input_count,
                        field_name=field.name,
                    ),
                )
        else:
            for slot_index, slot_names in enumerate(map_input_buffer_names(input_fields, max_input_slots)):
                for field in input_fields:
                    bind(
                        slot_names[field.name],
                        ParamRoleBinding(
                            DataflowHandlerParamRole.INPUT_SLOT_FIELD,
                            slot_index=slot_index,
                            field_name=field.name,
                        ),
                    )
    elif operator_kind == "reduce":
        fields = resolve_output_fields(call)
        use_direct_slots = call.operator.reducer_contract is DataflowReducerContract.ASSOCIATIVE_BINARY or max_reduce_input_slots > 1
        if use_direct_slots:
            for slot_index in range(max_reduce_input_slots):
                input_names = field_buffer_names(
                    f"Item{slot_index}",
                    (field.name for field in fields),
                )
                for field in fields:
                    bind(
                        input_names[field.name],
                        ParamRoleBinding(
                            DataflowHandlerParamRole.INPUT_SLOT_FIELD,
                            slot_index=slot_index,
                            field_name=field.name,
                        ),
                    )
        else:
            input_names = field_buffer_names("Items", (field.name for field in fields))
            for field in fields:
                bind(
                    input_names[field.name],
                    ParamRoleBinding(
                        DataflowHandlerParamRole.INPUT_SLOT_FIELD,
                        slot_index=0,
                        field_name=field.name,
                    ),
                )
    elif operator_kind == "finalize":
        fields = resolve_input_fields(call)
        input_count = len(call.operator.input_types)
        for slot_index in range(input_count):
            input_names = field_buffer_names(
                "Inter" if input_count == 1 else f"Item{slot_index}",
                (field.name for field in fields),
            )
            for field in fields:
                bind(
                    input_names[field.name],
                    ParamRoleBinding(
                        DataflowHandlerParamRole.INPUT_SLOT_FIELD,
                        slot_index=slot_index,
                        field_name=field.name,
                    ),
                )

    for axis, task_param in enumerate(task_params):
        bind(
            task_param.name,
            ParamRoleBinding(
                DataflowHandlerParamRole.TASK_COORD,
                task_coord_axis=axis,
            ),
        )
    bind("range_begin", ParamRoleBinding(DataflowHandlerParamRole.RANGE_BEGIN))
    bind("range_end", ParamRoleBinding(DataflowHandlerParamRole.RANGE_END))
    bind("input_count", ParamRoleBinding(DataflowHandlerParamRole.INPUT_COUNT))
    bind("task_id", ParamRoleBinding(DataflowHandlerParamRole.TASK_ID))
    bind(
        "dataflow_handoff_stage_count",
        ParamRoleBinding(DataflowHandlerParamRole.HANDOFF_STAGE_COUNT),
    )

    scalar_aliases = raw_scalar_param_aliases(body_ir)
    params: list[DataflowHandlerParam] = []
    for ordinal, parameter in enumerate(prim_func.params):
        buffer = prim_func.buffer_map.get(parameter)
        name = str(buffer.name) if buffer is not None else str(parameter)
        binding = bindings.get(name)
        if binding is None and buffer is None:
            alias_source = scalar_aliases.get(name)
            visited = {name}
            while alias_source is not None and alias_source not in visited:
                visited.add(alias_source)
                binding = bindings.get(alias_source)
                if binding is not None:
                    break
                alias_source = scalar_aliases.get(alias_source)
        next_task_coord = re.fullmatch(r"dataflow_next_task_coord_(\d+)", name)
        if binding is None and next_task_coord is not None:
            binding = ParamRoleBinding(
                DataflowHandlerParamRole.NEXT_TASK_COORD,
                task_coord_axis=int(next_task_coord.group(1)),
            )
        if binding is None:
            kind = "buffer" if buffer is not None else "scalar"
            raise DataflowPrimFuncLoweringError(
                f"handler {identity.to_dict()!r} has unclassified {kind} parameter {name!r}; "
                "declare a structured Dataflow handler parameter role before codegen"
            )
        if buffer is None and binding.role in {
            DataflowHandlerParamRole.TENSOR_ARG,
            DataflowHandlerParamRole.INPUT_SLOT_FIELD,
            DataflowHandlerParamRole.OUTPUT_SLOT_FIELD,
        }:
            raise DataflowPrimFuncLoweringError(f"handler parameter {name!r} with role {binding.role.value!r} must be a buffer")
        if buffer is not None and binding.role not in {
            DataflowHandlerParamRole.TENSOR_ARG,
            DataflowHandlerParamRole.INPUT_SLOT_FIELD,
            DataflowHandlerParamRole.OUTPUT_SLOT_FIELD,
        }:
            raise DataflowPrimFuncLoweringError(f"handler parameter {name!r} with role {binding.role.value!r} must be a scalar")
        dtype = str(buffer.dtype) if buffer is not None else str(parameter.dtype)
        params.append(
            DataflowHandlerParam(
                ordinal=ordinal,
                name=name,
                role=binding.role,
                dtype=dtype,
                c_type=(dataflow_buffer_cuda_c_type(dtype) if buffer is not None else dataflow_scalar_cuda_c_type(dtype)),
                is_pointer=buffer is not None,
                is_const=(prim_func_buffer_is_read_only(prim_func, buffer) if buffer is not None else False),
                tensor_arg_index=binding.tensor_arg_index,
                task_coord_axis=binding.task_coord_axis,
                slot_index=binding.slot_index,
                slot_extent=binding.slot_extent,
                field_name=binding.field_name,
            )
        )
    return tuple(params)


def raw_scalar_param_aliases(body_ir: DataflowBodyIR | None) -> dict[str, str]:
    if body_ir is None or not body_ir.tilelang_body:
        return {}
    try:
        module = ast.parse("\n".join(body_ir.tilelang_body))
    except SyntaxError as err:
        raise DataflowPrimFuncLoweringError(f"could not parse scalar aliases for operator {body_ir.operator_name!r}") from err
    aliases: dict[str, str] = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and isinstance(statement.value, ast.Name):
            aliases[target.id] = statement.value.id
    return aliases


def prim_func_buffer_is_read_only(prim_func: tir.PrimFunc, buffer: tir.Buffer) -> bool:
    written = False

    def visit(node: Any) -> None:
        nonlocal written
        if isinstance(node, tir.BufferStore) and node.buffer.data.same_as(buffer.data):
            written = True

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    return not written


def attach_handler_param_roles(
    prim_func: tir.PrimFunc,
    params: tuple[DataflowHandlerParam, ...],
) -> tir.PrimFunc:
    roles = tvm.runtime.convert([param.role.value for param in params])
    return prim_func.with_attr(
        "tl.dataflow_param_schema_version",
        DATAFLOW_HANDLER_PARAM_SCHEMA_VERSION,
    ).with_attr("tl.dataflow_param_roles", roles)


def prim_func_has_multicast_copy(prim_func: tir.PrimFunc) -> bool:
    found = False

    def visit(node: Any) -> None:
        nonlocal found
        if found or not (isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy"):
            return
        cluster_mask = node.annotations.get("cluster_mask")
        if cluster_mask is not None and int(cluster_mask) > 0:
            found = True

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    return found


def lower_program_handlers_to_primfuncs(
    program: DataflowProgram,
    wrapper_spec: DataflowWrapperSpec,
    tensor_arg_plan: DataflowTensorArgPlan,
    *,
    plan: InstructionPlan,
    target: str = "cuda",
    target_capabilities: TargetCapabilitySnapshot | None = None,
    lower_to_cuda: bool = True,
    allow_reduce_direct_output: bool = True,
    allow_reduce_inplace_direct_output: bool = False,
    allow_pair_reduce_direct_output: bool | None = None,
    iter_range_bucket_size: int | None = None,
    pass_configs: dict[str, Any] | None = None,
    thread_count_overrides: Mapping[int, int] | None = None,
    map_input_scope: str | None = None,
    precision_overrides: Mapping[tuple[int, str], Mapping[str, Any]] | None = None,
) -> DataflowPrimFuncLoweringResult:
    """Lower the current first-stage Dataflow handler subset to an IRModule."""

    if not isinstance(program, DataflowProgram):
        raise TypeError(f"lower_program_handlers_to_primfuncs expects DataflowProgram, got {program!r}")
    if not isinstance(wrapper_spec, DataflowWrapperSpec):
        raise TypeError(f"lower_program_handlers_to_primfuncs expects DataflowWrapperSpec, got {wrapper_spec!r}")
    if not isinstance(tensor_arg_plan, DataflowTensorArgPlan):
        raise TypeError(f"lower_program_handlers_to_primfuncs expects DataflowTensorArgPlan, got {tensor_arg_plan!r}")
    if not isinstance(plan, InstructionPlan):
        raise TypeError(f"lower_program_handlers_to_primfuncs expects InstructionPlan, got {plan!r}")
    if not isinstance(allow_reduce_direct_output, bool):
        raise TypeError(f"allow_reduce_direct_output must be a bool, got {allow_reduce_direct_output!r}")
    if not isinstance(allow_reduce_inplace_direct_output, bool):
        raise TypeError(f"allow_reduce_inplace_direct_output must be a bool, got {allow_reduce_inplace_direct_output!r}")
    if allow_pair_reduce_direct_output is not None:
        if not isinstance(allow_pair_reduce_direct_output, bool):
            raise TypeError(f"allow_pair_reduce_direct_output must be a bool when provided, got {allow_pair_reduce_direct_output!r}")
        allow_reduce_direct_output = allow_pair_reduce_direct_output
    allow_reduce_direct_output = allow_reduce_direct_output and (
        allow_reduce_inplace_direct_output or not plan_reduce_input_output_storage_may_alias(plan)
    )
    if map_input_scope not in {None, "shared", "shared.dyn"}:
        raise ValueError(f"map_input_scope must be None, 'shared', or 'shared.dyn', got {map_input_scope!r}")
    normalized_thread_count_overrides: dict[int, int] = {}
    for raw_handler_id, raw_thread_count in (thread_count_overrides or {}).items():
        handler_id = int(raw_handler_id)
        thread_count = int(raw_thread_count)
        if handler_id < 0 or thread_count <= 0:
            raise ValueError(
                "thread_count_overrides requires non-negative handler ids and "
                f"positive thread counts, got {raw_handler_id!r}: "
                f"{raw_thread_count!r}"
            )
        normalized_thread_count_overrides[handler_id] = thread_count
    if target_capabilities is not None:
        if target != "cuda" and target != target_capabilities.target:
            raise ValueError(f"target conflicts with target_capabilities: {target!r} != {target_capabilities.target!r}")
        target = target_capabilities.target

    effective_pass_configs = dict(pass_configs or {})
    if target_capabilities is not None and target_capabilities.max_dynamic_shared_memory is not None:
        effective_pass_configs.setdefault(
            "tl.logical_gemm_max_shared_memory_bytes",
            int(target_capabilities.max_dynamic_shared_memory),
        )

    program.validate()
    if not plan.task_range_lengths:
        raise DataflowPrimFuncLoweringError("Dataflow PrimFunc lowering requires at least one task range length")
    range_extent = range_extent_from_plan(plan)
    max_reduce_input_slots = compute_max_reduce_input_slots(plan)
    max_reduce_input_slots_by_variant = compute_max_reduce_input_slots_by_variant(plan)
    max_input_slots_by_handler = compute_max_input_slots_by_handler(plan)

    handler_registry = build_handler_registry(program)
    normalized_precision_overrides = {
        (int(operator_id), str(operator_kind)): dict(constants)
        for (operator_id, operator_kind), constants in (precision_overrides or {}).items()
    }
    unknown_precision_bindings = normalized_precision_overrides.keys() - handler_registry.bindings_by_key.keys()
    if unknown_precision_bindings:
        raise DataflowPrimFuncLoweringError(
            f"precision overrides reference unknown Dataflow handler identities: {sorted(unknown_precision_bindings)!r}"
        )

    lowered_handlers, ir_module = build_primfunc_handlers(
        program,
        wrapper_spec,
        tensor_arg_plan,
        range_extent,
        plan.task_extents,
        max_reduce_input_slots,
        max_reduce_input_slots_by_variant,
        handler_registry,
        max_input_slots_by_handler,
        target_capabilities=target_capabilities,
        thread_count_overrides=normalized_thread_count_overrides,
        pass_configs=effective_pass_configs,
        allow_reduce_direct_output=allow_reduce_direct_output,
        iter_range_bucket_size=iter_range_bucket_size,
        map_input_scope=map_input_scope,
        precision_overrides=normalized_precision_overrides,
        range_coarsening_plans=plan.range_coarsening_plans,
        reshared_transport_plans=plan.reshared_transport_plans,
        cross_handler_handoff_plans=plan.cross_handler_handoff_plans,
    )

    cuda_source = ""
    dynamic_shared_bytes = 0
    tma_descriptors: tuple[DataflowTMADescriptorSpec, ...] = ()
    codegen_artifacts: tuple[DataflowHandlerCodegenArtifact, ...] = ()
    if lower_to_cuda:
        (
            cuda_source,
            dynamic_shared_bytes,
            tma_descriptors,
            handler_dynamic_shared_bytes,
            codegen_module,
        ) = lower_dataflow_primfunc_module_to_cuda(
            ir_module,
            target,
            tuple(lowered_handlers),
            pass_configs=effective_pass_configs,
        )
        lowered_handlers, codegen_artifacts = materialize_handler_codegen_artifacts(
            tuple(lowered_handlers),
            codegen_module,
            tma_descriptors,
            handler_dynamic_shared_bytes,
            target_fingerprint=(None if target_capabilities is None else target_capabilities.fingerprint),
        )
    return DataflowPrimFuncLoweringResult(
        handlers=tuple(lowered_handlers),
        ir_module=ir_module,
        cuda_source=cuda_source,
        dynamic_shared_bytes=dynamic_shared_bytes,
        tma_descriptors=tma_descriptors,
        codegen_artifacts=codegen_artifacts,
        target_fingerprint=(None if target_capabilities is None else target_capabilities.fingerprint),
    )


def attach_task_coordinate_assumptions(
    prim_func: tir.PrimFunc,
    task_params: tuple[TaskParam, ...],
    *,
    task_axes: tuple[Any, ...],
    task_extents: tuple[int, ...] | None,
) -> tir.PrimFunc:
    if not task_params or task_extents is None:
        return prim_func
    if len(task_axes) != len(task_extents):
        raise DataflowPrimFuncLoweringError(
            f"task-domain axes and scheduler extents must have the same rank: {len(task_axes)} != {len(task_extents)}"
        )

    extent_by_axis = {str(axis): int(extent) for axis, extent in zip(task_axes, task_extents)}
    scalar_params = {str(param): param for param in prim_func.params if param not in prim_func.buffer_map}
    assumptions: list[tir.Stmt] = []
    for task_param in task_params:
        try:
            extent = extent_by_axis[task_param.name]
        except KeyError as err:
            raise DataflowPrimFuncLoweringError(f"task parameter {task_param.name!r} has no task-domain extent") from err
        param = scalar_params.get(task_param.name)
        if param is None:
            # Source factories may constant-fold an unused task coordinate out
            # of the PrimFunc signature before this boundary is reached.
            continue
        condition = tir.And(param >= 0, param < extent)
        assumptions.append(tir.Evaluate(tir.assume(condition)))
    if not assumptions:
        return prim_func
    assumptions.append(prim_func.body)
    return prim_func.with_body(tir.SeqStmt(assumptions))


def build_primfunc_handlers(
    program: DataflowProgram,
    wrapper_spec: DataflowWrapperSpec,
    tensor_arg_plan: DataflowTensorArgPlan,
    range_extent: int,
    task_extents: tuple[int, ...] | None,
    max_reduce_input_slots: int,
    max_reduce_input_slots_by_variant: dict[DataflowHandlerVariantKey, int],
    handler_registry: DataflowHandlerRegistry,
    max_input_slots_by_handler: dict[DataflowHandlerIdentity, int],
    *,
    target_capabilities: TargetCapabilitySnapshot | None = None,
    thread_count_overrides: dict[int, int] | None = None,
    allow_reduce_direct_output: bool = True,
    iter_range_bucket_size: int | None = None,
    map_input_scope: str | None = None,
    pass_configs: dict[str, Any] | None = None,
    precision_overrides: Mapping[tuple[int, str], Mapping[str, Any]] | None = None,
    range_coarsening_plans: tuple[tuple[int, Any], ...] = (),
    reshared_transport_plans: tuple[tuple[int, DataflowResharedTransportPlan], ...] = (),
    cross_handler_handoff_plans: tuple[DataflowCrossHandlerHandoffPlan, ...] = (),
) -> tuple[list[DataflowPrimFuncHandler], tvm.IRModule]:
    lowered_handlers: list[DataflowPrimFuncHandler] = []
    functions: dict[str, tir.PrimFunc] = {}
    used_primfunc_symbols: set[str] = set()
    for handler in wrapper_spec.handlers:
        identity = handler.handler_identity
        variant_key = handler.handler_variant_key
        if identity is None or variant_key is None:
            raise DataflowPrimFuncLoweringError(
                f"cannot lower Dataflow handler {handler.handler_id} without a structured identity and variant key"
            )
        if variant_key.base_identity != identity:
            raise DataflowPrimFuncLoweringError(f"Dataflow handler {handler.handler_id} variant base does not match identity")
        try:
            binding = handler_registry.resolve(identity)
        except KeyError as err:
            raise DataflowPrimFuncLoweringError(f"cannot lower Dataflow handler identity {identity.to_dict()!r} to a PrimFunc") from err
        call = binding.call
        stage = binding.stage
        specialization_overrides = dict((precision_overrides or {}).get(identity.binding_key, {}))
        stage_id = getattr(stage, "stage_id", None)
        if isinstance(stage_id, int):
            selected_range_tiles = {
                range_plan.tiles_per_handler for plan_stage_id, range_plan in range_coarsening_plans if plan_stage_id == stage_id
            }
            if len(selected_range_tiles) > 1:
                raise DataflowPrimFuncLoweringError(
                    "one Dataflow stage cannot lower range plans with different "
                    f"tiles_per_handler values: stage_id={stage_id}, "
                    f"values={sorted(selected_range_tiles)!r}"
                )
            if selected_range_tiles:
                specialization_overrides[DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION] = next(iter(selected_range_tiles))
        handler_range_extent = range_extent
        fixed_iter_range_extent = None
        if handler.operator_kind == "iter":
            specialization = variant_key.iter_range
            if specialization is None:
                raise DataflowPrimFuncLoweringError(f"Dataflow ITER handler {handler.handler_id} requires range specialization")
            try:
                fixed_iter_range_extent = specialization.fixed_extent(
                    bucket_size=iter_range_bucket_size,
                )
            except ValueError as err:
                raise DataflowPrimFuncLoweringError(str(err)) from err
            if fixed_iter_range_extent is not None:
                handler_range_extent = fixed_iter_range_extent
        task_params = task_params_for_handler(
            program,
            handler.operator_kind,
            call,
            stage,
        )
        thread_count = (
            handler_thread_count(call, handler.operator_kind)
            if thread_count_overrides is None
            else thread_count_overrides.get(
                handler.handler_id,
                handler_thread_count(call, handler.operator_kind),
            )
        )
        prim_func_symbol = unique_primfunc_symbol(handler, used_primfunc_symbols)
        device_symbol = device_kernel_symbol(prim_func_symbol)
        source_factory = call.operator.attrs.get("primfunc_source_factory")
        contiguous_map_input_count = 0
        reduce_input_slot_count = 0
        if handler.operator_kind == "reduce":
            reduce_input_slot_count = count_reduce_input_slots(
                call,
                variant_key,
                max_reduce_input_slots=max_reduce_input_slots,
                max_reduce_input_slots_by_variant=max_reduce_input_slots_by_variant,
            )
        body_ir: DataflowBodyIR | None = None
        if source_factory is None:
            body_ir = lower_operator_call_to_body_ir(
                call,
                handler.operator_kind,
                specialization_constants=specialization_overrides,
            )
            contiguous_map_input_count = count_contiguous_map_inputs(
                body_ir,
                call,
                resolve_input_fields(call) if handler.operator_kind == "map" and call.operator.input_types else (),
                max_input_slots_by_handler.get(identity, 0),
            )
            prim_func = body_ir_to_prim_func(
                body_ir,
                call,
                prim_func_symbol,
                handler_range_extent,
                reduce_input_slot_count or max_reduce_input_slots,
                tensor_arg_plan,
                handler_variant_key=variant_key,
                thread_count=thread_count,
                task_params=task_params,
                max_input_slots=max_input_slots_by_handler.get(identity, 0),
                allow_reduce_direct_output=allow_reduce_direct_output,
                fixed_iter_range_extent=fixed_iter_range_extent,
                map_input_scope=map_input_scope,
                specialization_overrides=specialization_overrides,
            )
        else:
            prim_func = prim_func_from_source_factory(
                source_factory,
                call,
                stage,
                prim_func_symbol,
                handler_range_extent,
                reduce_input_slot_count or max_reduce_input_slots,
                max_input_slots_by_handler.get(identity, 0),
                tensor_arg_plan,
                task_params,
                specialization_overrides=specialization_overrides,
            )
        pipeline_request = stage.attrs.get(DATAFLOW_PIPELINE_CONTRACT_ATTR)
        reshared_transport_plan = reshared_transport_plan_for_stage(
            stage,
            reshared_transport_plans,
        )
        if reshared_transport_plan is not None:
            if (
                reshared_transport_plan.lowering_implementation_id == DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION
                and pipeline_request is None
            ):
                raise DataflowPrimFuncLoweringError("streamed producer-push transport requires a consumer pipeline contract")
            preliminary_params = build_logical_handler_params(
                prim_func,
                call,
                body_ir,
                handler.operator_kind,
                identity,
                tensor_arg_plan,
                task_params,
                max_reduce_input_slots=(reduce_input_slot_count or max_reduce_input_slots),
                max_input_slots=max_input_slots_by_handler.get(identity, 0),
                contiguous_map_input_count=contiguous_map_input_count,
            )
            source_buffer_names = tuple(
                param.name for param in preliminary_params if param.role is DataflowHandlerParamRole.INPUT_SLOT_FIELD
            )
            try:
                prim_func = bind_reshared_transport_plan(
                    prim_func,
                    reshared_transport_plan,
                    source_buffer_names=source_buffer_names,
                )
            except ValueError as err:
                raise DataflowPrimFuncLoweringError(str(err)) from err
        if wrapper_spec.cluster_size > 1 and (
            prim_func_has_multicast_copy(prim_func)
            or (reshared_transport_plan is not None and reshared_transport_plan.family == DATAFLOW_TRANSPORT_STREAMED)
        ):
            prim_func = prim_func.with_attr(
                "cluster_dims",
                tvm.runtime.convert([wrapper_spec.cluster_size, 1, 1]),
            )
        prim_func = attach_task_coordinate_assumptions(
            prim_func,
            task_params,
            task_axes=program.task_domain.axes,
            task_extents=task_extents,
        )
        pipeline_dataflow_plan = None
        if pipeline_request is not None:
            if not isinstance(pipeline_request, DataflowPipelineRequest):
                raise DataflowPrimFuncLoweringError("Dataflow stage has an unnormalized pipeline contract")
            try:
                prim_func, pipeline_dataflow_plan = bind_pipeline_dataflow_plan(
                    prim_func,
                    pipeline_request,
                )
            except ValueError as err:
                raise DataflowPrimFuncLoweringError(str(err)) from err
        handoff_plan, handoff_role = cross_handler_handoff_plan_for_stage(
            stage,
            cross_handler_handoff_plans,
        )
        if handoff_plan is not None:
            assert handoff_role is not None
            try:
                prim_func = bind_cross_handler_handoff_plan(
                    prim_func,
                    handoff_plan,
                    role=handoff_role,
                    pipeline_request=(pipeline_request if handoff_role == "consumer" else None),
                )
            except ValueError as err:
                raise DataflowPrimFuncLoweringError(str(err)) from err
        gemm_lowerings: tuple[GemmLoweringResolution, ...] = ()
        if target_capabilities is not None:
            with tvm.transform.PassContext(
                opt_level=3,
                config=pass_configs or {},
            ):
                prim_func, gemm_lowerings = specialize_primfunc_gemm_lowerings(
                    prim_func,
                    target_capabilities,
                    handler_id=handler.handler_id,
                    operator_name=handler.operator_name,
                    thread_count=thread_count,
                )
        params = build_logical_handler_params(
            prim_func,
            call,
            body_ir,
            handler.operator_kind,
            identity,
            tensor_arg_plan,
            task_params,
            max_reduce_input_slots=reduce_input_slot_count or max_reduce_input_slots,
            max_input_slots=max_input_slots_by_handler.get(identity, 0),
            contiguous_map_input_count=contiguous_map_input_count,
        )
        prim_func = attach_handler_param_roles(prim_func, params)
        lowered_handlers.append(
            DataflowPrimFuncHandler(
                handler_id=handler.handler_id,
                operator_kind=handler.operator_kind,
                operator_name=handler.operator_name,
                global_symbol=prim_func_symbol,
                device_symbol=device_symbol,
                prim_func=prim_func,
                handler_identity=identity,
                handler_variant_key=variant_key,
                params=params,
                thread_count=thread_count,
                task_param_names=tuple(param.name for param in task_params),
                task_param_dtypes=tuple(param.dtype for param in task_params),
                contiguous_map_input_count=contiguous_map_input_count,
                reduce_input_slot_count=reduce_input_slot_count,
                reducer_contract=(call.operator.reducer_contract if handler.operator_kind == "reduce" else None),
                gemm_lowerings=gemm_lowerings,
                pipeline_dataflow_plan=pipeline_dataflow_plan,
                reshared_transport_plan=reshared_transport_plan,
                cross_handler_handoff_plan=handoff_plan,
                cross_handler_handoff_role=handoff_role,
            )
        )
        functions[prim_func_symbol] = prim_func
    lowered_handlers, functions = materialize_cross_handler_handoffs(
        lowered_handlers,
        functions,
    )
    return lowered_handlers, tvm.IRModule(functions)


def reshared_transport_plan_for_stage(
    stage: DataflowStage | Any,
    reshared_transport_plans: tuple[tuple[int, DataflowResharedTransportPlan], ...],
) -> DataflowResharedTransportPlan | None:
    if not isinstance(stage, DataflowStage):
        return None
    upstream_stage_ids = set(stage.deps)
    matches = tuple(
        transport_plan for reshared_stage_id, transport_plan in reshared_transport_plans if reshared_stage_id in upstream_stage_ids
    )
    if len(matches) > 1:
        raise DataflowPrimFuncLoweringError(f"handler stage {stage.stage_id} has multiple reshared transport inputs")
    return None if not matches else matches[0]


def cross_handler_handoff_plan_for_stage(
    stage: DataflowStage | Any,
    plans: tuple[DataflowCrossHandlerHandoffPlan, ...],
) -> tuple[DataflowCrossHandlerHandoffPlan | None, str | None]:
    if not isinstance(stage, DataflowStage):
        return None, None
    matches = tuple(
        (plan, role)
        for plan in plans
        for role, stage_id in (
            ("producer", plan.producer_stage_id),
            ("consumer", plan.consumer_stage_id),
        )
        if stage.stage_id == stage_id
    )
    if len(matches) > 1:
        raise DataflowPrimFuncLoweringError(f"handler stage {stage.stage_id} has multiple handoff roles")
    return (None, None) if not matches else matches[0]


@dataclass(frozen=True)
class HandoffCopyContext:
    call: tir.Call
    pipeline_loop: tir.For
    let_bindings: tuple[tuple[tir.Var, tir.PrimExpr], ...]
    enclosing_loops: tuple[tir.For, ...]
    inner_loops: tuple[tir.For, ...]
    guards: tuple[tir.PrimExpr, ...]


def materialize_cross_handler_handoffs(
    handlers: list[DataflowPrimFuncHandler],
    functions: dict[str, tir.PrimFunc],
) -> tuple[list[DataflowPrimFuncHandler], dict[str, tir.PrimFunc]]:
    """Materialize compiler-owned producer prefixes and shared arena parameters."""

    grouped: dict[
        str,
        dict[str, list[int]],
    ] = {}
    plans: dict[str, DataflowCrossHandlerHandoffPlan] = {}
    for index, handler in enumerate(handlers):
        plan = handler.cross_handler_handoff_plan
        role = handler.cross_handler_handoff_role
        if plan is None or role is None or not plan.enabled:
            continue
        plans[plan.fingerprint] = plan
        grouped.setdefault(plan.fingerprint, {"producer": [], "consumer": []})[role].append(index)

    for fingerprint, roles in grouped.items():
        plan = plans[fingerprint]
        if not roles["producer"] or not roles["consumer"]:
            raise DataflowPrimFuncLoweringError("enabled cross-handler handoff requires producer and consumer PrimFuncs")
        consumer_source_func: tir.PrimFunc | None = None
        original_consumer_params: tuple[DataflowHandlerParam, ...] | None = None
        contexts: tuple[HandoffCopyContext, ...] | None = None
        consumer_transfer_buffers: tuple[tir.Buffer, ...] | None = None
        for consumer_index in roles["consumer"]:
            consumer = handlers[consumer_index]
            variant_source_func = functions[consumer.global_symbol]
            variant_contexts = collect_handoff_copy_contexts(
                variant_source_func,
                plan,
            )
            variant_original_params = consumer.params
            if consumer_source_func is None:
                consumer_source_func = variant_source_func
                original_consumer_params = variant_original_params
                contexts = variant_contexts
            elif not handoff_consumer_variants_compatible(
                original_consumer_params,
                contexts,
                variant_original_params,
                variant_contexts,
            ):
                raise DataflowPrimFuncLoweringError(
                    f"cross-handler handoff consumer variants have incompatible physical prefixes; plan={fingerprint}"
                )

            (
                consumer_func,
                consumer_params,
                variant_transfer_buffers,
            ) = promote_handoff_consumer_buffers(
                variant_source_func,
                variant_original_params,
                plan,
                variant_contexts,
            )
            if consumer_transfer_buffers is None:
                consumer_transfer_buffers = variant_transfer_buffers
            consumer_func, consumer_params, _ = ensure_handoff_scalar_param(
                consumer_func,
                consumer_params,
                name="dataflow_handoff_stage_count",
                dtype="uint32",
                role=DataflowHandlerParamRole.HANDOFF_STAGE_COUNT,
            )
            consumer_func = attach_handler_param_roles(
                consumer_func,
                consumer_params,
            )
            consumer = replace(
                consumer,
                prim_func=consumer_func,
                params=consumer_params,
            )
            handlers[consumer_index] = consumer
            functions[consumer.global_symbol] = consumer_func

        assert consumer_source_func is not None
        assert original_consumer_params is not None
        assert contexts is not None
        assert consumer_transfer_buffers is not None

        for producer_index in roles["producer"]:
            producer = handlers[producer_index]
            producer_func = functions[producer.global_symbol]
            producer_func, producer_params = materialize_handoff_producer_prefix(
                producer_func,
                producer.params,
                consumer_func=consumer_source_func,
                consumer_params=original_consumer_params,
                consumer_transfer_buffers=consumer_transfer_buffers,
                contexts=contexts,
                plan=plan,
            )
            producer_func = attach_handler_param_roles(
                producer_func,
                producer_params,
            )
            producer = replace(
                producer,
                prim_func=producer_func,
                params=producer_params,
            )
            handlers[producer_index] = producer
            functions[producer.global_symbol] = producer_func

    return handlers, functions


def handoff_consumer_variants_compatible(
    left_params: tuple[DataflowHandlerParam, ...],
    left_contexts: tuple[HandoffCopyContext, ...],
    right_params: tuple[DataflowHandlerParam, ...],
    right_contexts: tuple[HandoffCopyContext, ...],
) -> bool:
    if tuple(param.to_dict() for param in left_params) != tuple(param.to_dict() for param in right_params):
        return False
    if len(left_contexts) != len(right_contexts):
        return False

    def structurally_equal(left: Any, right: Any) -> bool:
        return bool(tvm.ir.structural_equal(left, right, map_free_vars=True))

    for left, right in zip(left_contexts, right_contexts):
        if not structurally_equal(left.pipeline_loop, right.pipeline_loop):
            return False
        for left_values, right_values in (
            (left.let_bindings, right.let_bindings),
            (left.enclosing_loops, right.enclosing_loops),
            (left.inner_loops, right.inner_loops),
            (left.guards, right.guards),
        ):
            if len(left_values) != len(right_values):
                return False
            if any(not structurally_equal(left_value, right_value) for left_value, right_value in zip(left_values, right_values)):
                return False
    return True


def collect_handoff_copy_contexts(
    prim_func: tir.PrimFunc,
    plan: DataflowCrossHandlerHandoffPlan,
) -> tuple[HandoffCopyContext, ...]:
    calls: dict[int, tir.Call] = {}

    def collect(node: Any) -> None:
        if not (isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy"):
            return
        transfer_index = node.annotations.get(DATAFLOW_CROSS_HANDLER_HANDOFF_TRANSFER_INDEX_ATTR)
        fingerprint = node.annotations.get(DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR)
        if transfer_index is None or fingerprint is None:
            return
        fingerprint_value = getattr(fingerprint, "value", str(fingerprint))
        if fingerprint_value != plan.fingerprint:
            raise DataflowPrimFuncLoweringError(
                f"handoff copy references a different typed plan: got={fingerprint_value!r}, expected={plan.fingerprint!r}"
            )
        index = int(transfer_index)
        if index in calls:
            raise DataflowPrimFuncLoweringError(f"handoff transfer {index} has multiple physical copy statements")
        calls[index] = node

    tir.stmt_functor.post_order_visit(prim_func.body, collect)
    expected = set(range(len(plan.transfer_plans)))
    if set(calls) != expected:
        raise DataflowPrimFuncLoweringError(
            f"consumer handoff copies do not cover the typed transfer plan: got={sorted(calls)}, expected={sorted(expected)}"
        )
    return tuple(find_handoff_copy_context(prim_func.body, calls[index], plan) for index in range(len(plan.transfer_plans)))


def find_handoff_copy_context(
    body: tir.Stmt,
    target: tir.Call,
    plan: DataflowCrossHandlerHandoffPlan,
) -> HandoffCopyContext:
    result: HandoffCopyContext | None = None

    def contains_target(expr: tir.PrimExpr) -> bool:
        found = False

        def visit(node: Any) -> None:
            nonlocal found
            if isinstance(node, tir.Call) and node.same_as(target):
                found = True

        tir.stmt_functor.post_order_visit(expr, visit)
        return found

    def walk(
        stmt: tir.Stmt,
        lets: tuple[tuple[tir.Var, tir.PrimExpr], ...],
        loops: tuple[tir.For, ...],
        guards: tuple[tir.PrimExpr, ...],
    ) -> None:
        nonlocal result
        if result is not None:
            return
        if isinstance(stmt, tir.SeqStmt):
            for child in stmt.seq:
                walk(child, lets, loops, guards)
            return
        if isinstance(stmt, tir.LetStmt):
            walk(stmt.body, (*lets, (stmt.var, stmt.value)), loops, guards)
            return
        if isinstance(stmt, tir.For):
            walk(stmt.body, lets, (*loops, stmt), guards)
            return
        if isinstance(stmt, tir.IfThenElse):
            walk(stmt.then_case, lets, loops, (*guards, stmt.condition))
            if stmt.else_case is not None:
                walk(
                    stmt.else_case,
                    lets,
                    loops,
                    (*guards, tir.Not(stmt.condition)),
                )
            return
        if isinstance(stmt, tir.AttrStmt):
            walk(stmt.body, lets, loops, guards)
            return
        if isinstance(stmt, tir.BlockRealize):
            walk(stmt.block.body, lets, loops, guards)
            return
        if isinstance(stmt, tir.Block):
            walk(stmt.body, lets, loops, guards)
            return
        if not isinstance(stmt, tir.Evaluate) or not contains_target(stmt.value):
            return
        pipeline_loops = tuple(
            loop
            for loop in loops
            if getattr(
                loop.annotations.get(
                    DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR,
                    "",
                ),
                "value",
                loop.annotations.get(
                    DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR,
                    "",
                ),
            )
            == plan.fingerprint
        )
        if len(pipeline_loops) != 1:
            raise DataflowPrimFuncLoweringError("handoff copy must be nested in exactly one typed consumer pipeline")
        pipeline_loop = pipeline_loops[0]
        pipeline_index = next(index for index, loop in enumerate(loops) if loop.same_as(pipeline_loop))
        result = HandoffCopyContext(
            call=target,
            pipeline_loop=pipeline_loop,
            let_bindings=lets,
            enclosing_loops=loops[:pipeline_index],
            inner_loops=loops[pipeline_index + 1 :],
            guards=guards,
        )

    walk(body, (), (), ())
    if result is None:
        raise DataflowPrimFuncLoweringError("could not recover the structured path to a handoff transfer")
    return result


def promote_handoff_consumer_buffers(
    prim_func: tir.PrimFunc,
    params: tuple[DataflowHandlerParam, ...],
    plan: DataflowCrossHandlerHandoffPlan,
    contexts: tuple[HandoffCopyContext, ...],
) -> tuple[
    tir.PrimFunc,
    tuple[DataflowHandlerParam, ...],
    tuple[tir.Buffer, ...],
]:
    old_buffers = tuple(_ffi_api.ParseOperator(context.call).dst for context in contexts)
    if len({buffer.data for buffer in old_buffers}) != len(old_buffers):
        raise DataflowPrimFuncLoweringError("handoff transfers must target distinct pipeline buffers")

    params_list = list(prim_func.params)
    buffer_map = dict(prim_func.buffer_map)
    logical_params = list(params)
    substitution: dict[tir.Var, tir.PrimExpr] = {}
    promoted: list[tir.Buffer] = []
    removed_data = {buffer.data for buffer in old_buffers}
    body = remove_allocated_buffers(prim_func.body, removed_data)
    for transfer, old_buffer in zip(plan.transfer_plans, old_buffers):
        handle = tir.Var(
            f"dataflow_handoff_transfer_{transfer.binding_index}_handle",
            "handle",
        )
        data = tir.Var(
            f"dataflow_handoff_transfer_{transfer.binding_index}",
            old_buffer.data.type_annotation,
        )
        promoted_buffer = clone_buffer(
            old_buffer,
            data=data,
            name=f"dataflow_handoff_transfer_{transfer.binding_index}",
        )
        params_list.append(handle)
        buffer_map[handle] = promoted_buffer
        substitution[old_buffer.data] = promoted_buffer.data
        promoted.append(promoted_buffer)
        logical_params.append(
            DataflowHandlerParam(
                ordinal=len(logical_params),
                name=promoted_buffer.name,
                role=DataflowHandlerParamRole.HANDOFF_TRANSFER_BUFFER,
                dtype=str(promoted_buffer.dtype),
                c_type=dataflow_buffer_cuda_c_type(str(promoted_buffer.dtype)),
                is_pointer=True,
                is_const=False,
                handoff_transfer_index=transfer.binding_index,
            )
        )
    body = tir.stmt_functor.substitute(body, substitution)
    return (
        tir.PrimFunc(
            params_list,
            body,
            prim_func.ret_type,
            buffer_map,
            prim_func.attrs,
            prim_func.span,
        ),
        tuple(logical_params),
        tuple(promoted),
    )


def materialize_handoff_producer_prefix(
    prim_func: tir.PrimFunc,
    params: tuple[DataflowHandlerParam, ...],
    *,
    consumer_func: tir.PrimFunc,
    consumer_params: tuple[DataflowHandlerParam, ...],
    consumer_transfer_buffers: tuple[tir.Buffer, ...],
    contexts: tuple[HandoffCopyContext, ...],
    plan: DataflowCrossHandlerHandoffPlan,
) -> tuple[tir.PrimFunc, tuple[DataflowHandlerParam, ...]]:
    pipeline_loops: list[tir.For] = []

    def collect_pipeline(node: Any) -> None:
        if isinstance(node, tir.For) and "tl.pipeline_dataflow_mode" in node.annotations:
            pipeline_loops.append(node)

    tir.stmt_functor.post_order_visit(prim_func.body, collect_pipeline)
    if len(pipeline_loops) != 1:
        raise DataflowPrimFuncLoweringError("handoff producer requires exactly one typed physical pipeline")
    producer_loop = pipeline_loops[0]
    (
        producer_enclosing_loops,
        producer_let_bindings,
        producer_guards,
    ) = loop_path_context(
        prim_func.body,
        producer_loop,
    )
    producer_func, producer_params, stage_count_var = ensure_handoff_scalar_param(
        prim_func,
        params,
        name="dataflow_handoff_stage_count",
        dtype="uint32",
        role=DataflowHandlerParamRole.HANDOFF_STAGE_COUNT,
    )
    param_substitution: dict[tir.Var, tir.PrimExpr] = {}
    handler_param_bindings(producer_func, producer_params)
    consumer_param_by_role = handler_param_bindings(consumer_func, consumer_params)
    referenced_vars: list[tir.Var] = []

    def collect_referenced_var(node: Any) -> None:
        candidates: tuple[tir.Var, ...] = ()
        if isinstance(node, tir.Var):
            candidates = (node,)
        elif isinstance(node, (tir.BufferLoad, tir.BufferStore)):
            candidates = (node.buffer.data,)
        for candidate in candidates:
            if not any(candidate.same_as(item) for item in referenced_vars):
                referenced_vars.append(candidate)

    for context in contexts:
        tir.stmt_functor.post_order_visit(context.call, collect_referenced_var)
        source_buffer = _ffi_api.ParseOperator(context.call).src
        if not any(source_buffer.data.same_as(item) for item in referenced_vars):
            referenced_vars.append(source_buffer.data)
        for guard in context.guards:
            tir.stmt_functor.post_order_visit(guard, collect_referenced_var)
        for _, value in context.let_bindings:
            tir.stmt_functor.post_order_visit(value, collect_referenced_var)
        for loop in (*context.enclosing_loops, *context.inner_loops):
            tir.stmt_functor.post_order_visit(loop.min, collect_referenced_var)

    def parameter_is_referenced(var: tir.Var, buffer: tir.Buffer | None) -> bool:
        candidates = (var,) if buffer is None else (var, buffer.data)
        return any(candidate.same_as(referenced) for candidate in candidates for referenced in referenced_vars)

    for consumer_param in consumer_params:
        consumer_binding = consumer_param_by_role[consumer_param.name]
        consumer_var, consumer_buffer = consumer_binding
        if not parameter_is_referenced(consumer_var, consumer_buffer):
            continue
        if consumer_param.role is DataflowHandlerParamRole.TENSOR_ARG:
            if consumer_buffer is None:
                raise DataflowPrimFuncLoweringError("handoff consumer tensor source is not represented by a buffer")
            (
                producer_func,
                producer_params,
                producer_buffer,
            ) = ensure_handoff_tensor_param(
                producer_func,
                producer_params,
                consumer_param=consumer_param,
                consumer_buffer=consumer_buffer,
            )
            handler_param_bindings(
                producer_func,
                producer_params,
            )
            param_substitution[consumer_buffer.data] = producer_buffer.data
        elif consumer_param.role is DataflowHandlerParamRole.TASK_COORD:
            assert consumer_param.task_coord_axis is not None
            name = f"dataflow_next_task_coord_{consumer_param.task_coord_axis}"
            producer_func, producer_params, peer_var = ensure_handoff_scalar_param(
                producer_func,
                producer_params,
                name=name,
                dtype=consumer_param.dtype,
                role=DataflowHandlerParamRole.NEXT_TASK_COORD,
                task_coord_axis=consumer_param.task_coord_axis,
            )
            handler_param_bindings(
                producer_func,
                producer_params,
            )
            param_substitution[consumer_var] = peer_var
        elif consumer_param.role in {
            DataflowHandlerParamRole.RANGE_BEGIN,
            DataflowHandlerParamRole.RANGE_END,
            DataflowHandlerParamRole.TASK_ID,
        }:
            name, role = {
                DataflowHandlerParamRole.RANGE_BEGIN: (
                    "dataflow_handoff_peer_range_begin",
                    DataflowHandlerParamRole.HANDOFF_PEER_RANGE_BEGIN,
                ),
                DataflowHandlerParamRole.RANGE_END: (
                    "dataflow_handoff_peer_range_end",
                    DataflowHandlerParamRole.HANDOFF_PEER_RANGE_END,
                ),
                DataflowHandlerParamRole.TASK_ID: (
                    "dataflow_handoff_peer_task_id",
                    DataflowHandlerParamRole.HANDOFF_PEER_TASK_ID,
                ),
            }[consumer_param.role]
            producer_func, producer_params, peer_var = ensure_handoff_scalar_param(
                producer_func,
                producer_params,
                name=name,
                dtype=consumer_param.dtype,
                role=role,
            )
            handler_param_bindings(
                producer_func,
                producer_params,
            )
            param_substitution[consumer_var] = peer_var
        elif consumer_param.role is DataflowHandlerParamRole.INPUT_SLOT_FIELD:
            raise DataflowPrimFuncLoweringError("handoff prefix sources from peer intermediate slots are not yet legal")
        else:
            raise DataflowPrimFuncLoweringError(
                "handoff consumer prefix references an unsupported handler parameter: "
                f"name={consumer_param.name!r}, role={consumer_param.role.value!r}"
            )

    producer_params_list = list(producer_func.params)
    producer_buffer_map = dict(producer_func.buffer_map)
    producer_logical_params = list(producer_params)
    producer_transfer_buffers: list[tuple[tir.Buffer, ...]] = []
    for transfer, consumer_buffer in zip(
        plan.transfer_plans,
        consumer_transfer_buffers,
    ):
        stage_buffers: list[tir.Buffer] = []
        for stage in range(transfer.buffer_stages):
            name = f"dataflow_handoff_transfer_{transfer.binding_index}_stage_{stage}"
            handle = tir.Var(f"{name}_handle", "handle")
            data = tir.Var(name, consumer_buffer.data.type_annotation)
            stage_buffer = clone_buffer(
                consumer_buffer,
                data=data,
                name=name,
            )
            producer_params_list.append(handle)
            producer_buffer_map[handle] = stage_buffer
            stage_buffers.append(stage_buffer)
            producer_logical_params.append(
                DataflowHandlerParam(
                    ordinal=len(producer_logical_params),
                    name=stage_buffer.name,
                    role=DataflowHandlerParamRole.HANDOFF_TRANSFER_BUFFER,
                    dtype=str(stage_buffer.dtype),
                    c_type=dataflow_buffer_cuda_c_type(str(stage_buffer.dtype)),
                    is_pointer=True,
                    is_const=False,
                    handoff_transfer_index=transfer.binding_index,
                    handoff_stage_index=stage,
                )
            )
        producer_transfer_buffers.append(tuple(stage_buffers))
    producer_func = tir.PrimFunc(
        producer_params_list,
        producer_func.body,
        producer_func.ret_type,
        producer_buffer_map,
        producer_func.attrs,
        producer_func.span,
    )

    cloned_stmts: list[tir.Stmt] = []
    for transfer, context, destinations in zip(
        plan.transfer_plans,
        contexts,
        producer_transfer_buffers,
    ):
        for stage, destination in enumerate(destinations):
            substitution = dict(param_substitution)
            substitution[context.pipeline_loop.loop_var] = tir.IntImm(
                context.pipeline_loop.loop_var.dtype,
                stage,
            )
            for loop in context.enclosing_loops:
                extent = tir.stmt_functor.substitute(
                    loop.extent,
                    substitution,
                )
                if not isinstance(extent, tir.IntImm) or int(extent) != 1:
                    raise DataflowPrimFuncLoweringError(f"handoff consumer pipeline enclosing loops must have unit extent, got {extent}")
                substitution[loop.loop_var] = tir.stmt_functor.substitute(
                    loop.min,
                    substitution,
                )
            for var, value in context.let_bindings:
                substitution[var] = tir.stmt_functor.substitute(
                    value,
                    substitution,
                )
            rewritten_call = tir.stmt_functor.substitute(
                context.call,
                substitution,
            )
            parsed = _ffi_api.ParseOperator(rewritten_call)
            src_region = tir.BufferRegion(parsed.src, parsed.src_range)
            dst_region = tir.BufferRegion(destination, parsed.dst_range)
            annotations = dict(rewritten_call.annotations)
            annotations.pop(DATAFLOW_CROSS_HANDLER_HANDOFF_TRANSFER_INDEX_ATTR, None)
            annotations.pop("tl.pipeline_buffer_versions", None)
            annotations[DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_TRANSFER_ATTR] = tir.IntImm("int32", transfer.binding_index)
            annotations[DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_STAGE_ATTR] = tir.IntImm(
                "int32",
                stage,
            )
            contract = rewritten_call.args[2] if len(rewritten_call.args) > 2 else None
            cloned_call = T.copy(
                src_region,
                dst_region,
                contract,
                annotations=annotations,
            )
            guards = [
                producer_loop.loop_var == producer_loop.min + producer_loop.extent - 1,
                *(
                    last_active_loop_iteration(
                        loop,
                        producer_let_bindings,
                        producer_guards,
                    )
                    for loop in producer_enclosing_loops
                ),
                stage_count_var > stage,
            ]
            guards.extend(tir.stmt_functor.substitute(guard, substitution) for guard in context.guards)
            condition = guards[0]
            for guard in guards[1:]:
                condition = tir.And(condition, guard)
            cloned_stmt: tir.Stmt = tir.IfThenElse(
                condition,
                tir.Evaluate(cloned_call),
                None,
            )
            for loop in reversed(context.inner_loops):
                cloned_stmt = tir.For(
                    loop.loop_var,
                    tir.stmt_functor.substitute(loop.min, substitution),
                    tir.stmt_functor.substitute(loop.extent, substitution),
                    loop.kind,
                    cloned_stmt,
                    loop.thread_binding,
                    loop.annotations,
                    loop.step,
                    getattr(loop, "span", None),
                )
            cloned_stmts.append(cloned_stmt)

    producer_body = append_to_pipeline_body(
        producer_func.body,
        producer_loop,
        tuple(cloned_stmts),
    )
    producer_func = producer_func.with_body(producer_body)
    return producer_func, tuple(producer_logical_params)


def ensure_handoff_tensor_param(
    prim_func: tir.PrimFunc,
    params: tuple[DataflowHandlerParam, ...],
    *,
    consumer_param: DataflowHandlerParam,
    consumer_buffer: tir.Buffer,
) -> tuple[tir.PrimFunc, tuple[DataflowHandlerParam, ...], tir.Buffer]:
    tensor_index = consumer_param.tensor_arg_index
    if tensor_index is None:
        raise DataflowPrimFuncLoweringError(f"handoff tensor parameter {consumer_param.name!r} has no tensor index")
    bindings = handler_param_bindings(prim_func, params)
    for param in params:
        if param.role is not DataflowHandlerParamRole.TENSOR_ARG or param.tensor_arg_index != tensor_index:
            continue
        _, buffer = bindings[param.name]
        if buffer is not None and buffers_have_compatible_views(
            buffer,
            consumer_buffer,
        ):
            return prim_func, params, buffer

    used_names = set(bindings)
    base_name = consumer_param.name
    if base_name in used_names:
        base_name = f"dataflow_handoff_tensor_{tensor_index}_{base_name}"
    name = base_name
    suffix = 1
    while name in used_names:
        name = f"{base_name}_{suffix}"
        suffix += 1
    handle = tir.Var(f"{name}_handle", "handle")
    data = tir.Var(name, consumer_buffer.data.type_annotation)
    buffer = clone_buffer(consumer_buffer, data=data, name=name)
    updated_params = (
        *params,
        DataflowHandlerParam(
            ordinal=len(params),
            name=name,
            role=DataflowHandlerParamRole.TENSOR_ARG,
            dtype=str(buffer.dtype),
            c_type=dataflow_buffer_cuda_c_type(str(buffer.dtype)),
            is_pointer=True,
            is_const=True,
            tensor_arg_index=tensor_index,
        ),
    )
    return (
        tir.PrimFunc(
            [*prim_func.params, handle],
            prim_func.body,
            prim_func.ret_type,
            {**prim_func.buffer_map, handle: buffer},
            prim_func.attrs,
            prim_func.span,
        ),
        updated_params,
        buffer,
    )


def buffers_have_compatible_views(left: tir.Buffer, right: tir.Buffer) -> bool:
    return (
        left.dtype == right.dtype
        and left.scope() == right.scope()
        and tvm.ir.structural_equal(left.shape, right.shape)
        and tvm.ir.structural_equal(left.strides, right.strides)
        and tvm.ir.structural_equal(left.elem_offset, right.elem_offset)
    )


def loop_path_context(
    body: tir.Stmt,
    target_loop: tir.For,
) -> tuple[
    tuple[tir.For, ...],
    tuple[tuple[tir.Var, tir.PrimExpr], ...],
    tuple[tir.PrimExpr, ...],
]:
    result: (
        tuple[
            tuple[tir.For, ...],
            tuple[tuple[tir.Var, tir.PrimExpr], ...],
            tuple[tir.PrimExpr, ...],
        ]
        | None
    ) = None

    def walk(
        stmt: tir.Stmt,
        loops: tuple[tir.For, ...],
        lets: tuple[tuple[tir.Var, tir.PrimExpr], ...],
        guards: tuple[tir.PrimExpr, ...],
    ) -> None:
        nonlocal result
        if result is not None:
            return
        if isinstance(stmt, tir.For):
            if stmt.same_as(target_loop):
                result = (loops, lets, guards)
                return
            walk(stmt.body, (*loops, stmt), lets, guards)
            return
        if isinstance(stmt, tir.SeqStmt):
            for child in stmt.seq:
                walk(child, loops, lets, guards)
            return
        if isinstance(stmt, tir.LetStmt):
            walk(stmt.body, loops, (*lets, (stmt.var, stmt.value)), guards)
            return
        if isinstance(stmt, tir.IfThenElse):
            walk(stmt.then_case, loops, lets, (*guards, stmt.condition))
            if stmt.else_case is not None:
                walk(
                    stmt.else_case,
                    loops,
                    lets,
                    (*guards, tir.Not(stmt.condition)),
                )
            return
        if isinstance(stmt, tir.AttrStmt):
            walk(stmt.body, loops, lets, guards)
            return
        if isinstance(stmt, tir.BlockRealize):
            walk(stmt.block.body, loops, lets, guards)
            return
        if isinstance(stmt, tir.Block):
            walk(stmt.body, loops, lets, guards)

    walk(body, (), (), ())
    if result is None:
        raise DataflowPrimFuncLoweringError("could not recover the structured path to the handoff producer pipeline")
    return result


def expr_uses_any_var(expr: tir.PrimExpr, variables: tuple[tir.Var, ...]) -> bool:
    found = False

    def visit(node: Any) -> None:
        nonlocal found
        if isinstance(node, tir.Var) and any(node.same_as(var) for var in variables):
            found = True

    tir.stmt_functor.post_order_visit(expr, visit)
    return found


def last_active_loop_iteration(
    loop: tir.For,
    let_bindings: tuple[tuple[tir.Var, tir.PrimExpr], ...],
    guards: tuple[tir.PrimExpr, ...],
) -> tir.PrimExpr:
    dependent_vars = [loop.loop_var]
    for var, value in let_bindings:
        if expr_uses_any_var(value, tuple(dependent_vars)):
            dependent_vars.append(var)
    dependent_guards = tuple(guard for guard in guards if expr_uses_any_var(guard, tuple(dependent_vars)))
    structural_last = loop.loop_var == loop.min + loop.extent - 1
    if not dependent_guards:
        return structural_last

    substitution: dict[tir.Var, tir.PrimExpr] = {
        loop.loop_var: loop.loop_var + 1,
    }
    for var, value in let_bindings:
        substitution[var] = tir.stmt_functor.substitute(value, substitution)
    next_guards = tuple(tir.stmt_functor.substitute(guard, substitution) for guard in dependent_guards)
    next_iteration_active = next_guards[0]
    for guard in next_guards[1:]:
        next_iteration_active = tir.And(next_iteration_active, guard)
    return tir.Or(structural_last, tir.Not(next_iteration_active))


def append_to_pipeline_body(
    body: tir.Stmt,
    target_loop: tir.For,
    suffix: tuple[tir.Stmt, ...],
) -> tir.Stmt:
    def append_inside(stmt: tir.Stmt) -> tir.Stmt:
        if isinstance(stmt, tir.BlockRealize):
            block = stmt.block
            new_block = tir.Block(
                block.iter_vars,
                block.reads,
                block.writes,
                block.name_hint,
                append_inside(block.body),
                block.init,
                block.alloc_buffers,
                block.match_buffers,
                block.annotations,
            )
            return tir.BlockRealize(stmt.iter_values, stmt.predicate, new_block)
        if isinstance(stmt, tir.LetStmt):
            return tir.LetStmt(stmt.var, stmt.value, append_inside(stmt.body))
        if isinstance(stmt, tir.IfThenElse) and stmt.else_case is None:
            return tir.IfThenElse(
                stmt.condition,
                append_inside(stmt.then_case),
                None,
            )
        if isinstance(stmt, tir.SeqStmt):
            return tir.SeqStmt([*stmt.seq, *suffix])
        return tir.SeqStmt([stmt, *suffix])

    def rewrite(node: Any) -> Any:
        if not isinstance(node, tir.For) or not node.same_as(target_loop):
            return node
        return tir.For(
            node.loop_var,
            node.min,
            node.extent,
            node.kind,
            append_inside(node.body),
            node.thread_binding,
            node.annotations,
            node.step,
            getattr(node, "span", None),
        )

    return tir.stmt_functor.ir_transform(body, None, rewrite, ["tir.For"])


def ensure_handoff_scalar_param(
    prim_func: tir.PrimFunc,
    params: tuple[DataflowHandlerParam, ...],
    *,
    name: str,
    dtype: str,
    role: DataflowHandlerParamRole,
    task_coord_axis: int | None = None,
) -> tuple[tir.PrimFunc, tuple[DataflowHandlerParam, ...], tir.Var]:
    bindings = handler_param_bindings(prim_func, params)
    if name in bindings:
        var, buffer = bindings[name]
        if buffer is not None:
            raise DataflowPrimFuncLoweringError(f"handoff scalar parameter {name!r} is a buffer")
        return prim_func, params, var
    var = tir.Var(name, dtype)
    updated_params = (
        *params,
        DataflowHandlerParam(
            ordinal=len(params),
            name=name,
            role=role,
            dtype=dtype,
            c_type=dataflow_scalar_cuda_c_type(dtype),
            task_coord_axis=task_coord_axis,
        ),
    )
    return (
        tir.PrimFunc(
            [*prim_func.params, var],
            prim_func.body,
            prim_func.ret_type,
            prim_func.buffer_map,
            prim_func.attrs,
            prim_func.span,
        ),
        updated_params,
        var,
    )


def handler_param_bindings(
    prim_func: tir.PrimFunc,
    params: tuple[DataflowHandlerParam, ...],
) -> dict[str, tuple[tir.Var, tir.Buffer | None]]:
    if len(prim_func.params) != len(params):
        raise DataflowPrimFuncLoweringError("logical handoff parameter metadata differs from the PrimFunc ABI")
    result: dict[str, tuple[tir.Var, tir.Buffer | None]] = {}
    for var, param in zip(prim_func.params, params):
        buffer = prim_func.buffer_map.get(var)
        actual_name = str(buffer.name) if buffer is not None else str(var)
        if actual_name != param.name:
            raise DataflowPrimFuncLoweringError(
                f"logical handoff parameter order changed before physical lowering: got={actual_name!r}, expected={param.name!r}"
            )
        result[param.name] = (var, buffer)
    return result


def clone_buffer(
    buffer: tir.Buffer,
    *,
    data: tir.Var,
    name: str,
    leading_extent: int | None = None,
) -> tir.Buffer:
    shape = list(buffer.shape)
    strides = list(buffer.strides) if buffer.strides else None
    if leading_extent is not None:
        shape.insert(0, tir.IntImm("int32", leading_extent))
        if strides:
            strides.insert(0, strides[0] * buffer.shape[0])
    return tir.decl_buffer(
        shape,
        buffer.dtype,
        name=name,
        data=data,
        strides=strides,
        elem_offset=buffer.elem_offset,
        scope=buffer.scope(),
        data_alignment=buffer.data_alignment,
        offset_factor=buffer.offset_factor,
        axis_separators=buffer.axis_separators,
    )


def remove_allocated_buffers(
    body: tir.Stmt,
    removed_data: set[tir.Var],
) -> tir.Stmt:
    def rewrite(node: Any) -> Any:
        if not isinstance(node, tir.Block):
            return node
        alloc_buffers = [buffer for buffer in node.alloc_buffers if not any(buffer.data.same_as(data) for data in removed_data)]
        if len(alloc_buffers) == len(node.alloc_buffers):
            return node
        return tir.Block(
            node.iter_vars,
            node.reads,
            node.writes,
            node.name_hint,
            node.body,
            node.init,
            alloc_buffers,
            node.match_buffers,
            node.annotations,
        )

    return tir.stmt_functor.ir_transform(body, None, rewrite, ["tir.Block"])


def unique_primfunc_symbol(
    handler: DataflowHandlerSpec,
    used_symbols: set[str],
) -> str:
    """Keep legacy symbols when unique and disambiguate compiled variants."""

    base_symbol = f"dataflow_primfunc_{handler.operator_name}_device"
    symbol = base_symbol
    if symbol in used_symbols:
        symbol = f"dataflow_primfunc_{handler.operator_name}__dataflow_variant_{handler.handler_id}_device"
    suffix = 1
    candidate = symbol
    while candidate in used_symbols:
        candidate = f"{symbol}_{suffix}"
        suffix += 1
    used_symbols.add(candidate)
    return candidate


def handler_operator_kind(stage: DataflowStage) -> str:
    if stage.call is None:
        raise DataflowPrimFuncLoweringError(f"Dataflow stage {stage.name!r} has no operator call")
    if stage.kind is DataflowStageKind.MAP:
        return stage.call.kind.value
    return stage.kind.value


def prim_func_from_source_factory(
    source_factory: Any,
    call: OperatorCall,
    stage: DataflowStage | None,
    global_symbol: str,
    range_extent: int,
    max_reduce_input_slots: int,
    max_input_slots: int,
    tensor_arg_plan: DataflowTensorArgPlan,
    task_params: tuple[TaskParam, ...],
    *,
    specialization_overrides: Mapping[str, Any] | None = None,
) -> tir.PrimFunc:
    if not callable(source_factory):
        raise DataflowPrimFuncLoweringError(f"primfunc_source_factory for operator {call.name!r} must be callable")
    source_or_func = call_primfunc_source_factory(
        source_factory,
        {
            "global_symbol": global_symbol,
            "range_extent": range_extent,
            "max_reduce_input_slots": max_reduce_input_slots,
            "max_input_slots": max_input_slots,
            "task_params": task_params,
            "operator_call": call,
            "stage": stage,
            "tensor_arg_plan": tensor_arg_plan,
        },
    )
    if isinstance(source_or_func, str):
        return compile_prim_func(
            source_or_func,
            global_symbol,
            specialization_constants=compile_namespace_for_call(
                call,
                overrides=specialization_overrides,
            ),
        )
    if isinstance(source_or_func, tir.PrimFunc):
        if specialization_overrides:
            raise DataflowPrimFuncLoweringError(
                f"primfunc_source_factory for operator {call.name!r} returned a PrimFunc "
                "that cannot consume compiler-owned precision specialization overrides"
            )
        return source_or_func
    raise DataflowPrimFuncLoweringError(
        f"primfunc_source_factory for operator {call.name!r} must return source str or tir.PrimFunc, got {type(source_or_func)!r}"
    )


def call_primfunc_source_factory(
    source_factory: Any,
    kwargs: dict[str, Any],
) -> Any:
    try:
        signature = inspect.signature(source_factory)
    except (TypeError, ValueError):
        return source_factory(**kwargs)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return source_factory(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return source_factory(**accepted)


def body_ir_to_prim_func(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    global_symbol: str,
    range_extent: int,
    max_reduce_input_slots: int,
    tensor_arg_plan: DataflowTensorArgPlan,
    *,
    handler_variant_key: DataflowHandlerVariantKey,
    thread_count: int = 1,
    task_params: tuple[TaskParam, ...] = (),
    max_input_slots: int = 0,
    allow_reduce_direct_output: bool = True,
    fixed_iter_range_extent: int | None = None,
    map_input_scope: str | None = None,
    specialization_overrides: Mapping[str, Any] | None = None,
) -> tir.PrimFunc:
    if body_ir.operator_kind in {"iter", "map"}:
        source = iter_source(
            body_ir,
            call,
            global_symbol,
            range_extent,
            tensor_arg_plan,
            task_params,
            thread_count=thread_count,
            max_input_slots=max_input_slots,
            fixed_range_extent=fixed_iter_range_extent,
            map_input_scope=map_input_scope,
        )
    elif body_ir.operator_kind == "reduce":
        source = reduce_source(
            body_ir,
            call,
            global_symbol,
            max_reduce_input_slots,
            handler_variant_key=handler_variant_key,
            thread_count=thread_count,
            allow_reduce_direct_output=allow_reduce_direct_output,
        )
    elif body_ir.operator_kind == "finalize":
        source = finalize_source(body_ir, call, global_symbol, task_params, thread_count=thread_count)
    else:
        raise DataflowPrimFuncLoweringError(f"unsupported Dataflow Body IR kind {body_ir.operator_kind!r}")
    return compile_prim_func(
        source,
        global_symbol,
        specialization_constants=compile_namespace_for_call(
            call,
            overrides=specialization_overrides,
        ),
        parallel_thread_count=(thread_count if body_ir.operator_kind in {"reduce", "finalize"} else None),
    )


def iter_source(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    global_symbol: str,
    range_extent: int,
    tensor_arg_plan: DataflowTensorArgPlan,
    task_params: tuple[TaskParam, ...],
    *,
    thread_count: int = 1,
    max_input_slots: int = 0,
    fixed_range_extent: int | None = None,
    map_input_scope: str | None = None,
) -> str:
    tensor_args = iter_tensor_arg_sources(body_ir, call, range_extent, tensor_arg_plan)
    input_fields = resolve_input_fields(call) if body_ir.operator_kind == "map" and call.operator.input_types else ()
    contiguous_map_input_count = count_contiguous_map_inputs(
        body_ir,
        call,
        input_fields,
        max_input_slots,
    )
    contiguous_input_names = field_buffer_names("Items", (field.name for field in input_fields)) if contiguous_map_input_count else None
    input_names = () if contiguous_input_names is not None else map_input_buffer_names(input_fields, max_input_slots)
    fields = resolve_output_fields(call)
    field_names = tuple(field.name for field in fields)
    output_names = field_buffer_names("Out", field_names)
    args = [
        field_buffer_arg_source(
            slot_names[field.name],
            field,
            preserve_shape=bool(body_ir.tilelang_body),
            scope=map_input_scope,
        )
        for slot_names in input_names
        for field in input_fields
    ]
    if contiguous_input_names is not None:
        args.extend(
            field_buffer_arg_source(
                contiguous_input_names[field.name],
                field,
                preserve_shape=True,
                leading_extent=contiguous_map_input_count,
                scope=map_input_scope,
            )
            for field in input_fields
        )
    args.extend(tensor_args)
    args.extend(
        field_buffer_arg_source(
            output_names[field.name],
            field,
            preserve_shape=bool(body_ir.tilelang_body),
            scope=None if body_ir.tilelang_body else "shared",
        )
        for field in fields
    )
    args.extend(f"{param.name}: T.{param.dtype}" for param in task_params)
    raw_special_scalars = raw_tilelang_special_scalars(body_ir.tilelang_body) if body_ir.tilelang_body else set()
    args.extend(f"{name}: T.uint32" for name in sorted(raw_special_scalars) if name.startswith("dataflow_next_task_coord_"))
    if "dataflow_handoff_stage_count" in raw_special_scalars:
        args.append("dataflow_handoff_stage_count: T.uint32")
    args.extend(["range_begin: T.uint32", "range_end: T.uint32", "task_id: T.uint32"])
    lines = function_header(global_symbol, args, thread_count=thread_count)
    if body_ir.tilelang_body:
        tilelang_body, direct_output_fields = direct_output_body(
            body_ir,
            call,
            fields,
            output_names,
        )
        lines.extend(
            slot_layout_annotation_lines(
                fields,
                output_names,
                intermediate=call.output_type,
            )
        )
        if body_ir.operator_kind == "map":
            if call.operator.input_types and has_explicit_primfunc_slot_layout(call.operator.input_types[0]):
                layout_input_names = (contiguous_input_names,) if contiguous_input_names is not None else input_names
                for slot_names in layout_input_names:
                    lines.extend(
                        slot_layout_annotation_lines(
                            input_fields,
                            slot_names,
                            intermediate=call.operator.input_types[0],
                        )
                    )
            if contiguous_input_names is not None:
                body_lines = raw_contiguous_map_body_source_lines(
                    tilelang_body,
                    call=call,
                    input_fields=input_fields,
                    input_names=contiguous_input_names,
                    indent="        ",
                )
            else:
                body_lines = raw_map_body_source_lines(
                    tilelang_body,
                    call=call,
                    input_fields=input_fields,
                    input_names=input_names,
                    indent="        ",
                )
        else:
            body_lines = tilelang_body_source_lines(tilelang_body, indent="        ")
        if fixed_range_extent is not None:
            body_lines = specialize_tilelang_iter_range_bounds(
                body_lines,
                fixed_range_extent=fixed_range_extent,
            )
        lines.extend(body_lines)
        output_store_lines = iter_output_store_lines(
            body_ir,
            tuple(field for field in fields if field.name not in direct_output_fields),
            output_names,
            loop=None,
        )
        return_warp_groups = resolve_return_warp_groups(call)
        if return_warp_groups and output_store_lines:
            group_args = ", ".join(str(group) for group in return_warp_groups)
            lines.append(f"        with T.ws({group_args}):")
            lines.extend(f"    {line}" if line else line for line in output_store_lines)
        else:
            lines.extend(output_store_lines)
        return "\n".join(lines) + "\n"

    loop = require_loop(body_ir)
    for accumulator in body_ir.accumulators:
        lines.append(f'        {accumulator.name} = T.alloc_var("{accumulator.dtype}", {expr_source(accumulator.init, loop=loop)})')
    iter_range_end = "range_end"
    if fixed_range_extent is not None:
        lines.append(f"        range_end = range_begin + {fixed_range_extent}")
        iter_range_end = "range_end"
    lines.append(f"        for {loop.var.name} in T.serial(range_begin, {iter_range_end}):")
    map_input_names = {
        (f"Item{slot_index}", field.name): slot_names[field.name]
        for slot_index, slot_names in enumerate(input_names)
        for field in input_fields
    }
    lines.extend(
        stmt_source(
            stmt,
            loop=loop,
            indent="            ",
            map_input_names=map_input_names,
        )
        for stmt in loop.body
    )
    lines.extend(iter_output_store_lines(body_ir, fields, output_names, loop=loop))
    return "\n".join(lines) + "\n"


def tilelang_body_source_lines(body: tuple[str, ...], *, indent: str) -> list[str]:
    lines: list[str] = []
    for statement in body:
        if not statement:
            continue
        lines.extend(textwrap.indent(statement, indent).splitlines())
    return lines


def count_contiguous_map_inputs(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    input_fields: tuple[PrimFuncField, ...],
    max_input_slots: int,
) -> int:
    enabled = operator_physical_contract(call.operator.attrs).input_slots == DATAFLOW_INPUT_SLOTS_CONTIGUOUS
    if not enabled:
        return 0
    if body_ir.operator_kind != "map" or not body_ir.tilelang_body:
        raise DataflowPrimFuncLoweringError(f"operator {call.name!r} contiguous input slots require a raw TileLang map body")
    if len(input_fields) != 1 or input_fields[0].shape is None:
        raise DataflowPrimFuncLoweringError(f"operator {call.name!r} contiguous input slots currently require exactly one tensor field")
    if max_input_slots <= 0:
        raise DataflowPrimFuncLoweringError(f"operator {call.name!r} contiguous input slots require at least one input slot")
    return max_input_slots


def direct_output_body(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    fields: tuple[PrimFuncField, ...],
    output_names: dict[str, str],
    *,
    enabled_override: bool | None = None,
) -> tuple[tuple[str, ...], frozenset[str]]:
    enabled = (
        operator_physical_contract(call.operator.attrs).output_slot == DATAFLOW_OUTPUT_SLOT_DIRECT
        if enabled_override is None
        else enabled_override
    )
    if not isinstance(enabled, bool):
        raise DataflowPrimFuncLoweringError(f"operator {call.name!r} direct output override must be a bool")
    if not enabled:
        return body_ir.tilelang_body, frozenset()
    if not body_ir.tilelang_body:
        raise DataflowPrimFuncLoweringError(f"operator {call.name!r} direct output requires a raw TileLang body")

    aliases: dict[str, tuple[str, str]] = {}
    for field in fields:
        if field.shape is None:
            continue
        returned = body_ir.returns[field.name]
        if not isinstance(returned, ScalarVarIR):
            raise DataflowPrimFuncLoweringError(
                f"operator {call.name!r} direct output tensor field {field.name!r} must return a named shared buffer"
            )
        if returned.name in aliases:
            other_field = aliases[returned.name][0]
            raise DataflowPrimFuncLoweringError(
                f"operator {call.name!r} direct output cannot alias fields "
                f"{other_field!r} and {field.name!r} to the same buffer {returned.name!r}"
            )
        aliases[returned.name] = (field.name, output_names[field.name])
    if not aliases:
        raise DataflowPrimFuncLoweringError(f"operator {call.name!r} direct output requires a tensor output field")

    try:
        module = ast.parse("\n".join(body_ir.tilelang_body))
    except SyntaxError as err:
        raise DataflowPrimFuncLoweringError(f"could not parse raw TileLang body for direct output operator {call.name!r}") from err

    allocation_indices: dict[str, int] = {}
    for statement_index, statement in enumerate(module.body):
        if not is_raw_tilelang_shared_allocation(statement):
            continue
        assert isinstance(statement, ast.Assign)
        target = statement.targets[0]
        assert isinstance(target, ast.Name)
        if target.id not in aliases:
            continue
        if target.id in allocation_indices:
            raise DataflowPrimFuncLoweringError(
                f"operator {call.name!r} direct output buffer {target.id!r} must have exactly one top-level T.alloc_shared assignment"
            )
        allocation_indices[target.id] = statement_index

    missing = aliases.keys() - allocation_indices.keys()
    if missing:
        names = ", ".join(repr(name) for name in sorted(missing))
        raise DataflowPrimFuncLoweringError(
            f"operator {call.name!r} direct output returned buffer(s) {names} must be allocated by a top-level T.alloc_shared assignment"
        )

    allocation_index_set = set(allocation_indices.values())
    for statement_index, statement in enumerate(module.body):
        if statement_index in allocation_index_set:
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id in aliases and isinstance(node.ctx, (ast.Store, ast.Del)):
                raise DataflowPrimFuncLoweringError(f"operator {call.name!r} direct output buffer {node.id!r} cannot be reassigned")

    class DirectOutputAliasRewriter(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            alias = aliases.get(node.id)
            if alias is None:
                return node
            return ast.copy_location(ast.Name(id=alias[1], ctx=node.ctx), node)

    rewriter = DirectOutputAliasRewriter()
    rewritten_body = []
    for statement_index, statement in enumerate(module.body):
        if statement_index in allocation_index_set:
            continue
        rewritten = rewriter.visit(statement)
        assert isinstance(rewritten, ast.stmt)
        rewritten_body.append(rewritten)
    module.body = rewritten_body
    ast.fix_missing_locations(module)
    return (
        tuple(ast.unparse(statement).rstrip() for statement in module.body),
        frozenset(field_name for field_name, _ in aliases.values()),
    )


def is_raw_tilelang_shared_allocation(statement: ast.stmt) -> bool:
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
    ):
        return False
    callee = statement.value.func
    return (
        isinstance(callee, ast.Attribute)
        and callee.attr == "alloc_shared"
        and isinstance(callee.value, ast.Name)
        and callee.value.id == "T"
    )


def resolve_return_warp_groups(call: OperatorCall) -> tuple[int, ...]:
    return operator_physical_contract(call.operator.attrs).return_warp_groups


def specialize_tilelang_iter_range_bounds(
    lines: list[str],
    *,
    fixed_range_extent: int,
) -> list[str]:
    specialized = []
    for line in lines:
        stripped = line.strip()
        if stripped in {
            'range_end = T.cast(T.dataflow_range_end(), "int32")',
            'range_end = T.cast(range_end, "int32")',
        }:
            prefix = line[: len(line) - len(line.lstrip())]
            specialized.append(f"{prefix}range_end = range_begin + {fixed_range_extent}")
            continue
        if stripped == "range_len = range_end - range_begin":
            prefix = line[: len(line) - len(line.lstrip())]
            specialized.append(f"{prefix}range_len = {fixed_range_extent}")
            continue
        specialized.append(line)
    return specialized


def raw_contiguous_map_body_source_lines(
    body: tuple[str, ...],
    *,
    call: OperatorCall,
    input_fields: tuple[PrimFuncField, ...],
    input_names: dict[str, str],
    indent: str,
) -> list[str]:
    collection_names = set(map_intermediate_collection_parameter_names(call))
    fields_by_name = {field.name: field for field in input_fields}
    try:
        module = ast.parse("\n".join(body))
    except SyntaxError as err:
        raise DataflowPrimFuncLoweringError(f"could not parse contiguous raw map body for operator {call.name!r}") from err

    def intermediate_access(node: ast.AST) -> tuple[ast.expr, PrimFuncField] | None:
        if not isinstance(node, ast.Attribute):
            return None
        field = fields_by_name.get(node.attr)
        owner = node.value
        if not (
            field is not None
            and isinstance(owner, ast.Subscript)
            and isinstance(owner.value, ast.Name)
            and owner.value.id in collection_names
        ):
            return None
        return owner.slice, field

    def subscript_indices(node: ast.expr) -> list[ast.expr]:
        if isinstance(node, ast.Tuple):
            return list(node.elts)
        return [node]

    class ContiguousMapInputRewriter(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            access = intermediate_access(node.value)
            if access is None:
                return self.generic_visit(node)
            slot, field = access
            indices = subscript_indices(node.slice)
            rewritten_indices = [self.visit(copy.deepcopy(slot))]
            rewritten_indices.extend(self.visit(copy.deepcopy(index)) for index in indices)
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id=input_names[field.name], ctx=ast.Load()),
                    slice=ast.Tuple(elts=rewritten_indices, ctx=ast.Load()),
                    ctx=node.ctx,
                ),
                node,
            )

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            access = intermediate_access(node)
            if access is None:
                return self.generic_visit(node)
            slot, field = access
            indices: list[ast.expr] = [self.visit(copy.deepcopy(slot))]
            if field.shape is None:
                indices.append(ast.Constant(0))
            else:
                indices.extend(ast.Slice(lower=None, upper=None, step=None) for _ in field.shape)
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id=input_names[field.name], ctx=ast.Load()),
                    slice=ast.Tuple(elts=indices, ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                node,
            )

    rewritten = ContiguousMapInputRewriter().visit(module)
    ast.fix_missing_locations(rewritten)
    for node in ast.walk(rewritten):
        if intermediate_access(node) is not None:
            raise DataflowPrimFuncLoweringError(f"operator {call.name!r} has an unsupported contiguous map input access")
    return [line for statement in rewritten.body for line in textwrap.indent(ast.unparse(statement).rstrip(), indent).splitlines()]


def raw_map_body_source_lines(
    body: tuple[str, ...],
    *,
    call: OperatorCall,
    input_fields: tuple[PrimFuncField, ...],
    input_names: tuple[dict[str, str], ...],
    indent: str,
) -> list[str]:
    collection_names = map_intermediate_collection_parameter_names(call)
    fields_by_name = {field.name: field for field in input_fields}
    lines: list[str] = []
    for statement in body:
        if not statement:
            continue
        rewritten = rewrite_raw_map_intermediate_fields(
            statement,
            collection_names=collection_names,
            fields_by_name=fields_by_name,
            input_names=input_names,
        )
        lines.extend(textwrap.indent(rewritten, indent).splitlines())
    return lines


def map_intermediate_collection_parameter_names(call: OperatorCall) -> tuple[str, ...]:
    names: list[str] = []
    for name, parameter in call.operator.signature.parameters.items():
        annotation = call.operator.annotations.get(name, parameter.annotation)
        origin = get_origin(annotation)
        if origin not in (list, tuple, collections.abc.Sequence, collections.abc.Iterable):
            continue
        if any(get_intermediate_type(arg) is not None for arg in get_args(annotation)):
            names.append(name)
    return tuple(names)


def rewrite_raw_map_intermediate_fields(
    source: str,
    *,
    collection_names: tuple[str, ...],
    fields_by_name: dict[str, PrimFuncField],
    input_names: tuple[dict[str, str], ...],
) -> str:
    if not collection_names or not fields_by_name or not input_names:
        return source
    source = expand_raw_map_intermediate_copy_source(
        source,
        collection_names=collection_names,
        fields_by_name=fields_by_name,
        input_names=input_names,
    )
    source = expand_raw_map_intermediate_tile_calls_source(
        source,
        collection_names=collection_names,
        fields_by_name=fields_by_name,
        slot_count=len(input_names),
    )
    collection_pattern = "|".join(re.escape(name) for name in collection_names)
    field_pattern = "|".join(re.escape(name) for name in fields_by_name)
    pattern = re.compile(
        rf"\b(?P<collection>{collection_pattern})\s*\[\s*(?P<slot>[^\]]+)\s*\]\s*"
        rf"\.\s*(?P<field>{field_pattern})(?:\s*\[\s*(?P<indices>[^\]]+)\s*\])?"
    )

    def replace(match: re.Match[str]) -> str:
        slot_index = match.group("slot")
        field_name = match.group("field")
        indices = match.group("indices")
        field = fields_by_name[field_name]
        if indices is None:
            if field.shape is not None:
                raise DataflowPrimFuncLoweringError(f"raw map intermediate tensor field {field_name!r} must be indexed")
            indices = "0"
        return direct_map_field_access_source(
            input_names,
            field_name,
            slot_index,
            indices,
        )

    return pattern.sub(replace, source)


def expand_raw_map_intermediate_tile_calls_source(
    source: str,
    *,
    collection_names: tuple[str, ...],
    fields_by_name: dict[str, PrimFuncField],
    slot_count: int,
) -> str:
    if slot_count <= 1:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        raise DataflowPrimFuncLoweringError("could not parse raw map source while expanding dynamic intermediate tile calls") from err

    collections_set = set(collection_names)
    fields_set = set(fields_by_name)

    def dynamic_selectors(node: ast.AST) -> list[ast.expr]:
        selectors: list[ast.expr] = []
        for candidate in ast.walk(node):
            if not isinstance(candidate, ast.Attribute) or candidate.attr not in fields_set:
                continue
            owner = candidate.value
            if not (isinstance(owner, ast.Subscript) and isinstance(owner.value, ast.Name) and owner.value.id in collections_set):
                continue
            selector = owner.slice
            if isinstance(selector, ast.Constant) and isinstance(selector.value, int):
                continue
            selectors.append(selector)
        return selectors

    class StaticSlotRewriter(ast.NodeTransformer):
        def __init__(self, selector_key: str, slot_index: int) -> None:
            self.selector_key = selector_key
            self.slot_index = slot_index

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            node = self.generic_visit(node)
            assert isinstance(node, ast.Attribute)
            owner = node.value
            if not (
                node.attr in fields_set
                and isinstance(owner, ast.Subscript)
                and isinstance(owner.value, ast.Name)
                and owner.value.id in collections_set
                and ast.dump(owner.slice, include_attributes=False) == self.selector_key
            ):
                return node
            owner.slice = ast.copy_location(ast.Constant(self.slot_index), owner.slice)
            return node

    class TileCallExpander(ast.NodeTransformer):
        changed = False

        def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
            node = self.generic_visit(node)
            assert isinstance(node, ast.Expr)
            call = node.value
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "T"
            ):
                return node
            selectors = dynamic_selectors(call)
            if not selectors:
                return node
            selector_key = ast.dump(selectors[0], include_attributes=False)
            if any(ast.dump(selector, include_attributes=False) != selector_key for selector in selectors[1:]):
                raise DataflowPrimFuncLoweringError("raw map tile call cannot select multiple dynamic intermediate slots")
            self.changed = True
            expanded: list[ast.stmt] = []
            for slot_index in range(slot_count):
                slot_call = StaticSlotRewriter(selector_key, slot_index).visit(copy.deepcopy(node))
                assert isinstance(slot_call, ast.Expr)
                condition = ast.Compare(
                    left=copy.deepcopy(selectors[0]),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(slot_index)],
                )
                expanded.append(ast.If(test=condition, body=[slot_call], orelse=[]))
            return expanded

    expander = TileCallExpander()
    rewritten = expander.visit(tree)
    if not expander.changed:
        return source
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten)


def expand_raw_map_intermediate_copy_source(
    source: str,
    *,
    collection_names: tuple[str, ...],
    fields_by_name: dict[str, PrimFuncField],
    input_names: tuple[dict[str, str], ...],
) -> str:
    collection_pattern = "|".join(re.escape(name) for name in collection_names)
    field_pattern = "|".join(re.escape(name) for name in fields_by_name)
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*)T\.copy\(\s*"
        rf"(?P<collection>{collection_pattern})\s*\[\s*(?P<slot>[^\]]+)\s*\]\s*"
        rf"\.\s*(?P<field>{field_pattern})(?:\s*\[\s*(?P<indices>[^\]]+)\s*\])?"
        rf"\s*,\s*(?P<dst>[A-Za-z_]\w*(?:\s*\[[^\]]+\])?)\s*,?\s*\)",
        re.DOTALL | re.MULTILINE,
    )

    def replace(match: re.Match[str]) -> str:
        indent = match.group("indent")
        slot_index = match.group("slot")
        field_name = match.group("field")
        indices = match.group("indices")
        dst = " ".join(match.group("dst").split())
        field = fields_by_name[field_name]
        expanded = direct_map_field_copy_source(
            input_names,
            field,
            field_name,
            slot_index,
            indices,
            dst,
        )
        return "\n".join(f"{indent}{line}" if line else line for line in expanded.splitlines())

    return pattern.sub(replace, source)


def direct_map_field_copy_source(
    input_names: tuple[dict[str, str], ...],
    field: PrimFuncField,
    field_name: str,
    slot_index: str,
    indices: str | None,
    dst: str,
) -> str:
    normalized_slot = slot_index.strip()
    if re.fullmatch(r"\d+", normalized_slot):
        slot = int(normalized_slot)
        if slot >= len(input_names):
            raise DataflowPrimFuncLoweringError(f"raw map intermediate slot index {slot} is out of bounds for {len(input_names)} slot(s)")
        return f"T.copy({map_field_region_source(input_names[slot][field_name], field, indices)}, {dst})"

    lines: list[str] = []
    for slot, slot_names in enumerate(input_names):
        lines.append(f"if {normalized_slot} == {slot}:")
        lines.append(f"    T.copy({map_field_region_source(slot_names[field_name], field, indices)}, {dst})")
    return "\n".join(lines)


def map_field_region_source(
    buffer_name: str,
    field: PrimFuncField,
    indices: str | None,
) -> str:
    if indices is not None:
        return f"{buffer_name}[{indices}]"
    if field.shape is None:
        return f"{buffer_name}[0]"
    return buffer_name


def direct_map_field_access_source(
    input_names: tuple[dict[str, str], ...],
    field_name: str,
    slot_index: str,
    indices: str,
) -> str:
    normalized_slot = slot_index.strip()
    if re.fullmatch(r"\d+", normalized_slot):
        slot = int(normalized_slot)
        if slot >= len(input_names):
            raise DataflowPrimFuncLoweringError(f"raw map intermediate slot index {slot} is out of bounds for {len(input_names)} slot(s)")
        return f"{input_names[slot][field_name]}[{indices}]"

    fallback = f"{input_names[-1][field_name]}[{indices}]"
    for slot in range(len(input_names) - 2, -1, -1):
        value = f"{input_names[slot][field_name]}[{indices}]"
        fallback = f"T.if_then_else({normalized_slot} == {slot}, {value}, {fallback})"
    return fallback


def raw_reduce_body_source_lines(
    body: tuple[str, ...],
    *,
    collection_name: str,
    item_names: dict[str, str] | None = None,
    slot_item_names: tuple[dict[str, str], ...] | None = None,
    indent: str,
) -> list[str]:
    lines: list[str] = []
    for statement in body:
        if not statement:
            continue
        if slot_item_names is not None:
            statement = expand_direct_slot_reduce_loop_source(
                statement,
                collection_name=collection_name,
                slot_count=len(slot_item_names),
            )
        rewritten = rewrite_raw_reduce_item_fields(
            statement,
            collection_name=collection_name,
            item_names=item_names,
            slot_item_names=slot_item_names,
        )
        lines.extend(textwrap.indent(rewritten, indent).splitlines())
    return lines


def expand_direct_slot_reduce_loop_source(
    source: str,
    *,
    collection_name: str,
    slot_count: int,
) -> str:
    """Specialize the canonical direct-slot reduce loop into guarded slot bodies.

    The generic `items[item_idx]` lowering uses `T.if_then_else` for direct slot
    fields, which can make CUDA codegen load every candidate slot before
    selecting one.  For the canonical Dataflow reduce loop, clone the loop body per
    physical slot so each guarded branch names exactly one slot.
    """

    if slot_count <= 1:
        return source
    match = re.fullmatch(
        r"for\s+([A-Za-z_]\w*)\s+in\s+T\.serial\(\s*0\s*,\s*"
        r"T\.cast\(\s*input_count\s*,\s*['\"]int32['\"]\s*\)\s*\):\n"
        r"(?P<body>(?:    .*(?:\n|$))*)",
        source,
    )
    if match is None:
        return source
    item_var = match.group(1)
    body = textwrap.dedent(match.group("body")).rstrip()
    if not body:
        return source

    expanded: list[str] = []
    for slot_index in range(slot_count):
        slot_body = re.sub(
            rf"\b{re.escape(collection_name)}\s*\[\s*{re.escape(item_var)}\s*\]",
            f"{collection_name}[{slot_index}]",
            body,
        )
        expanded.append(f'if T.cast(input_count, "int32") > {slot_index}:')
        expanded.extend(textwrap.indent(slot_body, "    ").splitlines())
    return "\n".join(expanded)


def rewrite_raw_reduce_item_fields(
    source: str,
    *,
    collection_name: str,
    item_names: dict[str, str] | None = None,
    slot_item_names: tuple[dict[str, str], ...] | None = None,
) -> str:
    if (item_names is None) == (slot_item_names is None):
        raise DataflowPrimFuncLoweringError("raw reduce rewriting needs exactly one item buffer mode")
    field_names = item_names.keys() if item_names is not None else slot_item_names[0].keys()
    field_pattern = "|".join(re.escape(field_name) for field_name in field_names)
    if not field_pattern:
        return source
    pattern = re.compile(
        rf"\b{re.escape(collection_name)}\s*\[\s*([^\]]+)\s*\]\s*"
        rf"\.\s*({field_pattern})(?:\s*\[\s*([^\]]+)\s*\])?"
    )

    def replace(match: re.Match[str]) -> str:
        item_index, field_name, indices = match.groups()
        if slot_item_names is not None:
            return direct_reduce_field_access_source(
                slot_item_names,
                field_name,
                item_index,
                "0" if indices is None else indices,
            )
        assert item_names is not None
        if indices is None:
            return f"{item_names[field_name]}[{item_index}]"
        return f"{item_names[field_name]}[{item_index}, {indices}]"

    return pattern.sub(replace, source)


def direct_reduce_field_access_source(
    slot_item_names: tuple[dict[str, str], ...],
    field_name: str,
    item_index: str,
    indices: str,
) -> str:
    if not slot_item_names:
        raise DataflowPrimFuncLoweringError("direct reduce field access requires at least one input slot")
    fallback = f"{slot_item_names[-1][field_name]}[{indices}]"
    for slot_index in range(len(slot_item_names) - 2, -1, -1):
        value = f"{slot_item_names[slot_index][field_name]}[{indices}]"
        fallback = f"T.if_then_else({item_index} == {slot_index}, {value}, {fallback})"
    return fallback


def iter_output_store_lines(
    body_ir: DataflowBodyIR,
    fields: tuple[PrimFuncField, ...],
    output_names: dict[str, str],
    *,
    loop: DataflowLoopIR | None,
) -> list[str]:
    lines: list[str] = []
    for field in fields:
        returned = body_ir.returns[field.name]
        if field.numel == 1:
            lines.append(f"        {output_names[field.name]}[0] = {expr_source(returned, loop=loop)}")
            continue
        if body_ir.tilelang_body:
            raw_named_tensor_lines = raw_tilelang_named_tensor_return_store_lines(
                returned,
                field,
                output_names[field.name],
            )
            if raw_named_tensor_lines is not None:
                lines.extend(raw_named_tensor_lines)
                continue
            raw_sliced_tensor_lines = raw_tilelang_sliced_tensor_return_store_lines(
                returned,
                field,
                output_names[field.name],
            )
            if raw_sliced_tensor_lines is not None:
                lines.extend(raw_sliced_tensor_lines)
                continue
            raw_tensor_lines = raw_tilelang_tensor_return_store_lines(
                returned,
                field,
                output_names[field.name],
            )
            if raw_tensor_lines is not None:
                lines.extend(raw_tensor_lines)
                continue
        for flat_index, value in enumerate(tuple_return_values(returned, field, body_ir.operator_name)):
            lines.append(f"        {output_names[field.name]}[{flat_index}] = {expr_source(value, loop=loop)}")
    return lines


def raw_tilelang_named_tensor_return_store_lines(
    returned: ExprIR,
    field: PrimFuncField,
    output_name: str,
) -> list[str] | None:
    if not isinstance(returned, ScalarVarIR):
        return None
    if field.shape is None:
        return [f"        {output_name}[0] = {returned.name}"]
    loop_vars = tuple(f"_dataflow_{field.name}_i{axis}" for axis, _ in enumerate(field.shape))
    extents = ", ".join(str(extent) for extent in field.shape)
    indices = ", ".join(loop_vars)
    return [
        f"        for {indices} in T.Parallel({extents}):",
        f"            {output_name}[{indices}] = {returned.name}[{indices}]",
    ]


def raw_tilelang_tensor_return_store_lines(
    returned: ExprIR,
    field: PrimFuncField,
    output_name: str,
) -> list[str] | None:
    if not isinstance(returned, TupleExprIR) or len(returned.values) != field.numel:
        return None
    if field.shape != (field.numel,):
        return None
    first = returned.values[0]
    if not isinstance(first, TensorLoadIR) or len(first.indices) < 1:
        return None
    prefix = first.indices[:-1]
    for flat_index, value in enumerate(returned.values):
        if not isinstance(value, TensorLoadIR):
            return None
        if value.tensor_name != first.tensor_name or value.indices[:-1] != prefix:
            return None
        last_index = value.indices[-1]
        if not isinstance(last_index, LiteralIR) or last_index.value != flat_index:
            return None
    loop_var = f"_dataflow_{field.name}_i"
    prefix_source = ", ".join(expr_source(index, loop=None) for index in prefix)
    tensor_index = f"{prefix_source}, {loop_var}" if prefix_source else loop_var
    return [
        f"        for {loop_var} in T.serial({field.numel}):",
        f"            {output_name}[{loop_var}] = {first.tensor_name}[{tensor_index}]",
    ]


def raw_tilelang_sliced_tensor_return_store_lines(
    returned: ExprIR,
    field: PrimFuncField,
    output_name: str,
) -> list[str] | None:
    if field.shape is None:
        return None
    if not isinstance(returned, TensorLoadIR):
        return None
    slice_count = sum(isinstance(index, SliceIR) for index in returned.indices)
    if slice_count != len(field.shape):
        return None
    loop_vars = tuple(f"_dataflow_{field.name}_i{axis}" for axis, _ in enumerate(field.shape))
    extents = ", ".join(str(extent) for extent in field.shape)
    output_indices = ", ".join(loop_vars)
    source_indices: list[str] = []
    loop_axis = 0
    for index in returned.indices:
        if isinstance(index, SliceIR):
            loop_var = loop_vars[loop_axis]
            loop_axis += 1
            if index.start is None or index.start == LiteralIR(0):
                source_indices.append(loop_var)
            else:
                source_indices.append(f"{loop_var} + {expr_source(index.start, loop=None)}")
        else:
            source_indices.append(expr_source(index, loop=None))
    source_index = ", ".join(source_indices)
    return [
        f"        for {output_indices} in T.Parallel({extents}):",
        f"            {output_name}[{output_indices}] = {returned.tensor_name}[{source_index}]",
    ]


def reduce_source(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    global_symbol: str,
    max_reduce_input_slots: int,
    *,
    handler_variant_key: DataflowHandlerVariantKey,
    thread_count: int = 1,
    allow_reduce_direct_output: bool = True,
) -> str:
    if body_ir.reducer_contract is DataflowReducerContract.ASSOCIATIVE_BINARY:
        return associative_reduce_source(
            body_ir,
            call,
            global_symbol,
            max_reduce_input_slots,
            handler_variant_key=handler_variant_key,
            thread_count=thread_count,
            allow_direct_output=allow_reduce_direct_output,
        )

    fields = resolve_output_fields(call)
    field_names = tuple(field.name for field in fields)
    output_names = field_buffer_names("Out", field_names)
    use_direct_slot_inputs = max_reduce_input_slots > 1
    if use_direct_slot_inputs:
        slot_item_names = tuple(field_buffer_names(f"Item{slot_index}", field_names) for slot_index in range(max_reduce_input_slots))
    else:
        item_names = field_buffer_names("Items", field_names)
    if body_ir.tilelang_body:
        if use_direct_slot_inputs:
            args = [
                field_buffer_arg_source(
                    slot_item_names[slot_index][field.name],
                    field,
                    preserve_shape=True,
                    scope=None,
                )
                for slot_index in range(max_reduce_input_slots)
                for field in fields
            ]
        else:
            args = [
                field_buffer_arg_source(
                    item_names[field.name],
                    field,
                    preserve_shape=True,
                    leading_extent=max_reduce_input_slots,
                )
                for field in fields
            ]
        args.extend(
            field_buffer_arg_source(
                output_names[field.name],
                field,
                preserve_shape=True,
                scope=None,
            )
            for field in fields
        )
        args.extend(["input_count: T.uint32", "task_id: T.uint32"])
        lines = function_header(global_symbol, args, thread_count=thread_count, noalias=not use_direct_slot_inputs)
        if use_direct_slot_inputs:
            for slot_names in slot_item_names:
                lines.extend(
                    slot_layout_annotation_lines(
                        fields,
                        slot_names,
                        intermediate=call.output_type,
                    )
                )
        else:
            lines.extend(
                slot_layout_annotation_lines(
                    fields,
                    item_names,
                    intermediate=call.output_type,
                )
            )
        lines.extend(
            slot_layout_annotation_lines(
                fields,
                output_names,
                intermediate=call.output_type,
            )
        )
        lines.extend(
            raw_reduce_body_source_lines(
                body_ir.tilelang_body,
                collection_name=first_parameter_name(call),
                item_names=None if use_direct_slot_inputs else item_names,
                slot_item_names=slot_item_names if use_direct_slot_inputs else None,
                indent="        ",
            )
        )
        lines.extend(iter_output_store_lines(body_ir, fields, output_names, loop=None))
        return "\n".join(lines) + "\n"

    loop = require_loop(body_ir)
    field_numels = {field.name: field.numel for field in fields}
    if use_direct_slot_inputs:
        args = [
            field_buffer_arg_source(
                slot_item_names[slot_index][field.name],
                field,
                scope="shared",
            )
            for slot_index in range(max_reduce_input_slots)
            for field in fields
        ]
    else:
        args = [f'{item_names[field.name]}: T.Tensor(({max_reduce_input_slots * field.numel},), "{field.dtype}")' for field in fields]
    args.extend(field_buffer_arg_source(output_names[field.name], field, scope="shared") for field in fields)
    args.extend(["input_count: T.uint32", "task_id: T.uint32"])
    lines = function_header(global_symbol, args, thread_count=thread_count, noalias=not use_direct_slot_inputs)
    for accumulator in body_ir.accumulators:
        lines.append(f'        {accumulator.name} = T.alloc_var("{accumulator.dtype}", {expr_source(accumulator.init, loop=loop)})')
    if use_direct_slot_inputs:
        lines.append(f'        for {loop.var.name} in T.serial(0, T.cast(input_count, "int32")):')
    else:
        lines.append(f"        for {loop.var.name} in T.serial(T.uint32(0), input_count):")
    lines.extend(
        stmt_source(
            stmt,
            loop=loop,
            indent="            ",
            reduce_items_name=None if use_direct_slot_inputs else item_names,
            reduce_slot_item_names=slot_item_names if use_direct_slot_inputs else None,
            field_numels=field_numels,
        )
        for stmt in loop.body
    )
    for field in fields:
        returned = body_ir.returns[field.name]
        if field.numel == 1:
            lines.append(
                f"        {output_names[field.name]}[0] = "
                f"{expr_source(returned, loop=loop, reduce_items_name=None if use_direct_slot_inputs else item_names, reduce_slot_item_names=slot_item_names if use_direct_slot_inputs else None, field_numels=field_numels)}"
            )
            continue
        for flat_index, value in enumerate(tuple_return_values(returned, field, body_ir.operator_name)):
            lines.append(
                f"        {output_names[field.name]}[{flat_index}] = "
                f"{expr_source(value, loop=loop, reduce_items_name=None if use_direct_slot_inputs else item_names, reduce_slot_item_names=slot_item_names if use_direct_slot_inputs else None, field_numels=field_numels)}"
            )
    return "\n".join(lines) + "\n"


def associative_reduce_source(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    global_symbol: str,
    input_slot_count: int,
    *,
    handler_variant_key: DataflowHandlerVariantKey,
    thread_count: int,
    allow_direct_output: bool,
) -> str:
    specialization = handler_variant_key.reduce_arity
    if specialization is None:
        raise DataflowPrimFuncLoweringError(f"associative reducer {call.name!r} requires a reduce arity variant")
    arity_class = specialization.arity_class
    expected_slot_count = {
        REDUCE_ARITY_PASSTHROUGH: 1,
        REDUCE_ARITY_BINARY: 2,
    }.get(arity_class)
    if expected_slot_count is not None and input_slot_count != expected_slot_count:
        raise DataflowPrimFuncLoweringError(
            f"associative reducer {call.name!r} variant {arity_class!r} requires "
            f"{expected_slot_count} input slot(s), got {input_slot_count}"
        )
    if arity_class == REDUCE_ARITY_GENERIC and input_slot_count < 3:
        raise DataflowPrimFuncLoweringError(
            f"associative reducer {call.name!r} generic variant requires at least three input slots, got {input_slot_count}"
        )

    fields = resolve_output_fields(call)
    field_names = tuple(field.name for field in fields)
    slot_names = tuple(field_buffer_names(f"Item{slot_index}", field_names) for slot_index in range(input_slot_count))
    output_names = field_buffer_names("Out", field_names)
    preserve_shape = bool(body_ir.tilelang_body)
    scope = None if preserve_shape else "shared"
    args = [
        field_buffer_arg_source(
            slot_names[slot_index][field.name],
            field,
            preserve_shape=preserve_shape,
            scope=scope,
        )
        for slot_index in range(input_slot_count)
        for field in fields
    ]
    args.extend(
        field_buffer_arg_source(
            output_names[field.name],
            field,
            preserve_shape=preserve_shape,
            scope=scope,
        )
        for field in fields
    )
    args.extend(["input_count: T.uint32", "task_id: T.uint32"])
    lines = function_header(
        global_symbol,
        args,
        thread_count=thread_count,
        noalias=False,
    )
    if preserve_shape:
        for names in (*slot_names, output_names):
            lines.extend(
                slot_layout_annotation_lines(
                    fields,
                    names,
                    intermediate=call.output_type,
                )
            )

    if arity_class == REDUCE_ARITY_PASSTHROUGH:
        lines.extend(
            copy_associative_fields_source_lines(
                fields,
                slot_names[0],
                output_names,
                preserve_shape=preserve_shape,
                thread_count=thread_count,
                indent="        ",
            )
        )
        return "\n".join(lines) + "\n"

    if arity_class == REDUCE_ARITY_BINARY:
        if body_ir.tilelang_body:
            lines.extend(
                raw_associative_binary_body_source_lines(
                    body_ir,
                    call,
                    fields,
                    slot_names[0],
                    slot_names[1],
                    output_names,
                    allow_direct_output=allow_direct_output,
                    indent="        ",
                )
            )
        else:
            lines.extend(
                materialized_associative_expression_store_lines(
                    body_ir,
                    fields,
                    slot_names[0],
                    slot_names[1],
                    output_names,
                    indent="        ",
                )
            )
        return "\n".join(lines) + "\n"

    if arity_class != REDUCE_ARITY_GENERIC:
        raise DataflowPrimFuncLoweringError(f"unsupported associative reduce arity class {arity_class!r}")
    if body_ir.tilelang_body:
        lines.extend(
            raw_associative_left_fold_source_lines(
                body_ir,
                call,
                fields,
                slot_names,
                output_names,
                thread_count=thread_count,
                allow_direct_output=allow_direct_output,
                indent="        ",
            )
        )
    else:
        lines.extend(
            expression_associative_left_fold_source_lines(
                body_ir,
                fields,
                slot_names,
                output_names,
                thread_count=thread_count,
                indent="        ",
            )
        )
    return "\n".join(lines) + "\n"


def copy_associative_fields_source_lines(
    fields: tuple[PrimFuncField, ...],
    source_names: dict[str, str],
    output_names: dict[str, str],
    *,
    preserve_shape: bool,
    thread_count: int,
    indent: str,
) -> list[str]:
    if thread_count <= 0:
        raise DataflowPrimFuncLoweringError(f"associative field copy requires a positive thread count, got {thread_count}")
    lines: list[str] = []
    for field in fields:
        chunk = f"_dataflow_forward_{field.name}_chunk"
        lane = f"_dataflow_forward_{field.name}_lane"
        flat_index = f"{chunk} * {thread_count} + {lane}"
        chunk_count = (field.numel + thread_count - 1) // thread_count
        lines.append(f"{indent}for {chunk} in T.serial({chunk_count}):")
        lines.append(f"{indent}    for {lane} in T.Parallel({thread_count}):")
        copy_indent = f"{indent}        "
        if field.numel % thread_count:
            lines.append(f"{copy_indent}if {flat_index} < {field.numel}:")
            copy_indent = f"{copy_indent}    "
        if preserve_shape and field.shape is not None:
            indices = []
            for axis, extent in enumerate(field.shape):
                stride = 1
                for trailing_extent in field.shape[axis + 1 :]:
                    stride *= trailing_extent
                component = f"({flat_index})"
                if stride != 1:
                    component = f"{component} // {stride}"
                if axis != 0:
                    component = f"{component} % {extent}"
                indices.append(component)
            index_source = ", ".join(indices)
        else:
            index_source = flat_index
        lines.append(f"{copy_indent}{output_names[field.name]}[{index_source}] = {source_names[field.name]}[{index_source}]")
    return lines


def materialized_associative_expression_store_lines(
    body_ir: DataflowBodyIR,
    fields: tuple[PrimFuncField, ...],
    left_names: dict[str, str],
    right_names: dict[str, str],
    output_names: dict[str, str],
    *,
    indent: str,
) -> list[str]:
    """Evaluate every result before writing an output that may alias an input."""

    input_names = associative_input_field_names(
        body_ir,
        fields,
        left_names,
        right_names,
    )
    lines: list[str] = []
    stores: list[str] = []
    for field in fields:
        returned = body_ir.returns[field.name]
        values = (returned,) if field.numel == 1 else tuple_return_values(returned, field, body_ir.operator_name)
        for flat_index, value in enumerate(values):
            temporary = f"_dataflow_merge_{field.name}_{flat_index}"
            lines.append(
                f'{indent}{temporary} = T.alloc_var("{field.dtype}", {expr_source(value, loop=None, map_input_names=input_names)})'
            )
            stores.append(f"{indent}{output_names[field.name]}[{flat_index}] = {temporary}")
    lines.extend(stores)
    return lines


def associative_input_field_names(
    body_ir: DataflowBodyIR,
    fields: tuple[PrimFuncField, ...],
    left_names: dict[str, str],
    right_names: dict[str, str],
) -> dict[tuple[str, str], str]:
    if len(body_ir.reducer_input_names) != 2:
        raise DataflowPrimFuncLoweringError(f"associative reducer {body_ir.operator_name!r} requires two input names")
    left_name, right_name = body_ir.reducer_input_names
    return {
        **{(left_name, field.name): left_names[field.name] for field in fields},
        **{(right_name, field.name): right_names[field.name] for field in fields},
    }


def associative_expression_store_lines(
    body_ir: DataflowBodyIR,
    fields: tuple[PrimFuncField, ...],
    left_names: dict[str, str],
    right_names: dict[str, str],
    output_names: dict[str, str],
    *,
    indent: str,
) -> list[str]:
    input_names = associative_input_field_names(
        body_ir,
        fields,
        left_names,
        right_names,
    )
    lines: list[str] = []
    for field in fields:
        returned = body_ir.returns[field.name]
        if field.numel == 1:
            lines.append(f"{indent}{output_names[field.name]}[0] = {expr_source(returned, loop=None, map_input_names=input_names)}")
            continue
        for flat_index, value in enumerate(tuple_return_values(returned, field, body_ir.operator_name)):
            lines.append(f"{indent}{output_names[field.name]}[{flat_index}] = {expr_source(value, loop=None, map_input_names=input_names)}")
    return lines


def rewrite_raw_associative_input_fields(
    source: str,
    *,
    body_ir: DataflowBodyIR,
    fields: tuple[PrimFuncField, ...],
    left_names: dict[str, str],
    right_names: dict[str, str],
) -> str:
    field_by_name = {field.name: field for field in fields}
    input_names = associative_input_field_names(
        body_ir,
        fields,
        left_names,
        right_names,
    )
    try:
        module = ast.parse(source)
    except SyntaxError as err:
        raise DataflowPrimFuncLoweringError(f"could not parse associative reducer body statement {source!r}") from err

    class InputFieldRewriter(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            value = node.value
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and (value.value.id, value.attr) in input_names:
                rewritten = ast.Subscript(
                    value=ast.Name(
                        id=input_names[(value.value.id, value.attr)],
                        ctx=ast.Load(),
                    ),
                    slice=self.visit(node.slice),
                    ctx=node.ctx,
                )
                return ast.copy_location(rewritten, node)
            return self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            if isinstance(node.value, ast.Name) and (node.value.id, node.attr) in input_names:
                field = field_by_name[node.attr]
                buffer = ast.Name(
                    id=input_names[(node.value.id, node.attr)],
                    ctx=ast.Load(),
                )
                if field.shape is not None:
                    return ast.copy_location(buffer, node)
                rewritten = ast.Subscript(
                    value=buffer,
                    slice=ast.Constant(0),
                    ctx=ast.Load(),
                )
                return ast.copy_location(rewritten, node)
            return self.generic_visit(node)

    rewritten = InputFieldRewriter().visit(module)
    ast.fix_missing_locations(rewritten)
    assert isinstance(rewritten, ast.Module)
    return "\n".join(ast.unparse(statement).rstrip() for statement in rewritten.body)


def raw_associative_direct_output_body(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    fields: tuple[PrimFuncField, ...],
    output_names: dict[str, str],
    *,
    allow_direct_output: bool,
) -> tuple[tuple[str, ...], frozenset[str]]:
    if not allow_direct_output:
        return body_ir.tilelang_body, frozenset()
    return direct_output_body(
        body_ir,
        call,
        fields,
        output_names,
        enabled_override=True,
    )


def raw_associative_binary_body_source_lines(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    fields: tuple[PrimFuncField, ...],
    left_names: dict[str, str],
    right_names: dict[str, str],
    output_names: dict[str, str],
    *,
    allow_direct_output: bool,
    indent: str,
) -> list[str]:
    body, direct_output_fields = raw_associative_direct_output_body(
        body_ir,
        call,
        fields,
        output_names,
        allow_direct_output=allow_direct_output,
    )
    if direct_output_fields == frozenset(field.name for field in fields):
        body = strip_terminal_cta_completion_sync(body)
    lines: list[str] = []
    for statement in body:
        rewritten = rewrite_raw_associative_input_fields(
            statement,
            body_ir=body_ir,
            fields=fields,
            left_names=left_names,
            right_names=right_names,
        )
        lines.extend(textwrap.indent(rewritten, indent).splitlines())
    lines.extend(
        iter_output_store_lines(
            body_ir,
            tuple(field for field in fields if field.name not in direct_output_fields),
            output_names,
            loop=None,
        )
    )
    return lines


def strip_terminal_cta_completion_sync(
    body: tuple[str, ...],
) -> tuple[str, ...]:
    """Leave one completion barrier at the linked-handler boundary.

    A fully direct-output binary reducer has no generated output-copy epilogue:
    its last top-level ``T.sync_threads()`` only publishes stores to the caller.
    The linked collective-handler adapter already performs exactly that barrier
    after the device call.  Dropping the duplicate here is safe without making
    assumptions about tensor shapes, reducer math, or which input aliases the
    output.  Internal barriers and partially direct-output reducers are left
    untouched.
    """

    if not body or body[-1].strip() != "T.sync_threads()":
        return body
    return body[:-1]


def partition_raw_associative_body(
    body: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    allocations: list[str] = []
    computations: list[str] = []
    for source in body:
        try:
            module = ast.parse(source)
        except SyntaxError as err:
            raise DataflowPrimFuncLoweringError(f"could not parse associative reducer body statement {source!r}") from err
        if len(module.body) == 1 and is_raw_tilelang_shared_allocation(module.body[0]):
            allocations.append(source)
        else:
            computations.append(source)
    return tuple(allocations), tuple(computations)


def raw_associative_left_fold_source_lines(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    fields: tuple[PrimFuncField, ...],
    slot_names: tuple[dict[str, str], ...],
    output_names: dict[str, str],
    *,
    thread_count: int,
    allow_direct_output: bool,
    indent: str,
) -> list[str]:
    body, direct_output_fields = raw_associative_direct_output_body(
        body_ir,
        call,
        fields,
        output_names,
        allow_direct_output=allow_direct_output,
    )
    allocations, computations = partition_raw_associative_body(body)
    fully_direct_output = direct_output_fields == frozenset(field.name for field in fields)
    fold_computations = strip_terminal_cta_completion_sync(computations) if fully_direct_output else computations
    lines = tilelang_body_source_lines(allocations, indent=indent)
    # Seed the fold with the first *binary* composition.  Copying Item0 into
    # Out and then composing Item1 is semantically valid, but it adds one full
    # intermediate-sized shared-memory pass and an extra CTA barrier to every
    # n-ary reduction.  The binary reducer already proves that Item0/Item1 may
    # be composed directly into Out, so use that same physical contract here.
    # Keep the one-input branch for a generic handler that is shared by a
    # passthrough instruction; requested arity does not become a verifier gate.
    lines.append(f'{indent}if T.cast(input_count, "int32") > 1:')
    seed_indent = f"{indent}    "
    for statement in fold_computations:
        rewritten = rewrite_raw_associative_input_fields(
            statement,
            body_ir=body_ir,
            fields=fields,
            left_names=slot_names[0],
            right_names=slot_names[1],
        )
        lines.extend(textwrap.indent(rewritten, seed_indent).splitlines())
    remaining_fields = tuple(field for field in fields if field.name not in direct_output_fields)
    if remaining_fields:
        store_lines = iter_output_store_lines(
            body_ir,
            remaining_fields,
            output_names,
            loop=None,
        )
        lines.extend(f"    {line}" for line in store_lines)
    lines.append(f"{indent}else:")
    lines.extend(
        copy_associative_fields_source_lines(
            fields,
            slot_names[0],
            output_names,
            preserve_shape=True,
            thread_count=thread_count,
            indent=f"{indent}    ",
        )
    )
    if thread_count > 1 and not fully_direct_output:
        lines.append(f"{indent}T.sync_threads()")
    for slot_index in range(2, len(slot_names)):
        lines.append(f'{indent}if T.cast(input_count, "int32") > {slot_index}:')
        branch_indent = f"{indent}    "
        if thread_count > 1 and fully_direct_output:
            # Only an actually executed next fold needs to wait for the
            # preceding direct-output writes.  Its adapter-level completion
            # barrier handles the final fold, so no unconditional terminal
            # barrier remains on the two-input path.
            lines.append(f"{branch_indent}T.sync_threads()")
        for statement in fold_computations:
            rewritten = rewrite_raw_associative_input_fields(
                statement,
                body_ir=body_ir,
                fields=fields,
                left_names=output_names,
                right_names=slot_names[slot_index],
            )
            lines.extend(textwrap.indent(rewritten, branch_indent).splitlines())
        if remaining_fields:
            store_lines = iter_output_store_lines(
                body_ir,
                remaining_fields,
                output_names,
                loop=None,
            )
            lines.extend(f"    {line}" for line in store_lines)
        if thread_count > 1 and not fully_direct_output:
            lines.append(f"{branch_indent}T.sync_threads()")
    return lines


def expression_associative_left_fold_source_lines(
    body_ir: DataflowBodyIR,
    fields: tuple[PrimFuncField, ...],
    slot_names: tuple[dict[str, str], ...],
    output_names: dict[str, str],
    *,
    thread_count: int,
    indent: str,
) -> list[str]:
    temporary_names = field_buffer_names("Merge", (field.name for field in fields))
    lines = [f'{indent}{temporary_names[field.name]} = T.alloc_shared(({field.numel},), "{field.dtype}")' for field in fields]
    lines.extend(
        copy_associative_fields_source_lines(
            fields,
            slot_names[0],
            output_names,
            preserve_shape=False,
            thread_count=thread_count,
            indent=indent,
        )
    )
    if thread_count > 1:
        lines.append(f"{indent}T.sync_threads()")
    for slot_index in range(1, len(slot_names)):
        lines.append(f'{indent}if T.cast(input_count, "int32") > {slot_index}:')
        branch_indent = f"{indent}    "
        lines.extend(
            associative_expression_store_lines(
                body_ir,
                fields,
                output_names,
                slot_names[slot_index],
                temporary_names,
                indent=branch_indent,
            )
        )
        if thread_count > 1:
            lines.append(f"{branch_indent}T.sync_threads()")
        lines.extend(
            copy_associative_fields_source_lines(
                fields,
                temporary_names,
                output_names,
                preserve_shape=False,
                thread_count=thread_count,
                indent=branch_indent,
            )
        )
        if thread_count > 1:
            lines.append(f"{branch_indent}T.sync_threads()")
    return lines


def finalize_source(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    global_symbol: str,
    task_params: tuple[TaskParam, ...],
    *,
    thread_count: int = 1,
) -> str:
    if body_ir.tilelang_body:
        return raw_finalize_source(
            body_ir,
            call,
            global_symbol,
            task_params,
            thread_count=thread_count,
        )
    if not body_ir.stores:
        raise DataflowPrimFuncLoweringError(f"finalize operator {body_ir.operator_name!r} must contain stores")
    fields = resolve_input_fields(call)
    fields_by_name = {field.name: field for field in fields}
    field_names = tuple(field.name for field in fields)
    referenced_fields = set(finalize_used_fields(body_ir))
    used_fields = tuple(field for field in fields if field.name in referenced_fields)
    inter_names = field_buffer_names("Inter", field_names)
    input_name = first_parameter_name(call)
    stores_by_tensor = {store.tensor_name: store for store in body_ir.stores}
    output_names = [name for name in finalize_output_parameter_order(call, input_name) if name in stores_by_tensor]
    args = [field_buffer_arg_source(inter_names[field.name], field) for field in used_fields]
    args.extend(finalize_output_arg_source(call, stores_by_tensor[name]) for name in output_names)
    args.extend(f"{param.name}: T.{param.dtype}" for param in task_params)
    special_scalars = finalize_special_scalars(body_ir)
    if "T.dataflow_range_begin" in special_scalars:
        args.append("range_begin: T.uint32")
    if "T.dataflow_range_end" in special_scalars:
        args.append("range_end: T.uint32")
    args.append("task_id: T.uint32")
    lines = function_header(global_symbol, args, thread_count=thread_count)
    field_numels = {name: field.numel for name, field in fields_by_name.items()}
    lines.extend(
        stmt_source(store, loop=None, indent="        ", finalize_inter_name=inter_names, field_numels=field_numels)
        for store in body_ir.stores
    )
    return "\n".join(lines) + "\n"


def raw_finalize_source(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    global_symbol: str,
    task_params: tuple[TaskParam, ...],
    *,
    thread_count: int = 1,
) -> str:
    fields = resolve_input_fields(call)
    field_names = tuple(field.name for field in fields)
    fields_by_name = {field.name: field for field in fields}
    input_names = intermediate_parameter_names(call)
    if len(input_names) != len(call.operator.input_types):
        raise DataflowPrimFuncLoweringError(f"raw finalize operator {body_ir.operator_name!r} requires explicit intermediate parameters")
    inter_names_by_input = {
        input_name: field_buffer_names(
            "Inter" if len(input_names) == 1 else f"Item{slot_index}",
            field_names,
        )
        for slot_index, input_name in enumerate(input_names)
    }
    used_fields_by_input = {
        input_name: raw_finalize_used_fields(
            body_ir.tilelang_body,
            input_name=input_name,
            fields=fields,
        )
        for input_name in input_names
    }
    output_names = finalize_output_parameter_order(call, input_names[0])
    if not output_names:
        raise DataflowPrimFuncLoweringError(f"raw finalize operator {body_ir.operator_name!r} must bind at least one output tensor")
    if not any(used_fields_by_input.values()):
        raise DataflowPrimFuncLoweringError(f"raw finalize operator {body_ir.operator_name!r} must read at least one intermediate field")

    args = [
        field_buffer_arg_source(
            inter_names_by_input[input_name][field.name],
            field,
            preserve_shape=True,
            scope="shared",
        )
        for input_name in input_names
        for field in used_fields_by_input[input_name]
    ]
    args.extend(finalize_output_arg_source_for_name(call, name) for name in output_names)
    args.extend(f"{param.name}: T.{param.dtype}" for param in task_params)
    special_scalars = raw_tilelang_special_scalars(body_ir.tilelang_body)
    if "range_begin" in special_scalars:
        args.append("range_begin: T.uint32")
    if "range_end" in special_scalars:
        args.append("range_end: T.uint32")
    args.append("task_id: T.uint32")
    lines = function_header(global_symbol, args, thread_count=thread_count)
    lines.extend(
        raw_finalize_body_source_lines(
            body_ir.tilelang_body,
            input_names=input_names,
            fields_by_name=fields_by_name,
            inter_names_by_input=inter_names_by_input,
            indent="        ",
        )
    )
    return "\n".join(lines) + "\n"


def raw_tilelang_special_scalars(body: tuple[str, ...]) -> set[str]:
    source = "\n".join(body)
    result = {
        name
        for name in (
            "range_begin",
            "range_end",
            "task_id",
            "dataflow_handoff_stage_count",
        )
        if re.search(rf"\b{re.escape(name)}\b", source)
    }
    result.update(re.findall(r"\bdataflow_next_task_coord_\d+\b", source))
    return result


def finalize_special_scalars(body_ir: DataflowBodyIR) -> set[str]:
    names: set[str] = set()
    for store in body_ir.stores:
        names.update(referenced_scalar_names_for_source(store.index))
        names.update(referenced_scalar_names_for_source(store.value))
    return {name for name in names if name.startswith("T.dataflow_")}


def referenced_scalar_names_for_source(expr: ExprIR) -> set[str]:
    if isinstance(expr, ScalarVarIR):
        return {expr.name}
    if isinstance(expr, CastIR):
        return referenced_scalar_names_for_source(expr.value)
    if isinstance(expr, BinaryOpIR):
        return referenced_scalar_names_for_source(expr.left) | referenced_scalar_names_for_source(expr.right)
    if isinstance(expr, CallIR):
        return {name for arg in expr.args for name in referenced_scalar_names_for_source(arg)}
    if isinstance(expr, TensorLoadIR):
        return {name for index in expr.indices for name in referenced_scalar_names_for_source(index)}
    if isinstance(expr, FieldAccessIR):
        return referenced_scalar_names_for_source(expr.base)
    if isinstance(expr, FieldElementAccessIR):
        return referenced_scalar_names_for_source(expr.base) | {
            name for index in expr.indices for name in referenced_scalar_names_for_source(index)
        }
    if isinstance(expr, TupleExprIR):
        return {name for value in expr.values for name in referenced_scalar_names_for_source(value)}
    return set()


def raw_finalize_used_fields(
    body: tuple[str, ...],
    *,
    input_name: str,
    fields: tuple[PrimFuncField, ...],
) -> tuple[PrimFuncField, ...]:
    source = "\n".join(body)
    return tuple(field for field in fields if re.search(rf"\b{re.escape(input_name)}\.{re.escape(field.name)}\b", source))


def field_buffer_arg_source(
    buffer_name: str,
    field: PrimFuncField,
    *,
    preserve_shape: bool = False,
    leading_extent: int | None = None,
    scope: str | None = None,
) -> str:
    shape = field.shape if preserve_shape and field.shape is not None else (field.numel,)
    if leading_extent is not None:
        shape = (leading_extent, *shape)
    scope_arg = "" if scope is None else f', scope="{scope}"'
    return f'{buffer_name}: T.Tensor({shape_source(shape)}, "{field.dtype}"{scope_arg})'


def has_explicit_primfunc_slot_layout(intermediate: IntermediateType) -> bool:
    return "primfunc_slot_layout" in intermediate.attrs or DATAFLOW_LAYOUT_CONTRACTS_ATTR in intermediate.attrs


def primfunc_slot_layouts(
    intermediate: IntermediateType | None,
    fields: tuple[PrimFuncField, ...],
) -> dict[str, str]:
    field_names = {field.name for field in fields if field.shape is not None}
    layout_contracts = () if intermediate is None else intermediate.attrs.get(DATAFLOW_LAYOUT_CONTRACTS_ATTR, ())
    raw_layouts = None if intermediate is None else intermediate.attrs.get("primfunc_slot_layout")
    if layout_contracts:
        if raw_layouts is not None:
            raise DataflowPrimFuncLoweringError("layout_contracts and primfunc_slot_layout cannot both be set")
        tensor_fields = tuple(field for field in fields if field.shape is not None)
        layouts = {field.name: "linear" for field in tensor_fields}
        for request in layout_contracts:
            if not isinstance(request, DataflowTensorLayoutRequest):
                raise DataflowPrimFuncLoweringError("Dataflow intermediate has an unnormalized layout contract")
            if request.field_index >= len(tensor_fields):
                raise DataflowPrimFuncLoweringError(
                    f"layout field_index {request.field_index} is out of range for {len(tensor_fields)} tensor fields"
                )
            field = tensor_fields[request.field_index]
            assert field.shape is not None
            if request.logical_rank != len(field.shape):
                raise DataflowPrimFuncLoweringError(
                    f"layout logical_rank for field {field.name!r} does not match its rank: {request.logical_rank} != {len(field.shape)}"
                )
            if request.alignment_bytes is not None or request.allow_padding:
                raise DataflowPrimFuncLoweringError(
                    "current Dataflow layout implementations do not lower explicit alignment or padding requests"
                )
            if request.layout_family == DATAFLOW_LAYOUT_MATRIX_SWIZZLE and request.major_axis not in {
                request.logical_rank - 2,
                request.logical_rank - 1,
            }:
                raise DataflowPrimFuncLoweringError("current matrix-swizzle implementation supports only the two matrix axes")
            implementation_id = layout_implementation_id(request)
            dataflow_implementation_registry().require_contract_compatible(
                implementation_id,
                request,
                selected_explicitly=True,
            )
            if request.layout_family == DATAFLOW_LAYOUT_LINEAR:
                layouts[field.name] = "linear"
            elif request.layout_family == DATAFLOW_LAYOUT_MATRIX_SWIZZLE:
                layouts[field.name] = "wgmma_k_major" if request.major_axis == request.logical_rank - 1 else "wgmma_non_k_major"
            else:
                raise DataflowPrimFuncLoweringError(f"unsupported typed layout family {request.layout_family!r}")
        return layouts
    if raw_layouts is None:
        return {field_name: "linear" for field_name in field_names}
    if isinstance(raw_layouts, str):
        layouts = {field_name: raw_layouts for field_name in field_names}
    elif isinstance(raw_layouts, collections.abc.Mapping):
        unknown_fields = set(raw_layouts) - field_names
        if unknown_fields:
            raise DataflowPrimFuncLoweringError(f"primfunc_slot_layout contains unknown or scalar fields: {sorted(unknown_fields)!r}")
        layouts = {field_name: str(raw_layouts.get(field_name, "linear")) for field_name in field_names}
    else:
        raise DataflowPrimFuncLoweringError(f"primfunc_slot_layout must be a string or field-name mapping, got {raw_layouts!r}")
    supported = {"linear", "wgmma_k_major", "wgmma_non_k_major"}
    for field_name, layout in layouts.items():
        if layout not in supported:
            raise DataflowPrimFuncLoweringError(f"unsupported primfunc_slot_layout for field {field_name!r}: {layout!r}")
    return layouts


def slot_layout_annotation_lines(
    fields: tuple[PrimFuncField, ...],
    buffer_names: dict[str, str],
    *,
    intermediate: IntermediateType | None = None,
) -> list[str]:
    lines: list[str] = []
    layouts = primfunc_slot_layouts(intermediate, fields)
    for field in fields:
        if field.shape is None:
            continue
        buffer_name = buffer_names[field.name]
        layout = layouts[field.name]
        if layout == "wgmma_k_major":
            lines.append(f"        T.annotate_layout({{{buffer_name}: T.make_wgmma_swizzled_layout({buffer_name}, k_major=True)}})")
            continue
        if layout == "wgmma_non_k_major":
            lines.append(f"        T.annotate_layout({{{buffer_name}: T.make_wgmma_swizzled_layout({buffer_name}, k_major=False)}})")
            continue
        layout_vars = tuple(f"_dataflow_{field.name}_layout_i{axis}" for axis, _ in enumerate(field.shape))
        shape = shape_source(field.shape)
        args = ", ".join(layout_vars)
        result = f"({args},)" if len(layout_vars) == 1 else f"({args})"
        lines.append(f"        T.annotate_layout({{{buffer_name}: T.Layout({shape}, lambda {args}: {result})}})")
    return lines


def raw_finalize_body_source_lines(
    body: tuple[str, ...],
    *,
    input_names: tuple[str, ...],
    fields_by_name: dict[str, PrimFuncField],
    inter_names_by_input: dict[str, dict[str, str]],
    indent: str,
) -> list[str]:
    lines: list[str] = []
    for statement in body:
        if not statement:
            continue
        rewritten = statement
        for input_name in input_names:
            rewritten = rewrite_raw_finalize_intermediate_fields(
                rewritten,
                input_name=input_name,
                fields_by_name=fields_by_name,
                inter_names=inter_names_by_input[input_name],
            )
        lines.extend(textwrap.indent(rewritten, indent).splitlines())
    return lines


def rewrite_raw_finalize_intermediate_fields(
    source: str,
    *,
    input_name: str,
    fields_by_name: dict[str, PrimFuncField],
    inter_names: dict[str, str],
) -> str:
    for field_name, field in fields_by_name.items():
        replacement = inter_names[field_name] if field.shape is not None else f"{inter_names[field_name]}[0]"
        source = re.sub(
            rf"\b{re.escape(input_name)}\.{re.escape(field_name)}\b",
            replacement,
            source,
        )
    return source


def function_header(
    global_symbol: str,
    args: list[str],
    *,
    thread_count: int = 1,
    noalias: bool = True,
) -> list[str]:
    attrs = {
        "global_symbol": global_symbol,
        "tl.dataflow_device_function": True,
    }
    if noalias:
        attrs["tir.noalias"] = True
    return [
        "@T.prim_func",
        f"def {global_symbol}({', '.join(args)}):",
        f"    T.func_attr({attrs!r})",
        f"    with T.Kernel(1, threads={thread_count}) as _dataflow_pid:",
    ]


def compile_prim_func(
    source: str,
    global_symbol: str,
    *,
    specialization_constants: dict[str, Any] | None = None,
    parallel_thread_count: int | None = None,
) -> tir.PrimFunc:
    filename = f"<tilelang-dataflow-primfunc-{global_symbol}>"
    if parallel_thread_count is not None:
        source = legalize_static_parallel_extents_source(
            source,
            specialization_constants or {},
            thread_count=parallel_thread_count,
        )
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace: dict[str, Any] = {"T": T}
    namespace.update(specialization_constants or {})
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    prim_func = namespace[global_symbol]
    if not isinstance(prim_func, tir.PrimFunc):
        raise DataflowPrimFuncLoweringError(f"generated {global_symbol!r} did not produce a tir.PrimFunc")
    return prim_func


class StaticParallelExtentLegalizer(ast.NodeTransformer):
    def __init__(self, constants: dict[str, Any], *, thread_count: int) -> None:
        self.constants = constants
        self.thread_count = thread_count
        self.changed = False

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        call = node.iter
        if (
            not isinstance(call, ast.Call)
            or call.keywords
            or not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "T"
            or call.func.attr not in {"Parallel", "parallel"}
        ):
            return node
        targets = tuple(node.target.elts) if isinstance(node.target, (ast.Tuple, ast.List)) else (node.target,)
        if len(targets) != len(call.args) or not all(isinstance(target, ast.Name) for target in targets):
            return node
        extents = tuple(static_int_source_expr(argument, self.constants) for argument in call.args)
        if any(extent is None or extent <= 0 for extent in extents):
            return node
        concrete_extents = tuple(int(extent) for extent in extents)
        total_extent = math.prod(concrete_extents)
        if total_extent <= self.thread_count or total_extent % self.thread_count == 0:
            return node

        prefix_extent = math.prod(concrete_extents[:-1])
        alignment = self.thread_count // math.gcd(prefix_extent, self.thread_count)
        last_extent = concrete_extents[-1]
        padded_last_extent = ((last_extent + alignment - 1) // alignment) * alignment
        call.args[-1] = ast.Constant(padded_last_extent)
        last_target = targets[-1]
        node.body = [
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id=last_target.id, ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(last_extent)],
                ),
                body=node.body,
                orelse=[],
            )
        ]
        self.changed = True
        return node


def static_int_source_expr(node: ast.expr, constants: dict[str, Any]) -> int | None:
    if isinstance(node, ast.Constant):
        value = node.value
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(node, ast.Name):
        value = constants.get(node.id)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(node, ast.UnaryOp):
        operand = static_int_source_expr(node.operand, constants)
        if operand is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        return None
    if isinstance(node, ast.BinOp):
        left = static_int_source_expr(node.left, constants)
        right = static_int_source_expr(node.right, constants)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right != 0:
            return left // right
        if isinstance(node.op, ast.Mod) and right != 0:
            return left % right
    return None


def legalize_static_parallel_extents_source(
    source: str,
    constants: dict[str, Any],
    *,
    thread_count: int,
) -> str:
    if thread_count <= 0:
        raise DataflowPrimFuncLoweringError(f"parallel extent legalization requires a positive thread count, got {thread_count}")
    module = ast.parse(source)
    legalizer = StaticParallelExtentLegalizer(
        constants,
        thread_count=thread_count,
    )
    module = legalizer.visit(module)
    if not legalizer.changed:
        return source
    ast.fix_missing_locations(module)
    return f"{ast.unparse(module)}\n"


def specialization_constants_for_call(call: OperatorCall) -> dict[str, Any]:
    constants: dict[str, Any] = dict(call.operator.specialization_values)
    for intermediate in (*call.operator.input_types, call.output_type):
        if intermediate is not None:
            constants.update(intermediate.attrs.get("specialization_constants", {}))
    constants.update(call.operator.attrs.get("specialization_constants", {}))
    return constants


def compile_namespace_for_call(
    call: OperatorCall,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    namespace = operator_definition_namespace(call)
    constants = specialization_constants_for_call(call)
    unknown = set(overrides or {}) - (set(constants) | {DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION})
    if unknown:
        raise DataflowPrimFuncLoweringError(
            f"precision specialization overrides for operator {call.name!r} reference unknown constants: {sorted(unknown)!r}"
        )
    constants.update(overrides or {})
    constants.pop(DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION, None)
    namespace.update(constants)
    namespace.pop(DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION, None)
    namespace.pop("T", None)
    return namespace


def operator_definition_namespace(call: OperatorCall) -> dict[str, Any]:
    try:
        closure_vars = inspect.getclosurevars(call.operator.func)
    except TypeError:
        return {}

    namespace: dict[str, Any] = {}
    namespace.update(closure_vars.globals)
    namespace.update(closure_vars.nonlocals)
    namespace.pop("__builtins__", None)
    return namespace


def device_kernel_symbol(prim_func_symbol: str) -> str:
    return f"{prim_func_symbol}_kernel"


def lower_dataflow_primfunc_module_to_cuda(
    ir_module: tvm.IRModule,
    target: str,
    handlers: tuple[DataflowPrimFuncHandler, ...],
    *,
    pass_configs: dict[str, Any] | None = None,
) -> tuple[
    str,
    int,
    tuple[DataflowTMADescriptorSpec, ...],
    dict[str, int],
    tvm.IRModule,
]:
    from tvm.ir import CallingConv

    from tilelang.engine.lower import canon_target_host, device_codegen_without_compile
    from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck
    from tilelang.utils.target import determine_target

    target_obj = determine_target(target)
    target_host = tvm.target.Target.canon_target(canon_target_host(target_obj, None))
    target_obj = tvm.target.Target(target_obj, target_host)

    with tvm.transform.PassContext(opt_level=3, config=pass_configs or {}), target_obj:
        PreLowerSemanticCheck(ir_module)
        validate_transfer_implementation_lifecycle(ir_module, target_obj)
        mod = LowerAndLegalize(ir_module, target_obj)
        mod = OptimizeForTarget(mod, target_obj)
        device_symbols = {handler.device_symbol for handler in handlers}
        mod = restore_dataflow_device_function_attrs(mod, device_symbols)
        mod = restore_dataflow_handler_codegen_abi(mod, handlers)
        tma_descriptors = collect_tma_descriptors(mod)
        dynamic_shared_bytes = max_dynamic_shared_bytes(mod)
        handler_dynamic_shared_bytes = dynamic_shared_bytes_by_global_symbol(mod)
        device_mod = tir.transform.Filter(
            lambda func: bool(func.attrs and func.attrs.get("calling_conv", CallingConv.DEFAULT) == CallingConv.DEVICE_KERNEL_LAUNCH)
        )(mod)
        codegen_mod = device_codegen_without_compile(device_mod, target_obj)
    source = codegen_mod.inspect_source()
    return source, dynamic_shared_bytes, tma_descriptors, handler_dynamic_shared_bytes, mod


def validate_transfer_implementation_lifecycle(
    mod: tvm.IRModule,
    target: tvm.target.Target,
) -> None:
    registry = dataflow_implementation_registry()

    def visit(node: Any) -> None:
        if not (isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy"):
            return
        parsed = _ffi_api.ParseOperator(node)
        if parsed.transfer_contract is None:
            return
        plan = _ffi_api.ResolveTransferLowering(node, target)
        if not bool(plan.supported):
            return
        implementation_id = str(plan.implementation_id)
        try:
            registry.require_selectable(
                implementation_id,
                selected_explicitly=False,
            )
        except ValueError as err:
            raise DataflowPrimFuncLoweringError(
                f"transfer lowering selected an implementation that failed lifecycle governance: {implementation_id!r}"
            ) from err

    for _, base_func in mod.functions.items():
        if isinstance(base_func, tir.PrimFunc):
            tir.stmt_functor.post_order_visit(base_func.body, visit)


def collect_tma_descriptors(mod: tvm.IRModule) -> tuple[DataflowTMADescriptorSpec, ...]:
    specs: dict[str, DataflowTMADescriptorSpec] = {}
    for _, base_func in mod.functions.items():
        if not isinstance(base_func, tir.PrimFunc) or not base_func.attrs:
            continue
        tma_descriptor_args = base_func.attrs.get("tma_descriptor_args")
        if not tma_descriptor_args:
            continue
        for desc_var, args in tma_descriptor_args.items():
            spec = tma_descriptor_spec_from_args(str(desc_var), args)
            previous = specs.get(spec.name)
            if previous is not None and previous != spec:
                raise DataflowPrimFuncLoweringError(f"conflicting TMA descriptor metadata for {spec.name!r}: {previous!r} vs {spec!r}")
            specs[spec.name] = spec
    return tuple(specs[name] for name in sorted(specs))


def tma_descriptor_spec_from_args(name: str, args: Any) -> DataflowTMADescriptorSpec:
    items = list(args)
    if len(items) < 5:
        raise DataflowPrimFuncLoweringError(f"TMA descriptor {name!r} has too few metadata fields")
    kind = string_value(items[0], context=f"TMA descriptor {name!r} kind")
    is_im2col = kind == "__tvm_tensormap_create_im2col"
    if kind != "__tvm_tensormap_create_tiled" and not is_im2col:
        raise DataflowPrimFuncLoweringError(f"unsupported TMA descriptor kind {kind!r} for {name!r}")
    dtype = constant_int(items[2], context=f"TMA descriptor {name!r} dtype")
    tensor_rank = constant_int(items[3], context=f"TMA descriptor {name!r} rank")
    if tensor_rank <= 0:
        raise DataflowPrimFuncLoweringError(f"TMA descriptor {name!r} rank must be positive, got {tensor_rank}")
    tensor_name = descriptor_tensor_name(items[4], name)
    remaining = items[5:]
    if not is_im2col:
        expected = 4 * tensor_rank + 4
        if len(remaining) < expected:
            raise DataflowPrimFuncLoweringError(f"TMA descriptor {name!r} has {len(remaining)} tiled args, expected at least {expected}")
        global_dim = tuple(constant_int(item, context=f"TMA descriptor {name!r} global_dim") for item in remaining[:tensor_rank])
        global_stride = tuple(
            constant_int(item, context=f"TMA descriptor {name!r} global_stride") for item in remaining[tensor_rank : 2 * tensor_rank]
        )
        box_dim = tuple(
            constant_int(item, context=f"TMA descriptor {name!r} box_dim") for item in remaining[2 * tensor_rank : 3 * tensor_rank]
        )
        element_strides = tuple(
            constant_int(item, context=f"TMA descriptor {name!r} element_strides") for item in remaining[3 * tensor_rank : 4 * tensor_rank]
        )
        interleave, swizzle, l2_promotion, oob_fill = (
            constant_int(item, context=f"TMA descriptor {name!r} option") for item in remaining[4 * tensor_rank : 4 * tensor_rank + 4]
        )
        return DataflowTMADescriptorSpec(
            name=name,
            tensor_name=tensor_name,
            dtype=dtype,
            tensor_rank=tensor_rank,
            global_dim=global_dim,
            global_stride=global_stride,
            box_dim=box_dim,
            element_strides=element_strides,
            interleave=interleave,
            swizzle=swizzle,
            l2_promotion=l2_promotion,
            oob_fill=oob_fill,
        )

    expected = 5 * tensor_rank + 2
    if len(remaining) < expected:
        raise DataflowPrimFuncLoweringError(f"TMA descriptor {name!r} has {len(remaining)} im2col args, expected at least {expected}")
    global_dim = tuple(constant_int(item, context=f"TMA descriptor {name!r} global_dim") for item in remaining[:tensor_rank])
    global_stride = tuple(
        constant_int(item, context=f"TMA descriptor {name!r} global_stride") for item in remaining[tensor_rank : 2 * tensor_rank]
    )
    element_strides = tuple(
        constant_int(item, context=f"TMA descriptor {name!r} element_strides") for item in remaining[2 * tensor_rank : 3 * tensor_rank]
    )
    lower_corner = tuple(
        constant_int(item, context=f"TMA descriptor {name!r} lower_corner") for item in remaining[3 * tensor_rank : 4 * tensor_rank - 2]
    )
    upper_corner = tuple(
        constant_int(item, context=f"TMA descriptor {name!r} upper_corner") for item in remaining[4 * tensor_rank - 2 : 5 * tensor_rank - 4]
    )
    smem_box_pixel, smem_box_channel, interleave, swizzle, l2_promotion, oob_fill = (
        constant_int(item, context=f"TMA descriptor {name!r} option") for item in remaining[5 * tensor_rank - 4 : 5 * tensor_rank + 2]
    )
    return DataflowTMADescriptorSpec(
        name=name,
        tensor_name=tensor_name,
        dtype=dtype,
        tensor_rank=tensor_rank,
        global_dim=global_dim,
        global_stride=global_stride,
        box_dim=(),
        element_strides=element_strides,
        interleave=interleave,
        swizzle=swizzle,
        l2_promotion=l2_promotion,
        oob_fill=oob_fill,
        is_im2col=True,
        lower_corner=lower_corner,
        upper_corner=upper_corner,
        smem_box_channel=smem_box_channel,
        smem_box_pixel=smem_box_pixel,
    )


def descriptor_tensor_name(expr: Any, desc_name: str) -> str:
    if isinstance(expr, tir.Var):
        return str(expr)
    raise DataflowPrimFuncLoweringError(f"TMA descriptor {desc_name!r} global address must be a tensor parameter, got {expr!r}")


def string_value(expr: Any, *, context: str) -> str:
    value = getattr(expr, "value", None)
    if isinstance(value, str):
        return value
    if isinstance(expr, str):
        return expr
    raise DataflowPrimFuncLoweringError(f"{context} must be a string constant, got {expr!r}")


def constant_int(expr: Any, *, context: str) -> int:
    value = getattr(expr, "value", None)
    if value is not None:
        return int(value)
    if isinstance(expr, int):
        return int(expr)
    raise DataflowPrimFuncLoweringError(f"{context} must be an integer constant, got {expr!r}")


def max_dynamic_shared_bytes(mod: tvm.IRModule) -> int:
    max_bytes = 0
    for _, base_func in mod.functions.items():
        if not isinstance(base_func, tir.PrimFunc) or not base_func.attrs:
            continue
        attr_value = base_func.attrs.get("dyn_shared_memory_buf")
        if attr_value is None:
            continue
        max_bytes = max(max_bytes, int_attr(attr_value))
    return max_bytes


def dynamic_shared_bytes_by_global_symbol(mod: tvm.IRModule) -> dict[str, int]:
    result: dict[str, int] = {}
    for _, base_func in mod.functions.items():
        if not isinstance(base_func, tir.PrimFunc) or not base_func.attrs:
            continue
        global_symbol = base_func.attrs.get("global_symbol")
        if global_symbol is None:
            continue
        attr_value = base_func.attrs.get("dyn_shared_memory_buf")
        if attr_value is None:
            continue
        result[str(global_symbol)] = int_attr(attr_value)
    return result


def int_attr(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    raw_value = getattr(value, "value", None)
    if raw_value is not None:
        return int(raw_value)
    raise TypeError(f"cannot convert PrimFunc attr {value!r} to int")


def restore_dataflow_handler_codegen_abi(
    mod: tvm.IRModule,
    handlers: tuple[DataflowPrimFuncHandler, ...],
) -> tvm.IRModule:
    """Restore the logical device ABI on lowered PrimFuncs, before CUDA printing."""

    handlers_by_symbol = {handler.device_symbol: handler for handler in handlers}
    updates: dict[tvm.ir.GlobalVar, tir.PrimFunc] = {}
    for global_var, base_func in mod.functions.items():
        if not isinstance(base_func, tir.PrimFunc) or not base_func.attrs:
            continue
        global_symbol = base_func.attrs.get("global_symbol")
        handler = handlers_by_symbol.get("" if global_symbol is None else str(global_symbol))
        if handler is None:
            continue
        updates[global_var] = restore_dataflow_handler_primfunc_abi(base_func, handler)
    if updates:
        mod.update(tvm.IRModule(updates))
    return mod


def restore_dataflow_handler_primfunc_abi(
    prim_func: tir.PrimFunc,
    handler: DataflowPrimFuncHandler,
) -> tir.PrimFunc:
    if handler.operator_kind == "map":
        return prim_func
    current_params = tuple(prim_func.params)
    current_by_name = {str(param): param for param in current_params}
    if len(current_by_name) != len(current_params):
        raise DataflowPrimFuncLoweringError(f"lowered handler {handler.device_symbol!r} has duplicate parameter names")

    logical_names = {param.name for param in handler.params}
    restored_params: list[tir.Var] = []
    for logical_param in handler.params:
        current = current_by_name.get(logical_param.name)
        if current is not None:
            restored_params.append(current)
        elif not logical_param.is_pointer:
            restored_params.append(tir.Var(logical_param.name, logical_param.dtype))
    restored_params.extend(param for param in current_params if str(param) not in logical_names)

    old_readonly_indices = codegen_readonly_param_indices(prim_func)
    readonly_names = {str(current_params[index]) for index in old_readonly_indices if 0 <= index < len(current_params)}
    restored = tir.PrimFunc(
        restored_params,
        prim_func.body,
        prim_func.ret_type,
        prim_func.buffer_map,
        prim_func.attrs,
        prim_func.span,
    )
    readonly_indices = [index for index, param in enumerate(restored_params) if str(param) in readonly_names]
    restored = restored.with_attr("tl.readonly_param_indices", readonly_indices)

    non_restrict_names = {str(param) for param in prim_func.attrs.get("tl.non_restrict_params", [])}
    if handler.operator_kind == "reduce" and any(param.role is DataflowHandlerParamRole.INPUT_SLOT_FIELD for param in handler.params):
        non_restrict_names.update(str(param) for param in restored_params if param.dtype == "handle")
    non_restrict_params = [param for param in restored_params if str(param) in non_restrict_names]
    if non_restrict_params:
        restored = restored.with_attr("tl.non_restrict_params", non_restrict_params)
    return restored


def codegen_dataflow_primfunc_link_module(
    lowering: DataflowPrimFuncLoweringResult,
    *,
    wrapper_thread_count: int,
    handler_scratch_base_offset: int,
    handler_scratch_offsets: dict[str, int],
    non_restrict_output_symbols: set[str],
) -> DataflowHandlerCodegenModule:
    """Specialize lowered handler IR for composition with one wrapper kernel."""

    from tvm.ir import CallingConv

    if wrapper_thread_count <= 0:
        raise DataflowPrimFuncLoweringError(f"wrapper_thread_count must be positive, got {wrapper_thread_count}")
    if handler_scratch_base_offset < 0:
        raise DataflowPrimFuncLoweringError(f"handler scratch base offset must be non-negative, got {handler_scratch_base_offset}")
    handlers_by_symbol = {handler.device_symbol: handler for handler in lowering.handlers}
    unknown_offsets = set(handler_scratch_offsets) - set(handlers_by_symbol)
    unknown_non_restrict = non_restrict_output_symbols - set(handlers_by_symbol)
    if unknown_offsets or unknown_non_restrict:
        raise DataflowPrimFuncLoweringError(
            "wrapper handler specialization references unknown device symbols: "
            f"scratch={sorted(unknown_offsets)!r}, "
            f"non_restrict={sorted(unknown_non_restrict)!r}"
        )
    if any(offset < 0 for offset in handler_scratch_offsets.values()):
        raise DataflowPrimFuncLoweringError(f"handler scratch offsets must be non-negative: {handler_scratch_offsets!r}")
    if not lowering.codegen_artifacts:
        raise DataflowPrimFuncLoweringError("Dataflow PrimFunc link codegen requires lowered handler artifacts")
    modules = {id(artifact.module): artifact.module for artifact in lowering.codegen_artifacts}
    if len(modules) != 1:
        raise DataflowPrimFuncLoweringError("Dataflow PrimFunc handlers must share one lowered codegen module")
    mod = next(iter(modules.values()))
    artifacts_by_symbol = {artifact.device_symbol: artifact for artifact in lowering.codegen_artifacts}

    updates: dict[tvm.ir.GlobalVar, tir.PrimFunc] = {}
    targets: dict[str, tvm.target.Target] = {}
    for global_var, base_func in mod.functions.items():
        if not isinstance(base_func, tir.PrimFunc) or not base_func.attrs:
            continue
        global_symbol = base_func.attrs.get("global_symbol")
        symbol = "" if global_symbol is None else str(global_symbol)
        handler = handlers_by_symbol.get(symbol)
        if handler is None:
            continue
        artifact = artifacts_by_symbol.get(symbol)
        if artifact is None:
            raise DataflowPrimFuncLoweringError(f"missing codegen artifact for handler device symbol {symbol!r}")
        specialized = base_func
        if 1 < handler.thread_count < wrapper_thread_count:
            if prim_func_has_cluster_collective(specialized):
                raise DataflowPrimFuncLoweringError(
                    f"handler {symbol!r} cannot be composed with "
                    f"thread_limit={handler.thread_count} inside a "
                    f"{wrapper_thread_count}-thread wrapper because it contains "
                    "cluster-wide synchronization"
                )
            specialized = specialized.with_attr("tl.dataflow_thread_limit", handler.thread_count)
            specialized = specialized.with_attr(
                "tl.dataflow_partial_barrier_id",
                allocate_dataflow_partial_barrier_id(specialized, symbol),
            )
        if artifact.dynamic_shared_bytes:
            dynamic_allocations = count_dynamic_shared_allocations(specialized)
            if dynamic_allocations != 1:
                raise DataflowPrimFuncLoweringError(
                    f"handler {symbol!r} reports {artifact.dynamic_shared_bytes} dynamic-shared "
                    f"bytes but has {dynamic_allocations} merged shared.dyn allocations"
                )
            absolute_offset = handler_scratch_base_offset + handler_scratch_offsets.get(symbol, 0)
            specialized = specialized.with_attr("tl.dataflow_dynamic_shared_offset", absolute_offset)
            specialized = specialized.with_attr(
                "tl.dataflow_dynamic_shared_alignment",
                DATAFLOW_CUDA_DYNAMIC_SHARED_ALIGNMENT_BYTES,
            )
        if symbol in non_restrict_output_symbols:
            output_names = {param.name for param in artifact.params if param.role is DataflowHandlerParamRole.OUTPUT_SLOT_FIELD}
            existing = {str(param) for param in specialized.attrs.get("tl.non_restrict_params", [])}
            non_restrict_params = [param for param in specialized.params if str(param) in existing or str(param) in output_names]
            specialized = specialized.with_attr("tl.non_restrict_params", non_restrict_params)
        target = specialized.attrs.get("target")
        if target is None:
            raise DataflowPrimFuncLoweringError(f"lowered handler {symbol!r} is missing its codegen target")
        targets[str(target)] = target
        updates[global_var] = specialized

    if not updates:
        raise DataflowPrimFuncLoweringError("lowered module contains no Dataflow handler functions")
    if len(targets) != 1:
        raise DataflowPrimFuncLoweringError(f"Dataflow handlers must share one codegen target, got {sorted(targets)!r}")
    specialized_mod = tvm.IRModule(mod.functions, attrs=mod.attrs)
    specialized_mod.update(tvm.IRModule(updates))
    device_mod = tir.transform.Filter(
        lambda func: bool(func.attrs and func.attrs.get("calling_conv", CallingConv.DEFAULT) == CallingConv.DEVICE_KERNEL_LAUNCH)
    )(specialized_mod)
    target_obj = next(iter(targets.values()))
    return DataflowHandlerCodegenModule(
        module=device_mod,
        target=target_obj,
        handler_symbols=tuple(handler.device_symbol for handler in lowering.handlers),
    )


def allocate_dataflow_partial_barrier_id(prim_func: tir.PrimFunc, symbol: str) -> int:
    if not prim_func.attrs:
        raise DataflowPrimFuncLoweringError(f"handler {symbol!r} is missing CUDA named-barrier metadata")
    total_count = prim_func.attrs.get("tl.cuda_named_barrier_count")
    reserved_count = prim_func.attrs.get("tl.cuda_reserved_named_barrier_count")
    if total_count is None or reserved_count is None:
        raise DataflowPrimFuncLoweringError(f"handler {symbol!r} is missing CUDA named-barrier capacity/reservation metadata")
    total_count = int_attr(total_count)
    reserved_count = int_attr(reserved_count)
    if not 0 <= reserved_count <= total_count:
        raise DataflowPrimFuncLoweringError(
            f"handler {symbol!r} has invalid CUDA named-barrier metadata: reserved={reserved_count}, total={total_count}"
        )
    used_ids: set[int] = set(range(reserved_count))

    def visit(node: Any) -> None:
        if not isinstance(node, tir.Call) or not isinstance(node.op, tvm.ir.Op) or node.op.name != "tir.tvm_storage_sync":
            return
        if len(node.args) < 2:
            return
        barrier_id = node.args[1]
        value = getattr(barrier_id, "value", None)
        if value is None:
            raise DataflowPrimFuncLoweringError(f"handler {symbol!r} uses a non-constant CUDA named barrier id")
        used_ids.add(int(value))

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    for barrier_id in range(total_count):
        if barrier_id not in used_ids:
            return barrier_id
    raise DataflowPrimFuncLoweringError(
        f"handler {symbol!r} uses all {total_count} CUDA named barriers; none remains for thread-limited wrapper composition"
    )


def prim_func_has_cluster_collective(prim_func: tir.PrimFunc) -> bool:
    found = False

    def visit(node: Any) -> None:
        nonlocal found
        if found or not isinstance(node, tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if node.op.name == "tl.cluster_sync":
            found = True
            return
        if (
            node.op.name == "tir.tvm_storage_sync"
            and node.args
            and isinstance(node.args[0], tir.StringImm)
            and node.args[0].value == "cluster"
        ):
            found = True

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    return found


def count_dynamic_shared_allocations(prim_func: tir.PrimFunc) -> int:
    count = 0

    def visit(node: Any) -> None:
        nonlocal count
        if not isinstance(node, tir.Allocate):
            return
        pointer_type = node.buffer_var.type_annotation
        if isinstance(pointer_type, tvm.ir.PointerType) and pointer_type.storage_scope == "shared.dyn":
            count += 1

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    return count


def materialize_handler_codegen_artifacts(
    handlers: tuple[DataflowPrimFuncHandler, ...],
    codegen_module: tvm.IRModule,
    tma_descriptors: tuple[DataflowTMADescriptorSpec, ...],
    dynamic_shared_bytes_by_symbol: dict[str, int],
    *,
    target_fingerprint: str | None,
) -> tuple[tuple[DataflowPrimFuncHandler, ...], tuple[DataflowHandlerCodegenArtifact, ...]]:
    functions = codegen_device_functions(codegen_module)
    decision_functions = {
        str(global_var.name_hint): base_func
        for global_var, base_func in codegen_module.functions.items()
        if isinstance(base_func, tir.PrimFunc)
    }
    descriptor_by_name = {descriptor.name: descriptor for descriptor in tma_descriptors}
    updated_handlers: list[DataflowPrimFuncHandler] = []
    artifacts: list[DataflowHandlerCodegenArtifact] = []
    for handler in handlers:
        try:
            codegen_func = functions[handler.device_symbol]
        except KeyError as err:
            raise DataflowPrimFuncLoweringError(f"lowered module is missing handler device symbol {handler.device_symbol!r}") from err
        decision_func = decision_functions.get(handler.global_symbol, codegen_func)
        thread_count = max(handler.thread_count, codegen_thread_count(codegen_func))
        dynamic_shared_bytes = dynamic_shared_bytes_by_symbol.get(handler.device_symbol, 0)
        params, handler_tma_descriptors = materialize_codegen_params(
            handler,
            codegen_func,
            descriptor_by_name,
        )
        updated_handler = replace(
            handler,
            thread_count=thread_count,
            dynamic_shared_bytes=dynamic_shared_bytes,
            pipeline_lowerings=pipeline_lowering_decisions(decision_func),
        )
        updated_handlers.append(updated_handler)
        artifacts.append(
            DataflowHandlerCodegenArtifact(
                handler_id=handler.handler_id,
                global_symbol=handler.global_symbol,
                device_symbol=handler.device_symbol,
                params=params,
                thread_count=thread_count,
                dynamic_shared_bytes=dynamic_shared_bytes,
                tma_descriptors=handler_tma_descriptors,
                module=codegen_module,
                target_fingerprint=target_fingerprint,
            )
        )
    return tuple(updated_handlers), tuple(artifacts)


def pipeline_lowering_decisions(
    prim_func: tir.PrimFunc,
) -> tuple[dict[str, Any], ...]:
    if not prim_func.attrs:
        return ()
    raw_decisions = prim_func.attrs.get("tl.pipeline_lowering_decisions")
    if raw_decisions is None:
        return ()
    schema_version = int_attr(prim_func.attrs.get("tl.pipeline_decision_schema_version"))
    if schema_version != 1:
        raise DataflowPrimFuncLoweringError(f"unsupported pipeline decision schema version {schema_version}; expected 1")
    decisions: list[dict[str, Any]] = []
    required = {
        "schema_version",
        "loop_index",
        "requested_stages",
        "selected_implementation",
        "fallback",
        "selection_reason",
    }
    for raw in raw_decisions:
        values = {str(key): value for key, value in raw.items()}
        missing = required.difference(values)
        if missing:
            raise DataflowPrimFuncLoweringError(f"pipeline decision is missing required fields {sorted(missing)!r}")
        decision_schema = int_attr(values["schema_version"])
        if decision_schema != schema_version:
            raise DataflowPrimFuncLoweringError(
                f"pipeline decision schema does not match PrimFunc schema: {decision_schema} != {schema_version}"
            )
        selected_implementation = pipeline_string_attr(values["selected_implementation"])
        try:
            dataflow_implementation_registry().require_selectable(
                selected_implementation,
                selected_explicitly=False,
            )
        except ValueError as err:
            raise DataflowPrimFuncLoweringError(
                f"pipeline lowering selected an implementation that failed lifecycle governance: {selected_implementation!r}"
            ) from err
        decisions.append(
            {
                "schema_version": decision_schema,
                "loop_index": int_attr(values["loop_index"]),
                "requested_stages": int_attr(values["requested_stages"]),
                "selected_implementation": selected_implementation,
                "fallback": bool(int_attr(values["fallback"])),
                "selection_reason": pipeline_string_attr(values["selection_reason"]),
            }
        )
    return tuple(decisions)


def pipeline_string_attr(value: Any) -> str:
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        raise DataflowPrimFuncLoweringError(f"pipeline decision string field must be a string, got {value!r}")
    return raw


def codegen_device_functions(mod: tvm.IRModule) -> dict[str, tir.PrimFunc]:
    functions: dict[str, tir.PrimFunc] = {}
    for _, base_func in mod.functions.items():
        if not isinstance(base_func, tir.PrimFunc) or not base_func.attrs:
            continue
        global_symbol = base_func.attrs.get("global_symbol")
        if global_symbol is None:
            continue
        symbol = str(global_symbol)
        if symbol in functions:
            raise DataflowPrimFuncLoweringError(f"lowered module contains duplicate global symbol {symbol!r}")
        functions[symbol] = base_func
    return functions


def codegen_thread_count(prim_func: tir.PrimFunc) -> int:
    if not prim_func.attrs:
        return 1
    thread_extent = prim_func.attrs.get("thread_extent")
    if thread_extent is None:
        return 1
    for thread_tag, extent in thread_extent.items():
        if str(thread_tag) == "threadIdx.x":
            count = int_attr(extent)
            if count <= 0:
                raise DataflowPrimFuncLoweringError(f"lowered PrimFunc has invalid threadIdx.x extent {count}")
            return count
    return 1


def materialize_codegen_params(
    handler: DataflowPrimFuncHandler,
    codegen_func: tir.PrimFunc,
    descriptor_by_name: dict[str, DataflowTMADescriptorSpec],
) -> tuple[tuple[DataflowHandlerParam, ...], tuple[DataflowTMADescriptorSpec, ...]]:
    codegen_param_names = tuple(str(param) for param in codegen_func.params)
    if len(codegen_param_names) != len(set(codegen_param_names)):
        raise DataflowPrimFuncLoweringError(f"lowered handler {handler.device_symbol!r} has duplicate codegen parameters")
    codegen_param_set = set(codegen_param_names)
    codegen_index = {name: index for index, name in enumerate(codegen_param_names)}
    readonly_indices = codegen_readonly_param_indices(codegen_func)

    params: list[DataflowHandlerParam] = []
    logical_by_name = {param.name: param for param in handler.params}

    def append_logical(logical_param: DataflowHandlerParam) -> None:
        is_const = logical_param.is_const
        if logical_param.is_pointer and logical_param.name in codegen_index:
            is_const = codegen_index[logical_param.name] in readonly_indices
        params.append(replace(logical_param, ordinal=len(params), is_const=is_const))

    descriptors: list[DataflowTMADescriptorSpec] = []

    def append_descriptor(name: str) -> None:
        descriptor = descriptor_by_name.get(name)
        if descriptor is None:
            raise DataflowPrimFuncLoweringError(
                f"lowered handler {handler.device_symbol!r} introduced unsupported parameter "
                f"{name!r}; only structured TMA descriptors may extend the logical ABI"
            )
        descriptors.append(descriptor)
        params.append(
            DataflowHandlerParam(
                ordinal=len(params),
                name=name,
                role=DataflowHandlerParamRole.TMA_DESCRIPTOR,
                dtype="tma_descriptor",
                c_type="CUtensorMap",
                is_const=True,
                descriptor_name=name,
            )
        )

    if handler.operator_kind == "map":
        for name in codegen_param_names:
            logical_param = logical_by_name.get(name)
            if logical_param is None:
                append_descriptor(name)
            else:
                append_logical(logical_param)
        return tuple(params), tuple(descriptors)

    for logical_param in handler.params:
        if logical_param.is_pointer and logical_param.name not in codegen_param_set:
            continue
        append_logical(logical_param)
    for name in codegen_param_names:
        if name not in logical_by_name:
            append_descriptor(name)
    return tuple(params), tuple(descriptors)


def codegen_readonly_param_indices(prim_func: tir.PrimFunc) -> set[int]:
    if not prim_func.attrs:
        return set()
    indices = prim_func.attrs.get("tl.readonly_param_indices")
    if indices is None:
        return set()
    return {int_attr(index) for index in indices}


def restore_dataflow_device_function_attrs(mod: tvm.IRModule, device_symbols: set[str]) -> tvm.IRModule:
    updates: dict[tvm.ir.GlobalVar, tvm.ir.BaseFunc] = {}
    for global_var, base_func in mod.functions.items():
        if not isinstance(base_func, tir.PrimFunc):
            continue
        global_symbol = base_func.attrs.get("global_symbol") if base_func.attrs else None
        if global_symbol is None:
            continue
        if str(global_symbol) in device_symbols:
            updates[global_var] = base_func.with_attr("tl.dataflow_device_function", True)
    if updates:
        mod.update(tvm.IRModule(updates))
    return mod


def range_extent_from_plan(plan: InstructionPlan) -> int:
    positive_extents = [int(extent) for extent in plan.task_range_lengths if int(extent) > 0]
    if not positive_extents:
        raise DataflowPrimFuncLoweringError(f"Dataflow PrimFunc lowering requires a positive range extent, got {plan.task_range_lengths!r}")
    return max(positive_extents)


def compute_max_reduce_input_slots(plan: InstructionPlan) -> int:
    return max(
        (
            len(instruction.input_slots)
            for instruction in plan.instructions
            if getattr(instruction.opcode, "value", instruction.opcode) in {"reduce", "reduce_update"}
        ),
        default=1,
    )


def compute_max_reduce_input_slots_by_variant(
    plan: InstructionPlan,
) -> dict[DataflowHandlerVariantKey, int]:
    result: dict[DataflowHandlerVariantKey, int] = {}
    for instruction in plan.instructions:
        if instruction.opcode not in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE}:
            continue
        variant_key = instruction.handler_variant_key
        if variant_key is None:
            raise DataflowPrimFuncLoweringError(
                f"PrimFunc lowering requires a structured reduce variant for instruction {instruction.instruction_id}"
            )
        result[variant_key] = max(
            result.get(variant_key, 0),
            len(instruction.input_slots),
        )
    return result


def plan_reduce_input_output_storage_may_alias(plan: InstructionPlan) -> bool:
    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    for instruction in plan.instructions:
        if instruction.opcode not in {DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE} or instruction.output_slot is None:
            continue
        output = slots_by_id[instruction.output_slot]
        for input_slot_id in instruction.input_slots:
            source = slots_by_id[input_slot_id]
            if (
                source.scratch_backed
                and output.scratch_backed
                and source.scratch_offset is not None
                and source.scratch_offset == output.scratch_offset
            ):
                return True
            if source.shared_storage_id is not None and source.shared_storage_id == output.shared_storage_id:
                return True
            if source.global_storage_id is not None and source.global_storage_id == output.global_storage_id:
                return True
    return False


def count_reduce_input_slots(
    call: OperatorCall,
    variant_key: DataflowHandlerVariantKey,
    *,
    max_reduce_input_slots: int,
    max_reduce_input_slots_by_variant: dict[DataflowHandlerVariantKey, int],
) -> int:
    if call.operator.reducer_contract is not DataflowReducerContract.ASSOCIATIVE_BINARY:
        return max_reduce_input_slots
    specialization = variant_key.reduce_arity
    if specialization is None:
        raise DataflowPrimFuncLoweringError(f"associative reduce operator {call.name!r} requires reduce arity specialization")
    if specialization.arity_class == REDUCE_ARITY_PASSTHROUGH:
        expected = 1
    elif specialization.arity_class == REDUCE_ARITY_BINARY:
        expected = 2
    elif specialization.arity_class == REDUCE_ARITY_GENERIC:
        expected = max_reduce_input_slots_by_variant.get(variant_key, 0)
        if expected < 3:
            raise DataflowPrimFuncLoweringError(
                f"generic associative reduce variant for {call.name!r} requires at least three inputs, got {expected}"
            )
    else:
        raise DataflowPrimFuncLoweringError(f"unsupported associative reduce arity class {specialization.arity_class!r}")
    actual = max_reduce_input_slots_by_variant.get(variant_key, 0)
    if actual != expected:
        raise DataflowPrimFuncLoweringError(
            f"associative reduce variant for {call.name!r} has inconsistent arity: "
            f"class={specialization.arity_class!r}, actual={actual}, expected={expected}"
        )
    return expected


def compute_max_input_slots_by_handler(plan: InstructionPlan) -> dict[DataflowHandlerIdentity, int]:
    result: dict[DataflowHandlerIdentity, int] = {}
    for instruction in plan.instructions:
        operator_kind = instruction_operator_kind(instruction.opcode)
        if operator_kind is None:
            continue
        identity = instruction.handler_identity
        if identity is None:
            raise DataflowPrimFuncLoweringError(
                "PrimFunc lowering requires structured handler identity for "
                f"instruction {instruction.instruction_id} ({instruction.operator_name!r})"
            )
        result[identity] = max(result.get(identity, 0), len(instruction.input_slots))
    return result


def instruction_operator_kind(opcode: DataflowOpcode) -> str | None:
    if opcode in (DataflowOpcode.EXIT, DataflowOpcode.CLUSTER_SYNC, DataflowOpcode.RESHARED):
        return None
    if opcode is DataflowOpcode.REDUCE_UPDATE:
        return DataflowOpcode.REDUCE.value
    return opcode.value


def handler_thread_count(call: OperatorCall, operator_kind: str) -> int:
    raw_value = call.operator.attrs.get("threads", 1)
    if isinstance(raw_value, bool):
        raise DataflowPrimFuncLoweringError(
            f"primfunc handler lowering requires operator {call.name!r} threads attr to be a positive integer"
        )
    try:
        thread_count = int(raw_value)
    except (TypeError, ValueError) as err:
        raise DataflowPrimFuncLoweringError(
            f"primfunc handler lowering requires operator {call.name!r} threads attr to be a positive integer"
        ) from err
    if thread_count <= 0:
        raise DataflowPrimFuncLoweringError(f"primfunc handler lowering requires operator {call.name!r} threads attr to be positive")
    if operator_kind not in {"iter", "map", "reduce", "finalize"} and thread_count != 1:
        raise DataflowPrimFuncLoweringError(
            "primfunc handler lowering currently supports threads > 1 for iter/map/reduce/finalize handlers only; "
            f"operator {call.name!r} is {operator_kind!r}"
        )
    return thread_count


def resolve_output_fields(call: OperatorCall) -> tuple[PrimFuncField, ...]:
    if call.output_type is None:
        return ()
    return tuple(field_from_intermediate_field(call.name, field.name, field.dtype, field.shape) for field in call.output_type.fields)


def resolve_input_fields(call: OperatorCall) -> tuple[PrimFuncField, ...]:
    if not call.operator.input_types:
        raise DataflowPrimFuncLoweringError(f"primfunc handler lowering requires operator {call.name!r} to consume an intermediate")
    return tuple(
        field_from_intermediate_field(call.name, field.name, field.dtype, field.shape) for field in call.operator.input_types[0].fields
    )


def field_from_intermediate_field(
    operator_name: str,
    field_name: str,
    dtype: str | None,
    shape: tuple[Any, ...] | None,
) -> PrimFuncField:
    fixed_shape = fixed_field_shape(shape, operator_name, field_name)
    return PrimFuncField(
        name=field_name,
        dtype=normalize_dtype(dtype) or "int32",
        shape=fixed_shape,
        numel=field_numel(fixed_shape),
    )


def fixed_field_shape(shape: tuple[Any, ...] | None, operator_name: str, field_name: str) -> tuple[int, ...] | None:
    if shape is None:
        return None
    extents = []
    for extent in shape:
        try:
            value = int(extent)
        except (TypeError, ValueError) as err:
            raise DataflowPrimFuncLoweringError(
                f"primfunc handler lowering requires field {field_name!r} on operator {operator_name!r} to have fixed integer extents"
            ) from err
        if value <= 0:
            raise DataflowPrimFuncLoweringError(
                f"primfunc handler lowering requires field {field_name!r} on operator {operator_name!r} to have positive extents"
            )
        extents.append(value)
    return tuple(extents)


def field_numel(shape: tuple[int, ...] | None) -> int:
    if shape is None:
        return 1
    result = 1
    for extent in shape:
        result *= extent
    return result


def task_params_for_handler(
    program: DataflowProgram,
    operator_kind: str,
    call: OperatorCall,
    stage: DataflowStage | None = None,
) -> tuple[TaskParam, ...]:
    if operator_kind not in {"iter", "map", "finalize"}:
        return ()
    if stage is None:
        if program.partial_stage is None:
            return ()
        task_arg_names = tuple(str(name) for name in program.partial_stage.task_args)
    else:
        task_arg_names = tuple(str(name) for name in task_args_for_stage(program, stage))
    params: list[TaskParam] = []
    for name in task_arg_names:
        parameter = call.operator.signature.parameters.get(name)
        if parameter is None:
            continue
        annotation = call.operator.annotations.get(name, parameter.annotation)
        dtype = scalar_annotation_dtype(annotation)
        if dtype is None:
            raise DataflowPrimFuncLoweringError(
                f"primfunc handler lowering requires task arg {name!r} on operator {call.name!r} to use a scalar annotation"
            )
        params.append(TaskParam(name=name, dtype=dtype))
    return tuple(params)


def task_args_for_stage(program: DataflowProgram, stage: DataflowStage) -> tuple[Any, ...]:
    if stage.task_args:
        return stage.task_args
    for dep_id in stage.deps:
        upstream = program.stage(dep_id)
        task_args = task_args_for_stage(program, upstream)
        if task_args:
            return task_args
    return ()


def scalar_annotation_dtype(annotation: Any) -> str | None:
    if annotation is None:
        return None
    normalized = normalize_dtype(str(annotation))
    info = dataflow_dtype_info(normalized)
    if info is not None and info.primfunc_scalar_supported:
        return normalized
    return None


def tuple_return_values(expr: ExprIR, field: PrimFuncField, operator_name: str) -> tuple[ExprIR, ...]:
    if not isinstance(expr, TupleExprIR):
        raise DataflowPrimFuncLoweringError(
            f"primfunc handler lowering requires tensor field {field.name!r} on operator "
            f"{operator_name!r} to return a tuple of {field.numel} value(s)"
        )
    if len(expr.values) != field.numel:
        raise DataflowPrimFuncLoweringError(
            f"primfunc handler lowering requires tensor field {field.name!r} on operator "
            f"{operator_name!r} to return {field.numel} value(s), got {len(expr.values)}"
        )
    return expr.values


def field_buffer_names(base_name: str, fields: Any) -> dict[str, str]:
    field_names = tuple(fields)
    if len(field_names) == 1:
        return {field_names[0]: base_name}
    return {field_name: f"{base_name}_{field_name}" for field_name in field_names}


def map_input_buffer_names(
    fields: tuple[PrimFuncField, ...],
    max_input_slots: int,
) -> tuple[dict[str, str], ...]:
    field_names = tuple(field.name for field in fields)
    return tuple(field_buffer_names(f"Item{slot_index}", field_names) for slot_index in range(max_input_slots))


def finalize_used_fields(body_ir: DataflowBodyIR) -> tuple[str, ...]:
    names: list[str] = []
    for store in body_ir.stores:
        for field_name in referenced_field_names(store.value):
            if field_name not in names:
                names.append(field_name)
    return tuple(names)


def referenced_field_names(expr: ExprIR) -> tuple[str, ...]:
    if isinstance(expr, FieldAccessIR):
        return (expr.field_name,)
    if isinstance(expr, FieldElementAccessIR):
        return (expr.field_name,)
    if isinstance(expr, CastIR):
        return referenced_field_names(expr.value)
    if isinstance(expr, BinaryOpIR):
        return referenced_field_names(expr.left) + referenced_field_names(expr.right)
    if isinstance(expr, CallIR):
        names: list[str] = []
        for arg in expr.args:
            names.extend(referenced_field_names(arg))
        return tuple(names)
    if isinstance(expr, TupleExprIR):
        names: list[str] = []
        for value in expr.values:
            names.extend(referenced_field_names(value))
        return tuple(names)
    return ()


def iter_tensor_arg_sources(
    body_ir: DataflowBodyIR,
    call: OperatorCall,
    range_extent: int,
    tensor_arg_plan: DataflowTensorArgPlan,
) -> list[str]:
    args: list[str] = []
    parameter_names = raw_tilelang_iter_tensor_parameter_names(call) if body_ir.tilelang_body else tensor_load_names(body_ir)
    for parameter_name in parameter_names:
        require_iter_tensor_binding(call, tensor_arg_plan, body_ir.operator_kind, parameter_name)
        annotation = call.operator.annotations.get(
            parameter_name,
            call.operator.signature.parameters[parameter_name].annotation,
        )
        dtype, shape = tensor_annotation_dtype_and_shape(parameter_name, annotation, range_extent)
        args.append(f'{parameter_name}: T.Tensor({shape_source(shape)}, "{dtype}")')
    return args


def raw_tilelang_iter_tensor_parameter_names(call: OperatorCall) -> list[str]:
    names: list[str] = []
    for parameter_name, parameter in call.operator.signature.parameters.items():
        annotation = call.operator.annotations.get(parameter_name, parameter.annotation)
        if is_tensor_annotation(annotation):
            names.append(parameter_name)
    return names


def require_iter_tensor_binding(
    call: OperatorCall,
    tensor_arg_plan: DataflowTensorArgPlan,
    operator_kind: str,
    parameter_name: str,
) -> None:
    for binding in tensor_arg_plan.bindings:
        if binding.operator_kind == operator_kind and binding.operator_name == call.name and binding.parameter_name == parameter_name:
            return
    raise DataflowPrimFuncLoweringError(
        f"primfunc handler lowering requires {operator_kind} tensor parameter {parameter_name!r} "
        f"to be present in tensor_arg_plan for operator {call.name!r}"
    )


def tensor_annotation_dtype_and_shape(
    parameter_name: str,
    annotation: Any,
    range_extent: int,
) -> tuple[str, tuple[int, ...]]:
    if not is_tensor_annotation(annotation):
        raise DataflowPrimFuncLoweringError(
            f"primfunc handler lowering requires tensor parameter {parameter_name!r} to use T.Tensor annotation"
        )
    dtype = normalize_dtype(str(annotation.dtype))
    if dtype is None:
        raise DataflowPrimFuncLoweringError(
            f"primfunc handler lowering requires tensor parameter {parameter_name!r} to use T.Tensor annotation"
        )
    if not annotation.shape:
        return dtype, (range_extent,)
    extents: list[int] = []
    for axis, extent in enumerate(annotation.shape):
        fixed_extent = fixed_positive_extent(extent)
        if fixed_extent is None:
            if axis == 0:
                fixed_extent = range_extent
            else:
                raise DataflowPrimFuncLoweringError(
                    f"primfunc handler lowering requires tensor parameter {parameter_name!r} "
                    "non-leading dimensions to have fixed positive extents"
                )
        extents.append(fixed_extent)
    if len(extents) == 1:
        extents[0] = max(range_extent, extents[0])
    return dtype, tuple(extents)


def shape_source(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(extent) for extent in shape) + ")"


def is_tensor_annotation(annotation: Any) -> bool:
    shape = getattr(annotation, "shape", None)
    return (
        annotation is not None
        and hasattr(annotation, "dtype")
        and shape is not None
        and hasattr(shape, "__len__")
        and hasattr(shape, "__getitem__")
    )


def fixed_positive_extent(extent: Any) -> int | None:
    value = getattr(extent, "value", extent)
    if isinstance(value, int) and value > 0:
        return value
    return None


def stmt_source(
    stmt: Any,
    *,
    loop: DataflowLoopIR | None,
    indent: str,
    reduce_items_name: dict[str, str] | None = None,
    reduce_slot_item_names: tuple[dict[str, str], ...] | None = None,
    map_input_names: dict[tuple[str, str], str] | None = None,
    finalize_inter_name: dict[str, str] | None = None,
    field_numels: dict[str, int] | None = None,
) -> str:
    if isinstance(stmt, AssignIR):
        target = expr_source(stmt.target, loop=loop)
        value = expr_source(
            stmt.value,
            loop=loop,
            reduce_items_name=reduce_items_name,
            reduce_slot_item_names=reduce_slot_item_names,
            map_input_names=map_input_names,
            field_numels=field_numels,
        )
        return f"{indent}{target} = {value}"
    if isinstance(stmt, AugAssignIR):
        target = expr_source(stmt.target, loop=loop)
        value = expr_source(
            stmt.value,
            loop=loop,
            reduce_items_name=reduce_items_name,
            reduce_slot_item_names=reduce_slot_item_names,
            map_input_names=map_input_names,
            field_numels=field_numels,
        )
        return f"{indent}{target} {stmt.op}= {value}"
    if isinstance(stmt, TensorStoreIR):
        index = expr_source(stmt.index, loop=loop, finalize_inter_name=finalize_inter_name, field_numels=field_numels)
        value = expr_source(
            stmt.value,
            loop=loop,
            finalize_inter_name=finalize_inter_name,
            field_numels=field_numels,
        )
        return f"{indent}{stmt.tensor_name}[{index}] = {value}"
    raise DataflowPrimFuncLoweringError(f"unsupported Dataflow Body IR statement {stmt!r}")


def expr_source(
    expr: ExprIR,
    *,
    loop: DataflowLoopIR | None,
    reduce_items_name: dict[str, str] | None = None,
    reduce_slot_item_names: tuple[dict[str, str], ...] | None = None,
    map_input_names: dict[tuple[str, str], str] | None = None,
    finalize_inter_name: dict[str, str] | None = None,
    field_numels: dict[str, int] | None = None,
) -> str:
    if isinstance(expr, ScalarVarIR):
        if expr.name == "T.dataflow_task_id":
            return "task_id"
        if expr.name == "T.dataflow_range_begin":
            return "range_begin"
        if expr.name == "T.dataflow_range_end":
            return "range_end"
        if loop is not None and expr.name == loop.var.name:
            return expr.name
        return expr.name
    if isinstance(expr, LiteralIR):
        return repr(expr.value)
    if isinstance(expr, CastIR):
        value = expr_source(
            expr.value,
            loop=loop,
            reduce_items_name=reduce_items_name,
            reduce_slot_item_names=reduce_slot_item_names,
            map_input_names=map_input_names,
            finalize_inter_name=finalize_inter_name,
            field_numels=field_numels,
        )
        return f"T.{expr.dtype}({value})"
    if isinstance(expr, TensorLoadIR):
        indices = ", ".join(
            expr_source(
                index,
                loop=loop,
                reduce_items_name=reduce_items_name,
                reduce_slot_item_names=reduce_slot_item_names,
                map_input_names=map_input_names,
                finalize_inter_name=finalize_inter_name,
                field_numels=field_numels,
            )
            for index in expr.indices
        )
        return f"{expr.tensor_name}[{indices}]"
    if isinstance(expr, FieldAccessIR):
        if (
            reduce_slot_item_names is not None
            and loop is not None
            and isinstance(expr.base, ScalarVarIR)
            and expr.base.name == loop.var.name
        ):
            return direct_reduce_field_access_source(
                reduce_slot_item_names,
                expr.field_name,
                loop.var.name,
                "0",
            )
        if reduce_items_name is not None and loop is not None and isinstance(expr.base, ScalarVarIR) and expr.base.name == loop.var.name:
            return f"{reduce_items_name[expr.field_name]}[{loop.var.name}]"
        if finalize_inter_name is not None and isinstance(expr.base, ScalarVarIR):
            return f"{finalize_inter_name[expr.field_name]}[0]"
        if map_input_names is not None and isinstance(expr.base, ScalarVarIR):
            key = (expr.base.name, expr.field_name)
            if key in map_input_names:
                return f"{map_input_names[key]}[0]"
        raise DataflowPrimFuncLoweringError(f"unsupported Dataflow field access expression {expr!r}")
    if isinstance(expr, FieldElementAccessIR):
        if (
            reduce_slot_item_names is not None
            and loop is not None
            and isinstance(expr.base, ScalarVarIR)
            and expr.base.name == loop.var.name
        ):
            return direct_reduce_field_access_source(
                reduce_slot_item_names,
                expr.field_name,
                loop.var.name,
                str(expr.flat_index),
            )
        if reduce_items_name is not None and loop is not None and isinstance(expr.base, ScalarVarIR) and expr.base.name == loop.var.name:
            numel = field_numel_for_expr(expr, field_numels)
            index = f"{loop.var.name} * {numel}"
            if expr.flat_index:
                index = f"{index} + {expr.flat_index}"
            return f"{reduce_items_name[expr.field_name]}[{index}]"
        if finalize_inter_name is not None and isinstance(expr.base, ScalarVarIR):
            return f"{finalize_inter_name[expr.field_name]}[{expr.flat_index}]"
        if map_input_names is not None and isinstance(expr.base, ScalarVarIR):
            key = (expr.base.name, expr.field_name)
            if key in map_input_names:
                return f"{map_input_names[key]}[{expr.flat_index}]"
        raise DataflowPrimFuncLoweringError(f"unsupported Dataflow field element access expression {expr!r}")
    if isinstance(expr, BinaryOpIR):
        left = expr_source(
            expr.left,
            loop=loop,
            reduce_items_name=reduce_items_name,
            reduce_slot_item_names=reduce_slot_item_names,
            map_input_names=map_input_names,
            finalize_inter_name=finalize_inter_name,
            field_numels=field_numels,
        )
        right = expr_source(
            expr.right,
            loop=loop,
            reduce_items_name=reduce_items_name,
            reduce_slot_item_names=reduce_slot_item_names,
            map_input_names=map_input_names,
            finalize_inter_name=finalize_inter_name,
            field_numels=field_numels,
        )
        return f"({left} {expr.op} {right})"
    if isinstance(expr, CallIR):
        args = ", ".join(
            expr_source(
                arg,
                loop=loop,
                reduce_items_name=reduce_items_name,
                reduce_slot_item_names=reduce_slot_item_names,
                map_input_names=map_input_names,
                finalize_inter_name=finalize_inter_name,
                field_numels=field_numels,
            )
            for arg in expr.args
        )
        return f"{expr.name}({args})"
    if isinstance(expr, TupleExprIR):
        return ", ".join(
            expr_source(
                value,
                loop=loop,
                reduce_items_name=reduce_items_name,
                reduce_slot_item_names=reduce_slot_item_names,
                map_input_names=map_input_names,
                finalize_inter_name=finalize_inter_name,
                field_numels=field_numels,
            )
            for value in expr.values
        )
    raise DataflowPrimFuncLoweringError(f"unsupported Dataflow Body IR expression {expr!r}")


def field_numel_for_expr(expr: FieldElementAccessIR, field_numels: dict[str, int] | None) -> int:
    if field_numels is None or expr.field_name not in field_numels:
        raise DataflowPrimFuncLoweringError(f"missing numel metadata for Dataflow tensor field {expr.field_name!r}")
    return field_numels[expr.field_name]


def require_loop(body_ir: DataflowBodyIR) -> DataflowLoopIR:
    if body_ir.loop is None:
        raise DataflowPrimFuncLoweringError(f"{body_ir.operator_kind} operator {body_ir.operator_name!r} needs one loop")
    return body_ir.loop


def tensor_load_names(body_ir: DataflowBodyIR) -> list[str]:
    names: list[str] = []

    def visit_expr(expr: ExprIR) -> None:
        if isinstance(expr, TensorLoadIR):
            if expr.tensor_name not in names:
                names.append(expr.tensor_name)
            for index in expr.indices:
                visit_expr(index)
        elif isinstance(expr, TupleExprIR):
            for value in expr.values:
                visit_expr(value)
        elif isinstance(expr, (CastIR, FieldAccessIR)):
            visit_expr(expr.value if isinstance(expr, CastIR) else expr.base)
        elif isinstance(expr, FieldElementAccessIR):
            visit_expr(expr.base)
        elif isinstance(expr, BinaryOpIR):
            visit_expr(expr.left)
            visit_expr(expr.right)
        elif isinstance(expr, CallIR):
            for arg in expr.args:
                visit_expr(arg)

    def visit_stmt(stmt: Any) -> None:
        if isinstance(stmt, (AssignIR, AugAssignIR)):
            visit_expr(stmt.target)
            visit_expr(stmt.value)
        elif isinstance(stmt, TensorStoreIR):
            visit_expr(stmt.index)
            visit_expr(stmt.value)

    for accumulator in body_ir.accumulators:
        visit_expr(accumulator.init)
    if body_ir.loop is not None:
        for stmt in body_ir.loop.body:
            visit_stmt(stmt)
    for expr in body_ir.returns.values():
        visit_expr(expr)
    for stmt in body_ir.stores:
        visit_stmt(stmt)
    return names


def finalize_dtype(call: OperatorCall, store: TensorStoreIR) -> str:
    if call.operator.input_types and call.operator.input_types[0].fields:
        return normalize_dtype(call.operator.input_types[0].fields[0].dtype) or "int32"
    annotation = call.operator.annotations.get(store.tensor_name)
    if isinstance(annotation, tir.Buffer):
        return normalize_dtype(str(annotation.dtype)) or "int32"
    return "int32"


def finalize_output_arg_source(call: OperatorCall, store: TensorStoreIR) -> str:
    return finalize_output_arg_source_for_name(
        call,
        store.tensor_name,
        fallback_dtype=finalize_dtype(call, store),
    )


def finalize_output_arg_source_for_name(
    call: OperatorCall,
    tensor_name: str,
    *,
    fallback_dtype: str = "int32",
) -> str:
    annotation = call.operator.annotations.get(tensor_name)
    if not is_tensor_annotation(annotation):
        raise DataflowPrimFuncLoweringError(
            f"primfunc handler lowering requires finalize output {tensor_name!r} to use T.Tensor annotation"
        )
    if not annotation.shape:
        raise DataflowPrimFuncLoweringError(f"primfunc handler lowering requires finalize output {tensor_name!r} to have a fixed extent")
    shape = []
    for extent in annotation.shape:
        fixed_extent = fixed_positive_extent(extent)
        if fixed_extent is None:
            raise DataflowPrimFuncLoweringError(
                f"primfunc handler lowering requires finalize output {tensor_name!r} to have fixed positive extents"
            )
        shape.append(fixed_extent)
    dtype = normalize_dtype(str(annotation.dtype)) or fallback_dtype
    return f'{tensor_name}: T.Tensor({shape_source(tuple(shape))}, "{dtype}")'


def finalize_output_parameter_order(call: OperatorCall, input_name: str) -> list[str]:
    names: list[str] = []
    for name, parameter in call.operator.signature.parameters.items():
        annotation = call.operator.annotations.get(name, parameter.annotation)
        if name == input_name or get_intermediate_type(annotation) is not None:
            continue
        if name in call.bound_arguments:
            names.append(name)
    return names


def intermediate_parameter_names(call: OperatorCall) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in call.operator.signature.parameters.items()
        if get_intermediate_type(call.operator.annotations.get(name, parameter.annotation)) is not None
    )


def first_parameter_name(call: OperatorCall) -> str:
    for name in call.operator.signature.parameters:
        return name
    raise DataflowPrimFuncLoweringError(f"operator {call.name!r} must have an input parameter")


def normalize_dtype(dtype: str | None) -> str | None:
    return normalize_dtype_name(dtype)
