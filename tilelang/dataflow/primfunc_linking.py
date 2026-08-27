"""Link lowered Dataflow PrimFunc device handlers into wrapper handler adapters."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tvm import tir

from .handler_abi import DataflowFieldLayout, DataflowHandlerABI
from .handler_codegen import (
    DataflowHandlerCodegenModule,
    DataflowHandlerCodegenArtifact,
    DataflowHandlerParam,
    DataflowHandlerParamRole,
)
from .handler_identity import (
    DataflowHandlerVariantKey,
    REDUCE_ARITY_BINARY,
    REDUCE_ARITY_GENERIC,
    REDUCE_ARITY_PASSTHROUGH,
)
from .ir import DataflowReducerContract
from .primfunc_lowering import (
    DataflowPrimFuncHandler,
    DataflowPrimFuncLoweringResult,
    codegen_dataflow_primfunc_link_module,
)
from .semantic_config import DataflowSemanticConfig
from .wrapper import DataflowHandlerSource, DataflowWrapperSpec


@dataclass(frozen=True)
class DataflowPrimFuncLinkedHandlers:
    codegen_module: DataflowHandlerCodegenModule
    helper_source: str
    handler_sources: tuple[DataflowHandlerSource, ...]


class DataflowPrimFuncLinkingError(ValueError):
    """Raised when structured handler metadata cannot satisfy the wrapper ABI."""


def link_primfunc_handlers_for_wrapper(
    lowering: DataflowPrimFuncLoweringResult,
    wrapper_spec: DataflowWrapperSpec,
    abi: DataflowHandlerABI,
    *,
    semantic_config: DataflowSemanticConfig | None = None,
    terminal_finalize_variants: frozenset[DataflowHandlerVariantKey] = frozenset(),
) -> DataflowPrimFuncLinkedHandlers:
    semantic_config = semantic_config or DataflowSemanticConfig()
    handlers_by_id = {handler.handler_id: handler for handler in lowering.handlers}
    artifacts_by_id = {artifact.handler_id: artifact for artifact in lowering.codegen_artifacts}
    if set(artifacts_by_id) != set(handlers_by_id):
        raise DataflowPrimFuncLinkingError(
            "PrimFunc linking requires exactly one codegen artifact per handler: "
            f"handlers={sorted(handlers_by_id)!r}, artifacts={sorted(artifacts_by_id)!r}"
        )
    for handler in wrapper_spec.handlers:
        lowered_handler = handlers_by_id[handler.handler_id]
        identity = handler.handler_identity
        variant_key = handler.handler_variant_key
        if identity is None or variant_key is None:
            raise DataflowPrimFuncLinkingError(
                f"PrimFunc linking requires structured identity and variant key for handler {handler.handler_id}"
            )
        if lowered_handler.handler_identity != identity:
            raise DataflowPrimFuncLinkingError(f"PrimFunc handler identity mismatch for handler {handler.handler_id}")
        if lowered_handler.handler_variant_key != variant_key:
            raise DataflowPrimFuncLinkingError(f"PrimFunc handler variant mismatch for handler {handler.handler_id}")
        validate_handler_codegen_artifact(
            lowered_handler,
            artifacts_by_id[handler.handler_id],
            abi,
            abi.layouts_for_identity(identity),
            abi.input_layouts_for_identity(identity),
        )
    wrapper_thread_count = lowering.max_thread_count
    codegen_module = codegen_dataflow_primfunc_link_module(
        lowering,
        wrapper_thread_count=wrapper_thread_count,
        handler_scratch_base_offset=wrapper_spec.primfunc_scratch_offset,
        handler_scratch_offsets=dict(wrapper_spec.primfunc_handler_scratch_offsets),
        non_restrict_output_symbols=set(wrapper_spec.primfunc_scratch_backed_iter_symbols),
    )
    helper_source = generate_helper_source(abi.max_reduce_input_slots, wrapper_spec)
    sources = []
    for handler in wrapper_spec.handlers:
        lowered_handler = handlers_by_id[handler.handler_id]
        identity = handler.handler_identity
        assert identity is not None and handler.handler_variant_key is not None
        fields = abi.layouts_for_identity(identity)
        input_fields = abi.input_layouts_for_identity(identity)
        artifact = artifacts_by_id[handler.handler_id]
        symbol = adapter_call_symbol(lowered_handler)
        if handler.operator_kind in {"iter", "map"}:
            tensor_indices = abi.tensor_indices_for_identity(identity)
            body = iter_adapter(
                symbol,
                fields,
                input_fields,
                tensor_indices,
                artifact.params,
                lowered_handler.thread_count,
                wrapper_spec.primfunc_use_global_slot_fields,
                lowered_handler.contiguous_map_input_count,
                restore_cross_handler_registers=(
                    requires_cross_handler_register_restore(
                        lowered_handler,
                        artifact,
                    )
                ),
                lowered_handler=lowered_handler,
                wrapper_spec=wrapper_spec,
            )
        elif handler.operator_kind == "reduce":
            body = reduce_adapter(
                symbol,
                fields,
                lowered_handler.reduce_input_slot_count or abi.max_reduce_input_slots,
                artifact.params,
                lowered_handler.thread_count,
                wrapper_spec.primfunc_use_global_slot_fields,
                semantic_config.direct_slot_seed_reduce,
                reducer_contract=lowered_handler.reducer_contract,
                arity_class=(
                    None if handler.handler_variant_key.reduce_arity is None else handler.handler_variant_key.reduce_arity.arity_class
                ),
            )
        elif handler.operator_kind == "finalize":
            tensor_indices = abi.tensor_indices_for_identity(identity)
            skip_finalize_post_sync = semantic_config.skip_finalize_post_sync or handler.handler_variant_key in terminal_finalize_variants
            body = finalize_adapter(
                symbol,
                fields,
                tensor_indices,
                artifact.params,
                lowered_handler.thread_count,
                wrapper_spec.primfunc_use_global_slot_fields,
                skip_finalize_post_sync,
            )
        else:
            raise NotImplementedError(f"cannot link PrimFunc handler kind {handler.operator_kind!r}")
        completion_synchronized = lowered_handler.thread_count > 1
        if handler.operator_kind == "finalize" and (
            semantic_config.skip_finalize_post_sync or handler.handler_variant_key in terminal_finalize_variants
        ):
            completion_synchronized = False
        sources.append(
            DataflowHandlerSource(
                handler_id=handler.handler_id,
                body_source=body,
                completion_synchronized=completion_synchronized,
            )
        )
    return DataflowPrimFuncLinkedHandlers(
        codegen_module=codegen_module,
        helper_source=helper_source,
        handler_sources=tuple(sources),
    )


def generate_helper_source(
    max_reduce_input_slots: int,
    wrapper_spec: DataflowWrapperSpec,
) -> str:
    global_slot_field_helpers = (
        ""
        if not wrapper_spec.primfunc_use_global_slot_fields
        else """

template <typename T>
TL_DEVICE T *dataflow_primfunc_slot_global_field(
    void *global_base,
    const tl::DataflowSlot &slot,
    uint32_t byte_offset) {
  return reinterpret_cast<T *>(
      reinterpret_cast<uint8_t *>(tl::dataflow_slot_global_ptr(global_base, slot)) +
      byte_offset);
}

template <typename T>
TL_DEVICE const T *dataflow_primfunc_slot_global_field(
    const void *global_base,
    const tl::DataflowSlot &slot,
    uint32_t byte_offset) {
  return reinterpret_cast<const T *>(
      reinterpret_cast<const uint8_t *>(tl::dataflow_slot_global_ptr(global_base, slot)) +
      byte_offset);
}
"""
    )
    return f"""template <typename T>
TL_DEVICE T *dataflow_primfunc_slot_field(
    void *shared_base,
    const tl::DataflowSlot &slot,
    uint32_t byte_offset) {{
  return reinterpret_cast<T *>(
      reinterpret_cast<uint8_t *>(tl_dataflow_generated::dataflow_slot_shared_ptr(shared_base, slot)) +
      byte_offset);
}}

template <typename T>
TL_DEVICE const T *dataflow_primfunc_slot_field(
    const void *shared_base,
    const tl::DataflowSlot &slot,
    uint32_t byte_offset) {{
  return reinterpret_cast<const T *>(
      reinterpret_cast<const uint8_t *>(tl_dataflow_generated::dataflow_slot_shared_ptr(shared_base, slot)) +
      byte_offset);
}}

template <typename T>
TL_DEVICE T *dataflow_primfunc_handoff_field(
    void *shared_base,
    uint32_t byte_offset) {{
  return reinterpret_cast<T *>(
      reinterpret_cast<uint8_t *>(shared_base) -
      kDataflowSharedSlotBaseOffset + byte_offset);
}}

{global_slot_field_helpers}
static constexpr uint32_t kDataflowPrimFuncMaxReduceInputs = {max_reduce_input_slots}u;
"""


def iter_adapter(
    symbol: str,
    fields: tuple[DataflowFieldLayout, ...],
    input_fields: tuple[DataflowFieldLayout, ...],
    tensor_indices: dict[str, int],
    params: tuple[DataflowHandlerParam, ...],
    thread_count: int,
    use_global_slot_fields: bool,
    contiguous_map_input_count: int = 0,
    *,
    restore_cross_handler_registers: bool,
    lowered_handler: DataflowPrimFuncHandler,
    wrapper_spec: DataflowWrapperSpec,
) -> str:
    max_tensor_index = max(tensor_indices.values(), default=0)
    contiguous_map_input = contiguous_map_input_param(
        params,
        input_fields,
        contiguous_map_input_count,
    )
    if contiguous_map_input is not None and use_global_slot_fields:
        raise NotImplementedError("linked contiguous PrimFunc map inputs require shared slots")
    map_input_buffers = {} if contiguous_map_input is not None else map_input_params(params, input_fields)
    required_input_slot_count = (
        contiguous_map_input_count
        if contiguous_map_input is not None
        else max((slot_index for slot_index, _ in map_input_buffers.values()), default=-1) + 1
    )
    loaded_tensor_pointer_params = tuple(param for param in params if param.role is DataflowHandlerParamRole.TENSOR_ARG)
    tensor_declarations = [tensor_arg_pointer_declaration(param) for param in loaded_tensor_pointer_params]
    if use_global_slot_fields:
        input_slot_declarations = [
            f"  const uint32_t input_slot{slot_index}_id = "
            f"input_slots[handler_args.input_slot_offset + {slot_index}u];\n"
            f"  const tl::DataflowSlot &input_slot{slot_index} = slots[input_slot{slot_index}_id];"
            for slot_index in range(required_input_slot_count)
        ]
        map_input_declarations = [
            f"  const {field.c_type} *{map_input_pointer_name(slot_index, field, input_fields)}_shared = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, input_slot{slot_index}, {field.offset}u);\n"
            f"  const {field.c_type} *{map_input_pointer_name(slot_index, field, input_fields)}_global = "
            f"dataflow_primfunc_slot_global_field<{field.c_type}>(global_base, input_slot{slot_index}, {field.offset}u);\n"
            f"  const {field.c_type} *{map_input_pointer_name(slot_index, field, input_fields)} = "
            f"(_dataflow_use_global_slots && tl::dataflow_slot_uses_hbm_direct_global(input_slot{slot_index})) ? "
            f"{map_input_pointer_name(slot_index, field, input_fields)}_global : "
            f"{map_input_pointer_name(slot_index, field, input_fields)}_shared;"
            for slot_index, field in map_input_buffers.values()
        ]
    else:
        input_slot_declarations = []
        map_input_declarations = []
    if use_global_slot_fields:
        output_declarations = [
            f"  {field.c_type} *{output_pointer_name(field, fields)}_shared = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, output_slot, {field.offset}u);\n"
            f"  {field.c_type} *{output_pointer_name(field, fields)}_global = "
            f"dataflow_primfunc_slot_global_field<{field.c_type}>(global_base, output_slot, {field.offset}u);\n"
            f"  {field.c_type} *{output_pointer_name(field, fields)} = "
            f"(_dataflow_use_global_slots && tl::dataflow_slot_uses_hbm_direct_global(output_slot)) ? "
            f"{output_pointer_name(field, fields)}_global : "
            f"{output_pointer_name(field, fields)}_shared;"
            for field in fields
        ]
        output_global_to_shared_copies = []
    else:
        output_declarations = [
            f"  {field.c_type} *{output_pointer_name(field, fields)} = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, output_slot, {field.offset}u);"
            for field in fields
        ]
        output_global_to_shared_copies = []
    output_args = {
        param.name: output_pointer_name(param_field(param, fields), fields)
        for param in params
        if param.role is DataflowHandlerParamRole.OUTPUT_SLOT_FIELD
    }
    if contiguous_map_input is not None:
        param, field = contiguous_map_input
        map_input_args = {
            param.name: (
                f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, "
                "slots[input_slots[handler_args.input_slot_offset]], "
                f"{field.offset}u)"
            )
        }
    elif use_global_slot_fields:
        map_input_args = {
            buffer_name: map_input_pointer_name(slot_index, field, input_fields)
            for buffer_name, (slot_index, field) in map_input_buffers.items()
        }
    else:
        map_input_args = {
            buffer_name: (
                f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, "
                f"slots[input_slots[handler_args.input_slot_offset + {slot_index}u]], "
                f"{field.offset}u)"
            )
            for buffer_name, (slot_index, field) in map_input_buffers.items()
        }
    tensor_args_by_name = {param.name: f"{param.name}_tensor" for param in loaded_tensor_pointer_params}
    handoff_transfer_args = build_handoff_transfer_args(
        params,
        lowered_handler,
        wrapper_spec,
    )
    call_args = handler_call_args(
        params,
        map_input_args | tensor_args_by_name | output_args | handoff_transfer_args | structured_scalar_args(params),
    )
    required_task_coord_count = count_required_task_coords(params)
    task_coord_guard = f" ||\n      handler_args.task_coord_count < {required_task_coord_count}u" if required_task_coord_count else ""
    handoff_guard = handoff_adapter_guard(
        params,
        lowered_handler,
        wrapper_spec,
    )
    uses_task_coords = any(
        param.role
        in {
            DataflowHandlerParamRole.TASK_COORD,
            DataflowHandlerParamRole.NEXT_TASK_COORD,
        }
        for param in params
    )
    task_coords_unused = "" if uses_task_coords else "\n  (void)task_coords;"
    input_slots_unused = "" if required_input_slot_count else "\n  (void)input_slots;"
    leader_guard = "!tl::dataflow_is_leader_thread() ||\n      " if thread_count == 1 else ""
    post_call_sync = "\n  __syncthreads();" if thread_count > 1 else ""
    if restore_cross_handler_registers:
        if thread_count <= 1:
            raise DataflowPrimFuncLinkingError("cross-handler register restoration requires a collective handler")
        if thread_count % 128:
            raise DataflowPrimFuncLinkingError(
                f"cross-handler register restoration requires a warp-group-aligned physical thread count, got {thread_count}"
            )
        post_call_sync += f"\n  if (static_cast<uint32_t>(threadIdx.x) < {thread_count}u) {{\n    tl::warpgroup_reg_alloc<128>();\n  }}"
    global_base_unused = "" if use_global_slot_fields else "\n  (void)global_base;"
    use_global_decl = (
        "\n  const bool _dataflow_use_global_slots = tl_dataflow_generated::kDataflowPrimFuncUseGlobalSlotFields;"
        if use_global_slot_fields
        else ""
    )
    output_guard = "handler_args.output_slot == tl::kDataflowInvalidIndex ||\n      " if fields else ""
    output_slot_declaration = "  const tl::DataflowSlot &output_slot = slots[handler_args.output_slot];" if fields else ""
    return f"""  (void)inst;
{input_slots_unused}{task_coords_unused}{global_base_unused}
  if ({leader_guard}{output_guard}handler_args.input_slot_count < {required_input_slot_count}u ||
      tl_dataflow_generated::kDataflowTensorArgCount <= {max_tensor_index}u{task_coord_guard}{handoff_guard}) {{
    return;
  }}
{use_global_decl}
{chr(10).join(tensor_declarations)}
{chr(10).join(input_slot_declarations)}
{chr(10).join(map_input_declarations)}
{output_slot_declaration}
{chr(10).join(output_declarations)}
  {symbol}({", ".join(call_args)});
{chr(10).join(output_global_to_shared_copies)}{post_call_sync}"""


def tensor_arg_pointer_declaration(param: DataflowHandlerParam) -> str:
    assert param.tensor_arg_index is not None
    const_prefix = "const " if param.is_const else ""
    pointer_type = f"{const_prefix}{param.c_type}"
    return f"""  {pointer_type} *{param.name}_tensor = reinterpret_cast<{pointer_type} *>(
      static_cast<uintptr_t>(tensor_args[{param.tensor_arg_index}].data_ptr));"""


def reduce_adapter(
    symbol: str,
    fields: tuple[DataflowFieldLayout, ...],
    max_reduce_input_slots: int,
    params: tuple[DataflowHandlerParam, ...],
    thread_count: int,
    use_global_slot_fields: bool,
    direct_slot_seed_reduce: bool,
    *,
    reducer_contract: DataflowReducerContract | None,
    arity_class: str | None,
) -> str:
    if reducer_contract is DataflowReducerContract.ASSOCIATIVE_BINARY:
        if arity_class == REDUCE_ARITY_PASSTHROUGH:
            if max_reduce_input_slots != 1:
                raise DataflowPrimFuncLinkingError("associative passthrough handler requires one input slot")
            return single_input_reduce_adapter(
                symbol,
                fields,
                thread_count,
                use_global_slot_fields,
            )
        if arity_class not in {REDUCE_ARITY_BINARY, REDUCE_ARITY_GENERIC}:
            raise DataflowPrimFuncLinkingError(f"unsupported associative reduce arity class {arity_class!r}")
    if max_reduce_input_slots == 1:
        return single_input_reduce_adapter(
            symbol,
            fields,
            thread_count,
            use_global_slot_fields,
        )
    if not uses_direct_reduce_slot_params(params, fields, max_reduce_input_slots):
        raise NotImplementedError("linked PrimFunc multi-input reduce requires direct physical slot field buffers")
    return direct_slot_reduce_adapter(
        symbol,
        fields,
        max_reduce_input_slots,
        params,
        thread_count,
        use_global_slot_fields,
        (direct_slot_seed_reduce if reducer_contract is not DataflowReducerContract.ASSOCIATIVE_BINARY else False),
        arity_class=(arity_class if reducer_contract is DataflowReducerContract.ASSOCIATIVE_BINARY else None),
    )


def direct_slot_reduce_adapter(
    symbol: str,
    fields: tuple[DataflowFieldLayout, ...],
    max_reduce_input_slots: int,
    params: tuple[DataflowHandlerParam, ...],
    thread_count: int,
    use_global_slot_fields: bool,
    direct_slot_seed_reduce: bool,
    *,
    arity_class: str | None,
) -> str:
    input_slot_declarations = [
        "  const uint32_t input_slot0_id = handler_args.input_slot_count == 0u\n"
        "      ? handler_args.output_slot\n"
        "      : input_slots[handler_args.input_slot_offset];\n"
        "  const tl::DataflowSlot &input_slot0 = slots[input_slot0_id];"
    ]
    input_slot_declarations.extend(
        f"  const uint32_t input_slot{slot_index}_id = handler_args.input_slot_count > {slot_index}u\n"
        f"      ? input_slots[handler_args.input_slot_offset + {slot_index}u]\n"
        "      : input_slot0_id;\n"
        f"  const tl::DataflowSlot &input_slot{slot_index} = slots[input_slot{slot_index}_id];"
        for slot_index in range(1, max_reduce_input_slots)
    )
    if use_global_slot_fields:
        input_declarations = [
            f"  const {field.c_type} *{direct_reduce_item_pointer_name(slot_index, field, fields)}_shared = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, input_slot{slot_index}, {field.offset}u);\n"
            f"  const {field.c_type} *{direct_reduce_item_pointer_name(slot_index, field, fields)}_global = "
            f"dataflow_primfunc_slot_global_field<{field.c_type}>(global_base, input_slot{slot_index}, {field.offset}u);\n"
            f"  const {field.c_type} *{direct_reduce_item_pointer_name(slot_index, field, fields)} = "
            f"(_dataflow_use_global_slots && tl::dataflow_slot_uses_hbm_direct_global(input_slot{slot_index})) ? "
            f"{direct_reduce_item_pointer_name(slot_index, field, fields)}_global : "
            f"{direct_reduce_item_pointer_name(slot_index, field, fields)}_shared;"
            for slot_index in range(max_reduce_input_slots)
            for field in fields
        ]
        output_declarations = [
            f"  {field.c_type} *{output_pointer_name(field, fields)}_shared = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, output_slot, {field.offset}u);\n"
            f"  {field.c_type} *{output_pointer_name(field, fields)}_global = "
            f"dataflow_primfunc_slot_global_field<{field.c_type}>(global_base, output_slot, {field.offset}u);\n"
            f"  {field.c_type} *{output_pointer_name(field, fields)} = "
            f"(_dataflow_use_global_slots && tl::dataflow_slot_uses_hbm_direct_global(output_slot)) ? "
            f"{output_pointer_name(field, fields)}_global : "
            f"{output_pointer_name(field, fields)}_shared;"
            for field in fields
        ]
        output_global_to_shared_copies = []
    else:
        input_declarations = [
            f"  const {field.c_type} *{direct_reduce_item_pointer_name(slot_index, field, fields)} = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, input_slot{slot_index}, {field.offset}u);"
            for slot_index in range(max_reduce_input_slots)
            for field in fields
        ]
        output_declarations = [
            f"  {field.c_type} *{output_pointer_name(field, fields)} = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, output_slot, {field.offset}u);"
            for field in fields
        ]
        output_global_to_shared_copies = []
    input_args = {
        param.name: direct_reduce_item_pointer_name(
            required_slot_index(param),
            param_field(param, fields),
            fields,
        )
        for param in params
        if param.role is DataflowHandlerParamRole.INPUT_SLOT_FIELD
    }
    output_args = {
        param.name: output_pointer_name(param_field(param, fields), fields)
        for param in params
        if param.role is DataflowHandlerParamRole.OUTPUT_SLOT_FIELD
    }
    call_args = handler_call_args(
        params,
        input_args | output_args | structured_scalar_args(params),
    )
    leader_guard = "!tl::dataflow_is_leader_thread() ||\n      " if thread_count == 1 else ""
    post_call_sync = "\n  __syncthreads();" if thread_count > 1 else ""
    global_base_unused = "" if use_global_slot_fields else "\n  (void)global_base;"
    use_global_decl = (
        "\n  const bool _dataflow_use_global_slots = tl_dataflow_generated::kDataflowPrimFuncUseGlobalSlotFields;"
        if use_global_slot_fields
        else ""
    )
    direct_seed_fast_path = ""
    if direct_slot_seed_reduce:
        field_copies = [
            f"""  for (uint32_t j = static_cast<uint32_t>(threadIdx.x); j < {field.numel}u; j += {thread_count}u) {{
    {output_pointer_name(field, fields)}[j] = {direct_reduce_item_pointer_name(0, field, fields)}[j];
  }}"""
            for field in fields
        ]
        direct_seed_fast_path = f"""
  // DATAFLOW direct-slot seed reduce fast path.
  if (handler_args.input_slot_count == 1u) {{
{chr(10).join(field_copies)}
{chr(10).join(output_global_to_shared_copies)}{post_call_sync}
    return;
  }}"""
    input_count_guard = (
        "handler_args.input_slot_count != 2u"
        if arity_class == REDUCE_ARITY_BINARY
        else (
            f"handler_args.input_slot_count < 3u ||\n      handler_args.input_slot_count > {max_reduce_input_slots}u"
            if arity_class == REDUCE_ARITY_GENERIC
            else "handler_args.input_slot_count > kDataflowPrimFuncMaxReduceInputs"
        )
    )
    return f"""  (void)inst;
  (void)tensor_args;
  (void)task_coords;{global_base_unused}
  if ({leader_guard}handler_args.output_slot == tl::kDataflowInvalidIndex ||
      {input_count_guard}) {{
    return;
  }}
{use_global_decl}
  const tl::DataflowSlot &output_slot = slots[handler_args.output_slot];
{chr(10).join(input_slot_declarations)}
{chr(10).join(input_declarations)}
{chr(10).join(output_declarations)}
{direct_seed_fast_path}
  {symbol}({", ".join(call_args)});
{chr(10).join(output_global_to_shared_copies)}{post_call_sync}"""


def single_input_reduce_adapter(
    symbol: str,
    fields: tuple[DataflowFieldLayout, ...],
    thread_count: int,
    use_global_slot_fields: bool,
) -> str:
    del symbol
    leader_guard = "!tl::dataflow_is_leader_thread() ||\n      " if thread_count == 1 else ""
    post_call_sync = "\n  __syncthreads();" if thread_count > 1 else ""
    if use_global_slot_fields:
        input_declarations = [
            f"  const {field.c_type} *{reduce_items_name(field, fields)}_shared = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, input_slot, {field.offset}u);\n"
            f"  {field.c_type} *{reduce_items_name(field, fields)}_global = "
            f"dataflow_primfunc_slot_global_field<{field.c_type}>(global_base, input_slot, {field.offset}u);\n"
            f"  const {field.c_type} *{reduce_items_name(field, fields)} = "
            f"(_dataflow_use_global_slots && tl::dataflow_slot_uses_hbm_direct_global(input_slot)) ? "
            f"{reduce_items_name(field, fields)}_global : "
            f"{reduce_items_name(field, fields)}_shared;"
            for field in fields
        ]
        output_declarations = [
            f"  {field.c_type} *{output_pointer_name(field, fields)}_shared = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, output_slot, {field.offset}u);\n"
            f"  {field.c_type} *{output_pointer_name(field, fields)}_global = "
            f"dataflow_primfunc_slot_global_field<{field.c_type}>(global_base, output_slot, {field.offset}u);\n"
            f"  {field.c_type} *{output_pointer_name(field, fields)} = "
            f"(_dataflow_use_global_slots && tl::dataflow_slot_uses_hbm_direct_global(output_slot)) ? "
            f"{output_pointer_name(field, fields)}_global : "
            f"{output_pointer_name(field, fields)}_shared;"
            for field in fields
        ]
    else:
        input_declarations = [
            f"  const {field.c_type} *{reduce_items_name(field, fields)} = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, input_slot, {field.offset}u);"
            for field in fields
        ]
        output_declarations = [
            f"  {field.c_type} *{output_pointer_name(field, fields)} = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, output_slot, {field.offset}u);"
            for field in fields
        ]
    field_copies = [
        f"""  if ({output_pointer_name(field, fields)} != {reduce_items_name(field, fields)}) {{
    for (uint32_t j = static_cast<uint32_t>(threadIdx.x); j < {field.numel}u; j += {thread_count}u) {{
      {output_pointer_name(field, fields)}[j] = {reduce_items_name(field, fields)}[j];
    }}
  }}"""
        for field in fields
    ]
    output_global_to_shared_copies = []
    use_global_decl = (
        "\n  const bool _dataflow_use_global_slots = tl_dataflow_generated::kDataflowPrimFuncUseGlobalSlotFields;"
        if use_global_slot_fields
        else ""
    )
    global_base_unused = "" if use_global_slot_fields else "\n  (void)global_base;"
    return f"""  (void)inst;
  (void)tensor_args;
  (void)task_coords;{global_base_unused}
  if ({leader_guard}handler_args.output_slot == tl::kDataflowInvalidIndex ||
      handler_args.input_slot_count > 1u) {{
    return;
  }}
  if (handler_args.input_slot_count == 0u) {{
    return;
  }}
{use_global_decl}
  const uint32_t slot_id = input_slots[handler_args.input_slot_offset];
  const tl::DataflowSlot &input_slot = slots[slot_id];
  const tl::DataflowSlot &output_slot = slots[handler_args.output_slot];
{chr(10).join(input_declarations)}
{chr(10).join(output_declarations)}
{chr(10).join(field_copies)}
{chr(10).join(output_global_to_shared_copies)}{post_call_sync}"""


def finalize_adapter(
    symbol: str,
    fields: tuple[DataflowFieldLayout, ...],
    tensor_indices: dict[str, int],
    params: tuple[DataflowHandlerParam, ...],
    thread_count: int,
    use_global_slot_fields: bool,
    skip_post_sync: bool,
) -> str:
    max_tensor_index = max(tensor_indices.values(), default=0)
    inter_params = tuple(param for param in params if param.role is DataflowHandlerParamRole.INPUT_SLOT_FIELD)
    inter_fields = tuple(param_field(param, fields) for param in inter_params)
    required_input_count = max(
        (0 if param.slot_index is None else param.slot_index + 1 for param in inter_params),
        default=0,
    )
    inter_pointer_names = tuple(
        (inter_pointer_name(field, fields) if required_input_count == 1 else param.name[:1].lower() + param.name[1:])
        for param, field in zip(inter_params, inter_fields)
    )
    input_slot_declarations = [
        f"  const uint32_t slot_id_{slot_index} = "
        f"input_slots[handler_args.input_slot_offset + {slot_index}u];\n"
        f"  const tl::DataflowSlot &input_slot_{slot_index} = slots[slot_id_{slot_index}];"
        for slot_index in range(required_input_count)
    ]
    output_tensors = tuple(param for param in params if param.role is DataflowHandlerParamRole.TENSOR_ARG)
    if use_global_slot_fields:
        inter_declarations = [
            f"  const {field.c_type} *{pointer_name}_shared = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, "
            f"input_slot_{param.slot_index or 0}, {field.offset}u);\n"
            f"  {field.c_type} *{pointer_name}_global = "
            f"dataflow_primfunc_slot_global_field<{field.c_type}>(global_base, "
            f"input_slot_{param.slot_index or 0}, {field.offset}u);\n"
            f"  const {field.c_type} *{pointer_name} = "
            f"(_dataflow_use_global_slots && tl::dataflow_slot_uses_hbm_direct_global("
            f"input_slot_{param.slot_index or 0})) ? "
            f"{pointer_name}_global : {pointer_name}_shared;"
            for param, field, pointer_name in zip(
                inter_params,
                inter_fields,
                inter_pointer_names,
            )
        ]
    else:
        inter_declarations = [
            f"  const {field.c_type} *{pointer_name} = "
            f"dataflow_primfunc_slot_field<{field.c_type}>(shared_base, "
            f"input_slot_{param.slot_index or 0}, {field.offset}u);"
            for param, field, pointer_name in zip(
                inter_params,
                inter_fields,
                inter_pointer_names,
            )
        ]
    output_declarations = [
        f"""  {param.c_type} *{param.name}_tensor = reinterpret_cast<{param.c_type} *>(
      static_cast<uintptr_t>(tensor_args[{param.tensor_arg_index}].data_ptr));"""
        for param in output_tensors
    ]
    inter_args = {param.name: pointer_name for param, pointer_name in zip(inter_params, inter_pointer_names)}
    output_args = {param.name: f"{param.name}_tensor" for param in output_tensors}
    call_args = handler_call_args(
        params,
        inter_args | output_args | structured_scalar_args(params),
    )
    required_task_coord_count = count_required_task_coords(params)
    task_coord_guard = f" ||\n      handler_args.task_coord_count < {required_task_coord_count}u" if required_task_coord_count else ""
    task_coords_unused = "" if required_task_coord_count else "\n  (void)task_coords;"
    leader_guard = "!tl::dataflow_is_leader_thread() ||\n      " if thread_count == 1 else ""
    post_call_sync = "\n  __syncthreads();" if thread_count > 1 and not skip_post_sync else ""
    global_base_unused = "" if use_global_slot_fields else "\n  (void)global_base;"
    use_global_decl = (
        "\n  const bool _dataflow_use_global_slots = tl_dataflow_generated::kDataflowPrimFuncUseGlobalSlotFields;"
        if use_global_slot_fields
        else ""
    )
    return f"""  (void)inst;
{task_coords_unused}{global_base_unused}
  if ({leader_guard}handler_args.input_slot_count < {required_input_count}u ||
      tl_dataflow_generated::kDataflowTensorArgCount <= {max_tensor_index}u{task_coord_guard}) {{
    return;
  }}
{use_global_decl}
{chr(10).join(input_slot_declarations)}
{chr(10).join(inter_declarations)}
{chr(10).join(output_declarations)}
  {symbol}({", ".join(call_args)});{post_call_sync}"""


def validate_handler_codegen_artifact(
    handler: DataflowPrimFuncHandler,
    artifact: DataflowHandlerCodegenArtifact,
    abi: DataflowHandlerABI,
    fields: tuple[DataflowFieldLayout, ...],
    input_fields: tuple[DataflowFieldLayout, ...],
) -> None:
    if artifact.handler_id != handler.handler_id or artifact.device_symbol != handler.device_symbol:
        raise DataflowPrimFuncLinkingError(
            "handler codegen artifact identity mismatch: "
            f"handler=({handler.handler_id}, {handler.device_symbol!r}), "
            f"artifact=({artifact.handler_id}, {artifact.device_symbol!r})"
        )
    if artifact.global_symbol != handler.global_symbol:
        raise DataflowPrimFuncLinkingError(
            f"handler {handler.handler_id} global symbol mismatch: {artifact.global_symbol!r} != {handler.global_symbol!r}"
        )
    if artifact.thread_count != handler.thread_count:
        raise DataflowPrimFuncLinkingError(
            f"handler {handler.handler_id} thread metadata mismatch: {artifact.thread_count} != {handler.thread_count}"
        )
    if artifact.dynamic_shared_bytes != handler.dynamic_shared_bytes:
        raise DataflowPrimFuncLinkingError(
            f"handler {handler.handler_id} dynamic-shared metadata mismatch: "
            f"{artifact.dynamic_shared_bytes} != {handler.dynamic_shared_bytes}"
        )

    tensor_indices = abi.tensor_indices_for_identity(handler.handler_identity)
    allowed_tensor_indices = set(tensor_indices.values())
    if handler.cross_handler_handoff_role == "producer" and handler.cross_handler_handoff_plan is not None:
        consumer_stage_id = handler.cross_handler_handoff_plan.consumer_stage_id
        allowed_tensor_indices.update(
            binding.tensor_index for binding in abi.tensor_arg_plan.bindings if binding.handler_identity.operator_id == consumer_stage_id
        )
    field_names = {field.name for field in fields}
    input_field_names = {field.name for field in input_fields}
    task_coord_axes: list[int] = []
    roles = {param.role for param in artifact.params}
    for param in artifact.params:
        if param.role is DataflowHandlerParamRole.TENSOR_ARG:
            if param.tensor_arg_index not in allowed_tensor_indices:
                raise DataflowPrimFuncLinkingError(
                    f"handler {handler.handler_id} tensor parameter {param.name!r} has "
                    f"unknown tensor_arg_index={param.tensor_arg_index!r}; "
                    f"available={sorted(allowed_tensor_indices)!r}"
                )
        elif param.role is DataflowHandlerParamRole.TASK_COORD:
            assert param.task_coord_axis is not None
            task_coord_axes.append(param.task_coord_axis)
        elif param.role is DataflowHandlerParamRole.INPUT_SLOT_FIELD:
            allowed_fields = input_field_names if handler.operator_kind == "map" else field_names
            if handler.operator_kind == "iter" or param.field_name not in allowed_fields:
                raise DataflowPrimFuncLinkingError(
                    f"handler {handler.handler_id} has invalid input-slot parameter "
                    f"{param.to_dict()!r} for operator kind {handler.operator_kind!r}"
                )
        elif param.role is DataflowHandlerParamRole.OUTPUT_SLOT_FIELD and (
            handler.operator_kind not in {"iter", "map", "reduce"} or param.field_name not in field_names
        ):
            raise DataflowPrimFuncLinkingError(
                f"handler {handler.handler_id} has invalid output-slot parameter "
                f"{param.to_dict()!r} for operator kind {handler.operator_kind!r}"
            )

    expected_task_coord_axes = list(range(len(handler.task_param_names)))
    invalid_task_coord_axes = sorted(set(task_coord_axes) - set(expected_task_coord_axes))
    if invalid_task_coord_axes:
        raise DataflowPrimFuncLinkingError(
            f"handler {handler.handler_id} task-coordinate axes {task_coord_axes!r} are outside declared axes {expected_task_coord_axes!r}"
        )
    required_roles = {
        "iter": {
            DataflowHandlerParamRole.RANGE_BEGIN,
            DataflowHandlerParamRole.RANGE_END,
            DataflowHandlerParamRole.TASK_ID,
        },
        "map": set(),
        "reduce": {
            DataflowHandlerParamRole.INPUT_COUNT,
            DataflowHandlerParamRole.TASK_ID,
        },
        "finalize": {DataflowHandlerParamRole.TASK_ID},
    }[handler.operator_kind]
    missing_roles = required_roles - roles
    if missing_roles:
        raise DataflowPrimFuncLinkingError(
            f"handler {handler.handler_id} is missing required parameter roles {sorted(role.value for role in missing_roles)!r}"
        )
    logical_tensor_indices = {
        param.name: param.tensor_arg_index for param in handler.params if param.role is DataflowHandlerParamRole.TENSOR_ARG
    }
    for descriptor in artifact.tma_descriptors:
        if descriptor.tensor_name not in logical_tensor_indices:
            raise DataflowPrimFuncLinkingError(
                f"handler {handler.handler_id} TMA descriptor {descriptor.name!r} references "
                f"unknown tensor binding {descriptor.tensor_name!r}"
            )


def adapter_call_symbol(handler: DataflowPrimFuncHandler) -> str:
    return handler.device_symbol


def requires_cross_handler_register_restore(
    handler: DataflowPrimFuncHandler,
    artifact: DataflowHandlerCodegenArtifact,
) -> bool:
    plan = handler.cross_handler_handoff_plan
    if plan is None or not plan.enabled or handler.cross_handler_handoff_role is None:
        return False

    matched_functions = []
    for base_func in artifact.module.functions.values():
        if not isinstance(base_func, tir.PrimFunc) or not base_func.attrs:
            continue
        if str(base_func.attrs.get("global_symbol", "")) == artifact.device_symbol:
            matched_functions.append(base_func)
    if len(matched_functions) != 1:
        raise DataflowPrimFuncLinkingError(
            f"cross-handler register restoration requires exactly one lowered device function for {artifact.device_symbol!r}"
        )

    uses_set_max_nreg = False

    def visit(node: object) -> None:
        nonlocal uses_set_max_nreg
        if isinstance(node, tir.Call) and getattr(node.op, "name", None) == "tl.set_max_nreg":
            uses_set_max_nreg = True

    tir.stmt_functor.post_order_visit(matched_functions[0].body, visit)
    return uses_set_max_nreg


def contiguous_map_input_param(
    params: tuple[DataflowHandlerParam, ...],
    input_fields: tuple[DataflowFieldLayout, ...],
    input_count: int,
) -> tuple[DataflowHandlerParam, DataflowFieldLayout] | None:
    if input_count <= 0:
        return None
    if len(input_fields) != 1 or input_fields[0].shape is None:
        raise NotImplementedError("linked contiguous PrimFunc map inputs require exactly one tensor field")
    field = input_fields[0]
    if field.offset != 0:
        raise NotImplementedError("linked contiguous PrimFunc map input field must begin at byte offset zero")
    input_params = tuple(param for param in params if param.role is DataflowHandlerParamRole.INPUT_SLOT_FIELD)
    if not input_params:
        return None
    if len(input_params) != 1:
        raise DataflowPrimFuncLinkingError(
            "linked contiguous PrimFunc map input requires one structured slot parameter, "
            f"got {[param.to_dict() for param in input_params]!r}"
        )
    param = input_params[0]
    if param.field_name != field.name or param.slot_index != 0 or param.slot_extent != input_count:
        raise DataflowPrimFuncLinkingError(
            f"linked contiguous PrimFunc map input metadata does not match its slot layout: {param.to_dict()!r}, input_count={input_count}"
        )
    return param, field


def map_input_params(
    params: tuple[DataflowHandlerParam, ...],
    input_fields: tuple[DataflowFieldLayout, ...],
) -> dict[str, tuple[int, DataflowFieldLayout]]:
    result: dict[str, tuple[int, DataflowFieldLayout]] = {}
    for param in params:
        if param.role is not DataflowHandlerParamRole.INPUT_SLOT_FIELD:
            continue
        result[param.name] = (
            required_slot_index(param),
            param_field(param, input_fields),
        )
    return result


def map_input_pointer_name(
    slot_index: int,
    field: DataflowFieldLayout,
    input_fields: tuple[DataflowFieldLayout, ...],
) -> str:
    if len(input_fields) == 1:
        return f"item{slot_index}"
    return f"item{slot_index}_{c_identifier(field.name)}"


def uses_direct_reduce_slot_params(
    params: tuple[DataflowHandlerParam, ...],
    fields: tuple[DataflowFieldLayout, ...],
    max_reduce_input_slots: int,
) -> bool:
    expected = {(slot_index, field.name) for slot_index in range(max_reduce_input_slots) for field in fields}
    actual = {(required_slot_index(param), param.field_name) for param in params if param.role is DataflowHandlerParamRole.INPUT_SLOT_FIELD}
    return expected <= actual


def param_field(
    param: DataflowHandlerParam,
    fields: tuple[DataflowFieldLayout, ...],
) -> DataflowFieldLayout:
    for field in fields:
        if field.name == param.field_name:
            return field
    raise DataflowPrimFuncLinkingError(f"handler parameter {param.name!r} references unknown field {param.field_name!r}")


def required_slot_index(param: DataflowHandlerParam) -> int:
    if param.slot_index is None:
        raise DataflowPrimFuncLinkingError(f"input-slot parameter {param.name!r} is missing slot_index metadata")
    return param.slot_index


def handler_call_args(
    params: tuple[DataflowHandlerParam, ...],
    args_by_name: dict[str, str],
) -> list[str]:
    call_args = []
    for param in params:
        if param.role is DataflowHandlerParamRole.TMA_DESCRIPTOR:
            call_args.append(param.name)
            continue
        try:
            call_args.append(args_by_name[param.name])
        except KeyError as err:
            raise DataflowPrimFuncLinkingError(
                f"linked PrimFunc handler has no adapter value for structured parameter {param.to_dict()!r}"
            ) from err
    return call_args


def structured_scalar_args(
    params: tuple[DataflowHandlerParam, ...],
) -> dict[str, str]:
    result: dict[str, str] = {}
    fixed_roles = {
        DataflowHandlerParamRole.RANGE_BEGIN: "handler_args.range_begin",
        DataflowHandlerParamRole.RANGE_END: "handler_args.range_end",
        DataflowHandlerParamRole.INPUT_COUNT: "handler_args.input_slot_count",
        DataflowHandlerParamRole.TASK_ID: "handler_args.task_id",
    }
    for param in params:
        fixed = fixed_roles.get(param.role)
        if fixed is not None:
            result[param.name] = fixed
        elif param.role is DataflowHandlerParamRole.TASK_COORD:
            assert param.task_coord_axis is not None
            result[param.name] = task_coord_call_arg(param.task_coord_axis)
        elif param.role is DataflowHandlerParamRole.NEXT_TASK_COORD:
            assert param.task_coord_axis is not None
            result[param.name] = (
                "(handler_args.handoff_peer_arg_offset == tl::kDataflowInvalidIndex "
                "? 0u : task_coords[tl::dataflow_handler_args(arg_base, "
                "handler_args.handoff_peer_arg_offset).task_coord_offset + "
                f"{param.task_coord_axis}u])"
            )
        elif param.role is DataflowHandlerParamRole.HANDOFF_STAGE_COUNT:
            result[param.name] = "handler_args.handoff_stage_count"
        elif param.role is DataflowHandlerParamRole.HANDOFF_ARENA_SLOT:
            result[param.name] = "(handler_args.handoff_stage_count == 0u ? 0u : handler_args.handoff_arena_slot)"
        elif param.role in {
            DataflowHandlerParamRole.HANDOFF_PEER_RANGE_BEGIN,
            DataflowHandlerParamRole.HANDOFF_PEER_RANGE_END,
            DataflowHandlerParamRole.HANDOFF_PEER_TASK_ID,
        }:
            field = {
                DataflowHandlerParamRole.HANDOFF_PEER_RANGE_BEGIN: "range_begin",
                DataflowHandlerParamRole.HANDOFF_PEER_RANGE_END: "range_end",
                DataflowHandlerParamRole.HANDOFF_PEER_TASK_ID: "task_id",
            }[param.role]
            result[param.name] = (
                "(handler_args.handoff_peer_arg_offset == tl::kDataflowInvalidIndex "
                f"? 0u : tl::dataflow_handler_args(arg_base, handler_args.handoff_peer_arg_offset).{field})"
            )
    return result


def build_handoff_transfer_args(
    params: tuple[DataflowHandlerParam, ...],
    handler: DataflowPrimFuncHandler,
    wrapper_spec: DataflowWrapperSpec,
) -> dict[str, str]:
    transfer_params = tuple(param for param in params if param.role is DataflowHandlerParamRole.HANDOFF_TRANSFER_BUFFER)
    if not transfer_params:
        return {}
    plan = handler.cross_handler_handoff_plan
    if plan is None or not plan.enabled or handler.cross_handler_handoff_role is None:
        raise DataflowPrimFuncLinkingError("handoff transfer parameters require an enabled typed handler plan")
    arenas = tuple(arena for arena in wrapper_spec.handoff_plan_arenas if arena.plan_fingerprint == plan.fingerprint)
    if len(arenas) != 1:
        raise DataflowPrimFuncLinkingError(f"handler handoff plan {plan.fingerprint!r} lacks one wrapper arena")
    arena = arenas[0]
    transfers = {transfer.binding_index: transfer for transfer in plan.transfer_plans}
    result: dict[str, str] = {}
    for param in transfer_params:
        assert param.handoff_transfer_index is not None
        transfer = transfers.get(param.handoff_transfer_index)
        if transfer is None:
            raise DataflowPrimFuncLinkingError(f"handoff parameter {param.name!r} references an unknown transfer")
        stage_index = 0 if param.handoff_stage_index is None else param.handoff_stage_index
        if param.handoff_stage_index is not None and stage_index >= transfer.buffer_stages:
            raise DataflowPrimFuncLinkingError(f"handoff parameter {param.name!r} exceeds its staged transfer")
        slot_stride = transfer.bytes_per_stage * transfer.physical_buffer_stages
        absolute_offset = arena.offset + transfer.arena_offset
        stage_offset = stage_index * transfer.bytes_per_stage
        if transfer.arena_offset + transfer.lookahead_slots * slot_stride > arena.bytes:
            raise DataflowPrimFuncLinkingError(f"handoff transfer {transfer.binding_index} exceeds its wrapper arena")
        result[param.name] = (
            f"dataflow_primfunc_handoff_field<{param.c_type}>(shared_base, "
            f"{absolute_offset}u + "
            "(handler_args.handoff_stage_count == 0u ? 0u : "
            "handler_args.handoff_arena_slot) * "
            f"{slot_stride}u + {stage_offset}u)"
        )
    return result


def handoff_adapter_guard(
    params: tuple[DataflowHandlerParam, ...],
    handler: DataflowPrimFuncHandler,
    wrapper_spec: DataflowWrapperSpec,
) -> str:
    handoff_roles = {
        DataflowHandlerParamRole.NEXT_TASK_COORD,
        DataflowHandlerParamRole.HANDOFF_STAGE_COUNT,
        DataflowHandlerParamRole.HANDOFF_ARENA_SLOT,
        DataflowHandlerParamRole.HANDOFF_PEER_RANGE_BEGIN,
        DataflowHandlerParamRole.HANDOFF_PEER_RANGE_END,
        DataflowHandlerParamRole.HANDOFF_PEER_TASK_ID,
        DataflowHandlerParamRole.HANDOFF_TRANSFER_BUFFER,
    }
    if not any(param.role in handoff_roles for param in params):
        return ""
    plan = handler.cross_handler_handoff_plan
    if plan is None or not plan.enabled:
        raise DataflowPrimFuncLinkingError("physical handoff metadata requires an enabled typed plan")
    if not any(arena.plan_fingerprint == plan.fingerprint for arena in wrapper_spec.handoff_plan_arenas):
        raise DataflowPrimFuncLinkingError("physical handoff plan lacks an arena")

    clauses = [
        f"handler_args.handoff_stage_count > {plan.selected_buffer_stages}u",
    ]
    if any(param.role is DataflowHandlerParamRole.HANDOFF_TRANSFER_BUFFER for param in params):
        clauses.append(
            "(handler_args.handoff_stage_count != 0u && "
            "(handler_args.handoff_arena_slot == tl::kDataflowInvalidIndex || "
            f"handler_args.handoff_arena_slot >= {plan.lookahead_distance}u))"
        )
    peer_roles = {
        DataflowHandlerParamRole.NEXT_TASK_COORD,
        DataflowHandlerParamRole.HANDOFF_PEER_RANGE_BEGIN,
        DataflowHandlerParamRole.HANDOFF_PEER_RANGE_END,
        DataflowHandlerParamRole.HANDOFF_PEER_TASK_ID,
    }
    if any(param.role in peer_roles for param in params):
        peer_count = required_peer_task_coord_count(params)
        peer_count_guard = (
            ""
            if peer_count == 0
            else f" || tl::dataflow_handler_args(arg_base, handler_args.handoff_peer_arg_offset).task_coord_count < {peer_count}u"
        )
        clauses.append(
            "(handler_args.handoff_stage_count != 0u && "
            "(handler_args.handoff_peer_arg_offset == tl::kDataflowInvalidIndex"
            f"{peer_count_guard}))"
        )
    return "".join(f" ||\n      {clause}" for clause in clauses)


def count_required_task_coords(
    params: tuple[DataflowHandlerParam, ...],
) -> int:
    axes = [param.task_coord_axis for param in params if param.role is DataflowHandlerParamRole.TASK_COORD]
    return max((int(axis) + 1 for axis in axes if axis is not None), default=0)


def required_peer_task_coord_count(
    params: tuple[DataflowHandlerParam, ...],
) -> int:
    axes = [param.task_coord_axis for param in params if param.role is DataflowHandlerParamRole.NEXT_TASK_COORD]
    return max((int(axis) + 1 for axis in axes if axis is not None), default=0)


def task_coord_call_arg(index: int) -> str:
    if index == 0:
        return "task_coords[handler_args.task_coord_offset]"
    return f"task_coords[handler_args.task_coord_offset + {index}u]"


def output_pointer_name(field: DataflowFieldLayout, fields: tuple[DataflowFieldLayout, ...]) -> str:
    if len(fields) == 1:
        return "output_value"
    return f"output_{c_identifier(field.name)}"


def inter_pointer_name(field: DataflowFieldLayout, fields: tuple[DataflowFieldLayout, ...]) -> str:
    if len(fields) == 1:
        return "inter_value"
    return f"inter_{c_identifier(field.name)}"


def reduce_items_name(field: DataflowFieldLayout, fields: tuple[DataflowFieldLayout, ...]) -> str:
    if len(fields) == 1:
        return "reduce_items"
    return f"reduce_items_{c_identifier(field.name)}"


def direct_reduce_item_pointer_name(
    slot_index: int,
    field: DataflowFieldLayout,
    fields: tuple[DataflowFieldLayout, ...],
) -> str:
    if len(fields) == 1:
        return f"item{slot_index}"
    return f"item{slot_index}_{c_identifier(field.name)}"


def c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        identifier = "value"
    if identifier[0].isdigit():
        identifier = f"value_{identifier}"
    return identifier
