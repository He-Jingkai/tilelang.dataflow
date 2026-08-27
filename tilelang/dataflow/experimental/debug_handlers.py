"""Synthetic Dataflow handlers for runtime, communication, and ABI probes.

This module is deliberately outside :mod:`tilelang.dataflow`'s public frontend.
The providers exercise the queue-driven wrapper without introducing a second
operator-body compiler. Production compilation supports PrimFunc handlers only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..implementation_registry import dataflow_implementation_registry
from ..program import DataflowProgram
from ..tensor_args import collect_tensor_arg_plan
from ..wrapper import DataflowHandlerSource, DataflowHandlerSpec, DataflowWrapperSpec


_SCALAR_U32_DTYPES = frozenset(("uint32", "int32"))


@dataclass(frozen=True)
class DataflowDebugHandlerProvider:
    """One named synthetic handler family used by explicit debug fixtures."""

    name: str
    kind: str
    executable: bool
    non_executable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experimental Dataflow debug handler requires a name")
        if self.executable and self.non_executable_reason is not None:
            raise ValueError("executable debug handlers cannot have a non-executable reason")
        if not self.executable and not self.non_executable_reason:
            raise ValueError("non-executable debug handlers require a reason")

    def validate(self, program: DataflowProgram) -> None:
        if not isinstance(program, DataflowProgram):
            raise TypeError(f"Dataflow debug handler expects DataflowProgram, got {program!r}")
        if self.kind == "empty":
            return
        if not program.is_complete:
            raise ValueError(f"{self.name} debug handler requires a complete Dataflow program")
        intermediate = program.intermediate_type
        if intermediate is None:
            raise ValueError(f"{self.name} debug handler requires a Dataflow intermediate")
        if len(intermediate.fields) != 1:
            raise NotImplementedError(f"{self.name} debug handler supports exactly one scalar intermediate field")
        field = intermediate.fields[0]
        if field.shape is not None:
            raise NotImplementedError(f"{self.name} debug handler supports scalar intermediate fields only")
        if field.dtype not in _SCALAR_U32_DTYPES:
            supported = ", ".join(sorted(_SCALAR_U32_DTYPES))
            raise NotImplementedError(f"{self.name} debug handler supports {supported} intermediate fields only, got {field.dtype!r}")
        if self.kind == "tensor_u32" and len(collect_tensor_arg_plan(program).specs) < 2:
            raise NotImplementedError(f"{self.name} debug handler requires at least one input tensor and one output tensor binding")

    def populate_wrapper_spec(self, spec: DataflowWrapperSpec) -> DataflowWrapperSpec:
        if not isinstance(spec, DataflowWrapperSpec):
            raise TypeError(f"populate_wrapper_spec expects DataflowWrapperSpec, got {spec!r}")
        return replace(
            spec,
            handler_lowering=self.name,
            handler_helper_source=generate_helper_source(self.kind),
            handler_sources=tuple(
                DataflowHandlerSource(
                    handler_id=handler.handler_id,
                    body_source=handler_body(handler, self.kind),
                )
                for handler in spec.handlers
            ),
        )


EMPTY_HANDLER = DataflowDebugHandlerProvider(
    name="experimental.empty",
    kind="empty",
    executable=False,
    non_executable_reason="empty experimental handlers do not perform program semantics",
)
SCALAR_U32_HANDLER = DataflowDebugHandlerProvider(
    name="experimental.scalar_u32",
    kind="scalar_u32",
    executable=True,
)
TENSOR_U32_HANDLER = DataflowDebugHandlerProvider(
    name="experimental.tensor_u32",
    kind="tensor_u32",
    executable=True,
)

_PROVIDERS = {provider.name: provider for provider in (EMPTY_HANDLER, SCALAR_U32_HANDLER, TENSOR_U32_HANDLER)}


def resolve_debug_handler(value: Any) -> DataflowDebugHandlerProvider:
    if isinstance(value, DataflowDebugHandlerProvider):
        provider = value
    else:
        try:
            provider = _PROVIDERS[str(value)]
        except KeyError as err:
            raise ValueError(f"Unknown experimental Dataflow debug handler {value!r}; expected one of {tuple(_PROVIDERS)!r}") from err
    dataflow_implementation_registry().require_selectable(
        provider.name,
        selected_explicitly=True,
    )
    return provider


def populate_wrapper_spec(
    spec: DataflowWrapperSpec,
    handler: DataflowDebugHandlerProvider = EMPTY_HANDLER,
) -> DataflowWrapperSpec:
    return resolve_debug_handler(handler).populate_wrapper_spec(spec)


def compile(
    program: DataflowProgram,
    *,
    handler: DataflowDebugHandlerProvider = EMPTY_HANDLER,
    **options: Any,
):
    """Compile an explicit synthetic-handler artifact for Dataflow tests/probes."""

    from ..compiler import compile as compile_dataflow

    provider = resolve_debug_handler(handler)
    if "mode" in options or "_experimental_debug_handler" in options:
        raise ValueError("experimental debug compile owns its mode and handler provider")
    return compile_dataflow(
        program,
        mode="debug",
        _experimental_debug_handler=provider.name,
        **options,
    )


def generate_helper_source(kind: str) -> str:
    if kind == "empty":
        return ""
    helpers = """TL_DEVICE uint32_t *dataflow_scalar_debug_shared_u32(
    void *shared_base,
    const tl::DataflowSlot &slot) {
  return reinterpret_cast<uint32_t *>(tl_dataflow_generated::dataflow_slot_shared_ptr(shared_base, slot));
}

TL_DEVICE const uint32_t *dataflow_scalar_debug_shared_u32(
    const void *shared_base,
    const tl::DataflowSlot &slot) {
  return reinterpret_cast<const uint32_t *>(tl_dataflow_generated::dataflow_slot_shared_ptr(shared_base, slot));
}

TL_DEVICE uint32_t *dataflow_scalar_debug_global_u32(
    void *global_base,
    const tl::DataflowSlot &slot) {
  return reinterpret_cast<uint32_t *>(tl::dataflow_slot_global_ptr(global_base, slot));
}"""
    if kind == "tensor_u32":
        helpers += """

TL_DEVICE const uint32_t *dataflow_tensor_debug_input_u32(
    const tl::DataflowTensorArg *tensor_args) {
  return reinterpret_cast<const uint32_t *>(
      static_cast<uintptr_t>(tensor_args[0].data_ptr));
}

TL_DEVICE uint32_t *dataflow_tensor_debug_output_u32(
    const tl::DataflowTensorArg *tensor_args) {
  return reinterpret_cast<uint32_t *>(
      static_cast<uintptr_t>(
          tensor_args[tl_dataflow_generated::kDataflowTensorArgCount - 1u].data_ptr));
}"""
    return helpers


def unused_body(comment: str) -> str:
    return f"""  // {comment}
  (void)inst;
  (void)handler_args;
  (void)slots;
  (void)tensor_args;
  (void)input_slots;
  (void)task_coords;
  (void)shared_base;
  (void)global_base;"""


def handler_body(handler: DataflowHandlerSpec, kind: str) -> str:
    if kind == "empty":
        return unused_body(f"Empty experimental Dataflow handler for operator {handler.operator_name}.")
    if kind == "scalar_u32":
        return scalar_handler_body(handler)
    if kind == "tensor_u32":
        return tensor_handler_body(handler)
    raise AssertionError(f"Unhandled experimental debug handler kind {kind!r}")


def scalar_handler_body(handler: DataflowHandlerSpec) -> str:
    common_unused = """  (void)inst;
  (void)tensor_args;
  (void)task_coords;"""
    if handler.operator_kind == "iter":
        return f"""{common_unused}
  (void)input_slots;
  (void)global_base;
  if (!tl::dataflow_is_leader_thread() ||
      handler_args.output_slot == tl::kDataflowInvalidIndex) {{
    return;
  }}
  const tl::DataflowSlot &output_slot = slots[handler_args.output_slot];
  uint32_t *output = dataflow_scalar_debug_shared_u32(shared_base, output_slot);
  *output = handler_args.range_end - handler_args.range_begin;"""
    if handler.operator_kind == "reduce":
        return f"""{common_unused}
  (void)global_base;
  if (!tl::dataflow_is_leader_thread() ||
      handler_args.output_slot == tl::kDataflowInvalidIndex) {{
    return;
  }}
  uint32_t value = 0;
  for (uint32_t i = 0; i < handler_args.input_slot_count; ++i) {{
    uint32_t slot_id = input_slots[handler_args.input_slot_offset + i];
    const tl::DataflowSlot &input_slot = slots[slot_id];
    value += *dataflow_scalar_debug_shared_u32(shared_base, input_slot);
  }}
  const tl::DataflowSlot &output_slot = slots[handler_args.output_slot];
  uint32_t *output = dataflow_scalar_debug_shared_u32(shared_base, output_slot);
  *output = value;"""
    if handler.operator_kind == "finalize":
        return f"""{common_unused}
  if (!tl::dataflow_is_leader_thread() || handler_args.input_slot_count == 0u) {{
    return;
  }}
  uint32_t slot_id = input_slots[handler_args.input_slot_offset];
  const tl::DataflowSlot &slot = slots[slot_id];
  const uint32_t *input = dataflow_scalar_debug_shared_u32(shared_base, slot);
  uint32_t *output = dataflow_scalar_debug_global_u32(global_base, slot);
  *output = *input;"""
    return unused_body(f"No scalar debug handler is available for operator {handler.operator_name}.")


def tensor_handler_body(handler: DataflowHandlerSpec) -> str:
    common_unused = """  (void)inst;
  (void)task_coords;"""
    if handler.operator_kind == "iter":
        return f"""{common_unused}
  (void)input_slots;
  (void)global_base;
  if (!tl::dataflow_is_leader_thread() ||
      handler_args.output_slot == tl::kDataflowInvalidIndex ||
      tl_dataflow_generated::kDataflowTensorArgCount < 2u) {{
    return;
  }}
  const uint32_t *input_tensor = dataflow_tensor_debug_input_u32(tensor_args);
  uint32_t value = 0;
  for (uint32_t i = handler_args.range_begin; i < handler_args.range_end; ++i) {{
    value += input_tensor[i];
  }}
  const tl::DataflowSlot &output_slot = slots[handler_args.output_slot];
  uint32_t *output = dataflow_scalar_debug_shared_u32(shared_base, output_slot);
  *output = value;"""
    if handler.operator_kind == "reduce":
        return f"""{common_unused}
  (void)tensor_args;
  (void)global_base;
  if (!tl::dataflow_is_leader_thread() ||
      handler_args.output_slot == tl::kDataflowInvalidIndex) {{
    return;
  }}
  uint32_t value = 0;
  for (uint32_t i = 0; i < handler_args.input_slot_count; ++i) {{
    uint32_t slot_id = input_slots[handler_args.input_slot_offset + i];
    const tl::DataflowSlot &input_slot = slots[slot_id];
    value += *dataflow_scalar_debug_shared_u32(shared_base, input_slot);
  }}
  const tl::DataflowSlot &output_slot = slots[handler_args.output_slot];
  uint32_t *output = dataflow_scalar_debug_shared_u32(shared_base, output_slot);
  *output = value;"""
    if handler.operator_kind == "finalize":
        return f"""{common_unused}
  (void)global_base;
  if (!tl::dataflow_is_leader_thread() ||
      handler_args.input_slot_count == 0u ||
      tl_dataflow_generated::kDataflowTensorArgCount < 2u) {{
    return;
  }}
  uint32_t slot_id = input_slots[handler_args.input_slot_offset];
  const tl::DataflowSlot &slot = slots[slot_id];
  const uint32_t *input = dataflow_scalar_debug_shared_u32(shared_base, slot);
  uint32_t *output_tensor = dataflow_tensor_debug_output_u32(tensor_args);
  output_tensor[handler_args.task_id] = *input;"""
    return unused_body(f"No tensor debug handler is available for operator {handler.operator_name}.")
