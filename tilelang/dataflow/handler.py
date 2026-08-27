"""Production handler lowering contract for Dataflow compilation."""

from __future__ import annotations

from typing import Any

from .dtype_registry import primfunc_intermediate_dtype_names, require_dataflow_dtype
from .ir import IntermediateType
from .program import DataflowProgram


PRIMFUNC_HANDLER_LOWERING = "primfunc"
SUPPORTED_HANDLER_LOWERINGS = (PRIMFUNC_HANDLER_LOWERING,)


def normalize_handler_lowering(value: Any | None) -> str:
    """Return the canonical Dataflow handler lowering mode."""

    if value is None:
        return PRIMFUNC_HANDLER_LOWERING
    lowering = str(value)
    if lowering not in SUPPORTED_HANDLER_LOWERINGS:
        supported = ", ".join(SUPPORTED_HANDLER_LOWERINGS)
        raise ValueError(f"Unsupported Dataflow handler_lowering {value!r}; expected one of: {supported}")
    return lowering


def validate_handler_lowering(program: DataflowProgram, lowering: str) -> None:
    """Validate that a Dataflow program can use the requested handler lowering."""

    if not isinstance(program, DataflowProgram):
        raise TypeError(f"validate_handler_lowering expects DataflowProgram, got {program!r}")
    lowering = normalize_handler_lowering(lowering)
    if lowering == PRIMFUNC_HANDLER_LOWERING:
        if not program.is_complete:
            raise ValueError(f"{lowering} handler lowering requires a complete Dataflow program")
        intermediates = primfunc_intermediate_types(program)
        if not intermediates:
            raise ValueError(f"{lowering} handler lowering requires a Dataflow intermediate")
        for intermediate in intermediates:
            validate_primfunc_intermediate(intermediate, lowering)
        return

    raise AssertionError(f"Unhandled Dataflow handler lowering mode {lowering!r}")


def primfunc_intermediate_types(program: DataflowProgram) -> tuple[IntermediateType, ...]:
    if not program.is_stage_graph:
        return () if program.intermediate_type is None else (program.intermediate_type,)

    seen: set[str] = set()
    result: list[IntermediateType] = []
    for stage in program.stages:
        for intermediate in (stage.output_type, stage.physical_output_type):
            if intermediate is None or intermediate.name in seen:
                continue
            seen.add(intermediate.name)
            result.append(intermediate)
    return tuple(result)


def validate_primfunc_intermediate(intermediate: IntermediateType, lowering: str) -> None:
    if not intermediate.fields:
        raise NotImplementedError(f"{lowering} handler lowering requires at least one intermediate field")
    for field in intermediate.fields:
        try:
            dtype_info = require_dataflow_dtype(field.dtype)
        except NotImplementedError:
            dtype_info = None
        if dtype_info is None or not dtype_info.primfunc_intermediate_supported:
            supported = ", ".join(primfunc_intermediate_dtype_names())
            raise NotImplementedError(
                f"{lowering} handler lowering currently supports {supported} intermediate fields only, got {field.dtype!r}"
            )
        if field.shape is not None:
            validate_fixed_shape(field.shape, lowering, field.name)


def validate_fixed_shape(shape: tuple[Any, ...], lowering: str, field_name: str) -> None:
    for extent in shape:
        try:
            extent_value = int(extent)
        except (TypeError, ValueError) as err:
            raise NotImplementedError(
                f"{lowering} handler lowering currently supports fixed-shape tensor "
                f"intermediate fields only, got dynamic extent {extent!r} on field {field_name!r}"
            ) from err
        if extent_value <= 0:
            raise NotImplementedError(
                f"{lowering} handler lowering requires positive fixed-shape tensor extents, got {extent!r} on field {field_name!r}"
            )
