"""Decorators for declaring Dataflow operators."""

from __future__ import annotations

import collections.abc
import contextlib
import inspect
from typing import Any, get_args, get_origin, get_type_hints
from collections.abc import Callable

from .ir import (
    IntermediateType,
    DATAFLOW_INTERMEDIATE_ATTR,
    DataflowOperator,
    DataflowOperatorKind,
    DataflowReducerContract,
    get_intermediate_type,
    make_intermediate_field,
)
from .precision import normalize_accumulator_contracts
from .physical_contract import (
    DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_ATTR,
    DATAFLOW_OUTPUT_SLOT_NONE,
    normalize_operator_physical_contract,
    operator_physical_contract,
    reject_retired_raw_physical_attrs,
)
from .operation_contracts import (
    DATAFLOW_LAYOUT_CONTRACTS_ATTR,
    normalize_layout_contracts,
)
from .specialization import (
    DATAFLOW_SPECIALIZATION_SNAPSHOT_ATTR,
    capture_operator_specialization,
)
from .tensor_layout import (
    DATAFLOW_TENSOR_ARGUMENT_LAYOUTS_ATTR,
    normalize_tensor_argument_layouts,
)


def specialization_constants(attrs: collections.abc.Mapping[str, Any]) -> dict[str, Any]:
    constants = attrs.get("specialization_constants")
    if constants is None:
        return {}
    if not isinstance(constants, collections.abc.Mapping):
        raise TypeError("Dataflow specialization_constants must be a mapping")
    normalized: dict[str, Any] = {}
    for name, value in constants.items():
        if not isinstance(name, str) or not name.isidentifier() or name == "T":
            raise TypeError(f"Dataflow specialization constant names must be identifiers other than 'T', got {name!r}")
        normalized[name] = value
    return normalized


def normalize_operator_attributes(attrs: collections.abc.Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(attrs)
    reject_retired_raw_physical_attrs(normalized)
    if DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_ATTR in normalized:
        normalized[DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_ATTR] = normalize_operator_physical_contract(
            normalized[DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_ATTR]
        )
    if DATAFLOW_TENSOR_ARGUMENT_LAYOUTS_ATTR in normalized:
        layouts = normalize_tensor_argument_layouts(normalized[DATAFLOW_TENSOR_ARGUMENT_LAYOUTS_ATTR])
        if layouts:
            normalized[DATAFLOW_TENSOR_ARGUMENT_LAYOUTS_ATTR] = layouts
        else:
            normalized.pop(DATAFLOW_TENSOR_ARGUMENT_LAYOUTS_ATTR, None)
    constants = specialization_constants(normalized)
    if constants:
        normalized["specialization_constants"] = constants
    else:
        normalized.pop("specialization_constants", None)
    contracts = normalize_accumulator_contracts(normalized)
    if contracts:
        normalized["accumulator_contracts"] = contracts
    else:
        normalized.pop("accumulator_contracts", None)
    return normalized


def capture_specialization(attrs: dict[str, Any], target: Callable[..., Any]) -> dict[str, Any]:
    if DATAFLOW_SPECIALIZATION_SNAPSHOT_ATTR in attrs:
        raise ValueError("specialization_snapshot is compiler-owned metadata")
    snapshot = capture_operator_specialization(target)
    if snapshot.entries:
        attrs[DATAFLOW_SPECIALIZATION_SNAPSHOT_ATTR] = snapshot
    return attrs


def normalized_intermediate_attrs(
    attrs: collections.abc.Mapping[str, Any],
) -> dict[str, Any]:
    normalized = normalize_operator_attributes(attrs)
    if DATAFLOW_LAYOUT_CONTRACTS_ATTR not in normalized:
        return normalized
    if "primfunc_slot_layout" in normalized:
        raise ValueError("layout_contracts and legacy primfunc_slot_layout cannot both be set")
    contracts = normalize_layout_contracts(normalized[DATAFLOW_LAYOUT_CONTRACTS_ATTR])
    if contracts:
        normalized[DATAFLOW_LAYOUT_CONTRACTS_ATTR] = contracts
    else:
        normalized.pop(DATAFLOW_LAYOUT_CONTRACTS_ATTR, None)
    return normalized


def annotation_locals(obj: Callable[..., Any] | type, extra_locals: dict[str, Any]) -> dict[str, Any] | None:
    localns: dict[str, Any] = {}
    if inspect.isfunction(obj):
        with contextlib.suppress(TypeError):
            localns.update(inspect.getclosurevars(obj).nonlocals)
    elif isinstance(obj, type):
        localns.update(vars(obj))
    localns.update(extra_locals)
    return localns or None


def caller_locals() -> dict[str, Any]:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return {}
    return dict(frame.f_back.f_back.f_locals)


def annotation_extra_locals(attrs: dict[str, Any], definition_locals: dict[str, Any]) -> dict[str, Any]:
    localns = dict(definition_locals)
    localns.update(attrs.get("specialization_constants", {}))
    return localns


def safe_annotations(obj: Callable[..., Any] | type, *, extra_locals: dict[str, Any] | None = None) -> dict[str, Any]:
    globalns = getattr(obj, "__globals__", None)
    if globalns is None:
        module = inspect.getmodule(obj)
        globalns = vars(module) if module is not None else None
    localns = annotation_locals(obj, extra_locals or {})

    get_annotations = getattr(inspect, "get_annotations", None)
    if get_annotations is not None:
        try:
            return get_annotations(obj, globals=globalns, locals=localns, eval_str=True)
        except Exception:
            pass

    try:
        return get_type_hints(obj, globalns=globalns, localns=localns, include_extras=True)
    except Exception:
        annotations = dict(getattr(obj, "__annotations__", {}))
        if globalns is None:
            return annotations
        resolved = {}
        for name, annotation in annotations.items():
            if isinstance(annotation, str):
                try:
                    resolved[name] = eval(annotation, globalns, localns)  # pylint: disable=eval-used
                    continue
                except Exception:
                    pass
            resolved[name] = annotation
        return resolved


def resolve_return_annotation(func: Callable[..., Any], annotations: dict[str, Any]) -> Any:
    if "return" in annotations:
        return annotations["return"]
    return inspect.signature(func).return_annotation


def require_intermediate(annotation: Any, context: str) -> IntermediateType:
    intermediate = get_intermediate_type(annotation)
    if intermediate is None:
        raise TypeError(f"{context} must be annotated with a @T.dataflow_intermediate type, got {annotation!r}")
    return intermediate


def extract_collection_intermediate(annotation: Any) -> IntermediateType | None:
    origin = get_origin(annotation)
    if origin in (list, tuple, collections.abc.Sequence, collections.abc.Iterable):
        args = get_args(annotation)
        if len(args) == 1:
            return get_intermediate_type(args[0])
    return None


def extract_intermediate_inputs(func: Callable[..., Any], annotations: dict[str, Any]) -> tuple[IntermediateType, ...]:
    signature = inspect.signature(func)
    input_types: list[IntermediateType] = []
    for param in signature.parameters.values():
        annotation = annotations.get(param.name, param.annotation)
        if annotation is inspect.Signature.empty:
            continue
        collection_type = extract_collection_intermediate(annotation)
        if collection_type is not None:
            input_types.append(collection_type)
            continue
        intermediate = get_intermediate_type(annotation)
        if intermediate is not None:
            input_types.append(intermediate)
    return tuple(input_types)


def ensure_same_intermediate(types: tuple[IntermediateType, ...], context: str) -> IntermediateType:
    if not types:
        raise TypeError(f"{context} must consume at least one @T.dataflow_intermediate value")
    first = types[0]
    if any(item is not first for item in types):
        names = ", ".join(item.name for item in types)
        raise TypeError(f"{context} must use the same intermediate type for all intermediate inputs, got {names}")
    return first


def reduce_parameter_contract(
    target: Callable[..., Any],
    annotations: dict[str, Any],
    *,
    associative: bool,
) -> tuple[tuple[IntermediateType, ...], DataflowReducerContract]:
    signature = inspect.signature(target)
    parameters = tuple(signature.parameters.values())
    context = f"Dataflow reduce operator {target.__name__!r}"

    if associative:
        if len(parameters) != 2:
            raise TypeError(f"{context} associative contract requires exactly two intermediate parameters, got {len(parameters)}")
        input_types: list[IntermediateType] = []
        for parameter in parameters:
            annotation = annotations.get(parameter.name, parameter.annotation)
            intermediate = get_intermediate_type(annotation)
            if intermediate is None:
                raise TypeError(
                    f"{context} associative parameter {parameter.name!r} must be annotated with a @T.dataflow_intermediate type"
                )
            input_types.append(intermediate)
        ensure_same_intermediate(tuple(input_types), context)
        return tuple(input_types), DataflowReducerContract.ASSOCIATIVE_BINARY

    if len(parameters) != 1:
        raise TypeError(
            f"{context} legacy n-ary contract requires one collection parameter; "
            "use @T.dataflow.reduce(associative=True) for a binary reducer"
        )
    parameter = parameters[0]
    annotation = annotations.get(parameter.name, parameter.annotation)
    intermediate = extract_collection_intermediate(annotation)
    if intermediate is None:
        raise TypeError(
            f"{context} must consume at least one @T.dataflow_intermediate value; "
            f"legacy n-ary parameter {parameter.name!r} must be annotated as a collection"
        )
    return (intermediate,), DataflowReducerContract.LEGACY_NARY


def dataflow_intermediate(cls: type | None = None, **attrs: Any) -> type | Callable[[type], type]:
    """Declare a class as a Dataflow intermediate schema."""

    definition_locals = caller_locals()

    def decorate(target: type) -> type:
        normalized_attrs = normalized_intermediate_attrs(attrs)
        if not isinstance(target, type):
            raise TypeError(f"@T.dataflow_intermediate expects a class, got {target!r}")
        annotations = safe_annotations(
            target,
            extra_locals=annotation_extra_locals(normalized_attrs, definition_locals),
        )
        if not annotations:
            raise TypeError(f"Dataflow intermediate {target.__name__!r} must declare at least one annotated field")
        fields = tuple(make_intermediate_field(name, annotation) for name, annotation in annotations.items())
        intermediate = IntermediateType(
            name=target.__name__,
            fields=fields,
            python_class=target,
            attrs=normalized_attrs,
        )
        setattr(target, DATAFLOW_INTERMEDIATE_ATTR, intermediate)
        return target

    if cls is None:
        return decorate
    return decorate(cls)


def dataflow_iter(
    func: Callable[..., Any] | None = None,
    *,
    range: Any | None = None,  # pylint: disable=redefined-builtin
    **attrs: Any,
) -> DataflowOperator | Callable[[Callable[..., Any]], DataflowOperator]:
    """Declare an Iter operator.

    Calling the decorated object records an operator call; it does not execute
    the Python function body.
    """

    definition_locals = caller_locals()

    def decorate(target: Callable[..., Any]) -> DataflowOperator:
        normalized_attrs = normalize_operator_attributes(attrs)
        if not inspect.isfunction(target):
            raise TypeError(f"@T.dataflow.iter expects a function, got {target!r}")
        normalized_attrs = capture_specialization(normalized_attrs, target)
        annotations = safe_annotations(
            target,
            extra_locals=annotation_extra_locals(normalized_attrs, definition_locals),
        )
        output_type = require_intermediate(
            resolve_return_annotation(target, annotations), f"Dataflow iter operator {target.__name__!r} return"
        )
        return DataflowOperator(
            kind=DataflowOperatorKind.ITER,
            func=target,
            signature=inspect.signature(target),
            annotations=annotations,
            output_type=output_type,
            range_spec=range,
            attrs=normalized_attrs,
        )

    if func is None:
        return decorate
    return decorate(func)


def dataflow_map(
    func: Callable[..., Any] | None = None,
    *,
    range: Any | None = None,  # pylint: disable=redefined-builtin
    **attrs: Any,
) -> DataflowOperator | Callable[[Callable[..., Any]], DataflowOperator]:
    """Declare a generic map operator.

    A map operator is a compute stage that may consume zero or more Dataflow
    intermediates and produces one Dataflow intermediate.  Calling the decorated
    object records an operator call; it does not execute the Python function
    body.
    """

    definition_locals = caller_locals()

    def decorate(target: Callable[..., Any]) -> DataflowOperator:
        normalized_attrs = normalize_operator_attributes(attrs)
        if not inspect.isfunction(target):
            raise TypeError(f"@T.dataflow.map expects a function, got {target!r}")
        normalized_attrs = capture_specialization(normalized_attrs, target)
        annotations = safe_annotations(
            target,
            extra_locals=annotation_extra_locals(normalized_attrs, definition_locals),
        )
        return_annotation = resolve_return_annotation(target, annotations)
        physical = operator_physical_contract(normalized_attrs)
        if return_annotation in (inspect.Signature.empty, None, type(None)):
            if physical.output_slot != DATAFLOW_OUTPUT_SLOT_NONE:
                raise TypeError(
                    f"Dataflow map operator {target.__name__!r} without an intermediate "
                    "return requires physical_contract.output_slot='none'"
                )
            output_type = None
        else:
            if physical.output_slot == DATAFLOW_OUTPUT_SLOT_NONE:
                raise TypeError(f"outputless Dataflow map operator {target.__name__!r} must return None")
            output_type = require_intermediate(
                return_annotation,
                f"Dataflow map operator {target.__name__!r} return",
            )
        return DataflowOperator(
            kind=DataflowOperatorKind.MAP,
            func=target,
            signature=inspect.signature(target),
            annotations=annotations,
            input_types=extract_intermediate_inputs(target, annotations),
            output_type=output_type,
            range_spec=range,
            attrs=normalized_attrs,
        )

    if func is None:
        return decorate
    return decorate(func)


def dataflow_reduce(
    func: Callable[..., Any] | None = None,
    *,
    associative: bool = False,
    **attrs: Any,
) -> DataflowOperator | Callable[[Callable[..., Any]], DataflowOperator]:
    """Declare a legacy n-ary or associative binary Reduce operator."""

    if not isinstance(associative, bool):
        raise TypeError(f"Dataflow reduce associative must be a bool, got {associative!r}")

    definition_locals = caller_locals()

    def decorate(target: Callable[..., Any]) -> DataflowOperator:
        normalized_attrs = normalize_operator_attributes(attrs)
        if not inspect.isfunction(target):
            raise TypeError(f"@T.dataflow.reduce expects a function, got {target!r}")
        normalized_attrs = capture_specialization(normalized_attrs, target)
        annotations = safe_annotations(
            target,
            extra_locals=annotation_extra_locals(normalized_attrs, definition_locals),
        )
        input_types, reducer_contract = reduce_parameter_contract(
            target,
            annotations,
            associative=associative,
        )
        input_type = ensure_same_intermediate(input_types, f"Dataflow reduce operator {target.__name__!r}")
        output_type = require_intermediate(
            resolve_return_annotation(target, annotations), f"Dataflow reduce operator {target.__name__!r} return"
        )
        if output_type is not input_type:
            raise TypeError(
                f"Dataflow reduce operator {target.__name__!r} must return the same intermediate type it consumes, "
                f"got input {input_type.name} and output {output_type.name}"
            )
        return DataflowOperator(
            kind=DataflowOperatorKind.REDUCE,
            func=target,
            signature=inspect.signature(target),
            annotations=annotations,
            input_types=input_types,
            output_type=output_type,
            attrs=normalized_attrs,
            reducer_contract=reducer_contract,
        )

    if func is None:
        return decorate
    return decorate(func)


def dataflow_finalize(
    func: Callable[..., Any] | None = None,
    **attrs: Any,
) -> DataflowOperator | Callable[[Callable[..., Any]], DataflowOperator]:
    """Declare a Finalize operator."""

    definition_locals = caller_locals()

    def decorate(target: Callable[..., Any]) -> DataflowOperator:
        normalized_attrs = normalize_operator_attributes(attrs)
        if not inspect.isfunction(target):
            raise TypeError(f"@T.dataflow.finalize expects a function, got {target!r}")
        normalized_attrs = capture_specialization(normalized_attrs, target)
        annotations = safe_annotations(
            target,
            extra_locals=annotation_extra_locals(normalized_attrs, definition_locals),
        )
        input_types = extract_intermediate_inputs(target, annotations)
        ensure_same_intermediate(input_types, f"Dataflow finalize operator {target.__name__!r}")
        return_annotation = resolve_return_annotation(target, annotations)
        if return_annotation not in (inspect.Signature.empty, None, type(None)):
            raise TypeError(f"Dataflow finalize operator {target.__name__!r} must return None, got {return_annotation!r}")
        return DataflowOperator(
            kind=DataflowOperatorKind.FINALIZE,
            func=target,
            signature=inspect.signature(target),
            annotations=annotations,
            input_types=input_types,
            output_type=None,
            attrs=normalized_attrs,
        )

    if func is None:
        return decorate
    return decorate(func)
