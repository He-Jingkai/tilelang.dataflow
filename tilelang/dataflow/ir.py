"""Python-side IR nodes for the Dataflow frontend.

These classes intentionally sit above TileLang/TIR.  Dataflow lowering turns this
dataflow IR into ordinary TileLang PrimFuncs later in the compile path.
"""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import Any, get_args, get_origin
from collections.abc import Callable

from tvm import tir


DATAFLOW_INTERMEDIATE_ATTR = "__tilelang_dataflow_intermediate__"


class DataflowOperatorKind(str, Enum):
    ITER = "iter"
    MAP = "map"
    REDUCE = "reduce"
    FINALIZE = "finalize"


class DataflowReducerContract(str, Enum):
    """Semantic contract declared by a Dataflow reduce operator."""

    LEGACY_NARY = "legacy_nary"
    ASSOCIATIVE_BINARY = "associative_binary"


@dataclass(frozen=True)
class IntermediateField:
    name: str
    annotation: Any
    shape: tuple[Any, ...] | None = None
    dtype: str | None = None
    scope: str | None = None


@dataclass(frozen=True, eq=False)
class IntermediateType:
    name: str
    fields: tuple[IntermediateField, ...]
    python_class: type
    attrs: dict[str, Any] = field(default_factory=dict)

    def field(self, name: str) -> IntermediateField:
        for item in self.fields:
            if item.name == name:
                return item
        raise KeyError(f"Unknown Dataflow intermediate field {name!r} on {self.name}")


@dataclass(frozen=True)
class OperatorCall:
    operator: DataflowOperator
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    bound_arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> DataflowOperatorKind:
        return self.operator.kind

    @property
    def name(self) -> str:
        return self.operator.name

    @property
    def output_type(self) -> IntermediateType | None:
        return self.operator.output_type


@dataclass
class DataflowOperator:
    kind: DataflowOperatorKind
    func: Callable[..., Any]
    signature: inspect.Signature
    annotations: dict[str, Any]
    input_types: tuple[IntermediateType, ...] = ()
    output_type: IntermediateType | None = None
    range_spec: Any | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    reducer_contract: DataflowReducerContract | None = None

    def __post_init__(self) -> None:
        if self.kind is DataflowOperatorKind.REDUCE:
            if self.reducer_contract is None:
                self.reducer_contract = DataflowReducerContract.LEGACY_NARY
        elif self.reducer_contract is not None:
            raise ValueError(f"Dataflow {self.kind.value} operator cannot declare reducer contract {self.reducer_contract.value!r}")
        self.name = self.func.__name__
        self.__name__ = self.func.__name__
        self.__qualname__ = self.func.__qualname__
        self.__doc__ = self.func.__doc__
        self.__module__ = self.func.__module__
        self.__wrapped__ = self.func

    def __call__(self, *args: Any, **kwargs: Any) -> OperatorCall:
        try:
            bound = self.signature.bind_partial(*args, **kwargs)
        except TypeError as err:
            raise TypeError(f"Invalid arguments for Dataflow {self.kind.value} operator {self.name!r}: {err}") from err
        explicit_arguments = dict(bound.arguments)
        bound_arguments = {
            name: explicit_arguments.get(name, name)
            for name in self.signature.parameters
            if name in explicit_arguments or self.is_external_tensor_parameter(name)
        }
        return OperatorCall(
            operator=self,
            args=tuple(args),
            kwargs=dict(kwargs),
            bound_arguments=bound_arguments,
        )

    def is_external_tensor_parameter(self, parameter_name: str) -> bool:
        parameter = self.signature.parameters[parameter_name]
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            return False
        annotation = self.annotations.get(parameter_name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            return True
        if isinstance(annotation, tir.Buffer):
            return True
        if get_intermediate_type(annotation) is not None:
            return False
        origin = get_origin(annotation)
        if origin in (
            list,
            tuple,
            collections.abc.Sequence,
            collections.abc.Iterable,
        ) and any(get_intermediate_type(item) is not None for item in get_args(annotation)):
            return False
        return False

    @property
    def specialization_values(self) -> dict[str, Any]:
        """Return closure values after validating the canonical snapshot."""

        from .specialization import (
            specialization_snapshot_from_attrs,
            validated_operator_specialization_values,
        )

        return validated_operator_specialization_values(
            self.func,
            specialization_snapshot_from_attrs(self.attrs),
        )


def is_intermediate_class(value: Any) -> bool:
    return isinstance(value, type) and hasattr(value, DATAFLOW_INTERMEDIATE_ATTR)


def get_intermediate_type(value: Any) -> IntermediateType | None:
    if isinstance(value, IntermediateType):
        return value
    if is_intermediate_class(value):
        return getattr(value, DATAFLOW_INTERMEDIATE_ATTR)
    return None


def make_intermediate_field(name: str, annotation: Any) -> IntermediateField:
    if isinstance(annotation, tir.Buffer):
        return IntermediateField(
            name=name,
            annotation=annotation,
            shape=tuple(annotation.shape),
            dtype=str(annotation.dtype),
            scope=annotation.scope(),
        )
    dtype = str(annotation) if annotation is not None else None
    return IntermediateField(name=name, annotation=annotation, dtype=dtype)
