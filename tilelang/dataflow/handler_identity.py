"""Structured handler identities shared across Dataflow compilation stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

from .ir import OperatorCall
from .program import DataflowProgram, DataflowStage, DataflowStageKind


ITER_RANGE_GENERIC = "generic"
ITER_RANGE_TILE_COUNT = "tile_count"
ITER_RANGE_EXACT_LENGTH = "exact_length"
ITER_RANGE_SPECIALIZATION_KIND = "iter_range"
_ITER_RANGE_SPECIALIZATION_MODES = frozenset((ITER_RANGE_GENERIC, ITER_RANGE_TILE_COUNT, ITER_RANGE_EXACT_LENGTH))
REDUCE_ARITY_PASSTHROUGH = "passthrough"
REDUCE_ARITY_BINARY = "binary"
REDUCE_ARITY_GENERIC = "generic"
REDUCE_ARITY_SPECIALIZATION_KIND = "reduce_arity"
_REDUCE_ARITY_CLASSES = frozenset((REDUCE_ARITY_PASSTHROUGH, REDUCE_ARITY_BINARY, REDUCE_ARITY_GENERIC))
_DATAFLOW_HANDLER_KINDS = frozenset(("iter", "map", "reduce", "finalize"))
DATAFLOW_HANDLER_VARIANT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IterRangeSpecialization:
    """Structured ITER range specialization selected by the scheduler."""

    mode: str = ITER_RANGE_GENERIC
    value: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError(f"Dataflow ITER specialization mode must be a string, got {self.mode!r}")
        if self.mode not in _ITER_RANGE_SPECIALIZATION_MODES:
            expected = ", ".join(sorted(_ITER_RANGE_SPECIALIZATION_MODES))
            raise ValueError(f"Unsupported Dataflow ITER specialization mode {self.mode!r}; expected one of: {expected}")
        if self.mode == ITER_RANGE_GENERIC:
            if self.value is not None:
                raise ValueError("generic Dataflow ITER specialization cannot have a value")
            return
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value <= 0:
            raise ValueError(f"Dataflow ITER specialization {self.mode!r} requires a positive integer value, got {self.value!r}")

    def fixed_extent(self, *, bucket_size: int | None) -> int | None:
        if self.mode == ITER_RANGE_GENERIC:
            return None
        assert self.value is not None
        if self.mode == ITER_RANGE_EXACT_LENGTH:
            return self.value
        if isinstance(bucket_size, bool) or not isinstance(bucket_size, int) or bucket_size <= 0:
            raise ValueError("tile-count Dataflow ITER specialization requires a positive bucket_size")
        return self.value * bucket_size

    def display_name(self, base_name: str) -> str:
        """Return a diagnostic name; callers must not parse it for semantics."""

        if self.mode == ITER_RANGE_GENERIC:
            return base_name
        assert self.value is not None
        marker = "tiles" if self.mode == ITER_RANGE_TILE_COUNT else "len"
        return f"{base_name}__dataflow_iter_{marker}_{self.value}"

    @property
    def canonical_key(self) -> tuple[str, str, int | None]:
        return (ITER_RANGE_SPECIALIZATION_KIND, self.mode, self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "value": self.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IterRangeSpecialization:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow ITER specialization payload must be a mapping, got {type(value)!r}")
        return cls(mode=value.get("mode", ITER_RANGE_GENERIC), value=value.get("value"))


@dataclass(frozen=True)
class ReduceAritySpecialization:
    """Typed reducer variant selected from an instruction's input arity."""

    arity_class: str = REDUCE_ARITY_GENERIC

    def __post_init__(self) -> None:
        if not isinstance(self.arity_class, str):
            raise TypeError(f"Dataflow reduce arity class must be a string, got {self.arity_class!r}")
        if self.arity_class not in _REDUCE_ARITY_CLASSES:
            expected = ", ".join(sorted(_REDUCE_ARITY_CLASSES))
            raise ValueError(f"Unsupported Dataflow reduce arity class {self.arity_class!r}; expected one of: {expected}")

    @classmethod
    def for_arity(cls, arity: int | None) -> ReduceAritySpecialization:
        if arity is None:
            return cls(REDUCE_ARITY_GENERIC)
        if isinstance(arity, bool) or not isinstance(arity, int) or arity <= 0:
            raise ValueError(f"Dataflow reduce arity must be a positive integer or None, got {arity!r}")
        if arity == 1:
            return cls(REDUCE_ARITY_PASSTHROUGH)
        if arity == 2:
            return cls(REDUCE_ARITY_BINARY)
        return cls(REDUCE_ARITY_GENERIC)

    @property
    def canonical_key(self) -> tuple[str, str]:
        return (REDUCE_ARITY_SPECIALIZATION_KIND, self.arity_class)

    def to_dict(self) -> dict[str, str]:
        return {"arity_class": self.arity_class}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReduceAritySpecialization:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow reduce arity specialization payload must be a mapping, got {type(value)!r}")
        return cls(arity_class=value.get("arity_class", REDUCE_ARITY_GENERIC))


DataflowHandlerSpecialization: TypeAlias = IterRangeSpecialization | ReduceAritySpecialization


def specialization_kind(value: DataflowHandlerSpecialization) -> str:
    if isinstance(value, IterRangeSpecialization):
        return ITER_RANGE_SPECIALIZATION_KIND
    if isinstance(value, ReduceAritySpecialization):
        return REDUCE_ARITY_SPECIALIZATION_KIND
    raise TypeError(f"Dataflow handler specializations must be typed specialization objects, got {value!r}")


def specialization_from_dict(
    value: Mapping[str, Any],
) -> DataflowHandlerSpecialization:
    if not isinstance(value, Mapping):
        raise TypeError(f"Dataflow handler specialization entry must be a mapping, got {type(value)!r}")
    kind = value.get("kind")
    payload = value.get("value")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Dataflow handler specialization {kind!r} requires a mapping value")
    if kind == ITER_RANGE_SPECIALIZATION_KIND:
        return IterRangeSpecialization.from_dict(payload)
    if kind == REDUCE_ARITY_SPECIALIZATION_KIND:
        return ReduceAritySpecialization.from_dict(payload)
    raise ValueError(f"Unsupported Dataflow handler specialization kind {kind!r}")


@dataclass(frozen=True)
class DataflowHandlerIdentity:
    """Stable program binding identity shared by all compiled variants."""

    operator_id: int
    operator_kind: str

    def __post_init__(self) -> None:
        if isinstance(self.operator_id, bool) or not isinstance(self.operator_id, int):
            raise TypeError(f"Dataflow handler operator_id must be an integer, got {self.operator_id!r}")
        if self.operator_id < 0:
            raise ValueError(f"Dataflow handler operator_id must be non-negative, got {self.operator_id}")
        if not isinstance(self.operator_kind, str):
            raise TypeError(f"Dataflow handler operator_kind must be a string, got {self.operator_kind!r}")
        if self.operator_kind not in _DATAFLOW_HANDLER_KINDS:
            expected = ", ".join(sorted(_DATAFLOW_HANDLER_KINDS))
            raise ValueError(f"Unsupported Dataflow handler operator_kind {self.operator_kind!r}; expected one of: {expected}")

    @property
    def binding_key(self) -> tuple[int, str]:
        """Canonical program binding key, independent of specialization and names."""

        return (self.operator_id, self.operator_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "operator_kind": self.operator_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowHandlerIdentity:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow handler identity payload must be a mapping, got {type(value)!r}")
        if "specialization" in value:
            raise ValueError("Dataflow handler identity cannot contain specialization metadata; use DataflowHandlerVariantKey")
        return cls(
            operator_id=value.get("operator_id"),
            operator_kind=value.get("operator_kind"),
        )


@dataclass(frozen=True)
class DataflowHandlerVariantKey:
    """Canonical typed specialization tuple for one base handler identity."""

    base_identity: DataflowHandlerIdentity
    typed_specializations: tuple[DataflowHandlerSpecialization, ...] = ()
    schema_version: int = DATAFLOW_HANDLER_VARIANT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.base_identity, DataflowHandlerIdentity):
            raise TypeError(f"Dataflow handler variant requires a DataflowHandlerIdentity, got {self.base_identity!r}")
        if self.schema_version != DATAFLOW_HANDLER_VARIANT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Dataflow handler variant schema {self.schema_version}; expected {DATAFLOW_HANDLER_VARIANT_SCHEMA_VERSION}"
            )
        if not isinstance(self.typed_specializations, tuple):
            raise TypeError("Dataflow handler variant typed_specializations must be a tuple")
        kinds = []
        normalized = []
        for specialization in self.typed_specializations:
            kind = specialization_kind(specialization)
            kinds.append(kind)
            normalized.append(specialization)
        if len(kinds) != len(set(kinds)):
            raise ValueError(f"Dataflow handler variant has duplicate specialization kinds: {kinds!r}")
        normalized.sort(key=lambda item: item.canonical_key)
        object.__setattr__(self, "typed_specializations", tuple(normalized))

        expected_kinds = {
            "iter": (ITER_RANGE_SPECIALIZATION_KIND,),
            "reduce": (REDUCE_ARITY_SPECIALIZATION_KIND,),
            "map": (),
            "finalize": (),
        }[self.base_identity.operator_kind]
        normalized_kinds = tuple(specialization_kind(item) for item in self.typed_specializations)
        if normalized_kinds != expected_kinds:
            raise ValueError(
                "Dataflow handler variant specialization kinds do not match base identity: "
                f"operator_kind={self.base_identity.operator_kind!r}, "
                f"expected={expected_kinds!r}, got={normalized_kinds!r}"
            )

    @classmethod
    def default_for_identity(
        cls,
        identity: DataflowHandlerIdentity,
        *,
        reduce_arity: int | None = None,
    ) -> DataflowHandlerVariantKey:
        if identity.operator_kind == "iter":
            typed_specializations: tuple[DataflowHandlerSpecialization, ...] = (IterRangeSpecialization(),)
        elif identity.operator_kind == "reduce":
            typed_specializations = (ReduceAritySpecialization.for_arity(reduce_arity),)
        else:
            typed_specializations = ()
        return cls(
            base_identity=identity,
            typed_specializations=typed_specializations,
        )

    @property
    def binding_key(self) -> tuple[int, str]:
        return self.base_identity.binding_key

    @property
    def canonical_key(self) -> tuple[Any, ...]:
        return (
            self.schema_version,
            self.base_identity.binding_key,
            tuple(item.canonical_key for item in self.typed_specializations),
        )

    def specialization(
        self,
        kind: str,
    ) -> DataflowHandlerSpecialization | None:
        for specialization in self.typed_specializations:
            if specialization_kind(specialization) == kind:
                return specialization
        return None

    @property
    def iter_range(self) -> IterRangeSpecialization | None:
        value = self.specialization(ITER_RANGE_SPECIALIZATION_KIND)
        return value if isinstance(value, IterRangeSpecialization) else None

    @property
    def reduce_arity(self) -> ReduceAritySpecialization | None:
        value = self.specialization(REDUCE_ARITY_SPECIALIZATION_KIND)
        return value if isinstance(value, ReduceAritySpecialization) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_identity": self.base_identity.to_dict(),
            "typed_specializations": [
                {
                    "kind": specialization_kind(specialization),
                    "value": specialization.to_dict(),
                }
                for specialization in self.typed_specializations
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowHandlerVariantKey:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow handler variant payload must be a mapping, got {type(value)!r}")
        base_identity = value.get("base_identity")
        typed_specializations = value.get("typed_specializations", ())
        if not isinstance(base_identity, Mapping):
            raise TypeError("Dataflow handler variant requires base_identity metadata")
        if not isinstance(typed_specializations, (tuple, list)):
            raise TypeError("Dataflow handler variant typed_specializations payload must be a sequence")
        return cls(
            base_identity=DataflowHandlerIdentity.from_dict(base_identity),
            typed_specializations=tuple(specialization_from_dict(item) for item in typed_specializations),
            schema_version=value.get("schema_version", DATAFLOW_HANDLER_VARIANT_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class DataflowHandlerBinding:
    """Program operator and stage resolved from a canonical handler identity."""

    identity: DataflowHandlerIdentity
    call: OperatorCall
    stage: DataflowStage


@dataclass(frozen=True)
class DataflowHandlerRegistry:
    """Immutable program-local registry used by all semantic handler lookups."""

    bindings_by_key: Mapping[tuple[int, str], DataflowHandlerBinding]
    identities_by_stage_id: Mapping[int, DataflowHandlerIdentity]
    identities_by_call: Mapping[tuple[int, str], DataflowHandlerIdentity]

    def resolve(self, identity: DataflowHandlerIdentity) -> DataflowHandlerBinding:
        try:
            return self.bindings_by_key[identity.binding_key]
        except KeyError as err:
            raise KeyError(
                f"Unknown Dataflow handler identity: operator_id={identity.operator_id}, operator_kind={identity.operator_kind!r}"
            ) from err

    def resolve_variant(
        self,
        variant_key: DataflowHandlerVariantKey,
    ) -> DataflowHandlerBinding:
        if not isinstance(variant_key, DataflowHandlerVariantKey):
            raise TypeError(f"Dataflow handler registry expects DataflowHandlerVariantKey, got {variant_key!r}")
        return self.resolve(variant_key.base_identity)

    def identity_for_stage_id(self, stage_id: int) -> DataflowHandlerIdentity:
        try:
            return self.identities_by_stage_id[stage_id]
        except KeyError as err:
            raise KeyError(f"Dataflow stage {stage_id} does not define a handler") from err

    def identity_for_call(
        self,
        call: OperatorCall,
        operator_kind: str,
    ) -> DataflowHandlerIdentity:
        try:
            return self.identities_by_call[(id(call), operator_kind)]
        except KeyError as err:
            raise KeyError(f"Dataflow program does not register {operator_kind!r} call {call.name!r}") from err


def build_handler_registry(program: DataflowProgram) -> DataflowHandlerRegistry:
    """Build stable handler bindings in program stage order without using names as keys."""

    if not isinstance(program, DataflowProgram):
        raise TypeError(f"build_handler_registry expects DataflowProgram, got {program!r}")

    bindings: dict[tuple[int, str], DataflowHandlerBinding] = {}
    by_stage_id: dict[int, DataflowHandlerIdentity] = {}
    by_call: dict[tuple[int, str], DataflowHandlerIdentity] = {}
    for stage in program.stages:
        if stage.call is None or stage.kind is DataflowStageKind.RESHARED:
            continue
        operator_kind = handler_operator_kind(stage)
        identity = DataflowHandlerIdentity(
            operator_id=len(bindings),
            operator_kind=operator_kind,
        )
        call_key = (id(stage.call), operator_kind)
        if call_key in by_call:
            previous = by_call[call_key]
            raise ValueError(
                "Dataflow program reuses one operator call for multiple handler stages: "
                f"existing_operator_id={previous.operator_id}, duplicate_stage_id={stage.stage_id}, "
                f"operator_kind={operator_kind!r}"
            )
        binding = DataflowHandlerBinding(identity=identity, call=stage.call, stage=stage)
        bindings[identity.binding_key] = binding
        by_stage_id[stage.stage_id] = identity
        by_call[call_key] = identity
    if program.finalize_stage is not None and program.finalize_stage.fused_reduce_calls:
        stage = next(
            stage
            for stage in reversed(program.stages)
            if stage.kind is DataflowStageKind.FINALIZE and stage.call is program.finalize_stage.finalize_call
        )
        operator_kind = DataflowStageKind.FINALIZE.value
        for call in program.finalize_stage.fused_reduce_calls:
            identity = DataflowHandlerIdentity(
                operator_id=len(bindings),
                operator_kind=operator_kind,
            )
            call_key = (id(call), operator_kind)
            if call_key in by_call:
                raise ValueError("Dataflow program reuses a fused-reduce finalizer call for another handler stage")
            binding = DataflowHandlerBinding(identity=identity, call=call, stage=stage)
            bindings[identity.binding_key] = binding
            by_call[call_key] = identity
    return DataflowHandlerRegistry(
        bindings_by_key=MappingProxyType(bindings),
        identities_by_stage_id=MappingProxyType(by_stage_id),
        identities_by_call=MappingProxyType(by_call),
    )


def handler_operator_kind(stage: DataflowStage) -> str:
    if stage.call is None:
        raise ValueError(f"Dataflow stage {stage.name!r} has no operator call")
    if stage.kind is DataflowStageKind.MAP:
        return stage.call.kind.value
    return stage.kind.value
