"""Typed physical semantics for Dataflow operator bodies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from collections.abc import Mapping


DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_SCHEMA_VERSION = 2
_LEGACY_DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_SCHEMA_VERSION = 1
DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_ATTR = "physical_contract"
DATAFLOW_OPERATOR_PHYSICAL_IMPLEMENTATION = "dataflow.primfunc.physical_contract.v1"
DATAFLOW_TERMINAL_SIDE_EFFECT_IMPLEMENTATION = "dataflow.terminal.side_effect.v1"

DATAFLOW_INPUT_SLOTS_INDEXED = "indexed"
DATAFLOW_INPUT_SLOTS_CONTIGUOUS = "contiguous"
DATAFLOW_INPUT_SLOT_MODES = (
    DATAFLOW_INPUT_SLOTS_INDEXED,
    DATAFLOW_INPUT_SLOTS_CONTIGUOUS,
)

DATAFLOW_OUTPUT_SLOT_STORED = "stored"
DATAFLOW_OUTPUT_SLOT_DIRECT = "direct"
DATAFLOW_OUTPUT_SLOT_NONE = "none"
DATAFLOW_OUTPUT_SLOT_MODES = (
    DATAFLOW_OUTPUT_SLOT_STORED,
    DATAFLOW_OUTPUT_SLOT_DIRECT,
    DATAFLOW_OUTPUT_SLOT_NONE,
)

_RETIRED_RAW_ATTRS = frozenset(
    {
        "raw_tilelang_contiguous_map_inputs",
        "raw_tilelang_direct_output",
        "raw_tilelang_return_warp_groups",
    }
)


def choice(value: Any, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {value!r}")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"unsupported {name} {value!r}; expected one of {choices!r}")
    return normalized


@dataclass(frozen=True)
class DataflowOperatorPhysicalContract:
    input_slots: str = DATAFLOW_INPUT_SLOTS_INDEXED
    output_slot: str = DATAFLOW_OUTPUT_SLOT_STORED
    output_alias_input_indices: tuple[int, ...] = ()
    return_warp_groups: tuple[int, ...] = ()
    schema_version: int = DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {
            _LEGACY_DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_SCHEMA_VERSION,
            DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported Dataflow operator physical contract schema version {self.schema_version}")
        object.__setattr__(
            self,
            "input_slots",
            choice(self.input_slots, "input slot mode", DATAFLOW_INPUT_SLOT_MODES),
        )
        object.__setattr__(
            self,
            "output_slot",
            choice(self.output_slot, "output slot mode", DATAFLOW_OUTPUT_SLOT_MODES),
        )
        groups = tuple(self.return_warp_groups)
        if any(isinstance(group, bool) or not isinstance(group, int) or group < 0 for group in groups):
            raise ValueError("Dataflow return_warp_groups must contain non-negative integers")
        if len(groups) != len(set(groups)):
            raise ValueError("Dataflow return_warp_groups must be unique")
        object.__setattr__(self, "return_warp_groups", groups)
        alias_indices = tuple(self.output_alias_input_indices)
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in alias_indices):
            raise ValueError("Dataflow output_alias_input_indices must contain non-negative integers")
        if len(alias_indices) != len(set(alias_indices)):
            raise ValueError("Dataflow output_alias_input_indices must be unique")
        object.__setattr__(self, "output_alias_input_indices", alias_indices)
        if self.output_slot == DATAFLOW_OUTPUT_SLOT_NONE and groups:
            raise ValueError("an outputless Dataflow operator cannot declare return_warp_groups")
        if self.output_slot == DATAFLOW_OUTPUT_SLOT_NONE and alias_indices:
            raise ValueError("an outputless Dataflow operator cannot declare output aliases")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(include_fingerprint=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "implementation": DATAFLOW_OPERATOR_PHYSICAL_IMPLEMENTATION,
            "input_slots": self.input_slots,
            "output_slot": self.output_slot,
            "output_alias_input_indices": list(self.output_alias_input_indices),
            "return_warp_groups": list(self.return_warp_groups),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowOperatorPhysicalContract:
        if not isinstance(value, Mapping):
            raise TypeError("Dataflow operator physical contract must be a mapping")
        allowed = {
            "schema_version",
            "implementation",
            "input_slots",
            "output_slot",
            "output_alias_input_indices",
            "return_warp_groups",
            "fingerprint",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"Dataflow operator physical contract has unknown fields {sorted(unknown)!r}")
        contract = cls(
            input_slots=value.get("input_slots", DATAFLOW_INPUT_SLOTS_INDEXED),
            output_slot=value.get("output_slot", DATAFLOW_OUTPUT_SLOT_STORED),
            output_alias_input_indices=tuple(value.get("output_alias_input_indices", ())),
            return_warp_groups=tuple(value.get("return_warp_groups", ())),
            schema_version=int(
                value.get(
                    "schema_version",
                    DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_SCHEMA_VERSION,
                )
            ),
        )
        if value.get("implementation") not in {
            None,
            DATAFLOW_OPERATOR_PHYSICAL_IMPLEMENTATION,
        }:
            raise ValueError("Dataflow operator physical implementation is stale")
        if value.get("fingerprint") not in {None, contract.fingerprint}:
            raise ValueError("Dataflow operator physical contract fingerprint is stale")
        return contract


def normalize_operator_physical_contract(
    value: DataflowOperatorPhysicalContract | Mapping[str, Any] | None,
) -> DataflowOperatorPhysicalContract:
    if value is None:
        return DataflowOperatorPhysicalContract()
    if isinstance(value, DataflowOperatorPhysicalContract):
        return value
    if isinstance(value, Mapping):
        return DataflowOperatorPhysicalContract.from_dict(value)
    raise TypeError("physical_contract must be DataflowOperatorPhysicalContract or a mapping")


def reject_retired_raw_physical_attrs(attrs: Mapping[str, Any]) -> None:
    retired = sorted(_RETIRED_RAW_ATTRS.intersection(attrs))
    if retired:
        raise TypeError(f"raw TileLang operator attrs were retired; use DataflowOperatorPhysicalContract: {retired!r}")


def operator_physical_contract(
    attrs: Mapping[str, Any],
) -> DataflowOperatorPhysicalContract:
    value = attrs.get(DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_ATTR)
    if value is None:
        return DataflowOperatorPhysicalContract()
    if not isinstance(value, DataflowOperatorPhysicalContract):
        raise TypeError("Dataflow operator physical contract attr must be typed")
    return value


__all__ = [
    "DATAFLOW_INPUT_SLOT_MODES",
    "DATAFLOW_INPUT_SLOTS_CONTIGUOUS",
    "DATAFLOW_INPUT_SLOTS_INDEXED",
    "DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_ATTR",
    "DATAFLOW_OPERATOR_PHYSICAL_CONTRACT_SCHEMA_VERSION",
    "DATAFLOW_OPERATOR_PHYSICAL_IMPLEMENTATION",
    "DATAFLOW_OUTPUT_SLOT_DIRECT",
    "DATAFLOW_OUTPUT_SLOT_MODES",
    "DATAFLOW_OUTPUT_SLOT_NONE",
    "DATAFLOW_OUTPUT_SLOT_STORED",
    "DATAFLOW_TERMINAL_SIDE_EFFECT_IMPLEMENTATION",
    "DataflowOperatorPhysicalContract",
    "normalize_operator_physical_contract",
    "operator_physical_contract",
    "reject_retired_raw_physical_attrs",
]
