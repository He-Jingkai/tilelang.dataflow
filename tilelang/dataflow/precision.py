"""Typed accumulator precision contracts and compiler-owned selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from collections.abc import Mapping

from .abi_schema import DATAFLOW_TENSOR_DTYPE_BFLOAT, DATAFLOW_TENSOR_DTYPE_FLOAT
from .dtype_registry import normalize_dtype_name, require_dataflow_dtype
from .implementation_registry import dataflow_implementation_registry


DATAFLOW_PRECISION_POLICY_SCHEMA_VERSION = 1
DATAFLOW_PRECISION_PLAN_SCHEMA_VERSION = 1
DATAFLOW_PRECISION_IMPLEMENTATION_VERSION = "dataflow.precision.accumulator.v1"

DATAFLOW_PRECISION_STRICT = "strict"
DATAFLOW_PRECISION_FAST = "fast"
DATAFLOW_PRECISION_EXPLICIT = "explicit"
DATAFLOW_PRECISION_MODES = (
    DATAFLOW_PRECISION_STRICT,
    DATAFLOW_PRECISION_FAST,
    DATAFLOW_PRECISION_EXPLICIT,
)


def finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite non-negative number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return normalized


def normalized_accumulator_dtype(value: Any, context: str) -> str:
    normalized = normalize_dtype_name(value)
    info = require_dataflow_dtype(normalized)
    if info.tensor_dtype_code not in {
        DATAFLOW_TENSOR_DTYPE_FLOAT,
        DATAFLOW_TENSOR_DTYPE_BFLOAT,
    }:
        raise TypeError(f"{context} must be a floating-point dtype, got {value!r}")
    if not info.primfunc_scalar_supported:
        raise NotImplementedError(f"{context} dtype {info.name!r} is not supported for Dataflow scalar accumulation")
    return info.name


@dataclass(frozen=True)
class DataflowErrorBudget:
    """User-owned numerical budget for an explicitly requested fast policy."""

    max_absolute_error: float = 0.0
    max_relative_error: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_absolute_error",
            finite_nonnegative(self.max_absolute_error, "max_absolute_error"),
        )
        object.__setattr__(
            self,
            "max_relative_error",
            finite_nonnegative(self.max_relative_error, "max_relative_error"),
        )

    @property
    def permits_approximation(self) -> bool:
        return self.max_absolute_error > 0 or self.max_relative_error > 0

    def to_dict(self) -> dict[str, float]:
        return {
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowErrorBudget:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow error budget must be a mapping, got {type(value)!r}")
        return cls(
            max_absolute_error=value.get("max_absolute_error", 0.0),
            max_relative_error=value.get("max_relative_error", 0.0),
        )


@dataclass(frozen=True)
class DataflowPrecisionPolicy:
    """Compile-boundary policy for all declared accumulator contracts."""

    mode: str = DATAFLOW_PRECISION_STRICT
    accumulator_dtype: str | None = None
    error_budget: DataflowErrorBudget | None = None
    schema_version: int = DATAFLOW_PRECISION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_PRECISION_POLICY_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Dataflow precision policy schema version "
                f"{self.schema_version}; expected {DATAFLOW_PRECISION_POLICY_SCHEMA_VERSION}"
            )
        if not isinstance(self.mode, str):
            raise TypeError(f"Dataflow precision mode must be a string, got {self.mode!r}")
        mode = self.mode.strip().lower()
        if mode not in DATAFLOW_PRECISION_MODES:
            expected = ", ".join(DATAFLOW_PRECISION_MODES)
            raise ValueError(f"Unsupported Dataflow precision mode {self.mode!r}; expected one of: {expected}")
        object.__setattr__(self, "mode", mode)
        dtype = self.accumulator_dtype
        if mode == DATAFLOW_PRECISION_EXPLICIT:
            if dtype is None:
                raise ValueError("explicit Dataflow precision mode requires accumulator_dtype")
            dtype = normalized_accumulator_dtype(dtype, "explicit accumulator")
        elif dtype is not None:
            raise ValueError(f"Dataflow precision mode {mode!r} cannot set accumulator_dtype; use mode='explicit'")
        object.__setattr__(self, "accumulator_dtype", dtype)
        budget = self.error_budget
        if budget is not None and not isinstance(budget, DataflowErrorBudget):
            if not isinstance(budget, Mapping):
                raise TypeError("Dataflow precision error_budget must be DataflowErrorBudget or a mapping")
            budget = DataflowErrorBudget.from_dict(budget)
            object.__setattr__(self, "error_budget", budget)
        if mode == DATAFLOW_PRECISION_FAST:
            if budget is None or not budget.permits_approximation:
                raise ValueError("fast Dataflow precision mode requires a non-zero explicit error_budget")
        elif budget is not None:
            raise ValueError("Dataflow precision error_budget is only valid for mode='fast'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "accumulator_dtype": self.accumulator_dtype,
            "error_budget": None if self.error_budget is None else self.error_budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowPrecisionPolicy:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow precision policy must be a mapping, got {type(value)!r}")
        raw_budget = value.get("error_budget")
        return cls(
            schema_version=int(value.get("schema_version", DATAFLOW_PRECISION_POLICY_SCHEMA_VERSION)),
            mode=value.get("mode", DATAFLOW_PRECISION_STRICT),
            accumulator_dtype=value.get("accumulator_dtype"),
            error_budget=(None if raw_budget is None else DataflowErrorBudget.from_dict(raw_budget)),
        )


@dataclass(frozen=True)
class DataflowAccumulatorContract:
    """Operator declaration for one accumulator specialization constant."""

    contract_id: str
    allowed_dtypes: tuple[str, ...]
    strict_dtype: str
    minimum_dtype: str | None = None
    reduction_order: str = "operator_declared"

    def __post_init__(self) -> None:
        if not isinstance(self.contract_id, str) or not self.contract_id.strip():
            raise ValueError("Dataflow accumulator contract_id must be a non-empty string")
        object.__setattr__(self, "contract_id", self.contract_id.strip())
        if not isinstance(self.allowed_dtypes, (tuple, list)) or not self.allowed_dtypes:
            raise ValueError("Dataflow accumulator allowed_dtypes must be a non-empty sequence")
        normalized = tuple(normalized_accumulator_dtype(dtype, "Dataflow accumulator contract") for dtype in self.allowed_dtypes)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Dataflow accumulator allowed_dtypes contains duplicates: {normalized!r}")
        object.__setattr__(self, "allowed_dtypes", normalized)
        strict_dtype = normalized_accumulator_dtype(
            self.strict_dtype,
            "Dataflow strict accumulator",
        )
        if strict_dtype not in normalized:
            raise ValueError(f"Dataflow strict accumulator dtype {strict_dtype!r} is not in allowed_dtypes={normalized!r}")
        object.__setattr__(self, "strict_dtype", strict_dtype)
        minimum_dtype = self.minimum_dtype
        if minimum_dtype is not None:
            minimum_dtype = normalized_accumulator_dtype(
                minimum_dtype,
                "Dataflow minimum accumulator",
            )
            minimum_bits = require_dataflow_dtype(minimum_dtype).dtype_bits
            below_minimum = tuple(dtype for dtype in normalized if require_dataflow_dtype(dtype).dtype_bits < minimum_bits)
            if below_minimum:
                raise ValueError(f"Dataflow accumulator dtypes {below_minimum!r} are below minimum_dtype {minimum_dtype!r}")
            object.__setattr__(self, "minimum_dtype", minimum_dtype)
        if not isinstance(self.reduction_order, str) or not self.reduction_order.strip():
            raise ValueError("Dataflow accumulator reduction_order must be a non-empty string")
        object.__setattr__(self, "reduction_order", self.reduction_order.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "allowed_dtypes": list(self.allowed_dtypes),
            "strict_dtype": self.strict_dtype,
            "minimum_dtype": self.minimum_dtype,
            "reduction_order": self.reduction_order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowAccumulatorContract:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow accumulator contract must be a mapping, got {type(value)!r}")
        return cls(
            contract_id=str(value.get("contract_id", "")),
            allowed_dtypes=tuple(value.get("allowed_dtypes", ())),
            strict_dtype=value.get("strict_dtype"),
            minimum_dtype=value.get("minimum_dtype"),
            reduction_order=value.get("reduction_order", "operator_declared"),
        )


def normalize_accumulator_contracts(
    attrs: Mapping[str, Any],
) -> dict[str, DataflowAccumulatorContract]:
    raw_contracts = attrs.get("accumulator_contracts")
    if raw_contracts is None:
        return {}
    if not isinstance(raw_contracts, Mapping):
        raise TypeError("Dataflow accumulator_contracts must be a mapping")
    constants = attrs.get("specialization_constants", {})
    if not isinstance(constants, Mapping):
        raise TypeError("Dataflow specialization_constants must be a mapping")
    contracts: dict[str, DataflowAccumulatorContract] = {}
    for constant_name, raw_contract in raw_contracts.items():
        if not isinstance(constant_name, str) or not constant_name.isidentifier() or constant_name == "T":
            raise TypeError(f"Dataflow accumulator contract keys must be specialization constant identifiers, got {constant_name!r}")
        if constant_name not in constants:
            raise ValueError(f"Dataflow accumulator contract {constant_name!r} does not name a declared specialization constant")
        contract = (
            raw_contract if isinstance(raw_contract, DataflowAccumulatorContract) else DataflowAccumulatorContract.from_dict(raw_contract)
        )
        declared_dtype = normalized_accumulator_dtype(
            constants[constant_name],
            f"Dataflow accumulator specialization constant {constant_name!r}",
        )
        if declared_dtype != contract.strict_dtype:
            raise ValueError(
                f"Dataflow accumulator specialization constant {constant_name!r} defaults to "
                f"{declared_dtype!r}, expected strict_dtype={contract.strict_dtype!r}"
            )
        contracts[constant_name] = contract
    return contracts


@dataclass(frozen=True)
class DataflowAccumulatorResolution:
    operator_id: int
    operator_kind: str
    specialization_constant: str
    contract: DataflowAccumulatorContract
    policy_mode: str
    selected_dtype: str
    selection_reason: str
    error_budget: DataflowErrorBudget | None
    implementation_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "operator_kind": self.operator_kind,
            "specialization_constant": self.specialization_constant,
            "contract": self.contract.to_dict(),
            "policy_mode": self.policy_mode,
            "selected_dtype": self.selected_dtype,
            "selection_reason": self.selection_reason,
            "error_budget": None if self.error_budget is None else self.error_budget.to_dict(),
            "reduction_order": self.contract.reduction_order,
            "implementation_id": self.implementation_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowAccumulatorResolution:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow accumulator resolution must be a mapping, got {type(value)!r}")
        raw_budget = value.get("error_budget")
        return cls(
            operator_id=int(value["operator_id"]),
            operator_kind=str(value["operator_kind"]),
            specialization_constant=str(value["specialization_constant"]),
            contract=DataflowAccumulatorContract.from_dict(value["contract"]),
            policy_mode=str(value["policy_mode"]),
            selected_dtype=normalized_accumulator_dtype(
                value["selected_dtype"],
                "Dataflow selected accumulator",
            ),
            selection_reason=str(value["selection_reason"]),
            error_budget=(None if raw_budget is None else DataflowErrorBudget.from_dict(raw_budget)),
            implementation_id=str(value["implementation_id"]),
        )


@dataclass(frozen=True)
class DataflowPrecisionPlan:
    policy: DataflowPrecisionPolicy
    resolutions: tuple[DataflowAccumulatorResolution, ...] = ()
    schema_version: int = DATAFLOW_PRECISION_PLAN_SCHEMA_VERSION
    implementation_version: str = DATAFLOW_PRECISION_IMPLEMENTATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_PRECISION_PLAN_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Dataflow precision plan schema version "
                f"{self.schema_version}; expected {DATAFLOW_PRECISION_PLAN_SCHEMA_VERSION}"
            )
        if self.implementation_version != DATAFLOW_PRECISION_IMPLEMENTATION_VERSION:
            raise ValueError(f"Unsupported Dataflow precision implementation version {self.implementation_version!r}")
        if not isinstance(self.policy, DataflowPrecisionPolicy):
            raise TypeError("Dataflow precision plan policy must be DataflowPrecisionPolicy")
        keys = [
            (
                item.operator_id,
                item.operator_kind,
                item.specialization_constant,
            )
            for item in self.resolutions
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Dataflow precision plan contains duplicate accumulator resolutions")

    def specialization_overrides(self) -> dict[tuple[int, str], dict[str, str]]:
        overrides: dict[tuple[int, str], dict[str, str]] = {}
        for item in self.resolutions:
            overrides.setdefault(
                (item.operator_id, item.operator_kind),
                {},
            )[item.specialization_constant] = item.selected_dtype
        return overrides

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "implementation_version": self.implementation_version,
            "policy": self.policy.to_dict(),
            "resolutions": [item.to_dict() for item in self.resolutions],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowPrecisionPlan:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow precision plan must be a mapping, got {type(value)!r}")
        return cls(
            schema_version=int(value.get("schema_version", DATAFLOW_PRECISION_PLAN_SCHEMA_VERSION)),
            implementation_version=str(
                value.get(
                    "implementation_version",
                    DATAFLOW_PRECISION_IMPLEMENTATION_VERSION,
                )
            ),
            policy=DataflowPrecisionPolicy.from_dict(value.get("policy", {})),
            resolutions=tuple(DataflowAccumulatorResolution.from_dict(item) for item in value.get("resolutions", ())),
        )


def select_accumulator_dtype(
    contract: DataflowAccumulatorContract,
    policy: DataflowPrecisionPolicy,
) -> tuple[str, str]:
    if policy.mode == DATAFLOW_PRECISION_STRICT:
        return contract.strict_dtype, "strict_contract_default"
    if policy.mode == DATAFLOW_PRECISION_EXPLICIT:
        assert policy.accumulator_dtype is not None
        selected = policy.accumulator_dtype
        if selected not in contract.allowed_dtypes:
            raise ValueError(
                f"explicit accumulator dtype {selected!r} is not allowed by contract "
                f"{contract.contract_id!r}; allowed={contract.allowed_dtypes!r}"
            )
        return selected, "explicit_compile_policy"

    selected = min(
        contract.allowed_dtypes,
        key=lambda dtype: (
            require_dataflow_dtype(dtype).dtype_bits,
            contract.allowed_dtypes.index(dtype),
        ),
    )
    return selected, "fast_policy_lowest_legal_dtype"


def resolve_program_precision(
    program: Any,
    policy: DataflowPrecisionPolicy,
) -> DataflowPrecisionPlan:
    """Resolve all program accumulator declarations using stable handler identities."""

    from .handler_identity import build_handler_registry
    from .program import DataflowProgram

    if not isinstance(program, DataflowProgram):
        raise TypeError(f"resolve_program_precision expects DataflowProgram, got {program!r}")
    if not isinstance(policy, DataflowPrecisionPolicy):
        raise TypeError(f"resolve_program_precision expects DataflowPrecisionPolicy, got {policy!r}")
    registry = build_handler_registry(program)
    resolutions: list[DataflowAccumulatorResolution] = []
    for binding_key in sorted(registry.bindings_by_key):
        binding = registry.bindings_by_key[binding_key]
        contracts = normalize_accumulator_contracts(binding.call.operator.attrs)
        for constant_name, contract in sorted(contracts.items()):
            selected_dtype, reason = select_accumulator_dtype(contract, policy)
            implementation_id = f"{DATAFLOW_PRECISION_IMPLEMENTATION_VERSION}:{selected_dtype}"
            dataflow_implementation_registry().require_selectable(
                implementation_id,
                selected_explicitly=policy.mode == DATAFLOW_PRECISION_EXPLICIT,
            )
            resolutions.append(
                DataflowAccumulatorResolution(
                    operator_id=binding.identity.operator_id,
                    operator_kind=binding.identity.operator_kind,
                    specialization_constant=constant_name,
                    contract=contract,
                    policy_mode=policy.mode,
                    selected_dtype=selected_dtype,
                    selection_reason=reason,
                    error_budget=policy.error_budget,
                    implementation_id=implementation_id,
                )
            )
    return DataflowPrecisionPlan(policy=policy, resolutions=tuple(resolutions))
