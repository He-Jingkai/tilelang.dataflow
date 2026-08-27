"""Versioned compiler-owned memory placement policy for Dataflow artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence

from .implementation_registry import dataflow_implementation_registry

DATAFLOW_MEMORY_POLICY_SCHEMA_VERSION = 1
DATAFLOW_MEMORY_PLAN_SCHEMA_VERSION = 1
DATAFLOW_MEMORY_PLANNER_VERSION = "dataflow.memory.v1"

DATAFLOW_MEMORY_AUTO = "auto"
DATAFLOW_MEMORY_SHARED = "shared"
DATAFLOW_MEMORY_SCRATCH_BACKED = "scratch_backed"
DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL = "hbm_direct_global"
DATAFLOW_MEMORY_PLACEMENTS = (
    DATAFLOW_MEMORY_AUTO,
    DATAFLOW_MEMORY_SHARED,
    DATAFLOW_MEMORY_SCRATCH_BACKED,
    DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL,
)


class DataflowMemoryPlanningError(RuntimeError):
    """Raised when no requested memory placement satisfies the resource contract."""


@dataclass(frozen=True)
class DataflowMemoryPolicy:
    mode: str = DATAFLOW_MEMORY_AUTO
    schema_version: int = DATAFLOW_MEMORY_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_MEMORY_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Dataflow memory policy schema version {self.schema_version}; expected {DATAFLOW_MEMORY_POLICY_SCHEMA_VERSION}"
            )
        if not isinstance(self.mode, str):
            raise TypeError(f"Dataflow memory policy mode must be a string, got {self.mode!r}")
        mode = self.mode.strip().lower()
        if mode not in DATAFLOW_MEMORY_PLACEMENTS:
            expected = ", ".join(DATAFLOW_MEMORY_PLACEMENTS)
            raise ValueError(f"Unsupported Dataflow memory policy mode {self.mode!r}; expected one of: {expected}")
        object.__setattr__(self, "mode", mode)

    @property
    def considers_scratch(self) -> bool:
        return self.mode in {DATAFLOW_MEMORY_AUTO, DATAFLOW_MEMORY_SCRATCH_BACKED}

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "mode": self.mode}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowMemoryPolicy:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow memory policy must be a mapping, got {type(value)!r}")
        return cls(
            schema_version=int(value.get("schema_version", DATAFLOW_MEMORY_POLICY_SCHEMA_VERSION)),
            mode=value.get("mode", DATAFLOW_MEMORY_AUTO),
        )


def resolve_memory_policy(
    value: DataflowMemoryPolicy | Mapping[str, Any] | str | None,
) -> DataflowMemoryPolicy:
    if value is None:
        return DataflowMemoryPolicy()
    if isinstance(value, DataflowMemoryPolicy):
        return value
    if isinstance(value, str):
        return DataflowMemoryPolicy(mode=value)
    if isinstance(value, Mapping):
        return DataflowMemoryPolicy.from_dict(value)
    raise TypeError(f"Dataflow memory_policy must be DataflowMemoryPolicy, a mode string, or a mapping, got {type(value)!r}")


@dataclass(frozen=True)
class DataflowMemoryCandidate:
    candidate_id: str
    placement: str
    shared_memory_bytes: int
    shared_slot_bytes: int
    primfunc_scratch_bytes: int
    scratch_backed_slot_count: int
    scratch_backed_slot_bytes: int
    hbm_direct_global_slot_count: int
    target_shared_memory_limit: int | None
    legal: bool
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("Dataflow memory candidate_id must be a non-empty string")
        if self.placement not in DATAFLOW_MEMORY_PLACEMENTS[1:]:
            raise ValueError(f"Unsupported Dataflow memory placement {self.placement!r}")
        for name in (
            "shared_memory_bytes",
            "shared_slot_bytes",
            "primfunc_scratch_bytes",
            "scratch_backed_slot_count",
            "scratch_backed_slot_bytes",
            "hbm_direct_global_slot_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Dataflow memory candidate {name} must be non-negative")
        if self.target_shared_memory_limit is not None and (
            isinstance(self.target_shared_memory_limit, bool)
            or not isinstance(self.target_shared_memory_limit, int)
            or self.target_shared_memory_limit < 0
        ):
            raise ValueError("target_shared_memory_limit must be non-negative or None")
        if self.legal == bool(self.rejection_reasons):
            raise ValueError(
                "legal Dataflow memory candidates cannot have rejection reasons and illegal candidates require at least one reason"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "placement": self.placement,
            "shared_memory_bytes": self.shared_memory_bytes,
            "shared_slot_bytes": self.shared_slot_bytes,
            "primfunc_scratch_bytes": self.primfunc_scratch_bytes,
            "scratch_backed_slot_count": self.scratch_backed_slot_count,
            "scratch_backed_slot_bytes": self.scratch_backed_slot_bytes,
            "hbm_direct_global_slot_count": self.hbm_direct_global_slot_count,
            "target_shared_memory_limit": self.target_shared_memory_limit,
            "legal": self.legal,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowMemoryCandidate:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow memory candidate must be a mapping, got {type(value)!r}")
        return cls(
            candidate_id=str(value["candidate_id"]),
            placement=str(value["placement"]),
            shared_memory_bytes=int(value["shared_memory_bytes"]),
            shared_slot_bytes=int(value["shared_slot_bytes"]),
            primfunc_scratch_bytes=int(value["primfunc_scratch_bytes"]),
            scratch_backed_slot_count=int(value["scratch_backed_slot_count"]),
            scratch_backed_slot_bytes=int(value["scratch_backed_slot_bytes"]),
            hbm_direct_global_slot_count=int(value["hbm_direct_global_slot_count"]),
            target_shared_memory_limit=(
                None if value.get("target_shared_memory_limit") is None else int(value["target_shared_memory_limit"])
            ),
            legal=bool(value["legal"]),
            rejection_reasons=tuple(str(item) for item in value.get("rejection_reasons", ())),
        )


def make_memory_candidate(
    *,
    candidate_id: str,
    placement: str,
    shared_memory_bytes: int,
    shared_slot_bytes: int,
    primfunc_scratch_bytes: int,
    scratch_backed_slot_count: int = 0,
    scratch_backed_slot_bytes: int = 0,
    hbm_direct_global_slot_count: int = 0,
    target_shared_memory_limit: int | None,
    extra_rejection_reasons: Sequence[str] = (),
) -> DataflowMemoryCandidate:
    reasons = [str(reason) for reason in extra_rejection_reasons if str(reason)]
    if target_shared_memory_limit is not None and shared_memory_bytes > target_shared_memory_limit:
        reasons.append(f"target_shared_memory_limit_exceeded: required={shared_memory_bytes}, limit={target_shared_memory_limit}")
    return DataflowMemoryCandidate(
        candidate_id=candidate_id,
        placement=placement,
        shared_memory_bytes=int(shared_memory_bytes),
        shared_slot_bytes=int(shared_slot_bytes),
        primfunc_scratch_bytes=int(primfunc_scratch_bytes),
        scratch_backed_slot_count=int(scratch_backed_slot_count),
        scratch_backed_slot_bytes=int(scratch_backed_slot_bytes),
        hbm_direct_global_slot_count=int(hbm_direct_global_slot_count),
        target_shared_memory_limit=target_shared_memory_limit,
        legal=not reasons,
        rejection_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class DataflowMemoryPlan:
    policy: DataflowMemoryPolicy
    selected_candidate_id: str
    selection_reason: str
    candidates: tuple[DataflowMemoryCandidate, ...]
    schema_version: int = DATAFLOW_MEMORY_PLAN_SCHEMA_VERSION
    planner_version: str = DATAFLOW_MEMORY_PLANNER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_MEMORY_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Dataflow memory plan schema version {self.schema_version}; expected {DATAFLOW_MEMORY_PLAN_SCHEMA_VERSION}"
            )
        if self.planner_version != DATAFLOW_MEMORY_PLANNER_VERSION:
            raise ValueError(f"Unsupported Dataflow memory planner version {self.planner_version!r}")
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Dataflow memory plan candidate ids must be unique")
        selected = tuple(item for item in self.candidates if item.candidate_id == self.selected_candidate_id)
        if len(selected) != 1 or not selected[0].legal:
            raise ValueError("Dataflow memory plan must select exactly one legal candidate")
        if not self.selection_reason:
            raise ValueError("Dataflow memory plan requires a selection reason")

    @property
    def selected_candidate(self) -> DataflowMemoryCandidate:
        return next(item for item in self.candidates if item.candidate_id == self.selected_candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planner_version": self.planner_version,
            "policy": self.policy.to_dict(),
            "selected_candidate_id": self.selected_candidate_id,
            "selection_reason": self.selection_reason,
            "candidates": [item.to_dict() for item in self.candidates],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowMemoryPlan:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow memory plan must be a mapping, got {type(value)!r}")
        return cls(
            schema_version=int(value.get("schema_version", DATAFLOW_MEMORY_PLAN_SCHEMA_VERSION)),
            planner_version=str(value.get("planner_version", DATAFLOW_MEMORY_PLANNER_VERSION)),
            policy=DataflowMemoryPolicy.from_dict(value.get("policy", {})),
            selected_candidate_id=str(value["selected_candidate_id"]),
            selection_reason=str(value["selection_reason"]),
            candidates=tuple(DataflowMemoryCandidate.from_dict(item) for item in value.get("candidates", ())),
        )


def select_memory_plan(
    policy: DataflowMemoryPolicy,
    candidates: Sequence[DataflowMemoryCandidate],
    *,
    allow_deferred_hbm: bool = False,
) -> DataflowMemoryPlan | None:
    if not isinstance(policy, DataflowMemoryPolicy):
        raise TypeError(f"select_memory_plan expects DataflowMemoryPolicy, got {policy!r}")
    ordered = tuple(candidates)
    by_placement = {item.placement: item for item in ordered}

    if policy.mode != DATAFLOW_MEMORY_AUTO:
        selected = by_placement.get(policy.mode)
        if selected is None:
            if allow_deferred_hbm and policy.mode == DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL:
                return None
            raise DataflowMemoryPlanningError(f"requested Dataflow memory placement {policy.mode!r} was not generated")
        if not selected.legal:
            raise DataflowMemoryPlanningError(
                f"requested Dataflow memory placement {policy.mode!r} is illegal: " + "; ".join(selected.rejection_reasons)
            )
        return validated_memory_plan(
            policy=policy,
            selected_candidate_id=selected.candidate_id,
            selection_reason=f"explicit_policy:{policy.mode}",
            candidates=ordered,
        )

    legal = [item for item in ordered if item.legal]
    if legal:
        placement_priority = {
            DATAFLOW_MEMORY_SHARED: 0,
            DATAFLOW_MEMORY_SCRATCH_BACKED: 1,
            DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL: 2,
        }
        selected = min(
            legal,
            key=lambda item: (
                item.shared_memory_bytes,
                placement_priority[item.placement],
                item.candidate_id,
            ),
        )
        return validated_memory_plan(
            policy=policy,
            selected_candidate_id=selected.candidate_id,
            selection_reason=(
                "minimum_legal_shared_memory"
                if selected.placement != DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL
                else "shared_and_scratch_illegal_hbm_fallback"
            ),
            candidates=ordered,
        )
    if allow_deferred_hbm and DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL not in by_placement:
        return None
    reasons = "; ".join(f"{item.candidate_id}: {', '.join(item.rejection_reasons)}" for item in ordered)
    raise DataflowMemoryPlanningError("Dataflow memory planner found no legal placement candidate. " + reasons)


def validated_memory_plan(
    *,
    policy: DataflowMemoryPolicy,
    selected_candidate_id: str,
    selection_reason: str,
    candidates: tuple[DataflowMemoryCandidate, ...],
) -> DataflowMemoryPlan:
    plan = DataflowMemoryPlan(
        policy=policy,
        selected_candidate_id=selected_candidate_id,
        selection_reason=selection_reason,
        candidates=candidates,
    )
    dataflow_implementation_registry().require_selectable(
        f"{plan.planner_version}:{plan.selected_candidate.placement}",
        selected_explicitly=policy.mode != DATAFLOW_MEMORY_AUTO,
    )
    return plan
