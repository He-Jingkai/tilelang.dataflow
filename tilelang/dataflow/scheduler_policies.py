"""Stable scheduler policy identities and validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .implementation_registry import dataflow_implementation_registry


class DataflowSchedulerPolicy(str, Enum):
    ROUND_ROBIN = "round_robin"
    CLUSTER_LOCAL = "cluster_local"


class DataflowReduceStrategy(str, Enum):
    ALL_AT_ONCE = "all_at_once"
    STREAMING = "streaming"
    STREAMING_TREE = "streaming_tree"


@dataclass(frozen=True)
class DataflowSchedulerPolicySpec:
    policy: DataflowSchedulerPolicy
    reduce_strategies: tuple[DataflowReduceStrategy, ...]

    def supports(self, strategy: DataflowReduceStrategy) -> bool:
        return strategy in self.reduce_strategies


DATAFLOW_SCHEDULER_POLICY_SPECS = (
    DataflowSchedulerPolicySpec(
        DataflowSchedulerPolicy.ROUND_ROBIN,
        (DataflowReduceStrategy.ALL_AT_ONCE,),
    ),
    DataflowSchedulerPolicySpec(
        DataflowSchedulerPolicy.CLUSTER_LOCAL,
        tuple(DataflowReduceStrategy),
    ),
)
_POLICY_SPECS_BY_NAME = {spec.policy.value: spec for spec in DATAFLOW_SCHEDULER_POLICY_SPECS}


def normalize_scheduler_policy(value: str | DataflowSchedulerPolicy) -> str:
    normalized = value.value if isinstance(value, DataflowSchedulerPolicy) else str(value)
    try:
        normalized = DataflowSchedulerPolicy(normalized).value
    except ValueError as err:
        expected = tuple(item.value for item in DataflowSchedulerPolicy)
        raise ValueError(f"Dataflow scheduler_policy must be one of {expected!r}, got {value!r}") from err
    dataflow_implementation_registry().require_selectable(
        normalized,
        selected_explicitly=True,
    )
    return normalized


def normalize_reduce_strategy(value: str | DataflowReduceStrategy) -> str:
    normalized = value.value if isinstance(value, DataflowReduceStrategy) else str(value)
    try:
        return DataflowReduceStrategy(normalized).value
    except ValueError as err:
        expected = tuple(item.value for item in DataflowReduceStrategy)
        raise ValueError(f"Dataflow reduce_strategy must be one of {expected!r}, got {value!r}") from err


def validate_scheduler_policy(
    policy: str | DataflowSchedulerPolicy,
    reduce_strategy: str | DataflowReduceStrategy,
) -> tuple[str, str]:
    normalized_policy = normalize_scheduler_policy(policy)
    normalized_strategy = normalize_reduce_strategy(reduce_strategy)
    spec = _POLICY_SPECS_BY_NAME[normalized_policy]
    strategy = DataflowReduceStrategy(normalized_strategy)
    if not spec.supports(strategy):
        supported = tuple(item.value for item in spec.reduce_strategies)
        raise ValueError(
            f"Dataflow scheduler policy {normalized_policy!r} supports reduce strategies {supported!r}, got {normalized_strategy!r}"
        )
    return normalized_policy, normalized_strategy
