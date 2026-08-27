"""Typed, immutable scheduler configuration."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import math
from typing import Any
from collections.abc import Callable

from .implementation_registry import (
    dataflow_implementation_registry,
    register_scheduler_option_implementation,
    scheduler_option_implementation_id,
)
from .scheduler_cost_model import DataflowCostModelConfig


DATAFLOW_SCHEDULER_CONFIG_SCHEMA_VERSION = 2


class DataflowSchedulerOptionCategory(str, Enum):
    SEMANTIC = "semantic"
    POLICY = "policy"
    SEARCH = "search"
    COST = "cost"
    DEBUG = "debug"


class DataflowSchedulerOptionType(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    INT_SET = "int_set"


@dataclass(frozen=True)
class DataflowSchedulerOptionSpec:
    option_name: str
    value_type: DataflowSchedulerOptionType
    category: DataflowSchedulerOptionCategory
    implementation_id: str

    def parse(self, value: Any) -> Any:
        if self.value_type is DataflowSchedulerOptionType.BOOL:
            if isinstance(value, bool):
                return value
            normalized = str(value).strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{self.option_name} must be a bool, got {value!r}")
        if self.value_type is DataflowSchedulerOptionType.INT:
            try:
                return int(value)
            except (TypeError, ValueError) as err:
                raise ValueError(f"{self.option_name} must be an integer, got {value!r}") from err
        if self.value_type is DataflowSchedulerOptionType.FLOAT:
            try:
                result = float(value)
            except (TypeError, ValueError) as err:
                raise ValueError(f"{self.option_name} must be a float, got {value!r}") from err
            if not math.isfinite(result):
                raise ValueError(f"{self.option_name} must be finite, got {value!r}")
            return result
        if self.value_type is DataflowSchedulerOptionType.INT_SET:
            if isinstance(value, str):
                items = value.replace(";", ",").split(",")
            else:
                try:
                    items = tuple(value)
                except TypeError as err:
                    raise TypeError(f"{self.option_name} must contain integer ids, got {value!r}") from err
            try:
                return tuple(sorted({int(item) for item in items if str(item).strip()}))
            except (TypeError, ValueError) as err:
                raise ValueError(f"{self.option_name} must contain integer ids, got {value!r}") from err
        return str(value)


_BOOL_OPTIONS = frozenset(
    {
        "avoid_tiny_segment_copack",
        "balanced_dag_local_search",
        "balanced_dag_sched",
        "balance_cluster_segment_chunks",
        "balance_critical_cluster_segment_chunks",
        "balance_tree_owner_work_chunks",
        "balance_tree_owner_work_dp",
        "balance_tree_leaf_pair_skew",
        "chunk_length_search",
        "chunk_length_search_all_donors",
        "chunk_swap_all_chunks",
        "chunk_swap_search",
        "critical_path_split",
        "cross_cluster_long_split",
        "cross_cluster_root_global_candidates",
        "direct_leaf_acc",
        "even_segment_chunks",
        "heap_streaming_chunks",
        "hier_cross_cluster_split",
        "global_capacity_chunks",
        "global_capacity_copacked_chunks",
        "global_capacity_minimax_chunks",
        "global_capacity_replay_refine",
        "joint_schedule",
        "joint_cluster_async_receive_pipeline",
        "joint_critical_hbm_permanent_prefetch",
        "joint_hbm_async_receive_pipeline",
        "joint_hbm_segmented_pipeline",
        "joint_hbm_spill_blocked_cluster_push",
        "joint_reuse_resident_handler_threads",
        "joint_scratch_release_queue_reorder",
        "joint_serialize_permanent_inbox_reuse",
        "level_bucket_tree",
        "merge_tiny_final_chunk",
        "ordered_interval_tree",
        "ordered_tree_min_height",
        "ordered_tree_adaptive_consumer",
        "ready_time_finalize",
        "ready_time_tree",
        "reduce_higher_level_producer",
        "root_reduce_cluster_candidates",
        "root_reduce_late_producer",
        "skip_cross_cluster_reduce",
        "skip_tiny_root_fragment",
        "schedule_pic",
    }
)
_INT_OPTIONS = frozenset(
    {
        "balanced_dag_beam",
        "balanced_dag_hbm_penalty_blocks",
        "balanced_dag_local_max_task_segments",
        "balanced_dag_local_steps",
        "balanced_dag_max_split",
        "balanced_dag_min_segment_blocks",
        "balanced_dag_split_gain_blocks",
        "chunk_length_search_max_evals",
        "chunk_length_search_min_blocks",
        "chunk_length_search_passes",
        "chunk_length_search_step_blocks",
        "chunk_swap_max_swaps",
        "critical_path_allowed_max_block_regression",
        "critical_path_allow_extra_tiny_chunks",
        "critical_path_beam",
        "critical_path_capacity_blocks",
        "critical_path_capacity_search_max_evals",
        "cross_cluster_root_candidate_limit",
        "critical_path_max_extra_chunks",
        "critical_path_max_steps",
        "critical_path_max_task_segments",
        "critical_path_min_chunk_blocks",
        "critical_path_min_segment_blocks",
        "critical_path_suffix_candidate_limit",
        "critical_path_target_cluster_limit",
        "critical_path_task_limit",
        "fused_reduce_finalize_max_arity",
        "global_capacity_replay_max_evals",
        "global_capacity_replay_max_steps",
        "iter_tail_penalty_min_blocks",
        "level0_replay_passes",
        "ordered_tree_max_reduce_arity",
        "skip_tiny_root_fragment_blocks",
        "tiny_final_chunk_blocks",
        "tiny_final_chunk_max_merged_blocks",
        "tiny_segment_copack_blocks",
    }
)
_FLOAT_OPTIONS = frozenset(
    {
        "balanced_dag_root_skew_weight",
        "balanced_dag_total_root_skew_weight",
        "balance_tree_leaf_pair_reduction_weight",
        "chunk_swap_mix_penalty_us",
        "cluster_comm_cost_us",
        "critical_path_extra_chunk_penalty_us",
        "critical_path_hbm_penalty_us",
        "critical_path_max_block_regression_penalty_us",
        "critical_path_min_gain_us",
        "critical_path_p95_weight",
        "critical_path_recv_weight",
        "critical_path_tiny_chunk_penalty_us",
        "critical_path_total_recv_weight",
        "hbm_comm_cost_us",
        "iter_tail_penalty_us",
        "ready_time_balance_weight",
        "ready_time_cluster_comm_weight",
        "ready_time_finalize_comm_penalty_us",
        "ready_time_queue_weight",
        "ready_time_recv_weight",
        "tree_leaf_order_critical_eps_us",
    }
)
_INT_SET_OPTIONS = frozenset(
    {
        "critical_path_suffix_blocks",
        "level0_priority_tasks",
    }
)
_STRING_OPTIONS = frozenset(
    {
        "balanced_dag_score",
        "chunk_swap_score",
        "cluster_task_assignment",
        "level0_queue_order",
        "manual_cluster_segments",
        "streaming_tree_consumer",
        "schedule_pic_dir",
    }
)

_SEMANTIC_OPTIONS = frozenset(
    {
        "direct_leaf_acc",
        "level_bucket_tree",
        "reduce_higher_level_producer",
        "root_reduce_late_producer",
        "skip_cross_cluster_reduce",
    }
)
_DEBUG_OPTIONS = frozenset({"schedule_pic", "schedule_pic_dir"})


def option_category(option_name: str) -> DataflowSchedulerOptionCategory:
    if option_name in _SEMANTIC_OPTIONS:
        return DataflowSchedulerOptionCategory.SEMANTIC
    if option_name in _DEBUG_OPTIONS:
        return DataflowSchedulerOptionCategory.DEBUG
    if option_name in _FLOAT_OPTIONS:
        return DataflowSchedulerOptionCategory.COST
    if (
        any(
            token in option_name
            for token in (
                "search",
                "beam",
                "max_evals",
                "max_steps",
                "passes",
                "max_swaps",
            )
        )
        or option_name in _INT_SET_OPTIONS
    ):
        return DataflowSchedulerOptionCategory.SEARCH
    return DataflowSchedulerOptionCategory.POLICY


def option_type(option_name: str) -> DataflowSchedulerOptionType:
    if option_name in _BOOL_OPTIONS:
        return DataflowSchedulerOptionType.BOOL
    if option_name in _INT_OPTIONS:
        return DataflowSchedulerOptionType.INT
    if option_name in _FLOAT_OPTIONS:
        return DataflowSchedulerOptionType.FLOAT
    if option_name in _INT_SET_OPTIONS:
        return DataflowSchedulerOptionType.INT_SET
    if option_name in _STRING_OPTIONS:
        return DataflowSchedulerOptionType.STRING
    raise AssertionError(f"Missing Dataflow scheduler option type for {option_name}")


DATAFLOW_SCHEDULER_OPTION_SPECS = tuple(
    DataflowSchedulerOptionSpec(
        option_name=name,
        value_type=option_type(name),
        category=option_category(name),
        implementation_id=scheduler_option_implementation_id(name),
    )
    for name in sorted(_BOOL_OPTIONS | _INT_OPTIONS | _FLOAT_OPTIONS | _INT_SET_OPTIONS | _STRING_OPTIONS)
)
_SPECS_BY_OPTION = {spec.option_name: spec for spec in DATAFLOW_SCHEDULER_OPTION_SPECS}
for _spec in DATAFLOW_SCHEDULER_OPTION_SPECS:
    register_scheduler_option_implementation(
        _spec.option_name,
        category=_spec.category.value,
    )


@dataclass(frozen=True)
class DataflowSchedulerOptionGroup:
    values: tuple[tuple[str, Any], ...] = ()

    def get(self, option_name: str, default: Any = None) -> Any:
        for name, value in self.values:
            if name == option_name:
                return value
        return default

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                list(value)
                if isinstance(value, tuple) and _SPECS_BY_OPTION[name].value_type is DataflowSchedulerOptionType.INT_SET
                else value
            )
            for name, value in self.values
        }


@dataclass(frozen=True)
class DataflowSchedulerSemanticConfig(DataflowSchedulerOptionGroup):
    """Correctness-affecting scheduling choices."""


@dataclass(frozen=True)
class DataflowSchedulerPolicyConfig(DataflowSchedulerOptionGroup):
    """Policy selection and deterministic assignment inputs."""


@dataclass(frozen=True)
class DataflowSearchBudget(DataflowSchedulerOptionGroup):
    """Compile-latency/search-quality limits."""


@dataclass(frozen=True)
class DataflowSchedulerCostOptions(DataflowSchedulerOptionGroup):
    """Additional scoring weights layered on the versioned cost model."""


@dataclass(frozen=True)
class DataflowSchedulerDebugConfig(DataflowSchedulerOptionGroup):
    """Visualization and diagnostic-only settings."""


@dataclass(frozen=True)
class DataflowSchedulerConfig:
    schema_version: int = DATAFLOW_SCHEDULER_CONFIG_SCHEMA_VERSION
    semantic: DataflowSchedulerSemanticConfig = DataflowSchedulerSemanticConfig()
    policy: DataflowSchedulerPolicyConfig = DataflowSchedulerPolicyConfig()
    search: DataflowSearchBudget = DataflowSearchBudget()
    cost_options: DataflowSchedulerCostOptions = DataflowSchedulerCostOptions()
    debug: DataflowSchedulerDebugConfig = DataflowSchedulerDebugConfig()
    cost_model: DataflowCostModelConfig = DataflowCostModelConfig()

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_SCHEDULER_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Dataflow scheduler config schema version "
                f"{self.schema_version}; expected {DATAFLOW_SCHEDULER_CONFIG_SCHEMA_VERSION}"
            )
        for name, expected_type in (
            ("semantic", DataflowSchedulerSemanticConfig),
            ("policy", DataflowSchedulerPolicyConfig),
            ("search", DataflowSearchBudget),
            ("cost_options", DataflowSchedulerCostOptions),
            ("debug", DataflowSchedulerDebugConfig),
            ("cost_model", DataflowCostModelConfig),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"Dataflow scheduler config {name} must be {expected_type.__name__}")

    def get(self, option_name: str, default: Any = None) -> Any:
        if option_name == "cluster_comm_cost_us":
            return self.cost_model.cluster_comm_us
        if option_name == "hbm_comm_cost_us":
            return self.cost_model.hbm_comm_us
        spec = _SPECS_BY_OPTION.get(option_name)
        if spec is None:
            raise KeyError(f"Unknown Dataflow scheduler option {option_name!r}")
        group = {
            DataflowSchedulerOptionCategory.SEMANTIC: self.semantic,
            DataflowSchedulerOptionCategory.POLICY: self.policy,
            DataflowSchedulerOptionCategory.SEARCH: self.search,
            DataflowSchedulerOptionCategory.COST: self.cost_options,
            DataflowSchedulerOptionCategory.DEBUG: self.debug,
        }[spec.category]
        return group.get(option_name, default)

    def contains(self, option_name: str) -> bool:
        """Return whether an option was explicitly supplied to this snapshot."""

        if option_name in {
            "cluster_comm_cost_us",
            "hbm_comm_cost_us",
        }:
            return True
        spec = _SPECS_BY_OPTION.get(option_name)
        if spec is None:
            raise KeyError(f"Unknown Dataflow scheduler option {option_name!r}")
        group = {
            DataflowSchedulerOptionCategory.SEMANTIC: self.semantic,
            DataflowSchedulerOptionCategory.POLICY: self.policy,
            DataflowSchedulerOptionCategory.SEARCH: self.search,
            DataflowSchedulerOptionCategory.COST: self.cost_options,
            DataflowSchedulerOptionCategory.DEBUG: self.debug,
        }[spec.category]
        return any(name == option_name for name, _ in group.values)

    def explicit_option_names(self) -> tuple[str, ...]:
        """Return the canonical names explicitly present in this snapshot."""

        return tuple(
            sorted(
                name
                for group in (
                    self.semantic,
                    self.policy,
                    self.search,
                    self.cost_options,
                    self.debug,
                )
                for name, _ in group.values
            )
        )

    def with_options(self, **options: Any) -> DataflowSchedulerConfig:
        """Return a copy with typed public option-name overrides applied."""

        if not options:
            return self
        serialized = self.to_dict()
        category_names = {
            DataflowSchedulerOptionCategory.SEMANTIC: "semantic",
            DataflowSchedulerOptionCategory.POLICY: "policy",
            DataflowSchedulerOptionCategory.SEARCH: "search",
            DataflowSchedulerOptionCategory.COST: "cost_options",
            DataflowSchedulerOptionCategory.DEBUG: "debug",
        }
        for option_name, raw_value in options.items():
            spec = _SPECS_BY_OPTION.get(option_name)
            if spec is None:
                raise TypeError(f"Unknown Dataflow scheduler option {option_name!r}")
            dataflow_implementation_registry().require_selectable(
                spec.implementation_id,
                selected_explicitly=True,
            )
            if spec.option_name == "cluster_comm_cost_us":
                serialized["cost_model"]["cluster_comm_us"] = spec.parse(raw_value)
                continue
            if spec.option_name == "hbm_comm_cost_us":
                serialized["cost_model"]["hbm_comm_us"] = spec.parse(raw_value)
                continue
            serialized[category_names[spec.category]][option_name] = spec.parse(raw_value)
        return DataflowSchedulerConfig.from_dict(serialized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic": self.semantic.to_dict(),
            "policy": self.policy.to_dict(),
            "search": self.search.to_dict(),
            "cost_options": self.cost_options.to_dict(),
            "debug": self.debug.to_dict(),
            "cost_model": self.cost_model.to_dict(),
        }

    @classmethod
    def from_options_mapping(
        cls,
        options: Mapping[str, Any],
        *,
        cost_model: DataflowCostModelConfig | None = None,
    ) -> DataflowSchedulerConfig:
        grouped: dict[DataflowSchedulerOptionCategory, list[tuple[str, Any]]] = {
            category: [] for category in DataflowSchedulerOptionCategory
        }
        resolved_cost_model = cost_model or DataflowCostModelConfig()
        for option_name, raw in options.items():
            spec = _SPECS_BY_OPTION.get(str(option_name))
            if spec is None:
                raise TypeError(f"Unknown Dataflow scheduler option {option_name!r}")
            parsed = spec.parse(raw)
            if spec.option_name == "cluster_comm_cost_us":
                resolved_cost_model = DataflowCostModelConfig.from_dict({**resolved_cost_model.to_dict(), "cluster_comm_us": parsed})
                continue
            if spec.option_name == "hbm_comm_cost_us":
                resolved_cost_model = DataflowCostModelConfig.from_dict({**resolved_cost_model.to_dict(), "hbm_comm_us": parsed})
                continue
            grouped[spec.category].append((spec.option_name, parsed))

        def canonical_values(
            category: DataflowSchedulerOptionCategory,
        ) -> tuple[tuple[str, Any], ...]:
            return tuple(sorted(grouped[category], key=lambda item: item[0]))

        cost_values = tuple(
            item
            for item in canonical_values(DataflowSchedulerOptionCategory.COST)
            if item[0] not in {"cluster_comm_cost_us", "hbm_comm_cost_us"}
        )
        return cls(
            semantic=DataflowSchedulerSemanticConfig(canonical_values(DataflowSchedulerOptionCategory.SEMANTIC)),
            policy=DataflowSchedulerPolicyConfig(canonical_values(DataflowSchedulerOptionCategory.POLICY)),
            search=DataflowSearchBudget(canonical_values(DataflowSchedulerOptionCategory.SEARCH)),
            cost_options=DataflowSchedulerCostOptions(cost_values),
            debug=DataflowSchedulerDebugConfig(canonical_values(DataflowSchedulerOptionCategory.DEBUG)),
            cost_model=resolved_cost_model,
        )

    @classmethod
    def from_options(
        cls,
        *,
        cost_model: DataflowCostModelConfig | None = None,
        **options: Any,
    ) -> DataflowSchedulerConfig:
        return cls.from_options_mapping(options, cost_model=cost_model)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowSchedulerConfig:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow scheduler config must be a mapping, got {type(value)!r}")
        options: dict[str, Any] = {}
        for category_name in ("semantic", "policy", "search", "cost_options", "debug"):
            category = value.get(category_name, {})
            if not isinstance(category, Mapping):
                raise TypeError(f"Dataflow scheduler config {category_name} must be a mapping")
            for option_name, option_value in category.items():
                spec = _SPECS_BY_OPTION.get(str(option_name))
                if spec is None:
                    raise ValueError(f"Unknown Dataflow scheduler config option {option_name!r}")
                if spec.category.value != ("cost" if category_name == "cost_options" else category_name):
                    raise ValueError(f"Dataflow scheduler option {option_name!r} belongs to {spec.category.value!r}, not {category_name!r}")
                options[spec.option_name] = option_value
        cost_model_value = value.get("cost_model", {})
        cost_model = DataflowCostModelConfig.from_dict(cost_model_value)
        result = cls.from_options_mapping(options, cost_model=cost_model)
        schema_version = int(value.get("schema_version", DATAFLOW_SCHEDULER_CONFIG_SCHEMA_VERSION))
        if schema_version != result.schema_version:
            raise ValueError(f"Unsupported Dataflow scheduler config schema version {schema_version}")
        return result


_ACTIVE_SCHEDULER_CONFIG: ContextVar[DataflowSchedulerConfig | None] = ContextVar(
    "tilelang_dataflow_scheduler_config",
    default=None,
)


def resolve_scheduler_config(
    value: DataflowSchedulerConfig | Mapping[str, Any] | None,
) -> DataflowSchedulerConfig:
    if value is None:
        return DataflowSchedulerConfig()
    if isinstance(value, DataflowSchedulerConfig):
        return value
    if isinstance(value, Mapping):
        return DataflowSchedulerConfig.from_dict(value)
    raise TypeError(f"Dataflow scheduler_config must be DataflowSchedulerConfig or a serialized mapping, got {type(value)!r}")


@contextmanager
def use_scheduler_config(config: DataflowSchedulerConfig):
    if not isinstance(config, DataflowSchedulerConfig):
        raise TypeError(f"use_scheduler_config expects DataflowSchedulerConfig, got {type(config)!r}")
    token = _ACTIVE_SCHEDULER_CONFIG.set(config)
    try:
        yield
    finally:
        _ACTIVE_SCHEDULER_CONFIG.reset(token)


def current_scheduler_config() -> DataflowSchedulerConfig:
    return _ACTIVE_SCHEDULER_CONFIG.get() or DataflowSchedulerConfig()


def activate_scheduler_config(func: Callable[..., Any]) -> Callable[..., Any]:
    """Resolve one config at the public schedule boundary and bind it for helpers."""

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        resolved = resolve_scheduler_config(kwargs.get("scheduler_config"))
        kwargs["scheduler_config"] = resolved
        with use_scheduler_config(resolved):
            return func(*args, **kwargs)

    return wrapped
