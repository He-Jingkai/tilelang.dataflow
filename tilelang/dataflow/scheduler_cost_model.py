"""Versioned cost model inputs for Dataflow scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from collections.abc import Mapping


DATAFLOW_COST_MODEL_VERSION = "dataflow.conservative.v1"


@dataclass(frozen=True)
class DataflowCostModelConfig:
    """Portable conservative defaults used by Dataflow scheduling heuristics.

    Device-specific calibration may replace these values, but the complete
    model and its version always participate in compile fingerprints and plan
    dumps.
    """

    version: str = DATAFLOW_COST_MODEL_VERSION
    cluster_comm_us: float = 1.0
    hbm_comm_us: float = 8.0
    iter_base_us: float = 5.0
    iter_per_block_us: float = 2.7
    reduce_single_us: float = 2.2
    reduce_multi_us: float = 3.2
    finalize_us: float = 3.2
    transfer_issue_us: float = 0.05
    retained_copy_us: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Dataflow cost model version must be non-empty")
        for name in (
            "cluster_comm_us",
            "hbm_comm_us",
            "iter_base_us",
            "iter_per_block_us",
            "reduce_single_us",
            "reduce_multi_us",
            "finalize_us",
            "transfer_issue_us",
            "retained_copy_us",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Dataflow cost model {name} must be a finite non-negative value, got {value!r}")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "cluster_comm_us": self.cluster_comm_us,
            "hbm_comm_us": self.hbm_comm_us,
            "iter_base_us": self.iter_base_us,
            "iter_per_block_us": self.iter_per_block_us,
            "reduce_single_us": self.reduce_single_us,
            "reduce_multi_us": self.reduce_multi_us,
            "finalize_us": self.finalize_us,
            "transfer_issue_us": self.transfer_issue_us,
            "retained_copy_us": self.retained_copy_us,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowCostModelConfig:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow cost model must be a mapping, got {type(value)!r}")
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown Dataflow cost model field(s): {', '.join(unknown)}")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__ if name in value})


def estimate_communication_cost_us(
    model: DataflowCostModelConfig,
    *,
    same_sm: bool,
    same_cluster: bool,
    force_hbm: bool = False,
) -> float:
    if same_sm:
        return 0.0
    if same_cluster and not force_hbm:
        return model.cluster_comm_us
    return model.hbm_comm_us


def estimate_iter_cost_us(
    model: DataflowCostModelConfig,
    *,
    range_length: int,
    block_size: int,
    tail_penalty_us: float = 0.0,
    tail_penalty_min_blocks: int = 1,
) -> float:
    if range_length < 0 or block_size <= 0 or tail_penalty_min_blocks <= 0:
        raise ValueError("Dataflow iter cost inputs must use non-negative lengths and positive blocks")
    blocks = max(1, math.ceil(range_length / block_size))
    cost = model.iter_base_us + model.iter_per_block_us * blocks
    if tail_penalty_us > 0.0 and blocks >= tail_penalty_min_blocks and range_length % block_size != 0:
        cost += tail_penalty_us
    return cost


def estimate_reduce_cost_us(model: DataflowCostModelConfig, *, input_count: int) -> float:
    if input_count < 0:
        raise ValueError("Dataflow reduce input count must be non-negative")
    return model.reduce_single_us if input_count <= 1 else model.reduce_multi_us


def estimate_finalize_cost_us(model: DataflowCostModelConfig) -> float:
    return model.finalize_us
