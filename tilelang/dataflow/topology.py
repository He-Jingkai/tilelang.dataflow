"""Topology metadata used by the Dataflow scheduler mock."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUTopology:
    sm_count: int
    cluster_size: int = 1

    def __post_init__(self) -> None:
        if self.sm_count <= 0:
            raise ValueError(f"GPUTopology sm_count must be positive, got {self.sm_count}")
        if self.cluster_size <= 0:
            raise ValueError(f"GPUTopology cluster_size must be positive, got {self.cluster_size}")
        if self.cluster_size > self.sm_count:
            raise ValueError(
                f"GPUTopology cluster_size must not exceed sm_count, got cluster_size={self.cluster_size}, sm_count={self.sm_count}"
            )

    @property
    def cluster_count(self) -> int:
        return (self.sm_count + self.cluster_size - 1) // self.cluster_size

    def cluster_id(self, sm_id: int) -> int:
        if sm_id < 0 or sm_id >= self.sm_count:
            raise ValueError(f"SM id {sm_id} is outside topology with {self.sm_count} SMs")
        return sm_id // self.cluster_size

    def cluster_rank(self, sm_id: int) -> int:
        if sm_id < 0 or sm_id >= self.sm_count:
            raise ValueError(f"SM id {sm_id} is outside topology with {self.sm_count} SMs")
        return sm_id % self.cluster_size

    def same_cluster(self, lhs_sm: int, rhs_sm: int) -> bool:
        return self.cluster_id(lhs_sm) == self.cluster_id(rhs_sm)
