"""ITER handler range-bucket variant helpers."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from .handler_identity import (
    ITER_RANGE_EXACT_LENGTH,
    ITER_RANGE_GENERIC,
    ITER_RANGE_TILE_COUNT,
    IterRangeSpecialization,
)

ITER_RANGE_BUCKET_AUTO: tuple[int | str, ...] = (1, 2, 4, ITER_RANGE_GENERIC)


def normalize_iter_range_buckets(value: Any) -> tuple[int | str, ...] | None:
    if value is None or value is False:
        return None
    if isinstance(value, str):
        raw_value = value.strip()
        if raw_value.lower() in {"", "0", "false", "none", "off"}:
            return None
        if raw_value.lower() == "auto":
            items: Sequence[Any] = ITER_RANGE_BUCKET_AUTO
        else:
            items = [item.strip() for item in raw_value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        items = value
    else:
        raise TypeError(
            "Dataflow iter_range_buckets must be None, 'auto', a comma-separated string, "
            f"or a sequence of positive integers/'generic', got {value!r}"
        )

    numeric: set[int] = set()
    has_generic = False
    for item in items:
        if isinstance(item, str) and item.strip().lower() == ITER_RANGE_GENERIC:
            has_generic = True
            continue
        try:
            bucket = int(item)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Dataflow iter_range_buckets entries must be positive integers or 'generic', got {item!r}") from err
        if bucket <= 0:
            raise ValueError(f"Dataflow iter_range_buckets entries must be positive integers, got {item!r}")
        numeric.add(bucket)
    if not numeric and not has_generic:
        return None
    return tuple(sorted(numeric)) + (ITER_RANGE_GENERIC,)


def normalize_iter_range_bucket_size(value: Any, *, fallback: int) -> int:
    raw_value = fallback if value is None else value
    try:
        result = int(raw_value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Dataflow iter_range_bucket_size must be a positive integer, got {raw_value!r}") from err
    if result <= 0:
        raise ValueError(f"Dataflow iter_range_bucket_size must be a positive integer, got {raw_value!r}")
    return result


def normalize_iter_range_exact_lengths(value: Any) -> tuple[int, ...] | None:
    if value is None or value is False:
        return None
    if isinstance(value, str):
        raw_value = value.strip()
        if raw_value.lower() in {"", "0", "false", "none", "off"}:
            return None
        items: Sequence[Any] = [item.strip() for item in raw_value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        items = value
    else:
        raise TypeError(
            f"Dataflow iter_range_exact_lengths must be None, a comma-separated string, or a sequence of positive integers, got {value!r}"
        )

    lengths: set[int] = set()
    for item in items:
        try:
            length = int(item)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Dataflow iter_range_exact_lengths entries must be positive integers, got {item!r}") from err
        if length <= 0:
            raise ValueError(f"Dataflow iter_range_exact_lengths entries must be positive integers, got {item!r}")
        lengths.add(length)
    if not lengths:
        return None
    return tuple(sorted(lengths))


def iter_range_bucket_for_length(
    range_length: int,
    *,
    bucket_size: int,
    buckets: tuple[int | str, ...] | None,
) -> int | str | None:
    if not buckets:
        return None
    if range_length <= 0:
        return ITER_RANGE_GENERIC
    if range_length % bucket_size != 0:
        return ITER_RANGE_GENERIC
    tile_count = max(1, math.ceil(range_length / bucket_size))
    return tile_count if tile_count in buckets else ITER_RANGE_GENERIC


def iter_range_specialization_for_length(
    range_length: int,
    *,
    bucket_size: int,
    buckets: tuple[int | str, ...] | None,
    exact_lengths: tuple[int, ...] | None = None,
) -> IterRangeSpecialization:
    """Select structured specialization metadata without encoding it in a name."""

    if exact_lengths and range_length in exact_lengths:
        return IterRangeSpecialization(
            mode=ITER_RANGE_EXACT_LENGTH,
            value=range_length,
        )
    bucket = iter_range_bucket_for_length(
        range_length,
        bucket_size=bucket_size,
        buckets=buckets,
    )
    if bucket is None or bucket == ITER_RANGE_GENERIC:
        return IterRangeSpecialization()
    return IterRangeSpecialization(
        mode=ITER_RANGE_TILE_COUNT,
        value=int(bucket),
    )
