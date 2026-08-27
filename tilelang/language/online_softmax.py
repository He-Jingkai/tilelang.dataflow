"""Inline semantic combinations for online softmax recurrences."""

from __future__ import annotations

from typing import Any

from tilelang.language import (
    Parallel,
    block,
    block_attr,
    copy,
    exp2,
    fill,
    gemm,
    infinity,
    macro,
    max as max_value,
    reduce_max,
    reduce_sum,
    sync_threads,
)


@macro
def online_softmax_initialize(
    scores: Any,
    running_max: Any,
    running_sum: Any,
    state_rows: int,
) -> None:
    """Initialize online-softmax state for the logical rows in ``scores``."""

    with block("logical_row_initialization"):
        block_attr(
            {
                "tl.logical_gemm_padding_source": scores.data,
                "tl.logical_gemm_padding_buffers": {
                    scores.data: 0,
                    running_max.data: 0,
                    running_sum.data: 0,
                },
                "tl.logical_gemm_padding_logical_extent": state_rows,
            }
        )
        fill(running_sum, 0)
        fill(running_max, -infinity("float32"))


@macro
def online_softmax_update(
    scores: Any,
    values: Any,
    probabilities: Any,
    output: Any,
    running_max: Any,
    previous_max: Any,
    history_scale: Any,
    tile_sum: Any,
    running_sum: Any,
    staged_history_scale: Any,
    state_rows: int,
    output_rows: int,
    columns: int,
    value_columns: int,
    stage_history_scale: bool,
    gemm_policy: Any,
) -> None:
    """Merge one score tile into online-softmax state and accumulate P @ V.

    ``scores`` must already contain scaled logits, including any semantic mask.
    The helper is target-independent: callers describe whether the history scale
    needs a shared staging buffer, while common lowering selects copy/GEMM
    implementations and pipeline synchronization.
    """

    row_buffers = {
        scores.data: 0,
        running_max.data: 0,
        previous_max.data: 0,
        history_scale.data: 0,
        tile_sum.data: 0,
        running_sum.data: 0,
    }
    if stage_history_scale:
        row_buffers[staged_history_scale.data] = 0
    with block("logical_row_recurrence"):
        block_attr(
            {
                "tl.logical_gemm_padding_source": scores.data,
                "tl.logical_gemm_padding_buffers": row_buffers,
                "tl.logical_gemm_padding_logical_extent": state_rows,
            }
        )
        copy(running_max, previous_max)
        fill(running_max, -infinity("float32"))
        reduce_max(
            scores[0:state_rows, 0:columns],
            running_max,
            dim=1,
            clear=False,
        )
        for row in Parallel(state_rows):
            running_max[row] = max_value(running_max[row], previous_max[row])
        for row in Parallel(state_rows):
            history_scale[row] = exp2(previous_max[row] - running_max[row])
        for row, column in Parallel(state_rows, columns):
            scores[row, column] = exp2(scores[row, column] - running_max[row])
        fill(tile_sum, 0)
        reduce_sum(scores[0:state_rows, 0:columns], tile_sum, dim=1)
        for row in Parallel(state_rows):
            running_sum[row] = running_sum[row] * history_scale[row] + tile_sum[row]

        if stage_history_scale:
            for row in Parallel(state_rows):
                staged_history_scale[row] = history_scale[row]
            sync_threads()
    for row, column in Parallel(output_rows, value_columns):
        if stage_history_scale:
            output[row, column] *= staged_history_scale[row]
        else:
            output[row, column] *= history_scale[row]

    for row, column in Parallel(output_rows, columns):
        probabilities[row, column] = scores[row, column]
    if stage_history_scale:
        sync_threads()
    gemm(probabilities, values, output, policy=gemm_policy)


__all__ = ["online_softmax_initialize", "online_softmax_update"]
