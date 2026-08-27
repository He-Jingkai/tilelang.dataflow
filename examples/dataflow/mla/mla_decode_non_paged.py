from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import tilelang.language as T
import tilelang.dataflow as df


def require_supported_tiled_nonpaged_config(config: Any) -> None:
    if config.kv_heads != 1:
        raise NotImplementedError(f"Dataflow non-paged tiled MLA currently supports kv_heads=1, got {config.kv_heads}")
    for name in (
        "batch",
        "heads",
        "kv_ctx",
        "dim",
        "pe_dim",
        "sm_count",
        "cluster_size",
        "block_n",
        "range_block_n",
        "block_h",
        "threads",
    ):
        value = getattr(config, name)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if config.output_dtype not in ("float16", "float32"):
        raise ValueError(f"output_dtype must be 'float16' or 'float32', got {config.output_dtype!r}")
    if not isinstance(config.finalize_staged_transfer, bool):
        raise TypeError("finalize_staged_transfer must be a bool")
    if config.block_n > config.kv_ctx:
        raise ValueError(f"block_n must be <= kv_ctx={config.kv_ctx}, got {config.block_n}")
    if config.range_block_n > config.kv_ctx:
        raise ValueError(f"range_block_n must be <= kv_ctx={config.kv_ctx}, got {config.range_block_n}")
    if config.block_h > config.heads:
        raise ValueError(f"block_h={config.block_h} must be <= heads={config.heads}")
    if config.heads % config.block_h != 0:
        raise ValueError(f"block_h={config.block_h} must divide heads={config.heads}")


def normalize_tiled_nonpaged_seq_lens(config: Any, seq_lens: Sequence[Any] | None = None) -> tuple[int, ...]:
    if seq_lens is None:
        return (config.kv_ctx,) * config.task_count
    head_blocks = getattr(config, "head_blocks", config.task_count // config.batch)
    if len(seq_lens) == config.batch and seq_lens:
        if is_seq_len_scalar(seq_lens[0]):
            flattened = tuple(truncate_seq_len(item) for item in seq_lens for _ in range(head_blocks))
        else:
            flattened = tuple(truncate_seq_len(item) for row in seq_lens for item in row)
    else:
        flattened = tuple(truncate_seq_len(item) for item in seq_lens)
    if len(flattened) != config.task_count:
        raise ValueError(
            f"seq_lens must provide {config.task_count} entries for batch={config.batch}, heads={config.heads}; got {len(flattened)}"
        )
    for seq_len in flattened:
        if seq_len < 1 or seq_len > config.kv_ctx:
            raise ValueError(f"seq_lens entries must be between 1 and {config.kv_ctx}, got {seq_len}")
    return flattened


def is_seq_len_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str))


def truncate_seq_len(value: Any) -> int:
    return int(float(value))


@dataclass(frozen=True)
class NonPagedTiledMLAConfig:
    batch: int = 1
    heads: int = 32
    kv_heads: int = 1
    kv_ctx: int = 64
    dim: int = 512
    pe_dim: int = 64
    sm_count: int = 132
    cluster_size: int = 2
    block_n: int = 64
    range_block_n: int = 64
    block_h: int = 32
    threads: int = 256
    output_dtype: str = "float16"
    finalize_staged_transfer: bool = False

    @property
    def head_blocks(self) -> int:
        return self.heads // self.block_h

    @property
    def task_count(self) -> int:
        return self.batch * self.head_blocks


@dataclass(frozen=True)
class NonPagedTiledMLAOperators:
    partial_type: type
    split_tile: Any
    reduce: Any
    reduce_finalizers: tuple[Any, ...]
    finalize: Any


def make_nonpaged_tiled_mla_operators(
    config: NonPagedTiledMLAConfig,
) -> NonPagedTiledMLAOperators:
    require_supported_tiled_nonpaged_config(config)
    batch_extent = config.batch
    head_extent = config.heads
    kv_head_extent = config.kv_heads
    kv_ctx_extent = config.kv_ctx
    dim = config.dim
    pe_dim = config.pe_dim
    block_n = config.block_n
    block_h = config.block_h
    iter_threads = config.threads
    output_dtype = config.output_dtype
    scale = 1.4426950408889634 / math.sqrt(config.dim + config.pe_dim)
    requested_pipeline_stages = 2 if block_n >= 64 and block_h >= 32 and dim >= 64 and pe_dim >= 64 else 0
    reduce_o_acc_dtype = "float32"
    pv_half_dim = dim // 2
    finalize_staged_transfer = config.finalize_staged_transfer

    constants = {
        "batch_extent": batch_extent,
        "head_extent": head_extent,
        "kv_head_extent": kv_head_extent,
        "kv_ctx_extent": kv_ctx_extent,
        "dim": dim,
        "pe_dim": pe_dim,
        "block_n": block_n,
        "block_h": block_h,
        "iter_threads": iter_threads,
        "output_dtype": output_dtype,
        "scale": scale,
        "requested_pipeline_stages": requested_pipeline_stages,
        "pv_half_dim": pv_half_dim,
        "finalize_staged_transfer": config.finalize_staged_transfer,
        "reduce_o_acc_dtype": reduce_o_acc_dtype,
    }

    @T.macro
    def prepare_scores(
        q,
        q_pe,
        kv,
        k_pe,
        scores,
        tile_start,
        range_end,
        masked,
    ):
        T.gemm(
            q,
            kv,
            scores,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullCol,
            clear_accum=True,
            logical_shape=(block_h, block_n, dim),
        )
        T.gemm(
            q_pe,
            k_pe,
            scores,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullCol,
            logical_shape=(block_h, block_n, pe_dim),
        )
        for row, column in T.Parallel(block_h, block_n):
            if masked:
                scores[row, column] = T.if_then_else(
                    tile_start + column < range_end,
                    scores[row, column] * T.float32(scale),
                    -T.infinity("float32"),
                )
            else:
                scores[row, column] *= T.float32(scale)

    @T.dataflow_intermediate(specialization_constants=constants)
    class NonPagedTiledMLAPartial:
        m: T.Tensor((block_h,), T.float32)
        l: T.Tensor((block_h,), T.float32)
        o_left: T.Tensor((block_h, pv_half_dim), T.float16)
        o_right: T.Tensor((block_h, pv_half_dim), T.float16)

    @T.dataflow.iter(
        range=("kv_begin", "kv_end"),
        threads=iter_threads,
        specialization_constants=constants,
        physical_contract=df.DataflowOperatorPhysicalContract(
            output_slot=df.DATAFLOW_OUTPUT_SLOT_DIRECT,
        ),
    )
    def nonpaged_tiled_mla_split_tile(
        batch: T.int32,
        head_block: T.int32,
        Q: T.Tensor((batch_extent, head_extent, dim), T.float16),
        QPe: T.Tensor((batch_extent, head_extent, pe_dim), T.float16),
        KV: T.Tensor(
            (batch_extent, kv_ctx_extent, kv_head_extent, dim),
            T.float16,
        ),
        KPe: T.Tensor(
            (batch_extent, kv_ctx_extent, kv_head_extent, pe_dim),
            T.float16,
        ),
    ) -> NonPagedTiledMLAPartial:
        Q_shared = T.alloc_shared((block_h, dim), "float16")
        Q_pe_shared = T.alloc_shared((block_h, pe_dim), "float16")
        KV_shared = T.alloc_shared((block_n, dim), "float16")
        K_pe_shared = T.alloc_shared((block_n, pe_dim), "float16")
        KV_tail_shared = T.alloc_shared((block_n, dim), "float16")
        K_pe_tail_shared = T.alloc_shared((block_n, pe_dim), "float16")
        scores = T.alloc_fragment((block_h, block_n), "float32")
        running_max = T.alloc_fragment((block_h,), "float32")
        previous_max = T.alloc_fragment((block_h,), "float32")
        history_scale = T.alloc_fragment((block_h,), "float32")
        tile_sum = T.alloc_fragment((block_h,), "float32")
        running_sum = T.alloc_fragment((block_h,), "float32")
        staged_history_scale = T.alloc_shared((block_h,), "float32")
        S_shared = T.alloc_shared((block_h, block_n), "float16")
        acc_o = T.alloc_fragment((block_h, dim), "float32")
        acc_o_left = T.alloc_shared(
            (block_h, pv_half_dim),
            "float32",
        )
        acc_o_right = T.alloc_shared(
            (block_h, pv_half_dim),
            "float32",
        )
        M_shared = T.alloc_shared((block_h,), "float32")
        L_shared = T.alloc_shared((block_h,), "float32")

        head_start = head_block * block_h
        T.copy(
            Q[batch, head_start : head_start + block_h, :],
            Q_shared,
        )
        T.copy(
            QPe[batch, head_start : head_start + block_h, :],
            Q_pe_shared,
        )

        range_begin = T.cast(T.dataflow_range_begin(), "int32")
        range_end = T.cast(T.dataflow_range_end(), "int32")
        T.fill(acc_o, 0)
        T.online_softmax_initialize(scores, running_max, running_sum, block_h)

        range_len = range_end - range_begin
        full_tile_count = T.floordiv(range_len, block_n)
        for tile in T.Pipelined(
            full_tile_count,
            num_stages=requested_pipeline_stages,
        ):
            kv_start = range_begin + tile * block_n
            T.copy(
                KV[batch, kv_start : kv_start + block_n, 0, :],
                KV_shared,
                eviction_policy="evict_first",
            )
            T.copy(
                KPe[batch, kv_start : kv_start + block_n, 0, :],
                K_pe_shared,
                eviction_policy="evict_first",
            )
            prepare_scores(
                Q_shared,
                Q_pe_shared,
                KV_shared,
                K_pe_shared,
                scores,
                kv_start,
                range_end,
                False,
            )
            T.online_softmax_update(
                scores,
                KV_shared,
                S_shared,
                acc_o,
                running_max,
                previous_max,
                history_scale,
                tile_sum,
                running_sum,
                staged_history_scale,
                block_h,
                block_h,
                block_n,
                dim,
                True,
                T.GemmWarpPolicy.FullCol,
            )

        tail_start = range_begin + full_tile_count * block_n
        if tail_start < range_end:
            T.copy(
                KV[batch, tail_start : tail_start + block_n, 0, :],
                KV_tail_shared,
                valid_region=KV[batch, range_begin:range_end, 0, :],
                oob_fill=0,
                eviction_policy="evict_first",
            )
            T.copy(
                KPe[batch, tail_start : tail_start + block_n, 0, :],
                K_pe_tail_shared,
                valid_region=KPe[batch, range_begin:range_end, 0, :],
                oob_fill=0,
                eviction_policy="evict_first",
            )
            prepare_scores(
                Q_shared,
                Q_pe_shared,
                KV_tail_shared,
                K_pe_tail_shared,
                scores,
                tail_start,
                range_end,
                True,
            )
            T.online_softmax_update(
                scores,
                KV_tail_shared,
                S_shared,
                acc_o,
                running_max,
                previous_max,
                history_scale,
                tile_sum,
                running_sum,
                staged_history_scale,
                block_h,
                block_h,
                block_n,
                dim,
                True,
                T.GemmWarpPolicy.FullCol,
            )

        T.copy(running_max, M_shared)
        T.copy(running_sum, L_shared)
        T.sync_threads()
        for row, column in T.Parallel(block_h, dim):
            if column < pv_half_dim:
                acc_o_left[row, column] = acc_o[row, column]
            else:
                acc_o_right[row, column - pv_half_dim] = acc_o[row, column]
        return NonPagedTiledMLAPartial(
            m=M_shared,
            l=L_shared,
            o_left=acc_o_left,
            o_right=acc_o_right,
        )

    @T.dataflow.reduce(
        associative=True,
        threads=iter_threads,
        specialization_constants=constants,
        physical_contract=df.DataflowOperatorPhysicalContract(
            output_alias_input_indices=(0, 1),
        ),
        accumulator_contracts={
            "reduce_o_acc_dtype": df.DataflowAccumulatorContract(
                contract_id="online_softmax_output_merge",
                allowed_dtypes=("float16", "float32"),
                strict_dtype="float32",
                minimum_dtype="float16",
                reduction_order="scheduler_plan_associative_binary",
            ),
        },
    )
    def nonpaged_tiled_mla_reduce(
        left: NonPagedTiledMLAPartial,
        right: NonPagedTiledMLAPartial,
    ) -> NonPagedTiledMLAPartial:
        M = T.alloc_shared((block_h,), "float32")
        L = T.alloc_shared((block_h,), "float32")
        O_left = T.alloc_shared(
            (block_h, pv_half_dim),
            reduce_o_acc_dtype,
        )
        O_right = T.alloc_shared(
            (block_h, pv_half_dim),
            reduce_o_acc_dtype,
        )
        new_m = T.alloc_shared((block_h,), "float32")
        old_scale = T.alloc_shared((block_h,), "float32")
        item_scale = T.alloc_shared((block_h,), "float32")

        for head in T.Parallel(block_h):
            new_m[head] = T.max(left.m[head], right.m[head])
            old_scale[head] = T.exp2(left.m[head] - new_m[head])
            item_scale[head] = T.exp2(right.m[head] - new_m[head])
        T.sync_threads()
        for head, column in T.Parallel(block_h, pv_half_dim):
            O_left[head, column] = T.cast(left.o_left[head, column], reduce_o_acc_dtype) * T.cast(
                old_scale[head], reduce_o_acc_dtype
            ) + T.cast(right.o_left[head, column], reduce_o_acc_dtype) * T.cast(item_scale[head], reduce_o_acc_dtype)
            O_right[head, column] = T.cast(left.o_right[head, column], reduce_o_acc_dtype) * T.cast(
                old_scale[head], reduce_o_acc_dtype
            ) + T.cast(right.o_right[head, column], reduce_o_acc_dtype) * T.cast(item_scale[head], reduce_o_acc_dtype)
        for head in T.Parallel(block_h):
            L[head] = left.l[head] * old_scale[head] + right.l[head] * item_scale[head]
            M[head] = new_m[head]
        T.sync_threads()
        return NonPagedTiledMLAPartial(
            m=M,
            l=L,
            o_left=O_left,
            o_right=O_right,
        )

    # Fused finalizers are separate static-arity ABI handlers. The scheduler
    # selects both the two- and three-input forms on production traces.
    @T.dataflow.finalize(
        threads=iter_threads,
        specialization_constants=constants,
    )
    def nonpaged_tiled_mla_reduce_finalize(
        left: NonPagedTiledMLAPartial,
        right: NonPagedTiledMLAPartial,
        batch: T.int32,
        head_block: T.int32,
        Output: T.Tensor(
            (batch_extent, head_extent, dim),
            output_dtype,
        ),
    ) -> None:
        left_scale = T.alloc_shared((block_h,), "float32")
        right_scale = T.alloc_shared((block_h,), "float32")
        for row in T.Parallel(block_h):
            new_m = T.max(left.m[row], right.m[row])
            left_weight = T.exp2(left.m[row] - new_m)
            right_weight = T.exp2(right.m[row] - new_m)
            inv_l = T.fast_fdiv(
                T.float32(1),
                left.l[row] * left_weight + right.l[row] * right_weight,
            )
            left_scale[row] = left_weight * inv_l
            right_scale[row] = right_weight * inv_l
        T.sync_threads()
        for row, column in T.Parallel(block_h, pv_half_dim):
            Output[
                batch,
                head_block * block_h + row,
                column,
            ] = T.cast(
                T.cast(left.o_left[row, column], "float32") * left_scale[row]
                + T.cast(right.o_left[row, column], "float32") * right_scale[row],
                output_dtype,
            )
            Output[
                batch,
                head_block * block_h + row,
                column + pv_half_dim,
            ] = T.cast(
                T.cast(left.o_right[row, column], "float32") * left_scale[row]
                + T.cast(right.o_right[row, column], "float32") * right_scale[row],
                output_dtype,
            )

    @T.dataflow.finalize(
        threads=iter_threads,
        specialization_constants=constants,
    )
    def nonpaged_tiled_mla_reduce_finalize_three(
        first: NonPagedTiledMLAPartial,
        second: NonPagedTiledMLAPartial,
        third: NonPagedTiledMLAPartial,
        batch: T.int32,
        head_block: T.int32,
        Output: T.Tensor(
            (batch_extent, head_extent, dim),
            output_dtype,
        ),
    ) -> None:
        first_scale = T.alloc_shared((block_h,), "float32")
        second_scale = T.alloc_shared((block_h,), "float32")
        third_scale = T.alloc_shared((block_h,), "float32")
        for row in T.Parallel(block_h):
            new_m = T.max(T.max(first.m[row], second.m[row]), third.m[row])
            first_weight = T.exp2(first.m[row] - new_m)
            second_weight = T.exp2(second.m[row] - new_m)
            third_weight = T.exp2(third.m[row] - new_m)
            inv_l = T.fast_fdiv(
                T.float32(1),
                first.l[row] * first_weight + second.l[row] * second_weight + third.l[row] * third_weight,
            )
            first_scale[row] = first_weight * inv_l
            second_scale[row] = second_weight * inv_l
            third_scale[row] = third_weight * inv_l
        T.sync_threads()
        for row, column in T.Parallel(block_h, pv_half_dim):
            Output[
                batch,
                head_block * block_h + row,
                column,
            ] = T.cast(
                T.cast(first.o_left[row, column], "float32") * first_scale[row]
                + T.cast(second.o_left[row, column], "float32") * second_scale[row]
                + T.cast(third.o_left[row, column], "float32") * third_scale[row],
                output_dtype,
            )
            Output[
                batch,
                head_block * block_h + row,
                column + pv_half_dim,
            ] = T.cast(
                T.cast(first.o_right[row, column], "float32") * first_scale[row]
                + T.cast(second.o_right[row, column], "float32") * second_scale[row]
                + T.cast(third.o_right[row, column], "float32") * third_scale[row],
                output_dtype,
            )

    @T.dataflow.finalize(
        threads=iter_threads,
        specialization_constants=constants,
    )
    def nonpaged_tiled_mla_finalize(
        inter: NonPagedTiledMLAPartial,
        batch: T.int32,
        head_block: T.int32,
        Output: T.Tensor(
            (batch_extent, head_extent, dim),
            output_dtype,
        ),
    ) -> None:
        inv_l = T.alloc_shared((block_h,), "float32")
        for row in T.Parallel(block_h):
            inv_l[row] = T.fast_fdiv(
                T.float32(1),
                inter.l[row],
            )
        T.sync_threads()
        if finalize_staged_transfer:
            output_shared = T.alloc_shared(
                (block_h, dim),
                output_dtype,
            )
            for row, column in T.Parallel(block_h, pv_half_dim):
                output_shared[row, column] = T.cast(
                    T.cast(inter.o_left[row, column], "float32") * inv_l[row],
                    output_dtype,
                )
                output_shared[row, column + pv_half_dim] = T.cast(
                    T.cast(inter.o_right[row, column], "float32") * inv_l[row],
                    output_dtype,
                )
            T.sync_threads()
            T.copy(
                output_shared,
                Output[
                    batch,
                    head_block * block_h : (head_block + 1) * block_h,
                    :,
                ],
            )
        else:
            for row, column in T.Parallel(block_h, pv_half_dim):
                Output[
                    batch,
                    head_block * block_h + row,
                    column,
                ] = T.cast(
                    T.cast(inter.o_left[row, column], "float32") * inv_l[row],
                    output_dtype,
                )
                Output[
                    batch,
                    head_block * block_h + row,
                    column + pv_half_dim,
                ] = T.cast(
                    T.cast(inter.o_right[row, column], "float32") * inv_l[row],
                    output_dtype,
                )

    return NonPagedTiledMLAOperators(
        partial_type=NonPagedTiledMLAPartial,
        split_tile=nonpaged_tiled_mla_split_tile,
        reduce=nonpaged_tiled_mla_reduce,
        reduce_finalizers=(
            nonpaged_tiled_mla_reduce_finalize,
            nonpaged_tiled_mla_reduce_finalize_three,
        ),
        finalize=nonpaged_tiled_mla_finalize,
    )


def make_nonpaged_tiled_split_mla_program(config: NonPagedTiledMLAConfig | None = None):
    config = NonPagedTiledMLAConfig() if config is None else config
    require_supported_tiled_nonpaged_config(config)
    operators = make_nonpaged_tiled_mla_operators(config)
    return (
        T.dataflow_program(task_domain=("batch", "head_block"), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            operators.split_tile(Q="Q", QPe="QPe", KV="KV", KPe="KPe"),
            task_args=("batch", "head_block"),
            range_axis="kv",
        )
        .reduce(operators.reduce())
        .finalize(
            operators.finalize(Output="Output"),
            fused_reduce=tuple(finalizer(Output="Output") for finalizer in operators.reduce_finalizers),
        )
    )


def normalize_tiled_nonpaged_range_offsets(
    config: NonPagedTiledMLAConfig,
    range_offsets: Sequence[Any] | None,
    seq_lens: Sequence[int],
) -> tuple[int, ...] | None:
    if range_offsets is None:
        return None
    head_blocks = config.head_blocks

    def to_int(value: Any) -> int:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    if len(range_offsets) == config.batch and range_offsets:
        first = range_offsets[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            flattened = tuple(to_int(item) for row in range_offsets for item in row)
        else:
            flattened = tuple(to_int(item) for item in range_offsets for _ in range(head_blocks))
    else:
        flattened = tuple(to_int(item) for item in range_offsets)
    if len(flattened) != config.task_count:
        raise ValueError(
            f"range_offsets must provide {config.task_count} entries for batch={config.batch}, heads={config.heads}; got {len(flattened)}"
        )
    for offset, seq_len in zip(flattened, seq_lens):
        if offset < 0:
            raise ValueError(f"range_offsets entries must be non-negative, got {offset}")
        if offset + seq_len > config.kv_ctx:
            raise ValueError(
                f"range_offsets entries must keep each KV range inside kv_ctx: offset={offset}, length={seq_len}, kv_ctx={config.kv_ctx}"
            )
    return flattened


@df.jit
def nonpaged_tiled_split_mla(
    config: NonPagedTiledMLAConfig,
    *,
    seq_lens: Sequence[Any],
    range_offsets: Sequence[Any] | None = None,
    compile_options: Mapping[str, Any] | None = None,
) -> df.DataflowKernelSpec:
    require_supported_tiled_nonpaged_config(config)
    normalized_seq_lens = normalize_tiled_nonpaged_seq_lens(config, seq_lens)
    normalized_range_offsets = normalize_tiled_nonpaged_range_offsets(
        config,
        range_offsets,
        normalized_seq_lens,
    )
    if compile_options is not None and not isinstance(compile_options, Mapping):
        raise TypeError(f"compile_options must be a mapping or None, got {type(compile_options)!r}")
    compile_options = dict(compile_options or {})
    operator_owned_options = {
        "program",
        "topology",
        "range_lengths",
        "range_offsets",
        "block_size",
        "task_extents",
        "include_exit",
        "block_dim",
        "mode",
        "options",
    }
    conflicts = operator_owned_options & compile_options.keys()
    if conflicts:
        raise ValueError(f"nonpaged_tiled_split_mla owns these compile options: {sorted(conflicts)!r}")
    options = {
        "wrapper_name": "dataflow_nonpaged_tiled_split_mla_decode",
        "scheduler_policy": df.DATAFLOW_SCHEDULER_AUTO,
        "reduce_strategy": df.DATAFLOW_SCHEDULER_AUTO,
        "force_hbm_comms": False,
        "partial_only": False,
        "iter_range_bucket_size": config.block_n,
        "compile_flags": ("--maxrregcount=168",),
        "pic": False,
        "pic_dir": "schedule_res",
        **compile_options,
    }
    if options["iter_range_bucket_size"] is None:
        options["iter_range_bucket_size"] = config.block_n
    compile_flags = options.get("compile_flags")
    if compile_flags is None:
        options.pop("compile_flags", None)
    else:
        options["compile_flags"] = tuple(compile_flags)

    return df.make_kernel_spec(
        make_nonpaged_tiled_split_mla_program(config),
        topology=(config.sm_count, config.cluster_size),
        range_lengths={"kv": list(normalized_seq_lens)},
        range_offsets=(None if normalized_range_offsets is None else {"kv": list(normalized_range_offsets)}),
        task_extents=(config.batch, config.head_blocks),
        block_size=config.range_block_n,
        block_dim=config.threads,
        mode="executable",
        **options,
    )
