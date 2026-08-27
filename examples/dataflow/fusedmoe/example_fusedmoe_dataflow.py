import math
import tilelang.language as T
import tilelang.dataflow as df

GATE_UP_INTERLEAVE_GRANULARITY = 8


def gate_up_weight_layout(n_experts, d_expert, d_hidden, *, granularity=GATE_UP_INTERLEAVE_GRANULARITY):
    return df.DataflowTensorArgumentLayout(
        parameter_name="routed_expert_gate_up",
        logical_shape=(n_experts, 2, d_expert, d_hidden),
        physical_shape=(n_experts, 2 * d_expert, d_hidden),
        physical_axes=((0,), (2, 1), (3,)),
        selector_axis=1,
        interleave_axis=2,
        interleave=granularity,
    )


def pack_gate_up_weights(gate, up, *, granularity=GATE_UP_INTERLEAVE_GRANULARITY):
    n_experts, d_expert, d_hidden = gate.shape
    packed = gate.new_empty((n_experts, 2 * d_expert, d_hidden))
    view = packed.view(n_experts, d_expert // granularity, 2, granularity, d_hidden)
    logical = (n_experts, d_expert // granularity, granularity, d_hidden)
    view[:, :, 0].copy_(gate.reshape(logical))
    view[:, :, 1].copy_(up.reshape(logical))
    return df.mark_tensor_layout(packed, gate_up_weight_layout(n_experts, d_expert, d_hidden, granularity=granularity))


def pipeline_contract(transfers, gemms, execution, *, outstanding=None, shared_budget=None):
    def consumers(buffer_index):
        return tuple(index for index, gemm in enumerate(gemms) if buffer_index in gemm.input_buffer_indices)

    lifetimes = tuple(
        df.DataflowPipelineBufferLifetime(
            buffer_index=index,
            producer_transfer_index=index,
            consumer_gemm_indices=consumers(index),
            release_after_gemm_index=consumers(index)[-1],
        )
        for index in range(len(transfers))
    )
    return df.DataflowPipelineRequest(
        transfers=transfers,
        gemms=gemms,
        buffer_lifetimes=lifetimes,
        stage_budget=execution.pipeline_stages,
        max_outstanding=execution.max_outstanding if outstanding is None else outstanding,
        max_shared_memory_bytes=shared_budget,
        producer_threads=128,
        consumer_threads=execution.consumer_threads,
        synchronization_owner=df.DATAFLOW_PIPELINE_SYNC_PIPELINE,
        completion_semantics=df.DATAFLOW_PIPELINE_OWNED_COMPLETION,
        release_semantics=df.DATAFLOW_PIPELINE_OWNED_RELEASE,
    )


@T.macro
def store_gate_up(output, logits, output_tile, output_begin, token_extent, expert_extent, active_rows, output_dtype):
    for common_token, common_expert in T.Parallel(token_extent, expert_extent):
        common_group = common_expert // GATE_UP_INTERLEAVE_GRANULARITY
        common_lane = common_expert % GATE_UP_INTERLEAVE_GRANULARITY
        common_gate_index = common_group * (2 * GATE_UP_INTERLEAVE_GRANULARITY) + common_lane
        common_gate_logit = logits[common_gate_index, common_token]
        common_up_logit = logits[common_gate_index + GATE_UP_INTERLEAVE_GRANULARITY, common_token]
        common_value = T.if_then_else(
            common_token < active_rows,
            T.cast(silu(common_gate_logit) * common_up_logit, output_dtype),
            T.cast(0, output_dtype),
        )
        output[output_tile, common_token, output_begin + common_expert] = common_value


@T.macro
def silu(gate_logit):
    return gate_logit * (1.0 / (1.0 + T.exp2(-gate_logit * 1.44269504)))


@T.macro
def store_weighted_output(output, output_value, routed_expert_weights, grouped_row, hidden_col, output_dtype):
    value = T.cast(output_value, "float32") * T.cast(routed_expert_weights[grouped_row], "float32")
    output[grouped_row, hidden_col] = T.cast(value, output_dtype)


@df.jit(cache=False)
def compile_dataflow_routed_moe(
    d_hidden,
    d_expert,
    n_routed_experts,
    group_sum,
    dtype=T.float16,
    input_dtype=None,
    weight_dtype=None,
    intermediate_dtype=None,
    semantic_config=None,
    execution_override=None,
    mode="executable",
    inspection_stage="ir",
    active_group_blocks=None,
    task_group_extents=None,
    stage_graph_task_weights=None,
    stage_graph_cluster_assignment=None,
    **compile_options,
):
    weight_dtype = dtype if weight_dtype is None else weight_dtype
    input_dtype = weight_dtype if input_dtype is None else input_dtype
    intermediate_dtype = dtype if intermediate_dtype is None else intermediate_dtype
    semantic = df.resolve_semantic_config(semantic_config)
    task_group_extents = () if task_group_extents is None else tuple(task_group_extents)
    if sum(task_group_extents) > group_sum:
        raise ValueError("scheduled task group extents exceed the input row count")
    execution_plan = df.plan_execution(
        df.DataflowExecutionRequest(
            task_extent=sum(task_group_extents) if task_group_extents else group_sum,
            task_group_extents=task_group_extents,
            stages=(
                df.DataflowExecutionStageRequest(
                    output_extent=d_expert,
                    reduction_extent=d_hidden,
                    input_dtype=input_dtype,
                    weight_dtype=weight_dtype,
                    output_dtype=intermediate_dtype,
                    projection_count=2,
                ),
                df.DataflowExecutionStageRequest(
                    output_extent=d_hidden,
                    reduction_extent=d_expert,
                    input_dtype=intermediate_dtype,
                    weight_dtype=weight_dtype,
                    output_dtype=dtype,
                    input_from_previous_stage=True,
                ),
            ),
            linked_stage_pairs=((0, 1),),
            uniform_stage_implementation=True,
            minimum_tile_extent=1 if mode == df.DATAFLOW_COMPILE_MODE_INSPECT and inspection_stage == "ir" else 16,
        ),
        override=execution_override,
    )
    execution = execution_plan.selected_candidate
    map1_execution, map2_execution = execution.stages
    cluster_size, block_token = execution.topology.cluster_size, execution.task_tile_extent
    map_block_dhidden, map_block_dexpert, map2_block_dhidden = (map1_execution.tile_k, map1_execution.tile_n, map2_execution.tile_n)
    expert_shard = d_expert // cluster_size
    map1_tiles = math.ceil(map1_execution.handler_extent / map_block_dexpert)

    group_blocks = math.ceil(group_sum / block_token) + n_routed_experts
    scheduled_blocks = tuple(range(group_blocks)) if active_group_blocks is None else tuple(int(block) for block in active_group_blocks)
    force_hbm_comms = execution.transport_family == df.DATAFLOW_TRANSPORT_HBM
    streamed_transport = execution.transport_family == df.DATAFLOW_TRANSPORT_STREAMED
    reshared_access_order = df.DATAFLOW_TRANSPORT_INDEPENDENT_ORDER if streamed_transport else df.DATAFLOW_TRANSPORT_LOGICAL_ORDER
    if force_hbm_comms:
        compile_options["memory_policy"] = df.DataflowMemoryPolicy(mode=df.DATAFLOW_MEMORY_HBM_DIRECT_GLOBAL)

    input_bytes = df.require_dataflow_dtype(input_dtype).element_bytes
    weight_bytes = df.require_dataflow_dtype(weight_dtype).element_bytes
    intermediate_bytes = df.require_dataflow_dtype(intermediate_dtype).element_bytes
    map1_async = map1_execution.common_pipeline and map1_execution.transfer_family == df.DATAFLOW_EXECUTION_TRANSFER_TMA
    map2_async = map2_execution.common_pipeline and map2_execution.transfer_family == df.DATAFLOW_EXECUTION_TRANSFER_TMA
    map1_fused = map1_execution.fused_transfers
    map1_weight_rows = 2 * map_block_dexpert if map1_fused else map_block_dexpert
    map1_weight_count = 1 if map1_fused else 2
    map1_partitions = map1_execution.split_producers and map1_async
    map1_transfers = (
        df.DataflowPipelineTransfer(
            destination_buffer_index=0,
            logical_extent=(block_token, map_block_dhidden),
            bytes_per_stage=block_token * map_block_dhidden * input_bytes,
            producer_partition=0 if map1_partitions else None,
            async_permitted=map1_async,
            multicast_permitted=cluster_size > 1,
            eviction_hint=df.DATAFLOW_PIPELINE_EVICT_LAST,
        ),
        *tuple(
            df.DataflowPipelineTransfer(
                destination_buffer_index=index + 1,
                logical_extent=(map1_weight_rows, map_block_dhidden),
                bytes_per_stage=map1_weight_rows * map_block_dhidden * weight_bytes,
                producer_partition=index + 1 if map1_partitions else None,
                async_permitted=map1_async,
                eviction_hint=map1_execution.eviction_policy,
            )
            for index in range(map1_weight_count)
        ),
    )
    map1_gemms = tuple(
        df.DataflowPipelineGemm(input_buffer_indices=(index + 1, 0), accumulator_index=index) for index in range(map1_weight_count)
    )
    map1_pipeline = pipeline_contract(
        map1_transfers,
        map1_gemms,
        map1_execution,
        outstanding=map1_execution.max_outstanding,
        shared_budget=execution_plan.resources.pipeline_shared_memory_budgets[0],
    )

    map2_resident = execution.transport_family == df.DATAFLOW_TRANSPORT_ALL_GATHER
    map2_transfers = (
        df.DataflowPipelineTransfer(
            destination_buffer_index=0,
            logical_extent=(1, 1, block_token, map_block_dexpert) if map2_resident else (block_token, map_block_dexpert),
            bytes_per_stage=block_token * map_block_dexpert * intermediate_bytes,
            materialization=df.DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT if map2_resident else df.DATAFLOW_PIPELINE_MATERIALIZE_COPY,
            async_permitted=map2_async and not map2_resident,
            eviction_hint=df.DATAFLOW_PIPELINE_EVICT_LAST,
        ),
        df.DataflowPipelineTransfer(
            destination_buffer_index=1,
            logical_extent=(map2_block_dhidden, map_block_dexpert),
            bytes_per_stage=map2_block_dhidden * map_block_dexpert * weight_bytes,
            async_permitted=map2_async,
            eviction_hint=map2_execution.eviction_policy,
        ),
    )
    map2_pipeline = pipeline_contract(
        map2_transfers,
        (df.DataflowPipelineGemm(input_buffer_indices=(1, 0), accumulator_index=0),),
        map2_execution,
    )
    map1_sync = df.DATAFLOW_PIPELINE_SYNC_PIPELINE if map1_execution.common_pipeline else df.DATAFLOW_PIPELINE_SYNC_TRANSFER
    map2_sync = df.DATAFLOW_PIPELINE_SYNC_PIPELINE if map2_execution.common_pipeline else df.DATAFLOW_PIPELINE_SYNC_TRANSFER
    map1_stages = df.plan_pipeline_dataflow(map1_pipeline).selected_stages if map1_execution.common_pipeline else map1_execution.loop_stages
    map2_stages = df.plan_pipeline_dataflow(map2_pipeline).selected_stages if map2_execution.common_pipeline else map2_execution.loop_stages
    map1_input_annotations = (
        {"cluster_mask": (1 << cluster_size) - 1}
        if map1_execution.common_pipeline and map1_execution.input_distribution == df.DATAFLOW_EXECUTION_INPUT_MULTICAST
        else None
    )
    map1_range = df.DataflowRangeCoarseningRequest(
        logical_tile_extent=map_block_dexpert,
        logical_range_extent=expert_shard,
        handler_range_extent=map1_execution.handler_extent,
        output_tile_arity=map1_tiles,
    )
    map2_range = df.DataflowRangeCoarseningRequest(
        logical_tile_extent=map2_block_dhidden,
        logical_range_extent=d_hidden // cluster_size,
        handler_range_extent=map2_execution.handler_extent,
        output_tile_arity=math.ceil(map2_execution.handler_extent / map2_block_dhidden),
    )
    map2_input_slots = (
        df.DATAFLOW_INPUT_SLOTS_INDEXED if execution.transport_family == df.DATAFLOW_TRANSPORT_HBM else df.DATAFLOW_INPUT_SLOTS_CONTIGUOUS
    )

    @T.dataflow_intermediate(
        layout_contracts=df.DataflowTensorLayoutRequest(
            field_index=0,
            logical_rank=3,
            layout_family=df.DATAFLOW_LAYOUT_MATRIX_SWIZZLE,
            major_axis=2,
        )
    )
    class DataflowRoutedUpShard:
        value: T.Tensor((map1_tiles, block_token, map_block_dexpert), intermediate_dtype)

    @T.dataflow.map(
        range=("expert_begin", "expert_end"),
        threads=map1_execution.consumer_threads if map1_execution.common_pipeline else map1_execution.compute_threads,
        physical_contract=df.DataflowOperatorPhysicalContract(output_slot=df.DATAFLOW_OUTPUT_SLOT_DIRECT),
        tensor_argument_layouts=gate_up_weight_layout(n_routed_experts, d_expert, d_hidden),
    )
    def dataflow_routed_map1(
        group_block: T.int32,
        input: T.Tensor((group_sum, d_hidden), input_dtype),
        routed_expert_gate_up: T.Tensor((n_routed_experts, 2 * d_expert, d_hidden), weight_dtype),
        group_sizes: T.Tensor((n_routed_experts,), T.int32),
        group_offsets: T.Tensor((n_routed_experts,), T.int32),
        group_padded_offsets: T.Tensor((n_routed_experts,), T.int32),
        group_idx_for_bx: T.Tensor((group_blocks,), T.int32),
    ) -> DataflowRoutedUpShard:
        result = T.alloc_shared((T.dataflow_range_tiles_per_handler(), block_token, map_block_dexpert), intermediate_dtype)
        input_shared = T.alloc_shared((block_token, map_block_dhidden), input_dtype)
        if map1_fused:
            weights = T.alloc_shared((2 * map_block_dexpert, map_block_dhidden), weight_dtype)
            logits = T.alloc_fragment((2 * map_block_dexpert, block_token), "float32")
        else:
            first_weights = T.alloc_shared((map_block_dexpert, map_block_dhidden), weight_dtype)
            second_weights = T.alloc_shared((map_block_dexpert, map_block_dhidden), weight_dtype)
            first_logits = T.alloc_fragment((map_block_dexpert, block_token), "float32")
            second_logits = T.alloc_fragment((map_block_dexpert, block_token), "float32")
        common_m_start_padded = group_block * block_token
        common_group_idx = group_idx_for_bx[group_block]
        common_group_size = group_sizes[common_group_idx]
        common_group_padded_start = group_padded_offsets[common_group_idx]
        common_m_start = common_m_start_padded - common_group_padded_start + group_offsets[common_group_idx]
        # TMA valid_region zero-fills the tail; shifting this coordinate changes row identity.
        common_safe_m_start = T.max(0, common_m_start)
        common_actual_rows = T.max(0, T.min(block_token, common_group_size - (common_m_start_padded - common_group_padded_start)))
        common_expert_begin = T.cast(T.dataflow_range_begin(), "int32")

        for output_tile in T.serial(T.dataflow_range_tiles_per_handler()):
            if map1_fused:
                T.clear(logits)
            else:
                T.clear(first_logits)
                T.clear(second_logits)
            expert_begin = common_expert_begin + output_tile * map_block_dexpert
            packed_begin = 2 * expert_begin
            if common_actual_rows > 0:
                for k in T.Pipelined(d_hidden // map_block_dhidden, num_stages=map1_stages):
                    hidden_begin = k * map_block_dhidden
                    T.copy(
                        input[common_safe_m_start : common_safe_m_start + block_token, hidden_begin : hidden_begin + map_block_dhidden],
                        input_shared,
                        valid_region=input[0:group_sum, 0:d_hidden],
                        allow_async=map1_async,
                        synchronization_owner=map1_sync,
                        eviction_policy=df.DATAFLOW_PIPELINE_EVICT_LAST,
                        annotations=map1_input_annotations,
                    )
                    if map1_fused:
                        T.copy(
                            routed_expert_gate_up[
                                common_group_idx,
                                packed_begin : packed_begin + 2 * map_block_dexpert,
                                hidden_begin : hidden_begin + map_block_dhidden,
                            ],
                            weights,
                            valid_region=routed_expert_gate_up[common_group_idx, 0 : 2 * d_expert, 0:d_hidden],
                            allow_async=map1_async,
                            synchronization_owner=map1_sync,
                            eviction_policy=map1_execution.eviction_policy,
                        )
                        T.gemm(
                            weights,
                            input_shared,
                            logits,
                            transpose_B=True,
                            logical_shape=(2 * map_block_dexpert, block_token, map_block_dhidden),
                        )
                    else:
                        T.copy(
                            routed_expert_gate_up[
                                common_group_idx,
                                packed_begin : packed_begin + map_block_dexpert,
                                hidden_begin : hidden_begin + map_block_dhidden,
                            ],
                            first_weights,
                            valid_region=routed_expert_gate_up[common_group_idx, 0 : 2 * d_expert, 0:d_hidden],
                            allow_async=map1_async,
                            synchronization_owner=map1_sync,
                            eviction_policy=map1_execution.eviction_policy,
                        )
                        T.copy(
                            routed_expert_gate_up[
                                common_group_idx,
                                packed_begin + map_block_dexpert : packed_begin + 2 * map_block_dexpert,
                                hidden_begin : hidden_begin + map_block_dhidden,
                            ],
                            second_weights,
                            valid_region=routed_expert_gate_up[common_group_idx, 0 : 2 * d_expert, 0:d_hidden],
                            allow_async=map1_async,
                            synchronization_owner=map1_sync,
                            eviction_policy=map1_execution.eviction_policy,
                        )
                        T.gemm(
                            first_weights,
                            input_shared,
                            first_logits,
                            transpose_B=True,
                            logical_shape=(map_block_dexpert, block_token, map_block_dhidden),
                        )
                        T.gemm(
                            second_weights,
                            input_shared,
                            second_logits,
                            transpose_B=True,
                            logical_shape=(map_block_dexpert, block_token, map_block_dhidden),
                        )
            if map1_fused:
                store_gate_up(result, logits, output_tile, 0, block_token, map_block_dexpert, common_actual_rows, intermediate_dtype)
            else:
                store_gate_up(
                    result, first_logits, output_tile, 0, block_token, map_block_dexpert // 2, common_actual_rows, intermediate_dtype
                )
                store_gate_up(
                    result,
                    second_logits,
                    output_tile,
                    map_block_dexpert // 2,
                    block_token,
                    map_block_dexpert // 2,
                    common_actual_rows,
                    intermediate_dtype,
                )
        return DataflowRoutedUpShard(value=result)

    @T.dataflow.map(
        range=("hidden_begin", "hidden_end"),
        threads=map2_execution.compute_threads,
        physical_contract=df.DataflowOperatorPhysicalContract(input_slots=map2_input_slots, output_slot=df.DATAFLOW_OUTPUT_SLOT_NONE),
    )
    def dataflow_routed_map2(
        parts: list[DataflowRoutedUpShard],
        group_block: T.int32,
        input: T.Tensor((group_sum, d_hidden), input_dtype),
        routed_expert_down: T.Tensor((n_routed_experts, d_hidden, d_expert), weight_dtype),
        routed_expert_weights: T.Tensor((group_sum,), dtype),
        group_sizes: T.Tensor((n_routed_experts,), T.int32),
        group_offsets: T.Tensor((n_routed_experts,), T.int32),
        group_padded_offsets: T.Tensor((n_routed_experts,), T.int32),
        group_idx_for_bx: T.Tensor((group_blocks,), T.int32),
        output: T.Tensor((group_sum, d_hidden), dtype),
    ) -> None:
        if not map2_resident:
            up_shared = T.alloc_shared((block_token, map_block_dexpert), intermediate_dtype)
        down_shared = T.alloc_shared((map2_block_dhidden, map_block_dexpert), weight_dtype)
        accumulator = T.alloc_fragment((map2_block_dhidden, block_token), "float32")
        common_m_start_padded = group_block * block_token
        common_group_idx = group_idx_for_bx[group_block]
        common_group_size = group_sizes[common_group_idx]
        common_group_padded_start = group_padded_offsets[common_group_idx]
        common_m_start = common_m_start_padded - common_group_padded_start + group_offsets[common_group_idx]
        common_actual_rows = T.max(0, T.min(block_token, common_group_size - (common_m_start_padded - common_group_padded_start)))
        range_begin = T.cast(T.dataflow_range_begin(), "int32")
        range_end = T.cast(T.dataflow_range_end(), "int32")

        for output_tile in T.serial(T.dataflow_range_tiles_per_handler()):
            hidden_begin = range_begin + output_tile * map2_block_dhidden
            if hidden_begin < range_end and common_actual_rows > 0:
                T.clear(accumulator)
                for linear_expert in T.Pipelined(cluster_size * (expert_shard // map_block_dexpert), num_stages=map2_stages):
                    source_part = linear_expert // map1_tiles
                    source_tile = linear_expert % map1_tiles
                    if map2_resident:
                        T.copy(
                            parts[source_part].value[source_tile, 0:block_token, 0:map_block_dexpert],
                            parts[source_part].value[source_tile, 0:block_token, 0:map_block_dexpert],
                            valid_region=parts[source_part].value[source_tile, 0:block_token, 0:map_block_dexpert],
                            synchronization_owner=map2_sync,
                            eviction_policy=df.DATAFLOW_PIPELINE_EVICT_LAST,
                        )
                    else:
                        T.copy(
                            parts[source_part].value[source_tile, 0:block_token, 0:map_block_dexpert],
                            up_shared,
                            valid_region=parts[source_part].value[source_tile, 0:block_token, 0:map_block_dexpert],
                            allow_async=map2_async,
                            synchronization_owner=map2_sync,
                            eviction_policy=df.DATAFLOW_PIPELINE_EVICT_LAST,
                        )
                    T.copy(
                        routed_expert_down[
                            common_group_idx,
                            hidden_begin : hidden_begin + map2_block_dhidden,
                            linear_expert * map_block_dexpert : (linear_expert + 1) * map_block_dexpert,
                        ],
                        down_shared,
                        valid_region=routed_expert_down[common_group_idx, 0:d_hidden, 0:d_expert],
                        allow_async=map2_async,
                        synchronization_owner=map2_sync,
                        eviction_policy=map2_execution.eviction_policy,
                    )
                    if map2_resident:
                        T.gemm(
                            down_shared,
                            parts[source_part].value[source_tile, 0:block_token, 0:map_block_dexpert],
                            accumulator,
                            transpose_B=True,
                            logical_shape=(map2_block_dhidden, block_token, map_block_dexpert),
                        )
                    else:
                        T.gemm(
                            down_shared,
                            up_shared,
                            accumulator,
                            transpose_B=True,
                            logical_shape=(map2_block_dhidden, block_token, map_block_dexpert),
                        )
                if map2_execution.common_pipeline:
                    for token, hidden in T.Parallel(block_token, map2_block_dhidden):
                        hidden_col = hidden_begin + hidden
                        if token < common_actual_rows and hidden_col < range_end and hidden_col < d_hidden:
                            store_weighted_output(
                                output, accumulator[hidden, token], routed_expert_weights, common_m_start + token, hidden_col, dtype
                            )
                else:
                    for hidden, token in T.Parallel(map2_block_dhidden, block_token):
                        hidden_col = hidden_begin + hidden
                        if token < common_actual_rows and hidden_col < range_end and hidden_col < d_hidden:
                            store_weighted_output(
                                output, accumulator[hidden, token], routed_expert_weights, common_m_start + token, hidden_col, dtype
                            )
        return None

    program = (
        T.dataflow_program(
            task_domain=("group_block",),
            dynamic_ranges={"expert_tile": "expert_tiles", "hidden_tile": "hidden_tiles"},
        )
        .map(
            dataflow_routed_map1(),
            task_args=("group_block",),
            range_axis="expert_tile",
            range_contract=map1_range,
            pipeline_contract=map1_pipeline if map1_execution.common_pipeline else None,
        )
        .reshared(
            input="routed_map1",
            name="routed_reshared",
            transport_contract=df.DataflowResharedTransportRequest(
                family=execution.transport_family,
                logical_output_arity=d_expert // map_block_dexpert,
                physical_output_arity=d_expert // map1_execution.handler_extent,
                consumer_access_order=reshared_access_order,
                field_mappings=(
                    df.DataflowResharedFieldMapping(
                        field_index=0,
                        physical_value_axis=2,
                        physical_tile_axis=0,
                    ),
                ),
            ),
        )
        .map(
            dataflow_routed_map2(),
            input="routed_reshared",
            task_args=("group_block",),
            range_axis="hidden_tile",
            range_contract=map2_range,
            pipeline_contract=map2_pipeline if map2_execution.common_pipeline else None,
            handoff_contract=(
                df.DataflowCrossHandlerHandoffRequest(
                    consumer_stage_id=0,
                    buffer_stages=execution.handoff_stages,
                    value_bindings=(0,),
                )
                if execution.handoff_stages
                else None
            ),
        )
    )
    return df.make_kernel_spec(
        program=program,
        topology=execution.topology,
        range_lengths={"expert_tile": [d_expert] * len(scheduled_blocks), "hidden_tile": [d_hidden] * len(scheduled_blocks)},
        block_size=block_token,
        task_extents=(group_blocks,) if active_group_blocks is None else None,
        task_coord_overrides=None if active_group_blocks is None else tuple((block,) for block in scheduled_blocks),
        stage_graph_task_weights=stage_graph_task_weights,
        stage_graph_cluster_assignment=stage_graph_cluster_assignment,
        include_exit=False,
        mode=mode,
        inspection_stage=inspection_stage,
        semantic_config=semantic,
        force_hbm_comms=force_hbm_comms,
        wrapper_name="dataflow_routed_moe",
        **compile_options,
    )
