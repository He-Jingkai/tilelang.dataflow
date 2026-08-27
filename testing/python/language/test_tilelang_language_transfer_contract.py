"""IR contract tests for target-independent bounded transfers."""

from __future__ import annotations

import pytest

import tilelang as tl
import tilelang.testing
import tilelang.language as T
from tilelang import _ffi_api, tvm
from tvm import ir, tir


def region(buffer, *bounds):
    return tir.BufferRegion(
        buffer,
        [ir.Range.from_min_extent(begin, extent) for begin, extent in bounds],
    )


def parse_copy(call):
    parsed = _ffi_api.ParseOperator(call)
    assert isinstance(parsed, tl.ir.Copy)
    return parsed


def raw_transfer_contract(valid_region, oob_fill=0, allow_async=True, owner=0):
    return tir.call_intrin(
        "handle",
        ir.Op.get("tl.transfer_contract"),
        valid_region,
        oob_fill,
        int(allow_async),
        owner,
    )


def resolve_transfer(call, arch):
    plan = _ffi_api.ResolveTransferLowering(
        call,
        tvm.target.Target(f"cuda -arch={arch}"),
    )
    assert isinstance(plan, tl.ir.TransferLoweringPlan)
    assert plan.schema_version == 1
    return plan


def test_transfer_contract_1d_aligned_is_first_class_and_serializable():
    source = tir.decl_buffer((128,), "float16", name="Source")
    destination = tir.decl_buffer((64,), "float16", name="Destination")
    call = T.copy(
        region(source, (32, 64)),
        destination,
        valid_region=region(source, (0, 128)),
        oob_fill=0,
        allow_async=True,
        synchronization_owner="transfer",
    )

    assert len(call.args) == 3
    assert isinstance(call.args[2], tir.Call)
    assert call.args[2].op.same_as(ir.Op.get("tl.transfer_contract"))

    parsed = parse_copy(call)
    contract = parsed.transfer_contract
    assert isinstance(contract, tl.ir.TransferContract)
    assert contract.schema_version == 1
    assert contract.allow_async is True
    assert contract.sync_owner == int(T.TransferSynchronizationOwner.TRANSFER)
    ir.assert_structural_equal(
        contract.src_valid_region,
        region(source, (0, 128)),
    )
    assert isinstance(contract.oob_fill, tir.IntImm)
    assert contract.oob_fill.value == 0

    restored = tvm.ir.load_json(tvm.ir.save_json(contract))
    ir.assert_structural_equal(contract, restored)
    assert tvm.ir.structural_hash(contract) == tvm.ir.structural_hash(restored)
    assert "T.transfer_contract" in str(call)
    assert "metadata" not in str(call)


def test_transfer_contract_2d_tail_preserves_dtype_scope_and_owner():
    valid_rows = tir.Var("valid_rows", "int32")
    valid_cols = tir.Var("valid_cols", "int32")
    source = tir.decl_buffer((32, 64), "float16", name="Source")
    destination = tir.decl_buffer(
        (16, 32),
        "float32",
        name="Destination",
        scope="shared",
    )
    call = T.copy(
        region(source, (8, 16), (16, 32)),
        destination,
        valid_region=region(source, (4, valid_rows), (8, valid_cols)),
        oob_fill=-3.25,
        synchronization_owner=T.TransferSynchronizationOwner.PIPELINE,
    )

    parsed = parse_copy(call)
    contract = parsed.transfer_contract
    assert str(parsed.src.dtype) == "float16"
    assert parsed.src.scope() == "global"
    assert str(parsed.dst.dtype) == "float32"
    assert parsed.dst.scope() == "shared"
    assert len(parsed.src_range) == 2
    assert len(contract.src_valid_region.region) == 2
    assert contract.allow_async is True
    assert contract.sync_owner == int(T.TransferSynchronizationOwner.PIPELINE)
    assert isinstance(contract.oob_fill, tir.FloatImm)
    assert contract.oob_fill.value == pytest.approx(-3.25)

    seen_vars = []
    tir.stmt_functor.post_order_visit(
        call.args[2],
        lambda node: seen_vars.append(node) if isinstance(node, tir.Var) else None,
    )
    assert any(var.same_as(valid_rows) for var in seen_vars)
    assert any(var.same_as(valid_cols) for var in seen_vars)


def test_transfer_contract_dynamic_bound_participates_in_substitution():
    source = tir.decl_buffer((64,), "float16", name="Source")
    destination = tir.decl_buffer((32,), "float16", name="Destination")
    old_bound = tir.Var("old_bound", "int32")
    new_bound = tir.Var("new_bound", "int32")
    call = T.copy(
        region(source, (16, 32)),
        destination,
        valid_region=region(source, (0, old_bound)),
    )

    rewritten = tir.stmt_functor.substitute(call, {old_bound: new_bound})
    parsed = parse_copy(rewritten)
    extent = parsed.transfer_contract.src_valid_region.region[0].extent
    assert isinstance(extent, tir.Var)
    assert extent.same_as(new_bound)


def test_transfer_contract_common_fallback_uses_valid_region_and_fill():
    source = tir.decl_buffer((8,), "float16", name="Source")
    destination = tir.decl_buffer((8,), "float32", name="Destination")
    valid_end = tir.Var("valid_end", "int32")
    call = T.copy(
        source,
        destination,
        valid_region=region(source, (2, valid_end - 2)),
        oob_fill=7.0,
        allow_async=False,
    )
    func = tir.PrimFunc(
        [source.data, destination.data, valid_end],
        tir.Evaluate(call),
        buffer_map={source.data: source, destination.data: destination},
    ).with_attr("target", tvm.target.Target("llvm"))
    mod = tvm.IRModule.from_expr(func)

    with tvm.target.Target("llvm"):
        lowered = tl.transform.LowerTileOp()(mod)["main"]

    conditional_values = []

    def visit(node):
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tir.if_then_else":
            conditional_values.append(node)

    tir.stmt_functor.post_order_visit(lowered.body, visit)
    assert len(conditional_values) == 1
    conditional = conditional_values[0]
    assert isinstance(conditional.args[2], tir.FloatImm)
    assert conditional.args[2].value == pytest.approx(7.0)
    predicate_vars = []
    tir.stmt_functor.post_order_visit(
        conditional.args[0],
        lambda node: predicate_vars.append(node) if isinstance(node, tir.Var) else None,
    )
    assert any(var.same_as(valid_end) for var in predicate_vars)


def test_transfer_contract_common_fallback_guards_unit_extent_axes():
    source = tir.decl_buffer((1, 8), "float16", name="Source")
    destination = tir.decl_buffer((8,), "float16", name="Destination")
    call = T.copy(
        region(source, (2, 1), (0, 8)),
        destination,
        valid_region=region(source, (0, 1), (0, 8)),
        oob_fill=5,
        allow_async=False,
    )
    func = tir.PrimFunc(
        [source.data, destination.data],
        tir.Evaluate(call),
        buffer_map={source.data: source, destination.data: destination},
    ).with_attr("target", tvm.target.Target("llvm"))

    with tvm.target.Target("llvm"):
        lowered = tl.transform.LowerTileOp()(tvm.IRModule.from_expr(func))["main"]

    loads = []
    fill_values = []

    def visit(node):
        if isinstance(node, tir.BufferLoad):
            loads.append(node)
        if isinstance(node, (tir.IntImm, tir.FloatImm)):
            fill_values.append(float(node.value))

    tir.stmt_functor.post_order_visit(lowered.body, visit)
    assert not loads
    assert 5.0 in fill_values


def test_legacy_copy_keeps_two_argument_contract_and_annotations():
    source = tir.decl_buffer((32,), "float16", name="Source")
    destination = tir.decl_buffer((32,), "float16", name="Destination")
    call = T.copy(source, destination, coalesced_width=4)

    assert len(call.args) == 2
    assert "disable_tma" not in call.annotations
    parsed = parse_copy(call)
    assert parsed.transfer_contract is None
    assert "disable_tma" not in parsed.annotations
    assert parsed.annotations["coalesced_width"].value == 4

    ordinary_call = tir.call_extern("int32", "not_a_tile_operator")
    assert _ffi_api.ParseOperator(ordinary_call) is None


def test_transfer_contract_rejects_ambiguous_or_invalid_ownership():
    source = tir.decl_buffer((32,), "float16", name="Source")
    other = tir.decl_buffer((32,), "float16", name="Other")
    destination = tir.decl_buffer((32,), "float16", name="Destination")

    with pytest.raises(ValueError, match="must own its synchronization"):
        T.transfer_contract(
            region(source, (0, 16)),
            allow_async=False,
            synchronization_owner="pipeline",
        )

    wrong_buffer = T.transfer_contract(region(other, (0, 16)))
    with pytest.raises(tvm.error.InternalError, match="copy source buffer"):
        parse_copy(T.copy(source, destination, wrong_buffer))

    contract = T.transfer_contract(region(source, (0, 16)))
    base_call = T.copy(source, destination, contract)
    conflicting = tir.Call(
        base_call.dtype,
        base_call.op,
        base_call.args,
        annotations={"src_upper_bound_0": tir.IntImm("int32", 16)},
    )
    with pytest.raises(tvm.error.InternalError, match="cannot be combined"):
        parse_copy(conflicting)

    with pytest.raises(ValueError, match="cannot read a buffer"):
        T.transfer_contract(
            region(source, (0, 16)),
            oob_fill=other[0],
        )

    encoded_read_region = T.transfer_contract(region(source, (0, 16))).args[0]
    unsafe_fill = raw_transfer_contract(encoded_read_region, other[0])
    with pytest.raises(tvm.error.InternalError, match="cannot read a buffer"):
        parse_copy(T.copy(source, destination, unsafe_fill))

    opaque_fill = T.transfer_contract(
        region(source, (0, 16)),
        oob_fill=tir.call_extern("float16", "opaque_fill"),
    )
    with pytest.raises(tvm.error.InternalError, match="pure scalar expression"):
        parse_copy(T.copy(source, destination, opaque_fill))

    write_region = tir.call_intrin(
        "handle",
        ir.Op.get("tl.tileop.region"),
        source[0],
        2,
        16,
    )
    with pytest.raises(ValueError, match="must use read access"):
        T.transfer_contract(write_region)

    write_contract = raw_transfer_contract(write_region)
    with pytest.raises(tvm.error.InternalError, match="must use read access"):
        parse_copy(T.copy(source, destination, write_contract))


@pytest.mark.parametrize("owner", ["pipeline", "caller"])
def test_unconsumed_external_sync_ownership_fails_closed(owner):
    source = tir.decl_buffer((16,), "float16", name="Source")
    destination = tir.decl_buffer((16,), "float16", name="Destination")
    call = T.copy(
        source,
        destination,
        valid_region=region(source, (0, 16)),
        synchronization_owner=owner,
    )
    func = tir.PrimFunc(
        [source.data, destination.data],
        tir.Evaluate(call),
        buffer_map={source.data: source, destination.data: destination},
    ).with_attr("target", tvm.target.Target("llvm"))

    with tvm.target.Target("llvm"), pytest.raises(tvm.error.InternalError, match="not consumed"):
        tl.transform.LowerTileOp()(tvm.IRModule.from_expr(func))


def pipeline_decision_records(func):
    records = func.attrs["tl.pipeline_lowering_decisions"]
    return [{str(key): getattr(value, "value", value) for key, value in record.items()} for record in records]


def logical_weight_first_pipeline(logical_m, num_stages):
    tokens = 64
    reduction = 32
    reduction_tiles = 2

    @T.prim_func
    def before(
        weights: T.Tensor(
            (logical_m, reduction * reduction_tiles),
            T.float16,
        ),
        activations: T.Tensor(
            (tokens, reduction * reduction_tiles),
            T.float16,
        ),
        output: T.Tensor((logical_m, tokens), T.float32),
    ):
        with T.Kernel(1, threads=128):
            weight_shared = T.alloc_shared(
                (logical_m, reduction),
                T.float16,
            )
            activation_shared = T.alloc_shared(
                (tokens, reduction),
                T.float16,
            )
            accumulator = T.alloc_fragment(
                (logical_m, tokens),
                T.float32,
            )
            T.clear(accumulator)
            for tile in T.Pipelined(
                reduction_tiles,
                num_stages=num_stages,
            ):
                T.copy(
                    weights[
                        :,
                        tile * reduction : tile * reduction + reduction,
                    ],
                    weight_shared,
                    valid_region=weights,
                    synchronization_owner="pipeline",
                )
                T.copy(
                    activations[
                        :,
                        tile * reduction : tile * reduction + reduction,
                    ],
                    activation_shared,
                    valid_region=activations,
                    synchronization_owner="pipeline",
                )
                T.gemm(
                    weight_shared,
                    activation_shared,
                    accumulator,
                    transpose_B=True,
                    logical_shape=(logical_m, tokens, reduction),
                )
            T.copy(accumulator, output)

    return before


@pytest.mark.parametrize("logical_m", [16, 32, 48, 64])
@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_sm90_pipeline_closes_logical_m_padding_and_stage_matrix(
    logical_m,
    num_stages,
):
    target = tvm.target.Target("cuda -arch=sm_90a")
    before = logical_weight_first_pipeline(logical_m, num_stages)
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)

    with target:
        mod = tl.transform.MaterializeLogicalGemm()(mod)
        plans = list(mod["main"].attrs["tl.gemm_lowering_plans"])
        assert len(plans) == 1
        assert plans[0].implementation_id == ("cuda.wgmma.async" if logical_m == 64 else "cuda.wgmma.rs.shared_a")
        assert [int(value) for value in plans[0].logical_shape] == [
            logical_m,
            64,
            32,
        ]
        assert [int(value) for value in plans[0].physical_shape] == [
            64,
            64,
            32,
        ]
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
        decisions = pipeline_decision_records(mod["main"])
        assert len(decisions) == 1
        assert decisions[0]["selected_implementation"] == "warp_specialized"
        assert int(decisions[0]["requested_stages"]) == num_stages
        assert int(decisions[0]["fallback"]) == 0
        barrier_arrive_counts = []
        elected_release_extents = []

        def collect_ws_contract(node):
            if isinstance(node, tir.Block) and "barrier_init" in node.annotations:
                for counts in node.annotations["barrier_init"].values():
                    barrier_arrive_counts.append([int(value) for value in counts])
            if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tl_shuffle_elect":
                elected_release_extents.append(int(node.args[0]))

        tir.stmt_functor.post_order_visit(
            mod["main"].body,
            collect_ws_contract,
        )
        assert barrier_arrive_counts == [
            [1] * (2 * num_stages),
        ]
        assert elected_release_extents == [128]
        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    thread_extents = []

    def collect(node):
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op):
            call_names.append(node.op.name)
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            thread_extents.append(int(node.value))

    tir.stmt_functor.post_order_visit(mod["main"].body, collect)
    assert "tl.tma_load" in call_names
    assert "tl.mbarrier_wait_parity" in call_names
    assert max(thread_extents) == 256


def test_pipeline_owner_is_consumed_by_sm80_software_pipeline():
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def before(
        source: T.Tensor((64,), T.float32),
        destination: T.Tensor((64,), T.float32),
    ):
        with T.Kernel(1, threads=16):
            shared = T.alloc_shared((16,), T.float32)
            for tile in T.Pipelined(4, num_stages=2):
                T.copy(
                    source[tile * 16 : tile * 16 + 16],
                    shared,
                    valid_region=source[0:64],
                    oob_fill=0,
                    synchronization_owner="pipeline",
                )
                for i in T.Parallel(16):
                    destination[tile * 16 + i] = shared[i]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.PipelinePlanning()(mod)
        decisions = pipeline_decision_records(mod["main"])
        assert len(decisions) == 1
        assert str(decisions[0]["selected_implementation"]) == "software_pipeline"
        assert int(decisions[0]["requested_stages"]) == 2
        assert int(decisions[0]["fallback"]) == 0
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
    )
    assert "tl.ptx_cp_async" in call_names


def test_pipeline_planning_records_structured_synchronous_fallback():
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def before(
        source: T.Tensor((4,), T.float32),
        destination: T.Tensor((4,), T.float32),
        choose_source: T.int32,
    ):
        with T.Kernel(1, threads=32):
            for i in T.Pipelined(4, num_stages=2):
                if choose_source != 0:
                    destination[i] = source[i]
                else:
                    destination[i] = T.float32(0)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.PipelinePlanning()(mod)

    decisions = pipeline_decision_records(mod["main"])
    assert len(decisions) == 1
    assert str(decisions[0]["selected_implementation"]) == "synchronous"
    assert int(decisions[0]["fallback"]) == 1
    assert "if/else" in str(decisions[0]["selection_reason"])

    pipeline_loops = []

    def visit(node):
        if isinstance(node, tir.For) and "num_stages" in node.annotations:
            pipeline_loops.append(node)

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert not pipeline_loops


def test_pipeline_owned_transfer_fallback_forces_synchronous_lowering():
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def before(
        source: T.Tensor((64,), T.float32),
        alternate: T.Tensor((64,), T.float32),
        destination: T.Tensor((16,), T.float32),
        choose_source: T.int32,
    ):
        with T.Kernel(1, threads=16):
            shared = T.alloc_shared((16,), T.float32)
            for tile in T.Pipelined(4, num_stages=2):
                if choose_source != 0:
                    T.copy(
                        source[tile * 16 : tile * 16 + 16],
                        shared,
                        valid_region=source,
                        synchronization_owner="pipeline",
                    )
                else:
                    T.copy(
                        alternate[tile * 16 : tile * 16 + 16],
                        shared,
                        valid_region=alternate,
                        synchronization_owner="pipeline",
                    )
            T.copy(shared, destination)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.PipelinePlanning()(mod)
        decisions = pipeline_decision_records(mod["main"])
        assert decisions[0]["selected_implementation"] == "synchronous"
        assert int(decisions[0]["fallback"]) == 1
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
    )
    assert "tl.ptx_cp_async" not in call_names
    assert "tl.tma_load" not in call_names
    assert call_names.count("tir.tvm_storage_sync") == 2


def test_rocm_pipeline_owner_records_synchronous_fallback():
    target = tvm.target.Target("rocm -mcpu=gfx942")

    @T.prim_func
    def before(
        source: T.Tensor((64,), T.float32),
        destination: T.Tensor((64,), T.float32),
    ):
        with T.Kernel(1, threads=16):
            shared = T.alloc_shared((16,), T.float32)
            for tile in T.Pipelined(4, num_stages=2):
                T.copy(
                    source[tile * 16 : tile * 16 + 16],
                    shared,
                    valid_region=source,
                    synchronization_owner="pipeline",
                )
                for i in T.Parallel(16):
                    destination[tile * 16 + i] = shared[i]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.PipelinePlanning()(mod)
        decisions = pipeline_decision_records(mod["main"])
        assert decisions[0]["selected_implementation"] == "synchronous"
        assert int(decisions[0]["requested_stages"]) == 2
        assert int(decisions[0]["fallback"]) == 1
        assert "validated software-pipeline path" in str(decisions[0]["selection_reason"])

        pipeline_loops = []
        consumed_modes = []

        def collect(node):
            if isinstance(node, tir.For) and "num_stages" in node.annotations:
                pipeline_loops.append(node)
            if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tileop.copy":
                mode = node.annotations.get("tl.transfer_pipeline_sync_consumed")
                if mode is not None:
                    consumed_modes.append(int(mode))

        tir.stmt_functor.post_order_visit(mod["main"].body, collect)
        assert not pipeline_loops
        assert consumed_modes == [2]


def test_pipeline_owner_is_consumed_by_sm90_warp_specialization():
    target = tvm.target.Target("cuda -arch=sm_90a")
    block = 64
    reduction = 32

    @T.prim_func
    def before(
        left: T.Tensor((block, reduction * 4), T.float16),
        right: T.Tensor((reduction * 4, block), T.float16),
        output: T.Tensor((block, block), T.float32),
    ):
        with T.Kernel(1, threads=128):
            left_shared = T.alloc_shared((block, reduction), T.float16)
            right_shared = T.alloc_shared((reduction, block), T.float16)
            accumulator = T.alloc_fragment((block, block), T.float32)
            T.clear(accumulator)
            for tile in T.Pipelined(4, num_stages=2):
                T.copy(
                    left[:, tile * reduction : tile * reduction + reduction],
                    left_shared,
                    valid_region=left,
                    synchronization_owner="pipeline",
                )
                T.copy(
                    right[tile * reduction : tile * reduction + reduction, :],
                    right_shared,
                    valid_region=right,
                    synchronization_owner="pipeline",
                )
                T.gemm(left_shared, right_shared, accumulator)
            T.copy(accumulator, output)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
        decisions = pipeline_decision_records(mod["main"])
        assert len(decisions) == 1
        assert decisions[0]["selected_implementation"] == "warp_specialized"
        assert int(decisions[0]["fallback"]) == 0

        gemm_wait_modes = []
        explicit_wgmma_drains = []

        def collect_static_wgmma_protocol(node):
            if isinstance(node, tir.Call) and isinstance(node.op, ir.Op):
                if node.op.name == "tl.tileop.gemm":
                    gemm_wait_modes.append(int(node.args[15]))
                if node.op.name == "tl.wait_wgmma":
                    explicit_wgmma_drains.append(int(node.args[0]))

        tir.stmt_functor.post_order_visit(
            mod["main"].body,
            collect_static_wgmma_protocol,
        )
        assert gemm_wait_modes == [-1]
        assert explicit_wgmma_drains == [0]

        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
    )
    assert "tl.tma_load" in call_names
    assert "tl.mbarrier_wait_parity" in call_names


def test_transfer_resolver_honors_compiler_pipeline_ownership_annotations():
    source = tir.decl_buffer((64, 64), "float16", name="Source")
    destination = tir.decl_buffer(
        (64, 64),
        "float16",
        name="Destination",
        scope="shared",
    )
    call = T.copy(
        source,
        destination,
        valid_region=source,
        synchronization_owner="pipeline",
    )

    def annotated(mode):
        annotations = dict(call.annotations)
        annotations["tl.transfer_pipeline_sync_consumed"] = tir.IntImm("int32", mode)
        return tir.Call(
            call.dtype,
            call.op,
            list(call.args),
            annotations=annotations,
            span=call.span,
        )

    managed = _ffi_api.ResolveTransferLowering(
        annotated(1),
        tvm.target.Target("cuda -arch=sm_90a"),
    )
    fallback = _ffi_api.ResolveTransferLowering(
        annotated(2),
        tvm.target.Target("cuda -arch=sm_90a"),
    )

    assert managed.supported is True
    assert managed.implementation_id == "cuda.tma.load.full"
    assert managed.asynchronous is True
    assert fallback.supported is True
    assert fallback.implementation_id == "common.simt"
    assert fallback.asynchronous is False
    assert "synchronous transfer fallback" in fallback.selection_reason


def test_sm90_warp_specialization_supports_sync_and_tma_transfers_together():
    target = tvm.target.Target("cuda -arch=sm_90a")
    block = 64
    reduction = 32

    @T.prim_func
    def before(
        weights: T.Tensor((reduction * 4, block), T.float16),
        output: T.Tensor((block, block), T.float32),
    ):
        with T.Kernel(1, threads=128):
            resident_shared = T.alloc_shared((block, reduction), T.float16)
            staged_shared = T.alloc_shared((block, reduction), T.float16)
            weight_shared = T.alloc_shared((reduction, block), T.float16)
            accumulator = T.alloc_fragment((block, block), T.float32)
            T.clear(resident_shared)
            T.clear(accumulator)
            for tile in T.Pipelined(4, num_stages=2):
                T.copy(
                    resident_shared,
                    staged_shared,
                    valid_region=resident_shared,
                    synchronization_owner="pipeline",
                )
                T.copy(
                    weights[
                        tile * reduction : tile * reduction + reduction,
                        :,
                    ],
                    weight_shared,
                    valid_region=weights,
                    synchronization_owner="pipeline",
                )
                T.gemm(staged_shared, weight_shared, accumulator)
            T.copy(accumulator, output)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
        decisions = pipeline_decision_records(mod["main"])
        assert len(decisions) == 1
        assert decisions[0]["selected_implementation"] == "warp_specialized"
        assert int(decisions[0]["fallback"]) == 0

        consumed_modes = []

        def collect_copy(node):
            if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tileop.copy":
                mode = node.annotations.get("tl.transfer_pipeline_sync_consumed")
                if mode is not None:
                    consumed_modes.append(int(mode))

        tir.stmt_functor.post_order_visit(mod["main"].body, collect_copy)
        assert 2 in consumed_modes

        ws_call_names = []
        tir.stmt_functor.post_order_visit(
            mod["main"].body,
            lambda node: ws_call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
        )
        assert ws_call_names.count("tl.mbarrier_wait_parity") == 4

        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
    )
    assert "tl.tma_load" in call_names
    assert "tl.mbarrier_wait_parity" in call_names


def nested_guarded_dual_copy_pipeline(batch_count=2, outer_sync=False):
    block = 64
    reduction = 32

    @T.prim_func
    def before(
        left: T.Tensor((batch_count, block, reduction * 4), T.float16),
        right: T.Tensor((batch_count, reduction * 4, block), T.float16),
        output: T.Tensor((batch_count, block, block), T.float32),
        active_batches: T.int32,
    ):
        with T.Kernel(1, threads=128):
            left_shared = T.alloc_shared((block, reduction), T.float16)
            right_shared = T.alloc_shared((reduction, block), T.float16)
            accumulator = T.alloc_fragment((block, block), T.float32)
            for batch in T.serial(batch_count):
                T.clear(accumulator)
                if outer_sync:
                    T.sync_threads()
                if batch < active_batches:
                    for tile in T.Pipelined(4, num_stages=2):
                        T.copy(
                            left[
                                batch,
                                :,
                                tile * reduction : tile * reduction + reduction,
                            ],
                            left_shared,
                            valid_region=left[batch, :, :],
                            synchronization_owner="pipeline",
                        )
                        T.copy(
                            right[
                                batch,
                                tile * reduction : tile * reduction + reduction,
                                :,
                            ],
                            right_shared,
                            valid_region=right[batch, :, :],
                            synchronization_owner="pipeline",
                        )
                        T.gemm(left_shared, right_shared, accumulator)
                    T.copy(accumulator, output[batch, :, :])

    return before


def test_sm90_warp_specialization_finds_pipeline_inside_loop_and_guard():
    target = tvm.target.Target("cuda -arch=sm_90a")
    before = nested_guarded_dual_copy_pipeline()

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
        decisions = pipeline_decision_records(mod["main"])
        assert len(decisions) == 1
        assert decisions[0]["selected_implementation"] == "warp_specialized"
        assert int(decisions[0]["fallback"]) == 0

        phase_counter_allocations = []
        wait_parities = []
        pipeline_loop_kinds = []
        gemm_wait_modes = []
        gemm_fence_after_modes = []
        explicit_wgmma_drains = []

        def collect_ws(node):
            if isinstance(node, tir.Allocate) and "phase_cnt" in node.buffer_var.name:
                phase_counter_allocations.append(node.buffer_var.name)
            if isinstance(node, tir.For) and node.loop_var.name == "tile":
                pipeline_loop_kinds.append(node.kind)
            if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.mbarrier_wait_parity":
                wait_parities.append(node.args[-1])
            if isinstance(node, tir.Call) and isinstance(node.op, ir.Op):
                if node.op.name == "tl.tileop.gemm":
                    gemm_wait_modes.append(int(node.args[15]))
                    fence_after = node.annotations.get("wgmma_emit_fence_after")
                    if fence_after is not None:
                        gemm_fence_after_modes.append(int(fence_after))
                if node.op.name == "tl.wait_wgmma":
                    explicit_wgmma_drains.append(int(node.args[0]))

        tir.stmt_functor.post_order_visit(mod["main"].body, collect_ws)
        assert phase_counter_allocations == []
        assert wait_parities
        assert all(not isinstance(parity, tir.IntImm) for parity in wait_parities)
        parity_vars = set()
        for parity in wait_parities:
            tir.stmt_functor.post_order_visit(
                parity,
                lambda node: parity_vars.add(node.name) if isinstance(node, tir.Var) else None,
            )
        assert {"batch", "tile"}.issubset(parity_vars)
        assert pipeline_loop_kinds == [
            tir.ForKind.UNROLLED,
            tir.ForKind.UNROLLED,
        ]
        assert gemm_wait_modes == [-1]
        assert gemm_fence_after_modes == [0]
        assert explicit_wgmma_drains == [0]

        consumer_batch_loops = []

        def collect_consumer_batch_loop(node):
            if not isinstance(node, tir.For) or node.loop_var.name != "batch":
                return
            call_names = []
            tir.stmt_functor.post_order_visit(
                node.body,
                lambda child: call_names.append(child.op.name) if isinstance(child, tir.Call) and isinstance(child.op, ir.Op) else None,
            )
            if "tl.tileop.gemm" in call_names:
                consumer_batch_loops.append(node)

        tir.stmt_functor.post_order_visit(
            mod["main"].body,
            collect_consumer_batch_loop,
        )
        assert len(consumer_batch_loops) == 1
        consumer_body = consumer_batch_loops[0].body
        assert isinstance(consumer_body, tir.SeqStmt)
        prefix_call_names = []
        tir.stmt_functor.post_order_visit(
            consumer_body.seq[0],
            lambda node: prefix_call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
        )
        assert "tl.tileop.fill" in prefix_call_names
        assert isinstance(consumer_body.seq[1], tir.IfThenElse)
        guarded_call_names = []
        tir.stmt_functor.post_order_visit(
            consumer_body.seq[1].then_case,
            lambda node: guarded_call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
        )
        assert "tl.tileop.fill" not in guarded_call_names

        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    thread_extents = []

    def collect(node):
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op):
            call_names.append(node.op.name)
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            thread_extents.append(int(node.value))

    tir.stmt_functor.post_order_visit(mod["main"].body, collect)
    assert "tl.tma_load" in call_names
    assert "tl.mbarrier_wait_parity" in call_names
    assert max(thread_extents) == 256


def test_sm90_single_iteration_pipeline_lifts_complete_role_scope():
    target = tvm.target.Target("cuda -arch=sm_90a")
    before = nested_guarded_dual_copy_pipeline(batch_count=1)
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)

    with target:
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)

    ws_scopes = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: ws_scopes.append(node) if isinstance(node, tir.AttrStmt) and node.attr_key == "kWarpSpecializationScope" else None,
    )
    assert len(ws_scopes) == 1
    branch = ws_scopes[0].body
    assert isinstance(branch, tir.IfThenElse)
    assert branch.else_case is not None

    def call_names(stmt):
        names = []
        tir.stmt_functor.post_order_visit(
            stmt,
            lambda node: names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
        )
        return names

    producer_calls = call_names(branch.then_case)
    consumer_calls = call_names(branch.else_case)
    assert "tl.tileop.tma_copy" in producer_calls
    assert "tl.tileop.fill" not in producer_calls
    assert "tl.tileop.fill" in consumer_calls
    assert "tl.tileop.gemm" in consumer_calls
    assert "tl.tileop.copy" in consumer_calls

    producer_loops = []
    consumer_loops = []
    tir.stmt_functor.post_order_visit(
        branch.then_case,
        lambda node: producer_loops.append(node.loop_var.name) if isinstance(node, tir.For) else None,
    )
    tir.stmt_functor.post_order_visit(
        branch.else_case,
        lambda node: consumer_loops.append(node.loop_var.name) if isinstance(node, tir.For) else None,
    )
    assert "batch" in producer_loops
    assert "batch" in consumer_loops

    with target:
        mod = tl.transform.AnnotateWarpGroupRegAlloc()(mod)
    script = mod.script()
    reg_dealloc = script.index("T.set_max_nreg(40, 0)")
    reg_alloc = script.index("T.set_max_nreg(232, 1)")
    fragment_fill = script.index("T.fill")
    assert reg_dealloc < reg_alloc < fragment_fill


def test_sm90_single_iteration_outer_collective_prevents_role_scope_lift():
    target = tvm.target.Target("cuda -arch=sm_90a")
    before = nested_guarded_dual_copy_pipeline(
        batch_count=1,
        outer_sync=True,
    )
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)

    with target:
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)

    batch_loops = []
    ws_scopes = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: (
            batch_loops.append(node)
            if isinstance(node, tir.For) and node.loop_var.name == "batch"
            else ws_scopes.append(node)
            if isinstance(node, tir.AttrStmt) and node.attr_key == "kWarpSpecializationScope"
            else None
        ),
    )
    assert len(batch_loops) == 1
    assert len(ws_scopes) == 1

    batch_calls = []
    batch_ws_scopes = []
    tir.stmt_functor.post_order_visit(
        batch_loops[0].body,
        lambda node: (
            batch_calls.append(node.op.name)
            if isinstance(node, tir.Call) and isinstance(node.op, ir.Op)
            else batch_ws_scopes.append(node)
            if isinstance(node, tir.AttrStmt) and node.attr_key == "kWarpSpecializationScope"
            else None
        ),
    )
    assert "tir.tvm_storage_sync" in batch_calls
    assert len(batch_ws_scopes) == 1


def nested_sparse_dual_copy_pipeline():
    block = 64
    reduction = 32

    @T.prim_func
    def before(
        left: T.Tensor((3, block, reduction * 4), T.float16),
        right: T.Tensor((3, reduction * 4, block), T.float16),
        output: T.Tensor((3, block, block), T.float32),
    ):
        with T.Kernel(1, threads=128):
            left_shared = T.alloc_shared((block, reduction), T.float16)
            right_shared = T.alloc_shared((reduction, block), T.float16)
            accumulator = T.alloc_fragment((block, block), T.float32)
            for batch in T.serial(3):
                T.clear(accumulator)
                if batch % 2 == 0:
                    for tile in T.Pipelined(4, num_stages=2):
                        T.copy(
                            left[
                                batch,
                                :,
                                tile * reduction : tile * reduction + reduction,
                            ],
                            left_shared,
                            valid_region=left[batch, :, :],
                            synchronization_owner="pipeline",
                        )
                        T.copy(
                            right[
                                batch,
                                tile * reduction : tile * reduction + reduction,
                                :,
                            ],
                            right_shared,
                            valid_region=right[batch, :, :],
                            synchronization_owner="pipeline",
                        )
                        T.gemm(left_shared, right_shared, accumulator)
                    T.copy(accumulator, output[batch, :, :])

    return before


def test_sm90_sparse_nested_pipeline_retains_persistent_phase_counters():
    target = tvm.target.Target("cuda -arch=sm_90a")
    before = nested_sparse_dual_copy_pipeline()
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)

    with target:
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)

    phase_counter_allocations = []
    pipeline_loop_kinds = []
    gemm_wait_modes = []
    explicit_wgmma_drains = []

    def collect_sparse_ws(node):
        if isinstance(node, tir.Allocate) and "phase_cnt" in node.buffer_var.name:
            phase_counter_allocations.append(node.buffer_var.name)
        if isinstance(node, tir.For) and node.loop_var.name == "tile":
            pipeline_loop_kinds.append(node.kind)
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op):
            if node.op.name == "tl.tileop.gemm":
                gemm_wait_modes.append(int(node.args[15]))
            if node.op.name == "tl.wait_wgmma":
                explicit_wgmma_drains.append(int(node.args[0]))

    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        collect_sparse_ws,
    )
    assert sorted(phase_counter_allocations) == [
        "consumer_phase_cnt",
        "producer_phase_cnt",
    ]
    assert pipeline_loop_kinds == [
        tir.ForKind.SERIAL,
        tir.ForKind.SERIAL,
    ]
    assert gemm_wait_modes == [0]
    assert explicit_wgmma_drains == []


@tl.testing.requires_cuda
@tl.testing.requires_cuda_compute_version_ge(9, 0)
def test_sm90_nested_pipeline_phase_persists_across_outer_loop_runtime():
    import torch

    kernel = tl.compile(
        nested_guarded_dual_copy_pipeline(),
        out_idx=[2],
    )
    left = torch.randn((2, 64, 128), device="cuda", dtype=torch.float16)
    right = torch.randn((2, 128, 64), device="cuda", dtype=torch.float16)
    actual = kernel(left, right, 2)
    expected = torch.stack([left[index].float() @ right[index].float() for index in range(2)])
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def static_multicast_dual_copy_pipeline(
    *,
    cluster_mask=3,
    cluster_dims=(2, 1, 1),
):
    block = 64
    reduction = 32
    reduction_tiles = 4

    @T.prim_func
    def before(
        left: T.Tensor((block, reduction * reduction_tiles), T.float16),
        right: T.Tensor(
            (2, reduction * reduction_tiles, block),
            T.float16,
        ),
        output: T.Tensor((2, block, block), T.float32),
    ):
        with T.Kernel(
            2,
            threads=128,
            cluster_dims=cluster_dims,
        ) as block_id:
            left_shared = T.alloc_shared((block, reduction), T.float16)
            right_shared = T.alloc_shared((reduction, block), T.float16)
            accumulator = T.alloc_fragment((block, block), T.float32)
            T.clear(accumulator)
            for tile in T.Pipelined(reduction_tiles, num_stages=2):
                T.copy(
                    left[
                        :,
                        tile * reduction : (tile + 1) * reduction,
                    ],
                    left_shared,
                    valid_region=left,
                    synchronization_owner="pipeline",
                    annotations={"cluster_mask": cluster_mask},
                )
                T.copy(
                    right[
                        block_id,
                        tile * reduction : (tile + 1) * reduction,
                        :,
                    ],
                    right_shared,
                    valid_region=right[block_id, :, :],
                    synchronization_owner="pipeline",
                )
                T.gemm(left_shared, right_shared, accumulator)
            T.copy(accumulator, output[block_id, :, :])

    return before


def dynamic_multicast_dual_copy_pipeline():
    block = 64
    reduction = 32

    @T.prim_func
    def before(
        left: T.Tensor((block, 128), T.float16),
        right: T.Tensor(
            (2, 128, block),
            T.float16,
        ),
        output: T.Tensor((2, block, block), T.float32),
        tile_count: T.int32,
    ):
        with T.Kernel(
            2,
            threads=128,
            cluster_dims=(2, 1, 1),
        ) as block_id:
            left_shared = T.alloc_shared((block, reduction), T.float16)
            right_shared = T.alloc_shared((reduction, block), T.float16)
            accumulator = T.alloc_fragment((block, block), T.float32)
            T.clear(accumulator)
            for tile in T.Pipelined(tile_count, num_stages=2):
                T.copy(
                    left[
                        :,
                        tile * reduction : (tile + 1) * reduction,
                    ],
                    left_shared,
                    valid_region=left,
                    synchronization_owner="pipeline",
                    annotations={"cluster_mask": 3},
                )
                T.copy(
                    right[
                        block_id,
                        tile * reduction : (tile + 1) * reduction,
                        :,
                    ],
                    right_shared,
                    valid_region=right[block_id, :, :],
                    synchronization_owner="pipeline",
                )
                T.gemm(left_shared, right_shared, accumulator)
            T.copy(accumulator, output[block_id, :, :])

    return before


def nested_multicast_dual_copy_pipeline():
    block = 64
    reduction = 32

    @T.prim_func
    def before(
        left: T.Tensor((block, 128), T.float16),
        right: T.Tensor((2, 128, block), T.float16),
        output: T.Tensor((2, block, block), T.float32),
    ):
        with T.Kernel(
            2,
            threads=128,
            cluster_dims=(2, 1, 1),
        ) as block_id:
            left_shared = T.alloc_shared((block, reduction), T.float16)
            right_shared = T.alloc_shared((reduction, block), T.float16)
            accumulator = T.alloc_fragment((block, block), T.float32)
            for _ in T.serial(2):
                T.clear(accumulator)
                for tile in T.Pipelined(4, num_stages=2):
                    T.copy(
                        left[
                            :,
                            tile * reduction : (tile + 1) * reduction,
                        ],
                        left_shared,
                        valid_region=left,
                        synchronization_owner="pipeline",
                        annotations={"cluster_mask": 3},
                    )
                    T.copy(
                        right[
                            block_id,
                            tile * reduction : (tile + 1) * reduction,
                            :,
                        ],
                        right_shared,
                        valid_region=right[block_id, :, :],
                        synchronization_owner="pipeline",
                    )
                    T.gemm(left_shared, right_shared, accumulator)
                T.copy(accumulator, output[block_id, :, :])

    return before


def apply_sm90_warp_specialization(program):
    target = tvm.target.Target("cuda -arch=sm_90a")
    mod = tvm.IRModule.from_expr(program.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.MaterializeLogicalGemm()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
    return mod, target


def test_sm90_multicast_pipeline_uses_cluster_wide_stage_backpressure():
    mod, _ = apply_sm90_warp_specialization(static_multicast_dual_copy_pipeline())
    decisions = pipeline_decision_records(mod["main"])
    assert len(decisions) == 1
    assert decisions[0]["selected_implementation"] == "warp_specialized"
    assert int(decisions[0]["fallback"]) == 0

    call_names = []
    cluster_arrive_counts = []

    def collect(node):
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op):
            call_names.append(node.op.name)
        if isinstance(node, tir.Block):
            barrier_init = node.annotations.get("barrier_init")
            if barrier_init is None:
                return
            for buffer in node.alloc_buffers:
                if buffer.scope() == "shared.cluster_barrier":
                    cluster_arrive_counts.append([int(value) for value in barrier_init[buffer.data]])

    tir.stmt_functor.post_order_visit(mod["main"].body, collect)
    assert cluster_arrive_counts == [[256, 256]]
    assert "tl.ptx_arrive_cluster_barrier" in call_names
    assert "tl.cluster_sync" in call_names
    assert "tl.mbarrier_wait_parity" in call_names


def test_sm90_dynamic_multicast_pipeline_selects_executable_sync_fallback():
    mod, target = apply_sm90_warp_specialization(dynamic_multicast_dual_copy_pipeline())
    with target:
        mod = tl.transform.PipelinePlanning()(mod)
    decisions = pipeline_decision_records(mod["main"])
    assert len(decisions) == 1
    assert decisions[0]["selected_implementation"] == "synchronous"
    assert int(decisions[0]["fallback"]) == 1
    assert "dynamic multicast loop extent" in decisions[0]["selection_reason"]

    consumed_modes = []
    pipeline_loops = []

    def collect(node):
        if isinstance(node, tir.For) and "num_stages" in node.annotations:
            pipeline_loops.append(node)
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tileop.copy":
            mode = node.annotations.get("tl.transfer_pipeline_sync_consumed")
            if mode is not None:
                consumed_modes.append(int(mode))

    tir.stmt_functor.post_order_visit(mod["main"].body, collect)
    assert pipeline_loops == []
    assert consumed_modes == [2, 2]

    with target:
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
    )
    assert "tl.tma_load" not in call_names
    assert "tl.ptx_arrive_cluster_barrier" not in call_names
    assert "tir.tvm_storage_sync" in call_names


def test_sm90_nested_multicast_pipeline_selects_sync_fallback():
    mod, target = apply_sm90_warp_specialization(nested_multicast_dual_copy_pipeline())
    with target:
        mod = tl.transform.PipelinePlanning()(mod)
    decisions = pipeline_decision_records(mod["main"])
    assert len(decisions) == 1
    assert decisions[0]["selected_implementation"] == "synchronous"
    assert int(decisions[0]["fallback"]) == 1
    assert "nested multicast pipeline" in decisions[0]["selection_reason"]


@pytest.mark.parametrize(
    ("cluster_mask", "cluster_dims", "reason"),
    [
        (4, (2, 1, 1), "outside cluster_dims"),
        (3, None, "multi-CTA cluster topology"),
    ],
)
def test_sm90_invalid_multicast_pipeline_selects_sync_fallback(
    cluster_mask,
    cluster_dims,
    reason,
):
    mod, _ = apply_sm90_warp_specialization(
        static_multicast_dual_copy_pipeline(
            cluster_mask=cluster_mask,
            cluster_dims=cluster_dims,
        )
    )
    target = tvm.target.Target("cuda -arch=sm_90a")
    with target:
        mod = tl.transform.PipelinePlanning()(mod)
    decisions = pipeline_decision_records(mod["main"])
    assert len(decisions) == 1
    assert decisions[0]["selected_implementation"] == "synchronous"
    assert int(decisions[0]["fallback"]) == 1
    assert reason in decisions[0]["selection_reason"]


@tl.testing.requires_cuda
@tl.testing.requires_cuda_compute_version_ge(9, 0)
def test_sm90_multicast_pipeline_runtime_is_stable():
    import torch

    kernel = tl.compile(
        static_multicast_dual_copy_pipeline(),
        out_idx=[2],
    )
    left = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    right = torch.randn(
        (2, 128, 64),
        device="cuda",
        dtype=torch.float16,
    )
    expected = torch.stack([left.float() @ right[index].float() for index in range(2)])
    baseline = None
    for _ in range(3):
        actual = kernel(left, right)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        if baseline is not None:
            torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        baseline = actual


@tl.testing.requires_cuda
@tl.testing.requires_cuda_compute_version_ge(9, 0)
def test_sm90_dynamic_multicast_sync_fallback_runtime():
    import torch

    kernel = tl.compile(
        dynamic_multicast_dual_copy_pipeline(),
        out_idx=[2],
    )
    left = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    right = torch.randn(
        (2, 128, 64),
        device="cuda",
        dtype=torch.float16,
    )
    actual = kernel(left, right, 3)
    expected = torch.stack([left[:, :96].float() @ right[index, :96, :].float() for index in range(2)])
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_contract_copy_stays_synchronous_through_pipeline_and_ws_passes():
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def before(
        source: T.Tensor((16,), T.float16),
        destination: T.Tensor((16,), T.float16),
        valid_end: T.int32,
    ):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16,), T.float16)
            for _ in T.Pipelined(1, num_stages=2):
                T.copy(
                    source,
                    shared,
                    valid_region=source[0:valid_end],
                    oob_fill=0,
                    allow_async=False,
                )
                for i in T.Parallel(16):
                    destination[i] = shared[i]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
    )
    assert "tir.if_then_else" in call_names
    assert "tl.ptx_cp_async" not in call_names
    assert "tl.tma_load" not in call_names


def test_cuda_transfer_lowering_plan_selects_full_and_tail_tma():
    source = tir.decl_buffer((96, 64), "float16", name="Source")
    destination = tir.decl_buffer(
        (64, 64),
        "float16",
        name="Destination",
        scope="shared",
    )

    full = resolve_transfer(
        T.copy(
            source[0:64, 0:64],
            destination,
            valid_region=region(source, (0, 96), (0, 64)),
        ),
        "sm_90",
    )
    assert full.supported is True
    assert full.implementation_id == "cuda.tma.load.full"
    assert full.asynchronous is True
    assert full.uses_tma_descriptor is True
    assert full.requires_post_fill is False

    tail = resolve_transfer(
        T.copy(
            source[64:128, 0:64],
            destination,
            valid_region=region(source, (0, 96), (0, 64)),
            oob_fill=-2,
        ),
        "sm_90",
    )
    assert tail.supported is True
    assert tail.implementation_id == "cuda.tma.load.tail_oob"
    assert tail.uses_tma_descriptor is True
    assert tail.requires_post_fill is True
    assert any(item.startswith("cuda.tma.load.full:") for item in tail.rejected_candidates)

    tensor_oob_zero = resolve_transfer(
        T.copy(
            source[64:128, 0:64],
            destination,
            valid_region=region(source, (0, 96), (0, 64)),
            oob_fill=0,
        ),
        "sm_90",
    )
    assert tensor_oob_zero.implementation_id == "cuda.tma.load.tail_oob"
    assert tensor_oob_zero.requires_post_fill is False
    assert "tensor-bound zero-fill" in str(tensor_oob_zero.selection_reason)


def test_cuda_transfer_tma_keeps_post_fill_for_in_bounds_contract_tail():
    source = tir.decl_buffer((128, 64), "float16", name="Source")
    destination = tir.decl_buffer(
        (64, 64),
        "float16",
        name="Destination",
        scope="shared",
    )
    plan = resolve_transfer(
        T.copy(
            source[32:96, 0:64],
            destination,
            valid_region=region(source, (32, 32), (0, 64)),
            oob_fill=0,
        ),
        "sm_90",
    )

    assert plan.implementation_id == "cuda.tma.load.tail_oob"
    assert plan.requires_post_fill is True


@pytest.mark.parametrize(
    ("arch", "fill", "expected"),
    [
        ("sm_80", 0, "cuda.cp_async"),
        ("sm_70", 0, "common.simt"),
        ("sm_80", 3, "common.simt"),
    ],
)
def test_cuda_transfer_lowering_plan_uses_capability_and_fill_fallbacks(
    arch,
    fill,
    expected,
):
    source = tir.decl_buffer((40, 16), "float16", name="Source")
    destination = tir.decl_buffer(
        (32, 16),
        "float16",
        name="Destination",
        scope="shared",
    )
    plan = resolve_transfer(
        T.copy(
            source[24:56, 0:16],
            destination,
            valid_region=region(source, (0, 40), (0, 16)),
            oob_fill=fill,
        ),
        arch,
    )
    assert plan.supported is True
    assert plan.implementation_id == expected
    assert plan.requires_post_fill is False


def test_cuda_transfer_lowering_plan_respects_async_permission_and_tma_legality():
    source = tir.decl_buffer((512, 16), "float16", name="Source")
    large_destination = tir.decl_buffer(
        (300, 16),
        "float16",
        name="LargeDestination",
        scope="shared",
    )
    tma_illegal = resolve_transfer(
        T.copy(
            source[0:300, 0:16],
            large_destination,
            valid_region=region(source, (0, 512), (0, 16)),
        ),
        "sm_90",
    )
    assert tma_illegal.implementation_id == "cuda.cp_async"
    assert any("non-innermost box extents" in item for item in tma_illegal.rejected_candidates)

    destination = tir.decl_buffer(
        (32, 16),
        "float16",
        name="Destination",
        scope="shared",
    )
    synchronous = resolve_transfer(
        T.copy(
            source[0:32, 0:16],
            destination,
            valid_region=region(source, (0, 512), (0, 16)),
            allow_async=False,
        ),
        "sm_90",
    )
    assert synchronous.implementation_id == "common.simt"
    assert synchronous.asynchronous is False
    assert len(synchronous.rejected_candidates) == 3

    full_nonzero_fill = resolve_transfer(
        T.copy(
            source[0:32, 0:16],
            destination,
            valid_region=region(source, (0, 512), (0, 16)),
            oob_fill=3,
        ),
        "sm_80",
    )
    assert full_nonzero_fill.implementation_id == "cuda.cp_async"

    tail_call = T.copy(
        source[480:512, 0:16],
        destination,
        valid_region=region(source, (0, 500), (0, 16)),
        oob_fill=0,
    )
    with tvm.transform.PassContext(config={"tl.disable_tma_lower": True}):
        tma_disabled = resolve_transfer(tail_call, "sm_90")
    assert tma_disabled.implementation_id == "cuda.cp_async"
    assert any("disabled by compile policy" in item for item in tma_disabled.rejected_candidates)


@pytest.mark.parametrize("thread_count", [128, 384])
def test_sm90_tail_tma_lowering_waits_then_repairs_invalid_lanes(thread_count):
    target = tvm.target.Target("cuda -arch=sm_90")

    @T.prim_func
    def before(
        source: T.Tensor((96, 64), T.float16),
        destination: T.Tensor((384,), T.float16),
        valid_end: T.int32,
    ):
        with T.Kernel(1, threads=thread_count):
            shared = T.alloc_shared((64, 64), T.float16)
            T.copy(
                source[64:128, 0:64],
                shared,
                valid_region=source[0:valid_end, 0:64],
                oob_fill=-2,
            )
            for i in T.Parallel(384):
                destination[i] = shared[T.floordiv(i, 64), i % 64]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    fill_values = []

    def visit(node):
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op):
            call_names.append(node.op.name)
        if isinstance(node, (tir.FloatImm, tir.IntImm)):
            fill_values.append(float(node.value))

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert "tl.tma_load" in call_names
    assert "tl.mbarrier_wait_parity" in call_names
    assert "tir.tvm_storage_sync" in call_names
    assert -2.0 in fill_values


def test_sm90_tensor_oob_zero_fill_needs_no_cooperative_repair():
    target = tvm.target.Target("cuda -arch=sm_90")

    @T.prim_func
    def before(
        source: T.Tensor((96, 64), T.float16),
        destination: T.Tensor((384,), T.float16),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            T.copy(
                source[64:128, 0:64],
                shared,
                valid_region=source[0:96, 0:64],
                oob_fill=0,
            )
            for i in T.Parallel(384):
                destination[i] = shared[T.floordiv(i, 64), i % 64]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
    )
    assert "tl.tma_load" in call_names
    assert "tl.mbarrier_wait_parity" in call_names
    assert "tir.tvm_storage_sync" not in call_names


def test_sm90_tensor_oob_proof_uses_symbolic_coordinate_assumptions():
    target = tvm.target.Target("cuda -arch=sm_90")

    @T.prim_func
    def before(
        source: T.Tensor((1, 32, 64), T.float16),
        destination: T.Tensor((384,), T.float16),
        head_block: T.int32,
    ):
        T.assume(head_block >= 0)
        T.assume(head_block < 1)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            head_start = head_block * 32
            T.copy(
                source[0, head_start : head_start + 64, 0:64],
                shared,
                valid_region=source[0, head_start : head_start + 32, 0:64],
                oob_fill=0,
            )
            for i in T.Parallel(384):
                destination[i] = shared[T.floordiv(i, 64), i % 64]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.InjectAssumes()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    call_names = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: call_names.append(node.op.name) if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) else None,
    )
    assert "tl.tma_load" in call_names
    assert "tl.mbarrier_wait_parity" in call_names
    assert "tir.tvm_storage_sync" not in call_names


@tl.testing.requires_cuda
@tl.testing.requires_cuda_compute_version_ge(9, 0)
def test_sm90_tail_tma_runtime_supports_unpadded_source_and_nonzero_fill():
    import torch

    @T.prim_func
    def main(
        source: T.Tensor((70, 64), T.float16),
        destination: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            T.copy(
                source[32:96, 0:64],
                shared,
                valid_region=source[0:70, 0:64],
                oob_fill=-2,
            )
            T.copy(shared, destination)

    kernel = tl.compile(
        main,
        out_idx=[1],
        pass_configs={
            tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    source = torch.randn((70, 64), device="cuda", dtype=torch.float16)
    actual = kernel(source)
    expected = torch.full(
        (64, 64),
        -2,
        device="cuda",
        dtype=torch.float16,
    )
    expected[:38] = source[32:70]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tl.testing.main()
