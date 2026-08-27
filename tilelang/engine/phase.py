from __future__ import annotations
from tvm import tir, IRModule
from tvm import ir as tvm_ir
from tvm.target import Target
import tilelang
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma, have_pdl


def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if (not is_cuda_target(target)) or (not have_tma(target)):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


def module_has_tma(mod: IRModule) -> bool:
    """Check if any function in the module was lowered with TMA operations.

    This reads the ``tl.has_tma`` attribute set by ``LowerTileOp`` during
    ``LowerAndLegalize``, which is the source of truth for whether TMA
    copies were actually generated.
    """
    return any(func.attrs and func.attrs.get("tl.has_tma", False) for _, func in mod.functions.items())


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize


def allow_global_thread_synchronization(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_global_thread_sync = pass_ctx.config.get("tir.detect_global_barrier", False)
    return enable_global_thread_sync


def should_enable_aggressive_merge(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_aggressive_merge = bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        # This is a workaround to avoid the bug in the MergeSharedMemoryAllocations pass
        # when warp specialization is enabled, as different warp threads may access different
        # buffers, but the liveness analysis is hard because we need to do pipeline.
        enable_aggressive_merge = False
    return enable_aggressive_merge


def should_force_let_inline(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_FORCE_LET_INLINE, False))


def should_enable_ast_print(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_AST_PRINT_ENABLE, False))


def should_enable_layout_visual(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False)
    return enabled


def should_enable_race_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, False)
    return enabled


def slice_handoff_layout_for_single_stage(layout, logical_shape):
    """Remove leading pipeline-stage dimensions from a handoff layout."""

    from tilelang.layout import Layout

    input_shape = list(layout.get_input_shape())
    logical_shape = list(logical_shape)
    leading = len(input_shape) - len(logical_shape)
    if leading < 0 or input_shape[leading:] != logical_shape:
        raise RuntimeError("handoff consumer layout does not match its logical transfer shape")
    if leading == 0:
        return layout

    # Fragment layouts prepend their optional replication variable to the
    # logical input variables returned here.  The actual layout inputs are
    # therefore the trailing ``len(input_shape)`` entries, not the entire
    # forward-var array.
    old_vars = list(layout.get_forward_vars())
    old_forward = list(layout.get_forward_index())
    if len(old_vars) < len(input_shape) or len(old_forward) < leading:
        raise RuntimeError("handoff consumer layout cannot be sliced by stage")
    input_vars = old_vars[-len(input_shape) :] if input_shape else []
    auxiliary_vars = old_vars[: len(old_vars) - len(input_vars)]

    def forward(*new_vars):
        substitution = {input_vars[index]: tir.IntImm("int32", 0) for index in range(leading)}
        substitution.update({input_vars[leading + index]: var for index, var in enumerate(new_vars)})
        mapped = [tir.stmt_functor.substitute(expr, substitution) for expr in old_forward[leading:]]
        for expr in mapped:
            undefined = tir.analysis.undefined_vars(expr, list(new_vars))
            if any(var.same_as(auxiliary) for var in undefined for auxiliary in auxiliary_vars):
                raise RuntimeError("handoff consumer layout index depends on a fragment replication variable and cannot be sliced by stage")
        return mapped

    return Layout(logical_shape, forward)


def bind_cross_handler_handoff_layouts(mod: IRModule) -> IRModule:
    """Bind producer arena views to the consumer's inferred single-stage layout."""

    from tilelang import _ffi_api

    fingerprint_attr = "tl.cross_handler_handoff_plan_fingerprint"
    consumer_transfer_attr = "tl.cross_handler_handoff_transfer_index"
    producer_transfer_attr = "tl.cross_handler_handoff_producer_transfer"
    role_attr = "tl.cross_handler_handoff_role"

    def string_value(value) -> str:
        return str(getattr(value, "value", value))

    def copy_buffer_by_transfer(func, annotation_name):
        result = {}

        def visit(node):
            if not (
                isinstance(node, tir.Call) and isinstance(node.op, tvm_ir.Op) and node.op.name in {"tl.tileop.copy", "tl.tileop.tma_copy"}
            ):
                return
            transfer = node.annotations.get(annotation_name)
            if transfer is None:
                return
            parsed = _ffi_api.ParseOperator(node)
            buffers = result.setdefault(int(transfer), [])
            if not any(parsed.dst.data.same_as(item.data) for item in buffers):
                buffers.append(parsed.dst)

        tir.stmt_functor.post_order_visit(func.body, visit)
        return result

    def layout_for_data(func, data):
        found = []

        def visit(node):
            if not isinstance(node, tir.Block):
                return
            layout_map = node.annotations.get("layout_map")
            if layout_map is None:
                return
            for key, layout in layout_map.items():
                key_data = key.data if isinstance(key, tir.Buffer) else key
                if isinstance(key_data, tir.Var) and key_data.same_as(data):
                    found.append(layout)

        tir.stmt_functor.post_order_visit(func.body, visit)
        if not found:
            raise RuntimeError("handoff consumer buffer lacks an inferred layout")
        first = found[0]
        if any(not tvm_ir.structural_equal(first, item) for item in found[1:]):
            raise RuntimeError("handoff consumer buffer has conflicting inferred layouts")
        return first

    consumer_layouts = {}
    for _, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc) or not func.attrs:
            continue
        if string_value(func.attrs.get(role_attr, "")) != "consumer":
            continue
        fingerprint = string_value(func.attrs.get(fingerprint_attr, ""))
        if not fingerprint:
            continue
        for transfer, buffers in copy_buffer_by_transfer(
            func,
            consumer_transfer_attr,
        ).items():
            if len(buffers) != 1:
                raise RuntimeError("handoff consumer transfer must target one buffer")
            buffer = buffers[0]
            layout = layout_for_data(func, buffer.data)
            key = (fingerprint, transfer)
            previous = consumer_layouts.get(key)
            if previous is not None and not tvm_ir.structural_equal(previous, layout):
                raise RuntimeError(f"handoff consumer variants infer conflicting layouts for transfer {transfer}")
            consumer_layouts[key] = layout

    updates = {}
    for global_var, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc) or not func.attrs:
            continue
        if string_value(func.attrs.get(role_attr, "")) != "producer":
            continue
        fingerprint = string_value(func.attrs.get(fingerprint_attr, ""))
        producer_buffers = copy_buffer_by_transfer(func, producer_transfer_attr)
        if not producer_buffers:
            continue
        layouts = []
        for transfer, buffers in producer_buffers.items():
            consumer_key = (fingerprint, transfer)
            if consumer_key not in consumer_layouts:
                raise RuntimeError(f"handoff producer transfer has no matching consumer layout: transfer={transfer}")
            for buffer in buffers:
                layouts.append(
                    (
                        buffer,
                        slice_handoff_layout_for_single_stage(
                            consumer_layouts[consumer_key],
                            buffer.shape,
                        ),
                    )
                )

        def rewrite(node, layouts=tuple(layouts)):
            if not isinstance(node, tir.Block):
                return node
            if "layout_map" not in node.annotations:
                return node
            annotations = dict(node.annotations)
            layout_map = dict(annotations.get("layout_map", {}))
            bound_data = []
            for key in tuple(layout_map):
                key_data = key.data if isinstance(key, tir.Buffer) else key
                if not isinstance(key_data, tir.Var):
                    continue
                for buffer, layout in layouts:
                    data = buffer.data
                    if key_data.same_as(data):
                        layout_map[key] = layout
                        if not any(item.same_as(data) for item in bound_data):
                            bound_data.append(data)
            for buffer, layout in layouts:
                if not any(data.same_as(buffer.data) for data in bound_data):
                    layout_map[buffer] = layout
            annotations["layout_map"] = layout_map
            return tir.Block(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                annotations,
                getattr(node, "span", None),
            )

        body = tir.stmt_functor.ir_transform(
            func.body,
            None,
            rewrite,
            ["tir.Block"],
        )
        updates[global_var] = func.with_body(body)
    if updates:
        mod.update(IRModule(updates))
    return mod


def should_enable_prelower_semantic_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK, False)
    return enabled


def get_layout_visual_formats(pass_ctx: PassContext | None = None) -> list[str]:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    formats_value = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS, "")
    if not formats_value:
        return ["txt"]

    formats_str = formats_value.strip().lower()
    valid_formats = ["txt", "png", "pdf", "svg", "all"]

    if formats_str == "all":
        return ["txt", "png", "pdf", "svg"]

    if "," in formats_str:
        formats_list = [f.strip() for f in formats_str.split(",")]
    else:
        formats_list = [formats_str]

    invalid_formats = [f for f in formats_list if f not in valid_formats]
    if invalid_formats:
        raise ValueError(
            f"Invalid formats for TL_LAYOUT_VISUALIZATION_FORMATS: {invalid_formats}. "
            f"Valid formats are: {valid_formats}. "
            f"You can choose one of the valid formats or a comma-separated list of formats.(e.g., 'txt,png,pdf')"
        )
    return formats_list


def LayoutVisual(mod: IRModule) -> None:
    """Apply layout visualization pass if enabled."""
    if should_enable_layout_visual():
        formats = get_layout_visual_formats()
        tilelang.analysis.LayoutVisual(formats=formats)(mod)


def PreLowerSemanticCheck(mod: IRModule) -> None:
    """
    Check whether the module is valid before lowering. If not, raise a user-friendly error
    in Python side instead of letting the error dive into the complicated TVM/C++ stack.
    Note: This is a validation-only pipeline of passes and does not modify or return the module.
    """

    if not should_enable_prelower_semantic_check():
        return

    # Print AST for debugging purpose
    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    # Check if there are any invalid nested loops.
    tilelang.analysis.NestedLoopChecker()(mod)
    # Check if there are any invalid symbolic T.Parallel + fragment access.
    tilelang.analysis.FragmentLoopChecker()(mod)


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # Bind the target device information to the module
    """
    Bind target information and progressively legalize and lower frontend Tile IR into a form suitable for downstream optimization and codegen.

    This pass pipeline:
    - Binds the provided target to the module.
    - Legalizes frontend Tile IR into TVM-compatible constructs.
    - Simplifies expressions.
    - Configures reducer layouts and performs layout inference for fragments and shared memory.
    - Lowers high-level tile operations and L2 persistent maps.
    - Legalizes vectorized loops and inserts safety checks for memory accesses.
    - Re-simplifies to remove redundancies introduced by safety checks.
    - Attempts loop vectorization for dynamic-shaped loops.

    Parameters:
        mod (IRModule): The input IR module containing frontend Tile IR.
        target (Target): Target device information to bind into the module.

    Returns:
        IRModule: The transformed module, ready for target-specific optimization passes.
    """
    mod = tir.transform.BindTarget(target)(mod)

    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = tilelang.transform.LetInline()(mod)
    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    # Normalize negative indices to canonical non-negative form
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Verify parallel loop correctness
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    # Resolve logical GEMM implementations and materialize any compiler-owned
    # physical padding before pipeline and layout transforms inspect shapes.
    mod = tilelang.transform.MaterializeLogicalGemm()(mod)
    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)
    # Tile-level warp specialization: runs before layout inference so that
    # producer/consumer split happens at the high-level tile-op IR.
    # The pass classifies copy ops as TMA/cp.async/sync inline (no prior
    # InstructionAnnotation pass needed). Shared buffers are multi-versioned
    # internally only for functions where the WS transformation actually
    # applies.
    if allow_warp_specialized(target=target):
        mod = tilelang.transform.ProducerConsumerWarpSpecialized()(mod)
    # Lower 2SM TCGEN5MMA and related on Blackwell target (must run before
    # LayoutInference so that the use_2cta annotation is visible to infer_layout)
    mod = tilelang.transform.LowerBlackwell2SM()(mod)
    # Run pipeline planning and software-pipeline rewriting before layout
    # inference so inferred layouts see the final pipelined structure directly.
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)
    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    mod = bind_cross_handler_handoff_layouts(mod)
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    # Lower l2 persistent map
    mod = tilelang.cuda.transform.LowerL2Persistent()(mod)
    # Decouple type cast vectorization constraints before vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Lower frontend pointer metadata op to standard tvm_access_ptr
    mod = tilelang.transform.LowerAccessPtr()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = tilelang.transform.Simplify()(mod)
    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()
    # Lower the shared.tmem into specific initialization slot
    mod = tilelang.transform.LowerSharedTmem()(mod)
    # which may be introduced by the LegalizeSafeMemoryAccess
    mod = tilelang.transform.IfStmtBinding()(mod)
    has_tma = module_has_tma(mod)
    # Pipeline barriers are now created at final expanded size by
    # InjectSoftwarePipeline, so no late MVB barrier fixup is needed.
    # Buffer allocation placement is handled uniformly for both paths.
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    if has_tma:
        mod = tilelang.transform.FuseMBarrierArriveExpectTx()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    # ConfigIndexBitwidth must be applied after FlattenBuffer
    # as it will flatten index computing
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.transform.LowerLDGSTG()(mod)
    mod = tilelang.cuda.transform.LowerHopperIntrin()(mod)
    # Global Barrier Synchronization must be applied before
    # SplitHostDevice pass, as the global barrier
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    # Mark the function contains pdl_sync or pdl_trigger
    mod = tilelang.transform.MarkCudaSyncCalls(have_pdl(target))(mod)

    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge)(mod)
    # InjectFenceProxy is a no-op on targets that lack the TMA / async-proxy
    # programming model; the pass itself checks the PrimFunc's target.
    mod = tilelang.transform.InjectFenceProxy()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    # Inject conservative tcgen05 fences on Blackwell (SM100+).
    # Must run after ThreadSync so that tvm_storage_sync calls are present.
    # The pass handles shared syncs and simple linear wait/use, use/arrive
    # handoffs, and is a no-op on non-SM100 targets or functions without TMEM.
    mod = tilelang.transform.InjectTcgen05Fence()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    # NOTE: LowerPTXAsyncCopy is applied earlier (before PipelinePlanning).
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    # Transform threadblock to persistent threadblock
    mod = tilelang.cuda.transform.PersistThreadblock()(mod)

    return mod
