"""Copy operations exposed on the TileLang language surface."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from tilelang._typing import BufferLikeType
from tilelang.language.utils import get_extent
from tilelang.utils.language import (
    legalize_pairwise_extents,
    to_buffer_region,
)
from tvm import ir, tir


class TransferSynchronizationOwner(IntEnum):
    """Owner responsible for making an asynchronously issued transfer visible."""

    TRANSFER = 0
    PIPELINE = 1
    CALLER = 2


_TRANSFER_OWNER_BY_NAME = {
    "transfer": TransferSynchronizationOwner.TRANSFER,
    "pipeline": TransferSynchronizationOwner.PIPELINE,
    "caller": TransferSynchronizationOwner.CALLER,
}
_UNSET = object()


def is_region_call(value: Any) -> bool:
    return isinstance(value, tir.Call) and isinstance(value.op, ir.Op) and value.op.name == "tl.tileop.region"


def region_call_extents(value: Any):
    if is_region_call(value):
        return list(value.args[2:])
    return get_extent(value)


def encode_region(value: Any, *, access_type: str, extents: list[Any]):
    if is_region_call(value):
        return value
    return to_buffer_region(value, access_type=access_type, extents=extents)


def normalize_transfer_owner(
    owner: str | int | TransferSynchronizationOwner,
) -> TransferSynchronizationOwner:
    if isinstance(owner, str):
        try:
            return _TRANSFER_OWNER_BY_NAME[owner]
        except KeyError as err:
            expected = ", ".join(sorted(_TRANSFER_OWNER_BY_NAME))
            raise ValueError(f"synchronization_owner must be one of {expected}, got {owner!r}") from err
    if isinstance(owner, bool):
        raise TypeError("synchronization_owner cannot be a bool")
    try:
        return TransferSynchronizationOwner(owner)
    except (TypeError, ValueError) as err:
        raise ValueError(f"invalid synchronization_owner {owner!r}") from err


def transfer_contract(
    valid_region: BufferLikeType,
    oob_fill: Any = 0,
    allow_async: bool = True,
    synchronization_owner: str | int | TransferSynchronizationOwner = "transfer",
) -> tir.Call:
    """Create a target-independent bounded-transfer contract.

    ``valid_region`` is an absolute rectangular region of the source buffer.
    Source coordinates outside that region produce ``oob_fill``. The contract
    records whether asynchronous execution is permitted and which layer owns
    completion; it does not select a target instruction.
    """

    valid_extents = region_call_extents(valid_region)
    if valid_extents is None:
        raise TypeError("valid_region must carry an explicit rectangular extent")
    if is_region_call(valid_region):
        access_mask = valid_region.args[1]
        if not isinstance(access_mask, tir.IntImm) or access_mask.value != 1:
            raise ValueError("valid_region must use read access")
    if not isinstance(allow_async, bool):
        raise TypeError(f"allow_async must be a bool, got {allow_async!r}")
    owner = normalize_transfer_owner(synchronization_owner)
    if not allow_async and owner is not TransferSynchronizationOwner.TRANSFER:
        raise ValueError("a transfer that disallows asynchronous execution must own its synchronization")
    if not isinstance(oob_fill, tir.PrimExpr):
        if not isinstance(oob_fill, (bool, int, float)):
            raise TypeError(f"oob_fill must be a scalar value, got {oob_fill!r}")
        oob_fill = tir.const(oob_fill)
    fill_buffer_loads = []
    tir.stmt_functor.post_order_visit(
        oob_fill,
        lambda node: fill_buffer_loads.append(node) if isinstance(node, tir.BufferLoad) else None,
    )
    if fill_buffer_loads:
        raise ValueError("oob_fill cannot read a buffer; pass a constant or scalar parameter")

    encoded_valid_region = encode_region(
        valid_region,
        access_type="r",
        extents=list(valid_extents),
    )
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.transfer_contract"),
        encoded_valid_region,
        oob_fill,
        int(allow_async),
        int(owner),
    )


def _normalize_copy_regions(
    src: BufferLikeType, dst: BufferLikeType
) -> tuple[
    tir.BufferRegion | tir.BufferLoad | tir.Buffer,
    tir.BufferRegion | tir.BufferLoad | tir.Buffer,
]:
    # If both side are buffers, we should make sure their shapes are equal
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    src_extent = region_call_extents(src)
    dst_extent = region_call_extents(dst)

    src_is_scalar_load = src_extent is None and isinstance(src, tir.BufferLoad)
    dst_is_scalar_load = dst_extent is None and isinstance(dst, tir.BufferLoad)

    # copy(buffer_a[i], buffer_b[i]) where both are BufferLoad nodes
    # In this case, lower it to a simple BufferStore: buffer_b[i] = buffer_a[i]
    if src_is_scalar_load and dst_is_scalar_load:
        return src, dst

    assert src_extent or dst_extent, "Can't deduce copy extents from args. Both src and dst miss extents info."
    # Treat missing extent as length-matched ones for convenience. This provides limited
    # broadcasting-like syntactic sugar, but does not implement general broadcasting support.
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    # Align and broadcast extents from the right (tail) side.
    # This is majorly for supporting some syntactic sugar, not the whole broadcasting ability of copy op.
    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    # Use legalized extents for src and dst respectively.
    src = encode_region(src, access_type="r", extents=src_extent)
    dst = encode_region(dst, access_type="w", extents=dst_extent)
    return src, dst


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    contract: tir.Call | None = None,
    *,
    valid_region: BufferLikeType | None = None,
    oob_fill: Any = _UNSET,
    allow_async: bool | None = None,
    synchronization_owner: str | int | TransferSynchronizationOwner | None = None,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt:
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Destination memory region
        contract (Optional[tir.Call]): Pre-built ``T.transfer_contract``. The positional
            form exists so printed TileLang IR can round-trip.
        valid_region (Optional[BufferLikeType], keyword-only): Absolute valid source
            region for an opt-in bounded transfer.
        oob_fill (Optional[scalar], keyword-only): Value written for source coordinates
            outside ``valid_region``. Defaults to zero when ``valid_region`` is set.
        allow_async (Optional[bool], keyword-only): Whether an implementation may issue
            the transfer asynchronously. Defaults to True for a bounded transfer.
        synchronization_owner (Optional[str], keyword-only): ``transfer``, ``pipeline``,
            or ``caller``. Defaults to ``transfer``.
        coalesced_width (Optional[int], keyword-only): Width for coalesced memory access. Defaults to None.
        disable_tma (bool, keyword-only): Whether to disable TMA acceleration. Defaults to False.
        eviction_policy (Optional[str], keyword-only): Cache eviction policy. Defaults to None.
        annotations (Optional[dict], keyword-only): Additional annotations dict. If provided,
            coalesced_width, disable_tma, and eviction_policy can also be specified here.
            Values in annotations take precedence over individual arguments.
        loop_layout (Optional[Fragment], keyword-only): A parallel loop layout hint for the SIMT copy
            (only valid for normal SIMT copy; incompatible with TMA/LDSM/STSM/TMem). When provided,
            it is attached to the outermost parallel loop generated by this copy.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation

    Range handling notes:
    - Accepts `Buffer`/`BufferRegion`/`BufferLoad` on either side. Extents are
      derived as follows: `Buffer -> shape`, `BufferRegion -> [r.extent]`,
      `BufferLoad -> extents from its inferred/encoded region`.
    - Normally, we require the extents of both sides to be the same. If they
      differ, the copy instruction follows an internal rule to select one side
      as the base range and create iteration space. This may generate unexpected
      code. And if some dimensions are 1, unexpected errors may happen.
    - Small Optimization: If both `src` and `dst` are scalar `BufferLoad` without
      region extents, lowers to a direct store: `dst[...] = src[...]`.
    - Syntactic Sugar: TileLang supports passing the head address of a buffer to represent
      the whole buffer if there are no ambiguity. For example, T.copy(A, A_shared[i, j]).
      To support this, we need some special shape checking. But remember currently we don't
      support something like "broadcast".
    - The finalized extents are encoded with `tl.region` via `to_buffer_region`
      and passed through to the backend; low-level loop construction and any
      scope-specific decisions happen during lowering.
    """
    src, dst = _normalize_copy_regions(src, dst)
    if isinstance(src, tir.BufferLoad) and isinstance(dst, tir.BufferLoad):
        if (
            contract is not None
            or valid_region is not None
            or oob_fill is not _UNSET
            or allow_async is not None
            or synchronization_owner is not None
        ):
            raise ValueError("scalar copy does not accept a transfer contract")
        return tir.BufferStore(dst.buffer, src, dst.indices)

    if contract is not None and valid_region is not None:
        raise ValueError("provide either contract or valid_region, not both")
    if contract is None and valid_region is not None:
        contract = transfer_contract(
            valid_region,
            0 if oob_fill is _UNSET else oob_fill,
            True if allow_async is None else allow_async,
            "transfer" if synchronization_owner is None else synchronization_owner,
        )
    elif contract is None and (oob_fill is not _UNSET or allow_async is not None or synchronization_owner is not None):
        raise ValueError("oob_fill, allow_async, and synchronization_owner require valid_region")
    elif contract is not None:
        if not (isinstance(contract, tir.Call) and isinstance(contract.op, ir.Op) and contract.op.name == "tl.transfer_contract"):
            raise TypeError("contract must be produced by T.transfer_contract")
        if oob_fill is not _UNSET or allow_async is not None or synchronization_owner is not None:
            raise ValueError("contract cannot be combined with transfer convenience arguments")

    # Build annotations dict
    ann = annotations.copy() if annotations else {}

    # Individual arguments take lower precedence than annotations
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if "disable_tma" not in ann and disable_tma:
        ann["disable_tma"] = disable_tma
    if "eviction_policy" not in ann and eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]

    # Parallel loop layout hint (Fragment). Mirrors T.Parallel(loop_layout=...)
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    args = [src, dst]
    if contract is not None:
        args.append(contract)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.copy"),
        *args,
        annotations=ann if ann else None,
    )


def copy_cluster(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    dst_block: int | tir.PrimExpr | None = None,
    cluster_mask: int | None = None,
    remote_barrier: tir.BufferLoad | None = None,
    leader_thread_extent: int | None = None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt:
    """Cluster-aware copy for TMA multicast or SM-to-SM shared-memory copy.

    Args:
        src: Source memory region.
        dst: Destination memory region.
        dst_block: Destination CTA rank in the cluster for SM-to-SM copy.
        cluster_mask: Bitmask of CTAs that participate in TMA multicast.
        remote_barrier: Shared-memory mbarrier for asynchronous SM-to-SM copy
            completion signalling.  The destination CTA should wait on its
            local copy of this barrier.
        leader_thread_extent: Logical thread-group size used to elect the lane
            that issues a bulk SM-to-SM copy. Defaults to the kernel's lowering
            thread extent. Use 128 when issuing from a dedicated producer warp
            group. This does not affect the cooperative SIMT fallback.
        eviction_policy: Cache eviction hint passed to the TMA instruction.
            Only relevant for the TMA multicast path (``cluster_mask`` set).
        coalesced_width: Vectorization width (in elements) for the SIMT loop
            used on the SM-to-SM fallback path (``dst_block`` set, no fast
            bulk-async route available).
        loop_layout: Parallel loop layout hint (Fragment) for the SIMT loop on
            the SM-to-SM fallback path. Incompatible with the TMA multicast
            path (``cluster_mask`` set).

    Returns:
        tir.Call: A handle to the copy operation.
    """
    src, dst = _normalize_copy_regions(src, dst)

    ann: dict = {}
    if dst_block is not None:
        ann["dst_block"] = dst_block
    if cluster_mask is not None:
        ann["cluster_mask"] = cluster_mask
    if remote_barrier is not None:
        ann["barrier"] = remote_barrier
    if leader_thread_extent is not None:
        if not isinstance(leader_thread_extent, int):
            raise TypeError(f"leader_thread_extent must be an int or None, got {type(leader_thread_extent).__name__}")
        if leader_thread_extent != 0 and (leader_thread_extent < 32 or leader_thread_extent % 32 != 0):
            raise ValueError(f"leader_thread_extent must be 0 or a positive multiple of 32, got {leader_thread_extent}")
        ann["leader_thread_extent"] = leader_thread_extent
    if eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]
    if coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if loop_layout is not None:
        ann["parallel_loop_layout"] = loop_layout

    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.copy"), src, dst, annotations=ann if ann else None)


def async_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    src_upper_bounds: dict[int, Any] | None = None,
    assume_src_in_bounds: bool = False,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt:
    """Asynchronous copy primitive lowered through cp.async.

    This operator is intended for explicitly asynchronous global->shared copy.
    The backend enforces cp.async constraints and emits:
      `ptx_cp_async(...)` + `ptx_commit_group()`.
    No wait is auto-inserted for `T.async_copy`; synchronization is explicit.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], keyword-only): Width for coalesced memory access. Defaults to None.
        src_upper_bounds (Optional[dict[int, PrimExpr]], keyword-only): Per-source-axis dynamic
            absolute upper bounds used to form the cp.async zero-fill predicate.
        assume_src_in_bounds (bool, keyword-only): Skip source bounds checks for copies whose source
            region is known to be in-bounds. The caller is responsible for keeping this true.
        annotations (Optional[dict], keyword-only): Additional annotations dict.
        loop_layout (Optional[Fragment], keyword-only): A parallel loop layout hint for the SIMT copy loop.

    Returns:
        tir.Call: A handle to the async copy operation
    """
    src, dst = _normalize_copy_regions(src, dst)
    if isinstance(src, tir.BufferLoad) and isinstance(dst, tir.BufferLoad):
        return tir.BufferStore(dst.buffer, src, dst.indices)

    ann = annotations.copy() if annotations else {}
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if src_upper_bounds:
        for axis, bound in src_upper_bounds.items():
            if not isinstance(axis, int):
                raise TypeError(f"src_upper_bounds axis must be int, got {type(axis).__name__}")
            if axis < 0:
                raise ValueError(f"src_upper_bounds axis must be non-negative, got {axis}")
            ann[f"src_upper_bound_{axis}"] = bound if isinstance(bound, tir.PrimExpr) else tir.const(bound, "int32")
    if assume_src_in_bounds:
        ann["assume_src_in_bounds"] = 1
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.async_copy"),
        src,
        dst,
        annotations=ann if ann else None,
    )


def tma_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    barrier=None,
    expect_transaction: bool = True,
    leader_thread_extent: int | None = None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
) -> tir.PrimExpr | tir.Stmt:
    """TMA copy with user-managed synchronization.

    For **loads** (global -> shared): issues expect_tx + tma_load (no wait) by default.
    Unlike T.copy() which emits a full synchronous TMA sequence (arrive + load + wait),
    T.tma_copy() emits only the producer part (expect_tx + tma_load).
    The user manages synchronization explicitly via T.barrier_arrive() and
    T.mbarrier_wait_parity(). ``barrier`` is required for loads.

    For **stores** (shared -> global): issues tma_store + tma_store_arrive (no wait).
    Unlike T.copy() which emits tma_store + tma_store_arrive + tma_store_wait,
    T.tma_copy() omits the wait so the user can batch multiple stores before
    calling T.tma_store_wait() explicitly. ``barrier`` is not needed for stores.

    Args:
        src: Source memory region (global or shared)
        dst: Destination memory region (shared or global)
        barrier: Mbarrier (from T.alloc_barrier()) for TMA load synchronization.
            Required for loads (global -> shared). Not needed for stores.
            The TMA load will arrive at this barrier with expected byte count.
            The user must wait on the same barrier via T.mbarrier_wait_parity().
        expect_transaction: Whether a load should emit ``mbarrier.expect_tx``
            immediately before the TMA instruction. Set this to ``False`` only
            when the caller has already armed the same barrier explicitly.
        leader_thread_extent: Logical thread-group size used to elect the lane
            that issues the TMA instruction. Defaults to the kernel's lowering
            thread extent. Use 32 for a dedicated single-warp TMA producer.
        eviction_policy: Cache eviction policy. Defaults to None.
        annotations: Additional annotations dict. Values in annotations take
            precedence over individual arguments.

    Returns:
        tir.Call: A handle to the tma_copy operation
    """
    # If both side are buffers, we should make sure their shapes are equal
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    assert src_extent or dst_extent, "Can't deduce copy extents from args. Both src and dst miss extents info."
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    src = to_buffer_region(src, access_type="r", extents=src_extent)
    dst = to_buffer_region(dst, access_type="w", extents=dst_extent)

    ann = annotations.copy() if annotations else {}

    if barrier is not None:
        from .builtin import _mbar_to_buffer_load

        ann["barrier"] = _mbar_to_buffer_load(barrier)

    if not expect_transaction and "skip_expect_transaction" not in ann:
        ann["skip_expect_transaction"] = 1

    if leader_thread_extent is not None and "leader_thread_extent" not in ann:
        if not isinstance(leader_thread_extent, int):
            raise TypeError(f"leader_thread_extent must be an int or None, got {type(leader_thread_extent).__name__}")
        if leader_thread_extent != 0 and (leader_thread_extent < 32 or leader_thread_extent % 32 != 0):
            raise ValueError(f"leader_thread_extent must be 0 or a positive multiple of 32, got {leader_thread_extent}")
        ann["leader_thread_extent"] = leader_thread_extent

    if "eviction_policy" not in ann and eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]

    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.tma_copy"), src, dst, annotations=ann if ann else None)


def transpose(
    src: BufferLikeType,
    dst: BufferLikeType,
) -> tir.PrimExpr:
    """Transpose a 2D buffer in shared memory: dst[j, i] = src[i, j].

    Both src and dst should be shared memory buffers.
    If src has shape (M, N), dst should have shape (N, M).

    Args:
        src: Source buffer or region of shape (..., M, N).
        dst: Destination buffer or region of shape (..., N, M).

    Returns:
        tir.Call: A handle to the transpose operation.
    """
    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    assert src_extent is not None, "Cannot deduce extent for transpose src."
    assert dst_extent is not None, "Cannot deduce extent for transpose dst."
    assert len(src_extent) >= 2, "Transpose requires at least 2D buffers."
    assert len(dst_extent) >= 2, "Transpose requires at least 2D buffers."

    src = to_buffer_region(src, access_type="r")
    dst = to_buffer_region(dst, access_type="w")

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.transpose"),
        src,
        dst,
    )


def c2d_im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tir.PrimExpr,
    c_step: tir.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
) -> tir.PrimExpr:
    """Perform im2col transformation for 2D convolution.

    Args:
        img (tir.Buffer): Input image buffer
        col (tir.Buffer): Output column buffer
        nhw_step (tir.PrimExpr): Step size for batch and spatial dimensions
        c_step (tir.PrimExpr): Step size for channel dimension
        kernel (int): Kernel size
        stride (int): Stride of the convolution
        dilation (int): Dilation rate
        pad (int): Padding size

    Returns:
        tir.Call: A handle to the im2col operation
    """
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    img_region = to_buffer_region(img, access_type="r")
    col_region = to_buffer_region(col, access_type="w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.c2d_im2col"),
        img_region,
        col_region,
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
        eviction_policy,
    )
