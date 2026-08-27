import tilelang
import tilelang.testing
from tilelang import tvm


def make_duplicate_name_shared_allocations(*, reverse_creation):
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared.dyn")
    if reverse_creation:
        second_var = tvm.tir.Var("scratch", pointer_type)
        first_var = tvm.tir.Var("scratch", pointer_type)
    else:
        first_var = tvm.tir.Var("scratch", pointer_type)
        second_var = tvm.tir.Var("scratch", pointer_type)

    first = tvm.tir.decl_buffer((64,), "float32", data=first_var, name="first")
    second = tvm.tir.decl_buffer((64,), "float32", data=second_var, name="second")
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.BufferStore(first, tvm.tir.FloatImm("float32", 1), [0]),
            tvm.tir.BufferStore(second, tvm.tir.FloatImm("float32", 2), [0]),
            tvm.tir.Evaluate(tvm.tir.BufferLoad(first, [0]) + tvm.tir.BufferLoad(second, [0])),
        ]
    )
    for buffer_var in (second_var, first_var):
        body = tvm.tir.Allocate(
            buffer_var,
            "float32",
            [64],
            tvm.tir.const(True, "bool"),
            body,
        )

    thread_var = tvm.tir.Var("threadIdx.x", "int32")
    thread_axis = tvm.tir.IterVar(
        tvm.ir.Range(0, 32),
        thread_var,
        tvm.tir.IterVar.ThreadIndex,
        "threadIdx.x",
    )
    body = tvm.tir.AttrStmt(thread_axis, "thread_extent", 32, body)
    func = tvm.tir.PrimFunc([], body).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": func})


def merged_store_offsets(*, reverse_creation):
    module = tilelang.transform.MergeSharedMemoryAllocations()(make_duplicate_name_shared_allocations(reverse_creation=reverse_creation))
    offsets_by_value = {}

    def collect_store(node):
        if isinstance(node, tvm.tir.BufferStore):
            offsets_by_value[int(node.value.value)] = int(node.indices[0].value)

    tvm.tir.stmt_functor.post_order_visit(module["main"].body, collect_store)
    return offsets_by_value


def test_merge_shared_memory_uses_structural_order_for_duplicate_names():
    expected = {1: 0, 2: 64}

    assert merged_store_offsets(reverse_creation=False) == expected
    assert merged_store_offsets(reverse_creation=True) == expected


def test_merge_shared_memory_preserves_external_shared_buffer_base():
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared.dyn")
    external_var = tvm.tir.Var("external", pointer_type)
    first_var = tvm.tir.Var("first", pointer_type)
    second_var = tvm.tir.Var("second", pointer_type)
    external = tvm.tir.decl_buffer((64,), "float32", data=external_var, name="external")
    first = tvm.tir.decl_buffer((64,), "float32", data=first_var, name="first")
    second = tvm.tir.decl_buffer((64,), "float32", data=second_var, name="second")
    external_access = tvm.tir.Call(
        "handle",
        tvm.tir.op.Op.get("tir.tvm_access_ptr"),
        [
            tvm.tir.const(0, "uint64"),
            external_var,
            tvm.tir.const(3, "int32"),
            tvm.tir.const(8, "int32"),
            tvm.tir.const(1, "int32"),
        ],
    )
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.BufferStore(first, tvm.tir.FloatImm("float32", 1), [0]),
            tvm.tir.BufferStore(second, tvm.tir.FloatImm("float32", 2), [0]),
            tvm.tir.BufferStore(external, tvm.tir.BufferLoad(first, [0]), [5]),
            tvm.tir.Evaluate(external_access),
        ]
    )
    for buffer_var in (second_var, first_var):
        body = tvm.tir.Allocate(
            buffer_var,
            "float32",
            [64],
            tvm.tir.const(True, "bool"),
            body,
        )
    thread_var = tvm.tir.Var("threadIdx.x", "int32")
    thread_axis = tvm.tir.IterVar(
        tvm.ir.Range(0, 32),
        thread_var,
        tvm.tir.IterVar.ThreadIndex,
        "threadIdx.x",
    )
    body = tvm.tir.AttrStmt(thread_axis, "thread_extent", 32, body)
    module = tvm.IRModule(
        {
            "main": tvm.tir.PrimFunc(
                [external_var],
                body,
                buffer_map={external_var: external},
            ).with_attr("global_symbol", "main")
        }
    )

    merged = tilelang.transform.MergeSharedMemoryAllocations()(module)
    external_stores = []
    access_ptrs = []

    def collect(node):
        if isinstance(node, tvm.tir.BufferStore) and node.buffer.name == "external":
            external_stores.append(node)
        if isinstance(node, tvm.tir.Call) and node.op.same_as(tvm.tir.op.Op.get("tir.tvm_access_ptr")):
            access_ptrs.append(node)

    tvm.tir.stmt_functor.post_order_visit(merged["main"].body, collect)
    assert len(external_stores) == 1
    assert external_stores[0].buffer.data.same_as(external_var)
    assert int(external_stores[0].indices[0]) == 5
    assert len(access_ptrs) == 1
    assert access_ptrs[0].args[1].same_as(external_var)
    assert int(access_ptrs[0].args[2]) == 3


if __name__ == "__main__":
    tilelang.testing.main()
