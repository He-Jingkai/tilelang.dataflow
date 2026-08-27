from __future__ import annotations

import pytest

import tilelang.dataflow as df
import tilelang.language as T
import tilelang.language.reduce_op as reduce_ops


BLOCK_H = 2
HEAD_DIM = 8


@T.dataflow_intermediate
class AttnInter:
    lse: T.float32
    o: T.Tensor((BLOCK_H, HEAD_DIM), T.float16)


@T.dataflow_intermediate
class OtherInter:
    value: T.float32


def test_dataflow_intermediate_schema():
    intermediate = df.get_intermediate_type(AttnInter)

    assert intermediate is not None
    assert intermediate.name == "AttnInter"
    assert [field.name for field in intermediate.fields] == ["lse", "o"]

    lse = intermediate.field("lse")
    assert lse.dtype == "float32"
    assert lse.shape is None

    o = intermediate.field("o")
    assert o.dtype == "float16"
    assert o.shape == (BLOCK_H, HEAD_DIM)
    assert o.scope == "global"


def test_dataflow_operator_decorators_and_calls_do_not_execute_body():
    @T.dataflow.iter(range=("kv_begin", "kv_end"))
    def split_kv(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
        raise AssertionError("Dataflow operator declaration body should not execute on call")

    @T.dataflow.reduce
    def combine(items: list[AttnInter]) -> AttnInter:
        raise AssertionError("Dataflow reduce body should not execute on call")

    @T.dataflow.finalize
    def finalize(inter: AttnInter, seq: T.int32, head: T.int32, Output) -> None:
        raise AssertionError("Dataflow finalize body should not execute on call")

    assert split_kv.kind is df.DataflowOperatorKind.ITER
    assert split_kv.name == "split_kv"
    assert split_kv.range_spec == ("kv_begin", "kv_end")
    assert split_kv.output_type is df.get_intermediate_type(AttnInter)

    assert combine.kind is df.DataflowOperatorKind.REDUCE
    assert combine.input_types == (df.get_intermediate_type(AttnInter),)
    assert combine.output_type is df.get_intermediate_type(AttnInter)

    assert finalize.kind is df.DataflowOperatorKind.FINALIZE
    assert finalize.input_types == (df.get_intermediate_type(AttnInter),)
    assert finalize.output_type is None

    split_call = split_kv(Q="Q", K="K", V="V")
    assert split_call.kind is df.DataflowOperatorKind.ITER
    assert split_call.name == "split_kv"
    assert split_call.bound_arguments == {"Q": "Q", "K": "K", "V": "V"}

    automatic_split_call = split_kv()
    assert automatic_split_call.bound_arguments == {
        "Q": "Q",
        "K": "K",
        "V": "V",
    }

    renamed_split_call = split_kv(Q="query")
    assert renamed_split_call.bound_arguments == {
        "Q": "query",
        "K": "K",
        "V": "V",
    }

    reduce_call = combine()
    assert reduce_call.kind is df.DataflowOperatorKind.REDUCE

    finalize_call = finalize(Output="O")
    assert finalize_call.kind is df.DataflowOperatorKind.FINALIZE
    assert finalize_call.bound_arguments == {"Output": "O"}
    assert finalize().bound_arguments == {"Output": "Output"}


def test_dataflow_reduce_accepts_binary_signature():
    @T.dataflow.reduce(associative=True)
    def combine(lhs: AttnInter, rhs: AttnInter) -> AttnInter:
        raise AssertionError("Dataflow reduce body should not execute on call")

    assert combine.input_types == (
        df.get_intermediate_type(AttnInter),
        df.get_intermediate_type(AttnInter),
    )
    assert combine.reducer_contract is df.DataflowReducerContract.ASSOCIATIVE_BINARY


def test_dataflow_operator_namespace_keeps_legacy_aliases_and_local_annotations():
    assert T.dataflow.finalize is T.dataflow_finalize
    assert T.dataflow.iter is T.dataflow_iter
    assert T.dataflow.map is T.dataflow_map
    assert T.dataflow.reduce is T.dataflow_reduce

    @T.dataflow_intermediate
    class LocalInter:
        value: T.float32

    @T.dataflow.reduce
    def short_reduce(items: list[LocalInter]) -> LocalInter:
        raise AssertionError("Dataflow reduce body should not execute on call")

    @T.dataflow_reduce
    def legacy_reduce(items: list[LocalInter]) -> LocalInter:
        raise AssertionError("Dataflow reduce body should not execute on call")

    local_type = df.get_intermediate_type(LocalInter)
    assert short_reduce.input_types == (local_type,)
    assert legacy_reduce.input_types == (local_type,)


def test_dataflow_reduce_namespace_does_not_shadow_buffer_reduce():
    assert T.reduce is reduce_ops.reduce
    assert T.reduce is not T.dataflow.reduce


def test_dataflow_reduce_rejects_implicit_binary_contract():
    with pytest.raises(TypeError, match="associative=True"):

        @T.dataflow.reduce
        def combine(lhs: AttnInter, rhs: AttnInter) -> AttnInter:
            raise AssertionError("Dataflow reduce body should not execute on call")


def test_dataflow_frontend_rejects_invalid_annotations():
    with pytest.raises(TypeError, match="return"):

        @T.dataflow.iter(range=(0, 1))
        def bad_iter() -> int:
            return 0

    with pytest.raises(TypeError, match="consume at least one"):

        @T.dataflow.reduce
        def bad_reduce(value: int) -> AttnInter:
            raise AssertionError

    with pytest.raises(TypeError, match="same intermediate type"):

        @T.dataflow.reduce(associative=True)
        def mismatched_reduce(lhs: AttnInter, rhs: OtherInter) -> AttnInter:
            raise AssertionError

    with pytest.raises(TypeError, match="must return None"):

        @T.dataflow.finalize
        def bad_finalize(inter: AttnInter) -> AttnInter:
            raise AssertionError
