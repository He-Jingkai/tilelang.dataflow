from __future__ import annotations

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.compile_config import canonical_fingerprint


@T.dataflow_intermediate
class AttnInter:
    lse: T.float32
    o: T.Tensor((2, 8), T.float16)


@T.dataflow_intermediate
class OtherInter:
    value: T.float32


@T.dataflow.iter(range=("kv_begin", "kv_end"))
def split_kv(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
    raise AssertionError("Dataflow iter body should not execute during graph construction")


@T.dataflow.reduce
def combine(items: list[AttnInter]) -> AttnInter:
    raise AssertionError("Dataflow reduce body should not execute during graph construction")


@T.dataflow.reduce
def combine_other(items: list[OtherInter]) -> OtherInter:
    raise AssertionError("Dataflow reduce body should not execute during graph construction")


@T.dataflow.finalize
def finalize(inter: AttnInter, seq: T.int32, head: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during graph construction")


@T.dataflow.finalize
def finalize_other(inter: OtherInter, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during graph construction")


def test_dataflow_program_chains_iter_reduce_finalize():
    program = T.dataflow_program(
        task_domain=("batch", "head"),
        dynamic_ranges={"kv": "seq_lens"},
        name="attention",
    )

    graph = (
        program.partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
            range_axis="kv",
            placement="cluster",
        )
        .reduce(combine(), arity="tree")
        .finalize(finalize(Output="O"))
    )

    assert graph is program
    assert graph.validate() is graph
    assert graph.is_complete

    assert graph.task_domain.axes == ("batch", "head")
    assert graph.dynamic_ranges == {"kv": "seq_lens"}
    assert graph.attrs == {"name": "attention"}

    assert graph.partial_stage is not None
    assert graph.partial_stage.iter_call.name == "split_kv"
    assert graph.partial_stage.task_args == ("seq", "head")
    assert graph.partial_stage.range_axis == "kv"
    assert graph.partial_stage.attrs == {"placement": "cluster"}
    assert graph.partial_stage.output_type is df.get_intermediate_type(AttnInter)

    assert graph.reduce_stage is not None
    assert graph.reduce_stage.reduce_call.name == "combine"
    assert graph.reduce_stage.input_type is df.get_intermediate_type(AttnInter)
    assert graph.reduce_stage.output_type is df.get_intermediate_type(AttnInter)
    assert graph.reduce_stage.attrs == {"arity": "tree"}

    assert graph.finalize_stage is not None
    assert graph.finalize_stage.finalize_call.name == "finalize"
    assert graph.finalize_stage.input_type is df.get_intermediate_type(AttnInter)


def test_dataflow_program_default_map_stage_name_normalizes_dataflow_prefix():
    @T.dataflow_intermediate
    class Value:
        value: T.float32

    @T.dataflow.map
    def dataflow_transform(Input) -> Value:
        raise AssertionError("Dataflow map body should not execute during graph construction")

    @T.dataflow.map
    def plain_transform(Input) -> Value:
        raise AssertionError("Dataflow map body should not execute during graph construction")

    prefixed = T.dataflow_program(task_domain=("task",)).map(dataflow_transform())
    explicit = T.dataflow_program(task_domain=("task",)).map(
        dataflow_transform(),
        name="transform",
    )
    plain = T.dataflow_program(task_domain=("task",)).map(plain_transform())

    assert prefixed.stages[0].name == "transform"
    assert prefixed.stages[0].call.name == "dataflow_transform"
    assert canonical_fingerprint(prefixed) == canonical_fingerprint(explicit)
    assert plain.stages[0].name == "plain_transform"


def test_dataflow_program_builds_map_reshared_map_finalize_graph():
    @T.dataflow_intermediate
    class UpShard:
        value: T.Tensor((2, 4), T.float16)

    @T.dataflow_intermediate
    class UpFull:
        value: T.Tensor((2, 8), T.float16)

    @T.dataflow_intermediate
    class HiddenShard:
        value: T.Tensor((2, 4), T.float16)

    @T.dataflow.map(range=("expert_begin", "expert_end"))
    def moe_map1(expert: T.int32, token: T.int32, Input, W1) -> UpShard:
        raise AssertionError("Dataflow map body should not execute during graph construction")

    @T.dataflow.map(range=("hidden_begin", "hidden_end"))
    def moe_map2(parts: list[UpShard], expert: T.int32, token: T.int32, W2) -> HiddenShard:
        raise AssertionError("Dataflow map body should not execute during graph construction")

    @T.dataflow.finalize
    def finalize_hidden(hidden: HiddenShard, expert: T.int32, token: T.int32, Output) -> None:
        raise AssertionError("Dataflow finalize body should not execute during graph construction")

    program = T.dataflow_program(
        task_domain=("expert", "token"),
        dynamic_ranges={"expert_tile": "expert_tiles", "hidden_tile": "hidden_tiles"},
        name="moe",
    )

    graph = (
        program.map(
            moe_map1(Input="Input", W1="W1"),
            name="map1",
            task_args=("expert", "token"),
            range_axis="expert_tile",
            placement="cluster",
        )
        .reshared(
            input="map1",
            name="gather_up",
            output_type=UpFull,
            physical_output_type=UpShard,
            output_arity=2,
            policy="hbm_all_gather",
        )
        .map(
            moe_map2(W2="W2"),
            name="map2",
            input="gather_up",
            task_args=("expert", "token"),
            range_axis="hidden_tile",
            placement="cluster",
        )
        .finalize(finalize_hidden(Output="Output"), input="map2")
    )

    assert graph is program
    assert graph.validate() is graph
    assert graph.is_complete
    assert graph.is_stage_graph

    assert [stage.stage_id for stage in graph.stages] == [0, 1, 2, 3]
    assert [stage.name for stage in graph.stages] == ["map1", "gather_up", "map2", "finalize"]
    assert [stage.kind for stage in graph.stages] == [
        df.DataflowStageKind.MAP,
        df.DataflowStageKind.RESHARED,
        df.DataflowStageKind.MAP,
        df.DataflowStageKind.FINALIZE,
    ]

    map1_stage = graph.stage("map1")
    gather_stage = graph.stage("gather_up")
    map2_stage = graph.stage("map2")
    finalize_stage = graph.stage("finalize")

    assert map1_stage.call is not None
    assert map1_stage.call.name == "moe_map1"
    assert map1_stage.output_type is df.get_intermediate_type(UpShard)
    assert map1_stage.physical_output_type is df.get_intermediate_type(UpShard)
    assert map1_stage.output_arity == 1
    assert map1_stage.task_args == ("expert", "token")
    assert map1_stage.range_axis == "expert_tile"
    assert map1_stage.attrs == {"placement": "cluster"}

    assert gather_stage.call is None
    assert gather_stage.deps == (map1_stage.stage_id,)
    assert gather_stage.input_type is df.get_intermediate_type(UpShard)
    assert gather_stage.output_type is df.get_intermediate_type(UpFull)
    assert gather_stage.physical_output_type is df.get_intermediate_type(UpShard)
    assert gather_stage.output_arity == 2
    assert gather_stage.attrs == {"policy": "hbm_all_gather"}

    assert map2_stage.call is not None
    assert map2_stage.deps == (gather_stage.stage_id,)
    assert map2_stage.input_type is df.get_intermediate_type(UpShard)
    assert map2_stage.output_type is df.get_intermediate_type(HiddenShard)
    assert map2_stage.task_args == ("expert", "token")
    assert map2_stage.range_axis == "hidden_tile"

    assert finalize_stage.call is not None
    assert finalize_stage.deps == (map2_stage.stage_id,)
    assert finalize_stage.input_type is df.get_intermediate_type(HiddenShard)
    assert finalize_stage.output_type is None


def test_dataflow_program_builds_map_reshared_terminal_map_graph():
    @T.dataflow_intermediate
    class UpShard:
        value: T.Tensor((2, 4), T.float16)

    @T.dataflow_intermediate
    class UpFull:
        value: T.Tensor((2, 8), T.float16)

    @T.dataflow_intermediate
    class Done:
        value: T.Tensor((1,), T.int32)

    @T.dataflow.map(range=("expert_begin", "expert_end"))
    def moe_map1(Input, W1) -> UpShard:
        raise AssertionError("Dataflow map body should not execute during graph construction")

    @T.dataflow.map(range=("hidden_begin", "hidden_end"))
    def moe_map2(parts: list[UpShard], W2, Output) -> Done:
        raise AssertionError("Dataflow map body should not execute during graph construction")

    graph = (
        T.dataflow_program(
            task_domain=("expert",),
            dynamic_ranges={"expert_tile": "expert_tiles", "hidden_tile": "hidden_tiles"},
        )
        .map(
            moe_map1(Input="Input", W1="W1"),
            name="map1",
            task_args=("expert",),
            range_axis="expert_tile",
        )
        .reshared(
            input="map1",
            name="gather_up",
            output_type=UpFull,
            physical_output_type=UpShard,
            output_arity=2,
            policy="hbm_all_gather",
        )
        .map(
            moe_map2(W2="W2", Output="Output"),
            name="map2",
            input="gather_up",
            task_args=("expert",),
            range_axis="hidden_tile",
        )
    )

    assert graph.validate() is graph
    assert graph.is_complete
    assert graph.is_stage_graph
    assert graph.finalize_stage is None
    assert [stage.stage_id for stage in graph.stages] == [0, 1, 2]
    assert [stage.kind for stage in graph.stages] == [
        df.DataflowStageKind.MAP,
        df.DataflowStageKind.RESHARED,
        df.DataflowStageKind.MAP,
    ]
    assert graph.stage("map2").output_type is df.get_intermediate_type(Done)


def test_dataflow_program_rejects_invalid_reshared_stage_metadata():
    @T.dataflow_intermediate
    class UpShard:
        value: T.Tensor((2, 4), T.float16)

    @T.dataflow_intermediate
    class UpFull:
        value: T.Tensor((2, 8), T.float16)

    @T.dataflow.map
    def moe_map1(Input) -> UpShard:
        raise AssertionError("Dataflow map body should not execute during graph construction")

    program = T.dataflow_program(task_domain=("expert",))

    with pytest.raises(ValueError, match="requires an upstream stage"):
        program.reshared(input="missing", output_type=UpFull, physical_output_type=UpShard, output_arity=2)

    with pytest.raises(ValueError, match="output_arity"):
        T.dataflow_program(task_domain=("expert",)).map(
            moe_map1(Input="Input"),
            name="map1",
            task_args=("expert",),
        ).reshared(input="map1", output_type=UpFull, physical_output_type=UpShard, output_arity=0)


def test_dataflow_program_rejects_invalid_stage_order_and_kinds():
    with pytest.raises(ValueError, match="partial stage first"):
        T.dataflow_program(task_domain=("batch", "head")).reduce(combine())

    with pytest.raises(ValueError, match="reduce stage first"):
        T.dataflow_program(task_domain=("batch", "head")).partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
        ).finalize(finalize(Output="O"))

    with pytest.raises(TypeError, match="iter call"):
        T.dataflow_program(task_domain=("batch", "head")).partial(combine())

    with pytest.raises(TypeError, match="reduce call"):
        T.dataflow_program(task_domain=("batch", "head")).partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
        ).reduce(finalize(Output="O"))

    with pytest.raises(TypeError, match="finalize call"):
        T.dataflow_program(task_domain=("batch", "head")).partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
        ).reduce(combine()).finalize(combine())


def test_dataflow_program_rejects_type_and_rank_mismatches():
    with pytest.raises(ValueError, match="task_args rank"):
        T.dataflow_program(task_domain=("batch", "head")).partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq",),
        )

    with pytest.raises(TypeError, match="reduce intermediate type mismatch"):
        T.dataflow_program(task_domain=("batch", "head")).partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
        ).reduce(combine_other())

    with pytest.raises(TypeError, match="finalize intermediate type mismatch"):
        T.dataflow_program(task_domain=("batch", "head")).partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
        ).reduce(combine()).finalize(finalize_other(Output="O"))


def test_dataflow_program_rejects_duplicate_or_incomplete_graphs():
    with pytest.raises(ValueError, match="at least one axis"):
        T.dataflow_program(task_domain=())

    program = T.dataflow_program(task_domain=("batch", "head"))
    with pytest.raises(ValueError, match="missing a partial stage"):
        program.validate()

    program.partial(split_kv(Q="Q", K="K", V="V"), task_args=("seq", "head"))
    with pytest.raises(ValueError, match="already has a partial stage"):
        program.partial(split_kv(Q="Q", K="K", V="V"), task_args=("seq", "head"))

    with pytest.raises(ValueError, match="missing a reduce stage"):
        program.validate()

    program.reduce(combine())
    with pytest.raises(ValueError, match="already has a reduce stage"):
        program.reduce(combine())

    with pytest.raises(ValueError, match="missing a finalize stage"):
        program.validate()

    program.finalize(finalize(Output="O"))
    with pytest.raises(ValueError, match="already finalized"):
        program.reduce(combine())
