from __future__ import annotations

import random

import pytest

import tilelang.language as T
import tilelang.dataflow as df


@T.dataflow_intermediate
class BarrierValue:
    value: T.uint32


def instruction(
    instruction_id: int,
    sm_id: int,
    *,
    input_slots: tuple[int, ...] = (),
    output_slot: int | None = None,
) -> df.Instruction:
    return df.Instruction(
        instruction_id=instruction_id,
        opcode=df.DataflowOpcode.MAP,
        operator_name=f"barrier_stage_{instruction_id}",
        task_id=0,
        sm_id=sm_id,
        input_slots=input_slots,
        output_slot=output_slot,
    )


def slot(slot_id: int, *, producer_instruction_id: int | None) -> df.SlotPlan:
    return df.SlotPlan(
        slot_id=slot_id,
        task_id=0,
        intermediate_type=df.get_intermediate_type(BarrierValue),
        role="barrier_test",
        producer_instruction_id=producer_instruction_id,
    )


def cluster_transfer(
    source_instruction: df.Instruction,
    target_instruction: df.Instruction,
    source_slot_id: int,
    target_slot_id: int,
) -> tuple[df.CommPlan, df.CommPlan]:
    assert source_instruction.sm_id is not None
    assert target_instruction.sm_id is not None
    common = {
        "source_instruction_id": source_instruction.instruction_id,
        "target_instruction_id": target_instruction.instruction_id,
        "source_slot_id": source_slot_id,
        "target_slot_id": target_slot_id,
        "producer_sm": source_instruction.sm_id,
        "consumer_sm": target_instruction.sm_id,
    }
    return (
        df.CommPlan(
            **common,
            kind=df.DataflowCommKind.CLUSTER_SEND,
            dispatch_instruction_id=source_instruction.instruction_id,
            peer_cta_rank=target_instruction.sm_id,
        ),
        df.CommPlan(
            **common,
            kind=df.DataflowCommKind.CLUSTER_RECV,
            dispatch_instruction_id=target_instruction.instruction_id,
        ),
    )


def plan(
    instructions: tuple[df.Instruction, ...],
    queues: dict[int, tuple[df.Instruction, ...]],
    slots: tuple[df.SlotPlan, ...],
    comms: tuple[df.CommPlan, ...],
) -> df.InstructionPlan:
    return df.InstructionPlan(
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        block_size=32,
        range_axis=None,
        scheduler_policy="barrier_live_range_test",
        reduce_strategy="tree",
        task_extents=(1,),
        task_range_lengths=(1,),
        instructions=instructions,
        queues=queues,
        slots=slots,
        comms=comms,
    )


def overlapping_receive_plan() -> df.InstructionPlan:
    source_a = instruction(0, 1, output_slot=0)
    source_b = instruction(1, 1, output_slot=2)
    target = instruction(2, 0, input_slots=(1, 3))
    comms = (
        *cluster_transfer(source_a, target, 0, 1),
        *cluster_transfer(source_b, target, 2, 3),
    )
    return plan(
        (source_a, source_b, target),
        {0: (target,), 1: (source_a, source_b)},
        (
            slot(0, producer_instruction_id=source_a.instruction_id),
            slot(1, producer_instruction_id=None),
            slot(2, producer_instruction_id=source_b.instruction_id),
            slot(3, producer_instruction_id=None),
        ),
        comms,
    )


def phase_reuse_plan() -> df.InstructionPlan:
    source_a = instruction(0, 1, output_slot=0)
    target_a_source_c = instruction(1, 0, input_slots=(1,), output_slot=2)
    target_c_source_b = instruction(2, 1, input_slots=(3,), output_slot=4)
    target_b = instruction(3, 0, input_slots=(5,))
    comms = (
        *cluster_transfer(source_a, target_a_source_c, 0, 1),
        *cluster_transfer(target_a_source_c, target_c_source_b, 2, 3),
        *cluster_transfer(target_c_source_b, target_b, 4, 5),
    )
    return plan(
        (source_a, target_a_source_c, target_c_source_b, target_b),
        {0: (target_a_source_c, target_b), 1: (source_a, target_c_source_b)},
        (
            slot(0, producer_instruction_id=source_a.instruction_id),
            slot(1, producer_instruction_id=None),
            slot(2, producer_instruction_id=target_a_source_c.instruction_id),
            slot(3, producer_instruction_id=None),
            slot(4, producer_instruction_id=target_c_source_b.instruction_id),
            slot(5, producer_instruction_id=None),
        ),
        comms,
    )


def test_overlapping_recv_live_ranges_receive_distinct_barrier_colors():
    plan = overlapping_receive_plan()
    allocation = df.plan_dataflow_barriers(plan).require_valid()
    first = allocation.assignment_for_slot(1)
    second = allocation.assignment_for_slot(3)

    assert first.consumer_sm == second.consumer_sm
    assert first.barrier_index != second.barrier_index
    assert (min(first.live_range_id, second.live_range_id), max(first.live_range_id, second.live_range_id)) in (
        allocation.interference_graph.edges
    )

    packed = df.pack_instruction_plan(plan)
    assert packed.slots[1].barrier_index == first.barrier_index
    assert packed.slots[3].barrier_index == second.barrier_index
    assert df.build_launch_package(packed).barrier_count == allocation.barrier_count


def test_ordered_recv_live_ranges_reuse_barrier_with_alternating_phase():
    plan = phase_reuse_plan()
    allocation = df.plan_dataflow_barriers(plan).require_valid()
    first = allocation.assignment_for_slot(1)
    second = allocation.assignment_for_slot(5)

    assert first.consumer_sm == second.consumer_sm
    assert first.barrier_index == second.barrier_index
    assert first.uses[0].phase != second.uses[0].phase
    assert (min(first.live_range_id, second.live_range_id), max(first.live_range_id, second.live_range_id)) not in (
        allocation.interference_graph.edges
    )

    packed = df.pack_instruction_plan(plan)
    recv_phase_by_slot = {comm.dst_slot_id: comm.flag_epoch for comm in packed.comms if comm.kind == 2}
    assert recv_phase_by_slot[1] == first.uses[0].phase
    assert recv_phase_by_slot[5] == second.uses[0].phase
    assert df.build_launch_package(packed).barrier_count == allocation.barrier_count


def test_random_barrier_interference_graph_coloring_is_always_legal():
    rng = random.Random(0)
    for node_count in range(32):
        node_ids = tuple(range(node_count))
        edges = tuple((first, second) for first in node_ids for second in node_ids[first + 1 :] if rng.random() < 0.25)
        graph = df.DataflowBarrierInterferenceGraph(node_ids=node_ids, edges=edges)
        colors = graph.color()

        graph.require_legal_coloring(colors)
        assert set(colors) == set(node_ids)
        assert all(colors[first] != colors[second] for first, second in graph.edges)
        if colors:
            assert set(colors.values()) == set(range(max(colors.values()) + 1))


def test_barrier_planner_fails_closed_on_unpaired_cluster_transfer():
    plan = overlapping_receive_plan()
    unpaired = df.InstructionPlan(
        topology=plan.topology,
        block_size=plan.block_size,
        range_axis=plan.range_axis,
        scheduler_policy=plan.scheduler_policy,
        reduce_strategy=plan.reduce_strategy,
        task_extents=plan.task_extents,
        task_range_lengths=plan.task_range_lengths,
        instructions=plan.instructions,
        queues=plan.queues,
        slots=plan.slots,
        comms=tuple(comm for comm in plan.comms if comm.kind is not df.DataflowCommKind.CLUSTER_RECV),
    )

    with pytest.raises(df.DataflowBarrierPlanningError, match="no matching receive"):
        df.plan_dataflow_barriers(unpaired)
