from __future__ import annotations

from pathlib import Path

from tilelang.dataflow.packed_plan_viz import (
    build_packed_plan_graph,
    render_packed_plan_svg,
    write_packed_plan_svg,
)


def tiny_packed_plan_dict() -> dict:
    return {
        "plan": {
            "scheduler_policy": "cluster_local",
            "reduce_strategy": "streaming",
        },
        "packed_plan": {
            "instruction_record_size": 32,
            "arg_record_size": 32,
            "queue_offsets": [0, 2],
            "queue_lengths": [2, 2],
            "input_slots": [0, 1, 2],
            "task_coords": [0, 0, 0, 0],
            "instructions": [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [2, 1, 0, 32, 0, 0, 1, 0],
                [1, 0, 0, 64, 0, 0, 2, 0],
                [2, 1, 0, 96, 0, 0, 3, 0],
            ],
            "args": [
                [0, 0, 1, 0, 64, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 1],
                [0, 1, 1, 64, 128, 0, 0, 2],
                [0, 1, 1, 0, 0, 1, 2, 3],
            ],
            "comms": [],
        },
    }


def test_packed_plan_graph_tracks_slot_dependencies_and_cross_sm_edges():
    graph = build_packed_plan_graph(tiny_packed_plan_dict(), task_id=0)

    edge_pairs = {(edge.source.slot_id, edge.target.slot_id, edge.slot_id) for edge in graph.edges}
    assert edge_pairs == {
        (0, 1, 0),
        (1, 3, 1),
        (2, 3, 2),
    }

    cross_edges = [edge for edge in graph.edges if edge.is_cross_sm]
    assert len(cross_edges) == 1
    assert cross_edges[0].source.slot_id == 1
    assert cross_edges[0].target.slot_id == 3


def test_packed_plan_svg_contains_iter_reduce_nodes_and_dependency_arrows():
    svg = render_packed_plan_svg(build_packed_plan_graph(tiny_packed_plan_dict(), task_id=0))

    assert "<svg" in svg
    assert "Iter" in svg
    assert "Red" in svg
    assert "q0" in svg
    assert "q1" in svg
    assert "slot 1" in svg
    assert 'class="edge cross"' in svg
    assert 'marker-end="url(#arrow-cross)"' in svg


def test_packed_plan_graph_can_limit_visible_queue_entries_per_sm():
    graph = build_packed_plan_graph(tiny_packed_plan_dict(), max_per_sm=1)

    assert [(node.cta, node.pc, node.kind) for node in graph.nodes] == [
        (0, 0, "Iter"),
        (1, 0, "Iter"),
    ]
    assert graph.edges == ()
    assert "first 1 per SM" in graph.title


def test_packed_plan_graph_can_filter_sm_window():
    graph = build_packed_plan_graph(tiny_packed_plan_dict(), sm_start=1, sm_count=1)

    assert graph.ctas == (1,)
    assert [(node.cta, node.pc, node.kind) for node in graph.nodes] == [
        (1, 0, "Iter"),
        (1, 1, "Red"),
    ]
    assert "SM1-1" in graph.title


def test_write_packed_plan_svg_reads_json_and_writes_output(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        __import__("json").dumps(tiny_packed_plan_dict()),
        encoding="utf-8",
    )
    output_path = tmp_path / "schedule.svg"

    written = write_packed_plan_svg(plan_path, output_path, task_id=0)

    assert written == output_path
    assert "Dataflow packed plan" in output_path.read_text(encoding="utf-8")
