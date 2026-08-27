"""SVG visualization for packed Dataflow runtime plans."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Sequence

from .abi_schema import UINT32_SENTINEL


OPCODE_NAMES = {
    0: "Exit",
    1: "Iter",
    2: "Red",
    3: "Final",
}

COMM_KIND_NAMES = {
    1: "cluster",
    2: "cluster",
    3: "hbm",
    4: "hbm",
}


@dataclass(frozen=True)
class PackedPlanNode:
    node_id: str
    opcode: int
    cta: int
    pc: int
    task_id: int
    output_slot: int | None
    input_slots: tuple[int, ...]
    range_begin: int | None
    range_end: int | None
    task_coords: tuple[int, ...]
    comm_count: int
    x: int = 0
    y: int = 0

    @property
    def kind(self) -> str:
        return OPCODE_NAMES.get(self.opcode, f"Op{self.opcode}")

    @property
    def slot_id(self) -> int | None:
        return self.output_slot


@dataclass(frozen=True)
class PackedPlanEdge:
    source: PackedPlanNode
    target: PackedPlanNode
    slot_id: int
    comm_kind: str | None

    @property
    def is_cross_sm(self) -> bool:
        return self.source.cta != self.target.cta


@dataclass(frozen=True)
class PackedPlanGraph:
    title: str
    nodes: tuple[PackedPlanNode, ...]
    edges: tuple[PackedPlanEdge, ...]
    ctas: tuple[int, ...]


def load_packed_plan_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_packed_plan_graph(
    plan_json: dict[str, Any],
    *,
    task_id: int | None = None,
    include_exit: bool = False,
    max_per_sm: int | None = None,
    sm_start: int = 0,
    sm_count: int | None = None,
) -> PackedPlanGraph:
    """Build an instruction/slot dependency graph from a packed plan JSON dict."""

    packed = packed_plan(plan_json)
    instructions = packed.get("instructions", ())
    args = packed.get("args", ())
    input_slots = packed.get("input_slots", ())
    task_coords = packed.get("task_coords", ())
    arg_record_size = int(packed.get("arg_record_size", 32))
    queue_offsets = tuple(int(v) for v in packed.get("queue_offsets", (0,)))
    queue_lengths = tuple(int(v) for v in packed.get("queue_lengths", (len(instructions),)))
    comm_kinds_by_slot = collect_comm_kinds_by_slot(packed.get("comms", ()))

    visible_nodes: list[PackedPlanNode] = []
    visual_index_by_cta: dict[int, int] = {}
    raw_nodes: list[PackedPlanNode] = []
    for cta, offset in enumerate(queue_offsets):
        if cta < sm_start:
            continue
        if sm_count is not None and cta >= sm_start + sm_count:
            continue
        queue_length = queue_lengths[cta] if cta < len(queue_lengths) else 0
        for pc in range(queue_length):
            instruction_index = offset + pc
            if instruction_index >= len(instructions):
                continue
            inst = instructions[instruction_index]
            opcode = int(inst[0])
            inst_task_id = int(inst[2])
            if opcode == 0 and not include_exit:
                continue
            if task_id is not None and inst_task_id != task_id:
                continue
            if max_per_sm is not None and visual_index_by_cta.get(cta, 0) >= max_per_sm:
                continue

            arg = arg_for_instruction(inst, args, arg_record_size)
            node_input_slots = input_slots_for_arg(arg, input_slots)
            node_task_coords = task_coords_for_arg(arg, task_coords)
            output_slot = output_slot_for_instruction(inst, arg)
            range_begin, range_end = range_for_arg(arg)
            visual_index = visual_index_by_cta.get(cta, 0)
            visual_index_by_cta[cta] = visual_index + 1
            node = PackedPlanNode(
                node_id=f"n{cta}_{pc}_{len(raw_nodes)}",
                opcode=opcode,
                cta=cta,
                pc=pc,
                task_id=inst_task_id,
                output_slot=output_slot,
                input_slots=node_input_slots,
                range_begin=range_begin,
                range_end=range_end,
                task_coords=node_task_coords,
                comm_count=int(inst[5]),
                x=96 + visual_index * 150,
                y=92 + len([row for row in visual_index_by_cta if row < cta]) * 68,
            )
            raw_nodes.append(node)
            visible_nodes.append(node)

    row_index = {cta: i for i, cta in enumerate(sorted({node.cta for node in visible_nodes}))}
    nodes = tuple(
        PackedPlanNode(
            node.node_id,
            node.opcode,
            node.cta,
            node.pc,
            node.task_id,
            node.output_slot,
            node.input_slots,
            node.range_begin,
            node.range_end,
            node.task_coords,
            node.comm_count,
            node.x,
            92 + row_index[node.cta] * 68,
        )
        for node in visible_nodes
    )
    producers = {node.output_slot: node for node in nodes if node.output_slot is not None}
    edges: list[PackedPlanEdge] = []
    for node in nodes:
        for slot_id in node.input_slots:
            producer = producers.get(slot_id)
            if producer is None:
                continue
            edges.append(
                PackedPlanEdge(
                    source=producer,
                    target=node,
                    slot_id=slot_id,
                    comm_kind=preferred_comm_kind(comm_kinds_by_slot.get(slot_id, ())),
                )
            )

    task_suffix = "" if task_id is None else f" task {task_id}"
    limit_suffix = "" if max_per_sm is None else f" first {max_per_sm} per SM"
    sm_suffix = "" if sm_count is None else f" SM{sm_start}-{sm_start + sm_count - 1}"
    return PackedPlanGraph(
        title=f"Dataflow packed plan{task_suffix}{limit_suffix}{sm_suffix}",
        nodes=nodes,
        edges=tuple(edges),
        ctas=tuple(sorted(row_index)),
    )


def render_packed_plan_svg(graph: PackedPlanGraph) -> str:
    """Render a packed plan graph as a standalone SVG string."""

    if not graph.nodes:
        width = 640
        height = 180
    else:
        width = max(node.x for node in graph.nodes) + 180
        height = max(node.y for node in graph.nodes) + 96
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/>',
        "</marker>",
        '<marker id="arrow-cross" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#d97706"/>',
        "</marker>",
        "<style>",
        ".bg{fill:#f8fafc}.title{font:600 18px ui-sans-serif,system-ui,sans-serif;fill:#0f172a}",
        ".row-label{font:600 13px ui-sans-serif,system-ui,sans-serif;fill:#334155}",
        ".node rect{stroke:#0f172a;stroke-width:2}.node text{font:600 13px ui-sans-serif,system-ui,sans-serif;fill:#0f172a;text-anchor:middle}",
        ".node .small{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#334155}",
        ".iter rect{fill:#bff3fb}.red rect{fill:#c7e8ff}.final rect{fill:#d7f9d0}.exit rect{fill:#e2e8f0}",
        ".edge{fill:none;stroke:#475569;stroke-width:1.7;marker-end:url(#arrow)}",
        ".edge.cross{stroke:#d97706;stroke-width:2.2;stroke-dasharray:6 4;marker-end:url(#arrow-cross)}",
        ".edge-label{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#475569;text-anchor:middle}",
        ".edge-label.cross{fill:#b45309;font-weight:700}",
        "</style>",
        "</defs>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
        f'<text class="title" x="24" y="34">{escape(graph.title)}</text>',
    ]

    if not graph.nodes:
        lines.append('<text class="row-label" x="24" y="86">No visible instructions</text>')
        lines.append("</svg>")
        return "\n".join(lines)

    for cta in graph.ctas:
        y = 116 + graph.ctas.index(cta) * 68
        lines.append(f'<text class="row-label" x="24" y="{y}">SM{cta}</text>')

    for edge in graph.edges:
        lines.extend(render_edge(edge))
    for node in graph.nodes:
        lines.extend(render_node(node))

    lines.append("</svg>")
    return "\n".join(lines)


def write_packed_plan_svg(
    plan_path: str | Path,
    output_path: str | Path | None = None,
    *,
    task_id: int | None = None,
    include_exit: bool = False,
    max_per_sm: int | None = None,
    sm_start: int = 0,
    sm_count: int | None = None,
) -> Path:
    plan_path = Path(plan_path)
    if output_path is None:
        suffix = "" if task_id is None else f".task{task_id}"
        output_path = plan_path.with_suffix(f"{suffix}.svg")
    output = Path(output_path)
    graph = build_packed_plan_graph(
        load_packed_plan_json(plan_path),
        task_id=task_id,
        include_exit=include_exit,
        max_per_sm=max_per_sm,
        sm_start=sm_start,
        sm_count=sm_count,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_packed_plan_svg(graph), encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render packed Dataflow plan dependencies as SVG.")
    parser.add_argument("plan_json", type=Path, help="Path to a Dataflow plan JSON file.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output SVG path.")
    parser.add_argument("--task-id", type=int, default=None, help="Only render one task_id.")
    parser.add_argument("--max-per-sm", type=int, default=None, help="Only render the first N visible queue entries per SM.")
    parser.add_argument("--sm-start", type=int, default=0, help="First SM/CTA row to render.")
    parser.add_argument("--sm-count", type=int, default=None, help="Number of SM/CTA rows to render.")
    parser.add_argument("--include-exit", action="store_true", help="Render exit instructions too.")
    args = parser.parse_args(argv)

    output = write_packed_plan_svg(
        args.plan_json,
        args.output,
        task_id=args.task_id,
        include_exit=args.include_exit,
        max_per_sm=args.max_per_sm,
        sm_start=args.sm_start,
        sm_count=args.sm_count,
    )
    print(output)
    return 0


def packed_plan(plan_json: dict[str, Any]) -> dict[str, Any]:
    packed = plan_json.get("packed_plan", plan_json)
    if not isinstance(packed, dict):
        raise TypeError("Dataflow plan JSON must contain a packed_plan object or be a packed plan object")
    return packed


def arg_for_instruction(inst: Sequence[int], args: Sequence[Sequence[int]], arg_record_size: int) -> Sequence[int] | None:
    arg_offset = int(inst[3])
    if arg_offset == UINT32_SENTINEL:
        return None
    index = arg_offset // arg_record_size
    if index < 0 or index >= len(args):
        return None
    return args[index]


def input_slots_for_arg(arg: Sequence[int] | None, input_slots: Sequence[int]) -> tuple[int, ...]:
    if arg is None:
        return ()
    offset = int(arg[5])
    count = int(arg[6])
    if count == 0 or offset == UINT32_SENTINEL:
        return ()
    return tuple(int(slot) for slot in input_slots[offset : offset + count])


def task_coords_for_arg(arg: Sequence[int] | None, task_coords: Sequence[int]) -> tuple[int, ...]:
    if arg is None:
        return ()
    offset = int(arg[1])
    count = int(arg[2])
    if count == 0 or offset == UINT32_SENTINEL:
        return ()
    return tuple(int(coord) for coord in task_coords[offset : offset + count])


def output_slot_for_instruction(inst: Sequence[int], arg: Sequence[int] | None) -> int | None:
    if arg is not None and int(arg[7]) != UINT32_SENTINEL:
        return int(arg[7])
    slot_id = int(inst[6])
    return None if slot_id == UINT32_SENTINEL else slot_id


def range_for_arg(arg: Sequence[int] | None) -> tuple[int | None, int | None]:
    if arg is None:
        return None, None
    begin = int(arg[3])
    end = int(arg[4])
    if begin == UINT32_SENTINEL or end == UINT32_SENTINEL:
        return None, None
    return begin, end


def collect_comm_kinds_by_slot(comms: Iterable[Sequence[int]]) -> dict[int, tuple[str, ...]]:
    grouped: dict[int, list[str]] = {}
    for comm in comms:
        if not comm:
            continue
        kind = COMM_KIND_NAMES.get(int(comm[0]))
        if kind is None:
            continue
        src_slot = int(comm[1])
        dst_slot = int(comm[2])
        grouped.setdefault(src_slot, []).append(kind)
        grouped.setdefault(dst_slot, []).append(kind)
    return {slot: tuple(kinds) for slot, kinds in grouped.items()}


def preferred_comm_kind(kinds: Sequence[str]) -> str | None:
    if "cluster" in kinds:
        return "cluster"
    if "hbm" in kinds:
        return "hbm"
    return None


def render_node(node: PackedPlanNode) -> list[str]:
    css = {
        1: "iter",
        2: "red",
        3: "final",
        0: "exit",
    }.get(node.opcode, "exit")
    x = node.x
    y = node.y
    label = f"{node.kind} q{node.pc}"
    detail = node_detail(node)
    return [
        f'<g id="{node.node_id}" class="node {css}">',
        f'<rect x="{x}" y="{y}" width="108" height="42" rx="7"/>',
        f'<text x="{x + 54}" y="{y + 17}">{escape(label)}</text>',
        f'<text class="small" x="{x + 54}" y="{y + 33}">{escape(detail)}</text>',
        "</g>",
    ]


def node_detail(node: PackedPlanNode) -> str:
    slot = "-" if node.output_slot is None else str(node.output_slot)
    if node.opcode == 1 and node.range_begin is not None and node.range_end is not None:
        return f"t{node.task_id} s{slot} [{node.range_begin},{node.range_end})"
    if node.input_slots:
        inputs = ",".join(str(slot_id) for slot_id in node.input_slots)
        return f"t{node.task_id} {inputs}->{slot}"
    return f"t{node.task_id} s{slot}"


def render_edge(edge: PackedPlanEdge) -> list[str]:
    sx = edge.source.x + 108
    sy = edge.source.y + 21
    tx = edge.target.x
    ty = edge.target.y + 21
    control = max(sx, tx) + 56 if tx <= sx else sx + max(36, (tx - sx) // 2)
    css = "edge cross" if edge.is_cross_sm else "edge"
    marker = "arrow-cross" if edge.is_cross_sm else "arrow"
    label_css = "edge-label cross" if edge.is_cross_sm else "edge-label"
    label = f"slot {edge.slot_id}"
    if edge.is_cross_sm and edge.comm_kind:
        label = f"{label} {edge.comm_kind}"
    lx = (sx + tx) // 2 if tx > sx else control
    ly = (sy + ty) // 2 - 7
    return [
        f'<path class="{css}" marker-end="url(#{marker})" d="M {sx} {sy} C {control} {sy}, {control} {ty}, {tx} {ty}"/>',
        f'<text class="{label_css}" x="{lx}" y="{ly}">{escape(label)}</text>',
    ]


if __name__ == "__main__":
    raise SystemExit(main())
