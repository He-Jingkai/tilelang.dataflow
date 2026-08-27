"""Human-readable schedule dumps for Dataflow instruction plans."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from collections.abc import Iterable

from .scheduler import CommPlan, Instruction, InstructionPlan, DataflowOpcode


def dump_schedule_visualization(
    plan: InstructionPlan,
    *,
    output_dir: str | Path = "schedule_res",
) -> tuple[Path, Path]:
    """Write text and Markdown views of a Dataflow schedule plan."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"schedule-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    text_path = directory / f"{stem}.txt"
    markdown_path = directory / f"{stem}.md"

    text = text_schedule(plan)
    markdown = markdown_schedule(plan, text)
    text_path.write_text(text, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return text_path, markdown_path


def text_schedule(plan: InstructionPlan) -> str:
    lines: list[str] = []
    lines.append("Dataflow Schedule")
    lines.append("=" * 14)
    lines.append("")
    lines.append("Topology:")
    lines.append(f"  sm_count: {plan.topology.sm_count}")
    lines.append(f"  cluster_size: {plan.topology.cluster_size}")
    lines.append(f"  cluster_count: {plan.topology.cluster_count}")
    for cluster_id in range(plan.topology.cluster_count):
        sm_list = ", ".join(f"SM{sm_id}" for sm_id in get_cluster_sms(plan, cluster_id))
        lines.append(f"  cluster {cluster_id}: {sm_list}")
    lines.append("")
    lines.append("Plan:")
    lines.append(f"  scheduler_policy: {plan.scheduler_policy}")
    lines.append(f"  reduce_strategy: {plan.reduce_strategy}")
    lines.append(f"  block_size: {plan.block_size}")
    lines.append(f"  range_axis: {plan.range_axis}")
    lines.append(f"  task_extents: {plan.task_extents}")
    lines.append(f"  task_range_lengths: {plan.task_range_lengths}")
    lines.append("")
    lines.append("Per-SM Queues:")
    comms_by_dispatch = group_comms_by_dispatch(plan.comms)
    for sm_id in range(plan.topology.sm_count):
        lines.append(f"  SM{sm_id} [cluster {plan.topology.cluster_id(sm_id)}]")
        queue = plan.queue(sm_id)
        if not queue:
            lines.append("    <empty>")
            continue
        for index, instruction in enumerate(queue):
            lines.append(f"    {index:02d}. {format_instruction(instruction)}")
            for comm in comms_by_dispatch.get(instruction.instruction_id, ()):
                lines.append(f"        {comm.dispatch_phase}: {format_comm(comm)}")
    lines.append("")
    lines.append("Slots:")
    for slot in plan.slots:
        lines.append(
            "  "
            f"slot {slot.slot_id}: task={slot.task_id} role={slot.role} "
            f"producer={slot.producer_instruction_id} "
            f"shared_storage={slot.shared_storage_id} global_storage={slot.global_storage_id}"
        )
    lines.append("")
    lines.append("Comms:")
    if not plan.comms:
        lines.append("  <none>")
    else:
        for comm in plan.comms:
            lines.append(f"  {format_comm(comm)} dispatch={comm.resolved_dispatch_instruction_id}")
    lines.append("")
    return "\n".join(lines)


def markdown_schedule(plan: InstructionPlan, text: str) -> str:
    return "\n".join(
        [
            "# Dataflow Schedule",
            "",
            "```text",
            text.rstrip(),
            "```",
            "",
            "```mermaid",
            mermaid_schedule(plan),
            "```",
            "",
        ]
    )


def mermaid_schedule(plan: InstructionPlan) -> str:
    lines = ["flowchart TD"]
    for cluster_id in range(plan.topology.cluster_count):
        lines.append(f'  subgraph C{cluster_id}["cluster {cluster_id}"]')
        for sm_id in get_cluster_sms(plan, cluster_id):
            lines.append(f'    SM{sm_id}["SM{sm_id} [cluster {cluster_id}]"]')
        lines.append("  end")

    for sm_id in range(plan.topology.sm_count):
        previous = f"SM{sm_id}"
        for index, instruction in enumerate(plan.queue(sm_id)):
            if instruction.opcode is DataflowOpcode.EXIT:
                continue
            node = f"SM{sm_id}_{index}"
            lines.append(f'  {node}["{mermaid_instruction_label(instruction)}"]')
            lines.append(f"  {previous} --> {node}")
            previous = node

    producer_nodes = {
        instruction.instruction_id: f"SM{instruction.sm_id}_{index}"
        for sm_id in range(plan.topology.sm_count)
        for index, instruction in enumerate(plan.queue(sm_id))
        if instruction.opcode is not DataflowOpcode.EXIT
    }
    for comm in plan.comms:
        source = producer_nodes.get(comm.source_instruction_id)
        target = producer_nodes.get(comm.target_instruction_id)
        if source is None or target is None:
            continue
        lines.append(f'  {source} -. "{comm.kind.value} slot {comm.source_slot_id}" .-> {target}')
    return "\n".join(lines)


def get_cluster_sms(plan: InstructionPlan, cluster_id: int) -> tuple[int, ...]:
    begin = cluster_id * plan.topology.cluster_size
    end = min(begin + plan.topology.cluster_size, plan.topology.sm_count)
    return tuple(range(begin, end))


def group_comms_by_dispatch(comms: Iterable[CommPlan]) -> dict[int, list[CommPlan]]:
    grouped: dict[int, list[CommPlan]] = {}
    for comm in comms:
        grouped.setdefault(comm.resolved_dispatch_instruction_id, []).append(comm)
    return grouped


def format_instruction(instruction: Instruction) -> str:
    coords = instruction.task_coords
    task = instruction.task_id
    if instruction.opcode is DataflowOpcode.ITER:
        task_range = instruction.task_range
        assert task_range is not None
        return f"ITER task={task} coords={coords} range=[{task_range.begin}, {task_range.end}) output={instruction.output_slot}"
    if instruction.opcode in (DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE):
        return (
            f"{instruction.opcode.value.upper()} task={task} coords={coords} "
            f"inputs={instruction.input_slots} output={instruction.output_slot}"
        )
    if instruction.opcode is DataflowOpcode.FINALIZE:
        return f"FINALIZE task={task} coords={coords} inputs={instruction.input_slots}"
    return instruction.opcode.value.upper()


def format_comm(comm: CommPlan) -> str:
    return (
        f"{comm.kind.value.upper()} slot {comm.source_slot_id} -> {comm.target_slot_id} "
        f"SM{comm.producer_sm}->SM{comm.consumer_sm} "
        f"phase={comm.barrier_phase} epoch={comm.flag_epoch}"
    )


def mermaid_instruction_label(instruction: Instruction) -> str:
    if instruction.opcode is DataflowOpcode.ITER:
        task_range = instruction.task_range
        assert task_range is not None
        return f"task{instruction.task_id} [{task_range.begin},{task_range.end})"
    if instruction.opcode in (DataflowOpcode.REDUCE, DataflowOpcode.REDUCE_UPDATE):
        return f"{instruction.opcode.value} task{instruction.task_id}"
    if instruction.opcode is DataflowOpcode.FINALIZE:
        return f"finalize task{instruction.task_id}"
    return instruction.opcode.value
