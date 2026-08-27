from __future__ import annotations

from dataclasses import replace

import pytest

import tilelang.dataflow as df


def build_schedule(topology, nodes, **kwargs):
    return df.schedule_joint_compute_communication(
        topology,
        nodes,
        cluster_transfer_us=0.5,
        hbm_transfer_us=1.0,
        transfer_issue_us=kwargs.pop("transfer_issue_us", 0.0),
        **kwargs,
    )


def test_joint_schedule_pipelines_hbm_store_and_load_segments():
    topology = df.GPUTopology(sm_count=4, cluster_size=2)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(
            node_id=1,
            cta_id=2,
            duration_us=1.0,
            input_values=(10,),
        ),
    )

    schedule = build_schedule(
        topology,
        nodes,
        value_nbytes={10: 1024},
        hbm_segment_bytes=256,
    )

    transfer = schedule.transfers[0]
    assert transfer.kind is df.DataflowTransportKind.HBM_STAGED
    assert len(transfer.segments) == 4
    for current, following in zip(transfer.segments, transfer.segments[1:]):
        current_load_issue = schedule.event(current.load_issue_event_id)
        current_load_ready = schedule.event(current.load_ready_event_id)
        following_store_issue = schedule.event(following.store_issue_event_id)
        following_store_ready = schedule.event(following.store_ready_event_id)
        assert following_store_issue.start_us < current_load_ready.end_us
        assert current_load_issue.start_us < following_store_ready.end_us
    assert schedule.require_valid(topology=topology, nodes=nodes) is schedule


def test_joint_schedule_retains_earlier_remote_inputs_instead_of_rejecting_them():
    topology = df.GPUTopology(sm_count=4, cluster_size=4)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(node_id=1, cta_id=1, duration_us=1.0, output_value=11),
        df.DataflowJointScheduleNode(
            node_id=2,
            cta_id=2,
            duration_us=1.0,
            input_values=(10, 11),
        ),
    )

    schedule = build_schedule(topology, nodes, retained_copy_us=0.2)

    copied, resident = schedule.transfers
    assert copied.retained_input_copy_event_id is not None
    assert resident.retained_input_copy_event_id is None
    assert copied.consumer_slot_epoch < resident.consumer_slot_epoch
    copied_interval = next(
        interval for interval in schedule.comm_slot_intervals if interval.cta_id == 2 and interval.epoch == copied.consumer_slot_epoch
    )
    resident_interval = next(
        interval for interval in schedule.comm_slot_intervals if interval.cta_id == 2 and interval.epoch == resident.consumer_slot_epoch
    )
    assert copied_interval.end_us <= resident_interval.begin_us
    assert any("remote inputs" in diagnostic for diagnostic in schedule.diagnostics)


def test_joint_schedule_keeps_two_remote_inputs_in_independent_inboxes():
    topology = df.GPUTopology(sm_count=4, cluster_size=4)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(node_id=1, cta_id=1, duration_us=1.0, output_value=11),
        df.DataflowJointScheduleNode(
            node_id=2,
            cta_id=2,
            duration_us=1.0,
            input_values=(10, 11),
            transient_prefetch_bytes=1024,
        ),
    )

    schedule = build_schedule(
        topology,
        nodes,
        retained_copy_us=0.2,
        value_nbytes={10: 1024, 11: 1024},
    )

    transient, permanent = schedule.transfers
    assert transient.retained_input_copy_event_id is None
    assert permanent.retained_input_copy_event_id is None
    assert transient.consumer_storage_kind is df.DataflowCommStorageKind.TRANSIENT_PREFETCH
    assert permanent.consumer_storage_kind is df.DataflowCommStorageKind.PERMANENT
    intervals = {
        interval.transfer_id: interval for interval in schedule.comm_slot_intervals if interval.mode is df.DataflowCommSlotMode.INBOX
    }
    assert intervals[transient.transfer_id].end_event_id == transient.consume_event_id
    assert intervals[permanent.transfer_id].end_event_id == permanent.consume_event_id
    assert max(
        intervals[transient.transfer_id].begin_us,
        intervals[permanent.transfer_id].begin_us,
    ) < min(
        intervals[transient.transfer_id].end_us,
        intervals[permanent.transfer_id].end_us,
    )
    assert schedule.require_valid(topology=topology, nodes=nodes) is schedule


def test_joint_schedule_keeps_fanout_source_live_for_every_remote_consumer():
    topology = df.GPUTopology(sm_count=4, cluster_size=4)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(
            node_id=1,
            cta_id=1,
            duration_us=1.0,
            input_values=(10,),
        ),
        df.DataflowJointScheduleNode(
            node_id=2,
            cta_id=2,
            duration_us=1.0,
            input_values=(10,),
        ),
    )

    schedule = build_schedule(topology, nodes, transfer_issue_us=0.05)

    first, second = schedule.transfers
    assert first.producer_slot_epoch == second.producer_slot_epoch
    assert first.producer_release_event_id == second.producer_release_event_id
    release = schedule.event(first.producer_release_event_id)
    assert all(release.end_us >= schedule.event(transfer.destination_ready_event_id).end_us for transfer in schedule.transfers)
    assert any("remote consumers" in diagnostic for diagnostic in schedule.diagnostics)


def test_joint_schedule_uses_ordinary_source_when_inbox_cannot_alias_output():
    topology = df.GPUTopology(sm_count=3, cluster_size=3)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(
            node_id=1,
            cta_id=1,
            duration_us=1.0,
            input_values=(10,),
            output_value=11,
        ),
        df.DataflowJointScheduleNode(
            node_id=2,
            cta_id=2,
            duration_us=1.0,
            input_values=(11,),
        ),
    )

    schedule = build_schedule(topology, nodes)

    outgoing = next(transfer for transfer in schedule.transfers if transfer.value_id == 11)
    assert outgoing.producer_slot_epoch is None
    assert any("cannot alias" in diagnostic for diagnostic in schedule.diagnostics)
    assert schedule.require_valid(topology=topology, nodes=nodes) is schedule


def test_joint_schedule_demotes_a_source_slot_instead_of_rejecting_capacity_cycle():
    topology = df.GPUTopology(sm_count=2, cluster_size=2)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(node_id=1, cta_id=1, duration_us=1.0, output_value=11),
        df.DataflowJointScheduleNode(
            node_id=2,
            cta_id=0,
            duration_us=1.0,
            input_values=(11,),
        ),
        df.DataflowJointScheduleNode(
            node_id=3,
            cta_id=1,
            duration_us=1.0,
            input_values=(10,),
        ),
    )

    schedule = build_schedule(topology, nodes, transfer_issue_us=0.05)

    assert sum(transfer.producer_slot_epoch is None for transfer in schedule.transfers) == 1
    assert any("capacity wait cycle" in diagnostic for diagnostic in schedule.diagnostics)
    assert schedule.require_valid(topology=topology, nodes=nodes) is schedule


def test_joint_schedule_reuses_one_arrival_for_duplicate_input_value():
    topology = df.GPUTopology(sm_count=2, cluster_size=2)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(
            node_id=1,
            cta_id=1,
            duration_us=1.0,
            input_values=(10, 10),
        ),
    )

    schedule = build_schedule(topology, nodes)

    assert len(schedule.transfers) == 1
    assert any("shares one physical arrival" in item for item in schedule.diagnostics)
    assert schedule.require_valid(topology=topology, nodes=nodes) is schedule


def test_joint_schedule_verifier_allows_non_contiguous_slot_epochs():
    topology = df.GPUTopology(sm_count=2, cluster_size=2)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(
            node_id=1,
            cta_id=1,
            duration_us=1.0,
            input_values=(10,),
        ),
    )
    schedule = build_schedule(topology, nodes)
    transfer = schedule.transfers[0]
    assert transfer.producer_slot_epoch is not None
    epoch_offset = 7
    relaxed = replace(
        schedule,
        transfers=(
            replace(
                transfer,
                producer_slot_epoch=transfer.producer_slot_epoch + epoch_offset,
                consumer_slot_epoch=transfer.consumer_slot_epoch + epoch_offset,
            ),
        ),
        comm_slot_intervals=tuple(replace(interval, epoch=interval.epoch + epoch_offset) for interval in schedule.comm_slot_intervals),
    )

    assert relaxed.require_valid(topology=topology, nodes=nodes) is relaxed


def test_joint_schedule_verifier_rejects_load_without_matching_store_dependency():
    topology = df.GPUTopology(sm_count=4, cluster_size=2)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(
            node_id=1,
            cta_id=2,
            duration_us=1.0,
            input_values=(10,),
        ),
    )
    schedule = build_schedule(
        topology,
        nodes,
        value_nbytes={10: 512},
        hbm_segment_bytes=256,
    )
    segment = schedule.transfers[0].segments[0]
    events = list(schedule.events)
    load_issue = events[segment.load_issue_event_id]
    events[segment.load_issue_event_id] = replace(load_issue, predecessor_event_ids=())
    unsafe = replace(schedule, events=tuple(events))

    with pytest.raises(df.DataflowJointScheduleError, match="violates store/load ordering"):
        unsafe.require_valid(topology=topology, nodes=nodes)


def test_joint_schedule_retains_early_inbox_when_later_local_dependency_needs_slot():
    topology = df.GPUTopology(sm_count=3, cluster_size=3)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(node_id=1, cta_id=2, duration_us=10.0, output_value=11),
        df.DataflowJointScheduleNode(
            node_id=2,
            cta_id=1,
            duration_us=1.0,
            input_values=(11,),
            output_value=12,
        ),
        df.DataflowJointScheduleNode(
            node_id=3,
            cta_id=1,
            duration_us=1.0,
            input_values=(10, 12),
        ),
    )

    schedule = build_schedule(topology, nodes, retained_copy_us=0.2)

    early = next(transfer for transfer in schedule.transfers if transfer.value_id == 10)
    assert early.retained_input_copy_event_id is not None
    assert any("inbox placement caused" in item for item in schedule.diagnostics)
    assert schedule.require_valid(topology=topology, nodes=nodes) is schedule


def test_joint_schedule_prioritizes_remote_tail_before_independent_long_work():
    topology = df.GPUTopology(sm_count=2, cluster_size=2)
    nodes = (
        df.DataflowJointScheduleNode(node_id=0, cta_id=0, duration_us=1.0, output_value=10),
        df.DataflowJointScheduleNode(
            node_id=1,
            cta_id=1,
            duration_us=1.0,
            input_values=(10,),
        ),
        df.DataflowJointScheduleNode(node_id=2, cta_id=0, duration_us=10.0),
    )

    schedule = build_schedule(topology, nodes)

    assert schedule.queue(0) == (0, 2)
    assert schedule.event(schedule.transfers[0].destination_ready_event_id).end_us < 3.0


def test_joint_schedule_fixed_queue_delays_later_inbox_without_retained_copy():
    topology = df.GPUTopology(sm_count=3, cluster_size=3)
    nodes = (
        df.DataflowJointScheduleNode(0, 0, 3.0, output_value=10),
        df.DataflowJointScheduleNode(1, 1, 2.0, output_value=11),
        df.DataflowJointScheduleNode(2, 2, 1.0, output_value=12),
        df.DataflowJointScheduleNode(
            3,
            1,
            2.0,
            input_values=(10, 11),
            output_value=13,
            output_alias_input_index=0,
        ),
        df.DataflowJointScheduleNode(
            4,
            1,
            2.0,
            input_values=(12, 13),
            output_value=14,
            output_alias_input_index=0,
        ),
    )

    schedule = df.schedule_joint_compute_communication(
        topology,
        nodes,
        cluster_transfer_us=1.0,
        hbm_transfer_us=8.0,
        fixed_queue_node_ids={0: (0,), 1: (1, 3, 4), 2: (2,)},
    )

    transfer_to_later_consumer = next(transfer for transfer in schedule.transfers if transfer.consumer_node_id == 4)
    earlier_handler = next(event for event in schedule.events if event.kind is df.DataflowScheduleEventKind.HANDLER and event.node_id == 3)
    issue = schedule.event(transfer_to_later_consumer.producer_issue_event_id)
    assert issue.start_us >= earlier_handler.end_us
    assert transfer_to_later_consumer.retained_input_copy_event_id is None
    assert schedule.queue(1) == (1, 3, 4)


def test_joint_schedule_prefetches_hbm_across_independent_fixed_queue_handlers():
    topology = df.GPUTopology(sm_count=4, cluster_size=2)
    nodes = (
        df.DataflowJointScheduleNode(0, 0, 1.0, output_value=10),
        df.DataflowJointScheduleNode(1, 2, 6.0, output_value=20),
        df.DataflowJointScheduleNode(2, 2, 3.0, input_values=(20,), output_value=21),
        df.DataflowJointScheduleNode(3, 2, 3.0, input_values=(21,), output_value=22),
        df.DataflowJointScheduleNode(4, 2, 1.0, input_values=(10, 22)),
    )

    schedule = df.schedule_joint_compute_communication(
        topology,
        nodes,
        cluster_transfer_us=1.0,
        hbm_transfer_us=8.0,
        fixed_queue_node_ids={0: (0,), 1: (), 2: (1, 2, 3, 4), 3: ()},
    )

    transfer = schedule.transfers[0]
    load_issue = schedule.event(transfer.consumer_issue_event_id)
    first_overlap = next(event for event in schedule.events if event.kind is df.DataflowScheduleEventKind.HANDLER and event.node_id == 2)
    consumer = schedule.event(transfer.consume_event_id)
    assert transfer.kind is df.DataflowTransportKind.HBM_STAGED
    assert transfer.consumer_storage_kind is df.DataflowCommStorageKind.PERMANENT
    assert load_issue.start_us <= first_overlap.start_us
    assert schedule.event(transfer.destination_ready_event_id).end_us <= consumer.start_us
    assert consumer.start_us == pytest.approx(12.0)
    assert schedule.require_valid(topology=topology, nodes=nodes) is schedule


def test_joint_schedule_pushes_into_permanent_inbox_during_prior_handler():
    topology = df.GPUTopology(sm_count=3, cluster_size=3)
    nodes = (
        df.DataflowJointScheduleNode(0, 0, 1.0, output_value=10),
        df.DataflowJointScheduleNode(1, 1, 6.0, output_value=20),
        df.DataflowJointScheduleNode(
            2,
            1,
            1.0,
            input_values=(10, 20),
            output_alias_input_index=1,
        ),
    )

    schedule = df.schedule_joint_compute_communication(
        topology,
        nodes,
        cluster_transfer_us=1.0,
        hbm_transfer_us=8.0,
        fixed_queue_node_ids={0: (0,), 1: (1, 2), 2: ()},
    )

    transfer = schedule.transfers[0]
    issue = schedule.event(transfer.producer_issue_event_id)
    prior_handler = next(event for event in schedule.events if event.kind is df.DataflowScheduleEventKind.HANDLER and event.node_id == 1)
    consumer = schedule.event(transfer.consume_event_id)
    assert transfer.consumer_storage_kind is df.DataflowCommStorageKind.PERMANENT
    assert issue.start_us < prior_handler.end_us
    assert consumer.start_us == pytest.approx(prior_handler.end_us)
    assert schedule.require_valid(topology=topology, nodes=nodes) is schedule


def test_joint_schedule_double_buffers_ordered_cluster_pushes_across_reduce():
    topology = df.GPUTopology(sm_count=4, cluster_size=4)
    nodes = (
        df.DataflowJointScheduleNode(0, 0, 1.0, output_value=10),
        df.DataflowJointScheduleNode(1, 2, 1.0, output_value=11),
        df.DataflowJointScheduleNode(2, 1, 6.0, output_value=20),
        df.DataflowJointScheduleNode(
            3,
            1,
            3.0,
            input_values=(10, 20),
            output_value=30,
            transient_prefetch_bytes=1024,
        ),
        df.DataflowJointScheduleNode(
            4,
            1,
            1.0,
            input_values=(11, 30),
            transient_prefetch_bytes=1024,
        ),
    )

    schedule = df.schedule_joint_compute_communication(
        topology,
        nodes,
        cluster_transfer_us=1.0,
        hbm_transfer_us=8.0,
        value_nbytes={10: 1024, 11: 1024, 20: 1024, 30: 1024},
        fixed_queue_node_ids={0: (0,), 1: (2, 3, 4), 2: (1,), 3: ()},
    )

    first, second = schedule.transfers
    first_consumer = schedule.event(first.consume_event_id)
    second_issue = schedule.event(second.producer_issue_event_id)
    assert first.consumer_storage_kind is df.DataflowCommStorageKind.PERMANENT
    assert second.consumer_storage_kind is df.DataflowCommStorageKind.TRANSIENT_PREFETCH
    assert second_issue.start_us <= first_consumer.start_us
    assert schedule.require_valid(topology=topology, nodes=nodes) is schedule


def test_joint_schedule_fixed_queue_requires_exact_node_ownership():
    topology = df.GPUTopology(sm_count=2, cluster_size=2)
    nodes = (
        df.DataflowJointScheduleNode(0, 0, 1.0),
        df.DataflowJointScheduleNode(1, 1, 1.0),
    )

    with pytest.raises(df.DataflowJointScheduleError, match="every compute node"):
        df.schedule_joint_compute_communication(
            topology,
            nodes,
            cluster_transfer_us=1.0,
            hbm_transfer_us=8.0,
            fixed_queue_node_ids={0: (0,)},
        )


def test_joint_execution_plan_round_trips_with_stable_fingerprint():
    topology = df.GPUTopology(sm_count=4, cluster_size=2)
    nodes = (
        df.DataflowJointScheduleNode(0, 0, 1.0, output_value=10),
        df.DataflowJointScheduleNode(1, 2, 1.0, input_values=(10,)),
    )
    schedule = build_schedule(
        topology,
        nodes,
        value_nbytes={10: 1024},
        hbm_segment_bytes=512,
        fixed_queue_node_ids={0: (0,), 1: (), 2: (1,), 3: ()},
    )
    plan = df.DataflowJointExecutionPlan(
        nodes=nodes,
        schedule=schedule,
        hbm_segment_bytes=512,
    ).require_valid(topology=topology)

    restored = df.DataflowJointExecutionPlan.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.fingerprint == plan.fingerprint

    stale = plan.to_dict()
    stale["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint is stale"):
        df.DataflowJointExecutionPlan.from_dict(stale)
