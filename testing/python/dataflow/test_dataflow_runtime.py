from __future__ import annotations

from dataclasses import replace
import random
import struct

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.tensor_args import TENSOR_ARG_STRUCT


@T.dataflow_intermediate
class AttnInter:
    lse: T.float32
    o: T.Tensor((2, 8), T.float16)


@T.dataflow.iter(range=("kv_begin", "kv_end"))
def split_kv(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
    raise AssertionError("Dataflow iter body should not execute during runtime packing")


@T.dataflow.reduce
def combine(items: list[AttnInter]) -> AttnInter:
    raise AssertionError("Dataflow reduce body should not execute during runtime packing")


@T.dataflow.finalize
def finalize(inter: AttnInter, seq: T.int32, head: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during runtime packing")


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
    raise AssertionError("Dataflow map body should not execute during runtime packing")


@T.dataflow.map(range=("hidden_begin", "hidden_end"))
def moe_map2(parts: list[UpShard], expert: T.int32, token: T.int32, W2) -> HiddenShard:
    raise AssertionError("Dataflow map body should not execute during runtime packing")


@T.dataflow.finalize
def finalize_hidden(hidden: HiddenShard, expert: T.int32, token: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during runtime packing")


def make_program():
    return (
        T.dataflow_program(task_domain=("batch", "head"), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
            range_axis="kv",
        )
        .reduce(combine())
        .finalize(finalize(Output="O"))
    )


def make_reshared_program():
    return (
        T.dataflow_program(
            task_domain=("expert", "token"),
            dynamic_ranges={"expert_tile": "expert_tiles", "hidden_tile": "hidden_tiles"},
        )
        .map(
            moe_map1(Input="Input", W1="W1"),
            name="map1",
            task_args=("expert", "token"),
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
            moe_map2(W2="W2"),
            name="map2",
            input="gather_up",
            task_args=("expert", "token"),
            range_axis="hidden_tile",
        )
        .finalize(finalize_hidden(Output="Output"), input="map2")
    )


def make_plan():
    return df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        block_size=128,
        task_extents=(2,),
    )


def make_reshared_plan():
    return df.schedule(
        make_reshared_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": [256], "hidden_tile": [256]},
        block_size=128,
        task_extents=(1,),
        include_exit=False,
        force_hbm_comms=True,
    )


def test_persistent_executable_skips_unchanged_tensor_args_h2d(monkeypatch):
    from tilelang.dataflow import executor

    allocations = []
    copies = []

    class FakeDriver:
        def cuMemFree(self, ptr):
            return (0,)

    def fake_alloc_empty(driver, allocation_list, byte_count):
        ptr = f"ptr{len(allocation_list)}"
        allocation_list.append(ptr)
        allocations.append((ptr, byte_count))
        return ptr

    def fake_copy_to(driver, ptr, data):
        copies.append((ptr, bytes(data)))

    monkeypatch.setattr(executor, "device_alloc_empty", fake_alloc_empty)
    monkeypatch.setattr(executor, "device_copy_to", fake_copy_to)

    executable = executor.DataflowPersistentExecutable(
        kernel_name="test",
        source="",
        launch_package=None,
        topology=None,
    )
    executable._driver = FakeDriver()

    executable.allocate_or_update_tensor_args(b"abc")
    executable.allocate_or_update_tensor_args(b"abc")
    executable.allocate_or_update_tensor_args(b"xyz")

    assert allocations == [("ptr0", 3)]
    assert copies == [("ptr0", b"abc"), ("ptr0", b"xyz")]


def test_persistent_executable_rejects_pointer_only_tma_descriptor_rebuild(
    monkeypatch,
):
    from tilelang.dataflow import executor

    rebuilds = []

    class FakeDriver:
        def cuMemFree(self, ptr):
            return (0,)

    monkeypatch.setattr(
        executor,
        "device_alloc_empty",
        lambda driver, allocation_list, byte_count: allocation_list.append("ptr0") or "ptr0",
    )
    monkeypatch.setattr(executor, "device_copy_to", lambda driver, ptr, data: None)

    def fake_build_tma_descriptor_handles(specs, *, tensor_data_ptrs):
        rebuilds.append((tuple(specs), dict(tensor_data_ptrs)))
        return ("handle",)

    monkeypatch.setattr(executor, "build_tma_descriptor_handles", fake_build_tma_descriptor_handles)
    monkeypatch.setattr(
        executor,
        "validate_tma_descriptor_runtime_tensors",
        lambda *args, **kwargs: None,
    )

    executable = executor.DataflowPersistentExecutable(
        kernel_name="test",
        source="",
        launch_package=None,
        topology=None,
        tma_descriptor_specs=("desc",),
        tma_tensor_indices={"A": 0},
    )
    executable._driver = FakeDriver()

    first = TENSOR_ARG_STRUCT.pack(123, 0, 0, 0, 0, 0)
    second = TENSOR_ARG_STRUCT.pack(456, 0, 0, 0, 0, 0)
    executable.allocate_or_update_tensor_args(first)
    executable.allocate_or_update_tensor_args(first)
    with pytest.raises(ValueError, match="pointer-only tensor_args_bytes"):
        executable.allocate_or_update_tensor_args(second)

    assert rebuilds == [
        (("desc",), {"A": 123}),
    ]
    assert executable.tensor_args_bytes == first
    assert executable._tensor_args_uploaded_bytes == first


def test_tma_descriptor_runtime_tensor_validation_is_fail_closed():
    from tilelang.dataflow.tensor_args import DataflowRuntimeTensorMetadata
    from tilelang.dataflow.tma_descriptors import (
        DataflowTMADescriptorSpec,
        validate_tma_descriptor_runtime_tensors,
    )

    spec = DataflowTMADescriptorSpec(
        name="A_desc",
        tensor_name="A",
        dtype=6,
        tensor_rank=2,
        global_dim=(16, 4),
        global_stride=(2, 32),
        box_dim=(16, 4),
        element_strides=(1, 1),
        interleave=0,
        swizzle=0,
        l2_promotion=2,
        oob_fill=0,
    )
    valid = DataflowRuntimeTensorMetadata(
        source="test",
        shape=(4, 16),
        strides_bytes=(32, 2),
        device_type="cuda",
        device_index=0,
        dtype_code=df.DATAFLOW_TENSOR_DTYPE_FLOAT,
        dtype_bits=16,
    )
    validate_tma_descriptor_runtime_tensors(
        (spec,),
        tensor_data_ptrs={"A": 0x1000},
        tensor_metadata={"A": valid},
        expected_device_ordinal=0,
    )

    invalid_cases = [
        (
            {"A": 0x1001},
            {"A": valid},
            "16-byte aligned",
        ),
        (
            {"A": 0x1000},
            {"A": replace(valid, shape=(8, 8))},
            "shape mismatch",
        ),
        (
            {"A": 0x1000},
            {"A": replace(valid, strides_bytes=(64, 2))},
            "byte-stride mismatch",
        ),
        (
            {"A": 0x1000},
            {"A": replace(valid, dtype_code=df.DATAFLOW_TENSOR_DTYPE_BFLOAT)},
            "dtype mismatch",
        ),
        (
            {"A": 0x1000},
            {"A": replace(valid, device_index=1)},
            "device mismatch",
        ),
        (
            {"A": 0x1000},
            {
                "A": DataflowRuntimeTensorMetadata(
                    source="raw_pointer",
                )
            },
            "raw pointer",
        ),
    ]
    for pointers, metadata, expected in invalid_cases:
        with pytest.raises(ValueError, match=expected):
            validate_tma_descriptor_runtime_tensors(
                (spec,),
                tensor_data_ptrs=pointers,
                tensor_metadata=metadata,
                expected_device_ordinal=0,
            )


def test_executor_detects_when_launch_package_needs_hbm_flags():
    from tilelang.dataflow import executor
    from tilelang.dataflow.launch import build_launch_package

    cluster_plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [256, 256]},
        block_size=128,
        task_extents=(2,),
        scheduler_policy="cluster_local",
    )
    hbm_plan = make_plan()

    cluster_package = build_launch_package(df.pack_instruction_plan(cluster_plan))
    hbm_package = build_launch_package(df.pack_instruction_plan(hbm_plan))

    assert executor.launch_package_needs_flag_reset(cluster_package) is False
    assert executor.launch_package_needs_flag_reset(hbm_package) is True


def test_pack_instruction_plan_creates_stable_abi_records():
    plan = make_plan()
    packed = df.pack_instruction_plan(plan)

    assert packed.abi_version == df.ABI_VERSION
    assert packed.instruction_record_size == df.INSTRUCTION_STRUCT.size == 32
    assert packed.slot_record_size == df.SLOT_STRUCT.size == 32
    assert packed.comm_record_size == df.COMM_STRUCT.size == 48
    assert packed.arg_record_size == df.ARG_STRUCT.size == 64

    assert packed.operator_table == {
        "split_kv": 0,
        "combine": 1,
        "finalize": 2,
    }
    assert [identity.to_dict() for identity in packed.handler_identities] == [
        {
            "operator_id": 0,
            "operator_kind": "iter",
        },
        {
            "operator_id": 1,
            "operator_kind": "reduce",
        },
        {
            "operator_id": 2,
            "operator_kind": "finalize",
        },
        {
            "operator_id": 1,
            "operator_kind": "reduce",
        },
    ]
    assert [
        (
            variant.base_identity.binding_key,
            None if variant.iter_range is None else variant.iter_range.canonical_key,
            None if variant.reduce_arity is None else variant.reduce_arity.canonical_key,
        )
        for variant in packed.handler_variant_keys
    ] == [
        ((0, "iter"), ("iter_range", "generic", None), None),
        ((1, "reduce"), None, ("reduce_arity", "generic")),
        ((2, "finalize"), None, None),
        ((1, "reduce"), None, ("reduce_arity", "binary")),
    ]
    assert packed.intermediate_type_table == {"AttnInter": 0}
    assert packed.queue_offsets == (0, 5, 7, 9)
    assert packed.queue_lengths == (5, 2, 2, 4)
    assert packed.input_slots == (0, 1, 2, 3, 4, 5, 6)
    assert packed.task_coords == (0, 0, 0, 1, 0, 0, 1, 1, 1)

    assert len(packed.instruction_bytes) == len(plan.instructions) * df.INSTRUCTION_STRUCT.size
    assert len(packed.slot_bytes) == len(plan.slots) * df.SLOT_STRUCT.size
    assert len(packed.comm_bytes) == len(plan.comms) * df.COMM_STRUCT.size
    assert len(packed.arg_bytes) == 9 * df.ARG_STRUCT.size
    assert len(packed.queue_offsets_bytes) == 4 * 4
    assert len(packed.queue_lengths_bytes) == 4 * 4
    assert len(packed.input_slots_bytes) == len(packed.input_slots) * 4
    assert len(packed.task_coords_bytes) == len(packed.task_coords) * 4


def test_handler_packing_uses_identity_instead_of_diagnostic_names():
    plan = make_plan()
    original_packed = df.pack_instruction_plan(plan)
    renamed_instructions = tuple(
        replace(
            instruction,
            operator_name=(
                instruction.operator_name if instruction.handler_identity is None else f"diagnostic_{instruction.instruction_id}"
            ),
        )
        for instruction in plan.instructions
    )
    renamed_by_id = {instruction.instruction_id: instruction for instruction in renamed_instructions}
    renamed_plan = replace(
        plan,
        instructions=renamed_instructions,
        queues={sm_id: tuple(renamed_by_id[instruction.instruction_id] for instruction in queue) for sm_id, queue in plan.queues.items()},
    )

    packed = df.pack_instruction_plan(renamed_plan)
    wrapper = df.build_wrapper_spec(packed)

    assert len(packed.handler_identities) == 4
    assert [identity for identity in packed.handler_identities] == [handler.handler_identity for handler in wrapper.handlers]
    assert list(packed.handler_variant_keys) == [handler.handler_variant_key for handler in wrapper.handlers]
    assert packed.handler_identities == original_packed.handler_identities
    assert packed.handler_variant_keys == original_packed.handler_variant_keys
    assert df.instruction_plan_fingerprint(renamed_plan) == df.instruction_plan_fingerprint(plan)
    ordered_instructions = []
    seen_instruction_ids = set()
    for sm_id in range(renamed_plan.topology.sm_count):
        for instruction in renamed_plan.queue(sm_id):
            ordered_instructions.append(instruction)
            seen_instruction_ids.add(instruction.instruction_id)
    ordered_instructions.extend(
        instruction for instruction in renamed_plan.instructions if instruction.instruction_id not in seen_instruction_ids
    )
    handler_ids_by_variant = {variant: handler_id for handler_id, variant in enumerate(packed.handler_variant_keys)}
    for instruction, packed_instruction, original_instruction in zip(
        ordered_instructions,
        packed.instructions,
        original_packed.instructions,
    ):
        if instruction.handler_variant_key is None:
            continue
        assert packed_instruction.handler_id == handler_ids_by_variant[instruction.handler_variant_key]
        assert packed_instruction.handler_id == original_instruction.handler_id

    reduce_handlers = [handler for handler in wrapper.handlers if handler.operator_kind == "reduce"]
    assert len(reduce_handlers) == 2
    assert len({handler.handler_id for handler in reduce_handlers}) == 2
    assert len({handler.handler_identity.binding_key for handler in reduce_handlers}) == 1
    assert {handler.handler_variant_key.reduce_arity.arity_class for handler in reduce_handlers} == {
        df.REDUCE_ARITY_BINARY,
        df.REDUCE_ARITY_GENERIC,
    }


def test_additional_variants_do_not_renumber_primary_base_handlers():
    plan = make_plan()
    instructions = list(plan.instructions)
    additional_reduce_index = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.handler_variant_key is not None
        and instruction.handler_variant_key.reduce_arity == df.ReduceAritySpecialization(df.REDUCE_ARITY_BINARY)
    )
    finalize_index = next(index for index, instruction in enumerate(instructions) if instruction.opcode.value == "finalize")
    additional_reduce = instructions.pop(additional_reduce_index)
    instructions.insert(finalize_index, additional_reduce)

    packed = df.pack_instruction_plan(replace(plan, instructions=tuple(instructions)))

    assert [identity.operator_kind for identity in packed.handler_identities] == [
        "iter",
        "reduce",
        "finalize",
        "reduce",
    ]
    assert packed.handler_variant_keys[1].reduce_arity.arity_class == (df.REDUCE_ARITY_GENERIC)
    assert packed.handler_variant_keys[3].reduce_arity.arity_class == (df.REDUCE_ARITY_BINARY)


def test_typed_handler_variants_round_trip_for_random_iter_and_reduce_inputs():
    rng = random.Random(0)
    iter_identity = df.DataflowHandlerIdentity(operator_id=7, operator_kind="iter")
    reduce_identity = df.DataflowHandlerIdentity(operator_id=11, operator_kind="reduce")

    for _ in range(100):
        mode = rng.choice(
            (
                df.ITER_RANGE_GENERIC,
                df.ITER_RANGE_TILE_COUNT,
                df.ITER_RANGE_EXACT_LENGTH,
            )
        )
        specialization = df.IterRangeSpecialization(
            mode=mode,
            value=None if mode == df.ITER_RANGE_GENERIC else rng.randint(1, 4096),
        )
        variant = df.DataflowHandlerVariantKey(
            base_identity=iter_identity,
            typed_specializations=(specialization,),
        )
        assert df.DataflowHandlerVariantKey.from_dict(variant.to_dict()) == variant
        assert variant.binding_key == iter_identity.binding_key

        arity = rng.randint(1, 64)
        reduce_variant = df.DataflowHandlerVariantKey.default_for_identity(
            reduce_identity,
            reduce_arity=arity,
        )
        assert df.DataflowHandlerVariantKey.from_dict(reduce_variant.to_dict()) == reduce_variant
        assert reduce_variant.binding_key == reduce_identity.binding_key
        assert reduce_variant.reduce_arity == df.ReduceAritySpecialization.for_arity(arity)

    with pytest.raises(ValueError, match="specialization kinds do not match"):
        df.DataflowHandlerVariantKey(base_identity=iter_identity)
    with pytest.raises(ValueError, match="cannot contain specialization metadata"):
        df.DataflowHandlerIdentity.from_dict(
            {
                "operator_id": 7,
                "operator_kind": "iter",
                "specialization": {"mode": "generic", "value": None},
            }
        )


@pytest.mark.parametrize(
    ("arity", "expected_class"),
    (
        (1, df.REDUCE_ARITY_PASSTHROUGH),
        (2, df.REDUCE_ARITY_BINARY),
        (3, df.REDUCE_ARITY_GENERIC),
        (64, df.REDUCE_ARITY_GENERIC),
    ),
)
def test_reduce_arity_maps_to_typed_variant_class(arity, expected_class):
    identity = df.DataflowHandlerIdentity(operator_id=0, operator_kind="reduce")
    variant = df.DataflowHandlerVariantKey.default_for_identity(
        identity,
        reduce_arity=arity,
    )

    assert variant.binding_key == identity.binding_key
    assert variant.reduce_arity.arity_class == expected_class


def test_runtime_packing_rejects_reduce_variant_with_wrong_arity_class():
    plan = make_plan()
    reduce_instruction = next(
        instruction for instruction in plan.instructions if instruction.opcode.value == "reduce" and len(instruction.input_slots) == 2
    )
    wrong_variant = df.DataflowHandlerVariantKey.default_for_identity(
        reduce_instruction.handler_identity,
        reduce_arity=3,
    )
    invalid_instruction = replace(
        reduce_instruction,
        handler_variant_key=wrong_variant,
    )
    invalid_plan = replace(
        plan,
        instructions=tuple(
            invalid_instruction if instruction.instruction_id == invalid_instruction.instruction_id else instruction
            for instruction in plan.instructions
        ),
    )

    with pytest.raises(ValueError, match="does not match instruction arity"):
        df.pack_instruction_plan(invalid_plan)


def test_iter_specialization_identity_round_trips_for_varied_lengths():
    with pytest.raises(ValueError, match="Unsupported Dataflow ITER specialization mode"):
        df.IterRangeSpecialization(mode="name_suffix", value=1)
    with pytest.raises(ValueError, match="requires a positive integer value"):
        df.IterRangeSpecialization(mode=df.ITER_RANGE_TILE_COUNT, value=0)
    iter_identity = df.DataflowHandlerIdentity(operator_id=0, operator_kind="iter")
    assert iter_identity.binding_key == (0, "iter")

    rng = random.Random(0)
    lengths = list(range(16, 129, 16)) + [17, 23, 47]
    rng.shuffle(lengths)
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": lengths},
        block_size=128,
        task_extents=(len(lengths),),
        iter_range_buckets=tuple(range(1, 9)) + ("generic",),
        iter_range_bucket_size=16,
        iter_range_exact_lengths=(17, 23, 47),
    )
    packed = df.pack_instruction_plan(plan)
    wrapper = df.build_wrapper_spec(packed)

    plan_variants = []
    for instruction in plan.instructions:
        variant = instruction.handler_variant_key
        if variant is None or variant.base_identity.operator_kind != "iter":
            continue
        if variant not in plan_variants:
            plan_variants.append(variant)
    packed_variants = [
        variant for variant in packed.handler_variant_keys if variant is not None and variant.base_identity.operator_kind == "iter"
    ]
    wrapper_variants = [handler.handler_variant_key for handler in wrapper.handlers if handler.operator_kind == "iter"]

    assert packed_variants == plan_variants
    assert wrapper_variants == plan_variants
    assert len({variant.binding_key for variant in plan_variants}) == 1
    assert {(variant.iter_range.mode, variant.iter_range.value) for variant in plan_variants} == {
        *(("tile_count", value) for value in range(1, 9)),
        ("exact_length", 17),
        ("exact_length", 23),
        ("exact_length", 47),
    }


def test_pack_instruction_plan_handles_hbm_reshared_stage_graph():
    plan = make_reshared_plan()
    packed = df.pack_instruction_plan(plan)

    assert packed.operator_table == {
        "moe_map1": 0,
        "moe_map2": 1,
        "finalize_hidden": 2,
    }
    assert packed.operator_kinds == {
        "moe_map1": "map",
        "moe_map2": "map",
        "finalize_hidden": "finalize",
    }
    assert packed.intermediate_type_table == {
        "UpShard": 0,
        "HiddenShard": 1,
    }
    assert len(packed.slots) == 8
    assert len(packed.comms) == 4

    reshared_instructions = [
        instruction for instruction in packed.instructions if instruction.handler_id == df.UINT32_SENTINEL and instruction.opcode == 1
    ]
    assert len(reshared_instructions) == 2
    assert [instruction.comm_count for instruction in reshared_instructions] == [1, 1]
    assert [instruction.slot_id for instruction in reshared_instructions] == [2, 4]

    assert packed.slots[0].shared_offset == packed.slots[2].shared_offset
    assert packed.slots[1].shared_offset == packed.slots[5].shared_offset

    assert [comm.kind for comm in packed.comms] == [3, 4, 3, 4]


def test_pack_instruction_plan_can_place_hbm_reshared_slots_directly_in_global_memory():
    plan = make_reshared_plan()
    packed = df.pack_instruction_plan(plan, hbm_direct_global=True)
    package = df.build_launch_package(packed)

    assert len(packed.slots) == 8
    assert all(slot.flags & df.DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL for slot in packed.slots)
    assert {slot.shared_offset for slot in packed.slots} == {0}
    assert package.shared_slot_bytes == 0
    assert package.shared_memory_bytes == package.shared_slot_base_offset
    assert package.global_staging_bytes


def test_pack_instruction_plan_can_select_individual_hbm_direct_global_slots():
    plan = make_plan()
    selected_slot_id = next(comm.target_slot_id for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_RECV)

    packed = df.pack_instruction_plan(
        plan,
        hbm_direct_global_slot_ids=frozenset((selected_slot_id,)),
    )

    direct_slot_ids = {slot_id for slot_id, slot in enumerate(packed.slots) if slot.flags & df.DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL}
    assert direct_slot_ids == {selected_slot_id}
    assert packed.slots[selected_slot_id].shared_offset == 0
    assert all(
        not (slot.flags & df.DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL)
        for slot_id, slot in enumerate(packed.slots)
        if slot_id != selected_slot_id
    )

    with pytest.raises(ValueError, match="cannot combine full and selective"):
        df.pack_instruction_plan(
            plan,
            hbm_direct_global=True,
            hbm_direct_global_slot_ids=frozenset((selected_slot_id,)),
        )
    with pytest.raises(ValueError, match="must exist in the plan"):
        df.pack_instruction_plan(
            plan,
            hbm_direct_global_slot_ids=frozenset((len(plan.slots),)),
        )


def test_pack_instruction_plan_round_trips_key_instruction_fields():
    packed = df.pack_instruction_plan(make_plan())

    first_iter = struct.unpack_from("<8I", packed.instruction_bytes, 0)
    assert first_iter == (
        1,  # ITER opcode
        0,  # split_kv handler_id
        0,  # task_id
        0,  # arg_offset
        0,  # comm_offset
        0,  # same-CTA dependency needs no comm
        0,  # output_slot
        0,  # flags
    )

    first_reduce_offset = 1 * df.INSTRUCTION_STRUCT.size
    first_reduce = struct.unpack_from("<8I", packed.instruction_bytes, first_reduce_offset)
    assert first_reduce == (
        2,  # REDUCE opcode
        1,  # combine handler_id
        0,  # task_id
        64,  # arg_offset
        0,  # receive-side comm_offset
        2,  # reduce waits on the two non-local partial slots
        3,  # reduced output slot
        1,  # one cluster receive precedes the HBM receive
    )

    first_exit = packed.instructions[4]
    assert first_exit.opcode == 0
    assert first_exit.handler_id == df.UINT32_SENTINEL
    assert first_exit.task_id == df.UINT32_SENTINEL
    assert first_exit.arg_offset == df.UINT32_SENTINEL
    assert first_exit.slot_id == df.UINT32_SENTINEL


def test_pack_instruction_plan_encodes_cluster_comm_batch_counts():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [512]},
        block_size=128,
        task_extents=(1,),
        scheduler_policy="cluster_local",
    )

    packed = df.pack_instruction_plan(plan)
    recv_instruction = next(instruction for instruction in packed.instructions if instruction.comm_count == 3)
    send_instructions = [instruction for instruction in packed.instructions if instruction.comm_count == 1]

    assert recv_instruction.flags & 0xFFFF == 3
    assert recv_instruction.flags >> 16 == 0
    assert [instruction.flags for instruction in send_instructions] == [1 << 16] * 3
    assert [comm.kind for comm in packed.comms] == [2, 2, 2, 1, 1, 1]


def test_pack_instruction_plan_round_trips_handler_args():
    packed = df.pack_instruction_plan(make_plan())

    first_iter_args = struct.unpack_from("<8I", packed.arg_bytes, 0)
    assert first_iter_args == (
        0,  # task_id
        0,  # task_coord_offset
        1,  # task_coord_count
        0,  # range_begin
        128,  # range_end
        0,  # input_slot_offset
        0,  # input_slot_count
        0,  # output_slot
    )

    first_reduce_args = struct.unpack_from("<8I", packed.arg_bytes, df.ARG_STRUCT.size)
    assert first_reduce_args == (
        0,
        1,
        1,
        0,
        0,
        0,  # reduce consumes input_slots[0:3]
        3,
        3,
    )

    first_finalize_args = struct.unpack_from("<8I", packed.arg_bytes, 2 * df.ARG_STRUCT.size)
    assert first_finalize_args == (
        0,
        2,
        1,
        0,
        0,
        3,  # finalize consumes input_slots[3:4]
        1,
        df.UINT32_SENTINEL,
    )

    task1_iter_on_sm0_args = struct.unpack_from("<8I", packed.arg_bytes, 3 * df.ARG_STRUCT.size)
    assert task1_iter_on_sm0_args == (
        1,
        3,
        1,
        128,
        256,
        4,
        0,
        5,
    )


def test_pack_instruction_plan_round_trips_slots_and_comms():
    packed = df.pack_instruction_plan(make_plan())

    first_slot = struct.unpack_from("<8I", packed.slot_bytes, 0)
    assert first_slot == (
        0,  # shared_offset
        0,  # global_offset
        48,  # aligned AttnInter slot capacity
        0,  # flag_index
        df.UINT32_SENTINEL,  # no cluster recv targets this slot
        0,  # owner CTA rank
        0,  # flags
        0,  # reserved
    )

    first_reduced_slot = struct.unpack_from("<8I", packed.slot_bytes, 3 * df.SLOT_STRUCT.size)
    assert first_reduced_slot == (
        144,
        144,
        48,
        3,
        df.UINT32_SENTINEL,
        0,  # reduced slot is produced by the reducer on SM/CTA 0
        0,
        0,
    )

    first_comm = struct.unpack_from("<8I", packed.comm_bytes, 0)
    assert first_comm == (
        2,  # CLUSTER_RECV
        1,  # source partial slot
        1,  # target partial slot on reducer CTA
        1,  # source CTA rank
        0,  # runtime uses slot bytes
        df.UINT32_SENTINEL,  # ordinary push does not need a lifetime ACK
        0,  # first barrier parity
        df.UINT32_SENTINEL,  # use target slot default barrier
    )

    first_recv = struct.unpack_from("<8I", packed.comm_bytes, df.COMM_STRUCT.size)
    assert first_recv == (
        4,  # HBM_RECV
        2,  # source staging slot
        2,  # target partial slot on reducer CTA
        df.UINT32_SENTINEL,  # HBM recv does not use peer CTA rank
        0,  # runtime uses slot bytes
        df.UINT32_SENTINEL,  # use source slot default flag
        1,  # first HBM epoch
        1,  # explicit local receive barrier
    )

    task0_cluster_send = struct.unpack_from("<8I", packed.comm_bytes, 3 * df.COMM_STRUCT.size)
    assert task0_cluster_send == (
        1,  # CLUSTER_SEND
        1,  # source slot
        1,  # target partial slot on reducer CTA
        0,  # reducer CTA rank
        0,  # runtime uses slot bytes
        df.UINT32_SENTINEL,  # ordinary push does not need a lifetime ACK
        0,  # cluster send does not use an epoch
        df.UINT32_SENTINEL,  # use destination slot default barrier
    )

    task0_hbm_send = struct.unpack_from("<8I", packed.comm_bytes, 4 * df.COMM_STRUCT.size)
    assert task0_hbm_send == (
        3,  # HBM_SEND
        2,  # source slot
        2,  # target staging slot
        df.UINT32_SENTINEL,  # HBM send does not use peer CTA rank
        0,  # runtime uses slot bytes
        df.UINT32_SENTINEL,  # use destination slot default flag
        1,  # first HBM epoch
        df.UINT32_SENTINEL,  # HBM send does not use a barrier
    )


def test_pack_instruction_plan_only_allocates_cluster_ack_for_released_transfer():
    from tilelang.dataflow.abi_schema import DATAFLOW_COMM_KIND_ABI_VALUES

    plan = make_plan()
    cluster_send = next(comm for comm in plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_SEND)
    cluster_release = replace(
        cluster_send,
        kind=df.DataflowCommKind.CLUSTER_RELEASE,
        dispatch_instruction_id=cluster_send.target_instruction_id,
        producer_sm=cluster_send.consumer_sm,
        consumer_sm=cluster_send.producer_sm,
        peer_cta_rank=plan.topology.cluster_rank(cluster_send.producer_sm),
    )

    packed = df.pack_instruction_plan(replace(plan, comms=plan.comms + (cluster_release,)))
    matching_cluster_comms = [
        comm
        for comm in packed.comms
        if comm.kind
        in {
            DATAFLOW_COMM_KIND_ABI_VALUES["cluster_send"],
            DATAFLOW_COMM_KIND_ABI_VALUES["cluster_recv"],
            DATAFLOW_COMM_KIND_ABI_VALUES["cluster_release"],
        }
    ]

    assert {comm.flag_index for comm in matching_cluster_comms} == {0}
    assert df.build_launch_package(packed).cluster_ack_count == 1


def test_pack_instruction_plan_preserves_segmented_hbm_byte_ranges_and_barrier_keys():
    plan = make_plan()
    recv = next(comm for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_RECV)
    transfer_key = (
        recv.source_instruction_id,
        recv.target_instruction_id,
        recv.source_slot_id,
        recv.target_slot_id,
    )
    segment_layout = ((0, 0, 16), (1, 16, 16), (2, 32, 16))
    comms = []
    for comm in plan.comms:
        key = (
            comm.source_instruction_id,
            comm.target_instruction_id,
            comm.source_slot_id,
            comm.target_slot_id,
        )
        if key != transfer_key or comm.kind not in {
            df.DataflowCommKind.HBM_SEND,
            df.DataflowCommKind.HBM_RECV,
        }:
            comms.append(comm)
            continue
        comms.extend(
            replace(
                comm,
                byte_offset=byte_offset,
                byte_count=byte_count,
                segment_id=segment_id,
                segment_count=len(segment_layout),
                flag_epoch=segment_id + 1,
            )
            for segment_id, byte_offset, byte_count in segment_layout
        )
    segmented = replace(plan, comms=tuple(comms))

    allocation = df.plan_dataflow_barriers(segmented).require_valid()
    segment_keys = {key for key in allocation.transfer_phases if key[:4] == transfer_key}
    assert {key[-1] for key in segment_keys} == {0, 1, 2}
    assert {allocation.transfer_barrier_indices[key] for key in segment_keys} == {allocation.transfer_barrier_indices[min(segment_keys)]}
    assert {allocation.transfer_phases[key] for key in segment_keys} == {0}

    packed = df.pack_instruction_plan(segmented)
    transfer_comms = [
        (index, comm)
        for index, comm in enumerate(packed.comms)
        if (
            comm.src_slot_id,
            comm.dst_slot_id,
        )
        == transfer_key[2:]
        and comm.kind in (3, 4)
    ]
    assert len(transfer_comms) == 6
    for kind in (3, 4):
        kind_comms = sorted(
            (item for item in transfer_comms if item[1].kind == kind),
            key=lambda item: item[1].segment_id,
        )
        assert [
            (
                comm.segment_id,
                comm.byte_offset,
                comm.byte_count,
                comm.segment_count,
            )
            for _, comm in kind_comms
        ] == [(segment_id, offset, count, 3) for segment_id, offset, count in segment_layout]
        for index, comm in kind_comms:
            raw = df.COMM_STRUCT.unpack_from(
                packed.comm_bytes,
                index * df.COMM_STRUCT.size,
            )
            assert raw[8:] == (
                comm.byte_offset,
                comm.byte_count,
                comm.segment_id,
                comm.segment_count,
            )


def test_pack_instruction_plan_marks_scratch_backed_slot_without_normal_shared_extent():
    plan = make_plan()
    slots = list(plan.slots)
    slots[1] = df.SlotPlan(
        slot_id=slots[1].slot_id,
        task_id=slots[1].task_id,
        intermediate_type=slots[1].intermediate_type,
        role=slots[1].role,
        producer_instruction_id=slots[1].producer_instruction_id,
        scratch_backed=True,
        scratch_offset=4096,
    )

    packed = df.pack_instruction_plan(
        df.InstructionPlan(
            topology=plan.topology,
            block_size=plan.block_size,
            range_axis=plan.range_axis,
            scheduler_policy=plan.scheduler_policy,
            reduce_strategy=plan.reduce_strategy,
            task_extents=plan.task_extents,
            task_range_lengths=plan.task_range_lengths,
            instructions=plan.instructions,
            queues=plan.queues,
            slots=tuple(slots),
            comms=plan.comms,
        )
    )

    scratch_slot = packed.slots[1]
    assert scratch_slot.shared_offset == 4096
    assert scratch_slot.bytes == 48
    assert scratch_slot.flags & df.DATAFLOW_SLOT_FLAG_SCRATCH_BACKED

    normal_offsets = [slot.shared_offset for slot in packed.slots if not (slot.flags & df.DATAFLOW_SLOT_FLAG_SCRATCH_BACKED)]
    assert normal_offsets == [0, 48, 96, 144, 192, 240]


def test_pack_instruction_plan_uses_cluster_local_peer_cta_rank():
    program = (
        T.dataflow_program(task_domain=("batch", "head"), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
            range_axis="kv",
        )
        .reduce(combine())
        .finalize(finalize(Output="O"))
    )
    plan = df.schedule(
        program,
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [256, 256]},
        block_size=128,
        task_extents=(2,),
    )

    cluster_sends = [comm for comm in plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_SEND]
    assert [(comm.producer_sm, comm.consumer_sm, comm.peer_cta_rank) for comm in cluster_sends] == [
        (1, 0, 0),
        (3, 2, 0),
    ]

    packed = df.pack_instruction_plan(plan)
    packed_cluster_sends = [comm for comm in packed.comms if comm.kind == 1]
    assert [comm.peer_cta_rank for comm in packed_cluster_sends] == [0, 0]


def test_pack_instruction_plan_assigns_monotonic_hbm_epochs_for_reused_flags():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [256, 256]},
        task_extents=(2,),
        block_size=128,
    )
    packed = df.pack_instruction_plan(plan)
    hbm_epochs = [comm.flag_epoch for comm in packed.comms if comm.kind in (3, 4)]
    assert hbm_epochs
    assert sorted(set(hbm_epochs)) == [1, 2]


def test_pack_instruction_plan_reuses_hbm_flag_with_monotonic_epochs_when_requested():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        task_extents=(2,),
        block_size=128,
    )

    packed = df.pack_instruction_plan(plan, reuse_hbm_flags=True)
    hbm_comms = [comm for comm in packed.comms if comm.kind in (3, 4)]

    assert hbm_comms
    assert packed.slots
    assert {slot.flag_index for slot in packed.slots} == {0}
    assert {comm.flag_index for comm in hbm_comms} == {0}
    assert sorted(set(comm.flag_epoch for comm in hbm_comms)) == [1, 2]


def test_pack_instruction_plan_allocates_barriers_for_hbm_recv_targets():
    plan = make_plan()
    hbm_recv_targets = {comm.target_slot_id for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_RECV}
    assert hbm_recv_targets

    packed = df.pack_instruction_plan(plan)
    barrier_slots = {slot_id for slot_id, slot in enumerate(packed.slots) if slot.barrier_index != df.UINT32_SENTINEL}

    assert hbm_recv_targets <= barrier_slots


def test_pack_instruction_plan_keeps_sparse_async_recv_barriers_when_hbm_flags_are_reused():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        task_extents=(2,),
        block_size=128,
    )

    packed = df.pack_instruction_plan(plan, reuse_hbm_flags=True)
    baseline = df.pack_instruction_plan(plan)
    cluster_comms = [comm for comm in packed.comms if comm.kind in (1, 2)]

    assert cluster_comms
    assert packed.barrier_allocation == baseline.barrier_allocation
    assert packed.barrier_allocation.require_valid() is packed.barrier_allocation
    assert {comm.barrier_index for comm in cluster_comms} == {df.UINT32_SENTINEL}
    assert [comm.flag_epoch for comm in cluster_comms] == [comm.flag_epoch for comm in baseline.comms if comm.kind in (1, 2)]


def test_pack_instruction_plan_only_allocates_barriers_for_async_recv_targets():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        task_extents=(2,),
        block_size=128,
    )

    async_recv_targets = {
        comm.target_slot_id for comm in plan.comms if comm.kind in (df.DataflowCommKind.CLUSTER_RECV, df.DataflowCommKind.HBM_RECV)
    }
    assert async_recv_targets

    packed = df.pack_instruction_plan(plan)
    barrier_slots = {slot_id for slot_id, slot in enumerate(packed.slots) if slot.barrier_index != df.UINT32_SENTINEL}

    assert barrier_slots == async_recv_targets
    allocation = packed.barrier_allocation.require_valid()
    assert set(allocation.slot_barrier_indices) == async_recv_targets
    assert all(0 <= barrier_index < allocation.barrier_count for barrier_index in allocation.slot_barrier_indices.values())
    assert df.build_launch_package(packed).barrier_count == allocation.barrier_count


def test_pack_instruction_plan_preserves_initial_cluster_barrier_phase_for_one_shot_slots():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [256, 256]},
        task_extents=(2,),
        block_size=128,
    )
    packed = df.pack_instruction_plan(plan)
    cluster_phases = [comm.flag_epoch for comm in packed.comms if comm.kind in (1, 2)]
    assert cluster_phases
    assert cluster_phases == [0, 0, 0, 0]


def test_pack_streaming_reduce_plan_reuses_physical_shared_slots():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [8192, 8192]},
        task_extents=(2,),
        block_size=64,
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
    )

    packed = df.pack_instruction_plan(plan)
    package = df.build_launch_package(packed)

    assert len(plan.slots) == 8
    assert package.slot_count == 8
    assert {slot.shared_offset for slot in packed.slots if slot.shared_offset == 0} == {0}
    assert sorted({slot.shared_offset for slot in packed.slots}) == [0, 48]
    assert package.shared_slot_bytes == 96
    assert package.global_staging_bytes == bytes(8 * 48)
    assert max(len(inst.input_slots) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE) == 2


def test_pack_instruction_plan_rejects_wrong_input_type():
    try:
        df.pack_instruction_plan(object())
    except TypeError as err:
        assert "InstructionPlan" in str(err)
    else:
        raise AssertionError("pack_instruction_plan should reject non-InstructionPlan inputs")
