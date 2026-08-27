from __future__ import annotations

from dataclasses import replace

import pytest

import tilelang.dataflow as df
from testing.python.dataflow.test_dataflow_primfunc_linking import make_scalar_program
from tilelang.dataflow.handler_abi import build_handler_abi
from tilelang.dataflow.primfunc_linking import link_primfunc_handlers_for_wrapper


_BLACKWELL_CROSS_TARGET = df.TargetCapabilitySnapshot.for_cuda(
    (12, 0),
    compiler_version=(12, 8),
)


def compile_scalar(*, inspection_stage: str):
    return df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode="inspect",
        inspection_stage=inspection_stage,
        target_override=_BLACKWELL_CROSS_TARGET,
    )


def test_logical_handler_params_have_structured_roles_before_cuda_codegen():
    compiled = compile_scalar(inspection_stage="ir")

    assert compiled.primfunc_lowering is not None
    assert compiled.primfunc_lowering.codegen_artifacts == ()
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    assert [param.role for param in handlers["iter"].params] == [
        df.DataflowHandlerParamRole.TENSOR_ARG,
        df.DataflowHandlerParamRole.OUTPUT_SLOT_FIELD,
        df.DataflowHandlerParamRole.RANGE_BEGIN,
        df.DataflowHandlerParamRole.RANGE_END,
        df.DataflowHandlerParamRole.TASK_ID,
    ]
    assert [param.role for param in handlers["reduce"].params] == [
        df.DataflowHandlerParamRole.INPUT_SLOT_FIELD,
        df.DataflowHandlerParamRole.INPUT_SLOT_FIELD,
        df.DataflowHandlerParamRole.OUTPUT_SLOT_FIELD,
        df.DataflowHandlerParamRole.INPUT_COUNT,
        df.DataflowHandlerParamRole.TASK_ID,
    ]
    for handler in handlers.values():
        assert int(handler.prim_func.attrs["tl.dataflow_param_schema_version"]) == (df.DATAFLOW_HANDLER_PARAM_SCHEMA_VERSION)
        assert tuple(str(role) for role in handler.prim_func.attrs["tl.dataflow_param_roles"]) == tuple(
            param.role.value for param in handler.params
        )


def test_cuda_handler_artifact_uses_lowered_module_metadata():
    compiled = compile_scalar(inspection_stage="cuda")

    assert compiled.primfunc_lowering is not None
    lowering = compiled.primfunc_lowering
    assert len(lowering.codegen_artifacts) == len(lowering.handlers) == 3
    for handler, artifact in zip(lowering.handlers, lowering.codegen_artifacts):
        assert artifact.handler_id == handler.handler_id
        assert artifact.device_symbol == handler.device_symbol
        assert artifact.thread_count == handler.thread_count
        assert artifact.dynamic_shared_bytes == handler.dynamic_shared_bytes
        assert artifact.target_fingerprint == compiled.target_fingerprint
        assert artifact.module is not lowering.ir_module
        assert artifact.param_names[-1] == "task_id"
        assert artifact.to_dict()["params"] == [param.to_dict() for param in artifact.params]


def test_linker_uses_param_roles_not_diagnostic_param_names():
    compiled = compile_scalar(inspection_stage="cuda")
    assert compiled.primfunc_lowering is not None
    lowering = compiled.primfunc_lowering
    iter_artifact = lowering.codegen_artifact(0)
    renamed_params = tuple(
        replace(param, name="diagnostic_output") if param.role is df.DataflowHandlerParamRole.OUTPUT_SLOT_FIELD else param
        for param in iter_artifact.params
    )
    renamed_artifact = replace(iter_artifact, params=renamed_params)
    renamed_lowering = replace(
        lowering,
        codegen_artifacts=(renamed_artifact, *lowering.codegen_artifacts[1:]),
    )
    abi = build_handler_abi(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        compiled.plan,
    )

    linked = link_primfunc_handlers_for_wrapper(
        renamed_lowering,
        compiled.wrapper_spec,
        abi,
    )

    assert len(linked.handler_sources) == 3
    assert "diagnostic_output" not in linked.handler_sources[0].body_source
    assert not any(source.completion_synchronized for source in linked.handler_sources)


def test_linker_composes_lowered_module_without_reading_cuda_source_text():
    compiled = compile_scalar(inspection_stage="cuda")
    assert compiled.primfunc_lowering is not None
    lowering = replace(
        compiled.primfunc_lowering,
        cuda_source="this is intentionally not CUDA source",
    )
    abi = build_handler_abi(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        compiled.plan,
    )

    linked = link_primfunc_handlers_for_wrapper(
        lowering,
        compiled.wrapper_spec,
        abi,
    )
    composed_source = df.generate_wrapper_source(
        replace(
            compiled.wrapper_spec,
            handler_codegen_module=linked.codegen_module,
            handler_helper_source=linked.helper_source,
            handler_sources=linked.handler_sources,
        )
    )

    assert len(linked.handler_sources) == len(lowering.handlers)
    assert all(handler.device_symbol in composed_source for handler in lowering.handlers)
    assert "intentionally not CUDA" not in composed_source


def test_linker_rejects_bad_tensor_index_from_structured_metadata():
    compiled = compile_scalar(inspection_stage="cuda")
    assert compiled.primfunc_lowering is not None
    lowering = compiled.primfunc_lowering
    iter_artifact = lowering.codegen_artifact(0)
    invalid_params = tuple(
        replace(param, tensor_arg_index=999) if param.role is df.DataflowHandlerParamRole.TENSOR_ARG else param
        for param in iter_artifact.params
    )
    invalid_artifact = replace(iter_artifact, params=invalid_params)
    invalid_lowering = replace(
        lowering,
        codegen_artifacts=(invalid_artifact, *lowering.codegen_artifacts[1:]),
    )
    abi = build_handler_abi(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        compiled.plan,
    )

    with pytest.raises(df.DataflowPrimFuncLinkingError, match="unknown tensor_arg_index=999"):
        link_primfunc_handlers_for_wrapper(
            invalid_lowering,
            compiled.wrapper_spec,
            abi,
        )
