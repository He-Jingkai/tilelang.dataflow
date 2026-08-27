from __future__ import annotations

import os

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from tilelang.contrib import nvcc
from tilelang.env import env


@T.dataflow_intermediate
class AttnInter:
    lse: T.float32
    o: T.Tensor((2, 8), T.float16)


@T.dataflow.iter(range=("kv_begin", "kv_end"))
def split_kv(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
    raise AssertionError("Dataflow iter body should not execute during wrapper compile test")


@T.dataflow.reduce
def combine(items: list[AttnInter]) -> AttnInter:
    raise AssertionError("Dataflow reduce body should not execute during wrapper compile test")


@T.dataflow.finalize
def finalize(inter: AttnInter, seq: T.int32, head: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during wrapper compile test")


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


def require_nvcc() -> None:
    try:
        compiler = nvcc.get_nvcc_compiler()
    except Exception as err:
        pytest.skip(f"CUDA toolkit/NVCC unavailable: {err}")
    if not os.path.exists(compiler):
        pytest.skip(f"NVCC not found at {compiler}")


def compile_wrapper_source(source: str, *, arch: str) -> tuple[bytes, str]:
    # Avoid TVM debug tempdir creation under /tmp, which can be unavailable on
    # shared test machines. Compile-only tests do not need retained artifacts.
    env.TILELANG_CLEANUP_TEMP_FILES = "1"

    return (
        bytes(
            nvcc.compile_cuda(
                source,
                target_format="ptx",
                arch=arch,
                options=nvcc.default_compile_options(),
                verbose=True,
            )
        ),
        arch,
    )


def test_generated_dataflow_wrapper_source_compiles_with_nvcc():
    require_nvcc()
    compiled = dataflow_debug.compile(
        make_program(),
        handler=dataflow_debug.EMPTY_HANDLER,
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        block_size=128,
        task_extents=(2,),
        wrapper_name="dataflow_compile_test_wrapper",
    )

    ptx, arch = compile_wrapper_source(
        compiled.wrapper_source,
        arch=compiled.target_capabilities.arch,
    )

    assert ptx
    assert arch == compiled.target_capabilities.arch
    assert b".entry dataflow_compile_test_wrapper" in ptx
