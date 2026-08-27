# Dataflow Frontend API

`tilelang.dataflow` describes a persistent GPU program as typed operators and a
dependency graph. The compiler schedules that graph across CTAs, lowers every
operator body through TileLang `PrimFunc`, and links the resulting device
handlers into one queue-driven CUDA wrapper.

The public examples are:

- `examples/dataflow/mla/mla_decode_non_paged.py`
- `examples/dataflow/fusedmoe/example_fusedmoe_dataflow.py`

They intentionally keep operator definitions separate from benchmark drivers.

## Imports

Operator declarations use TileLang language primitives. Planning and compile
configuration live in `tilelang.dataflow`:

```python
import tilelang.language as T
import tilelang.dataflow as df
```

## Intermediate values

An intermediate is the typed value passed between handlers. Fields may be
scalars or fixed-shape tensors.

```python
@T.dataflow_intermediate
class Partial:
    scale: T.float32
    value: T.Tensor((64,), T.float16)
```

Use `df.get_intermediate_type(Partial)` when metadata inspection is needed.
Field order is part of the layout contract.

## Operators

Dataflow operator decorators use concise names for compute stages:

- `@T.dataflow.iter` computes one partial over a dynamic range.
- `@T.dataflow.map` declares a general map stage.
- `@T.dataflow.reduce` combines compatible intermediates.
- `@T.dataflow.finalize` performs terminal writes and returns no value.

The former `T.dataflow_iter`, `T.dataflow_map`, `T.dataflow_reduce`, and
`T.dataflow_finalize` names remain compatibility aliases. A normal
`T.reduce(buffer, out, ...)` call still performs TileLang buffer reduction.

```python
@T.dataflow.iter(range=("begin", "end"), threads=256)
def split(task: T.int32, Input: T.Tensor((4096,), T.float16)) -> Partial:
    begin = T.cast(T.dataflow_range_begin(), "int32")
    end = T.cast(T.dataflow_range_end(), "int32")
    # TileLang computation omitted.
    ...


@T.dataflow.reduce
def combine(items: list[Partial]) -> Partial:
    ...


@T.dataflow.finalize
def writeback(item: Partial, Output: T.Tensor((1, 64), T.float16)) -> None:
    task = T.cast(T.dataflow_task_id(), "int32")
    ...
```

Calling a decorated function creates an `OperatorCall`; it does not execute the
body. External tensor parameters omitted from the call are bound to same-named
runtime arguments. Explicit bindings override those defaults.

Operator decorators accept typed contracts for specialization, physical slot
ownership, pipelines, tensor layouts, precision, and cross-handler handoff.
Those contracts are validated before CUDA source is produced.

## Programs

Create a program with a task domain and any dynamic range axes, then add stages:

```python
program = T.dataflow_program(
    task_domain=("batch",),
    dynamic_ranges={"kv": "seq_lens"},
    name="example",
)

program = (
    program.partial(
        split(),
        task_args=("batch",),
        range_axis="kv",
        placement="cluster",
    )
    .reduce(combine(), arity="tree")
    .finalize(writeback())
)
program.validate()
```

The frontend also supports general stage graphs through `map`, `reshared`, and
explicit dependencies. The compiler validates stage order, intermediate types,
terminal side effects, range contracts, and transport legality.

## Kernel factories

For reusable operators, return a typed `DataflowKernelSpec` from a JIT factory:

```python
@df.jit
def make_kernel(size: int, **compile_options):
    program = build_program(size)
    return df.make_kernel_spec(
        program,
        topology=df.GPUTopology(sm_count=120, cluster_size=2),
        range_lengths={"kv": (size,)},
        block_size=128,
        **compile_options,
    )
```

`@df.jit(cache=False)` disables the in-process compile cache. Cache identity is
derived from the complete typed compile configuration and target capability
snapshot, rather than from generated source text.

For one-off programs, call `df.compile` directly:

```python
compiled = df.compile(
    program,
    topology=df.GPUTopology(sm_count=120, cluster_size=2),
    range_lengths={"kv": (4096,)},
    block_size=128,
)
```

Compilation accepts typed scheduler, semantic, precision, memory, execution,
and target options. Prefer their dataclass forms over unstructured option
dictionaries when exposing a library API.

## Artifacts and inspection

Production compilation returns an executable `DataflowCompiledProgram`.
`mode="inspect"` returns a deliberately non-executable artifact for examining
an intermediate compiler stage.

Useful structured outputs include:

```python
compiled.compile_config
compiled.plan
compiled.memory_plan
compiled.primfunc_lowering
compiled.decision_artifact()
compiled.dump_plan()
```

Generated CUDA is an implementation detail. Tooling should consume the typed
decision artifact and plans instead of parsing source strings.

## Launch

Call the compiled program with the runtime tensors named by the operator
signatures:

```python
result = compiled(Input=input_tensor, Output=output_tensor)
```

Before launch, Dataflow validates tensor count, shape, dtype, device, stride,
and any declared logical-to-physical layout. The target snapshot captured at
compile time is checked against the launch device before loading the cubin.

## Runtime intrinsics

Operator bodies can access scheduler-owned coordinates with:

- `T.dataflow_task_id()`
- `T.dataflow_task_coord(axis)`
- `T.dataflow_range_begin()`
- `T.dataflow_range_end()`
- `T.dataflow_range_tiles_per_handler()`

These values are provided by the wrapper ABI and must not be replaced with
workload-specific constants.

## Compatibility

Only the typed `PrimFunc` production path is public. Synthetic debug handlers
are available under `tilelang.dataflow.experimental` for compiler diagnostics
and carry no compatibility guarantee.

When extending Dataflow, keep scheduling decisions in typed plans, keep
operator-specific computation in operator bodies, and keep transport and
lifetime validation independent of model or workload names.
